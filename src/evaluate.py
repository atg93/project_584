import json
import os
import numpy as np
import torch
import faiss
from tqdm import tqdm

from dataset import get_doc_dataloader, load_jsonl, load_qrels
from tot_retrieval import QueryEncoder, DocEncoder


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(ranked_doc_ids, relevant_doc_ids, k=1000):
    """
    NDCG@K with binary relevance.
    Primary metric in TREC 2023 ToT track.
    """
    relevant_set = set(relevant_doc_ids)

    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in relevant_set:
            dcg += 1.0 / np.log2(i + 2)  # position is 1-indexed → log2(rank+1)

    n_rel = min(len(relevant_doc_ids), k)
    idcg  = sum(1.0 / np.log2(i + 2) for i in range(n_rel))

    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_doc_ids, relevant_doc_ids, k=100):
    """
    R@K — fraction of relevant documents retrieved in top-K.
    Critical for measuring candidate set quality before re-ranking.
    """
    relevant_set = set(relevant_doc_ids)
    retrieved    = set(ranked_doc_ids[:k])
    return len(relevant_set & retrieved) / len(relevant_set) if relevant_set else 0.0


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_index(doc_encoder, doc_dataloader, device):
    """Encode all documents and build a FAISS flat inner-product index."""
    doc_encoder.eval()
    all_vecs = []
    all_ids  = []

    with torch.no_grad():
        for batch in tqdm(doc_dataloader, desc='Encoding documents'):
            vecs = doc_encoder(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device)
            )
            all_vecs.append(vecs.cpu().numpy())
            all_ids.extend(batch['doc_id'])

    all_vecs = np.vstack(all_vecs).astype('float32')
    faiss.normalize_L2(all_vecs)

    index = faiss.IndexFlatIP(all_vecs.shape[1])
    index.add(all_vecs)

    return index, all_ids  # all_ids[i] = doc_id for FAISS index position i


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(query_encoder, doc_encoder, query_dataloader, doc_dataloader,
             qrels, device, k_ndcg=1000, k_recall=100):
    """
    Full evaluation pipeline:
      1. Build FAISS index from all documents
      2. Encode each query with QueryEncoder
      3. Retrieve top-K documents
      4. Compute NDCG@1000 and R@100 per query, then macro-average

    Args:
        qrels: dict {query_id: [relevant_doc_id, ...]}

    Returns:
        dict with mean NDCG@1000, R@100, and per-query scores
    """
    # step 1 — build document index
    index, idx_to_docid = build_index(doc_encoder, doc_dataloader, device)

    # step 2 — encode queries and retrieve
    query_encoder.eval()
    ndcg_scores   = []
    recall_scores = []
    per_query     = {}

    with torch.no_grad():
        for batch in tqdm(query_dataloader, desc='Evaluating queries'):
            q_vecs = query_encoder(
                batch['sentence_ids'].to(device),
                batch['sentence_masks'].to(device)
            ).cpu().numpy().astype('float32')

            faiss.normalize_L2(q_vecs)
            _, doc_indices = index.search(q_vecs, k_ndcg)  # (batch, k_ndcg)

            for i, qid in enumerate(batch['query_id']):
                ranked_ids   = [idx_to_docid[j] for j in doc_indices[i]]
                relevant_ids = qrels.get(qid, [])

                ndcg   = ndcg_at_k(ranked_ids, relevant_ids, k=k_ndcg)
                recall = recall_at_k(ranked_ids, relevant_ids, k=k_recall)

                ndcg_scores.append(ndcg)
                recall_scores.append(recall)
                per_query[qid] = {'ndcg@1000': ndcg, f'r@{k_recall}': recall}

    results = {
        f'mean_ndcg@{k_ndcg}': float(np.mean(ndcg_scores)),
        f'mean_r@{k_recall}':   float(np.mean(recall_scores)),
        'per_query':            per_query,
    }

    return results


# ---------------------------------------------------------------------------
# TREC run file export — for official TREC evaluation with trec_eval
# ---------------------------------------------------------------------------

def save_candidates(query_encoder, doc_encoder, query_dataloader, doc_dataloader,
                    device, output_path, k=1000, index=None, idx_to_docid=None,
                    append=False):
    """
    Saves top-K retrieved doc IDs per query in jsonl format:
        {"query_id": "...", "retrieved": ["doc1", "doc2", ...]}
    Required input for RerankerDataset in reranker.py.

    Pass a pre-built (index, idx_to_docid) to avoid re-encoding documents when
    calling for multiple query sets (e.g. TREC + Reddit) against the same corpus.
    Set append=True for the second and subsequent calls to the same output_path.
    """
    if index is None or idx_to_docid is None:
        index, idx_to_docid = build_index(doc_encoder, doc_dataloader, device)

    query_encoder.eval()
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    mode = 'a' if append else 'w'

    with open(output_path, mode) as f, torch.no_grad():
        for batch in tqdm(query_dataloader, desc='Generating candidates'):
            q_vecs = query_encoder(
                batch['sentence_ids'].to(device),
                batch['sentence_masks'].to(device)
            ).cpu().numpy().astype('float32')

            faiss.normalize_L2(q_vecs)
            _, doc_indices = index.search(q_vecs, k)

            for i, qid in enumerate(batch['query_id']):
                retrieved = [idx_to_docid[j] for j in doc_indices[i]]
                f.write(json.dumps({'query_id': qid, 'retrieved': retrieved}) + '\n')

    print(f'Candidates {"appended to" if append else "saved to"} {output_path}')
    return index, idx_to_docid


def save_trec_run(query_encoder, doc_encoder, query_dataloader, doc_dataloader,
                  device, output_path, run_name='bert_gru_attn', k=1000):
    """
    Saves retrieval results in standard TREC run format:
        qid Q0 docid rank score run_name
    Compatible with the official trec_eval tool.
    """
    index, idx_to_docid = build_index(doc_encoder, doc_dataloader, device)
    query_encoder.eval()

    with open(output_path, 'w') as f, torch.no_grad():
        for batch in tqdm(query_dataloader, desc='Writing TREC run'):
            q_vecs = query_encoder(
                batch['sentence_ids'].to(device),
                batch['sentence_masks'].to(device)
            ).cpu().numpy().astype('float32')

            faiss.normalize_L2(q_vecs)
            scores, doc_indices = index.search(q_vecs, k)

            for i, qid in enumerate(batch['query_id']):
                for rank, (doc_idx, score) in enumerate(zip(doc_indices[i], scores[i])):
                    doc_id = idx_to_docid[doc_idx]
                    f.write(f'{qid} Q0 {doc_id} {rank + 1} {score:.6f} {run_name}\n')

    print(f'TREC run saved to {output_path}')


# ---------------------------------------------------------------------------
# Ablation: shuffle sentence order at eval time
# ---------------------------------------------------------------------------

def evaluate_shuffled(query_encoder, doc_encoder, query_dataloader, doc_dataloader,
                      qrels, device, n_runs=5, k_ndcg=1000, k_recall=100):
    """
    Runs evaluation with randomly shuffled sentence order N times and averages.
    Compare against evaluate() to measure how much sentence order matters to GRU.
    """
    index, idx_to_docid = build_index(doc_encoder, doc_dataloader, device)
    query_encoder.eval()

    all_ndcg   = []
    all_recall = []

    for run in range(n_runs):
        ndcg_scores   = []
        recall_scores = []

        with torch.no_grad():
            for batch in query_dataloader:
                sent_ids   = batch['sentence_ids']
                sent_masks = batch['sentence_masks']
                N          = sent_ids.shape[1]

                perm       = torch.randperm(N)
                sent_ids   = sent_ids[:, perm, :]
                sent_masks = sent_masks[:, perm, :]

                q_vecs = query_encoder(
                    sent_ids.to(device),
                    sent_masks.to(device)
                ).cpu().numpy().astype('float32')

                faiss.normalize_L2(q_vecs)
                _, doc_indices = index.search(q_vecs, k_ndcg)

                for i, qid in enumerate(batch['query_id']):
                    ranked_ids   = [idx_to_docid[j] for j in doc_indices[i]]
                    relevant_ids = qrels.get(qid, [])
                    ndcg_scores.append(ndcg_at_k(ranked_ids, relevant_ids, k=k_ndcg))
                    recall_scores.append(recall_at_k(ranked_ids, relevant_ids, k=k_recall))

        all_ndcg.append(np.mean(ndcg_scores))
        all_recall.append(np.mean(recall_scores))

    return {
        f'shuffled_mean_ndcg@{k_ndcg}': float(np.mean(all_ndcg)),
        f'shuffled_std_ndcg@{k_ndcg}':  float(np.std(all_ndcg)),
        f'shuffled_mean_r@{k_recall}':   float(np.mean(all_recall)),
    }


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_results(results, shuffled_results=None):
    print('\n' + '=' * 45)
    print('Evaluation Results')
    print('=' * 45)
    for k, v in results.items():
        if k != 'per_query':
            print(f'  {k:<25} {v:.4f}')

    if shuffled_results:
        print('\nShuffled Sentence Order (ablation):')
        for k, v in shuffled_results.items():
            print(f'  {k:<35} {v:.4f}')
    print('=' * 45 + '\n')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    from dataset import get_trec_dataloader, get_doc_dataloader, load_qrels

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    query_enc = QueryEncoder().to(device)
    doc_enc   = DocEncoder().to(device)

    # load a checkpoint if available
    # query_enc.load_state_dict(torch.load('checkpoints/query_enc.pt'))
    # doc_enc.load_state_dict(torch.load('checkpoints/doc_enc.pt'))

    qrels = load_qrels('data/trec/qrels.txt')

    query_loader = get_trec_dataloader(
        query_path='data/trec/dev_queries.jsonl',
        doc_path='data/trec/docs.jsonl',
        qrel_path='data/trec/qrels.txt',
        batch_size=16,
        shuffle=False
    )

    doc_loader = get_doc_dataloader(
        doc_path='data/trec/docs.jsonl',
        batch_size=64
    )

    # standard evaluation
    results = evaluate(query_enc, doc_enc, query_loader, doc_loader, qrels, device)

    # shuffled sentence order ablation
    shuffled = evaluate_shuffled(query_enc, doc_enc, query_loader, doc_loader, qrels, device)

    print_results(results, shuffled)

    # save TREC run file for official trec_eval scoring
    save_trec_run(
        query_enc, doc_enc, query_loader, doc_loader,
        device, output_path='runs/bert_gru_attn.txt'
    )
