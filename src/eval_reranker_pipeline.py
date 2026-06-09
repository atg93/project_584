"""
End-to-end pipeline evaluation: BERT-GRU bi-encoder (from train.py) +
LoRA reranker (from reranker.py).

Steps
-----
  1. Load QueryEncoder / DocEncoder from a train.py checkpoint dir
     (expects query_enc.pt, doc_enc.pt, meta.json).
  2. Build a FAISS index over docs and retrieve top-K dense candidates
     for every dev query.
  3. Load the LoRA reranker from a reranker.py checkpoint dir.
  4. Re-rank the top --rerank-top-k of each candidate list.
  5. Report NDCG@1000 and R@100 for both stages so the lift is visible.

Usage
-----
    python eval_reranker_pipeline.py \\
        --bi-checkpoint       checkpoints/best \\
        --reranker-checkpoint checkpoints/reranker/best \\
        --docs                data/trec/docs.jsonl \\
        --queries             data/trec/dev_queries.jsonl \\
        --qrels               data/trec/dev_qrels.txt
"""

import os

# Pin all CUDA work to GPU 0. Must be set BEFORE torch / transformers /
# bitsandbytes import or the visibility mask is ignored.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import argparse
import numpy as np
import torch
import faiss
from tqdm import tqdm

from dataset import get_trec_dataloader, get_doc_dataloader, load_jsonl, load_qrels
from tot_retrieval import QueryEncoder, DocEncoder
from evaluate import build_index, ndcg_at_k, recall_at_k
from reranker import LoRAReranker, rerank


# ---------------------------------------------------------------------------
# Cached document index — encode once, reuse on every re-run
# ---------------------------------------------------------------------------

def build_index_cached(doc_encoder, doc_loader, device, cache_key,
                       cache_dir, docs_path, refresh=False):
    """
    Like evaluate.build_index but caches the (expensive) document embeddings to
    disk as .npy + .ids.json, keyed by `cache_key` (the doc encoder's checkpoint)
    and the docs filename. Re-runs load from cache and skip the ~2h encode.

    The FAISS index itself is cheap, so it is rebuilt from the cached vectors
    each run. Pass refresh=True to force re-encoding.
    """
    os.makedirs(cache_dir, exist_ok=True)
    safe     = (cache_key + '__' + os.path.basename(docs_path)).replace('/', '_')
    vec_path = os.path.join(cache_dir, safe + '.npy')
    ids_path = os.path.join(cache_dir, safe + '.ids.json')

    if not refresh and os.path.exists(vec_path) and os.path.exists(ids_path):
        print(f'[cache] loading doc embeddings ← {vec_path}')
        all_vecs = np.load(vec_path)
        with open(ids_path) as f:
            all_ids = json.load(f)
    else:
        print(f'[cache] encoding documents (will cache → {vec_path})')
        doc_encoder.eval()
        vecs_list, all_ids = [], []
        with torch.no_grad():
            for batch in tqdm(doc_loader, desc='Encoding documents'):
                vecs = doc_encoder(batch['input_ids'].to(device),
                                   batch['attention_mask'].to(device))
                vecs_list.append(vecs.cpu().numpy())
                all_ids.extend(batch['doc_id'])
        all_vecs = np.vstack(vecs_list).astype('float32')
        np.save(vec_path, all_vecs)
        with open(ids_path, 'w') as f:
            json.dump(all_ids, f)
        print(f'[cache] saved {all_vecs.shape[0]} doc embeddings → {vec_path}')

    all_vecs = all_vecs.astype('float32')
    faiss.normalize_L2(all_vecs)
    index = faiss.IndexFlatIP(all_vecs.shape[1])
    index.add(all_vecs)
    return index, all_ids


# ---------------------------------------------------------------------------
# Dense retrieval — capture full ranked lists for downstream reranking
# ---------------------------------------------------------------------------

def dense_retrieve(query_enc, doc_enc, query_loader, doc_loader, device, k=1000,
                   cache_key=None, cache_dir=None, docs_path=None, refresh=False):
    """
    Run the bi-encoder and return {query_id: [doc_id, ...]} (top-k per query).
    Document embeddings are cached to disk (see build_index_cached) so re-runs
    skip the expensive encode.
    """
    index, idx_to_docid = build_index_cached(
        doc_enc, doc_loader, device, cache_key=cache_key,
        cache_dir=cache_dir, docs_path=docs_path, refresh=refresh)

    query_enc.eval()
    retrieved = {}

    with torch.no_grad():
        for batch in tqdm(query_loader, desc='Dense retrieval'):
            q_vecs = query_enc(
                batch['sentence_ids'].to(device),
                batch['sentence_masks'].to(device),
            ).cpu().numpy().astype('float32')

            faiss.normalize_L2(q_vecs)
            _, doc_indices = index.search(q_vecs, k)

            for i, qid in enumerate(batch['query_id']):
                retrieved[qid] = [idx_to_docid[j] for j in doc_indices[i]]

    return retrieved


# ---------------------------------------------------------------------------
# Metric helper — run NDCG@1000 / R@100 over a {qid: ranked_ids} mapping
# ---------------------------------------------------------------------------

def score_runs(run, qrels, k_ndcg=1000, k_recall=100):
    ndcg_scores   = []
    recall_scores = []
    per_query     = {}
    for qid, ranked in run.items():
        rel = qrels.get(qid, [])
        ndcg   = ndcg_at_k(ranked, rel, k=k_ndcg)
        recall = recall_at_k(ranked, rel, k=k_recall)
        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
        per_query[qid] = {f'ndcg@{k_ndcg}': ndcg, f'r@{k_recall}': recall}
    return {
        f'mean_ndcg@{k_ndcg}': float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        f'mean_r@{k_recall}':  float(np.mean(recall_scores)) if recall_scores else 0.0,
        'per_query':           per_query,
    }


# ---------------------------------------------------------------------------
# Rerank a full {qid: ranked_ids} run using the LoRA model
# ---------------------------------------------------------------------------

def rerank_run(reranker, dense_run, queries, docs, rerank_top_k=100, batch_size=8):
    reranked_run = {}
    for qid, ranked_ids in tqdm(dense_run.items(), desc='Re-ranking'):
        query      = queries.get(qid, {})
        query_text = query.get('text', ' '.join(query.get('sentences', [])))

        top    = rerank(reranker, query_text, ranked_ids, docs,
                        top_k=rerank_top_k, batch_size=batch_size)
        tail   = ranked_ids[rerank_top_k:]
        reranked_run[qid] = top + tail
    return reranked_run


# ---------------------------------------------------------------------------
# TREC run file writer
# ---------------------------------------------------------------------------

def save_trec_run(run, path, run_name):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        for qid, ranked in run.items():
            for rank, doc_id in enumerate(ranked, start=1):
                # synthetic descending score — many TREC tools just need ordering
                score = 1.0 / rank
                f.write(f'{qid} Q0 {doc_id} {rank} {score:.6f} {run_name}\n')
    print(f'TREC run saved → {path}')


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_block(title, results, k_ndcg, k_recall):
    print('\n' + '=' * 50)
    print(title)
    print('=' * 50)
    print(f'  mean_ndcg@{k_ndcg:<6}  {results[f"mean_ndcg@{k_ndcg}"]:.4f}')
    print(f'  mean_r@{k_recall:<9}  {results[f"mean_r@{k_recall}"]:.4f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- 1. load bi-encoder ----
    query_enc = QueryEncoder(
        bert_model=args.bert_model,
        gru_hidden=args.gru_hidden,
        sentence_dropout=args.sentence_dropout,
        proj_dim=args.proj_dim,
    ).to(device)
    doc_enc = DocEncoder(bert_model=args.bert_model, proj_dim=args.proj_dim).to(device)

    q_path = os.path.join(args.bi_checkpoint, 'query_enc.pt')
    d_path = os.path.join(args.bi_checkpoint, 'doc_enc.pt')
    query_enc.load_state_dict(torch.load(q_path, map_location=device))
    doc_enc.load_state_dict(torch.load(d_path, map_location=device))
    print(f'Bi-encoder loaded from {args.bi_checkpoint}')

    # ---- 2. dataloaders ----
    query_loader = get_trec_dataloader(
        query_path=args.queries,
        doc_path=args.docs,
        qrel_path=args.qrels,
        batch_size=args.batch_size,
        tokenizer_name=args.bert_model,
        shuffle=False,
        distributed=False,
    )
    doc_loader = get_doc_dataloader(
        args.docs,
        batch_size=args.eval_batch_size,
        tokenizer_name=args.bert_model,
    )

    qrels   = load_qrels(args.qrels)
    queries = {q['id']: q for q in load_jsonl(args.queries)}
    docs    = {d['id']: d for d in load_jsonl(args.docs)}

    # ---- 3. dense retrieval ----
    dense_run = dense_retrieve(query_enc, doc_enc, query_loader, doc_loader,
                               device, k=args.k_ndcg,
                               cache_key=args.bi_checkpoint,
                               cache_dir=args.cache_dir,
                               docs_path=args.docs,
                               refresh=args.refresh_cache)

    dense_results = score_runs(dense_run, qrels,
                               k_ndcg=args.k_ndcg, k_recall=args.k_recall)
    print_block('Bi-encoder only', dense_results, args.k_ndcg, args.k_recall)

    # free GPU memory before loading the LLM reranker
    del query_enc, doc_enc
    torch.cuda.empty_cache()

    # ---- 4. load reranker ----
    reranker = LoRAReranker(
        model_name=args.reranker_model,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=torch.float32 if not args.load_in_4bit else None,
        checkpoint_path=args.reranker_checkpoint,
    )

    # ---- 5. rerank top-K ----
    reranked_run = rerank_run(reranker, dense_run, queries, docs,
                              rerank_top_k=args.rerank_top_k,
                              batch_size=args.rerank_batch_size)
    rerank_results = score_runs(reranked_run, qrels,
                                k_ndcg=args.k_ndcg, k_recall=args.k_recall)
    print_block(f'Bi-encoder + LoRA reranker (top-{args.rerank_top_k})',
                rerank_results, args.k_ndcg, args.k_recall)

    # ---- 6. lift summary ----
    d_ndcg   = rerank_results[f'mean_ndcg@{args.k_ndcg}']   - dense_results[f'mean_ndcg@{args.k_ndcg}']
    d_recall = rerank_results[f'mean_r@{args.k_recall}']    - dense_results[f'mean_r@{args.k_recall}']
    print('\nDelta from re-ranking:')
    print(f'  Δ ndcg@{args.k_ndcg}   {d_ndcg:+.4f}')
    print(f'  Δ r@{args.k_recall:<6}   {d_recall:+.4f}')
    print('=' * 50 + '\n')

    # ---- 7. optional outputs ----
    if args.output_run:
        save_trec_run(reranked_run, args.output_run,
                      run_name=args.run_name)

    if args.output_scores:
        os.makedirs(os.path.dirname(args.output_scores) or '.', exist_ok=True)
        with open(args.output_scores, 'w') as f:
            json.dump({
                'dense':    {k: v for k, v in dense_results.items()  if k != 'per_query'},
                'reranked': {k: v for k, v in rerank_results.items() if k != 'per_query'},
                'per_query': {
                    'dense':    dense_results['per_query'],
                    'reranked': rerank_results['per_query'],
                },
            }, f, indent=2)
        print(f'Per-query scores saved → {args.output_scores}')


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate bi-encoder + LoRA reranker pipeline')

    # checkpoints
    p.add_argument('--bi-checkpoint',       default='checkpoints/best',
                   help='Directory containing query_enc.pt / doc_enc.pt from train.py')
    p.add_argument('--reranker-checkpoint', default='checkpoints/reranker/best',
                   help='LoRA adapter directory from reranker.py')

    # data
    p.add_argument('--docs',    default='data/trec/docs.jsonl')
    p.add_argument('--queries', default='data/trec/dev_queries.jsonl')
    p.add_argument('--qrels',   default='data/trec/dev_qrels.txt')

    # bi-encoder hyperparameters (must match training config)
    p.add_argument('--bert-model',       default='bert-base-uncased')
    p.add_argument('--gru-hidden',       type=int,   default=None)
    p.add_argument('--sentence-dropout', type=float, default=0.2)
    p.add_argument('--proj-dim',         type=int,   default=512)

    # reranker hyperparameters
    p.add_argument('--reranker-model',     default='Qwen/Qwen2.5-0.5B')
    p.add_argument('--load-in-4bit',       action='store_true')
    p.add_argument('--rerank-top-k',       type=int, default=100)
    p.add_argument('--rerank-batch-size',  type=int, default=8)

    # doc-embedding cache
    p.add_argument('--cache-dir', default='cache/doc_embeddings',
                   help='Where to cache per-checkpoint document embeddings')
    p.add_argument('--refresh-cache', action='store_true',
                   help='Force re-encoding documents even if a cache exists')

    # eval config
    p.add_argument('--batch-size',      type=int, default=16)
    p.add_argument('--eval-batch-size', type=int, default=256)
    p.add_argument('--k-ndcg',          type=int, default=1000)
    p.add_argument('--k-recall',        type=int, default=100)

    # outputs
    p.add_argument('--output-run',    default=None,
                   help='Optional path to write reranked TREC run file')
    p.add_argument('--output-scores', default=None,
                   help='Optional path to dump per-query metrics as JSON')
    p.add_argument('--run-name',      default='biencoder_lora_rerank')

    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
