"""
Three-stage ablation evaluation on the TREC ToT dev set:

    Stage 1  BERT only                 — sentence [CLS] mean-pool → proj  (GRU bypassed)
    Stage 2  BERT + GRU                 — full QueryEncoder.forward
    Stage 3  BERT + GRU + LLM reranker  — re-rank stage-2 top-K with the LoRA reranker

Reports NDCG@1000 and R@100 for every stage plus the deltas between them, so the
contribution of the GRU aggregation and of the LLM reranker are each isolated.

This is a NEW, read-only consumer of the existing modules — it imports from
evaluate.py / tot_retrieval.py / reranker.py and changes none of them.

What "BERT only" means here
---------------------------
The trained QueryEncoder is BERT (per-sentence [CLS]) -> GRU + attention -> proj.
To ablate the GRU we keep BERT and the trained `proj` but replace the GRU+attention
aggregation with a padding-aware mean over the sentence [CLS] embeddings. The doc
side (DocEncoder) is identical across all stages, so the same FAISS index is reused.

Usage
-----
    python eval_ablation_pipeline.py \\
        --bi-checkpoint       checkpoints/best \\
        --reranker-checkpoint checkpoints/reranker/best \\
        --docs                data/trec/docs.jsonl \\
        --queries             data/trec/dev_queries.jsonl \\
        --qrels               data/trec/dev_qrels.txt
"""

import os

# Pin all CUDA work to GPU 0. Must be set BEFORE torch / transformers / bitsandbytes
# import or the visibility mask is ignored. (mirrors eval_reranker_pipeline.py)
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import faiss
from tqdm import tqdm
from transformers import AutoModel

from dataset import get_trec_dataloader, get_doc_dataloader, load_jsonl, load_qrels
from tot_retrieval import QueryEncoder, DocEncoder
from tokenizer_eval import GenericQueryEncoder, GenericDocEncoder
from evaluate import build_index, ndcg_at_k, recall_at_k
from reranker import LoRAReranker, rerank

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


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
    each run. Pass refresh=True to force re-encoding (e.g. after retraining a
    checkpoint that kept the same path).
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
# Retrieval — encode every dev query and search a (matching) document index
# ---------------------------------------------------------------------------

def retrieve_run(query_enc, query_loader, index, idx_to_docid, device,
                 k=1000, tag='retrieval'):
    """
    Run a bi-encoder QueryEncoder over all dev queries against its OWN prebuilt
    FAISS index and return {query_id: [doc_id, ...]} (top-k).

    The query_enc and the index must come from the same checkpoint (same trained
    projection space) — each stage builds its own index from its own DocEncoder.
    """
    query_enc.eval()
    run = {}

    with torch.no_grad():
        for batch in tqdm(query_loader, desc=f'Retrieval [{tag}]'):
            q_vecs = query_enc(
                batch['sentence_ids'].to(device),
                batch['sentence_masks'].to(device),
            ).cpu().numpy().astype('float32')

            faiss.normalize_L2(q_vecs)
            _, doc_indices = index.search(q_vecs, k)

            for i, qid in enumerate(batch['query_id']):
                run[qid] = [idx_to_docid[j] for j in doc_indices[i]]

    return run


# ---------------------------------------------------------------------------
# Zero-shot pretrained BERT (no fine-tuning) — raw [CLS], no projection head
# ---------------------------------------------------------------------------

class _RawCLSDocEncoder(nn.Module):
    """Wraps a pretrained HF encoder so build_index() can use it: returns [CLS]."""
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.encoder(input_ids=input_ids,
                            attention_mask=attention_mask).last_hidden_state[:, 0, :]


def retrieve_pretrained(encoder, query_loader, index, idx_to_docid, device,
                        k=1000, tag='pretrained'):
    """
    Zero-shot query retrieval: encode each sentence's [CLS] with the pretrained
    model and mean-pool over the real (non-padding) sentences — no projection.
    """
    encoder.eval()
    run = {}
    with torch.no_grad():
        for batch in tqdm(query_loader, desc=f'Retrieval [{tag}]'):
            sent_ids   = batch['sentence_ids'].to(device)
            sent_masks = batch['sentence_masks'].to(device)
            B, N, L = sent_ids.shape

            cls = encoder.encoder(input_ids=sent_ids.view(B * N, L),
                                  attention_mask=sent_masks.view(B * N, L)
                                  ).last_hidden_state[:, 0, :].view(B, N, -1)   # (B, N, H)

            # padding-aware mean over sentences
            idx    = torch.arange(N, device=device).unsqueeze(0)
            valid  = (idx < batch['num_sentences'].to(device).unsqueeze(1)).float().unsqueeze(-1)
            pooled = (cls * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)

            q_vecs = pooled.cpu().numpy().astype('float32')
            faiss.normalize_L2(q_vecs)
            _, doc_indices = index.search(q_vecs, k)
            for i, qid in enumerate(batch['query_id']):
                run[qid] = [idx_to_docid[j] for j in doc_indices[i]]
    return run


# ---------------------------------------------------------------------------
# Rerank a full {qid: ranked_ids} run with the LoRA model
# ---------------------------------------------------------------------------

def rerank_run(reranker, dense_run, queries, docs, rerank_top_k=100, batch_size=8):
    reranked_run = {}
    for qid, ranked_ids in tqdm(dense_run.items(), desc='Re-ranking'):
        query      = queries.get(qid, {})
        query_text = query.get('text', ' '.join(query.get('sentences', [])))

        top  = rerank(reranker, query_text, ranked_ids, docs,
                      top_k=rerank_top_k, batch_size=batch_size)
        tail = ranked_ids[rerank_top_k:]
        reranked_run[qid] = top + tail
    return reranked_run


# ---------------------------------------------------------------------------
# Metrics over a {qid: ranked_ids} mapping
# ---------------------------------------------------------------------------

def score_runs(run, qrels, k_ndcg=1000, k_recall=100):
    """
    Computes NDCG@10, NDCG@{k_ndcg}, R@{k_recall} and R@1000 per query, then
    macro-averages. The four metric keys are returned under `metric_keys` so the
    callers (print / W&B / JSON) stay generic.
    """
    ndcg_cutoffs   = sorted({10, k_ndcg})
    recall_cutoffs = sorted({k_recall, 1000})
    metric_keys = ([f'ndcg@{k}' for k in ndcg_cutoffs]
                   + [f'r@{k}' for k in recall_cutoffs])

    sums      = {f'mean_{m}': 0.0 for m in metric_keys}
    per_query = {}
    n = 0
    for qid, ranked in run.items():
        rel = qrels.get(qid, [])
        pq  = {}
        for k in ndcg_cutoffs:
            pq[f'ndcg@{k}'] = ndcg_at_k(ranked, rel, k=k)
        for k in recall_cutoffs:
            pq[f'r@{k}'] = recall_at_k(ranked, rel, k=k)
        for m, v in pq.items():
            sums[f'mean_{m}'] += v
        per_query[qid] = pq
        n += 1

    results = {key: (val / n if n else 0.0) for key, val in sums.items()}
    results['per_query']   = per_query
    results['metric_keys'] = metric_keys
    return results


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_block(title, results, k_ndcg=None, k_recall=None):
    print('\n' + '=' * 52)
    print(title)
    print('=' * 52)
    for m in results['metric_keys']:
        print(f'  mean_{m:<12}  {results[f"mean_{m}"]:.4f}')


def print_delta(label, new, old, k_ndcg=None, k_recall=None):
    print(f'\n{label}')
    for m in new['metric_keys']:
        print(f'  Δ {m:<12}  {new[f"mean_{m}"] - old[f"mean_{m}"]:+.4f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_meta(checkpoint_dir, args):
    """
    If checkpoints/<...>/meta.json exists, use its bi-encoder hyperparameters so
    they always match the trained weights. CLI flags act as fallbacks/overrides
    only for keys missing from meta.json.
    """
    meta_path = os.path.join(checkpoint_dir, 'meta.json')
    if not os.path.exists(meta_path):
        print(f'No meta.json in {checkpoint_dir} — using CLI hyperparameters')
        return
    with open(meta_path) as f:
        meta = json.load(f)
    for cli_key, meta_keys in {
        'bert_model':       ('bert_model',),
        'gru_hidden':       ('gru_hidden',),
        'sentence_dropout': ('sentence_dropout',),
        'proj_dim':         ('proj_dim',),
    }.items():
        for mk in meta_keys:
            if mk in meta and meta[mk] is not None:
                setattr(args, cli_key, meta[mk])
                break
    print(f'Loaded bi-encoder config from {meta_path}: '
          f'bert_model={args.bert_model}, gru_hidden={args.gru_hidden}, '
          f'proj_dim={args.proj_dim}, sentence_dropout={args.sentence_dropout}')


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # ---- optional W&B run ----
    use_wandb = bool(args.wandb_project) and WANDB_AVAILABLE
    if args.wandb_project and not WANDB_AVAILABLE:
        print('wandb not installed — skipping W&B logging')
    if use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run, config={
            'pretrained_model':    args.pretrained_model,
            'bert_only_checkpoint': args.bert_only_checkpoint,
            'bi_checkpoint':       args.bi_checkpoint,
            'reranker_checkpoint': args.reranker_checkpoint,
            'reranker_model':      args.reranker_model,
            'rerank_top_k':        args.rerank_top_k,
            'k_ndcg':              args.k_ndcg,
            'k_recall':            args.k_recall,
            'queries':             args.queries,
        })
        print(f'W&B run: {wandb.run.url}')

    # ---- shared data lookups ----
    qrels   = load_qrels(args.qrels)
    queries = {q['id']: q for q in load_jsonl(args.queries)}
    docs    = {d['id']: d for d in load_jsonl(args.docs)}

    # ===================================================================
    # Stage 1 — Pretrained BERT, zero-shot (no fine-tuning)
    #   Off-the-shelf weights from the Hub. Raw [CLS], no projection head:
    #   query = mean of sentence [CLS]; doc = [CLS]. Floor baseline.
    # ===================================================================
    pt_query_loader = get_trec_dataloader(
        query_path=args.queries, doc_path=args.docs, qrel_path=args.qrels,
        batch_size=args.batch_size, tokenizer_name=args.pretrained_model,
        shuffle=False, distributed=False,
    )
    pt_doc_loader = get_doc_dataloader(
        args.docs, batch_size=args.eval_batch_size, tokenizer_name=args.pretrained_model)

    pt_encoder = _RawCLSDocEncoder(args.pretrained_model).to(device)
    print(f'Pretrained zero-shot model loaded: {args.pretrained_model} (raw [CLS], no fine-tuning)')

    pt_index, pt_idx_to_docid = build_index_cached(
        pt_encoder, pt_doc_loader, device,
        cache_key=f'pretrained_{args.pretrained_model}',
        cache_dir=args.cache_dir, docs_path=args.docs, refresh=args.refresh_cache)
    pretrained_run     = retrieve_pretrained(pt_encoder, pt_query_loader, pt_index,
                                             pt_idx_to_docid, device, k=args.k_ndcg,
                                             tag='pretrained')
    pretrained_results = score_runs(pretrained_run, qrels,
                                    k_ndcg=args.k_ndcg, k_recall=args.k_recall)
    print_block('Stage 1 — Pretrained BERT, zero-shot (no fine-tuning)',
                pretrained_results, args.k_ndcg, args.k_recall)

    del pt_encoder, pt_index
    torch.cuda.empty_cache()

    # ===================================================================
    # Stage 2 — BERT-only baseline checkpoint (tokenizer_comparison run)
    #   Loaded with tokenizer_eval's Generic encoders. NOTE: this checkpoint
    #   also contains a GRU; it is a *separately trained* baseline, not a
    #   GRU-free architecture. Each stage uses its OWN doc index because the
    #   projection space differs across checkpoints.
    # ===================================================================
    bo_query_loader = get_trec_dataloader(
        query_path=args.queries, doc_path=args.docs, qrel_path=args.qrels,
        batch_size=args.batch_size, tokenizer_name=args.bert_only_model,
        shuffle=False, distributed=False,
    )
    bo_doc_loader = get_doc_dataloader(
        args.docs, batch_size=args.eval_batch_size, tokenizer_name=args.bert_only_model)

    bo_query_enc = GenericQueryEncoder(
        args.bert_only_model, proj_dim=args.bert_only_proj_dim).to(device)
    bo_doc_enc   = GenericDocEncoder(
        args.bert_only_model, proj_dim=args.bert_only_proj_dim).to(device)
    bo_query_enc.load_state_dict(torch.load(
        os.path.join(args.bert_only_checkpoint, 'query_enc.pt'), map_location=device))
    bo_doc_enc.load_state_dict(torch.load(
        os.path.join(args.bert_only_checkpoint, 'doc_enc.pt'), map_location=device))
    print(f'BERT-only baseline loaded from {args.bert_only_checkpoint} '
          f'({args.bert_only_model}, proj_dim={args.bert_only_proj_dim})')

    bo_index, bo_idx_to_docid = build_index_cached(
        bo_doc_enc, bo_doc_loader, device,
        cache_key=args.bert_only_checkpoint,
        cache_dir=args.cache_dir, docs_path=args.docs, refresh=args.refresh_cache)
    bert_run     = retrieve_run(bo_query_enc, bo_query_loader, bo_index,
                                bo_idx_to_docid, device, k=args.k_ndcg, tag='bert_only')
    bert_results = score_runs(bert_run, qrels, k_ndcg=args.k_ndcg, k_recall=args.k_recall)
    print_block('Stage 2 — BERT-only baseline, fine-tuned (wordpiece_d512)',
                bert_results, args.k_ndcg, args.k_recall)
    print_delta('Δ Stage 2 - Stage 1 (fine-tuned vs zero-shot BERT):',
                bert_results, pretrained_results, args.k_ndcg, args.k_recall)

    del bo_query_enc, bo_doc_enc, bo_index
    torch.cuda.empty_cache()

    # ===================================================================
    # Stage 3 — main BERT + GRU bi-encoder (checkpoints/best)
    # ===================================================================
    _load_meta(args.bi_checkpoint, args)  # match trained hyperparameters

    query_enc = QueryEncoder(
        bert_model=args.bert_model,
        gru_hidden=args.gru_hidden,
        sentence_dropout=args.sentence_dropout,
        proj_dim=args.proj_dim,
    ).to(device)
    doc_enc = DocEncoder(bert_model=args.bert_model, proj_dim=args.proj_dim).to(device)
    query_enc.load_state_dict(
        torch.load(os.path.join(args.bi_checkpoint, 'query_enc.pt'), map_location=device))
    doc_enc.load_state_dict(
        torch.load(os.path.join(args.bi_checkpoint, 'doc_enc.pt'), map_location=device))
    print(f'Bi-encoder loaded from {args.bi_checkpoint}')

    query_loader = get_trec_dataloader(
        query_path=args.queries, doc_path=args.docs, qrel_path=args.qrels,
        batch_size=args.batch_size, tokenizer_name=args.bert_model,
        shuffle=False, distributed=False,
    )
    doc_loader = get_doc_dataloader(
        args.docs, batch_size=args.eval_batch_size, tokenizer_name=args.bert_model)

    index, idx_to_docid = build_index_cached(
        doc_enc, doc_loader, device,
        cache_key=args.bi_checkpoint,
        cache_dir=args.cache_dir, docs_path=args.docs, refresh=args.refresh_cache)
    gru_run     = retrieve_run(query_enc, query_loader, index, idx_to_docid,
                               device, k=args.k_ndcg, tag='gru')
    gru_results = score_runs(gru_run, qrels, k_ndcg=args.k_ndcg, k_recall=args.k_recall)
    print_block('Stage 3 — BERT + GRU + attention',
                gru_results, args.k_ndcg, args.k_recall)
    print_delta('Δ Stage 3 - Stage 2 (adding GRU/attention vs BERT-only baseline):',
                gru_results, bert_results, args.k_ndcg, args.k_recall)

    # free bi-encoder GPU memory before loading the LLM reranker
    del query_enc, doc_enc, index
    torch.cuda.empty_cache()

    # ---- Stage 4: BERT + GRU + LLM reranker ----
    reranker = LoRAReranker(
        model_name=args.reranker_model,
        load_in_4bit=args.load_in_4bit,
        torch_dtype=torch.float32 if not args.load_in_4bit else None,
        checkpoint_path=args.reranker_checkpoint,
    )
    rerank_run_d  = rerank_run(reranker, gru_run, queries, docs,
                               rerank_top_k=args.rerank_top_k,
                               batch_size=args.rerank_batch_size)
    rerank_results = score_runs(rerank_run_d, qrels,
                                k_ndcg=args.k_ndcg, k_recall=args.k_recall)
    print_block(f'Stage 4 — BERT + GRU + LLM reranker (top-{args.rerank_top_k})',
                rerank_results, args.k_ndcg, args.k_recall)
    print_delta('Δ from adding reranker (Stage 4 - Stage 3):',
                rerank_results, gru_results, args.k_ndcg, args.k_recall)

    # ---- stages (name, W&B prefix, results) ----
    stages = [
        ('Pretrained BERT (zero-shot)',      'pretrained', pretrained_results),
        ('BERT-only fine-tuned (wordpiece)', 'bert_only',  bert_results),
        ('BERT + GRU',                       'bert_gru',   gru_results),
        ('BERT + GRU + LLM reranker',        'reranked',   rerank_results),
    ]
    metric_keys = rerank_results['metric_keys']  # same set for every stage

    # ---- final summary table ----
    print('\n' + '=' * (42 + 11 * len(metric_keys)))
    print('SUMMARY')
    print('=' * (42 + 11 * len(metric_keys)))
    header = f'  {"stage":<40}' + ''.join(f'{m:>11}' for m in metric_keys)
    print(header)
    for name, _prefix, res in stages:
        row = f'  {name:<40}' + ''.join(f'{res[f"mean_{m}"]:>11.4f}' for m in metric_keys)
        print(row)
    print('=' * (42 + 11 * len(metric_keys)) + '\n')

    # ---- optional W&B logging ----
    if use_wandb:
        log = {}
        for _name, prefix, res in stages:
            for m in metric_keys:
                log[f'{prefix}/{m}'] = res[f'mean_{m}']
        # deltas between consecutive stages
        for (_, _, old), (label, prefix, new) in zip(stages[:-1], stages[1:]):
            for m in metric_keys:
                log[f'delta_{prefix}/{m}'] = new[f'mean_{m}'] - old[f'mean_{m}']
        wandb.log(log)

        # summary table as a W&B Table for side-by-side comparison
        table = wandb.Table(columns=['stage'] + metric_keys)
        for name, _prefix, res in stages:
            table.add_data(name, *[res[f'mean_{m}'] for m in metric_keys])
        wandb.log({'ablation_summary': table})
        wandb.finish()
        print('Results logged to W&B')

    # ---- optional JSON dump ----
    if args.output_scores:
        os.makedirs(os.path.dirname(args.output_scores) or '.', exist_ok=True)
        drop = ('per_query', 'metric_keys')
        with open(args.output_scores, 'w') as f:
            json.dump({
                **{prefix: {k: v for k, v in res.items() if k not in drop}
                   for _name, prefix, res in stages},
                'per_query': {prefix: res['per_query']
                              for _name, prefix, res in stages},
            }, f, indent=2)
        print(f'Per-query scores saved → {args.output_scores}')


# ---------------------------------------------------------------------------
# Args  (kept consistent with eval_reranker_pipeline.py)
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Ablation eval: BERT only vs BERT+GRU vs BERT+GRU+LLM reranker')

    # checkpoints
    p.add_argument('--pretrained-model',   default='bert-base-uncased',
                   help='HF model id for the zero-shot pretrained baseline (Stage 1)')
    p.add_argument('--bert-only-checkpoint',
                   default='checkpoints/tokenizer_comparison/wordpiece_d512',
                   help='Stage-1 baseline encoder dir (query_enc.pt / doc_enc.pt '
                        'from tokenizer_eval.py)')
    p.add_argument('--bert-only-model',    default='bert-base-uncased',
                   help='HF model id for the Stage-1 baseline encoder')
    p.add_argument('--bert-only-proj-dim', type=int, default=512,
                   help='proj_dim of the Stage-1 baseline checkpoint')
    p.add_argument('--bi-checkpoint',       default='checkpoints/best',
                   help='Directory with query_enc.pt / doc_enc.pt from train.py')
    p.add_argument('--reranker-checkpoint',
                   default='checkpoints/reranker_qwen0.5b_3epoch/epoch_1',
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
    p.add_argument('--output-scores', default=None,
                   help='Optional path to dump per-stage / per-query metrics as JSON')

    # W&B logging (off unless --wandb-project is given)
    p.add_argument('--wandb-project', default=None,
                   help='W&B project name; enables logging when set')
    p.add_argument('--wandb-run',     default='bert-gru-llm-ablation',
                   help='W&B run name')

    return p.parse_args()


if __name__ == '__main__':
    main(parse_args())
