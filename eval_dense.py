"""
Dense bi-encoder evaluation for TREC 2023 Tip-of-the-Tongue retrieval.

Loads a saved QueryEncoder + DocEncoder checkpoint and evaluates with
NDCG@1000 and R@100 — the same metrics reported by bm25_baseline.py.

Usage
-----
singularity exec --nv \\
    --bind /home/tg22/.cache/hf_cache:/opt/hf_cache \\
    --bind /datasets \\
    /home/tg22/containers/584.sig \\
    python eval_dense.py \\
        --checkpoint checkpoints_v0/best \\
        --docs       data/trec/docs.jsonl \\
        --queries    data/trec/dev_queries.jsonl \\
        --qrels      data/trec/dev_qrels.txt \\
        --trec-run   runs/dense_best.txt
"""

import os
import json
import argparse
import numpy as np
import torch

from dataset import get_trec_dataloader, get_reddit_dataloader, get_doc_dataloader, load_qrels
from tot_retrieval import QueryEncoder, DocEncoder
from evaluate import evaluate, save_trec_run, save_candidates, print_results


# ---------------------------------------------------------------------------
# Per-query score helpers
# ---------------------------------------------------------------------------

def save_scores(per_query: dict, path: str) -> None:
    """Save per-query scores to JSON for downstream significance testing."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(per_query, f, indent=2)
    print(f'Per-query scores saved → {path}')


def significance_test(dense_per_query: dict, baseline_path: str,
                      metric: str = 'ndcg@1000') -> None:
    """
    Wilcoxon signed-rank test comparing dense model vs a saved baseline
    (e.g. BM25) on per-query NDCG@1000.

    Paired on the intersection of query IDs present in both score files.
    """
    with open(baseline_path) as f:
        baseline_per_query = json.load(f)

    shared_qids = sorted(set(dense_per_query) & set(baseline_per_query))
    if len(shared_qids) < 2:
        print('Not enough shared queries for significance test.')
        return

    dense_scores    = [dense_per_query[q][metric]    for q in shared_qids]
    baseline_scores = [baseline_per_query[q][metric] for q in shared_qids]

    try:
        from scipy.stats import wilcoxon as _wilcoxon
    except ImportError:
        print('scipy not installed — run: pip install scipy')
        return

    stat, p = _wilcoxon(dense_scores, baseline_scores, alternative='two-sided')

    dense_mean    = float(np.mean(dense_scores))
    baseline_mean = float(np.mean(baseline_scores))

    print('\n' + '=' * 45)
    print(f'Wilcoxon signed-rank test  ({metric})')
    print('=' * 45)
    print(f'  Queries compared   {len(shared_qids)}')
    print(f'  Dense mean         {dense_mean:.4f}')
    print(f'  Baseline mean      {baseline_mean:.4f}')
    print(f'  Δ (dense−baseline) {dense_mean - baseline_mean:+.4f}')
    print(f'  statistic          {stat:.4f}')
    print(f'  p-value            {p:.4f}')
    print(f'  {"** SIGNIFICANT **" if p < 0.05 else "not significant"} at α = 0.05')
    print('=' * 45 + '\n')


# ---------------------------------------------------------------------------
# Checkpoint loader
# ---------------------------------------------------------------------------

def _torch_load(path, device):
    """Load a .pt file saved by either modern (zip) or legacy (pickle) torch.save."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except RuntimeError:
        # legacy pre-1.6 format: needs PyTorch's own unpickler for persistent tensor IDs
        import pickle
        import torch.serialization as ts
        with open(path, 'rb') as f:
            return ts._legacy_load(f, map_location=device, pickle_module=pickle)


def load_checkpoint(checkpoint_dir, query_enc, doc_enc, device):
    q_path = os.path.join(checkpoint_dir, 'query_enc.pt')
    d_path = os.path.join(checkpoint_dir, 'doc_enc.pt')
    m_path = os.path.join(checkpoint_dir, 'meta.json')

    query_enc.load_state_dict(_torch_load(q_path, device))
    doc_enc.load_state_dict(  _torch_load(d_path, device))

    meta = {}
    if os.path.exists(m_path):
        with open(m_path) as f:
            meta = json.load(f)
        print(f'Loaded checkpoint  epoch={meta.get("epoch")}  '
              f'saved NDCG@1000={meta.get("ndcg@1000", "?"):.4f}')
    else:
        print(f'Loaded checkpoint from {checkpoint_dir}')

    return meta


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Evaluate dense bi-encoder on ToT retrieval')

    p.add_argument('--checkpoint',      default='checkpoints_v0/best',
                   help='Directory containing query_enc.pt, doc_enc.pt, meta.json')

    # data
    p.add_argument('--docs',            default='data/trec/docs.jsonl')
    p.add_argument('--queries',         default='data/trec/dev_queries.jsonl')
    p.add_argument('--qrels',           default='data/trec/dev_qrels.txt')

    # model architecture — must match the trained checkpoint
    p.add_argument('--bert-model',      default='bert-base-uncased')
    p.add_argument('--proj-dim',        type=int,   default=512)
    p.add_argument('--gru-hidden',      type=int,   default=None,
                   help='GRU hidden size (default: same as BERT hidden size, i.e. 768)')
    p.add_argument('--sentence-dropout', type=float, default=0.0,
                   help='Set to 0 for evaluation (no dropout at inference)')

    # eval settings
    p.add_argument('--batch-size',      type=int,   default=16)
    p.add_argument('--eval-batch-size', type=int,   default=256)

    # output
    p.add_argument('--trec-run',        default=None,
                   help='Path to save TREC run file (e.g. runs/dense_best.txt)')
    p.add_argument('--run-name',        default='bert_gru_attn',
                   help='Run tag written into the TREC run file')
    p.add_argument('--save-scores',     default=None,
                   help='Path to save per-query scores JSON (e.g. scores/dense.json)')
    p.add_argument('--compare-scores',  default=None,
                   help='Path to baseline per-query scores JSON for Wilcoxon test '
                        '(e.g. scores/bm25.json produced by bm25_baseline.py --save-scores)')
    p.add_argument('--save-candidates', default=None,
                   help='Path to save top-1000 candidates jsonl for reranker training '
                        '(e.g. runs/bi_encoder_train_candidates.jsonl)')
    p.add_argument('--reddit-queries',  default=None,
                   help='Reddit ToT queries jsonl to include in candidate generation')
    p.add_argument('--reddit-domain',   default=None,
                   help='Filter Reddit queries to a specific domain (movie, book, etc.)')

    # optional wandb logging
    p.add_argument('--wandb-project',   default=None)
    p.add_argument('--wandb-run',       default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # --- models ---
    query_enc = QueryEncoder(
        bert_model=args.bert_model,
        gru_hidden=args.gru_hidden,
        sentence_dropout=args.sentence_dropout,
        proj_dim=args.proj_dim,
    ).to(device)

    doc_enc = DocEncoder(
        bert_model=args.bert_model,
        proj_dim=args.proj_dim,
    ).to(device)

    load_checkpoint(args.checkpoint, query_enc, doc_enc, device)

    # --- data ---
    qrels = load_qrels(args.qrels)

    query_loader = get_trec_dataloader(
        query_path=args.queries,
        doc_path=args.docs,
        qrel_path=args.qrels,
        batch_size=args.batch_size,
        tokenizer_name=args.bert_model,
        shuffle=False,
    )

    doc_loader = get_doc_dataloader(
        doc_path=args.docs,
        batch_size=args.eval_batch_size,
        tokenizer_name=args.bert_model,
    )

    print(f'Queries : {len(query_loader.dataset)}')
    print(f'Corpus  : {len(doc_loader.dataset)} docs\n')

    # --- evaluate ---
    results = evaluate(query_enc, doc_enc, query_loader, doc_loader, qrels, device)
    print_results(results)

    # --- per-query score export ---
    if args.save_scores:
        save_scores(results['per_query'], args.save_scores)

    # --- Wilcoxon significance test vs baseline ---
    if args.compare_scores:
        significance_test(results['per_query'], args.compare_scores, metric='ndcg@1000')

    # --- optional wandb logging ---
    try:
        import wandb
        if args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run or f'dense_eval_{os.path.basename(args.checkpoint)}',
                config=vars(args),
            )
            flat = {k: v for k, v in results.items() if k != 'per_query'}
            wandb.log(flat)
            wandb.finish()
    except ImportError:
        pass

    # --- reranker candidates ---
    if args.save_candidates:
        index, idx_to_docid = save_candidates(
            query_enc, doc_enc, query_loader, doc_loader,
            device, output_path=args.save_candidates,
        )
        if args.reddit_queries:
            reddit_loader = get_reddit_dataloader(
                query_path=args.reddit_queries,
                doc_path=args.docs,
                batch_size=args.batch_size,
                tokenizer_name=args.bert_model,
                shuffle=False,
                domain=args.reddit_domain,
            )
            print(f'Reddit queries loaded: {len(reddit_loader.dataset)}'
                  + (f' (domain={args.reddit_domain})' if args.reddit_domain else ''))
            save_candidates(
                query_enc, doc_enc, reddit_loader, doc_loader,
                device, output_path=args.save_candidates,
                index=index, idx_to_docid=idx_to_docid,
                append=True,
            )

    # --- TREC run file ---
    if args.trec_run:
        os.makedirs(os.path.dirname(args.trec_run) or '.', exist_ok=True)
        save_trec_run(
            query_enc, doc_enc, query_loader, doc_loader,
            device, output_path=args.trec_run, run_name=args.run_name,
        )
