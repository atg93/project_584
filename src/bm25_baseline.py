"""
BM25 baseline for TREC 2023 Tip-of-the-Tongue retrieval.

Runs a WandB Bayesian sweep over k1 and b, then reports the best
configuration evaluated with NDCG@1000 and R@100.

Usage
-----
# single run with fixed params
python bm25_baseline.py --docs data/trec/docs.jsonl \
    --queries data/trec/dev_queries.jsonl \
    --qrels   data/trec/dev_qrels.txt

# WandB sweep (n_trials Bayesian trials)
python bm25_baseline.py --docs data/trec/docs.jsonl \
    --queries data/trec/dev_queries.jsonl \
    --qrels   data/trec/dev_qrels.txt \
    --sweep --wandb-project tot-bm25-sweep --n-trials 30
"""

import os
import re
import math
import json
import argparse
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm

from dataset import load_jsonl, load_qrels
from evaluate import ndcg_at_k, recall_at_k

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def tokenize(text):
    """Lowercase, extract word tokens (strips punctuation)."""
    return re.findall(r'\b\w+\b', text.lower())


# ---------------------------------------------------------------------------
# BM25 index
# ---------------------------------------------------------------------------

class BM25:
    """
    Inverted-index BM25.  Build once, then call set_params() to change
    k1/b without rebuilding — used by the sweep loop.
    """

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1 = k1
        self.b  = b

        self.doc_ids = [d['id'] for d in docs]
        self.N       = len(docs)

        tokenized = []
        for doc in tqdm(docs, desc='Tokenizing corpus'):
            text = doc.get('title', '') + ' ' + doc.get('text', '')
            tokenized.append(tokenize(text))

        dl = [len(t) for t in tokenized]
        self.avgdl = sum(dl) / self.N if self.N else 1.0
        self.dl    = dl

        # inverted index: term -> [(doc_idx, tf), ...]
        self.inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
        df: dict[str, int] = defaultdict(int)

        for idx, tokens in enumerate(tokenized):
            for term, freq in Counter(tokens).items():
                self.inverted[term].append((idx, freq))
                df[term] += 1

        # Robertson-Sparck Jones IDF with smoothing
        self.idf: dict[str, float] = {
            term: math.log((self.N - n + 0.5) / (n + 0.5) + 1.0)
            for term, n in df.items()
        }

    # ------------------------------------------------------------------

    def set_params(self, k1: float, b: float) -> None:
        self.k1 = k1
        self.b  = b

    def retrieve(self, query_text: str, topk: int = 1000) -> list[str]:
        """Return ranked list of doc_ids (length <= topk)."""
        tokens = tokenize(query_text)
        scores: dict[int, float] = defaultdict(float)
        k1, b, avgdl = self.k1, self.b, self.avgdl

        for term in set(tokens):
            if term not in self.inverted:
                continue
            idf = self.idf[term]
            for doc_idx, tf in self.inverted[term]:
                dl    = self.dl[doc_idx]
                denom = tf + k1 * (1.0 - b + b * dl / avgdl)
                scores[doc_idx] += idf * tf * (k1 + 1.0) / denom

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:topk]
        return [self.doc_ids[i] for i, _ in ranked]


# ---------------------------------------------------------------------------
# Query text builder
# ---------------------------------------------------------------------------

def query_to_text(query: dict) -> str:
    """
    Concatenate sentence-level clues into a single BM25 query string.
    Falls back to the free-text 'text' field if sentences are absent.
    """
    sentences = query.get('sentences') or []
    if sentences:
        return ' '.join(sentences)
    return query.get('text', '')


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_bm25(bm25: BM25, queries: list[dict], qrels: dict,
                  k_ndcg: int = 1000, k_recall: int = 100) -> dict:
    ndcg_scores:   list[float] = []
    recall_scores: list[float] = []
    per_query:     dict        = {}

    for q in tqdm(queries, desc=f'BM25 (k1={bm25.k1:.2f}, b={bm25.b:.2f})'):
        qid        = q['id']
        query_text = query_to_text(q)
        ranked_ids = bm25.retrieve(query_text, topk=k_ndcg)
        rel        = qrels.get(qid, [])

        ndcg   = ndcg_at_k(ranked_ids, rel, k=k_ndcg)
        recall = recall_at_k(ranked_ids, rel, k=k_recall)

        ndcg_scores.append(ndcg)
        recall_scores.append(recall)
        per_query[qid] = {f'ndcg@{k_ndcg}': ndcg, f'r@{k_recall}': recall}

    return {
        f'mean_ndcg@{k_ndcg}': float(np.mean(ndcg_scores)),
        f'mean_r@{k_recall}':   float(np.mean(recall_scores)),
        'k1':       bm25.k1,
        'b':        bm25.b,
        'per_query': per_query,
    }


def print_results(results: dict) -> None:
    print('\n' + '=' * 45)
    print('BM25 Evaluation Results')
    print('=' * 45)
    for k, v in results.items():
        if k == 'per_query':
            continue
        print(f'  {k:<25} {v:.4f}' if isinstance(v, float) else f'  {k:<25} {v}')
    print('=' * 45 + '\n')


def save_scores(results: dict, path: str) -> None:
    """Save per-query scores to JSON for downstream significance testing."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(results['per_query'], f, indent=2)
    print(f'Per-query scores saved → {path}')


# ---------------------------------------------------------------------------
# WandB sweep
# ---------------------------------------------------------------------------

SWEEP_CONFIG = {
    'method': 'bayes',
    'metric': {'name': 'mean_ndcg@1000', 'goal': 'maximize'},
    'parameters': {
        'k1': {'distribution': 'uniform', 'min': 0.5,  'max': 3.0},
        'b':  {'distribution': 'uniform', 'min': 0.0,  'max': 1.0},
    },
}


def build_sweep_fn(bm25: BM25, queries: list[dict], qrels: dict):
    """Return a zero-argument callable for wandb.agent()."""
    def sweep_trial():
        with wandb.init():
            cfg = wandb.config
            bm25.set_params(k1=cfg.k1, b=cfg.b)
            results = evaluate_bm25(bm25, queries, qrels)
            wandb.log(results)
    return sweep_trial


def run_sweep(bm25: BM25, queries: list[dict], qrels: dict,
              project: str, n_trials: int = 30) -> None:
    if not WANDB_AVAILABLE:
        raise RuntimeError('wandb is not installed — pip install wandb')

    sweep_id = wandb.sweep(SWEEP_CONFIG, project=project)
    print(f'WandB sweep created: {sweep_id}  ({n_trials} trials)')

    sweep_fn = build_sweep_fn(bm25, queries, qrels)
    wandb.agent(sweep_id, function=sweep_fn, count=n_trials)


# ---------------------------------------------------------------------------
# TREC run file
# ---------------------------------------------------------------------------

def save_trec_run(bm25: BM25, queries: list[dict],
                  output_path: str, run_name: str = 'bm25', topk: int = 1000) -> None:
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w') as f:
        for q in tqdm(queries, desc='Writing TREC run'):
            qid        = q['id']
            query_text = query_to_text(q)
            ranked_ids = bm25.retrieve(query_text, topk=topk)
            for rank, doc_id in enumerate(ranked_ids, start=1):
                score = topk - rank  # monotone proxy score for trec_eval
                f.write(f'{qid} Q0 {doc_id} {rank} {score} {run_name}\n')
    print(f'TREC run saved → {output_path}')


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='BM25 baseline for ToT retrieval')

    p.add_argument('--docs',          default='data/trec/docs.jsonl')
    p.add_argument('--queries',       default='data/trec/dev_queries.jsonl')
    p.add_argument('--qrels',         default='data/trec/dev_qrels.txt')

    # BM25 params (used when not sweeping)
    p.add_argument('--k1',            type=float, default=1.5)
    p.add_argument('--b',             type=float, default=0.75)

    # WandB grid sweep (recommended) — logs each config as a separate run
    p.add_argument('--grid-sweep',    action='store_true',
                   help='Run a grid sweep over --k1-values and --b-values')
    p.add_argument('--k1-values',     nargs='+', type=float,
                   default=[0.5, 1.0, 1.5, 2.0, 2.5],
                   help='k1 values to try in the grid sweep')
    p.add_argument('--b-values',      nargs='+', type=float,
                   default=[0.25, 0.5, 0.75, 1.0],
                   help='b values to try in the grid sweep')
    p.add_argument('--wandb-project', default='tot-bm25-sweep')

    # WandB Bayesian sweep (alternative)
    p.add_argument('--sweep',         action='store_true',
                   help='Run a WandB Bayesian sweep over k1 and b')
    p.add_argument('--n-trials',      type=int,   default=30,
                   help='Number of Bayesian sweep trials')

    # output
    p.add_argument('--trec-run',      default=None,
                   help='Path to save TREC run file (e.g. runs/bm25.txt)')
    p.add_argument('--save-scores',   default=None,
                   help='Path to save per-query scores JSON (e.g. scores/bm25.json)')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()

    docs    = load_jsonl(args.docs)
    queries = load_jsonl(args.queries)
    qrels   = load_qrels(args.qrels)

    # keep only queries that have at least one relevant document
    queries = [q for q in queries if q['id'] in qrels]
    print(f'Corpus: {len(docs)} docs  |  Queries: {len(queries)}')

    bm25 = BM25(docs, k1=args.k1, b=args.b)

    if args.grid_sweep:
        import itertools
        if not WANDB_AVAILABLE:
            raise RuntimeError('wandb is not installed — pip install wandb')

        grid = list(itertools.product(args.k1_values, args.b_values))
        print(f'\nGrid sweep: {len(grid)} configurations '
              f'(k1={args.k1_values}, b={args.b_values})\n')

        best_ndcg, best_cfg = 0.0, None
        for k1, b in grid:
            bm25.set_params(k1=k1, b=b)
            results = evaluate_bm25(bm25, queries, qrels)
            print_results(results)

            wandb.init(
                project=args.wandb_project,
                name=f'bm25_k1={k1}_b={b}',
                config={'k1': k1, 'b': b},
                reinit=True,
            )
            flat = {k: v for k, v in results.items() if k != 'per_query'}
            wandb.log(flat)
            wandb.finish()

            if results['mean_ndcg@1000'] > best_ndcg:
                best_ndcg = results['mean_ndcg@1000']
                best_cfg  = (k1, b)
                if args.save_scores:
                    save_scores(results, args.save_scores)

        print(f'\nBest config: k1={best_cfg[0]}, b={best_cfg[1]}  '
              f'NDCG@1000={best_ndcg:.4f}')

    elif args.sweep:
        run_sweep(bm25, queries, qrels,
                  project=args.wandb_project,
                  n_trials=args.n_trials)
    else:
        results = evaluate_bm25(bm25, queries, qrels)
        print_results(results)

        if WANDB_AVAILABLE:
            wandb.init(project=args.wandb_project,
                       name=f'bm25_k1={args.k1}_b={args.b}',
                       config={'k1': args.k1, 'b': args.b})
            flat = {k: v for k, v in results.items() if k != 'per_query'}
            wandb.log(flat)
            wandb.finish()

        if args.save_scores:
            save_scores(results, args.save_scores)

    if args.trec_run:
        bm25.set_params(k1=args.k1, b=args.b)
        save_trec_run(bm25, queries, output_path=args.trec_run)
