"""
Statistical significance testing for the BERT+GRU+Qwen2.5-0.5B retrieval pipeline.

Satisfies the "Statistical Testing & Benchmarking" requirement: a rigorous,
paired comparison of the tuned neural retriever against its baselines using two
significance tests on per-query NDCG@1000.

What it compares (defaults)
---------------------------
    reranked  (BERT+GRU+Qwen, final)   vs   bert_gru (dense, BERT+GRU)
    reranked  (BERT+GRU+Qwen, final)   vs   bm25     (lexical baseline)

Why these tests
---------------
Both systems are scored on the SAME dev queries, so the data is *paired*. ToT
NDCG is far from normal (dominated by 0.0 and a few 1.0), which violates the
paired t-test's normality assumption. We therefore use:

  * Wilcoxon signed-rank test — non-parametric paired test on the ranks of the
    per-query score differences (scipy if available, else a numpy normal
    approximation with tie + continuity correction).

  * Randomised permutation (sign-flip) test — under the null "the reranker has
    no systematic effect", the sign of each per-query difference is exchangeable.
    We flip signs at random B times and compare the resulting mean differences
    to the observed one. Pure numpy; no third-party dependency.

A note on this particular reranker
----------------------------------
The reranker only reorders the dense top-100, so most queries have a difference
of exactly 0. Wilcoxon drops those zero differences automatically, so the test
is effectively run on the queries the reranker actually moved — the honest unit
of analysis. The permutation test keeps all queries (zeros contribute nothing
under sign-flips), so the two are consistent.

Score-file formats accepted
---------------------------
  * nested  (eval_ablation_pipeline.py / eval_reranker_pipeline.py --output-scores):
        {"per_query": {<stage>: {qid: {metric: value}}}}
  * flat    (bm25.json, dense.json):
        {qid: {metric: value}}

Usage
-----
    python significance_test.py \\
        --scores      scores/ablation.json \\
        --final-stage reranked \\
        --dense-stage bert_gru \\
        --bm25        scores/bm25_best.json \\
        --metric      ndcg@1000 \\
        --permutations 10000 \\
        --out         scores/significance.json

If you only have scores/reranker_pipeline.json, pass --dense-stage dense
(the script also auto-falls back across {bert_gru, dense}).
"""

import os
import json
import math
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Loading per-query scores (nested-by-stage OR flat)
# ---------------------------------------------------------------------------

def _stage_per_query(blob, stage_prefs):
    """Return the {qid: {metric: value}} for the first stage name in stage_prefs
    that exists. Handles both the nested {"per_query": {stage: ...}} layout and
    a flat {qid: {...}} file (treated as already being the requested stage)."""
    if isinstance(blob, dict) and 'per_query' in blob and isinstance(blob['per_query'], dict):
        pq = blob['per_query']
        for s in stage_prefs:
            if s in pq:
                return pq[s], s
        raise KeyError(f'none of {stage_prefs} in per_query stages {list(pq.keys())}')
    # flat file
    return blob, '(flat)'


def load_system(path, stage_prefs, metric):
    """Load {qid: float} for one system/stage and one metric."""
    if not os.path.exists(path):
        raise SystemExit(f'scores file not found: {path}')
    with open(path) as f:
        blob = json.load(f)
    per_q, resolved = _stage_per_query(blob, stage_prefs)
    scores = {qid: float(d.get(metric, 0.0)) for qid, d in per_q.items()}
    return scores, resolved


def paired_arrays(treat, base):
    """Align two {qid: value} dicts on shared queries (sorted for determinism)."""
    shared = sorted(set(treat) & set(base), key=lambda x: (len(x), x))
    a = np.array([treat[q] for q in shared], dtype=float)
    b = np.array([base[q]  for q in shared], dtype=float)
    return a, b, shared


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank test
# ---------------------------------------------------------------------------

def _phi(z):
    """Standard-normal CDF via erf (no scipy needed)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def wilcoxon_test(a, b):
    """
    Two-sided Wilcoxon signed-rank test on paired samples a vs b.
    Returns dict with statistic (W+), p-value, n_effective (non-zero diffs),
    and the backend used. Prefers scipy.stats.wilcoxon; falls back to a numpy
    normal approximation with average-rank tie handling + continuity correction.
    """
    d = a - b
    nz = d[d != 0.0]
    n = nz.size
    if n == 0:
        return {'statistic': 0.0, 'p_value': 1.0, 'n_effective': 0,
                'backend': 'none (all differences zero)'}

    # ---- exact / scipy path ----
    try:
        from scipy.stats import wilcoxon as _w
        res = _w(a, b, alternative='two-sided', zero_method='wilcox')
        return {'statistic': float(res.statistic), 'p_value': float(res.pvalue),
                'n_effective': int(n), 'backend': 'scipy'}
    except ImportError:
        pass

    # ---- numpy normal approximation ----
    order = np.argsort(np.abs(nz), kind='mergesort')
    absd  = np.abs(nz)[order]
    signs = np.sign(nz)[order]
    # average ranks for ties in |d|
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and absd[j] == absd[i]:
            j += 1
        ranks[i:j] = (i + 1 + j) / 2.0   # average of ranks (i+1 .. j)
        i = j
    w_plus  = float(ranks[signs > 0].sum())
    w_minus = float(ranks[signs < 0].sum())
    W = min(w_plus, w_minus)

    mean_w = n * (n + 1) / 4.0
    # tie correction term
    _, counts = np.unique(absd, return_counts=True)
    tie_term = (counts ** 3 - counts).sum()
    var_w = (n * (n + 1) * (2 * n + 1) - tie_term / 2.0) / 24.0
    if var_w <= 0:
        return {'statistic': w_plus, 'p_value': 1.0, 'n_effective': int(n),
                'backend': 'numpy-normal (degenerate variance)'}
    z = (W - mean_w + 0.5) / math.sqrt(var_w)     # continuity correction
    p = 2.0 * _phi(-abs(z))
    return {'statistic': w_plus, 'p_value': min(1.0, p), 'n_effective': int(n),
            'backend': 'numpy-normal-approx'}


# ---------------------------------------------------------------------------
# Randomised permutation (sign-flip) test
# ---------------------------------------------------------------------------

def permutation_test(a, b, n_perm=10000, seed=42):
    """
    Two-sided paired permutation test. Statistic = mean(a - b). Under the null,
    each per-query difference's sign is exchangeable; we flip signs at random and
    build the null distribution of the mean difference.

    p = (1 + #{|perm_mean| >= |obs_mean|}) / (n_perm + 1)   (add-one smoothing).
    """
    d = a - b
    obs = float(d.mean())
    if np.allclose(d, 0.0):
        return {'observed_mean_diff': 0.0, 'p_value': 1.0, 'n_perm': int(n_perm)}
    rng = np.random.default_rng(seed)
    n = d.size
    # vectorised: (n_perm, n) random ±1 sign matrix
    flips = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    perm_means = (flips * d).mean(axis=1)
    count = int(np.sum(np.abs(perm_means) >= abs(obs) - 1e-12))
    p = (1 + count) / (n_perm + 1)
    return {'observed_mean_diff': obs, 'p_value': float(p), 'n_perm': int(n_perm)}


# ---------------------------------------------------------------------------
# One comparison + pretty print
# ---------------------------------------------------------------------------

def compare(treat_name, base_name, treat, base, metric, n_perm, seed, alpha):
    a, b, shared = paired_arrays(treat, base)
    n = len(shared)
    d = a - b
    wins   = int((d > 1e-12).sum())
    losses = int((d < -1e-12).sum())
    ties   = n - wins - losses

    wil = wilcoxon_test(a, b)
    per = permutation_test(a, b, n_perm=n_perm, seed=seed)

    sig_w = wil['p_value'] < alpha
    sig_p = per['p_value'] < alpha

    print('\n' + '=' * 62)
    print(f'{treat_name}  vs  {base_name}      [metric: {metric}]')
    print('=' * 62)
    print(f'  paired queries        {n}')
    print(f'  mean {metric:<14} {a.mean():.4f}   (vs {b.mean():.4f})')
    print(f'  mean difference       {d.mean():+.4f}')
    print(f'  median difference     {np.median(d):+.4f}')
    print(f'  wins / losses / ties  {wins} / {losses} / {ties}')
    print(f'  --- Wilcoxon signed-rank ({wil["backend"]}) ---')
    print(f'    effective n (≠0)    {wil["n_effective"]}')
    print(f'    statistic (W+)      {wil["statistic"]:.1f}')
    print(f'    p-value             {wil["p_value"]:.2e}'
          f'   -> {"SIGNIFICANT" if sig_w else "not significant"} at α={alpha}')
    print(f'  --- Randomised permutation ({per["n_perm"]} resamples) ---')
    print(f'    observed mean diff  {per["observed_mean_diff"]:+.4f}')
    print(f'    p-value             {per["p_value"]:.2e}'
          f'   -> {"SIGNIFICANT" if sig_p else "not significant"} at α={alpha}')

    return {
        'treatment': treat_name, 'baseline': base_name, 'metric': metric,
        'n_paired': n, 'mean_treatment': float(a.mean()), 'mean_baseline': float(b.mean()),
        'mean_diff': float(d.mean()), 'median_diff': float(np.median(d)),
        'wins': wins, 'losses': losses, 'ties': ties,
        'wilcoxon': wil, 'permutation': per,
        'significant_wilcoxon': bool(sig_w), 'significant_permutation': bool(sig_p),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scores', default='scores/ablation.json',
                   help='nested per-stage scores JSON holding the final + dense stages')
    p.add_argument('--final-stage', default='reranked',
                   help='stage key for the final BERT+GRU+Qwen model')
    p.add_argument('--dense-stage', default='bert_gru',
                   help='stage key for the dense BERT+GRU baseline '
                        '(use "dense" for reranker_pipeline.json)')
    p.add_argument('--bm25', default='scores/bm25_best.json',
                   help='flat per-query scores JSON for the BM25 baseline')
    p.add_argument('--metric', default='ndcg@1000',
                   help='comma-separated metric keys, e.g. "ndcg@1000,r@100"')
    p.add_argument('--permutations', type=int, default=10000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--out', default=None, help='optional path to dump results JSON')
    args = p.parse_args()

    metrics = [m.strip() for m in args.metric.split(',') if m.strip()]
    # tolerate either naming for the dense stage
    dense_prefs = [args.dense_stage, 'bert_gru', 'dense']
    final_prefs = [args.final_stage, 'reranked']

    all_results = []
    for metric in metrics:
        treat, t_stage = load_system(args.scores, final_prefs, metric)
        dense, d_stage = load_system(args.scores, dense_prefs, metric)
        print(f'\n#### metric={metric}  '
              f'(final stage="{t_stage}", dense stage="{d_stage}") ####')

        all_results.append(compare(
            f'BERT+GRU+Qwen [{t_stage}]', f'BERT+GRU [{d_stage}]',
            treat, dense, metric, args.permutations, args.seed, args.alpha))

        if args.bm25 and os.path.exists(args.bm25):
            bm25, _ = load_system(args.bm25, ['(flat)'], metric)
            all_results.append(compare(
                f'BERT+GRU+Qwen [{t_stage}]', 'BM25',
                treat, bm25, metric, args.permutations, args.seed, args.alpha))
        else:
            print(f'\n(skipping BM25 comparison — {args.bm25} not found)')

    print('\n' + '=' * 62)
    print('SUMMARY (✓ = significant at α = %.2g on BOTH tests)' % args.alpha)
    print('=' * 62)
    for r in all_results:
        both = r['significant_wilcoxon'] and r['significant_permutation']
        print(f'  [{"✓" if both else " "}] {r["treatment"]} vs {r["baseline"]} '
              f'({r["metric"]}): Δ={r["mean_diff"]:+.4f}, '
              f'W p={r["wilcoxon"]["p_value"]:.1e}, perm p={r["permutation"]["p_value"]:.1e}')

    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'alpha': args.alpha, 'seed': args.seed,
                       'permutations': args.permutations, 'comparisons': all_results},
                      f, indent=2)
        print(f'\nResults written → {args.out}')


if __name__ == '__main__':
    main()
