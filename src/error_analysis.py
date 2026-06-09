"""
Error analysis — systematic isolation of failure modes.

Standalone script. Does NOT modify any existing file. Reads per-query
NDCG/R@100 dumps from scores/*.json and the raw dev queries / qrels to
identify the worst-performing queries and classify them into failure
modes used in the final report.

What it produces
----------------
  reports/error_analysis.csv      — full per-query table with metadata
                                    (n_sentences, n_words, n_proper_nouns,
                                    hedge_count, ndcg, r@100, category)
  reports/error_analysis_top.md   — Markdown table of the K worst queries
                                    with their categorised failure mode,
                                    one verbatim sample per category.
  reports/error_analysis_summary.json — counts per category, ready for
                                         the report's Discussion section.

Failure-mode taxonomy (heuristic, transparent)
----------------------------------------------
  sparse_query        n_sentences <= 3  OR  n_words < 25
  strong_false_memory ANY hedge phrase  (I think / I'm not sure / maybe /
                                         could be / pretty sure)  AND
                                         a specific claim (proper-noun
                                         heavy sentence containing a year
                                         or full name).
  out_of_distribution none of the above and per-query R@1000 == 0
                       (target was never retrieved at all).
  ranking_error       target was retrieved (R@1000 > 0) but missed the
                       top-K rerank pool — pure ordering failure.

These are heuristics, not ground truth. They give an objective starting
point that the report's Discussion section then qualifies with examples.

Usage
-----
    python error_analysis.py \\
        --scores         scores/dense.json \\
        --queries        data/trec/dev_queries.jsonl \\
        --qrels          data/trec/dev_qrels.txt \\
        --pipeline-scores scores/pipeline.json \\
        --top-k          15
"""

import os
import re
import csv
import json
import argparse
from collections import Counter, defaultdict


HEDGE_PATTERNS = [
    r"\bi\s*think\b",
    r"\bi'?m\s+not\s+sure\b",
    r"\bnot\s+sure\b",
    r"\bmaybe\b",
    r"\bperhaps\b",
    r"\bcould\s+be\b",
    r"\bmight\s+be\b",
    r"\bpretty\s+sure\b",
    r"\bif\s+i\s+remember\b",
    r"\bcan'?t\s+remember\b",
]

YEAR_RE        = re.compile(r"\b(19|20)\d{2}\b")
PROPER_NOUN_RE = re.compile(r"\b[A-Z][a-zA-Z\-]+\b")


def _load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            qid, did, rel = parts[0], parts[2], int(parts[3])
            if rel > 0:
                qrels.setdefault(qid, []).append(did)
    return qrels


def _full_text(query):
    if 'sentences' in query and query['sentences']:
        return ' '.join(query['sentences'])
    return query.get('text', '') or ''


def _query_features(query):
    text       = _full_text(query)
    sentences  = query.get('sentences') or [s for s in text.split('.') if s.strip()]
    words      = text.split()
    hedge_hits = sum(1 for p in HEDGE_PATTERNS if re.search(p, text, re.IGNORECASE))
    return {
        'n_sentences':     len(sentences),
        'n_words':         len(words),
        'n_proper_nouns':  len(PROPER_NOUN_RE.findall(text)),
        'has_year':        bool(YEAR_RE.search(text)),
        'hedge_count':     hedge_hits,
    }


def _categorise(feat, ndcg, r100, pipeline_r100=None):
    """
    Returns one of:
      sparse_query | strong_false_memory | out_of_distribution |
      ranking_error | other_failure | success
    """
    if ndcg >= 0.05:                                # roughly "found something usable"
        return 'success'
    # 1. sparse
    if feat['n_sentences'] <= 3 or feat['n_words'] < 25:
        return 'sparse_query'
    # 2. strong false memory — hedge + specific claim
    has_specific_claim = feat['has_year'] or feat['n_proper_nouns'] >= 3
    if feat['hedge_count'] >= 1 and has_specific_claim:
        return 'strong_false_memory'
    # 3. OOD vs ranking error — needs a wider-pool r@1000 signal to disambiguate
    if r100 == 0.0 and (pipeline_r100 is None or pipeline_r100 == 0.0):
        return 'out_of_distribution'
    if r100 == 0.0:
        return 'ranking_error'
    return 'other_failure'


def analyse(args):
    print(f'[1/3] loading scores from {args.scores} ...')
    with open(args.scores) as f:
        scores = json.load(f)

    pipeline_scores = {}
    if args.pipeline_scores and os.path.exists(args.pipeline_scores):
        with open(args.pipeline_scores) as f:
            pipeline_scores = json.load(f)

    print(f'[2/3] loading dev queries / qrels ...')
    queries = {q['id']: q for q in _load_jsonl(args.queries)}
    qrels   = _load_qrels(args.qrels)

    rows = []
    for qid, sc in scores.items():
        query = queries.get(qid)
        if query is None:
            continue
        feat = _query_features(query)
        ndcg = float(sc.get('ndcg@1000', 0.0))
        r100 = float(sc.get('r@100',     0.0))
        pipeline_r100 = float(pipeline_scores.get(qid, {}).get('r@100', r100)) \
                          if pipeline_scores else None

        category = _categorise(feat, ndcg, r100, pipeline_r100)
        rows.append({
            'qid':          qid,
            'n_sentences':  feat['n_sentences'],
            'n_words':      feat['n_words'],
            'n_proper_nouns': feat['n_proper_nouns'],
            'has_year':     int(feat['has_year']),
            'hedge_count':  feat['hedge_count'],
            'ndcg@1000':    f'{ndcg:.4f}',
            'r@100':        f'{r100:.4f}',
            'n_relevant':   len(qrels.get(qid, [])),
            'category':     category,
        })

    rows.sort(key=lambda r: float(r['ndcg@1000']))

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # ----- CSV --------------------------------------------------------
    csv_path = os.path.join(out_dir, 'error_analysis.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'  -> {csv_path}')

    # ----- Top-K markdown table --------------------------------------
    top_k  = args.top_k
    top    = [r for r in rows if r['category'] != 'success'][:top_k]
    md_path = os.path.join(out_dir, 'error_analysis_top.md')
    with open(md_path, 'w') as f:
        f.write(f'# Worst {top_k} dev queries (sorted by NDCG@1000 asc)\n\n')
        f.write('| qid | NDCG | R@100 | #sent | #words | #PN | hedges | year | category |\n')
        f.write('|---|---|---|---|---|---|---|---|---|\n')
        for r in top:
            f.write('| {qid} | {ndcg@1000} | {r@100} | {n_sentences} | {n_words} | '
                    '{n_proper_nouns} | {hedge_count} | {has_year} | {category} |\n'.format(**r))

        # one verbatim sample per category
        f.write('\n\n## Verbatim samples\n')
        seen = set()
        for r in top:
            cat = r['category']
            if cat in seen:
                continue
            seen.add(cat)
            q = queries.get(r['qid'], {})
            text = _full_text(q).replace('\n', ' ')
            f.write(f'\n### {cat}  (qid={r["qid"]}, NDCG={r["ndcg@1000"]})\n')
            f.write(f'> {text[:600]}{"..." if len(text) > 600 else ""}\n')
    print(f'  -> {md_path}')

    # ----- Summary JSON ----------------------------------------------
    counts = Counter(r['category'] for r in rows)
    fail_counts = Counter(r['category'] for r in rows if r['category'] != 'success')
    summary = {
        'total_queries':       len(rows),
        'categories':          dict(counts),
        'failure_categories':  dict(fail_counts),
        'mean_ndcg@1000':      sum(float(r['ndcg@1000']) for r in rows) / len(rows),
        'mean_r@100':          sum(float(r['r@100'])     for r in rows) / len(rows),
    }
    sum_path = os.path.join(out_dir, 'error_analysis_summary.json')
    with open(sum_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'  -> {sum_path}')

    # ----- Console summary -------------------------------------------
    print(f'\n[3/3] summary')
    print('  category               n     share')
    print('  ---------------------- ----- -----')
    for cat, n in counts.most_common():
        print(f'  {cat:<22} {n:>5} {n/len(rows):>5.1%}')
    print(f'\n  total: {len(rows)} queries')
    print(f'  mean NDCG@1000: {summary["mean_ndcg@1000"]:.4f}')
    print(f'  mean R@100    : {summary["mean_r@100"]:.4f}')

    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scores',          default='scores/dense.json',
                         help='per-query scores from the model under study')
    parser.add_argument('--pipeline-scores', default=None,
                         help='optional second scores file (e.g. after reranking) '
                              'used to disambiguate OOD vs ranking errors')
    parser.add_argument('--queries',         default='data/trec/dev_queries.jsonl')
    parser.add_argument('--qrels',           default='data/trec/dev_qrels.txt')
    parser.add_argument('--top-k',           type=int, default=15)
    parser.add_argument('--out-dir',         default='reports')
    args = parser.parse_args()

    analyse(args)


if __name__ == '__main__':
    main()
