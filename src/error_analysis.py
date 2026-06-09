"""
Systematic error analysis for the BERT+GRU+Qwen2.5-0.5B ToT pipeline.

Satisfies the "Error Analysis & Discussion" requirement: it isolates the
specific query--document pairs on which the final model underperforms, derives
per-query diagnostic features, and assigns each failure to a concrete,
data-driven failure mode. Nothing here is hand-picked: the failing queries are
selected by their cached per-query NDCG@1000 and categorised by rule.

What it produces (under --out-dir, default error_analysis/outputs/)
------------------------------------------------------------------
  failures.csv        one row per failing query: qid, ndcg@1000, r@100,
                      num_sentences, proper_noun_count, gold ranks, primary
                      failure mode, secondary tags, gold title(s).
  failure_cases.md    human-readable case studies (query text + gold doc +
                      diagnosis) for the report's Error Analysis section.
  summary.json        counts per failure mode and per tag.

Failure-mode taxonomy
---------------------
The primary mode is keyed to *where in the cascade* the query was lost, which
is the most actionable axis. If a TREC run file (--run) is supplied we read the
true rank of the gold document; otherwise we fall back to R@100.

  recall_miss_top1000   gold doc never retrieved by the dense stage
                        (rank > 1000 / absent)            -> first-stage recall.
  rerank_window_miss    gold doc retrieved (<=1000) but below the top-100
                        rerank pool (100 < rank <= 1000)  -> widen rerank pool.
  ranking_error         gold doc inside the top-100 the reranker saw, yet still
                        ranked poorly                     -> scoring / ordering.
  recall_miss_top100    (no run file) R@100 = 0: gold not in top-100; cannot
                        distinguish top-1000 from beyond without the run.

Independent query-side tags (can co-occur with any primary mode):
  sparse_query          <= 3 sentences (under-specified description).
  few_entities          <= 1 proper-noun-like token (little to anchor on).

Usage
-----
    python error_analysis/error_analysis.py \\
        --scores  scores/ablation.json \\
        --stage   reranked \\
        --queries data/trec/dev_queries.jsonl \\
        --qrels   data/trec/dev_qrels.txt \\
        --docs    data/trec/docs.jsonl \\
        --run     runs/reranked.trec        # optional but recommended \\
        --n 15

To get the run file, re-run the pipeline with --output-run, e.g.
    python eval_reranker_pipeline.py ... --output-run runs/reranked.trec
"""

import os
import re
import csv
import json
import argparse
from collections import Counter, OrderedDict


# ---------------------------------------------------------------------------
# Self-contained loaders (no dependency on repo cwd / dataset.py import path)
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qrels(path):
    """TREC qrel format: qid 0 docid rel  (keeps rel > 0)."""
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, did, rel = parts[0], parts[1], parts[2], int(parts[3])
            if rel > 0:
                qrels.setdefault(qid, []).append(did)
    return qrels


def load_run(path):
    """TREC run format: qid Q0 docid rank score runname -> {qid: {docid: rank}}."""
    run = {}
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4:
                continue
            qid, _, did, rank = parts[0], parts[1], parts[2], int(parts[3])
            run.setdefault(qid, {})[did] = rank
    return run


def load_scores(path, stage_prefs):
    """Return {qid: {metric: value}} for the first matching stage; handles the
    nested eval_*_pipeline --output-scores layout and flat {qid: {...}}."""
    with open(path) as f:
        blob = json.load(f)
    if isinstance(blob, dict) and 'per_query' in blob and isinstance(blob['per_query'], dict):
        pq = blob['per_query']
        for s in stage_prefs:
            if s in pq:
                return pq[s], s
        raise SystemExit(f'none of {stage_prefs} in per_query stages {list(pq.keys())}')
    return blob, '(flat)'


# ---------------------------------------------------------------------------
# Query-side diagnostics
# ---------------------------------------------------------------------------

def query_text_of(q):
    sents = q.get('sentences') or []
    if sents:
        return [s.strip() for s in sents if s.strip()]
    txt = q.get('text', '') or ''
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', txt) if s.strip()]


_STOP_CAPS = {'I', 'The', 'A', 'An', 'It', 'He', 'She', 'They', 'We', 'My',
              'There', 'This', 'That', 'But', 'And', 'So', 'In', 'On', 'At'}


def proper_noun_count(sentences):
    """Heuristic: capitalised tokens that are not sentence-initial and not common
    function words — a rough proxy for named entities (titles, actors, places)."""
    n = 0
    for s in sentences:
        toks = re.findall(r"[A-Za-z][A-Za-z'\-]+", s)
        for i, t in enumerate(toks):
            if i == 0:
                continue                      # skip sentence-initial capital
            if t[0].isupper() and t not in _STOP_CAPS:
                n += 1
    return n


def best_gold_rank(run, qid, gold_ids):
    """Smallest rank among the query's gold docs in the run, or None."""
    ranks = run.get(qid, {})
    found = [ranks[g] for g in gold_ids if g in ranks]
    return min(found) if found else None


# title field varies across corpus dumps; try the common names, then fall back
_TITLE_KEYS = ('title', 'page_title', 'name', 'doc_title', 'wiki_title',
               'movie_title', 'pageTitle')
_TEXT_KEYS  = ('text', 'plot', 'body', 'contents', 'summary')


def doc_title(docs, did):
    """Best-effort human-readable title for a doc id, robust to the field name."""
    d = docs.get(did)
    if not d:
        return did
    for k in _TITLE_KEYS:
        v = d.get(k)
        if v:
            return str(v).strip()
    for k in _TEXT_KEYS:                      # fall back to first words of the body
        v = d.get(k)
        if v:
            return ' '.join(str(v).split()[:8]) + '...'
    return did


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify(ndcg, r100, num_sent, pn_count, gold_rank, have_run,
             rerank_k=100, recall_k=1000):
    tags = []
    if num_sent <= 3:
        tags.append('sparse_query')
    if pn_count <= 1:
        tags.append('few_entities')

    if have_run:
        if gold_rank is None or gold_rank > recall_k:
            mode = 'recall_miss_top1000'
        elif gold_rank > rerank_k:
            mode = 'rerank_window_miss'
        else:
            mode = 'ranking_error'
    else:
        if r100 <= 0.0:
            mode = 'recall_miss_top100'
        else:
            mode = 'ranking_error'
    return mode, tags


MODE_EXPLAIN = {
    'recall_miss_top1000':
        'gold document never retrieved by the dense first stage; no reranker '
        'can recover it. Argues for stronger first-stage recall (hybrid '
        'sparse+dense, document expansion).',
    'rerank_window_miss':
        'gold document retrieved but ranked below the top-100 rerank pool, so '
        'the cross-encoder never scored it. Argues for a wider rerank window.',
    'ranking_error':
        'gold document was inside the top-100 the reranker saw, yet still '
        'ranked poorly. A genuine scoring/ordering limitation of the model.',
    'recall_miss_top100':
        'gold document not in the top-100 (R@100 = 0); without the run file we '
        'cannot tell whether it sits in 100..1000 or beyond.',
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scores',  default='scores/ablation.json')
    p.add_argument('--stage',   default='reranked')
    p.add_argument('--queries', default='data/trec/dev_queries.jsonl')
    p.add_argument('--qrels',   default='data/trec/dev_qrels.txt')
    p.add_argument('--docs',    default='data/trec/docs.jsonl')
    p.add_argument('--run',     default=None,
                   help='optional TREC run file of the final pipeline; enables '
                        'true gold-rank-based failure modes')
    p.add_argument('--metric',  default='ndcg@1000')
    p.add_argument('--n', type=int, default=15, help='number of worst queries')
    p.add_argument('--rerank-k', type=int, default=100)
    p.add_argument('--recall-k', type=int, default=1000)
    p.add_argument('--out-dir', default='error_analysis/outputs')
    args = p.parse_args()

    scores, stage = load_scores(args.scores, [args.stage, 'reranked'])
    queries = {q['id']: q for q in load_jsonl(args.queries)}
    qrels   = load_qrels(args.qrels)
    docs    = {d['id']: d for d in load_jsonl(args.docs)}
    run     = load_run(args.run) if args.run and os.path.exists(args.run) else None
    have_run = run is not None
    if not have_run:
        target = args.run or 'runs/reranked.trec'
        print('(note) no run file — failure modes collapse to recall_miss_top100 '
              '(cannot separate gold-missed >1000 from gold-in-100..1000).\n'
              '       To get the full breakdown, regenerate the run and re-run:\n'
              f'         python eval_reranker_pipeline.py ... --output-run {target}\n'
              f'         python error_analysis/error_analysis.py ... --run {target}')

    # rank queries by the metric (ascending = worst first), pick the n worst
    ranked = sorted(scores.items(), key=lambda kv: kv[1].get(args.metric, 0.0))
    worst  = ranked[:args.n]

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    mode_counts, tag_counts = Counter(), Counter()

    for qid, sc in worst:
        q = queries.get(qid, {})
        sents = query_text_of(q)
        num_sent = len(sents)
        pn = proper_noun_count(sents)
        gold_ids = qrels.get(qid, [])
        grank = best_gold_rank(run, qid, gold_ids) if have_run else None
        ndcg = float(sc.get('ndcg@1000', 0.0))
        r100 = float(sc.get('r@100', 0.0))

        mode, tags = classify(ndcg, r100, num_sent, pn, grank, have_run,
                              args.rerank_k, args.recall_k)
        mode_counts[mode] += 1
        for t in tags:
            tag_counts[t] += 1

        gold_titles = [doc_title(docs, g) for g in gold_ids]
        rows.append(OrderedDict(
            qid=qid, ndcg=round(ndcg, 4), r100=round(r100, 4),
            num_sentences=num_sent, proper_nouns=pn,
            gold_rank=('' if grank is None else grank),
            in_top100=('' if not have_run else int(grank is not None and grank <= args.rerank_k)),
            in_top1000=('' if not have_run else int(grank is not None and grank <= args.recall_k)),
            primary_mode=mode, tags='|'.join(tags),
            gold_ids='|'.join(gold_ids),
            gold_titles='|'.join(gold_titles),
            query=' '.join(sents),
        ))

    # ---- failures.csv ----
    csv_path = os.path.join(args.out_dir, 'failures.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['qid'])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # ---- summary.json ----
    summary = {
        'stage': stage, 'metric': args.metric, 'n_worst': len(rows),
        'used_run_file': have_run,
        'mode_counts': dict(mode_counts), 'tag_counts': dict(tag_counts),
    }
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ---- failure_cases.md ----
    md_path = os.path.join(args.out_dir, 'failure_cases.md')
    with open(md_path, 'w') as f:
        f.write(f'# Error analysis — {len(rows)} worst queries by {args.metric}\n\n')
        f.write(f'Stage: `{stage}`  |  run file used: `{have_run}`\n\n')
        f.write('## Failure-mode distribution\n\n')
        for m, c in mode_counts.most_common():
            f.write(f'- **{m}** ({c}): {MODE_EXPLAIN.get(m, "")}\n')
        f.write('\n## Query-side tags\n\n')
        for t, c in tag_counts.most_common():
            f.write(f'- `{t}`: {c}\n')
        f.write('\n## Case studies\n\n')
        for i, r in enumerate(rows, 1):
            f.write(f'### {i}. Query `{r["qid"]}` — {r["primary_mode"]}'
                    f'{" [" + r["tags"] + "]" if r["tags"] else ""}\n\n')
            f.write(f'- NDCG@1000 = {r["ndcg"]}, R@100 = {r["r100"]}, '
                    f'sentences = {r["num_sentences"]}, proper-nouns = {r["proper_nouns"]}'
                    f'{", gold rank = " + str(r["gold_rank"]) if r["gold_rank"] != "" else ""}\n')
            f.write(f'- Gold: {r["gold_titles"] or r["gold_ids"]}\n')
            qtxt = r['query']
            f.write(f'- Query: {qtxt[:500]}{"..." if len(qtxt) > 500 else ""}\n\n')

    # ---- console summary ----
    print(f'\nWorst {len(rows)} queries by {args.metric} (stage={stage}, run={have_run})')
    print('Failure modes:')
    for m, c in mode_counts.most_common():
        print(f'  {m:<22} {c}')
    print('Query-side tags:')
    for t, c in tag_counts.most_common():
        print(f'  {t:<22} {c}')
    print(f'\nWrote:\n  {csv_path}\n  {md_path}\n  '
          f'{os.path.join(args.out_dir, "summary.json")}')


if __name__ == '__main__':
    main()
