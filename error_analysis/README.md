# Error Analysis & Discussion

## Assignment requirement (5 pts)

> **Error Analysis & Discussion (5 pts):** Conduct a systematic investigation of
> instances where your model underperformed and identify common failure modes.
> Isolate specific query-document pairs where the model failed and discuss what
> these failures reveal about the limitations of your approach.

## What this folder provides

`error_analysis.py` is a reproducible, data-driven implementation of the above.
It is **not** hand-curated: it selects the worst queries by their cached
per-query NDCG@1000 and classifies each by a fixed rule set, then writes the
specific failing query–document pairs to disk.

### How to run

```bash
# Recommended: first dump a TREC run file of the final pipeline so the analysis
# can read the true rank of each gold document.
python eval_reranker_pipeline.py \
    --bi-checkpoint       checkpoints/best \
    --reranker-checkpoint checkpoints/reranker_qwen0.5b_3epoch/epoch_1 \
    --output-scores       scores/reranker_pipeline.json \
    --output-run          runs/reranked.trec

python error_analysis/error_analysis.py \
    --scores  scores/ablation.json \
    --stage   reranked \
    --queries data/trec/dev_queries.jsonl \
    --qrels   data/trec/dev_qrels.txt \
    --docs    data/trec/docs.jsonl \
    --run     runs/reranked.trec \
    --n 15
```

If no `--run` file is supplied the script still runs, but falls back to R@100
and cannot separate a *recall miss* (gold not in top-1000) from a
*rerank-window miss* (gold in 100..1000).

### Outputs (`error_analysis/outputs/`)

| File | Contents |
|---|---|
| `failures.csv` | one row per failing query: NDCG@1000, R@100, #sentences, proper-noun count, gold-doc rank, primary failure mode, tags, gold title(s), full query text |
| `failure_cases.md` | human-readable case studies (the isolated query–document pairs) for the report |
| `summary.json` | counts per failure mode and per query-side tag |

### Failure-mode taxonomy

The **primary mode** is keyed to *where in the cascade the query was lost* —
the most actionable axis:

- **`recall_miss_top1000`** — gold doc never retrieved by the dense stage
  (rank > 1000 / absent). No reranker can recover it → fix first-stage recall.
- **`rerank_window_miss`** — gold doc retrieved but below the top-100 rerank
  pool (100 < rank ≤ 1000) → widen the rerank window.
- **`ranking_error`** — gold doc was inside the top-100 the reranker scored,
  yet still ranked poorly → a genuine scoring/ordering limitation.
- **`recall_miss_top100`** — (no run file) R@100 = 0.

Independent **query-side tags** (can co-occur): `sparse_query` (≤ 3 sentences),
`few_entities` (≤ 1 proper-noun-like token).

## Narrative (as written in the report)

This is the qualitative discussion that accompanies the quantitative output
above; it appears in `reports/final_report.tex` under
**Error Analysis and Discussion**.

> We selected the 15 lowest-NDCG@1000 queries from the full pipeline and
> clustered them by failure mode. Three categories account for 13 of 15
> failures:
>
> - **Sparse query (5 queries).** Queries with ≤ 3 sentences and few proper
>   nouns, e.g. "a movie about a dog and a beach, I think it was sad". The GRU's
>   sequential pooler has little to attend to, and the dense retriever returns
>   generic plot matches.
> - **Strong false memory (5 queries).** Queries that confidently assert an
>   incorrect detail (e.g. a wrong actor or decade). The contrastive loss has no
>   mechanism to discount asserted-but-wrong attributes, and the reranker often
>   accepts the misleading claim. This points to a need for either
>   retrieval-time query rewriting or an LLM step that explicitly hedges over
>   user claims.
> - **Out-of-distribution domain (3 queries).** Queries about films outside the
>   TREC corpus's apparent coverage. These are irrecoverable without a
>   recall-augmenting document expansion step.
>
> The remaining two failures are pure ranking errors where the target was
> retrieved within the top-1000 but never lifted into the reranker pool
> (top-100). This is a direct argument for either widening the rerank pool or
> adding a sparse + dense hybrid retrieval stage in future work.

### Note on reconciling the narrative with the script

The narrative's categories map onto the script's output as follows:

| Narrative category | Script signal |
|---|---|
| Sparse query | `sparse_query` / `few_entities` tags |
| Out-of-distribution domain | `recall_miss_top1000` |
| Pure ranking error (in top-1000, not top-100) | `rerank_window_miss` |
| Ranking error (in top-100) | `ranking_error` |

**"Strong false memory"** is a *semantic* category that cannot be detected
automatically from ranks alone — it requires reading the query against the gold
document. Use `failure_cases.md` to confirm which `ranking_error` /
`recall_miss` cases are genuinely false-memory queries before quoting the
"5 queries" figure, so the report's counts match the actual data.
