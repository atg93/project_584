# Tip-of-the-Tongue Retrieval on TREC 2023

## Project Structure

- **src/** — all runnable Python modules: BM25 baseline, dense retrieval, reranker training, ablation/eval pipelines, tokenizer/dropout sweeps, significance tests, error analysis, UMAP interpretability, and W&B export utilities.
- **scores/** — JSON files with retrieval metric results (BM25, dense, and reranker pipelines) plus failure case logs from error analysis runs.
- **figures/** — generated visualizations: model architecture diagrams, UMAP plots of query/document embeddings, and reranking delta maps (PNG and PDF).
- **error_analysis/** — error analysis outputs including per-query failure breakdowns and summary statistics.
- **reports/** — proposal, interim, and final report PDFs/LaTeX source, plus report-specific figures.

## Setup

1. Clone the repo
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`

## Data

All data lives under a `data/` directory (not committed to the repo). Run the automated downloader to populate it:

```bash
python src/download_data.py --output-dir data
```

This fetches:

- **TREC 2023 ToT** via [`ir_datasets`](https://ir-datasets.com/) (dataset IDs `trec-tot/2023/train` and `trec-tot/2023/dev`). `ir_datasets` handles authentication and caching automatically. The document collection is ~231K Wikipedia articles and can take several minutes; pass `--skip-docs` to skip it if you already have `data/trec/docs.jsonl`.

- **Reddit TOMT corpus** from HuggingFace (`webis/tip-of-my-tongue-known-item-search-triplets`) via the `datasets` library. Answer documents are matched against the TREC Wikipedia collection by doc ID, with a title-based fallback. **Download the TREC docs first** — the Reddit downloader needs `data/trec/docs.jsonl` to resolve answer IDs.

After downloading, the expected layout is:

```
data/
  trec/
    train_queries.jsonl    # {id, text, sentences: [...]}
    dev_queries.jsonl      # {id, text, sentences: [...]}
    docs.jsonl             # {id, title, text}  — 231K Wikipedia docs
    train_qrels.txt        # TREC format: qid 0 docid rel
    dev_qrels.txt          # TREC format: qid 0 docid rel
  reddit/
    queries.jsonl          # {id, text, sentences, answer_id, domain}
```

Verify everything is in place with:

```bash
python src/download_data.py --verify-only
```

**Manual fallback (if `ir_datasets` cannot access TREC data):** Register at https://trec.nist.gov, request access to the 2023 ToT track, download the topics/qrels/Wikipedia dump, convert to the JSONL schema above, and place them in `data/trec/`.

## Reproducing Results

All scripts are run from the **repo root**. Data paths default to `data/trec/` and `data/reddit/` as laid out above. Supply your W&B API key via `wandb login` or the `WANDB_API_KEY` environment variable before running.

> **Singularity users (DOJO cluster):** prefix every command below with:
> ```
> singularity exec --nv --bind /datasets \
>   --env HF_HOME=/home/<user>/.cache/hf_cache \
>   --env TRANSFORMERS_CACHE=/home/<user>/.cache/hf_cache \
>   --env WANDB_API_KEY=<your-key> \
>   /home/<user>/containers/584.sig
> ```

### BM25 Baseline

```bash
python src/bm25_baseline.py
```

### To run the dropout experiment (RQ2)

Sweeps sentence dropout over {0.0, 0.1, 0.2, 0.3} for 10 epochs each, logs 4 runs to the `tot-rq2-dropout-sweep` W&B project.

```bash
python src/rq2_dropout_sweep.py
```

### To run the tokenizer experiment (RQ1)

Trains 12 configurations (4 tokenizer families × 3 projection dimensions {256, 512, 768}) for 3 epochs each, logs to `tot-tokenizer-sweep`.

```bash
python src/tokenizer_eval.py
```

### To run BERT–GRU training

Replace `100` with the desired number of epochs. Best checkpoint (by dev NDCG@1000) is saved to `checkpoints/best/`.

```bash
python src/train.py \
  --reddit-queries data/reddit/queries.jsonl \
  --epochs 100 \
  --wandb-project tot-retrieval \
  --wandb-run bert-gru-uncased-d512-p02
```

### To train the LLM reranker

First generate the reranker training dataset from the trained BERT–GRU model (encodes the full 231K-doc corpus — expect ~2 hours on a single GPU):

```bash
python src/eval_dense.py \
  --checkpoint checkpoints/best \
  --queries data/trec/train_queries.jsonl \
  --qrels data/trec/train_qrels.txt \
  --save-candidates runs/bi_encoder_train_candidates.jsonl \
  --reddit-queries data/reddit/queries.jsonl
```

Then train the reranker (Qwen2.5-0.5B, 4-bit NF4 quantization, LoRA r=16, 2 epochs; best adapter saved to `checkpoints/reranker/best/`, logs to `tot-reranker`):

```bash
python src/reranker.py
```

### Significance tests and error analysis

```bash
python src/significance_test.py
python src/error_analysis.py \
  --scores scores/ablation.json --stage reranked \
  --queries data/trec/dev_queries.jsonl --qrels data/trec/dev_qrels.txt \
  --docs data/trec/docs.jsonl --run runs/reranked.trec --n 15
python src/interpretability_umap.py
```

## Random Seeds

All scripts set `torch.manual_seed(42)`, `np.random.seed(42)`, and FAISS deterministic settings for reproducibility.

## Experiment Tracking (W&B)

| Experiment | W&B Project |
|---|---|
| RQ1 tokenizer/dimension sweep | [tot-tokenizer-sweep](https://wandb.ai/tugrul-gorgulu-metu-middle-east-technical-university/tot-tokenizer-sweep) |
| RQ2 sentence-dropout sweep | [tot-rq2-dropout-sweep](https://wandb.ai/tugrul-gorgulu-metu-middle-east-technical-university/tot-rq2-dropout-sweep) |
| LoRA Qwen2.5 reranker training | [tot-reranker](https://wandb.ai/tugrul-gorgulu-metu-middle-east-technical-university/tot-reranker) |
| BM25 baseline grid search | [tot-bm25-sweep](https://wandb.ai/tugrul-gorgulu-metu-middle-east-technical-university/tot-bm25-sweep) |
| Dense BERT-GRU retrieval runs | [tot-retrieval](https://wandb.ai/tugrul-gorgulu-metu-middle-east-technical-university/tot-retrieval) |

## Reports

See `reports/` for proposal, interim, and final PDFs.
