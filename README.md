# Tip-of-the-Tongue Retrieval on TREC 2023

## Project Structure
- **src/** — all runnable Python modules: BM25 baseline, dense retrieval, reranker training, ablation/eval pipelines, tokenizer/dropout sweeps, significance tests, error analysis, UMAP interpretability, and W&B export utilities.
- **scores/** — JSON files with retrieval metric results (BM25, dense, and reranker pipelines) plus failure case logs from error analysis runs.
- **figures/** — generated visualizations: model architecture diagrams, UMAP plots of query/document embeddings, and reranking delta maps (PNG and PDF).
- **error_analysis/** — error analysis outputs including per-query failure breakdowns and summary statistics.
- **reports/** — proposal, interim, and final report PDFs/LaTeX source, plus report-specific figures.

## Setup
1. Clone the repo
2. python -m venv venv && source venv/bin/activate
3. pip install -r requirements.txt

## Data

All data lives under a `data/` directory (not committed to the repo). Run the
automated downloader to populate it:


python src/download_data.py --output-dir data
This fetches:

TREC 2023 ToT via ir_datasets (dataset IDs
trec-tot/2023/train and trec-tot/2023/dev). ir_datasets handles
authentication and caching automatically. The document collection is ~231 K
Wikipedia articles and can take several minutes; pass --skip-docs to skip it
if you already have data/trec/docs.jsonl.

Reddit TOMT corpus from HuggingFace
(webis/tip-of-my-tongue-known-item-search-triplets) via the datasets
library. Answer documents are matched against the TREC Wikipedia collection by
doc ID, with a title-based fallback. Download the TREC docs first — the
Reddit downloader needs data/trec/docs.jsonl to resolve answer IDs.

After downloading, the expected layout is:

data/
├── trec/
│   ├── train_queries.jsonl   # {id, text, sentences: [...]}
│   ├── dev_queries.jsonl
│   ├── docs.jsonl            # {id, title, text}  — 231K Wikipedia docs
│   ├── train_qrels.txt       # TREC format: qid 0 docid rel
│   └── dev_qrels.txt
└── reddit/
    └── queries.jsonl         # {id, text, sentences, answer_id, domain}

Verify everything is in place.



## Reproducing Results
- BM25 baseline:        python src/bm25.py
- RQ1 tokenizer sweep:  python src/tokenizer_sweep.py
- RQ2 dropout sweep:    python src/dropout_sweep.py
- Reranker training:    python src/train_reranker.py
- Significance tests:   python src/significance_test.py
- Error analysis:       python src/error_analysis.py

## Random Seeds
All scripts set torch.manual_seed(42), np.random.seed(42), and FAISS
deterministic settings for reproducibility.

## Experiment Tracking (W&B)
 RQ1 tokenizer/dimension sweep
        https://wandb.ai/tugrul-gorgulu-metu-middle-east-technical-university/tot-tokenizer-sweep
  RQ2 sentence-dropout sweep
        https://wandb.ai/tugrul-gorgulu-metu-middle-east-technical-university/tot-rq2-dropout-sweep
  LoRA Qwen2.5 reranker training
        https://wandb.ai/tugrul-gorgulu-metu-middle-east-technical-university/tot-reranker
  BM25 baseline grid search
        https://wandb.ai/tugrul-gorgulu-metu-middle-east-technical-university/tot-bm25-sweep
  Dense BERT--GRU retrieval runs
        https://wandb.ai/tugrul-gorgulu-metu-middle-east-technical-university/tot-retrieval

## Reports
See reports/ for final PDF
