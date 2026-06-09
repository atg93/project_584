"""
Tokenizer diagnostic script.
Compares how different tokenizers handle TREC vs Reddit queries:
  - average tokens per sentence
  - truncation rate (sentences exceeding max_sent_len)
  - [UNK] token rate

Usage:
    python diagnose_tokenizer.py
    python diagnose_tokenizer.py --max_sent_len 64 --n_samples 200
"""

import argparse
import json
import numpy as np
from transformers import AutoTokenizer

# ── tokenizer presets matching the WandB run names ──────────────────────────
TOKENIZERS = {
    "bpe_d768":             "roberta-base",          # RoBERTa uses BPE
    "wordpiece_d512":       "bert-base-uncased",
    "wordpiece_d768":       "bert-base-uncased",
    "wordpiece_cased_d512": "bert-base-cased",
    "wordpiece_cased_d768": "bert-base-cased",
}

DATA = {
    "trec":   "data/trec/train_queries.jsonl",
    "reddit": "data/reddit/queries.jsonl",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def get_sentences(query):
    if "sentences" in query and query["sentences"]:
        return query["sentences"]
    return [s.strip() for s in query["text"].split(".") if s.strip()]


def analyze(tokenizer, samples, max_sent_len, tok_name):
    lengths, trunc, unk_counts, total_tokens = [], 0, 0, 0

    unk_id = tokenizer.unk_token_id  # None for BPE tokenizers without UNK

    for q in samples:
        for sent in get_sentences(q):
            ids = tokenizer.encode(sent, add_special_tokens=True)
            n = len(ids)
            lengths.append(n)
            if n > max_sent_len:
                trunc += 1
            if unk_id is not None:
                unk_counts += ids.count(unk_id)
            total_tokens += n

    lengths = np.array(lengths)
    return {
        "n_sentences":    len(lengths),
        "avg_len":        lengths.mean(),
        "p95_len":        np.percentile(lengths, 95),
        "max_len":        lengths.max(),
        "truncation_%":   100 * trunc / max(len(lengths), 1),
        "unk_%":          100 * unk_counts / max(total_tokens, 1),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_sent_len", type=int, default=64)
    parser.add_argument("--n_samples",    type=int, default=None,
                        help="cap number of queries per dataset (default: all)")
    args = parser.parse_args()

    # load datasets
    datasets = {}
    for name, path in DATA.items():
        try:
            data = load_jsonl(path)
            datasets[name] = data[:args.n_samples] if args.n_samples else data
            print(f"Loaded {len(datasets[name])} {name} queries from {path}")
        except FileNotFoundError:
            print(f"[WARN] {path} not found — skipping {name}")

    if not datasets:
        print("No data found. Check DATA paths at the top of the script.")
        return

    print(f"\nmax_sent_len = {args.max_sent_len}\n")
    print(f"{'Tokenizer':<28} {'Dataset':<8} {'Sentences':>10} {'Avg len':>8} "
          f"{'P95 len':>8} {'Max len':>8} {'Trunc %':>8} {'UNK %':>8}")
    print("-" * 92)

    for tok_label, tok_name in TOKENIZERS.items():
        try:
            tokenizer = AutoTokenizer.from_pretrained(tok_name)
        except Exception as e:
            print(f"[WARN] Could not load {tok_name}: {e}")
            continue

        for ds_name, samples in datasets.items():
            stats = analyze(tokenizer, samples, args.max_sent_len, tok_name)
            print(f"{tok_label:<28} {ds_name:<8} {stats['n_sentences']:>10} "
                  f"{stats['avg_len']:>8.1f} {stats['p95_len']:>8.1f} "
                  f"{stats['max_len']:>8} {stats['truncation_%']:>8.1f} "
                  f"{stats['unk_%']:>8.3f}")

    print()
    print("KEY:")
    print("  Trunc %  — sentences that exceed max_sent_len (get cut off during training)")
    print("  UNK %    — fraction of tokens that are [UNK] (WordPiece only; BPE uses subwords)")
    print("  High Avg/P95 len on Reddit but not TREC → fragmentation → information loss after truncation")


if __name__ == "__main__":
    main()
