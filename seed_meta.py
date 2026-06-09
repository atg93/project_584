"""
Seed meta.json for already-completed tokenizer runs so tokenizer_eval.py
skips retraining them. Run this once before launching tokenizer_eval.py.

Usage:
    python seed_meta.py
"""

import os
import json

CHECKPOINT_DIR = 'checkpoints/tokenizer_comparison'
EPOCHS = 3

# Runs to mark as completed
COMPLETED_RUNS = [
    ('wordpiece',       'bert-base-uncased', 'WordPiece (uncased)', 768),
    ('wordpiece',       'bert-base-uncased', 'WordPiece (uncased)', 512),
    ('wordpiece_cased', 'bert-base-cased',   'WordPiece (cased)',   768),
    ('wordpiece_cased', 'bert-base-cased',   'WordPiece (cased)',   512),
]

results_path = os.path.join(CHECKPOINT_DIR, 'tokenizer_results.json')
saved_results = {}
if os.path.exists(results_path):
    with open(results_path) as f:
        saved_results = json.load(f)

for key, model_name, tokenizer_type, proj_dim in COMPLETED_RUNS:
    run_key   = f'{key}_d{proj_dim}'
    ckpt_path = os.path.join(CHECKPOINT_DIR, run_key)
    meta_path = os.path.join(ckpt_path, 'meta.json')

    if not os.path.exists(os.path.join(ckpt_path, 'query_enc.pt')):
        print(f'  [{run_key}] No checkpoint weights found — skipping seed.')
        continue

    meta = {
        'model_name':       model_name,
        'tokenizer_type':   tokenizer_type,
        'proj_dim':         proj_dim,
        'epochs_completed': EPOCHS,
    }

    # pull in metrics from tokenizer_results.json if available
    if run_key in saved_results:
        r = saved_results[run_key]
        meta['ndcg@1000'] = r.get('ndcg@1000', 0.0)
        meta['r@100']     = r.get('r@100',     0.0)

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    metrics_str = (f"  NDCG@1000={meta['ndcg@1000']:.4f}  R@100={meta['r@100']:.4f}"
                   if 'ndcg@1000' in meta else '  (no metrics found in tokenizer_results.json)')
    print(f'  [{run_key}] meta.json written.{metrics_str}')
