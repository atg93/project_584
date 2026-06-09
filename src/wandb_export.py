"""
Export W&B run summaries to CSV.

Standalone script. Does NOT modify any existing file. Reads the W&B
public API and dumps every run's config + summary metrics into per-project
CSVs plus one combined CSV, so the exact numbers can be pasted directly
into the final report's tables.

What it produces
----------------
  reports/wandb/<project>.csv      — one row per run in that W&B project
  reports/wandb/all_runs.csv       — union over all projects (long format)
  reports/wandb/summary.md         — best-by-NDCG-per-project quick table

Defaults
--------
  Pulls these four projects (matching the W&B project names used in the
  existing training / eval scripts):

      tot-bm25-sweep
      tot-tokenizer-sweep
      tot-rq2-dropout-sweep
      tot-reranker

Usage
-----
  # one-time login (uses the API key in ~/.netrc or WANDB_API_KEY)
  wandb login

  # default — pull all four projects, write to reports/wandb/
  python wandb_export.py --entity tugrul-gorgulu-metu-mid

  # custom project list
  python wandb_export.py --entity tugrul-gorgulu-metu-mid \\
      --projects tot-bm25-sweep tot-tokenizer-sweep

  # also include each run's full history (per-step logs) — slow, optional
  python wandb_export.py --entity tugrul-gorgulu-metu-mid --include-history

Notes
-----
  * Reads, never writes, to W&B. Safe to re-run.
  * Picks up the API key from the local wandb_api_key file if present,
    otherwise from the WANDB_API_KEY env var, otherwise from ~/.netrc.
"""

import os
import csv
import json
import argparse
from collections import defaultdict


DEFAULT_PROJECTS = [
    'tot-bm25-sweep',
    'tot-tokenizer-sweep',
    'tot-rq2-dropout-sweep',
    'tot-reranker',
]

# metrics we want to surface at the top of the CSV (others get included too,
# this just controls column order so the report-relevant ones come first)
PRIORITY_METRICS = [
    'final_ndcg@1000', 'final_r@100',
    'ndcg@1000',       'r@100',
    'eval_ndcg@1000',  'eval_r@100',
    'trec_loss',       'reddit_loss',  'epoch_loss',
    'epoch',           'step',
]


def _read_api_key():
    """Look for a key file colocated with this script before falling back to env."""
    here = os.path.dirname(os.path.abspath(__file__))
    key_file = os.path.join(here, 'wandb_api_key')
    if os.path.exists(key_file):
        with open(key_file) as f:
            k = f.read().strip()
        if k:
            os.environ.setdefault('WANDB_API_KEY', k)
            return k
    return os.environ.get('WANDB_API_KEY')


def _flatten(d, prefix=''):
    """Flatten a dict one level so nested config / summary fits in CSV cells."""
    flat = {}
    for k, v in d.items():
        key = f'{prefix}{k}' if not prefix else f'{prefix}.{k}'
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


def _safe_str(v):
    if v is None:
        return ''
    if isinstance(v, (dict, list)):
        return json.dumps(v, default=str)
    return str(v)


def _ordered_fieldnames(rows):
    """Stable column order: id/name/state first, priority metrics next,
       then everything else alphabetically."""
    head    = ['project', 'run_id', 'name', 'state', 'created_at', 'tags']
    metrics = [m for m in PRIORITY_METRICS if any(m in r for r in rows)]
    rest    = set()
    for r in rows:
        rest.update(r.keys())
    rest -= set(head)
    rest -= set(metrics)
    return head + metrics + sorted(rest)


def export_project(api, entity, project, include_history=False):
    rows = []
    try:
        runs = api.runs(f'{entity}/{project}')
    except Exception as e:
        print(f'  [{project}] could not list runs: {e}')
        return rows

    for r in runs:
        row = {
            'project':    project,
            'run_id':     r.id,
            'name':       r.name,
            'state':      r.state,
            'created_at': str(r.created_at),
            'tags':       ','.join(r.tags or []),
        }
        # summary  (final metric values)
        try:
            summary = dict(r.summary._json_dict) if hasattr(r.summary, '_json_dict') \
                       else dict(r.summary)
        except Exception:
            summary = {}
        # drop internal wandb keys (_step, _runtime, etc.)
        summary = {k: v for k, v in summary.items() if not k.startswith('_')}
        row.update(_flatten(summary))

        # config (hyperparameters)
        try:
            cfg = {k: v.get('value') if isinstance(v, dict) and 'value' in v else v
                    for k, v in dict(r.config).items()}
        except Exception:
            cfg = {}
        cfg = {k: v for k, v in cfg.items() if not k.startswith('_')}
        row.update(_flatten(cfg, prefix='config'))

        # optional: pull full history (per-step logs) for later analysis
        if include_history:
            try:
                hist = r.history(pandas=False)  # list[dict]
                row['history_steps'] = len(hist)
            except Exception:
                row['history_steps'] = ''

        rows.append(row)

    print(f'  [{project}] {len(rows)} runs')
    return rows


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = _ordered_fieldnames(rows)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow({k: _safe_str(v) for k, v in r.items()})


def _best_row(rows, metric='final_ndcg@1000'):
    pool = [(r, r.get(metric)) for r in rows if r.get(metric) not in (None, '')]
    if not pool:
        # fall back to ndcg@1000 / eval_ndcg@1000
        for fallback in ('ndcg@1000', 'eval_ndcg@1000'):
            pool = [(r, r.get(fallback)) for r in rows if r.get(fallback) not in (None, '')]
            if pool:
                metric = fallback
                break
    if not pool:
        return None, None
    best = max(pool, key=lambda t: float(t[1]))
    return best[0], metric


def write_summary_md(path, by_project):
    lines = ['# W&B run summary (best NDCG@1000 per project)', '']
    lines.append('| project | best run | NDCG@1000 | R@100 | config |')
    lines.append('|---|---|---|---|---|')
    for project, rows in by_project.items():
        if not rows:
            lines.append(f'| {project} | _no runs_ | — | — | — |')
            continue
        best, metric_used = _best_row(rows)
        if best is None:
            lines.append(f'| {project} | _no NDCG logged_ | — | — | — |')
            continue
        ndcg = best.get(metric_used, '')
        r100 = best.get('final_r@100') or best.get('r@100') or best.get('eval_r@100') or ''
        cfg  = {k.replace('config.', ''): v for k, v in best.items() if k.startswith('config.')}
        # keep cfg short
        short_cfg = {k: cfg[k] for k in list(cfg)[:6]}
        lines.append(
            f'| {project} | {best.get("name", best.get("run_id"))} | '
            f'{ndcg} | {r100} | `{json.dumps(short_cfg, default=str)}` |'
        )
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--entity', required=True,
                         help='W&B entity / username (e.g. tugrul-gorgulu-metu-mid)')
    parser.add_argument('--projects', nargs='+', default=DEFAULT_PROJECTS,
                         help='W&B project names to export')
    parser.add_argument('--out-dir', default='reports/wandb')
    parser.add_argument('--include-history', action='store_true',
                         help='also count per-step history rows (slower)')
    args = parser.parse_args()

    _read_api_key()
    try:
        import wandb
    except ImportError:
        raise SystemExit('wandb not installed. Install with:  pip install wandb')
    api = wandb.Api(timeout=60)

    print(f'entity   : {args.entity}')
    print(f'projects : {args.projects}')
    print(f'out_dir  : {args.out_dir}\n')

    by_project = {}
    all_rows   = []
    for project in args.projects:
        rows = export_project(api, args.entity, project,
                                include_history=args.include_history)
        by_project[project] = rows
        all_rows.extend(rows)
        write_csv(os.path.join(args.out_dir, f'{project}.csv'), rows)

    write_csv(os.path.join(args.out_dir, 'all_runs.csv'), all_rows)
    write_summary_md(os.path.join(args.out_dir, 'summary.md'), by_project)

    print(f'\nWrote {len(all_rows)} rows across {len(by_project)} projects.')
    print(f'See {args.out_dir}/summary.md for the quick best-of-project table.')


if __name__ == '__main__':
    main()
