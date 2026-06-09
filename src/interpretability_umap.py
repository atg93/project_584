"""
Interpretability analysis — UMAP of the dense embedding space for the
BERT + GRU + Qwen2.5-0.5B final (hybrid) model.

Standalone script. Does NOT modify any existing file. Reads the same
data / checkpoint layout used by eval_ablation_pipeline.py and train.py.

The final model is a two-stage hybrid:
    1. BERT (per-sentence [CLS]) -> GRU + attention -> proj   (dense retriever)
    2. Qwen2.5-0.5B LoRA cross-encoder                        (re-ranks top-K)

A cross-encoder produces no standalone vector, so there is nothing to embed
for stage 2. The architecture-appropriate move is therefore:

  * UMAP the *dense* BERT+GRU space (the only geometry that exists), and
  * colour each query by the FINAL (reranked) per-query NDCG, and
  * draw a second, contrastive figure that colours each query by how much the
    Qwen reranker changed its NDCG (reranked - dense). That delta figure is the
    "contrastive analysis for a hybrid system" view: it shows *where in the
    dense geometry* the cross-encoder rescues (or hurts) queries.

What it produces
----------------
  reports/figures/umap_query_doc.pdf / .png
        UMAP scatter of dev queries (circles) + ground-truth docs (triangles),
        queries coloured by the FINAL per-query NDCG@1000.
  reports/figures/umap_rerank_delta.pdf / .png         (only if a dense baseline
        is available) same layout, queries coloured by (reranked - dense) NDCG
        on a diverging colormap — the Qwen reranker's contribution.
  reports/umap_coords.csv
        qid, did, q_x, q_y, d_x, d_y, ndcg_final, ndcg_dense, ndcg_delta, r@100

Where the scores come from
--------------------------
Run eval_ablation_pipeline.py with --output-scores to get a nested JSON that
holds every stage's per-query metrics, then point --scores at it:

    python eval_ablation_pipeline.py ... --output-scores scores/ablation.json
    python interpretability_umap.py --bi-checkpoint checkpoints/best \\
        --scores scores/ablation.json

--scores also accepts a flat {qid: {ndcg@1000, r@100}} file (e.g. the legacy
scores/dense.json); in that case there is no reranked stage, so the delta
figure is skipped and the colour is just the dense NDCG.

Usage
-----
    python interpretability_umap.py \\
        --bi-checkpoint  checkpoints/best \\
        --queries        data/trec/dev_queries.jsonl \\
        --qrels          data/trec/dev_qrels.txt \\
        --docs           data/trec/docs.jsonl \\
        --scores         scores/ablation.json \\
        --final-stage    reranked \\
        --dense-stage    bert_gru
"""

import os
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from tot_retrieval import QueryEncoder, DocEncoder
from dataset import TRECToTDataset, DocumentDataset, load_jsonl, load_qrels


def _collate(batch):
    return {
        'query_id':         [b['query_id']        for b in batch],
        'sentence_ids':     torch.stack([b['sentence_ids']     for b in batch]),
        'sentence_masks':   torch.stack([b['sentence_masks']   for b in batch]),
        'doc_input_ids':    torch.stack([b['doc_input_ids']    for b in batch]),
        'doc_attention_mask': torch.stack([b['doc_attention_mask'] for b in batch]),
    }


def _doc_collate(batch):
    return {
        'doc_id':         [b['doc_id'] for b in batch],
        'input_ids':      torch.stack([b['input_ids']      for b in batch]),
        'attention_mask': torch.stack([b['attention_mask'] for b in batch]),
    }


def _load_state(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    # tolerate keys saved from DataParallel ("module." prefix)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model


def _load_meta(args):
    """
    If <bi-checkpoint>/meta.json exists, use the trained bi-encoder hyper-
    parameters so the encoders here always match the weights on disk (same
    behaviour as eval_ablation_pipeline._load_meta). CLI flags are fallbacks
    for keys missing from meta.json.
    """
    meta_path = os.path.join(args.bi_checkpoint, 'meta.json')
    if not os.path.exists(meta_path):
        print(f'  no meta.json in {args.bi_checkpoint} — using CLI hyperparameters '
              f'(model={args.model}, proj_dim={args.proj_dim}, gru_hidden={args.gru_hidden})')
        return
    with open(meta_path) as f:
        meta = json.load(f)
    for cli_key, meta_key in {
        'model':            'bert_model',
        'gru_hidden':       'gru_hidden',
        'proj_dim':         'proj_dim',
    }.items():
        if meta.get(meta_key) is not None:
            setattr(args, cli_key, meta[meta_key])
    print(f'  loaded bi-encoder config from {meta_path}: '
          f'model={args.model}, gru_hidden={args.gru_hidden}, proj_dim={args.proj_dim}')


def extract_embeddings(args, device):
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print(f'[1/4] loading dev queries / qrels / docs ...')
    dev_set = TRECToTDataset(args.queries, args.docs, args.qrels, tokenizer,
                              max_sent_len=64, max_doc_len=512)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size, shuffle=False,
                             collate_fn=_collate, num_workers=2)

    print(f'[2/4] loading encoders from {args.bi_checkpoint} ...')
    query_enc = QueryEncoder(bert_model=args.model, gru_hidden=args.gru_hidden,
                              proj_dim=args.proj_dim,
                              sentence_dropout=0.0).to(device).eval()
    doc_enc   = DocEncoder(  bert_model=args.model, proj_dim=args.proj_dim).to(device).eval()
    _load_state(query_enc, os.path.join(args.bi_checkpoint, 'query_enc.pt'), device)
    _load_state(doc_enc,   os.path.join(args.bi_checkpoint, 'doc_enc.pt'),   device)

    print('[3/4] computing query + ground-truth-doc embeddings ...')
    qids, q_vecs, dids, d_vecs = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc='  Embedding'):
            q = query_enc(batch['sentence_ids'].to(device),
                          batch['sentence_masks'].to(device)).cpu().numpy()
            d = doc_enc(batch['doc_input_ids'].to(device),
                        batch['doc_attention_mask'].to(device)).cpu().numpy()
            qids.extend(batch['query_id'])
            q_vecs.append(q)
            d_vecs.append(d)

    q_vecs = np.vstack(q_vecs).astype('float32')
    d_vecs = np.vstack(d_vecs).astype('float32')

    # ground-truth doc id for each query (first relevant doc, same convention
    # the rest of the pipeline already uses)
    qrels = load_qrels(args.qrels)
    dids  = [qrels[q][0] if qrels.get(q) else '' for q in qids]

    return qids, q_vecs, dids, d_vecs


def _stage_per_query(blob, stage):
    """
    Pull one stage's {qid: {metric: value}} mapping out of a scores file that
    may be either:
      * the eval_ablation_pipeline --output-scores JSON, which nests stages
        under a top-level "per_query" dict: {"per_query": {stage: {qid: {...}}}}
      * a flat {qid: {metric: value}} file (legacy scores/dense.json).
    Returns {} if the requested stage is absent.
    """
    if isinstance(blob, dict) and 'per_query' in blob and isinstance(blob['per_query'], dict):
        return blob['per_query'].get(stage, {})
    # flat file: treat it as already being the requested (single) stage
    return blob if isinstance(blob, dict) else {}


def _load_scores(path, final_stage, dense_stage):
    """
    Returns (final_scores, dense_scores) where each is {qid: {metric: value}}.
    dense_scores is {} when the file has no separate dense stage (flat file or
    missing stage) — the caller then skips the contrastive delta figure.
    """
    if not os.path.exists(path):
        print(f'  (warning) scores file not found: {path} — colouring with zeros')
        return {}, {}
    with open(path) as f:
        blob = json.load(f)

    nested = isinstance(blob, dict) and 'per_query' in blob
    if nested:
        final = _stage_per_query(blob, final_stage)
        dense = _stage_per_query(blob, dense_stage)
        if not final:
            print(f'  (warning) stage "{final_stage}" not in {path}; '
                  f'available: {list(blob["per_query"].keys())}')
        if dense_stage == final_stage:
            dense = {}
        print(f'  scores: final="{final_stage}" ({len(final)} q), '
              f'dense="{dense_stage}" ({len(dense)} q)')
        return final, dense

    # flat legacy file — single stage, no delta
    print(f'  scores: flat file, {len(blob)} queries (no reranked stage → delta skipped)')
    return blob, {}


def _ndcg_vec(scores, qids):
    return np.array([scores.get(q, {}).get('ndcg@1000', 0.0) for q in qids])


def _base_scatter(ax, q_xy, d_xy, ndcg):
    """Connector lines + query/doc markers shared by both figures (no colour)."""
    for (qx, qy), (dx, dy) in zip(q_xy, d_xy):
        ax.plot([qx, dx], [qy, dy], color='lightgray', linewidth=0.4, alpha=0.6)


def _reduce(joint, method):
    """2-D projection of the joint query+doc matrix.

    method='umap' uses umap-learn; 'tsne' uses scikit-learn; 'pca' is a
    dependency-free numpy SVD. 'auto' tries them in that order and falls back
    to whatever is installed. Returns (coords, label) where label names the
    technique actually used (for axis titles). All three are valid
    "dense-vector" interpretability projections per the rubric.
    """
    if method in ('auto', 'umap'):
        try:
            import umap
            print('  reducer: UMAP')
            coords = umap.UMAP(n_neighbors=30, min_dist=0.1, metric='cosine',
                               random_state=42, n_components=2).fit_transform(joint)
            return coords, 'UMAP'
        except ImportError:
            if method == 'umap':
                raise SystemExit('umap-learn not installed. '
                                 'Install with: pip install --user umap-learn')
            print('  umap-learn not found → falling back to t-SNE')

    if method in ('auto', 'tsne'):
        try:
            from sklearn.manifold import TSNE
            n = joint.shape[0]
            perp = max(5, min(30, (n - 1) // 3))
            print(f'  reducer: t-SNE (sklearn, perplexity={perp})')
            coords = TSNE(n_components=2, metric='cosine', init='pca',
                          perplexity=perp, random_state=42).fit_transform(joint)
            return coords, 't-SNE'
        except ImportError:
            if method == 'tsne':
                raise SystemExit('scikit-learn not installed. '
                                 'Install with: pip install --user scikit-learn')
            print('  sklearn not found → falling back to PCA')

    # PCA via numpy SVD — no third-party deps
    print('  reducer: PCA (numpy SVD)')
    X = joint - joint.mean(axis=0, keepdims=True)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    return U[:, :2] * S[:2], 'PCA'


def run_umap(qids, q_vecs, dids, d_vecs, final_scores, dense_scores, out_dir,
             reducer='auto'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import csv

    print('[4/4] running projection and writing artefacts ...')

    # joint projection so queries and docs share a coordinate system
    joint = np.vstack([q_vecs, d_vecs])
    coords, proj = _reduce(joint, reducer)
    n      = len(qids)
    q_xy   = coords[:n]
    d_xy   = coords[n:]

    ndcg_final = _ndcg_vec(final_scores, qids)
    r100       = np.array([final_scores.get(q, {}).get('r@100', 0.0) for q in qids])
    have_dense = bool(dense_scores)
    ndcg_dense = _ndcg_vec(dense_scores, qids) if have_dense else np.zeros(n)
    ndcg_delta = ndcg_final - ndcg_dense

    os.makedirs(out_dir, exist_ok=True)
    figures_dir = os.path.join(out_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # ----- figure 1: final-model NDCG ----------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))
    _base_scatter(ax, q_xy, d_xy, ndcg_final)
    vmax = max(0.4, float(ndcg_final.max()) if n else 0.4)
    sc = ax.scatter(q_xy[:, 0], q_xy[:, 1], c=ndcg_final, cmap='viridis',
                     marker='o', s=42, edgecolor='black', linewidth=0.4,
                     label='Query', vmin=0.0, vmax=vmax)
    ax.scatter(d_xy[:, 0], d_xy[:, 1], c=ndcg_final, cmap='viridis',
                marker='^', s=42, edgecolor='black', linewidth=0.4,
                label='Ground-truth document', vmin=0.0, vmax=vmax)
    cb = plt.colorbar(sc, ax=ax, shrink=0.85)
    cb.set_label('Final (BERT+GRU+Qwen) per-query NDCG@1000')
    ax.set_xlabel(f'{proj}-1'); ax.set_ylabel(f'{proj}-2')
    ax.set_title(f'{proj} of TREC 2023 ToT dev queries + ground-truth documents\n'
                 'BERT + GRU dense space, coloured by final hybrid-model NDCG')
    ax.legend(loc='best', frameon=True)
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        p = os.path.join(figures_dir, f'umap_query_doc.{ext}')
        fig.savefig(p, dpi=300 if ext == 'png' else None)
        print(f'  -> {p}')
    plt.close(fig)

    # ----- figure 2: contrastive reranker delta ------------------------
    if have_dense:
        fig, ax = plt.subplots(figsize=(7, 6))
        _base_scatter(ax, q_xy, d_xy, ndcg_delta)
        lim = max(0.05, float(np.abs(ndcg_delta).max()) if n else 0.05)
        sc = ax.scatter(q_xy[:, 0], q_xy[:, 1], c=ndcg_delta, cmap='coolwarm',
                         marker='o', s=46, edgecolor='black', linewidth=0.4,
                         label='Query', vmin=-lim, vmax=lim)
        ax.scatter(d_xy[:, 0], d_xy[:, 1], c='lightgray',
                    marker='^', s=30, edgecolor='black', linewidth=0.3,
                    label='Ground-truth document')
        cb = plt.colorbar(sc, ax=ax, shrink=0.85)
        cb.set_label('NDCG@1000 change from Qwen reranker (reranked - dense)')
        ax.set_xlabel(f'{proj}-1'); ax.set_ylabel(f'{proj}-2')
        ax.set_title('Where the Qwen2.5-0.5B reranker helps\n'
                     'red = reranker improves the query, blue = it hurts')
        ax.legend(loc='best', frameon=True)
        fig.tight_layout()
        for ext in ('pdf', 'png'):
            p = os.path.join(figures_dir, f'umap_rerank_delta.{ext}')
            fig.savefig(p, dpi=300 if ext == 'png' else None)
            print(f'  -> {p}')
        plt.close(fig)
        improved = int((ndcg_delta > 1e-6).sum())
        hurt     = int((ndcg_delta < -1e-6).sum())
        print(f'  reranker delta: {improved} queries improved, {hurt} hurt, '
              f'{n - improved - hurt} unchanged; mean Δ={ndcg_delta.mean():+.4f}')

    # ----- coords CSV ---------------------------------------------------
    csv_path = os.path.join(out_dir, 'umap_coords.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['qid', 'did', 'q_x', 'q_y', 'd_x', 'd_y',
                    'ndcg_final', 'ndcg_dense', 'ndcg_delta', 'r@100'])
        for i, (q, d) in enumerate(zip(qids, dids)):
            w.writerow([q, d, f'{q_xy[i,0]:.4f}', f'{q_xy[i,1]:.4f}',
                        f'{d_xy[i,0]:.4f}', f'{d_xy[i,1]:.4f}',
                        f'{ndcg_final[i]:.4f}',
                        f'{ndcg_dense[i]:.4f}' if have_dense else '',
                        f'{ndcg_delta[i]:.4f}' if have_dense else '',
                        f'{r100[i]:.4f}'])
    print(f'  -> {csv_path}')

    return q_xy, d_xy, ndcg_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bi-checkpoint', required=True,
                         help='dir with query_enc.pt and doc_enc.pt (final BERT+GRU encoder)')
    parser.add_argument('--queries', default='data/trec/dev_queries.jsonl')
    parser.add_argument('--qrels',   default='data/trec/dev_qrels.txt')
    parser.add_argument('--docs',    default='data/trec/docs.jsonl')
    parser.add_argument('--scores',  default='scores/ablation.json',
                         help='per-query metrics: eval_ablation_pipeline --output-scores '
                              'JSON (nested by stage) or a flat {qid:{...}} file')
    parser.add_argument('--final-stage', default='reranked',
                         help='stage key to colour by in a nested scores file '
                              '(the final hybrid model)')
    parser.add_argument('--dense-stage', default='bert_gru',
                         help='stage key used as the dense baseline for the '
                              'contrastive reranker-delta figure')
    parser.add_argument('--model',     default='bert-base-uncased',
                         help='HF model id; overridden by meta.json if present')
    parser.add_argument('--gru-hidden', type=int, default=None,
                         help='GRU hidden size; overridden by meta.json if present')
    parser.add_argument('--proj-dim',  type=int, default=512,
                         help='projection dim; overridden by meta.json if present')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--reducer', default='auto',
                         choices=['auto', 'umap', 'tsne', 'pca'],
                         help='2-D projection: auto tries UMAP→t-SNE→PCA by '
                              'availability (default auto)')
    parser.add_argument('--out-dir',   default='reports')
    parser.add_argument('--seed',      type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')

    _load_meta(args)  # match the trained final-model hyperparameters
    qids, q_vecs, dids, d_vecs = extract_embeddings(args, device)
    final_scores, dense_scores = _load_scores(args.scores, args.final_stage, args.dense_stage)
    run_umap(qids, q_vecs, dids, d_vecs, final_scores, dense_scores, args.out_dir,
             reducer=args.reducer)
    print('\nDone.')


if __name__ == '__main__':
    main()
