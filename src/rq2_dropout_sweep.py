"""
RQ2 — Noise Reduction via GRU Tuning
Addresses RQ2: "Can noise reduction techniques be further improved for retrieval performance?"

Fixed configuration: wordpiece_cased_d512
  - Model     : bert-base-cased  (WordPiece cased tokenizer)
  - proj_dim  : 512

Sweep: sentence_dropout ∈ {0.0, 0.1, 0.2, 0.3}

Fully self-contained — no local project imports required.
Multi-GPU: --gpus 0,1  (pass container-remapped indices, e.g. CUDA_VISIBLE_DEVICES=2,3 → 0,1)
W&B: per-epoch trec_loss / reddit_loss + dev NDCG@1000 / R@100 logged every --eval-every epochs.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import faiss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer
from scipy.stats import wilcoxon
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

MODEL_NAME     = 'bert-base-cased'
TOKENIZER_TYPE = 'WordPiece (cased)'
PROJ_DIM       = 512
DROPOUT_RATES  = [0.0, 0.1, 0.2, 0.3]


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_qrels(path):
    qrels = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            qid, did, rel = parts[0], parts[2], int(parts[3])
            if rel > 0:
                qrels.setdefault(qid, []).append(did)
    return qrels


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TRECDataset(Dataset):
    def __init__(self, query_path, doc_path, qrel_path, tokenizer,
                 max_sentences=30, max_sent_len=64, max_doc_len=512):
        self.tokenizer     = tokenizer
        self.max_sentences = max_sentences
        self.max_sent_len  = max_sent_len
        self.max_doc_len   = max_doc_len

        queries  = load_jsonl(query_path)
        qrels    = load_qrels(qrel_path)
        docs_raw = {d['id']: d for d in load_jsonl(doc_path)}

        self.examples = []
        for q in queries:
            qid = q['id']
            if qid not in qrels:
                continue
            pos_doc_id = qrels[qid][0]
            if pos_doc_id not in docs_raw:
                continue
            self.examples.append({
                'qid':       qid,
                'sentences': q.get('sentences', [q.get('text', '')])[:max_sentences],
                'doc':       docs_raw[pos_doc_id],
            })

    def _enc_sentences(self, sentences):
        enc = self.tokenizer(sentences, max_length=self.max_sent_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return enc['input_ids'], enc['attention_mask']

    def _enc_doc(self, doc):
        text = doc.get('title', '') + ' ' + doc.get('text', '')
        enc  = self.tokenizer(text, max_length=self.max_doc_len,
                              padding='max_length', truncation=True, return_tensors='pt')
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        sent_ids, sent_masks = self._enc_sentences(ex['sentences'])
        N = sent_ids.shape[0]
        if N < self.max_sentences:
            pad_ids   = torch.zeros(self.max_sentences - N, self.max_sent_len, dtype=torch.long)
            pad_masks = torch.zeros(self.max_sentences - N, self.max_sent_len, dtype=torch.long)
            sent_ids   = torch.cat([sent_ids,   pad_ids],   dim=0)
            sent_masks = torch.cat([sent_masks, pad_masks], dim=0)
        doc_ids, doc_masks = self._enc_doc(ex['doc'])
        return {
            'query_id':           ex['qid'],
            'sentence_ids':       sent_ids,
            'sentence_masks':     sent_masks,
            'doc_input_ids':      doc_ids,
            'doc_attention_mask': doc_masks,
        }


class RedditDataset(Dataset):
    def __init__(self, query_path, doc_path, tokenizer,
                 max_sentences=30, max_sent_len=64, max_doc_len=512):
        self.tokenizer     = tokenizer
        self.max_sentences = max_sentences
        self.max_sent_len  = max_sent_len
        self.max_doc_len   = max_doc_len

        docs_raw = {d['id']: d for d in load_jsonl(doc_path)}

        self.examples = []
        for q in load_jsonl(query_path):
            answer_id = q.get('answer_id')
            if not answer_id or answer_id not in docs_raw:
                continue
            sentences = q.get('sentences') or [
                s.strip() for s in q.get('text', '').split('.') if s.strip()
            ]
            if not sentences:
                continue
            self.examples.append({
                'qid':       q['id'],
                'sentences': sentences[:max_sentences],
                'doc':       docs_raw[answer_id],
            })

    def _enc_sentences(self, sentences):
        enc = self.tokenizer(sentences, max_length=self.max_sent_len,
                             padding='max_length', truncation=True, return_tensors='pt')
        return enc['input_ids'], enc['attention_mask']

    def _enc_doc(self, doc):
        text = doc.get('title', '') + ' ' + doc.get('text', '')
        enc  = self.tokenizer(text, max_length=self.max_doc_len,
                              padding='max_length', truncation=True, return_tensors='pt')
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        sent_ids, sent_masks = self._enc_sentences(ex['sentences'])
        N = sent_ids.shape[0]
        if N < self.max_sentences:
            pad_ids   = torch.zeros(self.max_sentences - N, self.max_sent_len, dtype=torch.long)
            pad_masks = torch.zeros(self.max_sentences - N, self.max_sent_len, dtype=torch.long)
            sent_ids   = torch.cat([sent_ids,   pad_ids],   dim=0)
            sent_masks = torch.cat([sent_masks, pad_masks], dim=0)
        doc_ids, doc_masks = self._enc_doc(ex['doc'])
        return {
            'query_id':           ex['qid'],
            'sentence_ids':       sent_ids,
            'sentence_masks':     sent_masks,
            'doc_input_ids':      doc_ids,
            'doc_attention_mask': doc_masks,
        }


class DocumentDataset(Dataset):
    def __init__(self, doc_path, tokenizer, max_length=512):
        self.docs      = load_jsonl(doc_path)
        self.tokenizer = tokenizer
        self.max_len   = max_length

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc  = self.docs[idx]
        text = doc.get('title', '') + ' ' + doc.get('text', '')
        enc  = self.tokenizer(text, max_length=self.max_len,
                              padding='max_length', truncation=True, return_tensors='pt')
        return {
            'doc_id':         doc['id'],
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }


def collate_fn(batch):
    return {
        'query_id':           [b['query_id'] for b in batch],
        'sentence_ids':       torch.stack([b['sentence_ids']       for b in batch]),
        'sentence_masks':     torch.stack([b['sentence_masks']     for b in batch]),
        'doc_input_ids':      torch.stack([b['doc_input_ids']      for b in batch]),
        'doc_attention_mask': torch.stack([b['doc_attention_mask'] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class QueryEncoder(nn.Module):
    def __init__(self, model_name, gru_hidden=256, sentence_dropout=0.2, proj_dim=512):
        super().__init__()
        self.encoder          = AutoModel.from_pretrained(model_name)
        bert_hidden           = self.encoder.config.hidden_size
        self.gru              = nn.GRU(input_size=bert_hidden, hidden_size=gru_hidden, batch_first=True)
        self.attention        = nn.Linear(gru_hidden * 2, 1)
        self.proj             = nn.Linear(gru_hidden, proj_dim)
        self.sentence_dropout = sentence_dropout

    def forward(self, sentence_ids, sentence_masks):
        B, N, L = sentence_ids.shape
        cls = self.encoder(
            input_ids=sentence_ids.view(B * N, L),
            attention_mask=sentence_masks.view(B * N, L),
        ).last_hidden_state[:, 0, :].view(B, N, -1)
        if self.training and self.sentence_dropout > 0:
            mask = (torch.rand(B, N, 1, device=cls.device) > self.sentence_dropout).float()
            cls  = cls * mask
        all_hidden, _ = self.gru(cls)
        final   = all_hidden[:, -1, :].unsqueeze(1).expand(-1, N, -1)
        concat  = torch.cat([all_hidden, final], dim=-1)
        weights = torch.softmax(self.attention(concat), dim=1)
        return self.proj((weights * all_hidden).sum(dim=1))


class DocEncoder(nn.Module):
    def __init__(self, model_name, proj_dim=512):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj    = nn.Linear(self.encoder.config.hidden_size, proj_dim)

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(input_ids=input_ids,
                           attention_mask=attention_mask).last_hidden_state[:, 0, :]
        return self.proj(cls)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def contrastive_loss(q_vecs, d_vecs, temperature=0.05):
    scores = torch.matmul(q_vecs, d_vecs.T) / temperature
    labels = torch.arange(scores.shape[0], device=scores.device)
    return nn.CrossEntropyLoss()(scores, labels)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(ranked, relevant, k=1000):
    rel_set = set(relevant)
    dcg  = sum(1.0 / np.log2(i + 2) for i, d in enumerate(ranked[:k]) if d in rel_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked, relevant, k=100):
    rel_set = set(relevant)
    return len(rel_set & set(ranked[:k])) / len(rel_set) if rel_set else 0.0


def build_index(doc_enc, doc_loader, device):
    doc_enc.eval()
    all_vecs, all_ids = [], []
    with torch.no_grad():
        for batch in tqdm(doc_loader, desc='  Indexing docs', leave=False):
            vecs = doc_enc(batch['input_ids'].to(device),
                           batch['attention_mask'].to(device))
            all_vecs.append(vecs.cpu().numpy())
            all_ids.extend(batch['doc_id'])
    all_vecs = np.vstack(all_vecs).astype('float32')
    faiss.normalize_L2(all_vecs)
    index = faiss.IndexFlatIP(all_vecs.shape[1])
    index.add(all_vecs)
    return index, all_ids


# ---------------------------------------------------------------------------
# Multi-GPU helpers
# ---------------------------------------------------------------------------

def _setup_device(gpu_ids):
    if gpu_ids and torch.cuda.is_available():
        return torch.device(f'cuda:{gpu_ids[0]}')
    return torch.device('cpu')


def _make_models(sentence_dropout, gpu_ids):
    device    = _setup_device(gpu_ids)
    query_enc = QueryEncoder(MODEL_NAME, sentence_dropout=sentence_dropout,
                              proj_dim=PROJ_DIM).to(device)
    doc_enc   = DocEncoder(MODEL_NAME, proj_dim=PROJ_DIM).to(device)
    if gpu_ids and len(gpu_ids) > 1 and torch.cuda.is_available():
        print(f'  DataParallel across GPUs: {gpu_ids}')
        query_enc = nn.DataParallel(query_enc, device_ids=gpu_ids)
        doc_enc   = nn.DataParallel(doc_enc,   device_ids=gpu_ids)
    return query_enc, doc_enc, device


def _unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(query_enc, doc_enc, dev_loader, doc_loader, qrels, device,
                    k_ndcg=1000, k_recall=100):
    index, idx_to_docid = build_index(doc_enc, doc_loader, device)
    query_enc.eval()
    ndcg_scores, recall_scores, per_query = [], [], {}
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc='  Evaluating', leave=False):
            q_vecs = query_enc(
                batch['sentence_ids'].to(device),
                batch['sentence_masks'].to(device),
            ).cpu().numpy().astype('float32')
            faiss.normalize_L2(q_vecs)
            _, doc_indices = index.search(q_vecs, k_ndcg)
            for i, qid in enumerate(batch['query_id']):
                ranked   = [idx_to_docid[j] for j in doc_indices[i]]
                relevant = qrels.get(qid, [])
                ndcg     = ndcg_at_k(ranked, relevant, k=k_ndcg)
                recall   = recall_at_k(ranked, relevant, k=k_recall)
                ndcg_scores.append(ndcg)
                recall_scores.append(recall)
                per_query[qid] = {'ndcg@1000': ndcg, 'r@100': recall}
    return {
        'ndcg@1000': float(np.mean(ndcg_scores)),
        'r@100':     float(np.mean(recall_scores)),
        'per_query': per_query,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _run_one_loader(query_enc, doc_enc, loader, optimizer, scheduler, device, desc):
    query_enc.train()
    doc_enc.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc=desc, leave=False):
        q_vecs = query_enc(batch['sentence_ids'].to(device),
                           batch['sentence_masks'].to(device))
        d_vecs = doc_enc(batch['doc_input_ids'].to(device),
                         batch['doc_attention_mask'].to(device))
        loss = contrastive_loss(q_vecs, d_vecs)
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(query_enc.parameters()) + list(doc_enc.parameters()), max_norm=1.0
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(loader)


def train_and_eval(sentence_dropout, train_loader, dev_loader, doc_loader,
                    qrels, gpu_ids, epochs=10, lr=2e-5, reddit_loader=None,
                    eval_every=5):
    query_enc, doc_enc, device = _make_models(sentence_dropout, gpu_ids)

    optimizer    = AdamW(list(query_enc.parameters()) + list(doc_enc.parameters()),
                         lr=lr, weight_decay=0.01)
    total_steps  = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                             total_iters=warmup_steps)

    best_ndcg         = 0.0
    best_metrics      = {}
    best_query_state  = None
    best_doc_state    = None

    for epoch in range(1, epochs + 1):
        trec_loss = _run_one_loader(query_enc, doc_enc, train_loader, optimizer,
                                     scheduler, device, desc=f'  Epoch {epoch} [TREC]')
        log = {'epoch': epoch, 'trec_loss': trec_loss}

        if reddit_loader is not None:
            reddit_loss = _run_one_loader(query_enc, doc_enc, reddit_loader, optimizer,
                                           scheduler, device,
                                           desc=f'  Epoch {epoch} [Reddit]')
            log['reddit_loss'] = reddit_loss
            print(f'  Epoch {epoch}/{epochs}  trec_loss={trec_loss:.4f}  reddit_loss={reddit_loss:.4f}')
        else:
            print(f'  Epoch {epoch}/{epochs}  trec_loss={trec_loss:.4f}')

        # dev evaluation every eval_every epochs and always on the last epoch
        if epoch % eval_every == 0 or epoch == epochs:
            metrics = evaluate_model(query_enc, doc_enc, dev_loader, doc_loader,
                                      qrels, device)
            log['ndcg@1000'] = metrics['ndcg@1000']
            log['r@100']     = metrics['r@100']
            print(f'  [eval]  NDCG@1000={metrics["ndcg@1000"]:.4f}  R@100={metrics["r@100"]:.4f}', end='')
            if metrics['ndcg@1000'] > best_ndcg:
                best_ndcg        = metrics['ndcg@1000']
                best_metrics     = metrics
                # snapshot best weights (CPU copy to avoid GPU memory pressure)
                best_query_state = {k: v.cpu().clone() for k, v in _unwrap(query_enc).state_dict().items()}
                best_doc_state   = {k: v.cpu().clone() for k, v in _unwrap(doc_enc).state_dict().items()}
                print(f'  ← best', end='')
            print()

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(log)

    # restore best weights into the model before returning
    if best_query_state is not None:
        _unwrap(query_enc).load_state_dict(best_query_state)
        _unwrap(doc_enc).load_state_dict(best_doc_state)
    else:
        # eval_every never triggered — evaluate final model
        best_metrics = evaluate_model(query_enc, doc_enc, dev_loader, doc_loader,
                                       qrels, device)

    return query_enc, doc_enc, device, best_metrics


# ---------------------------------------------------------------------------
# Significance test
# ---------------------------------------------------------------------------

def _wilcoxon_ndcg(results_a, results_b, qids):
    scores_a = [results_a['per_query'][q]['ndcg@1000'] for q in qids if q in results_a['per_query']]
    scores_b = [results_b['per_query'][q]['ndcg@1000'] for q in qids if q in results_b['per_query']]
    stat, p  = wilcoxon(scores_a, scores_b)
    return stat, p


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_results_table(results):
    print(f'\n{"=" * 70}')
    print('  GRU Sentence Dropout Sweep Results  (RQ2)')
    print(f'{"=" * 70}')
    print(f'  {"Run":<22} {"Dropout":>9} {"NDCG@1000":>12} {"R@100":>8}')
    print(f'  {"-" * 55}')
    best_key = max(results, key=lambda k: results[k]['ndcg@1000'])
    for run_key, r in results.items():
        marker = ' *' if run_key == best_key else '  '
        print(f'{marker} {run_key:<22} {r["sentence_dropout"]:>9.1f} '
              f'{r["ndcg@1000"]:>12.4f} {r["r@100"]:>8.4f}')
    print(f'\n  * best: dropout={results[best_key]["sentence_dropout"]:.1f}  '
          f'NDCG@1000={results[best_key]["ndcg@1000"]:.4f}')
    print(f'{"=" * 70}\n')


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_dropout_sweep(data_cfg, epochs=10, batch_size=16, lr=2e-5,
                       dropout_rates=None, checkpoint_dir='checkpoints/rq2_dropout_sweep',
                       wandb_project=None, reddit_query_path=None,
                       gpu_ids=None, eval_every=5):
    if dropout_rates is None:
        dropout_rates = DROPOUT_RATES
    if gpu_ids is None:
        gpu_ids = []

    qrels     = load_qrels(data_cfg['dev_qrel_path'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results   = {}

    for dropout in dropout_rates:
        run_key = f'dropout_{dropout:.1f}'

        print(f'\n{"=" * 60}')
        print(f'  Run              : {run_key}')
        print(f'  Model            : {MODEL_NAME}')
        print(f'  proj_dim         : {PROJ_DIM}')
        print(f'  sentence_dropout : {dropout}')
        print(f'  GPUs             : {gpu_ids if gpu_ids else "CPU/auto"}')
        print(f'  Data             : TREC' + (' + Reddit' if reddit_query_path else ''))
        print(f'{"=" * 60}')

        ckpt_path = os.path.join(checkpoint_dir, run_key)
        meta_path = os.path.join(ckpt_path, 'meta.json')

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                saved_meta = json.load(f)
            if saved_meta.get('epochs_completed') == epochs and 'ndcg@1000' in saved_meta:
                print(f'  Already completed — skipping.')
                results[run_key] = {
                    'sentence_dropout': dropout,
                    'ndcg@1000':        saved_meta['ndcg@1000'],
                    'r@100':            saved_meta['r@100'],
                    'per_query':        saved_meta.get('per_query', {}),
                }
                continue
            print(f'  Incomplete checkpoint — retraining.')

        if WANDB_AVAILABLE and wandb_project:
            wandb.init(
                project=wandb_project,
                name=run_key,
                config={
                    'model_name':       MODEL_NAME,
                    'tokenizer_type':   TOKENIZER_TYPE,
                    'proj_dim':         PROJ_DIM,
                    'sentence_dropout': dropout,
                    'epochs':           epochs,
                    'batch_size':       batch_size,
                    'lr':               lr,
                    'eval_every':       eval_every,
                    'data':             'TREC+Reddit' if reddit_query_path else 'TREC',
                },
                reinit=True,
            )

        train_dataset = TRECDataset(data_cfg['train_query_path'], data_cfg['doc_path'],
                                     data_cfg['train_qrel_path'], tokenizer)
        dev_dataset   = TRECDataset(data_cfg['dev_query_path'],   data_cfg['doc_path'],
                                     data_cfg['dev_qrel_path'],   tokenizer)
        doc_dataset   = DocumentDataset(data_cfg['doc_path'], tokenizer)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                   collate_fn=collate_fn, num_workers=2)
        dev_loader   = DataLoader(dev_dataset,   batch_size=batch_size, shuffle=False,
                                   collate_fn=collate_fn, num_workers=2)
        doc_loader   = DataLoader(doc_dataset,   batch_size=64, shuffle=False, num_workers=2)

        reddit_loader = None
        if reddit_query_path:
            reddit_dataset = RedditDataset(reddit_query_path, data_cfg['doc_path'], tokenizer)
            reddit_loader  = DataLoader(reddit_dataset, batch_size=batch_size, shuffle=True,
                                         collate_fn=collate_fn, num_workers=2)
            print(f'  Reddit queries : {len(reddit_dataset)}')
        print(f'  TREC train     : {len(train_dataset)}')
        print(f'  TREC dev       : {len(dev_dataset)}')

        query_enc, doc_enc, device, metrics = train_and_eval(
            dropout, train_loader, dev_loader, doc_loader, qrels,
            gpu_ids=gpu_ids, epochs=epochs, lr=lr,
            reddit_loader=reddit_loader, eval_every=eval_every,
        )

        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(_unwrap(query_enc).state_dict(), os.path.join(ckpt_path, 'query_enc.pt'))
        torch.save(_unwrap(doc_enc).state_dict(),   os.path.join(ckpt_path, 'doc_enc.pt'))

        meta = {
            'model_name':       MODEL_NAME,
            'tokenizer_type':   TOKENIZER_TYPE,
            'proj_dim':         PROJ_DIM,
            'sentence_dropout': dropout,
            'epochs_completed': epochs,
            'ndcg@1000':        metrics['ndcg@1000'],
            'r@100':            metrics['r@100'],
            'per_query':        metrics['per_query'],
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        results[run_key] = {'sentence_dropout': dropout, **metrics}

        print(f'  NDCG@1000 : {metrics["ndcg@1000"]:.4f}')
        print(f'  R@100     : {metrics["r@100"]:.4f}')

        if WANDB_AVAILABLE and wandb_project and wandb.run is not None:
            wandb.log({'final_ndcg@1000': metrics['ndcg@1000'],
                       'final_r@100':     metrics['r@100']})
            wandb.finish()

    # summary JSON (no per_query to keep it small)
    summary = {k: {kk: vv for kk, vv in v.items() if kk != 'per_query'}
               for k, v in results.items()}
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_path = os.path.join(checkpoint_dir, 'rq2_results.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nSummary saved to {out_path}')

    print_results_table(summary)

    # pairwise Wilcoxon
    all_qids = list(qrels.keys())
    run_keys = list(results.keys())
    if len(run_keys) > 1:
        print('  Pairwise Wilcoxon signed-rank tests (per-query NDCG@1000):')
        for i in range(len(run_keys)):
            for j in range(i + 1, len(run_keys)):
                ka, kb = run_keys[i], run_keys[j]
                if results[ka].get('per_query') and results[kb].get('per_query'):
                    stat, p = _wilcoxon_ndcg(results[ka], results[kb], all_qids)
                    sig = 'significant (p<0.05)' if p < 0.05 else 'not significant'
                    print(f'    {ka} vs {kb}: stat={stat:.1f}  p={p:.4f}  -> {sig}')

    return results


# ---------------------------------------------------------------------------
# Eval-only
# ---------------------------------------------------------------------------

def evaluate_saved_checkpoints(data_cfg, dropout_rates=None,
                                 checkpoint_dir='checkpoints/rq2_dropout_sweep',
                                 gpu_ids=None):
    if dropout_rates is None:
        dropout_rates = DROPOUT_RATES
    if gpu_ids is None:
        gpu_ids = []

    device    = _setup_device(gpu_ids)
    qrels     = load_qrels(data_cfg['dev_qrel_path'])
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    results   = {}

    for dropout in dropout_rates:
        run_key   = f'dropout_{dropout:.1f}'
        ckpt_path = os.path.join(checkpoint_dir, run_key)
        meta_path = os.path.join(ckpt_path, 'meta.json')

        if not os.path.exists(os.path.join(ckpt_path, 'query_enc.pt')):
            print(f'  [{run_key}] No checkpoint — skipping.')
            continue

        print(f'\n  Evaluating {run_key}')
        dev_dataset = TRECDataset(data_cfg['dev_query_path'], data_cfg['doc_path'],
                                   data_cfg['dev_qrel_path'], tokenizer)
        doc_dataset = DocumentDataset(data_cfg['doc_path'], tokenizer)
        dev_loader  = DataLoader(dev_dataset, batch_size=16, shuffle=False,
                                  collate_fn=collate_fn, num_workers=2)
        doc_loader  = DataLoader(doc_dataset, batch_size=64, shuffle=False, num_workers=2)

        query_enc = QueryEncoder(MODEL_NAME, sentence_dropout=dropout,
                                  proj_dim=PROJ_DIM).to(device)
        doc_enc   = DocEncoder(MODEL_NAME, proj_dim=PROJ_DIM).to(device)
        query_enc.load_state_dict(torch.load(os.path.join(ckpt_path, 'query_enc.pt'),
                                              map_location=device))
        doc_enc.load_state_dict(  torch.load(os.path.join(ckpt_path, 'doc_enc.pt'),
                                              map_location=device))
        if gpu_ids and len(gpu_ids) > 1 and torch.cuda.is_available():
            query_enc = nn.DataParallel(query_enc, device_ids=gpu_ids)
            doc_enc   = nn.DataParallel(doc_enc,   device_ids=gpu_ids)

        metrics = evaluate_model(query_enc, doc_enc, dev_loader, doc_loader, qrels, device)
        results[run_key] = {'sentence_dropout': dropout, **metrics}

        if os.path.exists(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            meta.update({'ndcg@1000': metrics['ndcg@1000'], 'r@100': metrics['r@100']})
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2)

        print(f'    NDCG@1000 : {metrics["ndcg@1000"]:.4f}')
        print(f'    R@100     : {metrics["r@100"]:.4f}')

    summary = {k: {kk: vv for kk, vv in v.items() if kk != 'per_query'}
               for k, v in results.items()}
    out_path = os.path.join(checkpoint_dir, 'rq2_results.json')
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nResults saved to {out_path}')
    print_results_table(summary)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='RQ2 — sentence dropout sweep on bert-base-cased / proj_dim=512'
    )
    parser.add_argument('--eval-only',    action='store_true')
    parser.add_argument('--epochs',       type=int,   default=10)
    parser.add_argument('--batch-size',   type=int,   default=16)
    parser.add_argument('--lr',           type=float, default=2e-5)
    parser.add_argument('--eval-every',   type=int,   default=5,
                        help='Evaluate on dev set and log to W&B every N epochs (default 5)')
    parser.add_argument('--wandb-project',     type=str, default='tot-rq2-dropout-sweep')
    parser.add_argument('--checkpoint-dir',    type=str, default='checkpoints/rq2_dropout_sweep')
    parser.add_argument('--reddit-query-path', type=str, default='data/reddit/queries.jsonl')
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated GPU indices with no spaces, e.g. --gpus 0,1')
    args = parser.parse_args()

    # parse --gpus "0,1" → [0, 1]
    gpu_ids = [int(g.strip()) for g in args.gpus.split(',')] if args.gpus else []

    if gpu_ids:
        print(f'Using GPUs : {gpu_ids}  (primary: cuda:{gpu_ids[0]})')
    else:
        print(f'Using      : {"cuda:0" if torch.cuda.is_available() else "CPU"}')

    data_cfg = {
        'train_query_path': 'data/trec/train_queries.jsonl',
        'train_qrel_path':  'data/trec/train_qrels.txt',
        'dev_query_path':   'data/trec/dev_queries.jsonl',
        'dev_qrel_path':    'data/trec/dev_qrels.txt',
        'doc_path':         'data/trec/docs.jsonl',
    }

    if args.eval_only:
        evaluate_saved_checkpoints(data_cfg, checkpoint_dir=args.checkpoint_dir,
                                    gpu_ids=gpu_ids)
    else:
        run_dropout_sweep(
            data_cfg,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            eval_every=args.eval_every,
            dropout_rates=DROPOUT_RATES,
            checkpoint_dir=args.checkpoint_dir,
            wandb_project=args.wandb_project,
            reddit_query_path=args.reddit_query_path,
            gpu_ids=gpu_ids,
        )
