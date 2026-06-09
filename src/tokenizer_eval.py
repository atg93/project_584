"""
Tokenizer comparison experiment — addresses RQ1:
  "Does changing the tokenizer affect retrieval performance on the ToT task?"

Four configurations evaluated:
  - WordPiece (uncased) : bert-base-uncased       (BERT)
  - WordPiece (cased)   : bert-base-cased         (BERT, preserves proper-noun casing)
  - BPE                 : roberta-base            (RoBERTa)
  - Unigram             : albert-base-v2          (ALBERT / SentencePiece)

Each tokenizer is trained with projection dimensions swept over {256, 512, 768},
giving 16 total runs. Every run is logged as a separate W&B run.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from dataset import load_jsonl, load_qrels
from evaluate import ndcg_at_k, recall_at_k, build_index
from tot_retrieval import contrastive_loss


# ---------------------------------------------------------------------------
# Experiment grid
# ---------------------------------------------------------------------------

TOKENIZER_CONFIGS = {
    'wordpiece': {
        'model_name':     'bert-base-uncased',
        'tokenizer_type': 'WordPiece (uncased)',
    },
    'wordpiece_cased': {
        'model_name':     'bert-base-cased',
        'tokenizer_type': 'WordPiece (cased)',
    },
    'bpe': {
        'model_name':     'roberta-base',
        'tokenizer_type': 'BPE (Byte Pair Encoding)',
    },
    'unigram': {
        'model_name':     'albert-base-v2',
        'tokenizer_type': 'Unigram (SentencePiece)',
    },
}

PROJ_DIMS = [768, 512, 256]


# ---------------------------------------------------------------------------
# Encoders — proj_dim is now a first-class parameter
# ---------------------------------------------------------------------------

class GenericQueryEncoder(nn.Module):
    """
    Any HuggingFace encoder → GRU → query-aware attention → proj_dim embedding.
    Hidden sizes are derived from the loaded model config, not hardcoded.
    """

    def __init__(self, model_name, gru_hidden=768, sentence_dropout=0.2, proj_dim=256):
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
            attention_mask=sentence_masks.view(B * N, L)
        ).last_hidden_state[:, 0, :].view(B, N, -1)          # (B, N, bert_hidden)

        if self.training and self.sentence_dropout > 0:
            mask = (torch.rand(B, N, 1, device=cls.device) > self.sentence_dropout).float()
            cls  = cls * mask

        all_hidden, _ = self.gru(cls)                          # (B, N, gru_hidden)
        final  = all_hidden[:, -1, :].unsqueeze(1).expand(-1, N, -1)
        concat = torch.cat([all_hidden, final], dim=-1)        # (B, N, gru_hidden*2)
        weights = torch.softmax(self.attention(concat), dim=1) # (B, N, 1)
        query_vec = (weights * all_hidden).sum(dim=1)          # (B, gru_hidden)
        return self.proj(query_vec)                            # (B, proj_dim)


class GenericDocEncoder(nn.Module):
    def __init__(self, model_name, proj_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.proj    = nn.Linear(self.encoder.config.hidden_size, proj_dim)

    def forward(self, input_ids, attention_mask):
        cls = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]                          # (B, hidden)
        return self.proj(cls)                                  # (B, proj_dim)


# ---------------------------------------------------------------------------
# Dataset — shared across all tokenizer configs
# ---------------------------------------------------------------------------

class TokenizerToTDataset(Dataset):
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

    def _encode_sentences(self, sentences):
        enc = self.tokenizer(
            sentences,
            max_length=self.max_sent_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return enc['input_ids'], enc['attention_mask']

    def _encode_doc(self, doc):
        text = doc.get('title', '') + ' ' + doc.get('text', '')
        enc  = self.tokenizer(
            text,
            max_length=self.max_doc_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex        = self.examples[idx]
        sent_ids, sent_masks = self._encode_sentences(ex['sentences'])

        N = sent_ids.shape[0]
        if N < self.max_sentences:
            pad_ids   = torch.zeros(self.max_sentences - N, self.max_sent_len, dtype=torch.long)
            pad_masks = torch.zeros(self.max_sentences - N, self.max_sent_len, dtype=torch.long)
            sent_ids   = torch.cat([sent_ids,   pad_ids],   dim=0)
            sent_masks = torch.cat([sent_masks, pad_masks], dim=0)

        doc_ids, doc_masks = self._encode_doc(ex['doc'])

        return {
            'query_id':           ex['qid'],
            'sentence_ids':       sent_ids,
            'sentence_masks':     sent_masks,
            'doc_input_ids':      doc_ids,
            'doc_attention_mask': doc_masks,
        }


def collate_fn(batch):
    return {
        'query_id':           [b['query_id'] for b in batch],
        'sentence_ids':       torch.stack([b['sentence_ids']       for b in batch]),
        'sentence_masks':     torch.stack([b['sentence_masks']     for b in batch]),
        'doc_input_ids':      torch.stack([b['doc_input_ids']      for b in batch]),
        'doc_attention_mask': torch.stack([b['doc_attention_mask'] for b in batch]),
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
        enc  = self.tokenizer(
            text, max_length=self.max_len, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        return {
            'doc_id':         doc['id'],
            'input_ids':      enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
        }


class RedditTokenizerDataset(Dataset):
    """
    Reddit ToT dataset for tokenizer comparison experiments.
    Uses answer_id as the single positive document (same doc collection as TREC).
    Sentences are taken from the 'sentences' field or split from 'text' on '.'.
    """

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

    def _encode_sentences(self, sentences):
        enc = self.tokenizer(
            sentences,
            max_length=self.max_sent_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return enc['input_ids'], enc['attention_mask']

    def _encode_doc(self, doc):
        text = doc.get('title', '') + ' ' + doc.get('text', '')
        enc  = self.tokenizer(
            text,
            max_length=self.max_doc_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex       = self.examples[idx]
        sent_ids, sent_masks = self._encode_sentences(ex['sentences'])

        N = sent_ids.shape[0]
        if N < self.max_sentences:
            pad_ids   = torch.zeros(self.max_sentences - N, self.max_sent_len, dtype=torch.long)
            pad_masks = torch.zeros(self.max_sentences - N, self.max_sent_len, dtype=torch.long)
            sent_ids   = torch.cat([sent_ids,   pad_ids],   dim=0)
            sent_masks = torch.cat([sent_masks, pad_masks], dim=0)

        doc_ids, doc_masks = self._encode_doc(ex['doc'])

        return {
            'query_id':           ex['qid'],
            'sentence_ids':       sent_ids,
            'sentence_masks':     sent_masks,
            'doc_input_ids':      doc_ids,
            'doc_attention_mask': doc_masks,
        }


# ---------------------------------------------------------------------------
# Train one (tokenizer × proj_dim) configuration
# ---------------------------------------------------------------------------

def _run_one_loader(query_enc, doc_enc, loader, optimizer, scheduler, device, desc):
    """One pass over a dataloader — returns average loss."""
    query_enc.train()
    doc_enc.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc=desc, leave=False):
        q_vecs = query_enc(
            batch['sentence_ids'].to(device),
            batch['sentence_masks'].to(device)
        )
        d_vecs = doc_enc(
            batch['doc_input_ids'].to(device),
            batch['doc_attention_mask'].to(device)
        )
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


def train_config(model_name, train_loader, device, epochs=10, lr=2e-5,
                 proj_dim=256, reddit_loader=None):
    query_enc = GenericQueryEncoder(model_name, proj_dim=proj_dim).to(device)
    doc_enc   = GenericDocEncoder(model_name,   proj_dim=proj_dim).to(device)

    optimizer    = AdamW(
        list(query_enc.parameters()) + list(doc_enc.parameters()),
        lr=lr, weight_decay=0.01
    )
    total_steps  = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)

    for epoch in range(1, epochs + 1):
        trec_loss = _run_one_loader(
            query_enc, doc_enc, train_loader, optimizer, scheduler, device,
            desc=f'  Epoch {epoch} [TREC]'
        )
        log = {'trec_loss': trec_loss, 'epoch': epoch}

        if reddit_loader is not None:
            reddit_loss = _run_one_loader(
                query_enc, doc_enc, reddit_loader, optimizer, scheduler, device,
                desc=f'  Epoch {epoch} [Reddit]'
            )
            log['reddit_loss'] = reddit_loss
            print(f'  Epoch {epoch}/{epochs}  TREC loss: {trec_loss:.4f}  Reddit loss: {reddit_loss:.4f}')
        else:
            print(f'  Epoch {epoch}/{epochs}  TREC loss: {trec_loss:.4f}')

        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(log)

    return query_enc, doc_enc


# ---------------------------------------------------------------------------
# Evaluate one configuration
# ---------------------------------------------------------------------------

def evaluate_config(query_enc, doc_enc, dev_loader, dev_doc_loader,
                    qrels, device, k_ndcg=1000, k_recall=100):
    import faiss

    index, idx_to_docid = build_index(doc_enc, dev_doc_loader, device)
    query_enc.eval()

    ndcg_scores   = []
    recall_scores = []

    _debug_printed = False
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc='  Evaluating', leave=False):
            q_vecs = query_enc(
                batch['sentence_ids'].to(device),
                batch['sentence_masks'].to(device)
            ).cpu().numpy().astype('float32')

            faiss.normalize_L2(q_vecs)
            _, doc_indices = index.search(q_vecs, k_ndcg)

            for i, qid in enumerate(batch['query_id']):
                ranked      = [idx_to_docid[j] for j in doc_indices[i]]
                relevant    = qrels.get(qid, [])

                # --- one-time debug snapshot ---
                if not _debug_printed:
                    _debug_printed = True
                    print(f'\n[DEBUG] qid={repr(qid)}  in_qrels={qid in qrels}')
                    print(f'[DEBUG] relevant docs : {relevant[:5]}')
                    print(f'[DEBUG] top-5 ranked  : {ranked[:5]}')
                    print(f'[DEBUG] q_vec norm={float(np.linalg.norm(q_vecs[i])):.4f}  '
                          f'any_nan={bool(np.isnan(q_vecs[i]).any())}')
                    print(f'[DEBUG] qrels sample  : {list(qrels.items())[:3]}\n')
                # --------------------------------

                ndcg_scores.append(  ndcg_at_k(  ranked, relevant, k=k_ndcg))
                recall_scores.append(recall_at_k(ranked, relevant, k=k_recall))

    return {
        f'ndcg@{k_ndcg}': float(np.mean(ndcg_scores)),
        f'r@{k_recall}':   float(np.mean(recall_scores)),
    }


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_comparison_table(results):
    print(f'\n{"=" * 80}')
    print(f'  Tokenizer Comparison Results  (RQ1)')
    print(f'{"=" * 80}')
    print(f'  {"Run":<28} {"Tokenizer":<28} {"proj_dim":>8} {"NDCG@1000":>10} {"R@100":>8}')
    print(f'  {"-" * 76}')

    best_key = max(results, key=lambda k: results[k]['ndcg@1000'])
    for run_key, r in results.items():
        marker = ' *' if run_key == best_key else '  '
        print(f'{marker} {run_key:<28} {r["tokenizer_type"]:<28} {r["proj_dim"]:>8} '
              f'{r["ndcg@1000"]:>10.4f} {r["r@100"]:>8.4f}')

    print(f'\n  * best run by NDCG@1000: {results[best_key]["tokenizer_type"]} '
          f'(proj_dim={results[best_key]["proj_dim"]}, NDCG@1000={results[best_key]["ndcg@1000"]:.4f})')
    print(f'{"=" * 80}\n')


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

def run_tokenizer_comparison(data_cfg, epochs=10, batch_size=16, lr=2e-5,
                              proj_dims=None,
                              checkpoint_dir='checkpoints/tokenizer_comparison',
                              wandb_project=None,
                              reddit_query_path=None):
    """
    Trains and evaluates all tokenizer × proj_dim combinations.
    Each combination is one W&B run (if wandb_project is given).
    Saves all results to tokenizer_results.json.

    data_cfg keys:
        train_query_path, train_qrel_path,
        dev_query_path,   dev_qrel_path,
        doc_path
    """
    if proj_dims is None:
        proj_dims = PROJ_DIMS

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qrels   = load_qrels(data_cfg['dev_qrel_path'])
    results = {}

    for proj_dim in proj_dims:
        for key, cfg in TOKENIZER_CONFIGS.items():
            run_key = f'{key}_d{proj_dim}'

            print(f'\n{"=" * 60}')
            print(f'  Run       : {run_key}')
            print(f'  Tokenizer : {cfg["tokenizer_type"]}')
            print(f'  Model     : {cfg["model_name"]}')
            print(f'  proj_dim  : {proj_dim}')
            print(f'{"=" * 60}')

            # skip only if training fully completed (meta.json records epochs_completed)
            ckpt_path   = os.path.join(checkpoint_dir, run_key)
            result_path = os.path.join(checkpoint_dir, 'tokenizer_results.json')
            meta_path   = os.path.join(ckpt_path, 'meta.json')
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    saved_meta = json.load(f)
                if saved_meta.get('epochs_completed') == epochs:
                    print(f'  Already completed — skipping. (delete {ckpt_path} to re-run)')
                    # prefer metrics from meta.json (written atomically after eval);
                    # fall back to the global results file for older checkpoints
                    if f'ndcg@1000' in saved_meta:
                        results[run_key] = {
                            'model_name':     saved_meta['model_name'],
                            'tokenizer_type': saved_meta['tokenizer_type'],
                            'proj_dim':       saved_meta['proj_dim'],
                            'ndcg@1000':      saved_meta['ndcg@1000'],
                            'r@100':          saved_meta['r@100'],
                        }
                    elif os.path.exists(result_path):
                        with open(result_path) as f:
                            saved = json.load(f)
                        if run_key in saved:
                            results[run_key] = saved[run_key]
                    continue
                else:
                    print(f'  Incomplete checkpoint (epochs_completed='
                          f'{saved_meta.get("epochs_completed")}, target={epochs}) — retraining.')

            # --- W&B run ---
            if WANDB_AVAILABLE and wandb_project:
                wandb.init(
                    project=wandb_project,
                    name=run_key,
                    config={
                        'tokenizer_key':  key,
                        'tokenizer_type': cfg['tokenizer_type'],
                        'model_name':     cfg['model_name'],
                        'proj_dim':       proj_dim,
                        'epochs':         epochs,
                        'batch_size':     batch_size,
                        'lr':             lr,
                    },
                    reinit=True,
                )

            tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

            train_dataset = TokenizerToTDataset(
                data_cfg['train_query_path'], data_cfg['doc_path'],
                data_cfg['train_qrel_path'], tokenizer
            )
            dev_dataset = TokenizerToTDataset(
                data_cfg['dev_query_path'], data_cfg['doc_path'],
                data_cfg['dev_qrel_path'], tokenizer
            )
            doc_dataset = DocumentDataset(data_cfg['doc_path'], tokenizer)

            train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True,  collate_fn=collate_fn, num_workers=2)
            dev_loader   = DataLoader(dev_dataset,   batch_size=batch_size,
                                      shuffle=False, collate_fn=collate_fn, num_workers=2)
            doc_loader   = DataLoader(doc_dataset,   batch_size=64,
                                      shuffle=False, num_workers=2)

            reddit_loader = None
            if reddit_query_path:
                reddit_dataset = RedditTokenizerDataset(
                    reddit_query_path, data_cfg['doc_path'], tokenizer
                )
                reddit_loader = DataLoader(reddit_dataset, batch_size=batch_size,
                                           shuffle=True, collate_fn=collate_fn, num_workers=2)
                print(f'  Reddit queries: {len(reddit_dataset)}')

            # RoBERTa (BPE) needs a lower LR to prevent catastrophic forgetting on small data
            effective_lr = 5e-6 if 'roberta' in cfg['model_name'] else lr

            # train
            query_enc, doc_enc = train_config(
                cfg['model_name'], train_loader, device,
                epochs=epochs, lr=effective_lr, proj_dim=proj_dim, reddit_loader=reddit_loader
            )

            # save checkpoint + meta
            os.makedirs(ckpt_path, exist_ok=True)
            torch.save(query_enc.state_dict(), os.path.join(ckpt_path, 'query_enc.pt'))
            torch.save(doc_enc.state_dict(),   os.path.join(ckpt_path, 'doc_enc.pt'))
            meta = {
                'model_name':      cfg['model_name'],
                'tokenizer_type':  cfg['tokenizer_type'],
                'proj_dim':        proj_dim,
                'epochs_completed': epochs,
            }
            with open(os.path.join(ckpt_path, 'meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)

            # evaluate
            metrics = evaluate_config(query_enc, doc_enc, dev_loader, doc_loader, qrels, device)

            # persist metrics into meta immediately so they survive even if the
            # global tokenizer_results.json write below is interrupted
            meta.update(metrics)
            with open(os.path.join(ckpt_path, 'meta.json'), 'w') as f:
                json.dump(meta, f, indent=2)

            results[run_key] = {
                **cfg,
                'proj_dim':  proj_dim,
                **metrics,
            }

            print(f'  NDCG@1000 : {metrics["ndcg@1000"]:.4f}')
            print(f'  R@100     : {metrics["r@100"]:.4f}')

            # log final metrics and close W&B run
            if WANDB_AVAILABLE and wandb_project and wandb.run is not None:
                wandb.log({
                    'ndcg@1000': metrics['ndcg@1000'],
                    'r@100':     metrics['r@100'],
                })
                wandb.finish()

    # persist all results
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'tokenizer_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print_comparison_table(results)
    return results


# ---------------------------------------------------------------------------
# Evaluate saved checkpoints without retraining
# ---------------------------------------------------------------------------

def evaluate_saved_checkpoints(data_cfg, proj_dims=None,
                                checkpoint_dir='checkpoints/tokenizer_comparison'):
    """
    Loads saved query/doc encoder checkpoints for every (tokenizer x proj_dim)
    combination and re-evaluates on the dev set. Useful after training completes
    or if you want to re-run metrics without retraining.
    """
    if proj_dims is None:
        proj_dims = PROJ_DIMS

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qrels   = load_qrels(data_cfg['dev_qrel_path'])
    results = {}

    for proj_dim in proj_dims:
        for key, cfg in TOKENIZER_CONFIGS.items():
            run_key   = f'{key}_d{proj_dim}'
            ckpt_path = os.path.join(checkpoint_dir, run_key)

            meta_path = os.path.join(ckpt_path, 'meta.json')
            if not os.path.exists(os.path.join(ckpt_path, 'query_enc.pt')):
                print(f'  [{run_key}] No checkpoint found — skipping.')
                continue
            if not os.path.exists(meta_path):
                print(f'  [{run_key}] No meta.json — checkpoint may be incomplete, skipping.')
                continue
            with open(meta_path) as f:
                ckpt_meta = json.load(f)

            print(f'\n  Evaluating {run_key}  ({cfg["tokenizer_type"]}, proj_dim={proj_dim})')

            tokenizer = AutoTokenizer.from_pretrained(cfg['model_name'])

            dev_dataset = TokenizerToTDataset(
                data_cfg['dev_query_path'], data_cfg['doc_path'],
                data_cfg['dev_qrel_path'], tokenizer
            )
            doc_dataset = DocumentDataset(data_cfg['doc_path'], tokenizer)

            dev_loader = DataLoader(dev_dataset, batch_size=16,
                                    shuffle=False, collate_fn=collate_fn, num_workers=2)
            doc_loader = DataLoader(doc_dataset, batch_size=64,
                                    shuffle=False, num_workers=2)

            query_enc = GenericQueryEncoder(cfg['model_name'], proj_dim=proj_dim).to(device)
            doc_enc   = GenericDocEncoder(cfg['model_name'],   proj_dim=proj_dim).to(device)
            query_enc.load_state_dict(torch.load(os.path.join(ckpt_path, 'query_enc.pt'),
                                                  map_location=device))
            doc_enc.load_state_dict(  torch.load(os.path.join(ckpt_path, 'doc_enc.pt'),
                                                  map_location=device))

            metrics = evaluate_config(query_enc, doc_enc, dev_loader, doc_loader, qrels, device)
            results[run_key] = {**cfg, 'proj_dim': proj_dim, **metrics}

            ckpt_meta.update(metrics)
            with open(meta_path, 'w') as f:
                json.dump(ckpt_meta, f, indent=2)

            print(f'    NDCG@1000 : {metrics["ndcg@1000"]:.4f}')
            print(f'    R@100     : {metrics["r@100"]:.4f}')

    out_path = os.path.join(checkpoint_dir, 'tokenizer_results.json')
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nResults saved to {out_path}')

    print_comparison_table(results)
    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-only', action='store_true',
                        help='Skip training — evaluate saved checkpoints only')
    args = parser.parse_args()

    data_cfg = {
        'train_query_path': 'data/trec/train_queries.jsonl',
        'train_qrel_path':  'data/trec/train_qrels.txt',
        'dev_query_path':   'data/trec/dev_queries.jsonl',
        'dev_qrel_path':    'data/trec/dev_qrels.txt',
        'doc_path':         'data/trec/docs.jsonl',
    }

    if args.eval_only:
        evaluate_saved_checkpoints(data_cfg, proj_dims=PROJ_DIMS)
    else:
        run_tokenizer_comparison(
            data_cfg,
            epochs=3,
            batch_size=16,
            lr=2e-5,
            proj_dims=PROJ_DIMS,          # [768, 512, 256]
            wandb_project='tot-tokenizer-sweep',
            reddit_query_path='data/reddit/queries.jsonl',
        )
