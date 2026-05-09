"""
Tokenizer comparison experiment — addresses RQ1:
  "Does changing the tokenizer affect retrieval performance on the ToT task?"

Three configurations evaluated:
  - WordPiece : bert-base-uncased       (BERT)
  - BPE       : roberta-base            (RoBERTa)
  - Unigram   : albert-base-v2          (ALBERT / SentencePiece)
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

from dataset import load_jsonl, load_qrels
from evaluate import ndcg_at_k, recall_at_k, build_index
from tot_retrieval import contrastive_loss


# ---------------------------------------------------------------------------
# Tokenizer configurations
# ---------------------------------------------------------------------------

TOKENIZER_CONFIGS = {
    'wordpiece': {
        'model_name': 'bert-base-uncased',
        'tokenizer_type': 'WordPiece',
        'hidden_size': 768,
    },
    'bpe': {
        'model_name': 'roberta-base',
        'tokenizer_type': 'BPE (Byte Pair Encoding)',
        'hidden_size': 768,
    },
    'unigram': {
        'model_name': 'albert-base-v2',
        'tokenizer_type': 'Unigram (SentencePiece)',
        'hidden_size': 768,
    },
}


# ---------------------------------------------------------------------------
# Generic encoder — works with BERT, RoBERTa, ALBERT via AutoModel
# ---------------------------------------------------------------------------

class GenericQueryEncoder(nn.Module):
    """
    BERT/RoBERTa/ALBERT → GRU → query-aware attention → query vector.
    Uses AutoModel so any HuggingFace encoder can be plugged in.
    """

    def __init__(self, model_name, hidden_size=768, gru_hidden=768, sentence_dropout=0.2):
        super().__init__()
        self.encoder         = AutoModel.from_pretrained(model_name)
        self.gru             = nn.GRU(input_size=hidden_size, hidden_size=gru_hidden, batch_first=True)
        self.attention       = nn.Linear(gru_hidden * 2, 1)
        self.sentence_dropout = sentence_dropout

    def forward(self, sentence_ids, sentence_masks):
        B, N, L = sentence_ids.shape

        cls = self.encoder(
            input_ids=sentence_ids.view(B * N, L),
            attention_mask=sentence_masks.view(B * N, L)
        ).last_hidden_state[:, 0, :].view(B, N, -1)  # (B, N, hidden)

        if self.training and self.sentence_dropout > 0:
            mask = (torch.rand(B, N, 1, device=cls.device) > self.sentence_dropout).float()
            cls  = cls * mask

        all_hidden, _ = self.gru(cls)                                          # (B, N, gru_hidden)
        final  = all_hidden[:, -1, :].unsqueeze(1).expand(-1, N, -1)          # (B, N, gru_hidden)
        concat = torch.cat([all_hidden, final], dim=-1)                        # (B, N, gru_hidden*2)
        weights = torch.softmax(self.attention(concat), dim=1)                 # (B, N, 1)
        return (weights * all_hidden).sum(dim=1)                               # (B, gru_hidden)


class GenericDocEncoder(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        return self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ).last_hidden_state[:, 0, :]  # (B, hidden)


# ---------------------------------------------------------------------------
# Dataset for tokenizer comparison
# ---------------------------------------------------------------------------

class TokenizerToTDataset(Dataset):
    """
    Tokenizes queries sentence-by-sentence and documents using the given tokenizer.
    Shared structure for all three tokenizer configurations.
    """

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


# ---------------------------------------------------------------------------
# Train one configuration
# ---------------------------------------------------------------------------

def train_config(model_name, train_loader, device, epochs=10, lr=2e-5):
    hidden = TOKENIZER_CONFIGS[
        next(k for k, v in TOKENIZER_CONFIGS.items() if v['model_name'] == model_name)
    ]['hidden_size']

    query_enc = GenericQueryEncoder(model_name, hidden_size=hidden).to(device)
    doc_enc   = GenericDocEncoder(model_name).to(device)

    optimizer    = AdamW(
        list(query_enc.parameters()) + list(doc_enc.parameters()),
        lr=lr, weight_decay=0.01
    )
    total_steps  = epochs * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)

    for epoch in range(1, epochs + 1):
        query_enc.train()
        doc_enc.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f'  Epoch {epoch}', leave=False):
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
        print(f'  Epoch {epoch}/{epochs}  loss: {total_loss / len(train_loader):.4f}')

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
                ndcg_scores.append(  ndcg_at_k(  ranked, relevant, k=k_ndcg))
                recall_scores.append(recall_at_k(ranked, relevant, k=k_recall))

    return {
        f'ndcg@{k_ndcg}': float(np.mean(ndcg_scores)),
        f'r@{k_recall}':   float(np.mean(recall_scores)),
    }


# ---------------------------------------------------------------------------
# Document dataloader — tokenizer-specific
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Main comparison runner
# ---------------------------------------------------------------------------

def run_tokenizer_comparison(data_cfg, epochs=10, batch_size=16, lr=2e-5,
                              checkpoint_dir='checkpoints/tokenizer_comparison'):
    """
    Trains and evaluates all three tokenizer configurations.
    Saves results to tokenizer_results.json and prints a comparison table.

    data_cfg keys:
        train_query_path, train_qrel_path,
        dev_query_path,   dev_qrel_path,
        doc_path
    """
    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    qrels   = load_qrels(data_cfg['dev_qrel_path'])
    results = {}

    for key, cfg in TOKENIZER_CONFIGS.items():
        print(f'\n{"=" * 55}')
        print(f' Tokenizer: {cfg["tokenizer_type"]}')
        print(f' Model    : {cfg["model_name"]}')
        print(f'{"=" * 55}')

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

        # train
        query_enc, doc_enc = train_config(
            cfg['model_name'], train_loader, device, epochs=epochs, lr=lr
        )

        # save checkpoint
        ckpt_path = os.path.join(checkpoint_dir, key)
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(query_enc.state_dict(), os.path.join(ckpt_path, 'query_enc.pt'))
        torch.save(doc_enc.state_dict(),   os.path.join(ckpt_path, 'doc_enc.pt'))

        # evaluate
        metrics = evaluate_config(query_enc, doc_enc, dev_loader, doc_loader, qrels, device)
        results[key] = {**cfg, **metrics}
        print(f'  NDCG@1000 : {metrics["ndcg@1000"]:.4f}')
        print(f'  R@100     : {metrics["r@100"]:.4f}')

    # save results
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'tokenizer_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print_comparison_table(results)
    return results


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_comparison_table(results):
    print(f'\n{"=" * 65}')
    print(f'  Tokenizer Comparison Results  (RQ1)')
    print(f'{"=" * 65}')
    print(f'  {"Config":<12} {"Tokenizer Type":<30} {"NDCG@1000":>10} {"R@100":>8}')
    print(f'  {"-" * 61}')
    for key, r in results.items():
        print(f'  {key:<12} {r["tokenizer_type"]:<30} {r["ndcg@1000"]:>10.4f} {r["r@100"]:>8.4f}')
    print(f'{"=" * 65}\n')

    # highlight best
    best_key = max(results, key=lambda k: results[k]['ndcg@1000'])
    print(f'  Best tokenizer by NDCG@1000: {results[best_key]["tokenizer_type"]} '
          f'({results[best_key]["ndcg@1000"]:.4f})\n')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    data_cfg = {
        'train_query_path': 'data/trec/train_queries.jsonl',
        'train_qrel_path':  'data/trec/train_qrels.txt',
        'dev_query_path':   'data/trec/dev_queries.jsonl',
        'dev_qrel_path':    'data/trec/dev_qrels.txt',
        'doc_path':         'data/trec/docs.jsonl',
    }

    results = run_tokenizer_comparison(
        data_cfg,
        epochs=10,
        batch_size=16,
        lr=2e-5,
    )
