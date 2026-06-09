import json
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoTokenizer, BertTokenizer


def load_jsonl(path):
    with open(path, 'r') as f:
        return [json.loads(line) for line in f if line.strip()]


def load_qrels(path):
    """Standard TREC qrel format: query_id 0 doc_id relevance"""
    qrels = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            qid, _, did, rel = parts[0], parts[1], parts[2], int(parts[3])
            if rel > 0:
                qrels.setdefault(qid, []).append(did)
    return qrels


# ---------------------------------------------------------------------------
# Document Dataset — shared by both TREC and Reddit
# ---------------------------------------------------------------------------

class DocumentDataset(Dataset):
    """
    Encodes Wikipedia documents for FAISS indexing.
    Each document uses title + first 400 words of text.
    """
    def __init__(self, doc_path, tokenizer, max_length=512):
        self.docs = load_jsonl(doc_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.id_to_idx = {doc['id']: i for i, doc in enumerate(self.docs)}

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, idx):
        doc = self.docs[idx]
        text = doc['title'] + ' ' + doc.get('text', '')
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'doc_id':          doc['id'],
            'input_ids':       encoded['input_ids'].squeeze(0),
            'attention_mask':  encoded['attention_mask'].squeeze(0),
        }


# ---------------------------------------------------------------------------
# TREC 2023 ToT Dataset
# ---------------------------------------------------------------------------

class TRECToTDataset(Dataset):
    """
    TREC 2023 Tip-of-the-Tongue dataset.

    Expected files:
      queries.jsonl  — each line: {id, sentences: [...], text}
      docs.jsonl     — each line: {id, title, text}
      qrels.txt      — standard TREC format: qid 0 docid rel

    Each query already has sentence-level annotations in the dataset.
    """
    def __init__(self, query_path, doc_path, qrel_path, tokenizer,
                 max_sentences=30, max_sent_len=64, max_doc_len=512):
        self.queries      = load_jsonl(query_path)
        self.qrels        = load_qrels(qrel_path)
        self.doc_dataset  = DocumentDataset(doc_path, tokenizer, max_doc_len)
        self.tokenizer    = tokenizer
        self.max_sentences = max_sentences
        self.max_sent_len  = max_sent_len

        # keep only queries that have at least one relevant document
        self.queries = [q for q in self.queries if q['id'] in self.qrels]

    def __len__(self):
        return len(self.queries)

    def _encode_sentences(self, sentences):
        encoded = self.tokenizer(
            sentences,
            max_length=self.max_sent_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded['input_ids'], encoded['attention_mask']

    def __getitem__(self, idx):
        query   = self.queries[idx]
        qid     = query['id']
        sentences = query['sentences'][:self.max_sentences]

        sent_ids, sent_masks = self._encode_sentences(sentences)

        # pad to max_sentences if query has fewer sentences
        N = sent_ids.shape[0]
        if N < self.max_sentences:
            pad_ids   = torch.zeros(self.max_sentences - N, self.max_sent_len, dtype=torch.long)
            pad_masks = torch.zeros(self.max_sentences - N, self.max_sent_len, dtype=torch.long)
            sent_ids   = torch.cat([sent_ids,   pad_ids],   dim=0)
            sent_masks = torch.cat([sent_masks, pad_masks], dim=0)

        # pick first relevant document as positive
        pos_doc_id  = self.qrels[qid][0]
        pos_doc_idx = self.doc_dataset.id_to_idx[pos_doc_id]
        pos_doc     = self.doc_dataset[pos_doc_idx]

        return {
            'query_id':         qid,
            'sentence_ids':     sent_ids,                        # (max_sentences, max_sent_len)
            'sentence_masks':   sent_masks,                      # (max_sentences, max_sent_len)
            'num_sentences':    torch.tensor(N),
            'doc_input_ids':    pos_doc['input_ids'],            # (max_doc_len,)
            'doc_attention_mask': pos_doc['attention_mask'],
        }


# ---------------------------------------------------------------------------
# Reddit ToT Dataset
# ---------------------------------------------------------------------------

class RedditToTDataset(Dataset):
    """
    Multi-domain Reddit Tip-of-the-Tongue dataset.
    Used for cross-domain generalization experiments.

    Expected files:
      reddit_queries.jsonl — each line: {id, text, sentences: [...], answer_id, domain}
      docs.jsonl           — same document collection as TREC

    Sentences are extracted by splitting query text on '.' if not pre-annotated.
    """
    def __init__(self, query_path, doc_path, tokenizer,
                 max_sentences=30, max_sent_len=64, max_doc_len=512,
                 domain=None):
        self.queries      = load_jsonl(query_path)
        self.doc_dataset  = DocumentDataset(doc_path, tokenizer, max_doc_len)
        self.tokenizer    = tokenizer
        self.max_sentences = max_sentences
        self.max_sent_len  = max_sent_len

        # optionally filter by domain (movie, book, music, etc.)
        if domain is not None:
            self.queries = [q for q in self.queries if q.get('domain') == domain]

        # keep only queries with a valid answer document and non-empty sentences
        valid_ids = set(self.doc_dataset.id_to_idx.keys())
        self.queries = [q for q in self.queries
                        if q.get('answer_id') in valid_ids and self._get_sentences(q)]

    def __len__(self):
        return len(self.queries)

    def _get_sentences(self, query):
        # use pre-annotated sentences if available, else split on period
        if 'sentences' in query and query['sentences']:
            return query['sentences']
        return [s.strip() for s in query['text'].split('.') if s.strip()]

    def _encode_sentences(self, sentences):
        encoded = self.tokenizer(
            sentences,
            max_length=self.max_sent_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return encoded['input_ids'], encoded['attention_mask']

    def __getitem__(self, idx):
        query     = self.queries[idx]
        sentences = self._get_sentences(query)[:self.max_sentences]
        if not sentences:
            sentences = [query.get('text', 'unknown') or 'unknown']

        sent_ids, sent_masks = self._encode_sentences(sentences)

        N = sent_ids.shape[0]
        if N < self.max_sentences:
            pad_ids   = torch.zeros(self.max_sentences - N, self.max_sent_len, dtype=torch.long)
            pad_masks = torch.zeros(self.max_sentences - N, self.max_sent_len, dtype=torch.long)
            sent_ids   = torch.cat([sent_ids,   pad_ids],   dim=0)
            sent_masks = torch.cat([sent_masks, pad_masks], dim=0)

        pos_doc_idx = self.doc_dataset.id_to_idx[query['answer_id']]
        pos_doc     = self.doc_dataset[pos_doc_idx]

        return {
            'query_id':           query['id'],
            'domain':             query.get('domain', 'unknown'),
            'sentence_ids':       sent_ids,
            'sentence_masks':     sent_masks,
            'num_sentences':      torch.tensor(N),
            'doc_input_ids':      pos_doc['input_ids'],
            'doc_attention_mask': pos_doc['attention_mask'],
        }


# ---------------------------------------------------------------------------
# Collate function — handles variable num_sentences across batch
# ---------------------------------------------------------------------------

def collate_fn(batch):
    return {
        'query_id':           [b['query_id'] for b in batch],
        'sentence_ids':       torch.stack([b['sentence_ids']     for b in batch]),
        'sentence_masks':     torch.stack([b['sentence_masks']   for b in batch]),
        'num_sentences':      torch.stack([b['num_sentences']    for b in batch]),
        'doc_input_ids':      torch.stack([b['doc_input_ids']    for b in batch]),
        'doc_attention_mask': torch.stack([b['doc_attention_mask'] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Dataloader builders
# ---------------------------------------------------------------------------

def get_trec_dataloader(query_path, doc_path, qrel_path, batch_size=16,
                        tokenizer_name='bert-base-uncased', shuffle=True, distributed=False):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    dataset   = TRECToTDataset(query_path, doc_path, qrel_path, tokenizer)
    sampler   = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=(shuffle and not distributed),
                      sampler=sampler, collate_fn=collate_fn, num_workers=4)


def get_reddit_dataloader(query_path, doc_path, batch_size=16,
                          tokenizer_name='bert-base-uncased', shuffle=True,
                          domain=None, distributed=False):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    dataset   = RedditToTDataset(query_path, doc_path, tokenizer, domain=domain)
    sampler   = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(dataset, batch_size=batch_size,
                      shuffle=(shuffle and not distributed),
                      sampler=sampler, collate_fn=collate_fn, num_workers=4)


def get_doc_dataloader(doc_path, batch_size=64, tokenizer_name='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
    dataset   = DocumentDataset(doc_path, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=4)


# ---------------------------------------------------------------------------
# Usage example
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    trec_loader = get_trec_dataloader(
        query_path='data/trec/queries.jsonl',
        doc_path='data/trec/docs.jsonl',
        qrel_path='data/trec/qrels.txt',
        batch_size=16
    )

    reddit_loader = get_reddit_dataloader(
        query_path='data/reddit/queries.jsonl',
        doc_path='data/trec/docs.jsonl',   # same Wikipedia collection
        batch_size=16,
        domain='movie'                      # filter to movie domain only
    )

    doc_loader = get_doc_dataloader(
        doc_path='data/trec/docs.jsonl',
        batch_size=64
    )

    batch = next(iter(trec_loader))
    print("sentence_ids shape :", batch['sentence_ids'].shape)     # (16, 30, 64)
    print("doc_input_ids shape:", batch['doc_input_ids'].shape)    # (16, 512)
    print("num_sentences      :", batch['num_sentences'])
