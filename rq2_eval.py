"""
RQ2 Evaluation: "Can noise reduction techniques be further improved for retrieval performance?"

Ablation study comparing noise reduction configurations:

  Baselines:
    1. no_reduction       — standard GRU, final hidden state only, no dropout
    2. first_k_sentences  — only use first K sentences (simple truncation)
    3. random_subset      — random sentence subset (sanity check)

  Our noise reduction components (ablations):
    4. dropout_only       — sentence dropout, no attention
    5. attention_only     — query-aware attention, no dropout
    6. full_model         — sentence dropout + query-aware attention (full)

  Dropout rate sensitivity:
    7. dropout_0.1        — full model, dropout=0.1
    8. dropout_0.2        — full model, dropout=0.2  (default)
    9. dropout_0.3        — full model, dropout=0.3

Each configuration is trained from scratch and evaluated on dev NDCG@1000 and R@100.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import faiss
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader
from transformers import AutoModel, BertTokenizer
from tqdm import tqdm

from dataset import TRECToTDataset, collate_fn, load_qrels, get_doc_dataloader
from evaluate import ndcg_at_k, recall_at_k, build_index
from tot_retrieval import DocEncoder, contrastive_loss


# ---------------------------------------------------------------------------
# Configurable QueryEncoder — controls noise reduction components
# ---------------------------------------------------------------------------

class AblationQueryEncoder(nn.Module):
    """
    Unified query encoder with toggleable noise reduction components.
    Allows clean ablation of each contribution independently.
    """

    def __init__(self,
                 bert_model='bert-base-uncased',
                 gru_hidden=768,
                 use_dropout=False,
                 sentence_dropout=0.2,
                 use_attention=False,
                 first_k=None):
        """
        Args:
            use_dropout:      enable sentence dropout during training
            sentence_dropout: dropout probability (only used if use_dropout=True)
            use_attention:    enable query-aware attention over GRU hidden states
            first_k:          if set, truncate query to first K sentences (truncation baseline)
        """
        super().__init__()
        self.bert             = AutoModel.from_pretrained(bert_model)
        self.gru              = nn.GRU(input_size=768, hidden_size=gru_hidden, batch_first=True)
        self.use_dropout      = use_dropout
        self.sentence_dropout = sentence_dropout
        self.use_attention    = use_attention
        self.first_k          = first_k

        if use_attention:
            self.attention = nn.Linear(gru_hidden * 2, 1)

    def forward(self, sentence_ids, sentence_masks):
        B, N, L = sentence_ids.shape

        # truncation baseline: keep only first K sentences
        if self.first_k is not None:
            k          = min(self.first_k, N)
            sentence_ids   = sentence_ids[:, :k, :]
            sentence_masks = sentence_masks[:, :k, :]
            N = k

        cls = self.bert(
            input_ids=sentence_ids.view(B * N, L),
            attention_mask=sentence_masks.view(B * N, L)
        ).last_hidden_state[:, 0, :].view(B, N, -1)  # (B, N, 768)

        # sentence dropout
        if self.training and self.use_dropout and self.sentence_dropout > 0:
            mask = (torch.rand(B, N, 1, device=cls.device) > self.sentence_dropout).float()
            cls  = cls * mask

        all_hidden, final_hidden = self.gru(cls)  # (B, N, 768), (1, B, 768)

        if self.use_attention:
            # query-aware attention: score each hidden state against final world state
            final   = all_hidden[:, -1, :].unsqueeze(1).expand(-1, N, -1)  # (B, N, 768)
            concat  = torch.cat([all_hidden, final], dim=-1)                # (B, N, 1536)
            weights = torch.softmax(self.attention(concat), dim=1)          # (B, N, 1)
            return (weights * all_hidden).sum(dim=1)                        # (B, 768)
        else:
            return final_hidden.squeeze(0)                                  # (B, 768)


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    # baselines
    'no_reduction': {
        'use_dropout': False, 'sentence_dropout': 0.0,
        'use_attention': False, 'first_k': None,
        'label': 'No noise reduction (GRU final hidden only)',
        'group': 'Baseline',
    },
    'first_5_sentences': {
        'use_dropout': False, 'sentence_dropout': 0.0,
        'use_attention': False, 'first_k': 5,
        'label': 'Truncation — first 5 sentences',
        'group': 'Baseline',
    },
    'first_10_sentences': {
        'use_dropout': False, 'sentence_dropout': 0.0,
        'use_attention': False, 'first_k': 10,
        'label': 'Truncation — first 10 sentences',
        'group': 'Baseline',
    },

    # component ablations
    'dropout_only': {
        'use_dropout': True, 'sentence_dropout': 0.2,
        'use_attention': False, 'first_k': None,
        'label': 'Sentence dropout only (p=0.2)',
        'group': 'Ablation',
    },
    'attention_only': {
        'use_dropout': False, 'sentence_dropout': 0.0,
        'use_attention': True, 'first_k': None,
        'label': 'Query-aware attention only',
        'group': 'Ablation',
    },
    'full_model': {
        'use_dropout': True, 'sentence_dropout': 0.2,
        'use_attention': True, 'first_k': None,
        'label': 'Full model (dropout + attention)',
        'group': 'Ablation',
    },

    # dropout rate sensitivity
    'dropout_0.1': {
        'use_dropout': True, 'sentence_dropout': 0.1,
        'use_attention': True, 'first_k': None,
        'label': 'Full model — dropout p=0.1',
        'group': 'Dropout sensitivity',
    },
    'dropout_0.3': {
        'use_dropout': True, 'sentence_dropout': 0.3,
        'use_attention': True, 'first_k': None,
        'label': 'Full model — dropout p=0.3',
        'group': 'Dropout sensitivity',
    },
}


# ---------------------------------------------------------------------------
# Random sentence subset baseline
# ---------------------------------------------------------------------------

class RandomSubsetQueryEncoder(AblationQueryEncoder):
    """Uses a random subset of K sentences at inference — sanity check baseline."""

    def __init__(self, k=10, **kwargs):
        super().__init__(**kwargs)
        self.k = k

    def forward(self, sentence_ids, sentence_masks):
        B, N, L = sentence_ids.shape
        k       = min(self.k, N)
        perm    = torch.randperm(N, device=sentence_ids.device)[:k]
        return super().forward(sentence_ids[:, perm, :], sentence_masks[:, perm, :])


# ---------------------------------------------------------------------------
# Train and evaluate one configuration
# ---------------------------------------------------------------------------

def train_one(config_key, config, train_loader, device,
              bert_model='bert-base-uncased', epochs=10, lr=2e-5):
    query_enc = AblationQueryEncoder(
        bert_model=bert_model,
        use_dropout=config['use_dropout'],
        sentence_dropout=config['sentence_dropout'],
        use_attention=config['use_attention'],
        first_k=config['first_k'],
    ).to(device)

    doc_enc = DocEncoder(bert_model=bert_model).to(device)

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

        for batch in tqdm(train_loader, desc=f'  [{config_key}] Epoch {epoch}', leave=False):
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

        print(f'  [{config_key}] Epoch {epoch}/{epochs}  loss: {total_loss / len(train_loader):.4f}')

    return query_enc, doc_enc


def evaluate_one(query_enc, doc_enc, dev_loader, doc_loader,
                 qrels, device, k_ndcg=1000, k_recall=100):
    index, idx_to_docid = build_index(doc_enc, doc_loader, device)
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
                ranked   = [idx_to_docid[j] for j in doc_indices[i]]
                relevant = qrels.get(qid, [])
                ndcg_scores.append(  ndcg_at_k(  ranked, relevant, k=k_ndcg))
                recall_scores.append(recall_at_k(ranked, relevant, k=k_recall))

    return {
        f'ndcg@{k_ndcg}': float(np.mean(ndcg_scores)),
        f'r@{k_recall}':   float(np.mean(recall_scores)),
    }


# ---------------------------------------------------------------------------
# Results table
# ---------------------------------------------------------------------------

def print_rq2_table(results):
    groups = ['Baseline', 'Ablation', 'Dropout sensitivity']

    print(f'\n{"=" * 75}')
    print(f'  RQ2: Noise Reduction Ablation Study')
    print(f'{"=" * 75}')
    print(f'  {"Config":<22} {"Description":<38} {"NDCG@1000":>10} {"R@100":>7}')
    print(f'  {"-" * 71}')

    best_ndcg = max(r['ndcg@1000'] for r in results.values())

    for group in groups:
        group_results = {k: v for k, v in results.items() if v['group'] == group}
        if not group_results:
            continue
        print(f'\n  [{group}]')
        for key, r in group_results.items():
            marker = ' *' if r['ndcg@1000'] == best_ndcg else '  '
            print(f'{marker} {key:<22} {r["label"]:<38} {r["ndcg@1000"]:>10.4f} {r["r@100"]:>7.4f}')

    print(f'\n  * best configuration')
    print(f'{"=" * 75}')

    # delta table — gain over no_reduction baseline
    if 'no_reduction' in results:
        base_ndcg = results['no_reduction']['ndcg@1000']
        print(f'\n  Gain over no-reduction baseline (NDCG@1000):')
        for key, r in results.items():
            if key == 'no_reduction':
                continue
            delta = r['ndcg@1000'] - base_ndcg
            sign  = '+' if delta >= 0 else ''
            print(f'    {key:<22} {sign}{delta:.4f}')
    print()


# ---------------------------------------------------------------------------
# Main RQ2 runner
# ---------------------------------------------------------------------------

def run_rq2_evaluation(data_cfg, epochs=10, batch_size=16, lr=2e-5,
                       bert_model='bert-base-uncased',
                       checkpoint_dir='checkpoints/rq2',
                       configs_to_run=None):
    """
    Runs the full RQ2 ablation study.

    Args:
        configs_to_run: list of config keys to run (default: all)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}\n')

    tokenizer = BertTokenizer.from_pretrained(bert_model)
    qrels     = load_qrels(data_cfg['dev_qrel_path'])

    # shared dataloaders
    train_dataset = TRECToTDataset(
        data_cfg['train_query_path'], data_cfg['doc_path'],
        data_cfg['train_qrel_path'], tokenizer
    )
    dev_dataset = TRECToTDataset(
        data_cfg['dev_query_path'], data_cfg['doc_path'],
        data_cfg['dev_qrel_path'], tokenizer
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  collate_fn=collate_fn, num_workers=2)
    dev_loader   = DataLoader(dev_dataset,   batch_size=batch_size,
                              shuffle=False, collate_fn=collate_fn, num_workers=2)
    doc_loader   = get_doc_dataloader(data_cfg['doc_path'], batch_size=64)

    configs = {k: v for k, v in ABLATION_CONFIGS.items()
               if configs_to_run is None or k in configs_to_run}

    all_results = {}

    # also run random subset baseline separately
    if configs_to_run is None or 'random_subset' in configs_to_run:
        configs['random_subset'] = {
            'label': 'Random subset (K=10 sentences)',
            'group': 'Baseline',
        }

    for key, config in configs.items():
        print(f'\n{"─" * 55}')
        print(f'  Config: {key}')
        print(f'  {config["label"]}')
        print(f'{"─" * 55}')

        if key == 'random_subset':
            query_enc = RandomSubsetQueryEncoder(
                k=10, bert_model=bert_model,
                use_dropout=False, use_attention=False
            ).to(device)
            doc_enc = DocEncoder(bert_model=bert_model).to(device)
            # random subset doesn't need training — evaluate zero-shot
            print('  (no training — random subset is a zero-shot baseline)')
        else:
            query_enc, doc_enc = train_one(
                key, config, train_loader, device,
                bert_model=bert_model, epochs=epochs, lr=lr
            )

        # save checkpoint
        ckpt = os.path.join(checkpoint_dir, key)
        os.makedirs(ckpt, exist_ok=True)
        torch.save(query_enc.state_dict(), os.path.join(ckpt, 'query_enc.pt'))
        torch.save(doc_enc.state_dict(),   os.path.join(ckpt, 'doc_enc.pt'))

        metrics = evaluate_one(query_enc, doc_enc, dev_loader, doc_loader, qrels, device)
        all_results[key] = {**config, **metrics}

        print(f'  NDCG@1000 : {metrics["ndcg@1000"]:.4f}')
        print(f'  R@100     : {metrics["r@100"]:.4f}')

    # save and print
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(os.path.join(checkpoint_dir, 'rq2_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)

    print_rq2_table(all_results)
    return all_results


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

    # run full ablation
    results = run_rq2_evaluation(data_cfg, epochs=10, batch_size=16)

    # or run a subset for quick testing
    # results = run_rq2_evaluation(data_cfg, epochs=3,
    #     configs_to_run=['no_reduction', 'dropout_only', 'attention_only', 'full_model'])
