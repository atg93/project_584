import torch
import torch.nn as nn
import torch.distributed
from transformers import BertModel
import faiss
import numpy as np


class QueryEncoder(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', gru_hidden=None,
                 sentence_dropout=0.2, proj_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        bert_hidden = self.bert.config.hidden_size      # 768, 256, 512, etc.
        gru_hidden  = gru_hidden or bert_hidden         # default to BERT's own dim
        self.sentence_dropout = sentence_dropout
        self.gru       = nn.GRU(input_size=bert_hidden, hidden_size=gru_hidden, batch_first=True)
        self.attention = nn.Linear(gru_hidden * 2, 1)
        self.proj      = nn.Linear(gru_hidden, proj_dim)

    def forward(self, sentence_ids, sentence_masks):
        B, N, L = sentence_ids.shape

        flat_ids   = sentence_ids.view(B * N, L)
        flat_masks = sentence_masks.view(B * N, L)

        cls = self.bert(flat_ids, flat_masks).last_hidden_state[:, 0, :]  # (B*N, bert_hidden)
        sentence_embeddings = cls.view(B, N, -1)

        if self.training and self.sentence_dropout > 0:
            mask = (torch.rand(B, N, 1, device=sentence_embeddings.device) > self.sentence_dropout).float()
            sentence_embeddings = sentence_embeddings * mask

        all_hidden, _ = self.gru(sentence_embeddings)

        final        = all_hidden[:, -1, :].unsqueeze(1).expand(-1, N, -1)
        concat       = torch.cat([all_hidden, final], dim=-1)
        attn_weights = torch.softmax(self.attention(concat), dim=1)
        query_vec    = (attn_weights * all_hidden).sum(dim=1)

        return self.proj(query_vec)                                        # (B, proj_dim)


class DocEncoder(nn.Module):
    def __init__(self, bert_model='bert-base-uncased', proj_dim=256):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_model)
        bert_hidden = self.bert.config.hidden_size
        self.proj   = nn.Linear(bert_hidden, proj_dim)

    def forward(self, input_ids, attention_mask):
        cls = self.bert(input_ids, attention_mask).last_hidden_state[:, 0, :]
        return self.proj(cls)                                              # (B, proj_dim)


def contrastive_loss(query_vecs, doc_vecs, temperature=0.05):
    # gather across all GPUs when running under DDP so the full batch is used
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank       = torch.distributed.get_rank()

        all_q = [torch.zeros_like(query_vecs) for _ in range(world_size)]
        all_d = [torch.zeros_like(doc_vecs)   for _ in range(world_size)]
        torch.distributed.all_gather(all_q, query_vecs.detach())
        torch.distributed.all_gather(all_d, doc_vecs.detach())

        # keep gradients flowing through the local rank's slice
        all_q[rank] = query_vecs
        all_d[rank] = doc_vecs

        query_vecs = torch.cat(all_q, dim=0)
        doc_vecs   = torch.cat(all_d, dim=0)

    scores = torch.matmul(query_vecs, doc_vecs.T) / temperature
    labels = torch.arange(scores.shape[0]).to(scores.device)
    return nn.CrossEntropyLoss()(scores, labels)


def build_index(doc_encoder, dataloader, device):
    doc_encoder.eval()
    all_vecs = []
    with torch.no_grad():
        for ids, masks in dataloader:
            vecs = doc_encoder(ids.to(device), masks.to(device))
            all_vecs.append(vecs.cpu().numpy())

    all_vecs = np.vstack(all_vecs).astype('float32')
    faiss.normalize_L2(all_vecs)

    index = faiss.IndexFlatIP(all_vecs.shape[1])  # inner product = cosine after L2 norm
    index.add(all_vecs)
    return index


def retrieve(query_encoder, query, index, k=1000):
    query_encoder.eval()
    with torch.no_grad():
        q_vec = query_encoder(*query).cpu().numpy().astype('float32')
    faiss.normalize_L2(q_vec)
    doc_indices, scores = index.search(q_vec, k)
    return doc_indices, scores


def train(query_enc, doc_enc, train_dataloader, device, epochs=10, lr=2e-5):
    query_enc.to(device)
    doc_enc.to(device)

    optimizer = torch.optim.AdamW(
        list(query_enc.parameters()) + list(doc_enc.parameters()), lr=lr
    )

    for epoch in range(epochs):
        query_enc.train()
        doc_enc.train()
        total_loss = 0.0

        for batch in train_dataloader:
            sentence_ids, sentence_masks, doc_ids, doc_masks = [x.to(device) for x in batch]

            q_vecs = query_enc(sentence_ids, sentence_masks)
            d_vecs = doc_enc(doc_ids, doc_masks)

            loss = contrastive_loss(q_vecs, d_vecs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} — Loss: {total_loss / len(train_dataloader):.4f}")


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    query_enc = QueryEncoder()
    doc_enc = DocEncoder()

    print("QueryEncoder and DocEncoder initialized.")
    print(f"Using device: {device}")
