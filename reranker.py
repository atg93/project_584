import os
import json
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm

from dataset import load_jsonl, load_qrels


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
You are an expert movie identification system.

A user is trying to recall a specific movie but cannot remember its title. \
Below is their description followed by a candidate movie.

User description:
{query}

Candidate movie:
Title: {title}
{doc_text}

Is this candidate the movie the user is describing? Answer with yes or no.
Answer:"""


def build_prompt(query_text, doc_title, doc_text, max_doc_words=200):
    truncated = ' '.join(doc_text.split()[:max_doc_words])
    return PROMPT_TEMPLATE.format(
        query=query_text,
        title=doc_title,
        doc_text=truncated
    )


# ---------------------------------------------------------------------------
# LoRA Reranker
# ---------------------------------------------------------------------------

class LoRAReranker:
    """
    Wraps a causal LLM (Llama / Qwen) with LoRA for pointwise re-ranking.
    Relevance score = log P("yes") - log P("no") given the prompt.
    """

    def __init__(self, model_name='Qwen/Qwen1.5-1.8B', load_in_4bit=True,
                 lora_r=16, lora_alpha=32, lora_dropout=0.1,
                 checkpoint_path=None):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ) if load_in_4bit else None

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map='auto',
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        )

        if checkpoint_path and os.path.exists(checkpoint_path):
            # load fine-tuned LoRA weights
            self.model = PeftModel.from_pretrained(base_model, checkpoint_path)
            print(f'LoRA weights loaded from {checkpoint_path}')
        else:
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                bias='none',
            )
            self.model = get_peft_model(base_model, lora_config)
            self.model.print_trainable_parameters()

        # get token ids for "yes" and "no"
        self.yes_id = self.tokenizer(' yes', add_special_tokens=False).input_ids[-1]
        self.no_id  = self.tokenizer(' no',  add_special_tokens=False).input_ids[-1]

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f'LoRA reranker saved to {path}')

    def score(self, prompts, batch_size=8):
        """
        Returns relevance scores for a list of prompts.
        Score = log P(yes) - log P(no) at the final token position.
        """
        self.model.eval()
        all_scores = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i: i + batch_size]
            inputs = self.tokenizer(
                batch_prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.model.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits  # (B, seq_len, vocab)

            last_logits = logits[:, -1, :]  # logits at final position → predicts next token
            log_probs   = torch.log_softmax(last_logits, dim=-1)

            scores = (log_probs[:, self.yes_id] - log_probs[:, self.no_id]).cpu().tolist()
            all_scores.extend(scores)

        return all_scores


# ---------------------------------------------------------------------------
# Re-ranking Dataset
# ---------------------------------------------------------------------------

class RerankerDataset(Dataset):
    """
    Builds (query, doc, label) training examples for LoRA fine-tuning.

    Positive: (query, relevant_doc) → label = "yes"
    Negatives: (query, top-K retrieved but irrelevant docs) → label = "no"

    Requires bi-encoder run file or pre-retrieved candidates.
    """

    def __init__(self, query_path, doc_path, qrel_path,
                 candidates_path, n_negatives=3, max_doc_words=200):
        queries    = {q['id']: q for q in load_jsonl(query_path)}
        docs       = {d['id']: d for d in load_jsonl(doc_path)}
        qrels      = load_qrels(qrel_path)

        # candidates_path: jsonl with {query_id, retrieved: [doc_id, ...]}
        candidates = {c['query_id']: c['retrieved'] for c in load_jsonl(candidates_path)}

        self.examples = []
        for qid, query in queries.items():
            if qid not in qrels or qid not in candidates:
                continue

            relevant_ids = set(qrels[qid])
            query_text   = query.get('text', ' '.join(query.get('sentences', [])))

            # positive examples
            for doc_id in qrels[qid]:
                if doc_id in docs:
                    self.examples.append({
                        'prompt': build_prompt(
                            query_text,
                            docs[doc_id].get('title', ''),
                            docs[doc_id].get('text', ''),
                            max_doc_words
                        ),
                        'label': 'yes'
                    })

            # negative examples from retrieved but irrelevant docs
            negatives = [d for d in candidates[qid] if d not in relevant_ids]
            for doc_id in negatives[:n_negatives]:
                if doc_id in docs:
                    self.examples.append({
                        'prompt': build_prompt(
                            query_text,
                            docs[doc_id].get('title', ''),
                            docs[doc_id].get('text', ''),
                            max_doc_words
                        ),
                        'label': 'no'
                    })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_reranker(reranker, dataset, epochs=3, lr=2e-4, batch_size=4,
                   checkpoint_dir='checkpoints/reranker'):
    """
    Fine-tunes the LoRA reranker with binary cross-entropy over yes/no logits.
    Uses a small lr since we only train LoRA adapter weights.
    """
    yes_id = reranker.yes_id
    no_id  = reranker.no_id

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, reranker.model.parameters()), lr=lr
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=lambda b: b)

    reranker.model.train()
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        progress   = tqdm(loader, desc=f'Reranker Epoch {epoch}')

        for batch in progress:
            prompts = [ex['prompt'] for ex in batch]
            labels  = [1 if ex['label'] == 'yes' else 0 for ex in batch]

            inputs = reranker.tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024
            ).to(reranker.model.device)

            logits     = reranker.model(**inputs).logits[:, -1, :]  # (B, vocab)
            yes_no     = logits[:, [yes_id, no_id]]                 # (B, 2)
            targets    = torch.tensor(labels, device=yes_no.device)

            loss = nn.CrossEntropyLoss()(yes_no, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(reranker.model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            progress.set_postfix(loss=f'{loss.item():.4f}')

        avg_loss = total_loss / len(loader)
        print(f'Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            reranker.save(os.path.join(checkpoint_dir, 'best'))

    reranker.save(os.path.join(checkpoint_dir, 'latest'))
    print(f'Reranker training complete. Best loss: {best_loss:.4f}')


# ---------------------------------------------------------------------------
# Re-ranking inference
# ---------------------------------------------------------------------------

def rerank(reranker, query_text, candidate_doc_ids, docs, top_k=100, batch_size=8):
    """
    Re-ranks a list of candidate doc_ids for a single query.
    Returns doc_ids sorted by relevance score descending.
    """
    candidates = candidate_doc_ids[:top_k]
    prompts    = [
        build_prompt(query_text, docs[d].get('title', ''), docs[d].get('text', ''))
        for d in candidates if d in docs
    ]
    valid_ids = [d for d in candidates if d in docs]

    scores  = reranker.score(prompts, batch_size=batch_size)
    ranked  = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked]


# ---------------------------------------------------------------------------
# Full pipeline evaluation: bi-encoder → LoRA reranker → metrics
# ---------------------------------------------------------------------------

def evaluate_pipeline(reranker, bi_encoder_results, queries, docs, qrels,
                       rerank_top_k=100, k_ndcg=1000, k_recall=100):
    """
    bi_encoder_results: dict {query_id: [ranked_doc_ids (up to 1000)]}
    Reranks top rerank_top_k, appends the rest unchanged.
    """
    from evaluate import ndcg_at_k, recall_at_k

    ndcg_scores   = []
    recall_scores = []

    for qid, ranked_ids in tqdm(bi_encoder_results.items(), desc='Re-ranking'):
        query      = queries.get(qid, {})
        query_text = query.get('text', ' '.join(query.get('sentences', [])))

        # re-rank top-K, keep the rest in original order
        reranked     = rerank(reranker, query_text, ranked_ids, docs, top_k=rerank_top_k)
        remainder    = [d for d in ranked_ids[rerank_top_k:]]
        final_ranked = reranked + remainder

        relevant_ids = qrels.get(qid, [])
        ndcg_scores.append(  ndcg_at_k(  final_ranked, relevant_ids, k=k_ndcg))
        recall_scores.append(recall_at_k(final_ranked, relevant_ids, k=k_recall))

    return {
        f'pipeline_mean_ndcg@{k_ndcg}': float(np.mean(ndcg_scores)),
        f'pipeline_mean_r@{k_recall}':   float(np.mean(recall_scores)),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    # 1. initialize reranker (Qwen 1.8B with 4-bit quantization)
    reranker = LoRAReranker(
        model_name='Qwen/Qwen1.5-1.8B',
        load_in_4bit=True,
        lora_r=16,
        lora_alpha=32,
    )

    # 2. build training dataset (needs bi-encoder candidates pre-generated)
    dataset = RerankerDataset(
        query_path='data/trec/train_queries.jsonl',
        doc_path='data/trec/docs.jsonl',
        qrel_path='data/trec/train_qrels.txt',
        candidates_path='runs/bi_encoder_train_candidates.jsonl',
        n_negatives=3
    )
    print(f'Reranker training examples: {len(dataset)}')

    # 3. fine-tune
    train_reranker(reranker, dataset, epochs=3, lr=2e-4, batch_size=4)

    # 4. load best checkpoint and evaluate
    reranker_best = LoRAReranker(
        model_name='Qwen/Qwen1.5-1.8B',
        load_in_4bit=True,
        checkpoint_path='checkpoints/reranker/best'
    )
