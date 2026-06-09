import os
import json
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from tqdm import tqdm

from dataset import load_jsonl, load_qrels

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# DDP helpers — mirror the pattern in train.py
# ---------------------------------------------------------------------------

def setup_ddp():
    """Initialise NCCL process group; return this rank's local GPU id."""
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(hours=4))
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main():
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def is_ddp():
    return 'LOCAL_RANK' in os.environ


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
                 torch_dtype=None, checkpoint_path=None, device_id=0):

        # device_id picks which GPU this rank owns under DDP; defaults to 0 for
        # single-GPU runs. Each DDP rank should pass its own LOCAL_RANK here so
        # that the model lands on the correct device.
        self.device_id = device_id

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        ) if load_in_4bit else None

        if torch_dtype is None:
            torch_dtype = torch.bfloat16 if not load_in_4bit else None

        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map={'': device_id},
            torch_dtype=torch_dtype,
        )

        # required for LoRA training with device_map='auto' (multi-GPU)
        base_model.enable_input_require_grads()

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
        # unwrap DDP if present so we save the underlying PEFT model
        model = self.model.module if hasattr(self.model, 'module') else self.model
        model.save_pretrained(path)
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
    Optionally accepts reddit_query_path to include Reddit ToT queries
    (uses answer_id as the single relevant document).
    """

    def __init__(self, query_path, doc_path, qrel_path,
                 candidates_path, n_negatives=3, max_doc_words=200,
                 reddit_query_path=None):
        docs       = {d['id']: d for d in load_jsonl(doc_path)}
        candidates = {c['query_id']: c['retrieved'] for c in load_jsonl(candidates_path)}

        self.examples = []

        # --- TREC queries ---
        trec_queries = {q['id']: q for q in load_jsonl(query_path)}
        trec_qrels   = load_qrels(qrel_path)
        self._add_examples(trec_queries, trec_qrels, candidates,
                           docs, n_negatives, max_doc_words)
        print(f'TREC examples: {len(self.examples)}')

        # --- Reddit queries (pseudo-qrels from answer_id) ---
        if reddit_query_path:
            n_before = len(self.examples)
            reddit_queries = {q['id']: q for q in load_jsonl(reddit_query_path)
                              if q.get('answer_id') in docs}
            # build pseudo-qrels: {qid: [answer_id]}
            reddit_qrels   = {q['id']: [q['answer_id']] for q in reddit_queries.values()}
            self._add_examples(reddit_queries, reddit_qrels, candidates,
                               docs, n_negatives, max_doc_words, is_reddit=True)
            print(f'Reddit examples: {len(self.examples) - n_before}')

    def _add_examples(self, queries, qrels, candidates,
                      docs, n_negatives, max_doc_words, is_reddit=False):
        for qid, query in queries.items():
            if qid not in qrels or qid not in candidates:
                continue

            relevant_ids = set(qrels[qid])

            if is_reddit:
                sentences = query.get('sentences') or []
                if not sentences:
                    sentences = [s.strip() for s in query.get('text', '').split('.') if s.strip()]
                query_text = ' '.join(sentences) or query.get('text', '')
            else:
                query_text = query.get('text', ' '.join(query.get('sentences', [])))

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
# Per-epoch pipeline validation helpers
# ---------------------------------------------------------------------------

def build_dev_candidates(bi_checkpoint, queries_path, qrels_path, docs_path,
                         bert_model='bert-base-uncased', proj_dim=512,
                         device=None, subset=None, k=1000,
                         batch_size=16, doc_batch_size=256):
    """
    Run the BERT+GRU bi-encoder ONCE over the dev set and return
    {qid: [top-k doc_ids]}. Called before reranker training starts so we don't
    re-encode 231K documents (~1h51m) on every epoch.

    Returns (candidates_dict, queries_dict, docs_dict, qrels_dict).
    """
    # local imports — keeps reranker.py importable without bi-encoder deps
    import faiss
    from tot_retrieval import QueryEncoder, DocEncoder
    from dataset import get_trec_dataloader, get_doc_dataloader
    from evaluate import build_index

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    query_enc = QueryEncoder(bert_model=bert_model, proj_dim=proj_dim).to(device)
    doc_enc   = DocEncoder(bert_model=bert_model, proj_dim=proj_dim).to(device)
    query_enc.load_state_dict(torch.load(os.path.join(bi_checkpoint, 'query_enc.pt'),
                                         map_location=device))
    doc_enc.load_state_dict(torch.load(os.path.join(bi_checkpoint, 'doc_enc.pt'),
                                       map_location=device))
    print(f'[per-epoch eval] bi-encoder loaded from {bi_checkpoint}')

    query_loader = get_trec_dataloader(
        query_path=queries_path, doc_path=docs_path, qrel_path=qrels_path,
        batch_size=batch_size, tokenizer_name=bert_model,
        shuffle=False, distributed=False,
    )
    doc_loader = get_doc_dataloader(docs_path, batch_size=doc_batch_size,
                                    tokenizer_name=bert_model)

    index, idx_to_docid = build_index(doc_enc, doc_loader, device)

    query_enc.eval()
    candidates = {}
    with torch.no_grad():
        for batch in tqdm(query_loader, desc='[per-epoch eval] dev retrieval'):
            q_vecs = query_enc(
                batch['sentence_ids'].to(device),
                batch['sentence_masks'].to(device),
            ).cpu().numpy().astype('float32')
            faiss.normalize_L2(q_vecs)
            _, doc_indices = index.search(q_vecs, k)
            for i, qid in enumerate(batch['query_id']):
                candidates[qid] = [idx_to_docid[j] for j in doc_indices[i]]

    if subset is not None and subset < len(candidates):
        # deterministic subset — first N in iteration order
        candidates = dict(list(candidates.items())[:subset])
        print(f'[per-epoch eval] using subset of {len(candidates)} dev queries')

    queries = {q['id']: q for q in load_jsonl(queries_path)}
    docs    = {d['id']: d for d in load_jsonl(docs_path)}
    qrels   = load_qrels(qrels_path)

    # free bi-encoder GPU memory before reranker training starts
    del query_enc, doc_enc, index
    torch.cuda.empty_cache()
    print(f'[per-epoch eval] candidates cached for {len(candidates)} queries; '
          f'bi-encoder freed from GPU')

    return candidates, queries, docs, qrels


def _pipeline_eval(reranker, candidates, queries, docs, qrels,
                   rerank_top_k=100, batch_size=8):
    """
    Lightweight eval — assumes bi-encoder candidates are already cached.
    Returns dict with mean_ndcg@1000, mean_r@100.
    """
    from evaluate import ndcg_at_k, recall_at_k

    # use the unwrapped PEFT model for inference — avoids DDP overhead and
    # spurious gradient-sync warnings inside @torch.no_grad
    saved_model = reranker.model
    if hasattr(reranker.model, 'module'):
        reranker.model = reranker.model.module

    reranker.model.eval()
    ndcg_scores, recall_scores = [], []
    try:
        for qid, ranked_ids in tqdm(candidates.items(),
                                    desc='[per-epoch eval] re-ranking',
                                    leave=False):
            query      = queries.get(qid, {})
            query_text = query.get('text', ' '.join(query.get('sentences', [])))
            reranked   = rerank(reranker, query_text, ranked_ids, docs,
                                top_k=rerank_top_k, batch_size=batch_size)
            remainder  = ranked_ids[rerank_top_k:]
            final      = reranked + remainder
            rel        = qrels.get(qid, [])
            ndcg_scores.append(ndcg_at_k(final, rel, k=1000))
            recall_scores.append(recall_at_k(final, rel, k=100))
    finally:
        # restore DDP wrapper and train mode regardless of errors
        reranker.model = saved_model
        reranker.model.train()

    return {
        'mean_ndcg@1000': float(np.mean(ndcg_scores)),
        'mean_r@100':     float(np.mean(recall_scores)),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_reranker(reranker, dataset, epochs=3, lr=5e-5, batch_size=4,
                   checkpoint_dir='checkpoints/reranker',
                   wandb_project=None, wandb_run=None,
                   warmup_ratio=0.05,
                   eval_candidates=None, eval_queries=None,
                   eval_docs=None, eval_qrels=None,
                   eval_rerank_top_k=100, eval_batch_size=8):
    """
    Fine-tunes the LoRA reranker with binary cross-entropy over yes/no logits.
    Uses cosine LR schedule with linear warmup to prevent late-training divergence.

    DDP-aware: if launched under torchrun, each rank trains on its own GPU and
    only rank 0 logs / saves checkpoints.
    """
    yes_id = reranker.yes_id
    no_id  = reranker.no_id

    ddp = is_ddp() and dist.is_initialized()

    # wrap in DDP after the model is fully loaded on its device.
    # PEFT needs find_unused_parameters=True because not every base-model
    # parameter participates in the loss (only LoRA adapters get gradients).
    if ddp and not isinstance(reranker.model, DDP):
        reranker.model = DDP(
            reranker.model,
            device_ids=[reranker.device_id],
            output_device=reranker.device_id,
            find_unused_parameters=True,
        )

    # DistributedSampler shards the dataset across ranks; on a single GPU we
    # fall back to a normal shuffled DataLoader.
    if ddp:
        sampler = DistributedSampler(dataset, shuffle=True)
        loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                             collate_fn=lambda b: b)
    else:
        sampler = None
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=lambda b: b)

    total_steps  = len(loader) * epochs
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    if is_main() and WANDB_AVAILABLE and wandb_project:
        wandb.init(project=wandb_project, name=wandb_run, config={
            'epochs': epochs, 'lr': lr, 'batch_size': batch_size,
            'warmup_steps': warmup_steps, 'total_steps': total_steps,
            'world_size': dist.get_world_size() if ddp else 1,
        })
        print(f'WandB run: {wandb.run.url}')

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, reranker.model.parameters()),
        lr=lr, weight_decay=0.01
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    reranker.model.train()
    best_loss = float('inf')
    best_ndcg = -float('inf')
    eval_enabled = eval_candidates is not None
    global_step = 0

    # use the underlying model's device for tokenizer .to(...) calls
    device = (reranker.model.module.device
              if hasattr(reranker.model, 'module')
              else reranker.model.device)

    for epoch in range(1, epochs + 1):
        # ensures each rank sees a different shuffle each epoch
        if sampler is not None:
            sampler.set_epoch(epoch)

        total_loss = 0.0
        progress = tqdm(loader, desc=f'Reranker Epoch {epoch}',
                        disable=not is_main())

        for batch in progress:
            prompts = [ex['prompt'] for ex in batch]
            labels  = [1 if ex['label'] == 'yes' else 0 for ex in batch]

            inputs = reranker.tokenizer(
                prompts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=1024
            ).to(device)

            logits     = reranker.model(**inputs).logits[:, -1, :]  # (B, vocab)
            no_yes     = logits[:, [no_id, yes_id]]                 # (B, 2) — col 0 = no, col 1 = yes
            targets    = torch.tensor(labels, device=no_yes.device) # 1 = yes (col 1), 0 = no (col 0)

            loss = nn.CrossEntropyLoss()(no_yes, targets)
            loss.backward()

            nn.utils.clip_grad_norm_(reranker.model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            total_loss += loss.item()
            global_step += 1
            current_lr = scheduler.get_last_lr()[0]
            if is_main():
                progress.set_postfix(loss=f'{loss.item():.4f}', lr=f'{current_lr:.2e}')

                if WANDB_AVAILABLE and wandb_project:
                    wandb.log({'batch_loss': loss.item(), 'lr': current_lr, 'step': global_step})

        avg_loss = total_loss / len(loader)
        if is_main():
            log = {'epoch_loss': avg_loss, 'epoch': epoch}
            print(f'Epoch {epoch}/{epochs}  Loss: {avg_loss:.4f}', end='')

            # save the per-epoch snapshot first (always)
            reranker.save(os.path.join(checkpoint_dir, f'epoch_{epoch}'))

            # ---- per-epoch pipeline eval (BERT+GRU candidates already cached) ----
            if eval_enabled:
                eval_results = _pipeline_eval(
                    reranker, eval_candidates, eval_queries, eval_docs, eval_qrels,
                    rerank_top_k=eval_rerank_top_k, batch_size=eval_batch_size,
                )
                eval_ndcg   = eval_results['mean_ndcg@1000']
                eval_recall = eval_results['mean_r@100']
                log.update({'eval_ndcg@1000': eval_ndcg, 'eval_r@100': eval_recall})
                print(f'  |  eval NDCG@1000: {eval_ndcg:.4f}  R@100: {eval_recall:.4f}')

                # use NDCG as the best-checkpoint criterion when eval is enabled
                if eval_ndcg > best_ndcg:
                    best_ndcg = eval_ndcg
                    reranker.save(os.path.join(checkpoint_dir, 'best'))
                    print(f'  ★ new best NDCG@1000 — saved → {checkpoint_dir}/best')
            else:
                print()  # newline after loss
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    reranker.save(os.path.join(checkpoint_dir, 'best'))

            if WANDB_AVAILABLE and wandb_project:
                wandb.log(log)

        # keep ranks in sync before next epoch (no rank starts forward pass
        # while rank 0 is still mid-save / mid-eval)
        if ddp:
            dist.barrier(device_ids=[reranker.device_id])

    if is_main():
        reranker.save(os.path.join(checkpoint_dir, 'latest'))
        if eval_enabled:
            print(f'Reranker training complete. Best dev NDCG@1000: {best_ndcg:.4f}')
        else:
            print(f'Reranker training complete. Best loss: {best_loss:.4f}')

    if is_main() and WANDB_AVAILABLE and wandb_project:
        wandb.finish()


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
    # 0. DDP setup — when launched via torchrun, each process gets its own
    # LOCAL_RANK and trains on a different GPU. Single-GPU runs skip this.
    if is_ddp():
        local_rank = setup_ddp()
    else:
        local_rank = 0

    if is_main():
        world_size = dist.get_world_size() if is_ddp() else 1
        gpus = ', '.join(f'cuda:{r}' for r in range(world_size))
        print(f'DDP: {is_ddp()}  |  World size: {world_size}  |  GPUs in use: [{gpus}]')

    # 1. initialize reranker — model lands on this rank's GPU
    reranker = LoRAReranker(
        model_name='Qwen/Qwen2.5-1.5B',   # was 0.5B
        load_in_4bit=False,                # fp32 fits comfortably on 24 GB GPUs
        lora_r=16,
        lora_alpha=32,
        torch_dtype=torch.float32,
        device_id=local_rank,
    )

    # 2. build training dataset (needs bi-encoder candidates pre-generated)
    dataset = RerankerDataset(
        query_path='data/trec/train_queries.jsonl',
        doc_path='data/trec/docs.jsonl',
        qrel_path='data/trec/train_qrels.txt',
        candidates_path='runs/bi_encoder_train_candidates.jsonl',
        n_negatives=3,
        reddit_query_path='data/reddit/queries.jsonl',
    )
    if is_main():
        print(f'Total reranker training examples: {len(dataset)}')

    # 2b. per-epoch validation pipeline (rank 0 only — needs full BERT+GRU
    # bi-encoder briefly to compute dev candidates, then frees the GPU before
    # reranker training starts). Set EVAL_BI_CHECKPOINT to None to disable.
    EVAL_BI_CHECKPOINT = 'checkpoints/best'
    EVAL_SUBSET        = 30            # 1.5B inference is ~2.5× slower than 0.5B,
                                       # so trim subset to keep epoch overhead similar
    EVAL_RERANK_TOP_K  = 100

    eval_candidates = eval_queries_d = eval_docs_d = eval_qrels_d = None
    if is_main() and EVAL_BI_CHECKPOINT is not None:
        device = torch.device(f'cuda:{local_rank}')
        eval_candidates, eval_queries_d, eval_docs_d, eval_qrels_d = build_dev_candidates(
            bi_checkpoint=EVAL_BI_CHECKPOINT,
            queries_path='data/trec/dev_queries.jsonl',
            qrels_path='data/trec/dev_qrels.txt',
            docs_path='data/trec/docs.jsonl',
            bert_model='bert-base-uncased',
            proj_dim=512,
            device=device,
            subset=EVAL_SUBSET,
            k=1000,
        )

    # sync ranks so non-zero ranks don't start training before rank 0 finishes
    # bi-encoder retrieval (avoids GPU memory contention)
    if is_ddp():
        dist.barrier(device_ids=[local_rank])

    # 3. fine-tune
    # 2 epochs — gives one extra dev-NDCG data point vs 1 epoch, so we can see
    # whether the model is still improving or already saturating. batch_size=2
    # keeps memory headroom (effective batch=4 across 2 ranks).
    train_reranker(reranker, dataset, epochs=2, lr=5e-5, batch_size=2,
                   wandb_project='tot-reranker', wandb_run='qwen2.5-1.5b-lora',
                   warmup_ratio=0.05,
                   eval_candidates=eval_candidates,
                   eval_queries=eval_queries_d,
                   eval_docs=eval_docs_d,
                   eval_qrels=eval_qrels_d,
                   eval_rerank_top_k=EVAL_RERANK_TOP_K)

    # 4. load best checkpoint (rank 0 only — eval is single-GPU)
    if is_main():
        reranker_best = LoRAReranker(
            model_name='Qwen/Qwen2.5-1.5B',   # must match training model
            load_in_4bit=False,
            torch_dtype=torch.float32,
            checkpoint_path='checkpoints/reranker/best',
            device_id=local_rank,
        )

    cleanup_ddp()
