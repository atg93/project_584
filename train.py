import os
import json
import argparse
import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from tqdm import tqdm

from tot_retrieval import QueryEncoder, DocEncoder, contrastive_loss
from dataset import get_trec_dataloader, get_reddit_dataloader, get_doc_dataloader, load_qrels
from evaluate import evaluate, print_results

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def setup_ddp():
    dist.init_process_group(backend='nccl', timeout=datetime.timedelta(hours=4))
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_ddp():
    dist.destroy_process_group()


def is_main():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(query_enc, doc_enc, optimizer, epoch, metric, checkpoint_dir):
    os.makedirs(checkpoint_dir, exist_ok=True)
    # unwrap DDP before saving
    q_state = query_enc.module.state_dict() if hasattr(query_enc, 'module') else query_enc.state_dict()
    d_state = doc_enc.module.state_dict()   if hasattr(doc_enc,   'module') else doc_enc.state_dict()
    torch.save(q_state, os.path.join(checkpoint_dir, 'query_enc.pt'))
    torch.save(d_state, os.path.join(checkpoint_dir, 'doc_enc.pt'))
    meta = {'epoch': epoch, 'ndcg@1000': metric}
    with open(os.path.join(checkpoint_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print(f'  Checkpoint saved → {checkpoint_dir}  (NDCG@1000: {metric:.4f})')


def load_checkpoint(query_enc, doc_enc, checkpoint_dir):
    query_enc.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'query_enc.pt')))
    doc_enc.load_state_dict(  torch.load(os.path.join(checkpoint_dir, 'doc_enc.pt')))
    with open(os.path.join(checkpoint_dir, 'meta.json')) as f:
        meta = json.load(f)
    print(f'  Loaded checkpoint from epoch {meta["epoch"]}  (NDCG@1000: {meta["ndcg@1000"]:.4f})')
    return meta


# ---------------------------------------------------------------------------
# One training epoch
# ---------------------------------------------------------------------------

def train_one_epoch(query_enc, doc_enc, dataloader, optimizer, scheduler, device, epoch):
    query_enc.train()
    doc_enc.train()

    # sync all ranks to the same epoch start
    if isinstance(dataloader.sampler, DistributedSampler):
        dataloader.sampler.set_epoch(epoch)

    total_loss = 0.0
    progress   = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False, disable=not is_main())

    for batch in progress:
        sentence_ids   = batch['sentence_ids'].to(device)
        sentence_masks = batch['sentence_masks'].to(device)
        doc_ids        = batch['doc_input_ids'].to(device)
        doc_masks      = batch['doc_attention_mask'].to(device)

        q_vecs = query_enc(sentence_ids, sentence_masks)
        d_vecs = doc_enc(doc_ids, doc_masks)

        loss = contrastive_loss(q_vecs, d_vecs)
        loss.backward()

        nn.utils.clip_grad_norm_(
            list(query_enc.parameters()) + list(doc_enc.parameters()), max_norm=1.0
        )

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if is_main():
            progress.set_postfix(loss=f'{loss.item():.4f}')

    return total_loss / len(dataloader)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args):
    # --- DDP setup ---
    ddp = 'LOCAL_RANK' in os.environ
    if ddp:
        local_rank = setup_ddp()
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0

    if is_main():
        print(f'Device: {device}  |  DDP: {ddp}  |  World size: {dist.get_world_size() if ddp else 1}')
        if WANDB_AVAILABLE and args.wandb_project:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run,
                config=vars(args),
            )
            print(f'WandB run: {wandb.run.url}')

    # --- models ---
    query_enc = QueryEncoder(
        bert_model=args.bert_model,
        gru_hidden=args.gru_hidden,
        sentence_dropout=args.sentence_dropout,
        proj_dim=args.proj_dim,
    ).to(device)

    doc_enc = DocEncoder(bert_model=args.bert_model, proj_dim=args.proj_dim).to(device)

    # optionally resume from checkpoint
    best_ndcg = 0.0
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        meta = load_checkpoint(query_enc, doc_enc, args.resume)
        best_ndcg   = meta['ndcg@1000']
        start_epoch = meta['epoch'] + 1

    # wrap with DDP after loading weights
    if ddp:
        query_enc = DDP(query_enc, device_ids=[local_rank], find_unused_parameters=True)
        doc_enc   = DDP(doc_enc,   device_ids=[local_rank], find_unused_parameters=True)

    # --- optimizer & scheduler ---
    optimizer = AdamW(
        list(query_enc.parameters()) + list(doc_enc.parameters()),
        lr=args.lr, weight_decay=args.weight_decay
    )

    # --- dataloaders (DistributedSampler when DDP) ---
    trec_loader = get_trec_dataloader(
        query_path=args.trec_train_queries,
        doc_path=args.docs,
        qrel_path=args.trec_train_qrels,
        batch_size=args.batch_size,
        tokenizer_name=args.bert_model,
        shuffle=not ddp,
        distributed=ddp,
    )

    total_steps  = args.epochs * len(trec_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler    = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)

    reddit_loader = None
    if args.reddit_queries:
        reddit_loader = get_reddit_dataloader(
            query_path=args.reddit_queries,
            doc_path=args.docs,
            batch_size=args.batch_size,
            tokenizer_name=args.bert_model,
            shuffle=not ddp,
            domain=args.reddit_domain,
            distributed=ddp,
        )
        if is_main():
            print(f'Reddit ToT dataset loaded — {len(reddit_loader.dataset)} queries'
                  + (f' (domain={args.reddit_domain})' if args.reddit_domain else ''))

    dev_loader = None
    dev_qrels  = None
    doc_loader = None
    if args.trec_dev_queries:
        dev_loader = get_trec_dataloader(
            query_path=args.trec_dev_queries,
            doc_path=args.docs,
            qrel_path=args.trec_dev_qrels,
            batch_size=args.batch_size,
            tokenizer_name=args.bert_model,
            shuffle=False,
            distributed=False,  # eval always on main process only
        )
        dev_qrels  = load_qrels(args.trec_dev_qrels)
        doc_loader = get_doc_dataloader(args.docs, batch_size=args.eval_batch_size,
                                        tokenizer_name=args.bert_model)

    if is_main():
        print(f'\nTREC training queries : {len(trec_loader.dataset)}')
        print(f'Total epochs          : {args.epochs}')
        print(f'Batch size per GPU    : {args.batch_size}')
        print(f'Learning rate         : {args.lr}\n')

    # --- training loop ---
    for epoch in range(start_epoch, args.epochs + 1):

        trec_loss = train_one_epoch(
            query_enc, doc_enc, trec_loader, optimizer, scheduler, device, epoch
        )
        if is_main():
            print(f'Epoch {epoch}/{args.epochs}  TREC loss: {trec_loss:.4f}', end='')

        reddit_loss = None
        if reddit_loader:
            reddit_loss = train_one_epoch(
                query_enc, doc_enc, reddit_loader, optimizer, scheduler, device, epoch
            )
            if is_main():
                print(f'  Reddit loss: {reddit_loss:.4f}', end='')

        if is_main():
            print()

        # eval and checkpointing only on rank 0
        if is_main():
            log = {'epoch': epoch, 'trec_loss': trec_loss}
            if reddit_loss is not None:
                log['reddit_loss'] = reddit_loss

            if dev_loader and epoch % args.eval_every == 0:
                print(f'  Evaluating on dev set...')
                q_eval = query_enc.module if hasattr(query_enc, 'module') else query_enc
                d_eval = doc_enc.module   if hasattr(doc_enc,   'module') else doc_enc
                results = evaluate(q_eval, d_eval, dev_loader, doc_loader, dev_qrels, device)
                print_results(results)

                ndcg = results['mean_ndcg@1000']
                log['ndcg@1000'] = ndcg
                log['r@100']     = results['mean_r@100']

                if ndcg > best_ndcg:
                    best_ndcg = ndcg
                    save_checkpoint(query_enc, doc_enc, optimizer, epoch,
                                    ndcg, os.path.join(args.checkpoint_dir, 'best'))

            if WANDB_AVAILABLE and args.wandb_project:
                wandb.log(log)

            save_checkpoint(query_enc, doc_enc, optimizer, epoch,
                            best_ndcg, os.path.join(args.checkpoint_dir, 'latest'))

        # sync all ranks after rank 0 finishes eval/checkpoint
        if ddp:
            dist.barrier(device_ids=[local_rank])

    if is_main():
        print(f'\nTraining complete. Best NDCG@1000: {best_ndcg:.4f}')
        if WANDB_AVAILABLE and args.wandb_project:
            wandb.finish()

    if ddp:
        cleanup_ddp()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Train BERT-GRU bi-encoder for ToT retrieval')

    # data
    p.add_argument('--docs',               default='data/trec/docs.jsonl')
    p.add_argument('--trec-train-queries', default='data/trec/train_queries.jsonl')
    p.add_argument('--trec-train-qrels',   default='data/trec/train_qrels.txt')
    p.add_argument('--trec-dev-queries',   default='data/trec/dev_queries.jsonl')
    p.add_argument('--trec-dev-qrels',     default='data/trec/dev_qrels.txt')
    p.add_argument('--reddit-queries',     default=None)
    p.add_argument('--reddit-domain',      default=None)

    # model
    p.add_argument('--bert-model',         default='bert-base-uncased')
    p.add_argument('--gru-hidden',         type=int,   default=None,
                   help='GRU hidden size (default: same as BERT hidden size)')
    p.add_argument('--sentence-dropout',   type=float, default=0.2)
    p.add_argument('--proj-dim',           type=int,   default=512)

    # training
    p.add_argument('--epochs',             type=int,   default=10)
    p.add_argument('--batch-size',         type=int,   default=16)
    p.add_argument('--lr',                 type=float, default=2e-5)
    p.add_argument('--weight-decay',       type=float, default=0.01)
    p.add_argument('--eval-every',         type=int,   default=1)
    p.add_argument('--eval-batch-size',    type=int,   default=256,
                   help='Batch size for document encoding during evaluation')

    # checkpoints
    p.add_argument('--checkpoint-dir',     default='checkpoints')
    p.add_argument('--resume',             default=None)

    # wandb
    p.add_argument('--wandb-project',      default=None,
                   help='WandB project name (omit to disable WandB)')
    p.add_argument('--wandb-run',          default=None,
                   help='WandB run name (optional)')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = parse_args()
    train(args)
