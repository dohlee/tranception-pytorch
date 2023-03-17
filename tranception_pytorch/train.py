import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import tranception_pytorch.util as util
import wandb
from tranception_pytorch import Tranception
from tranception_pytorch.data import MaskedProteinDataset

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Performance drops, so commenting out for now.
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

def cycle(loader, n):
    """Cycle through a dataloader indefinitely."""
    cnt, to_stop = 0, False
    while not to_stop:
        for batch in loader:
            yield batch

            cnt += 1
            if cnt == n:
                to_stop = True
                break

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', help='Output prefix.', required=True)
    parser.add_argument('--batch-size', type=int, default=1024)             # Taken from Table 8.
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1)
    parser.add_argument('--annealing-steps', type=int, default=10_000)      # Taken from Appendix B.3.
    parser.add_argument('--total-steps', type=int, default=150_000)         # Taken from Table 8.
    parser.add_argument('--peak-lr', type=float, default=3e-4)              # Taken from Table 8.
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--use-wandb', action='store_true', default=False)
    args = parser.parse_args()

    seed_everything(args.seed)
    if not args.use_wandb:
        os.environ['WANDB_MODE'] = 'disabled'

    wandb.init(project='tranception-pytorch', config=args, reinit=True)

    # Testing Tranception S. Hyperparameters taken from Table 5.
    num_heads = 12
    num_layers = 12
    embed_dim = 768
    max_length = 1024

    model = Tranception(
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        max_length=max_length,
    )
    model = model.cuda()

    optimizer = optim.AdamW(  # AdamW taken from Table 8.
        model.parameters(),
        lr=args.peak_lr,
        weight_decay=1e-4,
    )
    scheduler = util.LinearAnnealingLR(
        optimizer,
        num_annealing_steps=args.annealing_steps,
        num_total_steps=args.total_steps,
    )
    criterion = nn.CrossEntropyLoss(reduction='none')

    train_set = MaskedProteinDataset(
        args.input,
        mask_prob=0.15,
        mask_token=20,
        max_len=1024,
        p_reverse=0.5,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=32,
        pin_memory=True
    )

    print('Starting training...')
    model.train()

    cnt = 0
    optimizer.zero_grad()
    running_loss = []
    for batch in cycle(train_loader, args.total_steps * args.gradient_accumulation_steps):
        seq, masked_seq, mask = batch['seq'].cuda(), batch['masked_seq'].cuda(), batch['mask'].cuda()
        # Note that seq is not one-hot encoded. It's just a sequence of integers.
        out = model(masked_seq)             # (batch_size, seq_len, vocab_size)

        # Loss is computed only for masked positions. (where mask==1)
        loss = criterion(out.view(-1, 21), seq.view(-1)) * mask.view(-1)   # (batch_size, seq_len)
        loss = loss.sum() / mask.sum()
        running_loss.append(loss.item())

        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        if cnt % args.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        if cnt % 100 == 0:
            print(f'Iteration {cnt}, loss={np.mean(running_loss):.4f}')
            wandb.log({
                'train/loss': np.mean(running_loss),
                'train/lr': get_lr(optimizer),
                'train/step': cnt // args.gradient_accumulation_steps
            })
            running_loss = []

        if (cnt // args.gradient_accumulation_steps) % 25000 == 0:
            if not os.path.exists(os.path.dirname(args.output)):
                os.makedirs(os.path.dirname(args.output))

            idx = cnt // args.gradient_accumulation_steps
            torch.save(model.state_dict(), f'{args.output}_{idx}.pt')

        cnt += 1

if __name__ == '__main__':
    main()
