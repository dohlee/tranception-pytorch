import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import tqdm
import os
import wandb
import util

from torch.utils.data import Dataset, DataLoader
from scipy.stats import pearsonr, spearmanr
from torch.distributions.multinomial import Multinomial

from tranception_pytorch import Tranception
from tranception_pytorch.data import MaskedProteinDataset

def train(model, train_loader, optimizer, criterion, metrics_f):
    model.train()

    running_profiles, running_total_counts = [], []
    running_profile_labels, running_total_count_labels = [], []

    # Training loop with progressbar.
    bar = tqdm.tqdm(train_loader, total=len(train_loader), leave=False)
    for idx, batch in enumerate(bar):
        seq = batch['seq'].cuda()
        profile = batch['profile'].cuda()
        total_count = batch['total_count'].cuda()

        optimizer.zero_grad()
        out = model(seq)
        loss = multinomial_nll(out['profile'], profile) + F.mse_loss(torch.log(1 + out['total_count']), torch.log(1 + total_count))
        loss.backward()
        optimizer.step()

        running_profiles.append(out['profile'].detach().cpu())
        running_total_counts.append(out['total_count'].detach().cpu())
        running_profile_labels.append(profile.cpu())
        running_total_count_labels.append(total_count.cpu())

        if idx % 100 == 0:
            running_profiles = torch.cat(running_profiles, dim=0)
            running_total_counts = torch.cat(running_total_counts, dim=0)
            running_profile_labels = torch.cat(running_profile_labels, dim=0)
            running_total_count_labels = torch.cat(running_total_count_labels, dim=0)

            running_loss = multinomial_nll(running_profiles, running_profile_labels) + F.mse_loss(torch.log(1 + running_total_counts), torch.log(1 + running_total_count_labels))

            loss = running_loss.item()
            bar.set_postfix(loss=loss)
            wandb.log({
                'train/loss': loss,
            })

            running_output, running_label = [], []

def validate(model, val_loader, criterion, metrics_f):
    model.eval()

    out_fwd, out_rev, label = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            wt_emb, mut_emb = batch['wt_emb'].cuda(), batch['mut_emb'].cuda()
            _label = batch['label'].cuda().flatten()

            _out_fwd = model(wt_emb, mut_emb).flatten()
            _out_rev = model(mut_emb, wt_emb).flatten()  # Swap wt_emb and mut_emb.

            out_fwd.append(_out_fwd.cpu())
            out_rev.append(_out_rev.cpu())

            label.append(_label.cpu())
        
    out_fwd = torch.cat(out_fwd, dim=0)
    out_rev = torch.cat(out_rev, dim=0)
    label = torch.cat(label, dim=0)

    loss = criterion(out_fwd, label).item()
    metrics = {k: f(out_fwd, label) for k, f in metrics_f.items()}

    # Add antisymmetry metrics.
    metrics['pearson_fr'] = pearsonr(out_fwd, out_rev)[0] 
    metrics['delta'] = torch.cat([out_fwd, out_rev], dim=0).mean()

    wandb.log({
        'val/loss': loss,
        'val/pearson': metrics['pearson'],
        'val/spearman': metrics['spearman'],
        'val/pearson_fr': metrics['pearson_fr'],
        'val/delta': metrics['delta'],
    })

    return loss, metrics

def test(model, val_loader, criterion, metrics_f):
    model.eval()

    out_fwd, out_rev, label = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            wt_emb, mut_emb = batch['wt_emb'].cuda(), batch['mut_emb'].cuda()
            _label = batch['label'].cuda().flatten()

            _out_fwd = model(wt_emb, mut_emb).flatten()
            _out_rev = model(mut_emb, wt_emb).flatten()  # Swap wt_emb and mut_emb.

            out_fwd.append(_out_fwd.cpu())
            out_rev.append(_out_rev.cpu())

            label.append(_label.cpu())
        
    out_fwd = torch.cat(out_fwd, dim=0)
    out_rev = torch.cat(out_rev, dim=0)
    label = torch.cat(label, dim=0)

    loss = criterion(out_fwd, label).item()
    metrics = {k: f(out_fwd, label) for k, f in metrics_f.items()}

    # Add antisymmetry metrics.
    metrics['pearson_fr'] = pearsonr(out_fwd, out_rev)[0] 
    metrics['delta'] = torch.cat([out_fwd, out_rev], dim=0).mean()

    wandb.log({
        'test/loss': loss,
        'test/pearson': metrics['pearson'],
        'test/spearman': metrics['spearman'],
        'test/pearson_fr': metrics['pearson_fr'],
        'test/delta': metrics['delta'],
    })

    return loss, metrics

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
    cnt, stop_flag = 0, False
    while stop_flag is False:
        for batch in loader:
            yield batch

            cnt += 1
            if cnt == n:
                stop_flag = True
                break

def main():
    import pandas as pd
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--batch-size', type=int, default=1024)             # Taken from Table 8.
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

    train_set = MaskedProteinDataset(args.input)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)

    model = Tranception()
    model = model.cuda()

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)  # AdamW taken from Table 8.
    scheduler = util.LinearAnnealingLR(optimizer, args.steps, args.peak_lr)
    criterion = nn.CrossEntropyLoss(reduction='none')

    model.train()
    for batch in cycle(train_loader, args.steps):
        # train(model, train_loader, optimizer, criterion)
        seq, masked_seq, mask = batch['seq'].cuda(), batch['masked_seq'].cuda(), batch['mask'].cuda()

        optimizer.zero_grad()
        out = model(masked_seq)
        loss = criterion(out, seq) * mask
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            print(loss)

        scheduler.step()

if __name__ == '__main__':
    main()
