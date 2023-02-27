import torch
import torch.nn as nn
import random
import gzip

from torch.utils.data import Dataset, DataLoader
from Bio import SeqIO

a2i = dict(zip('ACDEFGHIKLMNPQRSTVWY', range(20)))

class MaskedProteinDataset(Dataset):
    def __init__(self, fasta_fp, mask_prob=0.15, mask_token=20, max_len=1024):
        if fasta_fp.endswith('.gz'):
            with gzip.open(fasta_fp, 'rt') as fp:
                self.records = list(SeqIO.parse(fp, 'fasta'))
        else:
            self.records = list(SeqIO.parse(fasta_fp, 'fasta'))
        
        print(f'Loaded {len(self.records)} records from {fasta_fp}.')

        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.max_len = max_len
    
    def _crop_seq_to_max_len(self, seq):
        start = random.randint(0, len(seq) - self.max_len)
        return seq[start:start + self.max_len]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        seq = self.records[idx].seq.upper()

        # Crop sequence.
        if len(seq) > self.max_len:
            seq = self._crop_seq_to_max_len(seq)
        
        # Convert to integer sequence.
        seq = torch.tensor([a2i[a] for a in seq], dtype=torch.long)
        
        # Mask sequence.
        masked_seq = seq.clone()
        mask = torch.rand(len(seq)) < self.mask_prob
        masked_seq[mask] = self.mask_token

        # Pad sequence.
        if len(seq) < self.max_len:
            seq = torch.cat([seq, torch.zeros(self.max_len - len(seq), dtype=torch.long)])
            masked_seq = torch.cat([masked_seq, torch.zeros(self.max_len - len(masked_seq), dtype=torch.long)])
            mask = torch.cat([mask, torch.zeros(self.max_len - len(mask), dtype=torch.bool)])

        return {
            'seq': seq,
            'masked_seq': masked_seq,
            'mask': mask,
        }
    
if __name__ == '__main__':
    dataset = MaskedProteinLMDataset('../data/uniprot_sprot.fasta.gz')
    loader = DataLoader(dataset, batch_size=16)

    for batch in loader:
        print(batch['seq'])
        print(batch['seq'].shape)

        print(batch['mask'])
        print(batch['mask'].shape)

        break

