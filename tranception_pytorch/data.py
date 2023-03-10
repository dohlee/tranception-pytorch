import torch
import random
import h5py

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

a2i = dict(zip('ACDEFGHIKLMNPQRSTVWY', range(20)))
replace_dict = {
    'X': 'ACDEFGHIKLMNPQRSTVWY',
    'B': 'DN',
    'Z': 'EQ',
    'J': 'IL',

    # Just dummy
    'O': 'A',
    'U': 'A',
}
def is_valid_sequence(seq):
    return not any(a in 'OU' for a in seq) and seq.count('X') < 2

def replace_with_random(seq, replace_dict):
    return ''.join([random.choice(replace_dict.get(a, a)) for a in seq])

class MaskedProteinDataset(Dataset):
    def __init__(self, h5_fp, mask_prob=0.15, mask_token=20, max_len=1024, p_reverse=0.5):
        f = h5py.File(h5_fp, 'r')
        self.records = f['seq']
        
        self.mask_prob = mask_prob
        self.mask_token = mask_token
        self.max_len = max_len
        self.p_reverse = p_reverse

    def _preprocess_records(self, records):
        # Remove records with non-standard amino acids (O and U).
        # And remove records with two or more X's.
        records = [r for r in tqdm(records) if is_valid_sequence(r.seq)]
        return records
    
    def _crop_seq_to_max_len(self, seq):
        start = random.randint(0, len(seq) - self.max_len)
        return seq[start:start + self.max_len]

    def __len__(self):
        return len(self.records)

    def __del__(self):
        self.records.file.close()

    def __getitem__(self, idx):
        seq = self.records[idx].decode()  # Convert bytes to string.
        valid = is_valid_sequence(seq)

        # Impute non-standard amino acids.
        seq = replace_with_random(seq, replace_dict)

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
            'seq': seq if random.random() < self.p_reverse else seq.flip(0),
            'masked_seq': masked_seq,
            # Invalid sequences do not participate in training.
            'mask': mask if valid else torch.zeros_like(mask),
        }
    
if __name__ == '__main__':
    dataset = MaskedProteinDataset('../data/uniref50.h5')
    loader = DataLoader(dataset, num_workers=16, batch_size=32)

    for batch in tqdm(loader):
        # print(batch['seq'])
        # print(batch['seq'].shape)

        # print(batch['mask'])
        # print(batch['mask'].shape)
        pass
