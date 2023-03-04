import argparse
import gzip
import re
import random
random.seed(42)

from Bio import SeqIO
from Bio.Seq import Seq
from tqdm import tqdm
from collections import Counter

replace_dict = {
    'X': 'ACDEFGHIKLMNPQRSTVWY',
    'B': 'DN',
    'Z': 'EQ',
    'J': 'IL',
}

def get_cluster_size(r):
    return int(re.search('n=(\d+)', r.description).group(1))

def process_seq(seq, replace_dict):
    return ''.join([random.choice(replace_dict.get(a, a)) for a in seq])

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-s', '--sample', type=float, default=1.0)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    to_sample = args.sample != 1.0
    
    final_records = []
    with gzip.open(args.input, 'rt') as inFile:
        for r in tqdm(SeqIO.parse(inFile, 'fasta')):
            if to_sample and random.random() >= args.sample:
                continue

            if get_cluster_size(r) == 1:
                continue
            
            aa_count = Counter(r.seq)
            if ('O' in aa_count) or ('U' in aa_count) or aa_count['X'] >= 2:
                continue

            r.seq = Seq(process_seq(r.seq, replace_dict))
            final_records.append(r)
    
    with open(args.output, 'w') as outFile:
        SeqIO.write(final_records, outFile, 'fasta')

    print(f'Done! Parsed {len(final_records)} sequences.')
