#!/usr/bin/env python3

import sys

import numpy as np

class PREPROCESS_MSA():
    """
    Preprocessing for an MSA to remove sequences and columns with too many gaps
    """
    def __init__(self,names, seqs, row = 0.25, col = 0.75):
 
        #filter thresholds
        self.row = row
        self.column = col
 
        #load msa information into numpy
        seqs = np.array([[*aas] for aas in seqs], dtype="<U1")
        names = np.array(names, dtype=np.string_)
 
        print('MSA started at {:} rows and {:} columns'.format(seqs.shape[0],seqs.shape[1]))
 
        print("Checking for gaps in query sequence...")
        remove_columns = np.where(seqs[0] == '-')
        self.seqs = np.delete(seqs, remove_columns, axis=1) #initial msa with query gaps removed
        self.names = names
        self.aa = np.array([idx for idx, aa in enumerate(self.seqs[0])])
        #__________________________________________________________
        print('MSA has been reduced to {:} rows and {:} columns'.format(self.seqs.shape[0],self.seqs.shape[1]))
 
    def reduce_gaps(self):
        """
        remove rows and columns that contain too many gaps for coevolutionary analysis
        """
        #
        #__________________________________________________________
        gap_count = np.sum(self.seqs == '-', axis=1)
        gap_percent = gap_count / self.seqs.shape[1]
        remove_rows = np.where(gap_percent >= 0.25) #if row is more than 25% gaps remove seq
        self.seqs = np.delete(self.seqs, remove_rows, axis=0)
        self.names = np.delete(self.names, remove_rows, axis=0)
        self.names = [names.decode('utf-8') for names in self.names]
        print('MSA has been reduced to {:} rows and {:} columns'.format(self.seqs.shape[0],self.seqs.shape[1]))
        #
        #__________________________________________________________
        gap_count = np.sum(self.seqs == '-', axis=0)
        gap_percent = gap_count / self.seqs.shape[0]
        remove_cols = np.where(gap_percent >= 0.75) #if column is more than 75% gaps remove seq
        self.seqs = np.delete(self.seqs, remove_cols, axis=1)
        self.aa = np.delete(self.aa, remove_cols)
        print('MSA has been reduced to {:} rows and {:} columns'.format(self.seqs.shape[0],self.seqs.shape[1]))
        #__________________________________________________________

def CL_input():
    """
    Parse command line arguments being passed in
    """
    if not any((True if _ == '-fasta' else False for _ in sys.argv)) or len(sys.argv) < 3:
        print('Missing command line arguments!')
        print('Available flags:')
        print("-fasta ####.fa           | multi-fasta file of multiple sequence alignment")
        print("-filter_msa True/[False] | remove sequences that are more than 25% gap and columns that are more than 75% gaps")
        print("-filter_msa_row [0.25]   | remove sequences that are more than [x]% gap")
        print("-filter_msa_col [0.75]   | remove  columns  that are more than [x]% gap")
        sys.exit()

    fasta = sys.argv[[idx for idx, _ in enumerate(sys.argv) if '-fasta' == _][0] + 1]
    filter_msa = True if any((True if _ == '-filter_msa' else False for _ in sys.argv)) else False
    filter_msa_row = 0.25 if not [idx for idx, _ in enumerate(sys.argv) if '-filter_msa_row' == _] else sys.argv[[idx for idx, _ in enumerate(sys.argv) if '-filter_msa_row' == _][0] + 1]

    filter_msa_col = 0.75 if not [idx for idx, _ in enumerate(sys.argv) if '-filter_msa_col' == _] else sys.argv[[idx for idx, _ in enumerate(sys.argv) if '-filter_msa_col' == _][0] + 1]
    return fasta, filter_msa, filter_msa_row, filter_msa_col

def Read_MSA(file_path: str):
    """
    read in multi-fasta format and separate names and sequences
    """
    with open(file_path, 'r') as file:
        _ = np.array([line.strip() for line in file])
    names = _[::2]; seqs =  _[1::2]
    return names, seqs


def main(fasta, filter_msa, filter_msa_row, filter_msa_col):
    print(f"Running {fasta.split('.')[0]}...")
    preprocess_msa = PREPROCESS_MSA(*Read_MSA(fasta))

    print(filter_msa)
    if filter_msa:
        preprocess_msa.reduce_gaps()
        seqs = preprocess_msa.seqs
        names = preprocess_msa.names
        aa = preprocess_msa.aa
    else:
        print(f"Running MSA without preprocessing")
        seqs = preprocess_msa.seqs
        names = preprocess_msa.names
        aa = preprocess_msa.aa
   
    print(seqs)
if  __name__ == '__main__':
    main(*CL_input())

