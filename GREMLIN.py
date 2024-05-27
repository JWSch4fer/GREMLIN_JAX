#!/usr/bin/env python3

import sys

import numpy as np

class Edit_MSA():
    def __init__(self,msa,name):
 
        #filter thresholds
        self.row = 0.25
        self.column = 0.75
 
        #load msa information into numpy
        msa = np.array([[*aas] for aas in msa], dtype="<U1")
        name = np.array(name, dtype=np.string_)
 
        print('msa started at {:} rows and {:} columns'.format(msa.shape[0],msa.shape[1]))
 
        remove_columns = np.where(msa[0] == '-')
        self.msa = np.delete(msa, remove_columns, axis=1) #initial msa with query gaps removed
        self.name = name
        self.aa = np.array([idx for idx, aa in enumerate(self.msa[0])])
        #__________________________________________________________
        print('msa has been reduced to {:} rows and {:} columns'.format(self.msa.shape[0],self.msa.shape[1]))
 
    def Clean_msa(self):
        """
        remove rows and columns that contain too many gaps for coevolutionary analysis
        """
 
        #
        #__________________________________________________________
        gap_count = np.sum(self.msa == '-', axis=1)
        gap_percent = gap_count / self.msa.shape[1]
        remove_rows = np.where(gap_percent >= 0.25) #if row is more than 25% gaps remove seq
        self.msa = np.delete(self.msa, remove_rows, axis=0)
        self.name = np.delete(self.name, remove_rows, axis=0)
        self.name = [name.decode('utf-8') for name in self.name]
        print('msa has been reduced to {:} rows and {:} columns'.format(self.msa.shape[0],self.msa.shape[1]))
        #
        #__________________________________________________________
        gap_count = np.sum(self.msa == '-', axis=0)
        gap_percent = gap_count / self.msa.shape[0]
        remove_cols = np.where(gap_percent >= 0.75) #if column is more than 75% gaps remove seq
        self.msa = np.delete(self.msa, remove_cols, axis=1)
        self.aa = np.delete(self.aa, remove_cols)
        print('msa has been reduced to {:} rows and {:} columns'.format(self.msa.shape[0],self.msa.shape[1]))
        return pd.DataFrame(self.msa,index=(self.name),columns=self.aa)
        #__________________________________________________________

def CL_input():
    """
    Parse command line arguments being passed in
    """
    if not any((True if _ == '-pdb' else False for _ in sys.argv)) or len(sys.argv) < 2:
        print('Missing command line arguments!')
        print('Available flags:')
        print("-fasta ####.fa         | multi-fasta file of multiple sequence alignment")
        print("-filter_msa [Y]/N      | remove sequences that are more than 25% gap and columns that are more than 75% gaps")
        print("-filter_msa_row [0.25] | remove sequences that are more than [x]% gap")
        print("-filter_msa_col [0.75] | remove  columns  that are more than [x]% gap")
        sys.exit()

    fasta = sys.argv[[idx for idx, _ in enumerate(sys.argv) if '-fasta' == _][0] + 1]
    filter_msa = 'Y' if not [idx for idx, _ in enumerate(sys.argv) if '-filter_msa' == _] else sys.argv[[idx for idx, _ in enumerate(sys.argv) if '-filter_msa' == _][0] + 1]
    filter_msa_row = 0.25 if not [idx for idx, _ in enumerate(sys.argv) if '-filter_msa_row' == _] else sys.argv[[idx for idx, _ in enumerate(sys.argv) if '-filter_msa_row' == _][0] + 1]
    filter_msa_col = 0.75 if not [idx for idx, _ in enumerate(sys.argv) if '-filter_msa_col' == _] else sys.argv[[idx for idx, _ in enumerate(sys.argv) if '-filter_msa_col' == _][0] + 1]
    return fasta, filter_msa, filter_msa_row, filter_msa_col

def main(fasta, filter_msa, filter_msa_row, filter_msa_col):
    print(f"Running {fasta.split('.')[0]}...")
    if filter_msa == 'Y':
        print(f"Removing MSA sequnces with too many gaps: {filter_msa_row}")
        print(f"Removing MSA columns  with too many gaps: {filter_msa_col}")
    else:
        print(f"Running MSA without preprocessing")
    

if  __name__ == '__main__':
    main(*CL_input())

