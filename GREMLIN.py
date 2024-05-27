#!/usr/bin/env python3

import sys

from scipy.spatial.distance import pdist, squareform
import numpy as np
import optax
import jax
import jax.numpy as jnp
from jax.numpy import array
from jax.scipy.special import logsumexp

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
        # shuffle sequences before running MRF
        #__________________________________________________________
        np.random.shuffle(self.seqs[1:])


 
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
        self.aa_edit = np.delete(self.aa, remove_cols)
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

def Translate_Msa(seqs : array):
    # + = positively charged at pH 7.4
    # - = negatively charged at pH 7.4
    # P =    polar at pH 7.4
    # H = nonpolar at pH 7.4
    # s = special cases
    # - = gap in multiple sequence alignment
    translation = {letter:idx for idx, letter in enumerate([*"RHKDESTNQAVILMFYWCGP-"])}
    seqs = np.array([[*seq] for seq in seqs], dtype='U1')
    translate = lambda x: translation.get(x, 20)
    msa_translation = np.vectorize(translate)(seqs)
    return msa_translation

def Neff(seqs: array, eff_cutoff=0.8):
    '''
        Neff calculation over array using 1 - hamming distance
        squareform(pdist()) calculates a square all v all distance matrix
        example available in scipy documentation

        weights correspond to how redundant an aa sequence is in the alignment
        if there are a lot of similar sequences each of them will have a low weights
        similiarity is based on hamming distance

        return weights 
    '''
    seqs_h_dist = 1.0 - squareform(pdist(seqs, "hamming"))
    #binary array to create weights
    seqs_weights = np.where(seqs_h_dist >= eff_cutoff, 1.0, 0.0)
    seqs_weights = 1.0/np.sum(seqs_weights, axis=-1)
    return seqs_weights


def GREMLIN(seqs: array, iterations=100, gaps=True):

    #use translate_msa to change seqs into integer version
    seqs_int = Translate_Msa(seqs)
    seqs_weights = Neff(seqs_int)

    # Initialize parameters
    states = 21 #20 aa + gap
    #use a bitmap to encode the integer version of the msa
    msa_bitmap = jax.nn.one_hot(seqs_int, states)
    NO_GAP = False
    if not gaps:
        states = states - 1
        NO_GAP = 1.0 - msa_bitmap[...,-1] 
        msa_bitmap = msa_bitmap[...,:states]

    #prep one-body term
    one_body = jnp.tensordot(msa_bitmap, seqs_weights, axes=[0,0])
    pseudo_count = 0.01 * jnp.log(jnp.sum(seqs_weights))
    one_body = jnp.log(jnp.add(one_body , pseudo_count))
    one_body = jnp.add(one_body,-1*jnp.mean(one_body, axis=1, keepdims=True))
    one_body_init = one_body
    
    #prep two-body term
    two_body_init = jnp.zeros((one_body.shape[0], states, one_body.shape[0], states))

    @jax.jit
    def loss(body, one_body_dims = one_body_init.shape, two_body_dims = two_body_init.shape):
        _ = one_body_dims[0] * one_body_dims[1]
        one_body = body[0:_].reshape(one_body_dims)
        two_body = body[_:].reshape(two_body_dims)
        # Symmetrize W
        two_body = two_body + jnp.transpose(two_body, (2, 3, 0, 1))

        # Set diagonal to zero
        two_body = two_body * (1 - jnp.eye(one_body.shape[0]))[:, None, :, None]

        one_two = one_body + jnp.tensordot(msa_bitmap, two_body, axes=2)
        markov_blanket = jnp.sum(msa_bitmap * one_two, axis=-1)
        partition = logsumexp(one_two, axis=-1)
        PLL = markov_blanket - partition

        def if_true(PLL: array):
            return PLL 
        def if_false(PLL: array):
            return PLL * NO_GAP
        PLL = jax.lax.cond(gaps == True,if_true, if_false , PLL)
        PLL = jnp.sum(PLL, axis=-1)
        PLL_weighted = jnp.sum(seqs_weights * PLL) / jnp.sum(seqs_weights)

        # Regularization to prevent overfitting
        L2_V =  0.01 * jnp.sum(jnp.square(one_body))
        L2_W =  0.1 * jnp.sum(jnp.square(two_body)) * 0.5
        L2_W =  L2_W * (one_body.shape[0] - 1) * (states - 1)
        return -PLL_weighted + (L2_V + L2_W) / jnp.sum(seqs_weights)

    @jax.jit
    def opt_step(iterate, _):
        x, opt_state = iterate
        grad = jax.grad(loss)(x)
        updates, opt_state = solver.update(grad, opt_state, x)
        x = optax.apply_updates(x, updates)
        return (x, opt_state), loss(x)

    solver = optax.adadelta(learning_rate=1.)
    x_0 = jnp.concatenate([one_body_init.ravel(), two_body_init.ravel()])
    opt_state = solver.init(x_0)
    x_final, losses = jax.lax.scan(opt_step, (x_0, opt_state), None, length=iterations)
    print(losses)
    one_body_final = x_final[0].reshape(one_body_init.shape)
    two_body_final = x_final[1].reshape(two_body_init.shape)

    return one_body_final, two_body_final


def main(fasta, filter_msa, filter_msa_row, filter_msa_col):
    print(f"Running {fasta.split('.')[0]}...")
    preprocess_msa = PREPROCESS_MSA(*Read_MSA(fasta))

    print(filter_msa)
    if filter_msa:
        preprocess_msa.reduce_gaps()
        seqs = preprocess_msa.seqs
        names = preprocess_msa.names
        aa = preprocess_msa.aa
        aa_edit = preprocess_msa.aa_edit
    else:
        print(f"Running MSA without preprocessing")
        seqs = preprocess_msa.seqs
        names = preprocess_msa.names
        aa = preprocess_msa.aa

    one_body, two_body = GREMLIN(seqs)
    print(one_body)
   
if  __name__ == '__main__':
    main(*CL_input())

