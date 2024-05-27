#!/usr/bin/env python3

import sys

from scipy.spatial.distance import pdist, squareform
import numpy as np
import jax
import jax.numpy as jnp
from jax.numpy import array

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
    msa_one_hot = jax.nn.one_hot(seqs_int, states)
    if not gaps:
        states = states - 1
        NO_GAP = 1.0 - msa_one_hot[...,-1] 
        msa_one_hot = msa_one_hot[...,:states]

    V = jnp.tensordot(msa_one_hot, seqs_weights, axes=[0,0])
    pseudo_count = 0.01 * jnp.log(jnp.sum(seqs_weights))
    V = jnp.log(jnp.add(V , pseudo_count))
    V = jnp.add(V,-1*jnp.mean(V, axis=1, keepdims=True))
    V_init = V

    V_dims = V_init.shape; V_idx = jnp.prod(jnp.array(V_dims))
    W_init = jnp.zeros((ncol, states, ncol, states))
    W_dims = W_init.shape; W_idx = jnp.prod(jnp.array(W_dims))
    print(V_idx, W_idx)
    print(V_dims, W_dims)
    print(V_init.shape)
    print(W_init.shape)
    print(msa_one_hot.shape)
    print(weights.shape)
    print(sum(weights))
    print(msa['msa'])
    # Optimizer setup
    # optimizer = optax.adam(learning_rate=opt_rate)
    # opt_state = optimizer.init((V_init, W_init))
    # 
    # @jax.jit
    # def loss_fn(params, msa_one_hot, weights):
    #     V, W = params
    #     VW = V + jnp.tensordot(msa_one_hot, W, axes=2)
    #     H = jnp.sum(msa_one_hot * VW, axis=-1)
    #     Z = logsumexp(VW, axis=-1)
    #     PLL = H - Z
    #     PLL = jnp.sum(PLL, axis=-1)
    #     PLL_weighted = jnp.sum(weights * PLL) / jnp.sum(weights)
    #
    #     # Regularization
    #     L2_V = lam_v * jnp.sum(jnp.square(V))
    #     L2_W = lam_w * jnp.sum(jnp.square(W)) * 0.5
    #     if scale_lam_w:
    #         L2_W *= (ncol - 1) * (states - 1)
    #
    #     return -PLL_weighted + (L2_V + L2_W) / msa["neff"]
    # @jax.jit
    # def step(params, msa_one_hot, weights, opt_state):
    #     grads = jax.grad(loss_fn)(params, msa_one_hot, weights)
    #     updates, opt_state = optimizer.update(grads, opt_state)
    #     params = optax.apply_updates(params, updates)
    #     return params, opt_state

    # Initial parameters
    # params = (V_init, W_init)
    # @jax.jit
    def loss_fn(x, V_dims=V_dims, V_idx=V_idx, W_dims=W_dims, W_idx=W_idx):
        V = x[0:V_idx].reshape(V_dims)
        W_tmp = x[V_idx:].reshape(W_dims)

        # Symmetrize W
        W = W_tmp + jnp.transpose(W_tmp, (2, 3, 0, 1))

        # Set diagonal to zero
        W = W * (1 - jnp.eye(ncol))[:, None, :, None]

        VW = V + jnp.tensordot(msa_one_hot, W, axes=2)
        H = jnp.sum(msa_one_hot * VW, axis=-1)
        Z = logsumexp(VW, axis=-1)
        PLL = H - Z

        PLL = PLL * NO_GAP
        PLL = jnp.sum(PLL, axis=-1)
        PLL_weighted = jnp.sum(weights * PLL) / jnp.sum(weights)

        # Regularization
        L2_V =  0.01 * jnp.sum(jnp.square(V))
        L2_W =  0.1 * jnp.sum(jnp.square(W)) * 0.5
        L2_W =  L2_W * (ncol - 1) * (states - 1)
        return -PLL_weighted + (L2_V + L2_W) / jnp.sum(weights)
    # t_i = time.time()
    # res = minimize(loss_fn, W_init.ravel(), args=(V_init, W_init.shape),  method='BFGS', tol=1e-3)
    # t_f = time.time()
    #
    # print(res.x, res.fun, res.success, res.nfev, res.njev)
    # print('JAX calculation took {:0.6f} secs.'.format(t_f-t_i))

    solver = optax.adadelta(learning_rate=1.)
    # params = W_init
    x0 = jnp.concatenate([V_init.ravel(), W_init.ravel()])

    opt_state = solver.init(x0)
    print(loss_fn(x0) * jnp.sum(weights))
    for _ in range(100):
        grad = jax.grad(loss_fn)(x0)
        updates, opt_state = solver.update(grad, opt_state, x0)
        x0 = optax.apply_updates(x0, updates)
        print('Objective function: {:.2E}'.format(loss_fn(x0)* jnp.sum(weights)))
    # W_final = x0
    V_final = x0[0:V_idx].reshape(V_dims)
    # V_final = V_init
    W_final = x0[V_idx:].reshape(W_dims)

    # Training loop
    # temp = []
    # for i in range(opt_iter):
    #     params, opt_state = step(params, msa_one_hot, weights, opt_state)
    #     temp.append(loss_fn(params, msa_one_hot, weights))
    #     if (i + 1) % 10 == 0:
    #         print(f"Iteration {i+1} loss = {loss_fn(params, msa_one_hot, weights)}")
    #
    # V_final, W_final = params
    return {"v": V_final, "w": W_final, "v_idx": msa["v_idx"], "len": msa["ncol_ori"]}
    return np.array(temp)



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

    GREMLIN(seqs)
   
if  __name__ == '__main__':
    main(*CL_input())

