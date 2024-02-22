import numpy as np
from numba import jit, prange

def shuffle_column(x, column, random_state=0):
    np.random.seed(random_state)
    # Create a copy of the input array to avoid modifying the original data
    shuffled = x.copy()
    # Shuffle the specified column
    np.random.shuffle(shuffled[:, column])
    return shuffled

def batcher(temp_X, important_genes_idxs, random_state=0):
    n_samples, n_features = temp_X.shape
    n_important_genes = len(important_genes_idxs)
    batched = np.empty((n_samples * (n_important_genes + 1), n_features), dtype=temp_X.dtype)
    batched[:n_samples] = temp_X
    for idx, gene_idx in enumerate(important_genes_idxs, start=1):
        shuffled_data = shuffle_column(temp_X, gene_idx, random_state=random_state)
        batch_start = idx * n_samples
        batch_end = (idx + 1) * n_samples
        batched[batch_start:batch_end] = shuffled_data
    return batched

@jit(nopython=True, parallel=True)
def batcher_numba(temp_X, important_genes_idxs, random_state=0):
    n_samples, n_features = temp_X.shape
    n_important_genes = len(important_genes_idxs)
    batched = np.empty((n_samples * (n_important_genes + 1), n_features), dtype=temp_X.dtype)
    batched[:n_samples] = temp_X  # Copy original data to the first batch

    # Loop through each important gene index for shuffling
    for idx in prange(n_important_genes):  # prange for parallel execution
        # Note: Using gene_idx directly from important_genes_idxs, corrected loop range
        gene_idx = important_genes_idxs[idx]
        shuffled_data = shuffle_column_numba(temp_X, gene_idx, random_state)
        batch_start = (idx + 1) * n_samples  # Corrected calculation to match non-Numba version
        batch_end = (idx + 2) * n_samples
        batched[batch_start:batch_end] = shuffled_data
    return batched

@jit(nopython=True)
def shuffle_column_numba(x, column, random_state=0):
    np.random.seed(random_state)
    # Create a copy of the input array to avoid modifying the original data
    shuffled = x.copy()
    # Shuffle the specified column
    np.random.shuffle(shuffled[:, column])
    return shuffled


## -- Old version of functions, kept for reproducibility --

def shuffle_columns_old(x,random_state=0):
    
    #set the random state
    np.random.seed(random_state)
    
    shuffled = np.zeros_like(x)

    #draw a matrix of random indices of the same shape as x
    random_indices = np.random.randint(0,x.shape[0],size=x.shape)
    
    #shuffle each column of x independently
    for i in range(x.shape[1]):
        shuffled[:,i] = x[random_indices[:,i],i]
    
    return shuffled

def shuffle_compose(oringinal,shuffled,idx):
    
    if idx == 0:
        return np.concatenate((shuffled[:,idx].reshape(-1,1),oringinal[:,idx+1:]),axis=1)
    elif idx == shuffled.shape[1]-1:
        return np.concatenate((oringinal[:,:idx],shuffled[:,idx].reshape(-1,1)),axis=1)
    else:
        return np.concatenate((oringinal[:,:idx],shuffled[:,idx].reshape(-1,1),oringinal[:,idx+1:]),axis=1)


