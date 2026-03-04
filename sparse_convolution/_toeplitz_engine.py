import numpy as np
import scipy.sparse

def build_toeplitz_matrix(x_shape, k, dtype):
    """
    Builds the double block Toeplitz matrix for convolution.
    """
    so = ( (k.shape[0] + x_shape[0] - 1), (k.shape[1] + x_shape[1] - 1) )
    
    t = [scipy.sparse.diags(
        diagonals=np.ones((k.shape[1], x_shape[1]), dtype=dtype) * k_i[::-1][:,None], 
        offsets=np.arange(-k.shape[1]+1, 1), 
        shape=(so[1], x_shape[1]),
        dtype=dtype,
    ) for k_i in k]
    
    tc = scipy.sparse.vstack(t + [scipy.sparse.dia_matrix((t[0].shape), dtype=dtype)]*(x_shape[0]-1))
    
    dt = scipy.sparse.hstack([_roll_sparse(
        x=tc, 
        shift=(ii>0)*ii*(so[1])
    ) for ii in range(x_shape[0])]).tocsr()
    
    return dt, so

def _roll_sparse(x, shift):
    """
    Roll columns of a sparse matrix.
    """
    out = x.copy()
    out.row += shift
    return out

def compute_toeplitz(x, dt, so, k_shape, x_shape, mode, batching, issparse):
    """
    Computes convolution using the pre-calculated Toeplitz matrix.
    """
    if batching:
        x_v = x.T
    else:
        x_v = x.reshape(-1, 1)
    
    if issparse:
        x_v = x_v.tocsc()
    
    out_v = dt @ x_v
        
    if mode == 'full':
        p_t = 0
        p_b = so[0]+1
        p_l = 0
        p_r = so[1]+1
    elif mode == 'same':
        p_t = (k_shape[0]-1)//2
        p_b = -(k_shape[0]-1)//2
        p_l = (k_shape[1]-1)//2
        p_r = -(k_shape[1]-1)//2

        p_b = x_shape[0]+1 if p_b==0 else p_b
        p_r = x_shape[1]+1 if p_r==0 else p_r
    elif mode == 'valid':
        p_t = (k_shape[0]-1)
        p_b = -(k_shape[0]-1)
        p_l = (k_shape[1]-1)
        p_r = -(k_shape[1]-1)

        p_b = x_shape[0]+1 if p_b==0 else p_b
        p_r = x_shape[1]+1 if p_r==0 else p_r
    
    if batching:
        idx_crop = np.zeros((so), dtype=np.bool_)
        idx_crop[p_t:p_b, p_l:p_r] = True
        idx_crop = idx_crop.reshape(-1)
        out = out_v[idx_crop,:].T
    else:
        if issparse:
            out = out_v.reshape((so)).tocsc()[p_t:p_b, p_l:p_r]
        else:
            out = out_v.reshape((so))[p_t:p_b, p_l:p_r]

    return out
