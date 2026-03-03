from typing import Tuple, Optional, Union

import scipy.sparse
import numpy as np

class Toeplitz_convolution2d():
    """
    Convolve a 2D array with a 2D kernel using the Toeplitz matrix
    multiplication method. This class is ideal when 'x' is very sparse
    (density<0.01), 'x' is small (shape <(1000,1000)), 'k' is small (shape
    <(100,100)), and the batch size is large (e.g. 1000+). Generally, it is
    faster than scipy.signal.convolve2d when convolving multiple arrays with the
    same kernel. It maintains a low memory footprint by storing the toeplitz
    matrix as a sparse matrix.
    RH 2022

    Attributes:
        x_shape (Tuple[int, int]):
            The shape of the 2D array to be convolved.
        k (np.ndarray):
            2D kernel to convolve with.
        mode (str):
            Either ``'full'``, ``'same'``, or ``'valid'``. See
            scipy.signal.convolve2d for details.
        dtype (Optional[np.dtype]):
            The data type to use for the Toeplitz matrix.
            If ``None``, then the data type of the kernel is used.

    Args:
        x_shape (Tuple[int, int]):
            The shape of the 2D array to be convolved.
        k (np.ndarray):
            2D kernel to convolve with.
        mode (str):
            Convolution method to use, either ``'full'``, ``'same'``, or
            ``'valid'``.
            See scipy.signal.convolve2d for details. (Default is 'same')
        dtype (Optional[np.dtype]):
            The data type to use for the Toeplitz matrix. Ideally, this matches
            the data type of the input array. If ``None``, then the data type of
            the kernel is used. (Default is ``None``)

    Example:
        .. highlight:: python
        .. code-block:: python

            # create Toeplitz_convolution2d object
            toeplitz_convolution2d = Toeplitz_convolution2d(
                x_shape=(100,30),
                k=np.random.rand(10,10),
                mode='same',
            )
            toeplitz_convolution2d(
                x=scipy.sparse.csr_matrix(np.random.rand(5,3000)),
                batch_size=True,
            )
    """
    def __init__(
        self,
        x_shape: Tuple[int, int],
        k: np.ndarray,
        mode: str = 'same',
        dtype: Optional[np.dtype] = None,
        verbose: Union[bool, int] = False,
    ):
        """
        Initializes the Toeplitz_convolution2d object.
        """
        ## Type checking
        assert isinstance(x_shape, (tuple, list)), f"x_shape must be a tuple. Found: {type(x_shape)}"
        assert all([isinstance(s, (int, float, np.integer, np.floating)) for s in x_shape]), f"x_shape must be a tuple of integers. Found: {[type(s) for s in x_shape]}"
        x_shape = (int(x_shape[0]), int(x_shape[1]))

        assert isinstance(k, np.ndarray), "k must be a numpy array"
        assert k.ndim == 2, "k must be a 2D array"

        assert isinstance(mode, str), "mode must be a string"
        assert mode in ['full', 'same', 'valid'], "mode must be 'full', 'same', or 'valid'"

        self.k = k.copy()
        self.mode = mode
        self.x_shape = x_shape
        self.dtype = k.dtype if dtype is None else dtype

        if mode == 'valid':
            assert x_shape[0] >= k.shape[0] and x_shape[1] >= k.shape[1], "x must be larger than k in both dimensions for mode='valid'"
    
    def __call__(
        self,
        x: Union[np.ndarray, scipy.sparse.csc_matrix, scipy.sparse.csr_matrix],
        batching: bool = True,
        mode: Optional[str] = None,
    ) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
        """
        Convolve the input array with the kernel using an optimized broadcasting approach.
        Instead of pre-building complete matrices, it extracts non-zero coordinates 
        (COO format) from both input and kernel and broadcasts their indices and values 
        to instantaneously calculate valid convolution points, offering huge memory 
        savings and fast execution time.

        Args:
            x (Union[np.ndarray, scipy.sparse.csc_matrix,
            scipy.sparse.csr_matrix]): 
                Input array(s) (i.e. image(s)) to convolve with the kernel. \n
                * If ``batching==False``: Single 2D array to convolve with the
                  kernel. Shape: *(self.x_shape[0], self.x_shape[1])*
                * If ``batching==True``: Multiple 2D arrays that have been
                  flattened into row vectors (with order='C'). \n
                Shape: *(n_arrays, self.x_shape[0]*self.x_shape[1])*

            batching (bool): 
                * ``False``: x is a single 2D array.
                * ``True``: x is a 2D array where each row is a flattened 2D
                  array. \n
                (Default is ``True``)

            mode (Optional[str]): 
                Defines the mode of the convolution. Options are 'full', 'same'
                or 'valid'. See `scipy.signal.convolve2d` for details. Overrides
                the mode set in __init__. (Default is ``None``)

        Returns:
            (Union[np.ndarray, scipy.sparse.csr_matrix]):
                out (Union[np.ndarray, scipy.sparse.csr_matrix]): 
                    * ``batching==True``: Multiple convolved 2D arrays that have
                      been flattened into row vectors (with order='C'). Shape:
                      *(n_arrays, height*width)*
                    * ``batching==False``: Single convolved 2D array of shape
                      *(height, width)*
        """
        if mode is None:
            mode = self.mode  ## use the mode that was set in the init if not specified
        issparse = scipy.sparse.issparse(x)

        if mode == 'full':
            H_out = self.x_shape[0] + self.k.shape[0] - 1
            W_out = self.x_shape[1] + self.k.shape[1] - 1
            t = l = 0
        elif mode == 'same':
            H_out = self.x_shape[0]
            W_out = self.x_shape[1]
            t = (self.k.shape[0] - 1) // 2
            l = (self.k.shape[1] - 1) // 2
        elif mode == 'valid':
            H_out = self.x_shape[0] - self.k.shape[0] + 1
            W_out = self.x_shape[1] - self.k.shape[1] + 1
            t = self.k.shape[0] - 1
            l = self.k.shape[1] - 1
            if H_out <= 0 or W_out <= 0:
                raise ValueError("x must be larger than k in both dimensions for mode='valid'")
        else:
            raise ValueError("mode must be 'full', 'same', or 'valid'")

        k_coo = scipy.sparse.coo_matrix(self.k)
        k_r = k_coo.row
        k_c = k_coo.col
        k_d = k_coo.data

        x_coo = scipy.sparse.coo_matrix(x)
        if batching:
            B = x.shape[0]
            batch_idx = x_coo.row
            x_idx = x_coo.col
            x_r = x_idx // self.x_shape[1]
            x_c = x_idx % self.x_shape[1]
        else:
            batch_idx = np.zeros_like(x_coo.row)
            x_r = x_coo.row
            x_c = x_coo.col
        
        # Broadcasting logic
        new_r = x_r[:, None] + k_r[None, :] - t
        new_c = x_c[:, None] + k_c[None, :] - l
        new_b = np.repeat(batch_idx[:, None], len(k_d), axis=1)
        new_d = x_coo.data[:, None] * k_d[None, :]

        # Ravel for valid mask and csr sparse input
        new_r = new_r.ravel()
        new_c = new_c.ravel()
        new_b = new_b.ravel()
        new_d = new_d.ravel()

        # Filter out-of-bounds bounds
        valid_mask = (new_r >= 0) & (new_r < H_out) & (new_c >= 0) & (new_c < W_out)
        
        new_r = new_r[valid_mask]
        new_c = new_c[valid_mask]
        new_b = new_b[valid_mask]
        new_d = new_d[valid_mask]

        if batching:
            out_idx = new_r * W_out + new_c
            out = scipy.sparse.csr_matrix(
                (new_d, (new_b, out_idx)),
                shape=(B, H_out * W_out)
            )
            # Ensure return type matches dense arrays if input is a dense array
            if not issparse:
                out = out.toarray()
        else:
            out = scipy.sparse.csr_matrix(
                (new_d, (new_r, new_c)),
                shape=(H_out, W_out)
            )
            # Ensure return type matches dense arrays if input is a dense array
            if not issparse:
                out = out.toarray()

        return out