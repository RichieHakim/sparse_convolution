import numpy as np
import scipy.sparse
import scipy.signal

x = np.zeros((5, 5))
x[2, 2] = 1
x[1, 1] = 2

k = np.array([[1, 2], [3, 4]])

# Scipy convolve2d
out_scipy = scipy.signal.convolve2d(x, k, mode='full')

# Broadcast method
x_coo = scipy.sparse.coo_matrix(x)
k_coo = scipy.sparse.coo_matrix(k)

new_r = x_coo.row[:, None] + k_coo.row[None, :]
new_c = x_coo.col[:, None] + k_coo.col[None, :]
new_d = x_coo.data[:, None] * k_coo.data[None, :]

out_sparse = scipy.sparse.csr_matrix((new_d.ravel(), (new_r.ravel(), new_c.ravel())), shape=(x.shape[0]+k.shape[0]-1, x.shape[1]+k.shape[1]-1))

print("Is allclose?", np.allclose(out_scipy, out_sparse.toarray()))
print("Scipy:")
print(out_scipy)
print("Sparse:")
print(out_sparse.toarray())
