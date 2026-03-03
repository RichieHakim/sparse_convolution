import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sparse_convolution as sc
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import time

# --- CONFIGURATION ---
dim = 50
points = [(10, 10), (10, 40), (25, 25), (40, 10), (40, 40)]
kernel_size = 7

print(f"--- Visual Test Started ({dim}x{dim}) ---")

# 1. DATA PREPARATION
A = scipy.sparse.lil_matrix((dim, dim))
for r, c in points:
    A[r, c] = 1.0
A = A.tocsr()

k = np.zeros((kernel_size, kernel_size))
k[kernel_size//2, :] = 1.0
k[:, kernel_size//2] = 1.0
k = k / k.sum()

# 2. INITIALIZATION PHASE (Instant initialization code)
start_init = time.time()
conv = sc.Toeplitz_convolution2d(x_shape=A.shape, k=k, mode='same')
end_init = time.time()

# 3. COMPUTATION PHASE (Core computation)
start_calc = time.time()
C = conv(x=A, batching=False)
end_calc = time.time()

# 4. VISUALIZATION PHASE (Save image file)
start_plot = time.time()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(A.toarray(), cmap='hot')
ax1.set_title("Original")
ax2.imshow(C.toarray(), cmap='hot')
ax2.set_title("Convolution Output")
plt.savefig("demo_result.png")
end_plot = time.time()

# --- FINAL REPORT ---
print(f"\n⏱️  TIME RESULTS:")
print(f"   - Initialization: {end_init - start_init:.6f} s")
print(f"   - Math computation: {end_calc - start_calc:.6f} s")
print(f"   - Plot creation: {end_plot - start_plot:.6f} s")
print(f"\n✅ Total time: {time.time() - start_init:.4f} s")
print("file saved in demo_result.png")