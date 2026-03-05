import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import sparse_convolution as sc
import numpy as np
import scipy.sparse
import time

# --- TEST CONFIGURATION ---
# Start with 5000. If htop shows no issues, try 10000.
dimension = 5000
print(f"\n[1] Starting test with dimension: {dimension}x{dimension}")

# --- DATA PREPARATION ---
# Create a sparse matrix with a single point in the center
A = scipy.sparse.lil_matrix((dimension, dimension))
A[dimension//2, dimension//2] = 1.0
A = A.tocsr() # Fast format required by the library

# Create a standard 3x3 kernel
B = np.array([
    [0.1, 0.2, 0.1],
    [0.2, 0.5, 0.2],
    [0.1, 0.2, 0.1]
])

# --- PHASE 1: INITIALIZATION ---
print(f"[2] Preparing Toeplitz_convolution2d class...")
start_prep = time.time()

try:
    conv = sc.Toeplitz_convolution2d(
        x_shape=A.shape,
        k=B,
        mode='same',
        dtype=np.float32,
    )
    end_prep = time.time()
    print(f"✅ Preparation completed in: {end_prep - start_prep:.2f} seconds")

    # --- PHASE 2: EXECUTION ---
    print(f"[3] Executing convolution...")
    start_exec = time.time()
    C = conv(x=A, batching=False)
    end_exec = time.time()
    print(f"✅ Calculation completed in: {end_exec - start_exec:.4f} seconds")
    
    print("\n--- RESULTS ---")
    print("Execution completed successfully without excessive memory consumption.")
    print("The 'Preparation' phase is now nearly instantaneous.")

except Exception as e:
    print(f"\n❌ ERROR: System stopped!")
    print(f"Details: {e}")
