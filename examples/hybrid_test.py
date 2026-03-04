import numpy as np
import scipy.sparse
from sparse_convolution import Toeplitz_convolution2d

# Configurazione test
shape = (100, 100)
kernel = np.ones((3, 3))
# Creiamo una matrice molto sparsa (1% di densità)
x_dense = np.zeros(shape)
x_dense[10, 10] = 1.0
x_dense[50, 50] = 5.0
x_sparse = scipy.sparse.csr_matrix(x_dense)

print("--- 🧪 INIZIO TEST IBRIDO ---")

# 1. Test Modalità LAZY (il tuo broadcasting)
conv_lazy = Toeplitz_convolution2d(shape, kernel, precompute=False, dtype=np.float32)
out_lazy = conv_lazy(x_sparse, batching=False)

# 2. Test Modalità PRECOMPUTE (il vecchio metodo)
conv_pre = Toeplitz_convolution2d(shape, kernel, precompute=True, dtype=np.float32)
out_pre = conv_pre(x_sparse, batching=False)

# VERIFICA ACCURATEZZA
diff = np.abs((out_lazy - out_pre).data).sum() if scipy.sparse.issparse(out_lazy) else np.abs(out_lazy - out_pre).sum()
print(f"✅ Coerenza risultati: {'OK' if diff < 1e-5 else 'ERRORE'}")

# VERIFICA FIX P2 (Dtype)
print(f"✅ Dtype Lazy: {out_lazy.dtype} | Dtype Pre: {out_pre.dtype} (Atteso: float32)")

# VERIFICA FIX P1 (Errore dimensioni)
print("--- 🛡️ TEST VALIDAZIONE (P1) ---")
try:
    x_wrong = scipy.sparse.csr_matrix((1, 50)) # Dimensione sbagliata (atteso 10000)
    conv_lazy(x_wrong, batching=True)
except ValueError as e:
    print(f"✅ Validazione P1 funzionante: {e}")

print("--- 🚀 TEST COMPLETATI ---")