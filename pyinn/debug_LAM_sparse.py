import sys
sys.path.append('/home/cpm1402/pyinn/GenTen/build/python')
import numpy as np
import pygenten

# 1. Create a random sparse tensor in COO format
shape = (30, 40, 50)
density = 0.05
num_nonzero = int(np.prod(shape) * density)
coords = np.vstack([np.random.randint(0, s, num_nonzero) for s in shape])
values = np.random.randn(num_nonzero)

# GenTen expects a list of tuples for coordinates
indices = list(zip(coords[0], coords[1], coords[2]))

# 2. Create a sparse tensor in GenTen
tensor = pygenten.Sptensor(shape, indices, values)

# 3. Perform CP-ALS decomposition (sparse, efficient)
rank = 5
cp_result = pygenten.cp_als(tensor, rank=rank, maxiters=50, tol=1e-6)

# 4. Access the factor matrices and lambda (weights)
factors = cp_result.factors  # List of numpy arrays, one per mode
lambdas = cp_result.lambdas  # 1D numpy array

print("CP Decomposition complete.")
print("Lambdas:", lambdas)
print("\n\n\n\n\n")
for mode, factor in enumerate(factors):
    print(f"Factor matrix for mode {mode}: shape {factor.shape}")

# 5. (Optional) Visualize the first factor matrix
# import matplotlib.pyplot as plt
# plt.imshow(factors[0], aspect='auto')
# plt.title('First factor matrix (mode-0)')
# plt.colorbar()
# plt.show() 