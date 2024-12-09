import numpy as np
import tensorly as tl
from tensorly.decomposition import parafac, tucker
import matplotlib.pyplot as plt
import os
np.set_printoptions(precision=4, suppress=True)

tensor = tl.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
                        [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

# ### CP decomposition ###
# factors = parafac(tensor, rank=2)

# # # for (weight, factor) in factors:
# # #     print(weight)
# # #     print(factor)

# tensor_recovered = tl.cp_to_tensor(factors)
# print(tensor - tensor_recovered)


# ### Tucker decomposition ###
def tucker_to_tensor(core, factors):
    """
    Reconstructs the original tensor from the Tucker decomposition for n-way tensors.
    
    Args:
    core (ndarray): Core tensor.
    factors (list of ndarrays): List of factor matrices.
    
    Returns:
    ndarray: Reconstructed tensor.
    """
    # Start with the core tensor
    reconstructed = core
    
    # Iterate over each mode and apply mode-n product
    for factor in factors:
        # Use tensordot to perform the mode-n product
        reconstructed = np.tensordot(reconstructed, factor, axes=(0, 1))
    return reconstructed


# core, factors = tucker(tensor, rank=[2,3])
# # print(core)
# # for factor in factors:
# #     print(factor)

# tensor_recovered = tucker_to_tensor(core, factors)
# print(tensor - tensor_recovered)


## FDM benchmark
nx = 51  # Number of points in x
ny = 51  # Number of points in y
time_steps = 10000  # Number of time steps

parent_dir = os.getcwd()
save_dir = os.path.join(parent_dir, 'data')
file_name = f"FDM_{nx}_{ny}_{time_steps}.npy"
file_path = os.path.join(save_dir, file_name)
T_tensor = np.load(file_path)
n_org = T_tensor.size

# plt.figure()
# plt.imshow(T_tensor[:,:,-1], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
# plt.colorbar(label='Temperature')
# plt.title("Final Temperature Distribution")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.show()
# plt.close()


### CP decomposition ###
rank_CP = 25
CP = parafac(T_tensor, rank=rank_CP)
weights, factors_CP = CP[0], CP[1]
# compute number of components in CP
n_CP = 0
for factor in factors_CP:
    n_CP += factor.size
# print(n_CP)
T_recovered_CP = tl.cp_to_tensor(CP)
err_CP = np.linalg.norm(T_tensor - T_recovered_CP) / np.linalg.norm(T_tensor)
# print(err_CP)


### Tucker decomposition
rank_Tucker = [25,25,10]
core, factors_Tucker = tucker(T_tensor, rank=rank_Tucker)
# compute number of components in Tucker
n_Tucker = core.size
for factor in factors_Tucker:
    n_Tucker += factor.size
# print(n_Tucker)
T_recovered_Tucker = tucker_to_tensor(core, factors_Tucker)
err_Tucker = np.linalg.norm(T_tensor - T_recovered_Tucker) / np.linalg.norm(T_tensor)
# print(err_Tucker)


### plot
fig, axes = plt.subplots(2, 3, figsize=(12,8))

ax = axes[0,0]
c0 = ax.imshow(T_tensor[:,:,-1], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
cbar = fig.colorbar(c0, ax=ax)
ax.set_title(f"Original \n n:{n_org:.2e}")

ax = axes[0,1]
c1 = ax.imshow(T_recovered_CP[:,:,-1], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
cbar = fig.colorbar(c1, ax=ax)
ax.set_title(f"CP {rank_CP}\n n:{n_CP:.2e}, err:{err_CP:.2e}")

ax = axes[0,2]
c2 = ax.imshow(T_recovered_Tucker[:,:,-1], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
cbar = fig.colorbar(c2, ax=ax)
ax.set_title(f"Tucker {rank_Tucker}\n n:{n_Tucker:.2e}, err:{err_Tucker:.2e}")
# plt.show()

ax = axes[1,0]
c0 = ax.imshow(T_tensor[:,:,int(0.5*time_steps)], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
cbar = fig.colorbar(c0, ax=ax)
# ax.set_title(f"Original \n n:{n_org:.2e}")

ax = axes[1,1]
c1 = ax.imshow(T_recovered_CP[:,:,int(0.5*time_steps)], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
cbar = fig.colorbar(c1, ax=ax)
# ax.set_title(f"CP \n n:{n_CP:.2e}, err:{err_CP:.2e}")

ax = axes[1,2]
c2 = ax.imshow(T_recovered_Tucker[:,:,int(0.5*time_steps)], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
cbar = fig.colorbar(c2, ax=ax)
# ax.set_title(f"Tucker \n n:{n_Tucker:.2e}, err:{err_Tucker:.2e}")

# plt.show()
path_figure = os.path.join(parent_dir, 'plots')
file_name = f"FDM_{nx}_{ny}_{time_steps}.png"
fig.savefig(os.path.join(path_figure, file_name) , dpi=300)