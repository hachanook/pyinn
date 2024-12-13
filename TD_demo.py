import numpy as np
import jax.numpy as jnp
import tensorly as tl
import tensortools as tt
from tensorly.decomposition import parafac, tucker
import matplotlib.pyplot as plt
import os
np.set_printoptions(precision=4, suppress=True)

# tensor = tl.tensor([[ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.],
#                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
#                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
#                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
#                         [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
#                         [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
#                         [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
#                         [ 0.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  0.],
#                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
#                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
#                         [ 0.,  0.,  0.,  0.,  1.,  1.,  1.,  1.,  0.,  0.,  0.,  0.],
#                         [ 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])

# ### CP decomposition ###
# factors = parafac(tensor, rank=2)
# print(type(factors))



# for (weight, factor) in factors:
#     print(weight)
#     print(factor)

# # tensor_recovered = tl.cp_to_tensor(factors)
# # print(tensor - tensor_recovered)


def cp_to_tensor(factors):
    """ This is an inhouse code that assembles factors (list) to a tensor
    factors: list of factor matrices of shape (nmode, nnode) each
    """

    for imode in range(factors[0].shape[0]):
        tensor = factors[0][imode,:]
        for factor in factors[1:]:
            tensor = np.tensordot(tensor, factor[imode,:], axes=0)
    
        if imode == 0:
            Tensor = tensor
        else:
            Tensor += tensor
    return Tensor
        



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


################## FDM benchmark ########################

nx = 101  # Number of points in x
ny = 101  # Number of points in y
time_steps = 2000  # Number of time steps
alpha = 0.1

parent_dir = os.getcwd()
save_dir = os.path.join(parent_dir, 'data')
file_name = f"FDM_{nx}_{ny}_{time_steps}_alpha{alpha}.npy"
file_path = os.path.join(save_dir, file_name)
T_tensor = np.load(file_path)
n_org = T_tensor.size

## CP decomposition ###

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
# # print(err_CP)


### Tucker decomposition 
rank_Tucker = [30,30,300]
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
fig, axes = plt.subplots(2, 3, figsize=(15,10))

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
plt.tight_layout()
path_figure = os.path.join(parent_dir, 'plots')
file_name = f"FDM_{nx}_{ny}_{time_steps}_alpha{alpha}.png"
fig.savefig(os.path.join(path_figure, file_name) , dpi=300)



################# Jiachen TD benchmark #########################

# ## Read the data file
# nelem_x, nelem_y, nelem_z, nelem_t = 500, 50, 3, 100
# num_mode, num_layer, r, num_max_nelem_t = 800, 1, 0.1, 100
# Layer= 0
# file_name = f"linear_bc_{nelem_x}_{nelem_y}_{nelem_z}_{nelem_t}_m{num_mode}_L{num_layer}_r{r}_max_elemt{num_max_nelem_t}.npz"                


# parent_dir = os.getcwd()
# save_dir = os.path.join(parent_dir, 'data')
# file_path = os.path.join(save_dir, file_name)
# loaded_dict = np.load(file_path)
# loaded_sol = {key: loaded_dict[key] for key in loaded_dict}
# U_x, U_y, U_z, U_t = loaded_dict['L'+ str(Layer) + '_U_x'], loaded_dict['L'+ str(Layer) + '_U_y'], loaded_dict['L'+ str(Layer) + '_U_z'], loaded_dict['L'+ str(Layer) + '_U_t']
# # shape: (num_mode, nnode)
# factors_CP = [U_x, U_y, U_z, U_t]

# ## Recover original tensor
# T_tensor = cp_to_tensor(factors_CP)
# n_org = T_tensor.size
# n_CP = U_x.size + U_y.size + U_z.size + U_t.size

# ### CP decomposition ###
# # rank_CP = 4
# # CP = parafac(T_tensor, rank=rank_CP)
# # weights, factors_CP = CP[0], CP[1]
# # # compute number of components in CP
# # n_CP = 0
# # for factor in factors_CP:
# #     n_CP += factor.size
# # # print(n_CP)
# # T_recovered_CP = tl.cp_to_tensor(CP)

# # err_CP = np.linalg.norm(T_tensor - T_recovered_CP) / np.linalg.norm(T_tensor)
# # # print(err_CP)


# ### Tucker decomposition
# rank_Tucker = [40,4,4,50]
# core, factors_Tucker = tucker(T_tensor, rank=rank_Tucker)
# # compute number of components in Tucker
# n_Tucker = core.size
# for factor in factors_Tucker:
#     n_Tucker += factor.size
# # print(n_Tucker)
# T_recovered_Tucker = tucker_to_tensor(core, factors_Tucker)
# err_Tucker = np.linalg.norm(T_tensor - T_recovered_Tucker) / np.linalg.norm(T_tensor)
# # print(err_Tucker)



# ### plot
# fig, axes = plt.subplots(2, 2, figsize=(10,10))

# ax = axes[0,0]
# c0 = ax.imshow(T_tensor[:,:,0,-1], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
# cbar = fig.colorbar(c0, ax=ax)
# ax.set_title(f"Original \n n:{n_org:.2e} / n_CP: {n_CP:.2e}")

# # ax = axes[0,1]
# # c1 = ax.imshow(T_recovered_CP[:,:,0,-1], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
# # cbar = fig.colorbar(c1, ax=ax)
# # ax.set_title(f"CP {rank_CP}\n n:{n_CP:.2e}, err:{err_CP:.2e}")

# ax = axes[0,1]
# c2 = ax.imshow(T_recovered_Tucker[:,:,0,-1], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
# cbar = fig.colorbar(c2, ax=ax)
# ax.set_title(f"Tucker {rank_Tucker}\n n:{n_Tucker:.2e}, err:{err_Tucker:.2e}")

# ax = axes[1,0]
# c0 = ax.imshow(T_tensor[:,:,0,int(nelem_t/2)], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
# cbar = fig.colorbar(c0, ax=ax)
# ax.set_title(f"Original \n n:{n_org:.2e}")

# # ax = axes[1,1]
# # c1 = ax.imshow(T_recovered_CP[:,:,0,int(nelem_t/2)], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
# # cbar = fig.colorbar(c1, ax=ax)
# # ax.set_title(f"CP {rank_CP}\n n:{n_CP:.2e}, err:{err_CP:.2e}")

# ax = axes[1,1]
# c2 = ax.imshow(T_recovered_Tucker[:,:,0,int(nelem_t/2)], extent=[0, 1, 0, 1], origin='lower', cmap='hot', aspect='auto')
# cbar = fig.colorbar(c2, ax=ax)
# ax.set_title(f"Tucker {rank_Tucker}\n n:{n_Tucker:.2e}, err:{err_Tucker:.2e}")

# # plt.show()
# plt.tight_layout()
# path_figure = os.path.join(parent_dir, 'plots')
# file_name = f"linear_bc_{nelem_x}_{nelem_y}_{nelem_z}_{nelem_t}_m{num_mode}_L{num_layer}_r{r}_max_elemt{num_max_nelem_t}.png"    
# fig.savefig(os.path.join(path_figure, file_name) , dpi=300)