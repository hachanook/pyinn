import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from model import *
from dataset import *


def plot_regression(plot_bool, model, plot_in_axis, plot_out_axis, data_name, color_map="viridis", vmin=0, vmax=1, marker_size=20):
    

    if len(plot_in_axis)==2 and len(plot_out_axis)==1 and plot_bool:
        plot_2D_1D(model, plot_in_axis, plot_out_axis, data_name, color_map="viridis", vmin=0, vmax=1, marker_size=20)

def plot_2D_1D(model, plot_in_axis, plot_out_axis, data_name, color_map="viridis", vmin=0, vmax=1, marker_size=20):
    """ This function plots 2D input and 1D output data
    """
    
    ## create mesh and data
    x_nds_fig = jnp.linspace(0, 1, 101, dtype=jnp.float64) 
    y_nds_fig = jnp.linspace(0, 1, 101, dtype=jnp.float64) 
    X,Y = jnp.meshgrid(x_nds_fig, y_nds_fig)
    XY = jnp.dstack((X, Y)) # (101,101,2)
    U_pred = vv_forward(model.params, model.x_dms_nds, XY) # (101,101,1)
    U_exact = globals()["vv_fun_"+data_name](XY) # (101,101,1)
    
    fig = plt.figure(figsize=(20,10))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    surf1 = ax1.pcolormesh(X, Y, jnp.squeeze(U_pred), cmap=color_map)
    ax1.set_xlabel(f"x_{str(plot_in_axis[0]+1)}", fontsize=16)
    ax1.set_ylabel(f"x_{str(plot_in_axis[1]+1)}", fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    cbar1 = fig.colorbar(surf1, shrink=0.8, aspect=20, pad=0.02)
    # cbar1.set_label(f'u(x,y,z={z_value})', fontsize=14)
    cbar1.ax.tick_params(labelsize=12)

    surf2 = ax2.pcolormesh(X, Y, jnp.squeeze(U_exact), cmap=color_map)
    ax2.set_xlabel(f"x_{str(plot_in_axis[0]+1)}", fontsize=16)
    ax2.set_ylabel(f"x_{str(plot_in_axis[1]+1)}", fontsize=16)
    ax2.tick_params(axis='both', labelsize=12)
    cbar2 = fig.colorbar(surf2, shrink=0.8, aspect=20, pad=0.02)
    # cbar1.set_label(f'u(x,y,z={z_value})', fontsize=14)
    cbar2.ax.tick_params(labelsize=12)

    # plt.tight_layout()
    parent_dir = os.path.abspath(os.getcwd())
    path_figure = os.path.join(parent_dir, 'plots')
    fig.savefig(os.path.join(path_figure, data_name) , dpi=300)
    plt.close()





        


    