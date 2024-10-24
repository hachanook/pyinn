import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from model import *
from dataset import *


def plot_regression(bool_plot, model, plot_in_axis, plot_out_axis, cls_data, color_map="viridis", vmin=0, vmax=1, marker_size=20):
    

    if len(plot_in_axis)==2 and len(plot_out_axis)==1 and bool_plot and cls_data.bool_normalize == False:
        # we will plot the error only when there is no normalization on the original data.
        plot_2D_1D(model, plot_in_axis, plot_out_axis, cls_data, color_map, vmin, vmax, marker_size)
    # else:
        # print("\n[Warning] plotting should not be conducted on normalized data\n")

def plot_2D_1D(model, plot_in_axis, plot_out_axis, cls_data, color_map="viridis", vmin=0, vmax=1, marker_size=20):
    """ This function plots 2D input and 1D output data. By default, this function should work on original data only
    plot_in_axis: [axis1, axis2]
    plot_out_axis: [axis1]
    """
    ## create mesh and data
    ### in normalized space, create prediction
    xmin, xmax = cls_data.x_data_minmax["min"][plot_in_axis[0]], cls_data.x_data_minmax["max"][plot_in_axis[0]]
    ymin, ymax = cls_data.x_data_minmax["min"][plot_in_axis[1]], cls_data.x_data_minmax["max"][plot_in_axis[1]]
    umin, umax = cls_data.u_data_minmax["min"][plot_out_axis[0]], cls_data.u_data_minmax["max"][plot_out_axis[0]]

    x_nds = jnp.linspace(xmin, xmax, 101, dtype=jnp.float64)
    y_nds = jnp.linspace(ymin, ymax, 101, dtype=jnp.float64)
    X,Y = jnp.meshgrid(x_nds, y_nds) # (101,101) each
    XY = jnp.dstack((X, Y)) # (101,101,2)
    U_pred = vv_forward(model.params, model.x_dms_nds, XY) # (101,101,1)
    U_exact = globals()["vv_fun_"+cls_data.data_name](XY) # (101,101,1)    

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    plt.subplots_adjust(wspace=0.4)  # Increase the width space between subplots

    surf1 = ax1.pcolormesh(X, Y, jnp.squeeze(U_pred), cmap=color_map, vmin=umin, vmax=umax)
    ax1.set_xlabel(f"x_{str(plot_in_axis[0]+1)}", fontsize=16)
    ax1.set_ylabel(f"x_{str(plot_in_axis[1]+1)}", fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    cbar1 = fig.colorbar(surf1, shrink=0.8, aspect=20, pad=0.02)
    cbar1.set_label(f'u', fontsize=14)
    cbar1.ax.tick_params(labelsize=12)
    ax1.set_title('INN prediction', fontsize=16)

    surf2 = ax2.pcolormesh(X, Y, jnp.squeeze(U_exact), cmap=color_map, vmin=umin, vmax=umax)
    ax2.set_xlabel(f"x_{str(plot_in_axis[0]+1)}", fontsize=16)
    ax2.set_ylabel(f"x_{str(plot_in_axis[1]+1)}", fontsize=16)
    ax2.tick_params(axis='both', labelsize=12)
    cbar2 = fig.colorbar(surf2, shrink=0.8, aspect=20, pad=0.02)
    cbar2.set_label(f'u', fontsize=14)
    cbar2.ax.tick_params(labelsize=12)
    ax2.set_title('Original function', fontsize=16)

    # plt.tight_layout()
    parent_dir = os.path.abspath(os.getcwd())
    path_figure = os.path.join(parent_dir, 'plots')
    fig.savefig(os.path.join(path_figure, cls_data.data_name) , dpi=300)
    plt.close()





        


    