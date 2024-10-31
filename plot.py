import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from model import *
from dataset_regression import *
from dataset_classification import *


def plot_regression(model, cls_data, config):
    bool_plot = config['PLOT']['bool_plot']
    plot_in_axis = config['PLOT']['plot_in_axis']
    plot_out_axis = config['PLOT']['plot_out_axis']

    if bool_plot:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        if len(plot_in_axis)==2 and  cls_data.bool_normalize == False:
        
            # we will plot the error only when there is no normalization on the original data.
            plot_2D_1D(model, cls_data, plot_in_axis, plot_out_axis)
    else:
        print("\nPlotting deactivated\n")
        import sys
        sys.exit()

def plot_2D_1D(model, cls_data, plot_in_axis, plot_out_axis, color_map="viridis", vmin=0, vmax=1, marker_size=20):
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
    if model.interp_method == "linear" or model.interp_method == "nonlinear":
        U_pred = model.vv_forward(model.params, model.x_dms_nds, XY) # (101,101,L)
    elif model.interp_method == "MLP":
        U_pred = model.vv_forward(model.params, model.activation, XY) # (101,101,L)
        

    
    U_exact = globals()["vv_fun_"+cls_data.data_name](XY) # (101,101,L)    

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    plt.subplots_adjust(wspace=0.4)  # Increase the width space between subplots

    surf1 = ax1.pcolormesh(X, Y, U_pred[:,:,plot_out_axis[0]], cmap=color_map, vmin=umin, vmax=umax)
    ax1.set_xlabel(f"x_{str(plot_in_axis[0]+1)}", fontsize=16)
    ax1.set_ylabel(f"x_{str(plot_in_axis[1]+1)}", fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    cbar1 = fig.colorbar(surf1, shrink=0.8, aspect=20, pad=0.02)
    cbar1.set_label(f'u', fontsize=14)
    cbar1.ax.tick_params(labelsize=12)
    ax1.set_title('INN prediction', fontsize=16)

    surf2 = ax2.pcolormesh(X, Y, U_exact[:,:,plot_out_axis[0]], cmap=color_map, vmin=umin, vmax=umax)
    ax2.set_xlabel(f"x_{str(plot_in_axis[0]+1)}", fontsize=16)
    ax2.set_ylabel(f"x_{str(plot_in_axis[1]+1)}", fontsize=16)
    ax2.tick_params(axis='both', labelsize=12)
    cbar2 = fig.colorbar(surf2, shrink=0.8, aspect=20, pad=0.02)
    cbar2.set_label(f'u', fontsize=14)
    cbar2.ax.tick_params(labelsize=12)
    ax2.set_title('Original function', fontsize=16)

    parent_dir = os.path.abspath(os.getcwd())
    path_figure = os.path.join(parent_dir, 'plots')
    fig.savefig(os.path.join(path_figure, cls_data.data_name + "_" + model.interp_method) , dpi=300)
    plt.close()





        


    