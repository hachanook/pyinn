import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from .model import *
from .dataset_regression import *
from .dataset_classification import *


def plot_regression(model, cls_data, config):
    bool_plot = config['PLOT']['bool_plot']
    
    if bool_plot:

        plot_in_axis = config['PLOT']['plot_in_axis']
        plot_out_axis = config['PLOT']['plot_out_axis']


        # make a directory if there isn't any
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        if len(plot_in_axis)==2 and  cls_data.bool_normalize == False:
            # we will plot the error only when there is no normalization on the original data.
            plot_2D_1D(model, cls_data, plot_in_axis, plot_out_axis)

            if config['interp_method'] != "MLP":
                plot_modes(model, cls_data, plot_in_axis, plot_out_axis)
        
        elif len(plot_in_axis)==1 and len(plot_out_axis)==1 and  cls_data.bool_normalize == False:
            # we will plot the error only when there is no normalization on the original data.
            plot_1D_1D(model, cls_data, plot_in_axis, plot_out_axis)

        elif len(plot_in_axis)==1 and len(plot_out_axis)==2 and  cls_data.bool_normalize == False:
            # we will plot the error only when there is no normalization on the original data.
            plot_1D_2D(model, cls_data, plot_in_axis, plot_out_axis)

        elif len(plot_in_axis)==2 and len(plot_out_axis)==1 and  cls_data.bool_normalize == False:
            # for spiral classification
            plot_2D_classification(model, cls_data, plot_in_axis, plot_out_axis)

        elif len(plot_in_axis)==3 and len(plot_out_axis)==1 and  cls_data.bool_normalize == True:
            # we will plot the error only when there is no normalization on the original data.
            # plot_2D_1D(model, cls_data, [0,1], plot_out_axis)
            if config['interp_method'] != "MLP":
                plot_modes(model, cls_data, plot_in_axis, plot_out_axis)

    else:
        print("\nPlotting deactivated\n")
        import sys
        sys.exit()

def plot_classification(model, cls_data, config):
    bool_plot = config['PLOT']['bool_plot']
    
    if bool_plot:
        plot_in_axis = config['PLOT']['plot_in_axis']
        plot_out_axis = config['PLOT']['plot_out_axis']


        # make a directory if there isn't any
        if not os.path.exists('plots'):
            os.makedirs('plots')
        
        if len(plot_in_axis)==2 and len(plot_out_axis)==1:
            # for spiral classification
            plot_2D_classification(model, cls_data, plot_in_axis, plot_out_axis)
    else:
        print("\nPlotting deactivated\n")
        import sys
        sys.exit()


def plot_1D_1D(model, cls_data, plot_in_axis, plot_out_axis):
    """ This function plots 2D input and 1D output data. By default, this function should work on original data only
    plot_in_axis: [axis1]
    plot_out_axis: [axis1]
    """

    ## create mesh and data
    ### in normalized space, create prediction
    xmin, xmax = cls_data.x_data_minmax["min"][plot_in_axis[0]], cls_data.x_data_minmax["max"][plot_in_axis[0]]
    
    x_nds = jnp.linspace(xmin, xmax, 101, dtype=jnp.float64).reshape(-1,1) # (101,1)
    if model.interp_method == "linear" or model.interp_method == "nonlinear":
        U_pred = model.v_forward(model.params, x_nds) # (101,L)
    elif model.interp_method == "MLP":
        U_pred = model.v_forward(model.params, model.activation, x_nds) # (101,L)
    
    U_exact = globals()["v_fun_"+cls_data.data_name](x_nds) # (101,L)    

    fig = plt.figure(figsize=(6,5))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0])
    # plt.subplots_adjust(wspace=0.4)  # Increase the width space between subplots

    ax1.plot(x_nds, U_exact, '-', color='k', linewidth = 4,  label='Original function')
    ax1.plot(x_nds, U_pred, '-', color='g', linewidth = 4,  label='Prediction')
    ax1.set_xlabel(fr"$x_{str(plot_in_axis[0]+1)}$", fontsize=16)
    ax1.set_ylabel(fr"$u_{str(plot_out_axis[0]+1)}$", fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    # ax1.set_title('INN prediction', fontsize=16)
    ax1.legend(shadow=True, borderpad=1, fontsize=14, loc='best')
    plt.tight_layout()

    parent_dir = os.path.abspath(os.getcwd())
    path_figure = os.path.join(parent_dir, 'plots')
    fig.savefig(os.path.join(path_figure, cls_data.data_name + "_" + model.interp_method) , dpi=300)
    plt.close()

def plot_modes(model, cls_data, plot_in_axis, plot_out_axis):
    """ This function plots multiple subplots in a dimension of (M, I). Each subplot represents 1D plot of input i at mode m.
    plot_in_axis: [axis1, axis2, ...]
    plot_out_axis: [axis1]
    """
    ## create mesh and data
    ### in normalized space, create prediction
    TD_type = model.config["TD_type"]

    if TD_type == "CP":
        params = model.params
    elif TD_type == "Tucker":
        params = model.params[1]
        print("Core tensor of Tucker product")
        print(model.params[0]) # print out 

    M = params.shape[0] # nmode
    I = params.shape[1] # dim
    J = params.shape[3] # nnode

    fig, axes = plt.subplots(M, I) 
    for idx, ax in enumerate(axes.flat):
        m = idx // I # m-th mode
        i = idx % I # i-th dimension

        xmin, xmax = cls_data.x_data_minmax["min"][plot_in_axis[i]], cls_data.x_data_minmax["max"][plot_in_axis[i]]
        x_nds = jnp.linspace(xmin, xmax, J, dtype=jnp.float64) # (J,)
        u_nds = params[m,i,plot_out_axis[0],:] # (J,)
        ax.plot(x_nds, u_nds, 'o')
        ax.set_title(f"{m+1}-th mode, {i+1}-th dim")

    plt.tight_layout()

    parent_dir = os.path.abspath(os.getcwd())
    path_figure = os.path.join(parent_dir, 'plots')
    TD_type = model.config['TD_type'] 
    fig.savefig(os.path.join(path_figure, cls_data.data_name + f"_{TD_type}_" + model.interp_method + f"_{M}modes") , dpi=300)
    plt.close()

def plot_1D_2D(model, cls_data, plot_in_axis, plot_out_axis, color_map="viridis", vmin=0, vmax=1, marker_size=20):
    """ This function plots 2D input and 1D output data. By default, this function should work on original data only
    plot_in_axis: [axis1, axis2]
    plot_out_axis: [axis1]
    """
    ## create mesh and data
    ### in normalized space, create prediction
    xmin, xmax = cls_data.x_data_minmax["min"][plot_in_axis[0]], cls_data.x_data_minmax["max"][plot_in_axis[0]]
    
    x_nds = jnp.linspace(xmin, xmax, 101, dtype=jnp.float64).reshape(-1,1) # (101,1)
    if model.interp_method == "linear" or model.interp_method == "nonlinear":
        U_pred = model.v_forward(model.params, x_nds) # (101,L)
    elif model.interp_method == "MLP":
        U_pred = model.v_forward(model.params, model.activation, x_nds) # (101,L)
    
    U_exact = globals()["v_fun_"+cls_data.data_name](x_nds) # (101,L)    

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    plt.subplots_adjust(wspace=0.4)  # Increase the width space between subplots

    ax1.plot(x_nds, U_exact[:,[plot_out_axis[0]]], '-', color='k', linewidth = 4,  label='Original function')
    ax1.plot(x_nds, U_pred[:,[plot_out_axis[0]]], '-', color='g', linewidth = 4,  label='Prediction')
    ax1.set_xlabel(fr"$x_{str(plot_in_axis[0]+1)}$", fontsize=16)
    ax1.set_ylabel(fr"$u_{str(plot_out_axis[0]+1)}$", fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    # ax1.set_title('INN prediction', fontsize=16)
    ax1.legend(shadow=True, borderpad=1, fontsize=14, loc='best')
    
    ax2.plot(x_nds, U_exact[:,[plot_out_axis[1]]], '-', color='k', linewidth = 4,  label='Original function')
    ax2.plot(x_nds, U_pred[:,[plot_out_axis[1]]], '-', color='g', linewidth = 4,  label='Prediction')
    ax2.set_xlabel(fr"$x_{str(plot_in_axis[0]+1)}$", fontsize=16)
    ax2.set_ylabel(fr"$u_{str(plot_out_axis[1]+1)}$", fontsize=16)
    ax2.tick_params(axis='both', labelsize=12)
    # ax1.set_title('INN prediction', fontsize=16)
    ax2.legend(shadow=True, borderpad=1, fontsize=14, loc='best')
    plt.tight_layout()


    parent_dir = os.path.abspath(os.getcwd())
    path_figure = os.path.join(parent_dir, 'plots')
    fig.savefig(os.path.join(path_figure, cls_data.data_name + "_" + model.interp_method) , dpi=300)
    plt.close()


def plot_2D_1D(model, cls_data, plot_in_axis, plot_out_axis, color_map="viridis", vmin=0, vmax=1, marker_size=20):
    """ This function plots 2D input and 1D output data. By default, this function should work on original data only
    plot_in_axis: [axis1, axis2]
    plot_out_axis: [axis1]
    """

    TD_type = model.config["TD_type"]

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
        U_pred = model.vv_forward(model.params, XY) # (101,101,L)
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
    ax1.set_title('Prediction', fontsize=16)

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
    fig.savefig(os.path.join(path_figure, cls_data.data_name + f"_{TD_type}_" + model.interp_method) , dpi=300)
    plt.close()


def plot_2D_classification(model, cls_data, plot_in_axis, plot_out_axis):

    TD_type = model.config["TD_type"]

    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    
    x_nds = jnp.linspace(xmin, xmax, 101, dtype=jnp.float64)
    y_nds = jnp.linspace(ymin, ymax, 101, dtype=jnp.float64)
    X,Y = jnp.meshgrid(x_nds, y_nds) # (101,101) each
    XY = jnp.dstack((X, Y)) # (101,101,2)
    if model.interp_method == "linear" or model.interp_method == "nonlinear":
        U_pred = model.vv_forward(model.params, XY) # (101,101,L)
    elif model.interp_method == "MLP":
        U_pred = model.vv_forward(model.params, model.activation, XY) # (101,101,L)
    U_pred_single = jnp.argmax(U_pred, axis=2)

    plt.figure(figsize=(6, 5))
    plt.set_cmap(plt.cm.Paired)
    plt.pcolormesh(X, Y, U_pred_single)


    ## debug
    all_inputs, all_labels = [], []

    for inputs, labels in model.test_dataloader:
        # Move data to CPU if it’s on GPU
        inputs = inputs.cpu().numpy()
        labels = labels.cpu().numpy()
        
        all_inputs.append(inputs)
        all_labels.append(labels)

    # Concatenate along the batch dimension
    x_data = np.concatenate(all_inputs, axis=0)
    u_data = np.concatenate(all_labels, axis=0)
    u_data_single = jnp.argmax(u_data, axis=1)
    
    plt.scatter(x_data[:,0], x_data[:,1], c=u_data_single, edgecolors='black')
    
    plt.xlabel(rf"$x_{plot_in_axis[0]+1}$", fontsize = 20)
    plt.ylabel(rf"$x_{plot_in_axis[1]+1}$", fontsize = 20)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.tight_layout()
    
    ## Save plot
    parent_dir = os.path.abspath(os.getcwd())
    path_figure = os.path.join(parent_dir, 'plots')
    plt.savefig(os.path.join(path_figure, cls_data.data_name + f"_{TD_type}_" + model.interp_method) , dpi=300)
    plt.close()

    


    