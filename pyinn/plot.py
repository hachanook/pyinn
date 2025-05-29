import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as animation
import os
# from .model import * ## when using pyinn
# from .dataset_regression import *
# from .dataset_classification import *
from model import * ## when debugging
from dataset_regression import *
from dataset_classification import *
# from pyinn.model import * ## when debugging on streamlit
# from pyinn.dataset_regression import *
# from pyinn.dataset_classification import *

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

            if config['interp_method'] != "MLP" and isinstance(config['MODEL_PARAM']['nelem'], int):
                plot_modes(model, cls_data, plot_in_axis, plot_out_axis)
        
        elif len(plot_in_axis)==1 and len(plot_out_axis)==1:
            # we will plot the error only when there is no normalization on the original data.
            plot_1D_1D(model, cls_data, plot_in_axis, plot_out_axis)

        elif len(plot_in_axis)==1 and len(plot_out_axis)==2:
            # we will plot the error only when there is no normalization on the original data.
            plot_1D_2D(model, cls_data, plot_in_axis, plot_out_axis)

        # elif len(plot_in_axis)==2 and len(plot_out_axis)==1 and  cls_data.bool_normalize == False:
        #     # for spiral classification
        #     plot_2D_classification(model, cls_data, plot_in_axis, plot_out_axis)

        elif len(plot_in_axis)==3 and len(plot_out_axis)==1 and  cls_data.bool_normalize == True:
            # we will plot the error only when there is no normalization on the original data.
            # plot_2D_1D(model, cls_data, [0,1], plot_out_axis)
            if config['interp_method'] != "MLP" and isinstance(config['MODEL_PARAM']['nelem'], int):
                plot_modes(model, cls_data, plot_in_axis, plot_out_axis)

        elif "turbulence" in cls_data.data_name:
            plot_turbulence(model, cls_data, plot_in_axis, plot_out_axis)

        

        ## plot loss landscape
        plot_loss_landscape(model, cls_data) 


    else:
        print("\nPlotting deactivated\n")
        # import sys
        # sys.exit()





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
        # import sys
        # sys.exit()


def plot_1D_1D(model, cls_data, plot_in_axis, plot_out_axis):
    """ This function plots 2D input and 1D output data in the original space.
    plot_in_axis: [axis1]
    plot_out_axis: [axis1]
    """

    ## when the data is normalized
    if model.config['DATA_PARAM']['bool_normalize']:
        ### in normalized space, create prediction
        x_pred = jnp.linspace(0, 1, 101, dtype=jnp.float64).reshape(-1,1) # (101,1)
        U_pred = model.v_forward(model.params, x_pred) # (101,1)
        x_pred_org, U_pred_org = cls_data.denormalize(x_pred, U_pred)
        if model.interp_method == "linear" or model.interp_method == "nonlinear": # for INNs
            x_grid = jnp.linspace(0, 1, model.nnode, dtype=jnp.float64).reshape(-1,1)
            U_grid = model.v_forward(model.params, x_grid) # for grid points
            x_grid_org, U_grid_org = cls_data.denormalize(x_grid, U_grid)
        
    ## when the data is not normalized
    else:
        ### in original space, create prediction
        xmin, xmax = cls_data.x_data_minmax["min"][plot_in_axis[0]], cls_data.x_data_minmax["max"][plot_in_axis[0]]
        x_pred_org = jnp.linspace(xmin, xmax, 101, dtype=jnp.float64).reshape(-1,1) # (101,1)
        U_pred_org = model.v_forward(model.params, x_pred_org) # (101,1)
        if model.interp_method == "linear" or model.interp_method == "nonlinear": # for INNs
            x_grid_org = jnp.linspace(xmin, xmax, model.nnode, dtype=jnp.float64).reshape(-1,1)
            U_grid_org = model.v_forward(model.params, x_grid_org) # for grid points
        
    
    U_exact_org = globals()["v_fun_"+cls_data.data_name](x_pred_org) # (101,1)    

    fig = plt.figure(figsize=(6,5))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0])
    # plt.subplots_adjust(wspace=0.4)  # Increase the width space between subplots

    ax1.plot(x_pred_org, U_exact_org, '-', color='k', linewidth = 4,  label='Original function')
    ax1.plot(x_pred_org, U_pred_org, '-', color='g', linewidth = 4,  label='Prediction')
    if model.interp_method == "linear" or model.interp_method == "nonlinear":
        ax1.plot(x_grid_org, U_grid_org, 'o', color='r',  markersize=5, label='Grid points')
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
    
    ## when the data is normalized
    if model.config['DATA_PARAM']['bool_normalize']:
        ### in normalized space, create prediction
        x_pred = jnp.linspace(0, 1, 101, dtype=jnp.float64).reshape(-1,1) # (101,1)
        U_pred = model.v_forward(model.params, x_pred) # (101,L=2)
        x_pred_org, U_pred_org = cls_data.denormalize(x_pred, U_pred)
        if model.interp_method == "linear" or model.interp_method == "nonlinear": # for INNs
            x_grid = jnp.linspace(0, 1, model.nnode, dtype=jnp.float64).reshape(-1,1)
            U_grid = model.v_forward(model.params, x_grid) # for grid points
            x_grid_org, U_grid_org = cls_data.denormalize(x_grid, U_grid)
        
    ## when the data is not normalized
    else:
        ### in original space, create prediction
        xmin, xmax = cls_data.x_data_minmax["min"][plot_in_axis[0]], cls_data.x_data_minmax["max"][plot_in_axis[0]]
        x_pred_org = jnp.linspace(xmin, xmax, 101, dtype=jnp.float64).reshape(-1,1) # (101,1)
        U_pred_org = model.v_forward(model.params, x_pred_org) # (101,1)
        if model.interp_method == "linear" or model.interp_method == "nonlinear": # for INNs
            x_grid_org = jnp.linspace(xmin, xmax, model.nnode, dtype=jnp.float64).reshape(-1,1)
            U_grid_org = model.v_forward(model.params, x_grid_org) # for grid points

    U_exact_org = globals()["v_fun_"+cls_data.data_name](x_pred_org) # (101,L=2)    

    fig = plt.figure(figsize=(10,5))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    plt.subplots_adjust(wspace=0.4)  # Increase the width space between subplots

    ax1.plot(x_pred_org, U_exact_org[:,[plot_out_axis[0]]], '-', color='k', linewidth = 4,  label='Original function')
    ax1.plot(x_pred_org, U_pred_org[:,[plot_out_axis[0]]], '-', color='g', linewidth = 4,  label='Prediction')
    if model.interp_method == "linear" or model.interp_method == "nonlinear":
        ax1.plot(x_grid_org, U_grid_org[:,[plot_out_axis[0]]], 'o', color='r',  markersize=5, label='Grid points')
    ax1.set_xlabel(fr"$x_{str(plot_in_axis[0]+1)}$", fontsize=16)
    ax1.set_ylabel(fr"$u_{str(plot_out_axis[0]+1)}$", fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    ax1.legend(shadow=True, borderpad=1, fontsize=14, loc='best')
    
    ax2.plot(x_pred_org, U_exact_org[:,[plot_out_axis[1]]], '-', color='k', linewidth = 4,  label='Original function')
    ax2.plot(x_pred_org, U_pred_org[:,[plot_out_axis[1]]], '-', color='g', linewidth = 4,  label='Prediction')
    if model.interp_method == "linear" or model.interp_method == "nonlinear":
        ax2.plot(x_grid_org, U_grid_org[:,[plot_out_axis[1]]], 'o', color='r',  markersize=5, label='Grid points')
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



def plot_loss_landscape(model, cls_data):
    """ This function plots training and validation loss landscape."""
    
    fig = plt.figure(figsize=(6,5))
    gs = gridspec.GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[0])
    # plt.subplots_adjust(wspace=0.4)  # Increase the width space between subplots

    ax1.plot(model.errors_epoch, model.errors_train, '-', color='k', linewidth = 3,  label='Training loss')
    ax1.plot(model.errors_epoch, model.errors_val, '--', color='g', linewidth = 3,  label='Validation loss')
    ax1.set_xlabel('Epoch', fontsize=16)
    ax1.set_ylabel('Loss', fontsize=16)
    ax1.tick_params(axis='both', labelsize=12)
    nepoch = model.errors_epoch[-1]
    ax1.set_xlim(0, nepoch)
    ax1.set_yscale('log')
    ax1.set_ylim(1e-4, 1e0)
    # ax1.set_title('INN prediction', fontsize=16)
    ax1.legend(shadow=True, borderpad=1, fontsize=14, loc='best')
    plt.tight_layout()

    parent_dir = os.path.abspath(os.getcwd())
    path_figure = os.path.join(parent_dir, 'plots')
    fig.savefig(os.path.join(path_figure, cls_data.data_name + "_" + model.interp_method + f"_loss_{nepoch}epoch") , dpi=300)
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
    U_pred = model.vv_forward(model.params, XY) # (101,101,L)
        

    
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


def plot_turbulence(model, cls_data, plot_in_axis, plot_out_axis):

    size = 256
    outer_steps = 200

    data = cls_data.data # (200*256*256, 5)
    x_data_org = data[:,:3] # (200*256*256, 3) for x,y,t
    x_data = cls_data.normalize(x_data = x_data_org)[0]
    
    nblock = 1000 # number of blocks
    ndata = x_data_org.shape[0]
    data_idx_list = jnp.array_split(jnp.arange(ndata, dtype=jnp.int64), nblock, axis=0)
    x_data_list = jnp.array_split(x_data, nblock, axis=0)
    u_data = jnp.zeros((ndata, 2), dtype=jnp.float64)
    for data_idx_block, x_data_block in zip(data_idx_list, x_data_list):
        u_data_block = model.v_forward(model.params, x_data_block) # (200*256*256, 2)
        u_data = u_data.at[data_idx_block,:].set(u_data_block)
    u_data_org = cls_data.denormalize(u_data = u_data)[0]

    # Reshape u and v trajectories for visualization
    u_traj_reshaped = u_data_org[:,0].reshape(outer_steps, size, size)
    v_traj_reshaped = u_data_org[:,1].reshape(outer_steps, size, size)

    # Create a figure and axes for the animation
    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle("Time Evolution of u and v")

    # Initialize the plots
    u_plot = ax_u.imshow(u_traj_reshaped[0], origin='lower', extent=(0, 2 * jnp.pi, 0, 2 * jnp.pi), cmap='viridis')
    v_plot = ax_v.imshow(v_traj_reshaped[0], origin='lower', extent=(0, 2 * jnp.pi, 0, 2 * jnp.pi), cmap='viridis')

    ax_u.set_title("u velocity")
    ax_v.set_title("v velocity")
    ax_u.set_xlabel("x")
    ax_u.set_ylabel("y")
    ax_v.set_xlabel("x")
    ax_v.set_ylabel("y")

    # Add colorbars
    fig.colorbar(u_plot, ax=ax_u)
    fig.colorbar(v_plot, ax=ax_v)

    # Update function for the animation
    def update(frame):
        u_plot.set_data(u_traj_reshaped[frame])
        v_plot.set_data(v_traj_reshaped[frame])
        return u_plot, v_plot

    # Create the animation
    ani = animation.FuncAnimation(fig, update, frames=outer_steps, interval=50, blit=True)

    # Save the animation as a video file
    parent_dir = os.path.abspath(os.getcwd())
    path_figure = os.path.join(parent_dir, 'plots')
    ani.save(os.path.join(path_figure, cls_data.data_name + "_" + model.interp_method + "_timeseries.gif"), writer="ffmpeg", fps=20)

    # Close the plot to avoid displaying it
    plt.close(fig)



def plot_2D_classification(model, cls_data, plot_in_axis, plot_out_axis):

    TD_type = model.config["TD_type"]

    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    
    x_nds = jnp.linspace(xmin, xmax, 101, dtype=jnp.float64)
    y_nds = jnp.linspace(ymin, ymax, 101, dtype=jnp.float64)
    X,Y = jnp.meshgrid(x_nds, y_nds) # (101,101) each
    XY = jnp.dstack((X, Y)) # (101,101,2)
    U_pred = model.vv_forward(model.params, XY) # (101,101,L)
    U_pred_single = jnp.argmax(U_pred, axis=2)

    plt.figure(figsize=(6, 5))
    plt.set_cmap(plt.cm.Paired)
    plt.pcolormesh(X, Y, U_pred_single)


    ## debug
    all_inputs, all_labels = [], []

    for inputs, labels in cls_data.test_dataloader:
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

    


def read_mesh_ANSYS(inp_filename):
    """Can read and identify 2D plane elements - CPE4, CPE6, CPE8 and 3D Tetrahedral and Brick elements - C3D4, C3D10, C3D8, C3D20"""
    file_directory = os.path.dirname(__file__)
    file_folder = '\\data'
    filename = inp_filename
    path = os.path.join(file_directory, file_folder, filename)
    mesh_file = open(path,'r')
    lines = mesh_file.readlines()
    xy, elem_nodes_list = [], []
    for count,line in enumerate(lines):
        if 'N,UNBL,LOC' in line:
            Coor_list = [items for items in line.strip().split(",")]
            Nodal_Coor = [float(items) for items in Coor_list[6:]]
            xy.append(Nodal_Coor)

        if 'EN,UNBL,ATTR' in line:
            connectivity_list_temp = []
            for count2,line2 in enumerate(lines[count+1:]):
                if 'EN,UNBL,NODE' in line2:
                    line2 = line2.strip().split(",")
                    connectivity_list_temp2 = [items for items in line2[3:] if items]
                    connectivity_list_temp.extend(connectivity_list_temp2)
                else:
                    Nodal_Connectivity = [float(items) for items in connectivity_list_temp]
                    elem_nodes_list.append(Nodal_Connectivity)
                    break

    elem_nodes = np.array(elem_nodes_list) - 1
    elem_nodes = np.array(elem_nodes, dtype=np.int64)

    XY = np.array(xy)

    if np.all(XY[:,2] == 0):
        XY = XY[:,:-1]
    if XY.shape[1] == 2:
        n = elem_nodes.shape[1]
        elem_type = 'CPE' + str(n)
    else:
        d = XY.shape[1]
        n = elem_nodes.shape[1]
        elem_type = 'C' + str(d) + 'D' + str(n)

    return XY, elem_nodes, elem_type
