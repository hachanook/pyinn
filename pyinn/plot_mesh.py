import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pyvista as pv
from stpyvista import stpyvista


def plot_mesh(model, cls_data, config):

    inp_filename = "Mesh_Hook.inp"
    xy, elem_nodes, elem_type = read_mesh_ANSYS(inp_filename)

    # Prepare data for the UnstructuredGrid
    num_cells = elem_nodes.shape[0]
    cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)

    # Connectivity array for PyVista
    connectivity = np.hstack([np.full((elem_nodes.shape[0], 1), 4), elem_nodes])

    # Create the unstructured grid
    grid = pv.UnstructuredGrid(connectivity, cell_types, xy)

    # Add the nodal data
    if model.interp_method == "linear" or model.interp_method == "nonlinear" or model.interp_method == "INN":
        U_pred = model.v_forward(model.params, xy) # (101,L)
    elif model.interp_method == "MLP":
        U_pred = model.v_forward(model.params, model.activation, xy) # (101,L)
    
    for i in range(U_pred.shape[1]):
        grid.point_data[f'u_{i+1}'] = U_pred[:, i]

    # Plot the mesh with the scalar field
    plotter = pv.Plotter()
    plotter.add_mesh(grid, scalars='u_1', cmap='viridis', show_edges=True)
    plotter.show()


    # grid.save('output.vtu')






def read_mesh_ANSYS(inp_filename):
    """Can read and identify 2D plane elements - CPE4, CPE6, CPE8 and 3D Tetrahedral and Brick elements - C3D4, C3D10, C3D8, C3D20"""
    # file_directory = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.getcwd())
    file_folder = 'data\\'
    path = os.path.join(parent_dir, file_folder, inp_filename)
    mesh_file = open(path,'r')
    lines = mesh_file.readlines()
    xy_list, elem_nodes_list = [], []
    for count,line in enumerate(lines):
        if 'N,UNBL,LOC' in line:
            # Coor_list = [items for items in line.strip().split(",")]
            # Nodal_Coor = [float(items) for items in Coor_list[6:]]
            Nodal_Coor = [float(items) for items in line.strip().split(",")[6:]]
            xy_list.append(Nodal_Coor)

        # elif 'EN,UNBL,ATTR' in line:
        #     connectivity_list_temp = []
        #     for count2,line2 in enumerate(lines[count+1:]):
        #         if 'EN,UNBL,NODE' in line2:
        #             line2 = line2.strip().split(",")
        #             connectivity_list_temp2 = [items for items in line2[3:] if items]
        #             connectivity_list_temp.extend(connectivity_list_temp2)
        #         else:
        #             Nodal_Connectivity = [float(items) for items in connectivity_list_temp]
        #             elem_nodes_list.append(Nodal_Connectivity)
        #             break
        elif 'EN,UNBL,NODE' in line:
            Nodal_Connectivity = [int(items) for items in line.strip().split(",")[3:] if items]
            elem_nodes_list.append(Nodal_Connectivity)


    elem_nodes = np.array(elem_nodes_list) - 1
    # elem_nodes = np.array(elem_nodes, dtype=np.int64)

    xy = np.array(xy_list)

    if np.all(xy[:,2] == 0):
        xy = xy[:,:-1]
    if xy.shape[1] == 2:
        n = elem_nodes.shape[1]
        elem_type = 'CPE' + str(n)
    else:
        d = xy.shape[1]
        n = elem_nodes.shape[1]
        elem_type = 'C' + str(d) + 'D' + str(n)

    return xy, elem_nodes, elem_type



# # Example data
# xy = np.random.rand(10, 3)  # Nodal coordinates (10 nodes in 3D space)
# elem_nodes = np.random.randint(0, 10, size=(20, 4))  # Connectivity (20 tetrahedral elements, 4 nodes each)
# u = np.random.rand(10, 1)  # Nodal output values

# # Prepare data for the UnstructuredGrid
# num_cells = elem_nodes.shape[0]
# cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)

# # Flatten the connectivity array for PyVista
# # connectivity = np.hstack([[4] + list(elem) for elem in elem_nodes])  # "4" is the number of points per tetrahedron
# connectivity = np.hstack([np.full((elem_nodes.shape[0], 1), 4), elem_nodes])

# # Create the unstructured grid
# grid = pv.UnstructuredGrid(connectivity, cell_types, xy)

# # Add the nodal data
# grid.point_data['Output'] = u.flatten()

# # Plot the mesh with the scalar field
# plotter = pv.Plotter()
# plotter.add_mesh(grid, scalars='Output', cmap='viridis', show_edges=True)
# plotter.show()