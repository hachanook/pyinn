import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os, sys
from pyinn import model, train, plot
sys.path.append('/pyinn')
from pyinn.dataset_regression import Data_regression_streamlit
from jax import config
import pyvista as pv
from stpyvista import stpyvista
config.update("jax_enable_x64", True) 


# Set page configuration
st.set_page_config(
    page_title="Space-Parameter",
    page_icon="📊",
    layout="centered",
)

# Page title
st.title("Space-Parameter")

st.markdown(
    """
    ### Problem statement
    If your physical simulation is static and depends on variable parameters such as geometry, material properties, 
    and boundary conditions, the problem is called "Space-Parameter" problem. 

    The inputs to the model are spatial variables (i.e., x,y,z coordinates) and variable parameters (p1, p2, ...) 
    while the outputs can be physical properties such as temperature and displacement.

    ##### Example - Crane Hook Design
    """
)

problem = Image.open("SP_problem_statement.jpg")
col1, col2 = st.columns([1, 1]) 
with col1: st.image(problem, use_container_width=True)

st.markdown(
    """
    
    For example, consider a crane hook design problem where the physical domain is represented in a 3D x,y,z coordinate system and three variable geometric parameters: 
    - $p_1 = t_x$; thickness in x-direction
    - $p_2 = t_y$; thickness in y-direction
    - $p_3 = t_z$; thickness in z-direction, \n
    making it a 6-dimensional problem. The outputs are x,y,z deformations and equivalent stress. 

    The training data shall be acquired by physical simulations or experiments, and be concatenated as a .csv file format. 
    In this problem, a dataset may look like:
    """
)

# Define the columns and a single row of data
columns = [" index "," &nbsp;&nbsp; 0 &nbsp;&nbsp;","&nbsp;&nbsp;&nbsp; 1 &nbsp;&nbsp;&nbsp;","&nbsp;&nbsp; 2 &nbsp;&nbsp;",
           "&nbsp;&nbsp; 3 &nbsp;&nbsp;","&nbsp;&nbsp; 4 &nbsp;&nbsp;","&nbsp;&nbsp; 5 &nbsp;&nbsp;","&nbsp;&nbsp; 6 &nbsp;&nbsp;",
           "&nbsp;&nbsp; 7 &nbsp;&nbsp;","&nbsp;&nbsp; 8 &nbsp;&nbsp;","&nbsp;&nbsp; 9 &nbsp;&nbsp;","&nbsp;&nbsp; 10 &nbsp;&nbsp;"]
table_header = "| " + " | ".join(columns) + " |"
table_separator = "| " + " | ".join(["---"] * 12) + " |"
st.markdown(f"{table_header}\n{table_separator}")
data_path = './data/6D_4D_ansys_29285.csv'
data = pd.read_csv(data_path)
st.dataframe(data.head())  # Display as an interactive table

st.markdown(
    """
    
    Then you may want to specify which columns are the inputs and outputs. In this problem, columns 1,2,3,4,5,6 are the inputs and 7,8,9,10 are the outputs.
    After setting hyperparameters for training, now you will be asked to run "Train the model". 

    """

)



if 'Start training' not in st.session_state:
    st.session_state.start_training = False
if 'Start plotting' not in st.session_state:
    st.session_state.start_plotting = False

# User input for uploading CSV data
st.markdown("### Upload Your Training Data")
uploaded_file = st.file_uploader("Drop your data (CSV) file here:", type=["csv"], help="Please upload a .csv file.")

if uploaded_file is not None:
    try:
        # Extract file name
        data_name = uploaded_file.name
        st.write(f"Uploaded data file: {data_name}")

        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)
        st.success("Data file uploaded successfully!")

        # Convert data to numpy array (removing headings)
        data_array = data.to_numpy()

        # Display the data
        st.markdown("### Preview of Your Data")
        st.dataframe(data.head())

        # Display basic info about the dataset
        st.markdown("### Dataset Information")
        st.write(f"Number of rows: {data.shape[0]}")
        st.write(f"Number of columns: {data.shape[1]}")

        # Input for specifying columns
        st.markdown("### Specify Columns")
        input_space_columns = st.text_input("Enter column indices for spatial inputs (comma-separated):", help="Specify the indices of the columns to be used as spatial inputs (e.g., 0, 1, 2).")
        input_parameter_columns = st.text_input("Enter column indices for parametric inputs (comma-separated):", help="Specify the indices of the columns to be used as parametric inputs (e.g., 3, 4, 5).")
        output_columns = st.text_input("Enter output columns (comma-separated):", help="Specify the indices of the columns to be used as outputs (e.g., 6, 7).")

    except Exception as e:
        st.error(f"An error occurred while processing the data file: {e}")
else:
    st.info("Awaiting file upload...")



# Convert input to list of integers
if 'output_columns' in globals():
    try:
        input_space_col = [int(col.strip()) for col in input_space_columns.split(",") if col.strip()] if input_space_columns else []
        input_parameter_col = [int(col.strip()) for col in input_parameter_columns.split(",") if col.strip()] if input_parameter_columns else []
        st.session_state.input_parameter_col = input_parameter_col
        input_col = input_space_col + input_parameter_col
        output_col = [int(col.strip()) for col in output_columns.split(",") if col.strip()] if output_columns else []
        # Ask user to choose the model type
        st.markdown("### Choose Model Type")
        interp_method = st.selectbox("Select the model type:", options=["INN", "MLP"], help="Choose the model type for training.")
        # st.write(f"**Selected Model Type:** {interp_method}")
        if interp_method == "INN":
            interp_method = "nonlinear" # use nonlinear INN as default

        # Additional user inputs
        st.markdown("### Additional Parameters")
        if interp_method == "INN" or "nonlinear":
            nelem = st.number_input("Number of elements per dimension:", min_value=1, step=1, help="Specify the number of elements per dimension.", value=40)
            nmode = st.number_input("Number of modes of CP decomposition:", min_value=1, step=1, help="Specify the number of modes of CP decomposition.", value=50)
            
        elif interp_method == "MLP":
            nlayers = st.number_input("Number of hidden layers:", min_value=1, step=1, help="Specify the number of hidden layers.", value=3)
            nneurons = st.number_input("Number of neurons per layer:", min_value=1, step=1, help="Specify the number of neurons per layer.", value=200)
        nepoch = st.number_input("Number of epochs:", min_value=1, step=1, help="Specify the number of epochs for training.", value=2)

        config = {"MODEL_PARAM": {"nelem": 40, "nmode": 50, "s_patch": 2, "alpha_dil": 20, "p_order": 2, "radial_basis": "cubicSpline", "INNactivation": "polynomial",
                                "nlayers": 3, "nneurons": 200, "activation": "sigmoid"},
                "DATA_PARAM": {"input_col": input_col, "output_col": output_col, "bool_normalize": True, "bool_random_split": True, "split_ratio": [0.8, 0.2]},
                "TRAIN_PARAM": {"num_epochs_INN": nepoch, "num_epochs_MLP": nepoch, "batch_size": 128, "learning_rate": 1e-3, "bool_train_acc": False, "validation_period": 10},
                "PLOT": {"bool_plot": False, "plot_in_axis": [3,4], "plot_out_axis": [0]}}
        config["interp_method"] = interp_method
        config["data_name"] = data_name
        config["TD_type"] = "CP"
        if interp_method == "INN" or "nonlinear":
            config["MODEL_PARAM"]["nelem"] = nelem
            config["MODEL_PARAM"]["nmode"] = nmode
        elif interp_method == "MLP":
            config["MODEL_PARAM"]["nlayers"] = nlayers
            config["MODEL_PARAM"]["nneurons"] = nneurons 

        ## data import
        data = Data_regression_streamlit(data_array, config)

    except ValueError:
        st.error("Please enter valid integer indices for the columns.")

    
## train
if "data" in globals():
    st.markdown("### Train the Model")
    if st.button("Start Training") or st.session_state.start_training:
        st.session_state.start_training = True
        # status_placeholder = st.empty()  # Create a placeholder for training status

        if interp_method == "INN" or "nonlinear":
            regressor = train.Regression_INN(data, config)  # HiDeNN-TD regressor class
        elif interp_method == "MLP":
            regressor = train.Regression_MLP(data, config)  # HiDeNN-TD regressor class
        regressor.train()  # Train module

        st.session_state.model = regressor # store the trained model as a global variable in streamlit
        st.success("Training completed!")


## plot 
st.markdown("### Plot trained model")

st.markdown("##### Upload Your Mesh (.inp) File")
uploaded_inp_file = st.file_uploader("Drop your mesh (.inp) file here:", type=["inp"], help="Please upload a .inp file.")

if uploaded_inp_file is not None:
    try:
        # Read the .inp file line by line
        lines = uploaded_inp_file.read().decode("utf-8").splitlines()
        xy_list, elem_nodes_list = [], []
        for count, line in enumerate(lines):
            if 'N,UNBL,LOC' in line:
                Nodal_Coor = [float(items) for items in line.strip().split(",")[6:]]
                xy_list.append(Nodal_Coor)
            elif 'EN,UNBL,NODE' in line:
                Nodal_Connectivity = [int(items) for items in line.strip().split(",")[3:] if items]
                elem_nodes_list.append(Nodal_Connectivity)
        elem_nodes = np.array(elem_nodes_list) - 1
        xy = np.array(xy_list)
        st.success("Mesh file uploaded successfully!")
        
    except Exception as e:
        st.error(f"An error occurred while processing the data file: {e}")
else:
    st.info("Awaiting file upload...")


# if st.session_state.start_training:

output_idx = st.number_input("Output index to be plotted:", min_value=1, step=1, help="Specify the output index to be plotted.")
output_label = st.text_input("Output label in text:", help="A short description of the output to be plotted.")
# st.write(f"Output index to be plotted: {output_idx}")
st.session_state.output_idx = output_idx
st.session_state.output_label = output_label

if 'input_parameter_col' in globals():
    parameters = []
    for idx, col in enumerate(input_parameter_col):
        p = st.number_input(rf"$p_{idx+1}$:", min_value=0.0, step=0.0001, help=f"Specify the value of {idx+1}-th parameter.", value=0.015)
        parameters.append(p)
    st.markdown("Parametric inputs:")
    st.write(f"{parameters}")

    
if st.button("Start Plotting"):
    st.session_state.start_plotting = True
    output_idx = st.session_state.output_idx
#     try:
    ## Prepare data for the UnstructuredGrid
    num_cells = elem_nodes.shape[0]
    cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)

    ## Connectivity array for PyVista
    connectivity = np.hstack([np.full((elem_nodes.shape[0], 1), 4), elem_nodes])

    ## Create the unstructured grid
    mesh = pv.UnstructuredGrid(connectivity, cell_types, xy)

    p_org = np.tile(parameters, (xy.shape[0], 1)) 
    x_org = np.hstack((xy, p_org))

    ## Normalize inputs
    x = (x_org - st.session_state.model.cls_data.x_data_minmax["min"]) / (
        st.session_state.model.cls_data.x_data_minmax["max"] - st.session_state.model.cls_data.x_data_minmax["min"])
    
    ## Add the nodal data
    if st.session_state.model.interp_method == "linear" or st.session_state.model.interp_method == "nonlinear" or st.session_state.model.interp_method == "INN":
        U_pred = st.session_state.model.v_forward(st.session_state.model.params, x) # (101,L)
    elif st.session_state.model.interp_method == "MLP":
        U_pred = st.session_state.model.v_forward(st.session_state.model.params, st.session_state.model.activation, x) # (101,L)
    
    st.write(f"{U_pred.shape}")
    ## Denormalize
    U_pred_org = (st.session_state.model.cls_data.u_data_minmax["max"] - st.session_state.model.cls_data.u_data_minmax["min"]
                  ) * U_pred + st.session_state.model.cls_data.u_data_minmax["min"]

    for i in range(U_pred.shape[1]):
        mesh.point_data[f'u_{i+1}'] = U_pred_org[:, i]

    plotter = pv.Plotter()
    plotter.add_mesh(mesh, scalars=f"u_{output_idx}", show_edges=True, color="lightblue", cmap="viridis", 
                     show_scalar_bar=True, scalar_bar_args={"title": f"{st.session_state.output_label}"})
    # plotter.add_axes()         
    plotter.view_isometric()            
    # Display the mesh using stpyvista Plotter
    stpyvista(plotter, key="pv_cube")

    # except Exception as e:
    #     st.error(f"An error occurred while plotting results: {e}")

    st.success("Plotting completed!")

                
        
            






        

        






