import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import os, sys, time
sys.path.append('/pyinn')
from jax import config
import pyvista as pv
from stpyvista import stpyvista
config.update("jax_enable_x64", True) 


# from pyinn import dataset_classification, dataset_regression, model, train, plot # with pyinn library
# import sys # for debugging
# sys.path.append('../pyinn')
from pyinn import dataset_classification, dataset_regression, model, train, plot # for debugging


gpu_idx = 7 # set which GPU to run on Athena
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing

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
data_path = './data/6D_4D_ansys_test.csv'
data = pd.read_csv(data_path)
st.dataframe(data.head())  # Display as an interactive table

st.markdown(
    """
    
    Then you may want to specify which columns are the inputs and outputs. In this problem, columns 1,2,3,4,5,6 are the inputs and 7,8,9,10 are the outputs.
    After setting hyperparameters for training, now you will be asked to run "Train the model". 

    """

)



# if 'Start training' not in st.session_state:
#     st.session_state.start_training = False
# if 'Start plotting' not in st.session_state:
#     st.session_state.start_plotting = False
# print(st.session_state)
if  st.session_state.get("complete_config_input") is None:
    # print(st.session_state)
    st.session_state.complete_config_input = False
    st.session_state.complete_data = False
    st.session_state.complete_train = False
    st.session_state.complete_mesh_file = False
    st.session_state.complete_config_plot = False
    st.session_state.complete_plot = False
    # print(st.session_state)


## Data input

data_type = st.selectbox("Select the data type:", options=["Entire data with split ratio", 
                                                            "Train and test data", 
                                                            "Train, validation, and test data"], help="Choose the type of data to be imported.")
if data_type == "Entire data with split ratio":
    # User input for uploading CSV data
    st.markdown("### Upload Your Entire Data")
    uploaded_file = st.file_uploader("Drop your data (CSV) file here:", type=["csv"], help="Please upload a .csv file.")
    if uploaded_file is not None:
        # Extract file name
        data_name = uploaded_file.name
        st.write(f"Uploaded data file: {data_name}")

        # Read the uploaded CSV file
        data = pd.read_csv(uploaded_file)
        st.success("Data file uploaded successfully!")

        # Display the data
        st.markdown("### Preview of Your Data")
        st.dataframe(data.head())

        # Convert data to numpy array (removing headings)
        data = data.to_numpy()
        data_list = [data]

        # Display basic info about the dataset
        st.markdown("### Dataset Information")
        st.write(f"Number of rows: {data.shape[0]}")
        st.write(f"Number of columns: {data.shape[1]}")

        # Input for specifying columns
        st.markdown("### Specify Columns")
        input_space_columns = st.text_input("Enter column indices for spatial inputs (comma-separated):", help="Specify the indices of the columns to be used as spatial inputs (e.g., 0, 1, 2).")
        input_parameter_columns = st.text_input("Enter column indices for parametric inputs (comma-separated):", help="Specify the indices of the columns to be used as parametric inputs (e.g., 3, 4, 5).")
        output_columns = st.text_input("Enter output columns (comma-separated):", help="Specify the indices of the columns to be used as outputs (e.g., 6, 7).")

        split_ratio_input = st.text_input("Enter data split ratio:", help="Specify the split ratio of the data (e.g., 0.7, 0.15, 0.15).")
        st.session_state.split_ratio = [float(col.strip()) for col in split_ratio_input.split(",") if col.strip()] if split_ratio_input else []

        while not split_ratio_input:
            # st.warning("Please enter split ratio to proceed...")
            time.sleep(1)
        # st.rerun()
        st.session_state.complete_config_input = True

elif data_type == "Train and test data":
    print("debugging")

    # User input for uploading CSV data
    st.markdown("### Upload Your Train Data")
    uploaded_file_train = st.file_uploader("Drop your data (CSV) file here:", type=["csv"], help="Please upload a .csv file.", key=123)

    st.markdown("### Upload Your Test Data")
    uploaded_file_test = st.file_uploader("Drop your data (CSV) file here:", type=["csv"], help="Please upload a .csv file.", key=456)


    if uploaded_file_train is not None and uploaded_file_test is not None:
        # Extract file name
        data_name_train = uploaded_file_train.name
        st.write(f"Uploaded data file: {data_name_train}")
        data_name_test = uploaded_file_test.name
        st.write(f"Uploaded data file: {data_name_test}")
        data_name = data_name_train

        # Read the uploaded CSV file
        data_train = pd.read_csv(uploaded_file_train)
        data_test = pd.read_csv(uploaded_file_test)
        st.success("Data file uploaded successfully!")

        # Display the data
        st.markdown("### Preview of Your Data")
        st.dataframe(data_train.head())

        # Convert data to numpy array (removing headings)
        data_train = data_train.to_numpy()
        data_test = data_test.to_numpy()
        data_list = [data_train, data_test]

        # Display basic info about the dataset
        st.markdown("### Dataset Information")
        st.write(f"Number of rows: {data.shape[0]}")
        st.write(f"Number of columns: {data.shape[1]}")

        # Input for specifying columns
        st.markdown("### Specify Columns")
        input_space_columns = st.text_input("Enter column indices for spatial inputs (comma-separated):", help="Specify the indices of the columns to be used as spatial inputs (e.g., 0, 1, 2).")
        input_parameter_columns = st.text_input("Enter column indices for parametric inputs (comma-separated):", help="Specify the indices of the columns to be used as parametric inputs (e.g., 3, 4, 5).")
        output_columns = st.text_input("Enter output columns (comma-separated):", help="Specify the indices of the columns to be used as outputs (e.g., 6, 7).")

        st.session_state.complete_config_input = True

elif data_type == "Train, validation, and test data":
    print("debugging")
        
else:
    st.info("Awaiting file upload...")




# Create data class
# if st.button("Create data class") :
if st.session_state.complete_config_input == True and st.session_state.complete_data == False:

    input_space_col = [int(col.strip()) for col in input_space_columns.split(",") if col.strip()] if input_space_columns else []
    input_parameter_col = [int(col.strip()) for col in input_parameter_columns.split(",") if col.strip()] if input_parameter_columns else []
    st.session_state.input_parameter_col = input_parameter_col
    input_col = input_space_col + input_parameter_col
    output_col = [int(col.strip()) for col in output_columns.split(",") if col.strip()] if output_columns else []
    # Ask user to choose the model type
    st.markdown("### Choose Model Type")
    model_type = st.selectbox("Select the model type:", options=["INN", "MLP"], help="Choose the model type for training.")
    st.session_state.model_type = model_type
    # st.write(f"**Selected Model Type:** {interp_method}")
    st.markdown("### Additional Model Parameters")
    
    # Additional user inputs
    if model_type == "INN":
        st.session_state.interp_method = "nonlinear" # use nonlinear INN as default
        nseg = st.number_input("Number of segments per dimension:", min_value=1, step=1, help="Specify the number of segments per dimension.", value=None)
        nmode = st.number_input("Number of modes of CP decomposition:", min_value=1, step=1, help="Specify the number of modes of CP decomposition.", value=None)
        while nseg is None or nmode is None :
            # st.warning("Please enter nseg and nmode to proceed...")
            time.sleep(1)
            # st.rerun()
    elif model_type == "MLP":
        st.session_state.interp_method = "MLP" # 
        nlayers = st.number_input("Number of hidden layers:", min_value=1, step=1, help="Specify the number of hidden layers.", value=None)
        nneurons = st.number_input("Number of neurons per layer:", min_value=1, step=1, help="Specify the number of neurons per layer.", value=None)
        while nlayers is None or nneurons is None :
            # st.warning("Please enter nlayers and nneurons to proceed...")
            time.sleep(1)
            # st.rerun()    

    config = {"MODEL_PARAM": {"nseg": 20, 
                                "nmode": 100, 
                                "s_patch": 4, 
                                "alpha_dil": 20, 
                                "p_order": 1, 
                                "radial_basis": "cubicSpline", 
                                "INNactivation": "polynomial",
                                "nlayers": 3, 
                                "nneurons": 100, 
                                "activation": "sigmoid"},
            "DATA_PARAM": {"input_col": input_col, 
                            "output_col": output_col, 
                            "bool_data_generation": False,
                            "bool_normalize": True, 
                            "bool_shuffle": True},
            "TRAIN_PARAM": {"num_epochs_INN": 100, 
                            "num_epochs_MLP": 100, 
                            "batch_size": 128, 
                            "learning_rate": 1e-3, 
                            "bool_train_acc": False, 
                            "validation_period": 1,
                            "bool_denormalize": False,
                            "error_type": "rmse",
                            "patience": 5},
            "PLOT": {"bool_plot": False, 
                        "plot_in_axis": [3,4], 
                        "plot_out_axis": [0]}}
    if st.session_state.get("split_ratio") is not None:
        config["DATA_PARAM"]["split_ratio"] = st.session_state.split_ratio
        # print("split ratio is added \n\n")

    config["interp_method"] = st.session_state.interp_method
    config["data_name"] = data_name
    config["TD_type"] = "CP"
    if (st.session_state.interp_method == "INN" or "nonlinear") and 'nseg' in globals():
        config["MODEL_PARAM"]["nseg"] = nseg
        config["MODEL_PARAM"]["nmode"] = nmode
    elif st.session_state.interp_method == "MLP" and 'nlayers' in globals():
        config["MODEL_PARAM"]["nlayers"] = nlayers
        config["MODEL_PARAM"]["nneurons"] = nneurons 

    
    st.session_state.config = config
    ## data import
    # print(st.session_state.complete_train, st.session_state.complete_mesh_file)
    st.session_state.data = dataset_regression.Data_regression(data_name, config, data_list)
    st.session_state.complete_data = True

    

    
## train
if st.session_state.complete_data == True and st.session_state.complete_train == False:
    st.markdown("### Train the Model")

    nepoch = st.number_input("Number of epochs:", min_value=1, step=1, help="Specify the number of epochs for training.", value=None)
    while nepoch is None:
        # st.warning("Please enter nepoch to proceed...")
        time.sleep(1)
        # st.rerun()
    st.session_state.config["TRAIN_PARAM"]["num_epochs_INN"] = nepoch
    st.session_state.config["TRAIN_PARAM"]["num_epochs_MLP"] = nepoch


    if st.button("Start Training"):

        if st.session_state.interp_method == "INN" or st.session_state.interp_method == "nonlinear":
            regressor = train.Regression_INN(st.session_state.data, st.session_state.config)  # HiDeNN-TD regressor class
        elif st.session_state.interp_method == "MLP":
            regressor = train.Regression_MLP(st.session_state.data, st.session_state.config)  # HiDeNN-TD regressor class
        regressor.train()  # Train module

        st.session_state.model = regressor # store the trained model as a global variable in streamlit
        st.success("Training completed!")
        st.session_state.complete_train = True


############################# plot ###############################
# st.markdown("### Plot trained model")
if st.session_state.complete_train == True and st.session_state.complete_mesh_file == False:
    st.markdown("### Plot trained model")

    st.markdown("##### Upload Your Mesh (.inp) File")
    uploaded_inp_file = st.file_uploader("Drop your mesh (.inp) file here:", type=["inp"], help="Please upload a .inp file.")

    if uploaded_inp_file is not None:
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
        st.session_state.elem_nodes = np.array(elem_nodes_list) - 1
        st.session_state.xy = np.array(xy_list)
        st.success("Mesh file uploaded successfully!")
        st.session_state.complete_mesh_file = True
            
    else:
        st.info("Awaiting file upload...")


if st.session_state.complete_mesh_file == True and st.session_state.complete_config_plot == False:
    
    output_idx = st.number_input("Output index to be plotted:", min_value=1, step=1, help="Specify the output index to be plotted.", value=None)
    output_label = st.text_input("Output label in text:", help="A short description of the output to be plotted.", value=None)
    st.session_state.output_idx = output_idx
    st.session_state.output_label = output_label
    while output_idx is None or output_label is None :
        # st.warning("Please enter output_idx and output_label to proceed...")
        time.sleep(1)
        # st.rerun()
    # st.write(f"Output index to be plotted: {output_idx}")
    # st.session_state.output_idx = output_idx
    # st.session_state.output_label = output_label

    if st.session_state.get("input_parameter_col") is not None:
        parameters = []
        p_ref = [0.01228624, 0.01671141, 0.01481995]
        for idx, col in enumerate(st.session_state.input_parameter_col):
            p = st.number_input(rf"$p_{idx+1}$:", min_value=0.0, step=0.0001, format="%.4f", 
                                help=f"Specify the value of {idx+1}-th parameter.", value=p_ref[idx])
            # while p is None:
            #     # st.warning("Please enter parameters to proceed...")
            #     time.sleep(1)
            #     # st.rerun()
            parameters.append(p)
        st.markdown("Parametric inputs:")
        st.write(f"{parameters}")
        st.session_state.parameters = parameters
    
    st.session_state.complete_config_plot = True
    

    
if st.session_state.complete_config_plot:
    if st.button("Start Plotting"):
        # st.session_state.start_plotting = True
        output_idx = st.session_state.output_idx

        num_cells = st.session_state.elem_nodes.shape[0]
        cell_types = np.full(num_cells, pv.CellType.TETRA, dtype=np.uint8)

        ## Connectivity array for PyVista
        connectivity = np.hstack([np.full((st.session_state.elem_nodes.shape[0], 1), 4), st.session_state.elem_nodes])

        ## Create the unstructured grid
        mesh = pv.UnstructuredGrid(connectivity, cell_types, st.session_state.xy)


        p_org = np.tile(st.session_state.parameters, (st.session_state.xy.shape[0], 1)) 
        x_org = np.hstack((st.session_state.xy, p_org))

        ## Normalize inputs
        x = (x_org - st.session_state.data.x_data_minmax["min"]) / (
            st.session_state.data.x_data_minmax["max"] - st.session_state.data.x_data_minmax["min"])
        
        ## Add the nodal data
        if st.session_state.interp_method == "linear" or st.session_state.interp_method == "nonlinear" or st.session_state.interp_method == "INN":
            U_pred = st.session_state.model.v_forward(st.session_state.model.params, x) # (101,L)
        elif st.session_state.interp_method == "MLP":
            U_pred = st.session_state.model.v_forward(st.session_state.model.params, st.session_state.model.activation, x) # (101,L)
        st.write(f"{U_pred.shape}")

        ## Denormalize
        U_pred_org = (st.session_state.data.u_data_minmax["max"] - st.session_state.data.u_data_minmax["min"]
                    ) * U_pred + st.session_state.data.u_data_minmax["min"]

        for i in range(U_pred.shape[1]):
            mesh.point_data[f'u_{i+1}'] = U_pred_org[:, i]

        plotter = pv.Plotter()
        plotter.add_mesh(
            mesh, 
            scalars=f"u_{output_idx}", 
            show_edges=False,  # Remove element boundaries
            color="lightblue", 
            cmap="viridis", 
            show_scalar_bar=True, 
            clim= (np.min(U_pred_org[:,output_idx-1]), 
            np.max(U_pred_org[:,output_idx-1])), 
            scalar_bar_args={"title": f"{st.session_state.output_label}"})
        # plotter.add_axes()         
        plotter.view_isometric()            
        # Display the mesh using stpyvista Plotter
        # stpyvista(plotter, key="pv_cube")


        # Create reference plot, data generated from Ansys
        ## Create the unstructured grid
        mesh_ref = pv.UnstructuredGrid(connectivity, cell_types, st.session_state.xy)

        # p_org = np.tile(parameters, (st.session_state.xy.shape[0], 1)) 
        # x_org = np.hstack((st.session_state.xy, p_org))

        ## read the test data
        data_path = './data/6D_4D_ansys_test.csv'
        data = pd.read_csv(data_path)
        data = data.to_numpy()
        U_ref_org = data[:,st.session_state.config["DATA_PARAM"]["output_col"]]
        print(U_ref_org.shape)

        for i in range(U_ref_org.shape[1]):
            mesh_ref.point_data[f'u_{i+1}'] = U_ref_org[:, i]

        plotter_ref = pv.Plotter()
        plotter_ref.add_mesh(
            mesh_ref,
            scalars=f"u_{output_idx}",
            show_edges=False,  # Remove element boundaries
            color="lightblue",
            cmap="viridis",
            show_scalar_bar=True,
            clim=(np.min(U_pred_org[:, output_idx - 1]), np.max(U_pred_org[:, output_idx - 1])),
            scalar_bar_args={
            "title": f"{st.session_state.output_label}",
            "width": 0.03  # Reduce scalar bar width (default is 0.05)
            }
        )
        # plotter_ref.add_mesh(mesh_ref, scalars=f"u_{output_idx}", show_edges=True, color="lightblue", cmap="viridis", 
        #                 show_scalar_bar=True, clim= (np.min(U_pred_org[:,output_idx-1]), 
        #                                                 np.max(U_pred_org[:,output_idx-1])), 
        #                     scalar_bar_args={"title": f"{st.session_state.output_label}"})
        # plotter.add_axes()         
        plotter_ref.view_isometric()            
        # Display the mesh using stpyvista Plotter
        # stpyvista(plotter_ref, key="pv_cube_ref")

        

        st.title("Results")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Test dataset - Ansys")
            # plotter1 = create_plotter("red")
            # stpyvista.render_plot(plotter_ref)
            stpyvista(plotter_ref, key="pv_cube_ref")
        
        with col2:
            st.write(f"### {st.session_state.model_type} prediction")
            # plotter2 = create_plotter("blue")
            # stpyvista.render_plot(plotter)
            stpyvista(plotter, key="pv_cube")


        st.success("Plotting completed!")
        st.session_state.complete_plot = True

## restart the program if needed
if st.session_state.complete_plot == True:
    
    st.session_state.complete_config_input = False
    st.session_state.complete_data = False
    st.session_state.complete_train = False
    st.session_state.complete_mesh_file = False
    st.session_state.complete_config_plot = False
    st.session_state.complete_plot = False

    if st.button("Restart"):
        st.session_state.complete_config_input = False
        st.session_state.complete_data = False
        st.session_state.complete_train = False
        st.session_state.complete_mesh_file = False
        st.session_state.complete_config_plot = False
        st.session_state.complete_plot = False


                    
            
                






        

            






