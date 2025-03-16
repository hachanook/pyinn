from kan import KAN, create_dataset
# from pyinn import dataset_regression, train
import dataset_classification, dataset_regression, model, train, plot # for debugging
import torch
import os, yaml


# %% User Set up
with open('./pyinn/settings.yaml','r') as file:
    settings = yaml.safe_load(file)

gpu_idx = settings['GPU']['gpu_idx']  # set which GPU to run on Athena
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # GPU indexing
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)  # GPU indexing

run_type = settings['PROBLEM']["run_type"]
interp_method = settings['PROBLEM']["interp_method"]
data_name = settings['DATA']["data_name"]

with open(f'./config/{data_name}.yaml','r') as file_dataConfig:
    config = yaml.safe_load(file_dataConfig)
    config['interp_method'] = settings['PROBLEM']["interp_method"]
    config['TD_type'] = settings['PROBLEM']["TD_type"]
    

# --------------------- Regression --------------------------
Ms = [50,100,200,300,400] # neurons
if run_type == "regression":
    
    ## data import
    data = dataset_regression.Data_regression(data_name, config)

    ## Prepare KAN dataset
    regressor = train.Regression_INN(data, config)
    regressor.data_split()
    train_input, train_label = [], []
    for inputs, outputs in regressor.train_dataloader:
        train_input.append(inputs)   # Store batch of inputs
        train_label.append(outputs)  # Store batch of outputs
    test_input, test_label = [], []
    for inputs, outputs in regressor.test_dataloader:
        test_input.append(inputs)   # Store batch of inputs
        test_label.append(outputs)  # Store batch of outputs
    dataset = {}
    dataset['train_input'], dataset['train_label'] = torch.cat(train_input, dim=0), torch.cat(train_label, dim=0)
    dataset['test_input'], dataset['test_label'] = torch.cat(test_input, dim=0), torch.cat(test_label, dim=0)

    for M in Ms:
        ## Define hyperparameters
        L = 3 # layers
        # M = 50 # neurons
        J = 10 # grid points
        I,V = 6,1 # inputs/outputs
        k = 3 # polynomial order
        print(f"------------KAN -------------")
        print(f"# of training parameters: {(J+k-1)*(M*I+M**2*(L-1)+M*V)}")
        
        ## Train KAN
        ### Set the device to GPU
        # torch.cuda.set_device(gpu_idx)
        # device = torch.device(f'cuda:{gpu_idx}')
        # print(f"Using GPU {gpu_idx}: {torch.cuda.get_device_name(gpu_idx)}")
        
        model = KAN(width=[I,M,M,M,V], grid=J, k=k, seed=1) # , device=device
        # model.to(device)
        model.fit(dataset, opt='LBFGS', steps=regressor.num_epochs, batch=regressor.batch_size)

 