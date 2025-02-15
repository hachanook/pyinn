"""
INN trainer
----------------------------------------------------------------------------------
Copyright (C) 2024  Chanwook Park
 Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""

# import jax
# import jax.numpy as jnp
import numpy as np
import os, sys, csv
import pandas as pd
from typing import Sequence
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, random_split

def one_hot(labels, num_classes):
    # Initialize a zero array with shape (ndata, nclass)
    one_hot = np.zeros((len(labels), num_classes), dtype=int)
    
    # Set the appropriate elements to 1 using numpy indexing
    one_hot[np.arange(len(labels)), np.squeeze(labels)] = 1
    
    return one_hot

class Data_classification(Dataset):
    def __init__(self, data_name: str, config: dict, *args: list) -> None:
        """
        --- inputs ---
        data_name: name of the dataset. Used only when the dataset is generated internally.
        config: configuration file stored in a dictionary. Check /config
        *args[0]: list of imported dataset. This can be: 1) entire dataset with split ratio specified in the config file, 
                    2) [train, test] dataset, 3) [train, validation, test] dataset
        """

        if not os.path.exists('data'):
            os.makedirs('data')
        self.data_dir = 'data/'
        self.data_name = data_name
        
        self.bool_data_generation = config['DATA_PARAM']['bool_data_generation']
        self.bool_normalize = config['DATA_PARAM']['bool_normalize']
        self.bool_image = config['DATA_PARAM']['bool_image']
        self.bool_shuffle = config['DATA_PARAM']['bool_shuffle']
        self.batch_size = config['TRAIN_PARAM']['batch_size']

        
        ################## Data loading #################
        if self.bool_data_generation == True: # if we use data generator
            data_file = self.data_dir + data_name + '.csv'

            ## Data load or generate
            try:
                data = np.loadtxt(data_file, delimiter=",", dtype=np.float32, skiprows=1)
            except: 
                print(F"Data file {data_file} dose not exist. We will create the data.")
                data_generation_classification(data_name, config)
                data = np.loadtxt(data_file, delimiter=",", dtype=np.float32, skiprows=1)
            ndata = len(data)

            ## shuffle
            if self.bool_shuffle:
                indices = np.arange(ndata)
                np.random.shuffle(indices)
                data = data[indices]

            ## split
            split_ratio = config['DATA_PARAM']['split_ratio']
            if all(isinstance(item, float) for item in split_ratio): # split with a split ratio
                if len(split_ratio) == 2: # train & test
                    train_end = int(split_ratio[0] * ndata)
                    data_train = data[:train_end]
                    data_val = data[train_end:]
                    data_test = data[train_end:]
                
                elif len(split_ratio) == 3: # split with a fixed number of data
                    train_end = int(split_ratio[0] * ndata) 
                    val_end = train_end + int(split_ratio[1] * ndata)
                    data_train = data[:train_end]
                    data_val = data[train_end:val_end]
                    data_test = data[val_end:]

                else:
                    print(f"Error took place while generating data. Check split ratio")
                    sys.exit()
            
            elif all(isinstance(item, int) for item in split_ratio):
                if len(split_ratio) == 2: # train & test
                    train_end = split_ratio[0]
                    data_train = data[:train_end]
                    data_val = data[train_end:]
                    data_test = data[train_end:]
                
                elif len(split_ratio) == 3: # split with a fixed number of data
                    train_end = split_ratio[0]
                    val_end = train_end + split_ratio[1]
                    data_train = data[:train_end]
                    data_val = data[train_end:val_end]
                    data_test = data[val_end:]

                else:
                    print(f"Error took place while generating data. Check split ratio")
                    sys.exit()

        else:
            print("To be updated. Importing classification data")
            import sys
            sys.exit()


        ################# Data loading end ################
            
        
        ## divide into input and output data
        if 'mnist' in self.data_name: # for mnist data where input_col is too long
            self.input_col = np.arange(1,data.shape[1])
            self.output_col = [0]
        else:
            self.input_col = config['DATA_PARAM']['input_col']
            self.output_col = config['DATA_PARAM']['output_col']
        self.nclass = config['DATA_PARAM']['nclass']
        self.dim = len(self.input_col) # size of input
        self.var = self.nclass

        self.x_data_org = data[:, self.input_col]
        self.u_data_org = data[:, self.output_col].astype(np.int32)
        x_data_train_org = data_train[:, self.input_col]
        u_data_train_org = data_train[:, self.output_col].astype(np.int32)
        x_data_val_org = data_val[:, self.input_col]
        u_data_val_org = data_val[:, self.output_col].astype(np.int32)
        x_data_test_org = data_test[:, self.input_col]
        u_data_test_org = data_test[:, self.output_col].astype(np.int32)
        
        ## normalize data
        if self.bool_image: # when image is provided with 0~255 integer pixel values
            self.x_data_minmax = {"min" : np.zeros(self.x_data_org.shape[1], dtype=np.float64), 
                                "max" : np.ones(self.x_data_org.shape[1], dtype=np.float64) * np.max(self.x_data_org)}
        else: # general classification data like the spiral classification
            self.x_data_minmax = {"min" : self.x_data_org.min(axis=0), "max" : self.x_data_org.max(axis=0)}
        if self.bool_normalize:    
            self.x_data_train = (x_data_train_org - self.x_data_minmax["min"]) / (self.x_data_minmax["max"] - self.x_data_minmax["min"])
            self.x_data_val = (x_data_val_org - self.x_data_minmax["min"]) / (self.x_data_minmax["max"] - self.x_data_minmax["min"])
            self.x_data_test = (x_data_test_org - self.x_data_minmax["min"]) / (self.x_data_minmax["max"] - self.x_data_minmax["min"])    
        else: 
            self.x_data_train = x_data_train_org
            self.x_data_val = x_data_val_org
            self.x_data_test = x_data_test_org
        self.u_data_train = one_hot(u_data_train_org, self.nclass) # u_data is the one-hot vector (ndata, nclass)
        self.u_data_val = one_hot(u_data_val_org, self.nclass) # u_data is the one-hot vector (ndata, nclass)
        self.u_data_test = one_hot(u_data_test_org, self.nclass) # u_data is the one-hot vector (ndata, nclass)
        
        ## Create TensorDatasets
        train_dataset = TensorDataset(torch.tensor(self.x_data_train), torch.tensor(self.u_data_train))
        val_dataset = TensorDataset(torch.tensor(self.x_data_val), torch.tensor(self.u_data_val))
        test_dataset = TensorDataset(torch.tensor(self.x_data_test), torch.tensor(self.u_data_test))

        ## Define Dataloaders
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.bool_shuffle)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=self.bool_shuffle)
        self.test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.bool_shuffle)


        print(f'loaded {ndata} datapoints from {data_name} dataset')
            
    
    def __len__(self):
        return len(self.x_data_org)

    
def data_generation_classification(data_name: str, config):

    data_dir = './data'

    if data_name == 'mnist' or data_name == 'fashion_mnist':
        transform = transforms.ToTensor()

        if data_name == 'mnist':
            train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        elif data_name == 'fashion_mnist':
            train_dataset = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
            test_dataset = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)

        # Combine training and test datasets
        combined_data = [(int(label), *image.numpy().astype('float16').flatten()) for image, label in train_dataset]
        combined_data += [(int(label), *image.numpy().astype('float16').flatten()) for image, label in test_dataset]
        
        # Define column names for the header
        columns = ["label"] + [f"pixel{i}" for i in range(784)]
        
        # Convert to DataFrame and save as CSV
        combined_df = pd.DataFrame(combined_data, columns=columns)
        combined_df.to_csv(os.path.join(data_dir, f'{data_name}.csv'), index=False, header=True)

    elif data_name == 'spiral':
        halfSamples = 5000 # number of data points in each class, 5,000 -> total 10,000 data
        noise = 3
        N_SAMPLES = halfSamples * 2
        def genSpiral(deltaT, label, halfSamples, noise):
            points = np.zeros((halfSamples, 3), dtype=np.double)
            for i in range(halfSamples):
                r = i / halfSamples * 5 # radius
                # t = 1.75 * i / halfSamples * 2 * np.pi + deltaT # theta
                t = 3.43 * i / halfSamples * 2 * np.pi + deltaT # theta
                x = r * np.sin(t) + np.random.uniform(-0.1,0.1) * noise
                y = r * np.cos(t) + np.random.uniform(-0.1,0.1) * noise
                points[i] = np.array([x, y, label])
            return points

        points1 =genSpiral(0, 1, halfSamples, noise) # Positive examples
        points2 =genSpiral(np.pi, 0, halfSamples, noise) # Negative examples
        points = np.concatenate((points1, points2), axis=0)
        # shuffle
        indices = np.arange(N_SAMPLES)
        np.random.shuffle(indices)
        points = points[indices,:]
        df = pd.DataFrame(points, columns=['x1', 'x2', 'u'])
        
        # ## Plot ##
        # import matplotlib.pyplot as plt

        # plt.figure(figsize=(6, 5))
        # plt.set_cmap(plt.cm.Paired)
        # # plt.pcolormesh(xx, yy, ynew)
        # plt.scatter(points[:,0], points[:,1], c=points[:,2], edgecolors='black')

        # plt.xlabel(r'$p_1$', fontsize = 20)
        # plt.ylabel(r'$p_2$', fontsize = 20)
        # plt.xticks(fontsize=14)
        # plt.yticks(fontsize=14)
        
        # parent_dir = os.path.abspath(os.getcwd())
        # path_figure = os.path.join(parent_dir, 'plots')
        # plt.savefig(os.path.join(path_figure, 'dataset_spiral') , dpi=300)
        # plt.close()


        # Save the DataFrame to a CSV file
        df.to_csv(os.path.join(data_dir, f'{data_name}.csv'), index=False)

        