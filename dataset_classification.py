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
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import v2
import torch

def one_hot(labels, num_classes):
    # Initialize a zero array with shape (ndata, nclass)
    one_hot = np.zeros((len(labels), num_classes), dtype=int)
    
    # Set the appropriate elements to 1 using numpy indexing
    one_hot[np.arange(len(labels)), labels] = 1
    
    return one_hot


class Data_classification(Dataset):
    def __init__(self, data_name: str, config) -> None:
        if not os.path.exists('data'):
            os.makedirs('data')
        self.data_dir = 'data/'
        self.data_name = data_name
        self.nclass = config['MODEL_PARAM']['nclass']
        self.var = self.nclass
        self.split_ratio = config['MODEL_PARAM']['split_ratio']
        self.bool_normalize = config['MODEL_PARAM']['bool_normalize']

        data_file = self.data_dir + data_name + '.csv'
        try:
            data = np.loadtxt(data_file, delimiter=",", dtype=np.float32, skiprows=1)
        except: 
            print(F"Data file {data_file} dose not exist. We will create the data.")
            data_generation_classification(data_name, config)
            data = np.loadtxt(data_file, delimiter=",", dtype=np.float32, skiprows=1)

        if 'mnist' in self.data_name:
            self.x_data_org = data[:, 1:]
            self.u_data_org = data[:, 0].astype(np.int32)
        else:
            self.x_data_org = data[:, config['MODEL_PARAM']['input_col']]
            self.u_data_org = data[:, config['MODEL_PARAM']['output_col']].astype(np.int32)
        self.u_data = one_hot(self.u_data_org, self.nclass) # u_data is the one-hot vector (ndata, nclass)
        self.dim = self.x_data_org.shape[1]
        
        
        if self.bool_normalize:    
            self.x_data_minmax = {"min" : self.x_data_org.min(axis=0), "max" : self.x_data_org.max(axis=0)}
            self.x_data = (self.x_data_org - self.x_data_minmax["min"]) / (self.x_data_minmax["max"] - self.x_data_minmax["min"])
        else:
            self.x_data_minmax = {"min" : self.x_data_org.min(axis=0), "max" : self.x_data_org.max(axis=0)}
            self.x_data = self.x_data_org
            
        print('loaded ',len(self.x_data_org),'datapoints from',data_name,'dataset')

    
    def __len__(self):
        return len(self.x_data_org)

    def __getitem__(self, idx):
        # image = np.array(self.dataframe.loc[idx, self.dataframe.columns != 'label'].values)
        # target = np.array(self.dataframe.loc[idx, 'label'])
        # image = image.reshape((28,28))
        # image = torch.tensor(image).unsqueeze(0)
        
        # image = image.reshape((28,28,1))
        # return np.array(image), np.array(target)
        return self.x_data[idx], self.u_data[idx]
    
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
        