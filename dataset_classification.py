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
# from torchvision.transforms import v2
import torch

def one_hot(labels, num_classes):
    # Initialize a zero array with shape (ndata, nclass)
    one_hot = np.zeros((len(labels), num_classes), dtype=int)
    
    # Set the appropriate elements to 1 using numpy indexing
    one_hot[np.arange(len(labels)), np.squeeze(labels)] = 1
    
    return one_hot

class Data_classification(Dataset):
    def __init__(self, data_name: str, config) -> None:
        if not os.path.exists('data'):
            os.makedirs('data')
        self.data_dir = 'data/'
        self.data_name = data_name
        self.nclass = config['DATA_PARAM']['nclass']
        self.var = self.nclass
        self.split_ratio = config['DATA_PARAM']['split_ratio']
        self.bool_normalize = config['DATA_PARAM']['bool_normalize']
        self.bool_image = config['DATA_PARAM']['bool_image']

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
            self.x_data_org = data[:, config['DATA_PARAM']['input_col']]
            self.u_data_org = data[:, config['DATA_PARAM']['output_col']].astype(np.int32)
        self.u_data = one_hot(self.u_data_org, self.nclass) # u_data is the one-hot vector (ndata, nclass)
        self.dim = self.x_data_org.shape[1]
        
        
        if self.bool_normalize:
            if self.bool_image: # when image is provided with 0~255 integer pixel values
                self.x_data_minmax = {"min" : np.zeros(self.x_data_org.shape[1], dtype=np.int32), 
                                      "max" : np.ones(self.x_data_org.shape[1], dtype=np.int32) * 255}
                self.x_data = (self.x_data_org - self.x_data_minmax["min"]) / (self.x_data_minmax["max"] - self.x_data_minmax["min"])

            elif self.bool_image == False: # spiral classification
                self.x_data_minmax = {"min" : self.x_data_org.min(axis=0), "max" : self.x_data_org.max(axis=0)}
                self.x_data = (self.x_data_org - self.x_data_minmax["min"]) / (self.x_data_minmax["max"] - self.x_data_minmax["min"])
        
        else:
            if self.bool_image: # when image is provided with 0~1 floating pixel values; our MNIST dataset
                self.x_data_minmax = {"min" : np.zeros(self.x_data_org.shape[1], dtype=np.float64), 
                                      "max" : np.ones(self.x_data_org.shape[1], dtype=np.float64)}
                self.x_data = self.x_data_org
                
            elif self.bool_image == False: # 
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

        