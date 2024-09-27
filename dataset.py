"""
INN trainer
----------------------------------------------------------------------------------
Copyright (C) 2024  Chanwook Park
 Northwestern University, Evanston, Illinois, US, chanwookpark2024@u.northwestern.edu
"""

import jax
import jax.numpy as jnp
import numpy as np
import os, sys, csv
import pandas as pd
from typing import Sequence
from torch.utils.data import Dataset
from torchvision.transforms import v2
import torch


class Data_regression(Dataset):
    def __init__(self, data_name: str, data_size: int, input_col: Sequence[int], output_col: Sequence[int], 
                 split_ratio: Sequence[int | float]=[0.7,0.15,0.15], bool_normalize=True) -> None:
        self.root_dir = 'data/'
        self.data_name = data_name
        self.data_size = data_size
        self.dim = len(input_col) # size of input
        self.var = len(output_col) # size of output
        self.split_ratio = split_ratio
        # self.bool_shuffle = bool_shuffle
        # self.bool_normalize = bool_normalize
        # self.transform = transform
        
        data_file = self.root_dir + data_name + '_' + str(data_size) + '.csv'
        try:
            data = np.loadtxt(data_file, delimiter=",", dtype=np.float64, skiprows=1)
        except: 
            print(F"Data file {data_file} dose not exist. We will create the data.")
            data_generation(data_name, data_size, input_col)
            data = np.loadtxt(data_file, delimiter=",", dtype=np.float64, skiprows=1)
        
        self.x_data_org = data[:, input_col]
        self.u_data_org = data[:, output_col]
        
        if bool_normalize:    
            self.x_data_minmax = {"min" : self.x_data_org.min(axis=0), "max" : self.x_data_org.max(axis=0)}
            self.u_data_minmax = {"min" : self.u_data_org.min(axis=0), "max" : self.u_data_org.max(axis=0)}
            self.x_data = (self.x_data_org - self.x_data_minmax["min"]) / (self.x_data_minmax["max"] - self.x_data_minmax["min"])
            self.u_data = (self.u_data_org - self.u_data_minmax["min"]) / (self.u_data_minmax["max"] - self.u_data_minmax["min"])
        else:
            self.x_data = self.x_data_org
            self.u_data = self.u_data_org
            
        print('loaded ',len(self.x_data_org),'datapoints from',data_name,'dataset')
        
    def __len__(self):
        return len(self.x_data_org)

    def __getitem__(self, idx):
        return self.x_data[idx], self.u_data[idx]
    

def data_generation(data_name: str, data_size: int, input_col: Sequence[int]):

    x_data_org = np.random.rand(data_size, len(input_col))
        
    if data_name == "1D_1D_sine":
        u_data_org = np.sin(2*np.pi*x_data_org)
        cols = ['x1', 'u']
    elif data_name == "1D_1D_exp":
        u_data_org = np.exp(4*x_data_org**2 - 2*x_data_org -1)
        cols = ['x1', 'u']
    elif data_name == "1D_2D_sine_exp":
        u1_data_org = np.sin(np.pi* x_data_org)
        u2_data_org = np.exp(x_data_org**2 - x_data_org -1)
        u_data_org = np.concatenate((u1_data_org, u2_data_org), axis=1)
        cols = ['x1','u1','u2']
    elif data_name == "2D_1D_sine":
        u_data_org = np.sin(x_data_org[:,[0]]) * np.cos(x_data_org[:,[1]])
        cols = ['x1','x2','u']
    elif data_name == "2D_1D_exp":
        u_data_org = np.exp(x_data_org[:,[0]] + 2*x_data_org[:,[1]])
        cols = ['x1','x2','u']
    elif data_name == "8D_1D_physics": # borehole function; use JAX to create data

        x_min = jnp.array([0.05,    100,  63_070,  990, 63.1, 700, 1120,  9_855], dtype=jnp.double)
        x_max = jnp.array([0.15, 50_000, 115_600, 1110,  116, 820, 1680, 12_045], dtype=jnp.double)
        x_data_org = jnp.array(x_data_org) * (x_max-x_min) + x_min
        def borehole(p):
            p1, p2, p3, p4, p5, p6, p7, p8 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
            u = 2*jnp.pi* p1 * (p4-p6) * (jnp.log(p2/p3) * (1 + 2*(p7*p1) / (jnp.log(p2/p3)*p3**2*p8) + p1/p5))**(-1)
            return jnp.array([u], dtype=jnp.double)
        u_data_org = jax.vmap(borehole)(x_data_org)
        cols = ['x1','x2','x3','x4','x5','x6','x7','x8','u']

    elif data_name == "10D_5D_physics": # five physics function; use JAX to create data

        ## u1: Borehole function
        x_min = jnp.array([0.05,    100,  63_070,  990, 63.1, 700, 1120,  9_855], dtype=jnp.double)
        x_max = jnp.array([0.15, 50_000, 115_600, 1110,  116, 820, 1680, 12_045], dtype=jnp.double)
        x1_data_org = jnp.array(x_data_org[:,:8]) * (x_max-x_min) + x_min
        def borehole(p):
            p1, p2, p3, p4, p5, p6, p7, p8 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
            u = 2*jnp.pi* p1 * (p4-p6) * (jnp.log(p2/p3) * (1 + 2*(p7*p1) / (jnp.log(p2/p3)*p3**2*p8) + p1/p5))**(-1)
            return jnp.array([u], dtype=jnp.double)
        u1_data_org = jax.vmap(borehole)(x1_data_org)

        ## u2: Piston simulation function
        x_min = jnp.array([30, 0.005, 0.002, 1000, 90_000, 290, 340], dtype=jnp.double)
        x_max = jnp.array([60, 0.020, 0.010, 5000,110_000, 296, 360], dtype=jnp.double)
        x2_data_org = jnp.array(x_data_org[:,:7]) * (x_max-x_min) + x_min
        def piston(p):
            p1, p2, p3, p4, p5, p6, p7 = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
            A = p5*p2 + 19.62*p1 - p4*p3/p2
            V = p2/(2*p4)*((A**2 + 4*p4*p5*p3*p6/p7)**0.5 - A)
            u = 2*jnp.pi* (p1/(p4 + p2**2*p5*p3*p6/p7/V**2))**0.5
            return jnp.array([u], dtype=jnp.double)
        u2_data_org = jax.vmap(piston)(x2_data_org)

        ## u3: OTL circuit function
        x_min = jnp.array([50, 25, 0.5, 1.2, 0.25, 50], dtype=jnp.double)
        x_max = jnp.array([150,70, 3.0, 2.5, 1.20,300], dtype=jnp.double)
        x3_data_org = jnp.array(x_data_org[:,:6]) * (x_max-x_min) + x_min
        def OTL(p):
            p1, p2, p3, p4, p5, p6 = p[0], p[1], p[2], p[3], p[4], p[5]
            u = ( ((12*p2/(p1+p2) + 0.74) * p6*(p5+9) + 11.35*p3)/(p6*(p5+9)+p3) 
                + (0.74*p3*p6*(p5+9))/((p6*(p5+9) + p3)*p4) )
            return jnp.array([u], dtype=jnp.double)
        u3_data_org = jax.vmap(OTL)(x3_data_org)

        ## u4: Robot arm function
        x_min = jnp.array([     0.0,      0.0,      0.0,      0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.double)
        x_max = jnp.array([2*jnp.pi, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi, 1.0, 1.0, 1.0, 1.0], dtype=jnp.double)
        x4_data_org = jnp.array(x_data_org[:,:8]) * (x_max-x_min) + x_min
        def robot(p):
            x, y = 0, 0
            for i in range(4):
                angle = 0
                for j in range(i+1):
                    angle += p[j]
                x += p[i+4] * jnp.cos(angle)
                y += p[i+4] * jnp.sin(angle)
            u = (x**2 + y**2)**0.5
            return jnp.array([u], dtype=jnp.double)
        u4_data_org = jax.vmap(robot)(x4_data_org)

        ## u5: Wing weight function
        x_min = jnp.array([150, 220, 6,-10*jnp.pi/180, 16, 0.5, 0.08, 2.5, 1700, 0.025], dtype=jnp.double)
        x_max = jnp.array([200, 300,10, 10*jnp.pi/180, 45, 1.0, 0.18, 6.0, 2500, 0.080], dtype=jnp.double)
        x5_data_org = jnp.array(x_data_org[:,:10]) * (x_max-x_min) + x_min
        def wing(p):
            p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
            u = ( 0.036*p1**0.758*p2**0.0035 * (p3/(jnp.cos(p4))**2)**0.6 * p5**0.006*p6**0.04 * (100*p7/jnp.cos(p4))**(-0.3)
                * (p8*p9)**0.49 + p1*p10 )
            return jnp.array([u], dtype=jnp.double)
        u5_data_org = jax.vmap(wing)(x5_data_org)

        u_data_org = jnp.concatenate((u1_data_org, u2_data_org, u3_data_org, u4_data_org, u5_data_org), axis=1) # save normalized p_data
        cols=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'u1', 'u2', 'u3', 'u4', 'u5']

    data = np.concatenate((x_data_org, u_data_org), axis=1)
    df = pd.DataFrame(data, columns=cols)

    # Save the DataFrame to a CSV file
    parent_dir = os.path.abspath(os.getcwd())
    path_data = os.path.join(parent_dir, 'data')
    csv_filename = f"{data_name}_{data_size}.csv" # 1D input, 1D output, sine curve
    df.to_csv(os.path.join(path_data, csv_filename), index=False)


