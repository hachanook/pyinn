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
from scipy.stats import qmc


class Data_regression(Dataset):
    def __init__(self, data_name: str, config) -> None:
        if not os.path.exists('data'):
            os.makedirs('data')
        self.data_dir = 'data/'
        self.data_name = data_name
        self.data_size = config['DATA_PARAM']['data_size']
        self.input_col = config['DATA_PARAM']['input_col']
        self.output_col = config['DATA_PARAM']['output_col']
        self.dim = len(self.input_col) # size of input
        self.var = len(self.output_col) # size of output
        self.split_ratio = config['DATA_PARAM']['split_ratio']
        self.bool_normalize = config['DATA_PARAM']['bool_normalize']
        
        data_file = self.data_dir + data_name + '_' + str(self.data_size) + '.csv'
        try:
            data = np.loadtxt(data_file, delimiter=",", dtype=np.float64, skiprows=1)
        except: 
            print(F"Data file {data_file} dose not exist. We will create the data.")
            data_generation_regression(data_name, self.data_size, self.input_col)
            data = np.loadtxt(data_file, delimiter=",", dtype=np.float64, skiprows=1)
        
        self.x_data_org = data[:, self.input_col]
        self.u_data_org = data[:, self.output_col]
        
        if self.bool_normalize:    
            self.x_data_minmax = {"min" : self.x_data_org.min(axis=0), "max" : self.x_data_org.max(axis=0)}
            self.u_data_minmax = {"min" : self.u_data_org.min(axis=0), "max" : self.u_data_org.max(axis=0)}
            self.x_data = (self.x_data_org - self.x_data_minmax["min"]) / (self.x_data_minmax["max"] - self.x_data_minmax["min"])
            self.u_data = (self.u_data_org - self.u_data_minmax["min"]) / (self.u_data_minmax["max"] - self.u_data_minmax["min"])
        else:
            self.x_data_minmax = {"min" : self.x_data_org.min(axis=0), "max" : self.x_data_org.max(axis=0)}
            self.u_data_minmax = {"min" : self.u_data_org.min(axis=0), "max" : self.u_data_org.max(axis=0)}
            self.x_data = self.x_data_org
            self.u_data = self.u_data_org
            
        print('loaded ',len(self.x_data_org),'datapoints from',data_name,'dataset')
        
    def __len__(self):
        return len(self.x_data_org)

    def __getitem__(self, idx):
        return self.x_data[idx], self.u_data[idx]
    
    def denormalize(self, x_data=None, u_data=None):
        """ Denormalize both x_data and u_data
        x_data: (ndata, I)
        u_data: (ndata, L)
        """
        if x_data is not None:
            x_data_org = (self.x_data_minmax["max"] - self.x_data_minmax["min"]) * x_data + self.x_data_minmax["min"]
        if u_data is not None:
            u_data_org = (self.u_data_minmax["max"] - self.u_data_minmax["min"]) * u_data + self.u_data_minmax["min"]
        return x_data, u_data
    

def data_generation_regression(data_name: str, data_size: int, input_col: Sequence[int]):

    ## random sampling
    # x_data_org = jnp.array(np.random.rand(data_size, len(input_col)))

    ## Latin Hypercube sampling
    x_data_sampler = qmc.LatinHypercube(d=len(input_col))
    x_data_org = x_data_sampler.random(n=data_size)
        
    if data_name == "1D_1D_sine":
        u_data_org = v_fun_1D_1D_sine(x_data_org)
        cols = ['x1', 'u']

    elif data_name == "1D_1D_exp":
        u_data_org = v_fun_1D_1D_exp(x_data_org)
        cols = ['x1', 'u']

    elif data_name == "1D_2D_sine_exp":
        u_data_org = v_fun_1D_2D_sine_exp(x_data_org)
        cols = ['x1','u1','u2']

    elif data_name == "2D_1D_sine":
        u_data_org = v_fun_2D_1D_sine(x_data_org)
        cols = ['x1','x2','u']

    elif data_name == "2D_1D_exp":
        u_data_org = v_fun_2D_1D_exp(x_data_org)
        cols = ['x1','x2','u']

    elif data_name == "3D_1D_exp":
        u_data_org = v_fun_3D_1D_exp(x_data_org)
        cols = ['x1','x2','x3','u']

    elif data_name == "8D_1D_physics": # borehole function; use JAX to create data

        x_min = jnp.array([0.05,    100,  63_070,  990, 63.1, 700, 1120,  9_855], dtype=jnp.double)
        x_max = jnp.array([0.15, 50_000, 115_600, 1110,  116, 820, 1680, 12_045], dtype=jnp.double)
        x_data_org = x_data_org * (x_max-x_min) + x_min
        
        u_data_org = jax.vmap(fun_8D_1D_physics)(x_data_org)
        cols = ['x1','x2','x3','x4','x5','x6','x7','x8','u']

    elif data_name == "10D_5D_physics": # five physics function; use JAX to create data

        ## u1: Borehole function
        x_min = jnp.array([0.05,    100,  63_070,  990, 63.1, 700, 1120,  9_855], dtype=jnp.double)
        x_max = jnp.array([0.15, 50_000, 115_600, 1110,  116, 820, 1680, 12_045], dtype=jnp.double)
        x1_data_org = x_data_org[:,:8] * (x_max-x_min) + x_min
        
        ## u2: Piston simulation function
        x_min = jnp.array([30, 0.005, 0.002, 1000, 90_000, 290, 340], dtype=jnp.double)
        x_max = jnp.array([60, 0.020, 0.010, 5000,110_000, 296, 360], dtype=jnp.double)
        x2_data_org = x_data_org[:,:7] * (x_max-x_min) + x_min
        
        ## u3: OTL circuit function
        x_min = jnp.array([50, 25, 0.5, 1.2, 0.25, 50], dtype=jnp.double)
        x_max = jnp.array([150,70, 3.0, 2.5, 1.20,300], dtype=jnp.double)
        x3_data_org = x_data_org[:,:6] * (x_max-x_min) + x_min
        
        ## u4: Robot arm function
        x_min = jnp.array([     0.0,      0.0,      0.0,      0.0, 0.0, 0.0, 0.0, 0.0], dtype=jnp.double)
        x_max = jnp.array([2*jnp.pi, 2*jnp.pi, 2*jnp.pi, 2*jnp.pi, 1.0, 1.0, 1.0, 1.0], dtype=jnp.double)
        x4_data_org = x_data_org[:,:8] * (x_max-x_min) + x_min
        
        ## u5: Wing weight function
        x_min = jnp.array([150, 220, 6,-10*jnp.pi/180, 16, 0.5, 0.08, 2.5, 1700, 0.025], dtype=jnp.double)
        x_max = jnp.array([200, 300,10, 10*jnp.pi/180, 45, 1.0, 0.18, 6.0, 2500, 0.080], dtype=jnp.double)
        x5_data_org = x_data_org[:,:10] * (x_max-x_min) + x_min
        

        u_data_org = jax.vmap(fun_10D_5D_physics, in_axes=(0,0,0,0,0))(x1_data_org, x2_data_org, x3_data_org, x4_data_org, x5_data_org)
        cols=['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'u1', 'u2', 'u3', 'u4', 'u5']

    
    elif data_name == "IGAMapping2D":

        ## define IGA parameters
        
        x_min = jnp.array([0, 0], dtype=jnp.double)
        x_max = jnp.array([10, 10], dtype=jnp.double)
        
        x_data_org = x_data_org * (x_max-x_min) + x_min
        u_data_org = v_fun_IGAMapping2D(x_data_org)
        cols = ['x1','x2','u1','u2']


    data = np.concatenate((x_data_org, u_data_org), axis=1)
    df = pd.DataFrame(data, columns=cols)

    # Save the DataFrame to a CSV file
    parent_dir = os.path.abspath(os.getcwd())
    path_data = os.path.join(parent_dir, 'data')
    csv_filename = f"{data_name}_{data_size}.csv" # 1D input, 1D output, sine curve
    df.to_csv(os.path.join(path_data, csv_filename), index=False)




## define functions
def fun_1D_1D_exp(x_data_org):
    u_data_org =  jnp.exp(4*x_data_org**2 - 2*x_data_org - 1)
    return u_data_org.reshape(1,)
v_fun_1D_1D_exp = jax.vmap(fun_1D_1D_exp, in_axes = (0)) # output: (ndata, )
vv_fun_1D_1D_exp = jax.vmap(v_fun_1D_1D_exp, in_axes = (0)) # output: (ndata, ndata)


def fun_1D_1D_sine(x_data_org):
    u_data_org =  jnp.sin(2*jnp.pi*x_data_org)
    return u_data_org.reshape(1,)
v_fun_1D_1D_sine = jax.vmap(fun_1D_1D_sine, in_axes = (0)) # output: (ndata, )
vv_fun_1D_1D_sine = jax.vmap(v_fun_1D_1D_sine, in_axes = (0)) # output: (ndata, ndata)

def fun_1D_2D_sine_exp(x_data_org):
    u1 = jnp.sin(2*jnp.pi*x_data_org)
    u2 = jnp.exp(4*x_data_org**2 - 2*x_data_org - 1)
    return jnp.array([u1,u2], dtype=jnp.double).reshape(-1)
v_fun_1D_2D_sine_exp = jax.vmap(fun_1D_2D_sine_exp, in_axes = (0)) # output: (ndata, )
vv_fun_1D_2D_sine_exp = jax.vmap(v_fun_1D_2D_sine_exp, in_axes = (0)) # output: (ndata, ndata)

def fun_2D_1D_sine(x_data_org):
    u_data_org =  jnp.sin(x_data_org[0]) * jnp.cos(x_data_org[1])
    return u_data_org.reshape(1,)
v_fun_2D_1D_sine = jax.vmap(fun_2D_1D_sine, in_axes = (0)) # output: (ndata, )
vv_fun_2D_1D_sine = jax.vmap(v_fun_2D_1D_sine, in_axes = (0)) # output: (ndata, ndata)

def fun_2D_1D_exp(x_data_org):
    u_data_org =  jnp.exp(x_data_org[0] + 2*x_data_org[1])
    return u_data_org.reshape(1,)
v_fun_2D_1D_exp = jax.vmap(fun_2D_1D_exp, in_axes = (0)) # output: (ndata, )
vv_fun_2D_1D_exp = jax.vmap(v_fun_2D_1D_exp, in_axes = (0)) # output: (ndata, ndata)

def fun_3D_1D_exp(x_data_org):
    u_data_org =  (2*x_data_org[2]*jnp.sin(x_data_org[1]) - 3*x_data_org[0]) / jnp.exp(x_data_org[0] - x_data_org[1]**2)
    return u_data_org.reshape(1,)
v_fun_3D_1D_exp = jax.vmap(fun_3D_1D_exp, in_axes = (0)) # output: (ndata, )
vv_fun_3D_1D_exp = jax.vmap(v_fun_3D_1D_exp, in_axes = (0)) # output: (ndata, ndata)


def fun_8D_1D_physics(p): 
    p1, p2, p3, p4, p5, p6, p7, p8 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]

    ## u1: Borehole function
    u = 2*jnp.pi* p1 * (p4-p6) * (jnp.log(p2/p3) * (1 + 2*(p7*p1) / (jnp.log(p2/p3)*p3**2*p8) + p1/p5))**(-1)
    return jnp.array([u], dtype=jnp.double)

def fun_10D_5D_physics(x1,x2,x3,x4,x5): 
    
    ## u1: Borehole function
    p=x1
    p1, p2, p3, p4, p5, p6, p7, p8 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
    u1 = 2*jnp.pi* p1 * (p4-p6) * (jnp.log(p2/p3) * (1 + 2*(p7*p1) / (jnp.log(p2/p3)*p3**2*p8) + p1/p5))**(-1)

    ## u2: Piston simulation function
    p=x2
    p1, p2, p3, p4, p5, p6, p7 = p[0], p[1], p[2], p[3], p[4], p[5], p[6]
    A = p5*p2 + 19.62*p1 - p4*p3/p2
    V = p2/(2*p4)*((A**2 + 4*p4*p5*p3*p6/p7)**0.5 - A)
    u2 = 2*jnp.pi* (p1/(p4 + p2**2*p5*p3*p6/p7/V**2))**0.5

    ## u3: OTL circuit function
    p=x3
    p1, p2, p3, p4, p5, p6 = p[0], p[1], p[2], p[3], p[4], p[5]
    u3 = ( ((12*p2/(p1+p2) + 0.74) * p6*(p5+9) + 11.35*p3)/(p6*(p5+9)+p3) 
                + (0.74*p3*p6*(p5+9))/((p6*(p5+9) + p3)*p4) )
    
    ## u4: Robot arm function
    p=x4
    p1, p2, p3, p4, p5, p6, p7, p8 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]
    x, y = 0, 0
    for i in range(4):
        angle = 0
        for j in range(i+1):
            angle += p[j]
        x += p[i+4] * jnp.cos(angle)
        y += p[i+4] * jnp.sin(angle)
    u4 = (x**2 + y**2)**0.5

    ## u5: Wing weight function
    p=x5
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10 = p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9]
    u5 = ( 0.036*p1**0.758*p2**0.0035 * (p3/(jnp.cos(p4))**2)**0.6 * p5**0.006*p6**0.04 * (100*p7/jnp.cos(p4))**(-0.3)
        * (p8*p9)**0.49 + p1*p10 )
    
    return jnp.array([u1,u2,u3,u4,u5], dtype=jnp.double)


## IGA mapping

def in_range(xi, lb, ub):
    # lb: lower bound, floating number
    # ub: upper bound, floating number
    return jnp.heaviside(xi-lb,1) * jnp.heaviside(ub-xi, 0)

def NBasis(xi, L):
    xi /= L
    N1 = (1-xi)**2
    N2 = 2*xi*(1-xi)
    N3 = xi**2
    N_all = jnp.array([N1, N2, N3], dtype=jnp.float64)
    return N_all

def MBasis(eta, L):
    eta /= L
    M1 = (1-eta)**2
    M2 = 2*eta*(1-eta)
    M3 = eta**2
    M_all = jnp.array([M1, M2, M3], dtype=jnp.float64)
    return M_all

def Sum_fun(xieta, L, weights):
    xi, eta = xieta[0], xieta[1]
    N_all = NBasis(xi, L)
    M_all = MBasis(eta, L)
    NM_all = jnp.tensordot(N_all, M_all, axes=0) # (4,3)
    Sum = jnp.sum(NM_all * weights)
    return Sum

def fun_IGAMapping2D(xieta):
    # This basic mapping is nothing but the identity mapping
    # R_all * controlPts returns xi itselt.

    ## IGA parameters
    L = 10
    controlPts = np.zeros((3,3,2), dtype=np.double)
                
    controlPts[:,:,0] = np.array([[0,   0,  0],
                                    [10, 15, 20],
                                    [10, 15, 20]], dtype=np.float64)
    controlPts[:,:,1]= np.array([[10, 15, 20],
                                    [10, 15, 20],
                                    [0,   0,  0]], dtype=np.float64)
    controlPts = jnp.array(controlPts)
    weights = jnp.array([[1, 1, 1],
                        [0.5*jnp.sqrt(2), 0.5*jnp.sqrt(2), 0.5*jnp.sqrt(2)],
                        [1, 1, 1]], dtype=jnp.float64)

    xi, eta = xieta[0], xieta[1]
    N_all = NBasis(xi, L)
    M_all = MBasis(eta, L)
    NM_all = jnp.tensordot(N_all, M_all, axes=0) # (4,3)
    Sum = Sum_fun(xieta, L, weights)
    R_all = NM_all * weights / Sum # (4,3)
    xy = jnp.sum(R_all[:,:,None] * controlPts[:,:,:], axis=(0,1))
    return xy
v_fun_IGAMapping2D = jax.vmap(fun_IGAMapping2D, in_axes = (0))
vv_fun_IGAMapping2D = jax.vmap(v_fun_IGAMapping2D, in_axes = (0))
