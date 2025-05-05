import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from jax.experimental.sparse import BCOO
from jax.scipy.optimize import minimize
from functools import partial
from itertools import combinations
jax.config.update("jax_enable_x64", True)



## Gauss rules

def GaussSet(nGP = 2, return_type='jnp'):
    if nGP == 2:
        Gauss_Weight1D = [1, 1]
        Gauss_Point1D = [-1/jnp.sqrt(3), 1/jnp.sqrt(3)]
       
    elif nGP == 3:
        Gauss_Weight1D = [0.55555556, 0.88888889, 0.55555556]
        Gauss_Point1D = [-0.7745966, 0, 0.7745966]
       
        
    elif nGP == 4:
        Gauss_Weight1D = [0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538]
        Gauss_Point1D = [-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]

    elif nGP == 6: # double checked, 16 digits
        Gauss_Weight1D = [0.1713244923791704, 0.3607615730481386, 0.4679139345726910, 
                          0.4679139345726910, 0.3607615730481386, 0.1713244923791704]
        Gauss_Point1D = [-0.9324695142031521, -0.6612093864662645, -0.2386191860831969, 
                         0.2386191860831969, 0.6612093864662645, 0.9324695142031521]

       
    elif nGP == 8: # double checked, 20 digits
        Gauss_Weight1D=[0.10122853629037625915, 0.22238103445337447054, 0.31370664587788728733, 0.36268378337836198296,
                        0.36268378337836198296, 0.31370664587788728733, 0.22238103445337447054,0.10122853629037625915]
        Gauss_Point1D=[-0.960289856497536231684, -0.796666477413626739592,-0.525532409916328985818, -0.183434642495649804939,
                        0.183434642495649804939,  0.525532409916328985818, 0.796666477413626739592,  0.960289856497536231684]
        
    elif nGP == 10:
        Gauss_Weight1D=[0.0666713443086881, 0.1494513491505806, 0.2190863625159820, 0.2692667193099963, 0.2955242247147529,
                        0.2955242247147529, 0.2692667193099963, 0.2190863625159820, 0.1494513491505806, 0.0666713443086881]
        Gauss_Point1D=[-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.1488743389816312,  
                        0.1488743389816312,  0.4333953941292472,  0.6794095682990244,  0.8650633666889845,  0.9739065285171717]
        
    elif nGP == 20:
        Gauss_Weight1D=[0.017614007, 0.04060143, 0.062672048, 0.083276742,0.10193012, 0.118194532,0.131688638,
                        0.142096109, 0.149172986, 0.152753387,0.152753387,0.149172986, 0.142096109, 0.131688638,
                        0.118194532,0.10193012, 0.083276742,0.062672048,0.04060143,0.017614007]
            
        Gauss_Point1D=[-0.993128599, -0.963971927, -0.912234428, -0.839116972, -0.746331906, -0.636053681,
                        -0.510867002, -0.373706089, -0.227785851, -0.076526521, 0.076526521, 0.227785851,
                        0.373706089, 0.510867002, 0.636053681, 0.746331906, 0.839116972, 0.912234428, 0.963971927, 0.993128599]
    
    if return_type == 'jnp':
        return jnp.array(Gauss_Weight1D, dtype=jnp.float64), jnp.array(Gauss_Point1D, dtype=jnp.float64)
    elif return_type == 'list':
        return Gauss_Weight1D, Gauss_Point1D


def get_quad_points(Gauss_num, dim, elem_type):

    # Quad elements
    if elem_type == 'CPE4' or elem_type == 'CPE8' or elem_type.startswith('C3D8') or elem_type.startswith('C3D20') or elem_type.startswith('D1LN2N'):
        Gauss_Weight1D, Gauss_Point1D = GaussSet(Gauss_num, return_type='list')
        quad_points, quad_weights = [], []
    
        for ipoint, iweight in zip(Gauss_Point1D, Gauss_Weight1D):
            
            if dim < 2: # 1D elements
                quad_points.append([ipoint])
                quad_weights.append(iweight)
            else:
                for jpoint, jweight in zip(Gauss_Point1D, Gauss_Weight1D):
                    if dim < 3:
                        quad_points.append([ipoint, jpoint])
                        quad_weights.append(iweight * jweight)
                    else:
                        for kpoint, kweight in zip(Gauss_Point1D, Gauss_Weight1D):
                            quad_points.append([ipoint, jpoint, kpoint])
                            quad_weights.append(iweight * jweight * kweight)
                   
        quad_points = jnp.array(quad_points) # (quad_degree*dim, dim)
        quad_weights = jnp.array(quad_weights) # (quad_degree,)
        
    # Triangular elements
    elif (elem_type == 'CPE3' or elem_type == 'CPE6' or elem_type.startswith('C3D4') or elem_type.startswith('C3D10')) and dim == 2:
        if Gauss_num == 1:
            quad_weights = jnp.array([1.000000000000000], dtype = jnp.float64)
            quad_points = jnp.array([[0.333333333333333, 0.333333333333333]], dtype = jnp.float64)
        
        elif Gauss_num == 3:
            quad_weights = jnp.array([0.333333333333333, 0.333333333333333, 0.333333333333333], dtype = jnp.float64)
            quad_points = jnp.array([[0.666666666666667, 0.166666666666667],
                                    [0.166666666666667, 0.666666666666667],
                                    [0.166666666666667, 0.166666666666667]], dtype = jnp.float64)
           
        elif Gauss_num == 4:
            quad_weights= jnp.array([-0.562500000000000, 0.520833333333333, 0.520833333333333, 0.520833333333333], dtype = jnp.float64)
            quad_points = jnp.array([[0.333333333333333, 0.333333333333333],
                                    [0.600000000000000, 0.200000000000000],
                                    [0.200000000000000, 0.600000000000000],
                                    [0.200000000000000, 0.200000000000000]], dtype = jnp.float64)

        elif Gauss_num == 6:
            quad_weights = jnp.array([0.109951743655322, 0.109951743655322, 0.109951743655322, 
                                     0.223381589678011, 0.223381589678011, 0.223381589678011], dtype = jnp.float64)
            quad_points = jnp.array([[0.816847572980459, 0.091576213509771],
                                    [0.091576213509771, 0.816847572980459],
                                    [0.091576213509771, 0.091576213509771],
                                    [0.108103018168070, 0.445948490915965],
                                    [0.445948490915965, 0.108103018168070],
                                    [0.445948490915965, 0.445948490915965]], dtype = jnp.float64)
        
    # Tetrahedral elements
    elif (elem_type.startswith('C3D4') or elem_type.startswith('C3D10')) and dim == 3:
        # https://github.com/sigma-py/quadpy
        if Gauss_num == 1:
            quad_weights = jnp.array([1.000000000000000], dtype = jnp.float64)
            quad_points = jnp.array([[0.250000000000000, 0.250000000000000, 0.250000000000000]], dtype = jnp.float64)
        
        elif Gauss_num == 4:
            # quad_weights = jnp.array([0.041666666666667, 0.041666666666667, 0.041666666666667, 0.041666666666667], dtype = jnp.float64)
            quad_weights = jnp.array([0.250000000000000, 0.250000000000000, 0.250000000000000, 0.250000000000000], dtype = jnp.float64)
            quad_points = jnp.array([[0.585410196624968, 0.138196601125010, 0.138196601125010],
                                     [0.138196601125010, 0.585410196624968, 0.138196601125010],
                                     [0.138196601125010, 0.138196601125010, 0.585410196624968],
                                     [0.138196601125010, 0.138196601125010, 0.138196601125010]], dtype = jnp.float64)
           
        elif Gauss_num == 8:
            quad_weights= jnp.array([0.13621784, 0.11378216, 0.13621784, 0.11378216, 0.13621784,
                                    0.11378216, 0.13621784, 0.11378216], dtype = jnp.float64)
            quad_points = jnp.array([[0.3281633 , 0.10804725, 0.3281633 , 0.10804725, 0.3281633 ,
                                    0.10804725, 0.01551009, 0.67585825],
                                   [0.3281633 , 0.10804725, 0.3281633 , 0.10804725, 0.01551009,
                                    0.67585825, 0.3281633 , 0.10804725],
                                   [0.3281633 , 0.10804725, 0.01551009, 0.67585825, 0.3281633 ,
                                    0.10804725, 0.3281633 , 0.10804725]], dtype = jnp.float64).T
        elif Gauss_num == 14:
            quad_weights = jnp.array([0.07349304, 0.11268793, 0.07349304, 0.11268793, 0.07349304,
                                       0.11268793, 0.07349304, 0.11268793, 0.04254602, 0.04254602,
                                       0.04254602, 0.04254602, 0.04254602, 0.04254602], dtype = jnp.float64)
            quad_points = jnp.array([[0.09273525, 0.31088592, 0.09273525, 0.31088592, 0.09273525,
                                    0.31088592, 0.72179425, 0.06734224, 0.0455037 , 0.0455037 ,
                                    0.4544963 , 0.0455037 , 0.4544963 , 0.4544963 ],
                                   [0.09273525, 0.31088592, 0.09273525, 0.31088592, 0.72179425,
                                    0.06734224, 0.09273525, 0.31088592, 0.0455037 , 0.4544963 ,
                                    0.0455037 , 0.4544963 , 0.0455037 , 0.4544963 ],
                                   [0.09273525, 0.31088592, 0.72179425, 0.06734224, 0.09273525,
                                    0.31088592, 0.09273525, 0.31088592, 0.4544963 , 0.0455037 ,
                                    0.0455037 , 0.4544963 , 0.4544963 , 0.0455037 ]], dtype = jnp.float64).T
        elif Gauss_num == 24:
            quad_weights = jnp.array([0.03992275, 0.01007721, 0.05535718, 0.03992275, 0.01007721,
                                   0.05535718, 0.03992275, 0.01007721, 0.05535718, 0.03992275,
                                   0.01007721, 0.05535718, 0.04821429, 0.04821429, 0.04821429,
                                   0.04821429, 0.04821429, 0.04821429, 0.04821429, 0.04821429,
                                   0.04821429, 0.04821429, 0.04821429, 0.04821429], dtype = jnp.float64)
            quad_points = jnp.array([[0.21460287, 0.04067396, 0.32233789, 0.21460287, 0.04067396,
                                    0.32233789, 0.21460287, 0.04067396, 0.32233789, 0.35619139,
                                    0.87797812, 0.03298633, 0.063661  , 0.063661  , 0.26967233,
                                    0.063661  , 0.26967233, 0.26967233, 0.063661  , 0.063661  ,
                                    0.60300566, 0.063661  , 0.60300566, 0.60300566],
                                   [0.21460287, 0.04067396, 0.32233789, 0.21460287, 0.04067396,
                                    0.32233789, 0.35619139, 0.87797812, 0.03298633, 0.21460287,
                                    0.04067396, 0.32233789, 0.063661  , 0.26967233, 0.063661  ,
                                    0.26967233, 0.063661  , 0.60300566, 0.063661  , 0.60300566,
                                    0.063661  , 0.60300566, 0.063661  , 0.26967233],
                                   [0.21460287, 0.04067396, 0.32233789, 0.35619139, 0.87797812,
                                    0.03298633, 0.21460287, 0.04067396, 0.32233789, 0.21460287,
                                    0.04067396, 0.32233789, 0.26967233, 0.063661  , 0.063661  ,
                                    0.60300566, 0.60300566, 0.063661  , 0.60300566, 0.063661  ,
                                    0.063661  , 0.26967233, 0.26967233, 0.063661  ]], dtype = jnp.float64).T
        
    return quad_points, quad_weights


def get_Gauss_num(elem_type, dim):
    if elem_type.startswith('C3D8') or elem_type.startswith('C3D20'): # 3D Hex element
        Gauss_num_FEM = 2   # 2
        Gauss_num_CFEM = 3  # 6
        Gauss_num_norm = 4
        Gauss_num_tr = Gauss_num_FEM
        
        quad_num_FEM = Gauss_num_FEM ** dim
        quad_num_CFEM = Gauss_num_CFEM ** dim
        quad_num_norm = Gauss_num_norm ** dim
        quad_num_tr = Gauss_num_tr ** (dim-1)
        
        
    elif elem_type.startswith('C3D4') or elem_type.startswith('C3D10'): # 3D Tet element
        Gauss_num_FEM = 1   # 1, 4, 8, 14, 24
        Gauss_num_CFEM = 4  # 1, 4, 8, 14, 24
        Gauss_num_norm = 4
        Gauss_num_tr = 3    # 1, 4, 6
    
        quad_num_FEM = Gauss_num_FEM
        quad_num_CFEM = Gauss_num_CFEM
        quad_num_norm = Gauss_num_norm
        quad_num_tr = Gauss_num_tr
        
    return Gauss_num_FEM, Gauss_num_CFEM, Gauss_num_norm, Gauss_num_tr, quad_num_FEM, quad_num_CFEM, quad_num_norm, quad_num_tr


def get_quadrature(x_nds, nGP):
    """
    --- input ---
    x_nds: (Jx,)
    --- output ---
    xGP: (nseg X nGP) where nseg = Jx-1
    xJW: (nseg X nGP) where nseg = Jx-1
    """
    x_seg = jnp.concatenate((jnp.expand_dims(x_nds[:-1], axis=1),
                            jnp.expand_dims(x_nds[1 :], axis=1)), axis=1) # (nseg, 2)
    J_seg = 0.5 * (x_seg[:,1] - x_seg[:,0]) # (nseg,)
    
    GW, GP = GaussSet(nGP, return_type='jnp') # (nGP,)
    
    xGP = (x_seg[:,1]-x_seg[:,0])[:,None]*((GP+1)/2)[None,:] + x_seg[:,0][:,None]
    xJW = J_seg[:,None] * GW[None,:]
    
    return xGP.reshape(-1), xJW.reshape(-1)
v_get_quadrature = jax.vmap(get_quadrature, in_axes = (0,None)) # vectorize w.r.t. input x


def gauss_integrate(fval, xJW):
    # fval, xJW: (nGP X nGP,)
    return jnp.sum(fval * xJW)