import numpy as np
import jax
import jax.numpy as jnp
from itertools import product, combinations
from functools import partial
from jax.numpy import (asarray, broadcast_arrays, can_cast,
                       empty, nan, searchsorted, where, zeros)
from jax._src.tree_util import register_pytree_node
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact
from scipy.sparse import csc_matrix


class LinearInterpolator:
  def __init__(self, grid):
    """ 1D linear interpolation
    --- input --- 
    grid: (J,) 1D vector of the grid
    """

    self.grid = grid
    # self.grid_sort = grid.at[-1].add(0.0001)

  def __call__(self, xi, values):
    
    ielem, norm_distance = self._find_ielem(xi)
    result = self._interpolate(ielem, norm_distance, values)
    return result

  def _interpolate(self, ielem, norm_distance, values):
    
    value = values[ielem] * (1-norm_distance) + values[ielem+1] * (norm_distance)
    return value

  def _find_ielem(self, xi):
    i = searchsorted(self.grid, xi) - 1
    i = where(i < 0, 0, i) # i: element index
    norm_distance = (xi - self.grid[i]) / (self.grid[i + 1] - self.grid[i])
    return i, norm_distance


class NonlinearInterpolator(LinearInterpolator):
  def __init__(self, grid, config):
    super().__init__(grid) # prob being dropout probability

    self.nelem = config['MODEL_PARAM']['nelem']
    self.nnode = len(grid)
    self.s_patch = config['MODEL_PARAM']['s_patch']
    self.alpha_dil = config['MODEL_PARAM']['alpha_dil']
    self.p_order = config['MODEL_PARAM']['p_order']
    p_dict = {0:0, 1:2, 2:3, 3:4, 4:5, 5:6} 
    self.mbasis = p_dict[self.p_order] 
    self.radial_basis = config['MODEL_PARAM']['radial_basis']
    self.activation = config['MODEL_PARAM']['INNactivation']
    
    ## compute G_inv
    self.nodes_per_elem = 2
    self.d_c = 1.0/self.nelem     # characteristic length in physical coord.
    self.a_dil = self.alpha_dil * self.d_c

    self.Elem_nodes_host = np.zeros([self.nelem, self.nodes_per_elem], dtype=np.int64)
    for j in range(1, self.nnode):
      self.Elem_nodes_host[j-1, 0] = j-1
      self.Elem_nodes_host[j-1, 1] = j 

    
    self.indices, self.indptr = self.get_adj_mat()
    self.edex_max, self.ndex_max = self.get_dex_max()
    print(f'edex_max / ndex_max: {self.edex_max} / {self.ndex_max}')
    
    # start_patch = time.time()
    (self.Elemental_patch_nodes_st, self.edexes,
      self.Nodal_patch_nodes_st, self.Nodal_patch_nodes_bool, self.Nodal_patch_nodes_idx, self.ndexes
      ) = self.get_patch_info()       
    
    self.Gs = self.vv_get_G(self.ndexes, self.Nodal_patch_nodes_st, self.Nodal_patch_nodes_bool, grid.reshape(-1,1)) # (nelem, npe=2, ndex_max+mbasis, ndex_max+mbasis)
    self.Gs_inv = jnp.array(np.linalg.inv(self.Gs))


  def __call__(self, xi, values):
    """
    --- values: (J,) nodal values
    """
  
    ielem, norm_distance = self._find_ielem(xi)
    result = self.get_phi(xi, ielem, 0, values) * (1-norm_distance) + self.get_phi(xi, ielem, 1, values) * (norm_distance)
    return result

  def get_phi(self, xi, ielem, inode, values):
    Ginv = self.Gs_inv[ielem, inode]  # (K+P+1, K+P+1) where K: ndex_max
    nodal_patch_nodes = self.Nodal_patch_nodes_st[ielem,inode]
    RP = self.Compute_RadialBasis_1D(xi.reshape(1), self.grid[nodal_patch_nodes].reshape(-1,1), 
                                     self.ndexes[ielem,inode], self.Nodal_patch_nodes_bool[ielem,inode]) # (K+P+1)
    RP_Ginv = jnp.tensordot(RP, Ginv, axes=(0,1))[:self.ndex_max]
    result = jnp.sum(RP_Ginv * jnp.take(values, nodal_patch_nodes), keepdims=False)
    return result

  def get_adj_mat(self):
    ''' Compute Adjacency matrix from the Graph theory
    --- input ---
    Elem_nodes_host: elemental connectivity stored in CPU as numpy array, (nelem, nodes_per_elem)
    nnode: number of nodes, scalar integer
    s_patch: patch size s, scalar integer
    --- output ---
    indices, indptr: indices and index pointers used in scipy spare matrix library.
    '''
    
    
    # get adjacency matrix of graph theory based on nodal connectivity
    adj_rows, adj_cols = [], []
    
    # self 
    for inode in range(self.nnode):
      adj_rows += [inode]
      adj_cols += [inode]
    
    for ielem, elem_nodes in enumerate(self.Elem_nodes_host):
      for (inode, jnode) in combinations(list(elem_nodes), 2):
        adj_rows += [inode, jnode]
        adj_cols += [jnode, inode]
    adj_values = np.ones(len(adj_rows), dtype=np.int64)
    adj_rows = np.array(adj_rows, dtype=np.int64)
    adj_cols = np.array(adj_cols, dtype=np.int64)
    
    # build sparse matrix
    adj_sp = csc_matrix((adj_values, 
                              (adj_rows, adj_cols)),
                            shape = (self.nnode, self.nnode))
    adj_s = csc_matrix((adj_values, 
                              (adj_rows, adj_cols)),
                            shape = (self.nnode, self.nnode))
    
    # compute s th power of the adjacency matrix to get s th order of connectivity
    for itr in range(self.s_patch-1):
      adj_s = adj_s.dot(adj_sp)
    indices = adj_s.indices
    indptr = adj_s.indptr
    
    return indices, indptr


  def get_dex_max(self): # delete d_c, XY, nnode
    ''' For padding algorithm in GPU computation, we need to know the maximum number of elemental patch nodes and 
    nodal patch nodes, edex_max and ndex_max, respectively.
    --- input ---
    indices, indptr: sparce matrix form of the Adjacency matrix
    s_patch: patch size s
    Elem_nodes: elemental connectivity stored in CPU as numpy array, (nelem, nodes_per_elem)
    nelem, nodes_per_elem, dim: scalar integers..
    --- output ---
    edex_max: maximum number of elemental patch nodes
    ndex_max: maximum number of nodal patch nodes
    '''
    
    dim = 1
    edex_max = (2+2*self.s_patch)**dim # estimated value of edex_max
    edexes = np.zeros(self.nelem, dtype=np.int64) # (num_elements, )
    ndexes = np.zeros((self.nelem, self.nodes_per_elem), dtype=np.int64) # static, (nelem, nodes_per_elem)
    
    for ielem, elem_nodes in enumerate(self.Elem_nodes_host):
      if len(elem_nodes) == 2 and dim == 1: # 1D Linear element
        nodal_patch_nodes0 = self.indices[ self.indptr[elem_nodes[0]] : self.indptr[elem_nodes[0]+1] ] # global index, # node_idx 0
        nodal_patch_nodes1 = self.indices[ self.indptr[elem_nodes[1]] : self.indptr[elem_nodes[1]+1] ] # global index
        ndexes[ielem, :] = np.array([len(nodal_patch_nodes0),len(nodal_patch_nodes1)])
        elemental_patch_nodes = np.unique(np.concatenate((nodal_patch_nodes0, nodal_patch_nodes1)))  

      edexes[ielem] = len(elemental_patch_nodes)
    edex_max = np.max(edexes)
    ndex_max = np.max(ndexes)
    return edex_max, ndex_max

  def get_patch_info(self): # for block, delete s_patch, d_c, XY
    ''' Inside the C-HiDeNN shape function builder, compute the patch information for each node on each element.
    --- input ---
    indices, indptr: sparce matrix form of the Adjacency matrix
    edex_max: maximum number of elemental patch nodes
    ndex_max: maximum number of nodal patch nodes
    Elem_nodes: elemental connectivity stored in CPU as numpy array, (nelem, nodes_per_elem)
    nelem, nodes_per_elem, dim: scalar integers..
    --- output ---
    Elemental_patch_nodes_st: elemental patch nodes, (nelem, edex_max)
    edexes: number of patch nodes for each element, (nelem,)
    Nodal_patch_nodes_st: nodal patch nodes for each element, (nelem, nodes_per_elem, ndex_max)
    Nodal_patch_nodes_bool: the same shape as Nodal_patch_nodes_st, returns 1 if the node has non-zero convolution patch function value, 
                            returns 0 if the node has zero convolution patch function value.
    Nodal_patch_nodes_idx: the same shape as Nodal_patch_nodes_st, stores nodal index in the corresponding elemental patch nodes
    ndexes: number of patch nodes for each node, and each element, (nelem, nodes_per_elem)
    '''    
    dim = 1
    
    # Assign memory to variables
    ## Elemental patch
    Elemental_patch_nodes_st = np.zeros((self.nelem, self.edex_max), dtype=np.int64) # edex_max should be grater than 100!
    edexes = np.zeros(self.nelem, dtype=np.int64) # (num_elements, )
    ## Nodal patch
    Nodal_patch_nodes_st = (-1)*np.ones((self.nelem, self.nodes_per_elem, self.ndex_max), dtype=np.int64) # static, (nelem, nodes_per_elem, ndex_max)
    Nodal_patch_nodes_bool = np.zeros((self.nelem, self.nodes_per_elem, self.ndex_max), dtype=np.int64) # static, (nelem, nodes_per_elem, ndex_max)
    Nodal_patch_nodes_idx = (-1)*np.ones((self.nelem, self.nodes_per_elem, self.ndex_max), dtype=np.int64) # static, (nelem, nodes_per_elem, ndex_max)
    ndexes = np.zeros((self.nelem, self.nodes_per_elem), dtype=np.int64) # static, (nelem, nodes_per_elem)
    
    
    for ielem, elem_nodes in enumerate(self.Elem_nodes_host):
        
      # 1. for loop: nodal_patch_nodes in global nodal index
      for inode_idx, inode in enumerate(elem_nodes):
        nodal_patch_nodes = np.sort(self.indices[ self.indptr[elem_nodes[inode_idx]] : self.indptr[elem_nodes[inode_idx]+1] ]) # global index
        ndex = len(nodal_patch_nodes)
        ndexes[ielem, inode_idx] = ndex
        Nodal_patch_nodes_st[ielem, inode_idx, :ndex] = nodal_patch_nodes  # global nodal index
        Nodal_patch_nodes_bool[ielem, inode_idx, :ndex] = np.where(nodal_patch_nodes>=0, 1, 0)
      
      
      # 2. get elemental_patch_nodes    
      if len(elem_nodes) == 2 and dim == 1: # 1D Linear element
        elemental_patch_nodes = np.unique(np.concatenate((Nodal_patch_nodes_st[ielem, 0, :ndexes[ielem, 0]],
                                                            Nodal_patch_nodes_st[ielem, 1, :ndexes[ielem, 1]])))  # node_idx 1
          
      
      edex = len(elemental_patch_nodes)
      edexes[ielem] = edex
      Elemental_patch_nodes_st[ielem, :edex] = elemental_patch_nodes
      
      # 3. for loop: get nodal_patch_nodes_idx
      for inode_idx, inode in enumerate(elem_nodes):
        nodal_patch_nodes_idx = np.searchsorted(
            elemental_patch_nodes, Nodal_patch_nodes_st[ielem, inode_idx, :ndexes[ielem, inode_idx]]) # local index
        Nodal_patch_nodes_idx[ielem, inode_idx, :ndexes[ielem, inode_idx]] = nodal_patch_nodes_idx
            
    # Convert everything to device array
    Elemental_patch_nodes_st = jnp.array(Elemental_patch_nodes_st)
    edexes = jnp.array(edexes)
    Nodal_patch_nodes_st = jnp.array(Nodal_patch_nodes_st)
    Nodal_patch_nodes_bool = jnp.array(Nodal_patch_nodes_bool)
    Nodal_patch_nodes_idx = jnp.array(Nodal_patch_nodes_idx)
    ndexes = jnp.array(ndexes)
    
    return Elemental_patch_nodes_st, edexes, Nodal_patch_nodes_st, Nodal_patch_nodes_bool, Nodal_patch_nodes_idx, ndexes


  def in_range(self, xi, lb, ub):
    ''' Returns 1 when lb < xi < ub, 0 otherwise '''
    # lb: lower bound, floating number
    # ub: upper bound, floating number
    return jnp.heaviside(xi-lb,1) * jnp.heaviside(ub-xi, 0)

  @partial(jax.jit, static_argnames=['self'])
  def get_R_cubicSpline(self, xy, xvi, a_dil):
    ''' Cubic spline radial basis function '''
    zI = jnp.linalg.norm(xy - xvi)/a_dil
    R = ((2/3 - 4*zI**2 + 4*zI**3         ) * self.in_range(zI, 0.0, 0.5) +    # phi_i
                        (4/3 - 4*zI + 4*zI**2 - 4/3*zI**3) * self.in_range(zI, 0.5, 1.0))
    return R
  v_get_R_cubicSpline = jax.vmap(get_R_cubicSpline, in_axes = (None, None,0,None))

  @partial(jax.jit, static_argnames=['self'])
  def get_R_gaussian1(self, xy, xvi, a_dil):
    ''' Gaussian radial basis function, zI^1 '''
    zI = jnp.linalg.norm(xy - xvi)/a_dil
    R = jnp.exp(-zI)
    return R
  v_get_R_gaussian1 = jax.vmap(get_R_gaussian1, in_axes = (None, None,0,None))

  @partial(jax.jit, static_argnames=['self'])
  def get_R_gaussian2(self, xy, xvi, a_dil):
    ''' Gaussian radial basis function, zI^2 '''
    zI = jnp.linalg.norm(xy - xvi)/a_dil
    R = jnp.exp(-zI**2)
    return R
  v_get_R_gaussian2 = jax.vmap(get_R_gaussian2, in_axes = (None, None,0,None))

  @partial(jax.jit, static_argnames=['self'])
  def get_R_gaussian3(self, xy, xvi, a_dil):
    ''' Gaussian radial basis function, zI^3 '''
    zI = jnp.linalg.norm(xy - xvi)/a_dil
    R = jnp.exp(-zI**3)
    return R
  v_get_R_gaussian3 = jax.vmap(get_R_gaussian3, in_axes = (None, None,0,None))

  @partial(jax.jit, static_argnames=['self'])
  def get_R_gaussian4(self, xy, xvi, a_dil):
    ''' Gaussian radial basis function, zI^4 '''
    zI = jnp.linalg.norm(xy - xvi)/a_dil
    R = jnp.exp(-zI**4)
    return R
  v_get_R_gaussian4 = jax.vmap(get_R_gaussian4, in_axes = (None, None,0,None))

  @partial(jax.jit, static_argnames=['self'])
  def get_R_gaussian5(self, xy, xvi, a_dil):
    ''' Gaussian radial basis function, zI^5 '''
    zI = jnp.linalg.norm(xy - xvi)/a_dil
    R = jnp.exp(-zI**5)
    return R
  v_get_R_gaussian5 = jax.vmap(get_R_gaussian5, in_axes = (None, None,0,None))
  
  @partial(jax.jit, static_argnames=['self'])
  def get_R_cosine(self, xy, xvi, a_dil):
    ''' Cubic spline radial basis function '''
    zI = jnp.linalg.norm(xy - xvi)/a_dil
    R = 0.5* ( jnp.cos(zI) + 1 ) * self.in_range(zI, 0.0, jnp.pi)
    return R
  v_get_R_cosine = jax.vmap(get_R_cosine, in_axes = (None, None,0,None))
  
  

  @partial(jax.jit, static_argnames=['self']) # This will slower the function
  def Compute_RadialBasis_1D(self, xy, xv, ndex, nodal_patch_nodes_bool):
    """ 
    --- input ---
    # xy: point of interest (dim,)
    # xv: ndoal coordinates of patch nodes. ()
    # ndex: number of nodse in the nodal patch
    # ndex_max: max of ndex, precomputed value
    # nodal_patch_nodes_bool: boolean vector that tells ~~~
    # a_dil: dilation parameter for cubic spline
    # mbasis: number of polynomial terms
    --- output ---
    RP: [R(x), P(x)] vector, R(x) is the radial basis, P(x) is the polynomial basis, (ndex_max + m_basis,)
    """
    
    RP = jnp.zeros(self.ndex_max + self.mbasis, dtype=jnp.double)
    
    if self.radial_basis == 'cubicSpline':
        RP = RP.at[:self.ndex_max].set(self.v_get_R_cubicSpline(xy, xv, self.a_dil) * nodal_patch_nodes_bool)
    if self.radial_basis == 'gaussian1':
        RP = RP.at[:self.ndex_max].set(self.v_get_R_gaussian1(xy, xv, self.a_dil) * nodal_patch_nodes_bool)        
    if self.radial_basis == 'gaussian2':
        RP = RP.at[:self.ndex_max].set(self.v_get_R_gaussian2(xy, xv, self.a_dil) * nodal_patch_nodes_bool)        
    if self.radial_basis == 'gaussian3':
        RP = RP.at[:self.ndex_max].set(self.v_get_R_gaussian3(xy, xv, self.a_dil) * nodal_patch_nodes_bool)        
    if self.radial_basis == 'gaussian5':
        RP = RP.at[:self.ndex_max].set(self.v_get_R_gaussian3(xy, xv, self.a_dil) * nodal_patch_nodes_bool)
    if self.radial_basis == 'cosine':
        RP = RP.at[:self.ndex_max].set(self.v_get_R_cosine(xy, xv, self.a_dil) * nodal_patch_nodes_bool)
    
    
    if self.activation == 'polynomial':
      if self.mbasis > 0: # 1st
        RP = RP.at[self.ndex_max   : self.ndex_max+ 2].set(jnp.array([1 , xy[0] ]))   # N 1, x
          
      if self.mbasis > 2: # 2nd
        RP = RP.at[self.ndex_max+ 2: self.ndex_max+ 3].set(jnp.array([xy[0]**2]))   # N x^2
          
      if self.mbasis > 3: # 3rd
        RP = RP.at[self.ndex_max+ 3: self.ndex_max+ 4].set(jnp.array([xy[0]**3]))   # N x^3
          
      if self.mbasis > 4: # 4th
        RP = RP.at[self.ndex_max+ 4: self.ndex_max+ 5].set(jnp.array([xy[0]**4]))   # N x^4
      
      if self.mbasis > 5: # 4th
        RP = RP.at[self.ndex_max+ 5: self.ndex_max+ 6].set(jnp.array([xy[0]**5]))   # N x^5
      
    elif self.activation == 'sinusoidal':
      if self.mbasis > 0: # 1st
        RP = RP.at[self.ndex_max   : self.ndex_max+ 2].set(jnp.array([1 , jnp.sin(xy[0]) ]))   # N 1, sin(x)
          
      if self.mbasis > 2: # 2nd
        P = RP.at[self.ndex_max+ 2: self.ndex_max+ 3].set(jnp.sin(2*xy[0]))   # N sin(2x)
          
      if self.mbasis > 3: # 3rd
        RP = RP.at[self.ndex_max+ 3: self.ndex_max+ 4].set(jnp.sin(3*xy[0]))   # N sin(3x)
    
    elif self.activation == 'exponential':
      if self.mbasis > 0: # 1st
        RP = RP.at[self.ndex_max   : self.ndex_max+ 2].set(jnp.array([1 , jnp.exp(xy[0]) ]))   # N 1, exp(x)
          
      if self.mbasis > 2: # 2nd
        RP = RP.at[self.ndex_max+ 2: self.ndex_max+ 3].set(jnp.exp(2*xy[0]))   # N exp(2x)
          
      if self.mbasis > 3: # 3rd
        RP = RP.at[self.ndex_max+ 3: self.ndex_max+ 4].set(jnp.exp(3*xy[0]))   # N exp(3x)
    
    elif self.activation == 'sigmoid':
      if self.mbasis > 0: # 1st
        RP = RP.at[self.ndex_max   : self.ndex_max+ 2].set(jnp.array([1 , jax.nn.sigmoid(xy[0]) ]))   # N 1, exp(x)
          
      if self.mbasis > 2: # 2nd
        RP = RP.at[self.ndex_max+ 2: self.ndex_max+ 3].set(jax.nn.sigmoid(2*xy[0]))   # N exp(2x)
          
      if self.mbasis > 3: # 3rd
        RP = RP.at[self.ndex_max+ 3: self.ndex_max+ 4].set(jax.nn.sigmoid(3*xy[0]))   # N exp(3x)
    
    elif self.activation == 'tanh':
      if self.mbasis > 0: # 1st
        RP = RP.at[self.ndex_max   : self.ndex_max+ 2].set(jnp.array([1 , jax.nn.tanh(xy[0]) ]))   # N 1, exp(x)
          
      if self.mbasis > 2: # 2nd
        RP = RP.at[self.ndex_max+ 2: self.ndex_max+ 3].set(jax.nn.tanh(2*xy[0]))   # N exp(2x)
          
      if self.mbasis > 3: # 3rd
        RP = RP.at[self.ndex_max+ 3: self.ndex_max+ 4].set(jax.nn.tanh(3*xy[0]))   # N exp(3x)
    
    elif self.activation == 'gelu':
      if self.mbasis > 0: # 1st
        RP = RP.at[self.ndex_max   : self.ndex_max+ 2].set(jnp.array([1 , jax.nn.gelu(xy[0]) ]))   # N 1, exp(x)
          
      if self.mbasis > 2: # 2nd
        RP = RP.at[self.ndex_max+ 2: self.ndex_max+ 3].set(jax.nn.gelu(2*xy[0]))   # N exp(2x)
          
      if self.mbasis > 3: # 3rd
        RP = RP.at[self.ndex_max+ 3: self.ndex_max+ 4].set(jax.nn.gelu(3*xy[0]))   # N exp(3x)
    return RP

  v_Compute_RadialBasis_1D = jax.vmap(Compute_RadialBasis_1D, in_axes = (None, 0,None,None,None), out_axes=1)
  v2_Compute_RadialBasis_1D = jax.vmap(Compute_RadialBasis_1D, in_axes = (None, None,0,0,0), out_axes=0)

  @partial(jax.jit, static_argnames=['self']) # unneccessary
  def get_G(self, ndex, nodal_patch_nodes, nodal_patch_nodes_bool, XY):
    ''' Compute assembled moment matrix G. Refer to Section 2.2 of:
        "Park, Chanwook, et al. "Convolution hierarchical deep-learning neural network 
        (c-hidenn) with graphics processing unit (gpu) acceleration." Computational Mechanics (2023): 1-27."
    --- input ---
    ndex: number of nodal patch nodes, scalar integer
    nodal_patch_nodes: nodal patch nodes, (ndex_max,)
    nodal_patch_nodes_bool: nodal patch nodes boolean, (ndex_max,)
    XY: nodal coorinates, (nnode, dim)
    ndex_max: maximum number of ndexes
    a_dil: dilation parameter a, scalar double
    mbasis: number of polynomial basis functions, scalar integer
    radial_basis: type of radial basis function, string
    --- output ---
    G: assembled moment matrix, (ndex_max + m_basis, ndex_max + m_basis)
    '''
    # nodal_patch_nodes_bool: (ndex_max,)
    G = jnp.zeros((self.ndex_max + self.mbasis, self.ndex_max + self.mbasis), dtype=jnp.double)
    xv = XY[nodal_patch_nodes,:]
    RPs = self.v_Compute_RadialBasis_1D(xv, xv, ndex, nodal_patch_nodes_bool) # (ndex_max + mbasis, ndex_max)
    
    G = G.at[:,:self.ndex_max].set(RPs * nodal_patch_nodes_bool[None,:])                        
    
    # Make symmetric matrix
    G = jnp.tril(G) + jnp.triu(G.T, 1)
    
    # Build diagonal terms to nullify dimensions
    Imat = jnp.eye(self.ndex_max) * jnp.abs(nodal_patch_nodes_bool-1)[:,None]
    G = G.at[:self.ndex_max,:self.ndex_max].add(Imat)
    return G # G matrix
  v_get_G = jax.vmap(get_G, in_axes = (None, 0,0,0,None))
  vv_get_G = jax.vmap(v_get_G, in_axes = (None, 0,0,0,None))

     

     







