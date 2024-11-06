from itertools import product

from jax.numpy import (asarray, broadcast_arrays, can_cast,
                       empty, nan, searchsorted, where, zeros)
from jax._src.tree_util import register_pytree_node
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact



class LinearInterpolator:
  def __init__(self,grid,values):
    """ 1D linear interpolation
    --- input --- 
    grid: (J,) 1D vector of the grid
    values: (J,) 1D vector of nodal values
    """

    self.grid = grid
    self.values = values

  def __call__(self, xi):
    
    indice, norm_distance = self._find_indices(xi)
    result = self._evaluate_linear(indice, norm_distance)
    return result
  
  def _evaluate_linear(self, indice, norm_distance):
    
    value = self.values[indice] * (1-norm_distance) + self.values[indice+1] * (norm_distance)
    return value

  def _find_indices(self, xi):
    i = searchsorted(self.grid, xi) - 1
    i = where(i < 0, 0, i) # i: element index
    norm_distance = (xi - self.grid[i]) / (self.grid[i + 1] - self.grid[i])
    return i, norm_distance










