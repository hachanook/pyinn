from itertools import product

from jax.numpy import (asarray, broadcast_arrays, can_cast,
                       empty, nan, searchsorted, where, zeros)
from jax._src.tree_util import register_pytree_node
from jax._src.numpy.util import check_arraylike, promote_dtypes_inexact



class LinearInterpolator:
  def __init__(self,grid):
    """ 1D linear interpolation
    --- input --- 
    grid: (J,) 1D vector of the grid
    """

    self.grid = grid

  def __call__(self, xi, values):
    
    indice, norm_distance = self._find_indices(xi)
    result = self._evaluate_linear(indice, norm_distance, values)
    return result
  
  def _evaluate_linear(self, indice, norm_distance, values):
    
    value = values[indice] * (1-norm_distance) + values[indice+1] * (norm_distance)
    return value

  def _find_indices(self, xi):
    i = searchsorted(self.grid, xi) - 1
    i = where(i < 0, 0, i) # i: element index
    norm_distance = (xi - self.grid[i]) / (self.grid[i + 1] - self.grid[i])
    return i, norm_distance










