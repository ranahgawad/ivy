# global
import numpy as np
from typing import Union, Tuple, Optional
from collections import namedtuple
from packaging import version

def intersection(
    arr1: np.ndarray,
    arr2: np.ndarray,
    assume_unique: Optional[bool] = False,
    return_indices: Optional[bool] = False,
    validate_indices: Optional[bool] = True,
    /,
    *,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    return np.intersect1d(arr1, arr2, assume_unique, return_indices)