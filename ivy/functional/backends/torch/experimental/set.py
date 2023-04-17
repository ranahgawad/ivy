# global
import torch
from typing import Tuple, Optional
from collections import namedtuple

# local
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

def intersection(
    arr1: torch.Tensor,
    arr2: torch.Tensor,
    assume_unique: Optional[bool] = False,
    return_indices: Optional[bool] = False,
    validate_indices: Optional[bool] = True,
    /,
    *,
    out: torch.Tensor = None,
) -> torch.Tensor:
    arr1 = torch.unique(arr1)
    arr2 = torch.unique(arr2)
    

    aux = torch.cat((arr1, arr2))
    aux = aux.sort()

    mask = aux[0][1:] == aux[0][:-1]
    int1d = aux[0][:-1][mask]

    return int1d