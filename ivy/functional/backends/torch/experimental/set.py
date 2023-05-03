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
    if return_indices:
        ar1 = arr1.flatten()
        perm = arr1.argsort()
        aux = ar1[perm]
        mask = torch.empty(aux.shape, dtype=torch.bool)
        mask[:1] = True
        mask[1:] = aux[1:] != aux[:-1]
        ret1 = (aux[mask],)
        ret1 += (perm[mask],)
        ar1, ind1 = ret1

        ar2 = arr2.flatten()
        perm = arr2.argsort()
        aux = ar2[perm]
        mask = torch.empty(aux.shape, dtype=torch.bool)
        mask[:1] = True
        mask[1:] = aux[1:] != aux[:-1]
        ret2 = (aux[mask],)
        ret2 += (perm[mask],)
        ar2, ind2 = ret2
    else:
        ar1 = torch.unique(arr1)
        ar2 = torch.unique(arr2)
 

    aux = torch.cat((ar1, ar2))
    if return_indices:
        aux_sort_indices = aux.argsort()
        aux = aux[aux_sort_indices]
    else:
         aux.sort()

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - torch.numel(ar1)
        ar1_indices = ind1[ar1_indices]
        ar2_indices = ind2[ar2_indices]
        return int1d, ar1_indices, ar2_indices
    else:
        return int1d

