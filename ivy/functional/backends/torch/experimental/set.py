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
    ar1 = torch.unique(ar1)
    ar2 = torch.unique(ar2)
    
    aux = torch.cat((ar1, ar2))
    aux = aux.sort()

    mask = aux[0][1:] == aux[0][:-1]
    int1d = aux[0][:-1][mask]

    if return_indices:
        idx1 = a.unsqueeze(2) == b.unsqueeze(1)
        idx1 = idx1.nonzero()
        idx1_ = idx1[:, :2]

        matches_len = idx1[:,0].unique(return_counts=True)[1]
        if (matches_len == matches_len[0]).all():
            ar1_indices = idx1[:, 1].contiguous().view(-1, matches_len[0])

        ar1_indices = ar1_indices[torch.arange(ar1_indices.size(0)).unsqueeze(1), idx1[:, 2].view_as(ar1_indices)].flatten()
     

        idx2 = c.unsqueeze(2) == b.unsqueeze(1)
        idx2 = idx2.nonzero() 

        idx2_ = idx2[:, :2]


        matches_len = idx2[:,0].unique(return_counts=True)[1]
        if (matches_len == matches_len[0]).all():
            ar2_indices = idx2[:, 1].contiguous().view(-1, matches_len[0])
            

        ar2_indices = ar2_indices[torch.arange(ar2_indices.size(0)).unsqueeze(1), idx2[:, 2].view_as(ar2_indices)].flatten()
       

        return int1d, ar1_indices, ar2_indices
    else:
        return int1d
