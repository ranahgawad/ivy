# global
import paddle
from typing import Tuple, Optional
from collections import namedtuple
from ivy.func_wrapper import with_unsupported_device_and_dtypes

# local

from . import backend_version

def intersection(
    arr1: paddle.Tensor,
    arr2: paddle.Tensor,
    assume_unique: Optional[bool] = False,
    return_indices: Optional[bool] = False,
    validate_indices: Optional[bool] = True,
    /,
    *,
    out: paddle.Tensor = None,
) -> paddle.Tensor:
    if not assume_unique:
        if return_indices:
            ar1, ind1 = paddle.unique(ar1, return_index=True)
            ar2, ind2 = paddle.unique(ar2, return_index=True)
        else:
            ar1 = paddle.unique(ar1)
            ar2 = paddle.unique(ar2)
    else:
        ar1 = ar1.ravel()
        ar2 = ar2.ravel()

    aux = paddle.concat((ar1, ar2))
    if return_indices:
        aux_sort_indices = paddle.argsort(aux)
        aux = aux[aux_sort_indices]
    else:
        aux = paddle.sort(aux)

    mask = aux[1:] == aux[:-1]
    int1d = aux[:-1][mask]

    if return_indices:
        ar1_indices = aux_sort_indices[:-1][mask]
        ar2_indices = aux_sort_indices[1:][mask] - ar1.size
        if not assume_unique:
            ar1_indices = ind1[ar1_indices]
            ar2_indices = ind2[ar2_indices]

        return int1d, ar1_indices, ar2_indices
    else:
        return int1d