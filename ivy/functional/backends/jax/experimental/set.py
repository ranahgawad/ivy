# global
import jax.numpy as jnp
from typing import Tuple, Optional
from collections import namedtuple

# local
from ivy.functional.backends.jax import JaxArray


def intersection(
    arr1: JaxArray,
    arr2: JaxArray,
    assume_unique: Optional[bool] = False,
    return_indices: Optional[bool] = False,
    validate_indices: Optional[bool] = True,
    /,
    *,
    out: Optional[JaxArray] = None,
) -> JaxArray:
    return jnp.intersect1d(arr1, arr2, assume_unique, return_indices=False)