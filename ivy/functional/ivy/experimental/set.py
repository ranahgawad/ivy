from typing import Optional, Union, Tuple
import ivy
from ivy.func_wrapper import (
    handle_out_argument,
    to_native_arrays_and_back,
    handle_nestable,
)
from ivy.utils.exceptions import handle_exceptions


@to_native_arrays_and_back
@handle_out_argument
@handle_nestable
@handle_exceptions
def intersection(
    arr1: Union[ivy.Array, ivy.NativeArray],
    arr2: Union[ivy.Array, ivy.NativeArray],
    assume_unique: Optional[bool] = False,
    return_indices: Optional[bool] = False,
    validate_indices: Optional[bool] = True,
    /,
    *,
    out: Optional[ivy.Array] = None,
) -> ivy.Array:
    return ivy.current_backend().intersection(arr1, arr2, assume_unique, return_indices, validate_indices)