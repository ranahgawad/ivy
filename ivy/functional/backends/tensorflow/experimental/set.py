# global
import tensorflow as tf
from typing import Tuple, Union, Optional
from collections import namedtuple
from ivy.func_wrapper import with_unsupported_dtypes
from . import backend_version

def intersection(
    arr1: Union[tf.Tensor, tf.Variable],
    arr2: Union[tf.Tensor, tf.Variable],
    assume_unique: Optional[bool] = False,
    return_indices: Optional[bool] = False,
    validate_indices: Optional[bool] = True,
    /,
    *,
    out: Union[tf.Tensor, tf.Variable] = None,
) -> Union[tf.Tensor, tf.Variable]:
    flat_arr1 = tf.reshape(arr1, -1)
    flat_arr2 = tf.reshape(arr2, -1)
    return tf.sets.intersection([flat_arr1], [flat_arr2], validate_indices).values