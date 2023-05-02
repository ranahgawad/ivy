# global
import tensorflow as tf
from typing import Tuple, Union, Optional
from collections import namedtuple
from ivy.func_wrapper import with_supported_dtypes
from . import backend_version

@with_supported_dtypes({"2.9.1 and below": ("int32", "uint8", "int16", "int8", "int64", "uint16",)}, backend_version)
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
    ret = tf.sets.intersection([flat_arr1], [flat_arr2], validate_indices).values
    if return_indices:
        expanded_b = ret[..., None]
        tiled_a = tf.tile(arr1[None, ...], [tf.shape(ret)[0], 1])
        tiled_c = tf.tile(arr2[None, ...], [tf.shape(ret)[0], 1])
        mult_a = tf.cast(tf.equal(expanded_b, tiled_a), tf.float32)
        sub_a = tf.cast(tf.math.equal(tf.reduce_sum(mult_a, -1), 0), tf.int64)
        res_a = tf.argmax(mult_a, axis=-1) - sub_a
        mult_c = tf.cast(tf.equal(expanded_b, tiled_c), tf.float32)
        sub_c = tf.cast(tf.math.equal(tf.reduce_sum(mult_c, -1), 0), tf.int64)
        res_c = tf.argmax(mult_c, axis=-1) - sub_c
        return ret, res_a, res_c
    else:
        return ret