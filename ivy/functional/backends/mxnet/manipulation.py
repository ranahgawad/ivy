import mxnet as mx
from numbers import Number
from typing import Union, Tuple, Optional, List, Sequence
import ivy


def concat(
    xs: Union[(Tuple[(None, ...)], List[None])],
    /,
    *,
    axis: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.concat Not Implemented")


def expand_dims(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    axis: Union[(int, Sequence[int])] = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.expand_dims Not Implemented")


def flip(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.flip Not Implemented")


def permute_dims(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    axes: Tuple[(int, ...)],
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.permute_dims Not Implemented")


def reshape(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    shape: Union[(ivy.NativeShape, Sequence[int])],
    *,
    copy: Optional[bool] = None,
    order: str = "C",
    allowzero: bool = True,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.reshape Not Implemented")


def roll(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    shift: Union[(int, Sequence[int])],
    *,
    axis: Optional[Union[(int, Sequence[int])]] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.roll Not Implemented")


def squeeze(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    axis: Union[(int, Sequence[int])],
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.squeeze Not Implemented")


def stack(
    arrays: Union[(Tuple[None], List[None])],
    /,
    *,
    axis: int = 0,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.stack Not Implemented")


def split(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    num_or_size_splits: Optional[Union[(int, Sequence[int])]] = None,
    axis: int = 0,
    with_remainder: bool = False,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.split Not Implemented")


def repeat(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    repeats: Union[(int, List[int])],
    *,
    axis: int = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.repeat Not Implemented")


def tile(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    repeats: Sequence[int],
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    return mx.nd.tile(x, repeats)


def constant_pad(
    x, /, pad_width, *, value=0, out: Optional[Union[(None, mx.ndarray.NDArray)]] = None
):
    raise NotImplementedError("mxnet.constant_pad Not Implemented")


def zero_pad(
    x, /, pad_width, *, out: Optional[Union[(None, mx.ndarray.NDArray)]] = None
):
    raise NotImplementedError("mxnet.zero_pad Not Implemented")


def swapaxes(
    x,
    axis0,
    axis1,
    /,
    *,
    copy: Optional[bool] = None,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
):
    raise NotImplementedError("mxnet.swapaxes Not Implemented")


def clip(
    x: Union[(None, mx.ndarray.NDArray)],
    x_min: Union[(Number, None, mx.ndarray.NDArray)],
    x_max: Union[(Number, None, mx.ndarray.NDArray)],
    /,
    *,
    out: Optional[Union[(None, mx.ndarray.NDArray)]] = None,
) -> Union[(None, mx.ndarray.NDArray)]:
    raise NotImplementedError("mxnet.clip Not Implemented")


def unstack(
    x: Union[(None, mx.ndarray.NDArray)],
    /,
    *,
    copy: Optional[bool] = None,
    axis: int = 0,
    keepdims: bool = False,
) -> List[None]:
    raise NotImplementedError("mxnet.unstack Not Implemented")
