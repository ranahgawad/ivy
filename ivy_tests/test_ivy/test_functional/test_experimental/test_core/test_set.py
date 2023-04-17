# global
import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test

@st.composite
def _generate_intersection_args(draw):
    dtype, arr1 = draw(
        helpers.dtype_and_values(
        min_num_dims=1,
        max_num_dims=1,
        available_dtypes=helpers.get_dtypes("integer"),
        )
    )

    arr2_shape = len(arr1[0]) - 1
    dtype, arr2 = draw(
        helpers.dtype_and_values(
        min_num_dims=1,
        max_num_dims=1,
        available_dtypes=helpers.get_dtypes("integer"),
        )
    )
    
    assume_unique = draw(st.booleans())
    return_indices = draw(st.booleans())
    validate_indices = draw(st.booleans())
    return dtype, arr1, arr2, assume_unique, return_indices, validate_indices

@handle_test(
    fn_tree="functional.ivy.experimental.intersection",
    args_packet = _generate_intersection_args(),
)
def test_intersection(
    *,
    args_packet,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
):
    dtype, arr1, arr2, assume_unique, return_indices, validate_indices = args_packet
    test_flags.with_out = False
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        test_flags=test_flags,
        arr1=arr1[0],
        arr2=arr2[0],
        fw = backend_fw,
        fn_name = fn_name,
        on_device=on_device,
        input_dtypes=dtype,
        assume_unique=assume_unique,
        return_indices=return_indices,
        validate_indices=validate_indices,
    )

