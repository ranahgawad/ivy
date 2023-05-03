# global
import numpy as np
from hypothesis import assume
from hypothesis import strategies as st

# local
import ivy_tests.test_ivy.helpers as helpers
from ivy_tests.test_ivy.helpers import handle_test


@handle_test(
    fn_tree="functional.ivy.experimental.intersection",
    args_packet = helpers.dtype_and_values(
        available_dtypes=helpers.get_dtypes("valid"),
        num_arrays=2,
        shared_dtype=True
        ),
    
    assume_unique = st.just(False),
    return_indices = st.booleans(),
    validate_indices = st.booleans(),
    test_with_out = st.just(False)
)
def test_intersection(
    *,
    args_packet,
    test_flags,
    backend_fw,
    fn_name,
    on_device,
    ground_truth_backend,
    assume_unique,
    return_indices,
    validate_indices
):
    dtype, arr = args_packet
    helpers.test_function(
        ground_truth_backend=ground_truth_backend,
        test_flags=test_flags,
        arr1=arr[0],
        arr2=arr[1],
        fw = backend_fw,
        fn_name = fn_name,
        on_device=on_device,
        input_dtypes=dtype,
        assume_unique=assume_unique,
        return_indices=return_indices,
        validate_indices=validate_indices,
    )

