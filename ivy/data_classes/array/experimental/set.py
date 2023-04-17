# global
import abc
from typing import Optional, Tuple


import ivy


class _ArrayWithSetExperimental(abc.ABC):
    def intersection(
        self: ivy.Array,
        arr2: ivy.Array,
        assume_unique: Optional[bool] = False,
        return_indices: Optional[bool] = False,
        validate_indices: Optional[bool] = True,
        /,
        *,
        out: Optional[ivy.Array] = None,
    ) -> ivy.Array:
        return ivy.intersection(self, arr2, assume_unique, return_indices, validate_indices)
