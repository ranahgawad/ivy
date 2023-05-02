from ivy.data_classes.container.base import ContainerBase
from ivy.data_classes.container.base import ContainerBase
from typing import Union, List, Dict, Optional, Tuple
import ivy

class _ContainerWithSetExperimental(ContainerBase):
    @staticmethod
    def static_intersection(
        arr1: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        arr2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        assume_unique: Optional[bool] = False,
        return_indices: Optional[bool] = False,
        validate_indices: Optional[bool] = True,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return ContainerBase.cont_multi_map_in_function(
            "intersection",
            arr1,
            arr2,
            assume_unique,
            return_indices,
            validate_indices,
            out=out,
        )
    
    def intersection(
        self,
        arr2: Union[ivy.Container, ivy.Array, ivy.NativeArray],
        assume_unique: Optional[bool] = False,
        return_indices: Optional[bool] = False,
        validate_indices: Optional[bool] = True,
        /,
        *,
        out: Optional[ivy.Container] = None,
    ) -> ivy.Container:
        return self.static_intersection(
            self,
            arr2,
            assume_unique,
            return_indices,
            validate_indices,
            out=out,
        )

