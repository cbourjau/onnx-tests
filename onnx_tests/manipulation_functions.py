from types import ModuleType

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw


@st.composite
def concat(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    n_arrays = draw(st.integers(min_value=1, max_value=100))
    base_shape = draw(hyn.array_shapes(min_dims=1, min_side=0, max_dims=3))
    rank = len(base_shape)
    axis = draw(st.integers(min_value=-rank, max_value=rank - 1))

    arrays = []
    for _ in range(n_arrays):
        shape = list(base_shape)
        shape[axis] = draw(st.integers(0, 3))
        arrays.append(draw(h.arrays(dtype=dtype, shape=tuple(shape))))

    return TestCaseDraw(
        inputs={"inputs": arrays}, attribute_kwargs={"axis": axis}, spox_fun=op.concat
    )
