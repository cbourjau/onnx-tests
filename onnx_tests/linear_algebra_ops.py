from types import ModuleType

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw


@st.composite
def det(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    shape = draw(hyn.array_shapes(min_dims=1, min_side=1))
    shape = shape + (shape[-1],)
    x = draw(h.arrays(dtype, shape=shape))
    return TestCaseDraw(inputs={"X": x}, attribute_kwargs={}, spox_fun=op.det)


@st.composite
def einsum(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    # TODO: Add more equations
    # trace
    equation = "ii"
    shape = draw(hyn.array_shapes(min_dims=2, min_side=0))
    arr = draw(h.arrays(dtype, shape))
    if arr.ndim > 2:
        equation = draw(st.sampled_from(["i...i", "ii...", "...ii"]))

    return TestCaseDraw(
        inputs={"Inputs": [arr]},
        attribute_kwargs={"equation": equation},
        spox_fun=op.einsum,
    )


@st.composite
def matmul(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    shapes = draw(h.matmul_shapes())
    x = draw(h.arrays(dtype, shape=shapes[0]))
    y = draw(h.arrays(dtype, shape=shapes[1]))

    return TestCaseDraw(inputs=[x, y], attribute_kwargs={}, spox_fun=op.matmul)
