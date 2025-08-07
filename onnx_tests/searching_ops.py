from collections.abc import Callable
from types import ModuleType

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw


@st.composite
def argmax(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_argmaxmin(dtype, op.arg_max))


@st.composite
def argmin(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_argmaxmin(dtype, op.arg_min))


@st.composite
def _argmaxmin(draw: st.DrawFn, dtype: np.dtype, spox_fun: Callable) -> TestCaseDraw:
    # TODO: The standard does not describe the behavior for rank-0 or 0-sized arrays
    # TODO: Behavior for NaN values is not specified by the ONNX standard
    data = draw(h.arrays(dtype, shape=hyn.array_shapes(min_dims=1, min_side=1)))
    return TestCaseDraw(
        inputs={"data": data},
        attribute_kwargs={
            "axis": draw(st.integers(0, data.ndim - 1)),
            "keepdims": draw(st.sampled_from([0, 1])),
            "select_last_index": draw(st.sampled_from([0, 1])),
        },
        spox_fun=spox_fun,
    )


@st.composite
def unique(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    shape = hyn.array_shapes(min_side=0)
    arr = draw(h.arrays(dtype=dtype, shape=shape))
    rank = arr.ndim
    axis = draw(st.one_of(st.none() | st.integers(-rank, rank - 1)))
    sorted = draw(st.sampled_from([0, 1]))

    return TestCaseDraw(
        inputs={"X": arr},
        attribute_kwargs={
            "axis": axis,
            "sorted": sorted,
        },
        spox_fun=op.unique,
    )


@st.composite
def where(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    shapes = draw(hyn.mutually_broadcastable_shapes(num_shapes=3))

    cond = draw(h.arrays(np.dtype(bool), shape=shapes.input_shapes[0]))
    x = draw(h.arrays(dtype, shape=shapes.input_shapes[1]))
    y = draw(h.arrays(dtype, shape=shapes.input_shapes[2]))

    return TestCaseDraw(
        inputs={"condition": cond, "X": x, "Y": y},
        attribute_kwargs={},
        spox_fun=op.where,
    )
