from collections.abc import Callable
from types import ModuleType

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw


@st.composite
def cumsum(
    draw: st.DrawFn, dtype: np.dtype, dtype_axis: np.dtype, op: ModuleType
) -> TestCaseDraw:
    x = draw(h.arrays(dtype, shape=hyn.array_shapes(min_dims=1, min_side=0)))
    rank = x.ndim

    axis = np.asarray(draw(st.integers(-rank, rank - 1)), dtype_axis)
    return TestCaseDraw(
        inputs={"x": x, "axis": axis},
        attribute_kwargs={
            "exclusive": draw(st.booleans().map(int)),
            "reverse": draw(st.booleans().map(int)),
        },
        spox_fun=op.cumsum,
    )


@st.composite
def reduce_l1(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_reduce(dtype, op.reduce_l1))


@st.composite
def reduce_l2(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_reduce(dtype, op.reduce_l2))


@st.composite
def reduce_log_sum(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_reduce(dtype, op.reduce_log_sum))


@st.composite
def reduce_log_sum_exp(
    draw: st.DrawFn, dtype: np.dtype, op: ModuleType
) -> TestCaseDraw:
    return draw(_reduce(dtype, op.reduce_log_sum_exp))


@st.composite
def reduce_max(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_reduce(dtype, op.reduce_max))


@st.composite
def reduce_mean(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_reduce(dtype, op.reduce_mean))


@st.composite
def reduce_min(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_reduce(dtype, op.reduce_min))


@st.composite
def reduce_prod(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_reduce(dtype, op.reduce_prod))


@st.composite
def reduce_sum(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_reduce(dtype, op.reduce_sum))


@st.composite
def reduce_sum_square(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_reduce(dtype, op.reduce_sum_square))


@st.composite
def _reduce(draw: st.DrawFn, dtype: np.dtype, spox_fun: Callable) -> TestCaseDraw:
    # TODO: onnxruntime fails badly for 0-rank and 0-sized inputs
    # array = data.draw(h.arrays(dtype, shape=hyn.array_shapes(min_dims=0, min_side=0)))

    data = draw(h.arrays(dtype, shape=hyn.array_shapes(min_dims=1, min_side=1)))
    axes = []
    for axis in range(data.ndim):
        maybe_axis = draw(st.sampled_from([None, axis, axis - data.ndim]))
        if maybe_axis is not None:
            axes.append(maybe_axis)

    rand = draw(st.randoms())
    rand.shuffle(axes)

    return TestCaseDraw(
        inputs={
            "data": data,
            "axes": axes if axes is None else np.asarray(axes, dtype=np.int64),
        },
        attribute_kwargs={
            "keepdims": draw(st.sampled_from([0, 1])),
            "noop_with_empty_axes": draw(st.sampled_from([0, 1])),
        },
        spox_fun=spox_fun,
    )
