from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
from hypothesis import assume
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn
from spox import Var

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw

# TODO: Add Pow


@st.composite
def abs(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.abs))


@st.composite
def acos(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.acos))


@st.composite
def acosh(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.acosh))


@st.composite
def asin(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.asin))


@st.composite
def asinh(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.asinh))


@st.composite
def atan(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.atan))


@st.composite
def atanh(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.atanh))


@st.composite
def ceil(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.ceil))


@st.composite
def cos(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.cos))


@st.composite
def cosh(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.cosh))


@st.composite
def erf(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.erf))


@st.composite
def exp(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.exp))


@st.composite
def floor(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.floor))


@st.composite
def hard_swish(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.hard_swish))


@st.composite
def hardmax(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.hardmax, with_axis=True))


@st.composite
def identity(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.identity))


@st.composite
def isinf(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.isinf))


@st.composite
def isnan(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.isnan))


@st.composite
def log(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.log))


@st.composite
def log_softmax(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.log_softmax, with_axis=True))


@st.composite
def neg(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.neg))


@st.composite
def not_(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.not_))


@st.composite
def reciprocal(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.reciprocal))


@st.composite
def round(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.round))


@st.composite
def sigmoid(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.sigmoid, dtype_args={"min_value": -5, "max_value": 5}))


@st.composite
def sign(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.sign))


@st.composite
def sin(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.sin))


@st.composite
def sinh(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.sinh))


@st.composite
def softmax(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.softmax, with_axis=True))


@st.composite
def sqrt(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.sqrt))


@st.composite
def tan(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.tan))


@st.composite
def tanh(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.tanh))


@st.composite
def mish(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_unary(dtype, op.mish))


@st.composite
def hard_sigmoid(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    attributes = {
        "alpha": draw(st.floats(1e-3, 0.3)),
        "beta": draw(st.floats(1e-3, 0.6)),
    }
    return draw(_unary(dtype, op.hard_sigmoid, independent_attributes=attributes))


@st.composite
def leaky_relu(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    attributes = {
        "alpha": draw(st.floats(1e-3, 0.1)),
    }
    return draw(_unary(dtype, op.leaky_relu, independent_attributes=attributes))


@st.composite
def add(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.add))


@st.composite
def and_(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.and_))


@st.composite
def bitwise_and(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.bitwise_and))


@st.composite
def bitwise_or(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.bitwise_or))


@st.composite
def bitwise_xor(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.bitwise_xor))


@st.composite
def bit_shift(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    # Over and underflow semantics are undefined behavior in most
    # languages. We only produce test cases that do not
    # over-/underflow.
    direction = draw(st.sampled_from(["LEFT", "RIGHT"]))
    x, y = draw(h.broadcastable_arrays(dtype))

    def shortest_leading_zeros(x: np.ndarray) -> int:
        mask = 1 << np.iinfo(x.dtype).bits - 1
        for shift in range(np.iinfo(dtype).bits - 1):
            if np.any((x << shift) & mask):
                break
        return shift  # type: ignore

    def shortest_trailing_zeros(x: np.ndarray) -> int:
        for shift in range(np.iinfo(dtype).bits - 1):
            if np.any(x & (1 << shift)):
                break
        return shift  # type: ignore

    # Find the maximal shift that we can do without overflowing any element.
    if y.size > 0:
        if direction == "LEFT":
            max_shift = shortest_leading_zeros(x)
        else:
            max_shift = shortest_trailing_zeros(x)
        y = np.minimum(y, np.min(max_shift).astype(dtype))

    return TestCaseDraw(
        inputs=[x, y], attribute_kwargs={"direction": direction}, spox_fun=op.bit_shift
    )


@st.composite
def div(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    test_case = draw(_binary(dtype, op.div))
    # discard integer arrays with zeros in the divisor
    if dtype.kind in "iu":
        assume(np.all(test_case.inputs[1] != 0))  # type: ignore
    return test_case


@st.composite
def equal(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.equal))


@st.composite
def greater(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.greater))


@st.composite
def greater_or_equal(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.greater_or_equal))


@st.composite
def less(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.less))


@st.composite
def less_or_equal(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.less_or_equal))


@st.composite
def mod(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    if dtype.kind in "iu":
        # fmod may only be set to 0 for integer data types
        fmod = draw(st.sampled_from([0, 1]))
    else:
        fmod = 1
    test_case = draw(_binary(dtype, op.mod, attributes={"fmod": fmod}))
    if dtype.kind in "iu":
        # Disallow division by 0
        assume(np.all(test_case.inputs[1] != 0))  # type: ignore

    return test_case


@st.composite
def mul(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.mul))


@st.composite
def or_(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.or_))


@st.composite
def sub(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.sub))


@st.composite
def xor(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    return draw(_binary(dtype, op.xor))


@st.composite
def _binary(
    draw: st.DrawFn,
    dtype: np.dtype,
    spox_fun: Callable[..., Var],
    attributes: dict[str, Any] = {},
) -> TestCaseDraw:
    array_tuple = draw(h.broadcastable_arrays(dtype))
    return TestCaseDraw(
        inputs=list(array_tuple), attribute_kwargs=attributes, spox_fun=spox_fun
    )


@st.composite
def _unary(
    draw: st.DrawFn,
    dtype: np.dtype,
    spox_fun: Callable[..., Var],
    independent_attributes: dict[str, Any] = {},
    dtype_args={},
    with_axis=False,
) -> TestCaseDraw:
    shape = hyn.array_shapes(min_dims=1 if with_axis else 0, min_side=0, max_dims=3)
    array = draw(h.arrays(dtype, shape=shape, **dtype_args))
    attributes = independent_attributes.copy()
    if with_axis:
        ndim = array.ndim
        attributes["axis"] = draw(st.integers(-ndim, ndim - 1))
    return TestCaseDraw(inputs=[array], attribute_kwargs=attributes, spox_fun=spox_fun)
