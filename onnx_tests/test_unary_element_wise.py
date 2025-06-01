from collections.abc import Callable
from typing import Protocol

import hypothesis.extra.numpy as hyn
import numpy as np
import pytest
import spox
import spox.opset.ai.onnx.v17 as op17
import spox.opset.ai.onnx.v18 as op18
import spox.opset.ai.onnx.v20 as op20
from hypothesis import given
from hypothesis import strategies as st
from spox import Var

from . import helpers as h
from .config import run_candidate


def assert_unary_against_reference(
    spox_fun: Callable[[Var], Var],
    x: h.ArrayWrapper,
    test: Callable[[np.ndarray, np.ndarray], None],
):
    model = spox.build({"x": x.spox_argument}, {"res": spox_fun(x.spox_argument)})

    expected, *_ = run_candidate(model, x=x.array).values()
    candidate, *_ = h.run_reference(model, x=x.array).values()

    test(candidate, expected)


def arrays(dtype: np.dtype) -> st.SearchStrategy[h.ArrayWrapper]:
    return h.arrays(
        dtype,
        hyn.array_shapes(
            min_dims=0,
            min_side=0,
        ),
    )


def dtypes(op_name: str, version: int, param_name: str) -> list[np.dtype]:
    # Identity uses "V" as the type parameter name (rather than "T" like everywhere else)
    return h.SCHEMAS["ai.onnx"][op_name][version].dtype_constraints[param_name]


def unary_element_wise_test(
    op_name: str,
    version: int,
    op: Callable[[Var], Var],
    test: Callable[[np.ndarray, np.ndarray], None] = np.testing.assert_almost_equal,
    param_name: str = "T",
) -> Callable:
    @given(data=st.data())
    @pytest.mark.parametrize("dtype", dtypes(op_name, version, param_name), ids=str)
    def test_function(data, dtype: np.dtype):
        array = data.draw(arrays(dtype))
        assert_unary_against_reference(op, array, test=test)

    return test_function


class AxisOp(Protocol):
    def __call__(self, x: Var, /, *, axis: int) -> Var: ...


def unary_element_wise_test_with_axis_param(
    op_name: str,
    version: int,
    op: AxisOp,
    test: Callable[[np.ndarray, np.ndarray], None] = np.testing.assert_almost_equal,
    param_name: str = "T",
) -> Callable:
    @given(data=st.data())
    @pytest.mark.parametrize("dtype", dtypes(op_name, version, param_name), ids=str)
    def test_function(data: st.DataObject, dtype: np.dtype):
        array = data.draw(arrays(dtype))
        ndim = array.array.ndim
        axis = data.draw(st.integers(-ndim, ndim))
        assert_unary_against_reference(lambda x: op(x, axis=axis), array, test=test)

    return test_function


def assert_allclose(actual: np.ndarray, desired: np.ndarray, /):
    """Like `numpy.testing.assert_allclose` but takes dtype into account for relative
    tolerance."""
    if actual.dtype != desired.dtype:
        raise TypeError(f"dtypes do not match `{actual.dtype}` != `{desired.dtype}`")
    kwargs = {}
    if actual.dtype == np.float16:
        kwargs = {"rtol": 1e-3}
    elif actual.dtype == np.float32:
        kwargs = {"rtol": 1e-5}
    elif actual.dtype == np.float64:
        kwargs = {"rtol": 1e-7}

    np.testing.assert_allclose(actual, desired, **kwargs)  # type: ignore


test_abs_v13 = unary_element_wise_test(
    "Abs", 13, op17.abs, np.testing.assert_array_equal
)
test_acos_v7 = unary_element_wise_test("Acos", 7, op17.acos, assert_allclose)
test_acosh_v9 = unary_element_wise_test("Acosh", 9, op17.acos, assert_allclose)
test_asin_v7 = unary_element_wise_test("Asin", 7, op17.asin, assert_allclose)
test_asinh_v9 = unary_element_wise_test("Asinh", 9, op17.asin, assert_allclose)
test_atan_v7 = unary_element_wise_test("Atan", 7, op17.atan, assert_allclose)
test_atanh_v9 = unary_element_wise_test("Atanh", 9, op17.atan, assert_allclose)

test_ceil_v13 = unary_element_wise_test(
    "Ceil", 13, op17.ceil, np.testing.assert_array_equal
)
test_cos_v7 = unary_element_wise_test("Cos", 7, op17.cos, assert_allclose)
test_cosh_v9 = unary_element_wise_test("Cosh", 9, op17.cos, assert_allclose)
test_erf_v13 = unary_element_wise_test("Erf", 13, op17.erf, assert_allclose)
test_exp_v13 = unary_element_wise_test("Exp", 13, op17.exp, assert_allclose)
test_floor_v13 = unary_element_wise_test(
    "Floor", 13, op17.floor, np.testing.assert_array_equal
)
test_hard_swish_v13 = unary_element_wise_test(
    "HardSwish", 14, op17.exp, assert_allclose
)
test_hardmax_v13 = unary_element_wise_test_with_axis_param(
    "Hardmax", 13, op17.hardmax, assert_allclose
)
test_identity_v16 = unary_element_wise_test(
    "Identity", 16, op17.identity, np.testing.assert_array_equal, param_name="V"
)
test_isinf_v10 = unary_element_wise_test(
    "IsInf", 10, op17.isinf, np.testing.assert_array_equal, param_name="T1"
)
test_isinf_v20 = unary_element_wise_test(
    "IsInf", 20, op20.isinf, np.testing.assert_array_equal, param_name="T1"
)
test_isnan_v13 = unary_element_wise_test(
    "IsNaN", 13, op17.isnan, np.testing.assert_array_equal, param_name="T1"
)
test_isnan_v20 = unary_element_wise_test(
    "IsNaN", 20, op20.isnan, np.testing.assert_array_equal, param_name="T1"
)
test_log_v13 = unary_element_wise_test("Log", 13, op17.log, assert_allclose)
test_log_softmax_v13 = unary_element_wise_test_with_axis_param(
    "LogSoftmax", 13, op17.log_softmax, assert_allclose
)
test_neg_v13 = unary_element_wise_test("Neg", 13, op17.neg, assert_allclose)
test_not_v1 = unary_element_wise_test(
    "Not", 1, op17.not_, np.testing.assert_array_almost_equal
)

test_reciprocal_v13 = unary_element_wise_test(
    "Reciprocal", 13, op17.reciprocal, assert_allclose
)
test_round_v11 = unary_element_wise_test(
    "Round", 11, op17.round, np.testing.assert_array_equal
)
# Bug in reference of sigmoid?
test_sigmoid_v13 = unary_element_wise_test("Sigmoid", 13, op17.sigmoid, assert_allclose)
# Standard does not specify NaN behavior. Presumably it should take the sign-bit, but does not say so.
test_sign_v13 = unary_element_wise_test("Sign", 13, op17.sign, assert_allclose)

test_sin_v7 = unary_element_wise_test("Sin", 7, op17.sin, assert_allclose)
test_sinh_v9 = unary_element_wise_test("Sinh", 9, op17.sinh, assert_allclose)
test_softmax_v13 = unary_element_wise_test_with_axis_param(
    "Softmax", 13, op17.softmax, assert_allclose
)
test_sqrt_v13 = unary_element_wise_test("Sqrt", 13, op17.sqrt, assert_allclose)
test_tan_v7 = unary_element_wise_test("Tan", 7, op17.tan, assert_allclose)
test_tanh_v13 = unary_element_wise_test("Tanh", 13, op17.tanh, assert_allclose)
# TODO: Bug in reference implementation of Mish
test_mish_v18 = unary_element_wise_test("Mish", 18, op18.mish, assert_allclose)


@given(data=st.data())
@pytest.mark.parametrize("dtype", dtypes("HardSigmoid", 6, "T"), ids=str)
def test_hard_sigmoid_v16(data: st.DataObject, dtype: np.dtype):
    array = data.draw(arrays(dtype))

    def do(x: Var) -> Var:
        alpha = data.draw(st.floats(1e-3, 0.3), "alpha")
        beta = data.draw(st.floats(1e-3, 0.6), "beta")
        return op17.hard_sigmoid(x, alpha=alpha, beta=beta)

    assert_unary_against_reference(do, array, test=assert_allclose)


@given(data=st.data())
@pytest.mark.parametrize("dtype", dtypes("LeakyRelu", 16, "T"), ids=str)
def test_leaky_relu_v16(data: st.DataObject, dtype: np.dtype):
    array = data.draw(arrays(dtype))

    def do(x: Var) -> Var:
        alpha = data.draw(st.floats(1e-3, 0.1), "alpha")
        return op17.leaky_relu(x, alpha=alpha)

    assert_unary_against_reference(do, array, test=assert_allclose)
