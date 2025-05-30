from collections.abc import Callable

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


def assert_unary_against_reference(
    spox_fun: Callable[[Var], Var],
    x: h.ArrayWrapper,
    test: Callable[[np.ndarray, np.ndarray], None],
):
    model = spox.build({"x": x.spox_argument}, {"res": spox_fun(x.spox_argument)})

    expected, *_ = h.run(model, x=x.array).values()
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


def unary_element_wise_test(
    dtypes: list[np.dtype],
    op: Callable[[Var], Var],
    test: Callable[[np.ndarray, np.ndarray], None] = np.testing.assert_almost_equal,
) -> Callable:
    @given(data=st.data())
    @pytest.mark.parametrize("dtype", dtypes, ids=str)
    def test_function(data, dtype: np.dtype):
        array = data.draw(arrays(dtype))
        assert_unary_against_reference(op, array, test=test)

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
    h.NUMERIC_DTYPES, op17.abs, np.testing.assert_array_equal
)
test_acos_v7 = unary_element_wise_test(h.FLOAT_DTYPES, op17.acos, assert_allclose)
test_acosh_v9 = unary_element_wise_test(h.FLOAT_DTYPES, op17.acos, assert_allclose)
test_asin_v7 = unary_element_wise_test(h.FLOAT_DTYPES, op17.asin, assert_allclose)
test_asinh_v9 = unary_element_wise_test(h.FLOAT_DTYPES, op17.asin, assert_allclose)
test_atan_v7 = unary_element_wise_test(h.FLOAT_DTYPES, op17.atan, assert_allclose)
test_atanh_v9 = unary_element_wise_test(h.FLOAT_DTYPES, op17.atan, assert_allclose)

test_ceil_v13 = unary_element_wise_test(
    h.FLOAT_DTYPES, op17.ceil, np.testing.assert_array_equal
)
test_cos_v7 = unary_element_wise_test(h.FLOAT_DTYPES, op17.cos, assert_allclose)
test_cosh_v9 = unary_element_wise_test(h.FLOAT_DTYPES, op17.cos, assert_allclose)
test_erf_v13 = unary_element_wise_test(h.NUMERIC_DTYPES, op17.erf, assert_allclose)
test_floor_v13 = unary_element_wise_test(
    h.FLOAT_DTYPES, op17.floor, np.testing.assert_array_equal
)
test_identity_v16 = unary_element_wise_test(
    h.DTYPES, op17.identity, np.testing.assert_array_equal
)
test_isinf_v10 = unary_element_wise_test(
    [np.dtype(np.float32), np.dtype(np.float64)],
    op17.isinf,
    np.testing.assert_array_equal,
)
test_isinf_v20 = unary_element_wise_test(
    h.FLOAT_DTYPES, op20.isinf, np.testing.assert_array_equal
)
test_isnan_v13 = unary_element_wise_test(
    h.FLOAT_DTYPES, op17.isnan, np.testing.assert_array_equal
)
test_log_v13 = unary_element_wise_test(h.FLOAT_DTYPES, op17.log, assert_allclose)
test_reciprocal_v13 = unary_element_wise_test(
    h.FLOAT_DTYPES, op17.reciprocal, assert_allclose
)
test_round_v11 = unary_element_wise_test(
    h.FLOAT_DTYPES, op17.round, np.testing.assert_array_equal
)
# Bug in reference of sigmoid?
test_sigmoid_v13 = unary_element_wise_test(
    h.FLOAT_DTYPES, op17.sigmoid, assert_allclose
)
# Standard does not specify NaN behavior. Presumably it should take the sign-bit, but does not say so.
test_sign_v13 = unary_element_wise_test(h.FLOAT_DTYPES, op17.sign, assert_allclose)

test_sin_v7 = unary_element_wise_test(h.FLOAT_DTYPES, op17.sin, assert_allclose)
test_sinh_v9 = unary_element_wise_test(h.FLOAT_DTYPES, op17.sinh, assert_allclose)
test_sqrt_v13 = unary_element_wise_test(h.FLOAT_DTYPES, op17.sqrt, assert_allclose)
test_tan_v7 = unary_element_wise_test(h.FLOAT_DTYPES, op17.tan, assert_allclose)
test_tanh_v13 = unary_element_wise_test(h.FLOAT_DTYPES, op17.tanh, assert_allclose)
# Bug in reference implementation of Mish
test_mish_v18 = unary_element_wise_test(h.FLOAT_DTYPES, op18.mish, assert_allclose)
