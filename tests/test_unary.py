from collections.abc import Callable

import hypothesis.extra.numpy as hyn
import numpy as np
import pytest
import spox
import spox.opset.ai.onnx.v21 as op21
from hypothesis import given
from hypothesis import strategies as st
from spox import Var

from . import helpers as h


def assert_unary(
    spox_fun: Callable[[Var], Var],
    x: h.ArrayWrapper,
    almost_equal: bool,
):
    model = spox.build({"x": x.spox_argument}, {"res": spox_fun(x.spox_argument)})

    expected, *_ = h.run(model, x=x.array).values()
    candidate, *_ = h.run_reference(model, x=x.array).values()

    if almost_equal:
        np.testing.assert_array_almost_equal(candidate, expected)
    else:
        np.testing.assert_array_equal(candidate, expected)


def arrays(dtype: np.dtype) -> st.SearchStrategy[h.ArrayWrapper]:
    return h.arrays(
        dtype,
        hyn.array_shapes(
            min_dims=0,
        ),
    )


@given(data=st.data())
@pytest.mark.parametrize("dtype", h.NUMERIC_DTYPES, ids=str)
def test_abs(data, dtype: np.dtype):
    array = data.draw(arrays(dtype))
    assert_unary(op21.abs, array, almost_equal=False)


@given(data=st.data())
@pytest.mark.parametrize("dtype", h.FLOAT_DTYPES, ids=str)
def test_sin(data, dtype: np.dtype):
    array = data.draw(arrays(dtype))
    assert_unary(op21.sin, array, almost_equal=True)


@given(data=st.data())
@pytest.mark.parametrize("dtype", h.FLOAT_DTYPES, ids=str)
def test_cos(data, dtype: np.dtype):
    array = data.draw(arrays(dtype))
    assert_unary(op21.cos, array, almost_equal=True)
