from collections.abc import Callable
from typing import NamedTuple

import hypothesis.extra.numpy as hyn
import numpy as np
import pytest
import spox
import spox.opset.ai.onnx.v17 as op17
from hypothesis import given
from hypothesis import strategies as st
from spox import Var

from . import helpers as h


class Params(NamedTuple):
    data: h.ArrayWrapper
    axis: int
    keepdims: int
    select_last_index: int


def make_params(data: st.DataObject, dtype: np.dtype) -> Params:
    # TODO: The standard does not describe the behavior for rank-0 or 0-sized arrays
    array = data.draw(h.arrays(dtype, shape=hyn.array_shapes(min_dims=1, min_side=1)))

    return Params(
        data=array,
        axis=data.draw(st.integers(0, array.array.ndim - 1), label="axis"),
        keepdims=data.draw(st.sampled_from([0, 1]), label="keepdims"),
        select_last_index=data.draw(st.sampled_from([0, 1]), label="select_last_index"),
    )


def assert_against_reference(
    spox_fun: Callable[[Var], Var],
    x: h.ArrayWrapper,
    test: Callable[[np.ndarray, np.ndarray], None],
):
    model = spox.build({"x": x.spox_argument}, {"res": spox_fun(x.spox_argument)})

    expected, *_ = h.run(model, x=x.array).values()
    candidate, *_ = h.run_reference(model, x=x.array).values()

    test(candidate, expected)


@given(data=st.data())
@pytest.mark.parametrize("dtype", h.NUMERIC_DTYPES, ids=str)
def test_argmax_v13(data, dtype):
    # TODO: onnxruntime and reference produce yield different results with NaN values
    params = make_params(data, dtype)

    def do(data: Var) -> Var:
        return op17.arg_max(
            data=data,
            axis=params.axis,
            keepdims=params.keepdims,
            select_last_index=params.select_last_index,
        )

    assert_against_reference(do, params.data, np.testing.assert_array_equal)


@given(data=st.data())
@pytest.mark.parametrize("dtype", h.NUMERIC_DTYPES, ids=str)
def test_argmin_v13(data, dtype):
    # TODO: onnxruntime and reference produce yield different results with NaN values
    params = make_params(data, dtype)

    def do(data: Var) -> Var:
        return op17.arg_min(
            data=data,
            axis=params.axis,
            keepdims=params.keepdims,
            select_last_index=params.select_last_index,
        )

    assert_against_reference(do, params.data, np.testing.assert_array_equal)
