from collections.abc import Callable
from typing import NamedTuple, Protocol

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
from .runtime_wrappers import run_ort


class Params(NamedTuple):
    data: h.ArrayWrapper
    axes: Var | None
    keepdims: int
    noop_with_empty_axes: int


def make_params(data: st.DataObject, dtype: np.dtype) -> Params:
    # TODO: onnxruntime fails badly for 0-rank and 0-sized inputs
    # array = data.draw(h.arrays(dtype, shape=hyn.array_shapes(min_dims=0, min_side=0)))
    array = data.draw(h.arrays(dtype, shape=hyn.array_shapes(min_dims=1, min_side=1)))
    axes = []
    for axis in range(array.array.ndim):
        if data.draw(st.booleans(), f"specify-axis-{axis}"):
            choices = [axis, axis - array.array.ndim]
            axes.append(data.draw(st.sampled_from(choices), f"axes-{axis}"))

    rand = data.draw(st.randoms())
    rand.shuffle(axes)

    return Params(
        data=array,
        axes=op17.const(np.asarray(axes, np.int64)) if axes else None,
        keepdims=data.draw(st.sampled_from([0, 1]), "keepdims"),
        noop_with_empty_axes=data.draw(st.sampled_from([0, 1]), "noop_with_empty_axes"),
    )


class ReduceFun(Protocol):
    def __call__(
        self,
        data: Var,
        axes: Var | None = None,
        *,
        keepdims: int = 1,
        noop_with_empty_axes: int = 0,
    ) -> Var: ...


def assert_against_reference(
    reduce_fun: ReduceFun,
    params: Params,
    test: Callable[[np.ndarray, np.ndarray], None],
):
    def do(data: Var, axes: Var | None) -> Var:
        return reduce_fun(
            data=data,
            axes=axes,
            keepdims=params.keepdims,
            noop_with_empty_axes=params.noop_with_empty_axes,
        )

    # Bake axes param into the graph as a constant.
    model = spox.build(
        {"data": params.data.spox_argument},
        {"res": reduce_fun(params.data.spox_argument, params.axes)},
    )

    kwargs = {"data": params.data.array}
    expected, *_ = run_candidate(model, **kwargs).values()
    candidate, *_ = h.run_reference(model, **kwargs).values()

    test(candidate, expected)


# TODO: Understand segfaults of onnxruntime


def make_test(fun: ReduceFun, op_name: str, version: int, skip_ort: bool):
    @given(data=st.data())
    @pytest.mark.parametrize(
        "dtype", h.SCHEMAS["ai.onnx"][op_name][version].dtype_constraints["T"], ids=str
    )
    def test_fun(data, dtype):
        params = make_params(data, dtype)
        assert_against_reference(fun, params, h.assert_allclose)

    if skip_ort:
        return pytest.mark.skipif(
            run_candidate == run_ort, reason="onnxruntime segfaults for some inputs"
        )(test_fun)
    return test_fun


test_reduce_l1_v18 = make_test(op18.reduce_l1, "ReduceL1", 18, False)
test_reduce_l2_v18 = make_test(op18.reduce_l2, "ReduceL2", 18, False)
test_reduce_logsum_v18 = make_test(op18.reduce_log_sum, "ReduceLogSum", 18, False)
test_reduce_logsum_exp_v18 = make_test(
    op18.reduce_log_sum_exp, "ReduceLogSumExp", 18, False
)

test_reduce_max_v18 = make_test(op18.reduce_max, "ReduceMax", 18, False)
test_reduce_max_v20 = make_test(op20.reduce_max, "ReduceMax", 20, False)
test_reduce_mean_v18 = make_test(op18.reduce_mean, "ReduceMean", 18, False)
test_reduce_min_v18 = make_test(op18.reduce_min, "ReduceMin", 18, False)
test_reduce_min_v20 = make_test(op20.reduce_min, "ReduceMin", 20, False)
test_reduce_prod_v18 = make_test(op18.reduce_prod, "ReduceProd", 18, False)
test_reduce_sum_v13 = make_test(op17.reduce_sum, "ReduceSum", 13, False)
test_reduce_sum_square_v18 = make_test(
    op18.reduce_sum_square, "ReduceSumSquare", 18, False
)
