import pytest
import spox.opset.ai.onnx.v21 as op
from hypothesis import given
from hypothesis import strategies as st

from onnx_tests import helpers as h
from onnx_tests import statistical_ops as strats
from onnx_tests.config import run_candidate
from onnx_tests.runtime_wrappers import run_reference

from .utils import make_test


@given(data=st.data())
@pytest.mark.parametrize(
    "dtype",
    h.SCHEMAS["ai.onnx"]["CumSum"][14].dtype_constraints["T"],
    ids=lambda el: f"{el}",
)
@pytest.mark.parametrize(
    "dtype_axis",
    h.SCHEMAS["ai.onnx"]["CumSum"][14].dtype_constraints["T2"],
    ids=lambda el: f"axis-dtype-{el}",
)
def test_CumSum_14(data: st.DataObject, dtype, dtype_axis):  # noqa
    state = data.draw(strats.cumsum(dtype, dtype_axis, op))
    model = state.build_model()

    (cand,) = run_candidate(model).values()
    (exp,) = run_reference(model).values()

    h.assert_allclose(cand, exp)


make_test("ReduceL1", 18, strats.reduce_l1, globals())
make_test("ReduceL2", 18, strats.reduce_l2, globals())
make_test("ReduceLogSum", 18, strats.reduce_log_sum, globals())
make_test("ReduceLogSumExp", 18, strats.reduce_log_sum_exp, globals())
make_test("ReduceMax", 18, strats.reduce_max, globals())
make_test("ReduceMax", 20, strats.reduce_max, globals())
make_test("ReduceMean", 18, strats.reduce_mean, globals())
make_test("ReduceMin", 18, strats.reduce_min, globals())
make_test("ReduceMin", 20, strats.reduce_min, globals())
make_test("ReduceProd", 18, strats.reduce_prod, globals())
make_test("ReduceSum", 13, strats.reduce_sum, globals())
make_test("ReduceSumSquare", 18, strats.reduce_sum_square, globals())
