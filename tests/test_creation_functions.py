import numpy as np
import pytest
import spox.opset.ai.onnx.v17 as op17
from hypothesis import given
from hypothesis import strategies as st

from onnx_tests import creation_functions as strats
from onnx_tests import helpers as h
from onnx_tests.config import run_candidate
from onnx_tests.runtime_wrappers import run_reference

from .utils import make_test

make_test("ConstantOfShape", 21, strats.constant_of_shape, globals(), type_var="T2")


@given(data=st.data())
@pytest.mark.parametrize(
    "dtype_in",
    h.SCHEMAS["ai.onnx"]["EyeLike"][9].dtype_constraints["T1"],
    ids=lambda el: f"in-{el}",
)
@pytest.mark.parametrize(
    "dtype_out",
    h.SCHEMAS["ai.onnx"]["EyeLike"][9].dtype_constraints["T2"],
    ids=lambda el: f"out-{el}",
)
def test_EyeLike_9(data: st.DataObject, dtype_in, dtype_out):  # noqa
    state = data.draw(strats.eye_like(np.dtype(dtype_in), np.dtype(dtype_out), op17))
    model = state.build_model()

    (cand,) = run_candidate(model).values()
    (exp,) = run_reference(model).values()

    h.assert_allclose(cand, exp)


# OneHot has 3 independent type constraints (T1 x T2 x T3 = 1573 combos).
# TODO: File upstream issue
# @given(data=st.data())
# @pytest.mark.parametrize(
#     "dtype_indices",
#     h.SCHEMAS["ai.onnx"]["OneHot"][11].dtype_constraints["T1"],
#     ids=lambda el: f"idx-{el}",
# )
# @pytest.mark.parametrize(
#     "dtype_depth",
#     h.SCHEMAS["ai.onnx"]["OneHot"][11].dtype_constraints["T2"],
#     ids=lambda el: f"depth-{el}",
# )
# @pytest.mark.parametrize(
#     "dtype_values",
#     h.SCHEMAS["ai.onnx"]["OneHot"][11].dtype_constraints["T3"],
#     ids=lambda el: f"val-{el}",
# )
# def test_OneHot_11(data: st.DataObject, dtype_indices, dtype_depth, dtype_values):  # noqa
#     state = data.draw(
#         strats.one_hot(
#             np.dtype(dtype_indices), np.dtype(dtype_depth), np.dtype(dtype_values), op17
#         )
#     )
#     model = state.build_model()
#
#     (cand,) = run_candidate(model).values()
#     (exp,) = run_reference(model).values()
#
#     h.assert_allclose(cand, exp)
