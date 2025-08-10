import numpy as np
import pytest
import spox.opset.ai.onnx.v21 as op
from hypothesis import given
from hypothesis import strategies as st

from onnx_tests import helpers as h
from onnx_tests import manipulation_functions
from onnx_tests.config import run_candidate
from onnx_tests.runtime_wrappers import run_reference

from .utils import make_test


@given(data=st.data())
@pytest.mark.parametrize(
    "dtype_out",
    h.SCHEMAS["ai.onnx"]["Cast"][21].dtype_constraints["T2"],
    ids=lambda el: f"to-{el}",
)
@pytest.mark.parametrize(
    "dtype_in",
    h.SCHEMAS["ai.onnx"]["Cast"][21].dtype_constraints["T1"],
    ids=lambda el: f"from-{el}",
)
def test_Cast_21(data: st.DataObject, dtype_in, dtype_out):  # noqa
    state = data.draw(manipulation_functions.cast(dtype_in, dtype_out, op))
    model = state.build_model()

    (cand,) = run_candidate(model).values()
    (exp,) = run_reference(model).values()

    if exp.dtype.kind in "UO":
        # Strings must match exactly
        np.testing.assert_array_equal(cand, exp)
    else:
        h.assert_allclose(cand, exp)


make_test("Concat", 13, manipulation_functions.concat, globals())
make_test("Compress", 11, manipulation_functions.compress, globals())
