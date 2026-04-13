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
    if (dtype_in.kind, dtype_out.kind) in [("b", "U"), ("U", "b")]:
        raise pytest.skip(reason="string representation of boolean values is undefined")
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
make_test("Flatten", 13, manipulation_functions.flatten, globals())
make_test("Reshape", 21, manipulation_functions.reshape, globals())
make_test("Squeeze", 13, manipulation_functions.squeeze, globals())
make_test("Unsqueeze", 13, manipulation_functions.unsqueeze, globals())
make_test("Transpose", 13, manipulation_functions.transpose, globals())
make_test("Expand", 13, manipulation_functions.expand, globals())
make_test("Tile", 13, manipulation_functions.tile, globals())
make_test("Gather", 13, manipulation_functions.gather, globals())
make_test("Split", 18, manipulation_functions.split, globals())
make_test("Pad", 21, manipulation_functions.pad, globals())
make_test("Trilu", 14, manipulation_functions.trilu, globals())
make_test("NonZero", 13, manipulation_functions.non_zero, globals())
make_test("GatherElements", 13, manipulation_functions.gather_elements, globals())
make_test("DepthToSpace", 13, manipulation_functions.depth_to_space, globals())
make_test("SpaceToDepth", 13, manipulation_functions.space_to_depth, globals())
