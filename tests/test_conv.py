import numpy as np
import pytest
import spox.opset.ai.onnx.v17 as op17  # Oldest opset currently provided by spox
from hypothesis import given
from hypothesis import strategies as st

from onnx_tests import helpers as h
from onnx_tests.config import run_candidate
from onnx_tests.conv import conv_2d, conv_integer_2d

from .utils import make_test

make_test("Conv", 11, conv_2d, globals())
# TODO: Fix Transpose
# make_test("ConvTranspose", 11, conv_transpose_2d, globals())


@given(data=st.data())
@pytest.mark.parametrize(
    "dtype_x", h.SCHEMAS["ai.onnx"]["ConvInteger"][10].dtype_constraints["T1"], ids=str
)
@pytest.mark.parametrize(
    "dtype_w", h.SCHEMAS["ai.onnx"]["ConvInteger"][10].dtype_constraints["T2"], ids=str
)
def test_conv_integer_10(data: st.DataObject, dtype_x: str, dtype_w: str):
    model = data.draw(
        conv_integer_2d(dtype_x=np.dtype(dtype_x), dtype_w=np.dtype(dtype_w), op=op17),
    ).build_model()

    (candidate,) = run_candidate(model).values()
    (expected,) = h.run_reference(model).values()

    h.assert_allclose(candidate, expected)
