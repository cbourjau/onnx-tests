import numpy as np
import pytest
import spox.opset.ai.onnx.v17 as op17  # Oldest opset currently provided by spox
from hypothesis import given
from hypothesis import strategies as st

from . import helpers as h
from .config import run_candidate
from .conv import conv_2d, conv_integer_2d, conv_transpose_2d


@given(data=st.data())
@pytest.mark.parametrize(
    "dtype", h.SCHEMAS["ai.onnx"]["Conv"][11].dtype_constraints["T"], ids=str
)
def test_conv_11(data: st.DataObject, dtype: str):
    # Opsets reexport earlier definitions if there has not been an
    # update. In the case of `Conv` this means that opset 17
    # re-exports the definition of opset 11

    model = data.draw(conv_2d(np.dtype(dtype)), label="full-state").build_model(op17)

    (candidate,) = run_candidate(model).values()
    (expected,) = h.run_reference(model).values()

    h.assert_allclose(candidate, expected)


@given(data=st.data())
@pytest.mark.parametrize(
    "dtype", h.SCHEMAS["ai.onnx"]["ConvTranspose"][11].dtype_constraints["T"], ids=str
)
def test_conv_transpose_11(data: st.DataObject, dtype: str):
    # Opsets reexport earlier definitions if there has not been an
    # update. In the case of `Conv` this means that opset 17
    # re-exports the definition of opset 11

    model = data.draw(
        conv_transpose_2d(np.dtype(dtype)), label="full-state"
    ).build_model(op17)

    (candidate,) = run_candidate(model).values()
    (expected,) = h.run_reference(model).values()

    h.assert_allclose(candidate, expected)


@pytest.mark.skip(reason="Work in progress")
@given(data=st.data())
@pytest.mark.parametrize(
    "dtype_x", h.SCHEMAS["ai.onnx"]["ConvInteger"][10].dtype_constraints["T1"], ids=str
)
@pytest.mark.parametrize(
    "dtype_w", h.SCHEMAS["ai.onnx"]["ConvInteger"][10].dtype_constraints["T2"], ids=str
)
def test_conv_integer_10(data: st.DataObject, dtype_x: str, dtype_w: str):
    model = data.draw(
        conv_integer_2d(dtype_x=np.dtype(dtype_x), dtype_w=np.dtype(dtype_w)),
        label="full-state",
    ).build_model(op17)

    (candidate,) = run_candidate(model).values()
    (expected,) = h.run_reference(model).values()

    h.assert_allclose(candidate, expected)
