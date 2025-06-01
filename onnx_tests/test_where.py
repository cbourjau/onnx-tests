import numpy as np
import pytest
import spox
import spox.opset.ai.onnx.v21 as op21
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn

from . import helpers as h
from .config import run_candidate


@given(data=st.data())
@pytest.mark.parametrize(
    "dtype", h.SCHEMAS["ai.onnx"]["Where"][16].dtype_constraints["T"], ids=str
)
def test_where_v16(data, dtype: np.dtype):
    shapes = data.draw(hyn.mutually_broadcastable_shapes(num_shapes=3))

    cond = data.draw(h.arrays(np.dtype(bool), shape=shapes.input_shapes[0]))
    x = data.draw(h.arrays(dtype, shape=shapes.input_shapes[1]))
    y = data.draw(h.arrays(dtype, shape=shapes.input_shapes[2]))

    res = op21.where(cond.spox_argument, x.spox_argument, y.spox_argument)
    model = spox.build(
        {"cond": cond.spox_argument, "x": x.spox_argument, "y": y.spox_argument},
        {"res": res},
    )

    candidate, *_ = run_candidate(model, cond=cond.array, x=x.array, y=y.array).values()
    # ONNX standard explicitly references `numpy.where` semantics
    expected = np.where(cond.array, x.array, y.array)

    np.testing.assert_array_equal(candidate, expected)
