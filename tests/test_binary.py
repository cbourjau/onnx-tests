import numpy as np
import pytest
import spox.opset.ai.onnx.v21 as op21
from hypothesis import given
from hypothesis import strategies as st

from . import helpers as h


@given(data=st.data())
@pytest.mark.parametrize("dtype", h.NUMERIC_DTYPES, ids=str)
def test_add(data, dtype: np.dtype):
    array_tuple = data.draw(h.broadcastable_arrays(dtype))
    h.assert_binary_numpy(np.add, op21.add, *array_tuple)
