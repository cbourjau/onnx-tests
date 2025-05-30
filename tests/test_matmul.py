import numpy as np
import pytest
import spox.opset.ai.onnx.v17 as op17  # Oldest opset currently provided by spox
from hypothesis import given
from hypothesis import strategies as st

from . import helpers as h


@given(data=st.data(), shapes=h.matmul_shapes())
@pytest.mark.parametrize("dtype", h.FLOAT_DTYPES, ids=str)
def test_matmul_13(data, dtype: np.dtype, shapes):
    op = op17
    array1 = data.draw(h.arrays(dtype, shape=shapes[0]))
    array2 = data.draw(h.arrays(dtype, shape=shapes[1]))

    # Standard explicitly defines behavior by NumPy semantics
    h.assert_binary_numpy(np.matmul, op.matmul, array1, array2)
