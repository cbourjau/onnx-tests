from types import ModuleType

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw


@st.composite
def size(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    shape = draw(hyn.array_shapes(min_side=0, min_dims=0))
    data = draw(h.arrays(dtype=dtype, shape=shape))
    return TestCaseDraw(inputs={"data": data}, attribute_kwargs={}, spox_fun=op.size)
