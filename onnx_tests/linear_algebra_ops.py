from types import ModuleType

import numpy as np
from hypothesis import strategies as st

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw


@st.composite
def matmul(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    shapes = draw(h.matmul_shapes())
    x = draw(h.arrays(dtype, shape=shapes[0]))
    y = draw(h.arrays(dtype, shape=shapes[1]))

    return TestCaseDraw(inputs=[x, y], attribute_kwargs={}, spox_fun=op.matmul)
