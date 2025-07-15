from types import ModuleType

import numpy as np
from hypothesis import strategies as st

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw
from onnx_tests._kernel_op import kernel_operation


@st.composite
def average_pool(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    kernel_op = draw(kernel_operation())
    N = draw(st.integers(min_value=1, max_value=2))
    C = draw(st.integers(min_value=1, max_value=3))

    x = draw(h.arrays(dtype=dtype, shape=(N, C) + kernel_op.input_spacial_shape))

    count_include_pad = draw(st.sampled_from([0, 1]), "count_include_pad")
    # TODO: ceil_mode is mutually exclusive with auto_pad, but the ONNX standard does not say so.
    if kernel_op.auto_pad == "NOTSET":
        ceil_mode = draw(st.sampled_from([0, 1]), "ceil_mode")
    else:
        ceil_mode = 0

    return TestCaseDraw(
        inputs={"X": x},
        attribute_kwargs={
            **kernel_op.attribute_kwargs(),
            "ceil_mode": ceil_mode,
            "count_include_pad": count_include_pad,
        },
        spox_fun=op.average_pool,
    )
