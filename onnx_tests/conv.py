from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Literal

import numpy as np
from hypothesis import strategies as st

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw
from onnx_tests._kernel_op import kernel_operation

AutoPad = Literal["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"]


@dataclass
class ConvKernelOperation:
    dilations: tuple[int, int]
    strides: tuple[int, int]
    pads: list[int] | None
    auto_pad: str
    kernel_shape: tuple[int, int]
    group: int


@st.composite
def conv_2d(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    x, w, params = draw(_conv_generic(dtype_x=dtype, dtype_w=dtype))
    b = draw(st.one_of([st.none(), h.arrays(dtype, (w.shape[0],))]))
    inputs = {"X": x, "W": w, "B": b}
    return TestCaseDraw(
        inputs=inputs, attribute_kwargs=params.__dict__, spox_fun=op.conv
    )


@st.composite
def conv_transpose_2d(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    x, w, params = draw(_conv_generic(dtype_x=dtype, dtype_w=dtype))
    b = draw(st.one_of([st.none(), h.arrays(dtype, (w.shape[0],))]))
    inputs = {"X": x, "W": w, "B": b}
    return TestCaseDraw(
        inputs=inputs,
        attribute_kwargs=params.__dict__,
        spox_fun=op.conv_transpose,
    )


@st.composite
def conv_integer_2d(
    draw: st.DrawFn, dtype_x: np.dtype, dtype_w: np.dtype, op: ModuleType
) -> TestCaseDraw:
    x, w, params = draw(_conv_generic(dtype_x=dtype_x, dtype_w=dtype_w))
    x_zero_point = draw(st.one_of([st.none(), h.arrays(dtype_x, ())]))
    w_zero_point = draw(
        st.one_of(
            [
                st.none(),
                h.arrays(dtype_w, ()),
                h.arrays(dtype_w, (w.shape[0],)),
            ]
        )
    )
    inputs = {
        "x": x,
        "w": w,
        "x_zero_point": x_zero_point,
        "w_zero_point": w_zero_point,
    }
    return TestCaseDraw(
        inputs=inputs,
        attribute_kwargs=params.__dict__,
        spox_fun=op.conv_transpose,
    )


@st.composite
def _conv_generic(
    draw: st.DrawFn, dtype_x: np.dtype, dtype_w: np.dtype
) -> tuple[np.ndarray, np.ndarray, ConvKernelOperation]:
    kernel_op = draw(kernel_operation())

    N = draw(st.integers(min_value=1, max_value=2))

    ch_per_group = draw(st.integers(min_value=1, max_value=2))
    group = draw(st.integers(min_value=1, max_value=2))
    C = ch_per_group * group

    M = draw(st.integers(min_value=1, max_value=2)) * group
    x = draw(h.arrays(dtype=dtype_x, shape=(N, C) + kernel_op.input_spacial_shape))
    w = draw(
        h.arrays(
            dtype=dtype_w, shape=(M, ch_per_group) + kernel_op.kernel_spacial_shape
        )
    )

    conv_kernel_op = ConvKernelOperation(group=group, **kernel_op.attribute_kwargs())

    return (x, w, conv_kernel_op)


__all__ = ["conv_2d", "conv_integer_2d", "conv_transpose_2d"]
