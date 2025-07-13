from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast, get_args

import numpy as np
import onnx
import spox
from hypothesis import strategies as st

from . import helpers as h

AutoPad = Literal["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"]


@dataclass
class ConvDraw:
    """Data class containing all the state of a test case.

    Instead of having this class, we could also return a ready-made model, but that does
    not allow for a good debugging experience.
    """

    inputs: dict[str, np.ndarray | None]
    params: ConvParams

    op_name: Literal["Conv", "ConvInteger", "ConvTranspose"]

    def build_model(self, op) -> onnx.ModelProto:
        try:
            input_vars = {
                k: None if v is None else op.const(v) for k, v in self.inputs.items()
            }
        except Exception:
            breakpoint()

        if self.op_name == "Conv":
            fun = op.conv
        elif self.op_name == "ConvInteger":
            fun = op.conv_integer
        elif self.op_name == "ConvTranspose":
            fun = op.conv_transpose
        else:
            raise NotImplementedError
        res = fun(**input_vars, **self.params.__dict__)

        return spox.build({}, {"res": res})


@dataclass
class ConvParams:
    strides: tuple[int, ...]
    pads: list[int] | None
    auto_pad: str
    group: int
    kernel_shape: list[int]


@st.composite
def conv_2d(draw: st.DrawFn, dtype: np.dtype) -> ConvDraw:
    x, w, params = draw(_conv_generic(dtype_x=dtype, dtype_w=dtype))
    b = draw(
        st.one_of([st.none(), h.arrays(dtype, (w.shape[0],)).map(lambda el: el.array)])
    )
    inputs = {"X": x, "W": w, "B": b}
    return ConvDraw(inputs=inputs, params=params, op_name="Conv")


@st.composite
def conv_transpose_2d(draw: st.DrawFn, dtype: np.dtype) -> ConvDraw:
    x, w, params = draw(_conv_generic(dtype_x=dtype, dtype_w=dtype))
    b = draw(
        st.one_of([st.none(), h.arrays(dtype, (w.shape[0],)).map(lambda el: el.array)])
    )
    inputs = {"X": x, "W": w, "B": b}
    return ConvDraw(inputs=inputs, params=params, op_name="ConvTranspose")


@st.composite
def conv_integer_2d(draw: st.DrawFn, dtype_x: np.dtype, dtype_w: np.dtype) -> ConvDraw:
    x, w, params = draw(_conv_generic(dtype_x=dtype_x, dtype_w=dtype_w))
    x_zero_point = draw(
        st.one_of([st.none(), h.arrays(dtype_x, ()).map(lambda el: el.array)])
    )
    w_zero_point = draw(
        st.one_of(
            [
                st.none(),
                h.arrays(dtype_w, ()).map(lambda el: el.array),
                h.arrays(dtype_w, (w.shape[0],)).map(lambda el: el.array),
            ]
        )
    )
    inputs = {
        "x": x,
        "w": w,
        "x_zero_point": x_zero_point,
        "w_zero_point": w_zero_point,
    }
    return ConvDraw(inputs=inputs, params=params, op_name="ConvInteger")


@st.composite
def _conv_generic(
    draw: st.DrawFn, dtype_x: np.dtype, dtype_w: np.dtype
) -> tuple[np.ndarray, np.ndarray, ConvParams]:
    strides = draw(
        st.tuples(
            st.integers(min_value=1, max_value=2), st.integers(min_value=1, max_value=2)
        ),
        "strides",
    )
    auto_pad = cast(AutoPad, draw(st.sampled_from(get_args(AutoPad))))
    k_h, k_w = draw(
        st.tuples(
            st.integers(min_value=1, max_value=5), st.integers(min_value=1, max_value=5)
        ),
        "kernel_shape",
    )

    dilation = draw(st.integers(min_value=1, max_value=2))
    x_h, pads_h = draw(_length_and_pads(k_h, dilation, strides[0], auto_pad))
    x_w, pads_w = draw(_length_and_pads(k_w, dilation, strides[1], auto_pad))

    if auto_pad == "NOTSET":
        pads = [pads_h[0], pads_w[0], pads_h[1], pads_w[1]]  # type: ignore
    else:
        pads = None

    N = draw(st.integers(min_value=1, max_value=2))
    ch_per_group = draw(st.integers(min_value=1, max_value=2))
    group = draw(st.integers(min_value=1, max_value=2))
    C = ch_per_group * group

    M = draw(st.integers(min_value=1, max_value=2)) * group
    x = draw(h.arrays(dtype=dtype_x, shape=(N, C, x_h, x_w))).array
    w = draw(h.arrays(dtype=dtype_w, shape=(M, ch_per_group, k_h, k_w))).array

    return (
        x,
        w,
        ConvParams(
            strides=strides,
            pads=pads,
            auto_pad=auto_pad,
            group=group,
            # Must be present or else onnx shape inference fails.
            # `kernel_shape` only contains spatial parameters
            kernel_shape=list(w.shape[2:]),
        ),
    )


@st.composite
def _length_and_pads(
    draw: st.DrawFn, k: int, dilation: int, stride: int, auto_pad: AutoPad
) -> tuple[int, tuple[int, int] | None]:
    effective_k = k + (k - 1) * (dilation - 1)
    n_steps = draw(st.integers(min_value=1, max_value=3))
    total_length = n_steps * stride + effective_k

    pads = None
    if auto_pad == "NOTSET":
        pad_start = draw(
            st.integers(
                min_value=0, max_value=min(total_length - effective_k, effective_k)
            )
        )
        pad_end = draw(
            st.integers(
                min_value=0,
                max_value=min(total_length - pad_start - effective_k, effective_k),
            )
        )
        x_length = total_length - pad_start - pad_end
        pads = (pad_start, pad_end)
    else:
        misalign = draw(st.integers(min_value=0, max_value=effective_k - 1))
        x_length = total_length + misalign

    if x_length == 0:
        breakpoint()
    return (x_length, pads)
