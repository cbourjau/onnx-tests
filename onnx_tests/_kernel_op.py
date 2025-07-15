from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast, get_args

from hypothesis import strategies as st

AutoPad = Literal["NOTSET", "SAME_UPPER", "SAME_LOWER", "VALID"]


@dataclass
class KernelOperation:
    dilations: tuple[int, int]
    strides: tuple[int, int]
    pads: list[int] | None
    auto_pad: str
    input_spacial_shape: tuple[int, int]
    kernel_spacial_shape: tuple[int, int]

    def attribute_kwargs(self) -> dict[str, Any]:
        """Convert object to attribute kwargs expected by a corresponding constructor
        function."""
        return {
            "dilations": self.dilations,
            "strides": self.strides,
            "pads": self.pads,
            "auto_pad": self.auto_pad,
            # Rename kernel shape and leave out input shape
            "kernel_shape": self.kernel_spacial_shape,
        }


@st.composite
def kernel_operation(draw: st.DrawFn) -> KernelOperation:
    """Reusable strategy for kernel operations."""
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

    dilation = draw(
        st.tuples(
            st.integers(min_value=1, max_value=2), st.integers(min_value=1, max_value=2)
        )
    )
    x_h, pads_h = draw(_length_and_pads(k_h, dilation[0], strides[0], auto_pad))
    x_w, pads_w = draw(_length_and_pads(k_w, dilation[1], strides[1], auto_pad))

    if auto_pad == "NOTSET":
        pads = [pads_h[0], pads_w[0], pads_h[1], pads_w[1]]  # type: ignore
    else:
        pads = None

    return KernelOperation(
        dilations=dilation,
        strides=strides,
        pads=pads,
        auto_pad=auto_pad,
        input_spacial_shape=(x_h, x_w),
        kernel_spacial_shape=(k_h, k_w),
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
                min_value=0,
                max_value=min(total_length - effective_k - 1, effective_k - 1),
            )
        )
        pad_end = draw(
            st.integers(
                min_value=0,
                max_value=min(
                    total_length - pad_start - effective_k - 1, effective_k - 1
                ),
            )
        )
        x_length = total_length - pad_start - pad_end
        pads = (pad_start, pad_end)
    else:
        misalign = draw(st.integers(min_value=0, max_value=effective_k - 1))
        x_length = total_length + misalign

    return (x_length, pads)
