from typing import NamedTuple

import numpy as np
import pytest
import spox
import spox.opset.ai.onnx.v17 as op17  # Oldest opset currently provided by spox
from hypothesis import given
from hypothesis import strategies as st

from . import helpers as h
from .config import run_candidate


class Conv2DTestCase(NamedTuple):
    x: h.ArrayWrapper
    w: h.ArrayWrapper
    b: h.ArrayWrapper | None
    strides: tuple[int, ...]
    pads: list[int] | None
    auto_pad: str
    group: int
    kernel_shape: list[int]


def conv_kwargs(data: st.DataObject, dtype: np.dtype) -> Conv2DTestCase:
    # TODO: Test different groups
    # TODO: Test dilation
    group = 1
    N, H, W, C, M = data.draw(
        st.tuples(
            st.integers(min_value=1, max_value=2),
            st.integers(min_value=4, max_value=32),
            st.integers(min_value=4, max_value=32),
            st.integers(min_value=1, max_value=3),
            st.integers(min_value=1, max_value=3),  # number of feature maps
        ),
        label="X-shape",
    )
    k_h, k_w = data.draw(
        st.tuples(
            st.integers(min_value=1, max_value=4), st.integers(min_value=1, max_value=4)
        ),
        label="kernel-shape",
    )

    x = data.draw(h.arrays(dtype, shape=(N, C, H, W)))
    w = data.draw(h.arrays(dtype, (M, int(C / group), k_h, k_w)))

    # Optional bias
    b = (
        data.draw(h.arrays(dtype, (M,)))
        if data.draw(st.booleans(), label="with-bias")
        else None
    )

    strides = data.draw(
        st.tuples(
            st.integers(min_value=1, max_value=2), st.integers(min_value=1, max_value=2)
        ),
        "strides",
    )
    if data.draw(st.booleans(), "with-explicit-padding"):
        # Set pads and thus not auto_pad
        pads = list(
            data.draw(
                st.tuples(
                    st.integers(min_value=0, max_value=2),
                    st.integers(min_value=0, max_value=2),
                    st.integers(min_value=0, max_value=2),
                    st.integers(min_value=0, max_value=2),
                ),
                label="pads",
            )
        )
        auto_pad = "NOTSET"
    else:
        pads = None
        auto_pad = data.draw(
            st.sampled_from(["SAME_UPPER", "SAME_LOWER", "VALID"]), "auto_pad"
        )

    return Conv2DTestCase(
        x=x,
        w=w,
        b=b,
        strides=strides,
        pads=pads,
        auto_pad=auto_pad,
        group=group,
        kernel_shape=[k_h, k_w],  # onnx shape inference fails if this is not set
    )


@given(data=st.data())
@pytest.mark.parametrize(
    "dtype", h.SCHEMAS["ai.onnx"]["Conv"][11].dtype_constraints["T"], ids=str
)
def test_conv_11(data: st.DataObject, dtype: str):
    # Opsets reexport earlier definitions if there has not been an
    # update. In the case of `Conv` this means that opset 17
    # re-exports the definition of opset 11
    op = op17

    kwargs = conv_kwargs(data, np.dtype(dtype))

    x = kwargs.x
    w = kwargs.w
    b = kwargs.b
    spox_arguments = {
        "X": x.spox_argument,
        "W": w.spox_argument,
    } | ({} if b is None else {"B": b.spox_argument})
    res = op.conv(
        **spox_arguments,  # type: ignore
        strides=kwargs.strides,
        pads=kwargs.pads,
        auto_pad=kwargs.auto_pad,
        kernel_shape=kwargs.kernel_shape,
    )
    model = spox.build(spox_arguments.copy(), {"res": res})

    array_args = {
        "X": x.array,
        "W": w.array,
    } | ({} if b is None else {"B": b.array})
    expected, *_ = h.run_reference(model, **array_args).values()
    candidate, *_ = run_candidate(model, **array_args).values()

    np.testing.assert_equal(candidate, expected)
