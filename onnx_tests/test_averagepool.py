from typing import NamedTuple

import numpy as np
import pytest
import spox
import spox.opset.ai.onnx.v19 as op19
from hypothesis import given
from hypothesis import strategies as st

from . import helpers as h
from .config import run_candidate
from .runtime_wrappers import run_reference


class AveragePool2DTestCase(NamedTuple):
    x: h.ArrayWrapper
    strides: tuple[int, ...]
    pads: list[int] | None
    auto_pad: str
    kernel_shape: list[int]
    ceil_mode: int
    count_include_pad: int
    dilations: list[int] | None


def avg_pool_kwargs(data: st.DataObject, dtype: np.dtype) -> AveragePool2DTestCase:
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

    strides = data.draw(
        st.tuples(
            st.integers(min_value=1, max_value=2), st.integers(min_value=1, max_value=2)
        ),
        "strides",
    )
    count_include_pad = data.draw(st.sampled_from([0, 1]), "count_include_pad")
    if data.draw(st.booleans(), "with-explicit-padding"):
        # TODO: Padding should be smaller than kernel, but the ONNX
        # standard makes no such recommendation.
        pads = list(
            data.draw(
                st.tuples(
                    st.integers(min_value=0, max_value=k_h - 1),
                    st.integers(min_value=0, max_value=k_h - 1),
                    st.integers(min_value=0, max_value=k_w - 1),
                    st.integers(min_value=0, max_value=k_w - 1),
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
    # TODO: ceil_mode is mutually exclusive with auto_pad, but the ONNX standard does not say so.
    if auto_pad == "NOTSET":
        ceil_mode = data.draw(st.sampled_from([0, 1]), "ceil_mode")
    else:
        ceil_mode = 0

    dilations = None
    if data.draw(st.booleans(), "with-explicit-dilation"):
        dilations = list(
            data.draw(
                st.tuples(
                    st.integers(min_value=1, max_value=2),
                    st.integers(min_value=1, max_value=2),
                ),
                label="dilations",
            )
        )

    return AveragePool2DTestCase(
        x=x,
        strides=strides,
        pads=pads,
        auto_pad=auto_pad,
        kernel_shape=[k_h, k_w],
        ceil_mode=ceil_mode,
        count_include_pad=count_include_pad,
        dilations=dilations,
    )


@given(data=st.data())
@pytest.mark.parametrize(
    "dtype", h.SCHEMAS["ai.onnx"]["AveragePool"][19].dtype_constraints["T"], ids=str
)
def test_average_pool_19(data: st.DataObject, dtype: str):
    kwargs = avg_pool_kwargs(data, np.dtype(dtype))

    x = kwargs.x
    res = op19.average_pool(
        x.spox_argument,
        strides=kwargs.strides,
        pads=kwargs.pads,
        auto_pad=kwargs.auto_pad,
        kernel_shape=kwargs.kernel_shape,
        ceil_mode=kwargs.ceil_mode,
        count_include_pad=kwargs.count_include_pad,
        dilations=kwargs.dilations,
    )
    model = spox.build({"X": x.spox_argument}, {"res": res})

    array_args = {"X": x.array}
    expected, *_ = run_reference(model, **array_args).values()
    candidate, *_ = run_candidate(model, **array_args).values()

    np.testing.assert_equal(candidate, expected)
