from typing import NamedTuple

import hypothesis.extra.numpy as hyn
import numpy as np
import pytest
import spox
import spox.opset.ai.onnx.v21 as op21
from hypothesis import given
from hypothesis import strategies as st

from . import helpers as h

_MAX_INT64 = np.iinfo(np.int64).max
_MIN_INT64 = np.iinfo(np.int64).min
_MAX_INT32 = np.iinfo(np.int32).max
_MIN_INT32 = np.iinfo(np.int32).min


class SliceInfo(NamedTuple):
    starts: h.ArrayWrapper
    ends: h.ArrayWrapper
    axes: h.ArrayWrapper | None
    steps: h.ArrayWrapper | None


@st.composite
def starts_stops_steps(draw: st.DrawFn, data: np.ndarray) -> SliceInfo:
    starts, stops, steps, axes = [], [], [], []
    index_dtype = draw(st.sampled_from([np.int32, np.int64]))
    max_idx = _MAX_INT64 if index_dtype == np.int64 else _MAX_INT32
    min_idx = _MIN_INT64 if index_dtype == np.int64 else _MIN_INT32

    for axis, dim_len in enumerate(data.shape):
        include_dim = draw(st.booleans())
        if not include_dim:
            # skip this axis
            continue
        # positive indices
        start = draw(st.integers(0, dim_len))
        starts.append(start)
        step = draw(st.sampled_from([-2, -1, 1, 2]))
        steps.append(step)
        stops.append(
            draw(
                st.integers(start, dim_len)
                | (st.just(max_idx) if step > 0 else st.just(min_idx))
            )
        )
        axes.append(axis)

    include_steps = draw(st.booleans())
    include_axes = draw(st.booleans())
    if include_axes:
        shuffle_axes = draw(st.booleans())
        if shuffle_axes:
            axes = np.random.default_rng(seed=42).permutation(axes).tolist()  # type: ignore

    return SliceInfo(
        starts=h.ArrayWrapper(np.asarray(starts, dtype=index_dtype)),
        ends=h.ArrayWrapper(np.asarray(stops, dtype=index_dtype)),
        axes=h.ArrayWrapper(np.asarray(axes, dtype=index_dtype))
        if include_axes
        else None,
        steps=h.ArrayWrapper(np.asarray(steps, dtype=index_dtype))
        if include_steps
        else None,
    )


def slices(info: SliceInfo):
    out = []
    axes = list(range(len(info.starts.array))) if info.axes is None else info.axes.array
    steps = [1 for _ in info.starts.array] if info.steps is None else info.steps.array

    for start, end, step in zip(info.starts.array, info.ends.array, steps):
        out.append(slice(start, end, step))

    if len(out) == 0:
        return out
    # apply axes sorting
    padded_out = [slice(None) for _ in range(np.max(axes) + 1)]
    for ax, s in zip(axes, out):
        padded_out[ax] = s
    return padded_out


@given(data=st.data())
@pytest.mark.parametrize("dtype", h.DTYPES)
def test_slice(data, dtype: np.dtype):
    shape = data.draw(hyn.array_shapes(min_dims=1))
    array = data.draw(h.arrays(dtype=dtype, shape=shape))

    info = data.draw(starts_stops_steps(data=array.array))

    res = op21.slice(
        data=array.spox_argument,
        starts=info.starts.spox_argument,
        ends=info.ends.spox_argument,
        axes=info.axes.spox_argument if info.axes else None,
        steps=info.steps.spox_argument if info.steps else None,
    )

    if info.axes:
        axes_arg = {"axes": info.axes.spox_argument}
        axes_arr_arg = {"axes": info.axes.array}
    else:
        axes_arg = {}
        axes_arr_arg = {}
    if info.steps:
        steps_arg = {"steps": info.steps.spox_argument}
        steps_arr_arg = {"steps": info.steps.array}
    else:
        steps_arg = {}
        steps_arr_arg = {}

    model = spox.build(
        {
            "data": array.spox_argument,
            "starts": info.starts.spox_argument,
            "ends": info.ends.spox_argument,
        }
        | axes_arg
        | steps_arg,
        {"res": res},
    )

    candidate, *_ = h.run(
        model,
        data=array.array,
        starts=info.starts.array,
        ends=info.ends.array,
        **steps_arr_arg,
        **axes_arr_arg,
    ).values()
    expected = array.array[*slices(info)]

    np.testing.assert_array_equal(candidate, expected)
