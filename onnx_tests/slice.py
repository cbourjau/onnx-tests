from types import ModuleType

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw

_MAX_INT64 = np.iinfo(np.int64).max
_MIN_INT64 = np.iinfo(np.int64).min
_MAX_INT32 = np.iinfo(np.int32).max
_MIN_INT32 = np.iinfo(np.int32).min


@st.composite
def slice(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    shape = draw(hyn.array_shapes(min_dims=1))
    data = draw(h.arrays(dtype=dtype, shape=shape))

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

    return TestCaseDraw(
        inputs={
            "data": data,
            "starts": np.asarray(starts, dtype=index_dtype),
            "ends": np.asarray(stops, dtype=index_dtype),
            "axes": np.asarray(axes, dtype=index_dtype) if include_axes else None,
            "steps": np.asarray(steps, dtype=index_dtype) if include_steps else None,
        },
        attribute_kwargs={},
        spox_fun=op.slice,
    )
