from types import ModuleType

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw


@st.composite
def constant_of_shape(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    shape = draw(hyn.array_shapes(min_dims=0, min_side=0, max_dims=4))
    input_shape = np.asarray(shape, dtype=np.int64)
    # TODO: Clarify in spec that this is a 1D tensor (or allow
    # scalars, which would fit better).
    value = draw(h.arrays(dtype, shape=(1,)))
    return TestCaseDraw(
        inputs={"input": input_shape},
        attribute_kwargs={"value": value},
        spox_fun=op.constant_of_shape,
    )


@st.composite
def eye_like(
    draw: st.DrawFn, dtype_in: np.dtype, dtype_out: np.dtype, op: ModuleType
) -> TestCaseDraw:
    rows = draw(st.integers(0, 5))
    cols = draw(st.integers(0, 5))
    input_arr = draw(h.arrays(dtype_in, shape=(rows, cols)))
    k = draw(st.integers(-rows, cols) if rows > 0 and cols > 0 else st.just(0))

    attrs = {}
    if draw(st.booleans()):
        # Dtype attribute is optional
        attrs["dtype"] = dtype_out
    return TestCaseDraw(
        inputs={"input": input_arr},
        attribute_kwargs=attrs | {"k": k},
        spox_fun=op.eye_like,
    )


@st.composite
def one_hot(
    draw: st.DrawFn,
    dtype_indices: np.dtype,
    dtype_depth: np.dtype,
    dtype_values: np.dtype,
    op: ModuleType,
) -> TestCaseDraw:
    indices_shape = draw(
        hyn.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=4)
    )
    depth_val = draw(st.integers(1, 6))

    # Generate indices in range [-depth, depth-1] plus some out-of-range values
    indices = draw(
        h.arrays(
            dtype_indices,
            shape=indices_shape,
            min_value=-depth_val,
            max_value=depth_val - 1,
        )
    )
    depth = np.asarray(depth_val, dtype=dtype_depth)
    values = draw(h.arrays(dtype_values, shape=(2,)))

    rank = len(indices_shape)
    axis = draw(st.integers(-rank - 1, rank))

    return TestCaseDraw(
        inputs={"indices": indices, "depth": depth, "values": values},
        attribute_kwargs={"axis": axis},
        spox_fun=op.one_hot,
    )
