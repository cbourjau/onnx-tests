from types import ModuleType

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw


@st.composite
def cast(
    draw: st.DrawFn, dtype_in: np.dtype, dtype_out: np.dtype, op: ModuleType
) -> TestCaseDraw:
    shape = draw(hyn.array_shapes(min_dims=1, min_side=0, max_dims=3))
    if dtype_in.kind in "OU":
        # strings
        # TODO: The standard is very vague on strings!
        if dtype_out is bool:
            fun = np.vectorize(str)
        elif dtype_out.kind == "f":
            format_spec = draw(st.sampled_from(["e", "E", "f", "F"]))
            fun = np.vectorize(lambda el: format(el, format_spec), otypes=[dtype_in])
        elif dtype_out.kind in "iu":
            fun = np.vectorize(str, otypes=[dtype_in])
        elif dtype_out.kind == "b":
            # TODO: Unclear how booleans should be formatted
            fun = np.vectorize(str, otypes=[dtype_in])
        elif dtype_out.kind in "OU":
            # Noop
            fun = np.vectorize(lambda el: el, otypes=[dtype_in])
        else:
            raise NotImplementedError

        arr = fun(draw(h.arrays(dtype=dtype_out, shape=shape)))
    elif dtype_in.kind == "f" and dtype_out.kind in "iu":
        # Out-of-range is UB
        min_val = float(np.asarray(np.iinfo(dtype_out).min, dtype_in))
        max_val = float(np.asarray(np.iinfo(dtype_out).max, dtype_in))
        arr = draw(
            h.arrays(
                dtype=dtype_in,
                shape=shape,
                max_value=max_val,
                min_value=min_val,
                allow_nan=False,
            )
        )
    else:
        # Booleans should work for any float input
        arr = draw(h.arrays(dtype=dtype_out, shape=shape))
    return TestCaseDraw(
        inputs={"input": arr}, attribute_kwargs={"to": dtype_out}, spox_fun=op.cast
    )


@st.composite
def concat(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    n_arrays = draw(st.integers(min_value=1, max_value=100))
    base_shape = draw(hyn.array_shapes(min_dims=1, min_side=0, max_dims=3))
    rank = len(base_shape)
    axis = draw(st.integers(min_value=-rank, max_value=rank - 1))

    arrays = []
    for _ in range(n_arrays):
        shape = list(base_shape)
        shape[axis] = draw(st.integers(0, 3))
        arrays.append(draw(h.arrays(dtype=dtype, shape=tuple(shape))))

    return TestCaseDraw(
        inputs={"inputs": arrays}, attribute_kwargs={"axis": axis}, spox_fun=op.concat
    )


@st.composite
def compress(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    shape = hyn.array_shapes(min_side=0)
    arr = draw(h.arrays(dtype=dtype, shape=shape))
    rank = arr.ndim
    axis = draw(st.one_of(st.none() | st.integers(-rank, rank - 1)))

    if axis is None:
        # may be shorter than axis or flattened buffer
        cond_shape = (draw(st.integers(0, arr.size)),)
    else:
        cond_shape = (draw(st.integers(0, arr.shape[axis])),)
    cond = draw(h.arrays(dtype=np.dtype("bool"), shape=cond_shape))

    return TestCaseDraw(
        inputs={"input": arr, "condition": cond},
        attribute_kwargs={"axis": axis},
        spox_fun=op.compress,
    )
