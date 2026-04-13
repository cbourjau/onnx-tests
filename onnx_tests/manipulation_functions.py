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
        # Create valid arrays and then cast to 'dtype_in'
        arr = draw(
            h.arrays(
                dtype=dtype_out,
                shape=shape,
                allow_nan=False,
            )
        )
        # After the cast we may end up with `inf` values which we clip away
        arr = arr.astype(dtype_out).clip(
            np.iinfo(dtype_out).min, np.iinfo(dtype_out).max
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


@st.composite
def flatten(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    arr = draw(
        h.arrays(
            dtype=dtype, shape=hyn.array_shapes(min_dims=0, min_side=0, max_dims=4)
        )
    )
    rank = arr.ndim
    # axis range is [-rank, rank] inclusive
    axis = draw(st.integers(min_value=-rank, max_value=rank)) if rank > 0 else 0
    return TestCaseDraw(
        inputs={"input": arr},
        attribute_kwargs={"axis": axis},
        spox_fun=op.flatten,
    )


@st.composite
def squeeze(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    arr = draw(
        h.arrays(
            dtype=dtype, shape=hyn.array_shapes(min_dims=0, min_side=0, max_dims=4)
        )
    )
    rank = arr.ndim

    # Find axes with size 1
    ones = [i for i, s in enumerate(arr.shape) if s == 1]

    if draw(st.booleans()) and ones:
        # Squeeze specific axes (subset of size-1 axes)
        axes_list = draw(st.lists(st.sampled_from(ones), min_size=1, unique=True))
        # Randomly replace some with negative equivalents
        axes_list = [i - rank if draw(st.booleans()) else i for i in axes_list]
        axes = np.array(axes_list, dtype=np.int64)
    else:
        # Squeeze all size-1 axes (pass None)
        axes = None

    return TestCaseDraw(
        inputs={"data": arr, "axes": axes},
        attribute_kwargs={},
        spox_fun=op.squeeze,
    )


@st.composite
def reshape(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    shape_a, shape_b = draw(_compatible_shapes())
    shape_b_lst = list(shape_b)  # we need to mutate shape_b
    arr = draw(h.arrays(dtype=dtype, shape=shape_a))
    # If the input is zero-sized we can only reshape to shapes that
    # will also contain a zero. We can't use -1 since that would be
    # ambiguous.
    if arr.size == 0:
        allowzero = 1
    else:
        allowzero = draw(st.sampled_from([0, 1]))

    if allowzero == 0 and arr.size != 0:
        # Optionally use zeros to copy elements from input shape
        for i, (ax1, ax2) in enumerate(zip(shape_a, shape_b)):
            if ax1 == ax2 and draw(st.booleans()):
                shape_b_lst[i] = 0

    if arr.size != 0:
        if draw(st.booleans()) and shape_b_lst:
            # Optionally use a single -1 to infer the number of elements
            # This is only possible if the input is not zero-size
            idx = draw(st.integers(0, len(shape_b_lst) - 1))
            shape_b_lst[idx] = -1

    return TestCaseDraw(
        inputs={"data": arr, "shape": np.asarray(shape_b_lst, dtype=np.int64)},
        attribute_kwargs={"allowzero": allowzero},
        spox_fun=op.reshape,
    )


@st.composite
def _compatible_shapes(draw: st.DrawFn) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Produce two shapes with the same number of elements."""
    shape_a = draw(hyn.array_shapes(min_dims=0, min_side=0, max_dims=4))
    shape_a_perm = draw(st.permutations(shape_a))
    shape_b: list[int] = []
    # merge some axes
    for ax in shape_a_perm:
        if draw(st.booleans()) and len(shape_b):
            # merge with previous one
            shape_b[-1] *= ax
        else:
            shape_b.append(ax)

    # shape_a always has more elements than shape_b
    if draw(st.booleans()):
        return shape_a, tuple(shape_b)
    return tuple(shape_b), shape_a


@st.composite
def unsqueeze(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    arr = draw(
        h.arrays(
            dtype=dtype, shape=hyn.array_shapes(min_dims=0, min_side=0, max_dims=3)
        )
    )

    n_new = draw(st.integers(min_value=1, max_value=3))
    new_rank = arr.ndim + n_new
    # Pick unique axis positions in [0, new_rank - 1]
    axes_list = draw(
        st.lists(
            st.integers(min_value=0, max_value=new_rank - 1),
            min_size=n_new,
            max_size=n_new,
            unique=True,
        )
    )
    # Randomly replace some with negative equivalents
    axes_list = [i - new_rank if draw(st.booleans()) else i for i in axes_list]
    axes = np.array(axes_list, dtype=np.int64)

    return TestCaseDraw(
        inputs={"data": arr, "axes": axes},
        attribute_kwargs={},
        spox_fun=op.unsqueeze,
    )


@st.composite
def transpose(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    arr = draw(
        h.arrays(
            dtype=dtype, shape=hyn.array_shapes(min_dims=0, min_side=0, max_dims=4)
        )
    )
    rank = arr.ndim

    if rank > 0 and draw(st.booleans()):
        perm = draw(st.permutations(list(range(rank))))
    else:
        perm = None

    return TestCaseDraw(
        inputs={"data": arr},
        attribute_kwargs={"perm": perm},
        spox_fun=op.transpose,
    )
