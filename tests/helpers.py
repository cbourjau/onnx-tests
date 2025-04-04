from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from math import prod
from typing import Any

import numpy as np
import onnx
import spox
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn
from spox import Tensor, Var, argument

SIGNED_INTEGER_DTYPES = [
    np.dtype(el)
    for el in [
        "int8",
        "int16",
        "int32",
        "int64",
    ]
]
UNSIGNED_INTEGER_DTYPES = [
    np.dtype(el)
    for el in [
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    ]
]

FLOAT_DTYPES = [np.dtype(el) for el in ["float16", "float32", "float64"]]

INTEGER_DTYPES = SIGNED_INTEGER_DTYPES + UNSIGNED_INTEGER_DTYPES
NUMERIC_DTYPES = INTEGER_DTYPES + FLOAT_DTYPES

DTYPES = NUMERIC_DTYPES + [np.dtype("str"), np.dtype(bool)]


@dataclass
class Shape:
    concrete: tuple[int, ...]

    @property
    def onnx(self) -> tuple[str | None | int, ...]:
        return tuple(None for _ in self.concrete)

    def size(self) -> int:
        if self.concrete:
            return prod(self.concrete)
        return 1


class ArrayWrapper:
    array: np.ndarray
    shape: Shape

    def __init__(self, array: np.ndarray):
        self.array = array
        self.shape = Shape(array.shape)

    def __repr__(self) -> str:
        return repr(self.array)

    @property
    @cache
    def spox_argument(self) -> Var:
        return argument(Tensor(self.array.dtype, shape=self.shape.onnx))


def arrays(
    dtype: np.dtype | st.SearchStrategy[np.dtype],
    shape: tuple[int, ...] | st.SearchStrategy[tuple[int, ...]],
    *,
    elements: st.SearchStrategy[Any] | Mapping[str, Any] | None = None,
    fill: st.SearchStrategy[Any] | None = None,
    unique: bool = False,
) -> st.SearchStrategy[ArrayWrapper]:
    return hyn.arrays(dtype, shape, elements=elements, fill=fill, unique=unique).map(
        ArrayWrapper
    )


@st.composite
def broadcastable_arrays(
    draw: st.DrawFn, dtype: np.dtype
) -> tuple[ArrayWrapper, ArrayWrapper]:
    shapes = draw(hyn.mutually_broadcastable_shapes(num_shapes=2))

    array1 = draw(arrays(dtype, shape=shapes.input_shapes[0]))
    array2 = draw(arrays(dtype, shape=shapes.input_shapes[1]))

    return array1, array2


def create_session(model: onnx.ModelProto):
    import onnxruntime as ort  # type: ignore

    return ort.InferenceSession(model.SerializeToString())


def run(model: onnx.ModelProto, **kwargs: np.ndarray) -> dict[str, np.ndarray]:
    sess = create_session(model)
    output_names = [meta.name for meta in sess.get_outputs()]
    return {k: v for k, v in zip(output_names, sess.run(None, kwargs))}


def assert_binary_numpy(
    np_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    spox_fun: Callable[[Var, Var], Var],
    arr1: ArrayWrapper,
    arr2: ArrayWrapper,
):
    x1, x2 = arr1.spox_argument, arr2.spox_argument
    model = spox.build({"x1": x1, "x2": x2}, {"res": spox_fun(x1, x2)})

    candidate, *_ = run(model, x1=arr1.array, x2=arr2.array).values()
    expected = np_fun(arr1.array, arr2.array)

    np.testing.assert_array_equal(candidate, expected)
