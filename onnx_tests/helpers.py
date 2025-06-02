from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import cache
from math import prod
from typing import Any

import numpy as np
import spox
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn
from onnx.defs import OpSchema, get_all_schemas_with_history
from spox import Tensor, Var, argument

from .config import run_candidate
from .runtime_wrappers import run_reference


@dataclass
class Shape:
    concrete: tuple[int, ...]
    dynamic_axes: tuple[int, ...]

    @property
    def onnx(self) -> tuple[str | None | int, ...]:
        return tuple(
            None if i in self.dynamic_axes else side_len
            for i, side_len in enumerate(self.concrete)
        )

    def size(self) -> int:
        if self.concrete:
            return prod(self.concrete)
        return 1


class ArrayWrapper:
    array: np.ndarray
    shape: Shape

    def __init__(self, array: np.ndarray, dynamic_axes=None):
        self.array = array
        self.shape = Shape(array.shape, dynamic_axes=dynamic_axes or ())

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
    def ensure_scalars_are_rank0_arrays(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr)

    def clip_very_large_floats(arr: np.ndarray) -> np.ndarray:
        if arr.dtype.kind == "f":
            max_ = np.sqrt(np.finfo(arr.dtype).max)
            min_ = -max_
            return np.clip(arr, min_, max_)
        return arr

    return (
        hyn.arrays(dtype, shape, elements=elements, fill=fill, unique=unique)
        .map(clip_very_large_floats)
        .map(ensure_scalars_are_rank0_arrays)
        .map(ArrayWrapper)
    )


@st.composite
def broadcastable_arrays(
    draw: st.DrawFn, dtype: np.dtype
) -> tuple[ArrayWrapper, ArrayWrapper]:
    shapes = draw(hyn.mutually_broadcastable_shapes(num_shapes=2, min_side=0))

    array1 = draw(arrays(dtype, shape=shapes.input_shapes[0]))
    array2 = draw(arrays(dtype, shape=shapes.input_shapes[1]))

    return array1, array2


@st.composite
def matmul_shapes(draw: st.DrawFn) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shapes = draw(hyn.mutually_broadcastable_shapes(signature="(m?,k),(k,n?)->(m?,n?)"))

    # We do our own prepending of broadcastable shapes
    shape1, shape2 = shapes.input_shapes
    shape1 = shape1[-2:]
    shape2 = shape2[-2:]

    pre1, pre2 = draw(
        hyn.mutually_broadcastable_shapes(num_shapes=2, min_side=0)
    ).input_shapes
    if len(shape1) >= 2:
        shape1 = pre1 + shape1
    if len(shape2) >= 2:
        shape2 = pre2 + shape2
    return shape1, shape2


def assert_binary_numpy(
    np_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    spox_fun: Callable[[Var, Var], Var],
    arr1: ArrayWrapper,
    arr2: ArrayWrapper,
):
    x1, x2 = arr1.spox_argument, arr2.spox_argument
    model = spox.build({"x1": x1, "x2": x2}, {"res": spox_fun(x1, x2)})

    expected = np_fun(arr1.array, arr2.array)
    candidate, *_ = run_candidate(model, x1=arr1.array, x2=arr2.array).values()

    np.testing.assert_allclose(candidate, expected)


def assert_binary_against_reference(
    spox_fun: Callable[[Var, Var], Var],
    arr1: ArrayWrapper,
    arr2: ArrayWrapper,
):
    x1, x2 = arr1.spox_argument, arr2.spox_argument
    model = spox.build({"x1": x1, "x2": x2}, {"res": spox_fun(x1, x2)})

    x1_arr = arr1.array
    x2_arr = arr2.array
    # Reference runtime cannot handle NumPy string types of different width
    if x1_arr.dtype.kind == "U":
        x1_arr = x1_arr.astype(object)
    if x2_arr.dtype.kind == "U":
        x2_arr = x2_arr.astype(object)
    kwargs = {
        "x1": x1_arr,
        "x2": x2_arr,
    }
    expected, *_ = run_reference(model, **kwargs).values()
    candidate, *_ = run_candidate(model, **kwargs).values()

    np.testing.assert_equal(candidate, expected)


def assert_allclose(actual: np.ndarray, desired: np.ndarray, /):
    """Like `numpy.testing.assert_allclose` but takes dtype into account for relative
    tolerance."""
    if actual.dtype != desired.dtype:
        raise TypeError(f"dtypes do not match `{actual.dtype}` != `{desired.dtype}`")
    kwargs = {}
    if actual.dtype == np.float16:
        kwargs = {"rtol": 1e-3}
    elif actual.dtype == np.float32:
        kwargs = {"rtol": 1e-5}
    elif actual.dtype == np.float64:
        kwargs = {"rtol": 1e-7}

    np.testing.assert_allclose(actual, desired, **kwargs)  # type: ignore


Domain = str
Version = int
Name = str


class SchemaWrapper:
    _schema: OpSchema

    def __init__(self, schema: OpSchema):
        self._schema = schema

    @property
    def dtype_constraints(self) -> dict[str, list[np.dtype]]:
        # TODO: support all data types
        dtype_map: dict[str, np.dtype] = {
            "tensor(uint8)": np.dtype("uint8"),
            "tensor(uint16)": np.dtype("uint16"),
            "tensor(uint32)": np.dtype("uint32"),
            "tensor(uint64)": np.dtype("uint64"),
            "tensor(int8)": np.dtype("int8"),
            "tensor(int16)": np.dtype("int16"),
            "tensor(int32)": np.dtype("int32"),
            "tensor(int64)": np.dtype("int64"),
            "tensor(float16)": np.dtype("float16"),
            "tensor(float)": np.dtype("float32"),
            "tensor(double)": np.dtype("float64"),
            "tensor(string)": np.dtype("str"),
            "tensor(bool)": np.dtype("bool"),
        }

        out = {}
        for item in self._schema.type_constraints:
            out[item.type_param_str] = [
                dtype_map[ty] for ty in item.allowed_type_strs if ty in dtype_map
            ]
        return out


def _get_op_schemas() -> dict[Domain, dict[Name, dict[Version, SchemaWrapper]]]:
    ALL_SCHEMAS: list[OpSchema] = get_all_schemas_with_history()  # type: ignore
    out: dict[Domain, dict[Name, dict[Version, SchemaWrapper]]] = {}
    for schema in ALL_SCHEMAS:
        domain_name = schema.domain or "ai.onnx"
        domain = out.setdefault(domain_name, {})
        versions_of_op = domain.setdefault(schema.name, {})
        versions_of_op[schema.since_version] = SchemaWrapper(schema)
    return out


SCHEMAS = _get_op_schemas()
