from collections.abc import Callable
from typing import Any

import numpy as np
import spox
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn
from onnx.defs import OpSchema, get_all_schemas_with_history
from spox import Var
from spox._future import initializer

from .config import run_candidate
from .runtime_wrappers import run_reference


def arrays(
    dtype: np.dtype | st.SearchStrategy[np.dtype],
    shape: tuple[int, ...] | st.SearchStrategy[tuple[int, ...]],
    *,
    fill: st.SearchStrategy[Any] | None = None,
    unique: bool = False,
    allow_nan: bool | None = None,
    max_value: int | float | None = None,
    min_value: int | float | None = None,
) -> st.SearchStrategy[np.ndarray]:
    def ensure_scalars_are_rank0_arrays(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr)

    def clip_very_large_floats(arr: np.ndarray) -> np.ndarray:
        if arr.dtype.kind == "f":
            max_ = np.sqrt(np.finfo(arr.dtype).max)
            min_ = -max_
            return np.clip(arr, min_, max_)
        return arr

    # mapping passed to from_dtype
    elements: dict[str, Any] = {"alphabet": st.characters(codec="utf-8")}
    if allow_nan is not None:
        elements["allow_nan"] = allow_nan
    if max_value is not None:
        elements["max_value"] = max_value
    if min_value is not None:
        elements["min_value"] = min_value
    return (
        hyn.arrays(dtype, shape, fill=fill, unique=unique, elements=elements)
        .map(clip_very_large_floats)
        .map(ensure_scalars_are_rank0_arrays)
    )


@st.composite
def broadcastable_arrays(
    draw: st.DrawFn, dtype: np.dtype, num_shapes: int = 2, **arrays_kwargs
) -> list[np.ndarray]:
    shapes = draw(hyn.mutually_broadcastable_shapes(num_shapes=num_shapes, min_side=0))

    return [
        draw(arrays(dtype, shape=shape, **arrays_kwargs))
        for shape in shapes.input_shapes
    ]


@st.composite
def matmul_shapes(draw: st.DrawFn) -> tuple[tuple[int, ...], tuple[int, ...]]:
    shapes = draw(hyn.mutually_broadcastable_shapes(signature="(m,k),(k,n)->(m,n)"))

    # We do our own prepending of broadcastable shapes
    shape1, shape2 = shapes.input_shapes
    shape1 = shape1[-2:]
    shape2 = shape2[-2:]

    # ONNX MatMul requires both arrays to have the same rank (contrary to NumPy)
    extra_dims = draw(st.integers(min_value=0, max_value=4))
    pre1, pre2 = draw(
        hyn.mutually_broadcastable_shapes(
            num_shapes=2, min_side=0, min_dims=extra_dims, max_dims=extra_dims
        )
    ).input_shapes
    shape1 = pre1 + shape1
    shape2 = pre2 + shape2
    assert len(shape1) == len(shape2)
    return shape1, shape2


def assert_binary_numpy(
    np_fun: Callable[[np.ndarray, np.ndarray], np.ndarray],
    spox_fun: Callable[[Var, Var], Var],
    arr1: np.ndarray,
    arr2: np.ndarray,
):
    res = spox_fun(initializer(arr1), initializer(arr2))
    model = spox.build({}, {"res": res})

    expected = np_fun(arr1, arr2)
    (candidate,) = run_candidate(model).values()

    np.testing.assert_allclose(candidate, expected)


def assert_binary_against_reference(
    spox_fun: Callable[[Var, Var], Var],
    arr1: np.ndarray,
    arr2: np.ndarray,
):
    res = spox_fun(initializer(arr1), initializer(arr2))
    model = spox.build({}, {"res": res})

    (expected,) = run_reference(model).values()
    (candidate,) = run_candidate(model).values()

    np.testing.assert_equal(candidate, expected)


def assert_allclose(
    actual: np.ndarray, desired: np.ndarray, /, *, err_msg: str | None = None
):
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

    np.testing.assert_allclose(actual, desired, **kwargs, err_msg=err_msg)  # type: ignore


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
