from collections.abc import Callable
from types import ModuleType
from typing import Any

import numpy as np
import pytest
import spox.opset.ai.onnx.v17 as op17
import spox.opset.ai.onnx.v18 as op18
import spox.opset.ai.onnx.v19 as op19
import spox.opset.ai.onnx.v20 as op20
import spox.opset.ai.onnx.v21 as op21
import spox.opset.ai.onnx.v22 as op22
from hypothesis import given, reproduce_failure
from hypothesis import strategies as st

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw
from onnx_tests.config import run_candidate
from onnx_tests.runtime_wrappers import run_reference


def get_opset(version: int):
    if version <= 17:
        op: ModuleType = op17
    elif version <= 18:
        op = op18
    elif version <= 19:
        op = op19
    elif version <= 20:
        op = op20
    elif version <= 21:
        op = op21
    elif version <= 22:
        op = op22
    else:
        raise NotImplementedError
    return op


def dtype_params(
    op_name: str, version: int, domain: str = "ai.onnx", type_var: str = "T"
) -> list[tuple[ModuleType, np.dtype]]:
    """Construct a list of opset modules and dtype suitable for
    ``pytest.mark.parametrize("op, dtype", ...)``."""
    out = []
    op = get_opset(version)
    dtypes = h.SCHEMAS[domain][op_name][version].dtype_constraints[type_var]
    for dtype in dtypes:
        out.append((op, dtype))

    return out


def format(val) -> str | None:
    if isinstance(val, np.dtype):
        return str(val)
    if isinstance(val, ModuleType):
        return val.__name__.split(".")[-1]
    return None


def make_test(
    op_name: str,
    version: int,
    strategy_factory: Callable[[np.dtype, ModuleType], st.SearchStrategy[TestCaseDraw]],
    global_namespace: dict[str, Any],
    *,
    type_var: str = "T",
    repro_hash: tuple[str, bytes] | None = None,
):
    @given(data=st.data())
    @pytest.mark.parametrize(
        "op,dtype", dtype_params(op_name, version, type_var=type_var), ids=format
    )
    def test_fun(data: st.DataObject, op, dtype):
        state = data.draw(strategy_factory(dtype, op))
        model = state.build_model()

        (candidate,) = run_candidate(model).values()
        (expected,) = run_reference(model).values()

        if expected.dtype.kind in "UO":
            # Strings must match exactly
            np.testing.assert_array_equal(candidate, expected)
        else:
            h.assert_allclose(candidate, expected)

    if repro_hash is not None:
        test_fun = reproduce_failure(*repro_hash)(test_fun)

    test_name = f"test_{op_name}_{version}"
    if test_name in global_namespace:
        raise ValueError(f"test with name `{test_name}` is already defined")

    global_namespace[test_name] = test_fun
