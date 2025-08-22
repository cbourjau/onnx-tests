import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from onnx_tests import linear_algebra_ops as cases
from onnx_tests.helpers import assert_allclose, run_candidate

from .utils import dtype_params, format, make_test

make_test("Det", 22, cases.det, globals())
make_test("Einsum", 12, cases.einsum, globals())


@given(data=st.data())
@pytest.mark.parametrize("op,dtype", dtype_params("MatMul", 13), ids=format)
def test_MatMul_13(data: st.DataObject, op, dtype):  # noqa
    # MatMul specs explicitly reference NumPy behavior as its semantics
    # onnxruntime fails on zero-sized inputs

    state = data.draw(cases.matmul(dtype, op))
    model = state.build_model()

    expected = np.matmul(*state.inputs)  # type: ignore
    (candidate,) = run_candidate(model).values()

    assert_allclose(candidate, expected)
