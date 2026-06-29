import pytest
from hypothesis import given
from hypothesis import strategies as st

from onnx_tests import helpers as h
from onnx_tests.attention import attention
from onnx_tests.config import run_candidate
from onnx_tests.runtime_wrappers import run_reference

from .utils import dtype_params, format


@given(data=st.data())
@pytest.mark.parametrize(
    "op,dtype", dtype_params("Attention", 24, type_var="T1"), ids=format
)
@pytest.mark.parametrize("use_past", [False, True], ids=["no_past", "past"])
def test_Attention_24(data: st.DataObject, op, dtype, use_past: bool):  # noqa: N802
    # ``use_past`` is parametrized rather than drawn so that the KV-cache path and
    # the (mutually exclusive) ``nonpad_kv_seqlen`` path produce independent signals.
    state = data.draw(attention(dtype, op, use_past=use_past))
    model = state.build_model()

    candidate = run_candidate(model)
    expected = run_reference(model)

    for i, (cand, exp) in enumerate(
        zip(candidate.values(), expected.values(), strict=True)
    ):
        err_msg = (
            f"output `{i}` of `Attention` did not meet expectation\n"
            f"  inputs: `{state.inputs}`\n"
            f"  attributes: `{state.attribute_kwargs}`"
        )
        h.assert_allclose(cand, exp, err_msg=err_msg)
