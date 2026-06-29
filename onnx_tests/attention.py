from types import ModuleType

import numpy as np
from hypothesis import strategies as st

from onnx_tests import helpers as h
from onnx_tests._base_draw import TestCaseDraw


def _make_spox_fun(op: ModuleType, *, want_present: bool, want_qk: bool):
    """Wrap ``op.attention`` so that only the requested optional outputs are part of the
    built graph.

    Per the spec the present key/value outputs "shall be used together" with the past
    key/value inputs, so they may not always be requested. The ``qk_matmul_output`` is
    independent and requested on demand.
    """

    def attention(**kwargs):
        y, present_key, present_value, qk = op.attention(**kwargs)
        outputs = [y]
        if want_present:
            outputs += [present_key, present_value]
        if want_qk:
            outputs.append(qk)
        return tuple(outputs) if len(outputs) > 1 else outputs[0]

    return attention


@st.composite
def attention(
    draw: st.DrawFn, dtype: np.dtype, op: ModuleType, *, use_past: bool
) -> TestCaseDraw:
    batch_size = draw(st.integers(1, 2))
    kv_num_heads = draw(st.integers(1, 2))
    # Group-query attention requires ``q_num_heads % kv_num_heads == 0``.
    q_num_heads = kv_num_heads * draw(st.integers(1, 2))
    head_size = draw(st.integers(1, 3))
    v_head_size = draw(st.integers(1, 3))
    q_seq = draw(st.integers(1, 3))
    kv_seq = draw(st.integers(1, 3))

    attrs: dict = {}

    # The past/present cache and ``nonpad_kv_seqlen`` are mutually exclusive.
    past_seq = draw(st.integers(0, 2)) if use_past else 0
    total_seq = past_seq + kv_seq

    use_3d = draw(st.booleans())
    if use_3d:
        q = draw(h.arrays(dtype, (batch_size, q_seq, q_num_heads * head_size)))
        k = draw(h.arrays(dtype, (batch_size, kv_seq, kv_num_heads * head_size)))
        v = draw(h.arrays(dtype, (batch_size, kv_seq, kv_num_heads * v_head_size)))
        # The number of heads must be passed explicitly for 3D inputs.
        attrs["q_num_heads"] = q_num_heads
        attrs["kv_num_heads"] = kv_num_heads
    else:
        q = draw(h.arrays(dtype, (batch_size, q_num_heads, q_seq, head_size)))
        k = draw(h.arrays(dtype, (batch_size, kv_num_heads, kv_seq, head_size)))
        v = draw(h.arrays(dtype, (batch_size, kv_num_heads, kv_seq, v_head_size)))

    inputs: dict = {"Q": q, "K": k, "V": v}

    # Attention mask: either a boolean mask or an additive float mask
    # broadcastable to (batch_size, q_num_heads, q_seq, total_seq).
    mask_kind = draw(st.sampled_from(["none", "bool", "float"]))
    if mask_kind == "bool":
        inputs["attn_mask"] = draw(
            h.arrays(np.dtype(bool), (batch_size, q_num_heads, q_seq, total_seq))
        )
    elif mask_kind == "float":
        inputs["attn_mask"] = draw(
            h.arrays(dtype, (batch_size, q_num_heads, q_seq, total_seq))
        )

    if use_past:
        inputs["past_key"] = draw(
            h.arrays(dtype, (batch_size, kv_num_heads, past_seq, head_size))
        )
        inputs["past_value"] = draw(
            h.arrays(dtype, (batch_size, kv_num_heads, past_seq, v_head_size))
        )
    elif draw(st.booleans()):
        inputs["nonpad_kv_seqlen"] = draw(
            h.arrays(np.dtype(np.int64), (batch_size,), min_value=0, max_value=kv_seq)
        )

    attrs["is_causal"] = draw(st.sampled_from([0, 1]))
    if draw(st.booleans()):
        attrs["scale"] = draw(st.floats(0.1, 2.0))
    if draw(st.booleans()):
        attrs["softcap"] = draw(st.floats(0.1, 5.0))

    want_qk = draw(st.booleans())
    if want_qk:
        attrs["qk_matmul_output_mode"] = draw(st.sampled_from([0, 1, 2, 3]))

    return TestCaseDraw(
        inputs=inputs,
        attribute_kwargs=attrs,
        spox_fun=_make_spox_fun(op, want_present=use_past, want_qk=want_qk),
    )


__all__ = ["attention"]
