from typing import Protocol

import ml_dtypes
import numpy as np
import onnx
import spox.opset.ai.onnx.v17 as op
from onnx.reference import ReferenceEvaluator
from spox import build, inline


class RunFunction(Protocol):
    def __call__(self, model: onnx.ModelProto) -> dict[str, np.ndarray]: ...


def run_ort(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Execute the given model using onnxruntime.

    The model must not require any inputs. Instead, any values required for the
    computation are serialized as constants in the model itself.
    """
    import onnxruntime as ort

    # ORT does not support ml_dtypes as outputs. We wrap the model
    # such that it returns a NumPy data type and then cast back to an
    # ml_dtype manually on the Python side.
    model, cast_back = _wrap_ml_dtype_outputs(model)
    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), sess_options=opt)
    output_names = [meta.name for meta in sess.get_outputs()]
    result = {k: v for k, v in zip(output_names, sess.run(None, {}))}  # type: ignore
    for k, dtype in cast_back.items():
        result[k] = result[k].astype(dtype)  # type: ignore

    return result  # type: ignore


def run_reference(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Execute the given model using the reference implementation.

    The model must not require any inputs. Instead, any values required for the
    computation are serialized as constants in the model itself.
    """
    sess = ReferenceEvaluator(model, optimized=False)
    result_list = sess.run(None, {})
    if not isinstance(result_list, list):
        raise TypeError(
            f"expected reference results as 'list', got `{type(result_list)}`"
        )

    non_str_names = [type(el) for el in sess.output_names if not isinstance(el, str)]
    if non_str_names:
        raise TypeError(
            f"expected output names to be of type 'str', got `{non_str_names}`"
        )
    return {k: v for k, v in zip(sess.output_names, result_list)}  # type: ignore


def _wrap_ml_dtype_outputs(
    mp: onnx.ModelProto,
) -> tuple[onnx.ModelProto, dict[str, np.dtype]]:
    cast_back: dict[str, np.dtype] = {}

    result = inline(mp)()
    for info in mp.graph.output:
        if info.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16:
            result[info.name] = op.cast(result[info.name], to=np.float32)
            cast_back[info.name] = np.dtype(ml_dtypes.bfloat16)

    if cast_back:
        return build({}, result), cast_back
    return mp, cast_back  # don't touch the default case
