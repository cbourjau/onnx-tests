from typing import Protocol

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator


class RunFunction(Protocol):
    def __call__(self, model: onnx.ModelProto) -> dict[str, np.ndarray]: ...


def run_ort(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Execute the given model using onnxruntime.

    The model must not require any inputs. Instead, any values required for the
    computation are serialized as constants in the model itself.
    """
    import onnxruntime as ort

    opt = ort.SessionOptions()
    opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model.SerializeToString(), sess_options=opt)
    output_names = [meta.name for meta in sess.get_outputs()]
    return {k: v for k, v in zip(output_names, sess.run(None, {}))}  # type: ignore


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
