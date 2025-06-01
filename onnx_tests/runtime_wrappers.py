from typing import Protocol

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator


class RunFunction(Protocol):
    def __call__(
        self, model: onnx.ModelProto, **kwargs: np.ndarray
    ) -> dict[str, np.ndarray]: ...


def run_ort(model: onnx.ModelProto, **kwargs: np.ndarray) -> dict[str, np.ndarray]:
    import onnxruntime as ort

    sess = ort.InferenceSession(model.SerializeToString())
    output_names = [meta.name for meta in sess.get_outputs()]
    return {k: v for k, v in zip(output_names, sess.run(None, kwargs))}


def run_reference(
    model: onnx.ModelProto, **kwargs: np.ndarray
) -> dict[str, np.ndarray]:
    sess = ReferenceEvaluator(model, optimized=False)
    result_list = sess.run(None, kwargs)
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
