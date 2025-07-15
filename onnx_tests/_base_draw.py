from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
import spox
from spox._future import initializer


@dataclass
class TestCaseDraw:
    """Base object encapsulating the drawn state of a test case."""

    inputs: dict[str, np.ndarray | None] | list[np.ndarray | None]
    attribute_kwargs: dict[str, Any]
    spox_fun: Callable[..., spox.Var]

    def input_vars(self) -> dict[str, spox.Var | None] | list[spox.Var | None]:
        """Convert input NumPy arrays into `spox.Var` objects (or leave them as `None`).

        This function uses "initializers" which are not bound to a specific opset
        version.
        """
        if isinstance(self.inputs, dict):
            return {
                k: None if v is None else initializer(v) for k, v in self.inputs.items()
            }
        return [None if v is None else initializer(v) for v in self.inputs]

    def build_model(self) -> onnx.ModelProto:
        input_vars = self.input_vars()
        if isinstance(input_vars, dict):
            res = self.spox_fun(**input_vars, **self.attribute_kwargs)
        else:
            res = self.spox_fun(*input_vars, **self.attribute_kwargs)

        return spox.build({}, {"res": res})
