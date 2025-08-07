from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import onnx
from spox import Var, build
from spox._future import initializer


@dataclass
class TestCaseDraw:
    """Base object encapsulating the drawn state of a test case."""

    inputs: dict[str, np.ndarray | None | list[np.ndarray]] | list[np.ndarray | None]
    attribute_kwargs: dict[str, Any]
    spox_fun: Callable[..., Var | tuple[Var, ...]]

    def __init__(
        self,
        inputs: Mapping[str, np.ndarray | None | list[np.ndarray]]
        | list[np.ndarray | None],
        attribute_kwargs: dict[str, Any],
        spox_fun: Callable[..., Var],
    ):
        # Keep mypy happy...
        self.inputs = dict(inputs.items()) if isinstance(inputs, Mapping) else inputs
        self.attribute_kwargs = attribute_kwargs
        self.spox_fun = spox_fun

    def input_vars(
        self,
    ) -> dict[str, Var | list[Var] | None] | list[Var | list[Var] | None]:
        """Convert input NumPy arrays into `spox.Var` objects (or leave them as `None`).

        This function uses "initializers" which are not bound to a specific opset
        version.
        """
        if isinstance(self.inputs, dict):
            return {
                k: None if v is None else create_vars(v) for k, v in self.inputs.items()
            }
        return [None if v is None else create_vars(v) for v in self.inputs]

    def build_model(self) -> onnx.ModelProto:
        input_vars = self.input_vars()
        if isinstance(input_vars, dict):
            res = self.spox_fun(**input_vars, **self.attribute_kwargs)
        else:
            res = self.spox_fun(*input_vars, **self.attribute_kwargs)

        if isinstance(res, Var):
            return build({}, {"res": res})
        return build({}, {f"res{i}": item for i, item in enumerate(res)})


def create_vars(obj: np.ndarray | list[np.ndarray]) -> Var | list[Var]:
    if isinstance(obj, np.ndarray):
        return initializer(obj)
    return [initializer(v) for v in obj]
