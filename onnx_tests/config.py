import importlib
from os import environ

from .runtime_wrappers import RunFunction


def _get_run_candidate() -> RunFunction:
    """Get run-function to compute candidate results."""
    run_candidate_path: str = environ.get(
        "RUN_CANDIDATE", "onnx_tests.runtime_wrappers.run_ort"
    )

    *segments, fun_name = run_candidate_path.split(".")
    module_path = ".".join(segments)

    return getattr(importlib.import_module(module_path), fun_name)


run_candidate = _get_run_candidate()

__all__ = ["run_candidate"]
