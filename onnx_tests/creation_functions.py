from types import ModuleType

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra import numpy as hyn

from onnx_tests._base_draw import TestCaseDraw


@st.composite
def range(draw: st.DrawFn, dtype: np.dtype, op: ModuleType) -> TestCaseDraw:
    # TODO:
    # File upstream issue to clarify what should happen if delta
    # points in the wrong direction. E.g. `range(1, -32768, 32767)`

    # TODO:
    # Investigate floating point issues and possibly file a bug
    # upstream (onnx or onnxruntime TBD).

    forward = draw(st.booleans())

    start = draw(
        hyn.arrays(
            dtype=dtype,
            shape=(),
            elements={"allow_nan": False, "allow_infinity": False},
        )
    )
    if forward:
        stop = draw(
            hyn.arrays(
                dtype=dtype,
                shape=(),
                elements={
                    "min_value": start.item(),
                    "allow_nan": False,
                    "allow_infinity": False,
                },
            )
        )
    else:
        stop = draw(
            hyn.arrays(
                dtype=dtype,
                shape=(),
                elements={
                    "max_value": start.item(),
                    "allow_nan": False,
                    "allow_infinity": False,
                },
            )
        )

    n_elements = draw(st.integers(1, 10))
    delta = np.asarray((stop - start) // n_elements, dtype=dtype)
    if delta == 0:
        delta = np.asarray(1, dtype=dtype)

    return TestCaseDraw(
        inputs={"start": start, "limit": stop, "delta": delta},
        attribute_kwargs={},
        spox_fun=op.range,
    )
