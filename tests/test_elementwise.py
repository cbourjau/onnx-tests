import numpy as np
import pytest
import spox.opset.ai.onnx.v17 as op17
from hypothesis import given
from hypothesis import strategies as st

from onnx_tests import elementwise_ops
from onnx_tests import helpers as h
from onnx_tests.config import run_candidate

from .utils import make_test as make_test

make_test("Add", 14, elementwise_ops.add, globals())
make_test("And", 7, elementwise_ops.and_, globals())
make_test("BitwiseAnd", 18, elementwise_ops.bitwise_and, globals())
make_test("BitwiseOr", 18, elementwise_ops.bitwise_or, globals())
make_test("BitwiseXor", 18, elementwise_ops.bitwise_xor, globals())
make_test("BitShift", 11, elementwise_ops.bit_shift, globals())
make_test("Clip", 13, elementwise_ops.clip, globals())
make_test("Celu", 12, elementwise_ops.celu, globals())

make_test("Div", 14, elementwise_ops.div, globals())
make_test("Equal", 13, elementwise_ops.equal, globals())
make_test("Equal", 19, elementwise_ops.equal, globals())

make_test("Greater", 13, elementwise_ops.greater, globals())
make_test("GreaterOrEqual", 16, elementwise_ops.greater_or_equal, globals())
make_test("Less", 13, elementwise_ops.less, globals())
make_test("LessOrEqual", 16, elementwise_ops.less_or_equal, globals())
make_test("Max", 13, elementwise_ops.max, globals())
make_test("Min", 13, elementwise_ops.min, globals())
make_test("Mod", 13, elementwise_ops.mod, globals())
make_test("Mul", 14, elementwise_ops.mul, globals())
make_test("Or", 7, elementwise_ops.or_, globals())
make_test("Sub", 14, elementwise_ops.sub, globals())
make_test("Xor", 7, elementwise_ops.xor, globals())

make_test("Abs", 13, elementwise_ops.abs, globals())
make_test("Acos", 7, elementwise_ops.acos, globals())
make_test("Acosh", 9, elementwise_ops.acosh, globals())
make_test("Asin", 7, elementwise_ops.asin, globals())
make_test("Asinh", 9, elementwise_ops.asinh, globals())
make_test("Atan", 7, elementwise_ops.atan, globals())
make_test("Atanh", 9, elementwise_ops.atanh, globals())
make_test("BitwiseNot", 18, elementwise_ops.bitwise_not, globals())
make_test("Ceil", 13, elementwise_ops.ceil, globals())
make_test("Cos", 7, elementwise_ops.cos, globals())
make_test("Cosh", 9, elementwise_ops.cosh, globals())
make_test("Erf", 13, elementwise_ops.erf, globals())
make_test("Exp", 13, elementwise_ops.exp, globals())
make_test("Floor", 13, elementwise_ops.floor, globals())
make_test("Hardmax", 13, elementwise_ops.hardmax, globals())
make_test("HardSwish", 14, elementwise_ops.hard_swish, globals())
make_test("Identity", 16, elementwise_ops.identity, globals(), type_var="V")
make_test("IsInf", 10, elementwise_ops.isinf, globals(), type_var="T1")
make_test("IsInf", 20, elementwise_ops.isinf, globals(), type_var="T1")
make_test("IsNaN", 13, elementwise_ops.isnan, globals(), type_var="T1")
make_test("IsNaN", 20, elementwise_ops.isnan, globals(), type_var="T1")
make_test("Log", 13, elementwise_ops.log, globals())
make_test("LogSoftmax", 13, elementwise_ops.log_softmax, globals())
make_test("Neg", 13, elementwise_ops.neg, globals())
make_test("Not", 1, elementwise_ops.not_, globals())
make_test("Reciprocal", 13, elementwise_ops.reciprocal, globals())
make_test("Round", 11, elementwise_ops.round, globals())
make_test("Sigmoid", 13, elementwise_ops.sigmoid, globals())
make_test("Sign", 13, elementwise_ops.sign, globals())
make_test("Sin", 7, elementwise_ops.sin, globals())
make_test("Sinh", 9, elementwise_ops.sinh, globals())

make_test("Softmax", 13, elementwise_ops.softmax, globals())
make_test("Sqrt", 13, elementwise_ops.sqrt, globals())
make_test("Tan", 7, elementwise_ops.tan, globals())
make_test("Tanh", 13, elementwise_ops.tanh, globals())
make_test("Mish", 18, elementwise_ops.mish, globals())

make_test("HardSigmoid", 6, elementwise_ops.hard_sigmoid, globals())
make_test("LeakyRelu", 16, elementwise_ops.leaky_relu, globals())

make_test("Relu", 14, elementwise_ops.relu, globals())
make_test("Elu", 6, elementwise_ops.elu, globals())
make_test("Selu", 6, elementwise_ops.selu, globals())
make_test("Softplus", 1, elementwise_ops.softplus, globals())
make_test("Softsign", 1, elementwise_ops.softsign, globals())
make_test("Gelu", 20, elementwise_ops.gelu, globals())
make_test("ThresholdedRelu", 10, elementwise_ops.thresholded_relu, globals())
make_test("Shrink", 9, elementwise_ops.shrink, globals())


@given(data=st.data())
@pytest.mark.parametrize(
    "dtype_x", h.SCHEMAS["ai.onnx"]["Pow"][15].dtype_constraints["T"], ids=str
)
@pytest.mark.parametrize(
    "dtype_y", h.SCHEMAS["ai.onnx"]["Pow"][15].dtype_constraints["T1"], ids=str
)
def test_Pow_13(data: st.DataObject, dtype_x: str, dtype_y: str):  # noqa
    # Pow requires special care due to the two interdependent data types
    state = data.draw(elementwise_ops.pow(np.dtype(dtype_x), np.dtype(dtype_y), op17))
    model = state.build_model()

    (expected,) = h.run_reference(model).values()
    (candidate,) = run_candidate(model).values()

    h.assert_allclose(candidate, expected)
