from onnx_tests.slice import slice

from .utils import make_test

make_test("Slice", 13, slice, globals())
