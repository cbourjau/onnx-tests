from onnx_tests import creation_functions

from .utils import make_test

make_test("Range", 11, creation_functions.range, globals())
