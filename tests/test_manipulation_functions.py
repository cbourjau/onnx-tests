from onnx_tests import manipulation_functions

from .utils import make_test as make_test

make_test("Concat", 13, manipulation_functions.concat, globals())
make_test("Compress", 11, manipulation_functions.compress, globals())
