from onnx_tests import searching_ops

from .utils import make_test

make_test("ArgMax", 13, searching_ops.argmax, globals())
make_test("ArgMin", 13, searching_ops.argmin, globals())
make_test("Max", 13, searching_ops.argmax, globals())
make_test("Min", 13, searching_ops.argmin, globals())
make_test("Unique", 11, searching_ops.unique, globals())
make_test("Where", 16, searching_ops.where, globals())
