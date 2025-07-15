from onnx_tests.average_pool import average_pool

from .utils import make_test

make_test("AveragePool", 19, average_pool, globals())
