from onnx_tests import statistical_ops as strats

from .utils import make_test as make_test

make_test("ReduceL1", 18, strats.reduce_l1, globals())
make_test("ReduceL2", 18, strats.reduce_l2, globals())
make_test("ReduceLogSum", 18, strats.reduce_log_sum, globals())
make_test("ReduceLogSumExp", 18, strats.reduce_log_sum_exp, globals())
make_test("ReduceMax", 18, strats.reduce_max, globals())
make_test("ReduceMax", 20, strats.reduce_max, globals())

make_test("ReduceMean", 18, strats.reduce_mean, globals())
make_test("ReduceMin", 18, strats.reduce_min, globals())
make_test("ReduceMin", 20, strats.reduce_min, globals())
make_test("ReduceProd", 18, strats.reduce_prod, globals())
make_test("ReduceSum", 13, strats.reduce_sum, globals())
make_test("ReduceSumSquare", 18, strats.reduce_sum_square, globals())
