from onnx_tests.searching_ops import argmax, argmin, where

from .utils import make_test

make_test("ArgMax", 13, argmax, globals())
make_test("ArgMin", 13, argmin, globals())
make_test("Where", 16, where, globals())
