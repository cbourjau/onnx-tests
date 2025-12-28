from onnx_tests import array_properties as array_props

from .utils import make_test

make_test("Size", 21, array_props.size, globals())
