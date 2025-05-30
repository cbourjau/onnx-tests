from collections.abc import Callable

import numpy as np
import pytest
import spox.opset.ai.onnx.v17 as op17  # Oldest opset currently provided by spox
import spox.opset.ai.onnx.v18 as op18
import spox.opset.ai.onnx.v19 as op19
from hypothesis import given
from hypothesis import strategies as st
from spox import Var

from . import helpers as h


def make_binary_element_wise_test(
    dtypes: list[np.dtype], op: Callable[[Var, Var], Var]
) -> Callable:
    @given(data=st.data())
    @pytest.mark.parametrize("dtype", dtypes, ids=str)
    def test_function(data, dtype: np.dtype):
        array_tuple = data.draw(h.broadcastable_arrays(dtype))
        h.assert_binary_against_reference(op, *array_tuple)

    return test_function


test_add_v14 = make_binary_element_wise_test(h.NUMERIC_DTYPES, op17.add)
test_and_v7 = make_binary_element_wise_test([np.dtype(bool)], op17.and_)
test_bitwise_and_v18 = make_binary_element_wise_test(h.INTEGER_DTYPES, op18.bitwise_and)
test_bitwise_or_v18 = make_binary_element_wise_test(h.INTEGER_DTYPES, op18.bitwise_or)
test_bitwise_xor_v18 = make_binary_element_wise_test(h.INTEGER_DTYPES, op18.bitwise_xor)

# Division by zero is not specified.
# test_div_v14 = make_binary_element_wise_test(h.NUMERIC_DTYPES, op17.div)

test_equal_v13 = make_binary_element_wise_test(h.NUMERIC_AND_BOOL_DTYPES, op17.equal)
test_equal_v19 = make_binary_element_wise_test(h.DTYPES, op19.equal)

test_greater_v13 = make_binary_element_wise_test(h.NUMERIC_DTYPES, op17.greater)
test_greater_or_equal_v16 = make_binary_element_wise_test(
    h.NUMERIC_DTYPES, op17.greater_or_equal
)
test_less_v13 = make_binary_element_wise_test(h.NUMERIC_DTYPES, op17.less)
test_less_or_equal_v16 = make_binary_element_wise_test(
    h.NUMERIC_DTYPES, op17.less_or_equal
)

# Division by 0 is unspecified
# test_mod_fmod0_v13 = make_binary_element_wise_test(h.INTEGER_DTYPES, lambda x,y: op17.mod(x,y, fmod=0))
test_mod_fmod1_v13 = make_binary_element_wise_test(
    h.NUMERIC_DTYPES, lambda x, y: op17.mod(x, y, fmod=1)
)

test_mul_v14 = make_binary_element_wise_test(h.NUMERIC_DTYPES, op17.mul)
test_or_v7 = make_binary_element_wise_test([np.dtype(bool)], op17.or_)

# Pow allows different type permutations and thus needs special handling
# test_pow_v15

test_sub_v14 = make_binary_element_wise_test(h.NUMERIC_DTYPES, op17.sub)
test_xor_v7 = make_binary_element_wise_test([np.dtype(bool)], op17.xor)
