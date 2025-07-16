# onnx-tests

The `onnx_test` framework provides property-based tests for the ONNX standard.
It is built on top of [Hypothesis](https://hypothesis.readthedocs.io/en/latest/index.html).
Its goal is to become the most comprehensive, user friendly, and canonical test suite of the ONNX standard.
An overview of the operators already covered can be found [here](https://github.com/cbourjau/onnx-tests/blob/main/report/coverage.md).

## Installation

The test suite is best run using the cross-platform and language-agnostic [pixi](https://pixi.sh/latest/) package manager that is based on conda-forge.
You can install the package in development mode using:

```bash
git clone https://github.com/cbourjau/onnx-tests
cd onnx-tests

pixi run pre-commit-install
pixi run postinstall
```

## Example

This project separates the generation of valid test data from defining a conformance test suite for runtimes.
The former is found in the `onnx_tests` package while the latter resides in the `tests` folder.

### Input generation

In the `onnx_tests` package every operator of the ONNX standard has a corresponding Hypothesis strategy producing valid inputs and attributes for the operator in question.
For instance, inputs for the convolution operator can be generated as follows:

```python
from onnx_tests.conv import conv_2d
import numpy as np
import spox.opset.ai.onnx.v21 as op

test_case = conv_2d(np.float32, op)
test_case.inputs
>>> {'X': array([[[[-0.0000000e+00,  1.0000000e-03,  3.1689131e-20],
         [ 0.0000000e+00,  1.0000000e-07,  0.0000000e+00],
         [-8.3047269e-21,  5.2521669e+16, -1.2681085e+16],
         [ 2.0419136e+16, -0.0000000e+00,  1.8446743e+19],
         [ 5.2909870e+16,  0.0000000e+00, -1.8446743e+19],
         [ 7.1820572e+16,  0.0000000e+00, -1.1920929e-07]]]],
      dtype=float32), 'W': array([[[[-3.9690695e+16],
         [ 4.7631236e+15],
         [-3.5537421e+16],
         [ 0.0000000e+00]]]], dtype=float32), 'B': None}

test_case.attributes_kwargs
>>> {'dilations': (1, 2), 'strides': (2, 1), 'pads': None, 'auto_pad': 'SAME_UPPER', 'kernel_shape': (4, 1), 'group': 1}

type(test_case.build_model())
>>> <class 'onnx.onnx_ml_pb2.ModelProto'>

```

However, Hypothesis strategies are best used in conjunction with pytest. Examples of such a usage can be found in the `tests` folder.

### Conformance test suite

The test suite can currently be run against two runtimes: onnxruntime and the reference runtime.
The latter may seem like an odd choice given that it is also used as the above-described "source-of-truth" inside the test suite.
However, running this test suite against the reference implementation itself reveals various bug in it or the `onnx` package.

Testing against the onnxruntime can be done by running

```bash
pixi run test-ort
```

while testing against the reference implementation is done by calling

```bash
pixi run test-reference
```

Both cases are run on CI and the [results](https://github.com/cbourjau/onnx-tests/actions/workflows/ci.yml) can be found in the respectively named jobs of the `ci.yml` workflow.

## Further Context

This project is inspired by [array-api-tests](https://github.com/data-apis/array-api-tests) and a little bit by PyTorch's OpInfo framework.
The API of the ONNX standard is large and complex.
A comprehensive test suite should test every operator of the standard at least along the following dimensions:

- All supported data types
- Various combinations of _interesting_ input shapes (e.g. scalars, and zero-sized tensors)
- _Interesting_ values based on each data type (e.g. max/min values for integers, empty strings, strings with null characters, floating point infs and NaN values, etc.)

Furthermore, a comprehensive test suite needs to parameterize over any further attributes that an operator may require.
These requirements lead to huge test matrices even for seemingly simple operations such as `Add`.
The `array-api-tests` projects is facing the same problem and chose a parameterized test suite which samples test cases at random ("fuzzing").
`array-api-tests` deploys the `hypothesis` framework for this task.
