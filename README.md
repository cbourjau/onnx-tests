# onnx-tests

Property-based tests for the ONNX standard.

## Context

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

This Proof of concept implements such a hypothesis-based test suite for the ONNX standard.

## State of this project

An overview of the tested operators can be found [here](https://github.com/cbourjau/onnx-tests/blob/main/covered_operators.md).

A common pattern in property-based testing is to compare candidate-output against a "source-of-truth".
If the ONNX standard explicitly states that an operator is following NumPy semantics test are written against NumPy.
Otherwise, tests are written against the "reference implementation" that is shipped with the `onnx` Python package.
However, the latter unfortunately is often buggy, but the hope is that this test suite can also aid improvements in the reference runtime (see "Running the test suite" below).

## Installation

The test suite is best run using the cross-platform and language-agnostic [pixi](https://pixi.sh/latest/) package manager that is based on conda-forge.
You can install the package in development mode using:

```bash
git clone https://github.com/cbourjau/onnx-tests
cd onnx-tests

pixi run pre-commit-install
pixi run postinstall
```

## Running the test suite

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

Both cases are run on CI and the [results](https://github.com/cbourjau/onnx-tests/actions/workflows/ci.yml) can be found in the respectively named jobs of the `CI.yml` workflow.
