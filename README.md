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

This project is a proof of concept at this point.
The tested runtime is currently hard-coded to be the onnxruntime but should naturally be configurable such that the test suite is easy to use by any implementation of the ONNX standard.

A common pattern in property-based testing is to compare candidate-output against a "source-of-truth". This Proof-of-concept is currently simply using NumPy as that source of truth, but there is an argument to be made that the reference runtime found in the `onnx` package is a better choice.

## Installation

You can install the package in development mode using:

```bash
git clone https://github.com/quantco/onnx-tests
cd onnx-tests

pixi run pre-commit-install
pixi run postinstall
pixi run test
```

## Running the test suite

```bash
pixi run test
```

The few existing tests reveal that the onnxruntime is lacking kernels for many data types that are required by the standard:

<details>
<summary>Test output</summary>

```
pixi run test
✨ Pixi task (test in default): pytest
[2K====================================================================== test session starts ======================================================================
platform darwin -- Python 3.13.3, pytest-8.3.5, pluggy-1.5.0
rootdir: /Users/c.bourjau/repos/quantco/onnx-tests
configfile: pyproject.toml
testpaths: tests
plugins: emoji-0.2.0, hypothesis-6.131.15, md-0.2.0, cov-6.1.1
collected 54 items

tests/test_binary.py ...........                                                                                                                          [ 20%]
tests/test_slice.py ...........F.                                                                                                                         [ 44%]
tests/test_unary.py ................F                                                                                                                     [ 75%]
tests/test_where.py FF...FFF...FF                                                                                                                         [100%]

=========================================================================== FAILURES ============================================================================
______________________________________________________________________ test_slice[dtype11] ______________________________________________________________________

dtype = dtype('<U')

    @given(data=st.data())
>   @pytest.mark.parametrize("dtype", h.DTYPES)

tests/test_slice.py:88:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/test_slice.py:127: in test_slice
    candidate, *_ = h.run(
tests/helpers.py:106: in run
    return {k: v for k, v in zip(output_names, sess.run(None, kwargs))}
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <onnxruntime.capi.onnxruntime_inference_collection.InferenceSession object at 0x117e887a0>, output_names = ['res']
input_feed = {'data': array(['\ud800'], dtype='<U1'), 'ends': array([], dtype=int32), 'starts': array([], dtype=int32)}, run_options = None

    def run(self, output_names, input_feed, run_options=None):
        """
        Compute the predictions.

        :param output_names: name of the outputs
        :param input_feed: dictionary ``{ input_name: input_value }``
        :param run_options: See :class:`onnxruntime.RunOptions`.
        :return: list of results, every result is either a numpy array,
            a sparse tensor, a list or a dictionary.

        ::

            sess.run([output_name], {input_name: x})
        """
        self._validate_input(list(input_feed.keys()))
        if not output_names:
            output_names = [output.name for output in self._outputs_meta]
        try:
>           return self._sess.run(output_names, input_feed, run_options)
E           onnxruntime.capi.onnxruntime_pybind11_state.Fail: <class 'UnicodeEncodeError'>: 'utf-8' codec can't encode character '\ud800' in position 0: surrogates not allowed
E           Falsifying example: test_slice(
E               dtype=dtype('<U'),
E               data=data(...),
E           )
E           Draw 1: (1,)
E           Draw 2: ArrayWrapper(array(['\ud800'], dtype='<U1'))
E           Draw 3: SliceInfo(starts=array([], dtype=int32), ends=array([], dtype=int32), axes=None, steps=None)
E           Explanation:
E               These lines were always and only run by failing examples:
E                   /Users/c.bourjau/repos/quantco/onnx-tests/.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:271

.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:270: Fail
_______________________________________________________________________ test_cos[float64] _______________________________________________________________________

dtype = dtype('float64')

    @given(data=st.data())
>   @pytest.mark.parametrize("dtype", h.FLOAT_DTYPES, ids=str)

tests/test_unary.py:56:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/test_unary.py:59: in test_cos
    assert_unary_numpy(np.cos, op21.cos, array, almost_equal=True)
tests/test_unary.py:23: in assert_unary_numpy
    candidate, *_ = h.run(model, x=x.array).values()
tests/helpers.py:104: in run
    sess = create_session(model)
tests/helpers.py:100: in create_session
    return ort.InferenceSession(model.SerializeToString())
.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:469: in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <onnxruntime.capi.onnxruntime_inference_collection.InferenceSession object at 0x117efba00>, providers = [], provider_options = []
disabled_optimizers = set()

    def _create_inference_session(self, providers, provider_options, disabled_optimizers=None):
        available_providers = C.get_available_providers()

        # Tensorrt can fall back to CUDA if it's explicitly assigned. All others fall back to CPU.
        if "TensorrtExecutionProvider" in available_providers:
            if (
                providers
                and any(
                    provider == "CUDAExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "CUDAExecutionProvider")
                    for provider in providers
                )
                and any(
                    provider == "TensorrtExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "TensorrtExecutionProvider")
                    for provider in providers
                )
            ):
                self._fallback_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        # MIGraphX can fall back to ROCM if it's explicitly assigned. All others fall back to CPU.
        elif "MIGraphXExecutionProvider" in available_providers:
            if providers and any(
                provider == "ROCMExecutionProvider"
                or (isinstance(provider, tuple) and provider[0] == "ROCMExecutionProvider")
                for provider in providers
            ):
                self._fallback_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        else:
            self._fallback_providers = ["CPUExecutionProvider"]

        # validate providers and provider_options before other initialization
        providers, provider_options = check_and_normalize_provider_args(
            providers, provider_options, available_providers
        )

        session_options = self._sess_options if self._sess_options else C.get_default_session_options()

        self._register_ep_custom_ops(session_options, providers, provider_options, available_providers)

        if self._model_path:
            sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
        else:
            sess = C.InferenceSession(session_options, self._model_bytes, False, self._read_config_from_model)

        if disabled_optimizers is None:
            disabled_optimizers = set()
        elif not isinstance(disabled_optimizers, set):
            # convert to set. assumes iterable
            disabled_optimizers = set(disabled_optimizers)

        # initialize the C++ InferenceSession
>       sess.initialize_session(providers, provider_options, disabled_optimizers)
E       onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Cos(7) node with name 'Cos_0'
E       Falsifying example: test_cos(
E           dtype=dtype('float64'),
E           data=data(...),
E       )
E       Draw 1: ArrayWrapper(array(0.))

.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:541: NotImplemented
_______________________________________________________________________ test_where[int8] ________________________________________________________________________

dtype = dtype('int8')

    @given(data=st.data())
>   @pytest.mark.parametrize("dtype", h.DTYPES, ids=str)

tests/test_where.py:13:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/test_where.py:27: in test_where
    candidate, *_ = h.run(model, cond=cond.array, x=x.array, y=y.array).values()
tests/helpers.py:104: in run
    sess = create_session(model)
tests/helpers.py:100: in create_session
    return ort.InferenceSession(model.SerializeToString())
.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:469: in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <onnxruntime.capi.onnxruntime_inference_collection.InferenceSession object at 0x117a920d0>, providers = [], provider_options = []
disabled_optimizers = set()

    def _create_inference_session(self, providers, provider_options, disabled_optimizers=None):
        available_providers = C.get_available_providers()

        # Tensorrt can fall back to CUDA if it's explicitly assigned. All others fall back to CPU.
        if "TensorrtExecutionProvider" in available_providers:
            if (
                providers
                and any(
                    provider == "CUDAExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "CUDAExecutionProvider")
                    for provider in providers
                )
                and any(
                    provider == "TensorrtExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "TensorrtExecutionProvider")
                    for provider in providers
                )
            ):
                self._fallback_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        # MIGraphX can fall back to ROCM if it's explicitly assigned. All others fall back to CPU.
        elif "MIGraphXExecutionProvider" in available_providers:
            if providers and any(
                provider == "ROCMExecutionProvider"
                or (isinstance(provider, tuple) and provider[0] == "ROCMExecutionProvider")
                for provider in providers
            ):
                self._fallback_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        else:
            self._fallback_providers = ["CPUExecutionProvider"]

        # validate providers and provider_options before other initialization
        providers, provider_options = check_and_normalize_provider_args(
            providers, provider_options, available_providers
        )

        session_options = self._sess_options if self._sess_options else C.get_default_session_options()

        self._register_ep_custom_ops(session_options, providers, provider_options, available_providers)

        if self._model_path:
            sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
        else:
            sess = C.InferenceSession(session_options, self._model_bytes, False, self._read_config_from_model)

        if disabled_optimizers is None:
            disabled_optimizers = set()
        elif not isinstance(disabled_optimizers, set):
            # convert to set. assumes iterable
            disabled_optimizers = set(disabled_optimizers)

        # initialize the C++ InferenceSession
>       sess.initialize_session(providers, provider_options, disabled_optimizers)
E       onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Where(16) node with name 'Where_0'
E       Falsifying example: test_where(
E           dtype=dtype('int8'),
E           data=data(...),
E       )
E       Draw 1: BroadcastableShapes(input_shapes=((), (), ()), result_shape=())
E       Draw 2: ArrayWrapper(array(False))
E       Draw 3: ArrayWrapper(array(0, dtype=int8))
E       Draw 4: ArrayWrapper(array(0, dtype=int8))

.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:541: NotImplemented
_______________________________________________________________________ test_where[int16] _______________________________________________________________________

dtype = dtype('int16')

    @given(data=st.data())
>   @pytest.mark.parametrize("dtype", h.DTYPES, ids=str)

tests/test_where.py:13:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/test_where.py:27: in test_where
    candidate, *_ = h.run(model, cond=cond.array, x=x.array, y=y.array).values()
tests/helpers.py:104: in run
    sess = create_session(model)
tests/helpers.py:100: in create_session
    return ort.InferenceSession(model.SerializeToString())
.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:469: in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <onnxruntime.capi.onnxruntime_inference_collection.InferenceSession object at 0x11803b860>, providers = [], provider_options = []
disabled_optimizers = set()

    def _create_inference_session(self, providers, provider_options, disabled_optimizers=None):
        available_providers = C.get_available_providers()

        # Tensorrt can fall back to CUDA if it's explicitly assigned. All others fall back to CPU.
        if "TensorrtExecutionProvider" in available_providers:
            if (
                providers
                and any(
                    provider == "CUDAExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "CUDAExecutionProvider")
                    for provider in providers
                )
                and any(
                    provider == "TensorrtExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "TensorrtExecutionProvider")
                    for provider in providers
                )
            ):
                self._fallback_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        # MIGraphX can fall back to ROCM if it's explicitly assigned. All others fall back to CPU.
        elif "MIGraphXExecutionProvider" in available_providers:
            if providers and any(
                provider == "ROCMExecutionProvider"
                or (isinstance(provider, tuple) and provider[0] == "ROCMExecutionProvider")
                for provider in providers
            ):
                self._fallback_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        else:
            self._fallback_providers = ["CPUExecutionProvider"]

        # validate providers and provider_options before other initialization
        providers, provider_options = check_and_normalize_provider_args(
            providers, provider_options, available_providers
        )

        session_options = self._sess_options if self._sess_options else C.get_default_session_options()

        self._register_ep_custom_ops(session_options, providers, provider_options, available_providers)

        if self._model_path:
            sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
        else:
            sess = C.InferenceSession(session_options, self._model_bytes, False, self._read_config_from_model)

        if disabled_optimizers is None:
            disabled_optimizers = set()
        elif not isinstance(disabled_optimizers, set):
            # convert to set. assumes iterable
            disabled_optimizers = set(disabled_optimizers)

        # initialize the C++ InferenceSession
>       sess.initialize_session(providers, provider_options, disabled_optimizers)
E       onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Where(16) node with name 'Where_0'
E       Falsifying example: test_where(
E           dtype=dtype('int16'),
E           data=data(...),
E       )
E       Draw 1: BroadcastableShapes(input_shapes=((), (), ()), result_shape=())
E       Draw 2: ArrayWrapper(array(False))
E       Draw 3: ArrayWrapper(array(0, dtype=int16))
E       Draw 4: ArrayWrapper(array(0, dtype=int16))

.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:541: NotImplemented
______________________________________________________________________ test_where[uint16] _______________________________________________________________________

dtype = dtype('uint16')

    @given(data=st.data())
>   @pytest.mark.parametrize("dtype", h.DTYPES, ids=str)

tests/test_where.py:13:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/test_where.py:27: in test_where
    candidate, *_ = h.run(model, cond=cond.array, x=x.array, y=y.array).values()
tests/helpers.py:104: in run
    sess = create_session(model)
tests/helpers.py:100: in create_session
    return ort.InferenceSession(model.SerializeToString())
.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:469: in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <onnxruntime.capi.onnxruntime_inference_collection.InferenceSession object at 0x117fe57e0>, providers = [], provider_options = []
disabled_optimizers = set()

    def _create_inference_session(self, providers, provider_options, disabled_optimizers=None):
        available_providers = C.get_available_providers()

        # Tensorrt can fall back to CUDA if it's explicitly assigned. All others fall back to CPU.
        if "TensorrtExecutionProvider" in available_providers:
            if (
                providers
                and any(
                    provider == "CUDAExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "CUDAExecutionProvider")
                    for provider in providers
                )
                and any(
                    provider == "TensorrtExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "TensorrtExecutionProvider")
                    for provider in providers
                )
            ):
                self._fallback_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        # MIGraphX can fall back to ROCM if it's explicitly assigned. All others fall back to CPU.
        elif "MIGraphXExecutionProvider" in available_providers:
            if providers and any(
                provider == "ROCMExecutionProvider"
                or (isinstance(provider, tuple) and provider[0] == "ROCMExecutionProvider")
                for provider in providers
            ):
                self._fallback_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        else:
            self._fallback_providers = ["CPUExecutionProvider"]

        # validate providers and provider_options before other initialization
        providers, provider_options = check_and_normalize_provider_args(
            providers, provider_options, available_providers
        )

        session_options = self._sess_options if self._sess_options else C.get_default_session_options()

        self._register_ep_custom_ops(session_options, providers, provider_options, available_providers)

        if self._model_path:
            sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
        else:
            sess = C.InferenceSession(session_options, self._model_bytes, False, self._read_config_from_model)

        if disabled_optimizers is None:
            disabled_optimizers = set()
        elif not isinstance(disabled_optimizers, set):
            # convert to set. assumes iterable
            disabled_optimizers = set(disabled_optimizers)

        # initialize the C++ InferenceSession
>       sess.initialize_session(providers, provider_options, disabled_optimizers)
E       onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Where(16) node with name 'Where_0'
E       Falsifying example: test_where(
E           dtype=dtype('uint16'),
E           data=data(...),
E       )
E       Draw 1: BroadcastableShapes(input_shapes=((), (), ()), result_shape=())
E       Draw 2: ArrayWrapper(array(False))
E       Draw 3: ArrayWrapper(array(0, dtype=uint16))
E       Draw 4: ArrayWrapper(array(0, dtype=uint16))

.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:541: NotImplemented
______________________________________________________________________ test_where[uint32] _______________________________________________________________________

dtype = dtype('uint32')

    @given(data=st.data())
>   @pytest.mark.parametrize("dtype", h.DTYPES, ids=str)

tests/test_where.py:13:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/test_where.py:27: in test_where
    candidate, *_ = h.run(model, cond=cond.array, x=x.array, y=y.array).values()
tests/helpers.py:104: in run
    sess = create_session(model)
tests/helpers.py:100: in create_session
    return ort.InferenceSession(model.SerializeToString())
.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:469: in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <onnxruntime.capi.onnxruntime_inference_collection.InferenceSession object at 0x117e00a10>, providers = [], provider_options = []
disabled_optimizers = set()

    def _create_inference_session(self, providers, provider_options, disabled_optimizers=None):
        available_providers = C.get_available_providers()

        # Tensorrt can fall back to CUDA if it's explicitly assigned. All others fall back to CPU.
        if "TensorrtExecutionProvider" in available_providers:
            if (
                providers
                and any(
                    provider == "CUDAExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "CUDAExecutionProvider")
                    for provider in providers
                )
                and any(
                    provider == "TensorrtExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "TensorrtExecutionProvider")
                    for provider in providers
                )
            ):
                self._fallback_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        # MIGraphX can fall back to ROCM if it's explicitly assigned. All others fall back to CPU.
        elif "MIGraphXExecutionProvider" in available_providers:
            if providers and any(
                provider == "ROCMExecutionProvider"
                or (isinstance(provider, tuple) and provider[0] == "ROCMExecutionProvider")
                for provider in providers
            ):
                self._fallback_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        else:
            self._fallback_providers = ["CPUExecutionProvider"]

        # validate providers and provider_options before other initialization
        providers, provider_options = check_and_normalize_provider_args(
            providers, provider_options, available_providers
        )

        session_options = self._sess_options if self._sess_options else C.get_default_session_options()

        self._register_ep_custom_ops(session_options, providers, provider_options, available_providers)

        if self._model_path:
            sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
        else:
            sess = C.InferenceSession(session_options, self._model_bytes, False, self._read_config_from_model)

        if disabled_optimizers is None:
            disabled_optimizers = set()
        elif not isinstance(disabled_optimizers, set):
            # convert to set. assumes iterable
            disabled_optimizers = set(disabled_optimizers)

        # initialize the C++ InferenceSession
>       sess.initialize_session(providers, provider_options, disabled_optimizers)
E       onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Where(16) node with name 'Where_0'
E       Falsifying example: test_where(
E           dtype=dtype('uint32'),
E           data=data(...),
E       )
E       Draw 1: BroadcastableShapes(input_shapes=((), (), ()), result_shape=())
E       Draw 2: ArrayWrapper(array(False))
E       Draw 3: ArrayWrapper(array(0, dtype=uint32))
E       Draw 4: ArrayWrapper(array(0, dtype=uint32))

.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:541: NotImplemented
______________________________________________________________________ test_where[uint64] _______________________________________________________________________

dtype = dtype('uint64')

    @given(data=st.data())
>   @pytest.mark.parametrize("dtype", h.DTYPES, ids=str)

tests/test_where.py:13:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/test_where.py:27: in test_where
    candidate, *_ = h.run(model, cond=cond.array, x=x.array, y=y.array).values()
tests/helpers.py:104: in run
    sess = create_session(model)
tests/helpers.py:100: in create_session
    return ort.InferenceSession(model.SerializeToString())
.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:469: in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <onnxruntime.capi.onnxruntime_inference_collection.InferenceSession object at 0x1180c3e10>, providers = [], provider_options = []
disabled_optimizers = set()

    def _create_inference_session(self, providers, provider_options, disabled_optimizers=None):
        available_providers = C.get_available_providers()

        # Tensorrt can fall back to CUDA if it's explicitly assigned. All others fall back to CPU.
        if "TensorrtExecutionProvider" in available_providers:
            if (
                providers
                and any(
                    provider == "CUDAExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "CUDAExecutionProvider")
                    for provider in providers
                )
                and any(
                    provider == "TensorrtExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "TensorrtExecutionProvider")
                    for provider in providers
                )
            ):
                self._fallback_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        # MIGraphX can fall back to ROCM if it's explicitly assigned. All others fall back to CPU.
        elif "MIGraphXExecutionProvider" in available_providers:
            if providers and any(
                provider == "ROCMExecutionProvider"
                or (isinstance(provider, tuple) and provider[0] == "ROCMExecutionProvider")
                for provider in providers
            ):
                self._fallback_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        else:
            self._fallback_providers = ["CPUExecutionProvider"]

        # validate providers and provider_options before other initialization
        providers, provider_options = check_and_normalize_provider_args(
            providers, provider_options, available_providers
        )

        session_options = self._sess_options if self._sess_options else C.get_default_session_options()

        self._register_ep_custom_ops(session_options, providers, provider_options, available_providers)

        if self._model_path:
            sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
        else:
            sess = C.InferenceSession(session_options, self._model_bytes, False, self._read_config_from_model)

        if disabled_optimizers is None:
            disabled_optimizers = set()
        elif not isinstance(disabled_optimizers, set):
            # convert to set. assumes iterable
            disabled_optimizers = set(disabled_optimizers)

        # initialize the C++ InferenceSession
>       sess.initialize_session(providers, provider_options, disabled_optimizers)
E       onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Where(16) node with name 'Where_0'
E       Falsifying example: test_where(
E           dtype=dtype('uint64'),
E           data=data(...),
E       )
E       Draw 1: BroadcastableShapes(input_shapes=((), (), ()), result_shape=())
E       Draw 2: ArrayWrapper(array(False))
E       Draw 3: ArrayWrapper(array(0, dtype=uint64))
E       Draw 4: ArrayWrapper(array(0, dtype=uint64))

.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:541: NotImplemented
________________________________________________________________________ test_where[<U0] ________________________________________________________________________

dtype = dtype('<U')

    @given(data=st.data())
>   @pytest.mark.parametrize("dtype", h.DTYPES, ids=str)

tests/test_where.py:13:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

data = data(...), dtype = dtype('<U')

    @given(data=st.data())
    @pytest.mark.parametrize("dtype", h.DTYPES, ids=str)
    def test_where(data, dtype: np.dtype):
        shapes = data.draw(hyn.mutually_broadcastable_shapes(num_shapes=3))

        cond = data.draw(h.arrays(np.dtype(bool), shape=shapes.input_shapes[0]))
        x = data.draw(h.arrays(dtype, shape=shapes.input_shapes[1]))
        y = data.draw(h.arrays(dtype, shape=shapes.input_shapes[2]))

        res = op21.where(cond.spox_argument, x.spox_argument, y.spox_argument)
        model = spox.build(
            {"cond": cond.spox_argument, "x": x.spox_argument, "y": y.spox_argument},
            {"res": res},
        )

        candidate, *_ = h.run(model, cond=cond.array, x=x.array, y=y.array).values()
        expected = np.where(cond.array, x.array, y.array)

>       np.testing.assert_array_equal(candidate, expected)
E       AssertionError:
E       Arrays are not equal
E
E       Mismatched elements: 1 / 1 (100%)
E        ACTUAL: array('', dtype=object)
E        DESIRED: array('\x000', dtype='<U2')
E       Falsifying example: test_where(
E           dtype=dtype('<U'),
E           data=data(...),
E       )
E       Draw 1: BroadcastableShapes(input_shapes=((), (), ()), result_shape=())
E       Draw 2: ArrayWrapper(array(False))
E       Draw 3: ArrayWrapper(array('', dtype='<U1'))
E       Draw 4: ArrayWrapper(array('\x000', dtype='<U2'))
E       Explanation:
E           These lines were always and only run by failing examples:
E               /Users/c.bourjau/repos/quantco/onnx-tests/.pixi/envs/default/lib/python3.13/site-packages/numpy/_core/_dtype.py:337
E               /Users/c.bourjau/repos/quantco/onnx-tests/.pixi/envs/default/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1558
E               /Users/c.bourjau/repos/quantco/onnx-tests/.pixi/envs/default/lib/python3.13/site-packages/numpy/_core/arrayprint.py:1604
E               /Users/c.bourjau/repos/quantco/onnx-tests/.pixi/envs/default/lib/python3.13/site-packages/numpy/testing/_private/utils.py:866

tests/test_where.py:30: AssertionError
_______________________________________________________________________ test_where[bool] ________________________________________________________________________

dtype = dtype('bool')

    @given(data=st.data())
>   @pytest.mark.parametrize("dtype", h.DTYPES, ids=str)

tests/test_where.py:13:
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _
tests/test_where.py:27: in test_where
    candidate, *_ = h.run(model, cond=cond.array, x=x.array, y=y.array).values()
tests/helpers.py:104: in run
    sess = create_session(model)
tests/helpers.py:100: in create_session
    return ort.InferenceSession(model.SerializeToString())
.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:469: in __init__
    self._create_inference_session(providers, provider_options, disabled_optimizers)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _

self = <onnxruntime.capi.onnxruntime_inference_collection.InferenceSession object at 0x117e53e10>, providers = [], provider_options = []
disabled_optimizers = set()

    def _create_inference_session(self, providers, provider_options, disabled_optimizers=None):
        available_providers = C.get_available_providers()

        # Tensorrt can fall back to CUDA if it's explicitly assigned. All others fall back to CPU.
        if "TensorrtExecutionProvider" in available_providers:
            if (
                providers
                and any(
                    provider == "CUDAExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "CUDAExecutionProvider")
                    for provider in providers
                )
                and any(
                    provider == "TensorrtExecutionProvider"
                    or (isinstance(provider, tuple) and provider[0] == "TensorrtExecutionProvider")
                    for provider in providers
                )
            ):
                self._fallback_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        # MIGraphX can fall back to ROCM if it's explicitly assigned. All others fall back to CPU.
        elif "MIGraphXExecutionProvider" in available_providers:
            if providers and any(
                provider == "ROCMExecutionProvider"
                or (isinstance(provider, tuple) and provider[0] == "ROCMExecutionProvider")
                for provider in providers
            ):
                self._fallback_providers = ["ROCMExecutionProvider", "CPUExecutionProvider"]
            else:
                self._fallback_providers = ["CPUExecutionProvider"]
        else:
            self._fallback_providers = ["CPUExecutionProvider"]

        # validate providers and provider_options before other initialization
        providers, provider_options = check_and_normalize_provider_args(
            providers, provider_options, available_providers
        )

        session_options = self._sess_options if self._sess_options else C.get_default_session_options()

        self._register_ep_custom_ops(session_options, providers, provider_options, available_providers)

        if self._model_path:
            sess = C.InferenceSession(session_options, self._model_path, True, self._read_config_from_model)
        else:
            sess = C.InferenceSession(session_options, self._model_bytes, False, self._read_config_from_model)

        if disabled_optimizers is None:
            disabled_optimizers = set()
        elif not isinstance(disabled_optimizers, set):
            # convert to set. assumes iterable
            disabled_optimizers = set(disabled_optimizers)

        # initialize the C++ InferenceSession
>       sess.initialize_session(providers, provider_options, disabled_optimizers)
E       onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could not find an implementation for Where(16) node with name 'Where_0'
E       Falsifying example: test_where(
E           dtype=dtype('bool'),
E           data=data(...),
E       )
E       Draw 1: BroadcastableShapes(input_shapes=((), (), ()), result_shape=())
E       Draw 2: ArrayWrapper(array(False))
E       Draw 3: ArrayWrapper(array(False))
E       Draw 4: ArrayWrapper(array(False))

.pixi/envs/default/lib/python3.13/site-packages/onnxruntime/capi/onnxruntime_inference_collection.py:541: NotImplemented
======================================================================= warnings summary ========================================================================
tests/test_binary.py::test_add[float64]
tests/test_binary.py::test_add[float64]
  /Users/c.bourjau/repos/quantco/onnx-tests/tests/helpers.py:119: RuntimeWarning: overflow encountered in add
    expected = np_fun(arr1.array, arr2.array)

tests/test_unary.py::test_sin[float16]
tests/test_unary.py::test_sin[float16]
tests/test_unary.py::test_sin[float16]
tests/test_unary.py::test_sin[float16]
tests/test_unary.py::test_sin[float16]
tests/test_unary.py::test_sin[float16]
tests/test_unary.py::test_sin[float64]
tests/test_unary.py::test_sin[float64]
  /Users/c.bourjau/repos/quantco/onnx-tests/tests/test_unary.py:24: RuntimeWarning: invalid value encountered in sin
    expected = np_fun(x.array)

tests/test_unary.py: 12 warnings
  /Users/c.bourjau/repos/quantco/onnx-tests/tests/test_unary.py:24: RuntimeWarning: invalid value encountered in cos
    expected = np_fun(x.array)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
==================================================================== short test summary info ====================================================================
FAILED tests/test_slice.py::test_slice[dtype11] - onnxruntime.capi.onnxruntime_pybind11_state.Fail: <class 'UnicodeEncodeError'>: 'utf-8' codec can't encode c...
FAILED tests/test_unary.py::test_cos[float64] - onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could n...
FAILED tests/test_where.py::test_where[int8] - onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could no...
FAILED tests/test_where.py::test_where[int16] - onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could n...
FAILED tests/test_where.py::test_where[uint16] - onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could ...
FAILED tests/test_where.py::test_where[uint32] - onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could ...
FAILED tests/test_where.py::test_where[uint64] - onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could ...
FAILED tests/test_where.py::test_where[<U0] - AssertionError:
FAILED tests/test_where.py::test_where[bool] - onnxruntime.capi.onnxruntime_pybind11_state.NotImplemented: [ONNXRuntimeError] : 9 : NOT_IMPLEMENTED : Could no...
========================================================== 9 failed, 45 passed, 22 warnings in 17.01s ===========================================================
```

</details>
