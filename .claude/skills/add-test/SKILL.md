---
name: add-test
description: Add ONNX conformance tests for a specified operator. Use when adding new operator tests to this suite.
---

# Instructions

## The spec is the source of truth

- Look up the operator's spec from the spox docstring (e.g. `spox.opset.ai.onnx.v21.add.__doc__`). The
  docstring contains the full signature, input/output optionality, type constraints, and semantics.
- **Never execute a model, spox, the reference, or onnxruntime to "discover" how an operator behaves.**
  The docstring and the ONNX spec fully describe the intended behaviour. Implementations may be buggy;
  observing them teaches you the bug, not the spec. Only run code to _verify a finished test_ (see below).
- The ONNX spec is the source of truth — not the reference implementation or onnxruntime.

## Writing the test

- Test cases use hypothesis: a `@st.composite` function builds a `TestCaseDraw` instance. The
  `spox_fun` may return a tuple for multi-output operators (see `Unique`, `Where`).
- Look at operators with identical/similar signatures for examples. Reuse logic (e.g. Sum, Min, Max, Mean).
- Uncommon signatures (e.g. `Pow`, `Conv`) show how to parametrize over tuples of input dtypes.
- Most tests wire up via `make_test(...)`, which parametrizes over `op, dtype` and draws everything else.
  **Deviate from `make_test` and write the test by hand when a single discrete choice cleanly separates a
  working code path from a failing/buggy one.** Make that choice a `@pytest.mark.parametrize` argument of
  the strategy (not an internal `draw`), so each variant gets its own test ID. This yields a sharp signal —
  e.g. only the buggy variant is xfailed and the working variant stays green — instead of a blanket xfail
  that hides passing behaviour. (See `test_attention.py` parametrizing over `use_past`.)
- Include interesting values and shapes (zero-sized, integer overflows, infs, nan). Don't restrict data
  generation unnecessarily — and never restrict it to dodge an implementation bug (xfail instead).
- Python supports negative indices; no need to convert them to positive integers.
- Keep it simple and Pythonic. Follow the file-naming schema from the array-api test suite where applicable:
  https://github.com/data-apis/array-api-tests/tree/master/array_api_tests

## Handling implementation bugs (xfails)

- If the reference implementation or onnxruntime disagrees with the spec, add the failing test to the
  appropriate `xfails-*.txt` file with a ` # reason` explaining the bug. Never adapt the test to the bug.
- **Write explicit, fully-qualified test IDs — do not use `*` wildcards.** List each failing parametrization
  on its own line. Scope as narrowly as the signal allows (a bug on one dtype or one variant should not
  xfail the others).
- `xfails-reference.txt` applies to both runs; `xfails-ort.txt` is for ORT-only failures. Don't duplicate a
  reference failure into the ORT file.

## Verifying

- Run the suite through the pixi tasks, which already set `RUN_CANDIDATE` and the right `--xfails-file`/
  `--skips-file` flags. Append extra pytest flags directly:
  - `pixi run test-reference -v -k <OpName> --hypothesis-max-examples=200`
  - `pixi run test-ort -v -k <OpName> --hypothesis-max-examples=200`
- A clean run shows your new tests as `PASSED` or `XFAIL` (never `FAILED` or unexpected `XPASS`). The
  conftest warns on xfail patterns that match no test — make sure none of _your_ patterns are listed.

## Finishing up

- Run `pixi run create-report && pixi run pre-commit-run -a` to regenerate the coverage report.
- `create-report` rewrites `report/coverage.md`; the raw git diff looks enormous because `pre-commit`'s
  `prettier` hook then reformats the whole file. After pre-commit runs (it may need a second pass until all
  hooks pass), the real diff is small — your operator's version flips to `[x]`.
- **If `create-report` crashes, fix the tooling — never hand-edit `report/coverage.md`.** The report is a
  generated artifact; editing it by hand leaves the generator broken and the report wrong.
