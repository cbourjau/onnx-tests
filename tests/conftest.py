import re
import warnings
from pathlib import Path
from typing import Any

import pytest
import spox._future
from hypothesis import settings
from onnx.defs import OpSchema, get_all_schemas_with_history


@pytest.fixture(autouse=True)
def disable_spox_value_prop():
    # Disable Value propagation in spox since it will raise a warning
    # whenever it fails or disagrees with onnxruntime. This is
    # happening a lot in this test suite, but is not actionable
    # information.
    with spox._future.value_prop_backend(spox._future.ValuePropBackend.NONE):
        yield


def _parse_xfails_file(path: Path) -> list[tuple[str, str]]:
    """Parse an xfails file into (pattern, reason) tuples.

    Lines starting with ``#`` and blank lines are ignored. A pattern
    may be followed by `` # reason`` to provide a reason for the xfail.
    """
    if not path.is_file():
        return []
    patterns: list[tuple[str, str]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if " # " in line:
            pattern, reason = line.split(" # ", 1)
            pattern = pattern.strip()
        else:
            pattern = line
            reason = ""
        patterns.append((pattern, reason))
    return patterns


def pytest_addoption(parser):
    parser.addoption(
        "--create-report",
        action="store_true",
        default=False,
        help="Create a report for the test coverage of the ONNX standard",
    )

    parser.addoption(
        "--hypothesis-max-examples",
        action="store",
        default=100,
        type=int,
        help="set the Hypothesis max_examples setting",
    )

    parser.addoption(
        "--xfails-file",
        action="append",
        default=[],
        help="Path to a file with patterns of tests to mark as xfail (use * as wildcard). Can be passed multiple times.",
    )

    parser.addoption(
        "--skips-file",
        action="append",
        default=[],
        help="Path to a file with patterns of tests to skip (use * as wildcard). Can be passed multiple times.",
    )


def pytest_configure(config: pytest.Config):
    # Hypothesis
    settings.register_profile(
        "onnx-tests",
        max_examples=config.getoption("--hypothesis-max-examples"),
        deadline=None,
    )
    settings.load_profile("onnx-tests")


def _apply_marker_from_files(config, items, option, marker_factory):
    """Apply a pytest marker to tests matching patterns from files."""
    patterns: list[tuple[str, str]] = []
    for path in config.getoption(option):
        patterns.extend(_parse_xfails_file(Path(path)))

    if not patterns:
        return

    matched: set[str] = set()
    for item in items:
        for pattern, reason in patterns:
            regex = re.escape(pattern).replace(r"\*", ".*")
            if re.fullmatch(regex, item.nodeid):
                item.add_marker(marker_factory(reason))
                matched.add(pattern)
                break
    for pattern in sorted({p for p, _ in patterns} - matched):
        warnings.warn(f"{option} pattern did not match any test: {pattern!r}")


def pytest_collection_modifyitems(session, config, items):
    _apply_marker_from_files(
        config, items, "--skips-file", lambda reason: pytest.mark.skip(reason=reason)
    )
    _apply_marker_from_files(
        config,
        items,
        "--xfails-file",
        lambda reason: pytest.mark.xfail(reason=reason),
    )

    if not config.getoption("--create-report"):
        return

    coverage = get_tested_ops(items)
    create_report(coverage)


def get_tested_ops(items) -> dict[str, list[int]]:
    tested_ops: dict[str, set[int]] = {}
    pattern = re.compile(r"test_(\w*)_(\d+)")
    for item in items:
        m = re.match(pattern, item.name)
        if m is None:
            raise ValueError(
                f"tests must be named as `test_{{OpType}}_{{version}}`, got `{item.name}`"
            )
        op_name, version = m.groups()
        tested_ops.setdefault(op_name, set()).add(int(version))

    return {k: sorted(v) for k, v in tested_ops.items()}


def create_report(coverage: dict[str, list[int]]):
    with open(Path(__file__).parent.parent / "report" / "coverage.md", "w") as f:
        for domain, schemas in _get_op_schemas().items():
            print(f"# {domain}", file=f)

            def make_url(opname):
                base_url = "https://github.com/onnx/onnx/blob/main/docs"
                if domain == "ai.onnx":
                    return f"{base_url}/Operators.md#{opname}"
                elif domain == "ai.onnx.ml":
                    return f"{base_url}/Operators-ml.md#aionnxml{opname.lower()}"
                elif domain == "ai.onnx.preview.training":
                    return f"{base_url}/Operators-ml.md#aionnxpreviewtraining{opname.lower()}"
                raise NotImplementedError(f"unexpected domain: `{domain}`")

            for name, versions in schemas.items():
                print(
                    f"  - [{name}]({make_url(name)})",
                    file=f,
                )
                for version in versions:
                    covered = name in coverage and version in coverage[name]
                    sigil = "x" if covered else " "
                    print(f"    - [{sigil}] {version}", file=f)


def _sort_by_key(dct: dict[Any, Any]) -> dict[Any, Any]:
    return dict(sorted(dct.items(), key=lambda item: item[0]))


Domain = str
Version = int
Name = str


def _get_op_schemas() -> dict[Domain, dict[Name, dict[Version, OpSchema]]]:
    ALL_SCHEMAS: list[OpSchema] = get_all_schemas_with_history()  # type: ignore
    out: dict[Domain, dict[Name, dict[Version, OpSchema]]] = {}
    for schema in ALL_SCHEMAS:
        domain_name = schema.domain or "ai.onnx"
        domain = out.setdefault(domain_name, {})
        versions_of_op = domain.setdefault(schema.name, {})
        versions_of_op[schema.since_version] = schema

    # sort domains
    out = _sort_by_key(out)
    # sort op names
    for domain_name, schemas in out.copy().items():
        for name, versions in schemas.copy().items():
            schemas[name] = _sort_by_key(versions)
        out[domain_name] = _sort_by_key(schemas)
    return out
