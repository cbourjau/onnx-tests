import re
from pathlib import Path
from typing import Any

import pytest
import spox._future
from onnx.defs import OpSchema, get_all_schemas_with_history


@pytest.fixture(autouse=True)
def disable_spox_value_prop():
    # Disable Value propagation in spox since it will raise a warning
    # whenever it fails or disagrees with onnxruntime. This is
    # happening a lot in this test suite, but is not actionable
    # information.
    with spox._future.value_prop_backend(spox._future.ValuePropBackend.NONE):
        yield


def pytest_addoption(parser):
    parser.addoption(
        "--create-report",
        action="store_true",
        default=False,
        help="Create a report for the test coverage of the ONNX standard",
    )


def pytest_collection_modifyitems(session, config, items):
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
            for name, versions in schemas.items():
                print(f"  - {name}", file=f)
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
