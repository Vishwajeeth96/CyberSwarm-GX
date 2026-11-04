"""Export one JSON Schema per registered record type (CONTRACTS section 2).

``python -m cyberswarm.contracts.export_schemas --out schemas`` writes
``schemas/<RecordName>.schema.json`` for every entry in
:data:`cyberswarm.contracts.registry.RECORD_TYPES`. Each file is produced from
pydantic v2 :meth:`~pydantic.BaseModel.model_json_schema` with two keys added
that pydantic does not emit itself:

* ``"$schema"`` - ``https://json-schema.org/draft/2020-12/schema`` (pydantic
  documents that its schemas follow JSON Schema Draft 2020-12 but omits the
  ``$schema`` key).
* ``"$id"`` - ``urn:cyberswarm:schema:<RecordName>:<SCHEMA_VERSION>``.

The output is deterministic and byte-identical across re-exports: keys are
sorted, the indent is fixed, ASCII is forced, and a single trailing newline is
written. A contract test re-runs this export and fails if the committed files
differ from a fresh one, so the committed ``schemas/`` tree is exactly this
module's output.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .common import SCHEMA_VERSION
from .registry import RECORD_TYPES

#: JSON Schema dialect the exported schemas declare (CONTRACTS section 2).
JSON_SCHEMA_DIALECT: str = "https://json-schema.org/draft/2020-12/schema"


def schema_id(record_name: str) -> str:
    """Return the stable ``$id`` URN for a record name (CONTRACTS section 2)."""

    return f"urn:cyberswarm:schema:{record_name}:{SCHEMA_VERSION}"


def schema_for(record_name: str) -> dict:
    """Build the full JSON Schema document for one registered record type.

    The returned mapping is the pydantic schema with ``$schema`` and ``$id``
    added. The caller serializes it; this function performs no I/O.
    """

    model_cls = RECORD_TYPES[record_name]
    schema = model_cls.model_json_schema()
    # Our two keys always win over anything pydantic might emit in future.
    schema["$schema"] = JSON_SCHEMA_DIALECT
    schema["$id"] = schema_id(record_name)
    return schema


def _render(schema: dict) -> str:
    """Serialize a schema deterministically (sorted keys, fixed indent, LF)."""

    return json.dumps(schema, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def export(out_dir: Path) -> list[Path]:
    """Write one ``<Name>.schema.json`` per registered record type.

    Returns the written file paths sorted by record name. The output directory
    is created if it does not exist. Existing files are overwritten with
    byte-identical content, so re-exporting leaves the working tree unchanged.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for record_name in sorted(RECORD_TYPES):
        path = out_dir / f"{record_name}.schema.json"
        path.write_text(_render(schema_for(record_name)), encoding="utf-8")
        written.append(path)
    return written


def _format_path(path: Path) -> str:
    """Show a path relative to the current directory when possible."""

    try:
        return os.path.relpath(path)
    except ValueError:  # e.g. different drive on Windows
        return str(path)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m cyberswarm.contracts.export_schemas``."""

    parser = argparse.ArgumentParser(
        prog="python -m cyberswarm.contracts.export_schemas",
        description="Export one JSON Schema per registered contract record type.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("schemas"),
        help="Output directory for <Name>.schema.json files (default: schemas).",
    )
    args = parser.parse_args(argv)

    written = export(args.out)
    for path in written:
        print(_format_path(path))
    print(f"wrote {len(written)} schema file(s) to {_format_path(args.out)}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via the CLI
    raise SystemExit(main())
