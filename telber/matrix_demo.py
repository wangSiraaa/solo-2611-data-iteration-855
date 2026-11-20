#!/usr/bin/env python3
"""Key-matrix demo: encode the booth example into files, reopen, verify.

The example (see ``schemas/matrix_keys_v1.json`` for the wire layout):

* a 2x2 machine, all four contacts installed, at most 3 simultaneous keys;
* record 0 — two consecutive frames in which all four positions are read.
  Each frame has exactly four three-contact truths (the four spanning trees
  of the 2x2 grid), and no single open/close turns one into another;
* record 1 — the same segment with the second frame corrected to only the
  first row's two read points, so two of the three-contact truths can legally
  release one key into the unique two-contact truth.

Three gesture tables are judged against the imported segments:

* ``gestures_distinct.json`` — each three-contact group fires its own action;
  any single action may appear, two consecutive different actions cannot;
* ``gestures_same.json`` — all four groups fire one shared action; that
  action must appear (exactly once);
* ``gestures_corrected.json`` — adds the two-contact gesture; against the
  corrected segment the consecutive pair ``trio, duo`` appears in every
  feasible history.

Run from the repository root::

    python3 tools/matrix_demo.py --out data/matrix
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telber import build as B  # noqa: E402
from telber import keymatrix as KM  # noqa: E402
from telber.repository import Repository  # noqa: E402
from telber.service import CompileService  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schemas" / "matrix_keys_v1.json"

STATION_ID = 41
INSTALLED = bytes([0x0F])  # contacts 0..3: (0,0) (0,1) (1,0) (1,1)
FRAMES_ORIGINAL = bytes([0x03, 0x03, 0x03, 0x03])  # both frames read all four
FRAMES_CORRECTED = bytes([0x03, 0x03, 0x03, 0x00])  # frame 1: first row only

GESTURE_TABLES = {
    "gestures_distinct.json": {
        "description": "Each three-contact group fires its own action.",
        "gestures": [
            {"required": [[0, 1], [1, 0], [1, 1]], "action": "alpha", "order": 1},
            {"required": [[0, 0], [1, 0], [1, 1]], "action": "beta", "order": 2},
            {"required": [[0, 0], [0, 1], [1, 1]], "action": "gamma", "order": 3},
            {"required": [[0, 0], [0, 1], [1, 0]], "action": "delta", "order": 4},
        ],
    },
    "gestures_same.json": {
        "description": "All four three-contact groups fire one shared action.",
        "gestures": [
            {"required": [[0, 1], [1, 0], [1, 1]], "action": "ensemble", "order": 1},
            {"required": [[0, 0], [1, 0], [1, 1]], "action": "ensemble", "order": 2},
            {"required": [[0, 0], [0, 1], [1, 1]], "action": "ensemble", "order": 3},
            {"required": [[0, 0], [0, 1], [1, 0]], "action": "ensemble", "order": 4},
        ],
    },
    "gestures_corrected.json": {
        "description": "The two-contact gesture plus the four three-contact groups.",
        "gestures": [
            {"required": [[0, 0], [0, 1]], "action": "duo", "order": 1},
            {"required": [[0, 1], [1, 0], [1, 1]], "action": "trio", "order": 2},
            {"required": [[0, 0], [1, 0], [1, 1]], "action": "trio", "order": 3},
            {"required": [[0, 0], [0, 1], [1, 1]], "action": "trio", "order": 4},
            {"required": [[0, 0], [0, 1], [1, 0]], "action": "trio", "order": 5},
        ],
    },
}


def segment_record(frames: bytes, *, note: bool = False) -> bytes:
    """Assemble one MatrixScanSegment record with the independent builder."""
    body = B.integer(STATION_ID, cls="context", number=0)
    model = B.integer(2, cls="context", number=0)
    model += B.integer(2, cls="context", number=1)
    model += B.integer(3, cls="context", number=2)
    model += B.octets(INSTALLED, cls="context", number=3)
    body += B.tlv(B.tag("application", True, 301), model)
    body += B.octets(frames, cls="context", number=4)
    if note:
        body += B.explicit(B.tag("context", True, 5), B.null())
    return B.tlv(B.tag("universal", True, 16), body)


def generate_stream() -> bytes:
    """Record 0: the original pair of frames.  Record 1: the corrected pair."""
    return segment_record(FRAMES_ORIGINAL, note=True) + segment_record(FRAMES_CORRECTED)


# (record ordinal, gesture table file, action string, expected verdict)
CHECKS = [
    (0, "gestures_distinct.json", ["alpha"], "some"),
    (0, "gestures_distinct.json", ["beta"], "some"),
    (0, "gestures_distinct.json", ["gamma"], "some"),
    (0, "gestures_distinct.json", ["delta"], "some"),
    (0, "gestures_distinct.json", ["alpha", "beta"], "none"),
    (0, "gestures_distinct.json", ["alpha", "alpha"], "none"),
    (0, "gestures_same.json", ["ensemble"], "all"),
    (0, "gestures_same.json", ["ensemble", "ensemble"], "none"),
    (0, "gestures_corrected.json", ["trio", "duo"], "none"),
    (1, "gestures_corrected.json", ["trio", "duo"], "all"),
    (1, "gestures_corrected.json", ["duo", "trio"], "none"),
    (1, "gestures_corrected.json", ["trio"], "all"),
    (1, "gestures_corrected.json", ["duo"], "all"),
]


def run(out: Path | None = None) -> dict:
    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    service = CompileService()
    product = service.compile(schema_text)
    stream = generate_stream()

    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "product.json").write_text(service.artifact(product), encoding="utf-8")
        (out / "matrix.bin").write_bytes(stream)
        for name, table in GESTURE_TABLES.items():
            (out / name).write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
        if (out / "matrix.sqlite").exists():
            (out / "matrix.sqlite").unlink()

    report: dict = {
        "product_id": product.product_id,
        "schema": product.schema_name,
        "schema_version": product.version,
        "stream_bytes": len(stream),
        "records": {},
        "queries": [],
        "checks_passed": 0,
        "checks_failed": 0,
    }

    def ingest(repo) -> int:
        repo.publish_product(schema_text)
        import_id = repo.create_import(product.product_id, source_name="matrix.bin")
        info = repo.ingest_bytes(import_id, stream, block_size=5)
        report["import"] = info.as_dict()
        return import_id

    tables = {name: json.loads(json.dumps(text)) for name, text in GESTURE_TABLES.items()}

    def verify(repo, import_id: int) -> None:
        dags = {}
        for ordinal in (0, 1):
            segment = KM.load_segment(repo, import_id, ordinal)
            dag = segment.build_dag()
            dags[ordinal] = dag
            report["records"][str(ordinal)] = dag.as_dict()
        for ordinal, table_name, actions, expected in CHECKS:
            table = KM.GestureTable.from_json(tables[table_name], dags[ordinal].model)
            result = KM.query_actions(dags[ordinal], table, actions)
            entry = result.as_dict()
            entry["record"] = ordinal
            entry["gestures"] = table_name
            entry["expected_verdict"] = expected
            entry["ok"] = result.verdict == expected
            report["queries"].append(entry)
            if entry["ok"]:
                report["checks_passed"] += 1
            else:
                report["checks_failed"] += 1
        # The machine's possible trajectories must not depend on the gesture
        # table: every query against one record carries the same DAG signature.
        for ordinal in (0, 1):
            sigs = {
                q["dag_signature"] for q in report["queries"] if q["record"] == ordinal
            }
            assert len(sigs) == 1 and sigs.pop() == dags[ordinal].signature()
        assert dags[0].signature() != dags[1].signature()

    if out is None:
        # An in-memory database cannot be reopened; ingest and verify share
        # the one connection.  The file-backed run below is the one that
        # demonstrates the close/reopen cycle.
        with Repository(":memory:") as repo:
            import_id = ingest(repo)
            verify(repo, import_id)
        return report

    # Ingest, close, and reopen: the archive reader works from the committed
    # facts alone, never from the original byte stream.
    db_path = str(out / "matrix.sqlite")
    with Repository(db_path) as repo:
        import_id = ingest(repo)
    with Repository(db_path) as repo:
        verify(repo, import_id)

    (out / "report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="write artifacts here")
    args = parser.parse_args(argv)
    report = run(args.out)
    printable = {k: v for k, v in report.items() if k not in ("records", "queries")}
    print(json.dumps(printable, indent=2))
    print("\nqueries:")
    for q in report["queries"]:
        mark = "ok " if q["ok"] else "FAIL"
        print(
            f"  [{mark}] record {q['record']} {q['gestures']}: "
            f"{','.join(q['query'])} -> {q['verdict']} (expected {q['expected_verdict']})"
        )
    if report["checks_failed"]:
        print(f"\n{report['checks_failed']} check(s) FAILED", file=sys.stderr)
        return 1
    print(f"\nall {report['checks_passed']} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
