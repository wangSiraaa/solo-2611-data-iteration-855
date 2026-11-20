"""Key-matrix product: ghost readout, history DAG, gesture queries, repository."""

from __future__ import annotations

import json
import random

import pytest

from conftest import field, make_schema
from telber import build as B
from telber import keymatrix as KM
from telber.cli import main as cli_main
from telber.codec import encode_record
from telber.errors import MatrixError
from telber.repository import Repository
from telber.service import CompileService

SCHEMA_PATH = __file__.rsplit("/", 2)[0] + "/schemas/matrix_keys_v1.json"

MODEL_2X2 = KM.MachineModel(2, 2, 3, frozenset({(0, 0), (0, 1), (1, 0), (1, 1)}))
ALL_READ = (0b11, 0b11)


def segment_bytes(frames, *, rows=2, cols=2, max_keys=3, installed=0x0F, note=False):
    """One MatrixScanSegment record built with the independent byte builder."""
    body = B.integer(41, cls="context", number=0)
    model = B.integer(rows, cls="context", number=0)
    model += B.integer(cols, cls="context", number=1)
    model += B.integer(max_keys, cls="context", number=2)
    model += B.octets(bytes([installed]), cls="context", number=3)
    body += B.tlv(B.tag("application", True, 301), model)
    body += B.octets(bytes(frames), cls="context", number=4)
    if note:
        body += B.explicit(B.tag("context", True, 5), B.null())
    return B.tlv(B.tag("universal", True, 16), body)


def gesture_table(entries):
    return KM.GestureTable.from_json({"gestures": entries}, MODEL_2X2)


# ---------------------------------------------------------------------------
# Circuit model: readout and ghosting


def test_three_real_contacts_read_as_four():
    # The classic ghost: (0,0), (1,0), (1,1) closed -> all four positions read.
    state = MODEL_2X2.mask_of({(0, 0), (1, 0), (1, 1)})
    assert KM.readout(MODEL_2X2, state) == ALL_READ


def test_ghost_chain_across_rows():
    model = KM.MachineModel(
        3, 4, 4, frozenset({(0, 0), (1, 0), (1, 3), (2, 3)})
    )
    state = model.mask_of({(0, 0), (1, 0), (1, 3), (2, 3)})
    # Every row of the chain reads both columns 0 and 3.
    assert KM.readout(model, state) == (0b1001, 0b1001, 0b1001)


def test_readout_of_empty_state_is_empty():
    assert KM.readout(MODEL_2X2, 0) == (0, 0)


def test_single_contact_reads_exactly_itself():
    state = MODEL_2X2.mask_of({(1, 0)})
    assert KM.readout(MODEL_2X2, state) == (0, 0b01)


# ---------------------------------------------------------------------------
# Model and frame parsing


def test_model_bounds():
    with pytest.raises(MatrixError):
        KM.MachineModel(4, 2, 3, frozenset())
    with pytest.raises(MatrixError):
        KM.MachineModel(2, 5, 3, frozenset())
    with pytest.raises(MatrixError):
        KM.MachineModel(2, 2, 5, frozenset())
    with pytest.raises(MatrixError):
        KM.MachineModel(2, 2, 0, frozenset())
    with pytest.raises(MatrixError):
        KM.MachineModel(2, 2, 3, frozenset({(2, 0)}))


def test_installed_mask_roundtrip_and_padding():
    installed = KM.parse_installed(2, 2, bytes([0b0101]))
    assert installed == frozenset({(0, 0), (1, 0)})
    with pytest.raises(MatrixError) as exc:
        KM.parse_installed(2, 2, bytes([0b0001_0000]))
    assert exc.value.code == "installed_padding_bits"
    with pytest.raises(MatrixError) as exc:
        KM.parse_installed(3, 4, bytes([0xFF]))
    assert exc.value.code == "installed_length"


def test_frames_parsing_errors():
    with pytest.raises(MatrixError) as exc:
        KM.parse_frames(MODEL_2X2, bytes([0x03]))  # 1 octet for 2 rows
    assert exc.value.code == "frame_bytes_misaligned"
    with pytest.raises(MatrixError) as exc:
        KM.parse_frames(MODEL_2X2, bytes([0x08, 0x00]))  # bit 3 with 2 cols
    assert exc.value.code == "frame_padding_bits"
    assert exc.value.frame_index == 0
    with pytest.raises(MatrixError) as exc:
        KM.parse_frames(MODEL_2X2, bytes(2 * 81))
    assert exc.value.code == "frame_count"
    with pytest.raises(MatrixError):
        KM.parse_frames(MODEL_2X2, b"")


def test_frames_roundtrip():
    frames = KM.parse_frames(MODEL_2X2, bytes([0x03, 0x03, 0x03, 0x00]))
    assert frames == [(0b11, 0b11), (0b11, 0)]


# ---------------------------------------------------------------------------
# Per-frame true states


def test_all_read_frame_has_exactly_four_three_key_truths():
    states = KM.explaining_states(MODEL_2X2, ALL_READ)
    assert len(states) == 4
    assert all(s.bit_count() == 3 for s in states)


def test_all_read_frame_with_four_keys_allowed_adds_full_set():
    model = KM.MachineModel(2, 2, 4, MODEL_2X2.installed)
    states = KM.explaining_states(model, ALL_READ)
    assert len(states) == 5
    assert model.installed_mask in states


def test_empty_observation_means_empty_state():
    assert KM.explaining_states(MODEL_2X2, (0, 0)) == [0]


def test_uninstalled_contacts_never_explain():
    model = KM.MachineModel(2, 2, 3, frozenset({(0, 0)}))
    # Row 0 reads column 1, but no installed contact can connect them.
    assert KM.explaining_states(model, (0b10, 0)) == []


# ---------------------------------------------------------------------------
# The history DAG


def test_two_all_read_frames_allow_only_identity_transitions():
    seg = KM.ScanSegment.from_values(
        rows=2, cols=2, max_keys=3, installed=bytes([0x0F]), frames=bytes([3, 3, 3, 3])
    )
    dag = seg.build_dag()
    assert dag.status == "ok"
    assert [len(layer) for layer in dag.layers] == [4, 4]
    # Distinct three-contact truths differ by two contacts, so each state can
    # only keep itself: no single open/close switches truths.
    for t, layer in enumerate(dag.transitions):
        for i, outs in enumerate(layer):
            assert outs == frozenset({i})
    assert len(list(dag.iter_histories())) == 4


def test_corrected_second_frame_trims_states_off_every_path():
    seg = KM.ScanSegment.from_values(
        rows=2, cols=2, max_keys=3, installed=bytes([0x0F]), frames=bytes([3, 3, 3, 0])
    )
    dag = seg.build_dag()
    assert dag.status == "ok"
    # Frame 0 alone still has four explaining truths...
    assert dag.explaining_counts == (4, 1)
    # ...but only the two that contain the first row's contacts can release
    # one key into the unique frame-1 truth; the others sit on no history.
    assert [len(layer) for layer in dag.layers] == [2, 1]
    stray = MODEL_2X2.mask_of({(0, 1), (1, 0), (1, 1)})  # missing (0, 0)
    assert stray not in dag.layers[0]
    for history in dag.iter_histories():
        assert history[0] != stray
        assert history[1] == MODEL_2X2.mask_of({(0, 0), (0, 1)})
        assert (history[0] ^ history[1]).bit_count() == 1  # one key released


def test_earliest_contradictory_frame_is_reported():
    # all-read needs 3 contacts; the empty frame needs none; 3 != 1 change.
    seg = KM.ScanSegment.from_values(
        rows=2, cols=2, max_keys=3, installed=bytes([0x0F]),
        frames=bytes([3, 3, 0, 0, 3, 3]),
    )
    dag = seg.build_dag()
    assert dag.status == "contradiction"
    assert dag.contradiction_frame == 1
    assert dag.analyzed_frames == 1  # the feasible prefix survives
    assert list(dag.iter_histories()) == []


def test_contradiction_at_the_first_frame():
    model = KM.MachineModel(2, 2, 3, frozenset({(0, 0)}))
    dag = KM.HistoryDag.build(model, [(0b10, 0)])
    assert dag.status == "contradiction"
    assert dag.contradiction_frame == 0
    assert dag.analyzed_frames == 0


def test_query_on_contradictory_segment_has_no_verdict():
    seg = KM.ScanSegment.from_values(
        rows=2, cols=2, max_keys=3, installed=bytes([0x0F]), frames=bytes([3, 3, 0, 0])
    )
    dag = seg.build_dag()
    table = gesture_table([{"required": [[0, 0]], "action": "a", "order": 1}])
    result = KM.query_actions(dag, table, ["a"])
    assert result.status == "contradiction"
    assert result.verdict is None
    assert result.contradiction_frame == 1


# ---------------------------------------------------------------------------
# Gesture table semantics


def test_frame_action_picks_most_required_contacts():
    table = gesture_table(
        [
            {"required": [[0, 0], [0, 1]], "action": "duo", "order": 1},
            {"required": [[0, 0], [0, 1], [1, 0]], "action": "trio", "order": 2},
        ]
    )
    state = MODEL_2X2.mask_of({(0, 0), (0, 1), (1, 0)})
    assert table.action_for(state) == "trio"
    assert table.action_for(MODEL_2X2.mask_of({(0, 0), (0, 1)})) == "duo"
    assert table.action_for(0) is None


def test_same_length_tie_broken_by_order():
    table = gesture_table(
        [
            {"required": [[0, 1]], "action": "q", "order": 2},
            {"required": [[0, 0]], "action": "p", "order": 1},
        ]
    )
    state = MODEL_2X2.mask_of({(0, 0), (0, 1)})
    assert table.action_for(state) == "p"


def test_gesture_table_validation():
    with pytest.raises(MatrixError):
        gesture_table([{"required": [], "action": "a", "order": 1}])
    with pytest.raises(MatrixError):
        gesture_table([{"required": [[0, 0], [0, 0]], "action": "a", "order": 1}])
    with pytest.raises(MatrixError):
        gesture_table([{"required": [[0, 2]], "action": "a", "order": 1}])
    with pytest.raises(MatrixError):
        gesture_table([{"required": [[0, 0]], "action": "", "order": 1}])
    with pytest.raises(MatrixError):
        gesture_table([{"required": [[0, 0]], "action": "a", "order": True}])
    with pytest.raises(MatrixError):  # same length, same precedence
        gesture_table(
            [
                {"required": [[0, 0]], "action": "a", "order": 1},
                {"required": [[0, 1]], "action": "b", "order": 1},
            ]
        )
    # Same precedence is fine across different lengths.
    gesture_table(
        [
            {"required": [[0, 0]], "action": "a", "order": 1},
            {"required": [[0, 0], [0, 1]], "action": "b", "order": 1},
        ]
    )


def test_trigger_stream_fires_only_on_change_to_non_empty():
    actions = [None, "a", "a", None, "a", "b", "b", None, None, "b"]
    assert KM.trigger_stream(actions) == ["a", "a", "b", "b"]


# ---------------------------------------------------------------------------
# Action-string queries


def test_query_input_validation():
    seg = KM.ScanSegment.from_values(
        rows=2, cols=2, max_keys=3, installed=bytes([0x0F]), frames=bytes([3, 3, 3, 3])
    )
    dag = seg.build_dag()
    table = gesture_table([{"required": [[0, 0]], "action": "a", "order": 1}])
    with pytest.raises(MatrixError):
        KM.query_actions(dag, table, [])
    with pytest.raises(MatrixError):
        KM.query_actions(dag, table, ["a", ""])


def test_query_longer_than_segment_is_none_with_counter_witness():
    seg = KM.ScanSegment.from_values(
        rows=2, cols=2, max_keys=3, installed=bytes([0x0F]), frames=bytes([3, 3, 3, 3])
    )
    dag = seg.build_dag()
    table = gesture_table([{"required": [[0, 0]], "action": "a", "order": 1}])
    result = KM.query_actions(dag, table, ["a", "a", "a"])
    assert result.verdict == "none"
    assert result.witness is None
    assert result.counter_witness is not None


def _brute_force_verdict(dag, table, query):
    flags = []
    for history in dag.iter_histories(limit=2_000_000):
        triggers = KM.trigger_stream([table.action_for(s) for s in history])
        m = len(query)
        flags.append(any(triggers[i : i + m] == list(query) for i in range(len(triggers) - m + 1)))
    assert flags, "DAG must yield at least one history"
    if all(flags):
        return "all"
    if any(flags):
        return "some"
    return "none"


def _check_witness(dag, table, query, result):
    if result.verdict in ("all", "some"):
        w = result.witness
        assert w is not None
        states = [dag.model.mask_of(s) for s in w["states"]]
        # The witness is a real history: explains every frame, one change max.
        for t, s in enumerate(states):
            assert KM.readout(dag.model, s) == dag.observations[t]
        for a, b in zip(states, states[1:]):
            assert (a ^ b).bit_count() <= 1
        triggers = KM.trigger_stream([table.action_for(s) for s in states])
        assert triggers == w["triggers"]
        m = len(query)
        assert any(
            triggers[i : i + m] == list(query) for i in range(len(triggers) - m + 1)
        )
    else:
        assert result.witness is None
    if result.verdict in ("some", "none"):
        cw = result.counter_witness
        assert cw is not None
        states = [dag.model.mask_of(s) for s in cw["states"]]
        triggers = KM.trigger_stream([table.action_for(s) for s in states])
        m = len(query)
        assert not any(
            triggers[i : i + m] == list(query) for i in range(len(triggers) - m + 1)
        )
    else:
        assert result.counter_witness is None


def test_random_queries_match_brute_force():
    rng = random.Random(20261120)
    grids = [(2, 2), (2, 3), (3, 2), (3, 3)]
    for trial in range(60):
        rows, cols = grids[trial % len(grids)]
        all_contacts = [(r, c) for r in range(rows) for c in range(cols)]
        installed = frozenset(
            rng.sample(all_contacts, rng.randint(max(2, len(all_contacts) - 3), len(all_contacts)))
        )
        max_keys = rng.randint(1, 4)
        model = KM.MachineModel(rows, cols, max_keys, installed)
        # A random legal walk produces guaranteed-feasible frames.
        frames = []
        state = 0
        for _ in range(rng.randint(1, 5)):
            if rng.random() < 0.7:
                bit = 1 << rng.randrange(rows * cols)
                if bit & state or (state.bit_count() < max_keys and bit & model.installed_mask):
                    state ^= bit
                    state &= model.installed_mask
            frames.append(KM.readout(model, state))
        dag = KM.HistoryDag.build(model, frames)
        assert dag.status == "ok"
        entries = []
        orders = {}
        for _ in range(rng.randint(1, 5)):
            size = rng.randint(1, 3)
            required = rng.sample(all_contacts, size)
            orders[size] = orders.get(size, 0) + 1
            entries.append(
                {
                    "required": [list(c) for c in required],
                    "action": rng.choice("abc"),
                    "order": orders[size],
                }
            )
        table = KM.GestureTable.from_json({"gestures": entries}, model)
        query = [rng.choice("abcd") for _ in range(rng.randint(1, 3))]
        result = KM.query_actions(dag, table, query)
        assert result.verdict == _brute_force_verdict(dag, table, query), (
            f"trial {trial}: query {query}"
        )
        _check_witness(dag, table, query, result)


def test_trajectories_do_not_depend_on_gesture_table():
    seg = KM.ScanSegment.from_values(
        rows=2, cols=2, max_keys=3, installed=bytes([0x0F]), frames=bytes([3, 3, 3, 3])
    )
    dag = seg.build_dag()
    t1 = gesture_table([{"required": [[0, 0]], "action": "a", "order": 1}])
    t2 = gesture_table([{"required": [[0, 0], [0, 1], [1, 0]], "action": "b", "order": 1}])
    r1 = KM.query_actions(dag, t1, ["a"])
    r2 = KM.query_actions(dag, t2, ["b"])
    assert r1.dag_signature == r2.dag_signature == dag.signature()


# ---------------------------------------------------------------------------
# The spec's 2x2 example, end to end through the repository


def _ingest_example(db_path):
    schema_text = open(SCHEMA_PATH, encoding="utf-8").read()
    stream = segment_bytes([3, 3, 3, 3], note=True) + segment_bytes([3, 3, 3, 0])
    with Repository(db_path) as repo:
        product = repo.publish_product(schema_text)
        import_id = repo.create_import(product.product_id, source_name="example.bin")
        info = repo.ingest_bytes(import_id, stream, block_size=5)
        assert info.status == "complete" and info.record_count == 2
    return import_id


DISTINCT = {
    "gestures": [
        {"required": [[0, 1], [1, 0], [1, 1]], "action": "alpha", "order": 1},
        {"required": [[0, 0], [1, 0], [1, 1]], "action": "beta", "order": 2},
        {"required": [[0, 0], [0, 1], [1, 1]], "action": "gamma", "order": 3},
        {"required": [[0, 0], [0, 1], [1, 0]], "action": "delta", "order": 4},
    ]
}
SAME = {
    "gestures": [
        {"required": [[0, 1], [1, 0], [1, 1]], "action": "ensemble", "order": 1},
        {"required": [[0, 0], [1, 0], [1, 1]], "action": "ensemble", "order": 2},
        {"required": [[0, 0], [0, 1], [1, 1]], "action": "ensemble", "order": 3},
        {"required": [[0, 0], [0, 1], [1, 0]], "action": "ensemble", "order": 4},
    ]
}
CORRECTED = {
    "gestures": [
        {"required": [[0, 0], [0, 1]], "action": "duo", "order": 1},
        {"required": [[0, 1], [1, 0], [1, 1]], "action": "trio", "order": 2},
        {"required": [[0, 0], [1, 0], [1, 1]], "action": "trio", "order": 3},
        {"required": [[0, 0], [0, 1], [1, 1]], "action": "trio", "order": 4},
        {"required": [[0, 0], [0, 1], [1, 0]], "action": "trio", "order": 5},
    ]
}


def test_example_reopened_repository(tmp_path):
    db = str(tmp_path / "matrix.sqlite")
    import_id = _ingest_example(db)

    # Reopen the import library: everything below runs on committed facts.
    with Repository(db) as repo:
        seg0 = KM.load_segment(repo, import_id, 0)
        seg1 = KM.load_segment(repo, import_id, 1)
    assert seg0.station_id == 41
    assert seg0.frames == (ALL_READ, ALL_READ)
    assert seg1.frames == (ALL_READ, (0b11, 0))

    dag0, dag1 = seg0.build_dag(), seg1.build_dag()
    distinct = gesture_table(DISTINCT["gestures"])
    same = gesture_table(SAME["gestures"])
    corrected = gesture_table(CORRECTED["gestures"])

    # Distinct actions: any single one may appear; two different in a row cannot.
    for action in ("alpha", "beta", "gamma", "delta"):
        assert KM.query_actions(dag0, distinct, [action]).verdict == "some"
    assert KM.query_actions(dag0, distinct, ["alpha", "beta"]).verdict == "none"

    # One shared action: it must appear, and it appears exactly once.
    assert KM.query_actions(dag0, same, ["ensemble"]).verdict == "all"
    assert KM.query_actions(dag0, same, ["ensemble", "ensemble"]).verdict == "none"

    # Corrected second frame: the two-contact gesture makes trio,duo certain.
    assert KM.query_actions(dag1, corrected, ["trio", "duo"]).verdict == "all"
    assert KM.query_actions(dag1, corrected, ["duo", "trio"]).verdict == "none"
    # ...while the same pair is impossible on the uncorrected segment.
    assert KM.query_actions(dag0, corrected, ["trio", "duo"]).verdict == "none"


def test_load_segment_errors(tmp_path):
    db = str(tmp_path / "matrix.sqlite")
    import_id = _ingest_example(db)
    with Repository(db) as repo:
        with pytest.raises(MatrixError) as exc:
            KM.load_segment(repo, import_id, 9)
        assert exc.value.code == "unknown_record"


def test_schema_product_roundtrip():
    service = CompileService()
    product = service.compile(open(SCHEMA_PATH, encoding="utf-8").read())
    values = {
        "station_id": 41,
        "model": {"rows": 2, "cols": 2, "max_keys": 3, "installed": bytes([0x0F])},
        "frames": bytes([3, 3, 3, 3]),
        "note": None,
    }
    data = service.encode(product, values)
    checked = service.check(product, data)
    assert checked.integers["model.rows"] == 2
    assert checked.octets["frames"] == bytes([3, 3, 3, 3])
    assert checked.nulls["note"] is True
    seg = KM.ScanSegment.from_values(
        rows=checked.integers["model.rows"],
        cols=checked.integers["model.cols"],
        max_keys=checked.integers["model.max_keys"],
        installed=checked.octets["model.installed"],
        frames=checked.octets["frames"],
    )
    assert seg.frames == (ALL_READ, ALL_READ)


# ---------------------------------------------------------------------------
# CLI and supported bounds


def test_cli_segment_and_query(tmp_path, capsys):
    db = str(tmp_path / "matrix.sqlite")
    import_id = _ingest_example(db)
    gestures = tmp_path / "gestures.json"
    gestures.write_text(json.dumps(CORRECTED), encoding="utf-8")

    assert cli_main(["matrix-segment", db, "--import-id", str(import_id), "--record", "1"]) == 0
    dump = json.loads(capsys.readouterr().out)
    assert dump["status"] == "ok"
    assert dump["frames"][0]["explaining_states"] == 4
    assert dump["frames"][0]["on_path_states"] == 2

    rc = cli_main(
        [
            "matrix-query", db, "--import-id", str(import_id), "--record", "1",
            "--gestures", str(gestures), "--actions", "trio,duo",
        ]
    )
    assert rc == 0
    result = json.loads(capsys.readouterr().out)
    assert result["verdict"] == "all"
    assert result["witness"]["triggers"] == ["trio", "duo"]

    assert cli_main(["matrix-segment", db, "--import-id", str(import_id), "--record", "7"]) == 2
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == "unknown_record"


def test_supported_bounds_3x4_4keys_80frames():
    rows, cols, max_keys = 3, 4, 4
    installed = frozenset((r, c) for r in range(rows) for c in range(cols))
    model = KM.MachineModel(rows, cols, max_keys, installed)
    rng = random.Random(7)
    state = 0
    frames = []
    for _ in range(80):
        bit = 1 << rng.randrange(rows * cols)
        if bit & state or state.bit_count() < max_keys:
            state ^= bit
        frames.extend(KM.readout(model, state))
    seg = KM.ScanSegment.from_values(
        rows=rows,
        cols=cols,
        max_keys=max_keys,
        installed=(model.installed_mask).to_bytes(2, "little"),
        frames=bytes(frames),
    )
    assert len(seg.frames) == 80
    dag = seg.build_dag()
    assert dag.status == "ok"
    assert dag.analyzed_frames == 80
    table = KM.GestureTable.from_json(
        {"gestures": [{"required": [[0, 0]], "action": "tap", "order": 1}]}, model
    )
    result = KM.query_actions(dag, table, ["tap"])
    assert result.status == "ok"
    assert result.verdict in ("all", "some", "none")
