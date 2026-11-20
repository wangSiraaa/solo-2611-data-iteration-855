"""Key-matrix scan segments: ghost-contact histories and gesture queries.

This module connects the matrix-keyboard scan product
(``schemas/matrix_keys_v1.json``) to the telber type directory, the real wire
import and the fact repository.  It is pure analysis over committed field
facts: it never re-reads the original byte stream, so an archive reader can
independently pull one whole segment of possible contact history out of the
repository and then judge it against the gesture table actually loaded on the
booth.  The original telemetry entry points are untouched.

Circuit model
-------------
The booth recorder scans a key matrix without isolation diodes, row by row;
one sample is the set of columns each row reads at the same moment.  A real
closed contact connects its row and its column; scanning a row reads every
column connected to it *through closed contacts* (transitively) — which is
why three real contacts can be read as four (the classic ghost).  The machine
model fixes the installed contacts and the maximum number of simultaneously
pressed keys.  Between any two adjacent frames at most one contact opens or
closes; the first frame may start from any legal state.

Possible histories
------------------
Reading a segment keeps *every* real key history that explains all frames
simultaneously.  The compact form is a layered DAG: one layer of true contact
sets per frame, an edge between adjacent layers exactly when the two sets
differ by at most one contact, trimmed so every kept state lies on at least
one complete history.  Per-frame feasible sets alone would be lossy — a state
that explains its own frame but no whole segment is dropped, so layers can
never be recombined arbitrarily.  Input that cannot be explained reports the
earliest contradictory frame.

Gesture table
-------------
Each gesture record fixes a set of must-press contacts, an action name and a
unique precedence used to break ties between satisfied records of the same
length.  A frame takes the satisfied record with the most must-press contacts
(ties by precedence, no hit means empty); an action truly triggers only when
the action name changes against the previous frame while the current frame is
non-empty (the stream starts empty).  Given an ordered action string, the
query reports whether it appears as a contiguous segment of the trigger
stream in *every* feasible history, only *some*, or *none*, and reads back a
representative contact history supporting the verdict.  A machine's possible
trajectories do not change when the gesture table is swapped: the DAG is
built before, and independently of, any table.

Supported bounds: at most 3 rows x 4 columns, 4 simultaneous keys, 80 frames
per segment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterator, Mapping, Sequence

from .errors import MatrixError
from .schema import canonical_json

MAX_ROWS = 3
MAX_COLS = 4
MAX_KEYS = 4
MAX_FRAMES = 80

VERDICT_ALL = "all"    # the action string appears in every feasible history
VERDICT_SOME = "some"  # it appears in some but not all feasible histories
VERDICT_NONE = "none"  # it appears in no feasible history


def _check_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MatrixError(f"{where} must be an integer", code="invalid_model")
    return value


# ---------------------------------------------------------------------------
# Machine model


@dataclass(frozen=True)
class MachineModel:
    """The booth machine: grid size, simultaneous-key limit, installed contacts.

    ``installed`` holds the physically installed contacts as ``(row, col)``
    pairs; a real closed contact can only ever be one of these.
    """

    rows: int
    cols: int
    max_keys: int
    installed: frozenset

    def __post_init__(self) -> None:
        rows = _check_int(self.rows, "model.rows")
        cols = _check_int(self.cols, "model.cols")
        max_keys = _check_int(self.max_keys, "model.max_keys")
        if not 1 <= rows <= MAX_ROWS:
            raise MatrixError(
                f"model.rows {rows} exceeds the supported maximum {MAX_ROWS}",
                code="unsupported_model",
            )
        if not 1 <= cols <= MAX_COLS:
            raise MatrixError(
                f"model.cols {cols} exceeds the supported maximum {MAX_COLS}",
                code="unsupported_model",
            )
        if not 1 <= max_keys <= MAX_KEYS:
            raise MatrixError(
                f"model.max_keys {max_keys} exceeds the supported maximum {MAX_KEYS}",
                code="unsupported_model",
            )
        contacts = set()
        for c in self.installed:
            if (
                not isinstance(c, (tuple, list))
                or len(c) != 2
                or any(isinstance(v, bool) or not isinstance(v, int) for v in c)
            ):
                raise MatrixError(
                    f"installed contact {c!r} must be a [row, col] pair of integers",
                    code="invalid_model",
                )
            r, col = int(c[0]), int(c[1])
            if not (0 <= r < rows and 0 <= col < cols):
                raise MatrixError(
                    f"installed contact [{r}, {col}] lies outside the "
                    f"{rows}x{cols} grid",
                    code="invalid_model",
                )
            contacts.add((r, col))
        object.__setattr__(self, "installed", frozenset(contacts))

    # -- contact indexing ---------------------------------------------------
    def contact_index(self, row: int, col: int) -> int:
        return row * self.cols + col

    def contact_at(self, index: int) -> tuple[int, int]:
        return divmod(index, self.cols)

    def mask_of(self, contacts) -> int:
        mask = 0
        for r, c in contacts:
            mask |= 1 << self.contact_index(r, c)
        return mask

    def contacts_of(self, mask: int) -> list[tuple[int, int]]:
        out = []
        for k in range(self.rows * self.cols):
            if mask >> k & 1:
                out.append(self.contact_at(k))
        return out

    @property
    def installed_mask(self) -> int:
        return self.mask_of(self.installed)

    def as_dict(self) -> dict:
        return {
            "rows": self.rows,
            "cols": self.cols,
            "max_keys": self.max_keys,
            "installed": [list(c) for c in sorted(self.installed)],
        }


def parse_installed(rows: int, cols: int, data: bytes) -> frozenset:
    """Decode the ``installed`` bitmask: bit ``row*cols + col`` of the string."""
    bits = rows * cols
    expected = (bits + 7) // 8
    if len(data) != expected:
        raise MatrixError(
            f"installed mask must be exactly {expected} octet(s) for a "
            f"{rows}x{cols} grid, got {len(data)}",
            code="installed_length",
        )
    mask = int.from_bytes(data, "little")
    if mask >> bits:
        raise MatrixError(
            f"installed mask sets bits beyond contact {bits - 1}",
            code="installed_padding_bits",
        )
    return frozenset(
        (k // cols, k % cols) for k in range(bits) if mask >> k & 1
    )


def parse_frames(model: MachineModel, data: bytes) -> list[tuple[int, ...]]:
    """Decode the ``frames`` string: one octet per row, bit c = column c read.

    Returns one ``tuple`` of per-row column masks per frame.
    """
    if len(data) == 0 or len(data) % model.rows != 0:
        raise MatrixError(
            f"frames string of {len(data)} octet(s) does not align to "
            f"{model.rows} row(s) per frame",
            code="frame_bytes_misaligned",
        )
    count = len(data) // model.rows
    if not 1 <= count <= MAX_FRAMES:
        raise MatrixError(
            f"segment holds {count} frames; supported range is 1..{MAX_FRAMES}",
            code="frame_count",
        )
    frames = []
    for t in range(count):
        row_masks = []
        for r in range(model.rows):
            mask = data[t * model.rows + r]
            if mask >> model.cols:
                raise MatrixError(
                    f"frame {t} row {r} reads column bits beyond "
                    f"{model.cols - 1}: 0x{mask:02x}",
                    code="frame_padding_bits",
                    frame_index=t,
                )
            row_masks.append(mask)
        frames.append(tuple(row_masks))
    return frames


def observation_contacts(model: MachineModel, obs: Sequence[int]) -> list[list[int]]:
    """The read points of one frame as sorted ``[row, col]`` pairs."""
    return [
        [r, c]
        for r in range(model.rows)
        for c in range(model.cols)
        if obs[r] >> c & 1
    ]


# ---------------------------------------------------------------------------
# Readout and per-frame true states


def readout(model: MachineModel, state: int) -> tuple[int, ...]:
    """The frame a true contact set produces, as per-row column masks.

    A row reads every column sharing its connected component in the graph
    whose edges are the closed contacts — this is where ghost reads come from.
    """
    parent: dict = {}

    def find(x):
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:
            parent[x], x = root, parent.get(x, x)
        return root

    for k in range(model.rows * model.cols):
        if state >> k & 1:
            r, c = model.contact_at(k)
            ra, rb = find(("r", r)), find(("c", c))
            if ra != rb:
                parent[ra] = rb
    obs = []
    for r in range(model.rows):
        mask = 0
        for c in range(model.cols):
            if find(("r", r)) == find(("c", c)):
                mask |= 1 << c
        obs.append(mask)
    return tuple(obs)


def explaining_states(model: MachineModel, obs: Sequence[int]) -> list[int]:
    """All true contact sets (bitmasks) that produce ``obs`` on this machine.

    A candidate is any subset of the installed contacts of at most
    ``max_keys`` members whose readout equals the observed frame.
    """
    contacts = [k for k in range(model.rows * model.cols) if model.installed_mask >> k & 1]
    target = tuple(obs)
    out = []
    for size in range(min(model.max_keys, len(contacts)) + 1):
        for combo in combinations(contacts, size):
            state = 0
            for k in combo:
                state |= 1 << k
            if readout(model, state) == target:
                out.append(state)
    return sorted(out)


def _adjacent(a: int, b: int) -> bool:
    """True iff the two contact sets differ by at most one open/close."""
    return (a ^ b).bit_count() <= 1


# ---------------------------------------------------------------------------
# The possible-history DAG


class HistoryDag:
    """Every real key history that explains a whole segment, in compact form.

    ``layers[t]`` holds the true contact sets (bitmasks, sorted) of frame
    ``t`` that lie on at least one complete history; ``transitions[t][i]``
    lists the layer-``t+1`` indices reachable from layer-``t`` state ``i`` by
    opening or closing at most one contact.  Complete histories are exactly
    the source-to-sink paths — the layers are *not* independently combinable.

    If the frames cannot be jointly explained, ``status`` is
    ``"contradiction"`` and ``contradiction_frame`` is the earliest frame
    index whose prefix has no explanation; the layers then cover the maximal
    explainable prefix ``[0, contradiction_frame)``.
    """

    def __init__(
        self,
        model: MachineModel,
        observations: tuple[tuple[int, ...], ...],
        layers: tuple[tuple[int, ...], ...],
        transitions: tuple[tuple[frozenset, ...], ...],
        explaining_counts: tuple[int, ...],
        status: str,
        contradiction_frame: int | None,
    ) -> None:
        self.model = model
        self.observations = observations
        self.layers = layers
        self.transitions = transitions
        self.explaining_counts = explaining_counts
        self.status = status
        self.contradiction_frame = contradiction_frame

    @classmethod
    def build(cls, model: MachineModel, observations: Sequence[Sequence[int]]) -> "HistoryDag":
        obs = tuple(tuple(o) for o in observations)
        if not obs:
            raise MatrixError("a segment must contain at least one frame", code="frame_count")
        for t, o in enumerate(obs):
            if len(o) != model.rows:
                raise MatrixError(
                    f"frame {t} has {len(o)} row mask(s), expected {model.rows}",
                    code="frame_bytes_misaligned",
                    frame_index=t,
                )
        explain = [explaining_states(model, o) for o in obs]

        # Forward pass: states reachable from frame 0 under the one-contact
        # transition rule.  The first frame with an empty set is the earliest
        # contradiction — frames [0, t) are still jointly explainable.
        forward: list[set[int]] = []
        contradiction: int | None = None
        for t, states in enumerate(explain):
            if t == 0:
                cur = set(states)
            else:
                prev = forward[-1]
                cur = {s for s in states if any(_adjacent(p, s) for p in prev)}
            if not cur:
                contradiction = t
                break
            forward.append(cur)
        end = len(obs) if contradiction is None else contradiction

        # Backward trim: keep only states that also reach the last feasible
        # frame, so every remaining state lies on a complete (prefix) history.
        layers: list[tuple[int, ...]] = []
        transitions: list[tuple[frozenset, ...]] = []
        if end > 0:
            keep = [set() for _ in range(end)]
            keep[end - 1] = forward[end - 1]
            for t in range(end - 2, -1, -1):
                keep[t] = {
                    s for s in forward[t] if any(_adjacent(s, n) for n in keep[t + 1])
                }
            layers = [tuple(sorted(k)) for k in keep]
            for t in range(end - 1):
                index = {s: i for i, s in enumerate(layers[t + 1])}
                transitions.append(
                    tuple(
                        frozenset(
                            index[n] for n in keep[t + 1] if _adjacent(s, n)
                        )
                        for s in layers[t]
                    )
                )
        return cls(
            model=model,
            observations=obs,
            layers=tuple(layers),
            transitions=tuple(transitions),
            explaining_counts=tuple(len(e) for e in explain),
            status="ok" if contradiction is None else "contradiction",
            contradiction_frame=contradiction,
        )

    # -- introspection ------------------------------------------------------
    @property
    def frame_count(self) -> int:
        return len(self.observations)

    @property
    def analyzed_frames(self) -> int:
        return len(self.layers)

    def signature(self) -> str:
        """Content hash of model + layers + transitions.

        The gesture table is not an input, so two queries against different
        tables always run on the identical trajectory set.
        """
        blob = canonical_json(
            {
                "model": self.model.as_dict(),
                "layers": [list(layer) for layer in self.layers],
                "transitions": [
                    [sorted(edge) for edge in layer] for layer in self.transitions
                ],
            }
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def iter_histories(self, limit: int | None = None) -> Iterator[tuple[int, ...]]:
        """Yield every complete history (tuple of state bitmasks), in order."""
        if self.status != "ok" or not self.layers:
            return
        produced = 0
        stack = [(i,) for i in range(len(self.layers[0]) - 1, -1, -1)]
        while stack:
            path = stack.pop()
            t = len(path) - 1
            if t == len(self.layers) - 1:
                yield tuple(self.layers[f][i] for f, i in enumerate(path))
                produced += 1
                if limit is not None and produced >= limit:
                    return
                continue
            for nxt in sorted(self.transitions[t][path[-1]], reverse=True):
                stack.append(path + (nxt,))

    def as_dict(self) -> dict:
        model = self.model
        return {
            "model": model.as_dict(),
            "status": self.status,
            "frame_count": self.frame_count,
            "analyzed_frames": self.analyzed_frames,
            "contradiction_frame": self.contradiction_frame,
            "signature": self.signature(),
            "frames": [
                {
                    "index": t,
                    "row_masks": list(obs),
                    "contacts": observation_contacts(model, obs),
                    "explaining_states": self.explaining_counts[t],
                    "on_path_states": (
                        len(self.layers[t]) if t < len(self.layers) else 0
                    ),
                }
                for t, obs in enumerate(self.observations)
            ],
            "layers": [
                [
                    {"index": i, "contacts": [list(c) for c in model.contacts_of(s)]}
                    for i, s in enumerate(layer)
                ]
                for layer in self.layers
            ],
            "transitions": [
                {
                    "from_frame": t,
                    "edges": [
                        [i, j] for i, outs in enumerate(layer) for j in sorted(outs)
                    ],
                }
                for t, layer in enumerate(self.transitions)
            ],
        }


# ---------------------------------------------------------------------------
# Gesture table


@dataclass(frozen=True)
class GestureEntry:
    """One gesture record: must-press contacts, action, tie-break precedence."""

    required: frozenset
    action: str
    order: int
    required_mask: int

    def as_dict(self) -> dict:
        return {
            "required": [list(c) for c in sorted(self.required)],
            "action": self.action,
            "order": self.order,
        }


class GestureTable:
    """An ordered gesture table, swappable per query.

    Entries are matched by ``required ⊆ state``; the satisfied entry with the
    most must-press contacts wins, ties go to the smaller ``order`` (unique
    within one must-press length), and no hit means an empty frame action.
    """

    def __init__(self, entries: Sequence[GestureEntry]) -> None:
        self.entries = tuple(entries)

    @classmethod
    def from_json(cls, obj: Any, model: MachineModel) -> "GestureTable":
        if isinstance(obj, list):
            obj = {"gestures": obj}
        if not isinstance(obj, dict):
            raise MatrixError("gesture table must be a JSON object", code="invalid_gesture_table")
        unknown = sorted(set(obj) - {"gestures", "description"})
        if unknown:
            raise MatrixError(
                f"gesture table: unknown key(s) {', '.join(unknown)}",
                code="invalid_gesture_table",
            )
        raw_entries = obj.get("gestures")
        if not isinstance(raw_entries, list):
            raise MatrixError(
                "gesture table must carry a 'gestures' array",
                code="invalid_gesture_table",
            )
        entries: list[GestureEntry] = []
        seen_order: dict[int, set[int]] = {}
        for pos, raw in enumerate(raw_entries):
            where = f"gestures[{pos}]"
            if not isinstance(raw, dict):
                raise MatrixError(f"{where} must be an object", code="invalid_gesture_table")
            unknown = sorted(set(raw) - {"required", "action", "order", "description"})
            if unknown:
                raise MatrixError(
                    f"{where}: unknown key(s) {', '.join(unknown)}",
                    code="invalid_gesture_table",
                )
            for key in ("required", "action", "order"):
                if key not in raw:
                    raise MatrixError(
                        f"{where}: missing required key {key!r}",
                        code="invalid_gesture_table",
                    )
            required_raw = raw["required"]
            if (
                not isinstance(required_raw, list)
                or not required_raw
                or any(
                    not isinstance(c, list)
                    or len(c) != 2
                    or any(isinstance(v, bool) or not isinstance(v, int) for v in c)
                    for c in required_raw
                )
            ):
                raise MatrixError(
                    f"{where}.required must be a non-empty array of [row, col] pairs",
                    code="invalid_gesture_table",
                )
            required = set()
            for r, c in required_raw:
                if not (0 <= r < model.rows and 0 <= c < model.cols):
                    raise MatrixError(
                        f"{where}.required contact [{r}, {c}] lies outside the "
                        f"{model.rows}x{model.cols} grid",
                        code="invalid_gesture_table",
                    )
                required.add((r, c))
            if len(required) != len(required_raw):
                raise MatrixError(
                    f"{where}.required lists a contact twice",
                    code="invalid_gesture_table",
                )
            action = raw["action"]
            if not isinstance(action, str) or not action:
                raise MatrixError(
                    f"{where}.action must be a non-empty string",
                    code="invalid_gesture_table",
                )
            order = raw["order"]
            if isinstance(order, bool) or not isinstance(order, int):
                raise MatrixError(
                    f"{where}.order must be an integer",
                    code="invalid_gesture_table",
                )
            length_orders = seen_order.setdefault(len(required), set())
            if order in length_orders:
                raise MatrixError(
                    f"{where}.order {order} is already used for another "
                    f"{len(required)}-contact record; the same-length "
                    f"precedence must be unique",
                    code="invalid_gesture_table",
                )
            length_orders.add(order)
            entries.append(
                GestureEntry(
                    required=frozenset(required),
                    action=action,
                    order=order,
                    required_mask=model.mask_of(required),
                )
            )
        # Most must-press contacts first; same length by ascending precedence.
        entries.sort(key=lambda e: (-len(e.required), e.order))
        return cls(entries)

    def action_for(self, state: int) -> str | None:
        """The frame action of one true contact set (None = empty)."""
        for entry in self.entries:
            if entry.required_mask & ~state == 0:
                return entry.action
        return None

    def as_dict(self) -> dict:
        return {"gestures": [e.as_dict() for e in self.entries]}


def trigger_stream(actions: Sequence[str | None]) -> list[str]:
    """The truly triggered actions of one frame-action sequence.

    An action fires only when it differs from the previous frame's action and
    is itself non-empty; the stream before the first frame is empty.
    """
    out: list[str] = []
    prev: str | None = None
    for action in actions:
        if action is not None and action != prev:
            out.append(action)
        prev = action
    return out


# ---------------------------------------------------------------------------
# Action-string query over the DAG


class _Matcher:
    """KMP automaton: has ``pattern`` appeared as a contiguous segment yet?

    The state is the longest matched prefix length; ``len(pattern)`` is the
    absorbing "found" state.
    """

    def __init__(self, pattern: Sequence[str]) -> None:
        self.pattern = list(pattern)
        self.m = len(self.pattern)
        pi = [0] * self.m
        k = 0
        for i in range(1, self.m):
            while k and self.pattern[i] != self.pattern[k]:
                k = pi[k - 1]
            if self.pattern[i] == self.pattern[k]:
                k += 1
            pi[i] = k
        self.pi = pi

    def step(self, k: int, event: str) -> int:
        if k == self.m:
            return self.m
        while k and self.pattern[k] != event:
            k = self.pi[k - 1]
        if self.pattern[k] == event:
            k += 1
        return k


@dataclass(frozen=True)
class QueryResult:
    """The verdict of one action-string query, with supporting histories."""

    status: str  # "ok" or "contradiction"
    verdict: str | None  # "all" | "some" | "none" (None when contradictory)
    query: tuple
    dag_signature: str
    frame_count: int
    contradiction_frame: int | None
    witness: dict | None  # a feasible history whose trigger stream matches
    counter_witness: dict | None  # a feasible history whose stream does not

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "verdict": self.verdict,
            "query": list(self.query),
            "dag_signature": self.dag_signature,
            "frame_count": self.frame_count,
            "contradiction_frame": self.contradiction_frame,
            "witness": self.witness,
            "counter_witness": self.counter_witness,
        }


def _first_path_indices(dag: HistoryDag) -> list[int]:
    """One source-to-sink path of state indices through the trimmed DAG."""
    indices = [0]
    for t in range(len(dag.layers) - 1):
        indices.append(min(dag.transitions[t][indices[-1]]))
    return indices


def _history_report(
    dag: HistoryDag,
    table: GestureTable,
    state_indices: Sequence[int],
    matcher: "_Matcher | None" = None,
    query_len: int = 0,
) -> dict:
    states = [dag.layers[t][i] for t, i in enumerate(state_indices)]
    actions = [table.action_for(s) for s in states]
    triggers = trigger_stream(actions)
    report = {
        "states": [
            [list(c) for c in dag.model.contacts_of(s)] for s in states
        ],
        "actions": actions,
        "triggers": triggers,
    }
    if matcher is not None and query_len:
        k = 0
        for pos, event in enumerate(triggers):
            k = matcher.step(k, event)
            if k == query_len:
                report["match_span"] = [pos - query_len + 1, pos + 1]
                break
    return report


def query_actions(
    dag: HistoryDag,
    table: GestureTable,
    actions: Sequence[str],
) -> QueryResult:
    """Judge an ordered action string against every feasible history.

    The answer is ``"all"`` when the string appears as a contiguous segment
    of the trigger stream of every feasible contact history, ``"some"`` when
    only part of the histories show it, and ``"none"`` when no history does.
    A witness history (and, when one exists, a counter-witness) is attached
    so the conclusion can be re-checked contact by contact.
    """
    query = tuple(actions)
    if not query or any(not isinstance(a, str) or not a for a in query):
        raise MatrixError(
            "the queried action string must be a non-empty sequence of "
            "non-empty action names",
            code="invalid_query",
        )
    if dag.status != "ok":
        return QueryResult(
            status=dag.status,
            verdict=None,
            query=query,
            dag_signature=dag.signature(),
            frame_count=dag.frame_count,
            contradiction_frame=dag.contradiction_frame,
            witness=None,
            counter_witness=None,
        )

    depth = len(dag.layers)
    if len(query) > depth:
        # At most one trigger fires per frame, so the string cannot fit.
        return QueryResult(
            status="ok",
            verdict=VERDICT_NONE,
            query=query,
            dag_signature=dag.signature(),
            frame_count=dag.frame_count,
            contradiction_frame=None,
            witness=None,
            counter_witness=_history_report(dag, table, _first_path_indices(dag)),
        )

    matcher = _Matcher(query)
    m = len(query)
    frame_actions = [
        tuple(table.action_for(s) for s in layer) for layer in dag.layers
    ]

    # Reachable (state index, matcher state) pairs per frame, each keeping
    # one predecessor for witness reconstruction.  Merging on the matcher
    # state is sound: the future depends only on the current contact set and
    # the matched prefix length.
    frontier: dict[tuple[int, int], tuple[int, int] | None] = {}
    for i in range(len(dag.layers[0])):
        a = frame_actions[0][i]
        k = matcher.step(0, a) if a is not None else 0
        frontier.setdefault((i, k), None)
    parents = [frontier]
    for t in range(1, depth):
        nxt: dict[tuple[int, int], tuple[int, int]] = {}
        for (i, k), _ in parents[-1].items():
            a_prev = frame_actions[t - 1][i]
            for j in dag.transitions[t - 1][i]:
                a_next = frame_actions[t][j]
                if a_next is not None and a_next != a_prev:
                    k2 = matcher.step(k, a_next)
                else:
                    k2 = k
                if (j, k2) not in nxt:
                    nxt[(j, k2)] = (i, k)
        parents.append(nxt)

    def backtrack(key: tuple[int, int]) -> list[int]:
        indices = [0] * depth
        cur = key
        for t in range(depth - 1, -1, -1):
            indices[t] = cur[0]
            cur = parents[t][cur]
        return indices

    last = parents[-1]
    found = sorted(key for key in last if key[1] == m)
    missing = sorted(key for key in last if key[1] < m)
    if found and not missing:
        verdict = VERDICT_ALL
    elif found:
        verdict = VERDICT_SOME
    else:
        verdict = VERDICT_NONE
    witness = (
        _history_report(dag, table, backtrack(found[0]), matcher, m) if found else None
    )
    counter = (
        _history_report(dag, table, backtrack(missing[0])) if missing else None
    )
    return QueryResult(
        status="ok",
        verdict=verdict,
        query=query,
        dag_signature=dag.signature(),
        frame_count=dag.frame_count,
        contradiction_frame=None,
        witness=witness,
        counter_witness=counter,
    )


# ---------------------------------------------------------------------------
# Scan segments and the repository reader


@dataclass(frozen=True)
class ScanSegment:
    """One imported scan segment: machine model plus the observed frames."""

    station_id: int | None
    model: MachineModel
    frames: tuple[tuple[int, ...], ...]
    import_id: int | None = None
    record_ordinal: int | None = None

    @classmethod
    def from_values(
        cls,
        *,
        rows: int,
        cols: int,
        max_keys: int,
        installed: bytes,
        frames: bytes,
        station_id: int | None = None,
    ) -> "ScanSegment":
        model = MachineModel(rows, cols, max_keys, parse_installed(rows, cols, installed))
        return cls(
            station_id=station_id,
            model=model,
            frames=tuple(parse_frames(model, frames)),
        )

    def build_dag(self) -> HistoryDag:
        return HistoryDag.build(self.model, self.frames)


def load_segment(repo, import_id: int, record_ordinal: int) -> ScanSegment:
    """Rebuild one scan segment from the repository's committed field facts.

    Only the SQLite fact relation is read; the original byte stream is not
    needed, so archive readers are independent of the ingest process.
    """
    rows = repo.conn.execute(
        "SELECT f.path, f.present, f.int_value, f.bytes_value "
        "FROM fields f JOIN records r ON r.record_id = f.record_id "
        "WHERE f.import_id = ? AND r.ordinal = ?",
        (import_id, record_ordinal),
    ).fetchall()
    if not rows:
        raise MatrixError(
            f"import {import_id} has no record with ordinal {record_ordinal}",
            code="unknown_record",
        )
    values: dict[str, Any] = {}
    for row in rows:
        if not row["present"]:
            continue
        if row["int_value"] is not None:
            values[row["path"]] = row["int_value"]
        elif row["bytes_value"] is not None:
            values[row["path"]] = bytes(row["bytes_value"])
    required = ("model.rows", "model.cols", "model.max_keys", "model.installed", "frames")
    missing = [p for p in required if p not in values]
    if missing:
        raise MatrixError(
            f"record {record_ordinal} of import {import_id} is not a complete "
            f"matrix scan segment; missing field(s): {', '.join(missing)}",
            code="segment_incomplete",
        )
    segment = ScanSegment.from_values(
        rows=values["model.rows"],
        cols=values["model.cols"],
        max_keys=values["model.max_keys"],
        installed=values["model.installed"],
        frames=values["frames"],
        station_id=values.get("station_id"),
    )
    return ScanSegment(
        station_id=segment.station_id,
        model=segment.model,
        frames=segment.frames,
        import_id=import_id,
        record_ordinal=record_ordinal,
    )
