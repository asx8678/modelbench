"""
SQLite storage. Raw responses are kept verbatim so you can re-grade or re-analyze
without re-running any model. SQLite is stdlib and the file is portable.
"""

import sqlite3
import json
import time
from typing import List
from generators import Problem

SCHEMA = """
CREATE TABLE IF NOT EXISTS dataset (
    item_id TEXT PRIMARY KEY,
    family TEXT, difficulty INTEGER,
    structure_seed INTEGER, surface_seed INTEGER,
    has_distractor INTEGER, probe TEXT, grp TEXT,
    answer_type TEXT, gold TEXT, choices TEXT, prompt TEXT, turns TEXT,
    subgold TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    model TEXT, base_url TEXT, params TEXT, created_at REAL
);
CREATE TABLE IF NOT EXISTS responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT, item_id TEXT, sample_idx INTEGER,
    raw TEXT, parsed TEXT, correct INTEGER, confidence INTEGER,
    latency_ms INTEGER, prompt_tokens INTEGER, completion_tokens INTEGER,
    metadata TEXT,
    UNIQUE(run_id, item_id, sample_idx)
);
CREATE INDEX IF NOT EXISTS idx_resp ON responses(run_id, item_id);
CREATE TABLE IF NOT EXISTS telemetry (
    run_id TEXT, item_id TEXT, sample_idx INTEGER,
    capabilities TEXT,
    reasoning_token_source TEXT,
    prompt_tokens INTEGER, completion_tokens INTEGER, reasoning_tokens INTEGER,
    reasoning_density_proxy REAL,
    ttft_ms INTEGER, first_reasoning_ms INTEGER, reasoning_wall_ms INTEGER, answer_wall_ms INTEGER,
    token_entropy_mean REAL, token_entropy_max REAL,
    friction_transitions INTEGER, logprob_divergence_spikes INTEGER,
    unobservable_fields TEXT,
    UNIQUE(run_id, item_id, sample_idx)
);
"""

ERROR_MARKER = "__ERROR__"


def connect(path: str) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.executescript(SCHEMA)
    return con


def save_dataset(con, items: List[Problem]):
    con.executemany(
        """INSERT OR REPLACE INTO dataset
           (item_id, family, difficulty, structure_seed, surface_seed,
            has_distractor, probe, grp, answer_type, gold, choices, prompt, turns, subgold)
           VALUES (:item_id,:family,:difficulty,:structure_seed,:surface_seed,
                   :has_distractor,:probe,:grp,:answer_type,:gold,:choices,:prompt,:turns,:subgold)""",
        [p.row() for p in items],
    )
    con.commit()

def load_dataset(con):
    return [dict(r) for r in con.execute("SELECT * FROM dataset")]


def new_run(con, run_id, model, base_url, params: dict):
    con.execute("INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?)",
                (run_id, model, base_url, json.dumps(params), time.time()))
    con.commit()


def save_response(con, run_id, item_id, sample_idx, raw, parsed, correct,
                  confidence, latency_ms, ptok, ctok, metadata=None):
    # INSERT OR REPLACE keyed on (run_id,item_id,sample_idx): re-running a run_id
    # overwrites rather than duplicating, so metrics never double-count.
    con.execute(
        """INSERT OR REPLACE INTO responses
           (run_id,item_id,sample_idx,raw,parsed,correct,confidence,latency_ms,prompt_tokens,completion_tokens,metadata)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, item_id, sample_idx, raw, parsed,
         int(correct) if correct is not None else None,
         confidence, latency_ms, ptok, ctok,
         json.dumps(metadata) if metadata is not None else None),
    )


def list_runs(con):
    return [dict(r) for r in con.execute("SELECT * FROM runs ORDER BY created_at")]


def done_items(con, run_id):
    """item_ids already scored OK for this run (for resume).

    Items whose only stored response is an error are NOT counted as done, so a
    transient failure gets retried on the next --resume instead of being frozen in
    as a wrong answer."""
    return {r["item_id"] for r in con.execute(
        "SELECT item_id FROM responses WHERE run_id=? GROUP BY item_id HAVING SUM(raw=?)=0",
        (run_id, ERROR_MARKER))}

def save_telemetry(con, run_id, item_id, sample_idx, *, capabilities=None,
                   reasoning_token_source=None, prompt_tokens=None,
                   completion_tokens=None, reasoning_tokens=None,
                   reasoning_density_proxy=None, ttft_ms=None,
                   first_reasoning_ms=None, reasoning_wall_ms=None,
                   answer_wall_ms=None, token_entropy_mean=None,
                   token_entropy_max=None, friction_transitions=None,
                   logprob_divergence_spikes=None, unobservable_fields=None):
    con.execute(
        """INSERT OR REPLACE INTO telemetry
           (run_id, item_id, sample_idx, capabilities, reasoning_token_source,
            prompt_tokens, completion_tokens, reasoning_tokens,
            reasoning_density_proxy, ttft_ms, first_reasoning_ms,
            reasoning_wall_ms, answer_wall_ms,
            token_entropy_mean, token_entropy_max,
            friction_transitions, logprob_divergence_spikes, unobservable_fields)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (run_id, item_id, sample_idx,
         json.dumps(capabilities) if capabilities is not None else None,
         reasoning_token_source,
         prompt_tokens, completion_tokens, reasoning_tokens,
         reasoning_density_proxy, ttft_ms, first_reasoning_ms,
         reasoning_wall_ms, answer_wall_ms,
         token_entropy_mean, token_entropy_max,
         friction_transitions, logprob_divergence_spikes,
         json.dumps(unobservable_fields) if unobservable_fields is not None else None),
    )


def load_telemetry(con, run_id, item_id, sample_idx):
    row = con.execute(
        "SELECT * FROM telemetry WHERE run_id=? AND item_id=? AND sample_idx=?",
        (run_id, item_id, sample_idx),
    ).fetchone()
    if row is None:
        return None
    d = dict(row)
    for k in ("capabilities", "unobservable_fields"):
        if d.get(k) is not None:
            d[k] = json.loads(d[k])
    return d
