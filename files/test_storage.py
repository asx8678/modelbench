import os
import sqlite3

import storage


def _db_path(tmp_path_factory):
    return str(tmp_path_factory.mktemp("storage") / "test.db")


def test_telemetry_table_created(tmp_path_factory):
    path = _db_path(tmp_path_factory)
    con = storage.connect(path)
    con.execute("SELECT 1 FROM telemetry").fetchone()
    con.close()


def test_telemetry_round_trip(tmp_path_factory):
    path = _db_path(tmp_path_factory)
    con = storage.connect(path)
    storage.save_telemetry(
        con,
        run_id="r1",
        item_id="i1",
        sample_idx=0,
        capabilities=["stream", "native_reasoning"],
        reasoning_token_source="native_usage",
        prompt_tokens=10,
        completion_tokens=20,
        reasoning_tokens=5,
        reasoning_density_proxy=0.25,
        ttft_ms=100,
        first_reasoning_ms=200,
        reasoning_wall_ms=300,
        answer_wall_ms=400,
        unobservable_fields={"reasoning_wall_ms": "no_stream"},
    )
    row = storage.load_telemetry(con, "r1", "i1", 0)
    assert row["run_id"] == "r1"
    assert row["item_id"] == "i1"
    assert row["sample_idx"] == 0
    assert row["capabilities"] == ["stream", "native_reasoning"]
    assert row["reasoning_token_source"] == "native_usage"
    assert row["prompt_tokens"] == 10
    assert row["completion_tokens"] == 20
    assert row["reasoning_tokens"] == 5
    assert row["reasoning_density_proxy"] == 0.25
    assert row["ttft_ms"] == 100
    assert row["first_reasoning_ms"] == 200
    assert row["reasoning_wall_ms"] == 300
    assert row["answer_wall_ms"] == 400
    assert row["unobservable_fields"] == {"reasoning_wall_ms": "no_stream"}
    con.close()


def test_idempotent_migration(tmp_path_factory):
    path = _db_path(tmp_path_factory)
    con1 = storage.connect(path)
    con1.execute("SELECT 1 FROM telemetry").fetchone()
    con1.close()
    con2 = storage.connect(path)
    con2.execute("SELECT 1 FROM telemetry").fetchone()
    con2.close()
