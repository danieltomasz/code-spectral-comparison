"""Local DuckDB result store keyed by a settings hash.

One ``.duckdb`` file holds a ``runs`` registry plus result tables (``peaks``,
``quality``, ``features``, ...). Every result row carries a ``run_id``; the
``runs`` table maps that id to the exact settings that produced it, together
with a short ``params_hash`` of those settings.

The point is provenance, not just storage: a consumer loads by ``params_hash``
and asserts the result is present, so cached results can never silently drift
from the live :mod:`pesco.config` settings — if the hash changed, the load
fails and you re-run the producer.

Usage
-----
>>> from pesco import config, store
>>> h = store.params_hash(config.SPECPARAM_SETTINGS, config.FREQ_RANGE,
...                       config.SELECTED_MODE)
>>> con = store.open_db("data/interim/analysis.duckdb")
>>> store.write_run(con, h, config.SPECPARAM_SETTINGS, config.FREQ_RANGE,
...                 config.SELECTED_MODE, {"peaks": peaks_df, "quality": q_df})
>>> peaks = store.load(con, "peaks", h)        # asserts rows exist for this hash
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Mapping

import duckdb
import pandas as pd


def params_hash(
    settings: Mapping,
    freq_range,
    selected_mode: Mapping,
    extra: Mapping | None = None,
) -> str:
    """Short stable hash of the analysis parameters.

    Any change to the specparam settings, fit range, mode choice, or ``extra``
    keys yields a different hash, so results computed under different parameters
    are never confused.
    """
    payload = {
        "settings": dict(settings),
        "freq_range": list(freq_range),
        "selected_mode": dict(selected_mode),
        "extra": dict(extra) if extra else {},
    }
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def open_db(path: str | Path) -> duckdb.DuckDBPyConnection:
    """Open (creating if needed) the store and ensure the ``runs`` registry."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(path))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id          VARCHAR,
            params_hash     VARCHAR,
            label           VARCHAR,
            freq_lo         DOUBLE,
            freq_hi         DOUBLE,
            peak_threshold  DOUBLE,
            min_peak_height DOUBLE,
            peak_width_lo   DOUBLE,
            peak_width_hi   DOUBLE,
            max_n_peaks     INTEGER,
            settings_json   VARCHAR,
            freq_range      VARCHAR,
            selected_mode   VARCHAR,
            git_sha         VARCHAR,
            created_at      TIMESTAMP
        )
        """
    )
    # Defensive migration for stores created before these columns existed.
    for col, typ in [
        ("label", "VARCHAR"), ("freq_lo", "DOUBLE"), ("freq_hi", "DOUBLE"),
        ("peak_threshold", "DOUBLE"), ("min_peak_height", "DOUBLE"),
        ("peak_width_lo", "DOUBLE"), ("peak_width_hi", "DOUBLE"),
        ("max_n_peaks", "INTEGER"),
    ]:
        con.execute(f"ALTER TABLE runs ADD COLUMN IF NOT EXISTS {col} {typ}")
    return con


def write_run(
    con: duckdb.DuckDBPyConnection,
    params_hash: str,
    settings: Mapping,
    freq_range,
    selected_mode: Mapping,
    tables: Mapping[str, pd.DataFrame],
    label: str | None = None,
) -> str:
    """Write a run and its result tables, replacing any prior data for this hash.

    ``run_id`` equals ``params_hash`` (one run per parameter set), so several runs
    with different settings / frequency ranges coexist in one store. ``label`` is
    a human name for the run (not part of its identity); re-running overwrites
    just that run's rows.
    """
    run_id = params_hash
    s = dict(settings)
    pw = list(s.get("peak_width_limits", [None, None])) + [None, None]
    con.execute("DELETE FROM runs WHERE run_id = ?", [run_id])
    con.execute(
        "INSERT INTO runs (run_id, params_hash, label, freq_lo, freq_hi, "
        "peak_threshold, min_peak_height, peak_width_lo, peak_width_hi, "
        "max_n_peaks, settings_json, freq_range, selected_mode, git_sha, "
        "created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        [
            run_id,
            params_hash,
            label,
            float(freq_range[0]),
            float(freq_range[1]),
            s.get("peak_threshold"),
            s.get("min_peak_height"),
            pw[0],
            pw[1],
            s.get("max_n_peaks"),
            json.dumps(s, default=str),
            json.dumps(list(freq_range)),
            json.dumps(dict(selected_mode)),
            _git_sha(),
            _dt.datetime.now(),
        ],
    )
    for name, df in tables.items():
        tagged = df.copy()
        tagged.insert(0, "run_id", run_id)
        con.register("_tmp_write", tagged)
        con.execute(
            f"CREATE TABLE IF NOT EXISTS {name} AS SELECT * FROM _tmp_write WHERE 1=0"
        )
        con.execute(f"DELETE FROM {name} WHERE run_id = ?", [run_id])
        con.execute(f"INSERT INTO {name} SELECT * FROM _tmp_write")
        con.unregister("_tmp_write")
    return run_id


def load(
    con: duckdb.DuckDBPyConnection,
    table: str,
    params_hash: str,
    required: bool = True,
) -> pd.DataFrame:
    """Load a result table for one parameter set; assert it exists by default.

    Raising on an empty result is the sync guarantee: if the live settings hash
    is not in the store, the consumer stops instead of silently using stale data.
    """
    exists = con.execute(
        "SELECT 1 FROM information_schema.tables WHERE table_name = ?", [table]
    ).fetchone()
    df = (
        con.execute(f"SELECT * FROM {table} WHERE run_id = ?", [params_hash]).df()
        if exists
        else pd.DataFrame()
    )
    if required and df.empty:
        raise ValueError(
            f"No rows in {table!r} for params_hash {params_hash!r}. "
            "Run the feature producer with the current pesco.config settings."
        )
    return df.drop(columns=["run_id"], errors="ignore")


def list_runs(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """The run registry, newest first."""
    return con.execute("SELECT * FROM runs ORDER BY created_at DESC").df()


def find_runs(
    con: duckdb.DuckDBPyConnection,
    since: str | None = None,
    until: str | None = None,
    **param_filters,
) -> pd.DataFrame:
    """Query the run registry by date and/or spectral parameter, newest first.

    ``since`` / ``until`` are dates/timestamps on ``created_at``. Keyword filters
    match the typed ``runs`` columns exactly, e.g.::

        find_runs(con, peak_threshold=2.0, freq_hi=40)
        find_runs(con, since="2026-06-01", label="canonical")

    Returns the matching run rows (use ``params_hash`` to load result tables).
    """
    where, params = [], []
    if since is not None:
        where.append("created_at >= ?")
        params.append(since)
    if until is not None:
        where.append("created_at <= ?")
        params.append(until)
    for col, val in param_filters.items():
        where.append(f"{col} = ?")
        params.append(val)
    clause = (" WHERE " + " AND ".join(where)) if where else ""
    return con.execute(
        f"SELECT * FROM runs{clause} ORDER BY created_at DESC", params
    ).df()


def find_run(
    con: duckdb.DuckDBPyConnection,
    label: str | None = None,
    params_hash: str | None = None,
) -> str:
    """Resolve a run to its ``params_hash`` by label (newest match) or hash.

    Lets consumers load a run by a readable name instead of recomputing a hash,
    which is convenient when sweeping several settings / frequency ranges.
    """
    if params_hash is not None:
        hit = con.execute(
            "SELECT params_hash FROM runs WHERE params_hash = ?", [params_hash]
        ).fetchone()
    elif label is not None:
        hit = con.execute(
            "SELECT params_hash FROM runs WHERE label = ? ORDER BY created_at DESC "
            "LIMIT 1",
            [label],
        ).fetchone()
    else:
        raise ValueError("Pass either label= or params_hash=.")
    if hit is None:
        raise ValueError(f"No run for label={label!r} / params_hash={params_hash!r}.")
    return hit[0]
