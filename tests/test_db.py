import pandas as pd
import pytest

from causal_fairness_sdg.experiments import db


@pytest.fixture
def conn():
    connection = db.get_connection(":memory:")
    yield connection
    connection.close()


def test_insert_run_and_query(conn):
    run_id = db.insert_run(
        conn,
        dataset="adult",
        sdg_method="mst",
        fairness_mechanism="cf",
        protected_attrs=["sex", "race"],
        outcome_attr="income",
        seed=1,
        admissible_attrs=["education"],
        epsilon=1.0,
        delta=1e-9,
        synth_size=1000,
    )
    assert run_id == 1

    runs = pd.read_sql_query("SELECT * FROM runs", conn)
    assert len(runs) == 1
    assert runs.loc[0, "status"] == "pending"
    assert runs.loc[0, "dataset"] == "adult"


def test_log_metric_round_trip(conn):
    run_id = db.insert_run(
        conn, dataset="adult", sdg_method="mst", fairness_mechanism="none",
        protected_attrs=["sex"], outcome_attr="income", seed=0,
    )
    db.log_metric(conn, run_id, "tvd_1way", 0.05)
    db.log_metrics(conn, run_id, {"dp_gap__sex": 0.12, "downstream_accuracy_mlp": 0.81})

    metrics = pd.read_sql_query("SELECT * FROM metrics WHERE run_id = ?", conn, params=(run_id,))
    assert len(metrics) == 3
    assert set(metrics["metric_name"]) == {"tvd_1way", "dp_gap__sex", "downstream_accuracy_mlp"}


def test_log_metric_upsert_overwrites_existing_value(conn):
    run_id = db.insert_run(
        conn, dataset="adult", sdg_method="mst", fairness_mechanism="none",
        protected_attrs=["sex"], outcome_attr="income", seed=0,
    )
    db.log_metric(conn, run_id, "tvd_1way", 0.05)
    db.log_metric(conn, run_id, "tvd_1way", 0.09)

    metrics = pd.read_sql_query("SELECT * FROM metrics WHERE run_id = ?", conn, params=(run_id,))
    assert len(metrics) == 1
    assert metrics.loc[0, "metric_value"] == pytest.approx(0.09)


def test_update_run_status(conn):
    run_id = db.insert_run(
        conn, dataset="adult", sdg_method="mst", fairness_mechanism="none",
        protected_attrs=["sex"], outcome_attr="income", seed=0,
    )
    db.update_run_status(conn, run_id, status="done", duration_seconds=12.3)

    runs = pd.read_sql_query("SELECT * FROM runs WHERE run_id = ?", conn, params=(run_id,))
    assert runs.loc[0, "status"] == "done"
    assert runs.loc[0, "duration_seconds"] == pytest.approx(12.3)


def test_log_edges(conn):
    run_id = db.insert_run(
        conn, dataset="adult", sdg_method="mst", fairness_mechanism="none",
        protected_attrs=["sex"], outcome_attr="income", seed=0,
    )
    db.log_edges(conn, run_id, [("A", "B"), ("B", "C")], weights={("A", "B"): 0.5})

    edges = pd.read_sql_query("SELECT * FROM graph_edges WHERE run_id = ?", conn, params=(run_id,))
    assert len(edges) == 2
    assert edges.loc[edges["node_a"] == "A", "weight"].iloc[0] == pytest.approx(0.5)


def test_query_runs_joins_metrics_wide_and_filters(conn):
    run1 = db.insert_run(
        conn, dataset="adult", sdg_method="mst", fairness_mechanism="none",
        protected_attrs=["sex"], outcome_attr="income", seed=0,
    )
    run2 = db.insert_run(
        conn, dataset="compas", sdg_method="mst", fairness_mechanism="cf",
        protected_attrs=["sex"], outcome_attr="two_year_recid", seed=0,
    )
    db.log_metric(conn, run1, "tvd_1way", 0.05)
    db.log_metric(conn, run2, "tvd_1way", 0.11)

    all_runs = db.query_runs(conn)
    assert len(all_runs) == 2
    assert "tvd_1way" in all_runs.columns

    adult_only = db.query_runs(conn, dataset="adult")
    assert len(adult_only) == 1
    assert adult_only.iloc[0]["tvd_1way"] == pytest.approx(0.05)


def test_query_runs_empty_db_returns_empty_dataframe(conn):
    result = db.query_runs(conn)
    assert result.empty
