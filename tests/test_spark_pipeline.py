"""
Offline/Online Rolling-Feature Parity
=====================================

The rolling-window block is computed twice in this codebase: in pandas at
request time (:func:`src.serving.feature_service.compute_rolling_features`) and
in Spark for the nightly materialisation
(:func:`src.feature_store.spark_pipeline.build_rolling_features`).

Two implementations of one definition is the standard way training–serving skew
gets in.  The fix is not to trust that they match — it is to run both over the
same history and compare every field, which is what this module does.

These tests are skipped when PySpark is not installed, so the rest of the suite
stays runnable without a JVM.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pyspark = pytest.importorskip("pyspark", reason="PySpark not installed")

from src.feature_store.spark_pipeline import (  # noqa: E402
    build_rolling_features,
    get_spark,
    latest_per_user,
)
from src.serving.feature_service import (
    ROLLING_FIELDS,
    compute_rolling_features,
)  # noqa: E402


@pytest.fixture(scope="module")
def spark():
    session = get_spark(app_name="profit-tests")
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def _make_history(n_users: int = 4, n_days: int = 60, seed: int = 11) -> pd.DataFrame:
    """
    Deterministic multi-user history.

    Loads are drawn from a set that straddles both thresholds the run-length
    features depend on — zero (inactive), below 180 (active but not hard), and
    at or above 180 (hard) — so streak, days-since-training and
    consecutive-hard-days all actually vary rather than sitting at a constant.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for user in range(n_users):
        for day in range(n_days):
            rows.append(
                {
                    "user_id": f"u{user}",
                    "date": day,
                    "hrv": float(rng.normal(52, 7)),
                    "resting_hr": float(rng.normal(58, 4)),
                    "sleep_hours": float(rng.normal(7.2, 0.8)),
                    "load": float(rng.choice([0.0, 0.0, 90.0, 180.0, 270.0])),
                    "completed": int(rng.integers(0, 2)),
                }
            )
    return pd.DataFrame(rows)


def test_spark_matches_pandas_on_latest_row(spark):
    """
    Core parity assertion: for each user's most recent day, the Spark pipeline
    and the pandas serving path must produce the same rolling block.
    """
    history = _make_history()
    sdf = spark.createDataFrame(history)

    spark_latest = (
        latest_per_user(build_rolling_features(sdf)).toPandas().set_index("user_id")
    )

    for user_id, group in history.groupby("user_id"):
        expected = compute_rolling_features(group.sort_values("date"))
        actual = spark_latest.loc[user_id]
        for field in ROLLING_FIELDS:
            assert actual[field] == pytest.approx(
                expected[field], rel=1e-9, abs=1e-9
            ), f"{user_id}.{field}: spark={actual[field]} pandas={expected[field]}"


def test_spark_matches_pandas_on_intermediate_days(spark):
    """
    Parity must hold mid-history too, not just at the end.

    The windows are still filling up early in a user's history — fewer than 7
    and fewer than 28 rows available — and that partial-window regime is where
    two implementations most easily disagree.
    """
    history = _make_history(n_users=2, n_days=40, seed=5)
    sdf = spark.createDataFrame(history)
    spark_all = build_rolling_features(sdf).toPandas()

    for user_id, group in history.groupby("user_id"):
        group = group.sort_values("date")
        for cutoff in (1, 2, 5, 7, 15, 28, 33):
            expected = compute_rolling_features(group[group["date"] <= cutoff])
            row = spark_all[
                (spark_all["user_id"] == user_id) & (spark_all["date"] == cutoff)
            ].iloc[0]
            for field in ROLLING_FIELDS:
                assert row[field] == pytest.approx(
                    expected[field], rel=1e-9, abs=1e-9
                ), (
                    f"{user_id} day {cutoff} {field}: "
                    f"spark={row[field]} pandas={expected[field]}"
                )


def test_pipeline_emits_every_declared_field(spark):
    history = _make_history(n_users=2, n_days=10)
    out = build_rolling_features(spark.createDataFrame(history))
    for field in ROLLING_FIELDS:
        assert field in out.columns


def test_pipeline_partitions_users_independently(spark):
    """
    A user's features must not be contaminated by other users' rows.

    Computing the same user's history alone and alongside others has to give
    the same answer; a missing ``partitionBy`` would silently pool them.
    """
    history = _make_history(n_users=3, n_days=30, seed=3)
    solo = history[history["user_id"] == "u1"].copy()

    together = (
        latest_per_user(build_rolling_features(spark.createDataFrame(history)))
        .toPandas()
        .set_index("user_id")
        .loc["u1"]
    )
    alone = (
        latest_per_user(build_rolling_features(spark.createDataFrame(solo)))
        .toPandas()
        .set_index("user_id")
        .loc["u1"]
    )

    for field in ROLLING_FIELDS:
        assert together[field] == pytest.approx(alone[field], rel=1e-9, abs=1e-9)
