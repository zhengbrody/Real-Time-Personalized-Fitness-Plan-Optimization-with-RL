"""
PySpark Offline Feature Pipeline
================================

Materialises the rolling-window feature block for every user-day in the logged
history, so the online path can serve it from cache instead of recomputing it
per request.

Relationship to the serving path
--------------------------------
:func:`src.serving.feature_service.compute_rolling_features` computes the same
block in pandas, for one user, at request time.  This module computes it for
every user and every day at once.  Two implementations of the same semantics is
exactly the setup that produces training–serving skew, so the equivalence is
not assumed — ``tests/test_spark_pipeline.py`` runs both over the same history
and asserts they agree to floating-point tolerance on every field.

If the two ever diverge, that test fails before the divergence reaches a model.

Why Spark rather than pandas here
---------------------------------
The per-request path handles one user's ~90 rows.  The offline path handles the
whole population's history every night, and the work is a set of partitioned
window aggregations — the shape Spark's window functions are built for, and one
that parallelises across users without any coordination.  The run-length
features (streak, days since training, consecutive hard days) are the only
awkward part; they are expressed here as gaps-and-islands queries rather than
by collecting each user's rows to the driver.

Usage
-----
    python -m src.feature_store.spark_pipeline \\
        --input data/history.parquet --output data/features.parquet
"""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, List, Optional

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pyspark.sql import DataFrame, SparkSession

__all__ = [
    "build_rolling_features",
    "get_spark",
    "SLEEP_TARGET_HOURS",
    "ACTIVE_LOAD_THRESHOLD",
    "HARD_LOAD_THRESHOLD",
]

# These thresholds are the contract with the serving-path implementation.
# Changing one here without changing it there is precisely what the parity test
# is watching for.
SLEEP_TARGET_HOURS = 7.5
ACTIVE_LOAD_THRESHOLD = 1.0
HARD_LOAD_THRESHOLD = 180.0

MAX_DAYS_SINCE = 7
MAX_STREAK = 14
MAX_CONSEC_HARD = 5

REQUIRED_COLUMNS = (
    "user_id",
    "date",
    "hrv",
    "resting_hr",
    "sleep_hours",
    "load",
    "completed",
)


def get_spark(app_name: str = "profit-feature-pipeline", local: bool = True):
    """Build (or fetch) a SparkSession."""
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.appName(app_name)
    if local:
        builder = builder.master("local[*]").config("spark.sql.shuffle.partitions", "8")
    return builder.getOrCreate()


def build_rolling_features(history: "DataFrame") -> "DataFrame":
    """
    Compute the rolling-window feature block for every user-day.

    Parameters
    ----------
    history
        One row per user per day with columns :data:`REQUIRED_COLUMNS`.

    Returns
    -------
    DataFrame
        ``user_id``, ``date``, and one column per rolling field.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    missing = [c for c in REQUIRED_COLUMNS if c not in history.columns]
    if missing:
        raise ValueError(f"history is missing required columns: {missing}")

    ordered = Window.partitionBy("user_id").orderBy("date")
    w7 = ordered.rowsBetween(-6, 0)
    w28 = ordered.rowsBetween(-27, 0)
    running = ordered.rowsBetween(Window.unboundedPreceding, 0)

    df = history.withColumn("rn", F.row_number().over(ordered))

    is_active = F.col("load") > ACTIVE_LOAD_THRESHOLD
    is_hard = F.col("load") >= HARD_LOAD_THRESHOLD

    df = (
        df
        # -- recovery ----------------------------------------------------
        .withColumn("hrv_7d_mean", F.avg("hrv").over(w7))
        .withColumn("hrv_28d_mean", F.avg("hrv").over(w28))
        # stddev_pop, not stddev_samp: the serving path uses ddof=0, and a
        # one-row window must yield 0.0 rather than NULL.
        .withColumn(
            "hrv_28d_std", F.coalesce(F.stddev_pop("hrv").over(w28), F.lit(0.0))
        )
        .withColumn("resting_hr_baseline", F.avg("resting_hr").over(w28))
        .withColumn(
            "sleep_debt_7d",
            F.sum(F.lit(SLEEP_TARGET_HOURS) - F.col("sleep_hours")).over(w7),
        )
        # -- load --------------------------------------------------------
        .withColumn("load_7d", F.sum("load").over(w7))
        .withColumn("_load_7d_mean", F.avg("load").over(w7))
        .withColumn("load_28d_mean", F.avg("load").over(w28))
        .withColumn(
            "acwr",
            F.when(
                F.col("load_28d_mean") > 1e-6,
                F.col("_load_7d_mean") / F.col("load_28d_mean"),
            ).otherwise(F.lit(1.0)),
        )
        # -- consistency -------------------------------------------------
        .withColumn(
            "completion_rate_7d", F.avg(F.col("completed").cast("double")).over(w7)
        )
        .withColumn(
            "completion_rate_28d", F.avg(F.col("completed").cast("double")).over(w28)
        )
        .withColumn(
            "sessions_7d", F.sum(F.when(is_active, 1.0).otherwise(0.0)).over(w7)
        )
        .withColumn(
            "days_active_28d", F.sum(F.when(is_active, 1.0).otherwise(0.0)).over(w28)
        )
    )

    # -- run-length features (gaps and islands) ---------------------------
    #
    # Each of these is "how many rows since the last row that broke the run".
    # Taking the running max of the row number of the breaking rows gives that
    # boundary directly, which avoids collecting a user's history to the driver
    # just to walk it backwards.
    df = (
        df.withColumn(
            "_last_active_rn",
            F.max(F.when(is_active, F.col("rn"))).over(running),
        )
        .withColumn(
            "_last_inactive_rn",
            F.max(F.when(~is_active, F.col("rn"))).over(running),
        )
        .withColumn(
            "_last_not_hard_rn",
            F.max(F.when(~is_hard, F.col("rn"))).over(running),
        )
        .withColumn(
            "days_since_training",
            F.least(
                F.col("rn") - F.coalesce(F.col("_last_active_rn"), F.lit(0)),
                F.lit(float(MAX_DAYS_SINCE)),
            ).cast("double"),
        )
        .withColumn(
            "streak",
            F.least(
                F.col("rn") - F.coalesce(F.col("_last_inactive_rn"), F.lit(0)),
                F.lit(float(MAX_STREAK)),
            ).cast("double"),
        )
        .withColumn(
            "consecutive_hard_days",
            F.least(
                F.col("rn") - F.coalesce(F.col("_last_not_hard_rn"), F.lit(0)),
                F.lit(float(MAX_CONSEC_HARD)),
            ).cast("double"),
        )
    )

    output_columns: List[str] = [
        "user_id",
        "date",
        "hrv_7d_mean",
        "hrv_28d_mean",
        "hrv_28d_std",
        "resting_hr_baseline",
        "sleep_debt_7d",
        "acwr",
        "load_7d",
        "load_28d_mean",
        "days_since_training",
        "consecutive_hard_days",
        "completion_rate_7d",
        "completion_rate_28d",
        "streak",
        "sessions_7d",
        "days_active_28d",
    ]
    return df.select(*output_columns)


def latest_per_user(features: "DataFrame") -> "DataFrame":
    """
    Keep only each user's most recent row — the snapshot the online cache holds.
    """
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    w = Window.partitionBy("user_id").orderBy(F.col("date").desc())
    return (
        features.withColumn("_rank", F.row_number().over(w))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
    )


def main() -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Offline rolling-feature pipeline")
    parser.add_argument("--input", required=True, help="history parquet/csv path")
    parser.add_argument("--output", required=True, help="output parquet path")
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="write only the newest row per user (the cache snapshot)",
    )
    args = parser.parse_args()

    spark = get_spark()
    reader = spark.read
    history = (
        reader.parquet(args.input)
        if args.input.endswith(".parquet")
        else reader.option("header", True).option("inferSchema", True).csv(args.input)
    )

    features = build_rolling_features(history)
    if args.latest_only:
        features = latest_per_user(features)

    features.write.mode("overwrite").parquet(args.output)
    print(f"wrote {features.count()} rows -> {args.output}")
    spark.stop()


if __name__ == "__main__":  # pragma: no cover
    main()
