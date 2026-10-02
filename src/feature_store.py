"""
Feature Store — SQLite-backed audit trail for live features.
=============================================================
Stores every feature vector computed during live trading so we can:
1. Audit predictions after the fact
2. Retrain on exact production features (no train/serve skew)
3. Detect feature drift via distribution comparison

Author: Om | Strategy Research Lab
"""

import os
import json
import sqlite3
import logging
import time
import numpy as np
import pandas as pd
from typing import Optional
from contextlib import contextmanager

logger = logging.getLogger("feature_store")

DB_PATH = os.getenv("FEATURE_STORE_DB", "data/feature_store.db")


# ------------------------------------------------------------------
# Database manager
# ------------------------------------------------------------------
class FeatureStore:
    """SQLite-backed feature store for audit and retraining."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    prediction INTEGER,
                    prediction_prob REAL,
                    actual_label INTEGER,
                    model_version TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS drift_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    feature_name TEXT NOT NULL,
                    mean REAL,
                    std REAL,
                    min REAL,
                    max REAL,
                    n_samples INTEGER
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_ts
                ON features(timestamp)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_features_symbol
                ON features(symbol)
            """)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------
    def log_features(
        self,
        timestamp: str,
        symbol: str,
        features: dict,
        prediction: Optional[int] = None,
        prediction_prob: Optional[float] = None,
        model_version: Optional[str] = None,
    ):
        """Log a feature vector with its prediction."""
        features_json = json.dumps(
            {k: float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
             for k, v in features.items()}
        )
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO features
                   (timestamp, symbol, features_json, prediction, prediction_prob, model_version)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (timestamp, symbol, features_json, prediction, prediction_prob, model_version),
            )

    def update_actual_label(self, feature_id: int, actual_label: int):
        """Update the actual label once known (for delayed feedback)."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE features SET actual_label = ? WHERE id = ?",
                (actual_label, feature_id),
            )

    def log_drift_snapshot(self, feature_stats: dict, timestamp: Optional[str] = None):
        """
        Log distribution statistics for drift detection.
        feature_stats: {feature_name: {mean, std, min, max, n_samples}}
        """
        ts = timestamp or time.strftime("%Y-%m-%d %H:%M:%S")
        with self._connect() as conn:
            for feat_name, stats in feature_stats.items():
                conn.execute(
                    """INSERT INTO drift_snapshots
                       (timestamp, feature_name, mean, std, min, max, n_samples)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (ts, feat_name,
                     stats.get("mean", 0), stats.get("std", 0),
                     stats.get("min", 0), stats.get("max", 0),
                     stats.get("n_samples", 0)),
                )

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------
    def get_recent_features(self, symbol: str = "BTCUSDT", limit: int = 1000) -> pd.DataFrame:
        """Fetch recent feature vectors as a DataFrame."""
        with self._connect() as conn:
            df = pd.read_sql_query(
                """SELECT id, timestamp, features_json, prediction,
                          prediction_prob, actual_label, model_version
                   FROM features WHERE symbol = ?
                   ORDER BY id DESC LIMIT ?""",
                conn,
                params=(symbol, limit),
            )

        if df.empty:
            return df

        # expand features_json into columns
        features_expanded = df["features_json"].apply(json.loads).apply(pd.Series)
        result = pd.concat([
            df[["id", "timestamp", "prediction", "prediction_prob", "actual_label", "model_version"]],
            features_expanded,
        ], axis=1)

        return result.sort_values("id").reset_index(drop=True)

    def get_training_data(self, symbol: str = "BTCUSDT", min_samples: int = 100) -> Optional[pd.DataFrame]:
        """
        Get features with actual labels for retraining.
        Only returns rows where actual_label is not NULL.
        """
        with self._connect() as conn:
            df = pd.read_sql_query(
                """SELECT features_json, actual_label
                   FROM features
                   WHERE symbol = ? AND actual_label IS NOT NULL
                   ORDER BY id""",
                conn,
                params=(symbol,),
            )

        if len(df) < min_samples:
            logger.info("Not enough labeled samples for retraining (%d < %d)", len(df), min_samples)
            return None

        features_expanded = df["features_json"].apply(json.loads).apply(pd.Series)
        features_expanded["target"] = df["actual_label"].astype(int)
        return features_expanded

    def count_records(self, symbol: str = "BTCUSDT") -> int:
        """Count total feature records for a symbol."""
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) FROM features WHERE symbol = ?",
                (symbol,),
            )
            return cursor.fetchone()[0]

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------
    def compute_feature_stats(self, symbol: str = "BTCUSDT", window: int = 500) -> dict:
        """Compute current feature distribution statistics."""
        df = self.get_recent_features(symbol=symbol, limit=window)
        if df.empty:
            return {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # exclude metadata columns
        exclude = {"id", "prediction", "prediction_prob", "actual_label"}
        feature_cols = [c for c in numeric_cols if c not in exclude]

        stats = {}
        for col in feature_cols:
            vals = df[col].dropna()
            if len(vals) > 0:
                stats[col] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                    "min": float(vals.min()),
                    "max": float(vals.max()),
                    "n_samples": len(vals),
                }

        return stats

    def detect_drift(self, symbol: str = "BTCUSDT", threshold: float = 2.0) -> list:
        """
        Simple drift detection: compare recent feature means to historical.
        Returns list of feature names where |zscore of mean shift| > threshold.
        """
        with self._connect() as conn:
            # get the two most recent snapshots
            snaps = pd.read_sql_query(
                """SELECT DISTINCT timestamp FROM drift_snapshots
                   ORDER BY timestamp DESC LIMIT 2""",
                conn,
            )

        if len(snaps) < 2:
            return []

        ts_new, ts_old = snaps.iloc[0]["timestamp"], snaps.iloc[1]["timestamp"]

        with self._connect() as conn:
            new_stats = pd.read_sql_query(
                "SELECT * FROM drift_snapshots WHERE timestamp = ?",
                conn, params=(ts_new,),
            )
            old_stats = pd.read_sql_query(
                "SELECT * FROM drift_snapshots WHERE timestamp = ?",
                conn, params=(ts_old,),
            )

        drifted = []
        old_map = {row["feature_name"]: row for _, row in old_stats.iterrows()}

        for _, row in new_stats.iterrows():
            fname = row["feature_name"]
            if fname in old_map:
                old = old_map[fname]
                if old["std"] > 1e-9:
                    z = abs(row["mean"] - old["mean"]) / old["std"]
                    if z > threshold:
                        drifted.append(fname)

        return drifted
