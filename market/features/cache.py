"""
market/features/cache.py — Caching layer for processed feature tensors.

Saves/loads processed sequences as .npz files in ~/.cache/autoresearch/market/.
Invalidates cache when data/ directory mtime changes.
"""

import os
from pathlib import Path

import numpy as np

from market.config import CACHE_DIR, DATA_DIR

CACHE_PATH = Path(CACHE_DIR)
CACHE_FILE = CACHE_PATH / "market_sequences.npz"
MTIME_FILE = CACHE_PATH / ".data_mtime"


def is_cache_valid() -> bool:
    """Check if the cached sequences are still fresh."""
    if not CACHE_FILE.exists() or not MTIME_FILE.exists():
        return False

    try:
        saved_mtime = float(MTIME_FILE.read_text().strip())
        current_mtime = _get_data_mtime()
        return abs(saved_mtime - current_mtime) < 1.0
    except (ValueError, OSError):
        return False


def save_sequences(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    feature_names: list[str],
) -> None:
    """Save processed sequences to cache."""
    CACHE_PATH.mkdir(parents=True, exist_ok=True)

    np.savez(
        CACHE_FILE,
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features,
        val_labels=val_labels,
        feature_names=np.array(feature_names),
    )

    MTIME_FILE.write_text(str(_get_data_mtime()))


def load_sequences() -> dict:
    """
    Load cached sequences.

    Returns:
        Dict with keys: train_features, train_labels,
        val_features, val_labels, feature_names
    """
    data = np.load(CACHE_FILE, allow_pickle=False)
    return {
        "train_features": data["train_features"],
        "train_labels": data["train_labels"],
        "val_features": data["val_features"],
        "val_labels": data["val_labels"],
        "feature_names": list(data["feature_names"]),
    }


def _get_data_mtime() -> float:
    """Get the most recent modification time across all data/ files."""
    data_path = Path(DATA_DIR)
    if not data_path.exists():
        return 0.0

    mtimes = []
    for root, _dirs, files in os.walk(data_path):
        for f in files:
            if f.endswith(".parquet"):
                mtimes.append(os.path.getmtime(os.path.join(root, f)))

    return max(mtimes) if mtimes else 0.0
