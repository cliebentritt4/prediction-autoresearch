"""
prepare.py — FIXED file. Do not modify.

Contains:
- Constants (MAX_SEQ_LEN, TIME_BUDGET, EVAL_TOKENS)
- Data download + BPE tokenizer (text data from HuggingFace)
- DataLoader (text shards)
- MarketDataLoader (cached market feature arrays)
- evaluate_bpb() — the ground truth metric
- prepare_market_features() — entry point for market data prep

Usage:
    uv run prepare.py           # Download text data + train tokenizer
    uv run prepare.py --market  # Prepare market feature cache
"""

import os
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import tiktoken
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SEQ_LEN = 512
TIME_BUDGET = 300  # 5 minutes wall clock
EVAL_TOKENS = 100_000  # tokens to evaluate on
VOCAB_SIZE = 50257  # GPT-2 BPE vocab size

# Cache directories
TEXT_CACHE_DIR = Path(os.path.expanduser("~/.cache/autoresearch"))
TEXT_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace dataset for text shards
HF_REPO = "karpathy/climbmix-400b-shuffle"
BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
MAX_SHARD = 6542  # shard_00000.parquet through shard_06542.parquet
# Download just 10 shards for dev (enough for experiments)
NUM_SHARDS = 10
SHARD_FILENAMES = [f"shard_{i:05d}.parquet" for i in range(NUM_SHARDS)]

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def get_tokenizer():
    """Return the GPT-2 BPE tokenizer via tiktoken."""
    return tiktoken.get_encoding("gpt2")


def tokens_to_bytes_ratio() -> float:
    """Average bytes per token for BPB conversion."""
    # GPT-2 BPE: ~3.6 bytes per token on average English text
    return 3.6


# ---------------------------------------------------------------------------
# Text Data Download
# ---------------------------------------------------------------------------


def download_text_data():
    """Download text data shards from HuggingFace if not cached."""
    for shard_name in SHARD_FILENAMES:
        shard_path = TEXT_CACHE_DIR / shard_name
        if shard_path.exists():
            continue
        print(f"Downloading {shard_name}...")
        downloaded = hf_hub_download(
            repo_id=HF_REPO,
            filename=shard_name,
            repo_type="dataset",
            local_dir=str(TEXT_CACHE_DIR),
        )
        # Move from nested structure to flat cache dir
        downloaded_path = Path(downloaded)
        if downloaded_path != shard_path:
            import shutil

            shutil.move(str(downloaded_path), str(shard_path))
        print(f"  -> {shard_path}")
    print(f"All text shards cached in {TEXT_CACHE_DIR}")


def load_shard_tokens(shard_path: Path) -> np.ndarray:
    """Load a parquet shard and tokenize the text column."""
    import pyarrow.parquet as pq

    table = pq.read_table(shard_path)
    # climbmix parquet has a 'text' column
    texts = table.column("text").to_pylist()
    enc = get_tokenizer()
    all_tokens = []
    for text in texts:
        tokens = enc.encode(text, allowed_special={"<|endoftext|>"}, disallowed_special=())
        all_tokens.extend(tokens)
    return np.array(all_tokens, dtype=np.int32)


# ---------------------------------------------------------------------------
# DataLoader — text shards
# ---------------------------------------------------------------------------


class DataLoader:
    """
    Loads pre-tokenized text data shards and serves random batches.

    Each batch is (input_ids, target_ids) where target = input shifted by 1.
    """

    def __init__(
        self, split: str = "train", batch_size: int = 4, seq_len: int = MAX_SEQ_LEN
    ):
        self.batch_size = batch_size
        self.seq_len = seq_len

        # Use first N-1 shards for train, last 1 for val
        if split == "train":
            shard_indices = list(range(NUM_SHARDS - 1))
        else:
            shard_indices = [NUM_SHARDS - 1]

        # Load and concatenate all shard tokens
        all_tokens = []
        for idx in shard_indices:
            shard_path = TEXT_CACHE_DIR / SHARD_FILENAMES[idx]
            if shard_path.exists():
                tokens = load_shard_tokens(shard_path)
                all_tokens.append(tokens)

        if not all_tokens:
            raise RuntimeError(
                f"No text data shards found in {TEXT_CACHE_DIR}. "
                "Run `uv run prepare.py` first to download them."
            )

        self.tokens = np.concatenate(all_tokens).astype(np.int32)
        self.n_tokens = len(self.tokens)
        self._rng = np.random.default_rng(42 if split == "val" else int(time.time()))

    def next_batch(self) -> tuple[mx.array, mx.array]:
        """
        Return (inputs, targets) each of shape (batch_size, seq_len).
        Targets are inputs shifted right by 1.
        """
        starts = self._rng.integers(
            0, self.n_tokens - self.seq_len - 1, size=self.batch_size
        )

        input_batch = np.stack([self.tokens[s : s + self.seq_len] for s in starts])
        target_batch = np.stack(
            [self.tokens[s + 1 : s + self.seq_len + 1] for s in starts]
        )

        return mx.array(input_batch), mx.array(target_batch)


# ---------------------------------------------------------------------------
# MarketDataLoader — cached market feature arrays
# ---------------------------------------------------------------------------


class MarketDataLoader:
    """
    Loads cached market feature sequences and serves batches.

    Each batch is (features, labels, mask) where:
    - features: (batch_size, MAX_SEQ_LEN, n_features) — market time series
    - labels: (batch_size,) — resolution outcome (0 or 1)
    - mask: (batch_size, MAX_SEQ_LEN) — 1 where data exists, 0 where padded
    """

    def __init__(self, split: str = "train", batch_size: int = 16):
        from market.features.cache import load_sequences, is_cache_valid

        if not is_cache_valid():
            raise RuntimeError(
                "Market feature cache is missing or stale. "
                "Run `uv run prepare.py --market` to rebuild it."
            )

        data = load_sequences()

        if split == "train":
            self.features = mx.array(data["train_features"], dtype=mx.float32)
            self.labels = mx.array(data["train_labels"], dtype=mx.float32)
        else:
            self.features = mx.array(data["val_features"], dtype=mx.float32)
            self.labels = mx.array(data["val_labels"], dtype=mx.float32)

        self.feature_names = data["feature_names"]
        self.n_markets = self.features.shape[0]
        self.batch_size = min(batch_size, self.n_markets)
        self._rng = np.random.default_rng(42 if split == "val" else int(time.time()))

    def next_batch(self) -> tuple[mx.array, mx.array, mx.array]:
        """
        Return (features, labels, mask).

        features: (batch_size, MAX_SEQ_LEN, n_features)
        labels: (batch_size,)
        mask: (batch_size, MAX_SEQ_LEN) — 1 for real data, 0 for padding
        """
        indices = self._rng.integers(0, self.n_markets, size=self.batch_size)

        batch_features = self.features[indices]
        batch_labels = self.labels[indices]

        # Build mask: padding is all-zeros rows
        # Sum across features — if all features are 0, it's padding
        row_sums = mx.abs(batch_features).sum(axis=-1)  # (batch, seq_len)
        mask = (row_sums > 1e-8).astype(mx.float32)

        return batch_features, batch_labels, mask


# ---------------------------------------------------------------------------
# Evaluation — bits per byte (ground truth metric)
# ---------------------------------------------------------------------------


def evaluate_bpb(
    model, split: str = "val", seq_len: int = MAX_SEQ_LEN, n_tokens: int = EVAL_TOKENS
) -> float:
    """
    Evaluate bits-per-byte on text validation data.

    This is the ground truth metric. Lower is better.
    BPB = cross_entropy_in_nats / ln(2) / bytes_per_token

    Args:
        model: must accept (batch_size, seq_len) int inputs, return logits
        split: "train" or "val"
        seq_len: sequence length for evaluation
        n_tokens: total tokens to evaluate on

    Returns:
        float — bits per byte
    """
    loader = DataLoader(split=split, batch_size=4, seq_len=seq_len)

    total_loss = 0.0
    total_tokens = 0
    bpb_ratio = tokens_to_bytes_ratio()

    while total_tokens < n_tokens:
        inputs, targets = loader.next_batch()

        logits = model(inputs)

        # Cross-entropy loss in nats
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
        mx.eval(loss)

        batch_tokens = targets.size
        total_loss += loss.item()
        total_tokens += batch_tokens

    # Convert: nats -> bits -> bits per byte
    avg_nats = total_loss / total_tokens
    avg_bits = avg_nats / np.log(2)
    bpb = avg_bits / bpb_ratio

    return float(bpb)


# ---------------------------------------------------------------------------
# Market Feature Preparation
# ---------------------------------------------------------------------------


def prepare_market_features():
    """
    Extract market features from parquet data, build sequences,
    and cache as .npz for the training loop.
    """
    from market.features.extract import get_resolved_features
    from market.features.sequences import build_sequences
    from market.features.cache import save_sequences, is_cache_valid

    if is_cache_valid():
        print("Market feature cache is up to date. Use --force to rebuild.")
        return

    print("Extracting market features from parquet files...")
    df = get_resolved_features(source="all")
    print(f"  Found {len(df)} feature rows across {df['ticker'].nunique()} markets")

    if len(df) == 0:
        print("  No resolved market data found. Run indexers first (make index).")
        return

    print("Building sequences...")
    seqs = build_sequences(df)
    print(f"  Train: {seqs.train_features.shape[0]} markets")
    print(f"  Val: {seqs.val_features.shape[0]} markets")
    print(f"  Features: {seqs.feature_names}")

    print("Saving to cache...")
    save_sequences(
        train_features=np.array(seqs.train_features),
        train_labels=np.array(seqs.train_labels),
        val_features=np.array(seqs.val_features),
        val_labels=np.array(seqs.val_labels),
        feature_names=seqs.feature_names,
    )
    print("Done. Market features cached.")


# ---------------------------------------------------------------------------
# Main — download text data or prepare market features
# ---------------------------------------------------------------------------


def main():
    if "--market" in sys.argv:
        prepare_market_features()
    elif "--force" in sys.argv and "--market" in sys.argv:
        # Force rebuild market cache
        from market.features.cache import CACHE_FILE

        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
        prepare_market_features()
    else:
        print("=== Downloading text data shards ===")
        download_text_data()
        print()
        print("=== Tokenizer check ===")
        enc = get_tokenizer()
        test = enc.encode("Hello, prediction markets!")
        print(f"  Tokenizer OK: {len(test)} tokens")
        print()
        print("Text data ready. Run with --market to prepare market features.")


if __name__ == "__main__":
    main()
