# prediction-autoresearch

Autonomous ML research on prediction market data using MLX on Apple Silicon.

## Project Structure

- `train.py` — The model file. This is the ONLY file the autoresearch agent modifies.
- `prepare.py` — FIXED. Data download, tokenizer, DataLoader, MarketDataLoader, evaluate_bpb().
- `program.md` — Agent instructions for the autoresearch experiment loop.
- `results.tsv` — Experiment history (date, commit, val_bpb, peak_mem_mb, note).
- `market/` — Market data pipeline (indexers, features, analysis, API).
- `data/` — Collected parquet files from Kalshi and Polymarket (gitignored).

## Key Commands

```bash
uv sync                                    # Install deps
uv run prepare.py                          # Download text data shards
uv run prepare.py --market                 # Build market feature cache
uv run train.py                            # Run training experiment (5 min)
uv run -m market.indexers.kalshi.markets   # Collect Kalshi markets
uv run -m market.indexers.kalshi.trades    # Collect Kalshi trades
uv run -m market.indexers.polymarket.markets  # Collect Polymarket markets
uv run -m market.indexers.polymarket.trades   # Collect Polymarket trades
make serve                                 # Start prediction API on :8000
make test                                  # Run pytest
make lint                                  # Run ruff
```

## Autoresearch Rules

- ONLY modify `train.py` during experiments
- NEVER modify `prepare.py`, `market/*`, or `program.md`
- NEVER install new packages beyond what's in pyproject.toml
- Time budget is 5 minutes per experiment
- val_bpb is the ground truth metric (lower is better)
- Commit improvements, revert failures
- Record every experiment in results.tsv

## Tech Stack

- MLX (Apple Silicon ML framework, replaces PyTorch)
- DuckDB (SQL on parquet files)
- tiktoken GPT-2 BPE tokenizer
- FastAPI for prediction serving
- uv for package management (not pip/venv)
- GitHub Actions CI/CD with self-hosted macOS ARM64 runner
