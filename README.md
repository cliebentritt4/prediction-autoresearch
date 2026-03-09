# prediction-autoresearch

Autonomous ML research on prediction market data using MLX on Apple Silicon.

Merges two projects into a single autonomous research loop:
- **autoresearch-mlx** — Karpathy's autoresearch (5-min training loop) ported to MLX
- **prediction-market-analysis** — Polymarket/Kalshi data collection + DuckDB analysis

The autoresearch agent modifies `train.py`, runs a 5-minute experiment, evaluates whether
prediction market signal features improve model performance, and iterates overnight.

## Quick Start

```bash
# Install dependencies (requires uv)
make sync

# Download text data shards
uv run prepare.py

# Index market data from APIs (requires API keys in .env)
make index

# Prepare market features
uv run prepare.py --market

# Run baseline training (5 min)
make train
```

## Project Structure

```
prediction-autoresearch/
├── prepare.py          # FIXED — data prep, tokenizer, eval harness
├── train.py            # AGENT-MODIFIABLE — model + optimizer + loop
├── program.md          # Agent instructions
├── results.tsv         # Experiment log
├── market/             # Prediction market data pipeline
│   ├── config.py       # Paths, constants, env vars
│   ├── indexers/       # Data collection (Kalshi + Polymarket)
│   ├── features/       # Feature extraction → MLX arrays
│   └── analysis/       # Standalone analysis scripts
├── scripts/            # Utility scripts
├── tests/              # Unit tests
└── data/               # Raw market data (gitignored)
```

## How It Works

1. **Data Collection**: Indexers fetch market metadata and trade history from Kalshi and Polymarket APIs, storing results as parquet files.

2. **Feature Extraction**: DuckDB queries on parquet files extract per-market time series features (price, volume, spread, volatility, time-to-resolution, outcome).

3. **Sequence Building**: Features are converted to fixed-length MLX arrays with normalization and train/val splits by market ticker.

4. **Autoresearch Loop**: An AI agent modifies `train.py` to experiment with architectures and features. Each experiment runs for 5 minutes. The ground truth metric is `val_bpb` (bits per byte on text validation data).

## Market Features

Per market per hourly timestep:
- `yes_price` — normalized 0-1
- `volume` — log-scaled trade count
- `spread` — proxy for bid-ask spread
- `volatility` — rolling price standard deviation
- `time_to_resolution` — log-scaled seconds remaining
- `outcome` — binary resolution (resolved markets only)

## Requirements

- Python 3.11+
- Apple Silicon Mac (for MLX)
- [uv](https://github.com/astral-sh/uv) package manager
- API keys for Kalshi and/or Polymarket (in `.env`)

## Configuration

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

## Running Tests

```bash
make test
```

## License

MIT
