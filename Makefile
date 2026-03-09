.PHONY: setup sync prepare train index analyze compress test lint clean serve runner research

# Setup: install all dependencies via uv
setup: sync index
	@echo "Setup complete. Run 'make prepare' to prepare data."

# Sync dependencies
sync:
	uv sync

# Prepare data (text shards + tokenizer; add --market for market features)
prepare:
	uv run prepare.py

prepare-market:
	uv run prepare.py --market

# Train (agent runs this in a loop)
train:
	uv run train.py > run.log 2>&1
	@grep "^val_bpb:\|^peak_mem_mb:" run.log || echo "Training may have crashed. Check run.log"

# Index market data from APIs
index:
	uv run -m market.indexers.kalshi.markets
	uv run -m market.indexers.kalshi.trades
	uv run -m market.indexers.polymarket.markets
	uv run -m market.indexers.polymarket.trades

# Run analysis scripts interactively
analyze:
	uv run scripts/run_analysis.py

# Compress data directory for storage/transfer
compress:
	tar -cf - data/ | zstd -19 -T0 -o data.tar.zst
	@echo "Compressed to data.tar.zst"

# Download pre-collected data
download:
	bash scripts/download_data.sh

# Run tests
test:
	uv run pytest

# Lint
lint:
	uv run ruff check .
	uv run ruff format --check .

# Start prediction API server
serve:
	uv run uvicorn market.api.server:app --host 0.0.0.0 --port 8000 --reload

# Set up self-hosted GitHub Actions runner
runner:
	@echo "To set up the self-hosted runner:"
	@echo "1. Go to https://github.com/cliebentritt4/prediction-autoresearch/settings/actions/runners/new"
	@echo "2. Select macOS + ARM64"
	@echo "3. Follow the download and configure steps"
	@echo "4. Run: ./run.sh (or install as service with ./svc.sh install)"

# Run Claude Code autoresearch loop locally (5 experiments)
research:
	@echo "Starting autoresearch loop with Claude Code..."
	claude -p "Read program.md and follow the autoresearch experiment loop. Run 5 experiments. Only modify train.py. Record results in results.tsv. Commit improvements, revert failures." --allowedTools "Edit,Read,Write,Bash,Glob,Grep" --max-turns 100

# Clean generated files
clean:
	rm -rf output/*.png output/*.csv output/*.json
	rm -f run.log
	rm -rf __pycache__ market/__pycache__ tests/__pycache__
