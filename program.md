# prediction-autoresearch — Agent Program

You are an autonomous ML researcher. Your job is to improve val_bpb by
modifying train.py. You run experiments in a loop: edit → train (5 min) →
evaluate → keep or discard → repeat.

## Setup Checklist

1. Agree on a run tag (e.g. `mar8`). Branch: `autoresearch/<tag>`.
2. `git checkout -b autoresearch/<tag>` from main.
3. Read these files for full context:
   - `README.md` — project overview
   - `prepare.py` — FIXED constants, text data prep, market data loader, eval
   - `train.py` — the file you modify
   - `market/features/extract.py` — what market features are available
   - `market/features/sequences.py` — how features become MLX arrays
4. Verify data exists:
   - Text: `~/.cache/autoresearch/` has data shards and tokenizer
   - Market: `data/kalshi/` and `data/polymarket/` have parquet files
   - If text data missing → tell human to run `uv run prepare.py`
   - If market data missing → tell human to run `make setup`
   - If market feature cache missing → run `uv run prepare.py --market`
5. Initialize `results.tsv` with header and baseline.

## Available Data Sources

### Text Data (from autoresearch upstream)
- Loaded via `DataLoader` from `prepare.py`
- HuggingFace climbmix shards, BPE tokenized
- Standard next-token prediction objective

### Market Data (NEW)
- Loaded via `MarketDataLoader` from `prepare.py`
- Features per market per timestep:
  - `yes_price` (0-100 cents, normalized to 0-1)
  - `volume` (trade count in window, log-scaled)
  - `spread` (proxy: abs(yes_price - 50), normalized)
  - `volatility` (rolling std of price, 1h window)
  - `time_to_resolution` (seconds remaining, log-scaled)
  - `outcome` (0 or 1, for resolved markets only)
- Sequences are padded/truncated to MAX_SEQ_LEN
- Train/val split by market (no data leakage)

### Using Both Together
You can train on text alone, market data alone, or combine them.
Ideas for combination:
- Multi-task: predict next token AND market outcome
- Feature conditioning: use market state as context for text generation
- Architecture experiments: separate encoders with cross-attention
- Pure market prediction: ignore text, predict price sequences

## Experiment Loop

1. Have an idea. Write it in one sentence.
2. Edit `train.py` to implement it.
3. Run: `uv run train.py > run.log 2>&1`
4. Read results: `grep "^val_bpb:\|^peak_mem_mb:" run.log`
5. If grep empty → crashed. `tail -n 50 run.log`, attempt fix.
   After 3 failed fix attempts, give up and revert.
6. Record in results.tsv:
   `<run_id>\t<val_bpb>\t<peak_mem_mb>\t<description>\t<timestamp>`
7. If val_bpb improved → `git add -A && git commit -m "<description>"`
8. If val_bpb same or worse → `git checkout -- train.py`
9. Repeat from step 1.

## Constraints

- **Only modify `train.py`.** Never touch prepare.py, market/*, or program.md.
- **Do not install new packages.** Use only what's in pyproject.toml.
- **Do not modify the evaluation harness.** `evaluate_bpb()` is ground truth.
- **Time budget is fixed at 5 minutes.** All experiments are comparable.
- **Memory is a soft constraint.** Some increase is fine for meaningful
  val_bpb gains, but don't blow it up 10x.
- **Simplicity wins.** Equal results + less complexity = keep the simpler version.

## Output Format

train.py must print exactly these lines at the end:

```
val_bpb: <float>
peak_mem_mb: <float>
```

## Strategy Notes

- Start with the baseline (text-only) to establish your current val_bpb.
- Early experiments should be small, safe changes (lr, batch size, warmup).
- Market data experiments are higher risk/reward — try after you've
  squeezed the text-only baseline.
- The market data has rich temporal structure. Transformers should work
  well on the price sequences.
- Remember: val_bpb is computed on TEXT validation data. Market features
  should improve the model's general capability, not overfit to market
  patterns.
