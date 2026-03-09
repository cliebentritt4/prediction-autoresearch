# Autoresearch Reference Notes

Findings from community member running Karpathy's autoresearch on Mac Mini M4 (16GB RAM).
Directly applicable to our Mac Studio setup.

## Hardware Constraints (Apple Silicon, 16GB)

- 11.5M parameter GPT is what fits in 16GB — don't try to go bigger without checking memory first
- 50M params (depth 8) → OOM crash
- 26M params (depth 6, batch 8) → ran but val_bpb was worse than the tiny baseline
- **Key lesson: a small well-trained model beats a large undertrained one on limited compute**

## Best Hyperparameter Findings

- Halving batch size was the single biggest win
- Batch 32K → val_bpb 1.5960 (first improvement over baseline)
- Batch 16K → val_bpb 1.4787 (best result, 15.7% improvement over baseline)
- Why: smaller batch = more optimizer steps in the same time budget (102 → 370 steps)
- More steps > bigger batches when time is fixed at 5 minutes

## Performance Comparison

- Mac M4 (16GB, $600): val_bpb 1.4787
- Karpathy's H100 ($30K GPU): val_bpb 0.9979
- M4 is ~2.5x slower per cycle but 50x cheaper hardware
- Our Mac Studio should be faster than M4 (more RAM, more GPU cores)

## Autonomous Loop Setup (What Worked)

- launchd starts a tmux session at 9 PM
- Runs `claude -p` in a bash loop: read results → decide experiment → edit train.py → run → check → keep or revert → log → repeat
- Stops at 6 AM (9 hours of autonomous research per night)
- ~45 experiments per night, ~315 per week
- Telegram bot sends debrief with overnight stats at 6:30 AM

## Our Hardware: Mac Studio M4 Max (36GB)

- Chip: Apple M4 Max
- Memory: 36 GB unified (shared CPU/GPU)
- ~2.25x the RAM of the M4 Mini reference (16GB)
- M4 Max has 40 GPU cores vs M4's 10 — roughly 4x GPU throughput
- Can likely support 25-30M parameter models (vs 11.5M on 16GB)
- Or same model size with much larger effective batch / faster iterations

## Implications for Our Project

1. **Start with batch size tuning** — try 32K, 16K, 8K before anything else
2. **We CAN try larger models** — 36GB means 25-30M params is feasible, but still test carefully (the lesson about small well-trained > large undertrained still applies)
3. **More GPU cores = faster iterations** — we might get 60+ experiments per night vs their 45
4. **The overnight loop works** — our GitHub Actions autoresearch workflow does the same thing but with better tracking (results.tsv, PRs, commit comments)
5. **Market data is our edge** — this person got 1.4787 on text-only. If our market features push it lower, that's a novel finding nobody else has

## Multi-Agent Coordination Strategy

The repo is the shared brain. All agents read/write through git.

- **Claude Code (researcher)**: runs autoresearch loop overnight via GitHub Actions or `make research`. Edits train.py, runs experiments, commits improvements, reverts failures. Guided by program.md and results.tsv.
- **Cowork Claude (infrastructure)**: handles data pipeline, debugging, new features, CI/CD, monitoring. Steers research direction based on results.
- **Prediction API**: serves the best model checkpoint. Once training produces good results, the API enables signal generation and eventually trading.
- **GitHub Actions**: orchestrates everything — triggers training on train.py changes, runs data collection daily, monitors data freshness, opens PRs for review.

Flow: data collects → features extract → Claude Code experiments → good results committed → CI validates → API serves best model → signals generated
