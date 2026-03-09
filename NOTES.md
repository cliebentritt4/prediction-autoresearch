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

## Implications for Our Project

1. **Start with batch size tuning** — try 32K, 16K, 8K before anything else
2. **Don't scale model up** — focus on training efficiency at 11.5M params
3. **Our Mac Studio has more RAM** — we might be able to go slightly larger, but still prioritize training steps over model size
4. **The overnight loop works** — our GitHub Actions autoresearch workflow does the same thing but with better tracking (results.tsv, PRs, commit comments)
5. **Market data is our edge** — this person got 1.4787 on text-only. If our market features push it lower, that's a novel finding nobody else has
