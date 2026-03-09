"""
train.py — AGENT-MODIFIABLE baseline.

This is the file the autoresearch agent modifies. It contains:
- Model architecture (baseline: small GPT-style Transformer)
- Optimizer configuration
- Training loop with wall-clock time budget

The agent experiments by changing architecture, hyperparameters,
and optionally incorporating market data from MarketDataLoader.

Must print exactly:
    val_bpb: <float>
    peak_mem_mb: <float>
"""

import time
import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

from prepare import (
    DataLoader,
    # MarketDataLoader,  # Uncomment to use market data
    evaluate_bpb,
    MAX_SEQ_LEN,
    TIME_BUDGET,
    VOCAB_SIZE,
)

# ---------------------------------------------------------------------------
# Hyperparameters (agent tunes these)
# ---------------------------------------------------------------------------

BATCH_SIZE = 4
SEQ_LEN = MAX_SEQ_LEN  # 512
N_LAYERS = 4
N_HEADS = 4
D_MODEL = 128
D_FF = 512
DROPOUT = 0.0
LEARNING_RATE = 3e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.1


# ---------------------------------------------------------------------------
# Model — small GPT (agent can replace this entirely)
# ---------------------------------------------------------------------------


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def __call__(self, x):
        B, T, C = x.shape
        q = (
            self.query_proj(x)
            .reshape(B, T, self.n_heads, self.d_head)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.key_proj(x)
            .reshape(B, T, self.n_heads, self.d_head)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.value_proj(x)
            .reshape(B, T, self.n_heads, self.d_head)
            .transpose(0, 2, 1, 3)
        )

        scale = math.sqrt(self.d_head)
        attn = (q @ k.transpose(0, 1, 3, 2)) / scale

        # Causal mask
        mask = mx.triu(mx.full((T, T), float("-inf")), k=1)
        attn = attn + mask

        attn = mx.softmax(attn, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def __call__(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = [
            TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ]
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def __call__(self, x):
        B, T = x.shape
        positions = mx.arange(T)

        tok = self.token_emb(x)
        pos = self.pos_emb(positions)
        h = tok + pos

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.head(h)
        return logits


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------


def get_lr(step: int) -> float:
    """Linear warmup then cosine decay."""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS

    # Estimate total steps from time budget
    # ~10 steps/sec is a rough estimate for this model size
    total_steps = TIME_BUDGET * 10
    decay_ratio = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
    decay_ratio = min(decay_ratio, 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return LEARNING_RATE * max(coeff, 0.1 * LEARNING_RATE)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train():
    print("=== Baseline Training ===")
    print(f"  d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}, d_ff={D_FF}")
    print(f"  batch_size={BATCH_SIZE}, seq_len={SEQ_LEN}, lr={LEARNING_RATE}")

    # Model
    model = GPT(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=SEQ_LEN,
    )

    n_params = sum(
        p.size for p in model.parameters().values() if isinstance(p, mx.array)
    )

    # Handle nested parameters
    def count_params(tree):
        total = 0
        if isinstance(tree, mx.array):
            return tree.size
        elif isinstance(tree, dict):
            for v in tree.values():
                total += count_params(v)
        elif isinstance(tree, list):
            for v in tree:
                total += count_params(v)
        return total

    n_params = count_params(model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Data
    train_loader = DataLoader(split="train", batch_size=BATCH_SIZE, seq_len=SEQ_LEN)

    # Optimizer
    optimizer = optim.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    # Loss function
    def loss_fn(model, inputs, targets):
        logits = model(inputs)
        loss = nn.losses.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        )
        return loss

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    # Training
    start_time = time.time()
    step = 0
    running_loss = 0.0
    log_interval = 50

    print(f"\nTraining for {TIME_BUDGET}s...")

    while True:
        elapsed = time.time() - start_time
        if elapsed >= TIME_BUDGET:
            break

        # Update learning rate
        lr = get_lr(step)
        optimizer.learning_rate = lr

        # Forward + backward
        inputs, targets = train_loader.next_batch()
        loss, grads = loss_and_grad(model, inputs, targets)
        optimizer.update(model, grads)

        # Force computation
        mx.eval(model.parameters(), optimizer.state)

        running_loss += loss.item()
        step += 1

        if step % log_interval == 0:
            avg_loss = running_loss / log_interval
            elapsed = time.time() - start_time
            steps_per_sec = step / elapsed
            print(
                f"  step {step:5d} | loss {avg_loss:.4f} | lr {lr:.2e} | "
                f"{steps_per_sec:.1f} steps/s | {elapsed:.0f}s/{TIME_BUDGET}s"
            )
            running_loss = 0.0

    train_time = time.time() - start_time
    print(f"\nTraining complete: {step} steps in {train_time:.1f}s")

    # Evaluate
    print("Evaluating...")
    val_bpb = evaluate_bpb(model, split="val", seq_len=SEQ_LEN)

    # Peak memory (Apple Silicon unified memory)
    try:
        peak_mem_mb = mx.metal.get_peak_memory() / 1024 / 1024
    except Exception:
        # Fallback if metal stats not available
        peak_mem_mb = 0.0

    # Required output format
    print(f"\nval_bpb: {val_bpb:.6f}")
    print(f"peak_mem_mb: {peak_mem_mb:.1f}")

    return val_bpb, peak_mem_mb


if __name__ == "__main__":
    train()
