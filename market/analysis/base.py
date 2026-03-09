"""
market/analysis/base.py — Analysis base class.

Provides run(), save(), and progress() context manager for
standalone analysis scripts.
"""

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from pathlib import Path

from market.config import OUTPUT_DIR


class Analysis(ABC):
    """Base class for analysis scripts."""

    name: str = "unnamed"

    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def run(self) -> None:
        """Execute the analysis."""
        ...

    def save(self, fig, filename: str) -> Path:
        """Save a matplotlib figure to output_dir."""
        path = self.output_dir / filename
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        return path

    @contextmanager
    def progress(self, description: str = ""):
        """Context manager that prints timing info."""
        desc = description or self.name
        print(f"Running: {desc}...")
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            print(f"  Done in {elapsed:.1f}s")
