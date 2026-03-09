"""
scripts/run_analysis.py — Interactive analysis menu.

Lists available analysis scripts and lets you pick one to run.
Results are saved to the output/ directory.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from market.analysis.scripts import ANALYSES


def main():
    print("\n=== prediction-autoresearch — Analysis Menu ===\n")

    if not ANALYSES:
        print("No analysis scripts registered.")
        return

    names = sorted(ANALYSES.keys())
    for i, name in enumerate(names, 1):
        cls = ANALYSES[name]
        print(f"  {i}. {name} — {cls.__doc__.strip().splitlines()[0] if cls.__doc__ else 'No description'}")

    print(f"\n  0. Run all")
    print()

    try:
        choice = input("Pick an analysis (number): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nBye.")
        return

    if choice == "0":
        for name in names:
            ANALYSES[name]().run()
    elif choice.isdigit() and 1 <= int(choice) <= len(names):
        name = names[int(choice) - 1]
        ANALYSES[name]().run()
    else:
        print(f"Invalid choice: {choice}")
        return

    print("\nDone. Check output/ for results.")


if __name__ == "__main__":
    main()
