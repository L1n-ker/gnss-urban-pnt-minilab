"""Run the complete reproducible MiniLab experiment pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.make_summary_tables import make_all_summary_tables
from experiments.plot_all_results import plot_all_results
from reproduction.run_reproduction import run_reproduction


def run_all_experiments() -> list[Path]:
    """Generate all CSV tables, all core figures, and reproduction artifacts."""

    generated: list[Path] = []
    generated.extend(make_all_summary_tables())
    generated.extend(plot_all_results())
    generated.extend(run_reproduction().values())
    return generated


def main() -> None:
    for path in run_all_experiments():
        print(path)


if __name__ == "__main__":
    main()
