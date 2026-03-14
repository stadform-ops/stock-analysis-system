"""Unified experiment entrypoint for the research workspace."""

from __future__ import annotations

import importlib
from pathlib import Path

import yaml


def load_experiment_targets(config_path: str | Path) -> list[str]:
    path = Path(config_path)
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data.get("run_targets", [])


def run(config_path: str | Path = "stock_prediction_research/configs/experiment_config.yaml") -> None:
    targets = load_experiment_targets(config_path)
    if not targets:
        print("No experiments configured in run_targets.")
        return

    for target in targets:
        module_name = f"stock_prediction_research.experiments.{target}"
        print(f"[Runner] import -> {module_name}")
        importlib.import_module(module_name)


if __name__ == "__main__":
    run()
