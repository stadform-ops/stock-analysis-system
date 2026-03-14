"""Unified experiment entrypoint for the research workspace."""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
from typing import Any
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Load a minimal subset of YAML used by this project without external deps."""
    data: dict[str, Any] = {}
    current_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if line.startswith("  - ") and current_key:
            data.setdefault(current_key, []).append(line.replace("  - ", "", 1).strip())
            continue

        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            if value == "":
                data[key] = []
                current_key = key
            else:
                data[key] = value
                current_key = None

    return data


def load_config(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    text = path.read_text(encoding="utf-8")

    try:
        import yaml  # type: ignore

        return yaml.safe_load(text) or {}
    except ModuleNotFoundError:
        return _simple_yaml_load(text)


def load_experiment_targets(config_path: str | Path) -> list[str]:
    data = load_config(config_path)
    targets = data.get("run_targets", [])
    return targets if isinstance(targets, list) else []


def run(
    config_path: str | Path = "stock_prediction_research/configs/experiment_config.yaml",
    dry_run: bool = False,
) -> int:
    targets = load_experiment_targets(config_path)
    if not targets:
        print("No experiments configured in run_targets.")
        return 1

    for target in targets:
        module_name = f"stock_prediction_research.experiments.{target}"
        print(f"[Runner] import -> {module_name}")
        importlib.import_module(module_name)

    if dry_run:
        print(f"[Runner] dry-run ok: validated {len(targets)} experiment modules")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run configured stock research experiments")
    parser.add_argument(
        "--config",
        default="stock_prediction_research/configs/experiment_config.yaml",
        help="Path to experiment config file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate imports only",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(run(config_path=args.config, dry_run=args.dry_run))
