#!/usr/bin/env python
"""Collect per-dataset metrics.json files into result tables.

Usage:
    python scripts/eval/summarize.py [--results-dir ./results]

Reads  <results-dir>/states_<S>/<checkpoint>/<dataset>_eval/metrics.json
and prints the tables to stdout.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Display order and names of the benchmarks.
DATASETS = [
    ("partnet_mobility", "PartNet-Mobility"),
    ("acd", "ACD"),
    ("articraft", "ArtiCraft"),
]


def collect(results_dir: Path) -> dict[int, dict[str, dict]]:
    """Map number of input states -> dataset name -> metrics payload."""
    by_states: dict[int, dict[str, dict]] = {}
    for path in sorted(results_dir.glob("states_*/*/*_eval/metrics.json")):
        payload = json.loads(path.read_text())
        states = int(payload.get("num_input_states") or 0)
        by_states.setdefault(states, {})[payload["dataset"]] = payload
    return by_states


def _pct(value) -> str:
    return "-" if value is None else f"{100 * value:.1f}"


def render(by_states: dict[int, dict[str, dict]]) -> str:
    lines: list[str] = []
    for states in sorted(by_states, reverse=True):
        found = by_states[states]
        lines.append(f"## S={states}\n")
        lines.append(
            "| Benchmark | Objects | Seg P | Seg R | Seg F1 | # | M | MA | MAO | AE | OE |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for key, label in DATASETS:
            entry = found.get(key)
            if entry is None:
                continue
            seg, motion = entry["segmentation"], entry["motion"]
            axis, origin = motion["axis_error_deg"], motion["origin_error"]
            lines.append(
                f"| {label} | {entry['num_objects']} "
                f"| {_pct(seg['precision'])} | {_pct(seg['recall'])} | {_pct(seg['f1'])} "
                f"| {motion['matched_parts']} "
                f"| {_pct(motion['f1_M'])} | {_pct(motion['f1_MA'])} | {_pct(motion['f1_MAO'])} "
                f"| {'-' if axis is None else f'{axis:.1f}'} "
                f"| {'-' if origin is None else f'{origin:.3f}'} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("./results"))
    args = parser.parse_args()

    by_states = collect(args.results_dir)
    if not by_states:
        raise SystemExit(f"No metrics.json found under {args.results_dir}")

    print(render(by_states))


if __name__ == "__main__":
    main()
