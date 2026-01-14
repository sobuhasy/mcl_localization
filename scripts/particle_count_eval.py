#!/usr/bin/env python3
import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


@dataclass
class RunSummary:
    particle_count: int
    rmse_xy: float
    convergence_time_s: Optional[float]


def parse_run_arg(value: str) -> Tuple[int, Path]:
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            "Run must be formatted as <particle_count>:<csv_path>."
        )
    count_str, path_str = value.split(":", 1)
    try:
        count = int(count_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Particle count must be an integer.") from exc
    path = Path(path_str)
    return count, path


def load_errors(csv_path: Path) -> Tuple[List[float], List[float]]:
    times: List[float] = []
    errors_xy: List[float] = []
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if not row:
                continue
            time_val = float(row["time"])
            error_x = float(row["error_x"])
            error_y = float(row["error_y"])
            times.append(time_val)
            errors_xy.append(math.hypot(error_x, error_y))
    return times, errors_xy


def load_rmse(csv_path: Path) -> float:
    rmse = None
    with csv_path.open(newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row:
                rmse = float(row["rmse_xy"])
    if rmse is None:
        raise ValueError(f"No data found in {csv_path}")
    return rmse


def compute_convergence_time(
    times: List[float],
    errors_xy: List[float],
    threshold: float,
    hold_time: float,
) -> Optional[float]:
    if not times:
        return None
    start_time = None
    t0 = times[0]
    for t, e_xy in zip(times, errors_xy):
        if e_xy < threshold:
            if start_time is None:
                start_time = t
            if t - start_time >= hold_time:
                return t - t0
        else:
            start_time = None
    return None


def summarize_runs(
    runs: Iterable[Tuple[int, Path]],
    threshold: float,
    hold_time: float,
) -> List[RunSummary]:
    summaries: List[RunSummary] = []
    for count, path in runs:
        times, errors_xy = load_errors(path)
        rmse_xy = load_rmse(path)
        convergence_time = compute_convergence_time(times, errors_xy, threshold, hold_time)
        summaries.append(
            RunSummary(
                particle_count=count,
                rmse_xy=rmse_xy,
                convergence_time_s=convergence_time,
            )
        )
    return sorted(summaries, key=lambda item: item.particle_count)


def select_optimal(summaries: List[RunSummary]) -> Optional[RunSummary]:
    if not summaries:
        return None

    def sort_key(item: RunSummary) -> Tuple[float, float]:
        conv = item.convergence_time_s
        return (item.rmse_xy, conv if conv is not None else float("inf"))

    return min(summaries, key=sort_key)


def write_summary_csv(summaries: List[RunSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["particle_count", "rmse_xy", "convergence_time_s"])
        for summary in summaries:
            writer.writerow(
                [
                    summary.particle_count,
                    f"{summary.rmse_xy:.6f}",
                    "" if summary.convergence_time_s is None else f"{summary.convergence_time_s:.3f}",
                ]
            )


def write_summary_md(
    summaries: List[RunSummary],
    optimal: Optional[RunSummary],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Task B1 - Particle Count Variation",
        "",
        "## Summary",
        "",
        "| Particle Count | RMSE (m) | Convergence Time (s) |",
        "| --- | --- | --- |",
    ]
    for summary in summaries:
        conv = (
            "—" if summary.convergence_time_s is None else f"{summary.convergence_time_s:.2f}"
        )
        lines.append(f"| {summary.particle_count} | {summary.rmse_xy:.4f} | {conv} |")
    lines.append("")
    if optimal is None:
        lines.append("**Optimal particle count:** _not computed_ (no data provided).")
    else:
        conv_str = (
            "not converged"
            if optimal.convergence_time_s is None
            else f"{optimal.convergence_time_s:.2f}s"
        )
        lines.append(
            f"**Optimal particle count:** {optimal.particle_count} (RMSE {optimal.rmse_xy:.4f}, "
            f"convergence {conv_str})."
        )
    lines.extend(
        [
            "",
            "## Behavior Notes",
            "",
            "* Increasing particle count generally lowers RMSE but with diminishing returns once the "
            "measurement update dominates the estimate.",
            "* Convergence time often improves with more particles until the runtime cost slows each "
            "update; this is where the RMSE curve typically bends (the elbow).",
            "* Choose the smallest particle count near the elbow that still yields stable convergence.",
        ]
    )
    output_path.write_text("\n".join(lines))


def plot_rmse_vs_particles(summaries: List[RunSummary], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = [s.particle_count for s in summaries]
    rmses = [s.rmse_xy for s in summaries]

    plt.figure(figsize=(6, 4))
    plt.plot(counts, rmses, marker="o")
    plt.xlabel("Particle Count")
    plt.ylabel("RMSE (m)")
    plt.title("RMSE vs Particle Count")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize particle filter runs and plot RMSE vs particle count."
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_arg,
        required=True,
        help="Run format: <particle_count>:<csv_path>",
    )
    parser.add_argument(
        "--conv-threshold",
        type=float,
        default=0.25,
        help="Convergence threshold for position error (meters).",
    )
    parser.add_argument(
        "--conv-hold-time",
        type=float,
        default=2.0,
        help="Minimum time the error must stay under threshold to converge (seconds).",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("reports/task_b1_summary.csv"),
        help="Path to write the summary CSV.",
    )
    parser.add_argument(
        "--summary-md",
        type=Path,
        default=Path("reports/task_b1_particle_count.md"),
        help="Path to write the summary Markdown report.",
    )
    parser.add_argument(
        "--plot-path",
        type=Path,
        default=Path("reports/task_b1_rmse_vs_particles.png"),
        help="Path to write the RMSE vs particle count plot.",
    )

    args = parser.parse_args()
    summaries = summarize_runs(args.run, args.conv_threshold, args.conv_hold_time)
    optimal = select_optimal(summaries)

    write_summary_csv(summaries, args.summary_csv)
    write_summary_md(summaries, optimal, args.summary_md)
    plot_rmse_vs_particles(summaries, args.plot_path)


if __name__ == "__main__":
    main()
