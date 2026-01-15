import glob
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def compute_rmse_xy(df: pd.DataFrame) -> float:
    return math.sqrt((df["e_xy"] ** 2).mean())


def compute_convergence_time(
    df: pd.DataFrame,
    threshold: float = 0.25,
    hold_s: float = 2.0,
) -> float | None:
    """
    Converged = e_xy < threshold continuously for hold_s seconds.
    Returns the first time when that continuous window starts, otherwise None.
    """
    if len(df) < 2:
        return None

    t = df["t"].to_numpy()
    e = df["e_xy"].to_numpy()

    start_t = None
    for ti, ei in zip(t, e):
        if ei < threshold:
            if start_t is None:
                start_t = ti
            elif ti - start_t >= hold_s:
                return float(start_t)
        else:
            start_t = None
    return None


def main() -> None:
    report_dir = Path("reports")
    paths = sorted(glob.glob(str(report_dir / "mcl_localization_particle_count_*.csv")))
    if not paths:
        raise SystemExit("No files found: reports/mcl_localization_particle_count_*.csv")

    rows: list[dict[str, object]] = []
    for p in paths:
        match = re.search(r"mcl_localization_particle_count_(\d+)", Path(p).stem)
        if not match:
            continue
        particle_count = int(match.group(1))
        df = pd.read_csv(p)
        conv = compute_convergence_time(df, threshold=0.25, hold_s=2.0)

        rows.append(
            {
                "N": particle_count,
                "rmse_xy": compute_rmse_xy(df),
                "convergence_time_s": conv if conv is not None else float("nan"),
                "samples": len(df),
                "file": p,
            }
        )

    out = pd.DataFrame(rows).sort_values("N")
    out_path = report_dir / "mcl_localization_particle_count_eval.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote evaluation to: {out_path}")
    print(out)

    # Plot RMSE vs Particle Count
    plt.figure()
    plt.plot(out["N"], out["rmse_xy"], marker="o")
    plt.xscale("log")
    plt.xlabel("Particle Count (N)")
    plt.ylabel("RMSE XY (m)")
    plt.title("RMSE XY vs Particle Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()