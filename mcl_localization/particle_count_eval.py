import glob
import math
import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

def compute_rmse_xy(df: pd.DataFrame) -> float:
    return math.sqrt((df["e_xy"] ** 2).mean())

def compute_convergence_time(df: pd.DataFrame, threshold: float = 0.25, hold_s: float = 2.0) -> float:
    """
    Converged = e_xy < thresh_xy continuously for hold_s seconds.
    Returns diest time when that continous window starts, otherwise NaN.
    """
    if len(df) < 2:
        return None
    
    start_t = None
    for ti, ei in zip(t, e):
        if ei < thresh_xy:
            if start_t is None:
                start_t = ti
            elif ti - start_t >= hold_s:
                return start_t
        else:
            start_t = None
    return None


def main():
    report_dir = Path("reports")
    paths = sorted(glob.glob(str(report_dir / "mcl_localization_particle_count_*.csv")))
    if not paths:
        raise SystemExit("No files found: reports/mcl_localization_particle_count_*.csv")
    
    rows = []
    for p in paths:
        m = re.search(r"mcl_localization_particle_count_(\d+).csv", Path(p).stem)
        if not m:
            continue
        particle_count = int(m.group(1))
        df = pd.read_csv(p)
        conv = compute_convergence_time(df, threshold=0.25, hold_s=2.0)

        rows.append({
            "N": particle_count,
            "rmse_xy": compute_rmse_xy(df),
            "convergence_time_s": conv if conv is not None else float('nan'),
            "samples": len(df),
            "file": p,
        })

        out = pd.DataFrame(rows).sort_values("N")
        out_path = report_dir / "mcl_localization_particle_count_eval.csv"
        out.to_csv(out_path, index=False)
        print(f"Wrote evaluation to: {out_path}")
        print(out)

        # Plot RMSE vs Particle Count
        plt.figure()
        plt.plot(out["N"], out["rmse_xy"], marker='o')
        plt.xscale('log')
        plt.xlabel('Particle Count (N)')
        plt.ylabel('RMSE XY (m)')
        plt.title('RMSE XY vs Particle Count')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()