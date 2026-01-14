# Task B1 - Particle Count Variation Instructions

Use this guide to run five simulations with different particle counts and generate the RMSE vs. particle count plot.

## 1) Run simulations and log evaluation CSVs

For each particle count, launch the particle filter and evaluator with a distinct CSV output path.

Example (repeat for each particle count):

```bash
ros2 run mcl_localization mcl_localization_pf --ros-args -p num_particles:=100
ros2 run mcl_localization pf_eval --ros-args -p output_csv:=reports/run_100.csv \
  -p ground_truth_topic:=/ground_truth_pose -p estimated_pose_topic:=/mcl_pose
```

Suggested particle counts: **100, 500, 1000, 2000, 5000**.

## 2) Generate the summary and plot

After collecting the CSVs, run:

```bash
python3 scripts/particle_count_eval.py \
  --run 100:reports/run_100.csv \
  --run 500:reports/run_500.csv \
  --run 1000:reports/run_1000.csv \
  --run 2000:reports/run_2000.csv \
  --run 5000:reports/run_5000.csv
```

Outputs:

- `reports/task_b1_summary.csv` (numeric summary)
- `reports/task_b1_particle_count.md` (markdown report with the optimal count)
- `reports/task_b1_rmse_vs_particles.png` (plot)

## 3) Document the optimal particle count

Open `reports/task_b1_particle_count.md`, which includes:

- RMSE and convergence time per particle count
- The computed optimal particle count
- Notes describing expected performance trends

Update the behavioral notes if your observed trends differ from the defaults.
