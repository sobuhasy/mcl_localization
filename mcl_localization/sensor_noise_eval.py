import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

files = sorted(Path('.').glob('sensor_noise_*.csv'))

rows = []
for file in files:
    df = pd.read_csv(file)
    if len(df) == 0:
        continue

    rmse = df["rmse_xy"].iloc[-1]
    noise = f.stem.replace('sensor_noise_', '').replace('_', ' ')
    rows.append((float(noise), rmse))

    rows.sort()

    noise, rmse = zip(*rows)
    plt.plot(noise, rmse, marker='o', label='Sensor Noise Evaluation')
plt.xlabel('Sensor Noise Standard Deviation')
plt.ylabel('RMSE (m)')
plt.title('Sensor Noise vs RMSE')
plt.grid()
plt.legend()
plt.savefig('sensor_noise_eval.png')
plt.show()
    

