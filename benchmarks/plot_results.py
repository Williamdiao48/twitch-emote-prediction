"""Plot per-clip processing time by VOD position from results/clips.csv.

Usage: python benchmarks/plot_results.py
Writes assets/benchmark_per_clip_time.png.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CLIPS_CSV = ROOT / "benchmarks" / "results" / "clips.csv"
OUT_PNG = ROOT / "assets" / "benchmark_per_clip_time.png"

with open(CLIPS_CSV, newline="", encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f) if r["skipped"] != "True"]

pos_h = np.array([float(r["start_time_h"]) for r in rows])
old_total = np.array([float(r["old_total_s"]) for r in rows])
new_total = np.array([float(r["new_total_s"]) for r in rows])

fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
ax.scatter(pos_h, old_total, s=9, alpha=0.55, color="#d62728", label="Original pipeline")
ax.scatter(pos_h, new_total, s=9, alpha=0.55, color="#1f77b4", label="Current pipeline")

# Straight-line fits make the trend (or lack of one) explicit.
xs = np.linspace(pos_h.min(), pos_h.max(), 100)
for y, color in [(old_total, "#d62728"), (new_total, "#1f77b4")]:
    slope, intercept = np.polyfit(pos_h, y, 1)
    ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.5)

ax.set_xlabel("Clip position in VOD (hours)")
ax.set_ylabel("Processing time per clip (s)")
ax.set_title(f"Per-clip processing time on {len(rows)} identical clips")
ax.set_ylim(bottom=0)
ax.grid(alpha=0.3)
ax.legend(loc="upper left")
fig.tight_layout()
OUT_PNG.parent.mkdir(exist_ok=True)
fig.savefig(OUT_PNG)
print(f"Wrote {OUT_PNG}")
