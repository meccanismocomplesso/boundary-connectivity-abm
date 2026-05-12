import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# INPUT (usa output già generato)
# ============================================================

INPUT_FILE = "paper_submission_outputs/data/full_model_aggregated.csv"

OUTPUT_DIR = "supplementary_slope_outputs"
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

Path(FIG_DIR).mkdir(parents=True, exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_FILE)

# Ordina per p
df = df.sort_values("exploratory_fraction")

p = df["exploratory_fraction"].to_numpy()
C = df["giant_weak_component"].to_numpy()

# ============================================================
# NUMERICAL DERIVATIVE dC/dp
# ============================================================

# derivata numerica (centrale)
dCdp = np.gradient(C, p)

# trova massimo (punto di transizione)
idx_max = np.argmax(dCdp)
p_star = p[idx_max]
slope_max = dCdp[idx_max]

print(f"Peak transition at p ≈ {p_star:.4f}")
print(f"Max slope dC/dp ≈ {slope_max:.4f}")

# ============================================================
# PLOT
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: giant component
axes[0].plot(p, C, marker="o")
axes[0].axvline(p_star, linestyle="--", alpha=0.6)
axes[0].set_title("Giant weak component")
axes[0].set_xlabel("Exploratory fraction")
axes[0].set_ylabel("C_w")
axes[0].grid(True, alpha=0.3)

# Panel B: slope
axes[1].plot(p, dCdp, marker="o")
axes[1].axvline(p_star, linestyle="--", alpha=0.6, label=f"p* ≈ {p_star:.3f}")
axes[1].set_title("Transition slope dC/dp")
axes[1].set_xlabel("Exploratory fraction")
axes[1].set_ylabel("dC/dp")
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "Supplementary_Figure_S3_transition_slope.png"), dpi=300, bbox_inches="tight")
plt.close()

print("Done.")


