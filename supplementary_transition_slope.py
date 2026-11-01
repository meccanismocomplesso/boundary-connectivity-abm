from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
INPUT_FILE = Path('paper_submission_outputs/data/full_model_aggregated.csv')
OUTPUT_DIR = Path('supplementary_transition_slope_outputs')
DATA_DIR = OUTPUT_DIR / 'data'
FIG_DIR = OUTPUT_DIR / 'figures'
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def main():
    df = pd.read_csv(INPUT_FILE).sort_values('exploratory_fraction').copy()
    required = {'exploratory_fraction', 'giant_weak_component'}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f'Missing required columns in {INPUT_FILE}: {sorted(missing)}')
    p = df['exploratory_fraction'].to_numpy(dtype=float)
    connectivity = df['giant_weak_component'].to_numpy(dtype=float)
    if len(p) < 3:
        raise ValueError('At least three p values are required for the numerical derivative.')
    if np.any(np.diff(p) <= 0):
        raise ValueError('exploratory_fraction values must be strictly increasing.')
    slope = np.gradient(connectivity, p)
    idx_max = int(np.argmax(slope))
    p_star = float(p[idx_max])
    slope_max = float(slope[idx_max])
    result = pd.DataFrame({'exploratory_fraction': p, 'giant_weak_component': connectivity, 'dCw_dp': slope})
    result.to_csv(DATA_DIR / 'transition_slope.csv', index=False)
    print(f'Maximum observed connectivity-growth rate at p ≈ {p_star:.4f}')
    print(f'Maximum numerical slope dCw/dp ≈ {slope_max:.4f}')
    print('Interpret p* as a finite-grid diagnostic, not as a critical point estimate.')
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(p, connectivity, marker='o')
    axes[0].axvline(p_star, linestyle='--', alpha=0.6)
    axes[0].set_title('Giant weak component')
    axes[0].set_xlabel('Exploratory fraction')
    axes[0].set_ylabel('$C_w$')
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(p, slope, marker='o')
    axes[1].axvline(p_star, linestyle='--', alpha=0.6, label=f'max slope at p ≈ {p_star:.3f}')
    axes[1].set_title('Connectivity-growth rate $dC_w/dp$')
    axes[1].set_xlabel('Exploratory fraction')
    axes[1].set_ylabel('$dC_w/dp$')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'Supplementary_Figure_transition_slope.png', dpi=300, bbox_inches='tight')
    plt.close()
if __name__ == '__main__':
    main()
