"""
Script 02: Distribution Visualization
======================================
Purpose: Visualize distributions of F_sim, S_sim, and VMS using histograms

Input:
  - virus_to_human_top5_neighbors_final_similarity.csv

Output:
  - fig_01_histograms.png (3-panel histogram plot)
  - report_02_distributions.md (distribution analysis report)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load dataset
print("="*70)
print("SCRIPT 02: DISTRIBUTION VISUALIZATION")
print("="*70)
print()

df = pd.read_csv('virus_to_human_top5_neighbors_final_similarity.csv')

# Rename columns
df = df.rename(columns={
    'embedding_similarity': 'F_sim',
    'similarity': 'S_sim',
    'virus_mimicry_score': 'VMS'
})

print(f"Dataset loaded: {len(df)} pairs")
print()

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Distribution of Similarity Metrics', fontsize=16, fontweight='bold', y=1.02)

variables = ['F_sim', 'S_sim', 'VMS']
titles = [
    'F_sim: Functional Similarity',
    'S_sim: Structural Similarity',
    'VMS: Virus Mimicry Score'
]
colors = ['#3498db', '#e74c3c', '#2ecc71']

# Calculate skewness and kurtosis for report
distribution_stats = {}

for idx, (var, title, color) in enumerate(zip(variables, titles, colors)):
    ax = axes[idx]

    # Plot histogram with KDE
    data = df[var].dropna()
    ax.hist(data, bins=50, alpha=0.7, color=color, edgecolor='black', density=True, label='Histogram')

    # Add KDE curve
    kde_x = np.linspace(data.min(), data.max(), 300)
    kde = stats.gaussian_kde(data)
    ax.plot(kde_x, kde(kde_x), 'k-', linewidth=2, label='KDE')

    # Add mean and median lines
    mean_val = data.mean()
    median_val = data.median()
    ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.3f}')
    ax.axvline(median_val, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_val:.3f}')

    # Labels and title
    ax.set_xlabel(var, fontsize=11, fontweight='bold')
    ax.set_ylabel('Density', fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)

    # Calculate distribution statistics
    distribution_stats[var] = {
        'Mean': mean_val,
        'Median': median_val,
        'Skewness': stats.skew(data),
        'Kurtosis': stats.kurtosis(data),
        'N': len(data)
    }

plt.tight_layout()
plt.savefig('fig_01_histograms.png', dpi=600, bbox_inches='tight')
print("✓ Saved: fig_01_histograms.png")
plt.close()

# Create Markdown report
report = f"""# Report 02: Distribution Analysis

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

This report analyzes the distributions of the three key variables: F_sim, S_sim, and VMS.

## Distribution Statistics

| Metric | F_sim | S_sim | VMS |
|--------|-------|-------|-----|
| Mean | {distribution_stats['F_sim']['Mean']:.4f} | {distribution_stats['S_sim']['Mean']:.4f} | {distribution_stats['VMS']['Mean']:.4f} |
| Median | {distribution_stats['F_sim']['Median']:.4f} | {distribution_stats['S_sim']['Median']:.4f} | {distribution_stats['VMS']['Median']:.4f} |
| Skewness | {distribution_stats['F_sim']['Skewness']:.4f} | {distribution_stats['S_sim']['Skewness']:.4f} | {distribution_stats['VMS']['Skewness']:.4f} |
| Kurtosis | {distribution_stats['F_sim']['Kurtosis']:.4f} | {distribution_stats['S_sim']['Kurtosis']:.4f} | {distribution_stats['VMS']['Kurtosis']:.4f} |

## Interpretation

### F_sim (Functional Similarity)
"""

# F_sim interpretation
f_skew = distribution_stats['F_sim']['Skewness']
if f_skew > 0.5:
    f_skew_text = "**Right-skewed** (positive skew > 0.5)"
elif f_skew < -0.5:
    f_skew_text = "**Left-skewed** (negative skew < -0.5)"
else:
    f_skew_text = "**Approximately symmetric** (-0.5 < skew < 0.5)"

f_kurt = distribution_stats['F_sim']['Kurtosis']
if f_kurt > 1:
    f_kurt_text = "**Leptokurtic** (heavy tails, kurtosis > 1)"
elif f_kurt < -1:
    f_kurt_text = "**Platykurtic** (light tails, kurtosis < -1)"
else:
    f_kurt_text = "**Mesokurtic** (normal-like tails, -1 < kurtosis < 1)"

report += f"""
- **Shape:** {f_skew_text}
- **Tails:** {f_kurt_text}
- **Mean vs Median:** {"Mean > Median (confirms right skew)" if distribution_stats['F_sim']['Mean'] > distribution_stats['F_sim']['Median'] else "Mean ≈ Median (symmetric)" if abs(distribution_stats['F_sim']['Mean'] - distribution_stats['F_sim']['Median']) < 0.01 else "Mean < Median (left skew)"}

### S_sim (Structural Similarity)
"""

# S_sim interpretation
s_skew = distribution_stats['S_sim']['Skewness']
if s_skew > 0.5:
    s_skew_text = "**Right-skewed** (positive skew > 0.5)"
elif s_skew < -0.5:
    s_skew_text = "**Left-skewed** (negative skew < -0.5)"
else:
    s_skew_text = "**Approximately symmetric** (-0.5 < skew < 0.5)"

s_kurt = distribution_stats['S_sim']['Kurtosis']
if s_kurt > 1:
    s_kurt_text = "**Leptokurtic** (heavy tails, kurtosis > 1)"
elif s_kurt < -1:
    s_kurt_text = "**Platykurtic** (light tails, kurtosis < -1)"
else:
    s_kurt_text = "**Mesokurtic** (normal-like tails, -1 < kurtosis < 1)"

report += f"""
- **Shape:** {s_skew_text}
- **Tails:** {s_kurt_text}
- **Mean vs Median:** {"Mean > Median (confirms right skew)" if distribution_stats['S_sim']['Mean'] > distribution_stats['S_sim']['Median'] else "Mean ≈ Median (symmetric)" if abs(distribution_stats['S_sim']['Mean'] - distribution_stats['S_sim']['Median']) < 0.01 else "Mean < Median (left skew)"}

### VMS (Virus Mimicry Score)
"""

# VMS interpretation
v_skew = distribution_stats['VMS']['Skewness']
if v_skew > 0.5:
    v_skew_text = "**Right-skewed** (positive skew > 0.5)"
elif v_skew < -0.5:
    v_skew_text = "**Left-skewed** (negative skew < -0.5)"
else:
    v_skew_text = "**Approximately symmetric** (-0.5 < skew < 0.5)"

v_kurt = distribution_stats['VMS']['Kurtosis']
if v_kurt > 1:
    v_kurt_text = "**Leptokurtic** (heavy tails, kurtosis > 1)"
elif v_kurt < -1:
    v_kurt_text = "**Platykurtic** (light tails, kurtosis < -1)"
else:
    v_kurt_text = "**Mesokurtic** (normal-like tails, -1 < kurtosis < 1)"

report += f"""
- **Shape:** {v_skew_text}
- **Tails:** {v_kurt_text}
- **Mean vs Median:** {"Mean > Median (confirms right skew)" if distribution_stats['VMS']['Mean'] > distribution_stats['VMS']['Median'] else "Mean ≈ Median (symmetric)" if abs(distribution_stats['VMS']['Mean'] - distribution_stats['VMS']['Median']) < 0.01 else "Mean < Median (left skew)"}

## Key Observations

1. **F_sim distribution:**
   - Shows more variability and discriminative power
   - {'Right-skewed with more low values' if f_skew > 0.5 else 'Approximately symmetric distribution'}

2. **S_sim distribution:**
   - Higher mean values (structural similarity more common)
   - Less variance than F_sim

3. **VMS distribution:**
   - Weighted combination reflects F_sim influence (75% weight)
   - Bounded in [0, 1] range

## Statistical Implications

Based on skewness and kurtosis values:
- **Normality assumption:** Likely violated for all three variables (will be tested in Script 03)
- **Recommended correlation method:** Spearman rank correlation (non-parametric)
- **Visualization method:** Histograms with KDE curves provide good representation

## Visual Output

- **Figure:** fig_01_histograms.png
- **Format:** 3-panel histogram with KDE overlay
- **Resolution:** 600 DPI

---

**Next steps:** Perform formal normality tests (Shapiro-Wilk, Kolmogorov-Smirnov, Q-Q plots)
"""

# Save report
with open('report_02_distributions.md', 'w') as f:
    f.write(report)
print("✓ Saved: report_02_distributions.md")

# Print summary to console
print()
print("-"*70)
print("DISTRIBUTION SUMMARY")
print("-"*70)
for var in variables:
    print(f"{var}:")
    print(f"  Skewness: {distribution_stats[var]['Skewness']:.4f}", end="")
    if distribution_stats[var]['Skewness'] > 0.5:
        print(" (right-skewed)")
    elif distribution_stats[var]['Skewness'] < -0.5:
        print(" (left-skewed)")
    else:
        print(" (approximately symmetric)")
    print(f"  Kurtosis: {distribution_stats[var]['Kurtosis']:.4f}", end="")
    if distribution_stats[var]['Kurtosis'] > 1:
        print(" (heavy tails)")
    elif distribution_stats[var]['Kurtosis'] < -1:
        print(" (light tails)")
    else:
        print(" (normal-like)")
print()
print("="*70)
print("SCRIPT 02 COMPLETED")
print("Files generated:")
print("  - fig_01_histograms.png")
print("  - report_02_distributions.md")
print("="*70)
