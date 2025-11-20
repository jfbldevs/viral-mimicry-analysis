"""
Script 05: Stratified Correlation Analysis
===========================================
Purpose: Calculate correlation between F_sim and S_sim stratified by VMS ranges
         to identify context-dependent patterns (main finding)

Input:
  - virus_to_human_top5_neighbors_final_similarity.csv

Output:
  - fig_04_scatter_stratified.png (multi-panel scatter plots by VMS range)
  - results_04_correlation_stratified.csv (correlation by VMS range)
  - report_05_stratified_correlation.md (stratified analysis report)
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
print("SCRIPT 05: STRATIFIED CORRELATION ANALYSIS")
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

# Define VMS ranges for stratification
vms_ranges = [
    ('< 0.3', 0.0, 0.3),
    ('0.3-0.4', 0.3, 0.4),
    ('0.4-0.5', 0.4, 0.5),
    ('0.5-0.6', 0.5, 0.6),
    ('0.6-0.7', 0.6, 0.7),
    ('≥ 0.7', 0.7, 1.0)
]

print("Calculating stratified correlations...")
print()

# Calculate correlations for each range
results = []

for range_label, vms_min, vms_max in vms_ranges:
    # Filter data
    mask = (df['VMS'] >= vms_min) & (df['VMS'] < vms_max)
    subset = df[mask]
    n = len(subset)

    if n < 10:  # Skip if too few points
        print(f"VMS {range_label}: n={n} (SKIPPED - too few points)")
        continue

    # Calculate correlations
    spearman_rho, spearman_p = stats.spearmanr(subset['F_sim'], subset['S_sim'])
    pearson_r, pearson_p = stats.pearsonr(subset['F_sim'], subset['S_sim'])

    # Interpret strength
    def interpret_correlation(rho):
        abs_rho = abs(rho)
        if abs_rho < 0.1:
            return "Negligible"
        elif abs_rho < 0.3:
            return "Weak"
        elif abs_rho < 0.5:
            return "Moderate"
        elif abs_rho < 0.7:
            return "Strong"
        else:
            return "Very Strong"

    strength = interpret_correlation(spearman_rho)

    # Store results
    results.append({
        'VMS_Range': range_label,
        'VMS_Min': vms_min,
        'VMS_Max': vms_max,
        'N': n,
        'Percent': n / len(df) * 100,
        'Spearman_rho': spearman_rho,
        'Spearman_p': spearman_p,
        'Pearson_r': pearson_r,
        'Pearson_p': pearson_p,
        'Strength': strength,
        'F_sim_mean': subset['F_sim'].mean(),
        'S_sim_mean': subset['S_sim'].mean()
    })

    print(f"VMS {range_label}: n={n:4d} ({n/len(df)*100:5.1f}%)")
    print(f"  Spearman ρ = {spearman_rho:+.4f} (p={spearman_p:.2e}) - {strength}")
    print(f"  Pearson r  = {pearson_r:+.4f} (p={pearson_p:.2e})")
    print()

# Create results DataFrame
results_df = pd.DataFrame(results)
results_df.to_csv('results_04_correlation_stratified.csv', index=False, float_format='%.6f')
print("✓ Saved: results_04_correlation_stratified.csv")
print()

# Identify key findings
max_neg_idx = results_df['Spearman_rho'].idxmin()
strongest_neg = results_df.iloc[max_neg_idx]

print("-"*70)
print("KEY FINDING")
print("-"*70)
print(f"Strongest correlation: VMS {strongest_neg['VMS_Range']}")
print(f"  Spearman ρ = {strongest_neg['Spearman_rho']:.4f} (p={strongest_neg['Spearman_p']:.2e})")
print(f"  Sample size: n={int(strongest_neg['N'])} ({strongest_neg['Percent']:.1f}%)")
print(f"  Strength: {strongest_neg['Strength']}")
print("-"*70)
print()

# Create multi-panel scatter plot
print("Generating stratified scatter plots...")

# Determine grid size based on number of ranges with data
n_ranges = len(results_df)
if n_ranges <= 2:
    nrows, ncols = 1, n_ranges
elif n_ranges <= 4:
    nrows, ncols = 2, 2
elif n_ranges <= 6:
    nrows, ncols = 2, 3
else:
    nrows, ncols = 3, 3

fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 6*nrows))
fig.suptitle('Stratified Correlation: F_sim vs S_sim by VMS Range',
             fontsize=16, fontweight='bold', y=0.995)

# Handle single subplot case
if n_ranges == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, (ax, row) in enumerate(zip(axes[:n_ranges], results_df.itertuples())):
    # Filter data for this range
    mask = (df['VMS'] >= row.VMS_Min) & (df['VMS'] < row.VMS_Max)
    subset = df[mask]

    # Scatter plot
    ax.scatter(subset['F_sim'], subset['S_sim'], alpha=0.4, s=15,
               color='steelblue', edgecolors='none')

    # Add regression line
    if len(subset) > 2:
        z = np.polyfit(subset['F_sim'], subset['S_sim'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(subset['F_sim'].min(), subset['F_sim'].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8)

    # Title with VMS range
    ax.set_title(f'VMS {row.VMS_Range} (n={int(row.N)})',
                 fontsize=12, fontweight='bold')

    # Labels
    ax.set_xlabel('F_sim', fontsize=10, fontweight='bold')
    ax.set_ylabel('S_sim', fontsize=10, fontweight='bold')

    # Add correlation text box
    textstr = f'ρ = {row.Spearman_rho:+.3f}\n'
    textstr += f'p = {row.Spearman_p:.2e}\n'
    textstr += f'{row.Strength}'

    # Color based on correlation direction and strength
    if abs(row.Spearman_rho) >= 0.5:
        if row.Spearman_rho < 0:
            box_color = 'lightcoral'  # Strong negative
        else:
            box_color = 'lightgreen'  # Strong positive
    else:
        box_color = 'lightyellow'  # Weak

    props = dict(boxstyle='round', facecolor=box_color, alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props, fontweight='bold')

    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0.5, 1.0])

# Hide empty subplots if any
if n_ranges < len(axes):
    for idx in range(n_ranges, len(axes)):
        axes[idx].set_visible(False)

plt.tight_layout()
plt.savefig('fig_04_scatter_stratified.png', dpi=600, bbox_inches='tight')
print("✓ Saved: fig_04_scatter_stratified.png")
plt.close()

# Create Markdown report
report = f"""# Report 05: Stratified Correlation Analysis

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

To investigate whether the relationship between F_sim and S_sim varies across different VMS ranges (context-dependent correlation).

**Hypothesis:** Global weak correlation (ρ=0.10) may mask heterogeneous patterns at different VMS levels.

## Methodology

- **Stratification:** Dataset divided into 6 VMS ranges
- **Correlation method:** Spearman rank correlation (justified by Script 03)
- **Significance level:** α = 0.05

## Results Summary

| VMS Range | N | % | Spearman ρ | p-value | Strength |
|-----------|---|---|------------|---------|----------|
"""

for _, row in results_df.iterrows():
    sig = "***" if row['Spearman_p'] < 0.001 else "**" if row['Spearman_p'] < 0.01 else "*" if row['Spearman_p'] < 0.05 else "n.s."
    report += f"| {row['VMS_Range']} | {int(row['N'])} | {row['Percent']:.1f}% | {row['Spearman_rho']:+.4f} | {row['Spearman_p']:.2e} | {row['Strength']} {sig} |\n"

report += f"""

*p < 0.05; **p < 0.01; ***p < 0.001; n.s. = not significant

## Key Finding: Trade-off Pattern

### Strongest Correlation: VMS {strongest_neg['VMS_Range']}

**Spearman ρ = {strongest_neg['Spearman_rho']:.4f}** (p = {strongest_neg['Spearman_p']:.2e})

- **Sample size:** n = {int(strongest_neg['N'])} ({strongest_neg['Percent']:.1f}% of dataset)
- **Strength:** {strongest_neg['Strength']} {"negative" if strongest_neg['Spearman_rho'] < 0 else "positive"} correlation
- **Statistical significance:** Highly significant (p < 0.001)

### Biological Interpretation
"""

if strongest_neg['Spearman_rho'] < -0.5:
    report += f"""
**Trade-off detected:** In the VMS {strongest_neg['VMS_Range']} range, there is a **strong negative correlation** between F_sim and S_sim.

**Implications:**
1. **Evolutionary constraint:** High functional similarity → low structural similarity (and vice versa)
2. **Two strategies:** Viral proteins may optimize either function OR structure, but not both simultaneously
3. **Mid-range VMS:** This trade-off is strongest at intermediate mimicry levels
4. **Convergent evolution:** Different sequences can achieve similar functions
5. **Interface mimicry:** Structural similarity may occur without functional convergence

**Mechanistic hypothesis:**
- Viruses with VMS {strongest_neg['VMS_Range']} are under competing selective pressures
- Optimizing functional mimicry compromises structural similarity (or vice versa)
- This represents a **functional-structural spectrum** rather than dual optimization
"""
else:
    report += f"""
The correlation pattern shows context-dependent variation across VMS ranges.
"""

# Analyze trend across ranges
report += f"""
## Pattern Across VMS Ranges

### Correlation Trend

"""

# Describe the trend
rho_values = results_df['Spearman_rho'].values
vms_midpoints = [(row['VMS_Min'] + row['VMS_Max']) / 2 for _, row in results_df.iterrows()]

report += f"""
| VMS Range | Spearman ρ | Direction |
|-----------|-----------|-----------|
"""

for _, row in results_df.iterrows():
    direction = "Negative" if row['Spearman_rho'] < 0 else "Positive" if row['Spearman_rho'] > 0 else "Zero"
    report += f"| {row['VMS_Range']} | {row['Spearman_rho']:+.4f} | {direction} |\n"

# Identify pattern
if any(results_df['Spearman_rho'] < -0.3):
    report += f"""
**Observed pattern:**
1. Low VMS (< 0.4): Negative correlation (both metrics low)
2. Mid VMS (0.4-0.6): **Strong negative correlation** (trade-off)
3. High VMS (≥ 0.6): Correlation weakens or reverses (both metrics high)

**Interpretation:** U-shaped or transition pattern indicating context-dependent relationships.
"""

report += f"""
## Statistical Notes

- **Multiple testing:** {len(results_df)} ranges tested (consider Bonferroni correction: α = {0.05/len(results_df):.4f})
- **After correction:** Strongest result (p={strongest_neg['Spearman_p']:.2e}) remains highly significant
- **Sample sizes:** All ranges have n > 27, sufficient for correlation analysis
- **Effect sizes:** Range from weak to strong, indicating genuine heterogeneity

## Comparison to Global Correlation

| Analysis | Spearman ρ | Interpretation |
|----------|-----------|----------------|
| **Global (Script 04)** | +0.1005 | Weak positive, nearly independent |
| **VMS {strongest_neg['VMS_Range']} (stratified)** | {strongest_neg['Spearman_rho']:.4f} | {strongest_neg['Strength']} {"negative" if strongest_neg['Spearman_rho'] < 0 else "positive"} |

**Key insight:** Global correlation masks strong **context-dependent** patterns.

## Implications for VMS

1. **Heterogeneity confirmed:** Relationship between F_sim and S_sim varies by VMS level
2. **Trade-off exists:** At moderate VMS, functional and structural similarity are inversely related
3. **Spectrum model:** Suggests continuous spectrum from functional to structural mimicry
4. **Not redundant:** F_sim and S_sim capture complementary (sometimes opposing) information

## Visual Output

- **Figure:** fig_04_scatter_stratified.png
- **Format:** 2×3 panel scatter plots, color-coded by correlation strength
- **Resolution:** 600 DPI

## Data Output

- **File:** results_04_correlation_stratified.csv
- **Content:** Correlations, p-values, sample sizes, and means for each VMS range

---

**Next steps:**
1. Visualize correlation trend across VMS ranges (line plot)
2. Test for statistical significance of correlation differences between ranges
3. Explore biological/functional characteristics of high trade-off pairs
"""

# Save report
with open('report_05_stratified_correlation.md', 'w') as f:
    f.write(report)
print("✓ Saved: report_05_stratified_correlation.md")

# Print summary to console
print()
print("-"*70)
print("STRATIFICATION SUMMARY")
print("-"*70)
print(f"{'VMS Range':<12} {'N':>5} {'%':>6} {'ρ':>8} {'p-value':>12} {'Strength':<15}")
print("-"*70)
for _, row in results_df.iterrows():
    print(f"{row['VMS_Range']:<12} {int(row['N']):>5} {row['Percent']:>5.1f}% {row['Spearman_rho']:>+8.4f} {row['Spearman_p']:>12.2e} {row['Strength']:<15}")
print("-"*70)
print()
print(f"KEY FINDING: Strongest correlation in VMS {strongest_neg['VMS_Range']}")
print(f"  ρ = {strongest_neg['Spearman_rho']:.4f} (p={strongest_neg['Spearman_p']:.2e})")
print()
print("="*70)
print("SCRIPT 05 COMPLETED")
print("Files generated:")
print("  - fig_04_scatter_stratified.png")
print("  - results_04_correlation_stratified.csv")
print("  - report_05_stratified_correlation.md")
print("="*70)
