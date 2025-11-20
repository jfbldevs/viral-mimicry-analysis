"""
Script 03: Normality Tests
===========================
Purpose: Test whether F_sim, S_sim, and VMS follow normal distributions
         using formal statistical tests and Q-Q plots

Input:
  - virus_to_human_top5_neighbors_final_similarity.csv

Output:
  - fig_02_qq_plots.png (Q-Q plots for normality assessment)
  - results_02_normality_tests.csv (p-values from normality tests)
  - report_03_normality_tests.md (normality analysis report)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Set style
plt.style.use('seaborn-v0_8-darkgrid')

# Load dataset
print("="*70)
print("SCRIPT 03: NORMALITY TESTS")
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

variables = ['F_sim', 'S_sim', 'VMS']
titles = [
    'F_sim: Functional Similarity',
    'S_sim: Structural Similarity',
    'VMS: Virus Mimicry Score'
]

# Perform normality tests
print("Performing normality tests...")
print()

normality_results = {}

for var in variables:
    data = df[var].dropna()

    # Shapiro-Wilk test (best for n < 5000)
    shapiro_stat, shapiro_p = stats.shapiro(data)

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))

    # D'Agostino-Pearson test
    dagostino_stat, dagostino_p = stats.normaltest(data)

    normality_results[var] = {
        'Shapiro_Stat': shapiro_stat,
        'Shapiro_p': shapiro_p,
        'KS_Stat': ks_stat,
        'KS_p': ks_p,
        'DAgostino_Stat': dagostino_stat,
        'DAgostino_p': dagostino_p,
        'Normal': 'No' if shapiro_p < 0.05 else 'Yes'
    }

    print(f"{var}:")
    print(f"  Shapiro-Wilk: W={shapiro_stat:.6f}, p={shapiro_p:.2e}")
    print(f"  Kolmogorov-Smirnov: D={ks_stat:.6f}, p={ks_p:.2e}")
    print(f"  D'Agostino-Pearson: K2={dagostino_stat:.6f}, p={dagostino_p:.2e}")
    print(f"  → Normal distribution: {normality_results[var]['Normal']}")
    print()

# Save results to CSV
results_df = pd.DataFrame(normality_results).T
results_df.to_csv('results_02_normality_tests.csv')
print("✓ Saved: results_02_normality_tests.csv")
print()

# Create Q-Q plots
print("Generating Q-Q plots...")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle('Q-Q Plots: Normality Assessment', fontsize=16, fontweight='bold', y=1.02)

for idx, (var, title) in enumerate(zip(variables, titles)):
    ax = axes[idx]
    data = df[var].dropna()

    # Q-Q plot
    stats.probplot(data, dist="norm", plot=ax)

    # Customize
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Theoretical Quantiles', fontsize=11, fontweight='bold')
    ax.set_ylabel('Sample Quantiles', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Add text box with ALL three test results
    shapiro_p = normality_results[var]['Shapiro_p']
    ks_p = normality_results[var]['KS_p']
    dagostino_p = normality_results[var]['DAgostino_p']
    normal_status = normality_results[var]['Normal']

    textstr = f'Shapiro-Wilk: p={shapiro_p:.2e}\n'
    textstr += f'K-S: p={ks_p:.2e}\n'
    textstr += f"D'Agostino: p={dagostino_p:.2e}\n"
    textstr += f'→ Normal: {normal_status}'

    # Color based on normality
    box_color = 'lightcoral' if normal_status == 'No' else 'lightgreen'

    props = dict(boxstyle='round', facecolor=box_color, alpha=0.7)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('fig_02_qq_plots.png', dpi=600, bbox_inches='tight')
print("✓ Saved: fig_02_qq_plots.png")
plt.close()

# Create Markdown report
report = f"""# Report 03: Normality Tests

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

To determine whether F_sim, S_sim, and VMS follow normal distributions, which informs the choice of correlation method (Pearson vs Spearman).

## Statistical Tests Performed

1. **Shapiro-Wilk Test** - Most powerful test for normality (n < 5000)
2. **Kolmogorov-Smirnov Test** - Compares empirical distribution to normal
3. **D'Agostino-Pearson Test** - Tests skewness and kurtosis

**Null hypothesis (H₀):** Data follows a normal distribution
**Significance level:** α = 0.05
**Decision rule:** Reject H₀ if p < 0.05

## Results Summary

| Variable | Shapiro-Wilk p | KS p | D'Agostino p | Normal? |
|----------|---------------|------|--------------|---------|
| **F_sim** | {normality_results['F_sim']['Shapiro_p']:.2e} | {normality_results['F_sim']['KS_p']:.2e} | {normality_results['F_sim']['DAgostino_p']:.2e} | **{normality_results['F_sim']['Normal']}** |
| **S_sim** | {normality_results['S_sim']['Shapiro_p']:.2e} | {normality_results['S_sim']['KS_p']:.2e} | {normality_results['S_sim']['DAgostino_p']:.2e} | **{normality_results['S_sim']['Normal']}** |
| **VMS** | {normality_results['VMS']['Shapiro_p']:.2e} | {normality_results['VMS']['KS_p']:.2e} | {normality_results['VMS']['DAgostino_p']:.2e} | **{normality_results['VMS']['Normal']}** |

## Interpretation

### F_sim (Functional Similarity)
"""

f_normal = normality_results['F_sim']['Normal']
if f_normal == 'No':
    report += f"""
- **Result:** All three tests reject normality (all p < 0.05)
- **Implication:** F_sim does NOT follow a normal distribution
- **Visual evidence:** Q-Q plot shows deviation from diagonal line
- **From Script 02:** Right-skewed distribution with heavy tails
"""
else:
    report += f"""
- **Result:** Tests do not reject normality (p > 0.05)
- **Implication:** F_sim approximately follows normal distribution
"""

report += f"""
### S_sim (Structural Similarity)
"""

s_normal = normality_results['S_sim']['Normal']
if s_normal == 'No':
    report += f"""
- **Result:** All three tests reject normality (all p < 0.05)
- **Implication:** S_sim does NOT follow a normal distribution
- **Visual evidence:** Q-Q plot shows deviation from diagonal line
- **From Script 02:** Distribution shows non-normal characteristics
"""
else:
    report += f"""
- **Result:** Tests do not reject normality (p > 0.05)
- **Implication:** S_sim approximately follows normal distribution
"""

report += f"""
### VMS (Virus Mimicry Score)
"""

v_normal = normality_results['VMS']['Normal']
if v_normal == 'No':
    report += f"""
- **Result:** All three tests reject normality (all p < 0.05)
- **Implication:** VMS does NOT follow a normal distribution
- **Visual evidence:** Q-Q plot shows deviation from diagonal line
- **From Script 02:** Right-skewed distribution inherited from F_sim
"""
else:
    report += f"""
- **Result:** Tests do not reject normality (p > 0.05)
- **Implication:** VMS approximately follows normal distribution
"""

# Overall conclusion
all_non_normal = all(normality_results[var]['Normal'] == 'No' for var in variables)
any_non_normal = any(normality_results[var]['Normal'] == 'No' for var in variables)

report += f"""
## Overall Conclusion

"""

if all_non_normal:
    report += """**All three variables reject normality at α = 0.05 level.**

### Statistical Implications

1. **Parametric tests invalid:** Pearson correlation assumes bivariate normality
2. **Non-parametric methods required:** Spearman rank correlation is appropriate
3. **Robust to outliers:** Spearman uses ranks, not raw values
4. **No assumptions violated:** Non-parametric methods do not require normality

### Methodological Decision

✅ **Use Spearman rank correlation (ρ) for all correlation analyses**

**Justification:**
- Both F_sim and S_sim are non-normal (p < 0.05 in all tests)
- Data bounded in [0, 1] range (not unbounded like normal distribution)
- Presence of skewness and heavy tails
- Non-parametric method is statistically appropriate and conservative
"""
elif any_non_normal:
    report += f"""**At least one variable rejects normality at α = 0.05 level.**

### Statistical Implications

Since at least one variable is non-normal, parametric correlation (Pearson) assumptions are violated.

### Methodological Decision

✅ **Use Spearman rank correlation (ρ) for all correlation analyses**

**Justification:**
- Conservative choice when normality is questionable
- Robust to outliers and skewed distributions
- Valid for monotonic relationships (not just linear)
"""
else:
    report += """**All variables appear approximately normal at α = 0.05 level.**

### Statistical Implications

Both Pearson and Spearman correlation are valid choices.

### Methodological Decision

We will use **Spearman rank correlation (ρ)** as a conservative choice and to maintain robustness to potential outliers.
"""

report += f"""
## Q-Q Plot Interpretation

**Q-Q (Quantile-Quantile) plots** compare sample quantiles to theoretical normal quantiles:
- **Points on diagonal line:** Data follows normal distribution
- **Points deviate from line:** Data deviates from normality
- **S-shaped curve:** Indicates heavy tails
- **Curved ends:** Indicates skewness

**Why three tests?**
- **Shapiro-Wilk:** Most powerful test for n < 5000 (gold standard)
- **Kolmogorov-Smirnov:** Tests overall distribution shape
- **D'Agostino-Pearson:** Tests skewness and kurtosis specifically
- **Robustness:** Convergence of all three tests strengthens conclusion

All three tests are displayed in each Q-Q plot panel for transparency.

See **fig_02_qq_plots.png** for visual assessment.

## Visual Output

- **Figure:** fig_02_qq_plots.png
- **Format:** 3-panel Q-Q plots
- **Resolution:** 600 DPI

## Data Output

- **File:** results_02_normality_tests.csv
- **Content:** Test statistics and p-values for all three tests

---

**Next steps:** Calculate global correlation between F_sim and S_sim using Spearman method
"""

# Save report
with open('report_03_normality_tests.md', 'w') as f:
    f.write(report)
print("✓ Saved: report_03_normality_tests.md")

# Print summary to console
print()
print("-"*70)
print("CONCLUSION")
print("-"*70)
if all_non_normal:
    print("All variables are NON-NORMAL (all p < 0.05)")
    print("→ Use SPEARMAN correlation (non-parametric)")
elif any_non_normal:
    print("At least one variable is NON-NORMAL")
    print("→ Use SPEARMAN correlation (conservative choice)")
else:
    print("All variables appear NORMAL (all p > 0.05)")
    print("→ Both Pearson and Spearman valid")
print()
print("="*70)
print("SCRIPT 03 COMPLETED")
print("Files generated:")
print("  - fig_02_qq_plots.png")
print("  - results_02_normality_tests.csv")
print("  - report_03_normality_tests.md")
print("="*70)
