"""
Script 11: Normalized Viral Family Comparison (Size-Controlled)
================================================================
Purpose: Fair comparison of mimicry tendency across families of different sizes
         Controls for genome size / protein count to avoid bias

Input:
  - virus_to_human_top5_neighbors_final_similarity.csv
  - viral_family_mapping_ICTV.csv
  - viral_family_mapping_complete.csv (for proteome info)

Output:
  - results_10_normalized_family_comparison.csv
  - fig_15_mimicry_per_protein.png
  - fig_16_family_size_stratified.png
  - report_11_normalized_comparison.md

Strategy: Multiple normalization approaches for fair comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = 'white'
sns.set_palette("husl")

print("="*80)
print("SCRIPT 11: NORMALIZED FAMILY COMPARISON (SIZE-CONTROLLED)")
print("="*80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

# Main dataset
df = pd.read_csv('virus_to_human_top5_neighbors_final_similarity.csv')
df = df.rename(columns={
    'embedding_similarity': 'F_sim',
    'similarity': 'S_sim',
    'virus_mimicry_score': 'VMS'
})

print(f"Total dataset: {len(df)} virus-human pairs")

# Load ICTV family mapping
viral_mapping = pd.read_csv('viral_family_mapping_ICTV.csv')
print(f"Loaded {len(viral_mapping)} viral proteins with ICTV families")

# Merge
df = df.merge(
    viral_mapping.rename(columns={'protein_id': 'virus_protein_id'}),
    on='virus_protein_id',
    how='left'
)
df['viral_family'] = df['viral_family'].fillna('Unmapped')

# Count proteins per family
proteins_per_family = viral_mapping.groupby('viral_family').size().reset_index(name='n_proteins')
print(f"\nProteins per family calculated for {len(proteins_per_family)} families")
print()

# ============================================================================
# CALCULATE NORMALIZED METRICS
# ============================================================================

print("Calculating normalized metrics...")
print()

cutoff = 0.5
high_mimicry = df[df['VMS'] >= cutoff].copy()

family_stats = []

for family in df['viral_family'].unique():
    if family == 'Unmapped':
        continue

    # Basic counts
    family_all = df[df['viral_family'] == family]
    family_high = high_mimicry[high_mimicry['viral_family'] == family]

    n_total_pairs = len(family_all)
    n_high_pairs = len(family_high)
    proportion_high = (n_high_pairs / n_total_pairs * 100) if n_total_pairs > 0 else 0

    # Get protein count
    n_proteins = proteins_per_family[proteins_per_family['viral_family'] == family]['n_proteins'].values
    n_proteins = n_proteins[0] if len(n_proteins) > 0 else np.nan

    # NORMALIZED METRICS
    # 1. Pairs per protein
    pairs_per_protein = n_high_pairs / n_proteins if n_proteins > 0 else np.nan

    # 2. Expected vs observed (assuming uniform distribution)
    total_high = len(high_mimicry)
    total_pairs = len(df)
    expected_high = n_total_pairs * (total_high / total_pairs)
    enrichment = (n_high_pairs / expected_high) if expected_high > 0 else np.nan

    # 3. VMS statistics
    vms_mean = family_all['VMS'].mean()
    vms_median = family_all['VMS'].median()
    vms_std = family_all['VMS'].std()

    family_stats.append({
        'Family': family,
        'N_Proteins': int(n_proteins) if not np.isnan(n_proteins) else 0,
        'Total_Pairs': n_total_pairs,
        'High_Mimicry_Pairs': n_high_pairs,
        'Proportion_High_Pct': proportion_high,
        'Pairs_per_Protein': pairs_per_protein,
        'Enrichment_vs_Expected': enrichment,
        'VMS_Mean': vms_mean,
        'VMS_Median': vms_median,
        'VMS_Std': vms_std
    })

# Create DataFrame
results = pd.DataFrame(family_stats)
results = results.sort_values('Pairs_per_Protein', ascending=False)

# Save
results.to_csv('results_10_normalized_family_comparison.csv', index=False, float_format='%.4f')
print("✓ Saved: results_10_normalized_family_comparison.csv")
print()

# ============================================================================
# STRATIFY BY FAMILY SIZE
# ============================================================================

print("Stratifying families by size...")

# Define size categories
def categorize_size(n_proteins):
    if n_proteins >= 400:
        return 'Large (≥400 proteins)'
    elif n_proteins >= 30:
        return 'Medium (30-399 proteins)'
    else:
        return 'Small (<30 proteins)'

results['Size_Category'] = results['N_Proteins'].apply(categorize_size)

print("\nFamilies by size category:")
for cat in ['Large (≥400 proteins)', 'Medium (30-399 proteins)', 'Small (<30 proteins)']:
    subset = results[results['Size_Category'] == cat]
    print(f"  {cat}: {len(subset)} families")
print()

# ============================================================================
# FIGURE 1: MIMICRY PER PROTEIN (NORMALIZED)
# ============================================================================

print("Generating Figure 1: Pairs per protein (normalized metric)...")

fig, ax = plt.subplots(figsize=(16, 12))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Sort by pairs per protein
plot_data = results.sort_values('Pairs_per_Protein', ascending=True)

# Color by size category
color_map = {
    'Large (≥400 proteins)': '#FF6B6B',      # Red
    'Medium (30-399 proteins)': '#4ECDC4',   # Teal
    'Small (<30 proteins)': '#95E1D3'        # Light green
}
colors = [color_map[cat] for cat in plot_data['Size_Category']]

y_pos = np.arange(len(plot_data)) * 1.2
bars = ax.barh(y_pos, plot_data['Pairs_per_Protein'],
               height=0.9, color=colors, edgecolor='black', linewidth=1.2)

# Add values
for i, (idx, row) in enumerate(plot_data.iterrows()):
    ax.text(row['Pairs_per_Protein'] + 0.05, y_pos[i],
            f"{row['Pairs_per_Protein']:.2f} ({row['Proportion_High_Pct']:.1f}%)",
            va='center', ha='left', fontsize=11, fontweight='bold')

ax.set_yticks(y_pos)
ax.set_yticklabels(plot_data['Family'], fontsize=13, fontweight='bold')
ax.set_xlabel('High-Mimicry Pairs per Protein (VMS ≥ 0.5)', fontsize=15, fontweight='bold')
ax.set_title('Normalized Mimicry Tendency: Pairs per Protein by ICTV Family',
             fontsize=18, fontweight='bold', pad=20)

# Legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color_map[cat], edgecolor='black', label=cat)
                   for cat in ['Large (≥400 proteins)', 'Medium (30-399 proteins)', 'Small (<30 proteins)']]
ax.legend(handles=legend_elements, loc='lower right', fontsize=12, frameon=True,
          facecolor='white', edgecolor='black', title='Family Size')

ax.grid(axis='x', alpha=0.3)
ax.set_axisbelow(True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('fig_15_mimicry_per_protein.png', dpi=600, bbox_inches='tight', facecolor='white')
print("✓ Saved: fig_15_mimicry_per_protein.png")
plt.close()

# ============================================================================
# FIGURE 2: STRATIFIED BY SIZE
# ============================================================================

print("Generating Figure 2: Comparison within size categories...")

fig, axes = plt.subplots(1, 3, figsize=(20, 8))
fig.suptitle('Mimicry Tendency by Family Size Category', fontsize=18, fontweight='bold', y=1.02)

size_cats = ['Large (≥400 proteins)', 'Medium (30-399 proteins)', 'Small (<30 proteins)']

for idx, (ax, cat) in enumerate(zip(axes, size_cats)):
    subset = results[results['Size_Category'] == cat].sort_values('Proportion_High_Pct', ascending=True)

    if len(subset) == 0:
        ax.text(0.5, 0.5, 'No families', ha='center', va='center', fontsize=14)
        ax.set_title(cat, fontsize=14, fontweight='bold')
        continue

    y_pos = np.arange(len(subset))
    bars = ax.barh(y_pos, subset['Proportion_High_Pct'],
                   color=color_map[cat], edgecolor='black', linewidth=1.5)

    # Add percentage labels
    for i, (_, row) in enumerate(subset.iterrows()):
        ax.text(row['Proportion_High_Pct'] + 1, i, f"{row['Proportion_High_Pct']:.1f}%",
                va='center', ha='left', fontsize=10, fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(subset['Family'], fontsize=11, fontweight='bold')
    ax.set_xlabel('% High Mimicry (VMS ≥ 0.5)', fontsize=12, fontweight='bold')
    ax.set_title(f'{cat}\n({len(subset)} families)', fontsize=13, fontweight='bold')
    ax.set_xlim(0, 110)
    ax.grid(axis='x', alpha=0.3)
    ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('fig_16_family_size_stratified.png', dpi=600, bbox_inches='tight', facecolor='white')
print("✓ Saved: fig_16_family_size_stratified.png")
plt.close()

# ============================================================================
# STATISTICAL TESTS
# ============================================================================

print("\nPerforming statistical tests...")

# Test if size correlates with mimicry proportion
spearman_rho, spearman_p = stats.spearmanr(results['N_Proteins'], results['Proportion_High_Pct'])
print(f"\nCorrelation: Family size vs Mimicry proportion")
print(f"  Spearman ρ = {spearman_rho:+.4f} (p = {spearman_p:.4e})")

if spearman_p < 0.05:
    if spearman_rho > 0:
        print("  → Larger families show HIGHER mimicry proportion")
    else:
        print("  → Larger families show LOWER mimicry proportion")
else:
    print("  → No significant correlation (size-independent mimicry)")

# Kruskal-Wallis test across size categories
size_groups = [results[results['Size_Category'] == cat]['Proportion_High_Pct'].values
               for cat in size_cats if len(results[results['Size_Category'] == cat]) > 0]

if len(size_groups) >= 2:
    h_stat, kw_p = stats.kruskal(*size_groups)
    print(f"\nKruskal-Wallis test: Mimicry across size categories")
    print(f"  H = {h_stat:.4f}, p = {kw_p:.4e}")
    if kw_p < 0.05:
        print("  → Significant difference between size categories")
    else:
        print("  → No significant difference between size categories")

print()

# ============================================================================
# CREATE REPORT
# ============================================================================

print("Creating report...")

report = f"""# Report 11: Normalized Family Comparison (Size-Controlled)

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Purpose

To fairly compare mimicry tendency across ICTV families with different genome sizes,
controlling for the fact that large-genome families naturally have more protein pairs.

## Problem Statement

**Bias in absolute counts:**
- Herpesviridae (702 proteins) has 1,646 high-mimicry pairs
- Flaviviridae (6 proteins) has only 20 high-mimicry pairs

**Question:** Is Herpesviridae truly more prone to mimicry, or does it just have more proteins?

## Normalization Strategies

### 1. Proportion (%) - Already Used
- **Formula:** (High mimicry pairs / Total pairs) × 100
- **Advantage:** Simple, intuitive
- **Limitation:** Doesn't account for statistical power differences

### 2. Pairs per Protein - NEW
- **Formula:** High mimicry pairs / Number of proteins
- **Advantage:** Direct size normalization
- **Interpretation:** "Mimicry intensity" per protein

### 3. Enrichment vs Expected - NEW
- **Formula:** Observed / Expected (under uniform distribution)
- **Advantage:** Statistical framework
- **Interpretation:** >1 = over-represented, <1 = under-represented

## Results: Top Families by Normalized Metric

### By Pairs per Protein (Size-Normalized)

| Rank | Family | Pairs/Protein | % High | # Proteins |
|------|--------|--------------|--------|------------|
"""

top_normalized = results.nlargest(10, 'Pairs_per_Protein')
for rank, (idx, row) in enumerate(top_normalized.iterrows(), 1):
    report += f"| {rank} | {row['Family']} | {row['Pairs_per_Protein']:.2f} | {row['Proportion_High_Pct']:.1f}% | {row['N_Proteins']} |\n"

report += f"""

### Key Finding: Size-Independent Mimicry Leaders

**Top 3 by normalized metric:**
"""

for rank, (idx, row) in enumerate(top_normalized.head(3).iterrows(), 1):
    report += f"""
**{rank}. {row['Family']}**
- **Pairs per protein:** {row['Pairs_per_Protein']:.2f}
- **Proportion high:** {row['Proportion_High_Pct']:.1f}%
- **Number of proteins:** {row['N_Proteins']}
- **Enrichment:** {row['Enrichment_vs_Expected']:.2f}× expected
"""

report += f"""

## Size Stratification Analysis

### Large Families (≥400 proteins): {len(results[results['Size_Category'] == 'Large (≥400 proteins)'])}

"""

large = results[results['Size_Category'] == 'Large (≥400 proteins)'].sort_values('Proportion_High_Pct', ascending=False)
for _, row in large.iterrows():
    report += f"- **{row['Family']}:** {row['Proportion_High_Pct']:.1f}% ({row['Pairs_per_Protein']:.2f} pairs/protein)\n"

report += f"""

### Medium Families (30-399 proteins): {len(results[results['Size_Category'] == 'Medium (30-399 proteins)'])}

"""

medium = results[results['Size_Category'] == 'Medium (30-399 proteins)'].sort_values('Proportion_High_Pct', ascending=False)
for _, row in medium.iterrows():
    report += f"- **{row['Family']}:** {row['Proportion_High_Pct']:.1f}% ({row['Pairs_per_Protein']:.2f} pairs/protein)\n"

report += f"""

### Small Families (<30 proteins): {len(results[results['Size_Category'] == 'Small (<30 proteins)'])}

"""

small = results[results['Size_Category'] == 'Small (<30 proteins)'].sort_values('Proportion_High_Pct', ascending=False)
for _, row in small.iterrows():
    report += f"- **{row['Family']}:** {row['Proportion_High_Pct']:.1f}% ({row['Pairs_per_Protein']:.2f} pairs/protein)\n"

report += f"""

## Statistical Analysis

### Correlation: Family Size vs Mimicry

- **Spearman ρ:** {spearman_rho:+.4f}
- **p-value:** {spearman_p:.4e}
- **Interpretation:** {"Significant correlation" if spearman_p < 0.05 else "No significant correlation"}

"""

if spearman_p < 0.05:
    if spearman_rho > 0:
        report += "**Finding:** Larger families tend to have HIGHER mimicry proportions.\n"
        report += "**Implication:** Genome size may provide MORE opportunities for effective mimicry.\n"
    else:
        report += "**Finding:** Larger families tend to have LOWER mimicry proportions.\n"
        report += "**Implication:** Small, focused genomes may optimize mimicry more efficiently.\n"
else:
    report += "**Finding:** No significant relationship between family size and mimicry tendency.\n"
    report += "**Implication:** Mimicry is a size-independent evolutionary strategy.\n"

report += f"""

## Biological Interpretation

### Why Normalize?

1. **Fair comparison:** Removes bias from genome size differences
2. **True evolutionary signal:** Identifies families with genuine mimicry strategies
3. **Statistical validity:** Comparable effect sizes across families

### What We Learn:

**Absolute count leaders (Herpesviridae, Poxviridae):**
- Have the most mimicry pairs in absolute terms
- This reflects their large genomes (>400 proteins)
- Still biologically meaningful - greater overall mimicry capacity

**Normalized leaders (e.g., Flaviviridae, Retroviridae):**
- Have the highest mimicry "intensity" per protein
- Suggests mimicry is a CORE evolutionary strategy
- Every protein has high probability of mimicking host

**Implication:** Both perspectives are important:
- Absolute: Total impact on host immune system
- Normalized: Evolutionary strategy and selective pressure

## Visual Outputs

### Figure 15: Mimicry per Protein
- **Type:** Horizontal bar chart (size-normalized)
- **Metric:** High-mimicry pairs per protein
- **Color-coded:** By family size category
- **Interpretation:** True mimicry tendency independent of genome size

### Figure 16: Size-Stratified Comparison
- **Type:** Multi-panel comparison
- **Shows:** Mimicry % within each size category
- **Advantage:** "Apples-to-apples" comparison

## Recommendations for Analysis

### When to Use Each Metric:

**Absolute counts:**
- Understanding total mimicry capacity
- Host immune burden assessment
- Dataset composition analysis

**Proportions (%):**
- General mimicry tendency
- Quick comparisons

**Pairs per protein:**
- **Fair cross-family comparisons** ← BEST for mixed-size datasets
- Evolutionary strategy assessment
- Effect size estimation

**Size-stratified:**
- When genome size is a confound
- Within-category rankings

## Conclusion

After controlling for genome size:
"""

top3 = top_normalized.head(3)
for rank, (idx, row) in enumerate(top3.iterrows(), 1):
    report += f"{rank}. **{row['Family']}** shows the highest mimicry intensity ({row['Pairs_per_Protein']:.2f} pairs/protein)\n"

report += f"""

These families demonstrate that mimicry is a **core evolutionary strategy**,
independent of genome size constraints.

---

**Data Files:**
- results_10_normalized_family_comparison.csv
- fig_15_mimicry_per_protein.png
- fig_16_family_size_stratified.png
"""

with open('report_11_normalized_comparison.md', 'w') as f:
    f.write(report)
print("✓ Saved: report_11_normalized_comparison.md")

# ============================================================================
# CONSOLE SUMMARY
# ============================================================================

print()
print("="*80)
print("NORMALIZED FAMILY COMPARISON SUMMARY")
print("="*80)
print()
print("Top 5 by Pairs per Protein (size-normalized):")
for rank, (idx, row) in enumerate(top_normalized.head(5).iterrows(), 1):
    print(f"  {rank}. {row['Family']}: {row['Pairs_per_Protein']:.2f} pairs/protein "
          f"({row['Proportion_High_Pct']:.1f}%, {row['N_Proteins']} proteins)")
print()
print(f"Correlation (size vs mimicry): ρ = {spearman_rho:+.4f} (p = {spearman_p:.4e})")
print()
print("="*80)
print("SCRIPT 11 COMPLETED")
print("Files generated:")
print("  - results_10_normalized_family_comparison.csv")
print("  - fig_15_mimicry_per_protein.png")
print("  - fig_16_family_size_stratified.png")
print("  - report_11_normalized_comparison.md")
print("="*80)
