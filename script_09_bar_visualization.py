"""
Script 09: Horizontal Bar Visualization - High-Confidence Pairs
================================================================
Purpose: Create 4 professional horizontal bar chart visualizations for all
         high-confidence pairs (VMS ≥ 0.6), divided into 20 panels each for optimal legibility

Input:
  - virus_to_human_top5_neighbors_final_similarity.csv

Output:
  - fig_10_bars_all279.png (All high-confidence pairs in 20 panels, 5×4 layout)
  - fig_11_bars_components_all279.png (F_sim, S_sim, VMS comparison in 20 panels)
  - fig_12_bars_viral_families_all279.png (Grouped by family in 20 panels)
  - fig_13_bars_stacked_all279.png (Stacked composition in 20 panels)
  - results_08_highconfidence_pairs.csv
  - report_09_bar_analysis.md
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.facecolor'] = 'white'

# Load dataset
print("="*70)
print("SCRIPT 09: HORIZONTAL BAR VISUALIZATION - HIGH-CONFIDENCE PAIRS")
print("="*70)
print()

df = pd.read_csv('virus_to_human_top5_neighbors_final_similarity.csv')

# Rename columns
df = df.rename(columns={
    'embedding_similarity': 'F_sim',
    'similarity': 'S_sim',
    'virus_mimicry_score': 'VMS'
})

print(f"Total dataset: {len(df)} pairs")

# Filter high-confidence pairs
cutoff = 0.6
high_conf = df[df['VMS'] >= cutoff].copy()

print(f"High-confidence pairs (VMS ≥ {cutoff}): {len(high_conf)} ({len(high_conf)/len(df)*100:.1f}%)")
print()

# Clean protein names
def clean_protein_name(name):
    if '|' in name:
        parts = name.split('|')
        return parts[-1].replace('_', ' ')[:25]
    return str(name)[:25]

high_conf['virus_clean'] = high_conf['virus_protein_id'].apply(clean_protein_name)
high_conf['human_clean'] = high_conf['human_protein_id'].apply(clean_protein_name)
high_conf['pair_label'] = high_conf.apply(lambda x: f"{x['virus_clean']} → {x['human_clean']}", axis=1)

# Load viral family mapping from FASTA-based classification
print("Loading viral family mapping from FASTA files...")
viral_mapping = pd.read_csv('viral_family_mapping.csv')
print(f"  Loaded {len(viral_mapping)} viral proteins mapped to categories")

# Merge with high_conf to assign real viral families
high_conf = high_conf.merge(
    viral_mapping.rename(columns={'protein_id': 'virus_protein_id', 'viral_category': 'viral_family'}),
    on='virus_protein_id',
    how='left'
)

# Fill unmapped proteins with "Unmapped"
high_conf['viral_family'] = high_conf['viral_family'].fillna('Unmapped')

print("  Viral family distribution:")
for family, count in high_conf['viral_family'].value_counts().items():
    print(f"    {family}: {count}")
print()

# Save data
high_conf.to_csv('results_08_highconfidence_pairs.csv', index=False)
print("✓ Saved: results_08_highconfidence_pairs.csv")
print()

# Sort by VMS
all_pairs = high_conf.sort_values('VMS', ascending=False).reset_index(drop=True)

# Divide into 20 groups for optimal legibility (4×5 layout)
n_panels = 20
n_per_panel = int(np.ceil(len(all_pairs) / n_panels))
groups = [
    all_pairs.iloc[i*n_per_panel:(i+1)*n_per_panel].reset_index(drop=True)
    for i in range(n_panels)
]

print(f"Dividing {len(all_pairs)} pairs into {n_panels} panels:")
for i, group in enumerate(groups, 1):
    print(f"  Panel {i}: {len(group)} pairs")
print()

# ============================================================================
# FIGURE 1: ALL High-Confidence Pairs - Horizontal Bars by VMS (20 panels)
# ============================================================================

print(f"Generating Figure 1: All {len(all_pairs)} pairs in {n_panels} panels...")

fig, axes = plt.subplots(4, 5, figsize=(25, 24))
fig.suptitle(f'All {len(all_pairs)} High-Confidence Viral Mimicry Pairs (VMS ≥ {cutoff})',
             fontsize=18, fontweight='bold', y=0.998)

axes = axes.flatten()

for panel_idx, (ax, group) in enumerate(zip(axes, groups)):
    # Calculate global rank
    start_rank = panel_idx * n_per_panel + 1

    # Create colormap
    colors = plt.cm.RdYlGn(np.linspace(0.3, 1, len(group)))

    # Create bars
    y_positions = np.arange(len(group))
    ax.barh(y_positions, group['VMS'], height=0.85,
            color=colors[::-1], edgecolor='black', linewidth=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(group.iterrows()):
        ax.text(row['VMS'] + 0.005, i, f"{row['VMS']:.3f}",
                va='center', fontsize=6, fontweight='bold')

    # Labels
    ax.set_yticks(y_positions)
    labels = [f"{start_rank+i}. {row['pair_label']}" for i, (_, row) in enumerate(group.iterrows())]
    ax.set_yticklabels(labels, fontsize=6)

    ax.set_xlabel('VMS', fontsize=11, fontweight='bold')
    ax.set_title(f'Panel {panel_idx+1}: Ranks {start_rank}-{start_rank+len(group)-1}',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0.55, all_pairs['VMS'].max() + 0.02)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('fig_10_bars_all279.png', dpi=600, bbox_inches='tight')
print("✓ Saved: fig_10_bars_all279.png")
plt.close()

# ============================================================================
# FIGURE 2: ALL Pairs - Grouped Bars (F_sim, S_sim, VMS) in 20 panels
# ============================================================================

print(f"Generating Figure 2: Component comparison in {n_panels} panels...")

fig, axes = plt.subplots(4, 5, figsize=(25, 24))
fig.suptitle(f'All {len(all_pairs)} Pairs: F_sim, S_sim, and VMS Comparison',
             fontsize=18, fontweight='bold', y=0.998)

axes = axes.flatten()

for panel_idx, (ax, group) in enumerate(zip(axes, groups)):
    start_rank = panel_idx * n_per_panel + 1

    y_positions = np.arange(len(group))
    bar_height = 0.27

    # Plot three bars for each pair
    ax.barh(y_positions - bar_height, group['F_sim'], bar_height,
            label='F_sim' if panel_idx == 0 else '', color='#3498db',
            edgecolor='black', linewidth=0.3)
    ax.barh(y_positions, group['S_sim'], bar_height,
            label='S_sim' if panel_idx == 0 else '', color='#e74c3c',
            edgecolor='black', linewidth=0.3)
    ax.barh(y_positions + bar_height, group['VMS'], bar_height,
            label='VMS' if panel_idx == 0 else '', color='#2ecc71',
            edgecolor='black', linewidth=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(group.iterrows()):
        ax.text(row['F_sim'] + 0.005, i - bar_height, f"{row['F_sim']:.2f}",
                va='center', fontsize=5.5, fontweight='bold')
        ax.text(row['S_sim'] + 0.005, i, f"{row['S_sim']:.2f}",
                va='center', fontsize=5.5, fontweight='bold')
        ax.text(row['VMS'] + 0.005, i + bar_height, f"{row['VMS']:.2f}",
                va='center', fontsize=5.5, fontweight='bold')

    # Labels
    ax.set_yticks(y_positions)
    labels = [f"{start_rank+i}. {row['pair_label']}" for i, (_, row) in enumerate(group.iterrows())]
    ax.set_yticklabels(labels, fontsize=6)

    ax.set_xlabel('Similarity Score', fontsize=11, fontweight='bold')
    ax.set_title(f'Panel {panel_idx+1}: Ranks {start_rank}-{start_rank+len(group)-1}',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1.03)
    if panel_idx == 0:
        ax.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('fig_11_bars_components_all279.png', dpi=600, bbox_inches='tight')
print("✓ Saved: fig_11_bars_components_all279.png")
plt.close()

# ============================================================================
# FIGURE 3: ALL Pairs - Viral Families in 20 panels
# ============================================================================

print(f"Generating Figure 3: Viral families in {n_panels} panels...")

# Sort by family and VMS
high_conf_sorted = high_conf.sort_values(['viral_family', 'VMS'], ascending=[True, False])
all_pairs_family = high_conf_sorted.reset_index(drop=True)

# Divide into 20 groups
groups_family = [
    all_pairs_family.iloc[i*n_per_panel:(i+1)*n_per_panel].reset_index(drop=True)
    for i in range(n_panels)
]

fig, axes = plt.subplots(4, 5, figsize=(25, 24))
fig.suptitle(f'All {len(all_pairs)} Pairs Grouped by Viral Family',
             fontsize=18, fontweight='bold', y=0.998)

axes = axes.flatten()

family_colors = {
    'Arboviruses': '#e74c3c',              # Red
    'Enteric Viruses': '#3498db',          # Blue
    'Hemorrhagic Fever Viruses': '#f39c12',# Orange
    'Hepatotropic Viruses': '#1abc9c',     # Teal
    'Herpesviruses': '#9b59b6',            # Purple
    'Oncogenic': '#e67e22',                # Dark Orange
    'Poxviruses': '#34495e',               # Dark Gray
    'Respiratory Viruses': '#16a085',      # Sea Green
    'Retroviruses': '#c0392b',             # Dark Red
    'Unmapped': '#95a5a6'                  # Light Gray
}

for panel_idx, (ax, group) in enumerate(zip(axes, groups_family)):
    colors = [family_colors.get(fam, '#95a5a6') for fam in group['viral_family']]

    y_positions = np.arange(len(group))
    ax.barh(y_positions, group['VMS'], height=0.85,
            color=colors, edgecolor='black', linewidth=0.3)

    # Add value labels
    for i, (idx, row) in enumerate(group.iterrows()):
        ax.text(row['VMS'] + 0.005, i, f"{row['VMS']:.3f}",
                va='center', fontsize=6, fontweight='bold')

    # Labels
    ax.set_yticks(y_positions)
    labels = [f"[{row['viral_family']}] {row['pair_label']}"
              for _, row in group.iterrows()]
    ax.set_yticklabels(labels, fontsize=6)

    # Add separators between families
    current_family = None
    for i, (_, row) in enumerate(group.iterrows()):
        if row['viral_family'] != current_family and current_family is not None:
            ax.axhline(i - 0.5, color='white', linewidth=2, linestyle='-')
        current_family = row['viral_family']

    ax.set_xlabel('VMS', fontsize=11, fontweight='bold')
    ax.set_title(f'Panel {panel_idx+1}', fontsize=12, fontweight='bold')
    ax.set_xlim(0.55, all_pairs_family['VMS'].max() + 0.02)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

# Add legend (only on last panel)
legend_patches = [mpatches.Patch(color=color, label=family)
                  for family, color in family_colors.items()
                  if family in all_pairs_family['viral_family'].unique()]
axes[n_panels-1].legend(handles=legend_patches, loc='lower right', fontsize=9,
                        frameon=True, shadow=True, title='Families', title_fontsize=10)

plt.tight_layout()
plt.savefig('fig_12_bars_viral_families_all279.png', dpi=600, bbox_inches='tight')
print("✓ Saved: fig_12_bars_viral_families_all279.png")
plt.close()

# ============================================================================
# FIGURE 4: ALL Pairs - Stacked Bars in 20 panels
# ============================================================================

print(f"Generating Figure 4: Stacked bars in {n_panels} panels...")

# Calculate weighted contributions
all_pairs['F_sim_weighted'] = all_pairs['F_sim'] * 0.50
all_pairs['S_sim_weighted'] = all_pairs['S_sim'] * 0.50

# Divide into 20 groups
groups_stacked = [
    all_pairs.iloc[i*n_per_panel:(i+1)*n_per_panel].reset_index(drop=True)
    for i in range(n_panels)
]

fig, axes = plt.subplots(4, 5, figsize=(25, 24))
fig.suptitle(f'All {len(all_pairs)} Pairs: VMS Composition (0.50×F_sim + 0.50×S_sim)',
             fontsize=18, fontweight='bold', y=0.998)

axes = axes.flatten()

for panel_idx, (ax, group) in enumerate(zip(axes, groups_stacked)):
    start_rank = panel_idx * n_per_panel + 1

    y_positions = np.arange(len(group))

    # Create stacked bars
    ax.barh(y_positions, group['F_sim_weighted'], height=0.85,
            label='F_sim × 0.50' if panel_idx == 0 else '',
            color='#3498db', edgecolor='black', linewidth=0.3)
    ax.barh(y_positions, group['S_sim_weighted'], height=0.85,
            left=group['F_sim_weighted'],
            label='S_sim × 0.50' if panel_idx == 0 else '',
            color='#e74c3c', edgecolor='black', linewidth=0.3)

    # Add VMS value at end
    for i, (idx, row) in enumerate(group.iterrows()):
        total = row['F_sim_weighted'] + row['S_sim_weighted']
        ax.text(total + 0.005, i, f"{row['VMS']:.3f}",
                va='center', fontsize=6, fontweight='bold')

    # Labels
    ax.set_yticks(y_positions)
    labels = [f"{start_rank+i}. {row['pair_label']}" for i, (_, row) in enumerate(group.iterrows())]
    ax.set_yticklabels(labels, fontsize=6)

    ax.set_xlabel('VMS Composition', fontsize=11, fontweight='bold')
    ax.set_title(f'Panel {panel_idx+1}: Ranks {start_rank}-{start_rank+len(group)-1}',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, all_pairs['VMS'].max() + 0.05)
    if panel_idx == 0:
        ax.legend(loc='lower right', fontsize=10, frameon=True, shadow=True)
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.invert_yaxis()

plt.tight_layout()
plt.savefig('fig_13_bars_stacked_all279.png', dpi=600, bbox_inches='tight')
print("✓ Saved: fig_13_bars_stacked_all279.png")
plt.close()

# ============================================================================
# CREATE REPORT
# ============================================================================

report = f"""# Report 09: Bar Chart Analysis - High-Confidence Pairs

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview

**Cutoff:** VMS ≥ {cutoff}
**High-confidence pairs:** {len(high_conf)} ({len(high_conf)/len(df)*100:.1f}%)

**Visualization Format:** Each figure divided into {n_panels} panels (4×5 layout: 4 rows, 5 columns) with ~{n_per_panel} pairs per panel for optimal legibility

## Summary Statistics

| Metric | Mean | Std | Min | Max |
|--------|------|-----|-----|-----|
| VMS | {high_conf['VMS'].mean():.3f} | {high_conf['VMS'].std():.3f} | {high_conf['VMS'].min():.3f} | {high_conf['VMS'].max():.3f} |
| F_sim | {high_conf['F_sim'].mean():.3f} | {high_conf['F_sim'].std():.3f} | {high_conf['F_sim'].min():.3f} | {high_conf['F_sim'].max():.3f} |
| S_sim | {high_conf['S_sim'].mean():.3f} | {high_conf['S_sim'].std():.3f} | {high_conf['S_sim'].min():.3f} | {high_conf['S_sim'].max():.3f} |

## Top 10 Pairs

| Rank | Viral Protein | Human Protein | VMS | F_sim | S_sim |
|------|---------------|---------------|-----|-------|-------|
"""

top10 = high_conf.nlargest(10, 'VMS')
for rank, (_, row) in enumerate(top10.iterrows(), 1):
    report += f"| {rank} | {row['virus_clean']} | {row['human_clean']} | {row['VMS']:.3f} | {row['F_sim']:.3f} | {row['S_sim']:.3f} |\n"

report += f"""
## Visual Outputs

### Figure 10: ALL {len(high_conf)} Pairs - Horizontal Bars ({n_panels} panels)
- **Layout:** 4×5 panels (4 rows, 5 columns), ~{n_per_panel} pairs per panel
- **Bar length:** VMS value
- **Color:** Gradient (Red-Yellow-Green) by VMS
- **Labels:** Global rank + Virus → Human pair
- **Resolution:** 600 DPI, 25×24 inches

### Figure 11: ALL {len(high_conf)} Pairs - Component Comparison ({n_panels} panels)
- **Layout:** 4×5 panels (4 rows, 5 columns), ~{n_per_panel} pairs per panel
- **Bars per pair:** 3 (F_sim, S_sim, VMS)
- **Colors:** Blue (F_sim), Red (S_sim), Green (VMS)
- **Purpose:** Compare metrics side-by-side for all pairs
- **Resolution:** 600 DPI, 25×24 inches

### Figure 12: ALL {len(high_conf)} Pairs - Viral Families ({n_panels} panels)
- **Layout:** 4×5 panels (4 rows, 5 columns), ~{n_per_panel} pairs per panel
- **Sorting:** By viral family, then by VMS
- **Colors:** Family-specific colors
- **Separators:** White lines between families
- **Legend:** Panel {n_panels} shows family color coding
- **Resolution:** 600 DPI, 25×24 inches

### Figure 13: ALL {len(high_conf)} Pairs - Stacked Composition ({n_panels} panels)
- **Layout:** 4×5 panels (4 rows, 5 columns), ~{n_per_panel} pairs per panel
- **Components:** Stacked bars showing F_sim×0.50 (blue) + S_sim×0.50 (red)
- **Total:** Sum = VMS
- **Purpose:** Visualize VMS formula for all pairs
- **Resolution:** 600 DPI, 25×24 inches

## Viral Family Distribution

| Family | Count | Percentage |
|--------|-------|------------|
"""

family_counts = high_conf['viral_family'].value_counts()
for family, count in family_counts.items():
    report += f"| {family} | {count} | {count/len(high_conf)*100:.1f}% |\n"

report += f"""
## Data Output

- **results_08_highconfidence_pairs.csv**
  - All {len(high_conf)} high-confidence pairs
  - Includes: VMS, F_sim, S_sim, viral_family, cleaned names

---

**Design rationale:** Each of the 4 figures is divided into {n_panels} panels (4×5 layout: 4 rows, 5 columns) with ~{n_per_panel} pairs per panel to ensure all {len(high_conf)} high-confidence pairs are visible and legible at 600 DPI resolution. This layout balances comprehensive coverage with readability, allowing detailed examination of individual pairs while maintaining professional publication quality.
"""

with open('report_09_bar_analysis.md', 'w') as f:
    f.write(report)
print("✓ Saved: report_09_bar_analysis.md")

# ============================================================================
# CONSOLE SUMMARY
# ============================================================================

print()
print("="*70)
print(f"BAR VISUALIZATION SUMMARY ({n_panels}-PANEL FORMAT)")
print("="*70)
print(f"High-confidence pairs: {len(high_conf)} (VMS ≥ {cutoff})")
print(f"Panels per figure: {n_panels} (4×5 layout: 4 rows, 5 columns)")
print(f"Pairs per panel: ~{n_per_panel}")
print()
print("Top 5 Pairs:")
top5 = high_conf.nlargest(5, 'VMS')
for rank, (_, row) in enumerate(top5.iterrows(), 1):
    print(f"  {rank}. {row['virus_clean']} → {row['human_clean']}")
    print(f"     VMS={row['VMS']:.3f} (F_sim={row['F_sim']:.3f}, S_sim={row['S_sim']:.3f})")
print()
print("="*70)
print("SCRIPT 09 COMPLETED")
print("Files generated:")
print(f"  - fig_10_bars_all279.png ({n_panels} panels, 4×5 layout, 25×24 in)")
print(f"  - fig_11_bars_components_all279.png ({n_panels} panels, 4×5 layout, 25×24 in)")
print(f"  - fig_12_bars_viral_families_all279.png ({n_panels} panels, 4×5 layout, 25×24 in)")
print(f"  - fig_13_bars_stacked_all279.png ({n_panels} panels, 4×5 layout, 25×24 in)")
print("  - results_08_highconfidence_pairs.csv")
print("  - report_09_bar_analysis.md")
print("="*70)
