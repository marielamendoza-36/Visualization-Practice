"""
Word Recognition Study - Complete Data Analysis & Visualization

This script analyzes children's word recognition performance across age groups
and generates four comprehensive visualizations:
    1. Grouped bar chart - Word difficulty comparison by age
    2. Line plot - Recognition development trajectories
    3. Horizontal bar chart - Variance analysis ranking
    4. Interactive heatmap - Complete word × age matrix

Requirements:
    - pandas
    - matplotlib
    - seaborn
    - numpy
    - plotly

Usage:
    python analysis.py

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px

# ============================================================
# CONFIGURATION
# ============================================================

# Visualization settings
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 150

# File paths
DATA_FILE = 'vocabulary_dataset.csv'
OUTPUT_FILES = {
    'plot1': 'plot1_word_comparison_by_age.png',
    'plot2': 'plot2_recognition_development.png',
    'plot3': 'plot3_variance_ranking.png',
    'plot4': 'plot4_heatmap_interactive.html'
}


# ============================================================
# DATA LOADING
# ============================================================

print("\n" + "=" * 60)
print("WORD RECOGNITION STUDY - DATA ANALYSIS")
print("=" * 60 + "\n")

df = pd.read_csv(DATA_FILE)

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Words analyzed: {df['targetWord'].nunique()}")
print(f"Age groups: {sorted(df['age_group'].unique())}\n")


# ============================================================
# PLOT 1: Grouped Bar Chart - Word Difficulty by Age
# ============================================================

print("Generating Plot 1: Grouped bar chart...")

# Calculate word statistics
word_stats = df.groupby('targetWord')['pc'].agg(['mean', 'var']).reset_index()
word_stats.columns = ['targetWord', 'avg_pc', 'variance']

# Select representative words from each category
top_hardest = word_stats.nsmallest(5, 'avg_pc')['targetWord'].tolist()
top_easiest = word_stats.nlargest(5, 'avg_pc')['targetWord'].tolist()
top_variable = word_stats.nlargest(5, 'variance')['targetWord'].tolist()
selected_words = list(dict.fromkeys(top_hardest + top_easiest + top_variable))

# Prepare data
plot_data = df[df['targetWord'].isin(selected_words)].copy()
plot_data['age_group'] = plot_data['age_group'].astype(str)
age_order = [str(age) for age in sorted(df['age_group'].unique())]

# Create plot
fig, ax = plt.subplots(figsize=(16, 7))
sns.barplot(
    data=plot_data,
    x='targetWord',
    y='pc',
    hue='age_group',
    hue_order=age_order,
    palette='coolwarm',
    ax=ax,
    order=selected_words
)

# Formatting
ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, alpha=0.7)
ax.set_xlabel('Word', fontsize=12, fontweight='bold')
ax.set_ylabel('Proportion Correct', fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.1)
ax.set_title(
    'Word Recognition Performance by Age Group\n'
    'Top 5 Hardest · Top 5 Easiest · Top 5 Most Variable',
    fontsize=14, fontweight='bold', pad=15
)
plt.xticks(rotation=30, ha='right', fontsize=10)
ax.legend(title='Age Group', bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)

# Add category dividers
for x_pos in [4.5, 9.5]:
    ax.axvline(x=x_pos, color='gray', linestyle=':', linewidth=1.2, alpha=0.6)

ax.text(2, 1.07, 'Hardest', ha='center', fontsize=10, color='#c0392b', style='italic', fontweight='bold')
ax.text(7, 1.07, 'Easiest', ha='center', fontsize=10, color='#27ae60', style='italic', fontweight='bold')
ax.text(12, 1.07, 'Most Variable', ha='center', fontsize=10, color='#2980b9', style='italic', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_FILES['plot1'], bbox_inches='tight')
plt.close()
print(f"✓ Plot 1 saved: {OUTPUT_FILES['plot1']}\n")


# ============================================================
# PLOT 2: Line Plot - Recognition Development Over Age
# ============================================================

print("Generating Plot 2: Line plot...")

# Select diverse words for trajectory analysis
words_to_plot = [
    'omelet', 'headdress', 'telescope', 'marshmallow', 'aloe',
    'bamboo', 'caramel', 'ant', 'ball', 'carrot'
]
subset = df[df['targetWord'].isin(words_to_plot)].copy()

# Create plot
fig, ax = plt.subplots(figsize=(12, 7))
for word in words_to_plot:
    word_data = subset[subset['targetWord'] == word].sort_values('age_group')
    ax.plot(
        word_data['age_group'],
        word_data['pc'],
        marker='o',
        linewidth=2.5,
        markersize=7,
        label=word,
        alpha=0.85
    )

# Formatting
ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Chance Level')
ax.set_xlabel('Age Group (years)', fontsize=12, fontweight='bold')
ax.set_ylabel('Proportion Correct', fontsize=12, fontweight='bold')
ax.set_title('Word Recognition Development Across Age Groups', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(sorted(df['age_group'].unique()))
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_FILES['plot2'], bbox_inches='tight')
plt.close()
print(f"✓ Plot 2 saved: {OUTPUT_FILES['plot2']}\n")


# ============================================================
# PLOT 3: Horizontal Bar Chart - Variance Analysis
# ============================================================

print("Generating Plot 3: Variance analysis...")

# Calculate variance per word
word_variance = (
    df.groupby('targetWord')['pc']
    .var()
    .sort_values(ascending=False)
    .reset_index()
)
word_variance.columns = ['targetWord', 'variance']
top_20_variance = word_variance.head(20)

# Create plot
fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(
    top_20_variance['targetWord'],
    top_20_variance['variance'],
    color=sns.color_palette("flare", n_colors=20)
)

# Add value labels
for bar, value in zip(bars, top_20_variance['variance']):
    ax.text(
        bar.get_width() + 0.002,
        bar.get_y() + bar.get_height() / 2,
        f'{value:.3f}',
        va='center',
        fontsize=9,
        fontweight='bold'
    )

# Formatting
ax.set_xlabel('Variance in Proportion Correct', fontsize=12, fontweight='bold')
ax.set_ylabel('Word', fontsize=12, fontweight='bold')
ax.set_title(
    'Top 20 Words with Highest Recognition Variance\n'
    'Higher variance indicates greater change across age groups',
    fontsize=14, fontweight='bold', pad=15
)
ax.invert_yaxis()

plt.tight_layout()
plt.savefig(OUTPUT_FILES['plot3'], bbox_inches='tight')
plt.close()
print(f"✓ Plot 3 saved: {OUTPUT_FILES['plot3']}\n")


# ============================================================
# PLOT 4: Interactive Heatmap - Complete Word × Age Matrix
# ============================================================

print("Generating Plot 4: Interactive heatmap...")

# Prepare heatmap matrix
heatmap_matrix = df.pivot_table(
    index='targetWord',
    columns='age_group',
    values='pc',
    aggfunc='mean'
)

# Sort by difficulty
heatmap_matrix['_avg'] = heatmap_matrix.mean(axis=1)
heatmap_matrix = heatmap_matrix.sort_values('_avg', ascending=True).drop(columns='_avg')
heatmap_matrix.columns = [f'Age {int(col)}' for col in heatmap_matrix.columns]

# Create interactive heatmap
fig = px.imshow(
    heatmap_matrix,
    color_continuous_scale='RdYlGn',
    zmin=0,
    zmax=1,
    aspect='auto',
    title=(
        '<b>Word Recognition Performance Across Age Groups</b><br>'
        '<sup>Sorted by difficulty (hardest at top) · Hover for details · Scroll to zoom</sup>'
    ),
    labels={'x': 'Age Group', 'y': 'Word', 'color': 'Proportion Correct'}
)

# Customize layout
fig.update_layout(
    height=1800,
    width=900,
    title_font_size=14,
    title_x=0.5,
    font=dict(family="Arial, sans-serif", size=11),
    margin=dict(t=180, l=10, r=10, b=10),  # Larger top margin to prevent overlap
    coloraxis_colorbar=dict(
        title='Recognition Score',
        tickvals=[0, 0.25, 0.5, 0.75, 1.0],
        ticktext=['0.0 (Hardest)', '0.25', '0.5 (Chance)', '0.75', '1.0 (Easiest)'],
        len=0.7,
        thickness=20
    ),
    xaxis=dict(
        side='top',
        tickfont=dict(size=12),
        title=dict(text='<b>Age Group</b>', font=dict(size=13))
    ),
    yaxis=dict(
        tickfont=dict(size=10),
        title=dict(text='<b>Word</b>', font=dict(size=13))
    ),
    plot_bgcolor='white',
    paper_bgcolor='#f8f9fa'
)

# Customize hover
fig.update_traces(
    hovertemplate=(
        '<b>Word:</b> %{y}<br>'
        '<b>Age Group:</b> %{x}<br>'
        '<b>Recognition Score:</b> %{z:.3f}<br>'
        '<extra></extra>'
    )
)

fig.write_html(OUTPUT_FILES['plot4'])
print(f"✓ Plot 4 saved: {OUTPUT_FILES['plot4']}")
print("  (Open in browser or upload to Tiiny.host)\n")


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)

# Difficulty ranking
word_difficulty = df.groupby('targetWord')['pc'].mean().sort_values().reset_index()
word_difficulty.columns = ['targetWord', 'avg_pc']

print(f"\nTotal words analyzed: {len(word_difficulty)}")
print(f"Age range: {df['age_group'].min()} - {df['age_group'].max()} years")

print("\n" + "-" * 60)
print("TOP 5 HARDEST WORDS (Lowest Recognition)")
print("-" * 60)
for idx, row in word_difficulty.head(5).iterrows():
    print(f"  {row['targetWord']:20s} {row['avg_pc']:.3f}")

print("\n" + "-" * 60)
print("TOP 5 EASIEST WORDS (Highest Recognition)")
print("-" * 60)
for idx, row in word_difficulty.tail(5).iterrows():
    print(f"  {row['targetWord']:20s} {row['avg_pc']:.3f}")

print("\n" + "-" * 60)
print("TOP 5 MOST VARIABLE WORDS (Greatest Age Effect)")
print("-" * 60)
for idx, row in top_20_variance.head(5).iterrows():
    print(f"  {row['targetWord']:20s} {row['variance']:.4f}")

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
print("\nOutput files:")
for key, filename in OUTPUT_FILES.items():
    print(f"  • {filename}")
print("=" * 60 + "\n")