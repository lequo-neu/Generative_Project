"""
Generate publication-quality figures from COCO 500 evaluation data.
Run: python3 outputs/figures/gen_coco_figures.py
Saves all figures to outputs/figures/
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# ── Load eval data ────────────────────────────────────────────────────────────
EVAL_PATH = Path('/Users/kevin/Documents/GitHub/Python/VESKL/11.DAE/NEU/NEU_IE7615/Prj/Generative_Project/outputs/evalutation/eval_results_coco.json')
if not EVAL_PATH.exists():
    # try alternate spelling
    EVAL_PATH = Path(str(EVAL_PATH).replace('evalutation','evaluation'))
OUT_DIR   = Path('/Users/kevin/Documents/GitHub/Python/VESKL/11.DAE/NEU/NEU_IE7615/Prj/Generative_Project/outputs/figures')
OUT_DIR.mkdir(exist_ok=True)

with open(EVAL_PATH) as f:
    ev = json.load(f)

M  = ev['METRICS']
BD = ev['BEAM_DATA']
TD = ev['TEMP_DATA']

# Style
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 150,
})

COLORS = {
    'Frozen':  '#2563eb',
    'FT-4L':   '#d97706',
    'LoRA':    '#e86020',
    'beam':    '#1e40af',
    'greedy':  '#6b7280',
    'nucleus': '#7c3aed',
}

# ── Figure 1: All-model CIDEr grouped bar chart ───────────────────────────────
configs  = list(M.keys())
models   = ['Frozen', 'FT-4L', 'LoRA']
strats   = ['greedy', 'beam', 'nucleus']
strat_colors = {'greedy': '#94a3b8', 'beam': '#2563eb', 'nucleus': '#7c3aed'}

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(models))
w = 0.25
for j, strat in enumerate(strats):
    vals = [M[f'{m}/{strat}'].get('CIDEr', 0) for m in models]
    bars = ax.bar(x + (j-1)*w, vals, w, label=strat.capitalize(),
                  color=strat_colors[strat], alpha=0.88, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='600')

ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('CIDEr Score', fontsize=12)
ax.set_title('CIDEr by Model and Decoding Strategy\n(COCO val2017, 500 images)', fontsize=13, fontweight='bold')
ax.legend(title='Strategy', fontsize=10)
ax.set_ylim(0, 1.15)
ax.axhline(y=0.9364, color='#e86020', linestyle=':', alpha=0.5, linewidth=1.2, label='Best (FT-4L/beam)')
plt.tight_layout()
plt.savefig(OUT_DIR / 'coco_cider_by_model_strategy.png', bbox_inches='tight')
plt.close()
print("Saved: coco_cider_by_model_strategy.png")

# ── Figure 2: BLEU-4 grouped bar chart ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for j, strat in enumerate(strats):
    vals = [M[f'{m}/{strat}'].get('BLEU-4', 0) for m in models]
    bars = ax.bar(x + (j-1)*w, vals, w, label=strat.capitalize(),
                  color=strat_colors[strat], alpha=0.88, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8.5, fontweight='600')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylabel('BLEU-4 Score', fontsize=12)
ax.set_title('BLEU-4 by Model and Decoding Strategy\n(COCO val2017, 500 images)', fontsize=13, fontweight='bold')
ax.legend(title='Strategy', fontsize=10)
ax.set_ylim(0, 0.38)
plt.tight_layout()
plt.savefig(OUT_DIR / 'coco_bleu4_by_model_strategy.png', bbox_inches='tight')
plt.close()
print("Saved: coco_bleu4_by_model_strategy.png")

# ── Figure 3: Multi-metric radar chart ───────────────────────────────────────
metrics_radar = ['BLEU-1', 'BLEU-4', 'METEOR', 'CIDEr', 'ROUGE-L']
scale = {'BLEU-1':1.0, 'BLEU-4':0.4, 'METEOR':1.0, 'CIDEr':1.0, 'ROUGE-L':1.0}

fig, axes = plt.subplots(1, 3, figsize=(14, 5), subplot_kw=dict(polar=True))
fig.suptitle('Multi-Metric Radar: Best Strategy per Model\n(COCO val2017, 500 images)',
             fontsize=13, fontweight='bold', y=1.02)

for ax_i, (model, col) in enumerate(zip(models, ['#2563eb','#d97706','#e86020'])):
    ax = axes[ax_i]
    # Use beam strategy (best for all models)
    key = f'{model}/beam'
    vals = [M[key].get(met, 0) / scale[met] for met in metrics_radar]
    vals += [vals[0]]  # close polygon

    angles = np.linspace(0, 2*np.pi, len(metrics_radar), endpoint=False).tolist()
    angles += angles[:1]

    ax.plot(angles, vals, 'o-', color=col, linewidth=2)
    ax.fill(angles, vals, alpha=0.18, color=col)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics_radar, size=9)
    ax.set_ylim(0, 1)
    ax.set_title(f'{model} / Beam', fontsize=11, fontweight='bold', color=col, pad=12)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['','','',''], size=7)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT_DIR / 'coco_radar_all_models.png', bbox_inches='tight')
plt.close()
print("Saved: coco_radar_all_models.png")

# ── Figure 4: Beam width sensitivity ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle('Beam Width Sensitivity (LoRA model, COCO val2017)',
             fontsize=13, fontweight='bold')

ax1.plot(BD['labels'], BD['cider'], 'o-', color='#e86020', linewidth=2.2,
         markersize=7, markerfacecolor='white', markeredgewidth=2)
ax1.axvline(x=5, color='#2563eb', linestyle='--', alpha=0.6, linewidth=1.5, label='Peak w=5')
ax1.fill_between(BD['labels'], BD['cider'], alpha=0.1, color='#e86020')
ax1.set_xlabel('Beam Width (w)', fontsize=11)
ax1.set_ylabel('CIDEr Score', fontsize=11)
ax1.set_title('CIDEr vs Beam Width', fontsize=11)
ax1.legend(fontsize=9)
for xi, yi in zip(BD['labels'], BD['cider']):
    ax1.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=8.5)

ax2.plot(BD['labels'], BD['bleu4'], 's-', color='#2563eb', linewidth=2.2,
         markersize=7, markerfacecolor='white', markeredgewidth=2)
ax2.axvline(x=5, color='#e86020', linestyle='--', alpha=0.6, linewidth=1.5, label='Peak w=5')
ax2.fill_between(BD['labels'], BD['bleu4'], alpha=0.1, color='#2563eb')
ax2.set_xlabel('Beam Width (w)', fontsize=11)
ax2.set_ylabel('BLEU-4 Score', fontsize=11)
ax2.set_title('BLEU-4 vs Beam Width', fontsize=11)
ax2.legend(fontsize=9)
for xi, yi in zip(BD['labels'], BD['bleu4']):
    ax2.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=8.5)

plt.tight_layout()
plt.savefig(OUT_DIR / 'coco_beam_sensitivity.png', bbox_inches='tight')
plt.close()
print("Saved: coco_beam_sensitivity.png")

# ── Figure 5: Temperature sensitivity ────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
fig.suptitle('Temperature Sensitivity (LoRA / Nucleus, COCO val2017)',
             fontsize=13, fontweight='bold')

ax1.plot(TD['labels'], TD['cider'], 'o-', color='#7c3aed', linewidth=2.2,
         markersize=7, markerfacecolor='white', markeredgewidth=2)
ax1.axvline(x=0.3, color='#16a34a', linestyle='--', alpha=0.7, linewidth=1.5, label='Best t=0.3')
ax1.fill_between(TD['labels'], TD['cider'], alpha=0.1, color='#7c3aed')
ax1.set_xlabel('Temperature (t)', fontsize=11)
ax1.set_ylabel('CIDEr Score', fontsize=11)
ax1.set_title('CIDEr vs Temperature', fontsize=11)
ax1.legend(fontsize=9)
for xi, yi in zip(TD['labels'], TD['cider']):
    ax1.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=8.5)

ax2.plot(TD['labels'], TD['distinct2'], '^-', color='#16a34a', linewidth=2.2,
         markersize=7, markerfacecolor='white', markeredgewidth=2)
ax2.fill_between(TD['labels'], TD['distinct2'], alpha=0.1, color='#16a34a')
ax2.set_xlabel('Temperature (t)', fontsize=11)
ax2.set_ylabel('Distinct-2', fontsize=11)
ax2.set_title('Lexical Diversity vs Temperature', fontsize=11)
ax2.annotate('Accuracy-Diversity Tradeoff', xy=(0.7, 0.7), fontsize=9,
             color='gray', style='italic')
for xi, yi in zip(TD['labels'], TD['distinct2']):
    ax2.annotate(f'{yi:.3f}', (xi, yi), textcoords='offset points',
                 xytext=(0, 8), ha='center', fontsize=8.5)

plt.tight_layout()
plt.savefig(OUT_DIR / 'coco_temperature_sensitivity.png', bbox_inches='tight')
plt.close()
print("Saved: coco_temperature_sensitivity.png")

# ── Figure 6: Summary table heatmap ──────────────────────────────────────────
metric_keys = ['BLEU-1', 'BLEU-4', 'METEOR', 'CIDEr', 'ROUGE-L']
row_labels  = list(M.keys())  # 9 configs
data_mat    = np.array([[M[k].get(met, 0) for met in metric_keys] for k in row_labels])

fig, ax = plt.subplots(figsize=(9, 6))
im = ax.imshow(data_mat, aspect='auto', cmap='YlOrRd')
ax.set_xticks(range(len(metric_keys)))
ax.set_xticklabels(metric_keys, fontsize=11, fontweight='600')
ax.set_yticks(range(len(row_labels)))
ax.set_yticklabels(row_labels, fontsize=10)
for i in range(data_mat.shape[0]):
    for j in range(data_mat.shape[1]):
        val = data_mat[i, j]
        text_color = 'white' if val > data_mat[:, j].mean() + 0.5*data_mat[:, j].std() else '#1e2433'
        ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                fontsize=9, color=text_color, fontweight='600')
plt.colorbar(im, ax=ax, shrink=0.8, label='Score')
ax.set_title('Evaluation Heatmap: All Configs x All Metrics\n(COCO val2017, 500 images)',
             fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'coco_metrics_heatmap.png', bbox_inches='tight')
plt.close()
print("Saved: coco_metrics_heatmap.png")

# ── Figure 7: Model comparison bar (best strategy each) ──────────────────────
best_per_model = {m: max([M[f'{m}/{s}'] for s in strats],
                         key=lambda x: x.get('CIDEr',0)) for m in models}

fig, axes = plt.subplots(1, len(metric_keys), figsize=(13, 4))
fig.suptitle('Best-Strategy Metrics per Model (COCO val2017, 500 images)',
             fontsize=12, fontweight='bold')
for j, met in enumerate(metric_keys):
    ax = axes[j]
    vals = [best_per_model[m].get(met, 0) for m in models]
    cols = [COLORS[m] for m in models]
    bars = ax.bar(models, vals, color=cols, alpha=0.85, edgecolor='white')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='700')
    ax.set_title(met, fontsize=11, fontweight='600')
    ax.set_xticklabels(models, fontsize=9)
    ax.tick_params(axis='y', labelsize=8)
    ax.set_ylim(0, max(vals)*1.25 + 0.05)
plt.tight_layout()
plt.savefig(OUT_DIR / 'coco_best_model_comparison.png', bbox_inches='tight')
plt.close()
print("Saved: coco_best_model_comparison.png")

print(f"\nAll figures saved to {OUT_DIR}")
print("Files created:")
for f in sorted(OUT_DIR.glob('coco_*.png')):
    print(f"  {f.name}")
