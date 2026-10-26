"""Plot the two frozen learning trajectories without selecting checkpoints."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
summary = json.loads((root / 'formal-audit-v1/summary.json').read_text())
scores = {(r['seed'], r['arm'], r['family']): r['exact_rate'] * 100
          for r in summary['seed_records'] if r['budget'] == 65536}
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6), sharey=True)
for axis, family, title in zip(axes, ('composition', 'shifted'),
                               ('New motif compositions', 'Shifted programs')):
    for kind, color, label in (('full', '#a34b2b', 'Full prefix'), ('local', '#245c88', 'Local context')):
        arms = [f'{kind}_frozen', f'{kind}_after_8', f'{kind}_after_16', f'{kind}_trained']
        values = np.array([[scores[(seed, arm, family)] for arm in arms] for seed in summary['seeds']])
        for row in values:
            axis.plot([0, 8, 16, 24], row, color=color, alpha=.13, linewidth=.6)
        axis.plot([0, 8, 16, 24], values.mean(axis=0), 'o-', color=color, linewidth=2, label=label)
    uniform = np.mean([scores[(seed, 'uniform', family)] for seed in summary['seeds']])
    axis.axhline(uniform, color='#555555', linestyle='--', linewidth=1.1, label='Uniform search')
    axis.set(title=title, xlabel='Experience tasks attempted', xticks=[0, 8, 16, 24], ylim=(-2, 102))
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', alpha=.15)
axes[0].set_ylabel('Exact success (%)')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, .89))
fig.savefig(root / 'learning-progress.pdf', metadata={'Title': 'Study J: full and local context transfer'})
plt.close(fig)
