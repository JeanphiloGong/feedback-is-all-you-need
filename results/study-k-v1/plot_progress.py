"""Show every fixed probe checkpoint under both stream orders."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
summary = json.loads((root / 'formal-audit-v1/summary.json').read_text())
scores = {(row['seed'], row['order'], row['arm'], row['checkpoint'], row['family']): row['exact_rate'] * 100
          for row in summary['probe_seed_records'] if row['budget'] == 65536}
fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.5), sharey=True, sharex=True)
for row_index, (order, label) in enumerate((('composition_first', 'Composition then shifted'),
                                          ('shifted_first', 'Shifted then composition'))):
    for column, (family, title) in enumerate((('composition', 'Composition probes'), ('shifted', 'Shifted probes'))):
        axis = axes[row_index, column]
        for arm, color, style, arm_label in (('online', '#245c88', 'o-', 'Online accepted solutions'),
                                            ('replay', '#a34b2b', 's-', 'Initial-corpus replay'),
                                            ('frozen', '#666666', ':', 'Frozen after initial experience'),
                                            ('uniform', '#39733b', '--', 'Uniform search')):
            values = np.array([[scores[(seed, order, arm, checkpoint, family)] for checkpoint in (0, 8, 16)]
                               for seed in summary['seeds']])
            if arm == 'online':
                for trajectory in values:
                    axis.plot([0, 8, 16], trajectory, color=color, alpha=.12, linewidth=.6)
            axis.plot([0, 8, 16], values.mean(axis=0), style, color=color, linewidth=1.8,
                      markersize=4, label=arm_label)
        axis.axvline(8, color='#aaaaaa', linewidth=.8, linestyle=':')
        axis.set(title=f'{label}\n{title}', xticks=[0, 8, 16], ylim=(-2, 102))
        axis.spines[['top', 'right']].set_visible(False)
        axis.grid(axis='y', alpha=.15)
        if column == 0:
            axis.set_ylabel('Exact success (%)')
        if row_index == 1:
            axis.set_xlabel('Stream tasks attempted (family changes after 8)')
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, .90))
fig.savefig(root / 'learning-progress.pdf', metadata={'Title': 'Study K: continued learning and untouched transfer probes'})
plt.close(fig)
