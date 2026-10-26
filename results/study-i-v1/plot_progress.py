"""Plot the prespecified small-prior checkpoints from audited formal records."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
summary = json.loads((root / 'formal-audit-v1/summary.json').read_text())
arms = ['small_frozen', 'small_after_8', 'small_after_16', 'small_trained']
attempts = [0, 8, 16, 24]
rows = {(r['seed'], r['family'], r['arm']): 100 * r['exact_rate']
        for r in summary['seed_records'] if r['budget'] == 65536}
plt.rcParams.update({'font.size': 9.5, 'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42})
figure, axes = plt.subplots(1, 2, figsize=(7, 3.25), sharey=True)
for axis, family, title in zip(axes, ['composition', 'shifted'],
                               ['New motif compositions', 'Shifted programs']):
    trajectories = np.array([[rows[(seed, family, arm)] for arm in arms] for seed in summary['seeds']])
    for index, trajectory in enumerate(trajectories):
        axis.plot(attempts, trajectory, color='#808080', alpha=.2, linewidth=.7,
                  label='Individual environments' if index == 0 else None)
    axis.plot(attempts, trajectories.mean(axis=0), color='#1f4d7a', marker='o', linewidth=2,
              label='Small neural prior (mean)')
    uniform = np.mean([rows[(seed, family, 'uniform')] for seed in summary['seeds']])
    axis.axhline(uniform, color='#b15b24', linestyle='--', linewidth=1.5, label='Uniform prior (mean)')
    axis.set(title=title, xlabel='Experience tasks attempted', xticks=attempts, ylim=(-2, 102))
    axis.grid(axis='y', alpha=.15)
axes[0].set_ylabel('Exact success (%)')
handles, labels = axes[0].get_legend_handles_labels()
figure.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=8)
figure.tight_layout(rect=(0, 0, 1, .91))
figure.savefig(root / 'learning-progress.pdf', metadata={
    'Title': 'Study I: transfer across prespecified replay checkpoints',
    'CreationDate': None, 'ModDate': None})
plt.close(figure)
