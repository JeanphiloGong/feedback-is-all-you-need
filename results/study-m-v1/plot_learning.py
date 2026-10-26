"""Plot every fixed learning curve and the paired structured-world primary."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
summary = json.loads((root / 'formal-audit-v1/summary.json').read_text())
arms = (
    ('small_immediate', 'Small, immediate', '#ad3151', '-', 'o'),
    ('small_delayed', 'Small, delayed', '#ad3151', '--', 's'),
    ('small_frozen', 'Small, frozen', '#ad3151', ':', None),
    ('large_immediate', 'Large, immediate', '#246eaa', '-', '^'),
    ('large_delayed', 'Large, delayed', '#246eaa', '--', 'v'),
    ('large_frozen', 'Large, frozen', '#246eaa', ':', None),
    ('observed_table', 'Observed table', '#388057', '-.', 'd'),
    ('uniform', 'Uniform actions', '#666666', ':', 'x'),
)
plt.rcParams.update({'font.size': 9})
fig = plt.figure(figsize=(7.2, 6.0))
grid = fig.add_gridspec(2, 2)
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[1, :])]
for axis, family in zip(axes, ('structured', 'random')):
    for arm, label, color, style, marker in arms:
        row = next(r for r in summary['summaries'] if (r['family'], r['arm']) == (family, arm))
        axis.plot(row['checkpoints'], 100 * np.array(row['success_rates']), label=label,
                  color=color, linestyle=style, marker=marker, linewidth=1.5, markersize=3.5)
    axis.set(title=family.capitalize() + ' worlds', xlabel='Main interactions',
             ylabel='Held-out goal success (%)', ylim=(-2, 102), xticks=[0, 128, 256, 512])
    axis.grid(axis='y', alpha=.2)
axis = axes[2]
primary = summary['primary']
delta = 100 * np.array(primary['differences'])
axis.axhline(0, color='#777777', linewidth=.8)
axis.scatter(np.arange(32), delta, s=14, color='#ad3151')
mu = 100 * primary['mean']
lo, hi = 100 * np.array(primary['ci95'])
axis.errorbar(41, mu, yerr=[[mu - lo], [hi - mu]], fmt='D', color='#222222', capsize=4, markersize=5)
axis.set(title='Structured AUC: small - large', xlabel='Environment seed',
         ylabel='AUC difference (percentage points)', xticks=[0, 15, 31, 41],
         xticklabels=['15000', '15015', '15031', 'Mean'])
axis.grid(axis='y', alpha=.2)
for axis in axes:
    axis.spines[['top', 'right']].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, .91))
fig.savefig(root / 'learning-and-contrasts.pdf', metadata={
    'Title': 'Study M: retained predictive weights and held-out goal success',
    'CreationDate': None, 'ModDate': None})
plt.close(fig)
