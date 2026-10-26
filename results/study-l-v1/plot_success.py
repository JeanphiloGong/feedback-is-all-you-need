"""Plot registered budgets and all paired final environment differences."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
summary = json.loads((root / 'formal-audit-v1/summary.json').read_text())
budgets = (4096, 16384, 65536)
arms = (
    ('uniform', 'Uniform search', '#666666', '--', 'o'),
    ('frozen', 'Frozen learned weights', '#b68125', ':', 'v'),
    ('uniformizing', 'Uniformizing updates', '#237789', '-.', 's'),
    ('rejection', 'Rejection-loss updates', '#a63a51', '-', '^'),
)
scores = {(r['arm'], r['family'], r['budget']): 100 * r['exact_rate']
          for r in summary['summaries']}
fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.6))
for axis, (family, label) in zip(axes.flat, (
    ('composition', 'Composition tasks'),
    ('shifted', 'Shifted tasks'),
    ('balanced', 'Balanced mixture'),
)):
    for arm, name, color, style, marker in arms:
        axis.plot(budgets, [scores[(arm, family, b)] for b in budgets],
                  label=name, color=color, linestyle=style, marker=marker,
                  linewidth=1.7, markersize=5)
    axis.set_xscale('log', base=2)
    axis.set(title=label, ylim=(-2, 102), xticks=budgets,
             xticklabels=['4,096', '16,384', '65,536'],
             ylabel='Exact success (%)', xlabel='Executed primitive budget')
    axis.spines[['top', 'right']].set_visible(False)
    axis.grid(axis='y', alpha=.15)
axis = axes[1, 1]
primary = summary['primary']
differences = 100 * np.array(primary['differences'])
x = np.arange(len(differences))
axis.axhline(0, color='#888888', linewidth=.8)
axis.scatter(x, differences, color='#a63a51', s=19, label='Each environment')
lo, hi = np.array(primary['ci95']) * 100
mu = primary['mean'] * 100
axis.errorbar(40, mu, yerr=[[mu - lo], [hi - mu]], fmt='D', color='#222222',
              capsize=4, markersize=5, label='Mean and paired t95')
axis.set(title='Final rejection minus uniformizing', ylabel='Difference (percentage points)',
         xticks=[0, 15, 31, 40], xticklabels=['14000', '14015', '14031', 'Mean'],
         xlabel='Environment seed', ylim=(-25, 10))
axis.spines[['top', 'right']].set_visible(False)
axis.grid(axis='y', alpha=.15)
axis.legend(loc='lower left', fontsize=8, frameon=False)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2, frameon=False, fontsize=10)
fig.tight_layout(rect=(0, 0, 1, .91))
fig.savefig(root / 'success-and-contrasts.pdf', metadata={
    'Title': 'Study L: within-task weight updates and exact task success',
    'CreationDate': None, 'ModDate': None,
})
plt.close(fig)
