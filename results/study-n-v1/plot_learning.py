"""Display fixed prediction curves and the paired encoding-gain contrast."""

import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).resolve().parent
summary = json.loads((root / 'formal-audit-v1/summary.json').read_text())
arms = (
    ('small_online', 'Small, online', '#ad3151', '-', 'o'),
    ('large_online', 'Large, online', '#246eaa', '-', '^'),
    ('small_frozen', 'Small, frozen', '#ad3151', ':', None),
    ('large_frozen', 'Large, frozen', '#246eaa', ':', None),
)
plt.rcParams.update({'font.size': 9})
fig = plt.figure(figsize=(7.2, 6.4))
grid = fig.add_gridspec(2, 2)
axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])]
for axis, encoding in zip(axes, ('canonical', 'relabeled')):
    for arm, label, color, style, marker in arms:
        row = next(r for r in summary['direct_summaries'] if (r['encoding'], r['arm']) == (encoding, arm))
        axis.plot(summary['checkpoints'], 100 * np.array(row['exact_rates']), label=label,
                  color=color, linestyle=style, marker=marker, linewidth=1.5, markersize=3.5)
    axis.plot(summary['checkpoints'], [100 * r['permutation_completion_expectation'] for r in summary['symbolic_reference']],
              label='Permutation expectation', color='#444444', linestyle='--', linewidth=1.1)
    axis.plot(summary['checkpoints'], [0.] * 4, label='Observed table (abstains)', color='#888888', linewidth=1.)
    if encoding == 'canonical':
        axis.plot(summary['checkpoints'], [100 * r['exact_rate'] for r in summary['symbolic_reference']],
                  label='Signed-coordinate reference', color='#388057', linestyle='-.', marker='d', markersize=3.5)
    axis.set(title=encoding.capitalize() + ' encoding', xlabel='Observed transitions',
             ylabel='Exact unqueried accuracy (%)', ylim=(-3, 103), xticks=[0, 16, 32, 64])
    axis.grid(axis='y', alpha=.2)
axis = axes[2]
primary = summary['primary']
axis.axhline(0, color='#777777', linewidth=.8)
axis.scatter(np.arange(32), 100 * np.array(primary['differences']), s=14, color='#ad3151')
mu = 100 * primary['mean']
lo, hi = 100 * np.array(primary['ci95'])
axis.errorbar(41, mu, yerr=[[mu - lo], [hi - mu]], fmt='D', color='#222222', capsize=4, markersize=5)
axis.set(title='Small-model learning gain: canonical - relabeled', xlabel='Environment seed',
         ylabel='Gain difference (percentage points)', ylim=(-2, 85), xticks=[0, 15, 31, 41],
         xticklabels=['16000', '16015', '16031', 'Mean'])
axis.grid(axis='y', alpha=.2)
for axis in axes:
    axis.spines[['top', 'right']].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=8)
fig.tight_layout(rect=(0, 0, 1, .87))
fig.savefig(root / 'encoding-and-contrasts.pdf', metadata={
    'Title': 'Study N: state encoding and learning of unqueried transitions',
    'CreationDate': None, 'ModDate': None})
plt.close(fig)
