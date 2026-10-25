"""Verify Study A records and produce reproducible descriptive statistics."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def interval(values, confidence=0.95):
    x = np.array(values, dtype=float)
    mean = x.mean()
    sd = x.std(ddof=1)
    half = stats.t.ppf((1 + confidence) / 2, len(x) - 1) * sd / np.sqrt(len(x))
    return dict(n=len(x), mean=float(mean), sd=float(sd), low=float(mean-half),
                high=float(mean+half), confidence=confidence)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("results/study-a-v1"))
    args = parser.parse_args()
    meta = json.loads((args.input / "metadata.json").read_text())
    FAMILIES, WIDTHS, ARMS = meta["families"], meta["widths"], meta["arms"]
    with gzip.open(args.input / "results.jsonl.gz", "rt") as f:
        rows = [json.loads(line) for line in f]
    seeds = list(range(4000, 4020))
    assert meta["seeds"] == seeds
    keys = [(r["family"], r["width"], r["arm"], r["seed"], r["checkpoint"], r["endpoint"]) for r in rows]
    assert len(keys) == len(set(keys)) == 19200
    index = dict(zip(keys, rows))
    metrics = ["accuracy_"+suffix for suffix in ("post_gaussian", "post_uniform", "old_gaussian", "old_uniform")]
    for family in FAMILIES:
        for width in WIDTHS:
            for seed in seeds:
                frozen = index[(family, width, "frozen", seed, 0, "online")]
                for arm in ARMS:
                    first = index[(family, width, arm, seed, 0, "online")]
                    assert all(first[k] == frozen[k] for k in metrics)
                for step in range(0, 257, 16):
                    fixed = index[(family, width, "frozen", seed, step, "online")]
                    assert all(fixed[k] == frozen[k] for k in metrics)
                clean = index[(family, width, "clean", seed, 256, "online")]
                delayed = index[(family, width, "delay32", seed, 256, "flushed")]
                assert all(clean[k] == delayed[k] for k in metrics+["last_loss", "updates", "fresh_examples"])
    summary = []
    for family in FAMILIES:
        for width in WIDTHS:
            for arm in ARMS:
                chosen = [index[(family, width, arm, seed, 256, "online")] for seed in seeds]
                summary.append(dict(family=family,width=width,arm=arm,parameters=chosen[0]["parameters"],
                                    **{k: interval([r[k] for r in chosen]) for k in metrics}))
    contrasts = []
    for family in FAMILIES:
        for other_width, arm, step, label, confidence in [
            (256,"frozen",256,"small_clean_minus_large_frozen",0.975),
            (256,"clean",256,"small_clean_minus_large_clean",0.95),
            (256,"clean",16,"small_clean_minus_large_clean_at_equal_update_proxy",0.95),
            (16,"noise50",256,"small_clean_minus_small_noise50",0.95),
        ]:
            differences=[]
            for seed in seeds:
                a=index[(family,16,"clean",seed,256,"online")]
                b=index[(family,other_width,arm,seed,step,"online")]
                differences.append(a["accuracy_post_gaussian"]-b["accuracy_post_gaussian"])
                if "equal_update_proxy" in label:
                    assert a["adaptation_flops_proxy"] == b["adaptation_flops_proxy"]
            contrasts.append(dict(family=family,contrast=label,**interval(differences,confidence)))
    output = dict(summary=summary,contrasts=contrasts,verification={
        "rows":len(rows),"independent_seeds":len(seeds),
        "initial_state_scores_equal":True,"frozen_scores_constant":True,
        "delayed_flush_scores_and_last_loss_equal_clean":True,
        "source_revision":meta["git_revision"],
        "interval_unit":"independent seed; t intervals assume approximately normal seed means/differences",
        "secondary_intervals":"descriptive, not simultaneous confidence guarantees"})
    (args.input/"summary.json").write_text(json.dumps(output,indent=2)+"\n")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,layout="constrained")
    for ax,family in zip(axes,FAMILIES):
        for arm,marker in [("clean","o"),("frozen","s"),("noise50","^")]:
            entries=[r for r in summary if r["family"]==family and r["arm"]==arm]
            means=np.array([r["accuracy_post_gaussian"]["mean"] for r in entries])
            errors=np.array([r["accuracy_post_gaussian"]["high"] for r in entries])-means
            ax.errorbar([r["parameters"] for r in entries],means,yerr=errors,label=arm,marker=marker,capsize=3)
        ax.set(xscale="log",xlabel="Trainable parameters",title=family.upper(),ylim=(0,1.04))
        ax.axhline(.5,color="gray",linestyle=":",linewidth=1)
    axes[0].set_ylabel("Post-change test accuracy")
    axes[1].legend(loc="lower right")
    fig.savefig(args.input/"capacity-feedback.pdf")
    fig.savefig(args.input/"capacity-feedback.png",dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,layout="constrained")
    for ax,family in zip(axes,FAMILIES):
        for arm in ("clean","noise25","noise50","inverted","delay32","replay25"):
            values=[]
            for step in range(0,257,16):
                values.append(interval([index[(family,16,arm,s,step,"online")]["accuracy_post_gaussian"] for s in seeds]))
            steps=np.arange(0,257,16)
            ax.plot(steps,[v["mean"] for v in values],label=arm)
            ax.fill_between(steps,[v["low"] for v in values],[v["high"] for v in values],alpha=.12)
        ax.set(xlabel="Arriving feedback batches (32 examples each)",title=family.upper(),ylim=(0,1.04))
    axes[0].set_ylabel("Post-change test accuracy, 161 parameters")
    axes[1].legend(fontsize=8)
    fig.savefig(args.input/"learning-curves.pdf")
    fig.savefig(args.input/"learning-curves.png",dpi=180)
    plt.close(fig)
    for r in contrasts: print(json.dumps(r))
    print("Verified",len(rows),"records; wrote statistics and figures.")


if __name__ == "__main__":
    main()
