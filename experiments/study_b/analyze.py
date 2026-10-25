"""Analyze complete Study B runs; environments are the replication units."""

import argparse
import gzip
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def interval(values, bootstrap=False):
    a=np.asarray(values,dtype=float)
    mean=float(a.mean());sd=float(a.std(ddof=1))
    half=float(stats.t.ppf(.975,len(a)-1)*sd/np.sqrt(len(a)))
    result=dict(n=len(a),mean=mean,sd=sd,low=mean-half,high=mean+half)
    if bootstrap:
        generator=np.random.default_rng(817)
        samples=a[generator.integers(0,len(a),(10000,len(a)))].mean(axis=1)
        result["bootstrap95"]=np.quantile(samples,[.025,.975]).tolist()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=Path("results/study-b-v1"))
    args=parser.parse_args()
    seeds=list(range(3000,3024))
    all_values={};means=[];costs=[]
    for size in ("0.5B","1.5B","3B","7B","lora"):
        directory=args.input/size
        meta=json.loads((directory/"metadata.json").read_text())
        assert meta["seeds"]==seeds and meta["finished_unix"]>meta["started_unix"]
        assert not meta["development"]
        with gzip.open(directory/"predictions.jsonl.gz","rt") as f: rows=[json.loads(line) for line in f]
        conditions=("lora","lora_context") if size=="lora" else ("none","clean4","clean8","clean16","corrupted","oracle","self")
        expected={(s,c,split,i) for s in seeds for c in conditions for split,n in [("direct",16),("compose",32)] for i in range(n)}
        keys=[(r["seed"],r["condition"],r["split"],r["index"]) for r in rows]
        assert len(keys)==len(set(keys)) and set(keys)==expected
        for r in rows:
            assert r["correct"]==(r["prediction"]==r["target"])
            assert 0<=r["prediction"]<8
            assert r["target"]==r["query"]["target"]
        with gzip.open(directory/"training.jsonl.gz","rt") as f: training=[json.loads(line) for line in f]
        assert len(training)==(24 if size=="lora" else 384)
        for condition in conditions:
            for split in ("direct","compose"):
                chosen=[r for r in rows if r["condition"]==condition and r["split"]==split]
                values=[np.mean([r["correct"] for r in chosen if r["seed"]==s]) for s in seeds]
                all_values[(size,condition,split)]=values
                unconstrained=[np.mean([r["unconstrained_first_token_correct"] for r in chosen if r["seed"]==s]) for s in seeds]
                means.append(dict(model=size,condition=condition,split=split,accuracy=interval(values),unconstrained_first_token=interval(unconstrained)))
        # Batch fields repeat per row; dividing by batch size counts each once.
        inference_rows=rows+([] if size=="lora" else training)
        costs.append(dict(model=size,parameters=meta["parameters"],
            inference_input_tokens=sum(r["input_tokens"] for r in inference_rows),
            inference_padded_tokens=round(sum(r["padded_batch_tokens"]/r["batch_size"] for r in inference_rows)),
            forward_calls=round(sum(1/r["batch_size"] for r in inference_rows)),
            forward_wall_seconds=sum(r["batch_wall_seconds"]/r["batch_size"] for r in inference_rows),
            elapsed_seconds=meta["elapsed_seconds"],peak_allocated_bytes=meta["peak_allocated_bytes"],
            optimization_tokens=sum(r["processed_input_tokens"] for r in training) if size=="lora" else 0,
            optimization_seconds=sum(r["wall_seconds"] for r in training) if size=="lora" else 0))
    contrasts=[]
    for size in ("0.5B","1.5B","3B","7B"):
        for a,b in [("clean16","self"),("clean16","none"),("self","none")]:
            differences=np.array(all_values[(size,a,"compose")])-all_values[(size,b,"compose")]
            primary=size=="0.5B" and a=="clean16" and b=="self"
            contrasts.append(dict(model=size,contrast=a+"_minus_"+b,primary=primary,**interval(differences,bootstrap=primary)))
    for large_condition in ("none","clean16"):
        differences=np.array(all_values[("0.5B","clean16","compose")])-all_values[("7B",large_condition,"compose")]
        contrasts.append(dict(model="0.5B_vs_7B",contrast="small_clean16_minus_large_"+large_condition,primary=False,**interval(differences)))
    for condition in ("lora","lora_context"):
        differences=np.array(all_values[("lora",condition,"compose")])-all_values[("0.5B","none","compose")]
        contrasts.append(dict(model="lora",contrast=condition+"_minus_none",primary=False,**interval(differences)))
    output=dict(summary=means,contrasts=contrasts,costs=costs,
                interpretation="24 environment units; primary paired t and bootstrap intervals; all secondary intervals descriptive",
                offline_control="clean16 and offline have byte-identical prompts; no duplicate model calls treated as replications",
                symbolic_control="exact offsets recovered from clean examples; all direct and composition targets verified")
    (args.input/"summary.json").write_text(json.dumps(output,indent=2)+"\n")
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True,layout="constrained")
    sizes=("0.5B","1.5B","3B","7B")
    for ax,split in zip(axes,("direct","compose")):
        for condition in ("none","clean16","self","oracle"):
            items=[next(m for m in means if m["model"]==size and m["condition"]==condition and m["split"]==split)["accuracy"] for size in sizes]
            ax.errorbar(range(4),[i["mean"] for i in items],yerr=[i["high"]-i["mean"] for i in items],marker="o",capsize=3,label=condition)
        ax.set(xticks=range(4),xticklabels=sizes,xlabel="Released Qwen2.5-Instruct model",title=split,ylim=(0,1.04))
        ax.axhline(1/8,color="gray",linestyle=":",linewidth=1)
    axes[0].set_ylabel("Held-out constrained first-token accuracy")
    axes[1].legend()
    fig.savefig(args.input/"language-model-feedback.pdf")
    fig.savefig(args.input/"language-model-feedback.png",dpi=180)
    plt.close(fig)
    for c in contrasts: print(json.dumps(c))
    print('Verified four context runs and one adapter run.')


if __name__=="__main__": main()
