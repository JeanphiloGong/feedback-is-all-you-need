"""Audit the fixed-budget run; estimate environment-paired prefix effects."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from transformers import AutoTokenizer

from experiments.study_b.run import REVISIONS
from experiments.study_c.run import reasoning_prompt


def interval(values,bootstrap=False):
    values=np.asarray(values,dtype=float)
    mean=float(values.mean());sd=float(values.std(ddof=1))
    half=float(stats.t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values)))
    result=dict(n=len(values),mean=mean,sd=sd,low=mean-half,high=mean+half)
    if bootstrap:
        rng=np.random.default_rng(820)
        samples=values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1)
        result["bootstrap95"]=np.quantile(samples,[.025,.975]).tolist()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=Path("results/study-d-v1"))
    args=parser.parse_args()
    root=args.input
    meta=json.loads((root/"metadata.json").read_text())
    seeds=list(range(6000,6024));conditions=("none","clean16","oracle")
    assert meta["model"]=="Qwen/Qwen2.5-7B-Instruct" and meta["revision"]==REVISIONS["7B"]
    assert meta["seeds"]==seeds and not meta["development"]
    assert meta["git_revision"]=="8507524" and meta["finished_unix"]>meta["started_unix"]
    config=meta["generation_config"]
    assert config["repetition_penalty"]==1.05 and config["num_beams"]==1
    assert config["min_length"]==0 and config["min_new_tokens"] is None
    assert config["forced_eos_token_id"] is None and config["max_time"] is None
    assert meta["generation_overrides"]==dict(do_sample=False,max_new_tokens=512)
    assert set(meta["source_sha256"])=={"experiments/study_d/run.py","experiments/study_b/run.py",
                                        "experiments/study_b/tasks.py","experiments/study_c/run.py"}
    for source,digest in meta["source_sha256"].items():
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest()==digest
    raw=Path("experiments/study_d/data/environments-v1.json").read_bytes()
    assert meta["input_environments_sha256"]==hashlib.sha256(raw).hexdigest()
    actual=(root/"environments.json").read_bytes()
    assert meta["executed_environments_sha256"]==hashlib.sha256(actual).hexdigest()
    environments=json.loads(raw)["environments"]
    assert json.loads(actual)==environments
    envs={e["seed"]:e for e in environments}
    with gzip.open(root/"predictions.jsonl.gz","rt") as f:rows=[json.loads(line) for line in f]
    keys=[(r["seed"],r["condition"],r["split"],r["index"]) for r in rows]
    expected={(s,c,split,i) for s in seeds for c in conditions
              for split,n in [("direct",16),("compose",32)] for i in range(n)}
    assert len(keys)==len(set(keys)) and set(keys)==expected
    tokenizer=AutoTokenizer.from_pretrained(meta["model"],revision=meta["revision"],local_files_only=True)
    eos=set(meta["generation_config"]["eos_token_id"])
    for row in rows:
        assert row["query"]==envs[row["seed"]][row["split"]][row["index"]]
        assert row["target"]==row["query"]["target"]
        env=envs[row["seed"]]
        history=env["train"] if row["condition"]=="clean16" else []
        offsets=env["offsets"] if row["condition"]=="oracle" else None
        prompt=reasoning_prompt(tokenizer,row["query"],history,offsets)
        assert hashlib.sha256(prompt.encode()).hexdigest()==row["prompt_sha256"]
        assert len(tokenizer.encode(prompt,add_special_tokens=False))==row["input_tokens"]
        tokens=row["output_token_ids"]
        assert len(tokens)==row["output_tokens"] and 1<=len(tokens)<=512
        assert not any(t in eos for t in tokens[:-1])
        assert tokens[-1] in eos or len(tokens)==512
        for budget in (128,512):
            prefix=tokens[:budget];end=next((i+1 for i,t in enumerate(prefix) if t in eos),None)
            if end is not None:prefix=prefix[:end]
            text=tokenizer.decode(prefix,skip_special_tokens=True)
            lines=[line.strip() for line in text.splitlines() if line.strip()]
            match=re.fullmatch(r"Answer:\s*([0-7])",lines[-1],re.IGNORECASE) if lines else None
            prediction=int(match[1]) if match else None
            expected_score=dict(text=text,prediction=prediction,correct=prediction==row["target"],
                format_valid=prediction is not None,output_tokens=len(prefix),eos_terminated=end is not None,
                token_cap_reached=end is None and len(prefix)==budget)
            assert expected_score==row["evaluations"][str(budget)]
    # These fields describe each actual generation batch and repeat per row.
    batches={}
    for row in rows:
        key=(row["seed"],row["condition"],row["split"],row["batch_index"])
        batches.setdefault(key,[]).append(row)
    assert len(batches)==216
    for group in batches.values():
        assert len(group)==group[0]["batch_size"]==16
        for field in ("batch_wall_seconds","padded_batch_tokens","padded_generation_tokens"):
            assert len({r[field] for r in group})==1
        assert group[0]["padded_batch_tokens"]==16*max(r["input_tokens"] for r in group)
        assert group[0]["padded_generation_tokens"]==16*max(r["output_tokens"] for r in group)
    values={};summaries=[];transitions=[]
    for condition in conditions:
        for split in ("direct","compose"):
            group=[r for r in rows if (r["condition"],r["split"])==(condition,split)]
            for budget in (128,512):
                metrics={}
                for field in ("correct","format_valid","token_cap_reached","output_tokens"):
                    per_seed=[np.mean([r["evaluations"][str(budget)][field] for r in group if r["seed"]==s]) for s in seeds]
                    metrics[field]=interval(per_seed)
                    if field=="correct":values[(condition,split,budget)]=np.array(per_seed)
                summaries.append(dict(condition=condition,split=split,budget=budget,**metrics))
            transitions.append(dict(condition=condition,split=split,n=len(group),
                wrong_to_correct=sum(not r["evaluations"]["128"]["correct"] and r["evaluations"]["512"]["correct"] for r in group),
                correct_to_wrong=sum(r["evaluations"]["128"]["correct"] and not r["evaluations"]["512"]["correct"] for r in group)))
    contrasts=[]
    for split in ("direct","compose"):
        for condition in conditions:
            delta=values[(condition,split,512)]-values[(condition,split,128)]
            primary=condition=="clean16" and split=="compose"
            contrasts.append(dict(condition=condition,split=split,contrast="512_minus_128",primary=primary,
                                  **interval(delta,bootstrap=primary)))
        effects={b:values[("clean16",split,b)]-values[("none",split,b)] for b in (128,512)}
        for budget,effect in effects.items():
            contrasts.append(dict(condition="clean16_minus_none",split=split,budget=budget,primary=False,**interval(effect)))
        contrasts.append(dict(condition="clean16_minus_none",split=split,contrast="512_minus_128",primary=False,
                              **interval(effects[512]-effects[128])))
    costs=dict(generation_calls=len(batches),input_tokens=sum(r["input_tokens"] for r in rows),
        output_tokens=sum(r["output_tokens"] for r in rows),
        padded_input_tokens=sum(g[0]["padded_batch_tokens"] for g in batches.values()),
        padded_output_tokens=sum(g[0]["padded_generation_tokens"] for g in batches.values()),
        generation_wall_seconds=sum(g[0]["batch_wall_seconds"] for g in batches.values()),
        elapsed_seconds=meta["elapsed_seconds"],peak_allocated_bytes=meta["peak_allocated_bytes"],
        accounting="One actual 512-budget generation per query; 128-budget scores are prefixes without separately measured wall times.")
    summary=dict(summary=summaries,contrasts=contrasts,transitions=transitions,costs=costs,
                 interpretation="24 paired environment units; clean composition budget contrast primary; other intervals descriptive.")
    (root/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    fig,axes=plt.subplots(1,2,figsize=(9,4),sharey=True,constrained_layout=True)
    for ax,split in zip(axes,("direct","compose")):
        for budget in (128,512):
            scores=[next(s["correct"] for s in summaries if (s["condition"],s["split"],s["budget"])==(c,split,budget)) for c in conditions]
            ax.errorbar(range(3),[s["mean"] for s in scores],yerr=[s["high"]-s["mean"] for s in scores],marker="o",capsize=3,label=str(budget)+" tokens")
        ax.set(xticks=range(3),xticklabels=conditions,ylim=(-.03,1.04),title=split)
    axes[0].set_ylabel("Held-out accuracy");axes[1].legend()
    fig.savefig(root/"generation-budget.pdf");fig.savefig(root/"generation-budget.png",dpi=180)
    plt.close(fig)
    for c in contrasts:print(json.dumps(c))
    print("Verified all 3456 generation records and 216 actual batches.")


if __name__=="__main__":main()
