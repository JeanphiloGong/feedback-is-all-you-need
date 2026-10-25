"""Audit frozen generation runs and estimate paired environment contrasts."""

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


def interval(values, bootstrap=False):
    values=np.asarray(values,dtype=float)
    mean=float(values.mean())
    sd=float(values.std(ddof=1))
    half=float(stats.t.ppf(.975,len(values)-1)*sd/np.sqrt(len(values)))
    result=dict(n=len(values),mean=mean,sd=sd,low=mean-half,high=mean+half)
    if bootstrap:
        rng=np.random.default_rng(819)
        means=values[rng.integers(0,len(values),(10000,len(values)))].mean(axis=1)
        result["bootstrap95"]=np.quantile(means,[.025,.975]).tolist()
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--input",type=Path,default=Path("results/study-c-v1"))
    args=parser.parse_args()
    frozen=Path("experiments/study_c/data/environments-v1.json").read_bytes()
    environments=json.loads(frozen)["environments"]
    env_by_seed={e["seed"]:e for e in environments}
    seeds=list(range(5000,5024))
    interfaces=("first_token","generation")
    conditions=("none","clean16","self","oracle")
    summaries=[];costs=[];values={};self_summaries=[];audits=[]
    for size in ("0.5B","7B"):
        directory=args.input/size
        meta=json.loads((directory/"metadata.json").read_text())
        assert meta["model"]==f"Qwen/Qwen2.5-{size}-Instruct"
        assert meta["revision"]==REVISIONS[size]
        assert meta["generation_config"]["repetition_penalty"]=={"0.5B":1.1,"7B":1.05}[size]
        tokenizer=AutoTokenizer.from_pretrained(meta["model"],revision=meta["revision"],local_files_only=True)
        assert meta["seeds"]==seeds and not meta["development"]
        assert meta["finished_unix"]>meta["started_unix"]
        assert meta["git_revision"]=="f0a8841"
        for field,path in [("source_sha256","experiments/study_c/run.py"),
                           ("tasks_sha256","experiments/study_b/tasks.py"),
                           ("score_source_sha256","experiments/study_b/run.py")]:
            assert meta[field]==hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert meta["input_environments_sha256"]==hashlib.sha256(frozen).hexdigest()
        actual=(directory/"environments.json").read_bytes()
        assert meta["executed_environments_sha256"]==hashlib.sha256(actual).hexdigest()
        assert json.loads(actual)==environments
        assert meta["generation_overrides"]==dict(max_new_tokens=128,do_sample=False)
        with gzip.open(directory/"predictions.jsonl.gz","rt") as stream:
            rows=[json.loads(line) for line in stream]
        with gzip.open(directory/"self-history.jsonl.gz","rt") as stream:
            history=[json.loads(line) for line in stream]
        keys=[(r["seed"],r["interface"],r["condition"],r["split"],r["index"]) for r in rows]
        expected={(seed,interface,condition,split,i) for seed in seeds for interface in interfaces
                  for condition in conditions for split,n in [("direct",16),("compose",32)] for i in range(n)}
        assert len(keys)==len(set(keys)) and set(keys)==expected
        for row in rows:
            assert row["query"]==env_by_seed[row["seed"]][row["split"]][row["index"]]
        assert len(history)==24*2*16
        for seed in seeds:
            for interface in interfaces:
                group=[r for r in history if r["seed"]==seed and r["interface"]==interface]
                assert [r["query"] for r in group]==env_by_seed[seed]["train"]
        eos=set(meta["generation_config"]["eos_token_id"])
        for row in rows+history:
            assert row["target"]==row["query"]["target"]
            assert row["correct"]==(row["prediction"]==row["target"])
            assert len(row["prompt_sha256"])==64
            if row["interface"]=="generation":
                lines=[line.strip() for line in row["text"].splitlines() if line.strip()]
                match=re.fullmatch(r"Answer:\s*([0-7])",lines[-1],re.IGNORECASE) if lines else None
                answer=int(match[1]) if match else None
                assert answer==row["prediction"] and row["format_valid"]==(answer is not None)
                tokens=row["output_token_ids"]
                assert tokenizer.decode(tokens,skip_special_tokens=True)==row["text"]
                assert len(tokens)==row["output_tokens"] and 1<=len(tokens)<=128
                assert not any(token in eos for token in tokens[:-1])
                assert row["eos_terminated"]==(tokens[-1] in eos)
                assert row["token_cap_reached"]==(not row["eos_terminated"] and len(tokens)==128)
                assert row["eos_terminated"] or row["token_cap_reached"]
            else:
                assert 0<=row["prediction"]<8
        for interface in interfaces:
            for condition in conditions:
                for split in ("direct","compose"):
                    group=[r for r in rows if (r["interface"],r["condition"],r["split"])==(interface,condition,split)]
                    metrics={}
                    fields=["correct"]+(["format_valid","token_cap_reached","output_tokens"] if interface=="generation" else [])
                    for field in fields:
                        scores=[float(np.mean([r[field] for r in group if r["seed"]==seed])) for seed in seeds]
                        metrics[field]=interval(scores)
                        if field=="correct":values[(size,interface,condition,split)]=scores
                    summaries.append(dict(model=size,interface=interface,condition=condition,split=split,**metrics))
            group=[r for r in rows+history if r["interface"]==interface]
            costs.append(dict(model=size,interface=interface,
                input_tokens=sum(r["input_tokens"] for r in group),
                padded_input_tokens=round(sum(r["padded_batch_tokens"]/r["batch_size"] for r in group)),
                calls=round(sum(1/r["batch_size"] for r in group)),
                wall_seconds=sum(r["batch_wall_seconds"]/r["batch_size"] for r in group),
                generated_tokens=sum(r.get("output_tokens",0) for r in group),
                padded_generated_tokens=round(sum(r.get("padded_generation_tokens",0)/r["batch_size"] for r in group))))
            group=[r for r in history if r["interface"]==interface]
            self_summaries.append(dict(model=size,interface=interface,n=len(group),
                correct=sum(r["correct"] for r in group),
                invalid=sum(not r["format_valid"] for r in group) if interface=="generation" else 0,
                truncated=sum(r["token_cap_reached"] for r in group) if interface=="generation" else 0))
        audits.append(dict(model=size,predictions=len(rows),history=len(history),
                           revision=meta["revision"],repetition_penalty=meta["generation_config"]["repetition_penalty"],
                           elapsed_seconds=meta["elapsed_seconds"],peak_allocated_bytes=meta["peak_allocated_bytes"]))
    contrasts=[]
    for size in ("0.5B","7B"):
        for split in ("direct","compose"):
            effects={}
            for interface in interfaces:
                for control in ("none","self"):
                    delta=np.array(values[(size,interface,"clean16",split)])-values[(size,interface,control,split)]
                    contrasts.append(dict(model=size,split=split,interface=interface,
                        contrast="clean16_minus_"+control,primary=False,**interval(delta)))
                    if control=="none":effects[interface]=delta
                delta=np.array(values[(size,interface,"clean16","compose")])-values[(size,interface,"clean16","direct")]
                if split=="compose":
                    contrasts.append(dict(model=size,split="compose_minus_direct",interface=interface,
                        contrast="clean16",primary=False,**interval(delta)))
            primary=size=="0.5B" and split=="compose"
            contrasts.append(dict(model=size,split=split,interface="interaction",
                contrast="generation_feedback_effect_minus_first_token_feedback_effect",primary=primary,
                **interval(effects["generation"]-effects["first_token"],bootstrap=primary)))
    summary=dict(summary=summaries,contrasts=contrasts,costs=costs,self_history=self_summaries,audits=audits,
                 interpretation="Environment-paired intervals; one primary interaction; all other intervals descriptive. Examine component effects before interpreting interaction.")
    (args.input/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    fig,axes=plt.subplots(2,2,figsize=(10,7),sharey=True,constrained_layout=True)
    for row,size in enumerate(("0.5B","7B")):
        for col,split in enumerate(("direct","compose")):
            ax=axes[row,col]
            for k,interface in enumerate(interfaces):
                items=[next(s for s in summaries if (s["model"],s["interface"],s["condition"],s["split"])==(size,interface,c,split))["correct"] for c in conditions]
                ax.errorbar(np.arange(4)+(k-.5)*.12,[i["mean"] for i in items],
                            yerr=[i["high"]-i["mean"] for i in items],marker="o",capsize=3,label=interface)
            ax.set(xticks=range(4),xticklabels=conditions,ylim=(-.035,1.04),title=f"{size}: {split}")
            if col==0:ax.set_ylabel("Held-out accuracy")
    axes[0,1].legend()
    fig.savefig(args.input/"generation-interface.pdf")
    fig.savefig(args.input/"generation-interface.png",dpi=180)
    plt.close(fig)
    for contrast in contrasts:print(json.dumps(contrast))
    print("Verified two complete generation-interface runs.")


if __name__=="__main__":main()
