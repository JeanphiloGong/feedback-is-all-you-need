"""Compare fixed generation budgets using auditable deterministic prefixes."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

from experiments.study_b.run import REVISIONS
from experiments.study_b.tasks import make_environment
from experiments.study_c.run import parse_answer,reasoning_prompt


def evaluate_prefix(tokenizer,tokens,eos,budget,target):
    prefix=tokens[:budget]
    stop=next((i+1 for i,t in enumerate(prefix) if t in eos),None)
    if stop is not None:prefix=prefix[:stop]
    text=tokenizer.decode(prefix,skip_special_tokens=True)
    prediction=parse_answer(text)
    return dict(text=text,prediction=prediction,correct=prediction==target,
                format_valid=prediction is not None,output_tokens=len(prefix),
                eos_terminated=stop is not None,
                token_cap_reached=stop is None and len(prefix)==budget)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--development",action="store_true")
    args=parser.parse_args()
    raw=Path("experiments/study_d/data/environments-v1.json").read_bytes()
    envs=[make_environment(11)] if args.development else json.loads(raw)["environments"]
    args.output.mkdir(parents=True,exist_ok=False)
    name="Qwen/Qwen2.5-7B-Instruct"
    torch.set_num_threads(4);torch.manual_seed(0)
    tokenizer=AutoTokenizer.from_pretrained(name,revision=REVISIONS["7B"],local_files_only=True)
    tokenizer.padding_side="left"
    model=AutoModelForCausalLM.from_pretrained(name,revision=REVISIONS["7B"],
        dtype=torch.bfloat16,device_map="cuda",attn_implementation="sdpa",local_files_only=True).eval()
    config=model.generation_config
    assert config.num_beams==1 and config.min_length==0 and config.min_new_tokens is None
    assert config.forced_eos_token_id is None and config.max_time is None
    eos=set(config.eos_token_id)
    actual=(json.dumps(envs,sort_keys=True)+"\n").encode()
    (args.output/"environments.json").write_bytes(actual)
    sources={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
        ["experiments/study_d/run.py","experiments/study_b/run.py",
         "experiments/study_b/tasks.py","experiments/study_c/run.py"]}
    meta=dict(study="D",model=name,revision=REVISIONS["7B"],development=args.development,
        seeds=[e["seed"] for e in envs],source_sha256=sources,
        input_environments_sha256=hashlib.sha256(raw).hexdigest(),
        executed_environments_sha256=hashlib.sha256(actual).hexdigest(),
        git_revision=os.environ.get("FEEDBACK_GIT_REVISION"),
        generation_config=config.to_dict(),generation_overrides=dict(do_sample=False,max_new_tokens=512),
        packages={p:importlib.metadata.version(p) for p in ["torch","transformers","numpy"]},
        device=torch.cuda.get_device_name(),batch_size=16,started_unix=time.time())
    (args.output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")
    prefix_checks=0
    with (args.output/"predictions.jsonl").open("x") as out:
        for env in envs:
            for condition in ("none","clean16","oracle"):
                history=env["train"] if condition=="clean16" else []
                offsets=env["offsets"] if condition=="oracle" else None
                for split in ("direct","compose"):
                    queries=env[split]
                    for start in range(0,len(queries),16):
                        chunk=queries[start:start+16]
                        prompts=[reasoning_prompt(tokenizer,q,history,offsets) for q in chunk]
                        inputs=tokenizer(prompts,padding=True,add_special_tokens=False,return_tensors="pt").to(model.device)
                        torch.cuda.synchronize();began=time.perf_counter()
                        with torch.inference_mode():
                            generated=model.generate(**inputs,max_new_tokens=512,do_sample=False)
                        torch.cuda.synchronize();elapsed=time.perf_counter()-began
                        tokens=generated[:,inputs["input_ids"].shape[1]:].tolist()
                        if args.development:
                            with torch.inference_mode():
                                shorter=model.generate(**inputs,max_new_tokens=128,do_sample=False)
                            short_tokens=shorter[:,inputs["input_ids"].shape[1]:].tolist()
                            for long,short in zip(tokens,short_tokens):
                                long_end=next((i+1 for i,t in enumerate(long[:128]) if t in eos),128)
                                short_end=next((i+1 for i,t in enumerate(short) if t in eos),len(short))
                                assert long[:long_end]==short[:short_end],"Generation prefix mismatch"
                                prefix_checks+=1
                        lengths=inputs["attention_mask"].sum(-1).tolist()
                        for i,(q,prompt,sequence,length) in enumerate(zip(chunk,prompts,tokens,lengths)):
                            stop=next((i+1 for i,t in enumerate(sequence) if t in eos),len(sequence))
                            sequence=sequence[:stop]
                            evaluations={str(b):evaluate_prefix(tokenizer,sequence,eos,b,q["target"]) for b in (128,512)}
                            row=dict(seed=env["seed"],condition=condition,split=split,index=start+i,
                                query=q,target=q["target"],output_token_ids=sequence,evaluations=evaluations,
                                input_tokens=length,output_tokens=len(sequence),batch_index=start//16,
                                batch_size=len(chunk),batch_wall_seconds=elapsed,
                                padded_batch_tokens=inputs["input_ids"].numel(),
                                padded_generation_tokens=sum(map(len,tokens)),
                                prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
                            out.write(json.dumps(row)+"\n")
                    out.flush()
                    print(env["seed"],condition,split,"complete",flush=True)
    meta.update(finished_unix=time.time(),development_prefix_checks=prefix_checks,
                peak_allocated_bytes=torch.cuda.max_memory_allocated())
    meta["elapsed_seconds"]=meta["finished_unix"]-meta["started_unix"]
    (args.output/"metadata.json").write_text(json.dumps(meta,indent=2)+"\n")


if __name__=="__main__":main()
