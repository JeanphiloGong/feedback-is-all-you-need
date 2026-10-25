"""Prospective response-interface experiment on fresh hidden command languages."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import time

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

from experiments.study_b.tasks import messages
from experiments.study_b.run import REVISIONS,score

REASONING = 'Reason briefly about the state changes. Finish with a line exactly "Answer: N", where N is a digit from 0 to 7.'


def parse_answer(text):
    lines=[line.strip() for line in text.splitlines() if line.strip()]
    match=re.fullmatch(r"Answer:\s*([0-7])",lines[-1],re.IGNORECASE) if lines else None
    return int(match[1]) if match else None


def reasoning_prompt(tokenizer,query,history,offsets=None):
    prompt=messages(query,history,offsets)
    prompt[0]["content"]="You execute an unfamiliar command language. "+REASONING
    original="Return only the final state digit."
    assert original in prompt[1]["content"]
    prompt[1]["content"]=prompt[1]["content"].replace(original,REASONING)
    return tokenizer.apply_chat_template(prompt,tokenize=False,add_generation_prompt=True)


def generate(model,tokenizer,queries,history,batch_size,offsets=None):
    output=[]
    eos=model.generation_config.eos_token_id
    eos={eos} if isinstance(eos,int) else set(eos)
    for start in range(0,len(queries),batch_size):
        chunk=queries[start:start+batch_size]
        prompts=[reasoning_prompt(tokenizer,q,history,offsets) for q in chunk]
        inputs=tokenizer(prompts,padding=True,add_special_tokens=False,return_tensors="pt").to(model.device)
        torch.cuda.synchronize();began=time.perf_counter()
        with torch.inference_mode():
            generated=model.generate(**inputs,max_new_tokens=128,do_sample=False)
        torch.cuda.synchronize();elapsed=time.perf_counter()-began
        continuations=generated[:,inputs["input_ids"].shape[1]:].tolist()
        lengths=inputs["attention_mask"].sum(-1).tolist()
        for query,prompt,tokens,length in zip(chunk,prompts,continuations,lengths):
            stop=next((i+1 for i,t in enumerate(tokens) if t in eos),None)
            trimmed=tokens[:stop] if stop is not None else tokens
            text=tokenizer.decode(trimmed,skip_special_tokens=True)
            answer=parse_answer(text)
            output.append(dict(query=query,target=query["target"],prediction=answer,
                correct=answer==query["target"],text=text,format_valid=answer is not None,
                output_token_ids=trimmed,
                eos_terminated=stop is not None,token_cap_reached=stop is None and len(tokens)==128,
                output_tokens=len(trimmed),input_tokens=length,batch_index=start//batch_size,
                batch_size=len(chunk),padded_batch_tokens=inputs["input_ids"].numel(),
                batch_wall_seconds=elapsed,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest()))
    return output


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=("0.5B","7B"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--environments",type=Path,default=Path("experiments/study_c/data/environments-v1.json"))
    parser.add_argument("--development",action="store_true")
    args=parser.parse_args()
    raw=args.environments.read_bytes();envs=json.loads(raw)["environments"]
    if args.development:
        from experiments.study_b.tasks import make_environment
        envs=[make_environment(9)]
    args.output.mkdir(parents=True,exist_ok=False)
    name=f"Qwen/Qwen2.5-{args.model}-Instruct"
    torch.set_num_threads(4);torch.manual_seed(0)
    tokenizer=AutoTokenizer.from_pretrained(name,revision=REVISIONS[args.model],local_files_only=True)
    tokenizer.padding_side="left"
    digit_ids=[]
    for digit in range(8):
        ids=tokenizer.encode(str(digit),add_special_tokens=False);assert len(ids)==1
        digit_ids.append(ids[0])
    model=AutoModelForCausalLM.from_pretrained(name,revision=REVISIONS[args.model],dtype=torch.bfloat16,
                device_map="cuda",attn_implementation="sdpa",local_files_only=True).eval()
    actual=(json.dumps(envs,sort_keys=True)+"\n").encode()
    (args.output/"environments.json").write_bytes(actual)
    metadata=dict(study="C",model=name,revision=REVISIONS[args.model],development=args.development,
        seeds=[e["seed"] for e in envs],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tasks_sha256=hashlib.sha256(Path("experiments/study_b/tasks.py").read_bytes()).hexdigest(),
        score_source_sha256=hashlib.sha256(Path("experiments/study_b/run.py").read_bytes()).hexdigest(),
        input_environments_sha256=hashlib.sha256(raw).hexdigest(),executed_environments_sha256=hashlib.sha256(actual).hexdigest(),
        git_revision=os.environ.get("FEEDBACK_GIT_REVISION"),packages={p:importlib.metadata.version(p) for p in ["torch","transformers","numpy"]},
        device=torch.cuda.get_device_name(),batch_size=16,max_new_tokens=128,started_unix=time.time())
    metadata["generation_config"]=model.generation_config.to_dict()
    metadata["generation_overrides"]={"max_new_tokens":128,"do_sample":False}
    metadata["parameters"]=sum(p.numel() for p in model.parameters())
    (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    with (args.output/"predictions.jsonl").open("x") as out,(args.output/"self-history.jsonl").open("x") as train:
        for env in envs:
            for interface in ("first_token","generation"):
                guesses=[]
                for query in env["train"]:
                    if interface=="first_token":
                        row=score(model,tokenizer,digit_ids,[query],guesses,16)[0]
                    else:
                        row=generate(model,tokenizer,[query],guesses,16)[0]
                    train.write(json.dumps(dict(seed=env["seed"],interface=interface,**row))+"\n")
                    guesses.append(dict(query,target=row["prediction"] if row["prediction"] is not None else "invalid"))
                train.flush()
                histories={"none":[],"clean16":env["train"],"self":guesses,"oracle":[]}
                for condition,history in histories.items():
                    offsets=env["offsets"] if condition=="oracle" else None
                    for split in ("direct","compose"):
                        if interface=="first_token":
                            rows=score(model,tokenizer,digit_ids,env[split],history,16,offsets)
                        else:
                            rows=generate(model,tokenizer,env[split],history,16,offsets)
                        for i,row in enumerate(rows):
                            out.write(json.dumps(dict(seed=env["seed"],interface=interface,condition=condition,split=split,index=i,**row))+"\n")
                        print(env["seed"],interface,condition,split,sum(r["correct"] for r in rows)/len(rows),flush=True)
                    out.flush()
    metadata["finished_unix"]=time.time()
    metadata["elapsed_seconds"]=metadata["finished_unix"]-metadata["started_unix"]
    metadata["peak_allocated_bytes"]=torch.cuda.max_memory_allocated()
    (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__=="__main__":main()
