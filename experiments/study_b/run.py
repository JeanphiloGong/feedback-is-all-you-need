"""Run frozen Study B prompts, with optional persistent LoRA adaptation."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiments.study_b.tasks import make_environment, messages, execute, symbolic_offsets

REVISIONS = {
    "0.5B": "7ae557604adf67be50417f59c2c2f167def9a775",
    "1.5B": "989aa7980e4cf806f80c7fef2b1adb7bc71aa306",
    "3B": "aa8e72537993ba99e69dfaafa59ed015b17504d1",
    "7B": "a09a35458c702b33eeacc393d103063234e8bc28",
}


def text_prompt(tokenizer, query, history, offsets=None):
    return tokenizer.apply_chat_template(messages(query,history,offsets),
                                         tokenize=False,add_generation_prompt=True)


def inputs(tokenizer, prompts, device):
    batch=tokenizer(prompts,padding=True,return_tensors="pt",add_special_tokens=False).to(device)
    batch["position_ids"]=batch["attention_mask"].long().cumsum(-1)-1
    batch["position_ids"].masked_fill_(batch["attention_mask"]==0,0)
    return batch


def score(model, tokenizer, digit_ids, queries, history, batch_size, offsets=None):
    model.eval()
    predictions=[]
    for start in range(0,len(queries),batch_size):
        chunk=queries[start:start+batch_size]
        prompts=[text_prompt(tokenizer,q,history,offsets) for q in chunk]
        batch=inputs(tokenizer,prompts,model.device)
        torch.cuda.synchronize()
        started=time.perf_counter()
        with torch.inference_mode():
            logits=model(**batch,logits_to_keep=1).logits[:,-1,:]
            chosen=logits[:,digit_ids].argmax(-1).tolist()
            unrestricted=logits.argmax(-1).tolist()
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        lengths=batch["attention_mask"].sum(-1).tolist()
        for q,prompt,pred,raw,length in zip(chunk,prompts,chosen,unrestricted,lengths):
            predictions.append(dict(query=q,prediction=pred,target=q["target"],
                                    correct=pred==q["target"],unconstrained_token=raw,
                                    unconstrained_text=tokenizer.decode([raw]),
                                    unconstrained_first_token_correct=raw==digit_ids[q["target"]],
                                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                                    batch_index=start//batch_size,
                                    input_tokens=length,batch_size=len(chunk),
                                    padded_batch_tokens=int(batch["input_ids"].numel()),
                                    batch_wall_seconds=elapsed))
    return predictions


def train_adapter(base, tokenizer, digit_ids, env, output):
    from peft import LoraConfig, TaskType, get_peft_model
    torch.manual_seed(env["seed"]+4000000)
    config=LoraConfig(r=8,lora_alpha=16,lora_dropout=0,bias="none",
                      target_modules=["q_proj","v_proj"],task_type=TaskType.CAUSAL_LM,
                      revision=REVISIONS["0.5B"])
    model=get_peft_model(base,config)
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=2e-4,weight_decay=0)
    batches=[]
    for start in range(0,16,4):
        queries=env["train"][start:start+4]
        prompts=[text_prompt(tokenizer,q,[]) for q in queries]
        batch=inputs(tokenizer,prompts,model.device)
        targets=torch.tensor([digit_ids[q["target"]] for q in queries],device=model.device)
        batches.append((batch,targets))
    model.train()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started=time.perf_counter()
    losses=[]
    for epoch in range(32):
        for batch,targets in batches:
            optimizer.zero_grad(set_to_none=True)
            logits=model(**batch,logits_to_keep=1).logits[:,-1,:]
            loss=torch.nn.functional.cross_entropy(logits.float(),targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    elapsed=time.perf_counter()-started
    save=output/"adapters"/str(env["seed"])
    model.save_pretrained(save,save_embedding_layers=False)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in save.iterdir() if p.is_file()}
    accounting=dict(seed=env["seed"],updates=128,trainable_parameters=sum(p.numel() for p in parameters),
                    processed_input_tokens=32*sum(int(b[0]["attention_mask"].sum()) for b in batches),
                    padded_input_tokens=32*sum(b[0]["input_ids"].numel() for b in batches),
                    wall_seconds=elapsed,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    loss_history=losses,adapter_sha256=manifest)
    return model,accounting


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",choices=REVISIONS,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--environments",type=Path,default=Path("experiments/study_b/data/environments-v1.json"))
    parser.add_argument("--batch-size",type=int,default=16)
    parser.add_argument("--development",action="store_true")
    parser.add_argument("--lora",action="store_true")
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    if args.lora and args.model != "0.5B": raise ValueError("LoRA protocol is restricted to 0.5B")
    raw=args.environments.read_bytes()
    envs=[make_environment(7)] if args.development else json.loads(raw)["environments"]
    name=f"Qwen/Qwen2.5-{args.model}-Instruct"
    tokenizer=AutoTokenizer.from_pretrained(name,revision=REVISIONS[args.model],local_files_only=True)
    tokenizer.padding_side="left"
    if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    digit_ids=[]
    for digit in range(8):
        ids=tokenizer.encode(str(digit),add_special_tokens=False)
        assert len(ids)==1
        digit_ids.append(ids[0])
    torch.set_num_threads(4)
    torch.manual_seed(0)
    load_start=time.perf_counter()
    model=AutoModelForCausalLM.from_pretrained(name,revision=REVISIONS[args.model],
            dtype=torch.bfloat16,device_map="cuda",attn_implementation="sdpa",local_files_only=True)
    actual_env_bytes=(json.dumps(envs,sort_keys=True)+"\n").encode()
    (args.output/"environments.json").write_bytes(actual_env_bytes)
    peak_bytes=torch.cuda.max_memory_allocated()
    metadata=dict(study="B",model=name,revision=REVISIONS[args.model],development=args.development,
                  mode="lora" if args.lora else "context",seeds=[e["seed"] for e in envs],
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  tasks_sha256=hashlib.sha256(Path("experiments/study_b/tasks.py").read_bytes()).hexdigest(),
                  environments_sha256=hashlib.sha256(raw).hexdigest(),
                  executed_environments_sha256=hashlib.sha256(actual_env_bytes).hexdigest(),
                  git_revision=os.environ.get("FEEDBACK_GIT_REVISION"),
                  torch=torch.__version__,numpy=np.__version__,python=platform.python_version(),
                  transformers=importlib.metadata.version("transformers"),
                  peft=importlib.metadata.version("peft"),
                  device=torch.cuda.get_device_name(),parameters=sum(p.numel() for p in model.parameters()),
                  load_seconds=time.perf_counter()-load_start,digit_token_ids=digit_ids,
                  batch_size=args.batch_size,started_unix=time.time())
    (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    with (args.output/"predictions.jsonl").open("x") as out, (args.output/"training.jsonl").open("x") as training:
        for env in envs:
            histories={"none":[],"clean4":env["train"][:4],"clean8":env["train"][:8],
                       "clean16":env["train"],"corrupted":env["corrupted"],"oracle":[]}
            if args.lora:
                model,accounting=train_adapter(model,tokenizer,digit_ids,env,args.output)
                training.write(json.dumps(accounting)+"\n");training.flush()
                histories={"lora":[],"lora_context":env["train"]}
            else:
                guesses=[]
                for query in env["train"]:
                    guess=score(model,tokenizer,digit_ids,[query],guesses,args.batch_size)[0]
                    training.write(json.dumps(dict(seed=env["seed"],kind="self_guess",**guess))+"\n")
                    guesses.append(dict(query,target=guess["prediction"]))
                training.flush()
                histories["self"]=guesses
            for condition,history in histories.items():
                offsets=env["offsets"] if condition=="oracle" else None
                for split in ("direct","compose"):
                    if condition=="clean16":
                        for q in env[split]:
                            assert text_prompt(tokenizer,q,history)==text_prompt(tokenizer,q,list(env["train"]))
                    rows=score(model,tokenizer,digit_ids,env[split],history,args.batch_size,offsets)
                    for i,row in enumerate(rows):
                        out.write(json.dumps(dict(seed=env["seed"],condition=condition,split=split,index=i,**row))+"\n")
                    print(env["seed"],condition,split,round(sum(r["correct"] for r in rows)/len(rows),4),flush=True)
                out.flush()
            inferred=symbolic_offsets(env["train"])
            assert all(execute(q["state"],q["commands"],inferred)==q["target"] for q in env["direct"]+env["compose"])
            if args.lora:
                model=model.unload()
                torch.cuda.empty_cache()
            peak_bytes=max(peak_bytes,torch.cuda.max_memory_allocated())
    metadata["finished_unix"]=time.time()
    metadata["elapsed_seconds"]=metadata["finished_unix"]-metadata["started_unix"]
    metadata["peak_allocated_bytes"]=peak_bytes
    metadata["accounting_note"]="batch wall time and padded tokens repeat per row; sum each divided by batch_size; input_tokens is per query"
    (args.output/"metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")


if __name__=="__main__":
    main()
