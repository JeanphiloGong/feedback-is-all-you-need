import hashlib,json,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
name='Qwen/Qwen2.5-0.5B-Instruct'
revision='7ae557604adf67be50417f59c2c2f167def9a775'
torch.set_num_threads(4)
t=AutoTokenizer.from_pretrained(name,revision=revision,local_files_only=True)
m=AutoModelForCausalLM.from_pretrained(name,revision=revision,local_files_only=True,torch_dtype=torch.float32,device_map='cuda',attn_implementation='eager').eval()
print('transformers',__import__('transformers').__version__,flush=True)
questions=['What is 2 + 2?','What is 2 + 2? Answer with one digit.','Repeat exactly the word hello.','Write a Python function that adds two numbers.','What is the capital of France?']
for q in questions:
 p=t.apply_chat_template([dict(role='system',content='You are Qwen, created by Alibaba Cloud. You are a helpful assistant.'),dict(role='user',content=q)],tokenize=False,add_generation_prompt=True)
 x=t(p,return_tensors='pt',add_special_tokens=False).to(m.device)
 with torch.inference_mode():
  y=m.generate(**x,max_new_tokens=80,do_sample=False)
 print(json.dumps(dict(question=q,input_ids=x.input_ids[0].tolist(),text=t.decode(y[0,x.input_ids.shape[1]:],skip_special_tokens=False))),flush=True)
