"""Check PEFT reset behavior without model downloads or GPU requirements."""
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen2Config, Qwen2ForCausalLM


def test_unload_restores_base_and_new_adapter_starts_clean():
    torch.manual_seed(7)
    base=Qwen2ForCausalLM(Qwen2Config(vocab_size=64,hidden_size=16,
        intermediate_size=32,num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2))
    initial={k:v.detach().clone() for k,v in base.state_dict().items()}
    for cycle in range(2):
        model=get_peft_model(base,LoraConfig(r=4,lora_alpha=8,lora_dropout=0,
            target_modules=['q_proj','v_proj'],bias='none',task_type=TaskType.CAUSAL_LM))
        assert all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
        optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01)
        ids=torch.tensor([[1,2,3],[4,5,6]])
        loss=model(input_ids=ids,labels=ids).loss
        loss.backward(); optimizer.step()
        assert any(p.abs().sum()>0 for n,p in model.named_parameters() if 'lora_B' in n)
        base=model.unload()
        assert set(base.state_dict())==set(initial)
        assert all(torch.equal(v,initial[k]) for k,v in base.state_dict().items())
