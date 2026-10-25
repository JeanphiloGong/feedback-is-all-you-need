from types import SimpleNamespace

import torch
from transformers import BatchEncoding

from experiments.study_c.run import generate,parse_answer,reasoning_prompt,REASONING


def test_parser_only_accepts_a_complete_final_answer_line():
    assert parse_answer('2+3=5\nAnswer: 5')==5
    assert parse_answer('Reasoning\n answer: 7 \n\n')==7
    for invalid in ['', '5', 'Answer: 8', 'Answer: 12', 'Answer: 5\nActually unsure',
                    'The answer is 5.', 'Answer: 5 or 6']:
        assert parse_answer(invalid) is None


def test_reasoning_prompts_remove_both_conflicting_digit_only_instructions():
    class CaptureTokenizer:
        def apply_chat_template(self,messages,**kwargs):
            return '\n'.join(m['content'] for m in messages)
    text=reasoning_prompt(CaptureTokenizer(),dict(state=2,commands=['dax'],target=3),[])
    assert text.count(REASONING)==2
    assert 'return only one digit' not in text
    assert 'Return only the final state digit' not in text
    assert 'Known offsets:' not in text


def test_generation_counts_through_first_eos_and_keeps_invalid_truncations(monkeypatch):
    class Tokenizer:
        def apply_chat_template(self,messages,**kwargs):
            return '\n'.join(m['content'] for m in messages)

        def __call__(self,prompts,**kwargs):
            return BatchEncoding(dict(input_ids=torch.tensor([[0,1],[1,1]]),
                                      attention_mask=torch.tensor([[0,1],[1,1]])))

        def decode(self,tokens,**kwargs):
            assert kwargs['skip_special_tokens']
            return 'Answer: 3' if tokens==[3,8,9] else 'unfinished reasoning'

    class Model:
        device='cpu'
        generation_config=SimpleNamespace(eos_token_id=[9,10])

        def generate(self,**kwargs):
            assert kwargs['max_new_tokens']==128 and not kwargs['do_sample']
            # Token 8 represents a non-EOS special token removed in decoding.
            continuation=torch.tensor([[3,8,9]+[9]*125,[1]*128])
            return torch.cat((kwargs['input_ids'],continuation),dim=1)

    monkeypatch.setattr(torch.cuda,'synchronize',lambda:None)
    queries=[dict(state=0,commands=['dax'],target=target) for target in (3,7)]
    first,second=generate(Model(),Tokenizer(),queries,[],16)
    assert first['output_tokens']==3 and first['input_tokens']==1
    assert first['output_token_ids']==[3,8,9]
    assert first['correct'] and first['eos_terminated'] and not first['token_cap_reached']
    assert second['output_tokens']==128 and second['input_tokens']==2
    assert second['prediction'] is None and not second['correct']
    assert second['token_cap_reached'] and not second['eos_terminated']
    assert first['batch_wall_seconds']==second['batch_wall_seconds']
    assert first['batch_size']==second['batch_size']==2
    assert first['padded_generation_tokens']==second['padded_generation_tokens']==256
