from experiments.study_d.run import evaluate_prefix


class Tokenizer:
    def decode(self,tokens,skip_special_tokens):
        assert skip_special_tokens
        return ''.join({1:'Answer: ',2:'5',3:'\nActually uncertain',9:''}[t] for t in tokens)


def test_later_reasoning_can_invalidate_a_previously_correct_prefix():
    tokens=[1,2,3,9]
    short=evaluate_prefix(Tokenizer(),tokens,{9},2,5)
    long=evaluate_prefix(Tokenizer(),tokens,{9},4,5)
    assert short['correct'] and short['token_cap_reached']
    assert not long['correct'] and not long['format_valid']
    assert long['eos_terminated'] and not long['token_cap_reached']


def test_eos_stops_both_budgets_before_padding():
    tokens=[1,2,9,9,9]
    short=evaluate_prefix(Tokenizer(),tokens,{9},4,5)
    long=evaluate_prefix(Tokenizer(),tokens,{9},512,5)
    assert short==long
    assert long['correct'] and long['output_tokens']==3
