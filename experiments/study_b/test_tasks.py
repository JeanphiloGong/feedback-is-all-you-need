import json

from experiments.study_b.tasks import COMMANDS, execute, make_environment, messages, symbolic_offsets


def test_queries_are_disjoint_and_have_exact_labels():
    for seed in (3,7,19):
        env=make_environment(seed)
        train={(q['state'],tuple(q['commands'])) for q in env['train']}
        test={(q['state'],tuple(q['commands'])) for q in env['direct']+env['compose']}
        assert not train & test
        assert len(train)==16 and len(test)==48
        for q in env['train']+env['direct']+env['compose']:
            assert q['target']==execute(q['state'],q['commands'],env['offsets'])
        assert symbolic_offsets(env['train'])==env['offsets']


def test_partial_command_knowledge_cannot_enter_prompt():
    q=dict(state=3,commands=['wug','dax'],target=5)
    altered=dict(q,target=2)
    assert messages(q,[])==messages(altered,[])
    assert 'Known offsets:' not in messages(q,[])[1]['content']


def test_corrupted_outputs_are_wrong_exactly_when_marked():
    env=make_environment(3)
    for clean,noise in zip(env['train'],env['corrupted']):
        assert (clean['target'] != noise['target'])==noise['corrupt']
        assert (clean['state'],clean['commands'])==(noise['state'],noise['commands'])


def test_offline_and_accumulated_feedback_prompts_are_identical():
    env=make_environment(4)
    history=[]
    for q in env['train']: history.append(q)
    assert json.dumps(messages(env['compose'][0],history))==json.dumps(messages(env['compose'][0],env['train']))


def test_composition_is_modular_and_commutative():
    offsets=[1,2,3,4]
    assert execute(7,['dax','wug'],offsets)==2
    assert execute(7,['wug','dax'],offsets)==2
    assert len(COMMANDS)==4
