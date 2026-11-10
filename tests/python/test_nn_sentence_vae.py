"""Native masked GRU sentence VAE, explicit generation and collapse diagnostics."""
import gc
import math
import pytest
from pydatamunge import nn


def sentences():
    return nn.from_vector_int64([3,5,8,11,13,2,0,4,7,10,12,16,2,0],[2,7])


def test_eighteen_fields_own_lifetime_and_keep_parameter_and_noise_paths():
    t=nn.DType.Float64
    p=nn.ParameterStore(3109,t)
    noise=nn.full([2,4],.2,t).requires_grad_()
    out=nn.SequenceVariationalAutoencoder(p,sentences(),noise)
    names=['decoder_inputs','encoder_state','encoder_states','mu','log_variance','z',
           'decoder_initial','decoder_states','logits','probability','predicted_ids','mask',
           'token_nll','nll','kl','correct','valid_tokens','loss']
    fields={name:getattr(out,name) for name in names}
    del out
    gc.collect()
    assert p.size()==22 and p.count()==3243
    assert tuple(fields['logits'].shape())==(2,7,19)
    assert fields['predicted_ids'].dtype()==nn.DType.Int32
    assert list(fields['valid_tokens'].values())==[6,6]
    assert all(all(math.isfinite(v) for v in a.values()) for a in fields.values())
    gradients=nn.grad(fields['loss'],list(p.parameters())+[noise])
    assert len(gradients)==23
    assert all(all(math.isfinite(v) for v in g.values()) for g in gradients)
    assert any(abs(v)>1e-10 for v in gradients[-1].values())
    teacher=nn.SequenceTeacher(p,fields['z'],fields['decoder_inputs'])
    assert list(teacher.probability.values())==list(fields['probability'].values())


def test_masked_encoder_prefix_only_decoder_and_generation_fields_survive_collection():
    p=nn.ParameterStore(3109)
    encoded=nn.SequenceEncoder(p,sentences())
    state,history=encoded.state,encoded.states
    del encoded
    gc.collect()
    assert p.size()==7 and p.count()==1352
    padded=nn.concat([sentences(),nn.zeros([2,1],nn.DType.Int64)],1)
    assert list(nn.SequenceEncoder(p,padded).state.values())==list(state.values())
    assert tuple(history.shape())==(2,7,16)
    decoder=nn.ParameterStore(3109)
    z=nn.full([2,4],.2).requires_grad_()
    teacher=nn.SequenceTeacher(decoder,z,nn.SequenceDecoderInputs(sentences()))
    names=['initial','states','logits','probability','predicted_ids']
    fields=[getattr(teacher,name) for name in names]
    del teacher
    generated=nn.SequenceGenerate(decoder,z,True,nn.full([2,7],.5,nn.DType.Float64))
    generation=[getattr(generated,name) for name in ['ids','logits','probability','ended']]
    del generated
    gc.collect()
    assert decoder.size()==11 and decoder.count()==1755
    assert all(all(math.isfinite(v) for v in field.values()) for field in fields+generation)
    dz=nn.grad(nn.sum(generation[1]),[z])[0]
    assert any(abs(v)>1e-10 for v in dz.values())
    assert not generation[0].requires_grad()


def test_explicit_adam_and_annealing_have_no_hidden_counter():
    p=nn.ParameterStore(3109)
    out=nn.SequenceVariationalAutoencoder(p,sentences(),nn.zeros([2,4]))
    before=tuple(p.flatten())
    assert [nn.CollapseBeta(True,k) for k in [0,17,18]]==[0,1,1]
    assert nn.CollapseBeta(False,0)==1
    objective=nn.AnnealedSequenceObjective(out,0)
    assert objective.item()==pytest.approx(sum(out.nll.values())/2)
    config=nn.OptimizerConfig()
    config.kind=nn.OptimizerKind.Adam
    optimizer=nn.Optimizer(p,.003,config)
    optimizer.step(nn.grad(objective,list(p.parameters())))
    assert optimizer.iterations()==1 and tuple(p.flatten())!=before


def test_host_mixture_summary_is_owned_and_retains_signed_estimates():
    t=nn.DType.Float64
    mu,lv=nn.zeros([8,4],t),nn.zeros([8,4],t)
    summary=nn.CollapseStatistics(mu,lv,[mu])
    unit,variance=summary.unit_kl,summary.mu_variance
    del summary,mu,lv
    gc.collect()
    assert list(unit)==[0]*4 and list(variance)==[0]*4
    mu=nn.zeros([8,4],t)
    negative=nn.CollapseStatistics(mu,nn.full([8,4],math.log(2),t),[mu])
    assert negative.analytic_kl>0 and negative.aggregate_kl<0 and negative.sampled_conditional_kl<0
    assert negative.active_units==0
    assert negative.sampled_conditional_kl==pytest.approx(negative.mixture_information+negative.aggregate_kl,abs=1e-14)


def test_invalid_inputs_are_rejected_before_leaves():
    p=nn.ParameterStore(1)
    for x in [nn.zeros([2,7]),nn.zeros([2,6],nn.DType.Int64),nn.full([2,7],19,nn.DType.Int32)]:
        with pytest.raises(RuntimeError):nn.SequenceVariationalAutoencoder(p,x,nn.zeros([2,4]))
    for u in [0,1,math.inf]:
        with pytest.raises(RuntimeError):nn.SequenceGenerate(p,nn.zeros([2,4]),True,nn.full([2,7],u,nn.DType.Float64))
    assert p.size()==0
    with pytest.raises(RuntimeError):nn.CollapseBeta(True,-1)
    with pytest.raises(RuntimeError):nn.CollapseStatistics(nn.zeros([2,4]),nn.zeros([2,4]),[])


def test_native_midpoint_uniform_stream_is_explicit_owned_and_open_interval():
    raw=[3499211612,581869302,3890346734,3586334585,545404204,4161255391]
    u=nn.MidpointUniformNoise([3,2],5489)
    assert list(u.values())==[(r+.5)/4294967296 for r in raw]
    assert u.dtype()==nn.DType.Float64 and not u.requires_grad()
    assert list(nn.MidpointUniformNoise([6],5489).values())==list(u.values())
    for seed in [0,2**31,2**32-1]:
        v=nn.MidpointUniformNoise([20,7],seed)
        assert all(0<x<1 for x in v.values())
        assert list(v.values())==list(nn.MidpointUniformNoise([20,7],seed).values())
        assert tuple(nn.SequenceGenerate(nn.ParameterStore(3109),nn.zeros([20,4]),True,v).ids.shape())==(20,7)
    assert tuple(nn.MidpointUniformNoise([],5489).shape())==()
    assert nn.MidpointUniformNoise([2,0,3],0).numel()==0
    with pytest.raises(RuntimeError):nn.MidpointUniformNoise([0,-1],1)
