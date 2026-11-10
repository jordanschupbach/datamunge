"""Owning native diffusion tables, source rounding and live epsilon/reverse graphs."""
import gc
import math
import pytest
from pydatamunge import nn as n


def times():return n.from_vector_int64([0,31],[2])
def state(dtype=None):return n.reshape(n.sub(n.div(n.arange(0,128),128.),.5),[2,8,8,1]) if dtype is None else n.cast(state(),dtype)
def keep():
    o=n.GradOptions();o.retain_graph=True;o.create_graph=True;return o


def test_cosine_schedule_and_live_coefficient_table_selection():
    s=n.CosineDiffusionSchedule();assert list(s.alpha_bar.shape())==[32]
    assert list(s.alpha_bar.values())[-1]==2.369970271136327e-6
    assert list(s.posterior_variance.values())[0]==0 and list(s.beta.values())[-1]==.999
    table=n.from_vector_double([.1,.2,.3,.4],[4]).requires_grad_()
    out=n.DiffusionExtract(table,n.from_vector_int64([3,0,3],[3]),2,n.DType.Float64)
    assert list(out.values())==[.4,.1,.4]
    g=n.grad(n.sum(n.square(out)),[table],keep())[0]
    assert list(g.values())==pytest.approx([.2,0,0,1.6])
    assert list(n.grad(n.sum(g),[table])[0].values())==[2,0,0,4]
    values=s.beta;del s;gc.collect();assert len(values.values())==32


def test_source_float32_time_features_and_forward_coefficients_survive_double():
    f=n.DiffusionTimeEmbedding(times());d=n.DiffusionTimeEmbedding(times(),31,n.DType.Float64)
    assert n.equal_bits(n.cast(f,n.DType.Float64),d) and not d.requires_grad()
    assert list(d.values())[:8]==[0,0,0,0,1,1,1,1] and abs(list(d.values())[8])>1e-8
    s=n.CosineDiffusionSchedule();t=n.from_vector_int64([1,17],[2])
    x=n.full([2,3],.25,n.DType.Float64);e=n.full([2,3],-.1,n.DType.Float64)
    y=n.DiffusionNoised(x,e,t,s)
    signal=n.cast(n.DiffusionExtract(s.sqrt_alpha_bar,t,2),n.DType.Float64)
    sigma=n.cast(n.DiffusionExtract(s.sqrt_one_minus_alpha_bar,t,2),n.DType.Float64)
    assert n.equal_bits(y,n.add(n.mul(signal,x),n.mul(sigma,e)))
    full=n.add(n.mul(n.DiffusionExtract(s.sqrt_alpha_bar,t,2,n.DType.Float64),x),n.mul(n.DiffusionExtract(s.sqrt_one_minus_alpha_bar,t,2,n.DType.Float64),e))
    assert n.max(n.abs(n.sub(full,y))).item()>1e-10


def test_last_reverse_transition_ignores_innovation_and_clipping_is_explicit():
    s=n.CosineDiffusionSchedule();x=n.from_vector_double([-.25,.25,2,-2],[2,2]);e=n.zeros([2,2],n.DType.Float64)
    a=n.DdpmReverse(x,times(),e,n.ones([2,2],n.DType.Float64),s)
    b=n.DdpmReverse(x,times(),e,n.full([2,2],123,n.DType.Float64),s)
    assert list(a.variance.values())[0]==0
    assert n.equal_bits(n.slice_axis(a.value,0,0,1),n.slice_axis(b.value,0,0,1))
    assert list(a.x0_raw.values())[2]>100 and list(a.x0_clipped.values())[2:]==[1,-1]
    assert n.max(n.abs(n.sub(a.value,b.value))).item()>1


def test_model_census_host_owned_adam_and_flat_hidden_lifetimes():
    p=n.ParameterStore(11003,n.DType.Float64);x=state(n.DType.Float64).requires_grad_();o=n.DdpmEpsilon(p,x,times())
    assert p.count()==5504 and p.size()==6 and list(o.hidden1.shape())==[2,32]
    before=list(p.flatten());config=n.OptimizerConfig();config.kind=n.OptimizerKind.Adam;optimizer=n.Optimizer(p,.003,config)
    gradients=n.grad(n.mean(n.square(n.sub(o.epsilon,.2))),p.parameters());optimizer.step(gradients)
    assert optimizer.iterations()==1 and before!=list(p.flatten())
    o=n.DdpmEpsilon(p,x,times());fields=[o.epsilon,o.hidden1,o.hidden2];embedding=o.time_embedding;del o,p,optimizer;gc.collect()
    for value in fields:
        first=n.grad(n.sum(n.square(value)),[x],keep())[0]
        assert n.grad(n.sum(first),[x],keep())[0].defined()
    assert not embedding.requires_grad() and list(embedding.shape())==[2,8]


def test_noising_and_reverse_graphs_survive_parent_collection():
    s=n.CosineDiffusionSchedule();x=n.full([2,3],.2,n.DType.Float64).requires_grad_();e=n.full([2,3],.1,n.DType.Float64).requires_grad_();t=n.from_vector_int64([4,17],[2])
    state_value=n.DiffusionNoised(x,e,t,s);out=n.DdpmReverse(state_value,t,e,n.ones([2,3],n.DType.Float64),s)
    fields=[state_value,out.x0_raw,out.mean,out.value];del out,s;gc.collect()
    for value in fields:
        first=n.grad(n.sum(n.square(value)),[x],keep())[0]
        assert n.max(n.abs(first)).item()>1e-5 and n.grad(n.sum(first),[x],keep())[0].defined()


def test_strides_warm_immutability_and_no_additional_rng_draws():
    p=n.ParameterStore(11003);q=n.ParameterStore(11003);x=n.transpose(state(),[0,2,1,3])
    a=n.DdpmEpsilon(p,x,times());n.DdpmEpsilon(q,x,times());before=list(p.flatten())
    b=n.DdpmEpsilon(p,x.contiguous(),times());assert n.equal_bits(a.epsilon,b.epsilon) and before==list(p.flatten())
    assert n.equal_bits(p.get('extra',[7],n.init_normal(.1)),q.get('extra',[7],n.init_normal(.1)))


def test_invalid_geometry_times_options_and_unsafe_math_reject():
    for count in [-1,0,1]:
        with pytest.raises(RuntimeError):n.CosineDiffusionSchedule(count)
    for offset in [-1,math.nan,1e300]:
        with pytest.raises(RuntimeError):n.CosineDiffusionSchedule(32,offset)
    for t in [n.zeros([2]),n.from_vector_int64([-1],[1]),n.from_vector_int64([32],[1])]:
        with pytest.raises(RuntimeError):n.DiffusionExtract(n.CosineDiffusionSchedule().beta,t)
    p=n.ParameterStore(11003)
    for x in [n.zeros([2,64]),n.zeros([0,8,8,1]),n.full([2,8,8,1],math.nan),state(n.DType.Float64)]:
        with pytest.raises(RuntimeError):n.DdpmEpsilon(p,x,times())
    assert p.size()==0
    s=n.CosineDiffusionSchedule();s.posterior_variance=n.full([32],-1,n.DType.Float64)
    with pytest.raises(RuntimeError):n.DdpmReverse(n.zeros([2,3]),times(),n.zeros([2,3]),n.zeros([2,3]),s)
    with pytest.raises(RuntimeError):n.DdpmReverse(n.zeros([2,3]),times(),n.zeros([2,3]),n.zeros([2,3]),n.DiffusionScheduleOutput())
    with pytest.raises(RuntimeError):n.DiffusionTimeEmbedding(times(),0)
