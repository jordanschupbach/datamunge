"""Native continuous field traces, source RK4 rounding and owning unrolled graphs."""
import gc
import math
import struct
import pytest
from pydatamunge import nn as n


def keep():
    g=n.GradOptions();g.retain_graph=True;g.create_graph=True;return g


def points():return n.from_vector_double([.4,-.7,1.2,.3,-1,2],[3,2])


def test_field_actual_trace_both_probe_contractions_and_owning_higher_graphs():
    p=n.ParameterStore(10003,n.DType.Float64);x=points().requires_grad_();f=n.CnfField(p,x,.5)
    a=n.grad(n.sum(n.slice_axis(f.velocity,1,0,1)),[x],keep())[0];b=n.grad(n.sum(n.slice_axis(f.velocity,1,1,2)),[x],keep())[0]
    exact=n.add(n.select(a,1,0),n.select(b,1,1));plus=n.sum(n.add(a,b),[1])
    minus=n.add(n.sub(n.sub(n.select(a,1,0),n.select(a,1,1)),n.select(b,1,0)),n.select(b,1,1))
    assert n.max(n.abs(n.sub(exact,f.exact_trace))).item()<1e-16
    assert n.max(n.abs(n.sub(plus,f.hutchinson_plus))).item()<1e-16
    assert n.max(n.abs(n.sub(minus,f.hutchinson_minus))).item()<1e-16
    assert n.max(n.abs(n.sub(plus,exact))).item()>1e-4
    values=[f.exact_trace,f.hutchinson_plus,f.hutchinson_minus,n.mul(n.add(f.hutchinson_plus,f.hutchinson_minus),.5)]
    for mode,value in enumerate(values):assert n.equal_bits(n.CnfField(p,x,.5,mode).divergence,value)
    trace=f.exact_trace;velocity=f.velocity;del f,p;gc.collect()
    for value in [trace,velocity]:
        first=n.grad(n.sum(value),[x],keep())[0]
        assert n.grad(n.sum(n.square(first)),[x],keep())[0].defined()


def test_structural_time_rounds_to_float32_before_double_field_computation():
    p=n.ParameterStore(10003,n.DType.Float64);x=n.zeros([1,2],n.DType.Float64);n.CnfField(p,x,0)
    assert p.count()==50 and p.size()==4
    wi=[0.]*24;wi[16]=1;wo=[0.]*16;wo[0]=1
    p.set('field/hidden/W',n.from_vector_double(wi,[3,8]));p.set('field/output/W',n.from_vector_double(wo,[8,2]))
    p.set('field/hidden/b',n.zeros([8],n.DType.Float64));p.set('field/output/b',n.zeros([2],n.DType.Float64))
    time=.1+1e-10;out=n.CnfField(p,x,time);rounded=struct.unpack('f',struct.pack('f',time))[0]
    assert abs(list(out.velocity.values())[0]-.3*math.tanh(rounded))<1e-16
    assert abs(list(out.velocity.values())[0]-.3*math.tanh(time))>1e-12
    assert out.divergence.item()==0


def test_density_inverse_and_quarter_fields_survive_collection_with_higher_paths():
    p=n.ParameterStore(10003,n.DType.Float64);x=points().requires_grad_();out=n.CnfDensity(p,x)
    assert p.count()==52 and p.size()==5 and list(out.stage_divergence2.shape())==[3]
    assert n.equal_bits(out.stage3,out.unscaled)
    rec=n.CnfInverse(p,out.latent);assert n.max(n.abs(n.sub(rec.value,x))).item()<1e-7
    assert n.max(n.abs(n.add(rec.logdet,out.logdet))).item()<1e-7
    quarter=out.stage1;cumulative=out.stage_logdet2;endpoint=out.stage_velocity3;inverse=rec.value
    del out,rec,p;gc.collect()
    for value in [quarter,cumulative,endpoint,inverse]:
        first=n.grad(n.sum(value),[x],keep())[0]
        assert n.grad(n.sum(n.square(first)),[x],keep())[0].defined()


def test_host_owned_adam_and_reverse_solver_parameter_gradients():
    p=n.ParameterStore(10003);x=n.from_vector([.4,-.7,1.2,.3,-1,2],[3,2]);n.CnfDensity(p,x)
    config=n.OptimizerConfig();config.kind=n.OptimizerKind.Adam;optimizer=n.Optimizer(p,.003,config)
    before=list(p.flatten());out=n.CnfDensity(p,x);gradients=n.grad(n.NormalFlowDensity(out.latent,out.logdet).loss,p.parameters());optimizer.step(gradients)
    assert optimizer.iterations()==1 and list(p.flatten())!=before
    inv=n.CnfInverse(p,x);objective=n.add(n.mean(n.sum(n.square(inv.value),[1]),[0]),n.mul(n.mean(inv.logdet,[0]),.3))
    assert len(n.grad(objective,p.parameters()))==5


def test_step_counts_reverse_checkpoints_and_cumulative_trace_fields():
    p=n.ParameterStore(10003,n.DType.Float64);x=points()
    for inverse in [False,True]:
        for steps in [4,8,12,16]:
            out=n.CnfIntegrate(p,x,inverse,steps)
            assert n.equal_bits(out.latent,out.stage3) and n.equal_bits(out.logdet,out.stage_logdet3)
            assert list(out.stage_velocity1.shape())==[3,2]
            assert n.max(n.abs(n.sub(out.stage_logdet3,out.stage_logdet0))).item()>1e-5
    assert p.count()==50 and p.size()==4


def test_invalid_shapes_time_trace_modes_steps_and_global_scale_reject():
    p=n.ParameterStore(10003)
    for bad in [n.zeros([2]),n.zeros([1,3]),n.zeros([0,2]),n.full([1,2],float('nan'))]:
        with pytest.raises(RuntimeError):n.CnfDensity(p,bad)
    x=n.zeros([1,2])
    for steps in [0,2,3,5,6,10]:
        with pytest.raises(RuntimeError):n.CnfDensity(p,x,steps)
        with pytest.raises(RuntimeError):n.CnfInverse(p,x,steps)
    for mode in [-1,4]:
        with pytest.raises(RuntimeError):n.CnfField(p,x,.5,mode)
        with pytest.raises(RuntimeError):n.CnfIntegrate(p,x,False,8,mode)
    with pytest.raises(RuntimeError):n.CnfField(p,x,float('nan'))
    assert p.size()==0
    n.CnfDensity(p,x);p.set('log_scale',n.full([2],1000))
    with pytest.raises(RuntimeError):n.CnfDensity(p,x)
    with pytest.raises(RuntimeError):n.CnfInverse(p,x)
