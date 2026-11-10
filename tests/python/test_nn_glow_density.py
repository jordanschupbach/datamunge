"""Native explicit ActNorm proposals, signed mixing and live compact Glow fields."""
import gc
import math
import pytest
from pydatamunge import nn as n


def test_calibration_moments_are_stopped_and_installation_is_explicit():
    x=n.from_vector([1,2,3,6],[2,2]).requires_grad_()
    fit=n.FlowActNormCalibration(x)
    assert list(fit.mean.values())==[2,4] and list(fit.variance.values())==[1,4]
    assert list(fit.bias.values())==[-2,-4] and fit.mean.dtype()==n.DType.Float64
    assert fit.log_scale.dtype()==n.DType.Float32
    for field in [fit.mean,fit.variance,fit.bias,fit.log_scale]:assert not field.requires_grad()
    p=n.ParameterStore(9503)
    assert n.equal_bits(n.FlowActNorm(p,'norm',x).value,x)
    bias,scale=fit.bias,fit.log_scale;del fit;gc.collect()
    p.set('norm/bias',bias);p.set('norm/log_scale',scale);before=list(p.flatten())
    out=n.FlowActNorm(p,'norm',x);rec=n.FlowActNorm(p,'norm',out.value,True)
    assert n.max(n.abs(n.sub(rec.value,x))).item()<1e-6
    assert n.add(rec.logdet,out.logdet).item()==0
    n.FlowActNorm(p,'norm',n.full([2,2],100));assert list(p.flatten())==before


def test_negative_mixing_log_absolute_determinant_inverse_and_weight_higher_paths():
    p=n.ParameterStore(1,n.DType.Float64)
    p.set('mix/W',n.from_vector_double([1,2,3,4],[2,2]))
    x=n.from_vector_double([.2,-.7,1.1,.4,-.3,2],[3,2]).requires_grad_()
    out=n.FlowMix2D(p,'mix',x);rec=n.FlowMix2D(p,'mix',out.value,.35,True)
    assert abs(out.logdet.item()-math.log(2))<1e-14
    assert n.max(n.abs(n.sub(rec.value,x))).item()<1e-14
    assert n.add(out.logdet,rec.logdet).item()==0
    before=list(p.flatten());graph=n.GradOptions();graph.retain_graph=True;graph.create_graph=True
    loss=n.sub(n.sum(n.square(out.value)),out.logdet)
    gradient=n.grad(loss,[p.at('mix/W')],graph)[0]
    assert n.grad(n.sum(gradient),[x],graph)[0].defined()
    value=out.value;del out,rec,p;gc.collect()
    assert n.grad(n.sum(value),[x])[0].defined()


def test_full_glow_flat_fields_and_per_row_inverse_survive_collection_and_keep_store_immutable():
    p=n.ParameterStore(9503,n.DType.Float64)
    x=n.from_vector_double([.4,-.7,1.2,.3,-1,2],[3,2]).requires_grad_()
    path=n.GlowDensity(p,x);assert p.count()==296 and p.size()==28
    assert list(path.logdet.shape())==[3]
    before=list(p.flatten());rec=n.GlowInverse(p,path.latent)
    assert n.max(n.abs(n.sub(rec.value,x))).item()<1e-14
    assert n.max(n.abs(n.add(rec.logdet,path.logdet))).item()<1e-14
    density=n.NormalFlowDensity(path.latent,path.logdet,True)
    norm,mix,scale,shift,loss=path.actnorm_output1,path.mixed2,path.coupling_scale2,path.coupling_shift3,density.loss
    del path,density,rec;gc.collect()
    graph=n.GradOptions();graph.retain_graph=True;graph.create_graph=True
    first=n.grad(loss,[x],graph)[0];assert n.grad(n.sum(first),[x],graph)[0].defined()
    assert list(p.flatten())==before
    del p;gc.collect()
    for field in [norm,mix,scale,shift]:assert n.grad(n.sum(field),[x],graph)[0].defined()


def test_invalid_calibration_names_matrices_and_scales_reject_without_initial_writes():
    for epsilon in [0,-1,float('inf'),float('nan')]:
        with pytest.raises(RuntimeError):n.FlowActNormCalibration(n.zeros([2,2]),epsilon)
    p=n.ParameterStore(9503)
    for bad in [n.zeros([2]),n.zeros([1,3]),n.zeros([0,2]),n.full([1,2],float('nan'))]:
        with pytest.raises(RuntimeError):n.GlowDensity(p,bad)
        with pytest.raises(RuntimeError):n.GlowInverse(p,bad)
        assert p.size()==0
    with pytest.raises(RuntimeError):n.FlowActNorm(p,'',n.zeros([2,2]))
    with pytest.raises(RuntimeError):n.FlowMix2D(p,'mix',n.zeros([2,2]),float('inf'))
    assert p.size()==0
    n.GlowDensity(p,n.zeros([1,2]));p.set('mix0/W',n.zeros([2,2]))
    with pytest.raises(RuntimeError):n.GlowDensity(p,n.zeros([1,2]))
    with pytest.raises(RuntimeError):n.GlowInverse(p,n.zeros([1,2]))
    p.set('mix0/W',n.from_vector([1,0,0,1],[2,2]));p.set('actnorm0/log_scale',n.full([2],1000))
    with pytest.raises(RuntimeError):n.GlowDensity(p,n.zeros([1,2]))
    with pytest.raises(RuntimeError):n.GlowInverse(p,n.zeros([1,2]))
