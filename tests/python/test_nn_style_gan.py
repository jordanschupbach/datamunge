"""Native style modulation, supplied-noise contracts, ownership and explicit writes."""
import gc
import pytest
from pydatamunge import nn as n


def inputs():
    t=n.DType.Float64
    g,d=n.ParameterStore(6607,t),n.ParameterStore(6619,t)
    z=n.GaussianNoise([8],2,6659)
    a=n.GaussianNoise([4,4,1],2,6653);b=n.GaussianNoise([8,8,1],2,6661)
    return g,d,n.full([2,8,8,1],.3,t),n.cast(z,t),n.cast(a,t),n.cast(b,t)


def test_population_spatial_moments_and_live_higher_paths():
    x=n.from_vector_double([1,3,5,7],[1,2,2,1]).requires_grad_();y=n.spatial_instance_normalize(x)
    assert n.mean(y).item()==0
    assert n.mean(n.square(y)).item()==pytest.approx(5/(5+1e-8))
    graph=n.GradOptions();graph.create_graph=True
    weighted=n.mul(y,n.from_vector_double([1,0,2,-1],x.shape()))
    first=n.grad(n.sum(weighted),[x],graph)[0]
    assert n.max(n.abs(n.grad(n.sum(n.square(first)),[x])[0])).item()>0
    with pytest.raises(RuntimeError):n.spatial_instance_normalize(x,0)


def test_modulated_kernel_analytic_normalization_partials_and_owning_fields():
    t=n.DType.Float64;x=n.from_vector_double([1,0],[1,1,1,2]).requires_grad_();w=n.from_vector_double([3,4],[1,1,2,1]).requires_grad_();s=n.ones([1,2],t).requires_grad_()
    o=n.modulated_conv2d(x,w,s);image=o.image;factor=o.demodulator;del o;gc.collect()
    assert image.item()==pytest.approx(.6) and factor.item()==pytest.approx(.2)
    graph=n.GradOptions();graph.create_graph=True
    grads=n.grad(n.sum(image),[x,w,s],graph)
    assert list(grads[1].values())==pytest.approx([.128,-.096])
    assert list(grads[2].values())==pytest.approx([.384,-.384])
    assert n.max(n.abs(n.grad(n.sum(n.square(grads[1])),[w])[0])).item()>0
    plain=n.modulated_conv2d(x,w,s,False);assert plain.image.item()==3 and plain.demodulator.item()==1
    assert n.modulated_conv2d(x,w,n.zeros(s.shape(),t)).image.item()==0
    with pytest.raises(RuntimeError):n.modulated_conv2d(x,w,s,True,0)
    with pytest.raises(RuntimeError):n.modulated_conv2d(x,w,n.ones([2,2],t))


@pytest.mark.parametrize('v2',[False,True])
def test_census_mapping_style_mixing_initial_noise_and_read_only_inference(v2):
    g,d,x,z,a,b=inputs();o=n.StyleGan(g,d,x,z,a,b,v2)
    assert (g.count(),g.size(),d.count(),d.size())==(625 if v2 else 661,19 if v2 else 17,493,6)
    before=(list(g.flatten()),list(d.flatten()));w=n.StyleGanMap(g,z)
    plain=n.StyleGanSynthesize(g,w,w,a,b,v2)
    assert n.equal_bits(o.fake,plain.image) and n.equal_bits(o.mapped,w)
    assert n.equal_bits(plain.image,n.StyleGanSynthesize(g,w,w,n.mul(a,2),n.mul(b,3),v2).image)
    other=n.StyleGanMap(g,n.mul(z,-1));mix=n.StyleGanSynthesize(g,w,other,a,b,v2)
    assert n.equal_bits(plain.first,mix.first)
    assert n.max(n.abs(n.sub(plain.image,mix.image))).item()>1e-8
    if v2:
        scaled=n.StyleGanSynthesize(g,w,w,a,b,True,n.mul(plain.first_style,2))
        assert n.max(n.abs(n.sub(plain.image,scaled.image))).item()<3e-6
        assert n.max(n.abs(n.sub(n.div(n.mul(scaled.first_demodulator,2),plain.first_demodulator),1))).item()<2e-4
    assert (list(g.flatten()),list(d.flatten()))==before


@pytest.mark.parametrize('v2',[False,True])
def test_noise_learnability_and_psi_zero_needs_common_noise(v2):
    g,d,x,z,a,b=inputs();a.requires_grad_();o=n.StyleGan(g,d,x,z,a,b,v2)
    keep=n.GradOptions();keep.retain_graph=True
    assert n.sum(n.abs(n.grad(o.g_loss,[a],keep)[0])).item()==0
    partial=n.grad(o.g_loss,[g.at('layer4/noise_strength'),g.at('layer8/noise_strength')])
    assert sum(n.sum(n.abs(p)).item() for p in partial)>0
    for name in ['layer4/noise_strength','layer8/noise_strength']:g.set(name,n.full([4],.2,g.dtype()))
    w=n.StyleGanMap(g,z);center=n.tile(n.mean(w,[0],True),[2,1])
    varied=n.StyleGanSynthesize(g,center,center,a,b,v2).image
    common=n.StyleGanSynthesize(g,center,center,n.take_rows(a,[0,0]),n.take_rows(b,[0,0]),v2).image
    assert n.max(n.abs(n.sub(n.select(varied,0,0),n.select(varied,0,1)))).item()>1e-8
    assert n.max(n.abs(n.sub(n.select(common,0,0),n.select(common,0,1)))).item()<1e-13


def test_explicit_d_then_fresh_g_changes_gradient_and_respects_owners():
    g,d,x,z,a,b=inputs();o=n.StyleGan(g,d,x,z,a,b,True);keep=n.GradOptions();keep.retain_graph=True
    stale=n.grad(o.g_loss,g.parameters(),keep);dg=n.grad(o.d_loss,d.parameters());cfg=n.OptimizerConfig();cfg.kind=n.OptimizerKind.Adam;cfg.beta1=.5
    go,do=n.Optimizer(g,.0002,cfg),n.Optimizer(d,.0002,cfg);before=list(g.flatten());do.step(dg);assert list(g.flatten())==before
    dafter=list(d.flatten());gg=n.grad(n.StyleGan(g,d,x,z,a,b,True).g_loss,g.parameters())
    assert max(n.max(n.abs(n.sub(p,q))).item() for p,q in zip(stale,gg))>1e-12
    go.step(gg);assert list(d.flatten())==dafter and go.iterations()==do.iterations()==1
    for name in d.names():d.set(name,n.zeros(d.at(name).shape(),d.dtype()))
    out=n.StyleGan(g,d,x,z,a,b,True);assert out.d_loss.item()==.5 and out.g_loss.item()==1


def test_parent_collection_higher_paths_and_fail_before_parameter_creation():
    g,d,x,z,a,b=inputs();z.requires_grad_();o=n.StyleGan(g,d,x,z,a,b,True);loss=o.g_loss;image=o.fake;del o,g,d,x,a,b;gc.collect()
    graph=n.GradOptions();graph.create_graph=True;first=n.grad(loss,[z],graph)[0]
    assert list(image.shape())==[2,8,8,1] and n.max(n.abs(n.grad(n.sum(n.square(first)),[z])[0])).item()>0
    g,d,x,z,a,b=inputs()
    for args in [(g,g,x,z,a,b,True),(g,d,x,z,n.ones([1,4,4,1],g.dtype()),b,True)]:
        with pytest.raises(RuntimeError):n.StyleGan(*args)
    with pytest.raises(RuntimeError):n.StyleGanSynthesize(g,z,z,a,b,False,n.ones([2,4],g.dtype()))
    assert g.size()==d.size()==0
