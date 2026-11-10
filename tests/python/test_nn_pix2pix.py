"""Native paired translation contracts, branch ownership and explicit optimizer phases."""
import gc
import math
import pytest
from pydatamunge import nn as n


def inputs():
    t=n.DType.Float64
    return n.ParameterStore(6101,t),n.ParameterStore(6113,t),n.full([2,28,28,1],.2,t),n.ones([2,28,28,1],t)


def test_sobel_integer_threshold_zero_border_and_detached_ownership():
    x=n.ones([1,3,3,1],n.DType.Float64).requires_grad_()
    edge=n.quantized_sobel_edges(x)
    assert list(edge.shape())==[1,3,3,1] and not edge.requires_grad()
    assert n.sum(edge).item()==8 and edge.at([0,1,1,0])==0
    impulse=n.from_vector_double([0,0,0,0,64/255,0,0,0,0],[1,3,3,1])
    assert n.sum(n.quantized_sobel_edges(impulse,128)).item()==8
    assert n.sum(n.quantized_sobel_edges(impulse,129)).item()==0
    for bad in [n.ones([1,2,2,3]),n.full([1,2,2,1],-1),n.full([1,2,2,1],float('nan'))]:
        with pytest.raises(RuntimeError):n.quantized_sobel_edges(bad)
    with pytest.raises(RuntimeError):n.quantized_sobel_edges(x,-1)


def test_actual_skips_census_source_only_and_conditional_immutable_paths():
    g,d,x,c=inputs();generated=n.Pix2PixGenerate(g,c);o=n.Pix2Pix(g,d,x,c)
    assert (g.count(),g.size(),d.count(),d.size())==(1390,10,725,6)
    assert list(o.real_logit.shape())==[2,7,7,1]
    assert n.equal_bits(n.slice_axis(generated.joined14,3,0,4),generated.up14)
    assert n.equal_bits(n.slice_axis(generated.joined14,3,4,8),generated.down14)
    assert n.equal_bits(n.slice_axis(generated.joined28,3,2,3),c)
    before=(list(g.flatten()),list(d.flatten()))
    assert n.equal_bits(n.Pix2Pix(g,d,n.ones(x.shape(),x.dtype()),c).fake,o.fake)
    blank=n.zeros(c.shape(),c.dtype())
    assert not n.equal_bits(n.Pix2PixGenerate(g,blank).image,o.fake)
    assert not n.equal_bits(n.Pix2PixDiscriminate(d,blank,x),o.real_logit)
    assert (list(g.flatten()),list(d.flatten()))==before


def test_patch_mean_scales_and_paired_l1():
    g,d,x,c=inputs();n.Pix2Pix(g,d,x,c)
    for name in d.names():d.set(name,n.zeros(d.at(name).shape(),d.dtype()))
    o=n.Pix2Pix(g,d,x,c)
    assert o.discriminator_loss.item()==pytest.approx(2*math.log(2))
    assert o.g_adversarial.item()==pytest.approx(math.log(2))
    assert o.generator_loss.item()==pytest.approx(o.g_adversarial.item()+100*o.l1.item())
    assert n.equal_bits(n.Pix2Pix(g,d,x,c,0).generator_loss,o.g_adversarial)


def test_explicit_d_then_fresh_g_changes_only_selected_parameters():
    g,d,x,c=inputs();out=n.Pix2Pix(g,d,x,c);cfg=n.OptimizerConfig();cfg.kind=n.OptimizerKind.Adam;cfg.beta1=.5
    gd=n.grad(out.discriminator_loss,d.parameters());before_g=list(g.flatten());before_d=list(d.flatten())
    od=n.Optimizer(d,.0002,cfg);og=n.Optimizer(g,.0002,cfg);od.step(gd)
    assert list(g.flatten())==before_g and list(d.flatten())!=before_d
    after_d=list(d.flatten());fresh=n.Pix2Pix(g,d,x,c);og.step(n.grad(fresh.generator_loss,g.parameters()))
    assert list(d.flatten())==after_d and list(g.flatten())!=before_g
    assert (od.iterations(),og.iterations())==(1,1)


def test_owned_fields_higher_source_paths_and_invalid_before_initialization():
    g,d,x,c=inputs();c.requires_grad_();generated=n.Pix2PixGenerate(g,c);out=n.Pix2Pix(g,d,x,c)
    loss=out.generator_loss;join=generated.joined28;del out,generated,g,d,x;gc.collect()
    options=n.GradOptions();options.create_graph=True;first=n.grad(loss,[c],options)[0]
    second=n.grad(n.sum(n.square(first)),[c])[0]
    assert list(second.shape())==list(c.shape()) and n.max(n.abs(second)).item()>0
    assert n.equal_bits(n.slice_axis(join,3,2,3),c)
    g,d,x,c=inputs()
    for weight in [-1,float('inf'),float('nan')]:
        with pytest.raises(RuntimeError):n.Pix2Pix(g,d,x,c,weight)
    with pytest.raises(RuntimeError):n.Pix2Pix(g,g,x,c)
    with pytest.raises(RuntimeError):n.Pix2Pix(g,d,x,n.ones([1,28,28,1],x.dtype()))
    assert g.size()+d.size()==0
