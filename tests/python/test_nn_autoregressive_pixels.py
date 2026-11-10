"""Native causal masks, model/record lifetime and binary density contracts."""
import gc
import math
import pytest
from pydatamunge import nn as n


def test_binary_rounding_mask_geometry_and_midpoint_seed_are_native():
    out=n.BinaryPixels(n.full([2,28,28,1],.5).requires_grad_())
    gray,binary=out.gray,out.images
    del out;gc.collect()
    assert n.equal_bits(gray,n.full([2,8,8,1],.5))
    assert n.equal_bits(binary,n.ones([2,8,8,1])) and not gray.requires_grad()
    a=n.CausalPixelMask(1,8,False)
    assert list(a.shape())==[3,3,1,8] and n.sum(a).item()==32
    assert n.sum(n.CausalPixelMask(8,8,True)).item()==320
    u=n.BinaryPixelUniforms(2,8423)
    assert list(u.shape())==[2,64] and not u.requires_grad()
    assert n.equal_bits(u,n.BinaryPixelUniforms(2,8423))
    assert not n.equal_bits(u,n.BinaryPixelUniforms(2,8424))
    with pytest.raises(RuntimeError):n.BinaryPixels(n.full([1,28,28,1],1.1))
    with pytest.raises(RuntimeError):n.CausalPixelMask(0,8,False)
    with pytest.raises(RuntimeError):n.BinaryPixelUniforms(0,1)


def test_pixel_cnn_census_causal_partial_and_fields_survive_parent_and_store():
    p=n.ParameterStore(8401,n.DType.Float64)
    x=n.full([1,5,7,1],.25,n.DType.Float64).requires_grad_()
    out=n.PixelCnn(p,x)
    assert [p.count(),p.size()]==[1841,10]
    assert list(out.logits.shape())==[1,5,7,1]
    logits,hidden=out.logits,out.hidden
    leaves=list(p.parameters())
    del out,p;gc.collect()
    g=n.grad(n.sum(n.slice_axis(n.reshape(logits,[35]),0,17,18)),[x])[0]
    assert all(v==0 for v in list(g.values())[17:])
    assert list(hidden.shape())==[1,5,7,8] and len(leaves)==10
    fresh=n.ParameterStore(1)
    with pytest.raises(RuntimeError):n.PixelCnn(fresh,n.zeros([1,8,8,2]))
    assert fresh.size()==0
    with pytest.raises(RuntimeError):n.MaskedPixelConv(fresh,'bad',n.zeros([1,8,8,1]),n.full([3,3,1,8],.5))
    assert fresh.size()==0


def test_bernoulli_units_derivative_and_owning_flat_fields():
    z=n.zeros([2,3,5,1],n.DType.Float64).requires_grad_()
    out=n.BernoulliPixels(z,n.ones(z.shape(),z.dtype()))
    loss,per,pixel,q=out.loss,out.per_image_nll,out.pixel_nll,out.probability
    del out;gc.collect()
    assert loss.item()==pytest.approx(math.log(2),abs=1e-14)
    assert list(per.values())==pytest.approx([15*math.log(2)]*2,abs=1e-13)
    g=n.grad(loss,[z])[0]
    assert list(g.values())==pytest.approx([-.5/30]*30,abs=1e-15)
    assert list(pixel.shape())==[2,3,5,1] and n.mean(q).item()==.5
    with pytest.raises(RuntimeError):n.BernoulliPixels(z,n.full(z.shape(),.5,z.dtype()))
