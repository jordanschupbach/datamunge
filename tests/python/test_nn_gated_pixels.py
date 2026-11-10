"""Native dual-stream geometry, discrete labels, owning graphs and paired gates."""
import gc
import math
import pytest
from pydatamunge import nn as n


def labels(count):
    return n.zeros([count],n.DType.Int64)


def test_stream_masks_gate_halves_and_higher_derivatives():
    assert n.sum(n.GatedPixelMask(1,True,True)).item()==24
    assert n.sum(n.GatedPixelMask(4,False,False)).item()==64
    assert list(n.GatedPixelMask(2,False,True,3).shape())==[1,3,2,3]
    x=n.from_vector_double([-1,0,1,2,0,1,-1,2],[1,1,1,8]).requires_grad_()
    y=n.GatedPixelGate(x)
    assert list(y.values())==pytest.approx([math.tanh(a)/(1+math.exp(-b)) for a,b in zip([-1,0,1,2],[0,1,-1,2])],abs=1e-15)
    graph=n.GradOptions();graph.create_graph=True
    g=n.grad(n.sum(y),[x],graph)[0]
    assert n.max(n.abs(n.grad(n.sum(n.square(g)),[x])[0])).item()>.01
    with pytest.raises(RuntimeError):n.GatedPixelMask(0,True,True)
    with pytest.raises(RuntimeError):n.GatedPixelGate(n.zeros([1,2,3,4]))


def test_conditional_census_owning_records_and_strict_causal_input_graph():
    for conditional,count,leaves in [(False,1073,23),(True,1553,29)]:
        p=n.ParameterStore(8501,n.DType.Float64)
        x=n.full([1,5,7,1],.25,n.DType.Float64).requires_grad_()
        o=n.GatedPixelCnn(p,x,labels(1),conditional)
        assert [p.count(),p.size()]==[count,leaves]
        fields=[o.vertical0,o.horizontal0,o.vertical1,o.horizontal1,o.vertical,o.horizontal,o.logits]
        del o,p;gc.collect()
        g=n.grad(n.sum(n.slice_axis(n.reshape(fields[-1],[35]),0,17,18)),[x])[0]
        assert all(v==0 for v in list(g.values())[17:])
        assert all(list(v.shape())==[1,5,7,4] for v in fields[:-1])
    rgb=n.ParameterStore(8609)
    out=n.GatedPixelConvolutional(rgb,n.zeros([2,4,6,3]),input_channels=3,output_channels=30)
    assert [rgb.count(),rgb.size()]==[1410,23] and list(out.logits.shape())==[2,4,6,30]


def test_class_effect_is_explicit_and_invalid_labels_reject_before_any_leaf():
    x=n.full([2,3,5,1],.25)
    for conditional in [False,True]:
        p=n.ParameterStore(8501)
        a=n.GatedPixelCnn(p,x,labels(2),conditional).logits
        before=list(p.flatten())
        b=n.GatedPixelCnn(p,x,n.ones([2],n.DType.Int64),conditional).logits
        if conditional:assert n.max(n.abs(n.sub(a,b))).item()>1e-6
        else:assert n.equal_bits(a,b)
        assert before==list(p.flatten())
    p=n.ParameterStore(1)
    with pytest.raises(RuntimeError):n.GatedPixelCnn(p,x,n.full([2],10,n.DType.Int64),False)
    with pytest.raises(RuntimeError):n.GatedPixelCnn(p,x,n.zeros([2]),True)
    with pytest.raises(RuntimeError):n.GatedPixelConvolutional(p,x,conditional=True)
    assert p.size()==0
