"""Native Row-LSTM gate order, linked zeros, causal graphs and owning records."""
import gc
import math
import pytest
from pydatamunge import nn as n


def test_row_step_uses_forget_bias_and_has_live_state_paths():
    t=n.DType.Float64
    pre=n.zeros([1,1,3,16],t).requires_grad_()
    h=n.zeros([1,1,3,4],t).requires_grad_()
    c=n.full([1,1,3,4],2,t).requires_grad_()
    w=n.zeros([1,3,4,16],t).requires_grad_()
    o=n.PixelRowLstmStep(pre,h,c,w)
    assert list(o.forget_gate.values())==pytest.approx([1/(1+math.exp(-1))]*12,abs=1e-15)
    assert list(o.cell.values())==pytest.approx([2/(1+math.exp(-1))]*12,abs=1e-15)
    assert list(o.input_gate.values())==[.5]*12 and list(o.candidate.values())==[0]*12
    fields=[o.gates,o.input_gate,o.forget_gate,o.output_gate,o.candidate,o.cell,o.hidden]
    del o;gc.collect()
    graph=n.GradOptions();graph.create_graph=True
    g=n.grad(n.sum(fields[5]),[c],graph)[0]
    assert list(g.values())==pytest.approx([1/(1+math.exp(-1))]*12,abs=1e-15)
    assert n.max(n.abs(n.grad(n.sum(g),[pre])[0])).item()>.1
    with pytest.raises(RuntimeError):n.PixelRowLstmStep(n.zeros([1,2,3,16],t),h,c,w)


def test_model_census_zero_paths_causality_and_fields_survive_store_collection():
    t=n.DType.Float64;p=n.ParameterStore(8303,t)
    x=n.full([1,5,7,1],.25,t).requires_grad_()
    o=n.PixelRnn(p,x)
    assert [p.count(),p.size()]==[357,5]
    fields=[o.initial_hidden,o.initial_cell,o.input_gates,o.row_gates,o.hidden,o.cell,o.logits]
    del o,p;gc.collect()
    assert n.equal_bits(fields[0],fields[1]) and fields[0].requires_grad()
    keep=n.GradOptions();keep.retain_graph=True
    zero=n.grad(n.add(n.sum(fields[0]),n.sum(fields[1])),[x],keep)[0]
    assert n.max(n.abs(zero)).item()==0
    g=n.grad(n.sum(n.slice_axis(n.reshape(fields[-1],[35]),0,17,18)),[x])[0]
    assert all(v==0 for v in list(g.values())[17:])
    assert list(fields[2].shape())==[1,5,7,16] and list(fields[4].shape())==[1,5,7,4]


def test_rectangles_immutable_inference_and_invalid_inputs_before_leaves():
    p=n.ParameterStore(8303)
    x=n.full([2,3,5,1],.2)
    o=n.PixelRnn(p,x);before=list(p.flatten())
    assert n.equal_bits(n.take_rows(o.logits,[0]),n.PixelRnn(p,n.take_rows(x,[0])).logits)
    assert before==list(p.flatten())
    p=n.ParameterStore(1)
    with pytest.raises(RuntimeError):n.PixelRnn(p,n.zeros([1,3,5,2]))
    with pytest.raises(RuntimeError):n.PixelRnn(p,n.zeros([1,3,5,1],n.DType.Int64))
    assert p.size()==0
