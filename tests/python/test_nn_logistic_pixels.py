"""Native quantized RGB densities, bin tails and owning first/higher graphs."""
import gc
import math
import pytest
from pydatamunge import nn as n


def test_integer_area_half_up_and_owning_detached_records():
    raw=n.from_vector_int64([0,0,0,1,1,1,0,0,0,1,1,1],[1,2,2,3])
    out=n.RgbAreaPixels(raw,2);codes,image=out.codes,out.images
    del out,raw;gc.collect()
    assert list(codes.values())==[1,1,1] and codes.dtype()==n.DType.Int32
    assert list(image.shape())==[1,1,1,3]
    assert list(image.values())==pytest.approx([1/127.5-1]*3,abs=1e-7)
    assert not image.requires_grad()
    with pytest.raises(RuntimeError):n.RgbAreaPixels(n.zeros([1,4,4,3]))
    with pytest.raises(RuntimeError):n.RgbAreaPixels(n.full([1,4,4,3],256,n.DType.Int64))
    with pytest.raises(RuntimeError):n.RgbAreaPixels(n.zeros([1,4,4,3],n.DType.Int64),3)


def test_all_256_bins_include_normalized_endpoint_tails():
    image=n.from_vector_double([i/127.5-1 for i in range(256) for _ in range(3)],[256,3])
    out=n.DiscretizedLogisticRGB(image,n.zeros([256,10],n.DType.Float64),1)
    mass,weight,loss=out.channel_log_mass,out.mixture_log_probability,out.mean_channel_nll
    del out;gc.collect()
    assert list(n.sum(n.exp(mass),[0]).values())==pytest.approx([1]*3,abs=2e-5)
    assert list(weight.values())==[0]*256 and math.isfinite(loss.item())
    with pytest.raises(RuntimeError):n.DiscretizedLogisticRGB(image,n.zeros([256,20],n.DType.Float64))
    with pytest.raises(RuntimeError):n.DiscretizedLogisticRGB(n.full([1,3],1.001),n.zeros([1,30]))


def test_likelihood_retains_image_parameter_higher_paths_and_stops_clipped_scale():
    values=[.1]*10;values[4]=-8
    parameters=n.from_vector_double(values,[1,10]).requires_grad_()
    image=n.from_vector_double([-.6,.2,.8],[1,3]).requires_grad_()
    out=n.DiscretizedLogisticRGB(image,parameters,1)
    means,loss=out.means,out.mean_channel_nll;del out;gc.collect()
    assert list(means.values())==pytest.approx([.1,.1+math.tanh(.1)*(-.6),.1+math.tanh(.1)*(-.4)])
    graph=n.GradOptions();graph.create_graph=True
    g=n.grad(loss,[parameters,image],graph)
    assert list(g[0].values())[4]==0 and g[1].defined()
    second=n.grad(n.sum(g[0]),[parameters])[0]
    assert all(math.isfinite(v) for v in second.values())


def test_compact_model_record_graph_outlives_store_and_has_whole_pixel_causality():
    p=n.ParameterStore(8609,n.DType.Float64)
    x=n.full([1,2,3,3],.25,n.DType.Float64).requires_grad_()
    out=n.PixelCnnPP(p,x);assert [p.count(),p.size()]==[1410,23]
    head,loss,score,vertical=out.parameters,out.loss,out.per_image_nll,out.vertical
    assert score.item()/18==pytest.approx(loss.item(),abs=1e-12)
    del out,p;gc.collect()
    assert list(vertical.shape())==[1,2,3,4]
    g=n.grad(n.sum(n.slice_axis(n.reshape(head,[1,6,30]),1,2,3)),[x])[0]
    assert all(v==0 for v in list(g.values())[6:])
    fresh=n.ParameterStore(1)
    with pytest.raises(RuntimeError):n.PixelCnnPP(fresh,n.full([1,2,3,3],1.1))
    assert fresh.size()==0
