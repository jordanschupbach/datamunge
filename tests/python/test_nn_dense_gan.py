"""Native dense objectives, conditional paths, owning records and host-owned writes."""
import gc
import math
import pytest
from pydatamunge import nn


def test_raw_objective_algebra_and_hinge_boundary_derivatives():
    t = nn.DType.Float64
    real = nn.from_vector_double([0., 1., 2.],[3, 1]).requires_grad_()
    fake = nn.from_vector_double([-2., -1., 0., 1.],[4, 1]).requires_grad_()
    ls = nn.LeastSquaresGanObjectives(real, fake)
    assert ls.discriminator.item() == pytest.approx(.5*(2/3+1.5))
    assert ls.generator.item() == pytest.approx(1.75)
    assert list(nn.grad(ls.discriminator, [real])[0].values()) == pytest.approx([-1/3, 0, 1/3])
    hinge = nn.HingeGanObjectives(real, fake)
    assert hinge.discriminator.item() == pytest.approx(1/3+.75)
    assert hinge.generator.item() == .5
    assert list(nn.grad(hinge.discriminator, [real, fake])[0].values()) == pytest.approx([-1/3, 0, 0])
    assert list(nn.grad(nn.HingeGanObjectives(real, fake).discriminator, [fake])[0].values()) == [0, 0, .25, .25]
    assert list(nn.grad(nn.HingeGanObjectives(real, fake).generator, [fake])[0].values()) == [-.25]*4


@pytest.mark.parametrize('objective', ['Logistic', 'LeastSquares', 'Hinge'])
def test_default_census_unequal_pools_and_owned_five_fields(objective):
    g, d = nn.ParameterStore(4801), nn.ParameterStore(4807)
    x, z = nn.full([2, 784], .3), nn.full([3, 16], -.2)
    out = nn.DenseGan(g, d, x, z, getattr(nn.DenseGanObjective, objective))
    assert [g.count(), d.count(), g.size(), d.size()] == [26416, 25153, 4, 4]
    assert list(nn.MlpGanGenerate(g, z).values()) == list(out.fake.values())
    assert list(nn.MlpGanDiscriminate(d, x).values()) == list(out.real_score.values())
    fields = [getattr(out, name) for name in ['fake','real_score','fake_score','discriminator_loss','generator_loss']]
    del out, g, d
    gc.collect()
    assert tuple(fields[0].shape()) == (3,784)
    assert all(all(math.isfinite(v) for v in a.values()) for a in fields)


@pytest.mark.parametrize('objective', ['Logistic', 'LeastSquares', 'Hinge', 'Conditional'])
def test_continuous_gradients_and_d_then_fresh_g_writes(objective):
    t = nn.DType.Float64
    g, d = nn.ParameterStore(5101, t), nn.ParameterStore(5113, t)
    x, z = nn.full([2,3], .3, t).requires_grad_(), nn.full([2,2], -.2, t).requires_grad_()
    labels = nn.from_vector_int64([0,1],[2])
    def forward():
        if objective == 'Conditional': return nn.ConditionalGan(g,d,x,z,labels,3,2,3)
        return nn.DenseGan(g,d,x,z,getattr(nn.DenseGanObjective,objective),3,3)
    out = forward()
    assert all(any(abs(v)>1e-12 for v in a.values()) for a in nn.grad(out.generator_loss,[z]))
    config = nn.OptimizerConfig();config.kind=nn.OptimizerKind.Adam;config.beta1=.5
    stores = [g,d];opts = [nn.Optimizer(p,.0002,config) for p in stores]
    before_g = [tuple(v.values()) for v in nn.grad(forward().generator_loss,list(g.parameters()))]
    for group in [1,0]:
        before = [tuple(p.flatten()) for p in stores]
        if group==0: assert before_g != [tuple(v.values()) for v in nn.grad(forward().generator_loss,list(g.parameters()))]
        out = forward();loss = out.discriminator_loss if group==1 else out.generator_loss
        opts[group].step(nn.grad(loss,list(stores[group].parameters())))
        assert [(tuple(p.flatten())!=before[k]) for k,p in enumerate(stores)] == [group==0,group==1]
    assert [o.iterations() for o in opts] == [1,1]


def test_both_networks_conditioned_default_census_and_seven_owned_fields():
    g,d=nn.ParameterStore(5101),nn.ParameterStore(5113)
    x,z=nn.full([2,784],.3),nn.full([2,16],-.2)
    labels=nn.from_vector_int64([0,1],[2])
    out=nn.ConditionalGan(g,d,x,z,labels)
    assert [g.count(),d.count()] == [26736,25473]
    wrong=nn.from_vector_int64([1,2],[2])
    assert list(nn.ConditionalGanGenerate(g,z,wrong).values()) != list(out.fake.values())
    assert list(nn.ConditionalGanDiscriminate(d,x,wrong).values()) != list(out.real_logit.values())
    fields=[getattr(out,k) for k in ['fake','real_logit','fake_logit','real_probability','fake_probability','discriminator_loss','generator_loss']]
    del out,g,d
    gc.collect()
    assert all(all(math.isfinite(v) for v in a.values()) for a in fields)


def test_rejects_invalid_options_labels_and_stores_before_creating_leaves():
    g,d=nn.ParameterStore(1),nn.ParameterStore(2)
    x,z=nn.zeros([2,3]),nn.zeros([2,2])
    for bad in [nn.from_vector_int64([-1,0],[2]),nn.from_vector_int64([0,2],[2]),nn.zeros([1],nn.DType.Int32),nn.zeros([2])]:
        with pytest.raises(RuntimeError): nn.ConditionalGan(g,d,x,z,bad,3,2,3)
    with pytest.raises(RuntimeError): nn.DenseGan(g,g,x,z,nn.DenseGanObjective.Logistic,3,3)
    with pytest.raises(RuntimeError): nn.DenseGan(g,d,x,z,999,3,3)
    with pytest.raises(RuntimeError): nn.MlpGanDiscriminate(d,x,3,0)
    with pytest.raises(RuntimeError): nn.LeastSquaresGanObjectives(nn.zeros([0,1]),nn.zeros([2,1]))
    assert g.size()+d.size()==0
