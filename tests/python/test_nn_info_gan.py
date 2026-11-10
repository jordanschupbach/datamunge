"""Native generated codes, separate image-only Q, owning fields and explicit D-Q-G writes."""
import gc
import math
import pytest
from pydatamunge import nn

FIELDS = ['fake','real_logit','fake_logit','real_probability','fake_probability',
          'discriminator_loss','generator_loss','source_generator_loss','recognition_loss',
          'categorical_loss','continuous_loss','mi_lower_bound','q_logits','q_probability','q_mean']


def test_prior_words_rejection_final_cast_owned_fields_and_guards():
    a = nn.InfoPrior(2, 1, 1073741825, 2)
    assert list(a.raw.values()) == [1791095845.,4282876139.,3093770124.,491263.,550290313.,1298508491.]
    assert list(a.category.values()) == [717354020,491263]
    first = nn.InfoPrior(4, 5382)
    assert list(first.category.values()) == [0,9,1,1]
    wide = nn.InfoPrior(4, 5382, 10, 2, nn.DType.Float64)
    assert list(first.continuous.values()) == list(nn.cast(wide.continuous,nn.DType.Float32).values())
    fields = [first.category, first.continuous, first.raw]
    del first
    gc.collect()
    assert [tuple(v.shape()) for v in fields] == [(4,), (4,2), (4,3)]
    assert not any(v.requires_grad() for v in fields)
    for args in [(0,1),(2,1,0),(2,1,2147483648),(2,1,10,0),(2,1,10,2,nn.DType.Int32),(2**63-1,1)]:
        with pytest.raises(RuntimeError): nn.InfoPrior(*args)


def test_recognition_pool_algebra_gradients_and_unused_probability():
    q = nn.InfoRecognition()
    q.logits = nn.zeros([2,2],nn.DType.Float64).requires_grad_()
    q.mean = nn.zeros([2,1],nn.DType.Float64).requires_grad_()
    category = nn.from_vector_int64([0,1],[2])
    target = nn.from_vector_double([-1,1],[2,1]).requires_grad_()
    loss = nn.InfoGanRecognitionLoss(q,category,target)
    assert loss.categorical.item() == pytest.approx(math.log(2))
    assert loss.continuous.item() == pytest.approx((1+math.log(2*math.pi))/2)
    assert loss.lower_bound.item() == pytest.approx(2*math.log(2)-loss.total.item())
    assert loss.lower_bound.item() < 0
    grads = nn.grad(loss.total,[q.logits,q.mean,target])
    assert list(grads[0].values()) == [-.25,.25,.25,-.25]
    assert list(grads[1].values()) == [.5,-.5]
    assert list(grads[2].values()) == [-.5,.5]
    q.probability = nn.full([3], math.nan)
    assert nn.InfoGanRecognitionLoss(q,category,target).total.item() == loss.total.item()


def test_default_census_unequal_pools_and_image_only_standalone_q():
    g,d,q = [nn.ParameterStore(seed) for seed in [5303,5309,5317]]
    codes = nn.InfoPrior(3,5351)
    image,noise = nn.full([2,784],.3),nn.full([3,16],-.2)
    out = nn.InfoGan(g,d,q,image,noise,codes.category,codes.continuous)
    assert [s.count() for s in [g,d,q]] == [26800,25153,25516]
    assert [s.size() for s in [g,d,q]] == [4,4,6]
    assert tuple(out.fake.shape()) == (3,784) and tuple(out.real_logit.shape()) == (2,1)
    assert list(nn.InfoGanGenerate(g,noise,codes.category,codes.continuous).values()) == list(out.fake.values())
    assert list(nn.MlpGanDiscriminate(d,image).values()) == list(out.real_logit.values())
    pred = nn.InfoGanRecognize(q,out.fake)
    assert list(pred.probability.values()) == list(out.q_probability.values())
    assert list(pred.mean.values()) == list(out.q_mean.values())
    q.set('continuous/W',nn.zeros([32,2]));q.set('continuous/b',nn.full([2],4.))
    assert list(nn.InfoGanRecognize(q,image).mean.values()) == [4.]*4


def test_fifteen_fields_survive_collection_with_live_noise_and_continuous_paths():
    t = nn.DType.Float64
    g,d,q = [nn.ParameterStore(seed,t) for seed in [5303,5309,5317]]
    x = nn.full([2,3],.3,t)
    z,u = nn.full([3,2],-.2,t).requires_grad_(),nn.full([3,1],.1,t).requires_grad_()
    c = nn.from_vector_int64([0,1,0],[3])
    out = nn.InfoGan(g,d,q,x,z,c,u,3,2,3)
    fields = {name:getattr(out,name) for name in FIELDS}
    del out,g,d,q
    gc.collect()
    assert len(fields) == 15 and all(all(math.isfinite(v) for v in a.values()) for a in fields.values())
    gradients = nn.grad(fields['generator_loss'],[z,u])
    assert all(any(abs(v)>1e-12 for v in a.values()) for a in gradients)


def test_three_host_owned_adam_phases_and_q_partial_independent_of_d():
    t = nn.DType.Float64
    stores = [nn.ParameterStore(seed,t) for seed in [5303,5309,5317]]
    x,z,u,c = nn.full([2,3],.3,t),nn.full([2,2],-.2,t),nn.full([2,1],.1,t),nn.from_vector_int64([0,1],[2])
    def forward(): return nn.InfoGan(*stores,x,z,c,u,3,2,3)
    forward()
    config = nn.OptimizerConfig();config.kind=nn.OptimizerKind.Adam;config.beta1=.5
    opts = [nn.Optimizer(p,.0002,config) for p in stores]
    q_before = [tuple(a.values()) for a in nn.grad(forward().recognition_loss,list(stores[2].parameters()))]
    g_before = [tuple(a.values()) for a in nn.grad(forward().generator_loss,list(stores[0].parameters()))]
    for group in [1,2,0]:
        before = [tuple(p.flatten()) for p in stores]
        if group==2: assert q_before == [tuple(a.values()) for a in nn.grad(forward().recognition_loss,list(stores[2].parameters()))]
        if group==0: assert g_before != [tuple(a.values()) for a in nn.grad(forward().generator_loss,list(stores[0].parameters()))]
        out=forward();loss=out.generator_loss if group==0 else out.discriminator_loss if group==1 else out.recognition_loss
        opts[group].step(nn.grad(loss,list(stores[group].parameters())))
        assert [(tuple(p.flatten())!=before[k]) for k,p in enumerate(stores)] == [k==group for k in range(3)]
    assert [o.iterations() for o in opts] == [1,1,1]


def test_invalid_options_stores_codes_and_computed_square_overflow():
    t = nn.DType.Float64
    g,d,q = [nn.ParameterStore(seed,t) for seed in [1,2,3]]
    x,z,u,c = nn.zeros([2,3],t),nn.zeros([3,2],t),nn.zeros([3,1],t),nn.zeros([3],nn.DType.Int32)
    for bad in [nn.zeros([2,1],t),nn.full([3,1],2.,t),nn.full([3,1],math.nan,t),nn.zeros([3,1])]:
        with pytest.raises(RuntimeError): nn.InfoGan(g,d,q,x,z,c,bad,3,2,3)
    for bad in [nn.from_vector_int64([-1,0,1],[3]),nn.full([3],2,nn.DType.Int32),nn.zeros([3])]:
        with pytest.raises(RuntimeError): nn.InfoGan(g,d,q,x,z,bad,u,3,2,3)
    for a,b,h in [(g,g,q),(g,d,g),(g,d,d)]:
        with pytest.raises(RuntimeError): nn.InfoGan(a,b,h,x,z,c,u,3,2,3)
    with pytest.raises(RuntimeError): nn.InfoGan(g,d,q,x,z,c,u,3,0,3)
    with pytest.raises(RuntimeError): nn.InfoGanRecognize(q,x,3,2,2**63-1,3)
    assert g.size()+d.size()+q.size() == 0
    prediction=nn.InfoRecognition();prediction.logits=nn.zeros([2,2]);prediction.mean=nn.full([2,1],1e30)
    with pytest.raises(RuntimeError): nn.InfoGanRecognitionLoss(prediction,nn.zeros([2],nn.DType.Int32),nn.zeros([2,1]))
