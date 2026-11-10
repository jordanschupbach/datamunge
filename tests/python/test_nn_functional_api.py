"""Functional models are composed in Python using the actual C++ NN API."""
import importlib.util
import math
from pathlib import Path
import runpy
import subprocess

import pytest
from pydatamunge import nn

ROOT = Path(__file__).resolve().parents[2]
NATIVE = {
    'cnn_mnist': '1_convolutional_neural_networks/1.1_fundamentals',
    'convolution_im2col': '1_convolutional_neural_networks/1.1_fundamentals',
    'undercomplete_autoencoder': '2_autoencoders/2.1_deterministic_autoencoders',
}


def load_example(name):
    return runpy.run_path(str(ROOT / 'examples/python/nn' / NATIVE[name] / (name+'.py')))


def test_named_functional_model_gradients_optimizer_and_checkpoint(tmp_path):
    p = nn.ParameterStore(9)
    x = nn.from_vector([.1,.8,.7,.2],[2,2])
    code = nn.dense(p,'encoder',x,1,nn.Activation.Relu)
    prediction = nn.dense(p,'decoder',code,2,nn.Activation.Sigmoid)
    loss = nn.mean(nn.square(nn.sub(prediction,x)))
    gradients = nn.grad(loss,p.parameters())
    assert list(p.names()) == ['encoder/W','encoder/b','decoder/W','decoder/b']
    assert len(gradients) == 4 and p.count() == 7
    assert all(math.isfinite(value) for g in gradients for value in g.values())
    config = nn.OptimizerConfig(); config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p,.003,config)
    before = tuple(p.flatten())
    optimizer.step(gradients)
    assert tuple(p.flatten()) != before
    checkpoint = str(tmp_path/'parameters.npz')
    p.save(checkpoint)
    restored = nn.ParameterStore()
    restored.load(checkpoint,False)
    assert list(restored.names()) == list(p.names())
    assert tuple(restored.flatten()) == tuple(p.flatten())
    archived = dict(nn.load_npz(checkpoint))
    assert set(archived) == set(p.names())
    nn.save_npz(str(tmp_path/'copies.npz'),list(archived.items()))
    assert (tmp_path/'copies.npz').read_bytes() == (tmp_path/'parameters.npz').read_bytes()


def test_canonical_contractive_model_keeps_complete_gradients_and_owned_results():
    import gc
    p = nn.ParameterStore(1753)
    x = nn.full([2, 3], .25)
    result = nn.ContractiveAutoencoder(p, x, .01, 3, 2)
    assert tuple(result.code.shape()) == (2, 2)
    assert tuple(result.reconstruction.shape()) == (2, 3)
    assert p.count() == 17
    assert result.loss.item() == pytest.approx(result.mse.item() + .01 * result.penalty.mean.item(), abs=1e-7)
    gradients = nn.grad(result.loss, p.parameters())
    assert len(gradients) == 4 and all(math.isfinite(v) for g in gradients for v in g.values())
    penalty = result.penalty
    expected = tuple(penalty.per_example.values())
    del result
    gc.collect()
    assert tuple(penalty.per_example.values()) == expected
    mean = penalty.mean
    del penalty
    gc.collect()
    assert math.isfinite(mean.item())
    weights = nn.ones([3, 2]).requires_grad_()
    code = nn.sigmoid(nn.matmul(nn.zeros([4, 3]), weights))
    regularizer = nn.SigmoidEncoderContraction(code, weights)
    assert regularizer.mean.item() == pytest.approx(.375)
    assert list(nn.grad(regularizer.mean, [weights])[0].values()) == pytest.approx([.125] * 6)
    with pytest.raises(RuntimeError):
        nn.ContractiveAutoencoder(nn.ParameterStore(1), x, -1, 3, 2)


def test_canonical_spatial_autoencoder_keeps_decoder_gradients_and_owned_endpoints():
    import gc
    p = nn.ParameterStore(1801)
    image = nn.full([2, 8, 12, 1], .25).requires_grad_()
    result = nn.ConvolutionalAutoencoder(p, image)
    assert p.count() == 741 and p.size() == 8
    assert tuple(result.code.shape()) == (2, 2, 3, 4)
    assert tuple(result.reconstruction.shape()) == (2, 8, 12, 1)
    decoded = nn.ConvolutionalDecoder(p, result.code)
    assert list(decoded.reconstruction.values()) == list(result.reconstruction.values())
    assert list(decoded.hidden.values()) == list(result.decoder1.values())
    gradients = nn.grad(nn.mse(result.reconstruction, image), list(p.parameters()) + [image])
    assert len(gradients) == 9
    assert all(all(math.isfinite(v) for v in g.values()) for g in gradients)
    assert all(any(v != 0 for v in g.values()) for g in gradients)
    code, reconstruction = result.code, decoded.reconstruction
    expected = tuple(reconstruction.values())
    del result, decoded
    gc.collect()
    assert tuple(reconstruction.values()) == expected
    assert tuple(code.shape()) == (2, 2, 3, 4)
    empty = nn.ParameterStore(1)
    with pytest.raises(RuntimeError):
        nn.ConvolutionalAutoencoder(empty, nn.zeros([1, 5, 4, 1]))
    with pytest.raises(RuntimeError):
        nn.ConvolutionalDecoder(empty, nn.zeros([1, 1, 1, 3]))
    assert empty.size() == 0


def test_native_gaussian_banks_and_clean_target_corruption_are_reproducible_and_owned():
    import gc
    bank = nn.GaussianNoise([2, 2], 1, 1673)
    expected = tuple(bank.values())
    nn.GaussianNoise([13], 2, 991)
    assert tuple(nn.GaussianNoise([2, 2], 1, 1673).values()) == expected
    assert tuple(nn.GaussianNoise([2, 2], 1, 1674).values()) != expected
    assert list(nn.GaussianNoise([1], 1, 2147483648).values()) == pytest.approx([-0.9908843253272774], abs=1e-14)
    assert list(nn.GaussianNoise([1], 1, 4294967295).values()) == pytest.approx([1.6481125509743353], abs=1e-14)
    clean = nn.from_vector([0, .2, .8, 1], [2, 2]).requires_grad_()
    noise = nn.from_vector_double([-2, 1, -1, 2], [1, 2, 2]).requires_grad_()
    result = nn.CorruptForDenoising(clean, noise, .3)
    assert list(result.values()) == pytest.approx([0, .5, .5, 1], abs=6e-8)
    assert not result.requires_grad()
    tiny = nn.CorruptForDenoising(nn.zeros([1, 2]),
        nn.from_vector_double([-1e-308, 1e-308], [1, 1, 2]), 1).values()
    assert math.copysign(1, tiny[0]) == -1 and math.copysign(1, tiny[1]) == 1
    assert list(nn.CorruptForDenoising(clean, noise, 0).values()) == list(clean.values())
    del clean, noise
    gc.collect()
    assert list(result.values()) == pytest.approx([0, .5, .5, 1], abs=6e-8)
    with pytest.raises(RuntimeError):
        nn.CorruptForDenoising(nn.zeros([2, 2]), bank, -1)
    with pytest.raises(RuntimeError):
        nn.GaussianNoise([-1], 1, 12)


def test_canonical_competition_preserves_signed_gradients_and_owned_model_paths():
    import gc
    scores = nn.from_vector([-4, -1, -2, -3], [1, 4]).requires_grad_()
    mask = nn.CompetitiveMask(scores, False, 2)
    assert list(mask.values()) == [0, 1, 1, 0]
    assert not mask.requires_grad()
    assert list(nn.grad(nn.sum(nn.mul(scores, mask)), [scores])[0].values()) == [0, 1, 1, 0]
    p = nn.ParameterStore(1931)
    x = nn.from_vector([.1, .2, .8, .7, .3, .4], [2, 3])
    result = nn.CompetitiveAutoencoder(p, x, False, False, 3, 4, 2)
    assert p.count() == 31 and p.size() == 4
    assert sum(result.mask.values()) == 4
    assert list(result.code.values()) == list(result.training_code.values())
    assert list(result.reconstruction.values()) == list(result.training_reconstruction.values())
    gradients = nn.grad(nn.mse(result.training_reconstruction, x), p.parameters())
    assert len(gradients) == 4 and all(math.isfinite(v) for g in gradients for v in g.values())
    reconstruction = result.reconstruction
    expected = tuple(reconstruction.values())
    del result
    gc.collect()
    assert tuple(reconstruction.values()) == expected
    with pytest.raises(RuntimeError):
        nn.CompetitiveMask(nn.full([1, 4], math.nan), False, 2)


def test_contractive_example_uses_the_canonical_model(monkeypatch):
    example = runpy.run_path(str(ROOT / 'examples/python/nn/2_autoencoders/2.1_deterministic_autoencoders/contractive_autoencoder.py'))
    example['fixtures']()
    calls = []
    original = nn.ContractiveAutoencoder

    def forward(*args):
        calls.append(True)
        return original(*args)

    monkeypatch.setattr(nn, 'ContractiveAutoencoder', forward)
    model = example['ContractiveAutoencoderModel'](.01, training=False)
    assert len(calls) == 1
    assert tuple(model.forward(nn.zeros([2, 784])).code.shape()) == (2, 32)
    assert len(calls) == 2


def test_no_grad_nested_and_exception_restoration():
    previous = nn.is_grad_enabled()
    with pytest.raises(ValueError):
        with nn.no_grad():
            assert not nn.is_grad_enabled()
            with nn.no_grad():
                assert not nn.is_grad_enabled()
            assert not nn.is_grad_enabled()
            raise ValueError('fixture')
    assert nn.is_grad_enabled() == previous


def test_batch_order_and_rectangular_convolution_options():
    batches = nn.BatchOrder(7,3,42)
    batches.begin_epoch()
    rows=[]
    while True:
        batch=list(batches.next())
        if not batch: break
        rows.extend(batch)
    assert sorted(rows)==list(range(7))
    x = nn.ones([1,5,6,1])
    options=nn.Conv2DOptions(); options.strides=[2,1]; options.dilations=[1,2]
    y=nn.conv2d(x,nn.ones([2,2,1,3]),options)
    assert tuple(y.shape())==(1,3,6,3)
    with pytest.raises(Exception):
        nn.zeros([-1,3])


def test_native_examples_never_call_the_compiled_experiment_bridge(monkeypatch):
    import pydatamunge.nn_examples as reference
    def reject(*args,**kwargs):
        raise AssertionError('language port called a compiled experiment or subprocess')
    monkeypatch.setattr(reference,'run',reject)
    monkeypatch.setattr(subprocess,'run',reject)
    monkeypatch.setattr(subprocess,'Popen',reject)
    assert load_example('convolution_im2col')['main']([])==0
    load_example('undercomplete_autoencoder')['checks']()
    p=nn.ParameterStore(17)
    logits=load_example('cnn_mnist')['logits'](p,nn.zeros([2,784]))
    assert tuple(logits.shape())==(2,10)
    assert p.count()==105866
    with pytest.raises(Exception):
        nn.grad(nn.sum(logits),[nn.zeros([2,784])])


def test_fluent_tensor_return_owns_the_temporary_and_preserves_aliasing():
    import gc
    leaf = nn.zeros([3]).requires_grad_()
    gc.collect()
    assert leaf.defined() and leaf.requires_grad()
    assert tuple(nn.grad(nn.sum(nn.square(leaf)), [leaf])[0].values()) == (0., 0., 0.)
    original = nn.zeros([3])
    alias = original.fill_(4)
    assert alias.is_same(original)
    del original
    gc.collect()
    assert tuple(alias.values()) == (4., 4., 4.)


def test_confusion_matrix_accumulates_all_batches():
    confusion = nn.ConfusionMatrix(3)
    confusion.add(nn.from_vector_int64([0, 1, 2]), nn.from_vector_int64([0, 2, 2]))
    confusion.add(1, 1)
    assert tuple(confusion.to_tensor().values()) == (1., 0., 0., 0., 1., 1., 0., 0., 1.)
    assert confusion.accuracy() == .75


def test_clipping_pair_retains_tensor_handles_after_collection():
    import gc
    gradients, norm = nn.clip_by_global_norm([nn.from_vector([3., 4.])], 2.)
    gc.collect()
    assert norm == 5.
    assert tuple(gradients[0].values()) == pytest.approx((1.2, 1.6))


def test_schedule_namespace_uses_completed_updates_and_rejects_invalid_duration():
    assert nn.schedule.step_decay(.1, .5, 10, 9) == .1
    assert nn.schedule.step_decay(.1, .5, 10, 10) == .05
    assert nn.schedule.warmup(.1, 10, 0) == 0.
    with pytest.raises(RuntimeError):
        nn.schedule.cosine(.1, 0., 0, 1)


def test_monitor_decision_is_an_owned_value_and_weighted_mean_resets():
    import gc
    monitor = nn.EarlyStopping(2, .1)
    decision = monitor.observe(1., 0)
    del monitor
    gc.collect()
    assert decision.improved and not decision.stop
    metric = nn.WeightedMean()
    metric.add(2., 4)
    metric.add(10., 1)
    assert metric.count() == 5 and metric.value() == 3.6
    metric.reset()
    assert metric.count() == 0


@pytest.mark.parametrize('engine', ['mt19937', 'mt19937_64'])
def test_mt19937_binding_matches_standard_cpp_stream(tmp_path, engine):
    import shutil
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    source = tmp_path / 'rng_oracle.cpp'
    source.write_text('''#include <random>
#include <iostream>
#include <iomanip>
int main() {
  std::mt19937 engine(113);
  std::normal_distribution<float> n32(0.f, .3f);
  std::normal_distribution<double> n64(0., .5);
  std::uniform_real_distribution<float> u32(-1.f, 1.f);
  std::uniform_real_distribution<double> u64(-2., 2.);
  std::cout << std::setprecision(17);
  for (int i=0; i<20; ++i)
    std::cout << double(n32(engine)) << ' ' << n64(engine) << ' '
              << double(u32(engine)) << ' ' << u64(engine) << '\\n';
}
'''.replace('std::mt19937 engine(113)', 'std::' + engine + ' engine(113)'))
    executable = tmp_path / 'rng_oracle'
    subprocess.run([compiler, '-std=c++17', str(source), '-o', str(executable)], check=True)
    expected = [float(value) for value in subprocess.check_output([str(executable)], text=True).split()]
    generator = getattr(nn, 'Mt19937_64' if engine.endswith('_64') else 'Mt19937')(113)
    actual = []
    for _ in range(20):
        actual += [generator.normal32(0., .3), generator.normal64(0., .5),
                   generator.uniform32(-1., 1.), generator.uniform64(-2., 2.)]
    assert actual == expected


@pytest.mark.parametrize('engine', ['mt19937', 'mt19937_64'])
def test_standard_shuffle_and_modulo_stream_match_cpp(tmp_path, engine):
    import shutil
    compiler = shutil.which('c++')
    if compiler is None:
        pytest.skip('C++ compiler unavailable')
    source = tmp_path / 'shuffle.cpp'
    source.write_text('''#include <algorithm>
#include <random>
#include <numeric>
#include <vector>
#include <iostream>
int main() {
  std::ENGINE rng(7);
  std::vector<int> rows(64);
  std::iota(rows.begin(), rows.end(), 0);
  for (int epoch=0; epoch<5; ++epoch) {
    std::shuffle(rows.begin(), rows.end(), rng);
    for (int row : rows) std::cout << row << ' ';
    std::cout << rng() % 11 << '\\n';
  }
}
'''.replace('ENGINE', engine))
    executable = tmp_path / 'shuffle'
    subprocess.run([compiler, '-std=c++17', str(source), '-o', str(executable)], check=True)
    expected = [int(v) for v in subprocess.check_output([str(executable)], text=True).split()]
    rng = getattr(nn, 'Mt19937_64' if engine.endswith('_64') else 'Mt19937')(7)
    rows, actual = list(range(64)), []
    for _ in range(5):
        rows = list(rng.shuffle(rows))
        actual += rows + [rng.modulo(11)]
    assert actual == expected


@pytest.mark.parametrize('engine', [nn.Mt19937, nn.Mt19937_64])
@pytest.mark.parametrize('precision', [32, 64])
def test_bulk_normals_preserve_scalar_stream_and_odd_count_cache(engine, precision):
    bulk, scalar = engine(113), engine(113)
    draws = getattr(bulk, f'normal{precision}_values')
    one = getattr(scalar, f'normal{precision}')
    for count, mean, stddev in [(3, 1.25, .75), (0, 2., 3.), (5, -.2, .4), (2, 0., 1.)]:
        assert list(draws(count, mean, stddev)) == [one(mean, stddev) for _ in range(count)]
        assert bulk.uniform64() == scalar.uniform64()
    with pytest.raises(RuntimeError, match='nonnegative'):
        draws(-1)
    with pytest.raises(RuntimeError, match='positive'):
        draws(2, 0., 0.)


@pytest.mark.parametrize('implementation', [nn.AttentionImpl.Plain, nn.AttentionImpl.Fused])
def test_attention_keep_mask_excludes_all_keys_and_gradients(implementation):
    q = nn.ones([1, 2, 3], nn.DType.Float64).requires_grad_()
    k = nn.ones([1, 4, 3], nn.DType.Float64).requires_grad_()
    v = nn.ones([1, 4, 2], nn.DType.Float64).requires_grad_()
    options = nn.AttentionOptions()
    options.mask = nn.zeros([2, 4], nn.DType.Bool)
    options.impl = implementation
    attended = nn.scaled_dot_product_attention(q, k, v, options)
    assert tuple(attended.output.values()) == (0.,) * 4
    for gradient in nn.grad(nn.sum(attended.output), [q, k, v]):
        assert all(value == 0 for value in gradient.values())


def test_kv_cache_pairs_keep_tensor_storage_alive_after_reset():
    import gc
    cache = nn.KvCache()
    with nn.no_grad():
        keys, values = cache.append(nn.ones([1, 1, 2, 3]), nn.full([1, 1, 2, 3], 7))
        cache.append(nn.zeros([1, 1, 1, 3]), nn.zeros([1, 1, 1, 3]))
    assert cache.length() == 3 and tuple(cache.keys().shape()) == (1, 1, 3, 3)
    cache.reset()
    del cache
    gc.collect()
    assert tuple(keys.values()) == (1.,) * 6 and tuple(values.values()) == (7.,) * 6


def test_attention_rejects_zero_key_width_before_scaling():
    with pytest.raises(RuntimeError, match='key width'):
        nn.scaled_dot_product_attention(nn.zeros([1, 1, 0]), nn.zeros([1, 1, 0]),
                                        nn.ones([1, 1, 1]), nn.Tensor())
