"""Native convolutional Gaussian VAE, owning spatial stages and explicit host optimization."""
import gc
import math
import pytest
from pydatamunge import nn


def test_spatial_result_and_decoder_own_inherited_endpoints_after_collection():
    p = nn.ParameterStore(2011)
    x, epsilon = nn.full([2, 4, 4, 1], .3), nn.full([2, 1], .7)
    out = nn.ConvolutionalVariationalAutoencoder(p, x, epsilon, 1, 2, 1)
    assert p.size() == 14 and p.count() == 84
    assert tuple(out.pool2.shape()) == (2, 1, 1, 1)
    assert tuple(out.decoder1.shape()) == (2, 2, 2, 2)
    assert tuple(out.spatial_logits.shape()) == (2, 4, 4, 1)
    probability, mu, z, loss, base = out.probability, out.mu, out.z, out.loss, out.decoder_base
    expected = tuple(probability.values())
    assert loss.item() == out.unweighted_loss.item()
    del out
    gc.collect()
    decoded = nn.ConvolutionalVaeDecode(p, z, 4, 4, 1, 2, 1)
    assert tuple(decoded.probability.values()) == expected
    assert tuple(decoded.decoder_base.values()) == tuple(base.values())
    retained = decoded.decoder1
    del decoded
    gc.collect()
    assert tuple(retained.shape()) == (2, 2, 2, 2)
    zero = nn.ConvolutionalVariationalAutoencoder(p, x, nn.zeros([2, 1]), 1, 2, 1)
    assert tuple(zero.mu.values()) == tuple(mu.values())
    assert tuple(zero.z.values()) == tuple(zero.mu.values())
    assert tuple(zero.z.values()) != tuple(z.values())
    gradients = nn.grad(loss, list(p.parameters()))
    assert len(gradients) == 14 and all(all(math.isfinite(v) for v in g.values()) for g in gradients)
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, .003, config)
    before = tuple(p.flatten())
    optimizer.step(gradients)
    assert optimizer.iterations() == 1 and tuple(p.flatten()) != before


def test_default_spatial_census_and_decoder_has_no_encoder_leaves():
    p = nn.ParameterStore(2011)
    out = nn.ConvolutionalVariationalAutoencoder(p, nn.zeros([1, 28, 28, 1]), nn.zeros([1, 2]))
    assert p.count() == 2117 and p.size() == 14
    assert tuple(out.probability.shape()) == (1, 784)
    decoder = nn.ParameterStore(7)
    alone = nn.ConvolutionalVaeDecode(decoder, nn.zeros([1, 2]))
    assert decoder.count() == 957 and decoder.size() == 6
    assert tuple(alone.spatial_logits.shape()) == (1, 28, 28, 1)
    assert all(not key.startswith('encoder') for key in decoder.names())


def test_invalid_images_noise_and_decoder_geometry_fail_before_creation():
    p = nn.ParameterStore(1)
    for shape in [[0, 4, 4, 1], [1, 5, 4, 1], [1, 4, 6, 1], [1, 4, 4, 2], [4, 4, 1]]:
        with pytest.raises(RuntimeError):
            nn.ConvolutionalVariationalAutoencoder(p, nn.zeros(shape), nn.zeros([1, 2]))
    with pytest.raises(RuntimeError):
        nn.ConvolutionalVariationalAutoencoder(p, nn.full([1, 4, 4, 1], 1.1), nn.zeros([1, 2]))
    with pytest.raises(RuntimeError):
        nn.ConvolutionalVariationalAutoencoder(p, nn.zeros([1, 4, 4, 1]), nn.zeros([2, 2]))
    with pytest.raises(RuntimeError):
        nn.ConvolutionalVaeDecode(p, nn.zeros([1, 2]), 6, 4)
    assert p.size() == 0


def test_float64_non_square_image_geometry_and_target_free_decode():
    dtype = nn.DType.Float64
    p = nn.ParameterStore(2011, dtype)
    out = nn.ConvolutionalVariationalAutoencoder(p, nn.full([2, 4, 8, 1], .3, dtype), nn.ones([2, 1], dtype), 1, 2, 1)
    decoded = nn.ConvolutionalVaeDecode(p, out.z, 4, 8, 1, 2, 1)
    assert tuple(decoded.probability.values()) == tuple(out.probability.values())
    assert tuple(decoded.spatial_logits.shape()) == (2, 4, 8, 1)
    assert decoded.probability.dtype() == dtype
