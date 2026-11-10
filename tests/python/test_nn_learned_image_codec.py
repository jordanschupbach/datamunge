"""Native byte quantizer, exact TFLCODE1 packets and packet-only decoder ownership."""
import gc
import struct
import pytest
from pydatamunge import nn


def _packet():
    q = nn.CodecQuantizer()
    q.minimum, q.maximum = [0.0]*32, [1.0]*32
    p = nn.CodecPacket()
    p.quantizer, p.weights, p.bias = q, nn.zeros([32, 784]), nn.zeros([784])
    p.source_ids, p.symbols = [7, 2147483647], list(range(64))
    return p


def test_range_ownership_signed_strides_constants_and_exact_half_up():
    fit = nn.transpose(nn.from_vector([-2, 1, 4, .25, .25, .25], [2, 3]), [1, 0]).requires_grad_()
    q = nn.FitCodecQuantizer(fit)
    assert tuple(q.minimum) == (-2, .25) and tuple(q.maximum) == (4, .25)
    with nn.no_grad():
        fit.fill_(9)
    assert tuple(q.minimum) == (-2, .25)
    q.minimum, q.maximum = [0, .25], [255, .25]
    symbols = nn.QuantizeCodec(nn.from_vector([.5, -9, 1.5, .25, 2.5, 8], [3, 2]), q)
    assert tuple(symbols) == (1, 0, 2, 0, 3, 0)
    values = nn.DequantizeCodec(symbols, q)
    assert tuple(values.values()) == (1, .25, 2, .25, 3, .25)
    assert not values.requires_grad() and values.dtype() == nn.DType.Float32
    assert tuple(nn.SymbolTensor(symbols, 2).values()) == (1, 0, 2, 0, 3, 0)


def test_continuous_decoder_keeps_each_supplied_gradient_path():
    code = nn.from_vector_double([-.4, .8], [1, 2]).requires_grad_()
    weight = nn.from_vector_double([.3, -.2, .7, .4], [2, 2]).requires_grad_()
    bias = nn.from_vector_double([.1, -.3], [2]).requires_grad_()
    prediction = nn.LearnedCodecDecode(code, weight, bias)
    assert prediction.dtype() == nn.DType.Float64
    grads = nn.grad(nn.sum(prediction), [code, weight, bias])
    assert all(nn.sum(nn.abs(g)).item() > 0 for g in grads)
    del code, weight, bias
    gc.collect()
    assert tuple(prediction.shape()) == (1, 2)


def test_packet_bytes_offsets_round_trip_and_owned_nested_quantizer():
    p = _packet()
    data = bytes(nn.PackCodec(p))
    assert len(data) == 103844 and data[:8] == b'TFLCODE1'
    assert struct.unpack_from('<5I', data, 8) == (2, 32, 28, 28, 1)
    assert struct.unpack_from('<2f', data, 28) == (0, 1)
    assert struct.unpack_from('<2I', data, 103772) == (7, 2147483647)
    assert data[103780:] == bytes(range(64))
    parsed = nn.UnpackCodec(data)
    assert bytes(nn.PackCodec(parsed)) == data
    kept_ids, kept_symbols = parsed.source_ids, parsed.symbols
    kept = parsed.quantizer
    kept.minimum = [-1]*32
    assert tuple(parsed.quantizer.minimum) == (0,)*32
    del parsed, p
    gc.collect()
    assert tuple(kept.minimum) == (-1,)*32
    assert kept_ids == (7, 2147483647) and kept_symbols == bytes(range(64))


def test_fresh_decoder_uses_only_packet_and_stops_trainable_handles():
    p = _packet()
    p.weights.requires_grad_(); p.bias.requires_grad_()
    result = nn.DecodeCodec(p)
    assert not result.requires_grad() and result.is_contiguous()
    assert p.weights.requires_grad() and p.bias.requires_grad()
    assert tuple(result.shape()) == (2, 784) and tuple(result.values()) == (.5,)*1568
    fresh = nn.UnpackCodec(bytes(nn.PackCodec(p)))
    kept_weight = fresh.weights
    del p, fresh
    gc.collect()
    assert tuple(kept_weight.shape()) == (32, 784)
    assert tuple(result.values()) == (.5,)*1568


def test_malformed_header_fields_ranges_symbols_and_decoder_are_rejected():
    p = _packet(); data = bytes(nn.PackCodec(p))
    with pytest.raises(TypeError):
        p.source_ids = 2
    with pytest.raises(TypeError):
        nn.DequantizeCodec(32, p.quantizer)
    for bad in [b'', data[:27], data[:-1], data+b'\0', b'X'+data[1:], data[:8]+struct.pack('<I', 10001)+data[12:], data[:28]+struct.pack('<f', float('nan'))+data[32:]]:
        with pytest.raises(RuntimeError):
            nn.UnpackCodec(bad)
    p.source_ids = [-1, 2]
    with pytest.raises(RuntimeError):
        nn.PackCodec(p)
    p = _packet(); p.symbols = [0]
    with pytest.raises(RuntimeError):
        nn.DecodeCodec(p)
    q = nn.CodecQuantizer(); q.minimum, q.maximum = [2], [1]
    with pytest.raises(RuntimeError):
        nn.QuantizeCodec(nn.zeros([1, 1]), q)
    with pytest.raises(RuntimeError):
        nn.SymbolTensor([0], 0)
    with pytest.raises(RuntimeError):
        nn.FitCodecQuantizer(nn.ones([2, 2], nn.DType.Float64))
    with pytest.raises(RuntimeError):
        nn.LearnedCodecDecode(nn.ones([1, 2]), nn.ones([3, 1]), nn.zeros([1]))
