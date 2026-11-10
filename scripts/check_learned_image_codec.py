#!/usr/bin/env python3
"""Independent NumPy/stdlib oracle for native TFLCODE1 learned-codec outputs.

Reads saved arrays, the self-contained bitstream and actual PNG files. Never
prepares inputs or trains. Checks quantizer ties/constants/clipping, every
packet byte, packet-only decoding, real lossless PNG pixels and full rate costs.
"""
import argparse
import binascii
import csv
import json
import math
from pathlib import Path
import struct
import zlib
import numpy as np
from check_adversarial_autoencoder import arrays, exact
from check_contractive_autoencoder import sigmoid


def quantize(code, low, high):
    """Source half-up rounding after clamped Double normalization; constants encode zero."""
    out = np.zeros(code.shape, dtype=np.int32)
    for j in range(32):
        if high[j] > low[j]:
            value = (code[:, j].astype(np.float64)-float(low[j]))/(float(high[j])-float(low[j]))*255
            out[:, j] = np.floor(np.clip(value, 0, 255)+.5).astype(np.int32)
    return out


def dequantize(symbols, low, high):
    """One final Float32 cast after Double multiply/divide/add in source grouping."""
    return (low.astype(np.float64)+symbols.astype(np.float64)*(high.astype(np.float64)-low)/255).astype(np.float32)


def packet(data):
    """Strict independent TFLCODE1 little-endian parser, with no model/input dependency."""
    if len(data) < 28 or data[:8] != b'TFLCODE1':
        raise ValueError('invalid codec magic or header')
    count, width, height, columns, channels = struct.unpack_from('<5I', data, 8)
    if not 0 < count <= 10000 or (width, height, columns, channels) != (32, 28, 28, 1):
        raise ValueError('invalid fixed architecture or count')
    if len(data) != 28+256+103488+count*36:
        raise ValueError('invalid exact bitstream length')
    bounds = np.frombuffer(data, dtype='<f4', count=64, offset=28).reshape(32, 2).copy()
    weights = np.frombuffer(data, dtype='<f4', count=32*784, offset=284).reshape(32, 784).copy()
    bias = np.frombuffer(data, dtype='<f4', count=784, offset=284+32*784*4).copy()
    if not all(np.isfinite(v).all() for v in [bounds, weights, bias]) or (bounds[:, 0] > bounds[:, 1]).any():
        raise ValueError('nonfinite parameters or reversed quantizer bounds')
    ids = np.frombuffer(data, dtype='<u4', count=count, offset=103772).copy()
    if (ids > np.iinfo(np.int32).max).any():
        raise ValueError('invalid source identity')
    symbols = np.frombuffer(data, dtype=np.uint8, offset=103772+count*4).reshape(count, 32).copy()
    return bounds[:, 0].copy(), bounds[:, 1].copy(), weights, bias, ids.astype(np.int32), symbols.astype(np.int32)


def encode_packet(low, high, weights, bias, ids, symbols):
    """Independent serializer reproducing every original parameter/identity/payload byte."""
    bounds = np.stack([low, high], axis=1).astype('<f4')
    return (b'TFLCODE1'+struct.pack('<5I', len(ids), 32, 28, 28, 1)+bounds.tobytes()+
            weights.astype('<f4').tobytes()+bias.astype('<f4').tobytes()+
            ids.astype('<u4').tobytes()+symbols.astype(np.uint8).tobytes())


def png_pixels(path):
    """Validate CRCs and decode noninterlaced grayscale8 PNG with all five row filters."""
    data = path.read_bytes()
    assert data[:8] == b'\x89PNG\r\n\x1a\n'
    position = 8;compressed = bytearray();shape = None;finished = False
    while position < len(data):
        length = struct.unpack_from('>I', data, position)[0];kind = data[position+4:position+8]
        payload = data[position+8:position+8+length]
        assert len(payload) == length
        crc = struct.unpack_from('>I', data, position+8+length)[0]
        assert binascii.crc32(kind+payload) & 0xffffffff == crc
        position += 12+length
        if kind == b'IHDR':
            width, height, depth, color, compression, filtering, interlace = struct.unpack('>2I5B', payload)
            assert (depth, color, compression, filtering, interlace) == (8, 0, 0, 0, 0)
            shape = (height, width)
        elif kind == b'IDAT':
            compressed.extend(payload)
        elif kind == b'IEND':
            assert not payload and position == len(data);finished = True;break
    assert finished and shape == (28, 28)
    raw = zlib.decompress(compressed);height, width = shape
    assert len(raw) == height*(width+1)
    image = np.zeros(shape, dtype=np.uint8)
    for row in range(height):
        mode = raw[row*(width+1)];assert mode <= 4
        for col in range(width):
            left = int(image[row, col-1]) if col else 0
            above = int(image[row-1, col]) if row else 0
            corner = int(image[row-1, col-1]) if row and col else 0
            if mode == 0:predictor = 0
            elif mode == 1:predictor = left
            elif mode == 2:predictor = above
            elif mode == 3:predictor = (left+above)//2
            else:
                p = left+above-corner;da, db, dc = abs(p-left), abs(p-above), abs(p-corner)
                predictor = left if da <= db and da <= dc else above if db <= dc else corner
            image[row, col] = (raw[row*(width+1)+1+col]+predictor) & 255
    return image


def check(directory, upstream=None):
    """Verify source/report arrays, actual bitstream and lossless PNG comparison."""
    a, p = arrays(directory/'analysis.npz'), arrays(directory/'model_arrays.npz')
    algebra = arrays(directory/'quantizer_algebra.npz')
    if upstream:
        original = arrays(upstream)
        for key in p:
            exact(p[key], original[key])
    assert sum(v.size for v in p.values()) == 50992
    largest = 0.;forward_values = 0;identities = set()
    for split, count in [('fit', 1000), ('test', 200)]:
        ids, labels = a[split+'_source_ids'], a[split+'_labels']
        assert ids.shape == (count,) and len(set(ids)) == count and not identities.intersection(ids)
        identities.update(ids)
        lo, hi = (60000, 70000) if split == 'test' else (0, 60000)
        assert ((ids >= lo) & (ids < hi)).all()
        assert np.array_equal(np.bincount(labels, minlength=10), np.full(10, count//10))
        x = a[split+'_images'].reshape(count, 784).astype(np.float64)
        code = np.maximum(x@p['encoder/W'].astype(np.float64)+p['encoder/b'], 0)
        np.testing.assert_allclose(code, a[split+'_code'], atol=5e-6, rtol=5e-6)
        largest = max(largest, float(abs(code-a[split+'_code']).max()));forward_values += code.size
    for sample in [a, algebra]:
        fit = sample['fit_code'] if 'fit_code' in sample else sample['fit']
        low, high = fit.min(axis=0), fit.max(axis=0)
        exact(low, sample['minimum']);exact(high, sample['maximum'])
        code = sample['test_code'] if 'test_code' in sample else sample['probe']
        symbols = quantize(code, low, high);exact(symbols, sample['symbols'])
        exact(dequantize(symbols, low, high), sample['dequantized'])
    exact(algebra['symbols'][:, 0], np.array([0, 128, 255], dtype=np.int32))
    assert not algebra['symbols'][:, 1:].any()
    data = (directory/'test.tflcodec').read_bytes()
    low, high, weights, bias, ids, symbols = packet(data)
    exact(low, a['minimum']);exact(high, a['maximum']);exact(weights, p['decoder/W']);exact(bias, p['decoder/b'])
    exact(ids, a['test_source_ids']);exact(symbols, a['symbols'])
    assert encode_packet(low, high, weights, bias, ids, symbols) == data
    decoded_code = dequantize(symbols, low, high);exact(decoded_code, a['dequantized'])
    decoded = sigmoid(decoded_code.astype(np.float64)@weights.astype(np.float64)+bias)
    np.testing.assert_allclose(decoded, a['decoded_reconstruction'], atol=1e-6, rtol=3e-6)
    largest = max(largest, float(abs(decoded-a['decoded_reconstruction']).max()));forward_values += decoded.size
    ordinary = sigmoid(a['test_code'].astype(np.float64)@weights.astype(np.float64)+bias)
    np.testing.assert_allclose(ordinary, a['unquantized_reconstruction'], atol=1e-6, rtol=3e-6)
    largest = max(largest, float(abs(ordinary-a['unquantized_reconstruction']).max()));forward_values += ordinary.size
    broken = [b'X'+data[1:], data[:-1], data[:8]+b'\xff'*4+data[12:]]
    for value in broken:
        try:packet(value)
        except ValueError:pass
        else:raise AssertionError('malformed codec accepted')
    with (directory/'png_sizes.tsv').open() as stream:rows = list(csv.DictReader(stream, delimiter='\t'))
    assert len(rows) == 200
    png_bytes = 0
    for i, row in enumerate(rows):
        assert int(row['source_id']) == ids[i]
        file = directory/'original_png'/(str(ids[i])+'.png');size = file.stat().st_size
        assert int(row['bytes']) == size;png_bytes += size
        expected = np.floor(a['test_images'][i, :, :, 0].astype(np.float64)*255+.5).astype(np.uint8)
        exact(png_pixels(file), expected)
    metrics = json.loads((directory/'metrics.json').read_text())
    for key, value in dict(payload_bytes=6400, decoder_parameter_bytes=103488, quantizer_bytes=256,
                           source_identity_bytes=800, header_bytes=28, total_file_bytes=len(data),
                           original_raw_u8_bytes=156800, original_png_collection_bytes=png_bytes).items():
        assert metrics[key] == value
    assert len(data) == 6400+103488+256+800+28
    for key, size in [('payload_bits_per_pixel', 6400), ('total_bits_per_pixel', len(data)), ('png_bits_per_pixel', png_bytes)]:
        assert metrics[key] == size*8/156800
    target = a['test_images'].reshape(200, 784).astype(np.float64)
    mse = float(np.mean((a['decoded_reconstruction'].astype(np.float64)-target)**2))
    np.testing.assert_allclose(mse, metrics['decoded_test_mse'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.mean((a['unquantized_reconstruction'].astype(np.float64)-target)**2), metrics['unquantized_test_mse'], atol=1e-12, rtol=0)
    np.testing.assert_allclose(10*math.log10(1/mse), metrics['decoded_global_psnr_db'], atol=1e-11, rtol=0)
    assert metrics['constant_channels'] == int((low == high).sum()) == 3
    assert metrics['clipped_test_code_entries'] == int(((a['test_code'] < low) | (a['test_code'] > high)).sum()) == 9
    assert metrics['additional_training_updates'] == 0 and metrics['fresh_decoder_max_error'] == 0 and metrics['malformed_controls_rejected'] == 3
    return dict(passed=True, additional_training_updates=0, forward_values_checked=forward_values,
                maximum_forward_error=largest, exact_quantized_symbols=6496, exact_dequantized_values=6496,
                exact_bitstream_bytes=len(data), malformed_controls_rejected=3, lossless_PNG_files_checked=200,
                lossless_PNG_pixels_checked=156800, actual_PNG_collection_bytes=png_bytes,
                full_bitstream_larger_than_PNG=len(data)>png_bytes, upstream_parameters_exact=bool(upstream), metrics=metrics)


def main():
    """Read complete native outputs and optionally save an independent oracle receipt."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--upstream', type=Path)
    parser.add_argument('--report', type=Path)
    args = parser.parse_args();result = check(args.directory, args.upstream)
    text = json.dumps(result, indent=2)+'\n'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True);args.report.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
