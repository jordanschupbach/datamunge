# Native Python NN example. TFRecord writing, streaming reads, compression and tf.Example parsing.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.4_data_loading_and_preprocessing/tfrecord_io.py
"""TFRecord writing, streaming reads, compression and tf.Example parsing.

Port of tflow's tfrecord_io. Serialize float, integer and bytes features in
tf.Example records. Round-trip identical payloads through plain, GZIP and
ZLIB TFRecord files, then parse required and defaulted features with a
ParseSingleExample-style schema. Check empty records versus EOF, embedded
NUL bytes and an empty file. Corrupted length headers, corrupted payloads
and truncated records must fail checksum validation. Missing required
features must fail parsing.

Differences from TF: TFRecord and tf.Example are out of the nn library's
scope (no TensorFlow or protobuf runtime), so the codec of
examples/nn/common/tfrecord.hpp (used only by this example) is written out
below from the published formats: CRC-32C framing, gzip/zlib wrapping and a
minimal Example protobuf codec. Parsing is a host function; float and int64
features become nn tensors, bytes features Python bytes. DEFLATE uses
Python's zlib module (level 6) instead of datamunge's internal codec, which
nn does not expose: payloads and plain files are byte-identical to the C++
run, but compressed file sizes may differ. Files go to
output/nn/python/<mode>/fixtures/tfrecord/.

Option verify=DIR reads plain/GZIP/ZLIB files written by a TF run
(tflow's output/fixtures/tfrecord), checking CRCs and every parsed feature.

TFRecord framing: uint64 little-endian payload length, masked CRC-32C of
those 8 bytes, the payload, masked CRC-32C of the payload. Compression is
"", "GZIP" (one gzip member around the whole stream) or "ZLIB" (one zlib
stream). tf.Example: Example{features=1: Features{feature=1: map<string,
Feature>}}, Feature oneof {bytes_list=1, float_list=2, int64_list=3}, each a
repeated field 1 (floats and int64s packed). Map entries are written in key
order. The parser accepts packed and unpacked numbers and skips unknown fields.
"""
import struct
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import helper

checks = helper("checks")
binary = helper("binary")

MASK32, MASK64 = 0xFFFFFFFF, (1 << 64) - 1


# ---- CRC-32C (Castagnoli), masked as TensorFlow's crc32c::Mask ------------------------


def _crc_table():
    table = []
    for i in range(256):
        c = i
        for _ in range(8):
            c = 0x82F63B78 ^ (c >> 1) if c & 1 else c >> 1
        table.append(c)
    return table


CRC32C_TABLE = _crc_table()


def crc32c(data):
    crc = 0xFFFFFFFF
    for byte in data:
        crc = CRC32C_TABLE[(crc ^ byte) & 0xFF] ^ (crc >> 8)
    return crc ^ 0xFFFFFFFF


def masked_crc(data):
    crc = crc32c(data)
    return (((crc >> 15) | (crc << 17)) + 0xA282EAD8) & MASK32


def validate_compression(compression):
    if compression not in ("", "GZIP", "ZLIB"):
        raise ValueError("TFRecord compression must be empty, GZIP or ZLIB")


def put_le(value, count):
    return (value & ((1 << (8 * count)) - 1)).to_bytes(count, "little")


def get_le(data, at, count):
    return int.from_bytes(data[at : at + count], "little")


def deflate(raw):
    """A raw DEFLATE stream (zlib level 6, no zlib header/trailer)."""
    compressor = zlib.compressobj(6, zlib.DEFLATED, -15)
    return compressor.compress(raw) + compressor.flush()


def gzip_wrap(raw):
    header = bytes([0x1F, 0x8B, 8, 0, 0, 0, 0, 0, 0, 255])  # no flags, no mtime, unknown OS
    return header + deflate(raw) + put_le(zlib.crc32(raw), 4) + put_le(len(raw), 4)


def gunzip(b):
    if len(b) < 18 or b[0] != 0x1F or b[1] != 0x8B or b[2] != 8:
        raise RuntimeError("GZIP TFRecord: bad header")
    flags, at = b[3], 10
    if flags & 0x04:
        if at + 2 > len(b):
            raise RuntimeError("GZIP TFRecord: truncated header")
        at += 2 + (b[at] | (b[at + 1] << 8))
    for bit in (0x08, 0x10):
        if flags & bit:
            while at < len(b) and b[at]:
                at += 1
            at += 1
    if flags & 0x02:
        at += 2
    if at + 8 > len(b):
        raise RuntimeError("GZIP TFRecord: truncated stream")
    trailer = len(b) - 8
    crc, isize = get_le(b, trailer, 4), get_le(b, trailer + 4, 4)
    try:
        inflater = zlib.decompressobj(-15)
        out = inflater.decompress(b[at:trailer]) + inflater.flush()
    except zlib.error as error:
        raise RuntimeError("GZIP TFRecord: " + str(error)) from None
    if len(out) != isize:
        raise RuntimeError("GZIP TFRecord: inflated size mismatch")
    if zlib.crc32(out) != crc:
        raise RuntimeError("GZIP TFRecord: CRC-32 mismatch")
    return out


def write_tfrecords(path, records, compression=""):
    validate_compression(compression)
    stream = bytearray()
    for record in records:
        length = put_le(len(record), 8)
        stream += length + put_le(masked_crc(length), 4) + record + put_le(masked_crc(record), 4)
    stream = bytes(stream)
    if compression == "GZIP":
        stream = gzip_wrap(stream)
    if compression == "ZLIB":
        stream = zlib.compress(stream, 6)
    binary.write_binary(path, stream)


def read_tfrecords(path, compression=""):
    validate_compression(compression)
    stream = binary.read_binary(path)
    if compression == "GZIP":
        stream = gunzip(stream)
    if compression == "ZLIB":
        try:
            stream = zlib.decompress(stream)
        except zlib.error as error:
            raise RuntimeError("ZLIB TFRecord: " + str(error)) from None
    records, at = [], 0
    while at < len(stream):
        if len(stream) - at < 12:
            raise RuntimeError("truncated TFRecord length header")
        if masked_crc(stream[at : at + 8]) != get_le(stream, at + 8, 4):
            raise RuntimeError("corrupted TFRecord length (CRC-32C mismatch)")
        length = get_le(stream, at, 8)
        at += 12
        if length > len(stream) - at or len(stream) - at - length < 4:
            raise RuntimeError("truncated TFRecord payload")
        data = stream[at : at + length]
        if masked_crc(data) != get_le(stream, at + length, 4):
            raise RuntimeError("corrupted TFRecord payload (CRC-32C mismatch)")
        records.append(data)
        at += length + 4
    return records


# ---- tf.Example -------------------------------------------------------------------------

BYTES, FLOAT, INT64 = 1, 2, 3


@dataclass
class Feature:
    """One tf.train.Feature: exactly one of the three lists is used."""

    kind: int = FLOAT
    bytes: list = field(default_factory=list)
    floats: list = field(default_factory=list)
    ints: list = field(default_factory=list)


def bytes_feature(values):
    return Feature(BYTES, bytes=list(values))


def float_feature(values):
    return Feature(FLOAT, floats=[struct.unpack("<f", struct.pack("<f", v))[0] for v in values])


def int64_feature(values):
    return Feature(INT64, ints=list(values))


def varint(value):
    value &= MASK64
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def delimited(field_number, payload):
    return varint((field_number << 3) | 2) + varint(len(payload)) + payload


class Reader:
    def __init__(self, data):
        self.s, self.at = data, 0

    def done(self):
        return self.at >= len(self.s)

    def varint(self):
        value = 0
        for shift in range(0, 64, 7):
            if self.at >= len(self.s):
                raise ValueError("tf.Example: truncated varint")
            b = self.s[self.at]
            self.at += 1
            value |= (b & 0x7F) << shift
            if not b & 0x80:
                return value & MASK64
        raise ValueError("tf.Example: varint too long")

    def bytes(self):
        n = self.varint()
        if n > len(self.s) - self.at:
            raise ValueError("tf.Example: truncated field")
        out = self.s[self.at : self.at + n]
        self.at += n
        return out

    def skip(self, wire):
        if wire == 0:
            self.varint()
        elif wire == 1:
            self.at += 8
        elif wire == 2:
            self.bytes()
        elif wire == 5:
            self.at += 4
        else:
            raise ValueError("tf.Example: unsupported wire type")
        if self.at > len(self.s):
            raise ValueError("tf.Example: truncated field")


def signed64(value):
    return value - (1 << 64) if value >= 1 << 63 else value


def serialize_example(example):
    features = b""
    for key in sorted(example, key=lambda k: k.encode("utf-8")):
        feature = example[key]
        values = b""
        if feature.kind == BYTES:
            for item in feature.bytes:
                values += delimited(1, item)
        elif feature.kind == FLOAT:
            if feature.floats:
                values += delimited(1, b"".join(struct.pack("<f", v) for v in feature.floats))
        elif feature.ints:
            values += delimited(1, b"".join(varint(v) for v in feature.ints))
        value = delimited(feature.kind, values)
        entry = delimited(1, key.encode("utf-8")) + delimited(2, value)
        features += delimited(1, entry)
    return delimited(1, features)


def parse_feature(s):
    f = Feature()
    r = Reader(s)
    while not r.done():
        tag = r.varint()
        number, wire = tag >> 3, tag & 7
        if number < 1 or number > 3 or wire != 2:
            r.skip(wire)
            continue
        f = Feature(number)
        values = r.bytes()
        lr = Reader(values)
        while not lr.done():
            ltag = lr.varint()
            lfield, lwire = ltag >> 3, ltag & 7
            if lfield != 1:
                lr.skip(lwire)
                continue
            if number == BYTES:
                if lwire != 2:
                    raise ValueError("tf.Example: bad bytes_list")
                f.bytes.append(lr.bytes())
            elif number == FLOAT:
                if lwire == 2:
                    packed = lr.bytes()
                    if len(packed) % 4:
                        raise ValueError("tf.Example: bad packed float_list")
                    f.floats += [struct.unpack_from("<f", packed, i)[0] for i in range(0, len(packed), 4)]
                elif lwire == 5:
                    if lr.at + 4 > len(values):
                        raise ValueError("tf.Example: truncated float")
                    f.floats.append(struct.unpack_from("<f", values, lr.at)[0])
                    lr.at += 4
                else:
                    raise ValueError("tf.Example: bad float_list")
            else:
                if lwire == 2:
                    pr = Reader(lr.bytes())
                    while not pr.done():
                        f.ints.append(signed64(pr.varint()))
                elif lwire == 0:
                    f.ints.append(signed64(lr.varint()))
                else:
                    raise ValueError("tf.Example: bad int64_list")
    return f


def parse_example(serialized):
    """A serialized tf.Example as {key: Feature}; ValueError when malformed."""
    example = {}
    r = Reader(serialized)
    while not r.done():
        tag = r.varint()
        if tag >> 3 != 1 or tag & 7 != 2:
            r.skip(tag & 7)
            continue
        fr = Reader(r.bytes())
        while not fr.done():
            ftag = fr.varint()
            if ftag >> 3 != 1 or ftag & 7 != 2:
                fr.skip(ftag & 7)
                continue
            er = Reader(fr.bytes())
            key, value = b"", Feature()
            while not er.done():
                etag = er.varint()
                if etag >> 3 == 1 and etag & 7 == 2:
                    key = er.bytes()
                elif etag >> 3 == 2 and etag & 7 == 2:
                    value = parse_feature(er.bytes())
                else:
                    er.skip(etag & 7)
            example[key.decode("utf-8", "surrogateescape")] = value  # later entries win
    return example


@dataclass
class DenseSpec:
    """A fixed-length feature: size values of kind, or default when absent (None = required)."""

    key: str
    kind: int
    size: int
    default: Feature = None


def parse_single_example(serialized, specs):
    """TF ParseSingleExample for dense features: one Feature per spec, in spec order."""
    example = parse_example(serialized)
    out = []
    for spec in specs:
        if spec.key not in example:
            if spec.default is None:
                raise ValueError(f"Feature: {spec.key} (data type) is required but could not be found.")
            out.append(spec.default)
            continue
        f = example[spec.key]
        n = len(f.bytes if f.kind == BYTES else f.floats if f.kind == FLOAT else f.ints)
        if f.kind != spec.kind:
            raise ValueError(f"Feature: {spec.key} has the wrong data type")
        if n != spec.size:
            raise ValueError(f"Feature: {spec.key} has {n} values, expected {spec.size}")
        out.append(f)
    return out


# ---- the example ------------------------------------------------------------------------


def examples():
    records = []
    for i in range(3):
        example = {
            "x": float_feature([i + j * 0.25 for j in range(3)]),
            "label": int64_feature([i % 2]),
            "id": bytes_feature([f"sample-{i}".encode()]),
        }
        if i != 1:
            example["weight"] = float_feature([2])
        records.append(serialize_example(example))
    return records


SCHEMA = [
    DenseSpec("x", FLOAT, 3),
    DenseSpec("label", INT64, 1),
    DenseSpec("id", BYTES, 1),
    DenseSpec("weight", FLOAT, 1, float_feature([1])),
]


def check_parsed(records):
    """Parses records with the schema and checks every value."""
    checks.require(len(records) == 3, "three Example records")
    for i, record in enumerate(records):
        parsed = parse_single_example(record, SCHEMA)
        x = nn.from_vector(parsed[0].floats, [3])
        label = nn.reshape(nn.from_vector_int64(parsed[1].ints, [1]), [])
        xv = x.values()
        for j in range(3):
            checks.near(xv[j], i + j * 0.25, 0, "float feature")
        checks.require(label.item() == i % 2, "integer feature")
        checks.require(parsed[2].bytes[0] == f"sample-{i}".encode(), "bytes feature")
        checks.near(parsed[3].floats[0], 1 if i == 1 else 2, 0, "defaulted feature")


def parse_schema(records):
    check_parsed(records)
    # A valid empty Example proto lacks every required feature.
    checks.throws(lambda: parse_single_example(b"", SCHEMA), "missing required feature must fail")
    print("  parsed float[3], int64 label, bytes id and defaulted weight")
    print("  weights=(2,1,2); missing required features rejected")


def round_trip():
    root = checks.output_dir("fixtures/tfrecord")
    records = examples()
    for compression in ("", "GZIP", "ZLIB"):
        path = root / ((compression or "plain") + ".tfrecord")
        write_tfrecords(path, records, compression)
        restored = read_tfrecords(path, compression)
        checks.require(restored == records, "exact TFRecord payload round trip")
        print(f"  {compression or 'plain'}: {len(restored)} exact records, file bytes={path.stat().st_size}")
    payloads = [b"", b"a\0b", b"last"]
    write_tfrecords(root / "bytes.tfrecord", payloads)
    checks.require(read_tfrecords(root / "bytes.tfrecord") == payloads, "empty and binary records distinct from EOF")
    write_tfrecords(root / "empty.tfrecord", [])
    checks.require(not read_tfrecords(root / "empty.tfrecord"), "empty TFRecord file")
    original = binary.read_binary(root / "plain.tfrecord")
    failures = 0
    for mode in range(3):
        damaged = bytearray(original)
        if mode == 0:
            damaged[0] ^= 1
        elif mode == 1:
            damaged[12] ^= 1
        else:
            damaged.pop()
        path = root / f"damaged-{mode}.tfrecord"
        binary.write_binary(path, damaged)
        rejected = False
        try:
            read_tfrecords(path)
        except RuntimeError:
            rejected = True
        checks.require(rejected, "corrupt/truncated TFRecord must fail")
        failures += 1
    rejected = False
    try:
        read_tfrecords(root / "plain.tfrecord", "unknown")
    except ValueError:
        rejected = True
    checks.require(rejected, "unsupported compression cannot silently fall back")
    print(f"  empty/binary payloads and EOF passed; {failures} corrupt/truncated streams rejected")
    parse_schema(records)


def verify_tf(directory):
    for compression in ("", "GZIP", "ZLIB"):
        path = directory / ((compression or "plain") + ".tfrecord")
        check_parsed(read_tfrecords(path, compression))
        print(f"  TF-written {path.name} ({path.stat().st_size} bytes): CRCs valid, every feature identical")


def main(arguments=None):
    try:
        nn.set_num_threads(2)
        verify = None
        for arg in sys.argv[1:] if arguments is None else arguments:
            if arg in ("--quick", "--full"):
                continue
            if arg.startswith("verify=") and len(arg) > 7:
                verify = Path(arg[7:])
            else:
                raise ValueError("expected verify=DIR")
        print("== TFRecord round trips and schema parsing")
        round_trip()
        if verify is not None:
            verify_tf(verify)
        print("All TFRecord checks passed.")
    except Exception as error:  # noqa: BLE001 - mirrors the C++ catch-all
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
