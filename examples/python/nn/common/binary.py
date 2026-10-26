"""Bounded binary file reads and checked dimension arithmetic (examples/nn/common/binary.hpp).

Bytes are Python ``bytes``. ``to_tensor`` turns a byte buffer into an int32
tensor holding 0..255 (nn has no uint8 dtype, matching common/cifar.hpp and
nn::read_idx) by framing it as a ``|u1`` .npy array for nn.load_npy.
"""
import os
import tempfile
from pathlib import Path

from pydatamunge import nn

SIZE_MAX = (1 << 64) - 1


def read_binary(path, max_bytes=512 * 1024 * 1024):
    """data::ReadBinary: the whole file, rejecting files above max_bytes."""
    path = Path(path)
    try:
        size = path.stat().st_size
        handle = path.open("rb")
    except OSError:
        raise RuntimeError("cannot open " + str(path)) from None
    with handle:
        if size > max_bytes:
            raise RuntimeError("file exceeds byte limit: " + str(path))
        result = handle.read(size)
        if len(result) != size:
            raise RuntimeError("short read: " + str(path))
        if handle.read(1):
            raise RuntimeError("file grew while reading: " + str(path))
    return result


def write_binary(path, data):
    """data::WriteBinary: truncate and write all bytes."""
    try:
        with open(path, "wb") as out:
            out.write(bytes(data))
    except OSError:
        raise RuntimeError("cannot write " + str(path)) from None


def product(a, b):
    """data::Product: a * b, rejecting size_t overflow."""
    if b and a > SIZE_MAX // b:
        raise ValueError("dataset dimensions overflow")
    return a * b


def big_endian32(data, offset):
    """data::BigEndian32: the unsigned big-endian 32-bit integer at offset."""
    if offset > len(data) or len(data) - offset < 4:
        raise ValueError("truncated integer")
    return int.from_bytes(data[offset : offset + 4], "big")


def to_tensor(data, shape):
    """Bytes as an int32 tensor of the given shape (values 0..255)."""
    count = 1
    for dimension in shape:
        count = product(count, dimension)
    if count != len(data):
        raise ValueError("byte count does not fill the tensor shape")
    header = "{'descr': '|u1', 'fortran_order': False, 'shape': (%s), }" % "".join(
        f"{d}, " for d in shape
    )
    header += " " * (63 - (10 + len(header)) % 64) + "\n"
    descriptor, name = tempfile.mkstemp(suffix=".npy")
    try:
        with os.fdopen(descriptor, "wb") as out:
            out.write(b"\x93NUMPY\x01\x00" + len(header).to_bytes(2, "little"))
            out.write(header.encode("latin-1"))
            out.write(data)
        return nn.load_npy(name)
    finally:
        os.remove(name)
