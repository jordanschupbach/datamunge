"""CIFAR-10/CIFAR-100 binary records to NHWC nn tensors (examples/nn/common/cifar.hpp).

CIFAR-10 records have one label byte followed by 1024 red, green and blue
values. CIFAR-100 records begin with coarse then fine labels; both are kept and
the fine label is exposed as ``labels``. File order determines example order.
Incomplete/empty records and out-of-range labels are rejected before
allocating. Pixels are int32 holding 0..255 and labels int32 (nn has no uint8).
"""
from dataclasses import dataclass
from enum import Enum

from pydatamunge import nn
from pydatamunge.nn_workflows import helper

binary = helper("binary")

INT32_MAX = (1 << 31) - 1


class Cifar(Enum):
    k10 = 10
    k100 = 100


@dataclass
class CifarDataset:
    images: object = None  # int32 [N, 32, 32, 3], values 0..255
    labels: object = None  # int32 [N] (fine labels for CIFAR-100)
    coarse_labels: object = None  # int32 [N], CIFAR-100 only

    def size(self):
        return self.labels.numel() if self.labels is not None else 0


def parse_cifar(files, kind):
    """data::ParseCifar over a list of byte buffers."""
    if not files or kind not in (Cifar.k10, Cifar.k100):
        raise ValueError("need CIFAR files and valid format")
    labels = 2 if kind == Cifar.k100 else 1
    record = 3072 + labels
    count = 0
    for data in files:
        if not data or len(data) % record:
            raise ValueError("incomplete or empty CIFAR records")
        count += len(data) // record
        first = data[0::record]
        if max(first) >= (20 if labels == 2 else 10) or (labels == 2 and max(data[1::record]) >= 100):
            raise ValueError("CIFAR label out of range")
    pixels = binary.product(count, 3072)
    if pixels > INT32_MAX:
        raise ValueError("CIFAR tensor too large")
    # Records [N, labels + 3072] -> planar [N, 3, 1024] -> NHWC [N, 32, 32, 3].
    records = binary.to_tensor(b"".join(bytes(data) for data in files), [count, record])
    planar = nn.reshape(nn.slice_axis(records, 1, labels, record), [count, 3, 1024])
    images = nn.reshape(nn.transpose(planar, [0, 2, 1]).contiguous(), [count, 32, 32, 3])
    result = CifarDataset()
    result.images = images
    result.labels = nn.reshape(nn.slice_axis(records, 1, labels - 1, labels).contiguous(), [count])
    if labels == 2:
        result.coarse_labels = nn.reshape(nn.slice_axis(records, 1, 0, 1).contiguous(), [count])
    return result


def load_cifar(paths, kind):
    """data::LoadCifar: read every file, then parse them in order."""
    return parse_cifar([binary.read_binary(path) for path in paths], kind)
