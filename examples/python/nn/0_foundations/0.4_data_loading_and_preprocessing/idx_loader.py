# Native Python NN example. MNIST and Fashion-MNIST: strict IDX loading and nn preprocessing.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.4_data_loading_and_preprocessing/idx_loader.py
"""MNIST and Fashion-MNIST: strict IDX loading and nn preprocessing.

Port of tflow's idx_loader. Create a tiny on-disk IDX pair with known
bytes, load uint8 NHWC images and integer labels, shuffle matched
image/label rows, then cast and divide by 255 with nn ops. Check malformed
magic, truncated and trailing data, dimension overflow, count mismatches and
label ranges before allocation.

Also inspect cached real test sets under $DATAMUNGE_DATA_DIR/{mnist,
fashion_mnist} (default ~/projects/tflow_examples/output/datasets; fetch
there with python3 scripts/fetch_datasets.py both --split test). Optional
root=PATH changes that root; dir=PATH loads a required IDX pair directly, and
split=train|t10k selects its filename prefix. No network access occurs.

Differences from tflow: nn.read_idx reads one IDX file of any element type
(and gunzips ".gz" files itself), so the pair rules of tflow's ParseIdx
(uint8 rank-3 images, uint8 rank-1 labels, nonzero dimensions, equal counts,
labels < 10) are checked here in load_pair. nn has no uint8 dtype: pixels
come back as int32 holding 0..255. tflow's "bounded file read" check has no
nn counterpart and is dropped; instead the real test set is also loaded from
its .gz file and compared.
"""
import sys
from dataclasses import dataclass
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import data_root, helper
from pydatamunge.nn_components import dataset_assets

checks = helper("checks")


@dataclass
class Pair:
    images: object  # int32 [N, H, W, 1] with byte values
    labels: object  # int64 [N]


def append32(data, value):
    data += value.to_bytes(4, "big")


def write_binary(path, data):
    with open(path, "wb") as out:
        out.write(bytes(data))


def require_ubyte(path, rank):
    """The first four bytes of a (possibly gzip-compressed) IDX file: a uint8
    file's magic is 0x0000 08 <rank>."""
    try:
        with open(path, "rb") as handle:
            magic = handle.read(4)
    except OSError:
        magic = b""
    if len(magic) == 4 and magic[0] == 0x1F and magic[1] == 0x8B:
        return  # gzip: the payload is validated by read_idx
    if len(magic) < 4 or magic[0] or magic[1] or magic[2] != 0x08 or magic[3] != rank:
        raise ValueError(f"IDX requires uint8 rank-3 images and rank-1 labels: {path}")


def load_pair(images_path, labels_path, classes=10):
    """An MNIST-style IDX pair with tflow's ParseIdx rules."""
    require_ubyte(images_path, 3)
    require_ubyte(labels_path, 1)
    try:
        images = nn.read_idx(str(images_path))
        labels = nn.read_idx(str(labels_path))
    except RuntimeError as error:
        raise ValueError(str(error)) from None
    if images.dim() != 3 or labels.dim() != 1:
        raise ValueError("IDX requires rank-3 images and rank-1 labels")
    if any(d == 0 for d in images.shape()):
        raise ValueError("invalid IDX dimensions")
    if images.size(0) != labels.size(0):
        raise ValueError("IDX payload size/count mismatch")
    if nn.max(labels).item() >= classes:
        raise ValueError("IDX label out of range")
    return Pair(
        nn.reshape(images, [images.size(0), images.size(1), images.size(2), 1]),
        nn.cast(labels, nn.DType.Int64),
    )


def fixtures():
    images, labels = bytearray(), bytearray()
    for value in (2051, 2, 2, 3):
        append32(images, value)
    for value in (2049, 2):
        append32(labels, value)
    pixels = bytes([0, 51, 102, 153, 204, 255, 255, 204, 153, 102, 51, 0])
    images += pixels
    labels += bytes([7, 3])
    directory = checks.output_dir("fixtures/idx")
    write_binary(directory / "images.idx", images)
    write_binary(directory / "labels.idx", labels)
    dataset = load_pair(directory / "images.idx", directory / "labels.idx")
    checks.require(tuple(dataset.images.shape()) == (2, 2, 3, 1), "IDX NHWC shape")
    x = dataset.images.values()
    for i in range(12):
        checks.require(x[i] == pixels[i], "IDX pixel order")
    y = dataset.labels.values()
    checks.require(y[0] == 7 and y[1] == 3, "IDX label order")
    shuffled_x = nn.take_rows(dataset.images, [1, 0])
    shuffled_y = nn.take_rows(dataset.labels, [1, 0])
    checks.require(shuffled_x.at([0, 0, 0, 0]) == 255 and shuffled_y.at([0]) == 3, "paired row gathering")

    normalized = nn.div(nn.cast(shuffled_x, nn.DType.Float32), 255.0)
    means = nn.mean(normalized, [1, 2, 3])
    n, sx = normalized.values(), shuffled_x.values()
    for i in range(12):
        checks.near(n[i], sx[i] / 255.0, 1e-7, "pixel normalization")
    for m in means.values():
        checks.near(m, 0.5, 1e-7, "per-image mean")

    failures = 0

    def reject(bad_x, bad_y):
        nonlocal failures
        write_binary(directory / "bad_images.idx", bad_x)
        write_binary(directory / "bad_labels.idx", bad_y)
        checks.throws(
            lambda: load_pair(directory / "bad_images.idx", directory / "bad_labels.idx"),
            "invalid IDX pair must fail",
        )
        failures += 1

    def edit(data, index, value):
        copy = bytearray(data)
        copy[index] = value
        return copy

    reject(edit(images, 3, 0), labels)  # rank-0 image magic
    reject(images, edit(labels, 3, 0))  # rank-0 label magic
    reject(images[:-1], labels)  # truncated images
    reject(images + b"\x00", labels)  # trailing image byte
    reject(images, labels[:-1])  # truncated labels
    reject(images, labels + b"\x00")  # trailing label byte
    reject(edit(images, 7, 0), labels)  # zero images
    reject(edit(images, 11, 0), labels)  # zero rows
    reject(images, edit(labels, 7, 3))  # label count mismatch
    reject(images, edit(labels, len(labels) - 1, 10))  # label out of range
    overflow = bytearray(images)
    for i in range(4, 16):
        overflow[i] = 255  # dimension overflow
    reject(overflow, labels)
    reject(b"", labels)
    reject(images, b"")
    print("  2 images, shape [2,2,3,1], paired shuffle and normalization passed")
    print(f"  {failures} malformed IDX pairs rejected")


def inspect(directory, split):
    images = directory / (split + "-images-idx3-ubyte")
    labels = directory / (split + "-labels-idx1-ubyte")
    dataset = load_pair(images if images.exists() else Path(str(images) + ".gz"),
                        labels if labels.exists() else Path(str(labels) + ".gz"))
    n = dataset.labels.size(0)
    counts = [0] * 10
    for label in dataset.labels.values():
        counts[int(label)] += 1
    pixels = nn.cast(dataset.images, nn.DType.Float64)
    low, high = int(nn.min(pixels).item()), int(nn.max(pixels).item())
    print(
        f"  {directory.name}/{split}: {n} images, shape={nn.shape_str(dataset.images.shape())}, "
        f"normalized mean={'%g' % (nn.mean(pixels).item() / 255)}, byte range=[{low},{high}]"
    )
    print("    class counts:" + "".join(f" {count}" for count in counts))
    gz_images, gz_labels = Path(str(images) + ".gz"), Path(str(labels) + ".gz")
    if images.exists() and labels.exists() and gz_images.exists() and gz_labels.exists():
        gz = load_pair(gz_images, gz_labels)
        checks.require(
            nn.all(nn.equal(gz.images, dataset.images)).item() == 1
            and nn.all(nn.equal(gz.labels, dataset.labels)).item() == 1,
            "gzip IDX matches the decompressed file",
        )
        print("    .gz files decode to identical tensors")


def main(arguments=None):
    try:
        root, required, split = data_root(), None, "t10k"
        for arg in sys.argv[1:] if arguments is None else arguments:
            if arg in ("--quick", "--full"):
                continue
            key, _, value = arg.partition("=")
            if not value:
                raise ValueError("expected key=value")
            if key == "root":
                root = Path(value)
            elif key == "dir":
                required = Path(value)
            elif key == "split" and value in ("train", "t10k"):
                split = value
            else:
                raise ValueError("unknown IDX option: " + arg)
        print("== IDX fixture and validation checks")
        fixtures()
        print("\n== Real IDX datasets")
        if required is not None:
            inspect(required, split)
        else:
            for name in ("mnist", "fashion_mnist"):
                dataset_assets.EnsureMnist(str(root / name), name == "fashion_mnist", "train" if split == "train" else "test")
                inspect(root / name, split)
        print("All IDX loader checks passed.")
    except Exception as error:  # noqa: BLE001 - mirrors the C++ catch-all
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
