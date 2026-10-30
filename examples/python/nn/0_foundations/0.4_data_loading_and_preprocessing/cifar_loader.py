# Native Python NN example. CIFAR-10/CIFAR-100 binary decoding, planar-to-NHWC conversion and labels.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.4_data_loading_and_preprocessing/cifar_loader.py
"""CIFAR-10/CIFAR-100 binary decoding, planar-to-NHWC conversion and labels.

Port of tflow's cifar_loader. Write original-format records with
deterministic pixel patterns, read them from disk, and check every output
coordinate. CIFAR-100 exposes both fine and coarse labels. Multiple CIFAR-10
batch files concatenate in caller order. Normalize the pixels with nn ops
and verify channel means. Reject empty, truncated and trailing records,
invalid labels and bad formats.

Optional format=10|100 and repeated file=PATH arguments load real extracted
binary batches (the CIFAR-10 cache lives in
$DATAMUNGE_DATA_DIR/cifar10/data_batch_*.bin). Download the binary archives
from the dataset authors at https://www.cs.toronto.edu/~kriz/cifar.html;
Python-pickle files are different. Real CIFAR-10 files are also compared
with the library loader nn.load_cifar10 (float32 / 255).

Differences from TF: nn has no uint8 dtype, so common/cifar.py returns the
pixels as int32 holding 0..255 (labels int32 as in tflow). Fixtures are
written under output/nn/python/<mode>/fixtures/ (checks.output_dir).
"""
import sys
from pathlib import Path

from pydatamunge.nn_components import dataset_assets
from pydatamunge import nn
from pydatamunge.nn_workflows import helper

checks = helper("checks")
binary = helper("binary")
cifar = helper("cifar")
Cifar = cifar.Cifar


def record(kind, index):
    data = bytearray()
    if kind == Cifar.k100:
        data.append(4 + index)
    data.append(73 + index if kind == Cifar.k100 else 3 + index)
    for channel in range(3):
        for pixel in range(1024):
            data.append((index * 17 + channel * 61 + pixel) % 256)
    return bytes(data)


def fixtures(kind):
    a, b = record(kind, 0), record(kind, 1)
    directory = checks.output_dir("fixtures/cifar10" if kind == Cifar.k10 else "fixtures/cifar100")
    binary.write_binary(directory / "batch1.bin", a)
    binary.write_binary(directory / "batch2.bin", b)
    dataset = cifar.load_cifar([directory / "batch1.bin", directory / "batch2.bin"], kind)
    checks.require(tuple(dataset.images.shape()) == (2, 32, 32, 3), "CIFAR NHWC shape")
    images, labels = dataset.images.values(), dataset.labels.values()
    coarse = dataset.coarse_labels.values() if kind == Cifar.k100 else None
    for row in range(2):
        checks.require(labels[row] == (73 if kind == Cifar.k100 else 3) + row, "fine label/order")
        if kind == Cifar.k100:
            checks.require(coarse[row] == 4 + row, "coarse label/order")
        for pixel in range(1024):
            for channel in range(3):
                checks.require(
                    images[row * 3072 + pixel * 3 + channel] == (row * 17 + channel * 61 + pixel) % 256,
                    "channel-major bytes to NHWC",
                )
    normalized = nn.div(nn.cast(dataset.images, nn.DType.Float32), 255.0)
    means = nn.mean(normalized, [0, 1, 2]).values()
    for i in range(3):
        checks.near(means[i], 0.5, 2e-6, "channel mean")
    failures = 0

    def reject(data):
        nonlocal failures
        rejected = False
        try:
            cifar.parse_cifar([data], kind)
        except ValueError:
            rejected = True
        checks.require(rejected, "invalid CIFAR records must fail")
        failures += 1

    reject(b"")
    reject(a[:-1])
    reject(a + b"\x00")
    bad = bytearray(a)
    bad[0] = 20 if kind == Cifar.k100 else 10
    reject(bytes(bad))
    if kind == Cifar.k100:
        bad = bytearray(a)
        bad[1] = 100
        reject(bytes(bad))
    print(
        f"  CIFAR-{10 if kind == Cifar.k10 else 100}: 2 files -> [2,32,32,3], all 6144 pixels and labels verified\n"
        f"    normalized channel means=(0.5,0.5,0.5), {failures} malformed records rejected"
    )


def main(arguments=None):
    try:
        nn.set_num_threads(2)
        kind, files = Cifar.k10, []
        for arg in sys.argv[1:] if arguments is None else arguments:
            if arg in ("--quick", "--full"):
                continue
            key, _, value = arg.partition("=")
            if key == "file" and value:
                files.append(Path(value))
            elif key == "format" and value in ("10", "100"):
                kind = Cifar.k10 if value == "10" else Cifar.k100
            else:
                raise ValueError("expected format=10|100 or file=PATH")
        print("== CIFAR binary loader checks")
        fixtures(Cifar.k10)
        fixtures(Cifar.k100)
        if files:
            dataset = cifar.load_cifar(files, kind)
            histogram = [0] * (10 if kind == Cifar.k10 else 100)
            for label in dataset.labels.values():
                histogram[int(label)] += 1
            print(f"  loaded {len(files)} real file(s), shape=[{dataset.images.size(0)},32,32,3]")
            print("    fine-label counts:" + "".join(f" {n}" for n in histogram))
            # A single standard test_batch.bin can be checked against the library loader.
            if kind == Cifar.k10 and len(files) == 1 and files[0].name == "test_batch.bin":
                library = dataset_assets.LoadCifar10(str(files[0].parent), "test")
                ours = nn.div(nn.cast(dataset.images, nn.DType.Float32), 255.0)
                checks.require(
                    nn.all(nn.equal(ours, library.images)).item() == 1
                    and library.labels.values() == dataset.labels.values(),
                    "nn::load_cifar10 agrees with the byte decoder",
                )
                print("    identical to nn::load_cifar10 (pixels / 255, labels)")
        print("All CIFAR loader checks passed.")
    except Exception as error:  # noqa: BLE001 - mirrors the C++ catch-all
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
