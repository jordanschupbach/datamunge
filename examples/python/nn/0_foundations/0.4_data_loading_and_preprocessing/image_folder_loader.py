# Native Python NN example. Image-folder datasets with PNG/JPEG decoding and bilinear resize.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.4_data_loading_and_preprocessing/image_folder_loader.py
"""Image-folder datasets with PNG/JPEG decoding and bilinear resize.

Port of tflow's image_folder_loader. Generate small RGB PNG/JPEG and
grayscale PNG fixtures. Scan class directories in deterministic order,
decode mixed image formats to RGB, and assemble a shuffled float32 batch
with aligned labels. Compare every resized PNG pixel with an independent
half-pixel bilinear calculation and bound JPEG's lossy error. Check
grayscale expansion, stable labels when a validation class is absent, and
explicit corrupt-image errors.

Optional root=PATH scans an existing folder (one subdirectory per class) and
decodes its first four images to 32x32 RGB. The shared decoder stretches
aspect ratio; callers can add cropping or padding for a different policy.

Differences from tflow: decoding uses nn.load_image_rgb (datamunge::image's
PNG reader and the self-contained JPEG decoder, which reproduces TF's
DecodeJpeg bit for bit) and nn.scan_image_folder for class names and
validation. datamunge has no JPEG encoder, so the JPEG fixture is the exact
byte stream TF's EncodeJpeg(quality=100, chroma_downsampling=false) produces
for the 4x4 solid image, embedded below. Labels are int64 (TF: int32).
Fixtures go to output/nn/python/<mode>/fixtures/images{,_validation}.

Binding notes: the SWIG layer exposes ImageFolder.images
(std::vector<ImageFile>) only as an opaque pointer and has no
std::vector<std::size_t> for load_image_batch's rows, so image_files lists
the class folders with scan_image_folder's documented rules (sorted class
directories, .png/.jpg/.jpeg in any case, sorted paths) and load_batch stacks
nn.load_image_rgb results (what load_image_batch does). PNG fixtures are
written with nn.save_png from byte-valued tensors (low=0, high=255, exact).
The C++ image::decode_jpeg comparison uses nn.load_image_rgb at the native
4x4 size (an identity resize) and recovers the bytes.
"""
import math
import sys
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import helper

checks = helper("checks")

# tf.io.encode_jpeg(4x4 RGB (64, 128, 192), quality=100, chroma_downsampling=False), TF 2.20.
SOLID_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010101012c012c0000ffdb0043000101010101010101010101010101010101010101010101"
    "0101010101010101010101010101010101010101010101010101010101010101010101010101010101ffdb0043010101"
    "010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101"
    "0101010101010101010101010101ffc00011080004000403011100021101031101ffc4001f0000010501010101010100"
    "000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d010203000411051221"
    "31410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a"
    "434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a9293949596979899"
    "9aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1"
    "f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405060708090a0bffc400b51100"
    "020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f015"
    "6272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a63646566676869"
    "6a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4"
    "c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00f1"
    "faff00560ff35cffd9"
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


def write_binary(path, data):
    try:
        Path(path).write_bytes(bytes(data))
    except OSError:
        raise RuntimeError(f"write {path}") from None


def image_files(root, class_names=()):
    """(path, label) pairs in nn::scan_image_folder order; the scan itself
    validates the root and class names and supplies the label order."""
    folder = nn.scan_image_folder(str(root), list(class_names))
    files = []
    for label, name in enumerate(folder.class_names):
        directory = Path(root) / name
        if not directory.exists():
            continue  # a class absent from this split keeps its label
        paths = sorted(
            str(entry) for entry in directory.iterdir() if entry.is_file() and entry.suffix.lower() in IMAGE_EXTENSIONS
        )
        files += [(path, label) for path in paths]
    return list(folder.class_names), files


def load_batch(files, rows, height, width):
    """images [rows, height, width, 3] float32 in [0, 1] and int64 labels."""
    for row in rows:
        if row >= len(files):
            raise IndexError("load_image_batch: row index out of range")
    images = nn.stack([nn.load_image_rgb(files[row][0], height, width) for row in rows], 0)
    labels = nn.from_vector_int64([files[row][1] for row in rows], [len(rows)])
    return images, labels


def fixtures():
    root = checks.output_dir("fixtures/images")
    validation = checks.output_dir("fixtures/images_validation")
    for directory in (root / "ant", root / "zebra", validation / "zebra"):
        directory.mkdir(parents=True, exist_ok=True)
    rgb = [pixel * 40 + c * 30 for pixel in range(4) for c in range(3)]
    nn.save_png(nn.from_vector([float(v) for v in rgb], [2, 2, 3]), str(root / "ant" / "pattern.PNG"), 0, 255)
    nn.save_png(nn.full([2, 2, 1], 77), str(root / "zebra" / "gray.png"), 0, 255)
    write_binary(root / "zebra" / "solid.jpg", SOLID_JPEG)
    write_binary(validation / "zebra" / "solid.jpg", SOLID_JPEG)
    write_binary(root / "ant" / "notes.txt", b"ignore")

    class_names, files = image_files(root)
    checks.require(class_names == ["ant", "zebra"] and len(files) == 3, "sorted classes and supported files")
    checks.require(files[0][1] == 0 and files[1][1] == 1 and files[2][1] == 1, "deterministic class labels")
    _, held_out = image_files(validation, class_names)
    checks.require(len(held_out) == 1 and held_out[0][1] == 1, "missing validation class preserves training labels")

    images, labels_t = load_batch(files, [2, 0, 1], 4, 4)
    checks.require(tuple(images.shape()) == (3, 4, 4, 3), "resized NHWC batch")
    labels = labels_t.values()
    checks.require(labels[0] == 1 and labels[1] == 0 and labels[2] == 1, "batch label alignment")
    pixels = images.values()

    def at(n, y, x, c):
        return pixels[((n * 4 + y) * 4 + x) * 3 + c]

    png_error = jpeg_error = 0.0
    for y in range(4):
        for x in range(4):
            for c in range(3):
                sy = min(max((y + 0.5) / 2 - 0.5, 0.0), 1.0)
                sx = min(max((x + 0.5) / 2 - 0.5, 0.0), 1.0)
                y0, x0 = math.floor(sy), math.floor(sx)
                y1, x1 = min(y0 + 1, 1), min(x0 + 1, 1)
                wy, wx = sy - y0, sx - x0

                def value(row, col):
                    return rgb[(row * 2 + col) * 3 + c] / 255.0

                expected = (1 - wy) * ((1 - wx) * value(y0, x0) + wx * value(y0, x1)) + wy * (
                    (1 - wx) * value(y1, x0) + wx * value(y1, x1)
                )
                png_error = max(png_error, abs(at(1, y, x, c) - expected))
                jpeg_error = max(jpeg_error, abs(at(0, y, x, c) - (64 + c * 64) / 255.0))
                checks.near(at(1, y, x, c), expected, 1e-7, "bilinear half-pixel coordinates")
                checks.near(at(0, y, x, c), (64 + c * 64) / 255.0, 2.0 / 255, "JPEG lossy error")
                checks.near(at(2, y, x, c), 77.0 / 255, 1e-7, "grayscale expands to RGB")
    checks.throws(lambda: nn.scan_image_folder(str(root), ["ant"]), "unknown validation class must fail")
    # Put corrupt input outside class folders so subsequent scans stay stable.
    corrupt = root / "corrupt.png"
    write_binary(corrupt, bytes([1, 2, 3, 4]))
    bad = False
    try:
        nn.load_image_rgb(str(corrupt), 4, 4)
    except RuntimeError:
        bad = True
    checks.require(bad, "corrupt PNG must fail")
    # Extra (not in tflow): a truncated JPEG must fail too, and the decoded fixture
    # must equal TF's DecodeJpeg output (64, 128, 192 at quality 100: lossless here).
    truncated = root / "truncated.jpg"
    write_binary(truncated, SOLID_JPEG[: len(SOLID_JPEG) // 2])
    bad = False
    try:
        nn.load_image_rgb(str(truncated), 4, 4)
    except RuntimeError:
        bad = True
    checks.require(bad, "truncated JPEG must fail")
    decoded = nn.load_image_rgb(str(root / "zebra" / "solid.jpg"), 4, 4).values()
    for i, v in enumerate(decoded):
        checks.require(round(v * 255) == 64 + (i % 3) * 64, "JPEG fixture decodes to TF's pixels")
    print("  classes=(ant,zebra), 3 images, shuffled batch [3,4,4,3]")
    print(f"  max PNG resize error={'%g' % png_error}, max JPEG error={'%g' % jpeg_error}")
    print("  grayscale RGB expansion, stable validation labels and corrupt-file rejection passed")


def main(arguments=None):
    try:
        root = None
        for arg in sys.argv[1:] if arguments is None else arguments:
            if arg in ("--quick", "--full"):
                continue
            if not arg.startswith("root=") or len(arg) == 5:
                raise ValueError("expected root=PATH")
            root = arg[5:]
        print("== Image folder and PNG/JPEG preprocessing")
        fixtures()
        if root is not None:
            class_names, files = image_files(root)
            images, _ = load_batch(files, list(range(min(4, len(files)))), 32, 32)
            shape = images.shape()
            print(
                f"  requested folder: {len(class_names)} classes, {len(files)} files; "
                f"first batch=[{shape[0]},{shape[1]},{shape[2]},{shape[3]}]"
            )
        print("All image-folder checks passed.")
    except Exception as error:  # noqa: BLE001 - mirrors the C++ catch-all
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
