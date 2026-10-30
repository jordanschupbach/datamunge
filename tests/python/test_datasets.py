"""Built-in datasets, CSV streaming and the ImageLoader from Python (all offline)."""
import struct

import pytest


def _dm():
    from pydatamunge import datamunge

    return datamunge


def _write_idx(directory, prefix, n):
    # n images of 2x3 pixels: pixel p of image i is 10*i + p; label i % 10.
    with open(directory / f"{prefix}-images-idx3-ubyte", "wb") as f:
        f.write(struct.pack(">IIII", 0x803, n, 2, 3))
        f.write(bytes(10 * i + p for i in range(n) for p in range(6)))
    with open(directory / f"{prefix}-labels-idx1-ubyte", "wb") as f:
        f.write(struct.pack(">II", 0x801, n))
        f.write(bytes(i % 10 for i in range(n)))


def test_builtin_datasets_have_documented_shapes():
    dm = _dm()
    expected = {
        "wine": (178, 14),
        "titanic": (891, 12),
        "eight_schools": (8, 3),
        "faithful": (272, 2),
        "mtcars": (32, 12),
        "airquality": (153, 6),
        "breast_cancer": (569, 31),
        "diabetes": (442, 11),
        "digits": (1797, 65),
    }
    for name, (rows, cols) in expected.items():
        frame = getattr(dm.DataFrame, name)()
        assert (frame.nrows(), frame.ncols()) == (rows, cols), name
        same = dm.DataFrame.dataset(name)
        assert same.nrows() == rows
    titanic = dm.DataFrame.titanic()
    assert titanic.string_at("Sex", 0) == "male"
    assert titanic.is_null("Age", 5)
    assert dm.DataFrame.eight_schools().numeric_at("effect", 0) == 28


def test_catalog_and_unknown_names():
    dm = _dm()
    catalog = dm.DataFrame.dataset_catalog()
    assert catalog.nrows() == 22
    assert "diamonds" in list(dm.downloadable())
    with pytest.raises(ValueError):
        dm.DataFrame.dataset("no_such_dataset")
    with pytest.raises(RuntimeError):
        dm.DataFrame.read_csv("/definitely/not/here.csv")


def test_missing_image_files_raise_offline(tmp_path, monkeypatch):
    dm = _dm()
    monkeypatch.setenv("DATAMUNGE_DATA_OFFLINE", "1")
    with pytest.raises(RuntimeError, match="DATAMUNGE_DATA_OFFLINE"):
        dm.ImageLoader("mnist", "train", str(tmp_path))
    with pytest.raises(ValueError):
        dm.ImageLoader("imagenet", "train", str(tmp_path))


def test_csv_read_and_stream(tmp_path):
    dm = _dm()
    path = tmp_path / "rows.csv"
    path.write_text("i,name\n" + "".join(f"{i},row{i}\n" for i in range(25)))
    whole = dm.DataFrame.read_csv(str(path))
    assert whole.nrows() == 25
    assert whole.is_numeric_column("i") and not whole.is_numeric_column("name")

    reader = dm.CsvReader(str(path), 10)
    assert [chunk.nrows() for chunk in reader] == [10, 10, 5]
    assert reader.rows_read() == 25
    reader.reset()
    assert reader.next().string_at("name", 0) == "row0"

    parsed = dm.DataFrame.parse_csv("a;b\n1;x\n", ";")
    assert parsed.numeric_at("a", 0) == 1


def test_image_loader_batches_shuffles_and_converts(tmp_path):
    dm = _dm()
    _write_idx(tmp_path, "train", 10)
    _write_idx(tmp_path, "t10k", 2)
    loader = dm.ImageLoader("mnist", "train", str(tmp_path))
    # Chained setters return the same loader and keep it alive.
    same = loader.set_batch_size(4).set_normalize(False).set_layout("flat")
    assert same.batch_size() == 4
    assert len(loader) == 3

    first = loader.next()
    assert list(first.shape) == [4, 6]
    assert list(first.labels) == [0, 1, 2, 3]
    assert list(first.image(1)) == [10, 11, 12, 13, 14, 15]

    loader.reset()
    sizes = [batch.size() for batch in loader]
    assert sizes == [4, 4, 2]

    loader.set_shuffle(True).set_seed(3)
    order = [i for batch in loader for i in batch.indices]
    assert sorted(order) == list(range(10))

    frame = dm.DataFrame.from_image_batch(loader.take([4, 0]))
    assert frame.nrows() == 2 and frame.ncols() == 8
    assert frame.numeric_at("label", 0) == 4

    chained = dm.ImageLoader("mnist", "test", str(tmp_path)).set_batch_size(1)
    assert chained.num_batches() == 2
