# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/datasets_ex.py
# Tour of the example datasets: built-in tables, cached downloads, and streaming loaders.
# The built-in part runs offline; the download/stream parts fetch data once into
# dm.data_home() (set DATAMUNGE_DATA_OFFLINE=1 to skip them).
from pydatamunge import datamunge as dm

print(dm.DataFrame.dataset_catalog().to_md())

wine = dm.DataFrame.wine()
print(f"wine: {wine.nrows()} wines x {wine.ncols()} columns")

titanic = dm.DataFrame.titanic()
print(f"titanic: {titanic.nrows()} passengers; first is {titanic.string_at('Name', 0)}")

print(dm.DataFrame.eight_schools().to_md())

try:
    diamonds = dm.DataFrame.diamonds()
    print(f"diamonds: {diamonds.nrows()} rows, columns {list(diamonds.columns())}")
    # Stream the cached CSV 10000 rows at a time.
    reader = dm.CsvReader(dm.fetch("diamonds"), 10000)
    total = sum(chunk.nrows() for chunk in reader)
    print(f"streamed {total} diamonds in chunks of {reader.batch_size()}")
except RuntimeError as error:
    print(f"skipping diamonds: {error}")

try:
    train = dm.ImageLoader("mnist", "train")
    train.set_batch_size(256).set_shuffle(True).set_seed(1).set_layout("flat").set_limit(2048)
    print(f"mnist: using {train.size()} of {train.split_size()} images, {len(train)} batches per epoch")
    for epoch in range(2):
        counts = [0] * 10
        for batch in train:
            for label in batch.labels:
                counts[label] += 1
        print(f"  epoch {train.epoch()} label counts: {counts}")
        train.reset()  # next epoch, reshuffled
    first = dm.DataFrame.from_image_batch(train.take([0, 1, 2]))
    print(f"  three images as a table: {first.nrows()} x {first.ncols()}")
except RuntimeError as error:
    print(f"skipping mnist: {error}")
