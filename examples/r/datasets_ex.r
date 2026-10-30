# CMD: nix develop .#r --command Rscript examples/r/datasets_ex.r
# Tour of the example datasets: built-in tables, cached downloads, and streaming loaders.
# Uses the flat ClassName_method(obj, ...) call form (see examples/r/lm_ex.r). The download
# and stream parts fetch data once into data_home(); set DATAMUNGE_DATA_OFFLINE=1 to skip them.
library(datamunger)

cat(DataFrame_to_md(DataFrame_dataset_catalog()), "\n")

wine <- DataFrame_wine()
cat("wine:", DataFrame_nrows(wine), "wines x", DataFrame_ncols(wine), "columns\n")
titanic <- DataFrame_titanic()
cat("titanic: first passenger is", DataFrame_string_at(titanic, "Name", 0), "\n")
cat(DataFrame_to_md(DataFrame_eight_schools()), "\n")

tryCatch({
  diamonds <- DataFrame_diamonds()
  cat("diamonds:", DataFrame_nrows(diamonds), "rows\n")
  reader <- CsvReader(fetch("diamonds"), 10000L)
  total <- 0
  while (CsvReader_has_next(reader)) total <- total + DataFrame_nrows(CsvReader_next_chunk(reader))
  cat("streamed", total, "diamonds in chunks of", CsvReader_batch_size(reader), "\n")
}, error = function(e) cat("skipping diamonds:", conditionMessage(e), "\n"))

tryCatch({
  train <- ImageLoader("mnist", "train")
  ImageLoader_set_batch_size(train, 256L)
  ImageLoader_set_shuffle(train, TRUE)
  ImageLoader_set_seed(train, 1L)
  ImageLoader_set_layout(train, "flat")
  ImageLoader_set_limit(train, 2048L)
  cat("mnist: using", ImageLoader_size(train), "of", ImageLoader_split_size(train), "images\n")
  counts <- integer(10)
  while (ImageLoader_has_next(train)) {
    labels <- ImageBatch_labels_get(ImageLoader_next_batch(train))
    counts <- counts + tabulate(labels + 1L, nbins = 10)
  }
  cat("  label counts:", counts, "\n")
}, error = function(e) cat("skipping mnist:", conditionMessage(e), "\n"))
