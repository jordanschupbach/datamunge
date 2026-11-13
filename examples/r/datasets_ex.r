# CMD: nix develop .#r --command env R_LIBS_USER=build/r/library Rscript examples/r/datasets_ex.r
# REPL: nix develop .#r --command env R_LIBS_USER=build/r/library R --no-save --quiet
# Tour of the example datasets: built-in tables, cached downloads, and streaming loaders.
# The download and stream parts fetch data once into data_home(); set DATAMUNGE_DATA_OFFLINE=1 to skip them.
library(datamunger)

cat(data$catalog()$to_md(), "\n")

wine <- data$wine()
cat("wine:", wine$nrows(), "wines x", wine$ncols(), "columns\n")
titanic <- data$titanic()
cat("titanic: first passenger is", titanic$string_at("Name", 0), "\n")
cat(data$eight_schools()$to_md(), "\n")

tryCatch({
  diamonds <- data$diamonds()
  cat("diamonds:", diamonds$nrows(), "rows\n")
  reader <- CsvReader$new(fetch("diamonds"), 10000L)
  total <- 0
  while (reader$has_next()) total <- total + reader$next_chunk()$nrows()
  cat("streamed", total, "diamonds in chunks of", reader$batch_size(), "\n")
}, error = function(e) cat("skipping diamonds:", conditionMessage(e), "\n"))

tryCatch({
  train <- ImageLoader$new("mnist", "train")
  train$set_batch_size(256L)
  train$set_shuffle(TRUE)
  train$set_seed(1L)
  train$set_layout("flat")
  train$set_limit(2048L)
  cat("mnist: using", train$size(), "of", train$split_size(), "images\n")
  counts <- integer(10)
  while (train$has_next()) {
    labels <- train$next_batch()$labels
    counts <- counts + tabulate(labels + 1L, nbins = 10)
  }
  cat("  label counts:", counts, "\n")
}, error = function(e) cat("skipping mnist:", conditionMessage(e), "\n"))
