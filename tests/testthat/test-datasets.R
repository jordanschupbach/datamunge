write_idx <- function(dir, prefix, n) {
  # n images of 2x3 pixels: pixel p of image i is 10*i + p; label i %% 10.
  img <- file(file.path(dir, paste0(prefix, "-images-idx3-ubyte")), "wb")
  writeBin(as.integer(c(0x803, n, 2, 3)), img, size = 4, endian = "big")
  writeBin(as.raw(unlist(lapply(seq_len(n) - 1, function(i) 10 * i + 0:5))), img)
  close(img)
  lab <- file(file.path(dir, paste0(prefix, "-labels-idx1-ubyte")), "wb")
  writeBin(as.integer(c(0x801, n)), lab, size = 4, endian = "big")
  writeBin(as.raw((seq_len(n) - 1) %% 10), lab)
  close(lab)
}

test_that("built-in datasets load with their documented shapes", {
  shapes <- list(wine = c(178, 14), titanic = c(891, 12), eight_schools = c(8, 3),
                 faithful = c(272, 2), mtcars = c(32, 12), airquality = c(153, 6),
                 breast_cancer = c(569, 31), diabetes = c(442, 11), digits = c(1797, 65))
  for (name in names(shapes)) {
    frame <- DataFrame_dataset(name)
    expect_equal(c(DataFrame_nrows(frame), DataFrame_ncols(frame)), shapes[[name]], info = name)
  }
  expect_equal(DataFrame_numeric_at(DataFrame_faithful(), "waiting", 0), 79)
  expect_true(DataFrame_is_null(DataFrame_airquality(), "Ozone", 4))
  expect_equal(DataFrame_nrows(DataFrame_dataset_catalog()), 22)
  expect_error(DataFrame_dataset("no_such_dataset"))
})

test_that("CSV files can be read whole or streamed in chunks", {
  path <- tempfile(fileext = ".csv")
  writeLines(c("i,name", paste0(0:24, ",row", 0:24)), path)
  expect_equal(DataFrame_nrows(DataFrame_read_csv(path)), 25)
  reader <- CsvReader(path, 10L)
  sizes <- c()
  while (CsvReader_has_next(reader)) sizes <- c(sizes, DataFrame_nrows(CsvReader_next_chunk(reader)))
  expect_equal(sizes, c(10, 10, 5))
  CsvReader_reset(reader)
  expect_equal(DataFrame_string_at(CsvReader_next_chunk(reader), "name", 0), "row0")
})

test_that("ImageLoader streams configurable batches", {
  dir <- tempfile()
  dir.create(dir)
  write_idx(dir, "train", 10)
  write_idx(dir, "t10k", 2)
  loader <- ImageLoader("mnist", "train", dir)
  ImageLoader_set_batch_size(loader, 4L)
  ImageLoader_set_normalize(loader, FALSE)
  ImageLoader_set_layout(loader, "flat")
  expect_equal(ImageLoader_num_batches(loader), 3)

  batch <- ImageLoader_next_batch(loader)
  expect_equal(ImageBatch_shape_get(batch), c(4, 6))
  expect_equal(ImageBatch_labels_get(batch), 0:3)
  expect_equal(ImageBatch_image(batch, 1L), 10:15)

  ImageLoader_reset(loader)
  ImageLoader_set_shuffle(loader, TRUE)
  ImageLoader_set_seed(loader, 3L)
  seen <- c()
  while (ImageLoader_has_next(loader)) seen <- c(seen, ImageBatch_indices_get(ImageLoader_next_batch(loader)))
  expect_equal(sort(seen), 0:9)

  frame <- DataFrame_from_image_batch(ImageLoader_take(loader, c(4L, 0L)))
  expect_equal(DataFrame_ncols(frame), 8)
  expect_equal(DataFrame_numeric_at(frame, "label", 0), 4)
})
