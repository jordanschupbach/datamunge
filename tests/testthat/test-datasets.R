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
    frame <- DataFrame$dataset(name)
    expect_equal(c(frame$nrows(), frame$ncols()), shapes[[name]], info = name)
  }
  expect_equal(DataFrame$faithful()$numeric_at("waiting", 0), 79)
  expect_true(DataFrame$airquality()$is_null("Ozone", 4))
  expect_equal(DataFrame$dataset_catalog()$nrows(), 23)
  expect_true("rebot_devarm" %in% downloadable())
  expect_error(DataFrame$dataset("no_such_dataset"))
})

test_that("CSV files can be read whole or streamed in chunks", {
  path <- tempfile(fileext = ".csv")
  writeLines(c("i,name", paste0(0:24, ",row", 0:24)), path)
  expect_equal(DataFrame$read_csv(path)$nrows(), 25)
  reader <- CsvReader$new(path, 10L)
  sizes <- c()
  while (reader$has_next()) sizes <- c(sizes, reader$next_chunk()$nrows())
  expect_equal(sizes, c(10, 10, 5))
  reader$reset()
  expect_equal(reader$next_chunk()$string_at("name", 0), "row0")
})

test_that("ImageLoader streams configurable batches", {
  dir <- tempfile()
  dir.create(dir)
  write_idx(dir, "train", 10)
  write_idx(dir, "t10k", 2)
  loader <- ImageLoader$new("mnist", "train", dir)
  loader$set_batch_size(4L)
  loader$set_normalize(FALSE)
  loader$set_layout("flat")
  expect_equal(loader$num_batches(), 3)

  batch <- loader$next_batch()
  expect_equal(batch$shape, c(4, 6))
  expect_equal(batch$labels, 0:3)
  expect_equal(batch$image(1L), 10:15)

  loader$reset()
  loader$set_shuffle(TRUE)
  loader$set_seed(3L)
  seen <- c()
  while (loader$has_next()) seen <- c(seen, loader$next_batch()$indices)
  expect_equal(sort(seen), 0:9)

  frame <- DataFrame$from_image_batch(loader$take(c(4L, 0L)))
  expect_equal(frame$ncols(), 8)
  expect_equal(frame$numeric_at("label", 0), 4)
})
