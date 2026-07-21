# A minimal, test-only ESRI shapefile (.shp/.dbf) writer -- see tests/cpp/gis_shapefile_tests.cpp
# for the C++ equivalent. Deliberately independent code (not shared with the reader) so a
# reader bug can't be masked by a matching writer bug.

.gis_write_u16le <- function(con, v) writeBin(as.integer(v), con, size = 2, endian = "little")
.gis_write_u32le <- function(con, v) writeBin(as.integer(v), con, size = 4, endian = "little")
.gis_write_f64le <- function(con, v) writeBin(as.double(v), con, size = 8, endian = "little")
.gis_write_i32be <- function(con, v) writeBin(as.integer(v), con, size = 4, endian = "big")

.gis_write_shp <- function(path, rings_list) {
  content_for_shape <- function(rings) {
    con <- rawConnection(raw(0), "wb")
    writeBin(as.integer(5), con, size = 4, endian = "little") # Polygon
    all_pts <- do.call(rbind, rings)
    .gis_write_f64le(con, min(all_pts[, 1]))
    .gis_write_f64le(con, min(all_pts[, 2]))
    .gis_write_f64le(con, max(all_pts[, 1]))
    .gis_write_f64le(con, max(all_pts[, 2]))
    .gis_write_u32le(con, length(rings))
    .gis_write_u32le(con, nrow(all_pts))
    starts <- cumsum(c(0, sapply(rings, nrow)))[1:length(rings)]
    for (s in starts) .gis_write_u32le(con, s)
    for (ring in rings) {
      for (i in seq_len(nrow(ring))) {
        .gis_write_f64le(con, ring[i, 1])
        .gis_write_f64le(con, ring[i, 2])
      }
    }
    bytes <- rawConnectionValue(con)
    close(con)
    bytes
  }

  contents <- lapply(rings_list, content_for_shape)
  total_words <- sum(sapply(contents, function(c) 4 + length(c) / 2))

  con <- file(path, "wb")
  .gis_write_i32be(con, 9994)
  for (i in 1:5) .gis_write_i32be(con, 0)
  .gis_write_i32be(con, 50 + total_words)
  .gis_write_u32le(con, 1000)
  .gis_write_u32le(con, 5)
  .gis_write_f64le(con, -100)
  .gis_write_f64le(con, -100)
  .gis_write_f64le(con, 100)
  .gis_write_f64le(con, 100)
  for (i in 1:4) .gis_write_f64le(con, 0)
  for (i in seq_along(contents)) {
    .gis_write_i32be(con, i)
    .gis_write_i32be(con, length(contents[[i]]) / 2)
    writeBin(contents[[i]], con)
  }
  close(con)
}

.gis_write_dbf <- function(path, names_col, pops_col) {
  n <- length(names_col)
  header_size <- 32 + 2 * 32 + 1
  record_size <- 1 + 12 + 8
  con <- file(path, "wb")
  writeBin(as.integer(3), con, size = 1)
  writeBin(as.integer(c(0, 0, 0)), con, size = 1)
  .gis_write_u32le(con, n)
  .gis_write_u16le(con, header_size)
  .gis_write_u16le(con, record_size)
  writeBin(raw(20), con)

  write_field <- function(name, type, length) {
    nm <- charToRaw(name)
    writeBin(nm, con)
    writeBin(raw(11 - length(nm)), con)
    writeBin(charToRaw(type), con)
    writeBin(raw(4), con)
    writeBin(as.integer(length), con, size = 1)
    writeBin(raw(15), con)
  }
  write_field("NAME", "C", 12)
  write_field("POP", "N", 8)
  writeBin(as.raw(0x0D), con)

  pad <- function(s, len) formatC(substr(s, 1, len), width = -len)
  for (i in seq_len(n)) {
    writeBin(charToRaw(" "), con)
    writeBin(charToRaw(pad(names_col[i], 12)), con)
    writeBin(charToRaw(pad(pops_col[i], 8)), con)
  }
  close(con)
}

test_that("ShapeLayer reads a shapefile and draws a map via idiomatic $ chaining", {
  base <- file.path(tempdir(), "test_gis_counties")
  outer <- matrix(c(0, 0, 0, 10, 10, 10, 10, 0, 0, 0), ncol = 2, byrow = TRUE)
  hole <- matrix(c(3, 3, 4, 3, 4, 4, 3, 4, 3, 3), ncol = 2, byrow = TRUE)
  .gis_write_shp(paste0(base, ".shp"), list(list(outer, hole)))
  .gis_write_dbf(paste0(base, ".dbf"), c("Lakeside"), c("48231"))

  layer <- ShapeLayer_read(base)
  expect_equal(layer$size(), 1L)
  expect_equal(layer$shape_type(), "polygon")
  expect_equal(layer$num_parts(0), 2L)
  expect_equal(layer$part_x(0, 0), c(0, 0, 10, 10, 0))
  expect_equal(layer$part_x(0, 1), c(3, 4, 4, 3, 3))

  attrs <- layer$attributes()
  expect_equal(attrs$nrows(), 1L)
  expect_equal(attrs$to_string(), DataFrame_to_string(attrs))

  svg_path <- file.path(tempdir(), "test_gis_counties_map.svg")
  layer$plot()$save_svg(svg_path)
  svg_text <- paste(readLines(svg_path), collapse = "\n")
  expect_match(svg_text, "<svg")
  expect_match(svg_text, "polygon")
})

test_that("ShapeLayer_read accepts a path with or without the .shp extension", {
  base <- file.path(tempdir(), "test_gis_ext")
  .gis_write_shp(paste0(base, ".shp"), list(list(matrix(c(0, 0, 0, 1, 1, 1, 1, 0, 0, 0), ncol = 2, byrow = TRUE))))
  .gis_write_dbf(paste0(base, ".dbf"), c("A"), c("1"))

  layer_no_ext <- ShapeLayer_read(base)
  layer_with_ext <- ShapeLayer_read(paste0(base, ".shp"))
  expect_equal(layer_no_ext$size(), layer_with_ext$size())
})
