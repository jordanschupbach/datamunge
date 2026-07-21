# Demonstrates ShapeLayer: reading a shapefile (.shp geometry + .dbf attributes) and drawing it
# as a map. Real .shp/.dbf files are large binary bundles that don't belong in this repo, so
# this example first writes a tiny synthetic shapefile by hand (two "counties": one plain
# square, one with a lake-shaped hole) using the same ESRI byte layout ShapeLayer_read() expects
# -- see tests/testthat/test-gis.R for the shared writer helpers this is adapted from.
library(datamunger)

write_u16le <- function(con, v) writeBin(as.integer(v), con, size = 2, endian = "little")
write_u32le <- function(con, v) writeBin(as.integer(v), con, size = 4, endian = "little")
write_f64le <- function(con, v) writeBin(as.double(v), con, size = 8, endian = "little")
write_i32be <- function(con, v) writeBin(as.integer(v), con, size = 4, endian = "big")

write_counties_shp <- function(path, shapes) {
  # shapes: list of shapes, each a list of rings, each ring a matrix of x,y rows (closed)
  content_for_shape <- function(rings) {
    con <- rawConnection(raw(0), "wb")
    writeBin(as.integer(5), con, size = 4, endian = "little") # Polygon
    all_pts <- do.call(rbind, rings)
    write_f64le(con, min(all_pts[, 1]))
    write_f64le(con, min(all_pts[, 2]))
    write_f64le(con, max(all_pts[, 1]))
    write_f64le(con, max(all_pts[, 2]))
    write_u32le(con, length(rings))
    write_u32le(con, nrow(all_pts))
    starts <- cumsum(c(0, sapply(rings, nrow)))[1:length(rings)]
    for (s in starts) write_u32le(con, s)
    for (ring in rings) {
      for (i in seq_len(nrow(ring))) {
        write_f64le(con, ring[i, 1])
        write_f64le(con, ring[i, 2])
      }
    }
    bytes <- rawConnectionValue(con)
    close(con)
    bytes
  }

  contents <- lapply(shapes, content_for_shape)
  bounds <- do.call(rbind, lapply(shapes, function(s) do.call(rbind, s)))
  total_words <- sum(sapply(contents, function(c) 4 + length(c) / 2))

  con <- file(path, "wb")
  write_i32be(con, 9994)
  for (i in 1:5) write_i32be(con, 0)
  write_i32be(con, 50 + total_words)
  write_u32le(con, 1000)
  write_u32le(con, 5)
  write_f64le(con, min(bounds[, 1])); write_f64le(con, min(bounds[, 2]))
  write_f64le(con, max(bounds[, 1])); write_f64le(con, max(bounds[, 2]))
  for (i in 1:4) write_f64le(con, 0)
  for (i in seq_along(contents)) {
    write_i32be(con, i)
    write_i32be(con, length(contents[[i]]) / 2)
    writeBin(contents[[i]], con)
  }
  close(con)
}

write_counties_dbf <- function(path, names_col, pops_col) {
  n <- length(names_col)
  header_size <- 32 + 2 * 32 + 1
  record_size <- 1 + 12 + 8
  con <- file(path, "wb")
  writeBin(as.integer(3), con, size = 1)
  writeBin(as.integer(c(0, 0, 0)), con, size = 1)
  write_u32le(con, n)
  write_u16le(con, header_size)
  write_u16le(con, record_size)
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

base <- file.path(tempdir(), "datamunger_gis_ex_counties")

# Per the ESRI winding convention, outer rings are clockwise, holes counterclockwise.
plain <- list(matrix(c(10, 0, 10, 10, 20, 10, 20, 0, 10, 0), ncol = 2, byrow = TRUE))
with_lake <- list(
  matrix(c(0, 0, 0, 10, 10, 10, 10, 0, 0, 0), ncol = 2, byrow = TRUE),
  matrix(c(3, 3, 4, 3, 4, 4, 3, 4, 3, 3), ncol = 2, byrow = TRUE)
)

write_counties_shp(paste0(base, ".shp"), list(with_lake, plain))
write_counties_dbf(paste0(base, ".dbf"), c("Lakeside", "Plainview"), c("48231", "19876"))

counties <- ShapeLayer_read(base)

cat("shapes =", counties$size(), ", shape_type =", counties$shape_type(), "\n")
cat("bounds =", counties$bounds(), "\n\n")

cat("attributes\n")
cat(counties$attributes()$to_string(), "\n\n")

for (i in 0:(counties$size() - 1)) {
  cat(DataFrame_string_at(counties$attributes(), "NAME", i), ":", counties$num_parts(i), "ring(s)\n")
}

svg_path <- file.path(tempdir(), "datamunger_gis_ex_map.svg")
counties$plot()$save_svg(svg_path)
cat("\nmap saved to", svg_path, "\n")
