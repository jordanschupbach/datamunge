# Bounded binary file reads and checked dimension arithmetic (examples/nn/common/binary.hpp).
# Bytes are R raw vectors. binary_to_tensor turns bytes into an int32 tensor
# holding 0..255 (nn has no uint8 dtype) by framing them as a |u1 .npy array
# for nn$load_npy.

binary_size_max <- 2^64 - 1

binary_read <- function(path, max_bytes = 512 * 1024 * 1024) {
    info <- file.info(path)
    if (is.na(info$size) || isTRUE(info$isdir))
        stop("cannot open ", path, call. = FALSE)
    if (info$size > max_bytes)
        stop("file exceeds byte limit: ", path, call. = FALSE)
    connection <- file(path, "rb")
    on.exit(close(connection))
    result <- readBin(connection, "raw", n = info$size)
    if (length(result) != info$size)
        stop("short read: ", path, call. = FALSE)
    if (length(readBin(connection, "raw", n = 1L)))
        stop("file grew while reading: ", path, call. = FALSE)
    result
}

binary_write <- function(path, bytes) {
    connection <- tryCatch(file(path, "wb"), error = function(e) stop("cannot write ", path, call. = FALSE))
    on.exit(close(connection))
    writeBin(as.raw(bytes), connection)
    invisible(path)
}

binary_product <- function(a, b) {
    if (b != 0 && a > binary_size_max / b)
        stop("dataset dimensions overflow", call. = FALSE)
    a * b
}

binary_big_endian32 <- function(bytes, offset) {
    if (offset > length(bytes) || length(bytes) - offset < 4)
        stop("truncated integer", call. = FALSE)
    sum(as.integer(bytes[offset + 1:4]) * c(16777216, 65536, 256, 1))
}

# Bytes as an int32 tensor of the given shape (values 0..255).
binary_to_tensor <- function(bytes, shape) {
    count <- 1
    for (dimension in shape) count <- binary_product(count, dimension)
    if (count != length(bytes))
        stop("byte count does not fill the tensor shape", call. = FALSE)
    header <- sprintf("{'descr': '|u1', 'fortran_order': False, 'shape': (%s), }",
        paste0(sprintf("%.0f, ", shape), collapse = ""))
    header <- paste0(header, strrep(" ", 63 - (10 + nchar(header)) %% 64), "\n")
    path <- tempfile(fileext = ".npy")
    on.exit(unlink(path))
    length_bytes <- as.raw(c(nchar(header) %% 256, nchar(header) %/% 256))
    binary_write(path, c(as.raw(c(0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0)), length_bytes,
        charToRaw(header), bytes))
    nn$load_npy(path)
}
