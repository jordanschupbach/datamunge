# Native R NN example. TFRecord writing, streaming reads, compression and tf.Example parsing.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
# CMD: just r-example tfrecord_io
# Port of tflow's tfrecord_io. Serialize float, integer and bytes features in
# tf.Example records. Round-trip identical payloads through plain, GZIP and
# ZLIB TFRecord files, then parse required and defaulted features with a
# ParseSingleExample-style schema. Check empty records versus EOF, embedded
# NUL bytes and an empty file. Corrupted length headers, corrupted payloads
# and truncated records must fail checksum validation. Missing required
# features must fail parsing. Option verify=DIR reads plain/GZIP/ZLIB files
# written by a TF run (tflow's output/fixtures/tfrecord).
#
# The codec of examples/nn/common/tfrecord.hpp (used only by this example) is
# written out below over raw vectors: CRC-32C framing (unsigned 32-bit
# arithmetic in doubles), gzip/zlib wrapping and a minimal tf.Example protobuf
# codec. DEFLATE uses R's memCompress/memDecompress (zlib level 6) instead of
# datamunge's internal codec, which nn does not expose: payloads and plain
# files are byte-identical to the C++ run, compressed file sizes may differ.
# int64 features are R doubles, exact for |v| < 2^53 (negative values are
# not supported by this varint writer). Files go to output/nn/r/<mode>/fixtures/tfrecord.
nn_helper("checks")
nn_helper("binary")

tfr_two32 <- 4294967296

tfr_xor <- function(a, b) {
    bitwXor(a %/% 65536, b %/% 65536) * 65536 + bitwXor(a %% 65536, b %% 65536)
}

tfr_crc_table <- function(polynomial) {
    vapply(0:255, function(i) {
        c <- i
        for (k in 1:8) c <- if (c %% 2 == 1) tfr_xor(polynomial, c %/% 2) else c %/% 2
        c
    }, 0)
}

tfr_crc32c_table <- tfr_crc_table(0x82F63B78)
tfr_crc32_table <- tfr_crc_table(0xEDB88320)

# Reflected CRC-32 with the given table over a raw vector.
tfr_crc <- function(bytes, table) {
    crc <- 0xFFFFFFFF
    for (byte in as.integer(bytes)) crc <- tfr_xor(table[bitwXor(crc %% 256, byte) + 1L], crc %/% 256)
    tfr_xor(crc, 0xFFFFFFFF)
}

tfr_masked_crc <- function(bytes) {
    crc <- tfr_crc(bytes, tfr_crc32c_table)
    rotated <- crc %/% 32768 + (crc %% 32768) * 131072
    (rotated + 0xa282ead8) %% tfr_two32
}

tfr_validate_compression <- function(compression) {
    if (!compression %in% c("", "GZIP", "ZLIB"))
        stop("TFRecord compression must be empty, GZIP or ZLIB", call. = FALSE)
}

tfr_put_le <- function(value, count) as.raw((value %/% 256^(0:(count - 1L))) %% 256)

tfr_get_le <- function(bytes, at, count) sum(as.integer(bytes[at + seq_len(count)]) * 256^(0:(count - 1L)))

# A raw DEFLATE stream: R's zlib stream without its 2-byte header and adler trailer.
tfr_deflate <- function(raw_bytes) {
    z <- memCompress(raw_bytes, "gzip")
    z[3:(length(z) - 4L)]
}

tfr_gzip <- function(raw_bytes) {
    c(as.raw(c(0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255)), tfr_deflate(raw_bytes),
        tfr_put_le(tfr_crc(raw_bytes, tfr_crc32_table), 4L), tfr_put_le(length(raw_bytes) %% tfr_two32, 4L))
}

tfr_gunzip <- function(b) {
    v <- as.integer(b)
    if (length(v) < 18L || v[1] != 0x1f || v[2] != 0x8b || v[3] != 8L)
        stop("GZIP TFRecord: bad header", call. = FALSE)
    flags <- v[4]
    at <- 10L  # 0-based offset of the next byte
    if (bitwAnd(flags, 4L)) {
        if (at + 2L > length(v)) stop("GZIP TFRecord: truncated header", call. = FALSE)
        at <- at + 2L + v[at + 1L] + v[at + 2L] * 256L
    }
    for (bit in c(8L, 16L)) {
        if (!bitwAnd(flags, bit)) next
        while (at < length(v) && v[at + 1L] != 0L) at <- at + 1L
        at <- at + 1L
    }
    if (bitwAnd(flags, 2L)) at <- at + 2L
    if (at + 8L > length(v)) stop("GZIP TFRecord: truncated stream", call. = FALSE)
    trailer <- length(v) - 8L
    crc <- tfr_get_le(b, trailer, 4L)
    isize <- tfr_get_le(b, trailer + 4L, 4L)
    body <- if (trailer > at) b[(at + 1L):trailer] else raw(0)
    # Inflate the DEFLATE body through a minimal gzip member; CRC and size are checked below.
    member <- c(as.raw(c(0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 0, 255)), body, b[trailer + 1:8])
    out <- tryCatch(memDecompress(member, "gzip"),
        error = function(e) stop("GZIP TFRecord: ", conditionMessage(e), call. = FALSE))
    if (length(out) != isize) stop("GZIP TFRecord: inflated size mismatch", call. = FALSE)
    if (tfr_crc(out, tfr_crc32_table) != crc) stop("GZIP TFRecord: CRC-32 mismatch", call. = FALSE)
    out
}

# records: a list of raw vectors.
tfr_write_records <- function(path, records, compression = "") {
    tfr_validate_compression(compression)
    stream <- list()
    for (record in records) {
        length_bytes <- tfr_put_le(length(record), 8L)
        stream[[length(stream) + 1L]] <- c(length_bytes, tfr_put_le(tfr_masked_crc(length_bytes), 4L), record,
            tfr_put_le(tfr_masked_crc(record), 4L))
    }
    stream <- if (length(stream)) do.call(c, stream) else raw(0)
    if (compression == "GZIP") stream <- tfr_gzip(stream)
    if (compression == "ZLIB") stream <- memCompress(stream, "gzip")
    binary_write(path, stream)
}

tfr_read_records <- function(path, compression = "") {
    tfr_validate_compression(compression)
    stream <- binary_read(path)
    if (compression == "GZIP") stream <- tfr_gunzip(stream)
    if (compression == "ZLIB")
        stream <- tryCatch(memDecompress(stream, "gzip"),
            error = function(e) stop("ZLIB TFRecord: ", conditionMessage(e), call. = FALSE))
    records <- list()
    at <- 0
    while (at < length(stream)) {
        if (length(stream) - at < 12)
            stop("truncated TFRecord length header", call. = FALSE)
        if (tfr_masked_crc(stream[at + 1:8]) != tfr_get_le(stream, at + 8, 4L))
            stop("corrupted TFRecord length (CRC-32C mismatch)", call. = FALSE)
        size <- tfr_get_le(stream, at, 8L)
        at <- at + 12
        if (size > length(stream) - at || length(stream) - at - size < 4)
            stop("truncated TFRecord payload", call. = FALSE)
        data <- stream[at + seq_len(size)]
        if (tfr_masked_crc(data) != tfr_get_le(stream, at + size, 4L))
            stop("corrupted TFRecord payload (CRC-32C mismatch)", call. = FALSE)
        records[[length(records) + 1L]] <- data
        at <- at + size + 4
    }
    records
}

# ---- tf.Example -------------------------------------------------------------------------

tfr_bytes_kind <- 1L
tfr_float_kind <- 2L
tfr_int64_kind <- 3L

tfr_feature <- function(kind = tfr_float_kind, bytes = list(), floats = numeric(), ints = numeric())
    list(kind = kind, bytes = bytes, floats = floats, ints = ints)
tfr_float32 <- function(values) readBin(writeBin(as.numeric(values), raw(), size = 4L, endian = "little"),
    "double", n = length(values), size = 4L, endian = "little")
tfr_bytes_feature <- function(values) tfr_feature(tfr_bytes_kind, bytes = values)
tfr_float_feature <- function(values) tfr_feature(tfr_float_kind, floats = tfr_float32(values))
tfr_int64_feature <- function(values) tfr_feature(tfr_int64_kind, ints = as.numeric(values))

tfr_varint <- function(value) {
    if (value < 0 || value >= 2^53) stop("varint: value outside the exact double range", call. = FALSE)
    out <- integer()
    while (value >= 128) {
        out <- c(out, value %% 128 + 128)
        value <- value %/% 128
    }
    as.raw(c(out, value))
}

tfr_delimited <- function(field_number, payload) c(tfr_varint(field_number * 8 + 2), tfr_varint(length(payload)), payload)

# A protobuf reader over a raw vector; at is the 0-based cursor.
tfr_reader <- function(s) {
    env <- environment()
    at <- 0
    done <- function() at >= length(s)
    varint <- function() {
        value <- 0
        for (shift in seq(0, 63, by = 7)) {
            if (at >= length(s)) stop("tf.Example: truncated varint", call. = FALSE)
            b <- as.integer(s[at + 1])
            at <<- at + 1
            value <- value + (b %% 128) * 2^shift
            if (b < 128) return(value)
        }
        stop("tf.Example: varint too long", call. = FALSE)
    }
    bytes <- function() {
        n <- varint()
        if (n > length(s) - at) stop("tf.Example: truncated field", call. = FALSE)
        out <- s[at + seq_len(n)]
        at <<- at + n
        out
    }
    skip <- function(wire) {
        if (wire == 0) varint()
        else if (wire == 1) at <<- at + 8
        else if (wire == 2) bytes()
        else if (wire == 5) at <<- at + 4
        else stop("tf.Example: unsupported wire type", call. = FALSE)
        if (at > length(s)) stop("tf.Example: truncated field", call. = FALSE)
    }
    env
}

tfr_serialize_example <- function(example) {
    features <- raw(0)
    for (key in sort(names(example), method = "radix")) {
        feature <- example[[key]]
        values <- raw(0)
        if (feature$kind == tfr_bytes_kind) {
            for (item in feature$bytes) values <- c(values, tfr_delimited(1, item))
        } else if (feature$kind == tfr_float_kind) {
            if (length(feature$floats))
                values <- tfr_delimited(1, writeBin(feature$floats, raw(), size = 4L, endian = "little"))
        } else if (length(feature$ints)) {
            values <- tfr_delimited(1, do.call(c, lapply(feature$ints, tfr_varint)))
        }
        value <- tfr_delimited(feature$kind, values)
        entry <- c(tfr_delimited(1, charToRaw(key)), tfr_delimited(2, value))
        features <- c(features, tfr_delimited(1, entry))
    }
    tfr_delimited(1, features)
}

tfr_parse_feature <- function(s) {
    f <- tfr_feature()
    r <- tfr_reader(s)
    while (!r$done()) {
        tag <- r$varint()
        number <- tag %/% 8
        wire <- tag %% 8
        if (number < 1 || number > 3 || wire != 2) {
            r$skip(wire)
            next
        }
        f <- tfr_feature(number)
        values <- r$bytes()
        lr <- tfr_reader(values)
        while (!lr$done()) {
            ltag <- lr$varint()
            lfield <- ltag %/% 8
            lwire <- ltag %% 8
            if (lfield != 1) {
                lr$skip(lwire)
                next
            }
            if (number == tfr_bytes_kind) {
                if (lwire != 2) stop("tf.Example: bad bytes_list", call. = FALSE)
                f$bytes[[length(f$bytes) + 1L]] <- lr$bytes()
            } else if (number == tfr_float_kind) {
                if (lwire == 2) {
                    packed <- lr$bytes()
                    if (length(packed) %% 4) stop("tf.Example: bad packed float_list", call. = FALSE)
                    f$floats <- c(f$floats, readBin(packed, "double", n = length(packed) / 4, size = 4L,
                        endian = "little"))
                } else if (lwire == 5) {
                    if (lr$at + 4 > length(values)) stop("tf.Example: truncated float", call. = FALSE)
                    f$floats <- c(f$floats, readBin(values[lr$at + 1:4], "double", size = 4L, endian = "little"))
                    lr$at <- lr$at + 4
                } else {
                    stop("tf.Example: bad float_list", call. = FALSE)
                }
            } else {
                if (lwire == 2) {
                    pr <- tfr_reader(lr$bytes())
                    while (!pr$done()) f$ints <- c(f$ints, pr$varint())
                } else if (lwire == 0) {
                    f$ints <- c(f$ints, lr$varint())
                } else {
                    stop("tf.Example: bad int64_list", call. = FALSE)
                }
            }
        }
    }
    f
}

# A serialized tf.Example as a named list of features.
tfr_parse_example <- function(serialized) {
    example <- list()
    r <- tfr_reader(serialized)
    while (!r$done()) {
        tag <- r$varint()
        if (tag %/% 8 != 1 || tag %% 8 != 2) {
            r$skip(tag %% 8)
            next
        }
        fr <- tfr_reader(r$bytes())
        while (!fr$done()) {
            ftag <- fr$varint()
            if (ftag %/% 8 != 1 || ftag %% 8 != 2) {
                fr$skip(ftag %% 8)
                next
            }
            er <- tfr_reader(fr$bytes())
            key <- ""
            value <- tfr_feature()
            while (!er$done()) {
                etag <- er$varint()
                if (etag %/% 8 == 1 && etag %% 8 == 2) key <- rawToChar(er$bytes())
                else if (etag %/% 8 == 2 && etag %% 8 == 2) value <- tfr_parse_feature(er$bytes())
                else er$skip(etag %% 8)
            }
            example[[key]] <- value  # later entries win, as in protobuf maps
        }
    }
    example
}

tfr_dense_spec <- function(key, kind, size, default = NULL) list(key = key, kind = kind, size = size, default = default)

# TF ParseSingleExample for dense features: one feature per spec, in spec order.
tfr_parse_single_example <- function(serialized, specs) {
    example <- tfr_parse_example(serialized)
    lapply(specs, function(spec) {
        if (!spec$key %in% names(example)) {
            if (is.null(spec$default))
                stop(sprintf("Feature: %s (data type) is required but could not be found.", spec$key), call. = FALSE)
            return(spec$default)
        }
        f <- example[[spec$key]]
        n <- length(if (f$kind == tfr_bytes_kind) f$bytes else if (f$kind == tfr_float_kind) f$floats else f$ints)
        if (f$kind != spec$kind) stop(sprintf("Feature: %s has the wrong data type", spec$key), call. = FALSE)
        if (n != spec$size)
            stop(sprintf("Feature: %s has %d values, expected %d", spec$key, n, spec$size), call. = FALSE)
        f
    })
}

# ---- the example ------------------------------------------------------------------------

tfr_examples <- function() {
    lapply(0:2, function(i) {
        example <- list(x = tfr_float_feature(i + 0:2 * 0.25), label = tfr_int64_feature(i %% 2),
            id = tfr_bytes_feature(list(charToRaw(sprintf("sample-%d", i)))))
        if (i != 1) example$weight <- tfr_float_feature(2)
        tfr_serialize_example(example)
    })
}

tfr_schema <- list(
    tfr_dense_spec("x", tfr_float_kind, 3),
    tfr_dense_spec("label", tfr_int64_kind, 1),
    tfr_dense_spec("id", tfr_bytes_kind, 1),
    tfr_dense_spec("weight", tfr_float_kind, 1, tfr_float_feature(1)))

tfr_check_parsed <- function(records) {
    nn_require(length(records) == 3L, "three Example records")
    for (i in 0:2) {
        parsed <- tfr_parse_single_example(records[[i + 1L]], tfr_schema)
        x <- nn$from_vector(parsed[[1]]$floats, 3)
        label <- nn$reshape(nn$from_vector_int64(parsed[[2]]$ints, 1), numeric())
        xv <- x$values()
        for (j in 0:2) nn_near(xv[j + 1L], i + j * 0.25, 0, "float feature")
        nn_require(label$item() == i %% 2, "integer feature")
        nn_require(rawToChar(parsed[[3]]$bytes[[1]]) == sprintf("sample-%d", i), "bytes feature")
        nn_near(parsed[[4]]$floats[1], if (i == 1) 1 else 2, 0, "defaulted feature")
    }
}

tfr_parse_schema <- function(records) {
    tfr_check_parsed(records)
    # A valid empty Example proto lacks every required feature.
    nn_throws(tfr_parse_single_example(raw(0), tfr_schema), "missing required feature must fail")
    cat("  parsed float[3], int64 label, bytes id and defaulted weight\n")
    cat("  weights=(2,1,2); missing required features rejected\n")
}

tfr_round_trip <- function() {
    root <- nn_output_dir("fixtures/tfrecord")
    records <- tfr_examples()
    for (compression in c("", "GZIP", "ZLIB")) {
        label <- if (nzchar(compression)) compression else "plain"
        path <- file.path(root, paste0(label, ".tfrecord"))
        tfr_write_records(path, records, compression)
        restored <- tfr_read_records(path, compression)
        nn_require(identical(restored, records), "exact TFRecord payload round trip")
        cat(sprintf("  %s: %d exact records, file bytes=%.0f\n", label, length(restored), file.size(path)))
    }
    payloads <- list(raw(0), as.raw(c(0x61, 0, 0x62)), charToRaw("last"))
    tfr_write_records(file.path(root, "bytes.tfrecord"), payloads)
    nn_require(identical(tfr_read_records(file.path(root, "bytes.tfrecord")), payloads),
        "empty and binary records distinct from EOF")
    tfr_write_records(file.path(root, "empty.tfrecord"), list())
    nn_require(!length(tfr_read_records(file.path(root, "empty.tfrecord"))), "empty TFRecord file")
    original <- binary_read(file.path(root, "plain.tfrecord"))
    failures <- 0L
    for (mode in 0:2) {
        damaged <- original
        if (mode == 0) damaged[1] <- xor(damaged[1], as.raw(1))
        else if (mode == 1) damaged[13] <- xor(damaged[13], as.raw(1))
        else damaged <- damaged[-length(damaged)]
        path <- file.path(root, sprintf("damaged-%d.tfrecord", mode))
        binary_write(path, damaged)
        rejected <- tryCatch({
            tfr_read_records(path)
            FALSE
        }, error = function(e) TRUE)
        nn_require(rejected, "corrupt/truncated TFRecord must fail")
        failures <- failures + 1L
    }
    rejected <- tryCatch({
        tfr_read_records(file.path(root, "plain.tfrecord"), "unknown")
        FALSE
    }, error = function(e) grepl("compression must be", conditionMessage(e), fixed = TRUE))
    nn_require(rejected, "unsupported compression cannot silently fall back")
    cat(sprintf("  empty/binary payloads and EOF passed; %d corrupt/truncated streams rejected\n", failures))
    tfr_parse_schema(records)
}

tfr_verify <- function(directory) {
    for (compression in c("", "GZIP", "ZLIB")) {
        path <- file.path(directory, paste0(if (nzchar(compression)) compression else "plain", ".tfrecord"))
        tfr_check_parsed(tfr_read_records(path, compression))
        cat(sprintf("  TF-written %s (%.0f bytes): CRCs valid, every feature identical\n", basename(path), file.size(path)))
    }
}

main <- function(arguments = commandArgs(TRUE)) {
    tryCatch({
        nn$set_num_threads(2L)
        verify <- ""
        for (arg in arguments) {
            if (startsWith(arg, "verify=") && nchar(arg) > 7L) verify <- substring(arg, 8L)
            else stop("expected verify=DIR", call. = FALSE)
        }
        cat("== TFRecord round trips and schema parsing\n")
        tfr_round_trip()
        if (nzchar(verify)) tfr_verify(verify)
        cat("All TFRecord checks passed.\n")
        0L
    }, error = function(e) {
        message(conditionMessage(e))
        1L
    })
}

if (sys.nframe() == 0L) quit(status = main())
