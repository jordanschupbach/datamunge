# Native R NN example. CSV parsing and training-only numeric/categorical feature columns.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
# CMD: just r-example csv_features
# Port of tflow's csv_features. Read a table containing quoted commas, escaped
# quotes, embedded newlines, CRLF, missing values and held-out categories.
# Fit numeric means/scales and sorted categorical vocabularies on four
# training rows, then apply that fixed transformation to two validation rows.
# Numeric missingness and categorical missing/unknown values have separate
# indicator columns. Column lookup uses header names, so a reordered input
# table produces the same model features. The dense float32 matrix feeds an
# eager nn$matmul linear prediction.
#
# The parser and feature columns (examples/nn/common/csv.hpp, used only by this
# example) are written out below. R strings cannot hold NUL, so the parser's
# NUL rejection has nothing to check here. Sorting uses byte order (radix), as
# std::set<std::string>. The fixture goes to output/nn/r/<mode>/fixtures/tabular.
nn_helper("checks")
nn_helper("binary")

csv_number_pattern <- "^[+-]?([0-9]+\\.?[0-9]*|\\.[0-9]+)([eE][+-]?[0-9]+)?$"
csv_float_max <- 3.4028234663852886e38

# ---- CSV table (common/csv.hpp) ---------------------------------------------------------

csv_parse <- function(text, delimiter = ",") {
    if (delimiter %in% c("\"", "\n", "\r"))
        stop("invalid CSV delimiter", call. = FALSE)
    chars <- if (nzchar(text)) strsplit(enc2utf8(text), "")[[1]] else character()
    if (length(chars) && chars[1] == "﻿")
        chars <- chars[-1]
    records <- list()
    row <- character()
    field <- character()
    quoted <- closed <- started <- FALSE
    end_field <- function() {
        row[[length(row) + 1L]] <<- paste0(field, collapse = "")
        field <<- character()
        closed <<- started <<- FALSE
    }
    n <- length(chars)
    i <- 1L
    while (i <= n) {
        c <- chars[i]
        if (quoted) {
            if (c == "\"") {
                if (i + 1L <= n && chars[i + 1L] == "\"") {
                    field <- c(field, "\"")
                    i <- i + 1L
                } else {
                    quoted <- FALSE
                    closed <- TRUE
                }
            } else {
                field <- c(field, c)
            }
        } else if (c == delimiter) {
            end_field()
        } else if (c == "\n" || c == "\r") {
            if (c == "\r") {
                if (i == n || chars[i + 1L] != "\n")
                    stop("bare CR outside quoted field", call. = FALSE)
                i <- i + 1L
            }
            end_field()
            records[[length(records) + 1L]] <- row
            row <- character()
        } else if (c == "\"" && !started && !closed) {
            quoted <- started <- TRUE
        } else {
            if (closed || c == "\"")
                stop("invalid CSV quote placement", call. = FALSE)
            field <- c(field, c)
            started <- TRUE
        }
        i <- i + 1L
    }
    if (quoted)
        stop("unterminated CSV quote", call. = FALSE)
    if (started || closed || length(row)) {
        end_field()
        records[[length(records) + 1L]] <- row
    }
    if (!length(records))
        stop("CSV requires a header", call. = FALSE)
    header <- records[[1]]
    if (any(!nzchar(header)) || anyDuplicated(header))
        stop("empty or duplicate CSV header", call. = FALSE)
    rows <- list()
    for (i in seq_along(records)[-1]) {
        if (length(records[[i]]) != length(header))
            stop(sprintf("CSV record width mismatch at record %d", i), call. = FALSE)
        rows[[length(rows) + 1L]] <- records[[i]]
    }
    list(header = header, rows = rows)
}

csv_column <- function(table, name) {
    column <- match(name, table$header)
    if (is.na(column))
        stop("missing CSV column: ", name, call. = FALSE)
    column
}

# 0-based row, 1-based column.
csv_at <- function(table, row, column) table$rows[[row + 1L]][[column]]

csv_load <- function(path, delimiter = ",") {
    bytes <- binary_read(path, 128 * 1024 * 1024)
    text <- rawToChar(bytes)
    Encoding(text) <- "UTF-8"
    csv_parse(text, delimiter)
}

# NULL for an empty (ASCII-blank) cell, else a strict finite decimal value.
csv_numeric <- function(text) {
    text <- gsub("^[ \t\r\n]+|[ \t\r\n]+$", "", text)
    if (!nzchar(text))
        return(NULL)
    if (substr(text, 1L, 1L) == "+") {
        text <- substring(text, 2L)
        if (!nzchar(text) || substr(text, 1L, 1L) %in% c("+", "-"))
            stop("invalid numeric sign", call. = FALSE)
    }
    if (!grepl(csv_number_pattern, text))
        stop("invalid finite numeric field: ", text, call. = FALSE)
    value <- as.numeric(text)
    if (!is.finite(value))
        stop("invalid finite numeric field: ", text, call. = FALSE)
    value
}

# ---- fitted feature columns ---------------------------------------------------------------

csv_fit <- function(table, training_rows, specs) {
    if (!length(training_rows) || !length(specs))
        stop("feature fitting needs training rows and columns", call. = FALSE)
    if (anyDuplicated(names(specs)))
        stop("duplicate feature column", call. = FALSE)
    features <- list()
    feature_names <- character()
    for (name in names(specs)) {
        kind <- specs[[name]]
        column <- csv_column(table, name)
        fit <- list(name = name, kind = kind, mean = 0, scale = 1, vocabulary = character())
        if (kind == "numeric") {
            count <- 0
            m2 <- 0
            for (row in training_rows) {
                value <- csv_numeric(csv_at(table, row, column))
                if (!is.null(value)) {
                    count <- count + 1
                    delta <- value - fit$mean
                    fit$mean <- fit$mean + delta / count
                    m2 <- m2 + delta * (value - fit$mean)
                }
            }
            if (count > 0 && m2 > 0)
                fit$scale <- sqrt(m2 / count)
            if (!is.finite(fit$mean) || !is.finite(fit$scale))
                stop("numeric statistics overflow", call. = FALSE)
            feature_names <- c(feature_names, paste0(name, ":value"), paste0(name, ":missing"))
        } else if (kind == "categorical") {
            values <- vapply(training_rows, function(row) csv_at(table, row, column), "")
            values <- sort(unique(values[nzchar(values)]), method = "radix")
            fit$vocabulary <- values
            feature_names <- c(feature_names, paste0(name, ":missing"), paste0(name, ":unknown"),
                paste0(name, ":category=", values))
        } else {
            stop("invalid feature kind", call. = FALSE)
        }
        features[[length(features) + 1L]] <- fit
    }
    list(features = features, names = feature_names)
}

csv_transform <- function(columns, table, rows) {
    if (!length(columns$features))
        stop("feature columns are not fitted", call. = FALSE)
    width <- length(columns$names)
    result <- numeric(length(rows) * width)
    offset <- 0L
    for (feature in columns$features) {
        column <- csv_column(table, feature$name)
        for (r in seq_along(rows)) {
            value <- csv_at(table, rows[r], column)
            base <- (r - 1L) * width + offset
            if (feature$kind == "numeric") {
                number <- csv_numeric(value)
                normalized <- if (is.null(number)) 0 else (number - feature$mean) / feature$scale
                if (!is.finite(normalized) || abs(normalized) > csv_float_max)
                    stop("normalized feature overflows float32", call. = FALSE)
                result[base + 1L] <- normalized
                result[base + 2L] <- if (is.null(number)) 1 else 0
            } else {
                index <- if (nzchar(value)) 1L else 0L
                known <- match(value, feature$vocabulary)
                if (nzchar(value) && !is.na(known))
                    index <- 1L + known
                result[base + index + 1L] <- 1
            }
        }
        offset <- offset + if (feature$kind == "numeric") 2L else length(feature$vocabulary) + 2L
    }
    nn$reshape(nn$from_vector(result, length(rows) * width), c(length(rows), width))
}

# ---- checks ----------------------------------------------------------------------------

csv_rejects <- function(code) tryCatch({
    code
    FALSE
}, error = function(e) TRUE)

csv_parser_checks <- function() {
    quoted <- csv_parse("a,b\r\n\"one, two\",\"say \"\"hi\"\"\"\r\n\"line1\nline2\",\"\"\r\n")
    nn_require(length(quoted$rows) == 2L && csv_at(quoted, 0, 1) == "one, two" &&
        csv_at(quoted, 0, 2) == "say \"hi\"" && csv_at(quoted, 1, 1) == "line1\nline2" &&
        csv_at(quoted, 1, 2) == "", "quoted CSV fields")
    nn_require(csv_at(csv_parse("﻿a;b\n1;2", ";"), 0, 2) == "2", "BOM and alternate delimiter")
    nn_require(identical(csv_parse("a,b\n,\n")$rows, list(c("", ""))), "empty fields and terminal newline")
    nn_require(csv_at(csv_parse("a\n  x  "), 0, 1) == "  x  ", "preserve whitespace")
    rejected <- 0L
    for (malformed in c("", "a,a\n1,2", ",b\n1,2", "a,b\n1", "a\n1,2", "a\n\"open", "a\nraw\"quote",
        "a\n\"closed\"tail", "a\r1")) {
        nn_require(csv_rejects(csv_parse(malformed)), "malformed CSV must fail")
        rejected <- rejected + 1L
    }
    for (invalid in c("1x", "NaN", "inf", "1e999", "+-1")) {
        nn_require(csv_rejects(csv_numeric(invalid)), "invalid/nonfinite numeric must fail")
        rejected <- rejected + 1L
    }
    nn_near(csv_numeric(" +1.25e2 "), 125, 0, "numeric whitespace/sign/exponent")
    nn_require(is.null(csv_numeric(" \t")), "empty numeric cell is missing")
    cat("  quoting, multiline fields, CRLF/LF, BOM, delimiters and empty cells passed\n")
    cat(sprintf("  %d malformed CSV/numeric cases rejected\n", rejected))
}

csv_feature_checks <- function() {
    text <- paste0(
        "age,income,city,target,note\r\n",
        "20,100,Denver,0,first\r\n",
        "30,200,Boulder,1,\"said \"\"hello\"\"\"\r\n",
        ",300,Denver,0,\"two\nlines\"\r\n",
        "50,400,Austin,1,last training row\r\n",
        "1000,500,\"Rome, IT\",1,validation only\r\n",
        ",,,0,validation missing\r\n")
    path <- file.path(nn_output_dir("fixtures/tabular"), "features.csv")
    binary_write(path, charToRaw(enc2utf8(text)))
    table <- csv_load(path)
    specs <- list(age = "numeric", income = "numeric", city = "categorical")
    features <- csv_fit(table, 0:3, specs)
    nn_near(features$features[[1]]$mean, 100 / 3, 1e-12, "training age mean excludes validation")
    nn_near(features$features[[1]]$scale, sqrt(1400 / 9), 1e-12, "observed population age deviation")
    nn_near(features$features[[2]]$mean, 250, 0, "training income mean")
    nn_near(features$features[[2]]$scale, sqrt(12500), 1e-12, "population income deviation")
    nn_require(identical(features$features[[3]]$vocabulary, c("Austin", "Boulder", "Denver")),
        "training-only sorted vocabulary")
    training <- csv_transform(features, table, 0:3)
    validation <- csv_transform(features, table, 4:5)
    nn_require(identical(as.numeric(training$shape()), c(4, 9)) && identical(as.numeric(validation$shape()), c(2, 9)),
        "feature dimensions")
    tr <- training$values()
    va <- validation$values()
    nn_near(tr[2 * 9 + 1], 0, 0, "impute standardized mean")
    nn_near(tr[2 * 9 + 2], 1, 0, "numeric missing indicator")
    nn_near(va[1], (1000 - 100 / 3) / sqrt(1400 / 9), 1e-5, "held-out numeric transform")
    nn_near(va[6], 1, 0, "unknown category bucket")
    nn_near(va[9 + 5], 1, 0, "missing category bucket")
    reordered <- table
    reordered$header[c(1, 3)] <- reordered$header[c(3, 1)]
    reordered$rows <- lapply(reordered$rows, function(row) row[c(3, 2, 1, 4:length(row))])
    transformed <- csv_transform(features, reordered, 4:5)
    nn_require(identical(transformed$values(), va), "column reorder must preserve features")
    sparse <- csv_parse("constant,missing\n4,\n4,\n")
    fixed <- csv_fit(sparse, 0:1, list(constant = "numeric", missing = "numeric"))
    zeros <- csv_transform(fixed, sparse, 0:1)$values()
    for (row in 0:1) {
        nn_near(zeros[row * 4 + 1], 0, 0, "constant column")
        nn_near(zeros[row * 4 + 3], 0, 0, "all-missing imputation")
        nn_near(zeros[row * 4 + 4], 1, 0, "all-missing indicator")
    }
    weights <- nn$full(c(9, 1), 0.1)
    out <- nn$matmul(validation, weights)$values()
    for (row in 0:1) {
        expected <- 0
        for (col in 1:9) expected <- expected + va[row * 9 + col] * 0.1
        nn_near(out[row + 1L], expected, 1e-6, "nn consumes feature matrix")
    }
    cat(sprintf("  training rows=4, validation rows=2; features=%d\n", length(features$names)))
    for (name in features$names) cat("    ", name, "\n", sep = "")
    cat(sprintf("  training age mean=%g, scale=%g; held-out age 1000 -> %g\n", features$features[[1]]$mean,
        features$features[[1]]$scale, va[1]))
    cat("  unknown/missing buckets, fixed statistics and reordered columns passed\n")
    cat(sprintf("  nn validation predictions=(%g, %g)\n", out[1], out[2]))
}

main <- function(arguments = commandArgs(TRUE)) {
    tryCatch({
        nn$set_num_threads(2L)
        cat("== CSV parser checks\n")
        csv_parser_checks()
        cat("\n== Fitted feature columns\n")
        csv_feature_checks()
        cat("All CSV and feature-column checks passed.\n")
        0L
    }, error = function(e) {
        message(conditionMessage(e))
        1L
    })
}

if (sys.nframe() == 0L) quit(status = main())
