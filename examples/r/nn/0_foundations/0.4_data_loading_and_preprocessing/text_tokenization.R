# Native R NN example. Whitespace, Unicode characters, learned BPE and greedy WordPiece.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
# CMD: just r-example text_tokenization
# Port of tflow's text_tokenization. Learn ten BPE merges from a weighted toy
# vocabulary, inspect segmentation of unseen word combinations, and reload a
# saved merge model. Test Unicode scalar boundaries and invalid UTF-8,
# deterministic tie-breaking, WordPiece continuation markers, longest matches
# and whole-word unknown fallback. Convert WordPiece tokens to stable IDs,
# truncate/right-pad a batch and feed it to an nn embedding lookup with masked
# mean pooling; an empty sentence must produce a zero vector even when the PAD
# embedding is nonzero. References: https://arxiv.org/abs/1508.07909 and
# https://github.com/google-research/bert/blob/master/tokenization.py.
#
# The tokenizers (examples/nn/common/tokenization.hpp, used only by this
# example) are written out below over UTF-8 bytes. Non-ASCII literals use
# \u escapes and output is written with useBytes so the result does not depend
# on the R session locale. Ordering is byte order (radix sort), as std::map.
nn_helper("checks")
nn_helper("binary")

tok_space <- as.raw(c(0x20, 0x09, 0x0d, 0x0a, 0x0c, 0x0b))

tok_utf8 <- function(bytes) {
    text <- rawToChar(bytes)
    Encoding(text) <- "UTF-8"
    text
}

tok_out <- function(...) writeLines(enc2utf8(paste0(...)), useBytes = TRUE)

tok_sort <- function(x) sort(x, method = "radix")

# Unicode scalar tokens; text is a UTF-8 string or a raw vector of bytes.
tok_characters <- function(text) {
    data <- if (is.raw(text)) text else charToRaw(enc2utf8(text))
    bytes <- as.integer(data)
    tokens <- character()
    i <- 1L
    n <- length(bytes)
    while (i <= n) {
        lead <- bytes[i]
        if (lead < 0x80) {
            count <- 1L; value <- lead; minimum <- 0
        } else if (lead >= 0xc2 && lead <= 0xdf) {
            count <- 2L; value <- bitwAnd(lead, 31L); minimum <- 0x80
        } else if (lead >= 0xe0 && lead <= 0xef) {
            count <- 3L; value <- bitwAnd(lead, 15L); minimum <- 0x800
        } else if (lead >= 0xf0 && lead <= 0xf4) {
            count <- 4L; value <- bitwAnd(lead, 7L); minimum <- 0x10000
        } else {
            stop("invalid UTF-8 leading byte", call. = FALSE)
        }
        if (n - i + 1L < count)
            stop("truncated UTF-8", call. = FALSE)
        for (j in seq_len(count - 1L)) {
            byte <- bytes[i + j]
            if (bitwAnd(byte, 0xc0) != 0x80)
                stop("invalid UTF-8 continuation", call. = FALSE)
            value <- value * 64 + bitwAnd(byte, 63L)
        }
        if (value < minimum || value > 0x10ffff || (value >= 0xd800 && value <= 0xdfff))
            stop("invalid Unicode scalar", call. = FALSE)
        tokens <- c(tokens, tok_utf8(data[i:(i + count - 1L)]))
        i <- i + count
    }
    tokens
}

tok_whitespace <- function(text) {
    tok_characters(text)  # validate UTF-8 even though splitting operates on ASCII bytes
    words <- strsplit(enc2utf8(text), "[ \t\r\n\f\v]+", perl = TRUE, useBytes = TRUE)[[1]]
    words <- words[nzchar(words)]
    vapply(words, function(word) tok_utf8(charToRaw(word)), "", USE.NAMES = FALSE)
}

# std::quoted output: surrounding quotes, backslash-escaped quotes/backslashes.
tok_quoted <- function(value) paste0("\"", gsub("([\"\\\\])", "\\\\\\1", value), "\"")

# ---- BPE ------------------------------------------------------------------------------

tok_merge <- function(symbols, pair) {
    result <- character()
    i <- 1L
    while (i <= length(symbols)) {
        if (i < length(symbols) && symbols[i] == pair[1] && symbols[i + 1L] == pair[2]) {
            result <- c(result, paste0(pair[1], pair[2]))
            i <- i + 2L
        } else {
            result <- c(result, symbols[i])
            i <- i + 1L
        }
    }
    result
}

# frequencies: a named numeric vector word -> count.
tok_bpe_train <- function(frequencies, max_merges, min_count = 2) {
    if (!length(frequencies) || !min_count)
        stop("BPE needs a corpus and positive minimum count", call. = FALSE)
    model <- list(alphabet = character(), merges = list())
    words <- list()
    for (key in tok_sort(names(frequencies))) {
        if (!frequencies[[key]] || !identical(tok_whitespace(key), key))
            stop("BPE corpus keys must be nonempty words", call. = FALSE)
        symbols <- tok_characters(key)
        model$alphabet <- tok_sort(unique(c(model$alphabet, symbols)))
        words[[length(words) + 1L]] <- list(symbols = c(symbols, "</w>"), count = frequencies[[key]])
    }
    for (iteration in seq_len(max_merges)) {
        firsts <- seconds <- character()
        weights <- numeric()
        for (word in words) {
            s <- word$symbols
            if (length(s) > 1L) {
                firsts <- c(firsts, s[-length(s)])
                seconds <- c(seconds, s[-1L])
                weights <- c(weights, rep(word$count, length(s) - 1L))
            }
        }
        if (!length(weights))
            break
        # (first, second) in std::map order: "\001" sorts below every token byte.
        keys <- paste0(firsts, "\001", seconds)
        counts <- vapply(split(weights, keys), sum, 0)
        counts <- counts[tok_sort(names(counts))]
        best_key <- names(counts)[which.max(counts)]
        if (counts[[best_key]] < min_count)
            break
        best <- strsplit(best_key, "\001", fixed = TRUE)[[1]]
        model$merges[[length(model$merges) + 1L]] <- best
        words <- lapply(words, function(word) list(symbols = tok_merge(word$symbols, best), count = word$count))
    }
    model
}

tok_bpe_encode_word <- function(model, word) {
    symbols <- tok_characters(word)
    if (!length(symbols))
        return(character())
    symbols[!symbols %in% model$alphabet] <- "[UNK]"
    symbols <- c(symbols, "</w>")
    for (pair in model$merges) symbols <- tok_merge(symbols, pair)
    symbols
}

tok_bpe_save <- function(model) {
    lines <- c(sprintf("TFLOW_BPE_V1 %d %d", length(model$alphabet), length(model$merges)),
        tok_quoted(model$alphabet),
        vapply(model$merges, function(pair) paste(tok_quoted(pair[1]), tok_quoted(pair[2])), ""))
    paste0(paste0(lines, collapse = "\n"), "\n")
}

# The istream reads Bpe::Load uses over a character vector: word, quoted, ws/eof.
tok_stream <- function(text) {
    chars <- strsplit(enc2utf8(text), "")[[1]]
    at <- 1L
    space <- c(" ", "\t", "\r", "\n", "\f", "\v")
    skip <- function() while (at <= length(chars) && chars[at] %in% space) at <<- at + 1L
    word <- function() {
        skip()
        start <- at
        while (at <= length(chars) && !chars[at] %in% space) at <<- at + 1L
        if (start == at) stop("stream read failed", call. = FALSE)
        paste0(chars[start:(at - 1L)], collapse = "")
    }
    quoted <- function() {
        skip()
        if (at > length(chars)) stop("stream read failed", call. = FALSE)
        if (chars[at] != "\"") return(word())
        at <<- at + 1L
        result <- character()
        while (at <= length(chars)) {
            c <- chars[at]
            at <<- at + 1L
            if (c == "\\" && at <= length(chars)) {
                result <- c(result, chars[at])
                at <<- at + 1L
            } else if (c == "\"") {
                return(paste0(result, collapse = ""))
            } else {
                result <- c(result, c)
            }
        }
        stop("stream read failed", call. = FALSE)
    }
    eof <- function() {
        skip()
        at > length(chars)
    }
    list(word = word, quoted = quoted, eof = eof)
}

tok_bpe_load <- function(serialized) {
    stream <- tok_stream(serialized)
    header <- tryCatch(list(stream$word(), stream$word(), stream$word()),
        error = function(e) stop("invalid BPE model header", call. = FALSE))
    chars <- suppressWarnings(as.numeric(header[[2]]))
    merges <- suppressWarnings(as.numeric(header[[3]]))
    if (header[[1]] != "TFLOW_BPE_V1" || is.na(chars) || is.na(merges) || !chars || chars > 1e6 || merges > 1e6)
        stop("invalid BPE model header", call. = FALSE)
    model <- list(alphabet = character(), merges = list())
    for (i in seq_len(chars)) {
        symbol <- tryCatch(stream$quoted(), error = function(e) stop("invalid BPE alphabet", call. = FALSE))
        if (length(tok_characters(symbol)) != 1L || symbol %in% model$alphabet)
            stop("invalid BPE alphabet", call. = FALSE)
        model$alphabet <- c(model$alphabet, symbol)
    }
    model$alphabet <- tok_sort(model$alphabet)
    known <- c(model$alphabet, "</w>")
    for (i in seq_len(merges)) {
        pair <- tryCatch(c(stream$quoted(), stream$quoted()),
            error = function(e) stop("invalid BPE merge dependency", call. = FALSE))
        if (!pair[1] %in% known || !pair[2] %in% known)
            stop("invalid BPE merge dependency", call. = FALSE)
        known <- c(known, paste0(pair[1], pair[2]))
        model$merges[[length(model$merges) + 1L]] <- pair
    }
    if (!stream$eof())
        stop("trailing BPE model data", call. = FALSE)
    model
}

# ---- WordPiece -------------------------------------------------------------------------

tok_wordpiece <- function(vocabulary, max_chars = 100) {
    if (length(vocabulary) < 2L || vocabulary[1] != "[PAD]" || vocabulary[2] != "[UNK]" || !max_chars)
        stop("WordPiece needs PAD=0, UNK=1 and positive max length", call. = FALSE)
    for (token in vocabulary) tok_characters(token)
    if (any(!nzchar(vocabulary)) || anyDuplicated(vocabulary))
        stop("empty/duplicate WordPiece token", call. = FALSE)
    list(vocabulary = vocabulary, ids = setNames(seq_along(vocabulary) - 1L, vocabulary), max_chars = max_chars)
}

tok_wordpiece_encode_word <- function(wp, word) {
    chars <- tok_characters(word)
    if (length(chars) > wp$max_chars)
        return("[UNK]")
    tokens <- character()
    start <- 1L
    while (start <= length(chars)) {
        found <- ""
        end <- length(chars)
        while (end >= start) {
            candidate <- paste0(if (start > 1L) "##" else "", paste0(chars[start:end], collapse = ""))
            if (candidate %in% wp$vocabulary) {
                found <- candidate
                break
            }
            end <- end - 1L
        }
        if (!nzchar(found))
            return("[UNK]")
        tokens <- c(tokens, found)
        start <- end + 1L
    }
    tokens
}

tok_wordpiece_encode <- function(wp, sentence) {
    ids <- integer()
    for (word in tok_whitespace(sentence))
        ids <- c(ids, unname(wp$ids[tok_wordpiece_encode_word(wp, word)]))
    ids
}

# Right-truncated, right-padded IDs int32 [N, max_length] (PAD = 0) and mask float32.
tok_pad_tokens <- function(sequences, max_length) {
    if (!length(sequences) || !max_length)
        stop("token batching needs sequences and positive max length", call. = FALSE)
    ids <- numeric(length(sequences) * max_length)
    mask <- numeric(length(sequences) * max_length)
    for (i in seq_along(sequences)) {
        sequence <- sequences[[i]]
        for (j in seq_len(min(max_length, length(sequence)))) {
            if (sequence[j] < 0)
                stop("negative token ID", call. = FALSE)
            ids[(i - 1L) * max_length + j] <- sequence[j]
            mask[(i - 1L) * max_length + j] <- 1
        }
    }
    shape <- c(length(sequences), max_length)
    list(ids = nn$cast(nn$from_vector_int64(ids, shape), nn$DType$Int32), mask = nn$from_vector(mask, shape))
}

# ---- checks ----------------------------------------------------------------------------

tok_show <- function(tokens) paste0(sprintf("[%s] ", tokens), collapse = "")

tok_rejects <- function(code) tryCatch({
    code
    FALSE
}, error = function(e) TRUE)

tok_basic <- function() {
    words <- tok_whitespace("  café\tnewer\nlowest  ")
    nn_require(identical(words, c("café", "newer", "lowest")), "whitespace boundaries")
    chars <- tok_characters("café\U0001F642")
    nn_require(identical(chars, c("c", "a", "f", "é", "\U0001F642")), "Unicode scalar tokenization")
    nn_require(length(tok_characters("é")) == 2L, "combining marks remain distinct scalars")
    failures <- 0L
    invalid <- list(c(0xc0, 0x80), c(0xed, 0xa0, 0x80), c(0xf4, 0x90, 0x80, 0x80), c(0xe2, 0x82), 0x80,
        c(0xe2, 0x78, 0x80))
    for (bytes in invalid) {
        nn_require(tok_rejects(tok_characters(as.raw(bytes))), "malformed UTF-8 must fail")
        failures <- failures + 1L
    }
    tok_out("  whitespace: ", tok_show(words))
    tok_out("  characters: ", tok_show(chars))
    tok_out(sprintf("  %d invalid UTF-8 cases rejected", failures))
}

tok_bpe_checks <- function() {
    bpe <- tok_bpe_train(c(low = 5, lower = 2, newest = 6, widest = 3), 10)
    nn_require(length(bpe$merges) == 10L && identical(bpe$merges[[1]], c("e", "s")),
        "weighted pair counts and deterministic tie break")
    tok_out("  learned merges:", paste0(vapply(bpe$merges, function(pair) sprintf(" (%s,%s)", pair[1], pair[2]), ""),
        collapse = ""))
    path <- file.path(nn_output_dir("fixtures/text"), "bpe.model")
    serialized <- tok_bpe_save(bpe)
    binary_write(path, charToRaw(enc2utf8(serialized)))
    restored <- tok_bpe_load(tok_utf8(binary_read(path)))
    nn_require(identical(tok_bpe_save(restored), serialized), "BPE persistence preserves merge ranks/alphabet")
    for (word in c("low", "lower", "lowest", "newer", "widest")) {
        pieces <- tok_bpe_encode_word(bpe, word)
        nn_require(identical(tok_bpe_encode_word(restored, word), pieces), "reloaded BPE segmentation")
        nn_require(paste0(pieces, collapse = "") == paste0(word, "</w>"),
            "BPE preserves known character sequence and word boundary")
        tok_out("  ", word, ": ", tok_show(pieces))
    }
    nn_require(identical(tok_bpe_encode_word(bpe, "λ"), c("[UNK]", "</w>")), "BPE unseen character fallback")
    nn_require(tok_rejects(tok_bpe_load(paste0(serialized, " garbage"))), "invalid saved model rejected")
}

tok_wordpiece_and_batch <- function() {
    wp <- tok_wordpiece(c("[PAD]", "[UNK]", "low", "##est", "lower", "new", "##er", "wide", "##st", "caf",
        "##é", "##s"))
    nn_require(identical(tok_wordpiece_encode_word(wp, "lowest"), c("low", "##est")), "WordPiece continuations")
    nn_require(identical(tok_wordpiece_encode_word(wp, "lower"), "lower"), "WordPiece longest match")
    nn_require(identical(tok_wordpiece_encode_word(wp, "lowerx"), "[UNK]"), "whole word fallback after failed suffix")
    nn_require(identical(tok_wordpiece_encode_word(wp, "café"), c("caf", "##é")),
        "WordPiece Unicode boundaries")
    nn_require(identical(tok_wordpiece_encode_word(wp, strrep("a", 101)), "[UNK]"), "maximum word length")
    for (word in c("lowest", "lower", "newer", "café", "lowerx"))
        tok_out("  ", word, ": ", tok_show(tok_wordpiece_encode_word(wp, word)))
    batch <- tok_pad_tokens(list(tok_wordpiece_encode(wp, "lowest newer"), tok_wordpiece_encode(wp, "unknown"),
        tok_wordpiece_encode(wp, "")), 3L)
    nn_require(identical(as.numeric(batch$ids$shape()), c(3, 3)), "padded token shape")
    ids <- batch$ids$values()
    nn_require(ids[1] == 2 && ids[2] == 3 && ids[3] == 5, "right truncation and stable IDs")
    nn_near(batch$mask$values()[3 + 2], 0, 0, "padding mask")
    vocab <- length(wp$vocabulary)
    table <- as.numeric(rbind(0:(vocab - 1L), 100 + 0:(vocab - 1L)))
    embeddings <- nn$reshape(nn$from_vector(table, vocab * 2), c(vocab, 2))
    lookup <- nn$gather(embeddings, batch$ids, 0L)
    mask <- nn$expand_dims(batch$mask, 2L)
    total <- nn$sum(nn$mul(lookup, mask), 1L)
    denominator <- nn$maximum(nn$sum(mask, 1L), nn$scalar(1))
    pooled <- nn$div(total, denominator)$values()
    nn_near(pooled[1], 10 / 3, 1e-6, "truncated mean embedding")
    nn_near(pooled[2], 100 + 10 / 3, 1e-5, "embedding feature two")
    nn_near(pooled[3], 1, 0, "unknown embedding")
    for (c in 1:2) nn_near(pooled[4 + c], 0, 0, "empty sentence ignores PAD embedding")
    tok_out("  token batch [3,3], retained lengths=(3,1,0)")
    tok_out(sprintf("  masked mean embeddings=(%g, %g), (1,101), (0,0)", pooled[1], pooled[2]))
}

main <- function(arguments = commandArgs(TRUE)) {
    tryCatch({
        nn$set_num_threads(2L)
        tok_out("== Whitespace and characters")
        tok_basic()
        tok_out("\n== Learned BPE")
        tok_bpe_checks()
        tok_out("\n== WordPiece and nn token batching")
        tok_wordpiece_and_batch()
        tok_out("All tokenization checks passed.")
        0L
    }, error = function(e) {
        message(conditionMessage(e))
        1L
    })
}

if (sys.nframe() == 0L) quit(status = main())
