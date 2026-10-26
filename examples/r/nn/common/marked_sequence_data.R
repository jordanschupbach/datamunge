# Synthetic variable-length single-marker memory task; padding never reaches
# the model through the length mask (examples/nn/common/marked_sequence_data.hpp;
# tflow's common/ml/marked_sequence_data.h plus its per-example SaveData, Copy,
# Accuracy and Heatmap helpers).
#
# The generator is std::mt19937 exactly as in tflow, so every split is
# bit-identical to tflow's data.npz. The bound nn$Mt19937 cannot hand R its raw
# 32-bit draws (R integers are 31-bit), so the engine is written out here with
# exact unsigned 32-bit arithmetic on doubles.
if (!exists("recurrent_model_new", inherits = TRUE)) nn_helper("recurrent_model")

marked_sequence_data_xor32 <- function(a, b)
    bitwXor(a %/% 65536, b %/% 65536) * 65536 + bitwXor(a %% 65536, b %% 65536)
marked_sequence_data_and32 <- function(a, b)
    bitwAnd(a %/% 65536, b %/% 65536) * 65536 + bitwAnd(a %% 65536, b %% 65536)
marked_sequence_data_mul32 <- function(a, b) {
    ah <- a %/% 65536
    al <- a %% 65536
    bh <- b %/% 65536
    bl <- b %% 65536
    (((ah * bl + al * bh) %% 65536) * 65536 + al * bl) %% 4294967296
}

# std::mt19937(seed): returns a function draws(n) giving the next n raw 32-bit outputs.
marked_sequence_data_mt19937 <- function(seed) {
    xor32 <- marked_sequence_data_xor32
    and32 <- marked_sequence_data_and32
    mt <- numeric(624)
    mt[1] <- seed %% 4294967296
    for (i in 2:624) {
        previous <- mt[i - 1]
        mt[i] <- (marked_sequence_data_mul32(1812433253, xor32(previous, previous %/% 1073741824)) + (i - 1)) %% 4294967296
    }
    buffer <- numeric(0)
    twist <- function() {
        # new[i] = old[i + 397] ^ (y >> 1) ^ (odd(y) ? 0x9908b0df : 0), y = upper(mt[i]) | lower(mt[i + 1]);
        # computed in blocks so that every reference to an already-updated word sees its new value.
        mix <- function(i, j, k) {
            y <- (mt[i] %/% 2147483648) * 2147483648 + mt[j] %% 2147483648
            xor32(xor32(mt[k], y %/% 2), ifelse(y %% 2 == 1, 2567483615, 0))
        }
        a <- 1:227
        mt[a] <<- mix(a, a + 1, a + 397)
        b <- 228:454
        mt[b] <<- mix(b, b + 1, b - 227)
        c <- 455:623
        mt[c] <<- mix(c, c + 1, c - 227)
        mt[624] <<- mix(624, 1, 397)
        y <- mt
        y <- xor32(y, y %/% 2048)
        y <- xor32(y, and32((y * 128) %% 4294967296, 2636928640))
        y <- xor32(y, and32((y * 32768) %% 4294967296, 4022730752))
        buffer <<- c(buffer, xor32(y, y %/% 262144))
    }
    function(n) {
        while (length(buffer) < n) twist()
        out <- buffer[seq_len(n)]
        buffer <<- buffer[-seq_len(n)]
        out
    }
}

marked_sequence_data_int32 <- function(values) nn$cast(nn$from_vector_int64(as.numeric(values), length(values)), nn$DType$Int32)

# inputs float [n, 12, 2] (value, marker), lengths/labels/source_ids int32 [n], uniforms float64 [n, 14].
marked_sequence_data_new <- function(inputs, lengths, labels, source_ids, uniforms)
    list(inputs = inputs, lengths = lengths, labels = labels, source_ids = source_ids, uniforms = uniforms)

marked_sequence_data_feed <- function(x) recurrent_model_feed(x$inputs, x$lengths, x$labels)
marked_sequence_data_size <- function(x) x$inputs$shape()[1]

marked_sequence_data_make <- function(n, seed, offset) {
    draws <- marked_sequence_data_mt19937(seed)
    uniforms <- (draws(n * 14) + 0.5) / 4294967296
    inputs <- numeric(n * 24)
    lengths <- labels <- integer(n)
    for (b in seq_len(n)) {
        u <- uniforms[(b - 1) * 14 + 1:14]
        length <- 4L + as.integer(floor(9 * u[1]))
        cue <- as.integer(floor((length - 1) * u[2]))
        label <- (b - 1L) %% 2L
        lengths[b] <- length
        labels[b] <- label
        base <- (b - 1) * 24
        t <- 0:(length - 1)
        inputs[base + t * 2 + 1] <- ifelse(u[t + 3] < 0.5, -1, 1)
        inputs[base + cue * 2 + 1] <- if (label) 1 else -1
        inputs[base + cue * 2 + 2] <- 1
    }
    marked_sequence_data_new(nn$from_vector(inputs, c(n, 12, 2)), marked_sequence_data_int32(lengths),
        marked_sequence_data_int32(labels), marked_sequence_data_int32(offset + 0:(n - 1)),
        nn$from_vector_double(uniforms, c(n, 14)))
}

# Rows are 0-based like the C++.
marked_sequence_data_rows <- function(x, rows) lapply(x, function(v) nn$take_rows(v, rows))
marked_sequence_data_copy <- function(x) marked_sequence_data_rows(x, seq_len(marked_sequence_data_size(x)) - 1)

marked_sequence_data_save <- function(path, x)
    nn$save_npz(path, x[c("inputs", "lengths", "labels", "source_ids", "uniforms")])

marked_sequence_data_load <- function(path) {
    a <- nn$load_npz(path)
    marked_sequence_data_new(recurrent_model_find(a, "inputs"), recurrent_model_find(a, "lengths"),
        recurrent_model_find(a, "labels"), recurrent_model_find(a, "source_ids"), recurrent_model_find(a, "uniforms"))
}

# The three splits (512 fit / 128 validation / 128 test) and their data.npz arrays (attribute "arrays").
marked_sequence_data_pools <- function() {
    pools <- list()
    arrays <- list()
    splits <- c("fit", "validation", "test")
    for (k in 0:2) {
        x <- marked_sequence_data_make(if (k) 128L else 512L, 26001 + 2 * k, 10000 * k)
        pools[[k + 1]] <- x
        for (key in c("inputs", "lengths", "labels", "source_ids", "uniforms")) arrays[[paste0(splits[k + 1], "_", key)]] <- x[[key]]
    }
    attr(pools, "arrays") <- arrays
    pools
}

# 1 where logit 1 exceeds logit 0, per row.
marked_sequence_data_predictions <- function(logits) {
    l <- matrix(logits$values(), ncol = 2, byrow = TRUE)
    as.integer(l[, 2] > l[, 1])
}

# Fraction of rows whose logit 1 exceeds logit 0 exactly when the label is 1.
marked_sequence_data_accuracy <- function(logits, labels) {
    y <- labels$values()
    mean(marked_sequence_data_predictions(logits) == (y != 0))
}

# tflow's Heatmap: first 24 sequences, 12 columns of 29x15 cells; blue = -1,
# red = +1, white = 0, black strip = marker; gradients are scaled by the global
# peak |value| of channel 0.
marked_sequence_data_heatmap <- function(path, x, gradients = FALSE) {
    n <- min(24L, x$shape()[1])
    v <- x$values()
    rgb <- array(220, c(3L, 360L, n * 16L))  # column-major [channel, column, row] = row-major [row, column, channel]
    value0 <- function(b, t) v[(b * 12 + t) * 2 + 1]
    peak <- 1
    if (gradients) peak <- max(1e-12, max(0, abs(v[((rep(0:(n - 1), each = 12) * 12 + rep(0:11, n)) * 2) + 1])))
    for (b in 0:(n - 1)) {
        for (t in 0:11) {
            value <- value0(b, t) / peak
            colour <- c(floor(255 * (1 - max(0, -value)) + 0.5), floor(255 * (1 - abs(value)) + 0.5),
                floor(255 * (1 - max(0, value)) + 0.5))
            rgb[, t * 30 + 1:29, b * 16 + 1:15] <- colour
            if (!gradients && v[(b * 12 + t) * 2 + 2] > 0) rgb[, t * 30 + 1:29, b * 16 + 1:3] <- 0
        }
    }
    nn$save_png(nn$from_vector(as.vector(rgb), c(n * 16, 360, 3)), path, 0, 255)
}
