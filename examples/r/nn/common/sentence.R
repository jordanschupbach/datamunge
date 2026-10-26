# MR sentence data, Kim CNN-rand model wrapper and the generic 1-D convolution
# helpers (examples/nn/common/sentence.hpp): row selection, stable CE, metrics,
# confusion matrices, fixture parameter assignment, host finite differences and
# store persistence. No training loops live here.
if (!exists("nn_require", inherits = TRUE)) nn_helper("checks")
if (!exists("replay_get", inherits = TRUE)) nn_helper("replay")
if (!exists("kim_cnn_forward", inherits = TRUE)) nn_helper("kim_cnn")

# ---- formatting / float32 / fused multiply-add emulation ---------------------------------

# std::cout << value (default precision 6, %g).
sentence_fmt <- function(x) sprintf("%g", x)
# Integral counts printed as integers.
sentence_int <- function(x) sprintf("%.0f", x)
# std::filesystem::path printed by std::cout (quoted).
sentence_quoted <- function(path) paste0("\"", path, "\"")

# static_cast<float>(x), element-wise.
sentence_f32 <- function(x) readBin(writeBin(as.numeric(x), raw(), size = 4L), "double", n = length(x), size = 4L)

# Correctly rounded a * b + c (the C++ reference is built by clang with
# -ffp-contract=on, so a*b+c inside one expression is a hardware FMA). Exact
# Dekker product plus a compensated sum; vectorized.
sentence_fma <- function(a, b, c) {
    p <- a * b
    split <- 134217729
    ta <- split * a
    ah <- ta - (ta - a)
    al <- a - ah
    tb <- split * b
    bh <- tb - (tb - b)
    bl <- b - bh
    e <- ((ah * bh - p) + ah * bl + al * bh) + al * bl
    s <- p + c
    bb <- s - p
    t <- (p - (s - bb)) + (c - bb)
    s + (t + e)
}

sentence_ints <- function(tensor) as.integer(round(tensor$values()))
sentence_int_tensor <- function(values, shape) nn$cast(nn$from_vector_int64(as.numeric(values), as.integer(shape)), nn$DType$Int32)
sentence_shape <- function(tensor) as.numeric(tensor$shape())

# ---- generic helpers ----------------------------------------------------------------------

sentence_select <- function(x, rows) nn$take_rows(x, as.integer(rows))
sentence_range <- function(begin, end) if (end > begin) seq.int(begin, end - 1L) else integer(0)

# Overwrites every parameter of p (creation order v, flat index i, both 0-based)
# with value(v, i, shape) rounded to float32; returns the values as a list.
sentence_fix_parameters <- function(p, value) {
    names <- p$names()
    tensors <- nn$tensor_list(p$parameters())
    weights <- list()
    for (v in seq_along(tensors)) {
        shape <- as.integer(tensors[[v]]$shape())
        count <- prod(shape)
        data <- sentence_f32(vapply(seq_len(count) - 1L, function(i) value(v - 1L, i, shape), numeric(1)))
        p$set(names[[v]], nn$cast(nn$from_vector(data, shape), p$dtype()))
        weights[[v]] <- data
    }
    weights
}

# max |central difference of objective (step 1e-5) - gradient| over every coordinate.
sentence_host_gradient_error <- function(weights, gradients, objective) {
    error <- 0
    coordinates <- 0L
    for (v in seq_along(weights)) {
        g <- gradients[[v]]$values()
        for (i in seq_along(weights[[v]])) {
            plus <- weights
            minus <- weights
            plus[[v]][i] <- plus[[v]][i] + 1e-5
            minus[[v]][i] <- minus[[v]][i] - 1e-5
            error <- max(error, abs((objective(plus) - objective(minus)) / 2e-5 - g[i]))
            coordinates <- coordinates + 1L
        }
    }
    list(error = error, coordinates = coordinates)
}

# Sum of squares of each gradient (sequential fused accumulation as in C++).
sentence_squared_norms <- function(gradients) vapply(gradients, function(g) {
    s <- 0
    for (x in g$values()) s <- sentence_fma(x, x, s)
    s
}, numeric(1))

# Every coordinate finite and every tensor nonzero (unless may_vanish(name)); returns the nonzero count.
sentence_require_live_gradients <- function(p, gradients, what, may_vanish = NULL, floor = 0) {
    names <- p$names()
    nonzero <- 0L
    for (v in seq_along(gradients)) {
        values <- gradients[[v]]$values()
        nn_require(all(is.finite(values)), paste0("finite ", what, " gradient coordinate"))
        norm <- 0
        for (x in values) norm <- sentence_fma(x, x, norm)
        if (is.null(may_vanish) || !may_vanish(names[[v]]))
            nn_require(norm > floor, paste0("nonzero ", what, " gradient: ", names[[v]]))
        nonzero <- nonzero + as.integer(norm > floor)
    }
    nonzero
}

# -mean_n sum_c onehot(labels) log_softmax(logits) (tflow's SentenceCE).
sentence_ce <- function(logits, labels) {
    shape <- logits$shape()
    target <- nn$one_hot(labels, as.integer(shape[length(shape)]), 1, 0, logits$dtype())
    nn$neg(nn$mean(nn$sum(nn$mul(target, nn$log_softmax(logits)), 1L)))
}

sentence_npz <- function(arrays, key) replay_get(arrays, key)

# Clones of every tensor (tflow's Snapshot); compare with replay_same_bits.
sentence_snapshot <- function(stores) {
    out <- list()
    for (k in seq_along(stores)) for (name in stores[[k]]$names())
        out[[paste0(k - 1L, ":", name)]] <- stores[[k]]$at(name)$detach()$clone()
    out
}

# Parameters (and optional state) as one .npz with "p:"/"s:" prefixes.
sentence_save_stores <- function(path, p, s = NULL) {
    arrays <- list()
    for (n in p$names()) arrays[[paste0("p:", n)]] <- p$at(n)$detach()
    if (!is.null(s)) for (n in s$names()) arrays[[paste0("s:", n)]] <- s$at(n)$detach()
    nn$save_npz(path, arrays)
}

sentence_restore_stores <- function(path, p, s = NULL) {
    arrays <- nn$load_npz(path)
    nn_require(length(arrays) == p$size() + (if (is.null(s)) 0 else s$size()), "checkpoint covers exactly the model tensors")
    for (key in names(arrays)) {
        store <- if (startsWith(key, "p:")) p else s
        name <- substring(key, 3L)
        nn_require(store$contains(name), paste0("checkpoint tensor ", name, " belongs to the model"))
        store$set(name, nn$cast(arrays[[key]], store$dtype()))
    }
}

# Parameters as "p<i>" and state as "s<i>" (tflow's model_arrays.npz layout).
sentence_export_arrays <- function(path, p, s = NULL) {
    arrays <- list()
    values <- nn$tensor_list(p$parameters())
    for (i in seq_along(values)) arrays[[paste0("p", i - 1L)]] <- values[[i]]$detach()
    if (!is.null(s)) {
        values <- nn$tensor_list(s$parameters())
        for (i in seq_along(values)) arrays[[paste0("s", i - 1L)]] <- values[[i]]$detach()
    }
    nn$save_npz(path, arrays)
}

sentence_copy_if_exists <- function(from, to) if (file.exists(from)) invisible(file.copy(from, to, overwrite = TRUE))

# ---- MR sentence data ---------------------------------------------------------------------

sentence_data <- function(tokens, lengths, labels, source_ids) {
    d <- list(tokens = tokens, lengths = lengths, labels = labels, source_ids = source_ids)
    d$size <- sentence_shape(labels)[1L]
    d
}

sentence_rows <- function(d, rows) sentence_data(sentence_select(d$tokens, rows), sentence_select(d$lengths, rows),
    sentence_select(d$labels, rows), sentence_select(d$source_ids, rows))

# Returns list(pools = list(fit, validation, test), config) from <root>/sentence.npz.
sentence_load <- function(root) {
    arrays <- nn$load_npz(file.path(root, "sentence.npz"))
    config_tensor <- sentence_npz(arrays, "config")
    nn_require(config_tensor$dtype() == nn$DType$Int32 && identical(sentence_shape(config_tensor), 2),
               "sentence vocabulary metadata")
    cfg <- sentence_ints(config_tensor)
    config <- kim_cnn_config(cfg[1L], cfg[2L], 32L, 32L)
    nn_require(config$vocabulary >= 2L && config$vocabulary <= 8000L && config$time == 64L, "sentence fit vocabulary/time bounds")
    seen <- integer(0)
    pools <- list()
    for (split in c("fit", "validation", "test")) {
        d <- sentence_data(sentence_npz(arrays, paste0(split, "_tokens")), sentence_npz(arrays, paste0(split, "_lengths")),
                           sentence_npz(arrays, paste0(split, "_labels")), sentence_npz(arrays, paste0(split, "_source_ids")))
        n <- sentence_shape(d$tokens)[1L]
        nn_require(n > 0 && d$tokens$dtype() == nn$DType$Int32 && identical(sentence_shape(d$tokens), c(n, 64)),
                   "sentence token matrix")
        for (t in list(d$lengths, d$labels, d$source_ids))
            nn_require(t$dtype() == nn$DType$Int32 && identical(sentence_shape(t), n), "aligned sentence metadata")
        tokens <- matrix(sentence_ints(d$tokens), nrow = n, byrow = TRUE)
        lengths <- sentence_ints(d$lengths)
        labels <- sentence_ints(d$labels)
        sources <- sentence_ints(d$source_ids)
        nn_require(all(lengths >= 1L & lengths <= 64L & labels >= 0L & labels < 2L & sources >= 0L & sources < 10662L) &&
                       !anyDuplicated(c(seen, sources)), "valid unique sentence ID/label/length")
        seen <- c(seen, sources)
        inside <- col(tokens) <= lengths
        nn_require(all(ifelse(inside, tokens >= 1L & tokens < config$vocabulary, tokens == 0L)), "sentence legal tokens/right PAD")
        cat(sprintf(" MR %s:%d sentences,positive=%d\n", split, as.integer(n), sum(labels)))
        pools[[length(pools) + 1L]] <- d
    }
    list(pools = pools, config = config)
}

# ---- metrics and baselines ------------------------------------------------------------------

sentence_metrics <- function(name, labels_tensor, logits) {
    n <- sentence_shape(labels_tensor)[1L]
    nn_require(identical(sentence_shape(logits), c(n, 2)), "sentiment score shape")
    z <- matrix(logits$values(), ncol = 2L, byrow = TRUE)
    labels <- sentence_ints(labels_tensor)
    a <- z[, 1L]
    b <- z[, 2L]
    nn_require(all(is.finite(a) & is.finite(b)), "finite binary logits")
    prediction <- as.integer(b > a)
    m <- pmax(a, b)
    nll <- sum(m + log(exp(a - m) + exp(b - m)) - ifelse(labels == 1L, b, a))
    confusion <- c(sum(labels == 0L & prediction == 0L), sum(labels == 0L & prediction == 1L),
                   sum(labels == 1L & prediction == 0L), sum(labels == 1L & prediction == 1L))
    cat(sprintf(" %s:accuracy=%s%%,CE=%s,confusion=%s\n", name, sentence_fmt(100 * sum(labels == prediction) / n),
                sentence_fmt(nll / n), paste(confusion, collapse = "/")))
}

# Fit-only add-one unigram multinomial naive Bayes log scores [N, 2].
sentence_nb <- function(fit, test, vocabulary) {
    T <- sentence_shape(fit$tokens)[2L]
    ft <- matrix(sentence_ints(fit$tokens), ncol = T, byrow = TRUE)
    fl <- sentence_ints(fit$lengths)
    fy <- sentence_ints(fit$labels)
    count <- matrix(1, 2L, vocabulary)
    total <- c(vocabulary, vocabulary)
    prior <- c(1, 1)
    for (n in seq_len(fit$size)) {
        label <- fy[n] + 1L
        prior[label] <- prior[label] + 1
        words <- ft[n, seq_len(fl[n])]
        count[label, ] <- count[label, ] + tabulate(words + 1L, vocabulary)
        total[label] <- total[label] + fl[n]
    }
    tt <- matrix(sentence_ints(test$tokens), ncol = T, byrow = TRUE)
    tl <- sentence_ints(test$lengths)
    out <- numeric(2L * test$size)
    for (n in seq_len(test$size)) {
        words <- tt[n, seq_len(tl[n])] + 1L
        for (c in 1:2)
            out[2L * (n - 1L) + c] <- sum(c(log(prior[c] / (prior[1L] + prior[2L])), log(count[c, words] / total[c])))
    }
    nn$from_vector(out, c(test$size, 2L))
}

# 1 when logits[n,1] > logits[n,0].
sentence_positive <- function(logits) {
    z <- matrix(logits$values(), ncol = 2L, byrow = TRUE)
    as.integer(z[, 2L] > z[, 1L])
}

# cells[truth * classes + prediction + 1] counts (row-major, 0-based classes).
sentence_confusion <- function(labels_tensor, predictions, classes) {
    labels <- sentence_ints(labels_tensor)
    nn_require(length(labels) == length(predictions), "confusion labels/predictions alignment")
    tabulate(labels * classes + as.integer(predictions) + 1L, classes * classes)
}

# Prints a confusion matrix (rows = truth, columns = prediction).
sentence_print_confusion <- function(name, cells, classes) {
    cat(sprintf(" Confusion matrix %s (rows=truth, columns=prediction):\n", name))
    cat(" truth/pred", sprintf(" %5d", seq_len(classes) - 1L), "\n", sep = "")
    for (r in seq_len(classes))
        cat(sprintf(" %10d", r - 1L), sprintf(" %5d", as.integer(cells[(r - 1L) * classes + seq_len(classes)])), "\n", sep = "")
}

# ---- Kim CNN-rand model ---------------------------------------------------------------------

# Parameters (seed 317), Kim's Adadelta when training and the dropout stream (313).
sentence_model <- function(config, training = TRUE) {
    self <- new.env()
    self$parameters <- nn$ParameterStore(317)
    self$config <- config
    self$dropout <- nn$Generator(313)
    nn$no_grad(function() kim_cnn_forward(self$parameters, nn$zeros(c(1L, config$time), nn$DType$Int32),
                                          nn$full(1L, 1, nn$DType$Int32), config))
    self$optimizer <- if (training) kim_cnn_adadelta(self$parameters) else NULL
    self$forward <- function(d, training) kim_cnn_forward(self$parameters, d$tokens, d$lengths, self$config, training, .5, self$dropout)
    # Inference (dropout-free, no gradient) CE.
    self$loss <- function(d) {
        value <- nn$no_grad(function() sentence_ce(self$forward(d, FALSE)$logits, d$labels)$item())
        nn_require(is.finite(value) && value >= 0, "finite sentence CE")
        value
    }
    self$predict <- function(d) nn$no_grad(function() self$forward(d, FALSE)$logits$detach())
    # Predictions in batches of 50.
    self$predict_all <- function(d) {
        parts <- lapply(seq.int(0L, d$size - 1L, by = 50L), function(start)
            self$predict(sentence_rows(d, sentence_range(start, min(start + 50L, d$size)))))
        nn$concat(nn$tensor_vector(parts), 0L)
    }
    self$snapshot <- function() sentence_snapshot(list(self$parameters))
    self$save <- function(path) sentence_save_stores(path, self$parameters)
    self$restore <- function(path) sentence_restore_stores(path, self$parameters)
    self
}
