# Kim CNN-rand model pieces (examples/nn/common/kim_cnn.hpp): learned word
# embeddings with a structural zero PAD row, parallel 3/4/5 ReLU word
# convolutions, max-over-time pooling, dropout and a binary head, plus Kim's
# Adadelta (rho .95, eps 1e-6) with the softmax-column max-norm 3.

kim_cnn_config <- function(vocabulary = 8000L, time = 64L, embedding = 300L, filters = 100L)
    list(vocabulary = as.integer(vocabulary), time = as.integer(time), embedding = as.integer(embedding),
         filters = as.integer(filters))

# nn::pad of one 0-based axis with zeros (nn$pad's paddings vector is not bound; concat is equivalent).
kim_cnn_pad_axis <- function(x, axis, before, after) {
    shape <- as.integer(x$shape())
    parts <- list()
    if (before > 0L) {
        shape[axis + 1L] <- as.integer(before)
        parts[[length(parts) + 1L]] <- nn$zeros(shape, x$dtype())
    }
    parts[[length(parts) + 1L]] <- x
    if (after > 0L) {
        shape[axis + 1L] <- as.integer(after)
        parts[[length(parts) + 1L]] <- nn$zeros(shape, x$dtype())
    }
    if (length(parts) == 1L) x else nn$concat(nn$tensor_vector(parts), axis)
}

# [N, T] mask: t < lengths[n].
kim_cnn_length_mask <- function(lengths, time, dtype = nn$DType$Float32) {
    positions <- nn$arange(0, as.numeric(time), 1, nn$DType$Int32)
    keep <- nn$less(nn$expand_dims(positions, 0L), nn$expand_dims(nn$cast(lengths, nn$DType$Int32), 1L))
    nn$cast(keep, dtype)
}

# Kim's max-norm constraint: columns of w scaled to norm <= 3.
kim_cnn_max_norm <- function(w) {
    norm <- nn$sqrt(nn$sum(nn$square(w), 0L, TRUE))
    nn$div(w, nn$maximum(nn$scalar_like(1, w), nn$div(norm, 3)))
}

# tokens [N, T] int, lengths [N]; "embedding/nonpad" holds rows 1..V-1 (U(-.25, .25)).
kim_cnn_forward <- function(p, tokens, lengths, config = kim_cnn_config(), training = FALSE, keep = 1, gen = NULL) {
    if (config$vocabulary < 2L || config$time < 1L || config$embedding < 1L || config$filters < 1L)
        stop("Kim CNN dimensions")
    learned <- p$get("embedding/nonpad", c(config$vocabulary - 1L, config$embedding), nn$init$uniform(.25))
    embedding <- nn$concat(nn$tensor_vector(list(nn$zeros(c(1L, config$embedding), learned$dtype()), learned)), 0L)
    mask <- kim_cnn_length_mask(lengths, config$time, learned$dtype())
    x <- nn$mul(nn$gather(embedding, tokens, 0L), nn$expand_dims(mask, 2L))
    padded <- kim_cnn_pad_axis(x, 1L, 4L, 4L)
    maps <- list()
    pooled <- list()
    for (kernel in 3:5) {
        options <- nn$ConvLayerOptions()
        options$padding <- nn$Padding$Valid
        options$activation <- nn$Activation$Relu
        feature_map <- nn$conv1d(p, paste0("window", kernel), padded, config$filters, kernel, options)
        maps[[length(maps) + 1L]] <- feature_map
        pooled[[length(pooled) + 1L]] <- nn$max(feature_map, 1L)
    }
    features <- nn$concat(nn$tensor_vector(pooled), 1L)
    feature <- features
    if (training && keep < 1) feature <- nn$dropout(feature, 1 - keep, TRUE, gen)
    logits <- nn$dense(p, "classifier", feature, 2L)
    list(logits = logits, features = features, embedding = embedding, maps = maps)
}

# Adadelta rho=.95, eps=1e-6 and softmax-column norm <= 3 as in Kim. Slots
# "gradient<i>" / "update<i>" start at zero; every new value is computed
# before any parameter or slot is written (TF's all-RHS barrier).
kim_cnn_adadelta <- function(p) {
    self <- new.env()
    self$p <- p
    self$slots <- nn$ParameterStore(319)
    values <- nn$tensor_list(p$parameters())
    for (i in seq_along(values)) {
        shape <- as.integer(values[[i]]$shape())
        self$slots$get(paste0("gradient", i - 1L), shape, nn$init$zeros())
        self$slots$get(paste0("update", i - 1L), shape, nn$init$zeros())
    }
    self$step <- function(gradients) {
        if (length(gradients) != self$p$size()) stop("Adadelta gradient alignment")
        nn$no_grad(function() {
            names <- self$p$names()
            params <- list()
            slot_values <- list()
            for (i in seq_along(gradients)) {
                name <- names[[i]]
                g <- gradients[[i]]$detach()
                g2 <- self$slots$at(paste0("gradient", i - 1L))$detach()
                d2 <- self$slots$at(paste0("update", i - 1L))$detach()
                next_g <- nn$add(nn$mul(g2, .95), nn$mul(nn$square(g), .05))
                delta <- nn$mul(nn$div(nn$sqrt(nn$add(d2, 1e-6)), nn$sqrt(nn$add(next_g, 1e-6))), g)
                next_d <- nn$add(nn$mul(d2, .95), nn$mul(nn$square(delta), .05))
                next_p <- nn$sub(self$p$at(name)$detach(), delta)
                if (name == "classifier/W") next_p <- kim_cnn_max_norm(next_p)
                params[[name]] <- next_p
                slot_values[[paste0("gradient", i - 1L)]] <- next_g
                slot_values[[paste0("update", i - 1L)]] <- next_d
            }
            for (name in names(params)) self$p$set(name, params[[name]]$clone())
            for (name in names(slot_values)) self$slots$set(name, slot_values[[name]]$clone())
        })
        invisible(NULL)
    }
    self
}
