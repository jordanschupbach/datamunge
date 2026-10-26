# Autoencoder support mirroring examples/nn/common/autoencoder.hpp.
#
# Saliency MNIST pools (1000 fit / 200 validation / 200 test from
# $DATAMUNGE_DATA_DIR/saliency/saliency_data.npz), tensor/byte utilities, figure
# writers, the convolutional / variational / dense undercomplete autoencoder
# components, tflow's MT19937 Box-Muller Gaussian banks and denoising corruption.
# Only model pieces and numerical utilities live here: training loops stay in the
# example scripts (there is deliberately no update/train function).

autoencoder_require <- function(condition, message) if (!isTRUE(condition)) stop(message, call. = FALSE)

# ---- data ------------------------------------------------------------------------------

autoencoder_rows <- function(data, rows) {
    list(images = nn$take_rows(data$images, rows), labels = nn$take_rows(data$labels, rows),
         source = nn$take_rows(data$source, rows))
}

autoencoder_find <- function(arrays, name) {
    value <- arrays[[name]]
    if (is.null(value)) stop("missing array ", name, call. = FALSE)
    value
}

# list(fit, validation, test) of list(images, labels, source) with tflow's integrity checks.
autoencoder_load_saliency <- function(root) {
    file <- file.path(root, "saliency_data.npz")
    autoencoder_require(file.exists(file),
        paste0(file, " missing; run `just autoencoder-assets` in tflow_examples or pass data="))
    arrays <- nn$load_npz(file)
    result <- list()
    ids <- numeric()
    seen <- new.env(hash = TRUE, parent = emptyenv())
    for (split in c("fit", "validation", "test")) {
        x <- autoencoder_find(arrays, paste0(split, "_images"))$contiguous()
        n <- if (split == "fit") 1000L else 200L
        autoencoder_require(identical(as.numeric(x$shape()), c(n, 28, 28, 1)), "source28x28 grayscale digit images")
        labels <- autoencoder_find(arrays, paste0(split, "_labels"))$contiguous()
        source <- autoencoder_find(arrays, paste0(split, "_source_ids"))$contiguous()
        autoencoder_require(labels$dtype() == nn$DType$Int32 && source$dtype() == nn$DType$Int32 &&
            identical(as.numeric(labels$shape()), as.numeric(n)) && identical(as.numeric(source$shape()), as.numeric(n)),
            "aligned source/label vectors")
        label_values <- labels$values()
        source_values <- source$values()
        autoencoder_require(all(label_values >= 0 & label_values < 10) &&
            !anyDuplicated(c(ids, source_values)), "source IDs disjoint,labels valid")
        ids <- c(ids, source_values)
        boundary <- if (split == "test") source_values >= 60000 & source_values < 70000
                    else source_values >= 0 & source_values < 60000
        autoencoder_require(all(boundary), "original test/train boundary")
        pixels <- x$values()
        autoencoder_require(all(pixels >= 0 & pixels < 256), "uint8 pixels")
        for (b in seq_len(n)) {
            key <- paste(pixels[(b - 1L) * 784L + seq_len(784L)], collapse = ",")
            autoencoder_require(!exists(key, envir = seen, inherits = FALSE), "no duplicated input bytes across all pools")
            assign(key, TRUE, envir = seen)
        }
        autoencoder_require(all(tabulate(as.integer(label_values) + 1L, 10L) == n / 10L), "balanced digit pools")
        result[[split]] <- list(images = nn$div(nn$cast(x, nn$DType$Float32), 255), labels = labels, source = source)
    }
    unname(result)
}

# Copies tflow's provenance files next to the outputs when they exist.
autoencoder_copy_manifests <- function(data, root) {
    if (file.exists(file.path(data, "manifest.json")))
        file.copy(file.path(data, "manifest.json"), file.path(root, "data_manifest.json"), overwrite = TRUE)
    source <- file.path(nn_data_root(), "autoencoder", "source_manifest.json")
    if (file.exists(source)) file.copy(source, file.path(root, "source_manifest.json"), overwrite = TRUE)
    invisible(NULL)
}

# ---- tensor helpers ----------------------------------------------------------------------

# Raw float32 bytes, for exact-equality checks.
autoencoder_bytes <- function(tensor) writeBin(tensor$contiguous()$values(), raw(), size = 4L)

autoencoder_max_abs_diff <- function(a, b, count) {
    x <- a$contiguous()$values()[seq_len(count)]
    y <- b$contiguous()$values()[seq_len(count)]
    max(abs(x - y))
}

# Double-precision MSE (C++ summation order); the prediction must be finite in [0, 1].
autoencoder_mse <- function(prediction, original) {
    autoencoder_require(prediction$numel() == original$numel(), "aligned reconstruction pixels")
    p <- prediction$contiguous()$values()
    o <- original$contiguous()$values()
    autoencoder_require(all(is.finite(p) & p >= 0 & p <= 1), "finite bounded sigmoid reconstruction")
    autoencoder_sum((p - o)^2) / length(p)
}

# Left-to-right double sum (R's sum() accumulates in long double).
autoencoder_sum <- function(values) Reduce(`+`, values, accumulate = FALSE, 0)

# Per-pixel mean [784] of the fit images.
autoencoder_pixel_mean <- function(fit) {
    pixels <- matrix(fit$contiguous()$values(), ncol = 784L, byrow = TRUE)
    means <- vapply(seq_len(784L), function(i) autoencoder_sum(pixels[, i]) / nrow(pixels), numeric(1))
    nn$from_vector(means, 784L)
}

# `mean` [784] repeated n times as [n, 784].
autoencoder_constant <- function(mean, n) nn$tile(nn$reshape(mean, c(1L, 784L)), c(as.integer(n), 1L))$contiguous()

# Deep copies of every parameter, named, in creation order.
autoencoder_arrays <- function(store) nn_arrays(store)

autoencoder_snapshot <- function(store) lapply(store$names(), function(name) autoencoder_bytes(store$at(name)))

# Checks every gradient is finite and nonzero; returns them as a named list.
autoencoder_check_gradients <- function(store, gradients, finite, nonzero) {
    autoencoder_require(length(gradients) == store$size(), "one gradient per parameter")
    for (gradient in gradients) {
        autoencoder_require(gradient$defined(), nonzero)
        values <- gradient$contiguous()$values()
        autoencoder_require(all(is.finite(values)), finite)
        autoencoder_require(sum(values^2) > 0, nonzero)
    }
    setNames(gradients, store$names())
}

# ---- figures ------------------------------------------------------------------------------

# One row of 28x28 panels scaled 4x, 4 px apart on gray 220 (tflow's digit<k>.png).
autoencoder_panel_row <- function(path, panels) {
    tiles <- lapply(panels, function(p) nn$reshape(p, c(28L, 28L, 1L)))
    nn$save_image_grid(nn$stack(nn$tensor_vector(tiles), 0L), path, length(tiles), 0, 1, 4L, 220 / 255, 4L)
}

# The b-th (0-based) 784-pixel row of [n, 784] / [n, 28, 28, 1] values.
autoencoder_row <- function(x, b) nn$reshape(nn$slice_axis(x, 0L, b, b + 1L), 784L)

# ---- convolutional autoencoder -------------------------------------------------------------

autoencoder_relu_conv_options <- function() {
    options <- nn$ConvLayerOptions()
    options$init <- nn$init$he_uniform()
    options$activation <- nn$Activation$Relu
    options
}

# nearest-repeat2, Conv3x3 4->8 ReLU, nearest-repeat2, Conv3x3 8->1 sigmoid; list(decoder1, reconstruction).
autoencoder_conv_ae_decode <- function(p, code) {
    d1 <- nn$conv2d(p, "decoder1", nn$upsample_nearest(code, 2L), 8L, 3L, autoencoder_relu_conv_options())
    output <- nn$ConvLayerOptions()
    output$activation <- nn$Activation$Sigmoid
    list(decoder1 = d1, reconstruction = nn$conv2d(p, "decoder2", nn$upsample_nearest(d1, 2L), 1L, 3L, output))
}

# Conv3x3 1->8 ReLU, AvgPool2, Conv3x3 8->4 ReLU, AvgPool2 (7x7x4 code), then the decoder.
autoencoder_convolutional_autoencoder <- function(p, image) {
    out <- list()
    out$conv1 <- nn$conv2d(p, "encoder1", image, 8L, 3L, autoencoder_relu_conv_options())
    out$pool1 <- nn$avg_pool2d(out$conv1, 2L, 2L)
    out$conv2 <- nn$conv2d(p, "encoder2", out$pool1, 4L, 3L, autoencoder_relu_conv_options())
    out$code <- nn$avg_pool2d(out$conv2, 2L, 2L)
    decoded <- autoencoder_conv_ae_decode(p, out$code)
    out$decoder1 <- decoded$decoder1
    out$reconstruction <- decoded$reconstruction
    out
}

# Adam(.003), the optimiser of every short autoencoder example.
autoencoder_make_adam <- function(store, learning_rate = 0.003) {
    config <- nn$OptimizerConfig()
    config$kind <- nn$OptimizerKind$Adam
    nn$Optimizer(store, learning_rate, config)
}

# Parameters (seed 1801) + optional Adam; forward pieces only.
autoencoder_convolutional_model <- function(training = TRUE) {
    model <- new.env()
    model$parameters <- nn$ParameterStore(1801L)
    nn$no_grad(function() autoencoder_convolutional_autoencoder(model$parameters, nn$zeros(c(1L, 28L, 28L, 1L))))
    model$optimizer <- if (training) autoencoder_make_adam(model$parameters) else NULL
    model$loss <- function(reconstruction, image) nn$mean(nn$square(nn$sub(reconstruction, image)))
    model$forward <- function(images) autoencoder_convolutional_autoencoder(model$parameters, images)
    model$evaluate <- function(images) nn$no_grad(function() {
        net <- model$forward(images)
        list(net$code, net$reconstruction, model$loss(net$reconstruction, images))
    })
    model$activations <- function(images) nn$no_grad(function() {
        n <- model$forward(images)
        list(n$conv1, n$pool1, n$conv2, n$code, n$decoder1, n$reconstruction)
    })
    model$decode <- function(codes) nn$no_grad(function() autoencoder_conv_ae_decode(model$parameters, codes)$reconstruction)
    model
}

# ---- variational autoencoder ---------------------------------------------------------------

autoencoder_vae_reparameterize <- function(mu, log_variance, epsilon)
    nn$add(mu, nn$mul(nn$exp(nn$mul(log_variance, 0.5)), epsilon))

# KL(N(mu, exp(lv)) || N(0, I)) per row.
autoencoder_vae_kl <- function(mu, log_variance) {
    inner <- nn$sub(nn$sub(nn$add(nn$square(mu), nn$exp(log_variance)), 1), log_variance)
    nn$mul(nn$sum(inner, 1L), 0.5)
}

# Bernoulli NLL summed over pixels, stable for large |logits|.
autoencoder_vae_pixel_nll <- function(logits, target) nn$sum(nn$sub(nn$softplus(logits), nn$mul(target, logits)), 1L)

autoencoder_vae_decode_logits <- function(p, z) nn$dense(p, "output", nn$dense(p, "decoder", z, 64L, nn$Activation$Relu), 784L)

# 784 -> 64 ReLU -> (mu, log variance) [2]; z = mu + exp(lv/2) eps; 2 -> 64 ReLU -> 784 logits.
autoencoder_variational_autoencoder <- function(p, image, epsilon) {
    out <- list()
    h <- nn$dense(p, "encoder", image, 64L, nn$Activation$Relu)
    out$mu <- nn$dense(p, "mu", h, 2L)
    out$log_variance <- nn$dense(p, "log_variance", h, 2L)
    out$z <- autoencoder_vae_reparameterize(out$mu, out$log_variance, epsilon)
    out$logits <- autoencoder_vae_decode_logits(p, out$z)
    out$probability <- nn$sigmoid(out$logits)
    out$reconstruction_nll <- autoencoder_vae_pixel_nll(out$logits, image)
    out$kl <- autoencoder_vae_kl(out$mu, out$log_variance)
    out$loss <- nn$mean(nn$add(out$reconstruction_nll, out$kl), 0L)
    out
}

# Parameters (seed 1951) + optional Adam; forward pieces only.
autoencoder_variational_model <- function(training = TRUE) {
    model <- new.env()
    model$parameters <- nn$ParameterStore(1951L)
    nn$no_grad(function() autoencoder_variational_autoencoder(model$parameters, nn$zeros(c(1L, 784L)), nn$zeros(c(1L, 2L))))
    model$optimizer <- if (training) autoencoder_make_adam(model$parameters) else NULL
    model$flatten <- function(images) nn$reshape(images, c(images$size(0L), 784L))
    model$forward <- function(images, noise) autoencoder_variational_autoencoder(model$parameters, model$flatten(images), noise)
    # list(mu, log_variance, z, probability, reconstruction_nll, kl, loss), all checked finite.
    model$evaluate <- function(images, noise) {
        out <- nn$no_grad(function() {
            n <- model$forward(images, noise)
            lapply(list(n$mu, n$log_variance, n$z, n$probability, n$reconstruction_nll, n$kl, n$loss),
                   function(t) t$contiguous())
        })
        for (t in out) autoencoder_require(all(is.finite(t$values())), "ALL posterior/output/objective values finite")
        out
    }
    model$decode <- function(codes) nn$no_grad(function() nn$sigmoid(autoencoder_vae_decode_logits(model$parameters, codes)))
    model
}

# (a * b) mod 2^32 for 32-bit unsigned values held in doubles.
autoencoder_mul32 <- function(a, b) {
    ah <- a %/% 65536; al <- a %% 65536
    bh <- b %/% 65536; bl <- b %% 65536
    (al * bl + ((ah * bl + al * bh) %% 65536) * 65536) %% 4294967296
}

autoencoder_xor32 <- function(a, b) {
    bitwXor(as.integer(a %/% 65536), as.integer(b %/% 65536)) * 65536 +
        bitwXor(as.integer(a %% 65536), as.integer(b %% 65536))
}

# `count` raw std::mt19937(seed)() outputs: R's own Mersenne-Twister seeded with init_genrand(seed).
autoencoder_mt19937_raw <- function(count, seed) {
    state <- numeric(624L)
    state[1L] <- seed %% 4294967296
    for (i in 2L:624L) {
        previous <- state[i - 1L]
        state[i] <- (autoencoder_mul32(1812433253, autoencoder_xor32(previous, previous %/% 1073741824)) + (i - 1L)) %%
            4294967296
    }
    signed <- as.integer(ifelse(state >= 2147483648, state - 4294967296, state))
    had <- exists(".Random.seed", envir = globalenv(), inherits = FALSE)
    if (had) old <- get(".Random.seed", envir = globalenv())
    on.exit(if (had) assign(".Random.seed", old, envir = globalenv()) else rm(".Random.seed", envir = globalenv()))
    assign(".Random.seed", c(10403L, 624L, signed), envir = globalenv())
    stats::runif(count) * 4294967296
}

# tflow's Box-Muller stream from std::mt19937(seed): `count` doubles.
autoencoder_gaussian_values <- function(count, seed) {
    pairs <- (count + 1L) %/% 2L
    raw <- autoencoder_mt19937_raw(2 * pairs, seed)
    u <- (raw[c(TRUE, FALSE)] + 0.5) / 4294967296
    v <- (raw[c(FALSE, TRUE)] + 0.5) / 4294967296
    tau <- 2 * pi  # R_strtod may misround the long literal; 2 * pi is the exact double
    radius <- sqrt(-2 * log(u))
    bank <- as.vector(rbind(radius * cos(tau * v), radius * sin(tau * v)))
    bank[seq_len(count)]
}

# tflow's nn::GaussianNoise as float64 [samples, shape...].
autoencoder_gaussian_noise <- function(shape, samples, seed) {
    full <- as.integer(c(samples, shape))
    nn$from_vector_double(autoencoder_gaussian_values(prod(full), seed), full)
}

autoencoder_vae_noise <- function(n, samples, seed) {
    standard <- autoencoder_gaussian_noise(c(n, 2L), samples, seed)
    list(standard = standard, epsilon = nn$cast(standard, nn$DType$Float32)$contiguous())
}

autoencoder_vae_noise_at <- function(bank, k) nn$select(bank$epsilon, 0L, k)$contiguous()

# Monte Carlo c(reconstruction NLL, KL, negative ELBO) per image over every bank sample.
autoencoder_vae_score <- function(model, images, bank) {
    n <- images$size(0L)
    samples <- bank$epsilon$size(0L)
    reconstruction <- 0
    kl <- 0
    for (k in seq_len(samples) - 1L) {
        out <- model$evaluate(images, autoencoder_vae_noise_at(bank, k))
        reconstruction <- Reduce(`+`, out[[5L]]$values(), reconstruction)
        if (k == 0L) kl <- Reduce(`+`, out[[6L]]$values(), kl)
    }
    reconstruction <- reconstruction / (n * samples)
    kl <- kl / n
    c(reconstruction, kl, reconstruction + kl)
}

# ---- dense undercomplete autoencoder -------------------------------------------------------

# [n, 28, 28, 1] (or any n x 784 values) as [n, 784].
autoencoder_flatten <- function(images) nn$reshape(images, c(images$size(0L), 784L))

# "encoder" features -> latent ReLU, "decoder" latent -> features sigmoid; list(code, reconstruction).
autoencoder_undercomplete <- function(p, x, features = 784L, latent = 32L) {
    autoencoder_require(features > 1L && latent > 0L && latent < features, "strictly undercomplete positive bottleneck")
    code <- nn$dense(p, "encoder", x, as.integer(latent), nn$Activation$Relu)
    list(code = code, reconstruction = nn$dense(p, "decoder", code, as.integer(features), nn$Activation$Sigmoid))
}

# tflow's ReconstructionMse: mean over every element.
autoencoder_reconstruction_mse <- function(prediction, original) nn$mean(nn$square(nn$sub(prediction, original)))

# clamp(float(x + sigma z), 0, 1) with z one float64 bank [1, shape...].
autoencoder_corrupt_for_denoising <- function(clean, standard_noise, sigma) {
    sigma <- readBin(writeBin(sigma, raw(), size = 4L), "double", size = 4L)
    autoencoder_require(clean$dtype() == nn$DType$Float32 && clean$dim() >= 2L &&
        standard_noise$dtype() == nn$DType$Float64 &&
        identical(as.numeric(standard_noise$shape()), c(1, as.numeric(clean$shape()))) &&
        is.finite(sigma) && sigma >= 0, "aligned single Gaussian bank and nonnegative sigma")
    x <- clean$contiguous()$values()
    z <- standard_noise$contiguous()$values()
    autoencoder_require(all(is.finite(x) & x >= 0 & x <= 1 & is.finite(z)), "finite unit-range clean pixels/noise")
    # float32 rounding first (from_vector), then the clamp in float32.
    nn$clip(nn$from_vector(x + sigma * z, clean$shape()), 0, 1)
}

# tflow's data.npz record: the three pools (+ any extra named tensors).
autoencoder_save_data <- function(path, pools, extra = list()) {
    arrays <- list(fit_images = pools[[1L]]$images, validation_images = pools[[2L]]$images,
                   test_images = pools[[3L]]$images, fit_source_ids = pools[[1L]]$source,
                   validation_source_ids = pools[[2L]]$source, test_source_ids = pools[[3L]]$source,
                   test_labels = pools[[3L]]$labels)
    nn$save_npz(path, c(arrays, extra))
}

# 0-based index of the first record of `digit`.
autoencoder_first_of_digit <- function(labels, digit) {
    index <- which(labels$contiguous()$values() == digit)
    if (!length(index)) stop("no record of digit ", digit, call. = FALSE)
    index[1L] - 1L
}

# Writes digit<k>.png (row b of every panel); returns the html sections.
autoencoder_digit_gallery <- function(root, test, panels) {
    source <- test$source$contiguous()$values()
    sections <- character()
    for (digit in 0L:9L) {
        b <- autoencoder_first_of_digit(test$labels, digit)
        file <- paste0("digit", digit, ".png")
        autoencoder_panel_row(file.path(root, file), lapply(panels, function(t) autoencoder_row(t, b)))
        sections <- c(sections, sprintf("<section><h2>Digit%d,source%d</h2><img src='%s'></section>",
                                        digit, as.integer(source[b + 1L]), file))
    }
    sections
}

# Byte-identical parameter stores (matched-initialisation checks).
autoencoder_same_arrays <- function(a, b) {
    if (!identical(a$names(), b$names())) return(FALSE)
    all(vapply(a$names(), function(name) identical(autoencoder_bytes(a$at(name)), autoencoder_bytes(b$at(name))),
               logical(1)))
}

# ---- 2.3 applications: frozen checkpoints and case figures ---------------------------------

# tflow's ml::UndercompleteModel (784->32 ReLU->784 sigmoid, seed 1471) for frozen applications.
autoencoder_undercomplete_model <- function(training = TRUE) {
    model <- new.env()
    model$parameters <- nn$ParameterStore(1471L)
    nn$no_grad(function() autoencoder_undercomplete(model$parameters, nn$zeros(c(1L, 784L))))
    model$optimizer <- if (training) autoencoder_make_adam(model$parameters) else NULL
    model$evaluate <- function(images) nn$no_grad(function() {
        x <- autoencoder_flatten(images)
        n <- autoencoder_undercomplete(model$parameters, x)
        list(n$code, n$reconstruction, autoencoder_reconstruction_mse(n$reconstruction, x))
    })
    model$decode <- function(codes) nn$no_grad(function()
        nn$dense(model$parameters, "decoder", codes, 784L, nn$Activation$Sigmoid))
    model
}

# Restores best.npz (a training port) or model_arrays.npz (tflow); returns the file used.
autoencoder_load_frozen <- function(store, directory, what) {
    for (name in c("best.npz", "model_arrays.npz")) {
        file <- file.path(directory, name)
        if (file.exists(file)) {
            store$load(file)
            return(file)
        }
    }
    stop(what, ": no best.npz/model_arrays.npz in ", directory, " (run the training port first)", call. = FALSE)
}

# tflow's case_<k>.png: row b (0-based) of every panel, 3x nearest, 4 px gray 220 gaps, RGB.
autoencoder_case_strip3 <- function(path, panels, b) {
    tiles <- lapply(panels, function(p) nn$reshape(autoencoder_row(p, b), c(28L, 28L, 1L)))
    nn$save_image_grid(nn$tile(nn$stack(nn$tensor_vector(tiles), 0L), c(1L, 1L, 1L, 3L)), path, length(tiles),
                       0, 1, 4L, 220 / 255, 3L)
}

# [n, 784] rows, `columns` per row, 3x nearest, 4 px gray 220 gaps, RGB.
autoencoder_mosaic3 <- function(path, values, columns) {
    n <- values$size(0L)
    nn$save_image_grid(nn$tile(nn$reshape(values, c(n, 28L, 28L, 1L)), c(1L, 1L, 1L, 3L)), path, as.integer(columns),
                       0, 1, 4L, 220 / 255, 3L)
}
