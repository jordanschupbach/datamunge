# Native R NN example. Tied-weight autoencoder (one live encoder matrix, transposed decoder) with an untied control.
# CMD: nix develop .#r --command Rscript examples/r/nn/2_autoencoders/2.1_deterministic_autoencoders/tied_autoencoder.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

# ONE live 784x32 parameter for encoder AND transposed decoder; two independent biases.
# Three default epochs, matched-initial untied control, and explicit gradient-path checks.
#
# Port of tflow's tied_autoencoder (model: examples/nn/common/tied_autoencoder.hpp).
# Preparation (in ~/projects/tflow_examples): just autoencoder-assets; data are read from
# $DATAMUNGE_DATA_DIR/saliency. Outputs (tied/ and untied/ training records, initial_forward.npz,
# data.npz, index.html and digit<k>.png galleries) go to output/nn/models/tied_autoencoder.
# The tie is the decoder weight being recomputed as transpose(encoder/W) on every forward pass
# (no decoder parameter exists); it is checked coordinate-by-coordinate after every update.
# Options: epochs=N data=PATH out=PATH init=PATH (tied initial parameters; the untied control
# always starts from their exact transpose).

nn_helper("checks")
nn_helper("autoencoder")

fmt <- function(x) sprintf("%g", x)
f32 <- function(x) readBin(writeBin(x, raw(), size = 4L), "double", size = 4L, n = length(x))

# code = ReLU(x W + eb); reconstruction = sigmoid(code D + db), D = transpose(W) when tied.
tied_autoencoder <- function(p, image, features = 784L, latent = 32L, tied = TRUE) {
    nn_require(features > 1L && latent > 0L && latent < features, "positive strict undercomplete tied architecture")
    limit <- sqrt(6 / (features + latent))
    out <- list()
    out$encoder_weight <- p$get("encoder/W", c(features, latent), nn$init$uniform(limit))
    eb <- p$get("encoder/b", latent, nn$init$zeros())
    out$code <- nn$relu(nn$bias_add(nn$matmul(image, out$encoder_weight), eb))
    out$decoder_weight <- if (tied) nn$transpose(out$encoder_weight, c(1L, 0L))
                          else p$get("decoder/W", c(latent, features), nn$init$uniform(limit))
    db <- p$get("decoder/b", features, nn$init$zeros())
    out$reconstruction <- nn$sigmoid(nn$bias_add(nn$matmul(out$code, out$decoder_weight), db))
    out
}

# Parameters (seed 1853) + optional Adam of the tied or untied model (tied_autoencoder.hpp).
tied_model <- function(tied = TRUE, training = TRUE) {
    model <- new.env()
    model$tied <- tied
    model$parameters <- nn$ParameterStore(1853L)
    nn$no_grad(function() tied_autoencoder(model$parameters, nn$zeros(c(1L, 784L)), 784L, 32L, tied))
    model$optimizer <- if (training) autoencoder_make_adam(model$parameters) else NULL
    model$net <- function(images) tied_autoencoder(model$parameters, autoencoder_flatten(images), 784L, 32L, tied)
    # list(code, reconstruction, loss)
    model$evaluate <- function(images) nn$no_grad(function() {
        n <- model$net(images)
        list(n$code, n$reconstruction, autoencoder_reconstruction_mse(n$reconstruction, autoencoder_flatten(images)))
    })
    model$decode <- function(codes) nn$no_grad(function() {
        d <- if (tied) nn$transpose(model$parameters$at("encoder/W"), c(1L, 0L)) else model$parameters$at("decoder/W")
        nn$sigmoid(nn$bias_add(nn$matmul(codes, d), model$parameters$at("decoder/b")))
    })
    # Loads every parameter by name (all must be supplied with matching shapes).
    model$load <- function(arrays) {
        nn_require(length(arrays) == model$parameters$size(), "all initial model tensors supplied")
        for (name in model$parameters$names()) {
            value <- autoencoder_find(arrays, name)
            nn_require(value$dtype() == nn$DType$Float32 &&
                       identical(as.numeric(value$shape()), as.numeric(model$parameters$at(name)$shape())),
                       "matching initial shape/dtype")
            model$parameters$set(name, value)
        }
    }
    # The decoder weight the forward pass actually uses.
    model$decoder_weight <- function() nn$no_grad(function() model$net(nn$zeros(c(1L, 784L)))$decoder_weight$contiguous())
    model$check_tie <- function() {
        nn_require(!model$parameters$contains("decoder/W") && model$parameters$size() == 3L,
                   "decoder graph directly transposes same encoder parameter")
        d <- model$decoder_weight()
        nn_require(identical(as.numeric(d$shape()), c(32, 784)), "actual decoder transpose shape")
        decoder <- matrix(d$values(), nrow = 32L, byrow = TRUE)
        encoder <- matrix(model$parameters$at("encoder/W")$contiguous()$values(), nrow = 784L, byrow = TRUE)
        nn_require(identical(decoder, t(encoder)), "ALL25088current decoder coordinates equal current encoder transpose")
    }
    model
}

tied_fixtures <- function() {
    p <- nn$ParameterStore(1861L)
    nn$no_grad(function() tied_autoencoder(p, nn$zeros(c(1L, 3L)), 3L, 2L, TRUE))
    w <- f32(c(0.2, 0.1, -0.3, 0.4, 0.5, -0.2))
    eb <- f32(c(0.1, 0.2))
    db <- f32(c(-0.1, 0.05, 0.2))
    inputs <- f32(c(0.1, 0.2, 0.7, 0.8, 0.3, 0.4))
    p$set("encoder/W", nn$from_vector(w, c(3L, 2L)))
    p$set("encoder/b", nn$from_vector(eb, 2L))
    p$set("decoder/b", nn$from_vector(db, 3L))
    x <- nn$from_vector(inputs, c(2L, 3L))
    net <- tied_autoencoder(p, x, 3L, 2L, TRUE)
    loss <- autoencoder_reconstruction_mse(net$reconstruction, x)
    gradient <- nn$grad(loss, p$parameters())
    code <- net$code$values()
    recon <- net$reconstruction$values()
    gw <- gradient[[1L]]$values()
    geb <- gradient[[2L]]$values()
    gdb <- gradient[[3L]]$values()
    W <- function(i, j) w[(i - 1L) * 2L + j]
    encoder <- matrix(0, 3L, 2L)
    decoder <- matrix(0, 3L, 2L)
    ge <- numeric(2L)
    gd <- numeric(3L)
    mse <- 0
    for (b in 0L:1L) {
        c <- eb
        delta <- numeric(3L)
        for (j in 1L:2L) {
            for (i in 1L:3L) c[j] <- c[j] + inputs[b * 3L + i] * W(i, j)
            nn_require(c[j] > 0, "toy avoids ReLU kink")
            nn_near(code[b * 2L + j], c[j], 5e-8, "all4toy codes")
        }
        for (i in 1L:3L) {
            value <- db[i]
            for (j in 1L:2L) value <- value + c[j] * W(i, j)
            y <- 1 / (1 + exp(-value))
            error <- y - inputs[b * 3L + i]
            nn_near(recon[b * 3L + i], y, 8e-8, "all6toy sigmoid predictions")
            mse <- mse + error * error / 6
            delta[i] <- error / 3 * y * (1 - y)
            gd[i] <- gd[i] + delta[i]
            for (j in 1L:2L) decoder[i, j] <- decoder[i, j] + delta[i] * c[j]
        }
        for (j in 1L:2L) {
            dz <- 0
            for (i in 1L:3L) dz <- dz + delta[i] * W(i, j)
            ge[j] <- ge[j] + dz
            for (i in 1L:3L) encoder[i, j] <- encoder[i, j] + inputs[b * 3L + i] * dz
        }
    }
    nn_require(p$size() == 3L && p$count() == 11, "one sharedmatrix plus two biases,not a copied decoder parameter")
    nn_near(loss$item(), mse, 2e-8, "known tied MSE")
    for (i in 1L:3L) {
        nn_near(gdb[i], gd[i], 3e-8, "all3decoder bias derivatives")
        for (j in 1L:2L)
            nn_near(gw[(i - 1L) * 2L + j], encoder[i, j] + decoder[i, j], 3e-8,
                    "ALL6shared derivatives equal encoder-path PLUS decoder-path")
    }
    for (j in 1L:2L) nn_near(geb[j], ge[j], 3e-8, "all2encoder bias derivatives")
    nn_require(sum(abs(encoder)) > 1e-3 && sum(abs(decoder)) > 1e-3,
               "neither shared gradient contribution is silently missing")
    cat("known tied toy4codes/6predictions/MSE/ALL11derivatives/BOTHsharedmatrix contributions passed\n")
}

tied_shift <- function(codes) {
    n <- codes$size(0L)
    nn$take_rows(codes, c(seq_len(n - 1L), 0L))$contiguous()
}

# Adam training of the tied or untied model from matched initial arrays, then a fresh-restore evaluation.
tied_train_variant <- function(tied, pools, epochs, root, initial) {
    name <- if (tied) "tied" else "untied"
    directory <- file.path(root, name)
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    model <- tied_model(tied)
    model$load(initial)
    nn_require(model$parameters$size() == (if (tied) 3L else 4L) &&
               model$parameters$count() == (if (tied) 25904 else 50992), "independent tied/untied parameter census")
    if (tied) model$check_tie()
    nn$save_npz(file.path(directory, "initial_arrays.npz"), autoencoder_arrays(model$parameters))
    order <- nn$BatchOrder(1000L, 32L, 1867L)
    fit_source <- as.integer(pools[[1L]]$source$values())
    best_path <- file.path(directory, "best.npz")
    best <- Inf
    selected <- 0L
    step <- 0L
    history <- file(file.path(directory, "history.tsv"), "w")
    batches <- file(file.path(directory, "batches.tsv"), "w")
    on.exit({
        close(history)
        close(batches)
    }, add = TRUE)
    writeLines("epoch\tfit_preupdate_mse\tvalidation_mse", history)
    writeLines("epoch\tstep\trow\tsource_id", batches)
    started <- proc.time()[["elapsed"]]
    for (epoch in seq_len(epochs)) {
        order$begin_epoch()
        total <- 0
        seen <- 0L
        repeat {
            rows <- order$`_next`()
            if (!length(rows)) break
            input <- nn$take_rows(pools[[1L]]$images, rows)
            # Forward (live transposed decoder when tied), reconstruction MSE, gradients through BOTH paths.
            net <- model$net(input)
            loss <- autoencoder_reconstruction_mse(net$reconstruction, autoencoder_flatten(input))
            if (step == 0L) {
                keep <- nn$GradOptions()
                keep$retain_graph <- TRUE
                nn$save_npz(file.path(directory, "first_gradients.npz"),
                    autoencoder_check_gradients(model$parameters, nn$grad(loss, model$parameters$parameters(), keep),
                                                "all model MSE derivatives finite",
                                                "all model tensors receive full MSE gradients"))
                nn$save_npz(file.path(directory, "first_batch.npz"), list(images = input))
            }
            value <- loss$item()
            nn_require(is.finite(value) && value >= 0, "finite pre-update MSE")
            model$optimizer$minimize(loss)
            total <- total + length(rows) * value
            seen <- seen + length(rows)
            writeLines(sprintf("%d\t%d\t%d\t%d", epoch, step, rows, fit_source[rows + 1L]), batches)
            if (tied) model$check_tie()
            if (step == 0L) nn$save_npz(file.path(directory, "first_update_arrays.npz"), autoencoder_arrays(model$parameters))
            step <- step + 1L
        }
        nn_require(seen == 1000L, "all fitting records exactly once per epoch")
        mse <- autoencoder_mse(model$evaluate(pools[[2L]]$images)[[2L]], pools[[2L]]$images)
        writeLines(sprintf("%d\t%.17g\t%.17g", epoch, total / seen, mse), history)
        nn$save_npz(file.path(directory, paste0("epoch", epoch, "_arrays.npz")), autoencoder_arrays(model$parameters))
        if (mse < best) {
            best <- mse
            selected <- epoch
            model$parameters$save(best_path)  # common validation reconstruction MSE selects each model
        }
        cat(sprintf("%s epoch%d:validationMSE=%s\n", name, epoch, fmt(mse)))
    }
    seconds <- proc.time()[["elapsed"]] - started
    deployed <- tied_model(tied, FALSE)
    deployed$parameters$load(best_path)  # fresh tied/untied inference restore
    if (tied) deployed$check_tie()
    before <- autoencoder_snapshot(deployed$parameters)
    arrays <- list()
    splits <- c("fit", "validation", "test")
    for (split in 1L:3L) {
        out <- deployed$evaluate(pools[[split]]$images)
        arrays[[paste0(splits[split], "_code")]] <- out[[1L]]
        arrays[[paste0(splits[split], "_reconstruction")]] <- out[[2L]]
        if (split == 2L) nn_near(autoencoder_mse(out[[2L]], pools[[split]]$images), best, 1e-8, "fresh selected validation MSE")
    }
    test <- deployed$evaluate(pools[[3L]]$images)
    repeated <- deployed$evaluate(pools[[3L]]$images)
    nn_require(identical(autoencoder_bytes(test[[2L]]), autoencoder_bytes(repeated[[2L]])) &&
               identical(autoencoder_bytes(deployed$decode(test[[1L]])), autoencoder_bytes(test[[2L]])),
               "deterministic exact encode-decode")
    arrays$zero_reconstruction <- deployed$decode(nn$zeros(c(200L, 32L)))
    arrays$shuffled_reconstruction <- deployed$decode(tied_shift(test[[1L]]))
    one <- deployed$evaluate(autoencoder_rows(pools[[3L]], 0L)$images)
    error <- autoencoder_max_abs_diff(one[[2L]], test[[2L]], 784L)
    nn_require(error < 1e-6, "single/batch tied or untied inference parity")
    nn_require(identical(before, autoencoder_snapshot(deployed$parameters)),
               "all deployed parameters immutable through code controls")
    nn$save_npz(file.path(directory, "outputs.npz"), arrays)
    nn$save_npz(file.path(directory, "model_arrays.npz"), autoencoder_arrays(deployed$parameters))
    nn$save_npz(file.path(directory, "decoder_weight.npz"), list(decoder_weight = deployed$decoder_weight()))
    test_mse <- autoencoder_mse(test[[2L]], pools[[3L]]$images)
    writeLines(sprintf(paste0('{"tied":%s,"parameters":%d,"epochs":%d,"updates":%d,"selected_epoch":%d,',
                              '"test_mse":%.17g,"zero_mse":%.17g,"shuffled_mse":%.17g,"single_batch_max_error":%.17g,',
                              '"training_seconds":%.17g}'),
                       if (tied) "true" else "false", as.integer(model$parameters$count()), epochs, step, selected,
                       test_mse, autoencoder_mse(arrays$zero_reconstruction, pools[[3L]]$images),
                       autoencoder_mse(arrays$shuffled_reconstruction, pools[[3L]]$images), error, seconds),
               file.path(directory, "metrics.json"))
    cat(sprintf("%s:selected%d,parameters=%d,testMSE=%s,freshrestore/decode/repeat/immutable parameters passed\n",
                name, selected, as.integer(model$parameters$count()), fmt(test_mse)))
    seconds
}

main <- function(arguments = commandArgs(TRUE)) {
    args <- nn_options(arguments, list(epochs = 3L, data = file.path(nn_data_root(), "saliency"), out = "", init = "",
                                       full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs <- args$epochs
    root <- if (nzchar(args$out)) args$out else file.path(nn_output_root(args$full), "models/tied_autoencoder")
    nn_require(epochs > 0L && epochs <= 100L, "three default epochs or explicit1..100")
    tied_fixtures()

    pools <- autoencoder_load_saliency(args$data)
    dir.create(root, recursive = TRUE, showWarnings = FALSE)
    factory <- tied_model(TRUE, FALSE)
    if (nzchar(args$init)) factory$parameters$load(args$init)
    tied_initial <- autoencoder_arrays(factory$parameters)
    transposed <- nn$transpose(tied_initial[["encoder/W"]], c(1L, 0L))$contiguous()
    untied_initial <- c(tied_initial, list("decoder/W" = transposed))
    initial_control <- tied_model(FALSE, FALSE)
    initial_control$load(untied_initial)
    initial_probe <- autoencoder_rows(pools[[1L]], 0L:7L)$images
    initial_t <- factory$evaluate(initial_probe)
    initial_u <- initial_control$evaluate(initial_probe)
    nn_require(identical(autoencoder_bytes(initial_t[[1L]]), autoencoder_bytes(initial_u[[1L]])) &&
               identical(autoencoder_bytes(initial_t[[2L]]), autoencoder_bytes(initial_u[[2L]])),
               "native matched initial codes AND reconstructions are BITEXACT")
    nn$save_npz(file.path(root, "initial_forward.npz"),
                list(images = initial_probe, tied_code = initial_t[[1L]], untied_code = initial_u[[1L]],
                     tied_reconstruction = initial_t[[2L]], untied_reconstruction = initial_u[[2L]]))
    seconds <- tied_train_variant(TRUE, pools, epochs, root, tied_initial)
    seconds <- seconds + tied_train_variant(FALSE, pools, epochs, root, untied_initial)
    t_initial <- nn$load_npz(file.path(root, "tied", "initial_arrays.npz"))
    u_initial <- nn$load_npz(file.path(root, "untied", "initial_arrays.npz"))
    for (key in names(t_initial))
        nn_require(identical(autoencoder_bytes(autoencoder_find(u_initial, key)), autoencoder_bytes(t_initial[[key]])),
                   "matched initial encoder/bias bytes")
    nn_require(identical(autoencoder_bytes(autoencoder_find(u_initial, "decoder/W")), autoencoder_bytes(transposed)),
               "untied starts EXACTLY same numerical transpose,then independently trains")
    autoencoder_save_data(file.path(root, "data.npz"), pools)
    autoencoder_copy_manifests(args$data, root)
    t <- nn$load_npz(file.path(root, "tied", "outputs.npz"))
    u <- nn$load_npz(file.path(root, "untied", "outputs.npz"))
    test_images <- pools[[3L]]$images
    mean <- autoencoder_constant(autoencoder_pixel_mean(pools[[1L]]$images), 200L)
    t_recon <- t$test_reconstruction
    tied_mse <- autoencoder_mse(t_recon, test_images)
    untied_mse <- autoencoder_mse(u$test_reconstruction, test_images)
    mean_mse <- autoencoder_mse(mean, test_images)
    html <- paste0(
        "<!doctype html><meta charset='utf-8'><title>Actual tied-weight autoencoder</title><style>body{font:16px ",
        "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:",
        "25px 0}</style><h1>Tied weights:ONE parameter,two gradient paths</h1><p>Original,tied,untied,fit-only mean,",
        "absolute tied error,tied zero-code,tied shuffled-code. Fixed[0,1]grayscale/nearest4,no error amplification ",
        "or quality filtering.</p><p>784->32ReLU->784sigmoid. Tied decoder is the LIVE transpose of the SAME encoder ",
        "parameter,never a copied decoder parameter. Separate encoder/decoder biases;25904parameters versus50992",
        "untied. After EVERY tied update all25088decoder coordinates are checked EXACTLY against current encoder ",
        "transpose. Shared MSE gradient sums encoder and decoder contributions.</p><p>Both models start numerically ",
        "IDENTICAL,including untied decoder initialized to encoder transpose. Same source/order/Adam.003/", epochs,
        "epochs;each selects on common validation MSE. Parameter counts differ,so this is a shared-constraint ",
        "comparison,not equal capacity or statistical proof of superiority. Continuous32float code is not bit-rate ",
        "compression. Blurry/distorted shapes can persist.</p><p>TestMSE tied=", fmt(tied_mse), ",untied=",
        fmt(untied_mse), ",fitmean=", fmt(mean_mse),
        ". Zero/shuffle controls expose input-specific encoding;no classification/denoising/generative claim.</p>")
    html <- c(html, autoencoder_digit_gallery(root, pools[[3L]],
        list(test_images, t_recon, u$test_reconstruction, mean,
             nn$abs(nn$sub(t_recon, autoencoder_flatten(test_images))), t$zero_reconstruction,
             t$shuffled_reconstruction)))
    writeLines(paste(html, collapse = ""), file.path(root, "index.html"))
    cat(sprintf(paste0("Actualsharedparameter/transpose afterEVERYupdate/BOTHgradientpaths/numericallymatched initial ",
                       "untiedcontrol verified.\ntraining time %s s\nArtifacts:%s/index.html\n"), fmt(seconds), root))
    0L
}

if (sys.nframe() == 0L) quit(status = main())
