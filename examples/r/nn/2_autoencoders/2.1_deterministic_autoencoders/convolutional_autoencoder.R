# Native R NN example. Spatial 7x7x4-code convolutional autoencoder with latent and baseline controls.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

# CMD: just r-example convolutional_autoencoder
# Actual spatial 7x7x4 code / nearest-repeat convolutional decoder, 741 learned parameters.
# Three default epochs, mean/initial/pool-only/zero/shuffled-code controls.
#
# Port of tflow's convolutional_autoencoder. Preparation (in ~/projects/tflow_examples):
# just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
# (history, npz arrays, metrics.json, index.html and digit<k>.png galleries) go to
# output/nn/models/convolutional_autoencoder. Options: epochs=N data=PATH out=PATH
# init=PATH (start from saved parameters, e.g. tflow's initial_arrays.npz, to
# follow its trajectory exactly; by default nn's own initialiser is used).

nn_helper("checks")
nn_helper("autoencoder")

fmt <- function(x) sprintf("%g", x)

convolutional_fixtures <- function() {
    pixels <- nn$from_vector((0:15) / 15, 16L)$values()
    x <- nn$from_vector(pixels, c(1L, 4L, 4L, 1L))$requires_grad_()
    code <- nn$avg_pool2d(x, 2L, 2L)
    enlarged <- nn$upsample_nearest(code, 2L)
    loss <- nn$mean(nn$square(nn$sub(enlarged, x)))
    gradient <- nn$grad(loss, list(x))[[1L]]$values()
    code_values <- code$values()
    enlarged_values <- enlarged$values()
    mse <- 0
    for (y in 0:3) {
        for (a in 0:3) {
            mean <- 0
            for (dy in 0:1) for (dx in 0:1) mean <- mean + pixels[((y %/% 2) * 2 + dy) * 4 + (a %/% 2) * 2 + dx + 1] / 4
            nn_near(code_values[(y %/% 2) * 2 + a %/% 2 + 1], mean, 4e-8, "all4pool means")
            nn_near(enlarged_values[y * 4 + a + 1], mean, 4e-8, "all16nearestrepeat pixels,not tile-whole-image")
            error <- pixels[y * 4 + a + 1] - mean
            mse <- mse + error * error / 16
            nn_near(gradient[y * 4 + a + 1], error / 8, 2e-8, "all16analytic pool+repeat MSE input derivatives")
        }
    }
    nn_near(loss$item(), mse, 2e-8, "all16spatial reconstruction MSE terms")
    cat("known4x4->2x2->4x4 allpool/repeatpixels/MSE/ALL16analytic gradients passed\n")
}

main <- function(arguments = commandArgs(TRUE)) {
    args <- nn_options(arguments, list(epochs = 3L, data = file.path(nn_data_root(), "saliency"), out = "", init = "",
                                       full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs <- args$epochs
    root <- if (nzchar(args$out)) args$out else file.path(nn_output_root(args$full), "models/convolutional_autoencoder")
    nn_require(epochs > 0L && epochs <= 100L, "three default epochs or explicit1..100")
    convolutional_fixtures()

    pools <- autoencoder_load_saliency(args$data)
    test_images <- pools[[3L]]$images
    model <- autoencoder_convolutional_model()
    if (nzchar(args$init)) model$parameters$load(args$init)
    nn_require(model$parameters$size() == 8L && model$parameters$count() == 741,
               "independent3x3kernels+biases741params/8tensors")
    dir.create(root, recursive = TRUE, showWarnings = FALSE)
    nn$save_npz(file.path(root, "initial_arrays.npz"), autoencoder_arrays(model$parameters))
    initial <- model$evaluate(test_images)
    order <- nn$BatchOrder(1000L, 32L, 1807L)
    fit_source <- as.integer(pools[[1L]]$source$values())
    best_path <- file.path(root, "best.npz")
    best <- Inf
    selected <- 0L
    step <- 0L
    history <- file(file.path(root, "history.tsv"), "w")
    batches <- file(file.path(root, "batches.tsv"), "w")
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
            # Forward pass, reconstruction MSE, then (first step) every gradient and one Adam step.
            net <- autoencoder_convolutional_autoencoder(model$parameters, input)
            loss <- model$loss(net$reconstruction, input)
            if (step == 0L) {
                keep <- nn$GradOptions()
                keep$retain_graph <- TRUE
                gradients <- autoencoder_check_gradients(model$parameters,
                    nn$grad(loss, model$parameters$parameters(), keep),
                    "all convolutional parameter gradients finite",
                    "all8convolutional tensors receive reconstruction gradients")
                nn$save_npz(file.path(root, "first_gradients.npz"), gradients)
                nn$save_npz(file.path(root, "first_batch.npz"), list(images = input))
            }
            value <- loss$item()
            nn_require(is.finite(value) && value >= 0, "finite pre-update MSE")
            model$optimizer$minimize(loss)
            total <- total + length(rows) * value
            seen <- seen + length(rows)
            writeLines(sprintf("%d\t%d\t%d\t%d", epoch, step, rows, fit_source[rows + 1L]), batches)
            if (step == 0L) nn$save_npz(file.path(root, "first_update_arrays.npz"), autoencoder_arrays(model$parameters))
            step <- step + 1L
        }
        nn_require(seen == 1000L, "each fitting record once per epoch including last8")
        validation <- model$evaluate(pools[[2L]]$images)
        value <- autoencoder_mse(validation[[2L]], pools[[2L]]$images)
        writeLines(sprintf("%d\t%.17g\t%.17g", epoch, total / seen, value), history)
        nn$save_npz(file.path(root, paste0("epoch", epoch, "_arrays.npz")), autoencoder_arrays(model$parameters))
        if (value < best) {
            best <- value
            selected <- epoch
            model$parameters$save(best_path)  # validation MSE alone selects the checkpoint
        }
        cat(sprintf("epoch%d:fitPreMSE=%s,validationMSE=%s\n", epoch, fmt(total / seen), fmt(value)))
    }
    close(history)
    close(batches)
    train_seconds <- proc.time()[["elapsed"]] - started
    model$parameters$load(best_path)  # restore selected weights for the checkpoint-gradient probes
    deployed <- autoencoder_convolutional_model(FALSE)
    deployed$parameters$load(best_path)  # fresh spatial inference restore
    before <- autoencoder_snapshot(deployed$parameters)
    arrays <- list()
    splits <- c("fit", "validation", "test")
    for (split in 1:3) {
        name <- splits[split]
        images <- pools[[split]]$images
        out <- deployed$evaluate(images)
        nn_require(identical(as.numeric(out[[1L]]$shape()), c(images$size(0L), 7, 7, 4)),
                   "actual spatial196-coordinate bottleneck")
        nn_require(identical(as.numeric(out[[2L]]$shape()), as.numeric(images$shape())),
                   "decoder restores original28x28x1 geometry")
        arrays[[paste0(name, "_images")]] <- images
        arrays[[paste0(name, "_source_ids")]] <- pools[[split]]$source
        arrays[[paste0(name, "_code")]] <- out[[1L]]
        arrays[[paste0(name, "_reconstruction")]] <- out[[2L]]
        if (split == 2L) nn_near(autoencoder_mse(out[[2L]], images), best, 1e-8, "fresh restored validation MSE")
    }
    arrays$test_labels <- pools[[3L]]$labels
    arrays$initial_test_reconstruction <- initial[[2L]]
    test <- deployed$evaluate(test_images)
    repeated <- deployed$evaluate(test_images)
    nn_require(identical(autoencoder_bytes(test[[2L]]), autoencoder_bytes(repeated[[2L]])),
               "deterministic spatial reconstruction")
    nn_require(identical(autoencoder_bytes(deployed$decode(test[[1L]])), autoencoder_bytes(test[[2L]])),
               "standalone convolutional decoder EXACT encode/decode")
    probe <- autoencoder_rows(pools[[3L]], 0:7)
    stages <- deployed$activations(probe$images)
    probe_net <- autoencoder_convolutional_autoencoder(model$parameters, probe$images)
    probe_gradients <- nn$grad(model$loss(probe_net$reconstruction, probe$images), model$parameters$parameters())
    nn$save_npz(file.path(root, "probe_gradients.npz"), setNames(probe_gradients, model$parameters$names()))
    expected <- list(c(8, 28, 28, 8), c(8, 14, 14, 8), c(8, 14, 14, 4), c(8, 7, 7, 4), c(8, 14, 14, 8), c(8, 28, 28, 1))
    for (k in 1:6) {
        nn_require(identical(as.numeric(stages[[k]]$shape()), expected[[k]]), "all six actual intermediate spatial shapes")
        arrays[[paste0("probe_stage", k - 1L)]] <- stages[[k]]
    }
    arrays$probe_images <- probe$images
    single <- deployed$evaluate(autoencoder_rows(pools[[3L]], 0L)$images)
    batch_error <- autoencoder_max_abs_diff(single[[2L]], test[[2L]], 784L)
    nn_require(batch_error < 2e-6, "single/batch convolutional inference parity")
    # Decode an all-zero code and each image's code cyclically shifted by one record.
    zero <- deployed$decode(nn$zeros(c(200L, 7L, 7L, 4L)))
    shuffled <- deployed$decode(nn$concat(nn$tensor_vector(list(nn$slice_axis(test[[1L]], 0L, 1L, 200L),
                                                                nn$slice_axis(test[[1L]], 0L, 0L, 1L))), 0L))
    arrays$zero_reconstruction <- zero
    arrays$shuffled_reconstruction <- shuffled
    mean4 <- nn$reshape(autoencoder_constant(autoencoder_pixel_mean(pools[[1L]]$images), 200L), c(200L, 28L, 28L, 1L))
    arrays$mean_reconstruction <- mean4
    # 4x4 average then nearest repeat: a 49-coordinate input-dependent baseline.
    pooled <- nn$upsample_nearest(nn$avg_pool2d(test_images, 4L, 4L), 4L)$contiguous()
    arrays$pool_repeat_reconstruction <- pooled
    nn_require(identical(before, autoencoder_snapshot(deployed$parameters)),
               "all8parameters immutable through inference/zero/shuffledcode controls")
    nn$save_npz(file.path(root, "convolutional.npz"), arrays)
    nn$save_npz(file.path(root, "model_arrays.npz"), autoencoder_arrays(deployed$parameters))
    autoencoder_copy_manifests(args$data, root)
    mse <- autoencoder_mse(test[[2L]], test_images)
    mean_mse <- autoencoder_mse(mean4, test_images)
    pool_mse <- autoencoder_mse(pooled, test_images)
    writeLines(sprintf(paste0('{"epochs":%d,"updates":%d,"selected_epoch":%d,"parameters":741,"test_mse":%.17g,',
                              '"mean_mse":%.17g,"pool_repeat_mse":%.17g,"initial_mse":%.17g,"zero_mse":%.17g,',
                              '"shuffled_mse":%.17g,"single_batch_max_error":%.17g}'),
                       epochs, step, selected, mse, mean_mse, pool_mse, autoencoder_mse(initial[[2L]], test_images),
                       autoencoder_mse(zero, test_images), autoencoder_mse(shuffled, test_images), batch_error),
               file.path(root, "metrics.json"))
    html <- paste0(
        "<!doctype html><meta charset='utf-8'><title>Convolutional autoencoder</title><style>body{font:16px ",
        "system-ui;max-width:1100px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:",
        "25px 0}</style><h1>Convolutional autoencoder: spatial bottleneck</h1><p>Original, trained, absolute ",
        "error, fit-only mean, 4x4 average/nearest repeat, untrained, zero code, cyclic shuffled code. Fixed [0,1] ",
        "grayscale/nearest 4x, no error amplification or quality filtering.</p><p>Conv3x3 same 1->8 ReLU, AvgPool2, ",
        "Conv3x3 same 8->4 ReLU, AvgPool2 gives a 7x7x4 code. Decoder nearest-repeat 2, Conv3x3 same 4->8 ReLU, ",
        "nearest-repeat 2, Conv3x3 same 8->1 sigmoid. No skips/dense bypass; 741 untied parameters, ", epochs,
        " epochs, batch 32, Adam .003. Clean inputs are targets; no labels enter training or validation-MSE ",
        "selection. The 196 continuous code values are 4x fewer coordinates, not a bit-rate/file compression ",
        "claim.</p><p>Held-out MSE trained=", fmt(mse), ", fit mean=", fmt(mean_mse), ", pool/repeat=", fmt(pool_mse),
        ". Pool/repeat is a 49-coordinate input-dependent baseline, not a same-capacity learned comparator. ",
        "Latent zero/shuffle and initial-model controls expose whether learned spatial information matters. MSE ",
        "alone does not establish shape fidelity, classification, denoising or a generative prior.</p>")
    error <- nn$abs(nn$sub(test[[2L]], test_images))
    labels <- as.integer(pools[[3L]]$labels$values())
    source <- as.integer(pools[[3L]]$source$values())
    for (digit in 0:9) {
        b <- which(labels == digit)[1L] - 1L
        panels <- lapply(list(test_images, test[[2L]], error, mean4, pooled, initial[[2L]], zero, shuffled),
                         function(t) autoencoder_row(t, b))
        file <- paste0("digit", digit, ".png")
        autoencoder_panel_row(file.path(root, file), panels)
        html <- paste0(html, sprintf("<section><h2>Digit %d, source %d</h2><img src='%s'></section>",
                                     digit, source[b + 1L], file))
    }
    writeLines(html, file.path(root, "index.html"))
    cat(sprintf(paste0("Convolutional741params/8tensors,selectedepoch=%d,testMSE=%s,fitmean=%s,pool/repeat=%s\n",
                       "Spatialall6shapes/freshrestore/exactdecode/repeat/single-batch/immutable8weights/",
                       "all8gradienttensors passed.\ntraining time %s s\nArtifacts:%s/index.html\n"),
                selected, fmt(mse), fmt(mean_mse), fmt(pool_mse), fmt(train_seconds), root))
    0L
}

if (sys.nframe() == 0L) quit(status = main())
