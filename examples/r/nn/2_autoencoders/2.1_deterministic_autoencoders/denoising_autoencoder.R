# Native R NN example. Corrupted-input/clean-target denoising autoencoder with a matched ordinary control.
# CMD: nix develop .#r --command Rscript examples/r/nn/2_autoencoders/2.1_deterministic_autoencoders/denoising_autoencoder.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

# Actual corrupted-input/CLEAN-target autoencoder with fresh epoch noise and matched ordinary AE.
# Three epochs/default; fixed validation/test Gaussian banks, clean/low/strong/secondseed controls.
#
# Port of tflow's denoising_autoencoder. Preparation (in ~/projects/tflow_examples):
# just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
# (denoising/ and ordinary/ training records, data.npz with every noise bank, the
# per-epoch training corruptions, test_metrics.tsv, outputs.npz, index.html and
# digit<k>.png galleries) go to output/nn/models/denoising_autoencoder. The Gaussian
# banks (MT19937/Box-Muller) and therefore every corrupted image are bit-identical to
# tflow's. Options: epochs=N data=PATH out=PATH init=PATH (start both models from
# saved parameters; by default nn's own initialiser is used, identically for both).

nn_helper("checks")
nn_helper("autoencoder")
nn_helper("denoising_autoencoder")

sigma <- 0.35

fmt <- function(x) sprintf("%g", x)

denoising_fixtures <- function() {
    clean <- nn$from_vector(c(0, 0.2, 0.8, 1), c(2L, 2L))
    z <- nn$from_vector_double(c(-2, 1, -1, 2), c(1L, 2L, 2L))
    noisy <- autoencoder_corrupt_for_denoising(clean, z, 0.3)
    expected <- nn$from_vector(c(0, 0.5, 0.5, 1), 4L)$values()
    nv <- noisy$values()
    for (i in 1:4) nn_near(nv[i], expected[i], 6e-8, "known additiveGaussian/float32/clamp actual pixels")
    nn_require(identical(autoencoder_bytes(autoencoder_corrupt_for_denoising(clean, z, 0)), autoencoder_bytes(clean)),
               "zero sigma exact identity")
    a <- autoencoder_gaussian_noise(clean$shape(), 1L, 1673L)
    b <- autoencoder_gaussian_noise(clean$shape(), 1L, 1673L)
    c <- autoencoder_gaussian_noise(clean$shape(), 1L, 1674L)
    nn_require(identical(a$values(), b$values()) && !identical(a$values(), c$values()),
               "Gaussian seed repeat/secondseed distinction")
    nn_throws(autoencoder_corrupt_for_denoising(clean, z, -1), "negative corruption sigma rejected")
    prediction <- noisy$clone()$requires_grad_()
    loss <- autoencoder_reconstruction_mse(prediction, clean)
    gradient <- nn$grad(loss, list(prediction))[[1L]]$values()
    nn_near(loss$item(), 0.045, 2e-8, "nonzero error to CLEAN target despite prediction exactly noisy input")
    cv <- clean$values()
    for (i in 1:4) nn_near(gradient[i], (nv[i] - cv[i]) / 2, 2e-8, "all4clean-target MSE derivatives")
    cat("corruption/clamp/zero/seed/invalidsigma and CLEAN-vs-noisy target/MSE/4derivative fixtures passed\n")
}

# Trains the denoising (corrupted input) or ordinary (clean input) model on CLEAN targets.
denoising_train_variant <- function(name, pools, validation_noisy, epochs, root, init) {
    directory <- file.path(root, name)
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    model <- denoising_autoencoder_model()
    if (nzchar(init)) model$parameters$load(init)
    nn_require(model$parameters$count() == 50992 && model$parameters$size() == 4L, "784->32ReLU->784sigmoid inventory")
    nn$save_npz(file.path(directory, "initial_arrays.npz"), autoencoder_arrays(model$parameters))
    order <- nn$BatchOrder(1000L, 32L, 1663L)
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
    writeLines("epoch\tfit_preupdate_clean_target_mse\tvalidation_denoising_mse", history)
    writeLines("epoch\tstep\trow\tsource_id", batches)
    started <- proc.time()[["elapsed"]]
    for (epoch in seq_len(epochs)) {
        corruption <- NULL
        if (name == "denoising") {
            bank <- autoencoder_gaussian_noise(pools[[1L]]$images$shape(), 1L, 1671L + epoch)
            corruption <- autoencoder_corrupt_for_denoising(pools[[1L]]$images, bank, sigma)
            nn$save_npz(file.path(root, paste0("train_corruption_epoch", epoch, ".npz")),
                        list(standard_noise = bank, corrupted = corruption))
        }
        order$begin_epoch()
        total <- 0
        seen <- 0L
        repeat {
            rows <- order$`_next`()
            if (!length(rows)) break
            clean <- nn$take_rows(pools[[1L]]$images, rows)
            input <- if (name == "ordinary") clean else nn$take_rows(corruption, rows)
            # Forward the (corrupted or clean) input, score it against the CLEAN target.
            net <- model$forward(input)
            loss <- autoencoder_reconstruction_mse(net$reconstruction, autoencoder_flatten(clean))
            if (step == 0L) {
                keep <- nn$GradOptions()
                keep$retain_graph <- TRUE
                gradients <- autoencoder_check_gradients(model$parameters,
                    nn$grad(loss, model$parameters$parameters(), keep),
                    "every actual clean-target parameter derivative finite",
                    "all4parameters receive CLEAN-target loss gradients")
                nn$save_npz(file.path(directory, "first_gradients.npz"), gradients)
                nn$save_npz(file.path(directory, "first_batch.npz"), list(input = input, clean_target = clean))
            }
            value <- loss$item()
            nn_require(is.finite(value) && value >= 0, "finite pre-update MSE")
            model$optimizer$minimize(loss)
            total <- total + length(rows) * value
            seen <- seen + length(rows)
            writeLines(sprintf("%d\t%d\t%d\t%d", epoch, step, rows, fit_source[rows + 1L]), batches)
            if (step == 0L) nn$save_npz(file.path(directory, "first_update_arrays.npz"), autoencoder_arrays(model$parameters))
            step <- step + 1L
        }
        nn_require(seen == 1000L, "all1000fitting clean images once per epoch")
        value <- model$loss(validation_noisy, pools[[2L]]$images)
        writeLines(sprintf("%d\t%.17g\t%.17g", epoch, total / seen, value), history)
        nn$save_npz(file.path(directory, paste0("epoch", epoch, "_arrays.npz")), autoencoder_arrays(model$parameters))
        if (value < best) {
            best <- value
            selected <- epoch
            model$parameters$save(best_path)  # both variants select on SAME fixed noisy validation-to-clean MSE
        }
        cat(sprintf("%s epoch%d:fixed-noisy validation CLEAN-target MSE=%s\n", name, epoch, fmt(value)))
    }
    seconds <- proc.time()[["elapsed"]] - started
    nn$save_npz(file.path(directory, "last_arrays.npz"), autoencoder_arrays(model$parameters))
    model$parameters$load(best_path)  # restore selected denoising-task checkpoint
    nn$save_npz(file.path(directory, "model_arrays.npz"), autoencoder_arrays(model$parameters))
    writeLines(sprintf(paste0('{"epochs":%d,"updates":%d,"selected_epoch":%d,"validation_clean_target_mse":%.17g,',
                              '"training_corrupted_inputs":%s,"training_seconds":%.17g}'),
                       epochs, step, selected, best, if (name == "denoising") "true" else "false", seconds),
               file.path(directory, "metrics.json"))
    seconds
}

main <- function(arguments = commandArgs(TRUE)) {
    args <- nn_options(arguments, list(epochs = 3L, data = file.path(nn_data_root(), "saliency"), out = "", init = "",
                                       full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs <- args$epochs
    root <- if (nzchar(args$out)) args$out else file.path(nn_output_root(args$full), "models/denoising_autoencoder")
    nn_require(epochs > 0L && epochs <= 100L, "three default epochs or explicit1..100")
    denoising_fixtures()

    pools <- autoencoder_load_saliency(args$data)
    dir.create(root, recursive = TRUE, showWarnings = FALSE)
    test_images <- pools[[3L]]$images
    validation_bank <- autoencoder_gaussian_noise(pools[[2L]]$images$shape(), 1L, 1709L)
    test_bank <- autoencoder_gaussian_noise(test_images$shape(), 1L, 1711L)
    second_bank <- autoencoder_gaussian_noise(test_images$shape(), 1L, 1717L)
    validation_noisy <- autoencoder_corrupt_for_denoising(pools[[2L]]$images, validation_bank, sigma)
    test_noisy <- autoencoder_corrupt_for_denoising(test_images, test_bank, sigma)
    test_low <- autoencoder_corrupt_for_denoising(test_images, test_bank, 0.15)
    test_strong <- autoencoder_corrupt_for_denoising(test_images, test_bank, 0.6)
    test_second <- autoencoder_corrupt_for_denoising(test_images, second_bank, sigma)
    autoencoder_save_data(file.path(root, "data.npz"), pools,
        list(validation_standard_noise = validation_bank, test_standard_noise = test_bank,
             second_standard_noise = second_bank, validation_corrupted = validation_noisy,
             test_corrupted = test_noisy, test_low = test_low, test_strong = test_strong, test_second = test_second))
    seconds <- denoising_train_variant("denoising", pools, validation_noisy, epochs, root, args$init)
    seconds <- seconds + denoising_train_variant("ordinary", pools, validation_noisy, epochs, root, args$init)
    initial_d <- nn$load_npz(file.path(root, "denoising", "initial_arrays.npz"))
    initial_o <- nn$load_npz(file.path(root, "ordinary", "initial_arrays.npz"))
    for (key in names(initial_d))
        nn_require(identical(autoencoder_bytes(autoencoder_find(initial_o, key)), autoencoder_bytes(initial_d[[key]])),
                   "matched ordinary and denoising initial parameters")

    deployed <- denoising_autoencoder_model(FALSE)
    ordinary <- denoising_autoencoder_model(FALSE)
    deployed$parameters$load(file.path(root, "denoising", "best.npz"))  # fresh denoising inference restore
    ordinary$parameters$load(file.path(root, "ordinary", "best.npz"))  # fresh ordinary control restore
    before <- autoencoder_snapshot(deployed$parameters)
    ordinary_before <- autoencoder_snapshot(ordinary$parameters)
    # std::map order of the C++ cases.
    cases <- list(clean = test_images, low = test_low, noisy = test_noisy, second = test_second, strong = test_strong)
    outputs <- list()
    scores <- list()
    metrics <- file(file.path(root, "test_metrics.tsv"), "w")
    writeLines("case\tcorrupted_to_clean_mse\tdenoising_to_clean_mse\tordinary_to_clean_mse", metrics)
    for (name in names(cases)) {
        input <- cases[[name]]
        d <- deployed$predict(input)
        o <- ordinary$predict(input)
        outputs[[paste0(name, "_denoising_code")]] <- d[[1L]]
        outputs[[paste0(name, "_denoising_reconstruction")]] <- d[[2L]]
        outputs[[paste0(name, "_ordinary_reconstruction")]] <- o[[2L]]
        raw <- autoencoder_mse(input, test_images)
        denoised <- autoencoder_mse(d[[2L]], test_images)
        control <- autoencoder_mse(o[[2L]], test_images)
        scores[[name]] <- c(raw, denoised, control)
        writeLines(sprintf("%s\t%.17g\t%.17g\t%.17g", name, raw, denoised, control), metrics)
        cat(sprintf("%s:raw/denoised/ordinary CLEAN-target MSE=%s/%s/%s\n", name, fmt(raw), fmt(denoised), fmt(control)))
    }
    close(metrics)
    result <- deployed$predict(test_noisy)
    repeated <- deployed$predict(test_noisy)
    nn_require(identical(autoencoder_bytes(result[[2L]]), autoencoder_bytes(repeated[[2L]])),
               "deterministic target-free denoising inference")
    nn_require(identical(autoencoder_bytes(deployed$decode(result[[1L]])), autoencoder_bytes(result[[2L]])),
               "explicit encode-decode exact")
    one <- deployed$predict(nn$take_rows(test_noisy, 0L))
    error <- autoencoder_max_abs_diff(one[[2L]], result[[2L]], 784L)
    nn_require(error < 1e-6, "single/batch denoising parity")
    nn_require(identical(before, autoencoder_snapshot(deployed$parameters)) &&
               identical(ordinary_before, autoencoder_snapshot(ordinary$parameters)),
               "all2x4model parameters immutable through test controls")
    nn$save_npz(file.path(root, "outputs.npz"), outputs)
    autoencoder_copy_manifests(args$data, root)

    mean <- autoencoder_constant(autoencoder_pixel_mean(pools[[1L]]$images), 200L)
    baseline <- autoencoder_mse(mean, test_images)
    html <- c(paste0(
        "<!doctype html><meta charset='utf-8'><title>Clean-target denoising autoencoder</title><style>body{font:",
        "16px system-ui;max-width:1100px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{",
        "margin:25px 0}</style><h1>Denoising:corrupted inputs,CLEAN targets</h1><p>Clean original,corrupted ",
        "sigma.35,denoised,ordinary AE on SAME corruption,absolute denoising error,fit-only mean,denoiser on clean ",
        "input,denoiser on sigma.6. Fixed[0,1]grayscale/nearest4,no error amplification or quality filtering.</p>",
        "<p>784->32ReLU->784sigmoid/50992params,Adam.003,batch32,", epochs,
        "epochs each,matched initial parameters and order1663. Denoiser receives NEW epoch-specific Gaussian ",
        "draws,ordinary control trains clean->clean. BOTH select on SAME fixed noisy-validation-to-clean MSE. No ",
        "labels enter training or selection;NO clean target is required for inference.</p><p>Noise MT19937/",
        "Box-Muller standardGaussian,sigma.35 in original unit-pixel coordinates,castfloat32 THEN clamp[0,1]. ",
        "Clipping changes the corruption distribution and is NOT inverted. Epoch seeds1672onward;validation1709/",
        "test1711/second1717 independent. Low.15 and strong.6 share test standard draws for paired amplitude ",
        "controls. Clean-input test has sigma0. Raw corruptions,allnoise banks and clean targets saved.</p><p>",
        "Fit-only mean MSE=", fmt(baseline),
        ". All raw/denoised/ordinary CLEAN-target MSE values,including LOW-noise failure cases,in ",
        "test_metrics.tsv. A short undercomplete model may blur digit identity or be WORSE than leaving mild noise ",
        "unchanged. One chosen corruption family and tiny data budget do not demonstrate universal denoising,",
        "classification,generative quality or bit-rate compression.</p>"),
        "<table><tr><th>Case</th><th>Leave input unchanged</th><th>Noise-trained AE</th><th>Ordinary AE</th></tr>")
    for (name in names(scores))
        html <- c(html, sprintf("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>", name,
                                fmt(scores[[name]][1L]), fmt(scores[[name]][2L]), fmt(scores[[name]][3L])))
    verdict <- if (scores$noisy[2L] < scores$noisy[3L]) "noise-trained lower MSE" else "ordinary lower MSE in this short run"
    html <- c(html, paste0("</table><p>All values compare against CLEAN targets. Noise-trained versus ordinary on main ",
                           "noise: ", verdict, ". This baseline comparison remains visible regardless of result.</p>"))
    denoised <- outputs$noisy_denoising_reconstruction
    html <- c(html, autoencoder_digit_gallery(root, pools[[3L]],
        list(test_images, test_noisy, denoised, outputs$noisy_ordinary_reconstruction,
             nn$abs(nn$sub(denoised, autoencoder_flatten(test_images))), mean,
             outputs$clean_denoising_reconstruction, outputs$strong_denoising_reconstruction)))
    writeLines(paste(html, collapse = ""), file.path(root, "index.html"))
    cat(sprintf(paste0("Freshrestore/target-freeinference/encode-decode/repeat/single-batch/immutable2x4weights passed,",
                       "maxbatcherror=%s\nFit-onlymean CLEAN-target MSE=%s;weak/mildnoise failures retained.\n",
                       "training time %s s\nArtifacts:%s/index.html\n"), fmt(error), fmt(baseline), fmt(seconds), root))
    0L
}

if (sys.nframe() == 0L) quit(status = main())
