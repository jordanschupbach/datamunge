# Native R NN example. Lifetime winner-take-all autoencoder: training-only per-feature batch winners, matched plain control.
# CMD: nix develop .#r --command Rscript examples/r/nn/2_autoencoders/2.1_deterministic_autoencoders/winner_take_all_autoencoder.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

# Lifetime WTA: training-only per-feature batch winners, full ReLU deployment.
# Three default epochs per model; matched plain control.
#
# Port of tflow's winner_take_all_autoencoder (the model is examples/nn/common/competitive_autoencoder.hpp,
# mirrored by examples/r/nn/common/competitive_autoencoder.R and shared with
# k_sparse_autoencoder). Preparation (in ~/projects/tflow_examples): just autoencoder-assets; data are
# read from $DATAMUNGE_DATA_DIR/saliency. Outputs (competitive/ and plain/ training records,
# data.npz, index.html and digit<k>.png galleries) go to output/nn/models/winner_take_all_autoencoder.
# Options: epochs=N data=PATH out=PATH init=PATH (start both models from saved
# parameters; by default nn's own initialiser is used, identically for both models).

nn_helper("checks")
nn_helper("autoencoder")
nn_helper("competitive_autoencoder")

lifetime_competition <- TRUE

fmt <- function(x) sprintf("%g", x)

# Trains the competitive (or matched plain) model; returns its training seconds.
competitive_train_variant <- function(lifetime, plain, pools, epochs, root, initial) {
    directory <- file.path(root, if (plain) "plain" else "competitive")
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    model <- competitive_autoencoder_model(lifetime, plain)
    model$load(initial)
    nn_require(model$parameters$size() == 4L && model$parameters$count() == 101200,
               "hard-competition and matched plain parameter census")
    nn$save_npz(file.path(directory, "initial_arrays.npz"), autoencoder_arrays(model$parameters))
    initial_output <- model$evaluate(pools[[3L]]$images)
    nn$save_npz(file.path(directory, "initial_outputs.npz"),
                list(code = initial_output[[1L]], reconstruction = initial_output[[2L]]))
    order <- nn$BatchOrder(1000L, 32L, 1933L)
    fit_source <- as.integer(pools[[1L]]$source$values())
    best_path <- file.path(directory, "best.npz")
    label <- if (plain) "plain" else "competitive"
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
            images <- nn$take_rows(pools[[1L]]$images, rows)
            # Training forward (competition mask applied), reconstruction MSE against the input.
            x <- autoencoder_flatten(images)
            net <- model$forward(x)
            loss <- autoencoder_reconstruction_mse(net$training_reconstruction, x)
            if (step == 0L) {
                probe <- model$probe(images)
                competitive_autoencoder_check_support(probe, lifetime, plain)
                nn$save_npz(file.path(directory, "first_probe.npz"),
                            list(raw = probe[[1L]], mask = probe[[2L]], training_code = probe[[3L]],
                                 reconstruction = probe[[4L]]))
                keep <- nn$GradOptions()
                keep$retain_graph <- TRUE
                gradients <- autoencoder_check_gradients(model$parameters,
                    nn$grad(loss, model$parameters$parameters(), keep),
                    "all model MSE derivatives finite", "all model tensors receive full MSE gradients")
                nn$save_npz(file.path(directory, "first_gradients.npz"), gradients)
                nn$save_npz(file.path(directory, "first_batch.npz"), list(images = images))
            }
            if (step == 31L) {
                probe <- model$probe(images)
                competitive_autoencoder_check_support(probe, lifetime, plain)
                nn$save_npz(file.path(directory, "partial_probe.npz"),
                            list(images = images, raw = probe[[1L]], mask = probe[[2L]],
                                 training_code = probe[[3L]], reconstruction = probe[[4L]]))
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
        nn_require(seen == 1000L, "all fitting records exactly once per epoch")
        mse <- autoencoder_mse(model$evaluate(pools[[2L]]$images)[[2L]], pools[[2L]]$images)
        writeLines(sprintf("%d\t%.17g\t%.17g", epoch, total / seen, mse), history)
        nn$save_npz(file.path(directory, paste0("epoch", epoch, "_arrays.npz")), autoencoder_arrays(model$parameters))
        if (mse < best) {
            best <- mse
            selected <- epoch
            model$parameters$save(best_path)  # common validation reconstruction MSE selects each model
        }
        cat(sprintf("%s epoch%d:validationMSE=%s\n", label, epoch, fmt(mse)))
    }
    seconds <- proc.time()[["elapsed"]] - started

    deployed <- competitive_autoencoder_model(lifetime, plain, FALSE)
    deployed$parameters$load(best_path)  # fresh competitive/plain inference restore
    before <- autoencoder_snapshot(deployed$parameters)
    probe_images <- nn$take_rows(pools[[3L]]$images, 0:31)
    probe <- deployed$probe(probe_images)
    competitive_autoencoder_check_support(probe, lifetime, plain)
    nn$save_npz(file.path(directory, "selected_probe.npz"),
                list(images = probe_images, raw = probe[[1L]], mask = probe[[2L]], training_code = probe[[3L]],
                     reconstruction = probe[[4L]]))
    arrays <- list()
    splits <- c("fit", "validation", "test")
    for (split in 1:3) {
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
    zero <- deployed$decode(nn$zeros(c(200L, 64L)))
    shifted <- deployed$decode(nn$concat(nn$tensor_vector(list(nn$slice_axis(test[[1L]], 0L, 1L, 200L),
                                                               nn$slice_axis(test[[1L]], 0L, 0L, 1L))), 0L))
    arrays$zero_reconstruction <- zero
    arrays$shuffled_reconstruction <- shifted
    one <- deployed$evaluate(nn$take_rows(pools[[3L]]$images, 0L))
    error <- autoencoder_max_abs_diff(one[[2L]], test[[2L]], 784L)
    nn_require(error < 1e-6, "single/batch competitive/plain inference parity")
    nn_require(identical(before, autoencoder_snapshot(deployed$parameters)),
               "all deployed parameters immutable through code controls")
    nn$save_npz(file.path(directory, "outputs.npz"), arrays)
    nn$save_npz(file.path(directory, "model_arrays.npz"), autoencoder_arrays(deployed$parameters))
    test_mse <- autoencoder_mse(test[[2L]], pools[[3L]]$images)
    tf <- function(x) if (x) "true" else "false"
    writeLines(sprintf(paste0('{"lifetime":%s,"plain":%s,"parameters":%d,"epochs":%d,"updates":%d,"selected_epoch":%d,',
                              '"test_mse":%.17g,"zero_mse":%.17g,"shuffled_mse":%.17g,"single_batch_max_error":%.17g,',
                              '"training_seconds":%.17g}'),
                       tf(lifetime), tf(plain), as.integer(model$parameters$count()), epochs, step, selected, test_mse,
                       autoencoder_mse(zero, pools[[3L]]$images), autoencoder_mse(shifted, pools[[3L]]$images), error,
                       seconds), file.path(directory, "metrics.json"))
    cat(sprintf("%s:selected%d,parameters=%d,testMSE=%s,freshrestore/decode/repeat/immutable parameters passed (training %s s)\n",
                label, selected, as.integer(model$parameters$count()), fmt(test_mse), fmt(seconds)))
    seconds
}

main <- function(arguments = commandArgs(TRUE)) {
    args <- nn_options(arguments, list(epochs = 3L, data = file.path(nn_data_root(), "saliency"), out = "", init = "",
                                       full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs <- args$epochs
    root <- if (nzchar(args$out)) args$out else file.path(nn_output_root(args$full), "models/winner_take_all_autoencoder")
    nn_require(epochs > 0L && epochs <= 100L, "three default epochs or explicit1..100")
    competitive_autoencoder_checks()

    lifetime <- lifetime_competition
    pools <- autoencoder_load_saliency(args$data)
    dir.create(root, recursive = TRUE, showWarnings = FALSE)
    factory <- competitive_autoencoder_model(lifetime, FALSE, FALSE)
    if (nzchar(args$init)) factory$parameters$load(args$init)
    seconds <- competitive_train_variant(lifetime, FALSE, pools, epochs, root, factory$parameters)
    seconds <- seconds + competitive_train_variant(lifetime, TRUE, pools, epochs, root, factory$parameters)
    autoencoder_save_data(file.path(root, "data.npz"), pools)
    autoencoder_copy_manifests(args$data, root)
    t <- nn$load_npz(file.path(root, "competitive", "outputs.npz"))
    u <- nn$load_npz(file.path(root, "plain", "outputs.npz"))
    test_images <- pools[[3L]]$images
    mean <- autoencoder_constant(autoencoder_pixel_mean(pools[[1L]]$images), 200L)
    competitive_reconstruction <- autoencoder_find(t, "test_reconstruction")
    plain_reconstruction <- autoencoder_find(u, "test_reconstruction")
    competitive_mse <- autoencoder_mse(competitive_reconstruction, test_images)
    plain_mse <- autoencoder_mse(plain_reconstruction, test_images)
    mean_mse <- autoencoder_mse(mean, test_images)
    html <- paste0(
        "<!doctype html><meta charset='utf-8'><title>Hard competition autoencoder</title><style>body{font:16px ",
        "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:",
        "25px 0}</style><h1>",
        if (lifetime) "Lifetime winner-take-all" else "Per-image k-sparse",
        " autoencoder</h1><p>Original,competitive,plain control,fit-only mean,absolute competitive error,zero-code,",
        "shuffled-code. Fixed[0,1]gray/nearest4/no error amplification. First test record of each digit,unfiltered.",
        "</p><p>784->64->784sigmoid,101200parameters. ",
        if (lifetime) paste0("ReLU encoder,TRAINING ONLY lifetime winners:each feature keeps ceil(batch/8) examples,4of32 ",
                             "or1of8. Deployed encoder is full ReLU with NO lifetime gate;deployed code is NOT guaranteed ",
                             "sparse. This is fully-connected WTA,not spatial/convolutional WTA.")
        else paste0("LINEAR encoder,top8largest SIGNED activities per image,not top absolute magnitude;mask is ",
                    "retained in training AND inference. Exactly8selected coordinates,at most8nonzero values;",
                    "negative winners remain signed. No ReLU,no sparsity penalty or straight-through gradient."),
        " TopK ties select lower indices;derivatives pass ONLY selected coordinates. The discrete ranking has no ",
        "derivative;at ties this is one deterministic subgradient convention.</p><p>Matched initial parameter bytes ",
        "and source orders;plain control removes competition. ", epochs,
        "epochs EACH,Adam.003,batch32/final8. Separate validation reconstruction MSE selects each deployed model. ",
        "Sigmoid decoder/raw pixels/compact width are educational adaptations of the <a href='https://arxiv.org/abs/",
        if (lifetime) "1409.2752" else "1312.5663",
        "'>hard sparsity approach</a>,not the original scale/decoder/classification experiment. Same weights do not ",
        "imply same gated initial mapping. Labels only balance pools/captions. Blurry errors/weak quality remain ",
        "visible;no bit-rate or semantic/classification claim.</p><p>TestMSE competitive=", fmt(competitive_mse),
        ",plain=", fmt(plain_mse), ",mean=", fmt(mean_mse), ".</p>")
    html <- c(html, autoencoder_digit_gallery(root, pools[[3L]],
        list(test_images, competitive_reconstruction, plain_reconstruction, mean,
             nn$abs(nn$sub(competitive_reconstruction, autoencoder_flatten(test_images))),
             autoencoder_find(t, "zero_reconstruction"), autoencoder_find(t, "shuffled_reconstruction"))))
    writeLines(paste(html, collapse = ""), file.path(root, "index.html"))
    cat(sprintf(paste0("Hard winner competition/matched plain controls/fresh deployment verified.\nTestMSE competitive=",
                       "%s,plain=%s,mean=%s\ntraining time %s s\nArtifacts:%s/index.html\n"),
                fmt(competitive_mse), fmt(plain_mse), fmt(mean_mse), fmt(seconds), root))
    0L
}

if (sys.nframe() == 0L) quit(status = main())
