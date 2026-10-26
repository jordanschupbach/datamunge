# Native R NN example. Greedy layer-wise pretrained stacked sigmoid autoencoder with a scratch control.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

# CMD: just r-example stacked_autoencoder
# Greedy 784->64 and 64->16 pretraining, exact transfer into 784->64->16->64->784.
# Three epochs per stage/default; validation-only selection and same-stack scratch control.
#
# Port of tflow's stacked_autoencoder (model: examples/nn/common/stacked_autoencoder.hpp).
# Preparation (in ~/projects/tflow_examples): just autoencoder-assets; data are read from
# $DATAMUNGE_DATA_DIR/saliency. Outputs (stage1/, stage2/, finetune/, scratch/ training records,
# transferred_arrays.npz, stacked.npz, metrics.json, index.html and digit<k>.png galleries) go
# to output/nn/models/stacked_autoencoder. Options: epochs=N data=PATH out=PATH init=DIR (a
# tflow stacked_autoencoder output directory: stage1/, stage2/ and scratch/ initial_arrays.npz
# are loaded into stage1, stage2 and both full stacks; by default nn's own initialiser is used,
# with the full stacks identical because they share a seed).

nn_helper("checks")
nn_helper("autoencoder")

fmt <- function(x) sprintf("%g", x)
f32 <- function(x) readBin(writeBin(x, raw(), size = 4L), "double", size = 4L, n = length(x))

# Dense sigmoid layers "layer<i>" through mirrored, strictly decreasing widths; every layer output.
stacked_sigmoid_autoencoder <- function(p, image, widths) {
    count <- length(widths)
    nn_require(count >= 3L && count %% 2L == 1L, "odd symmetric width list includes input and output")
    for (i in seq_len(count)) {
        nn_require(widths[i] > 0L && widths[i] == widths[count + 1L - i], "positive mirrored encoder/decoder dimensions")
        if (i <= count %/% 2L) nn_require(widths[i] > widths[i + 1L], "strictly decreasing encoder widths")
    }
    outputs <- list()
    value <- image
    for (i in 2L:count) {
        value <- nn$dense(p, paste0("layer", i - 2L), value, as.integer(widths[i]), nn$Activation$Sigmoid)
        outputs[[i - 1L]] <- value
    }
    outputs
}

# Parameters + optional Adam of one symmetric sigmoid stack (stacked_autoencoder.hpp).
stacked_model <- function(widths, seed, training = TRUE) {
    model <- new.env()
    model$widths <- as.integer(widths)
    model$features <- model$widths[1L]
    model$code_index <- length(widths) %/% 2L - 1L
    model$parameters <- nn$ParameterStore(seed)
    nn$no_grad(function() stacked_sigmoid_autoencoder(model$parameters, nn$zeros(c(1L, max(model$features, 1L))),
                                                      model$widths))
    model$optimizer <- if (training) autoencoder_make_adam(model$parameters) else NULL
    # Every layer output, then the MSE (graph-building).
    model$net <- function(x) {
        nn_require(x$dim() == 2L && x$size(1L) == model$features, "stage input dimensionality")
        layers <- stacked_sigmoid_autoencoder(model$parameters, x, model$widths)
        c(layers, list(autoencoder_reconstruction_mse(layers[[length(layers)]], x)))
    }
    model$evaluate <- function(x) nn$no_grad(function() lapply(model$net(x), function(t) t$contiguous()))
    # The layers after the code layer, from an explicit code.
    model$decode <- function(code) nn$no_grad(function() {
        value <- code
        for (i in seq.int(model$code_index + 2L, length(model$widths) - 1L))
            value <- nn$dense(model$parameters, paste0("layer", i - 1L), value, model$widths[i + 1L],
                              nn$Activation$Sigmoid)
        value$contiguous()
    })
    # Explicit whole-model weight transfer (optimizer state unchanged).
    model$load <- function(arrays) {
        nn_require(length(arrays) == model$parameters$size(), "transfer all model tensors")
        for (name in model$parameters$names()) {
            value <- autoencoder_find(arrays, name)
            nn_require(value$dtype() == nn$DType$Float32 &&
                       identical(as.numeric(value$shape()), as.numeric(model$parameters$at(name)$shape())),
                       "exact transfer shape and dtype")
            model$parameters$set(name, value)
        }
    }
    model
}

stacked_fixtures <- function() {
    toy <- stacked_model(c(4L, 3L, 2L, 3L, 4L), 1543)
    widths <- c(4L, 3L, 2L, 3L, 4L)
    w <- list()
    b <- list()
    for (l in 0L:3L) {
        w[[l + 1L]] <- f32(0.1 * sin(seq_len(widths[l + 1L] * widths[l + 2L]) - 1 + l + 0.3))
        b[[l + 1L]] <- f32(0.01 * (seq_len(widths[l + 2L]) - 2))
        toy$parameters$set(paste0("layer", l, "/W"), nn$from_vector(w[[l + 1L]], c(widths[l + 1L], widths[l + 2L])))
        toy$parameters$set(paste0("layer", l, "/b"), nn$from_vector(b[[l + 1L]], widths[l + 2L]))
    }
    inputs <- f32(f32(0.1) * (1:8))
    x <- nn$from_vector(inputs, c(2L, 4L))
    actual <- toy$evaluate(x)
    # Explicit forward through every sigmoid layer, full-stack MSE, then reverse-mode gradients.
    toy_layers <- stacked_sigmoid_autoencoder(toy$parameters, x, toy$widths)
    gradient <- nn$grad(autoencoder_reconstruction_mse(toy_layers[[4L]], x), toy$parameters$parameters())
    forward_error <- 0
    activations <- list(inputs)
    for (l in 1L:4L) {
        got <- actual[[l]]$values()
        nin <- widths[l]
        nout <- widths[l + 1L]
        following <- numeric(2L * nout)
        for (n in 0L:1L) for (j in 1L:nout) {
            value <- b[[l]][j]
            for (i in 1L:nin) value <- value + activations[[l]][n * nin + i] * w[[l]][(i - 1L) * nout + j]
            following[n * nout + j] <- 1 / (1 + exp(-value))
            forward_error <- max(forward_error, abs(got[n * nout + j] - following[n * nout + j]))
        }
        activations[[l + 1L]] <- following
    }
    cat(sprintf("toy float32 sigmoid versus float64 maxerror=%s\n", fmt(forward_error)))
    nn_near(forward_error, 0, 2e-7, "all24toy sigmoid activations across four layers,float32 tolerance")
    delta <- numeric(8L)
    loss <- 0
    for (i in 1L:8L) {
        value <- activations[[5L]][i]
        error <- value - activations[[1L]][i]
        loss <- loss + error * error / 8
        delta[i] <- 2 * error / 8 * value * (1 - value)
    }
    nn_near(actual[[5L]]$item(), loss, 2e-8, "known full-stack MSE")
    checked <- 0L
    for (l in 4L:1L) {
        nin <- widths[l]
        nout <- widths[l + 1L]
        prior <- numeric(2L * nin)
        gw <- gradient[[2L * l - 1L]]$values()
        gb <- gradient[[2L * l]]$values()
        for (j in 1L:nout) {
            db <- 0
            for (n in 0L:1L) db <- db + delta[n * nout + j]
            nn_near(gb[j], db, 2e-8, "all toy bias chain derivatives")
            checked <- checked + 1L
            for (i in 1L:nin) {
                dw <- 0
                for (n in 0L:1L) {
                    dw <- dw + activations[[l]][n * nin + i] * delta[n * nout + j]
                    prior[n * nin + i] <- prior[n * nin + i] + delta[n * nout + j] * w[[l]][(i - 1L) * nout + j]
                }
                nn_near(gw[(i - 1L) * nout + j], dw, 2e-8, "all toy weight chain derivatives")
                checked <- checked + 1L
            }
        }
        delta <- prior * activations[[l]] * (1 - activations[[l]])
    }
    nn_require(checked == 48L, "independent toy gradient inventory48")
    nn_require(identical(autoencoder_bytes(toy$decode(actual[[2L]])), autoencoder_bytes(actual[[4L]])),
               "two-layer standalone decoder has same full result")
    nn_throws(stacked_model(c(4L, 4L, 4L), 1543), "non-undercomplete stack rejected")
    cat("toy four-layer sigmoid forward/MSE/ALL48analytic derivatives/transfer/decode/strictshape checks passed\n")
}

# Second-to-last evaluate output: the reconstruction.
stacked_reconstruction <- function(out) out[[length(out) - 1L]]

# Adam stage training with validation-only selection.
stacked_train_stage <- function(model, fit, validation, source, epochs, directory, include_zero) {
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    nn$save_npz(file.path(directory, "initial_arrays.npz"), autoencoder_arrays(model$parameters))
    initial_fit <- autoencoder_mse(stacked_reconstruction(model$evaluate(fit)), fit)
    epoch0 <- autoencoder_mse(stacked_reconstruction(model$evaluate(validation)), validation)
    best <- epoch0
    selected <- 0L
    best_path <- file.path(directory, "best.npz")
    if (include_zero) {
        model$parameters$save(best_path)  # epoch0 pretrained/scratch baseline eligible for validation selection
    } else {
        best <- Inf
    }
    nn$save_npz(file.path(directory, "epoch0_arrays.npz"), autoencoder_arrays(model$parameters))
    ids <- as.integer(source$values())
    order <- nn$BatchOrder(1000L, 32L, 1559L)
    step <- 0L
    history <- file(file.path(directory, "history.tsv"), "w")
    batches <- file(file.path(directory, "batches.tsv"), "w")
    on.exit({
        close(history)
        close(batches)
    }, add = TRUE)
    writeLines("epoch\ttrain_preupdate_mse\tvalidation_mse", history)
    writeLines("epoch\tstep\trow\tsource_id", batches)
    writeLines(sprintf("0\t%.17g\t%.17g", initial_fit, epoch0), history)
    started <- proc.time()[["elapsed"]]
    for (epoch in seq_len(epochs)) {
        order$begin_epoch()
        total <- 0
        seen <- 0L
        repeat {
            rows <- order$`_next`()
            if (!length(rows)) break
            x <- nn$take_rows(fit, rows)
            # Forward the stage, reconstruction MSE against its own input.
            layers <- stacked_sigmoid_autoencoder(model$parameters, x, model$widths)
            loss <- autoencoder_reconstruction_mse(layers[[length(layers)]], x)
            if (step == 0L) {
                keep <- nn$GradOptions()
                keep$retain_graph <- TRUE
                nn$save_npz(file.path(directory, "first_gradients.npz"),
                    autoencoder_check_gradients(model$parameters, nn$grad(loss, model$parameters$parameters(), keep),
                                                "all stage derivatives finite",
                                                "every stage tensor receives reconstruction gradient"))
                nn$save_npz(file.path(directory, "first_batch.npz"), list(images = x))
            }
            value <- loss$item()
            nn_require(is.finite(value) && value >= 0, "finite stage MSE")
            model$optimizer$minimize(loss)
            total <- total + length(rows) * value
            seen <- seen + length(rows)
            writeLines(sprintf("%d\t%d\t%d\t%d", epoch, step, rows, ids[rows + 1L]), batches)
            if (step == 0L) nn$save_npz(file.path(directory, "first_update_arrays.npz"), autoencoder_arrays(model$parameters))
            step <- step + 1L
        }
        nn_require(seen == 1000L, "each fitting record once per stage epoch including short last batch")
        mse <- autoencoder_mse(stacked_reconstruction(model$evaluate(validation)), validation)
        writeLines(sprintf("%d\t%.17g\t%.17g", epoch, total / seen, mse), history)
        nn$save_npz(file.path(directory, paste0("epoch", epoch, "_arrays.npz")), autoencoder_arrays(model$parameters))
        if (mse < best) {
            best <- mse
            selected <- epoch
            model$parameters$save(best_path)  # validation-only stage selection
        }
        cat(sprintf("%s epoch%d:fitPreMSE=%s,validationMSE=%s\n", basename(directory), epoch, fmt(total / seen), fmt(mse)))
    }
    seconds <- proc.time()[["elapsed"]] - started
    model$parameters$load(best_path)  # restore selected greedy/joint model
    fit_mse <- autoencoder_mse(stacked_reconstruction(model$evaluate(fit)), fit)
    nn_require(fit_mse < initial_fit || (include_zero && selected == 0L),
               "greedy stage learns target or eligible epoch0 is explicitly retained")
    nn$save_npz(file.path(directory, "model_arrays.npz"), autoencoder_arrays(model$parameters))
    writeLines(sprintf(paste0('{"epochs":%d,"updates":%d,"selected_epoch":%d,"include_epoch0":%s,',
                              '"initial_fit_mse":%.17g,"fit_mse":%.17g,"validation_mse":%.17g,"training_seconds":%.17g}'),
                       epochs, step, selected, if (include_zero) "true" else "false", initial_fit, fit_mse, best, seconds),
               file.path(directory, "stage.json"))
    list(selected = selected, initial_fit = initial_fit, fit = fit_mse, validation = best, seconds = seconds)
}

stacked_shift <- function(codes) {
    n <- codes$size(0L)
    nn$take_rows(codes, c(seq_len(n - 1L), 0L))$contiguous()
}

main <- function(arguments = commandArgs(TRUE)) {
    args <- nn_options(arguments, list(epochs = 3L, data = file.path(nn_data_root(), "saliency"), out = "", init = "",
                                       full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs <- args$epochs
    directory <- if (nzchar(args$out)) args$out else file.path(nn_output_root(args$full), "models/stacked_autoencoder")
    nn_require(epochs > 0L && epochs <= 100L, "three default epochs PER stage or explicit1..100")
    stacked_fixtures()

    pools <- autoencoder_load_saliency(args$data)
    fit <- autoencoder_flatten(pools[[1L]]$images)$contiguous()
    val <- autoencoder_flatten(pools[[2L]]$images)$contiguous()
    test <- autoencoder_flatten(pools[[3L]]$images)$contiguous()
    ids <- pools[[1L]]$source
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    full_widths <- c(784L, 64L, 16L, 64L, 784L)
    stage1 <- stacked_model(c(784L, 64L, 784L), 1549)
    stage2 <- stacked_model(c(64L, 16L, 64L), 1553)
    joint <- stacked_model(full_widths, 1553)
    scratch <- stacked_model(full_widths, 1553)
    if (nzchar(args$init)) {
        stage1$parameters$load(file.path(args$init, "stage1", "initial_arrays.npz"))
        stage2$parameters$load(file.path(args$init, "stage2", "initial_arrays.npz"))
        joint$parameters$load(file.path(args$init, "scratch", "initial_arrays.npz"))
        scratch$parameters$load(file.path(args$init, "scratch", "initial_arrays.npz"))
    }
    nn_require(stage1$parameters$count() == 101200 && stage2$parameters$count() == 2128 &&
               joint$parameters$count() == 103328, "independent greedy/full parameter inventories")
    nn_require(identical(autoencoder_snapshot(joint$parameters), autoencoder_snapshot(scratch$parameters)),
               "same full graph/initial seed before transfer")

    first <- stacked_train_stage(stage1, fit, val, ids, epochs, file.path(directory, "stage1"), FALSE)
    frozen <- autoencoder_snapshot(stage1$parameters)
    encoded_fit <- stage1$evaluate(fit)[[1L]]
    encoded_val <- stage1$evaluate(val)[[1L]]
    encoded_test <- stage1$evaluate(test)[[1L]]
    second <- stacked_train_stage(stage2, encoded_fit, encoded_val, ids, epochs, file.path(directory, "stage2"), FALSE)
    nn_require(identical(frozen, autoencoder_snapshot(stage1$parameters)),
               "stage2 cannot mutate frozen stage1 encoder/decoder")
    nn_require(identical(autoencoder_bytes(stage1$evaluate(fit)[[1L]]), autoencoder_bytes(encoded_fit)),
               "cached stage2 fitting targets remain exact frozen encoder outputs")

    a <- autoencoder_arrays(stage1$parameters)
    b <- autoencoder_arrays(stage2$parameters)
    transfer <- list()
    for (spec in list(c("layer0", "layer0", "1"), c("layer3", "layer1", "1"), c("layer1", "layer0", "2"),
                      c("layer2", "layer1", "2")))
        for (suffix in c("/W", "/b"))
            transfer[[paste0(spec[1L], suffix)]] <- autoencoder_find(if (spec[3L] == "1") a else b, paste0(spec[2L], suffix))
    joint$load(transfer)
    loaded <- autoencoder_arrays(joint$parameters)
    for (key in names(transfer))
        nn_require(identical(autoencoder_bytes(autoencoder_find(loaded, key)), autoencoder_bytes(transfer[[key]])),
                   "ALL8pretrained arrays transferred exactly including both decoders")
    nn$save_npz(file.path(directory, "transferred_arrays.npz"), loaded)
    pretrained <- joint$evaluate(test)
    nested1 <- stage1$decode(stage2$evaluate(encoded_test)[[2L]])
    nn_require(identical(autoencoder_bytes(pretrained[[4L]]), autoencoder_bytes(nested1)),
               "composed selected greedy decoders exactly match transferred full graph")

    fine <- stacked_train_stage(joint, fit, val, ids, epochs, file.path(directory, "finetune"), TRUE)
    plain <- stacked_train_stage(scratch, fit, val, ids, epochs, file.path(directory, "scratch"), TRUE)
    final <- joint$evaluate(test)
    scratch_result <- scratch$evaluate(test)
    deployed <- stacked_model(full_widths, 1553, FALSE)
    deployed$parameters$load(file.path(directory, "finetune", "best.npz"))  # fresh inference restore all8 tensors
    before <- autoencoder_snapshot(deployed$parameters)
    restored <- deployed$evaluate(test)
    nn_require(identical(autoencoder_bytes(restored[[4L]]), autoencoder_bytes(final[[4L]])) &&
               identical(autoencoder_bytes(restored[[2L]]), autoencoder_bytes(final[[2L]])),
               "fresh restore exact all test reconstructions/codes")
    nn_require(identical(autoencoder_bytes(deployed$decode(restored[[2L]])), autoencoder_bytes(restored[[4L]])),
               "full encode then two-layer decoder exact")
    nn_require(identical(before, autoencoder_snapshot(deployed$parameters)),
               "deployed inference leaves all8parameters immutable")
    nn_require(identical(frozen, autoencoder_snapshot(stage1$parameters)), "stage1 remains frozen during all later stages")

    shuffled <- deployed$decode(stacked_shift(final[[2L]]))
    zero <- deployed$decode(nn$zeros(c(200L, 16L)))
    mean <- autoencoder_constant(autoencoder_pixel_mean(fit), 200L)
    nn_require(identical(before, autoencoder_snapshot(deployed$parameters)),
               "zero/shuffled code controls leave all8deployed parameters immutable")
    first_reconstruction <- stage1$evaluate(test)[[2L]]
    pre_mse <- autoencoder_mse(pretrained[[4L]], test)
    fine_mse <- autoencoder_mse(final[[4L]], test)
    scratch_mse <- autoencoder_mse(scratch_result[[4L]], test)
    mean_mse <- autoencoder_mse(mean, test)
    nn$save_npz(file.path(directory, "stacked.npz"), list(
        fit_images = fit, validation_images = val, test_images = test, fit_source_ids = ids,
        validation_source_ids = pools[[2L]]$source, test_source_ids = pools[[3L]]$source,
        test_labels = pools[[3L]]$labels, stage1_fit_code = encoded_fit, stage1_validation_code = encoded_val,
        stage1_test_code = encoded_test, stage1_test_reconstruction = first_reconstruction,
        pretrained_test_code = pretrained[[2L]], pretrained_test_reconstruction = pretrained[[4L]],
        finetuned_test_code = final[[2L]], finetuned_test_reconstruction = final[[4L]],
        scratch_test_code = scratch_result[[2L]], scratch_test_reconstruction = scratch_result[[4L]],
        mean_reconstruction = mean, shuffled_reconstruction = shuffled, zero_reconstruction = zero))
    autoencoder_copy_manifests(args$data, directory)
    seconds <- first$seconds + second$seconds + fine$seconds + plain$seconds
    writeLines(sprintf(paste0('{"epochs_per_stage":%d,"staged_epochs":%d,"scratch_epochs":%d,"staged_updates":%d,',
                              '"scratch_updates":%d,"pretrained_test_mse":%.17g,"finetuned_test_mse":%.17g,',
                              '"scratch_test_mse":%.17g,"mean_test_mse":%.17g,"shuffled_test_mse":%.17g,',
                              '"zero_test_mse":%.17g,"training_seconds":%.17g}'),
                       epochs, epochs * 3L, epochs, epochs * 96L, epochs * 32L, pre_mse, fine_mse, scratch_mse, mean_mse,
                       autoencoder_mse(shuffled, test), autoencoder_mse(zero, test), seconds),
               file.path(directory, "metrics.json"))

    html <- paste0(
        "<!doctype html><meta charset='utf-8'><title>Greedy stacked autoencoder</title><style>body{font:16px ",
        "system-ui;max-width:1100px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:",
        "25px 0}</style><h1>Stacked autoencoder with actual layer-wise pretraining</h1><p>Original,stage1 ",
        "reconstruction,composed pretrained stack,pretrained plus fine-tuning,scratch joint training,fit-only mean,",
        "cyclic shuffled code,zero-code decoder. Fixed[0,1]grayscale/nearest4,all ten first heldout examples per ",
        "true digit,no quality filtering.</p><p>784->64->16->64->784,ALL sigmoid layers,103328untied parameters. ",
        "Stage1 learns784->64->784;selected encoder1 is frozen while stage2 learns64->16->64 from its actual cached ",
        "representations. All8arrays including both decoders transfer EXACTLY before joint fine-tuning. No labels ",
        "enter any loss or selection.</p><p>",
        epochs, "epochs PER stage;staged total", 3L * epochs, "epochs/", 96L * epochs,
        "updates across different-sized models,versus scratch", epochs, "epochs/", 32L * epochs,
        "full-model updates. Same full architecture and fitting order;equal joint phase,NOT equal total work or a ",
        "causal proof that pretraining is better. Each phase selects by its own validation MSE;epoch0 is eligible ",
        "for fine-tuning/scratch.</p><p>Heldout MSE pretrained=", fmt(pre_mse), ",fine-tuned=", fmt(fine_mse),
        ",scratch=", fmt(scratch_mse), ",fit-mean=", fmt(mean_mse),
        ". Greedy stage2 reconstruction quality need not preserve raw-image detail. Blurry average-looking ",
        "reconstructions and weak short-run results remain visible. This16float code is not a bit-rate compression ",
        "claim,classification or a generative prior.</p>")
    html <- c(html, autoencoder_digit_gallery(directory, pools[[3L]],
        list(test, first_reconstruction, pretrained[[4L]], final[[4L]], scratch_result[[4L]], mean, shuffled, zero)))
    writeLines(paste(html, collapse = ""), file.path(directory, "index.html"))
    cat(sprintf(paste0("Stage selections=%d/%d/%d,scratch=%d\nHeldoutMSE pretrained/fine/scratch/mean=%s/%s/%s/%s\n",
                       "Frozen stage1/cachedtargets/ALL8exacttransfer/nesteddecoder/freshrestore/encode-decode/",
                       "immutable8weights passed. Staged work3x scratch epochs;no superiority claim.\n",
                       "training time %s s\nArtifacts:%s/index.html\n"),
                first$selected, second$selected, fine$selected, plain$selected, fmt(pre_mse), fmt(fine_mse),
                fmt(scratch_mse), fmt(mean_mse), fmt(seconds), directory))
    0L
}

if (sys.nframe() == 0L) quit(status = main())
