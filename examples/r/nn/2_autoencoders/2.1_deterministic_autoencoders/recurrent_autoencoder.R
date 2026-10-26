# Native R NN example. Sequence-to-sequence tanh recurrent autoencoder with a final-state-only bottleneck.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

# CMD: just r-example recurrent_autoencoder
# Sequence-to-sequence tanh recurrent autoencoder with a final-state-only bottleneck.
#
# Port of tflow's recurrent_autoencoder. Preparation (in ~/projects/tflow_examples):
# just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
# (history, npz arrays, states.npz, metrics.json, index.html and digit<k>.png galleries)
# go to output/nn/models/recurrent_autoencoder. Checkpoints are best.npz files instead of
# TF SaveV2 bundles. Options: epochs=N data=PATH out=PATH init=PATH (start from saved
# parameters, e.g. tflow's initial_arrays.npz, to follow its trajectory exactly; by
# default nn's own initialiser is used).

nn_helper("checks")
nn_helper("autoencoder")

fmt <- function(x) sprintf("%g", x)

# ---- model (examples/nn/common/recurrent_autoencoder.hpp) --------------------------------

# Decoder only: repeat the code, NOT original/previous target rows; all steps share weights.
recurrent_decode <- function(p, code, steps = 28L, features = 28L) {
    h <- nn$zeros_like(code)
    states <- list()
    rows <- list()
    for (t in seq_len(steps)) {
        h <- nn$rnn_step(p, "decoder", code, h)
        states[[t]] <- h
        rows[[t]] <- nn$dense(p, "output", h, features, nn$Activation$Sigmoid)
    }
    list(code = code, reconstruction = nn$stack(nn$tensor_vector(rows), 1L),
         decoder_states = nn$stack(nn$tensor_vector(states), 1L))
}

# Tanh encoder over the rows of sequence [B, steps, features]; the final state is the only bridge.
recurrent_autoencoder <- function(p, sequence, steps = 28L, features = 28L, units = 32L) {
    nn_require(steps > 1L && features > 0L && units > 0L && units < steps * features, "strict sequence bottleneck")
    h <- nn$zeros(c(sequence$size(0L), units), sequence$dtype())
    states <- list()
    for (t in seq_len(steps)) {
        h <- nn$rnn_step(p, "encoder", nn$select(sequence, 1L, t - 1L), h)
        states[[t]] <- h
    }
    out <- recurrent_decode(p, h, steps, features)
    out$encoder_states <- nn$stack(nn$tensor_vector(states), 1L)
    out
}

# [n, 28, 28, 1] viewed as an ordered 28-row sequence, without resize.
recurrent_sequence <- function(images) nn$reshape(images, c(images$size(0L), 28L, 28L))

recurrent_mse <- function(reconstruction, sequence) nn$mean(nn$square(nn$sub(reconstruction, sequence)))

recurrent_model <- function(training = TRUE) {
    model <- new.env()
    model$parameters <- nn$ParameterStore(1871L)
    nn$no_grad(function() recurrent_autoencoder(model$parameters, nn$zeros(c(1L, 28L, 28L))))
    model$optimizer <- if (training) autoencoder_make_adam(model$parameters) else NULL
    # list(code, reconstruction [n, 28, 28], MSE)
    model$evaluate <- function(images) nn$no_grad(function() {
        x <- recurrent_sequence(images)
        n <- recurrent_autoencoder(model$parameters, x)
        list(n$code, n$reconstruction, recurrent_mse(n$reconstruction, x))
    })
    model$decode <- function(codes) nn$no_grad(function() recurrent_decode(model$parameters, codes)$reconstruction)
    model$states <- function(images) nn$no_grad(function() {
        n <- recurrent_autoencoder(model$parameters, recurrent_sequence(images))
        list(n$encoder_states, n$decoder_states)
    })
    model
}

# ---- experiment ------------------------------------------------------------------------------

# Trains, selects on validation MSE, and checks the restored model; returns training seconds.
recurrent_train <- function(pools, epochs, directory, init) {
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    test_images <- pools[[3L]]$images
    model <- recurrent_model()
    if (nzchar(init)) model$parameters$load(init)
    nn_require(model$parameters$size() == 8L && model$parameters$count() == 4956,
               "shared-over-time recurrent parameter census")
    nn$save_npz(file.path(directory, "initial_arrays.npz"), autoencoder_arrays(model$parameters))
    initial <- model$evaluate(test_images)
    nn$save_npz(file.path(directory, "initial_outputs.npz"), list(code = initial[[1L]], reconstruction = initial[[2L]]))
    order <- nn$BatchOrder(1000L, 32L, 1873L)
    fit_source <- as.integer(pools[[1L]]$source$values())
    best_path <- file.path(directory, "best.npz")
    best <- Inf
    selected <- 0L
    step <- 0L
    history <- file(file.path(directory, "history.tsv"), "w")
    batches <- file(file.path(directory, "batches.tsv"), "w")
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
            # Full backpropagation through all 28 encoder + 28 decoder steps.
            sequence <- recurrent_sequence(input)
            loss <- recurrent_mse(recurrent_autoencoder(model$parameters, sequence)$reconstruction, sequence)
            if (step == 0L) {
                keep <- nn$GradOptions()
                keep$retain_graph <- TRUE
                gradients <- autoencoder_check_gradients(model$parameters,
                    nn$grad(loss, model$parameters$parameters(), keep),
                    "all model MSE derivatives finite", "all model tensors receive full MSE gradients")
                nn$save_npz(file.path(directory, "first_gradients.npz"), gradients)
                nn$save_npz(file.path(directory, "first_batch.npz"), list(images = input))
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
        value <- model$evaluate(pools[[2L]]$images)
        mse <- autoencoder_mse(value[[2L]], pools[[2L]]$images)
        writeLines(sprintf("%d\t%.17g\t%.17g", epoch, total / seen, mse), history)
        nn$save_npz(file.path(directory, paste0("epoch", epoch, "_arrays.npz")), autoencoder_arrays(model$parameters))
        if (mse < best) {
            best <- mse
            selected <- epoch
            model$parameters$save(best_path)  # common validation reconstruction MSE selects the model
        }
        cat(sprintf("recurrent epoch%d:validationMSE=%s\n", epoch, fmt(mse)))
    }
    close(history)
    close(batches)
    seconds <- proc.time()[["elapsed"]] - started
    deployed <- recurrent_model(FALSE)
    deployed$parameters$load(best_path)  # fresh recurrent inference restore
    before <- autoencoder_snapshot(deployed$parameters)
    probe <- autoencoder_rows(pools[[3L]], 0:7)$images
    states <- deployed$states(probe)
    nn$save_npz(file.path(directory, "states.npz"),
                list(images = probe, encoder_states = states[[1L]], decoder_states = states[[2L]]))
    arrays <- list()
    splits <- c("fit", "validation", "test")
    for (split in 1:3) {
        out <- deployed$evaluate(pools[[split]]$images)
        arrays[[paste0(splits[split], "_code")]] <- out[[1L]]
        arrays[[paste0(splits[split], "_reconstruction")]] <- out[[2L]]
        if (split == 2L) nn_near(autoencoder_mse(out[[2L]], pools[[split]]$images), best, 1e-8,
                                 "fresh selected validation MSE")
    }
    test <- deployed$evaluate(test_images)
    repeated <- deployed$evaluate(test_images)
    nn_require(identical(autoencoder_bytes(test[[2L]]), autoencoder_bytes(repeated[[2L]])) &&
               identical(autoencoder_bytes(deployed$decode(test[[1L]])), autoencoder_bytes(test[[2L]])),
               "deterministic exact encode-decode")
    zero <- deployed$decode(nn$zeros(c(200L, 32L)))
    shuffled <- deployed$decode(nn$concat(nn$tensor_vector(list(nn$slice_axis(test[[1L]], 0L, 1L, 200L),
                                                                nn$slice_axis(test[[1L]], 0L, 0L, 1L))), 0L))
    arrays$zero_reconstruction <- zero
    arrays$shuffled_reconstruction <- shuffled
    # Rows in reverse order (y -> 27 - y): the ordered input is disrupted, output positions unchanged.
    reversed_rows <- nn$from_vector_int64(27:0, 28L)
    reversed <- nn$gather(recurrent_sequence(test_images), reversed_rows, 1L)$contiguous()
    reverse_out <- deployed$evaluate(reversed)
    arrays$reversed_code <- reverse_out[[1L]]
    arrays$reversed_reconstruction <- reverse_out[[2L]]
    one <- deployed$evaluate(nn$take_rows(test_images, 0L))
    error <- autoencoder_max_abs_diff(one[[2L]], test[[2L]], 784L)
    nn_require(error < 1e-6, "single/batch recurrent inference parity")
    nn_require(identical(before, autoencoder_snapshot(deployed$parameters)),
               "all deployed parameters immutable through code controls")
    nn$save_npz(file.path(directory, "outputs.npz"), arrays)
    nn$save_npz(file.path(directory, "model_arrays.npz"), autoencoder_arrays(deployed$parameters))
    test_mse <- autoencoder_mse(test[[2L]], test_images)
    writeLines(sprintf(paste0('{"parameters":%d,"epochs":%d,"updates":%d,"selected_epoch":%d,"test_mse":%.17g,',
                              '"zero_mse":%.17g,"shuffled_mse":%.17g,"single_batch_max_error":%.17g,',
                              '"training_seconds":%.17g}'),
                       as.integer(model$parameters$count()), epochs, step, selected, test_mse,
                       autoencoder_mse(zero, test_images), autoencoder_mse(shuffled, test_images), error, seconds),
               file.path(directory, "metrics.json"))
    cat(sprintf("recurrent:selected%d,parameters=%d,testMSE=%s,freshrestore/decode/repeat/immutable parameters passed\n",
                selected, as.integer(model$parameters$count()), fmt(test_mse)))
    seconds
}

main <- function(arguments = commandArgs(TRUE)) {
    args <- nn_options(arguments, list(epochs = 3L, data = file.path(nn_data_root(), "saliency"), out = "", init = "",
                                       full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs <- args$epochs
    root <- if (nzchar(args$out)) args$out else file.path(nn_output_root(args$full), "models/recurrent_autoencoder")
    nn_require(epochs > 0L && epochs <= 100L, "three default epochs or explicit1..100")
    pools <- autoencoder_load_saliency(args$data)
    dir.create(root, recursive = TRUE, showWarnings = FALSE)
    seconds <- recurrent_train(pools, epochs, root, args$init)
    autoencoder_save_data(file.path(root, "data.npz"), pools)
    autoencoder_copy_manifests(args$data, root)
    t <- nn$load_npz(file.path(root, "outputs.npz"))
    test_images <- pools[[3L]]$images
    mean <- autoencoder_constant(autoencoder_pixel_mean(pools[[1L]]$images), 200L)
    reconstruction <- t$test_reconstruction
    html <- paste0(
        "<!doctype html><meta charset='utf-8'><title>Recurrent sequence autoencoder</title><style>body{font:16px ",
        "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:",
        "25px 0}</style><h1>Sequence-to-sequence recurrent autoencoder</h1><p>Original,reconstruction,fit-only mean,",
        "absolute error,zero-code,shuffled-code,reversed encoder input. Fixed[0,1] grayscale,nearest4,no error ",
        "amplification. First test record per digit,unfiltered.</p><p>MNIST is a sequence of28rows of28pixels. Tanh ",
        "encoder32,final-state-only code32,repeated into tanh decoder32 with zero initial state,shared sigmoid row ",
        "projection. Distinct encoder/decoder weights,shared across time;4956parameters. No teacher forcing,",
        "original-input bypass,attention or target rows in decoder. Full backpropagation through all28encoder+",
        "28decoder steps. This is fixed-length row-sequence reconstruction,not language translation or ",
        "variable-length prediction.</p><p>", epochs,
        " epochs,Adam.003,batch32,1000fit/200validation/200test. Validation selects the completed epoch. Labels only ",
        "balance/captions. Reversing encoder rows deliberately disrupts the ordered input while output positions stay ",
        "unchanged;not an equivalent reconstruction task. Continuous code is not a compressed bitstream. Short ",
        "training may produce generic blurry rows or lose digit identity.</p><p>TestMSE=",
        fmt(autoencoder_mse(reconstruction, test_images)), ",fitmean=", fmt(autoencoder_mse(mean, test_images)), ".</p>")
    sections <- autoencoder_digit_gallery(root, pools[[3L]],
        list(test_images, reconstruction, mean,
             nn$abs(nn$sub(nn$reshape(reconstruction, c(200L, 784L)), autoencoder_flatten(test_images))),
             t$zero_reconstruction, t$shuffled_reconstruction, t$reversed_reconstruction))
    writeLines(paste0(html, paste(sections, collapse = "")), file.path(root, "index.html"))
    cat(sprintf("Full recurrent sequence reconstruction/fresh restore/latent controls verified.\ntraining time %s s\nArtifacts:%s/index.html\n",
                fmt(seconds), root))
    0L
}

if (sys.nframe() == 0L) quit(status = main())
