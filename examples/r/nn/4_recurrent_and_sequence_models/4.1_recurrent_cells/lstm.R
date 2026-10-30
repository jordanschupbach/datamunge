# Native R NN example. Full masked LSTM BPTT on the three-epoch marked-token memory task.
# CMD: nix develop .#r --command Rscript examples/r/nn/4_recurrent_and_sequence_models/4.1_recurrent_cells/lstm.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

# Full masked LSTM BPTT on the common three-epoch marked-token memory task.
#
# Port of tflow's lstm (common/ml/gated_sequence_demo.h with GatedCell::kLstm).
# No dataset: the 512/128/128 marked sequences are generated with tflow's
# std::mt19937 recipe (bit-identical data.npz). 746 parameters in 5 tensors,
# Adam .01, batch 32, three epochs (48 updates). Checks: parameter counts,
# bit-exact .npz restore, label-free prediction == evaluation, repeat and
# single-row parity, padding garbage cannot change any output or gradient,
# last-token flips cannot change earlier states, zero-length sequences, the
# LSTM gate algebra on every valid step, weights immutable under inference.
# Weights initialise from nn's generator (statistical parity); init=PATH
# (tflow's output/models/lstm/initial_arrays.npz) replays tflow's run and
# compares every record with tflow's run directory (tf=PATH).
# Options: epochs=3 out=PATH init=PATH tf=PATH. Output: <output>/models/lstm/.

nn_helper("checks")
nn_helper("recurrent_model")
nn_helper("marked_sequence_data")
nn_helper("marked_sequence_demo")
nn_helper("gated_sequence")
nn_helper("gated_sequence_demo")

name <- "lstm"
kind <- "lstm"

# Forward fields, loss and the full-BPTT gradients of the chosen logit with
# respect to the inputs and every zero initial state (tflow's Evaluate fetches).
evaluate <- function(model, feed) {
    inputs <- feed$inputs$detach()$clone()
    initials <- recurrent_model_initials(model, feed$inputs$shape()[1])
    inputs$requires_grad_(TRUE)
    for (state in initials) state$requires_grad_(TRUE)
    out <- model$forward(model$store, inputs, feed$lengths, initials)
    out$probability <- model$probability(out$logits)
    loss <- model$loss_fn(out, feed$labels)
    wrt <- c(list(inputs), initials)
    options <- nn$GradOptions()
    options$retain_graph <- TRUE
    options$allow_unused <- TRUE
    g <- nn$grad(list(loss$chosen), wrt, options)
    result <- setNames(lapply(model$fields, function(f) out[[f]]$detach()), model$fields)
    result$loss <- loss$loss$detach()
    result$input_gradient <- recurrent_model_defined_gradient(g[[1]], inputs)
    for (k in seq_along(model$initial_gradient_names)) {
        gradient_name <- model$initial_gradient_names[[k]]
        if (nzchar(gradient_name)) result[[gradient_name]] <- recurrent_model_defined_gradient(g[[k + 1]], initials[[k]])
    }
    result
}

# Loss gradient of every parameter in store order (zeros where unused).
loss_gradients <- function(model, feed) {
    out <- model$forward(model$store, feed$inputs, feed$lengths, recurrent_model_initials(model, feed$inputs$shape()[1]))
    loss <- model$loss_fn(out, feed$labels)$loss
    params <- nn$tensor_list(model$store$parameters())
    options <- nn$GradOptions()
    options$allow_unused <- TRUE
    g <- nn$grad(list(loss), params, options)
    mapply(recurrent_model_defined_gradient, g, params, SIMPLIFY = FALSE)
}

named_gradients <- function(model, gradients) setNames(gradients, model$store$names())

main <- function(arguments = commandArgs(TRUE)) {
    start <- proc.time()[["elapsed"]]
    args <- nn_options(arguments, marked_sequence_demo_defaults(name, nn_full()))
    marked_sequence_demo_check_epochs(args$epochs)
    nn$set_num_threads(4L)
    spec <- gated_sequence_demo_spec(kind)
    root <- args$out
    dir.create(root, recursive = TRUE, showWarnings = FALSE)
    path <- function(file) file.path(root, file)

    # Data: 512 fit / 128 validation / 128 test marked sequences.
    pools <- marked_sequence_data_pools()
    nn$save_npz(path("data.npz"), attr(pools, "arrays"))
    model <- gated_sequence_model(kind, TRUE)
    nn_require(model$store$count() == spec$parameters && model$store$size() == spec$tensors, spec$count_message)
    if (nzchar(args$init)) recurrent_model_load_arrays(model, args$init)
    initial_arrays <- recurrent_model_arrays(model)
    nn$save_npz(path("initial_arrays.npz"), initial_arrays)
    marked_sequence_demo_save_outputs(path("initial_outputs.npz"), evaluate(model, marked_sequence_data_feed(pools[[3]])))

    # Training: Adam .01, batch 32, BatchOrder(512, 32, 26019).
    history <- file(path("history.tsv"), "w")
    batches <- file(path("batches.tsv"), "w")
    writeLines("epoch\tvalidation_ce\tvalidation_accuracy", history)
    writeLines("epoch\tstep\trow", batches)
    order <- nn$BatchOrder(512, 32, 26019)
    updates <- 0L
    for (epoch in seq_len(args$epochs)) {
        order$begin_epoch()
        repeat {
            rows <- order$`_next`()
            if (!length(rows)) break
            x <- marked_sequence_data_rows(pools[[1]], rows)
            if (!updates) {
                marked_sequence_data_save(path("first_batch.npz"), x)
                marked_sequence_demo_save_outputs(path("first_forward.npz"), evaluate(model, marked_sequence_data_feed(x)))
                nn$save_npz(path("initial_gradients.npz"), named_gradients(model, loss_gradients(model, marked_sequence_data_feed(x))))
            }
            # One Adam update: forward, loss, gradient of every parameter, step.
            out <- model$forward(model$store, x$inputs, x$lengths, recurrent_model_initials(model, length(rows)))
            loss <- model$loss_fn(out, x$labels)$loss
            params <- nn$tensor_list(model$store$parameters())
            grad_options <- nn$GradOptions()
            grad_options$allow_unused <- TRUE
            g <- nn$grad(list(loss), params, grad_options)
            model$optimizer$step(nn$tensor_vector(mapply(recurrent_model_defined_gradient, g, params, SIMPLIFY = FALSE)))
            if (!updates) nn$save_npz(path("first_after.npz"), recurrent_model_arrays(model))
            writeLines(sprintf("%d\t%d\t%d", epoch, updates, as.integer(rows)), batches)
            updates <- updates + 1L
        }
        valid <- evaluate(model, marked_sequence_data_feed(pools[[2]]))
        ce <- valid$loss$item()
        accuracy <- marked_sequence_data_accuracy(valid$logits, pools[[2]]$labels)
        writeLines(sprintf("%d\t%.17g\t%.17g", epoch, ce, accuracy), history)
        nn$save_npz(path(sprintf("epoch%d_arrays.npz", epoch)), recurrent_model_arrays(model))
        marked_sequence_demo_save_outputs(path(sprintf("epoch%d_validation.npz", epoch)), valid)
        cat(sprintf("epoch%d validationCE=%g accuracy=%g\n", epoch, ce, accuracy))
    }
    flush(batches)
    first <- marked_sequence_data_feed(marked_sequence_data_load(path("first_batch.npz")))
    nn$save_npz(path("final_gradients.npz"), named_gradients(model, loss_gradients(model, first)))

    # Save -> restore into a fresh inference model.
    recurrent_model_save(model, path("final.npz"))
    deployed <- gated_sequence_model(kind, FALSE)
    recurrent_model_restore(deployed, path("final.npz"))
    nn_require(recurrent_model_same_bits(recurrent_model_arrays(model), recurrent_model_arrays(deployed)), spec$restore_message)
    before <- recurrent_model_arrays(deployed)
    test <- pools[[3]]
    test_feed <- marked_sequence_data_feed(test)
    output <- evaluate(deployed, test_feed)
    prediction <- recurrent_model_predict(deployed, test_feed)
    for (k in names(prediction))
        nn_require(recurrent_model_same_bits(prediction[[k]], output[[k]]), "labelFreePredictionMatchesEvaluation")
    nn$save_npz(path("model_arrays.npz"), recurrent_model_arrays(deployed))
    marked_sequence_demo_save_outputs(path("outputs.npz"), output)
    lengths <- as.integer(test$lengths$values())
    labels <- as.integer(test$labels$values())
    gated_sequence_demo_gate_algebra_audit(kind, output, lengths)
    repeat_prediction <- recurrent_model_predict(deployed, test_feed)
    for (k in names(prediction))
        nn_require(recurrent_model_same_bits(prediction[[k]], repeat_prediction[[k]]), "RNNPredictRepeatBitexact")
    single_logits <- recurrent_model_predict(deployed, marked_sequence_data_feed(marked_sequence_data_rows(test, 0)))$logits$values()
    logits <- output$logits$values()
    for (c in 1:2) nn_near(single_logits[c], logits[c], 2e-6, "singleBatchRNNParityWithResetState")

    # Counterfactual copies: padding garbage, flipped marked value, flipped last valid value.
    test_inputs <- test$inputs$values()
    input_gradient <- output$input_gradient$values()
    states <- output$states$values()
    padded <- cue_flip <- last_flip <- test_inputs
    idx <- function(b, t, c) (b * 12 + t) * 2 + c + 1
    sidx <- function(b, t, j) (b * 13 + t) * 12 + j + 1
    oracle_correct <- 0L
    last_correct <- 0L
    for (b in 0:127) {
        length <- lengths[b + 1]
        if (length < 12) {
            pad <- length:11
            padded[idx(b, pad, 0)] <- 123 + b + pad
            padded[idx(b, pad, 1)] <- -57
            nn_require(all(input_gradient[c(idx(b, pad, 0), idx(b, pad, 1))] == 0), "ALLpaddedInputLogitGradientExactlyZero")
            tt <- rep(pad, each = 12)
            jj <- rep(0:11, length(pad))
            nn_require(all(states[sidx(b, tt + 1, jj)] == states[sidx(b, tt, jj)]), "stateRetainedAcrossEveryPaddingStep")
        }
        marked <- which(test_inputs[idx(b, 0:11, 1)] > 0) - 1
        selected <- sum(test_inputs[idx(b, marked, 0)])
        cue_flip[idx(b, marked, 0)] <- -cue_flip[idx(b, marked, 0)]
        oracle_correct <- oracle_correct + ((selected > 0) == (labels[b + 1] != 0))
        last_correct <- last_correct + ((test_inputs[idx(b, length - 1, 0)] > 0) == (labels[b + 1] != 0))
        last_flip[idx(b, length - 1, 0)] <- -last_flip[idx(b, length - 1, 0)]
    }
    nn_require(oracle_correct == 128, "deterministicMarkerValueOracle")
    pad_data <- marked_sequence_data_copy(test)
    cue_data <- marked_sequence_data_copy(test)
    last_data <- marked_sequence_data_copy(test)
    pad_data$inputs <- nn$from_vector(padded, c(128, 12, 2))
    cue_data$inputs <- nn$from_vector(cue_flip, c(128, 12, 2))
    cue_data$labels <- marked_sequence_data_int32(1L - labels)
    last_data$inputs <- nn$from_vector(last_flip, c(128, 12, 2))
    pad_output <- evaluate(deployed, marked_sequence_data_feed(pad_data))
    cue_output <- evaluate(deployed, marked_sequence_data_feed(cue_data))
    last_output <- evaluate(deployed, marked_sequence_data_feed(last_data))
    for (k in names(output))
        nn_require(recurrent_model_same_bits(output[[k]], pad_output[[k]]),
            paste0("arbitraryPaddingValuesCannotAffectAnyOutputOrGradient:", k))
    last_states <- last_output$states$values()
    for (b in 0:127) {
        tt <- rep(0:(lengths[b + 1] - 1), each = 12)
        jj <- rep(0:11, lengths[b + 1])
        nn_require(all(last_states[sidx(b, tt, jj)] == states[sidx(b, tt, jj)]),
            "changingLastValidTokenCannotChangeEarlierHiddenStates")
    }
    marked_sequence_data_save(path("padding_inputs.npz"), pad_data)
    marked_sequence_data_save(path("cue_flip_inputs.npz"), cue_data)
    marked_sequence_data_save(path("last_flip_inputs.npz"), last_data)
    marked_sequence_demo_save_outputs(path("padding_outputs.npz"), pad_output)
    marked_sequence_demo_save_outputs(path("cue_flip_outputs.npz"), cue_output)
    marked_sequence_demo_save_outputs(path("last_flip_outputs.npz"), last_output)
    empty <- marked_sequence_data_rows(test, 0:3)
    empty$lengths <- nn$zeros(4, nn$DType$Int32)
    empty_output <- evaluate(deployed, marked_sequence_data_feed(empty))
    nn_require(all(empty_output$states$values() == 0), "zeroLengthSequenceRetainsZeroInitialState")
    nn_require(all(empty_output$input_gradient$values() == 0), "zeroLengthAllInputGradientsZero")
    marked_sequence_data_save(path("zero_length_inputs.npz"), empty)
    marked_sequence_demo_save_outputs(path("zero_length_outputs.npz"), empty_output)
    marked_sequence_data_heatmap(path("sequences.png"), test$inputs)
    marked_sequence_data_heatmap(path("input_gradients.png"), output$input_gradient, TRUE)
    predicted <- marked_sequence_data_predictions(output$logits)
    ids <- as.integer(test$source_ids$values())
    last_value <- as.integer(test_inputs[idx(0:127, lengths - 1, 0)] > 0)
    writeLines(c("source_id\tlength\tlabel\tpredicted\tmarked_oracle\tlast_value_baseline",
        sprintf("%d\t%d\t%d\t%d\t%d\t%d", ids, lengths, labels, predicted, labels, last_value)), path("predictions.tsv"))
    nn_require(recurrent_model_same_bits(before, recurrent_model_arrays(deployed)),
        "allRecurrentInferenceAndGradientControlsWeightsImmutable")

    # Test metrics, the confusion matrix and the report.
    test_ce <- output$loss$item()
    accuracy <- marked_sequence_data_accuracy(output$logits, test$labels)
    cue_accuracy <- marked_sequence_data_accuracy(cue_output$logits, cue_data$labels)
    last_accuracy <- marked_sequence_data_accuracy(last_output$logits, test$labels)
    confusion <- table(factor(labels, 0:1), factor(predicted, 0:1))
    cat("Confusion matrix (rows=truth, columns=prediction):\ntruth/pred     0     1\n")
    for (r in 1:2) cat(sprintf("%10d %5d %5d\n", r - 1L, confusion[r, 1], confusion[r, 2]))
    writeLines(paste0("{", sprintf(paste0("\"parameters\":%d,\"parameter_tensors\":%d,\"hidden_units\":12,\"maximum_steps\":12,",
        "\"epochs\":%d,\"optimizer_updates\":%d,\"test_ce\":%.17g,\"test_accuracy\":%.17g,\"cue_flip_accuracy\":%.17g,",
        "\"last_distractor_flip_accuracy\":%.17g,\"last_value_baseline_accuracy\":%.17g,"),
        as.integer(spec$parameters), as.integer(spec$tensors), as.integer(args$epochs), updates, test_ce, accuracy,
        cue_accuracy, last_accuracy, last_correct / 128),
        "\"majority_baseline_accuracy\":0.5,\"marked_value_oracle_accuracy\":1}"), path("metrics.json"))
    html <- paste0("<!doctype html><meta charset='utf-8'><title>", spec$title,
        "</title><style>body{font:16px system-ui;margin:30px;max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style>",
        "<h1>", spec$title, "</h1><p>", spec$description, "</p>",
        "<p>Synthetic marked-token memory task: 512 fit / 128 validation / 128 test, balanced labels, valid length 4..12, one ",
        "marker before the last token whose +/-1 value sets the binary target; padding zeros. Three epochs, 48 updates. The task ",
        "has a perfect deterministic marker-times-value oracle; it is not a natural-language benchmark.</p>",
        "<h2>First 24 held-out sequences</h2><p>12 time columns; blue=-1, red=+1, white=padding, black top strip=marked ",
        "token.</p><img src='sequences.png'><h2>Target-logit input-value gradients</h2><p>Same cases, global symmetric colour ",
        "peak. Masked padding derivatives are exactly zero.</p><img src='input_gradients.png'>",
        sprintf("<p>Test CE %g, accuracy %g, cue-flip accuracy %g, last-distractor-flip accuracy %g, last-value baseline %g.",
            test_ce, accuracy, cue_accuracy, last_accuracy, last_correct / 128),
        " See metrics.json and predictions.tsv.</p>")
    cat(html, file = path("index.html"))
    close(history)
    close(batches)
    seconds <- proc.time()[["elapsed"]] - start
    cat(sprintf("testCE=%g accuracy=%g cueFlip=%g lastBaseline=%g;%s PASS (%g s)\nArtifacts:%s\n", test_ce, accuracy,
        cue_accuracy, last_correct / 128, spec$pass_tag, seconds, marked_sequence_demo_quoted(path("index.html"))))
    marked_sequence_demo_compare_with_tflow(root, args$epochs, args$init, args$tf,
        c("initial_outputs.npz", "first_forward.npz", "initial_gradients.npz", "first_after.npz", "epoch1_arrays.npz",
            sprintf("epoch%d_arrays.npz", args$epochs), "final_gradients.npz", "outputs.npz", "cue_flip_outputs.npz",
            "last_flip_outputs.npz"))
    0L
}

if (sys.nframe() == 0L) quit(status = main())
