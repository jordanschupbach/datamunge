# Native R NN example: ntm, explicit loop over canonical C++ memory components.
# CMD: nix develop .#r --command Rscript examples/r/nn/4_recurrent_and_sequence_models/4.3_memory_augmented_networks/ntm.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT", getwd()), "examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R", package="datamungerNN", mustWork=TRUE)
source(.nn_common, local=TRUE)
rm(.nn_common)
nn_helper("memory_records")
component <- nn_components$memory_components
check <- nn_components$check
NAME <- "ntm"
DATASET <- "ntm"
KIND <- component$Kind_Ntm
COUNT <- 851L
TENSORS <- 8L
SEED <- 45107L
LR <- .005
FIRST_FORWARD <- FALSE
EMPTY_FIELDS <- c("keys", "values", "attention", "reads", "input_gradient")
STREAM_TOL <- 2e-5
PADDING <- .9
DESCRIPTION <- "<p>Binary copy: three-bit vectors, a delimiter, then recall. Canonical Ntm memory model, masked recall-bit cross-entropy, Adam and three default epochs.</p>"

# Options: epochs=N out=PATH init=PATH tf=PATH. Data: $DATAMUNGE_DATA_DIR/ntm.
main <- function(arguments=commandArgs(trailingOnly=TRUE)) {
    started <- proc.time()[[3L]]
    nn$set_num_threads(4L)
    args <- nn_options(arguments, list(epochs=3L, out="", init="",
        tf=file.path(path.expand("~"), "projects/tflow_examples/output/models", NAME)))
    epochs <- args$epochs
    check$Require(epochs > 0L && epochs <= 100L, "three cheap default epochs")
    root <- if (nzchar(args$out)) args$out else file.path(nn_output_root(), "models", NAME)
    dir.create(root, recursive=TRUE, showWarnings=FALSE)
    source <- file.path(nn_data_root(), DATASET)
    component$PrepareDataset(source, TRUE)
    for (filename in c("data.npz", "manifest.json")) if (file.exists(file.path(source, filename)))
        file.copy(file.path(source, filename), file.path(root, filename), overwrite=TRUE)
    component$Fixtures(KIND, root)
    all_arrays <- nn$load_npz(file.path(root, "data.npz"))
    keys <- c("input", "target", "lengths", "source_ids")
    pools <- lapply(c("fit", "validation", "test"), function(split)
        setNames(lapply(keys, function(key) all_arrays[[paste0(split, "_", key)]]), keys))
    fit <- pools[[1L]]
    validation <- pools[[2L]]
    test <- pools[[3L]]
    example <- memory_rows(fit, 0L)
    model <- component$Model(KIND, memory_records(example))
    params <- model$parameters()
    check$Require(params$count() == COUNT && params$size() == TENSORS, "canonical parameter census")
    if (nzchar(args$init)) model$Load(args$init)
    cfg <- nn$OptimizerConfig()
    cfg$kind <- nn$OptimizerKind$Adam
    optimizer <- nn$Optimizer(params, LR, cfg)
    named_gradients <- function(batch) {
        values <- model$Forward(model$Leaves(memory_records(batch)), memory_records(batch))
        loss <- model$Loss(values, memory_records(batch))
        grad_options <- nn$GradOptions()
        grad_options$allow_unused <- TRUE
        tensors <- nn$tensor_list(params$parameters())
        gradients <- nn$grad(loss, tensors, grad_options)
        nn$named_tensors(setNames(lapply(seq_along(gradients), function(i) {
            if (gradients[[i]]$defined()) gradients[[i]]$detach() else nn$zeros_like(tensors[[i]]$detach())
        }), params$names()))
    }
    memory_save(file.path(root, "initial_arrays.npz"), model$Arrays())
    memory_save(file.path(root, "initial_outputs.npz"), memory_evaluate(model, test))
    order <- nn$BatchOrder(fit[[1L]]$size(0L), 8L, SEED)
    updates <- 0L
    first <- NULL
    history <- file(file.path(root, "history.tsv"), "wt")
    batches <- file(file.path(root, "batches.tsv"), "wt")
    tryCatch({
        writeLines("epoch\tvalidation_bit_ce", history)
        writeLines("epoch\tstep\trow", batches)
        for (epoch in seq_len(epochs)) {
            order$begin_epoch()
            repeat {
                rows <- order$`_next`()
                if (!length(rows)) break
                batch <- memory_rows(fit, rows)
                if (!updates) {
                    first <- batch
                    memory_save(file.path(root, "first_batch.npz"), batch)
                    memory_save(file.path(root, "initial_gradients.npz"), named_gradients(batch))
                    if (FIRST_FORWARD) memory_save(file.path(root, "first_forward.npz"), memory_evaluate(model, batch))
                }
                leaves <- model$Leaves(memory_records(batch))
                values <- model$Forward(leaves, memory_records(batch))
                loss <- model$Loss(values, memory_records(batch))
                grad_options <- nn$GradOptions()
                grad_options$allow_unused <- TRUE
                tensors <- nn$tensor_list(params$parameters())
                gradients <- nn$grad(loss, tensors, grad_options)
                for (i in seq_along(gradients)) if (!gradients[[i]]$defined())
                    gradients[[i]] <- nn$zeros_like(tensors[[i]]$detach())
                check$Require(is.finite(loss$item()), "finite training loss")
                optimizer$step(nn$tensor_vector(gradients))
                if (!updates) memory_save(file.path(root, "first_after.npz"), model$Arrays())
                for (row in rows) writeLines(sprintf("%d\t%d\t%d", epoch, updates, row), batches)
                updates <- updates + 1L
            }
            ce <- memory_evaluate(model, validation)$loss$item()
            check$Require(is.finite(ce), "finite validation CE")
            writeLines(sprintf("%d\t%.6g", epoch, ce), history)
            cat(sprintf("epoch=%d validationBitCE=%.6g\n", epoch, ce))
        }
    }, finally={close(history); close(batches)})
    memory_save(file.path(root, "final_gradients.npz"), named_gradients(first))
    if (FIRST_FORWARD) memory_save(file.path(root, "final_first_forward.npz"), memory_evaluate(model, first))
    memory_save(file.path(root, "final.npz"), model$Arrays())
    deployed <- component$Model(KIND, memory_records(example))
    deployed$Load(file.path(root, "final.npz"))
    weights <- nn$named_list(deployed$Arrays())
    check$Require(memory_same(nn$named_list(model$Arrays()), weights), "restore bitexact")
    out <- memory_evaluate(deployed, test)
    logits <- deployed$Predict(memory_records(test))
    check$Require(memory_same(logits, out$logits) && memory_same(logits, deployed$Predict(memory_records(test))),
                   "label-free repeat inference")
    lengths <- as.integer(test$lengths$values())
    state_names <- deployed$StateNames()
    for (step_index in 0:8) {
        state <- setNames(lapply(state_names, function(key) nn$select(out[[key]], 1L, step_index)), state_names)
        step <- nn$named_list(deployed$Step(nn$select(test$input, 1L, step_index), memory_records(state)))
        active <- which(step_index <= 2L * lengths) - 1L
        for (key in names(step)) if (length(active)) {
            expected <- nn$select(out[[key]], 1L, step_index + as.integer(key %in% state_names))
            error <- nn$max(nn$abs(nn$sub(nn$take_rows(step[[key]], active), nn$take_rows(expected, active))))$item()
            check$Near(error, 0, STREAM_TOL, "native state/access streaming parity")
        }
    }
    normal_in <- memory_rows(test, 0:1)
    normal <- memory_evaluate(deployed, normal_in)
    padded_values <- normal_in$input$values()
    for (batch in 0:1) if (2L * lengths[batch + 1L] + 1L < 9L)
        for (step_index in seq.int(2L * lengths[batch + 1L] + 1L, 8L)) for (channel in 0:3)
            padded_values[(batch * 9L + step_index) * 4L + channel + 1L] <- PADDING
    pad <- normal_in
    pad$input <- nn$from_vector(padded_values, c(2L, 9L, 4L))
    padout <- memory_evaluate(deployed, pad)
    check$Require(memory_same(normal, padout), "all state/loss/gradient padding invariance")
    memory_save(file.path(root, "padding_inputs.npz"), pad)
    memory_save(file.path(root, "padding_outputs.npz"), padout)
    predictions <- logits$values()
    targets <- test$target$values()
    confusion <- matrix(0L, nrow = 2L, ncol = 2L)
    correct <- total <- exact <- 0L
    for (batch in seq_along(lengths) - 1L) {
        length <- lengths[batch + 1L]
        match <- TRUE
        for (step_index in seq.int(length + 1L, 2L * length)) for (bit in 0:2) {
            index <- (batch * 9L + step_index) * 3L + bit + 1L
            truth <- as.integer(targets[index] > 0)
            predicted <- as.integer(predictions[index] > 0)
            confusion[truth + 1L, predicted + 1L] <- confusion[truth + 1L, predicted + 1L] + 1L
            correct <- correct + as.integer(truth == predicted)
            total <- total + 1L
            match <- match && truth == predicted
        }
        exact <- exact + as.integer(match)
    }
    check$Require(sum(confusion) == total && sum(diag(confusion)) == correct, "all held-out recall bits counted")
    cat("confusion (truth rows, predicted columns; recall bits):\n")
    for (i in 1:2) cat(paste(confusion[i, ], collapse = " "), "\n", sep = "")
    metric_fields <- sprintf('"test_bit_ce":%.6g,"bit_accuracy":%.6g,"exact":%d', out$loss$item(), correct / total, exact)
    summary <- sprintf("testBitCE=%.6g bitAccuracy=%.6g exact=%d/32;restore/nativeStreaming/padding/labelFree/immutable PASS",
                       out$loss$item(), correct / total, exact)

    memory_save(file.path(root, "outputs.npz"), out)
    memory_save(file.path(root, "model_arrays.npz"), deployed$Arrays())
    check$Require(memory_same(nn$named_list(deployed$Arrays()), weights), "immutable inference")
    metrics <- sprintf('{"parameters":%d,"epochs":%d,"updates":%d,%s}', COUNT, epochs, updates, metric_fields)
    writeLines(metrics, file.path(root, "metrics.json"))
    writeLines(paste0("<!doctype html><meta charset='utf-8'><title>", NAME,
        "</title><style>body{font:16px system-ui;margin:30px;max-width:1000px}</style><h1>", NAME,
        "</h1>", DESCRIPTION, "<p>", summary, "</p><p>See metrics.json, history.tsv and the .npz records.</p>"), file.path(root, "index.html"))
    cat(summary, sprintf(" (%.3g s)\nArtifacts:", proc.time()[[3L]] - started), file.path(root, "index.html"), "\n", sep="")
    comparison <- c("first_batch.npz", "initial_arrays.npz", "initial_outputs.npz", "initial_gradients.npz",
                    "first_after.npz", "final_gradients.npz", "model_arrays.npz", "outputs.npz")
    if (FIRST_FORWARD) comparison <- c(comparison, "first_forward.npz", "final_first_forward.npz")
    component$CompareRecords(root, args$tf, args$init, comparison, "test_bit_ce")
    for (filename in c("address_fixture.npz")) if (nzchar(args$tf) && file.exists(file.path(args$tf, filename)))
        check$Require(component$CompareNpz(file.path(root, filename), file.path(args$tf, filename)) <= 1e-5,
                       paste(filename, "matches recorded fixture"))
    0L
}
if (sys.nframe() == 0L) main()
