# Native R NN example: key_value, explicit loop over canonical C++ memory components.
# CMD: nix develop .#r --command Rscript examples/r/nn/4_recurrent_and_sequence_models/4.3_memory_augmented_networks/key_value.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT", getwd()), "examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R", package="datamungerNN", mustWork=TRUE)
source(.nn_common, local=TRUE)
rm(.nn_common)
nn_helper("memory_records")
component <- nn_components$memory_components
check <- nn_components$check
NAME <- "key_value"
DATASET <- "memnet"
KIND <- component$Kind_KeyValue
COUNT <- 264L
TENSORS <- 4L
SEED <- 45407L
LR <- .01
FIRST_FORWARD <- FALSE
EMPTY_FIELDS <- c("keys", "values", "attention", "reads", "input_gradient")
STREAM_TOL <- 2e-5
PADDING <- .9
DESCRIPTION <- "<p>Synthetic question answering over up to six (key, value) token facts. A two-hop key-value memory network addresses keys and reads values through one shared embedding, applies a learned transform R_k after each hop and scores the four answers; softmax cross-entropy, three epochs.</p>"

# Options: epochs=N out=PATH init=PATH tf=PATH. Data: $DATAMUNGE_DATA_DIR/memnet.
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
    component$PrepareDataset(source, FALSE)
    for (filename in c("data.npz", "manifest.json")) if (file.exists(file.path(source, filename)))
        file.copy(file.path(source, filename), file.path(root, filename), overwrite=TRUE)
    component$Fixtures(KIND, root)
    all_arrays <- nn$load_npz(file.path(root, "data.npz"))
    keys <- c("facts", "query", "target", "lengths", "source_ids")
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
        writeLines("epoch\tvalidation_ce", history)
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
            cat(sprintf("epoch=%d validationCE=%.6g\n", epoch, ce))
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
    normal_in <- memory_rows(test, 0:1)
    normal <- memory_evaluate(deployed, normal_in)
    lengths <- as.integer(normal_in$lengths$values())
    facts <- as.integer(normal_in$facts$values())
    padded <- facts
    for (batch in 0:1) if (lengths[batch + 1L] < 6L)
        for (slot in seq.int(lengths[batch + 1L], 5L)) for (role in 0:1)
            padded[(batch * 6L + slot) * 2L + role + 1L] <- 11L
    pad <- normal_in
    pad$facts <- nn$cast(nn$from_vector(padded, c(2L, 6L, 2L)), nn$DType$Int32)
    padout <- memory_evaluate(deployed, pad)
    check$Require(memory_same(normal, padout), "all masked fields/partials invariant")
    memory_save(file.path(root, "padding_inputs.npz"), pad)
    memory_save(file.path(root, "padding_outputs.npz"), padout)
    empty <- normal_in
    empty$lengths <- nn$zeros(2L, nn$DType$Int32)
    emptyout <- memory_evaluate(deployed, empty)
    for (key in EMPTY_FIELDS) check$Require(all(emptyout[[key]]$values() == 0), "empty memory reads/partials zero")
    memory_save(file.path(root, "empty_inputs.npz"), empty)
    memory_save(file.path(root, "empty_outputs.npz"), emptyout)
    swapped <- facts
    for (batch in 0:1) if (lengths[batch + 1L] >= 2L)
        for (slot in seq_len(lengths[batch + 1L] %/% 2L) - 1L) for (role in 0:1) {
            a <- (batch * 6L + slot) * 2L + role + 1L
            b <- (batch * 6L + lengths[batch + 1L] - 1L - slot) * 2L + role + 1L
            swapped[c(a, b)] <- swapped[c(b, a)]
        }
    shuffled <- normal_in
    shuffled$facts <- nn$cast(nn$from_vector(swapped, c(2L, 6L, 2L)), nn$DType$Int32)
    shuffledout <- memory_evaluate(deployed, shuffled)
    for (key in c("logits", "states", "reads", "loss"))
        check$Near(nn$max(nn$abs(nn$sub(shuffledout[[key]], normal[[key]])))$item(), 0, 2e-6, "fact-order invariance")
    memory_save(file.path(root, "shuffled_inputs.npz"), shuffled)
    memory_save(file.path(root, "shuffled_outputs.npz"), shuffledout)
    prediction <- nn$argmax(logits, 1L)
    confusion <- nn$ConfusionMatrix(4L)
    confusion$add(test$target, prediction)
    cells <- matrix(confusion$to_tensor()$values(), nrow = 4L, byrow = TRUE)
    cat("confusion (truth rows, predicted columns):\n")
    for (i in 1:4) cat(paste(cells[i, ], collapse = " "), "\n", sep = "")
    correct <- sum(prediction$values() == test$target$values())
    metric_fields <- sprintf('"test_ce":%.6g,"correct":%d', out$loss$item(), correct)
    summary <- sprintf("testCE=%.6g correct=%d/32;restore/padding/empty/factOrder/labelFree/immutable PASS", out$loss$item(), correct)

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
    component$CompareRecords(root, args$tf, args$init, comparison, "test_ce")
    for (filename in character()) if (nzchar(args$tf) && file.exists(file.path(args$tf, filename)))
        check$Require(component$CompareNpz(file.path(root, filename), file.path(args$tf, filename)) <= 1e-5,
                       paste(filename, "matches recorded fixture"))
    0L
}
if (sys.nframe() == 0L) main()
