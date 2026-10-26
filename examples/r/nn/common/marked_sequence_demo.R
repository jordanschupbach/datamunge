# Records and tflow comparisons for the marked-token memory experiments of
# tflow's 4.1 recurrent-cell examples (examples/nn/common/marked_sequence_demo.hpp).
#
# The experiment itself (data pools, training loop, evaluations, counterfactual
# audits, records) is written out in each example script; this file holds the
# option defaults, .npz/JSON utilities and the comparison with tflow's run
# directory. Parity with tflow: the data and batch order are bit-identical;
# parameters initialise from nn's generator, so init=PATH (tflow's
# initial_arrays.npz) is needed to replay tflow's run.
if (!exists("recurrent_model_new", inherits = TRUE)) nn_helper("recurrent_model")

# tflow's output/models/<name> (where its examples write their runs).
marked_sequence_demo_tflow_run_dir <- function(name)
    file.path(path.expand("~"), "projects", "tflow_examples", "output", "models", name)

# Option defaults of ParseDemoOptions: epochs=3 out=PATH init=PATH tf=PATH.
marked_sequence_demo_defaults <- function(name, full = FALSE)
    list(epochs = 3L, out = file.path(nn_output_root(full), "models", name), init = "",
        tf = marked_sequence_demo_tflow_run_dir(name), full = full)

marked_sequence_demo_check_epochs <- function(epochs) nn_require(epochs > 0 && epochs <= 100, "threeCheapDefaultEpochs")

# C++ default ostream formatting of a double (%g, 6 significant digits).
marked_sequence_demo_g <- function(x) sprintf("%g", x)

# std::filesystem::path stream output (quoted).
marked_sequence_demo_quoted <- function(path) paste0("\"", path, "\"")

marked_sequence_demo_save_outputs <- function(path, outputs) nn$save_npz(path, recurrent_model_named(outputs))

# Max |ours - tflow| over the arrays both .npz files contain; -1 when tflow's file is missing.
marked_sequence_demo_compare_npz <- function(ours, theirs) {
    if (!file.exists(theirs)) return(-1)
    a <- nn$load_npz(ours)
    b <- nn$load_npz(theirs)
    worst <- 0
    for (name in intersect(names(a), names(b))) worst <- max(worst, recurrent_model_max_abs_diff(a[[name]], b[[name]]))
    worst
}

marked_sequence_demo_read_text <- function(path) readBin(path, "raw", file.size(path))

# The number after "key": in a flat JSON object (NaN when absent).
marked_sequence_demo_json_number <- function(text, key) {
    if (is.raw(text)) text <- rawToChar(text)
    match <- regmatches(text, regexec(paste0("\"", key, "\":([-+.0-9eE]+)"), text))[[1]]
    if (length(match) < 2L) NaN else as.numeric(match[2])
}

# Compares this run with tflow's run directory: data and batch order always
# (identical; with another epoch budget the shorter batch record is a prefix of
# the longer), and with init= every replayed record.
marked_sequence_demo_compare_with_tflow <- function(root, epochs, init, tf, records, test_key = "test_ce") {
    if (!nzchar(tf) || !file.exists(file.path(tf, "data.npz"))) {
        cat("tflow run ", marked_sequence_demo_quoted(tf), " not found; skipping the comparison\n", sep = "")
        return(invisible())
    }
    nn_require(marked_sequence_demo_compare_npz(file.path(root, "data.npz"), file.path(tf, "data.npz")) == 0,
        "dataBitIdenticalToTflow")
    if (file.exists(file.path(tf, "batches.tsv"))) {
        ours <- marked_sequence_demo_read_text(file.path(root, "batches.tsv"))
        theirs <- marked_sequence_demo_read_text(file.path(tf, "batches.tsv"))
        n <- min(length(ours), length(theirs))
        nn_require(identical(ours[seq_len(n)], theirs[seq_len(n)]), "batchOrderIdenticalToTflow")
    }
    line <- "vs tflow: data.npz and batches.tsv identical"
    if (!nzchar(init)) {
        cat(line, " (pass init=", file.path(tf, "initial_arrays.npz"), " to replay tflow's run)\n", sep = "")
        return(invisible())
    }
    worst_initial <- 0
    worst_final <- 0
    for (name in records) {
        d <- marked_sequence_demo_compare_npz(file.path(root, name), file.path(tf, name))
        if (d < 0) next
        line <- paste0(line, "\n  ", name, " max|diff|=", marked_sequence_demo_g(d))
        if (startsWith(name, "initial") || startsWith(name, "first")) worst_initial <- max(worst_initial, d)
        else worst_final <- max(worst_final, d)
    }
    cat(line, "\n", sep = "")
    nn_require(worst_initial <= 1e-5, "replayedInitialForwardGradientsAndFirstUpdateMatchTflow")
    ours <- marked_sequence_demo_json_number(marked_sequence_demo_read_text(file.path(root, "metrics.json")), test_key)
    theirs <- marked_sequence_demo_json_number(marked_sequence_demo_read_text(file.path(tf, "metrics.json")), test_key)
    if (!is.nan(theirs)) {
        cat("  ", test_key, " ours=", marked_sequence_demo_g(ours), " tflow=", marked_sequence_demo_g(theirs),
            " (trained records max|diff|=", marked_sequence_demo_g(worst_final), ")\n", sep = "")
        nn_near(ours, theirs, 5e-3, "replayedTrainedTestLossMatchesTflow")
    }
}
