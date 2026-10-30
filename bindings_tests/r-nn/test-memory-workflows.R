testthat::test_that("native memory examples run their complete workflows", {
    nn <- datamungerNN::nn
    data <- tempfile("memory-data-")
    datamungerNN::nn_components$memory_components$PrepareData(data)
    previous <- Sys.getenv("DATAMUNGE_DATA_DIR", unset=NA_character_)
    Sys.setenv(DATAMUNGE_DATA_DIR=data)
    on.exit({
        if (is.na(previous)) Sys.unsetenv("DATAMUNGE_DATA_DIR") else Sys.setenv(DATAMUNGE_DATA_DIR=previous)
        unlink(data, recursive=TRUE)
    }, add=TRUE)
    for (name in c("ntm", "dnc", "memnet", "key_value")) {
        output <- tempfile(paste0(name, "-"))
        testthat::expect_equal(datamungerNN::nn_example_run(name, c("epochs=3", paste0("out=", output), "tf=")), 0L)
        metrics <- readLines(file.path(output, "metrics.json"))
        testthat::expect_match(metrics, '"epochs":3,"updates":48')
        testthat::expect_length(readLines(file.path(output, "history.tsv")), 4L)
        first <- nn$load_npz(file.path(output, "initial_arrays.npz"))
        final <- nn$load_npz(file.path(output, "model_arrays.npz"))
        testthat::expect_setequal(names(first), names(final))
        testthat::expect_true(any(vapply(names(first), function(key) !nn$equal_bits(first[[key]], final[[key]]), logical(1))))
        testthat::expect_true(file.exists(file.path(output, "padding_outputs.npz")))
        testthat::expect_true(file.exists(file.path(output, "index.html")))
        unlink(output, recursive=TRUE)
    }
})
