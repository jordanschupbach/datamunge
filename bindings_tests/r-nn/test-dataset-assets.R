testthat::test_that("dataset preparation is shared with C++ and preserves cached data", {
    saved <- Sys.getenv(c("DATAMUNGE_DATA_DIR", "DATAMUNGE_DATA_OFFLINE"), unset=NA_character_)
    on.exit(for (name in names(saved)) {
        if (is.na(saved[[name]])) Sys.unsetenv(name)
        else do.call(Sys.setenv, setNames(list(saved[[name]]), name))
    })
    root <- tempfile("dataset-assets-")
    dir.create(root)
    on.exit(unlink(root, recursive=TRUE), add=TRUE)
    Sys.setenv(DATAMUNGE_DATA_DIR=root, DATAMUNGE_DATA_OFFLINE="1")
    assets <- nn_components$dataset_assets
    testthat::expect_identical(assets$DataRoot(), root)
    testthat::expect_error(assets$EnsureMnist(file.path(root, "missing")), "DATAMUNGE_DATA_OFFLINE")
    testthat::expect_false(dir.exists(file.path(root, "missing")))
    # A supplied test split is sufficient; no training archive is fetched.
    file <- file.path(root, "test_batch.bin")
    writeBin(as.raw(c(4, rep(128, 3072))), file)
    data <- assets$LoadCifar10(root, "test")
    testthat::expect_equal(data$labels$values(), 4)
    testthat::expect_equal(as.numeric(data$images$shape()), c(1,32,32,3))
    testthat::expect_identical(list.files(root), "test_batch.bin")
    memory_components <- nn_components$memory_components
    memory_components$PrepareDataset(file.path(root,"copy"), TRUE)
    testthat::expect_true(file.exists(file.path(root,"copy","data.npz")))
    testthat::expect_error(assets$LoadMnist(root,"invalid"), "split must be train or test")
    testthat::expect_error(assets$EnsureSentences(file.path(root,"sentence")), "DATAMUNGE_DATA_OFFLINE")
})
