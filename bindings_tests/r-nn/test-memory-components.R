memory_component_batch <- function(copy_task) {
    nn <- datamungerNN::nn
    integer_tensor <- function(values, shape) nn$cast(nn$from_vector(values, shape), nn$DType$Int32)
    if (!copy_task) return(list(
        facts = integer_tensor(c(1, 8, 2, 9, 0, 0, 0, 0, 0, 0, 0, 0,
                                 1, 8, 2, 9, 3, 10, 0, 0, 0, 0, 0, 0), c(2L, 6L, 2L)),
        query = integer_tensor(c(1, 2), 2L), target = integer_tensor(c(0, 1), 2L),
        lengths = integer_tensor(c(2, 3), 2L)))
    inputs <- numeric(72L)
    targets <- numeric(54L)
    for (batch in 0:1) {
        length <- c(2L, 3L)[batch + 1L]
        for (step in seq_len(length) - 1L) for (bit in 0:2) {
            value <- (batch + step + bit) %% 2L
            inputs[(batch * 9L + step) * 4L + bit + 1L] <- value
            targets[(batch * 9L + length + 1L + step) * 3L + bit + 1L] <- value
        }
        inputs[(batch * 9L + length) * 4L + 4L] <- 1
    }
    list(input = nn$from_vector(inputs, c(2L, 9L, 4L)),
         target = nn$from_vector(targets, c(2L, 9L, 3L)),
         lengths = integer_tensor(c(2, 3), 2L))
}

testthat::test_that("C++ memory models support R-owned autograd, Adam and restore", {
    nn <- datamungerNN::nn
    component <- datamungerNN::nn_components$memory_components
    for (name in c("Ntm", "Dnc", "MemNet", "KeyValue")) {
        copy_task <- name %in% c("Ntm", "Dnc")
        batch <- nn$named_tensors(memory_component_batch(copy_task))
        kind <- component[[paste0("Kind_", name)]]
        model <- component$Model(kind, batch)
        parameters <- model$parameters()
        testthat::expect_equal(parameters$count(), c(Ntm = 851, Dnc = 799, MemNet = 312, KeyValue = 264)[[name]])
        testthat::expect_equal(parameters$size(), c(Ntm = 8L, Dnc = 7L, MemNet = 3L, KeyValue = 4L)[[name]])
        before <- lapply(nn$named_list(model$Arrays()), function(tensor) tensor$values())
        cfg <- nn$OptimizerConfig()
        cfg$kind <- nn$OptimizerKind$Adam
        optimizer <- nn$Optimizer(parameters, if (copy_task) 0.005 else 0.01, cfg)
        outputs <- model$Forward(model$Leaves(batch), batch)
        loss <- model$Loss(outputs, batch)
        rm(outputs)
        invisible(gc())
        options <- nn$GradOptions()
        options$allow_unused <- TRUE
        gradients <- nn$grad(loss, nn$tensor_list(parameters$parameters()), options)
        testthat::expect_true(all(vapply(gradients, function(tensor) tensor$defined() &&
                                          all(is.finite(tensor$values())), logical(1))))
        optimizer$step(nn$tensor_vector(gradients))
        testthat::expect_true(is.finite(loss$item()) && loss$item() > 0)
        testthat::expect_false(identical(lapply(nn$named_list(model$Arrays()), function(tensor) tensor$values()), before))
        diagnostic <- nn$named_list(model$Evaluate(batch))
        testthat::expect_true(is.finite(diagnostic$loss$item()) && diagnostic$input_gradient$defined())
        testthat::expect_true(nn$equal_bits(diagnostic$logits, model$Predict(batch)))
        corrupt <- nn$named_list(batch)
        corrupt$target <- nn$zeros_like(corrupt$target)
        testthat::expect_true(nn$equal_bits(model$Predict(batch), model$Predict(nn$named_tensors(corrupt))))
        checkpoint <- tempfile(fileext = ".npz")
        nn$save_npz(checkpoint, nn$named_list(model$Arrays()))
        deployed <- component$Model(kind, batch)
        deployed$Load(checkpoint)
        testthat::expect_true(nn$equal_bits(model$Predict(batch), deployed$Predict(batch)))
        unlink(checkpoint)
    }
})

testthat::test_that("named tensor marshalling retains tensors after source collection", {
    nn <- datamungerNN::nn
    records <- nn$named_tensors(list(signal = nn$from_vector(c(2, 4, 8))))
    result <- nn$named_list(records)
    rm(records)
    invisible(gc())
    testthat::expect_equal(result$signal$values(), c(2, 4, 8))
    testthat::expect_error(nn$named_tensors(list(nn$ones(1L))), "unique nonempty names")
    testthat::expect_error(nn$named_tensors(setNames(list(nn$ones(1L), nn$zeros(1L)), c("x", "x"))),
                           "unique nonempty names")
})
