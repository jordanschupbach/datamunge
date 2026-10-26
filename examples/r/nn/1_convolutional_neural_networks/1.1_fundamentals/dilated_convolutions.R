# Native R NN example. Atrous convolutions: exact dilation, gridding and learning distant context.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
# CMD: just r-example dilated_convolutions
# A 3x3 filter at rate d has effective width 2d+1 and nine learned taps per
# channel pair. Compare zero-expanded filters with native dilated nn$conv2d at
# rates 1,2,3 under SAME/VALID padding, check input/filter derivatives
# numerically, and check that native and expanded-filter gradients agree.
# Compare stacks with rates [1,1,1], [2,2,2], [1,2,4]: positive-kernel input
# gradients reveal dense 7x7, sparse 13x13 and dense 15x15 dependency patterns.
# Finally fit a nine-tap, rate-2 teacher with nine-parameter rate-1/rate-2
# students on independent random images and compare held-out MSE. Inputs come
# from the same std::mt19937 streams as the C++ (bit-identical).
# Support arrays: output/nn/r/<mode>/visualizations/dilated_convolutions/supports.npz.
# C++ reference: examples/cpp/nn/1_convolutional_neural_networks/1.1_fundamentals/dilated_convolutions.cpp

nn_helper("checks")
nn_helper("dilated")
nn_helper("receptive_field")

f32 <- function(values) nn$from_vector(values, length(values))$values()

g <- function(value) sprintf("%.6g", value)

# A fresh std::normal_distribution<float>(0, .5) over the shared std::mt19937.
random_tensor <- function(shape, rng) {
    rng$reset_normal()
    nn$from_vector(rng$normal32_values(prod(shape), 0, 0.5), shape)
}

max_difference <- function(a, b) {
    nn_require(identical(as.numeric(a$shape()), as.numeric(b$shape())), "dilated output shape")
    max(f32(abs(a$contiguous()$values() - b$contiguous()$values())))
}

equivalence <- function() {
    rng <- nn$Mt19937(7)
    pixels <- random_tensor(c(2, 11, 12, 2), rng)
    filter <- random_tensor(c(3, 3, 2, 3), rng)
    for (dilation in 1:3) for (padding in c("SAME", "VALID")) {
        pad <- if (padding == "SAME") nn$Padding$Same else nn$Padding$Valid
        forward_and_gradients <- function(a, b, derivatives) {
            x <- a$clone()
            w <- b$clone()
            x$requires_grad_()
            w$requires_grad_()
            r <- list()
            r$native <- nn$conv2d(x, w, nn$conv_options(1L, pad, dilation))
            r$portable <- dilated_portable_dilated_conv2d(x, w, dilation, padding)
            r$objective <- nn$mean(nn$square(r$portable))
            if (derivatives) {
                r$gradients <- nn$grad(r$objective, list(x, w))
                r$native_gradients <- nn$grad(nn$mean(nn$square(r$native)), list(x, w))
            }
            r
        }
        result <- forward_and_gradients(pixels, filter, TRUE)
        maximum <- max_difference(result$native, result$portable)
        nn_near(maximum, 0, 2e-6, "native and expanded-filter dilation")
        native_gradient_error <- max(max_difference(result$gradients[[1L]], result$native_gradients[[1L]]),
                                     max_difference(result$gradients[[2L]], result$native_gradients[[2L]]))
        nn_near(native_gradient_error, 0, 2e-6, "native dilated backprop equals expanded-filter gradients")
        gradient_error <- 0
        for (which in 0:1) {
            source <- if (which == 0) pixels else filter
            base <- source$values()
            analytic <- result$gradients[[which + 1L]]$values()
            for (i in c(0, source$numel() %/% 2, source$numel() - 1)) {
                plus <- base
                minus <- base
                plus[i + 1] <- f32(plus[i + 1] + f32(0.002))
                minus[i + 1] <- f32(minus[i + 1] - f32(0.002))
                p <- nn$from_vector(plus, source$shape())
                m <- nn$from_vector(minus, source$shape())
                upper <- if (which == 0) forward_and_gradients(p, filter, FALSE) else forward_and_gradients(pixels, p, FALSE)
                lower <- if (which == 0) forward_and_gradients(m, filter, FALSE) else forward_and_gradients(pixels, m, FALSE)
                finite <- f32(upper$objective$item() - lower$objective$item()) / 0.004
                error <- abs(finite - analytic[i + 1])
                gradient_error <- max(gradient_error, error)
                nn_near(error, 0, 8e-5, "dilated finite-difference derivative")
            }
        }
        cat(sprintf("  rate=%d %s: forward error=%s; derivative error=%s; native-vs-expanded gradient error=%s\n",
                    dilation, padding, g(maximum), g(gradient_error), g(native_gradient_error)))
    }
}

gridding <- function() {
    arrays <- list()
    schedules <- list(c(1, 1, 1), c(2, 2, 2), c(1, 2, 4))
    kernel <- nn$ones(c(3, 3, 1, 1))
    for (schedule in seq_along(schedules)) {
        x <- nn$zeros(c(1, 33, 33, 1))
        x$requires_grad_()
        h <- x
        field <- receptive_field_new()
        name <- ""
        for (dilation in schedules[[schedule]]) {
            h <- dilated_portable_dilated_conv2d(h, kernel, dilation)
            field <- receptive_field_after(field, 3, 1, dilation, dilation)
            name <- paste0(name, dilation)
        }
        center <- nn$sum(nn$slice(h, c(0, 16, 16, 0), c(1, 1, 1, 1)))
        gradient <- nn$grad(center, list(x))[[1L]]$clone()
        connected <- matrix(gradient$values() > 0, nrow = 33, byrow = TRUE)
        bounds <- receptive_field_bounds(field, 16)
        coordinate <- 0:32
        inside_axis <- coordinate >= bounds[1L] & coordinate < bounds[2L]
        expected <- outer(inside_axis, inside_axis, "&")
        if (schedule == 2L) expected <- expected & outer(coordinate %% 2 == 0, coordinate %% 2 == 0, "&")
        nn_require(identical(connected, expected), "exact dense or gridded dependency pattern")
        arrays[[paste0("rates_", name)]] <- gradient
        cat(sprintf("  rates %s: bounding width=%d; connected pixels=%d/%d\n", name, as.integer(field$width),
                    sum(connected), as.integer(field$width^2)))
    }
    directory <- nn_output_dir("visualizations/dilated_convolutions")
    nn$save_npz(file.path(directory, "supports.npz"), arrays[sort(names(arrays), method = "radix")])
}

learn_context <- function() {
    parameters <- nn$ParameterStore(19)
    truth <- nn$from_vector(c(0.4, -0.7, 0.2, 0.6, 1, -0.3, -0.5, 0.8, 0.25), c(3, 3, 1, 1))
    coefficients <- truth$values()
    for (dilation in 1:2) parameters$get(paste0("rate", dilation), c(3, 3, 1, 1), nn$init$zeros())
    losses <- function(x) {
        teacher <- dilated_portable_dilated_conv2d(x, truth, 2)
        lapply(1:2, function(dilation) {
            prediction <- dilated_portable_dilated_conv2d(x, parameters$at(paste0("rate", dilation)), dilation)
            nn$mean(nn$squared_difference(prediction, teacher))
        })
    }
    config <- nn$OptimizerConfig()
    config$kind <- nn$OptimizerKind$Adam
    optimizer <- nn$Optimizer(parameters, 0.03, config)
    rng <- nn$Mt19937(101)
    training <- random_tensor(c(32, 13, 13, 1), rng)
    test <- random_tensor(c(64, 13, 13, 1), rng)
    for (step in 1:400) {
        loss <- losses(training)
        optimizer$minimize(nn$add(loss[[1L]], loss[[2L]]))
    }
    held_out <- nn$no_grad(function() sapply(losses(test), function(l) l$item()))
    cat(sprintf("  equal nine-parameter students, held-out MSE: rate1=%s, rate2=%s\n", g(held_out[1L]), g(held_out[2L])))
    nn_require(held_out[1L] > f32(0.2), "rate-1 student lacks independent distant context")
    nn_require(held_out[2L] < f32(1e-8), "rate-2 student recovers teacher")
    learned <- parameters$at("rate2")$values()
    for (i in 1:9) nn_near(learned[i], coefficients[i], 1e-4, "learned dilated coefficient")
}

main <- function(arguments = commandArgs(TRUE)) {
    nn_options(arguments, list(full = FALSE))
    equivalence()
    gridding()
    learn_context()
    cat("All dilated-convolution checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
