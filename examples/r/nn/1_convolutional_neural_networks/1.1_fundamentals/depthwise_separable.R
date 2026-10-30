# Native R NN example. Depthwise separable convolution: equivalence, gradients, rank and cost.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.1_fundamentals/depthwise_separable.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
# Compose HWIM depthwise and 1x1 pointwise filters into a dense HWIO filter,
# with no intervening activation. Check outputs, input gradients and both
# factor gradients against dense convolution for multipliers 1/2 and strides
# 1/2, and selected factor derivatives by central finite differences. A depth
# multiplier m limits each input channel's spatial-by-output kernel matrix to
# rank m; SVD quantifies the lost energy. Count parameters and MACs, then time
# matching dense and separable forwards (two threads). Inputs come from the
# same std::mt19937 streams as the C++ (bit-identical).
# C++ reference: examples/cpp/source/nn/1_convolutional_neural_networks/1.1_fundamentals/depthwise_separable.cpp

nn_helper("checks")
nn_helper("separable")

# Rounds doubles to float32 (vectorized), as the C++ float arithmetic does.
f32 <- function(values) nn$from_vector(values, length(values))$values()

g <- function(value) sprintf("%.6g", value)

# A fresh std::normal_distribution<float>(0, .2) over the shared std::mt19937.
random_tensor <- function(shape, rng) {
    rng$reset_normal()
    nn$from_vector(rng$normal32_values(prod(shape), 0, 0.2), shape)
}

difference <- function(a, b) {
    nn_require(identical(as.numeric(a$shape()), as.numeric(b$shape())), "matching tensor shapes")
    max(f32(abs(a$contiguous()$values() - b$contiguous()$values())))
}

equivalence <- function() {
    rng <- nn$Mt19937(41)
    for (multiplier in c(1, 2)) for (stride in c(1, 2)) {
        pixels <- random_tensor(c(2, 9, 10, 3), rng)
        spatial <- random_tensor(c(3, 3, 3, multiplier), rng)
        mixing <- random_tensor(c(1, 1, 3 * multiplier, 5), rng)
        forward_and_gradients <- function(a, b) {
            x <- pixels$clone()
            dw <- a$clone()
            pw <- b$clone()
            x$requires_grad_()
            dw$requires_grad_()
            pw$requires_grad_()
            r <- list()
            depthwise <- nn$depthwise_conv2d(x, dw, nn$conv_options(stride, nn$Padding$Same))
            r$separable <- nn$conv2d(depthwise, pw, nn$conv_options(1L, nn$Padding$Valid))
            r$kernel <- separable_compose_separable_kernel(dw, pw, 3, 3, multiplier, 5)
            r$dense <- nn$conv2d(x, r$kernel, nn$conv_options(stride, nn$Padding$Same))
            r$sep_loss <- nn$mean(nn$square(r$separable))
            full_loss <- nn$mean(nn$square(r$dense))
            r$sep_grad <- nn$grad(r$sep_loss, list(x, dw, pw))
            r$full_grad <- nn$grad(full_loss, list(x, dw, pw))
            r
        }
        result <- forward_and_gradients(spatial, mixing)
        nn_near(difference(result$separable, result$dense), 0, 2e-6, "separable forward equals composed dense kernel")
        gradient_error <- max(sapply(1:3, function(i) difference(result$sep_grad[[i]], result$full_grad[[i]])))
        nn_near(gradient_error, 0, 2e-6, "equivalent input and factor gradients")
        s <- spatial$values()
        m <- mixing$values()
        k <- result$kernel$values()
        for (y in 0:2) for (x in 0:2) for (i in 0:2) for (o in 0:4) {
            expected <- 0
            for (j in seq_len(multiplier) - 1)
                expected <- expected + s[((y * 3 + x) * 3 + i) * multiplier + j + 1] * m[(i * multiplier + j) * 5 + o + 1]
            nn_near(k[((y * 3 + x) * 3 + i) * 5 + o + 1], expected, 1e-7, "host filter contraction")
        }
        for (factor in 0:1) {
            value <- if (factor == 0) spatial else mixing
            base <- value$values()
            for (i in c(0, value$numel() %/% 2, value$numel() - 1)) {
                plus <- base
                minus <- base
                plus[i + 1] <- f32(plus[i + 1] + f32(0.001))
                minus[i + 1] <- f32(minus[i + 1] - f32(0.001))
                p <- nn$from_vector(plus, value$shape())
                q <- nn$from_vector(minus, value$shape())
                upper <- if (factor == 0) forward_and_gradients(p, mixing) else forward_and_gradients(spatial, p)
                lower <- if (factor == 0) forward_and_gradients(q, mixing) else forward_and_gradients(spatial, q)
                nn_near(f32(upper$sep_loss$item() - lower$sep_loss$item()) / 0.002,
                        result$sep_grad[[2 + factor]]$values()[i + 1], 2e-6, "factor finite difference")
            }
        }
        cat(sprintf("  multiplier=%d, stride=%d: forward error=%s; gradient error=%s\n", multiplier, stride,
                    g(difference(result$separable, result$dense)), g(gradient_error)))
    }
}

rank_restriction <- function() {
    rng <- nn$Mt19937(109)
    arbitrary <- random_tensor(c(3, 9, 5), rng)  # input channel, spatial tap, output channel
    svd <- nn$svd(arbitrary)
    a <- arbitrary$values()
    s <- svd$s$contiguous()$values()
    u <- svd$u$contiguous()$values()
    v <- svd$v$contiguous()$values()
    total <- sum(a^2)
    previous <- Inf
    for (multiplier in c(1, 2, 5)) {
        error <- 0
        tail_energy <- 0
        for (channel in 0:2) {
            if (multiplier < 5)
                for (j in multiplier:4) tail_energy <- tail_energy + s[channel * 5 + j + 1]^2
            for (tap in 0:8) for (out in 0:4) {
                ms <- seq_len(multiplier) - 1
                # float * float * float per term, accumulated in double (as in the C++).
                terms <- f32(f32(u[(channel * 9 + tap) * 5 + ms + 1] * s[channel * 5 + ms + 1]) *
                             v[(channel * 5 + out) * 5 + ms + 1])
                approximation <- 0
                for (term in terms) approximation <- approximation + term
                error <- error + (approximation - a[(channel * 9 + tap) * 5 + out + 1])^2
            }
        }
        nn_near(error, tail_energy, 2e-6, "best rank-m error equals discarded singular-value energy")
        nn_require(error < previous, "extra depth multiplier increases representational capacity")
        previous <- error
        if (multiplier == 1) nn_require(error / total > 0.3, "arbitrary dense filters are not generally rank one")
        if (multiplier == 5) nn_require(error / total < 1e-10, "full-rank factors reconstruct any kernel")
        cat(sprintf("  multiplier=%d: relative squared kernel error=%s\n", multiplier, g(error / total)))
    }
}

cost_and_timing <- function() {
    height <- 32; width <- 32; inputs <- 32; outputs <- 64; kernel <- 3
    dense_parameters <- kernel * kernel * inputs * outputs
    separable_parameters <- kernel * kernel * inputs + inputs * outputs
    rng <- nn$Mt19937(204)
    pixels <- random_tensor(c(1, height, width, inputs), rng)
    dw <- random_tensor(c(kernel, kernel, inputs, 1), rng)
    pw <- random_tensor(c(1, 1, inputs, outputs), rng)
    nn$no_grad(function() {
        composed <- separable_compose_separable_kernel(dw, pw, kernel, inputs, 1, outputs)
        full <- function() nn$conv2d(pixels, composed, nn$conv_options(1L, nn$Padding$Same))
        sep <- function() nn$conv2d(nn$depthwise_conv2d(pixels, dw, nn$conv_options(1L, nn$Padding$Same)), pw,
                                    nn$conv_options(1L, nn$Padding$Valid))
        nn_near(difference(full(), sep()), 0, 1e-5, "benchmark functions match")
        threads <- nn$get_num_threads()
        nn$set_num_threads(2L)
        for (variant in 0:1) {
            forward <- if (variant == 1) sep else full
            sink <- 0
            for (warmup in 1:10) sink <- sink + nn$mean(forward())$item()
            start <- as.numeric(Sys.time())
            for (repeat_index in 1:100) sink <- sink + nn$mean(forward())$item()
            milliseconds <- (as.numeric(Sys.time()) - start) * 1000 / 100
            nn_require(is.finite(sink), "timed forward is finite")
            parameters <- if (variant == 1) separable_parameters else dense_parameters
            cat(sprintf("  %s: kernel parameters=%d; MACs=%d; CPU forward mean=%s ms (100 runs, two threads)\n",
                        if (variant == 1) "separable" else "dense", parameters, height * width * parameters,
                        g(milliseconds)))
        }
        nn$set_num_threads(threads)
    })
    cat(sprintf("  theoretical kernel/MAC reduction=%sx; measured timings are eager nn calls\n",
                g(dense_parameters / separable_parameters)))
}

main <- function(arguments = commandArgs(TRUE)) {
    nn_options(arguments, list(full = FALSE))
    equivalence()
    rank_restriction()
    cost_and_timing()
    cat("All depthwise-separable checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
