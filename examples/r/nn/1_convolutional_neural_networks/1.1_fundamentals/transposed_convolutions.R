# Native R NN example. Transposed convolution, its adjoint identity and checkerboard artifacts.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.1_fundamentals/transposed_convolutions.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
# nn$conv2d_transpose (TF Conv2DBackpropInput) is the adjoint of a specified
# forward convolution, not generally its inverse. Verify SAME/VALID outputs
# against an independent scatter-and-add loop, the inner-product adjoint
# identity, and derivatives with respect to coarse input and filter. Then feed
# a constant 8x8 image to stride-2 transposed convolutions (3x3 ones, 4x4 ones,
# weighted 4x4) and to nearest resize + 3x3 convolution, and measure interior
# phase means/CV. Inputs come from the same std::mt19937(18) stream as the C++.
# Arrays: output/nn/r/<mode>/visualizations/transposed_convolutions/checkerboards.npz.
# C++ reference: examples/cpp/source/nn/1_convolutional_neural_networks/1.1_fundamentals/transposed_convolutions.cpp

nn_helper("checks")
nn_helper("im2col")

f32 <- function(values) nn$from_vector(values, length(values))$values()

g <- function(value) sprintf("%.6g", value)

# A fresh std::normal_distribution<float>(0, .25) over the shared std::mt19937.
random_tensor <- function(shape, rng) {
    rng$reset_normal()
    nn$from_vector(rng$normal32_values(prod(shape), 0, 0.25), shape)
}

difference <- function(a, b) {
    nn_require(identical(as.numeric(a$shape()), as.numeric(b$shape())), "transposed tensor shapes agree")
    max(f32(abs(a$contiguous()$values() - b$contiguous()$values())))
}

shape_str <- function(shape) paste0("[", paste(as.integer(shape), collapse = ","), "]")

# Independent scatter-and-add reference: every (n, y, x, a, b, out, in)
# contribution in the C++ loop order, accumulated per output pixel in float32.
scatter_reference <- function(coarse, kernel, height, width, geometry) {
    grid <- expand.grid(inp = 0:1, out = 0:2, b = 0:2, a = 0:2, x = 0:4, y = 0:3, n = 0:1)
    py <- 2 * grid$y + grid$a - geometry$top
    px <- 2 * grid$x + grid$b - geometry$left
    keep <- py >= 0 & py < height & px >= 0 & px < width
    grid <- grid[keep, ]
    target <- ((grid$n * height + py[keep]) * width + px[keep]) * 3 + grid$out + 1
    product <- f32(coarse$values()[((grid$n * 4 + grid$y) * 5 + grid$x) * 2 + grid$inp + 1] *
                   kernel$values()[((grid$a * 3 + grid$b) * 3 + grid$out) * 2 + grid$inp + 1])
    # Round r adds the r-th contribution of every pixel, preserving per-pixel order.
    rank <- ave(seq_along(target), target, FUN = seq_along)
    reference <- numeric(height * width * 3 * 2)
    for (round in seq_len(max(rank))) {
        selected <- rank == round
        reference[target[selected]] <- f32(reference[target[selected]] + product[selected])
    }
    reference
}

adjoint_checks <- function() {
    rng <- nn$Mt19937(18)
    for (padding in c("SAME", "VALID")) {
        height <- if (padding == "SAME") 8 else 9
        width <- if (padding == "SAME") 10 else 11
        coarse <- random_tensor(c(2, 4, 5, 2), rng)
        kernel <- random_tensor(c(3, 3, 3, 2), rng)
        fine <- random_tensor(c(2, height, width, 3), rng)
        geometry <- im2col_patch_geometry(fine$shape(), 3, 3, 2, 2, 1, 1, padding)
        reference <- scatter_reference(coarse, kernel, height, width, geometry)
        conv <- nn$conv_options(2L, if (padding == "SAME") nn$Padding$Same else nn$Padding$Valid)
        forward_and_gradients <- function(filter, derivatives) {
            x <- coarse$clone()
            w <- filter$clone()
            x$requires_grad_()
            w$requires_grad_()
            r <- list()
            r$transposed <- nn$conv2d_transpose(x, w, fine$shape(), conv)
            r$forward <- nn$conv2d(fine, w, conv)
            r$objective <- nn$sum(nn$mul(r$transposed, fine))
            r$paired <- nn$sum(nn$mul(r$forward, x))
            if (derivatives) r$gradients <- nn$grad(r$objective, list(x, w))
            r
        }
        result <- forward_and_gradients(kernel, TRUE)
        scatter_error <- difference(result$transposed, nn$from_vector(reference, fine$shape()))
        nn_near(scatter_error, 0, 2e-7, "transposed output equals scatter-and-add reference")
        nn_near(result$objective$item(), result$paired$item(), 2e-6, "convolution inner-product adjoint identity")
        nn_near(difference(result$forward, result$gradients[[1L]]), 0, 1e-7,
                "transposed input gradient is forward convolution")
        dw <- result$gradients[[2L]]$values()
        k <- kernel$values()
        for (index in c(0, kernel$numel() %/% 2, kernel$numel() - 1)) {
            plus <- k
            minus <- k
            plus[index + 1] <- f32(plus[index + 1] + f32(0.002))
            minus[index + 1] <- f32(minus[index + 1] - f32(0.002))
            upper <- forward_and_gradients(nn$from_vector(plus, kernel$shape()), FALSE)$objective$item()
            lower <- forward_and_gradients(nn$from_vector(minus, kernel$shape()), FALSE)$objective$item()
            nn_near((upper - lower) / 0.004, dw[index + 1], 2e-4, "transposed filter finite difference")
        }
        adjoint_error <- f32(abs(result$objective$item() - result$paired$item()))
        cat(sprintf("  %s output=%s; scatter error=%s; adjoint error=%s\n", padding,
                    shape_str(result$transposed$shape()), g(scatter_error), g(adjoint_error)))
    }
}

checkerboards <- function() {
    coarse <- nn$ones(c(1, 8, 8, 1))
    result <- list()
    for (variant in 0:2) {
        k <- if (variant == 0) 3 else 4
        filter <- rep(1, k * k)
        if (variant == 2) filter[1] <- 4
        result[[variant + 1L]] <- nn$conv2d_transpose(coarse, nn$from_vector(filter, c(k, k, 1, 1)),
                                                      c(1, 14 + k, 14 + k, 1), nn$conv_options(2L, nn$Padding$Valid))
    }
    size <- nn$ImageSize()
    size$height <- 16L
    size$width <- 16L
    resized <- nn$resize_nearest(coarse, size)
    result[[4L]] <- nn$conv2d(resized, nn$ones(c(3, 3, 1, 1)), nn$conv_options(1L, nn$Padding$Same))
    labels <- c("transpose_k3", "transpose_k4", "transpose_k4_weighted", "resize_conv")
    arrays <- list()
    for (variant in 0:3) {
        value <- result[[variant + 1L]]
        side <- as.numeric(value$shape())[3L]
        values <- value$values()
        # The same 10x10 interior window in all variants (five samples per phase and axis).
        means <- numeric(4)
        counts <- numeric(4)
        for (y in 4:13) for (x in 4:13) {
            phase <- 2 * (y %% 2) + x %% 2 + 1
            means[phase] <- means[phase] + values[y * side + x + 1]
            counts[phase] <- counts[phase] + 1
        }
        means <- means / counts
        average <- 0
        for (mean in means) average <- average + mean / 4
        variance <- 0
        for (mean in means) variance <- variance + (mean - average) * (mean - average) / 4
        cv <- sqrt(variance) / average
        if (variant %in% c(0, 2)) nn_require(cv > 0.1, "unequal phases produce checkerboard response")
        else nn_near(cv, 0, 0, "equal interior phases for uniform even kernel or resize-convolution")
        if (variant == 0)
            for (i in 1:4) nn_near(means[i], c(4, 2, 2, 1)[i], 0, "3x3 stride-2 overlap multiplicity")
        arrays[[labels[variant + 1L]]] <- value
        cat(sprintf("  %s: interior phase means %s; phase CV=%s\n", labels[variant + 1L],
                    paste(g(means), collapse = " "), g(cv)))
    }
    directory <- nn_output_dir("visualizations/transposed_convolutions")
    nn$save_npz(file.path(directory, "checkerboards.npz"), arrays[sort(names(arrays), method = "radix")])
}

main <- function(arguments = commandArgs(TRUE)) {
    nn_options(arguments, list(full = FALSE))
    adjoint_checks()
    checkerboards()
    cat("All transposed-convolution checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
