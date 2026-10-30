# Native R NN example. Receptive-field geometry and trained CNN feature derivatives in R.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.1_fundamentals/receptive_fields.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
nn_helper("checks")
nn_helper("cnn_baselines")
nn_helper("mnist")
nn_helper("receptive_field")

# Receptive fields and trained CNN feature maps, with numerical validation.
#
# Port of tflow's receptive_fields. Preparation: just r-example cnn_mnist
# (writes <output root>/models/cnn_mnist/best.npz), then run this example.
# MNIST comes from tflow's dataset cache (data=PATH). Restore the exact shared
# MNIST CNN, verify its test accuracy, and export intermediate activations for
# the first test example of every digit. Track receptive-field width, jump and
# center through conv/pool layers: width 1 -> 3 -> 4 -> 8 -> 10; final jump 4
# and first center 2. Check the full predicted support with positive
# convolutions/average pools, including boundary clipping. The trained model's
# input gradient for one pooled feature lies inside that box; ReLU and max
# pooling may reduce it. A finite difference checks an active derivative, and
# an outside pixel perturbation leaves the selected feature unchanged.
# Data, layer geometry: output/nn/visualizations/receptive_fields/.
# Options: data=PATH checkpoint=PATH threads=N.

g6 <- function(x) sprintf("%.6g", x)

# Fused multiply-add round(a * b + c) from exact double-double pieces
# (Veltkamp/Dekker two-product and two-sum). The C++ reference's compiler
# contracts its multiply-adds into hardware FMAs; R has no fma().
fma <- function(a, b, c) {
    split <- function(x) {
        t <- 134217729 * x
        hi <- t - (t - x)
        list(hi = hi, lo = x - hi)
    }
    p <- a * b
    sa <- split(a)
    sb <- split(b)
    e <- ((sa$hi * sb$hi - p) + sa$hi * sb$lo + sa$lo * sb$hi) + sa$lo * sb$lo
    s <- p + c
    bb <- s - p
    t <- (p - (s - bb)) + (c - bb)
    s + (t + e)
}

geometry <- function() {
    path <- nn_output_dir("visualizations/receptive_fields")
    lines <- c("stage\twidth\tjump\tfirst_center", "input\t1\t1\t0.5")
    field <- receptive_field_new()
    names <- c("conv1", "pool1", "conv2", "pool2")
    for (index in seq_along(names)) {
        pool <- index %% 2 == 0
        field <- receptive_field_after(field, if (pool) 2 else 3, if (pool) 2 else 1, if (pool) 0 else 1)
        lines <- c(lines, sprintf("%s\t%d\t%d\t%s", names[index], as.integer(field$width), as.integer(field$jump),
                                  g6(field$start)))
    }
    writeLines(lines, file.path(path, "geometry.tsv"))
    nn_require(field$width == 10 && field$jump == 4 && field$start == 2, "final receptive-field geometry")
    # The width of a dilated layer is an enclosing box, not a count of samples.
    nn_require(receptive_field_after(receptive_field_new(), 3, 1, 2, 2)$width == 5, "dilation expands effective width")
    field
}

support_checks <- function(field) {
    kernel <- nn$ones(c(3L, 3L, 1L, 1L))
    for (test in 0:1) {
        coordinate <- if (test == 0) 0 else 3
        x <- nn$zeros(c(1L, 28L, 28L, 1L))
        x$requires_grad_()
        first <- nn$avg_pool2d(nn$conv2d(x, kernel, nn$conv_options(1L, nn$Padding$Same)), 2L, 2L)
        second <- nn$avg_pool2d(nn$conv2d(first, kernel, nn$conv_options(1L, nn$Padding$Same)), 2L, 2L)
        feature <- nn$sum(nn$slice(second, c(0, coordinate, coordinate, 0), c(1, 1, 1, 1)))
        derivative <- matrix(nn$grad(feature, list(x))[[1L]]$values(), 28L, 28L, byrow = TRUE)
        bounds <- receptive_field_bounds(field, coordinate)
        inside_axis <- 0:27 >= bounds[1L] & 0:27 < bounds[2L]
        inside <- outer(inside_axis, inside_axis, "&")
        nn_require(identical(derivative > 0, inside), "analytic and numerical support agree")
        cat(sprintf("  positive linear network: %s support=%d pixels\n", if (test == 0) "corner" else "center",
                    sum(inside)))
    }
}

logits_of <- function(store, x, training) cnn_baselines_mnist_cnn(store, x)$logits

main <- function(arguments = commandArgs(TRUE)) {
    full <- nn_full()
    args <- nn_options(arguments, list(data = file.path(nn_data_root(), "mnist"),
                                       checkpoint = file.path(nn_output_root(full), "models/cnn_mnist/best.npz"),
                                       threads = 0, full = full))
    Sys.setenv(DATAMUNGE_EXAMPLES_FULL = if (args$full) "1" else "0")
    if (args$threads > 0) nn$set_num_threads(as.integer(args$threads))
    field <- geometry()
    support_checks(field)
    nn_require(file.exists(args$checkpoint), "run cnn_mnist first to create trained checkpoint")
    dataset <- nn_components$dataset_assets$LoadMnist(args$data, "test")
    count <- as.numeric(dataset$images$shape())[1L]
    pixels <- nn$reshape(dataset$images, c(count, 784))

    store <- nn$ParameterStore(17L)
    nn$no_grad(function() logits_of(store, nn$slice_axis(pixels, 0L, 0L, 1L), FALSE))
    store$load(args$checkpoint)
    evaluation <- mnist_evaluate(store, logits_of, pixels, dataset$labels, seq_len(count) - 1)
    nn_training_quality(evaluation$accuracy$value() >= .98, "restored shared CNN preserves test performance")
    cat(sprintf("  restored checkpoint test accuracy=%s\n", g6(evaluation$accuracy$value())))

    labels <- dataset$labels$values()
    rows <- match(0:9, labels) - 1
    nn_require(!anyNA(rows), "one example of each digit")
    selected_channel <- 0
    arrays <- nn$no_grad(function() {
        x <- nn$take_rows(pixels, rows)
        network <- cnn_baselines_mnist_cnn(store, x)
        for (c in 1:31)
            if (network$pool2$at(c(0, 3, 3, c)) > network$pool2$at(c(0, 3, 3, selected_channel))) selected_channel <<- c
        nn_require(network$pool2$at(c(0, 3, 3, selected_channel)) > 0, "active selected feature")
        list(conv1 = network$conv1, conv2 = network$conv2, images = network$image,
             labels = nn$take_rows(dataset$labels, rows), pool2 = network$pool2,
             predictions = nn$argmax(network$logits, 1L))
    })

    # Differentiate in Float64 so shrinking the finite-difference interval does
    # not amplify Float32 cancellation. A tiny deterministic texture breaks
    # blank-background max-pool ties in a lightly trained quick model.
    derivative_store <- nn$ParameterStore(17L, nn$DType$Float64)
    for (name in store$names()) derivative_store$set(name, store$at(name)$to(nn$DType$Float64))
    probe <- nn$take_rows(pixels, rows[1L])$values()
    if (!nn_full())
        probe <- fma(.0001, sin(fma(.37, seq_along(probe) - 1, .1)), probe)
    single <- nn$from_vector_double(probe, c(1L, 784L))
    arrays$gradient_probe <- single

    evaluate <- function(input, derivative) {
        x <- input$clone()
        if (derivative) x$requires_grad_()
        network <- cnn_baselines_mnist_cnn(derivative_store, x)
        selected <- nn$sum(nn$slice(network$pool2, c(0, 3, 3, selected_channel), c(1, 1, 1, 1)))
        gradient <- if (derivative) nn$grad(selected, list(x))[[1L]] else NULL
        list(value = selected$item(), gradient = gradient)
    }
    reference <- evaluate(single, TRUE)
    arrays$gradient <- reference$gradient
    dx <- reference$gradient$values()
    # The C++ stores each derivative as a float before testing it.
    derivative32 <- nn$from_vector(dx, length(dx))$values()
    bounds <- receptive_field_bounds(field, 3)
    inside_axis <- 0:27 >= bounds[1L] & 0:27 < bounds[2L]
    support <- as.numeric(t(outer(inside_axis, inside_axis, "&")))
    for (i in which(support == 0))
        nn_near(derivative32[i], 0, 0, "gradient stays inside theoretical receptive field")
    active <- sum(derivative32 != 0)
    strongest <- 1L
    for (i in seq_along(dx)) if (abs(derivative32[i]) > abs(dx[strongest])) strongest <- i
    nn_require(active > 0, "nonzero feature gradient")
    nn$no_grad(function() {
        base <- single$values()
        changed <- base
        changed[1L] <- changed[1L] + .5
        nn_near(evaluate(nn$from_vector_double(changed, c(1L, 784L)), FALSE)$value, reference$value, 0,
                "outside perturbation cannot affect feature")
        # A short-trained checkpoint can place a ReLU/max-pool boundary inside
        # the original .001 interval. Shrink the interval until both one-sided
        # slopes agree; retain the original derivative tolerance.
        difference <- 0
        stable <- FALSE
        for (h in c(.001, .0002, .0001, .00005, .00002, .000001, .00000001)) {
            plus <- base
            minus <- base
            plus[strongest] <- plus[strongest] + h
            minus[strongest] <- minus[strongest] - h
            forward <- (evaluate(nn$from_vector_double(plus, c(1L, 784L)), FALSE)$value - reference$value) / h
            backward <- (reference$value - evaluate(nn$from_vector_double(minus, c(1L, 784L)), FALSE)$value) / h
            if (abs(forward - backward) <= .001) {
                difference <- (forward + backward) / 2
                stable <- TRUE
                break
            }
        }
        nn_require(stable, "finite-difference interval stays within one active branch")
        nn_near(difference, dx[strongest], .001, "active derivative finite difference")
        arrays$support <<- nn$from_vector(support, c(28L, 28L))
        arrays$unit <<- nn$cast(nn$from_vector_int64(c(3, 3, selected_channel), 3L), nn$DType$Int32)
        out <- file.path(normalizePath(nn_output_dir("visualizations/receptive_fields")), "features.npz")
        nn$save_npz(out, arrays)
        cat(sprintf("  pool2[3,3,%d]: theoretical box=[%d,%d) squared; nonzero input derivatives=%d/100\n",
                    as.integer(selected_channel), as.integer(bounds[1L]), as.integer(bounds[2L]), as.integer(active)))
        cat(sprintf("  finite difference=%s; analytic=%s\n", g6(difference), g6(dx[strongest])))
        cat("  export: ", out, "\nAll receptive-field checks passed.\n", sep = "")
    })
    0L
}

if (sys.nframe() == 0L) quit(status = main())
