# Native R NN example. Contractive autoencoder with the exact sigmoid-encoder Jacobian penalty.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

# CMD: just r-example contractive_autoencoder
# Exact sigmoid-encoder Jacobian-Frobenius penalty, plain and strong matched controls.
# Three epochs/default; differentiate through activation slopes AND encoder weights.
#
# Port of tflow's contractive_autoencoder. Preparation (in ~/projects/tflow_examples):
# just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
# (per-variant plain/contractive/strong histories, npz arrays and metrics.json, data.npz,
# index.html and digit<k>.png galleries) go to output/nn/models/contractive_autoencoder.
# Options: epochs=N data=PATH out=PATH init=PATH (start all three variants from saved
# parameters; by default nn's own initialiser is used, identically for the three variants).

nn_helper("checks")
nn_helper("autoencoder")

fmt <- function(x) sprintf("%g", x)
float32 <- function(x) readBin(writeBin(x, raw(), size = 4L), "double", size = 4L, n = length(x))

# ---- model (examples/nn/common/contractive_autoencoder.hpp) ----------------------------------

# sum_j (h_j (1 - h_j))^2 sum_i W_ij^2 per example, and its batch mean.
sigmoid_encoder_contraction <- function(code, encoder_weight) {
    slope <- nn$mul(code, nn$sub(1, code))
    squared_column_norm <- nn$sum(nn$square(encoder_weight), 0L)
    per <- nn$sum(nn$mul(nn$square(slope), squared_column_norm), 1L)
    list(per_example = per, mean = nn$mean(per, 0L))
}

# 784->32 sigmoid->784 sigmoid (seed 1753) with loss MSE + weight * exact Jacobian penalty.
contractive_model <- function(weight, training = TRUE) {
    model <- new.env()
    model$weight <- float32(weight)
    nn_require(is.finite(model$weight) && model$weight >= 0, "finite nonnegative Jacobian weight")
    model$parameters <- nn$ParameterStore(1753L)
    model$decoder <- function(code) nn$dense(model$parameters, "decoder", code, 784L, nn$Activation$Sigmoid)
    model$forward <- function(image) {
        out <- list()
        out$code <- nn$dense(model$parameters, "encoder", image, 32L, nn$Activation$Sigmoid)
        out$reconstruction <- model$decoder(out$code)
        out$mse <- autoencoder_reconstruction_mse(out$reconstruction, image)
        out$penalty <- sigmoid_encoder_contraction(out$code, model$parameters$at("encoder/W"))
        out$loss <- nn$add(out$mse, nn$mul(out$penalty$mean, model$weight))
        out
    }
    # list(code, reconstruction, loss, mse, penalty mean, penalty per example)
    model$evaluate <- function(images) nn$no_grad(function() {
        n <- model$forward(autoencoder_flatten(images))
        list(n$code, n$reconstruction, n$loss, n$mse, n$penalty$mean, n$penalty$per_example)
    })
    # [n, 32, 784] image Jacobian of the code, one reverse pass per code unit.
    model$jacobian <- function(images) {
        x <- autoencoder_flatten(images)$contiguous()$detach()$requires_grad_()
        code <- nn$dense(model$parameters, "encoder", x, 32L, nn$Activation$Sigmoid)
        keep <- nn$GradOptions()
        keep$retain_graph <- TRUE
        rows <- lapply(0L:31L, function(j) nn$grad(nn$sum(nn$slice_axis(code, 1L, j, j + 1L)), list(x), keep)[[1L]])
        nn$stack(nn$tensor_vector(rows), 1L)$contiguous()
    }
    model$decode <- function(codes) nn$no_grad(function() model$decoder(codes))
    nn$no_grad(function() model$forward(nn$zeros(c(1L, 784L))))
    model$optimizer <- if (training) autoencoder_make_adam(model$parameters) else NULL
    model
}

contractive_fixtures <- function() {
    xv <- float32(c(0.1, 0.2, 0.7, 0.8, 0.3, 0.4))
    wv <- float32(c(0.3, -0.4, 0.7, 0.2, -0.2, 0.5))
    bv <- float32(c(0.1, -0.2))
    x <- nn$from_vector(xv, c(2L, 3L))$requires_grad_()
    w <- nn$from_vector(wv, c(3L, 2L))$requires_grad_()
    b <- nn$from_vector(bv, 2L)$requires_grad_()
    h <- nn$sigmoid(nn$bias_add(nn$matmul(x, w), b))
    p <- sigmoid_encoder_contraction(h, w)
    keep <- nn$GradOptions()
    keep$retain_graph <- TRUE
    grad <- nn$grad(p$mean, list(w, b), keep)
    rows <- lapply(0L:1L, function(j) nn$grad(nn$sum(nn$slice_axis(h, 1L, j, j + 1L)), list(x), keep)[[1L]])
    jac <- nn$stack(nn$tensor_vector(rows), 1L)$values()  # [2, 2, 3]
    penalty <- 0
    gw <- matrix(0, 3L, 2L)
    gb <- c(0, 0)
    for (n in 0L:1L) {
        for (j in 0L:1L) {
            z <- bv[j + 1L]
            norm <- 0
            for (i in 0L:2L) {
                z <- z + xv[n * 3L + i + 1L] * wv[i * 2L + j + 1L]
                norm <- norm + wv[i * 2L + j + 1L]^2
            }
            activation <- 1 / (1 + exp(-z))
            slope <- activation * (1 - activation)
            dz <- slope * slope * (1 - 2 * activation) * norm
            penalty <- penalty + slope * slope * norm / 2
            gb[j + 1L] <- gb[j + 1L] + dz
            for (i in 0L:2L) {
                nn_near(jac[(n * 2L + j) * 3L + i + 1L], slope * wv[i * 2L + j + 1L], 3e-8,
                        "ALL12true encoder-Jacobian entries")
                gw[i + 1L, j + 1L] <- gw[i + 1L, j + 1L] + xv[n * 3L + i + 1L] * dz + wv[i * 2L + j + 1L] * slope * slope
            }
        }
    }
    nn_near(p$mean$item(), penalty, 3e-8, "SUMallJacobian coordinates,MEANbatch")
    gwv <- grad[[1L]]$values()
    gbv <- grad[[2L]]$values()
    for (j in 0L:1L) {
        nn_near(gbv[j + 1L], gb[j + 1L], 3e-8, "ALL2penalty bias derivatives")
        for (i in 0L:2L)
            nn_near(gwv[i * 2L + j + 1L], gw[i + 1L, j + 1L], 3e-8, "ALL6penalty weight derivatives include direct2W term")
    }
    cat("known encoder:ALL12Jacobian coordinates/Frobeniusnorm/ALL8penalty parameter derivatives passed\n")
}

# Trains one matched variant on MSE + weight * Jacobian penalty; returns training seconds.
contractive_train_variant <- function(name, weight, pools, epochs, root, init) {
    directory <- file.path(root, name)
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    model <- contractive_model(weight)
    if (nzchar(init)) model$parameters$load(init)
    nn_require(model$parameters$count() == 50992 && model$parameters$size() == 4L,
               "independent784->32->784untied inventory")
    nn$save_npz(file.path(directory, "initial_arrays.npz"), autoencoder_arrays(model$parameters))
    order <- nn$BatchOrder(1000L, 32L, 1759L)
    fit_source <- as.integer(pools[[1L]]$source$values())
    best_path <- file.path(directory, "best.npz")
    best <- Inf
    selected <- 0L
    step <- 0L
    history <- file(file.path(directory, "history.tsv"), "w")
    batches <- file(file.path(directory, "batches.tsv"), "w")
    on.exit({
        close(history)
        close(batches)
    }, add = TRUE)
    writeLines("epoch\tfit_preupdate_loss\tvalidation_loss\tvalidation_mse\tvalidation_jacobian_squared", history)
    writeLines("epoch\tstep\trow\tsource_id", batches)
    started <- proc.time()[["elapsed"]]
    for (epoch in seq_len(epochs)) {
        order$begin_epoch()
        total <- 0
        seen <- 0L
        repeat {
            rows <- order$`_next`()
            if (!length(rows)) break
            images <- nn$take_rows(pools[[1L]]$images, rows)
            # Forward: MSE + weight * exact Jacobian penalty, differentiated through h AND W.
            loss <- model$forward(autoencoder_flatten(images))$loss
            if (step == 0L) {
                keep <- nn$GradOptions()
                keep$retain_graph <- TRUE
                gradients <- autoencoder_check_gradients(model$parameters,
                    nn$grad(loss, model$parameters$parameters(), keep),
                    "every full contractive gradient finite", "all4parameters receive complete gradients")
                nn$save_npz(file.path(directory, "first_gradients.npz"), gradients)
                nn$save_npz(file.path(directory, "first_batch.npz"), list(images = images))
            }
            value <- loss$item()
            nn_require(is.finite(value) && value >= 0, "finite pre-update regularized loss")
            model$optimizer$minimize(loss)
            total <- total + length(rows) * value
            seen <- seen + length(rows)
            writeLines(sprintf("%d\t%d\t%d\t%d", epoch, step, rows, fit_source[rows + 1L]), batches)
            if (step == 0L) nn$save_npz(file.path(directory, "first_update_arrays.npz"), autoencoder_arrays(model$parameters))
            step <- step + 1L
        }
        nn_require(seen == 1000L, "every fit image once per variant epoch")
        result <- model$evaluate(pools[[2L]]$images)
        value <- result[[3L]]$item()
        writeLines(sprintf("%d\t%.17g\t%.17g\t%.17g\t%.17g", epoch, total / seen, result[[3L]]$item(),
                           result[[4L]]$item(), result[[5L]]$item()), history)
        nn$save_npz(file.path(directory, paste0("epoch", epoch, "_arrays.npz")), autoencoder_arrays(model$parameters))
        if (value < best) {
            best <- value
            selected <- epoch
            model$parameters$save(best_path)  # own full contractive validation objective selects checkpoint
        }
        cat(sprintf("%s epoch%d:validation total/MSE/JacobianSquared=%s/%s/%s\n", name, epoch, fmt(value),
                    fmt(result[[4L]]$item()), fmt(result[[5L]]$item())))
    }
    seconds <- proc.time()[["elapsed"]] - started

    deployed <- contractive_model(weight, FALSE)
    deployed$parameters$load(best_path)  # fresh restore all contractive tensors
    before <- autoencoder_snapshot(deployed$parameters)
    arrays <- list()
    labels <- c("fit", "validation", "test")
    for (split in 1:3) {
        result <- deployed$evaluate(pools[[split]]$images)
        arrays[[paste0(labels[split], "_code")]] <- result[[1L]]
        arrays[[paste0(labels[split], "_reconstruction")]] <- result[[2L]]
        arrays[[paste0(labels[split], "_jacobian_squared")]] <- result[[6L]]
        arrays[[paste0(labels[split], "_components")]] <-
            nn$stack(nn$tensor_vector(list(result[[3L]], result[[4L]], result[[5L]])), 0L)$contiguous()
        if (split == 2L) nn_near(result[[3L]]$item(), best, 2e-7, "fresh selected-loss restore")
    }
    probe <- autoencoder_rows(pools[[3L]], 0L:7L)
    actual_j <- deployed$jacobian(probe$images)
    evaluation <- deployed$evaluate(probe$images)
    jv <- actual_j$values()
    per <- evaluation[[6L]]$values()
    norm_error <- 0
    for (n in 0L:7L) {
        total <- autoencoder_sum(jv[n * 32L * 784L + seq_len(32L * 784L)]^2)
        norm_error <- max(norm_error, abs(total - per[n + 1L]))
    }
    nn_require(norm_error < 2e-5, "analytic contraction equals independent ALLautodiff Jacobian Frobenius sum")
    arrays$probe_images <- probe$images
    arrays$probe_jacobian <- actual_j
    repeated <- deployed$evaluate(pools[[3L]]$images)
    nn_require(identical(autoencoder_bytes(repeated[[2L]]), autoencoder_bytes(arrays$test_reconstruction)),
               "deterministic contractive reconstruction")
    nn_require(identical(autoencoder_bytes(deployed$decode(repeated[[1L]])), autoencoder_bytes(repeated[[2L]])),
               "explicit contractive encode-decode exact")
    nn_require(identical(before, autoencoder_snapshot(deployed$parameters)), "all4contractive parameters immutable")
    nn$save_npz(file.path(directory, "outputs.npz"), arrays)
    nn$save_npz(file.path(directory, "model_arrays.npz"), autoencoder_arrays(deployed$parameters))
    writeLines(sprintf(paste0('{"weight":%.17g,"epochs":%d,"updates":%d,"selected_epoch":%d,"test_mse":%.17g,',
                              '"test_jacobian_squared":%.17g,"autodiff_norm_max_error":%.17g,"training_seconds":%.17g}'),
                       deployed$weight, epochs, step, selected, autoencoder_mse(repeated[[2L]], pools[[3L]]$images),
                       repeated[[5L]]$item(), norm_error, seconds), file.path(directory, "metrics.json"))
    cat(sprintf("%s:fullautodiff Jacobian-norm maxerror=%s,freshrestore/decode/repeat/immutable4weights passed (training %s s)\n",
                name, fmt(norm_error), fmt(seconds)))
    seconds
}

main <- function(arguments = commandArgs(TRUE)) {
    args <- nn_options(arguments, list(epochs = 3L, data = file.path(nn_data_root(), "saliency"), out = "", init = "",
                                       full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs <- args$epochs
    root <- if (nzchar(args$out)) args$out else file.path(nn_output_root(args$full), "models/contractive_autoencoder")
    nn_require(epochs > 0L && epochs <= 100L, "three default epochs or explicit1..100")
    contractive_fixtures()

    pools <- autoencoder_load_saliency(args$data)
    dir.create(root, recursive = TRUE, showWarnings = FALSE)
    variants <- list(plain = 0, contractive = 0.01, strong = 0.1)
    seconds <- 0
    for (name in names(variants))
        seconds <- seconds + contractive_train_variant(name, variants[[name]], pools, epochs, root, args$init)
    base <- nn$load_npz(file.path(root, "plain", "initial_arrays.npz"))
    for (name in names(variants)) {
        initial <- nn$load_npz(file.path(root, name, "initial_arrays.npz"))
        for (key in names(base))
            nn_require(identical(autoencoder_bytes(autoencoder_find(initial, key)), autoencoder_bytes(base[[key]])),
                       "all3initializations byteidentical")
    }
    autoencoder_save_data(file.path(root, "data.npz"), pools)
    autoencoder_copy_manifests(args$data, root)
    mean <- autoencoder_constant(autoencoder_pixel_mean(pools[[1L]]$images), 200L)
    outputs <- lapply(names(variants), function(name) nn$load_npz(file.path(root, name, "outputs.npz")))
    html <- paste0(
        "<!doctype html><meta charset='utf-8'><title>Contractive autoencoder</title><style>body{font:16px ",
        "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:",
        "25px 0}</style><h1>Contractive autoencoder:exact encoder Jacobian</h1><p>Original,plain,lambda.01,",
        "lambda.1,fit-only mean. Fixed[0,1]grayscale/nearest4,no quality filtering. 784->32sigmoid->784sigmoid/",
        "50992parameters,", epochs,
        "epochs each,matched initialization/order. Clean inputs reconstruct themselves;no corruption or class ",
        "targets.</p><p>Loss=MSE+lambda*MEANexamples SUMunits,inputs (dh_j/dx_i)^2. For sigmoid h:Jacobian ",
        "diag(h*(1-h))*W^T. Analytic exact norm differentiates through BOTH h and W,including direct weight ",
        "dependence;it is not weight decay or a noisy-input proxy. Coordinates are original[0,1]pixels. ",
        "Independent reverse-mode Jacobian rows validate the actual norm.</p><p>Each variant selects its OWN full ",
        "validation objective;objective numbers across different lambdas are not a common-MSE ranking. A small ",
        "encoder Jacobian can result from saturation or collapsed representations,not meaningful invariance. ",
        "Reconstruction detail,latent variance and saturation accompany the metric. Local contraction does not ",
        "prove robustness for finite changes or preservation of digit identity.</p><table><tr><th>Variant</th>",
        "<th>Test MSE</th><th>Mean Jacobian squared norm</th><th>Fraction saturated outside[.01,.99]</th><th>Mean ",
        "code coordinate std</th></tr>")
    for (k in seq_along(variants)) {
        codes <- outputs[[k]]$test_code$values()
        components <- outputs[[k]]$test_components$values()
        saturated <- sum(codes < 0.01 | codes > 0.99)
        code_matrix <- matrix(codes, ncol = 32L, byrow = TRUE)
        standard_deviation <- 0
        for (j in seq_len(32L)) {
            mean_j <- autoencoder_sum(code_matrix[, j] / 200)
            total <- autoencoder_sum((code_matrix[, j] - mean_j)^2 / 200)
            standard_deviation <- standard_deviation + sqrt(total) / 32
        }
        fraction <- saturated / length(codes)
        html <- c(html, sprintf("<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>", names(variants)[k],
                                fmt(components[2L]), fmt(components[3L]), fmt(fraction), fmt(standard_deviation)))
        cat(sprintf("%s: test MSE=%s, Jacobian squared=%s, saturated=%s, code std=%s\n", names(variants)[k],
                    fmt(components[2L]), fmt(components[3L]), fmt(fraction), fmt(standard_deviation)))
    }
    html <- c(html, paste0("</table><p>Short sigmoid runs may produce blurry average-looking outputs despite very small ",
                           "derivatives. Full-code variance/saturation and numerical evidence are saved. These are ",
                           "continuous32float coordinates,not a bit-rate compression or generative claim.</p>"))
    html <- c(html, autoencoder_digit_gallery(root, pools[[3L]],
        list(pools[[3L]]$images, outputs[[1L]]$test_reconstruction, outputs[[2L]]$test_reconstruction,
             outputs[[3L]]$test_reconstruction, mean)))
    writeLines(paste(html, collapse = ""), file.path(root, "index.html"))
    cat(sprintf(paste0("Three matched variants/%d epochs,actual squared encoder Jacobian penalty;no semantic/",
                       "finite-robustness superiority claim.\ntraining time %s s\nArtifacts:%s/index.html\n"),
                epochs, fmt(seconds), root))
    0L
}

if (sys.nframe() == 0L) quit(status = main())
