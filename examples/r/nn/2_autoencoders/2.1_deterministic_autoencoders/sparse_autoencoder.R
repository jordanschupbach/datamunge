# Native R NN example. Activation-L1 and Bernoulli-KL sparse autoencoders with matched controls.
# CMD: nix develop .#r --command Rscript examples/r/nn/2_autoencoders/2.1_deterministic_autoencoders/sparse_autoencoder.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

# BOTH activation-L1 and mean-unit Bernoulli-KL sparse autoencoders with matched controls.
# Four matched 64-sigmoid variants, three epochs/default, no label targets.
#
# Port of tflow's sparse_autoencoder (model and penalties: examples/nn/common/sparse_autoencoder.hpp).
# Preparation (in ~/projects/tflow_examples): just autoencoder-assets; data are read from
# $DATAMUNGE_DATA_DIR/saliency. Outputs (plain/, l1/, kl/, combined/ training records, data.npz,
# index.html and digit<k>.png galleries) go to output/nn/models/sparse_autoencoder.
# Options: epochs=N data=PATH out=PATH init=PATH (start all four variants from saved
# parameters; by default nn's own initialiser is used, identically for every variant).

nn_helper("checks")
nn_helper("autoencoder")

fmt <- function(x) sprintf("%g", x)
float32 <- function(x) readBin(writeBin(x, raw(), size = 4L), "double", size = 4L, n = length(x))

# ---- model and penalties (examples/nn/common/sparse_autoencoder.hpp) -------------------------

# l1 = mean |code|; kl = mean over units of KL(Bernoulli(rho) || Bernoulli(clamped batch mean));
# total = l1_weight l1 + kl_weight kl. All constants are float32 values as in tflow.
activation_sparsity <- function(code, l1_weight, kl_weight, rho = 0.05, epsilon = 1e-5) {
    l1_weight <- float32(l1_weight)
    kl_weight <- float32(kl_weight)
    rho <- float32(rho)
    epsilon <- float32(epsilon)
    nn_require(is.finite(l1_weight) && l1_weight >= 0 && is.finite(kl_weight) && kl_weight >= 0 && rho > 0 &&
               rho < 1 && epsilon > 0 && epsilon < 0.5, "valid sparsity weights/target/clamp")
    out <- list()
    out$l1 <- nn$mean(nn$abs(code))
    out$unit_mean <- nn$mean(code, 0L)
    out$clipped_mean <- nn$clip(out$unit_mean, epsilon, float32(1 - epsilon))
    complement <- float32(1 - rho)
    positive <- nn$mul(nn$log(nn$div(rho, out$clipped_mean)), rho)
    negative <- nn$mul(nn$log(nn$div(complement, nn$sub(1, out$clipped_mean))), complement)
    out$kl <- nn$mean(nn$add(positive, negative), 0L)
    out$total <- nn$add(nn$mul(out$l1, l1_weight), nn$mul(out$kl, kl_weight))
    out
}

# 784 -> 64 sigmoid -> 784 sigmoid; loss = MSE + sparsity penalties.
sparse_forward <- function(p, image, l1_weight, kl_weight) {
    out <- list()
    out$code <- nn$dense(p, "encoder", image, 64L, nn$Activation$Sigmoid)
    out$reconstruction <- nn$dense(p, "decoder", out$code, 784L, nn$Activation$Sigmoid)
    out$mse <- autoencoder_reconstruction_mse(out$reconstruction, image)
    out$penalty <- activation_sparsity(out$code, l1_weight, kl_weight)
    out$loss <- nn$add(out$mse, out$penalty$total)
    out
}

sparse_model <- function(l1, kl, training = TRUE) {
    model <- new.env()
    model$l1_weight <- float32(l1)
    model$kl_weight <- float32(kl)
    model$parameters <- nn$ParameterStore(1601L)
    nn$no_grad(function() sparse_forward(model$parameters, nn$zeros(c(1L, 784L)), model$l1_weight, model$kl_weight))
    model$optimizer <- if (training) autoencoder_make_adam(model$parameters) else NULL
    model$net <- function(images)
        sparse_forward(model$parameters, autoencoder_flatten(images), model$l1_weight, model$kl_weight)
    # list(code, reconstruction, loss, mse, l1, kl, unit_mean)
    model$evaluate <- function(images) nn$no_grad(function() {
        n <- model$net(images)
        list(n$code, n$reconstruction, n$loss, n$mse, n$penalty$l1, n$penalty$kl, n$penalty$unit_mean)
    })
    model$decode <- function(codes) nn$no_grad(function()
        nn$dense(model$parameters, "decoder", codes, 784L, nn$Activation$Sigmoid))
    model
}

sparse_fixtures <- function() {
    values <- float32(c(0, 0.05, 0.2, 1, 0, 0.05, 0.6, 1, 0, 0.05, 1, 1))
    h <- nn$from_vector(values, c(3L, 4L))$requires_grad_()
    p <- activation_sparsity(h, 0.1, 0.2)
    gradient <- nn$grad(p$total, list(h))[[1L]]$values()
    low <- float32(1e-5)
    high <- float32(1 - low)
    rho <- float32(0.05)
    complement <- float32(1 - rho)
    w1 <- float32(0.1)
    w2 <- float32(0.2)
    l1 <- autoencoder_sum(abs(values) / 12)
    kl <- 0
    for (j in 0L:3L) {
        q <- autoencoder_sum(values[0L:2L * 4L + j + 1L] / 3)
        clipped <- min(max(q, low), high)
        kl <- kl + (rho * log(rho / clipped) + complement * log(complement / (1 - clipped))) / 4
        for (b in 0L:2L) {
            expected <- if (values[b * 4L + j + 1L] == 0) 0 else w1 / 12
            if (q > low && q < high) expected <- expected + w2 * (-rho / q + complement / (1 - q)) / 12
            nn_near(gradient[b * 4L + j + 1L], expected, 3e-7, "all12L1+KLpartials including low/high clamp and targetrho")
        }
    }
    nn_near(p$l1$item(), l1, 4e-8, "activation L1 mean not parameter penalty")
    nn_near(p$kl$item(), kl, 4e-6, "clamped perunit Bernoulli KL mean including extreme means")
    nn_near(p$total$item(), w1 * l1 + w2 * kl, 1e-6, "weighted sum of distinct penalties")
    nn_throws(activation_sparsity(h, 0.1, 0.2, 1), "invalid targetrho rejected")
    cat("analytic L1/KL means/ALL12derivatives/zero/targetrho/clamped0+1means/invalidrho fixtures passed\n")
}

# Trains one matched sparse variant on its own regularized objective; returns training seconds.
sparse_train_variant <- function(name, l1, kl, pools, epochs, root, init) {
    directory <- file.path(root, name)
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    model <- sparse_model(l1, kl)
    if (nzchar(init)) model$parameters$load(init)
    nn_require(model$parameters$count() == 101200 && model$parameters$size() == 4L,
               "independent784x64+64+64x784+784 inventory")
    nn$save_npz(file.path(directory, "initial_arrays.npz"), autoencoder_arrays(model$parameters))
    order <- nn$BatchOrder(1000L, 32L, 1607L)
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
    writeLines("epoch\tfit_preupdate_loss\tvalidation_loss\tvalidation_mse\tvalidation_l1\tvalidation_kl", history)
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
            # Forward: reconstruction MSE + L1/KL activation penalties of this batch.
            loss <- model$net(images)$loss
            if (step == 0L) {
                keep <- nn$GradOptions()
                keep$retain_graph <- TRUE
                nn$save_npz(file.path(directory, "first_gradients.npz"),
                            autoencoder_check_gradients(model$parameters,
                                nn$grad(loss, model$parameters$parameters(), keep),
                                "finite all penalized parameter gradients", "all4tensors receive complete loss gradients"))
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
        nn_require(seen == 1000L, "each fitting image once per variant epoch")
        validation <- model$evaluate(pools[[2L]]$images)
        value <- validation[[3L]]$item()
        writeLines(sprintf("%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g", epoch, total / seen, validation[[3L]]$item(),
                           validation[[4L]]$item(), validation[[5L]]$item(), validation[[6L]]$item()), history)
        nn$save_npz(file.path(directory, paste0("epoch", epoch, "_arrays.npz")), autoencoder_arrays(model$parameters))
        if (value < best) {
            best <- value
            selected <- epoch
            model$parameters$save(best_path)  # validation own full regularized objective selects checkpoint
        }
        cat(sprintf("%s epoch%d:validation total/MSE/L1/KL=%s/%s/%s/%s\n", name, epoch, fmt(value),
                    fmt(validation[[4L]]$item()), fmt(validation[[5L]]$item()), fmt(validation[[6L]]$item())))
    }
    seconds <- proc.time()[["elapsed"]] - started

    deployed <- sparse_model(l1, kl, FALSE)
    deployed$parameters$load(best_path)  # fresh restore matched sparse variant
    before <- autoencoder_snapshot(deployed$parameters)
    arrays <- list()
    labels <- c("fit", "validation", "test")
    for (split in 1:3) {
        result <- deployed$evaluate(pools[[split]]$images)
        components <- vapply(3:6, function(i) result[[i]]$item(), numeric(1))
        arrays[[paste0(labels[split], "_code")]] <- result[[1L]]
        arrays[[paste0(labels[split], "_reconstruction")]] <- result[[2L]]
        arrays[[paste0(labels[split], "_unit_mean")]] <- result[[7L]]
        arrays[[paste0(labels[split], "_components")]] <- nn$from_vector(components, 4L)
        if (split == 2L) nn_near(result[[3L]]$item(), best, 1e-7, "selected regularized validation loss restored")
    }
    test <- deployed$evaluate(pools[[3L]]$images)
    nn_require(identical(autoencoder_bytes(test[[2L]]), autoencoder_bytes(arrays$test_reconstruction)),
               "deterministic sparse reconstruction")
    nn_require(identical(autoencoder_bytes(deployed$decode(test[[1L]])), autoencoder_bytes(test[[2L]])),
               "explicit sparse encode-decode exact")
    one <- deployed$evaluate(autoencoder_rows(pools[[3L]], 0L)$images)
    error <- autoencoder_max_abs_diff(one[[2L]], test[[2L]], 784L)
    nn_require(error < 1e-6, "inference single/batch parity despite batch-dependent KL objective")
    nn_require(identical(before, autoencoder_snapshot(deployed$parameters)), "all4sparse parameters immutable during evaluation")
    nn$save_npz(file.path(directory, "outputs.npz"), arrays)
    nn$save_npz(file.path(directory, "model_arrays.npz"), autoencoder_arrays(deployed$parameters))
    code <- arrays$test_code$contiguous()$values()
    mean <- autoencoder_sum(code) / length(code)
    fraction <- sum(code < float32(0.1)) / length(code)
    mse <- autoencoder_mse(test[[2L]], pools[[3L]]$images)
    writeLines(sprintf(paste0('{"l1_weight":%.17g,"kl_weight":%.17g,"rho":%.17g,"epsilon":%.17g,"epochs":%d,',
                              '"selected_epoch":%d,"updates":%d,"test_mse":%.17g,"test_mean_activation":%.17g,',
                              '"test_fraction_below_point1":%.17g,"single_batch_max_error":%.17g,"training_seconds":%.17g}'),
                       deployed$l1_weight, deployed$kl_weight, float32(0.05), float32(1e-5), epochs, selected, step,
                       mse, mean, fraction, error, seconds), file.path(directory, "metrics.json"))
    cat(sprintf("%s selected%d:testMSE=%s,meanactivation=%s,fractionbelow.1=%s;freshrestore/decode/repeat/batch/immutable4weights passed\n",
                name, selected, fmt(mse), fmt(mean), fmt(fraction)))
    seconds
}

main <- function(arguments = commandArgs(TRUE)) {
    args <- nn_options(arguments, list(epochs = 3L, data = file.path(nn_data_root(), "saliency"), out = "", init = "",
                                       full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs <- args$epochs
    directory <- if (nzchar(args$out)) args$out else file.path(nn_output_root(args$full), "models/sparse_autoencoder")
    nn_require(epochs > 0L && epochs <= 100L, "three default epochs or explicit1..100")
    sparse_fixtures()

    pools <- autoencoder_load_saliency(args$data)
    variants <- list(plain = c(0, 0), l1 = c(0.05, 0), kl = c(0, 0.05), combined = c(0.05, 0.05))
    seconds <- 0
    for (name in names(variants))
        seconds <- seconds + sparse_train_variant(name, variants[[name]][1L], variants[[name]][2L], pools, epochs,
                                                  directory, args$init)
    original <- nn$load_npz(file.path(directory, "plain", "initial_arrays.npz"))
    for (name in names(variants)) {
        initial <- nn$load_npz(file.path(directory, name, "initial_arrays.npz"))
        for (key in names(original))
            nn_require(identical(autoencoder_bytes(autoencoder_find(initial, key)), autoencoder_bytes(original[[key]])),
                       "ALL4variants identical initial parameters")
    }
    autoencoder_save_data(file.path(directory, "data.npz"), pools)
    autoencoder_copy_manifests(args$data, directory)
    mean <- autoencoder_constant(autoencoder_pixel_mean(pools[[1L]]$images), 200L)
    html <- paste0(
        "<!doctype html><meta charset='utf-8'><title>L1 and KL sparse autoencoders</title><style>body{font:16px ",
        "system-ui;max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}section{margin:",
        "25px 0}</style><h1>Sparse autoencoders:activation L1 and KL</h1><p>Original,plain,L1,KL,combined,fit-only ",
        "mean. ALL use784->64sigmoid->784sigmoid/101200parameters,identical initialization and batch order,", epochs,
        "epochs PER variant. Fixed[0,1]intensity/nearest4,no quality-filtering. Labels do not enter training or ",
        "selection.</p><p>Total=MSE+lambda*mean(abs(code))+beta*mean_unit KL(rho||mean_batch(code)). rho=.05,",
        "lambda=.05 for L1,beta=.05 for KL. Unit means clamp to[1e-5,1-1e-5]before logs;clamp derivative is flat ",
        "outside. KL is a penalty on each unit's BATCH mean,not each image's activation,not a softmax/class KL. ",
        "Minibatch KL and full-pool KL differ. Each variant selects on its OWN full validation objective,not ",
        "comparable objective numbers across penalties.</p><p>Sigmoid codes are generally positive:small activation ",
        "is not exact zero sparsity. Mean activation and fraction below.1 are descriptive,threshold not used in ",
        "optimization. Stronger sparsity can worsen reconstruction or collapse features;three short epochs do not ",
        "guarantee rho is reached. MSE alone can favor blurry averages. This is not a proof of semantic features,",
        "denoising,generative modeling or bit-rate compression.</p>")
    panels <- list(pools[[3L]]$images)
    for (name in names(variants)) {
        raw <- nn$load_npz(file.path(directory, name, "outputs.npz"))
        c <- raw$test_components$values()
        html <- c(html, paste0(sprintf("<p>%s:test total/MSE/L1/KL=%s/%s/%s/%s", name, fmt(c[1L]), fmt(c[2L]),
                                       fmt(c[3L]), fmt(c[4L])),
            ". Mean activation equals L1 for these nonnegative sigmoid codes. rho=.05 remains a target rather than an ",
            "attained guarantee;per-unit means/codes and threshold metrics are saved in variant artifacts.</p>"))
        panels[[length(panels) + 1L]] <- raw$test_reconstruction
    }
    panels[[length(panels) + 1L]] <- mean
    html <- c(html, autoencoder_digit_gallery(directory, pools[[3L]], panels))
    writeLines(paste(html, collapse = ""), file.path(directory, "index.html"))
    cat(sprintf(paste0("ALL4variants identicalinitial/sourceorder,96updates each at default3epochs. L1/KL tradeoffs ",
                       "saved without superiority claim.\ntraining time %s s\nArtifacts:%s/index.html\n"),
                fmt(seconds), directory))
    0L
}

if (sys.nframe() == 0L) quit(status = main())
