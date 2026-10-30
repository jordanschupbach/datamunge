# Native R NN example. Visible-only transformer masked autoencoder (75 percent patch masking).
# CMD: nix develop .#r --command Rscript examples/r/nn/2_autoencoders/2.1_deterministic_autoencoders/masked_autoencoder.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

# Visible-only transformer MAE with 75 percent masking and a smaller mask-token decoder.
#
# Port of tflow's masked_autoencoder. Preparation (in ~/projects/tflow_examples):
# just autoencoder-assets; data are read from $DATAMUNGE_DATA_DIR/saliency. Outputs
# (history, per-epoch masks and arrays, probe.npz, outputs.npz, metrics.json, index.html
# and digit<k>.png galleries) go to output/nn/models/masked_autoencoder. The patch masks
# are the same hash-derived selections as tflow's (bit-identical). Checkpoints are
# best.npz files. Options: epochs=N data=PATH out=PATH init=PATH (start from saved
# parameters, e.g. tflow's initial_arrays.npz; by default nn's own initialiser is used).

nn_helper("checks")
nn_helper("autoencoder")
nn_helper("masked_autoencoder")

fmt <- function(x) sprintf("%g", x)

masked_loss_checks <- function() {
    mask <- rep(c(rep(0, 4L), rep(1, 12L)), 2L)
    patches <- nn$full(c(2L, 16L, 49L), 0.7)
    y <- rep(0.3, 2L * 16L * 49L)
    prediction <- nn$from_vector(y, c(2L, 16L, 49L))$requires_grad_()
    m <- nn$from_vector(mask, c(2L, 16L))
    loss <- masked_autoencoder_loss(prediction, patches, m)
    gradient <- nn$grad(loss, list(prediction))[[1L]]$values()
    nn_near(loss$item(), 0.16, 3e-7, "loss denominator counts ONLY12hidden patches per image")
    index <- seq_along(gradient) - 1L
    expected <- ifelse((index %/% 49L) %% 16L < 4L, 0, -0.8 / (2 * 12 * 49))
    nn_require(all(abs(gradient - expected) <= 2e-10), "all1568prediction derivatives;visible EXACTzero")
    before <- loss$item()
    for (b in 0:1) y[(b * 16L) * 49L + 1:(4L * 49L)] <- 0.99
    nn_near(masked_autoencoder_loss(nn$from_vector(y, c(2L, 16L, 49L)), patches, m)$item(), before, 0,
            "visible predictions excluded exactly")
    cat("masked loss/denominator/ALL1568derivatives/visible perturbation invariance passed\n")
}

main <- function(arguments = commandArgs(TRUE)) {
    args <- nn_options(arguments, list(epochs = 3L, data = file.path(nn_data_root(), "saliency"), out = "", init = "",
                                       full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    epochs <- args$epochs
    root <- if (nzchar(args$out)) args$out else file.path(nn_output_root(args$full), "models/masked_autoencoder")
    nn_require(epochs > 0L && epochs <= 100L, "three default epochs or explicit1..100")
    masked_loss_checks()

    pools <- autoencoder_load_saliency(args$data)
    dir.create(root, recursive = TRUE, showWarnings = FALSE)
    model <- masked_autoencoder_model()
    if (nzchar(args$init)) model$parameters$load(args$init)
    nn_require(model$parameters$size() == 35L && model$parameters$count() == 13649,
               "independent tiny transformer parameter census")
    nn$save_npz(file.path(root, "initial_arrays.npz"), autoencoder_arrays(model$parameters))
    test_images <- pools[[3L]]$images
    test_selection <- masked_autoencoder_selection(pools[[3L]]$source)
    initial <- model$evaluate(test_images, test_selection)
    nn$save_npz(file.path(root, "initial_outputs.npz"), list(code = initial[[1L]], prediction = initial[[2L]]))
    order <- nn$BatchOrder(1000L, 32L, 1889L)
    fit_source <- as.integer(pools[[1L]]$source$values())
    best_path <- file.path(root, "best.npz")
    best <- Inf
    selected <- 0L
    step <- 0L
    history <- file(file.path(root, "history.tsv"), "w")
    batches <- file(file.path(root, "batches.tsv"), "w")
    writeLines("epoch\tfit_preupdate_masked_mse\tvalidation_masked_mse", history)
    writeLines("epoch\tstep\trow\tsource_id", batches)
    started <- proc.time()[["elapsed"]]
    for (epoch in seq_len(epochs)) {
        order$begin_epoch()
        epoch_selection <- masked_autoencoder_selection(pools[[1L]]$source, epoch)
        nn$save_npz(file.path(root, paste0("epoch", epoch, "_mask.npz")), list(selection = epoch_selection))
        total <- 0
        seen <- 0L
        repeat {
            rows <- order$`_next`()
            if (!length(rows)) break
            input <- nn$take_rows(pools[[1L]]$images, rows)
            selection <- nn$take_rows(epoch_selection, rows)
            # Visible-only encode, mask-token decode, hidden-patch-only loss.
            net <- masked_autoencoder_forward(model$parameters, input, selection)
            loss <- masked_autoencoder_loss(net$prediction, net$patches, net$mask)
            if (step == 0L) {
                keep <- nn$GradOptions()
                keep$retain_graph <- TRUE
                gradients <- autoencoder_check_gradients(model$parameters,
                    nn$grad(loss, model$parameters$parameters(), keep),
                    "all transformer gradients finite", "ALL35tensors receive nonzero masked-loss gradients")
                nn$save_npz(file.path(root, "first_gradients.npz"), gradients)
                nn$save_npz(file.path(root, "first_batch.npz"), list(images = input, selection = selection))
            }
            value <- loss$item()
            nn_require(is.finite(value) && value >= 0, "finite masked MSE")
            model$optimizer$minimize(loss)
            total <- total + length(rows) * value
            seen <- seen + length(rows)
            writeLines(sprintf("%d\t%d\t%d\t%d", epoch, step, rows, fit_source[rows + 1L]), batches)
            if (step == 0L) nn$save_npz(file.path(root, "first_update_arrays.npz"), autoencoder_arrays(model$parameters))
            step <- step + 1L
        }
        nn_require(seen == 1000L, "every fitting record once per epoch")
        val_selection <- masked_autoencoder_selection(pools[[2L]]$source)
        value <- model$evaluate(pools[[2L]]$images, val_selection)
        mse <- masked_autoencoder_mse(value[[2L]], pools[[2L]]$images, val_selection)
        nn_near(value[[3L]]$item(), mse, 2e-7, "graph and independent native hidden-only MSE")
        writeLines(sprintf("%d\t%.17g\t%.17g", epoch, total / seen, mse), history)
        nn$save_npz(file.path(root, paste0("epoch", epoch, "_arrays.npz")), autoencoder_arrays(model$parameters))
        if (mse < best) {
            best <- mse
            selected <- epoch
            model$parameters$save(best_path)  # fixed masked validation MSE selects epoch
        }
        cat(sprintf("MAE epoch%d:validation masked MSE=%s\n", epoch, fmt(mse)))
    }
    close(history)
    close(batches)
    seconds <- proc.time()[["elapsed"]] - started

    deployed <- masked_autoencoder_model(FALSE)
    deployed$parameters$load(best_path)  # fresh inference MAE restore
    before <- autoencoder_snapshot(deployed$parameters)
    saved <- list()
    split_names <- c("fit", "validation", "test")
    for (split in 1:3) {
        name <- split_names[split]
        selection <- masked_autoencoder_selection(pools[[split]]$source)
        out <- deployed$evaluate(pools[[split]]$images, selection)
        saved[[paste0(name, "_code")]] <- out[[1L]]
        saved[[paste0(name, "_prediction")]] <- out[[2L]]
        saved[[paste0(name, "_selection")]] <- selection
        if (split == 2L)
            nn_near(masked_autoencoder_mse(out[[2L]], pools[[split]]$images, selection), best, 1e-8,
                    "fresh selected validation metric")
    }
    test <- deployed$evaluate(test_images, test_selection)
    repeated <- deployed$evaluate(test_images, test_selection)
    nn_require(identical(autoencoder_bytes(test[[2L]]), autoencoder_bytes(repeated[[2L]])) &&
               identical(autoencoder_bytes(deployed$decode(test[[1L]], test_selection)), autoencoder_bytes(test[[2L]])),
               "deterministic exact target-free code decoding")
    images <- test_images$values()
    sel <- test_selection$values()
    hidden_pixels <- masked_autoencoder_hidden_pixels(sel, 200L)
    altered <- ifelse(hidden_pixels, 1 - images, images)
    shifted <- images[((rep(0:199, each = 784L) + 1L) %% 200L) * 784L + rep(1:784, 200L)]
    # Visible token slots in reverse order: slot k takes slot 3 - k.
    permuted <- as.vector(array(sel, c(16L, 4L, 200L))[, 4:1, ])
    image_shape <- test_images$shape()
    hidden <- deployed$evaluate(nn$from_vector(altered, image_shape), test_selection)
    nn_require(identical(autoencoder_bytes(hidden[[1L]]), autoencoder_bytes(test[[1L]])) &&
               identical(autoencoder_bytes(hidden[[2L]]), autoencoder_bytes(test[[2L]])),
               "changing EVERY hidden input pixel leaves codes/predictions BITEXACT;no target leakage")
    shuffled <- deployed$evaluate(nn$from_vector(shifted, image_shape), test_selection)
    permutation <- deployed$evaluate(test_images, nn$from_vector(permuted, c(200L, 4L, 16L)))
    second_selection <- masked_autoencoder_selection(pools[[3L]]$source, 0L, 1913)
    second <- deployed$evaluate(test_images, second_selection)
    zero_prediction <- deployed$decode(nn$zeros(c(200L, 4L, 32L)), test_selection)
    saved$zero_prediction <- zero_prediction
    saved$shuffled_prediction <- shuffled[[2L]]
    saved$second_selection <- second_selection
    saved$second_prediction <- second[[2L]]
    saved$permuted_prediction <- permutation[[2L]]
    single <- deployed$evaluate(nn$take_rows(test_images, 0L), nn$take_rows(test_selection, 0L))
    single_error <- autoencoder_max_abs_diff(single[[2L]], test[[2L]], 784L)
    permutation_error <- autoencoder_max_abs_diff(permutation[[2L]], test[[2L]], 200L * 784L)
    nn_require(single_error < 2e-6 && permutation_error < 2e-6,
               "batch parity and visible token permutation/restoration equivariance")
    nn_require(identical(before, autoencoder_snapshot(deployed$parameters)),
               "inference and controls leave ALL35parameter tensors immutable")
    first8 <- 0:7
    probe <- nn$take_rows(test_images, first8)
    probe_selection <- nn$take_rows(test_selection, first8)
    stages <- deployed$probe(probe, probe_selection)
    nn$save_npz(file.path(root, "probe.npz"),
                list(images = probe, selection = probe_selection, patches = stages[[1L]], visible = stages[[2L]],
                     mask = stages[[3L]], encoder_attention = stages[[4L]], decoder_attention = stages[[5L]]))
    nn$save_npz(file.path(root, "outputs.npz"), saved)
    nn$save_npz(file.path(root, "model_arrays.npz"), autoencoder_arrays(deployed$parameters))
    autoencoder_save_data(file.path(root, "data.npz"), pools)
    autoencoder_copy_manifests(args$data, root)
    test_mse <- masked_autoencoder_mse(test[[2L]], test_images, test_selection)
    zero_mse <- masked_autoencoder_mse(zero_prediction, test_images, test_selection)
    shuffled_mse <- masked_autoencoder_mse(shuffled[[2L]], test_images, test_selection)
    second_mse <- masked_autoencoder_mse(second[[2L]], test_images, second_selection)
    writeLines(sprintf(paste0('{"parameters":13649,"epochs":%d,"updates":%d,"selected_epoch":%d,',
                              '"test_masked_mse":%.17g,"zero_masked_mse":%.17g,"shuffled_masked_mse":%.17g,',
                              '"second_masked_mse":%.17g,"single_batch_max_error":%.17g,',
                              '"visible_permutation_max_error":%.17g,"training_seconds":%.17g}'),
                       epochs, step, selected, test_mse, zero_mse, shuffled_mse, second_mse, single_error,
                       permutation_error, seconds), file.path(root, "metrics.json"))

    # Gallery panels: original, visible input (gray hidden), raw prediction, visible-copy composite,
    # fit-only mean, hidden-only absolute error, zero-code raw prediction.
    predicted <- masked_autoencoder_image(test[[2L]])$values()
    visible <- ifelse(hidden_pixels, 0.5, images)
    composite <- ifelse(hidden_pixels, predicted, images)
    # |float32 difference|: the exact double difference rounds to the same float.
    error <- ifelse(hidden_pixels, abs(predicted - images), 0)
    html <- paste0(
        "<!doctype html><meta charset='utf-8'><title>Tiny transformer MAE</title><style>body{font:16px system-ui;",
        "max-width:1000px;margin:30px auto}img{max-width:100%;image-rendering:pixelated}</style><h1>Visible-only ",
        "transformer masked autoencoder</h1><p>Original,visible input(gray hidden),raw prediction,visible-copy ",
        "composite,fit-only mean,hidden-only absolute error,zero-code raw prediction. Fixed[0,1]gray/nearest4/no ",
        "error amplification. First heldout example per digit,unfiltered. Composite copies known pixels;raw ",
        "prediction does not.</p><p>16nonoverlapping7x7patches;75percent masked,encoder ONLY4visible tokens,width32/",
        "one2-head transformer block. Decoder width16/one2-head block receives restored projected codes and learned ",
        "mask tokens plus fixed2D sin/cos positions. MLP ratio2,tanh-GELU,prenorm;no CLS token or dropout. Sigmoid ",
        "raw-pixel targets;loss ONLY12hidden patches. Small educational adaptation of <a ",
        "href='https://arxiv.org/abs/2111.06377'>He et al.MAE</a>,not an ImageNet reproduction or transfer-learning ",
        "benchmark.</p><p>", epochs,
        "epochs,13649parameters,Adam.003,batch32. New reproducible per-example masks each training epoch;fixed ",
        "disjoint validation/test masks. Test masked MSE=", fmt(test_mse),
        ". Zero/shuffled content and second mask diagnostics remain in metrics. Hidden-input perturbation leaves ",
        "codes/predictions BITEXACT. Short training can produce blurry generic patches;no classification/semantic or ",
        "compression claim.</p>")
    gallery <- autoencoder_digit_gallery(root, pools[[3L]],
        list(test_images, nn$from_vector(visible, c(200L, 784L)), nn$from_vector(predicted, c(200L, 784L)),
             nn$from_vector(composite, c(200L, 784L)), autoencoder_constant(autoencoder_pixel_mean(pools[[1L]]$images), 200L),
             nn$from_vector(error, c(200L, 784L)), masked_autoencoder_image(zero_prediction)))
    writeLines(paste(c(html, gallery), collapse = ""), file.path(root, "index.html"))
    cat(sprintf(paste0("MAE selected epoch%d,test masked MSE=%s,zero=%s,shuffled=%s,second mask=%s;fresh restore/",
                       "exact decode/hidden leakage/permutation/batch/immutable checks passed\ntraining time %s s\n",
                       "Artifacts:%s/index.html\n"),
                selected, fmt(test_mse), fmt(zero_mse), fmt(shuffled_mse), fmt(second_mse), fmt(seconds), root))
    0L
}

if (sys.nframe() == 0L) quit(status = main())
