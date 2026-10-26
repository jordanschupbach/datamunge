# Native R NN example. Model, differentiation and training loops run in R.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT", getwd()), "examples/r/nn/common.R")
if (!file.exists(.nn_common))
    .nn_common <- system.file("examples/nn/common.R", package = "datamungerNN", mustWork = TRUE)
source(.nn_common, local = TRUE)
rm(.nn_common)

undercomplete_model <- function(training = TRUE, features = 784L, latent = 32L) {
    stopifnot(latent > 0L, latent < features)
    model <- new.env(parent = environment())
    model$parameters <- nn$ParameterStore(1471L)
    model$decode <- function(code) nn$dense(model$parameters, "decoder", code, features, nn$Activation$Sigmoid)
    model$forward <- function(images) {
        flat <- nn$reshape(images, c(images$size(0L), features))
        code <- nn$dense(model$parameters, "encoder", flat, latent, nn$Activation$Relu)
        list(code, model$decode(code))
    }
    model$loss <- function(images) nn$mean(nn$square(nn$sub(model$forward(images)[[2L]], nn$reshape(images, 
        c(images$size(0L), features)))))
    model$evaluate <- function(images) nn$no_grad(function() model$forward(images))
    nn$no_grad(function() model$forward(nn$zeros(c(1L, features))))
    if (training) {
        config <- nn$OptimizerConfig()
        config$kind <- nn$OptimizerKind$Adam
        model$optimizer <- nn$Optimizer(model$parameters, 0.003, config)
    }
    model$update <- function(images) {
        loss <- model$loss(images)
        value <- loss$item()
        stopifnot(is.finite(value), value >= 0)
        model$optimizer$minimize(loss)
        value
    }
    model
}

undercomplete_checks <- function() {
    model <- undercomplete_model(features = 2L, latent = 1L)
    fixed <- list(c(0.3, 0.2), 0.1, c(0.4, -0.5), c(-0.2, 0.1))
    names <- model$parameters$names()
    for (i in seq_along(names)) model$parameters$set(names[i], nn$from_vector(fixed[[i]], model$parameters$at(names[i])$shape()))
    inputs <- c(0.1, 0.8, 0.7, 0.2)
    x <- nn$from_vector(inputs, c(2L, 2L))
    result <- model$forward(x)
    loss <- nn$mean(nn$square(nn$sub(result[[2L]], x)))
    gradients <- unlist(lapply(nn$grad(loss, model$parameters$parameters()), function(t) t$values()))
    expected <- numeric(7L)
    mse <- 0
    for (row in 0L:1L) {
        code <- 0.1 + sum(inputs[row * 2L + 1L:2L] * fixed[[1L]])
        stopifnot(abs(result[[1L]]$values()[row + 1L] - code) < 3e-08)
        dz <- 0
        for (j in 1L:2L) {
            prediction <- 1/(1 + exp(-(code * fixed[[3L]][j] + fixed[[4L]][j])))
            stopifnot(abs(result[[2L]]$values()[row * 2L + j] - prediction) < 5e-08)
            error <- prediction - inputs[row * 2L + j]
            mse <- mse + error^2/4
            delta <- error/2 * prediction * (1 - prediction)
            expected[3L + j] <- expected[3L + j] + code * delta
            expected[5L + j] <- expected[5L + j] + delta
            dz <- dz + delta * fixed[[3L]][j]
        }
        expected[1L:2L] <- expected[1L:2L] + inputs[row * 2L + 1L:2L] * dz
        expected[3L] <- expected[3L] + dz
    }
    stopifnot(abs(loss$item() - mse) < 2e-08, max(abs(gradients - expected)) < 2e-08)
}

load_saliency <- function(directory) {
    arrays <- nn$load_npz(file.path(directory, "saliency_data.npz"))
    result <- list()
    seen <- numeric()
    seen_images <- new.env(hash = TRUE, parent = emptyenv())
    for (split in c("fit", "validation", "test")) {
        n <- if (split == "fit") 
            1000L
        else 200L
        images <- arrays[[paste0(split, "_images")]]
        labels <- arrays[[paste0(split, "_labels")]]
        ids <- arrays[[paste0(split, "_source_ids")]]
        stopifnot(identical(as.numeric(images$shape()), c(n, 28, 28, 1)), length(labels$values()) == 
            n, length(ids$values()) == n)
        stopifnot(!anyDuplicated(c(seen, ids$values())))
        seen <- c(seen, ids$values())
        stopifnot(all(tabulate(as.integer(labels$values()) + 1L, 10L) == n/10L))
        stopifnot(if (split == "test") 
            all(ids$values() >= 60000 & ids$values() < 70000)
        else all(ids$values() >= 0 & ids$values() < 60000))
        pixels <- images$values()
        stopifnot(all(pixels >= 0 & pixels < 256))
        for (row in seq_len(n)) {
            key <- paste(pixels[(row - 1L) * 784L + seq_len(784L)], collapse = ",")
            stopifnot(!exists(key, seen_images, inherits = FALSE))
            assign(key, TRUE, seen_images)
        }
        result[[split]] <- list(images = nn$div(nn$cast(images, nn$DType$Float32), 255), labels = labels, 
            ids = ids)
    }
    result
}

pixel_mse <- function(prediction, original) {
    values <- prediction$values()
    stopifnot(length(values) == original$numel(), all(is.finite(values)), all(values >= 0 & values <= 
        1))
    mean((values - original$values())^2)
}

main <- function(arguments = commandArgs(TRUE)) {
    args <- nn_options(arguments, list(epochs = 3L, data = file.path(nn_data_root(), "saliency"), out = "", 
        init = "", full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    stopifnot(args$epochs >= 1L, args$epochs <= 100L)
    undercomplete_checks()
    directory <- if (nzchar(args$out)) 
        args$out
    else file.path(nn_output_root(args$full), "models/undercomplete_autoencoder")
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    pools <- load_saliency(args$data)
    fit <- pools$fit$images
    validation <- pools$validation$images
    test <- pools$test$images
    model <- undercomplete_model()
    if (nzchar(args$init)) 
        model$parameters$load(args$init)
    stopifnot(model$parameters$size() == 4L, model$parameters$count() == 50992L)
    nn$save_npz(file.path(directory, "initial_arrays.npz"), nn_arrays(model$parameters))
    initial_fit <- model$evaluate(fit)
    initial_test <- model$evaluate(test)
    initial_mse <- pixel_mse(initial_fit[[2L]], fit)
    pixels <- matrix(fit$values(), ncol = 784L, byrow = TRUE)
    pixelmean <- nn$from_vector(colMeans(pixels), 784L)
    mean_prediction <- nn$tile(nn$reshape(pixelmean, c(1L, 784L)), c(200L, 1L))$contiguous()
    black <- nn$zeros(c(200L, 784L))
    batches <- nn$BatchOrder(1000L, 32L, 1483L)
    best <- Inf
    selected <- step <- 0L
    history <- file(file.path(directory, "history.tsv"), "w")
    provenance <- file(file.path(directory, "batches.tsv"), "w")
    on.exit({
        close(history)
        close(provenance)
    }, add = TRUE)
    writeLines("epoch\ttrain_preupdate_mse\tvalidation_mse", history)
    writeLines("epoch\tstep\trow\tsource_id", provenance)
    started <- proc.time()[["elapsed"]]
    for (epoch in seq_len(args$epochs)) {
        batches$begin_epoch()
        total <- 0
        seen <- 0L
        repeat {
            rows <- batches$`_next`()
            if (!length(rows)) 
                break
            images <- nn$take_rows(fit, rows)
            if (step == 0L) {
                first_images <- images
                first_ids <- nn$take_rows(pools$fit$ids, rows)
                gradients <- nn$grad(model$loss(images), model$parameters$parameters())
                for (gradient in gradients) stopifnot(all(is.finite(gradient$values())), any(gradient$values() != 
                  0))
                nn$save_npz(file.path(directory, "first_gradients.npz"), setNames(gradients, model$parameters$names()))
            }
            total <- total + length(rows) * model$update(images)
            seen <- seen + length(rows)
            for (row in rows) writeLines(sprintf("%d\t%d\t%d\t%d", epoch, step, row, as.integer(pools$fit$ids$values()[row + 
                1L])), provenance)
            if (step == 0L) 
                nn$save_npz(file.path(directory, "first_update_arrays.npz"), nn_arrays(model$parameters))
            step <- step + 1L
        }
        stopifnot(seen == 1000L)
        mse <- pixel_mse(model$evaluate(validation)[[2L]], validation)
        writeLines(sprintf("%d\t%.17g\t%.17g", epoch, total/seen, mse), history)
        nn$save_npz(file.path(directory, paste0("epoch", epoch, "_arrays.npz")), nn_arrays(model$parameters))
        if (mse < best) {
            best <- mse
            selected <- epoch
            model$parameters$save(file.path(directory, "best.npz"))
        }
        cat(sprintf("epoch%d:preupdate fitMSE=%.7g,validationMSE=%.7g\n", epoch, total/seen, mse))
    }
    training_seconds <- proc.time()[["elapsed"]] - started
    deployed <- undercomplete_model(FALSE)
    deployed$parameters$load(file.path(directory, "best.npz"))
    before <- deployed$parameters$flatten()
    fit_result <- deployed$evaluate(fit)
    val_result <- deployed$evaluate(validation)
    result <- deployed$evaluate(test)
    stopifnot(abs(pixel_mse(val_result[[2L]], validation) - best) < 1e-08, identical(result[[2L]]$values(), 
        deployed$evaluate(test)[[2L]]$values()))
    controls <- nn$no_grad(function() {
        stopifnot(identical(result[[2L]]$values(), deployed$decode(result[[1L]])$values()))
        shifted <- nn$take_rows(result[[1L]], c(1L:199L, 0L))$contiguous()
        list(shifted = shifted, shuffled = deployed$decode(shifted), zero = deployed$decode(nn$zeros(c(200L, 
            32L))))
    })
    stopifnot(pixel_mse(fit_result[[2L]], fit) < initial_mse * 0.8, identical(before, deployed$parameters$flatten()))
    archive <- list(fit_images = fit, fit_source_ids = pools$fit$ids, validation_images = validation, 
        validation_source_ids = pools$validation$ids, test_images = test, test_labels = pools$test$labels, 
        test_source_ids = pools$test$ids, fit_code = fit_result[[1L]], fit_reconstruction = fit_result[[2L]], 
        validation_code = val_result[[1L]], validation_reconstruction = val_result[[2L]], test_code = result[[1L]], 
        test_reconstruction = result[[2L]], initial_test_reconstruction = initial_test[[2L]], fit_pixelmean = pixelmean, 
        mean_reconstruction = mean_prediction, black_reconstruction = black, shifted_codes = controls$shifted, 
        shuffled_reconstruction = controls$shuffled, zero_reconstruction = controls$zero, first_batch_images = first_images, 
        first_batch_source_ids = first_ids)
    nn$save_npz(file.path(directory, "autoencoder.npz"), archive)
    nn$save_npz(file.path(directory, "model_arrays.npz"), nn_arrays(deployed$parameters))
    model$parameters$load(file.path(directory,"best.npz"))
    stopifnot(identical(model$evaluate(test)[[2L]]$values(),result[[2L]]$values()))
    single <- deployed$evaluate(nn$take_rows(test,0L))[[2L]]$values()
    batch_error <- max(abs(single-head(result[[2L]]$values(),784L)))
    stopifnot(batch_error<1e-6)
    metrics <- list(epochs=args$epochs,selected_epoch=selected,updates=step,fit_initial_mse=initial_mse,
      fit_mse=pixel_mse(fit_result[[2L]],fit),validation_mse=best,test_mse=pixel_mse(result[[2L]],test),
      test_initial_mse=pixel_mse(initial_test[[2L]],test),test_mean_mse=pixel_mse(mean_prediction,test),
      test_black_mse=pixel_mse(black,test),test_shuffled_mse=pixel_mse(controls$shuffled,test),
      test_zero_code_mse=pixel_mse(controls$zero,test),single_batch_max_error=batch_error,training_seconds=training_seconds)
    writeLines(paste0("{",paste(sprintf('"%s":%.17g',names(metrics),unlist(metrics)),collapse=","),"}"),file.path(directory,"metrics.json"))
    panels <- list(test,result[[2L]],nn$abs(nn$sub(result[[2L]],nn$reshape(test,c(200L,784L)))),
      mean_prediction,initial_test[[2L]],controls$zero,controls$shuffled)
    sections <- character()
    for(digit in 0L:9L) {
      row <- which(pools$test$labels$values()==digit)[1L]-1L
      tiles <- lapply(panels,function(panel)nn$reshape(nn$slice_axis(panel,0L,row,row+1L),c(28L,28L,1L)))
      filename <- paste0("digit",digit,".png")
      nn$save_image_grid(nn$stack(nn$tensor_vector(tiles),0L),file.path(directory,filename),7L,0,1,4L,220/255,4L)
      sections <- c(sections,sprintf("<section><h2>Digit%d,source%d</h2><img src='%s'></section>",digit,as.integer(pools$test$ids$values()[row+1L]),filename))
    }
    writeLines(c("<!doctype html><meta charset='utf-8'><title>Undercomplete MNIST autoencoder</title>",
      "<h1>784 to32 to784 undercomplete autoencoder</h1><p>Original, reconstruction, absolute error, fit-only mean, initial model, zero-code decoder, shuffled-code decoder. Fixed[0,1]intensity, nearest4 enlargement. Reconstruction-only training, validation MSE selects the checkpoint. Fewer code coordinates do not imply equivalent file compression.</p>",sections),file.path(directory,"index.html"))
    for(pair in list(c(file.path(args$data,"manifest.json"),"data_manifest.json"),c(file.path(nn_data_root(),"autoencoder/source_manifest.json"),"source_manifest.json")))
      if(file.exists(pair[1L]))file.copy(pair[1L],file.path(directory,pair[2L]),overwrite=TRUE)
    cat(sprintf("undercomplete_autoencoder:%d updates,selected=%d,test MSE=%.7g\n", step, selected, pixel_mse(result[[2L]], 
        test)))
    0L
}

if (sys.nframe() == 0L) quit(status = main())

