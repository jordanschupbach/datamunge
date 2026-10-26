# Native R NN example. Kim CNN-rand sentence classification with Adadelta written out in R.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
# CMD: just r-example kim_sentence_cnn
# Kim2014 CNN-rand: learned embeddings,parallel3/4/5-ReLU word convolutions,
# max-over-time pooling,dropout.5,softmax and Adadelta with head maxnorm3.
# Real pinned MR v1.0 sentences;fit-only vocabulary,duplicate-group holdout.
# Full architecture and independent every-coordinate host gradient fixtures.
# Three compact default epochs;untrained/majority/fit-only unigram NB controls.
#
# Port of tflow's kim_sentence_cnn. Data: tflow's `just sentence-assets`
# (output/datasets/sentence/sentence.npz, found via nn_dataset). Deviations:
# .npz checkpoints; tflow's GraphDef placeholder audit has no eager
# counterpart. Numbers match the C++ port exactly. Host fixtures mirror the
# C++ build's fused multiply-adds (clang -ffp-contract=on) with sentence_fma.
# Options: epochs=3 data=PATH checks_only=0|1 threads=4.
# Output: <output root>/models/kim_sentence_cnn/.

nn_helper("checks")
nn_helper("replay")
nn_helper("kim_cnn")
nn_helper("sentence")
fma <- sentence_fma
f32 <- sentence_f32
fmt <- sentence_fmt

train_sentence <- function(root, epochs) {
    nn_require(epochs > 0L, "positive short sentiment epochs")
    data <- sentence_load(root)
    fit <- data$pools[[1L]]
    validation <- data$pools[[2L]]
    test <- data$pools[[3L]]
    config <- data$config
    model <- sentence_model(config)
    directory <- nn_output_dir("models/kim_sentence_cnn")
    prefix <- file.path(directory, "best.npz")
    probe <- sentence_rows(fit, sentence_range(0L, 50L))
    initial <- model$loss(probe)
    untrained <- model$predict_all(test)
    cat(sprintf(paste0("CNN-rand compact:%s parameters/%s learned tensors,18 Adadelta slots;E32,F32,windows3/4/5,dropout.5,",
                       "maxnorm3;%d epochs,batch50,rho.95,eps1e-6,initial fixed CE=%s\n"),
                sentence_int(model$parameters$count()), sentence_int(model$parameters$size()), as.integer(epochs), fmt(initial)))
    nn_require(model$optimizer$slots$size() == 18, "two Adadelta slots per learned tensor")
    probe_loss <- sentence_ce(model$forward(probe, TRUE)$logits, probe$labels)
    sentence_require_live_gradients(model$parameters, nn$grad(probe_loss, model$parameters$parameters()), "embedding/CNN/head")
    cat(" all9 actual training gradient tensors finite/nonzero\n")
    history <- file(file.path(directory, "history.tsv"), "w")
    writeLines("epoch\ttrain_dropout_ce\tvalidation_ce", history)
    order <- nn$BatchOrder(fit$size, 50L, 331L)
    best <- Inf
    selected <- 0L
    for (epoch in seq_len(epochs)) {
        order$begin_epoch()
        total <- 0
        seen <- 0L
        repeat {
            rows <- order$`_next`()
            if (!length(rows)) break
            # Dropout training graph, CE, all gradients and one Adadelta/max-norm step.
            batch <- sentence_rows(fit, rows)
            loss <- sentence_ce(model$forward(batch, TRUE)$logits, batch$labels)
            model$optimizer$step(nn$grad(loss, model$parameters$parameters()))
            value <- loss$item()
            nn_require(is.finite(value) && value >= 0, "finite sentence CE")
            total <- total + length(rows) * value
            seen <- seen + length(rows)
        }
        nn_require(seen == fit$size, "each fitting sentence once per epoch")
        score <- model$loss(validation)
        if (score < best) {
            best <- score
            selected <- epoch
            model$save(prefix)
        }
        writeLines(sprintf("%d\t%s\t%s", epoch, fmt(total / seen), fmt(score)), history)
        flush(history)
        cat(sprintf(" epoch %d:trainCE=%s,validationCE=%s\n", epoch, fmt(total / seen), fmt(score)))
    }
    close(history)
    model$restore(prefix)
    nn_near(model$loss(validation), best, 1e-6, "restored selected validation CE")
    learned <- model$loss(probe)
    nn_require(learned < initial, "short-run fixed fitting sentiment CE improves")
    # The deployed model is a fresh inference-only model (no labels, no optimizer).
    deployed <- sentence_model(config, FALSE)
    deployed$restore(prefix)
    nn_require(deployed$parameters$size() == 9 && is.null(deployed$optimizer), "inference model holds only the nine learned tensors")
    before <- deployed$snapshot()
    logits <- deployed$predict_all(test)
    reference <- model$predict_all(test)
    repeated <- deployed$predict_all(test)
    nn_near(replay_max_abs_diff(logits, reference), 0, 1e-6, "training/inference independent restore parity")
    nn_require(replay_max_abs_diff(logits, repeated) == 0, "dropout-free deterministic inference")
    one <- sentence_rows(test, 0L)
    single <- deployed$predict(one)
    nn_near(replay_max_abs_diff(single, nn$slice_axis(logits, 0L, 0L, 1L)), 0, 1e-6, "single/batch sentence agreement")
    length_0 <- sentence_ints(one$lengths)[1L]
    tokens <- sentence_ints(one$tokens)
    if (length_0 < 64L) tokens[(length_0 + 1L):64L] <- 1L
    one$tokens <- sentence_int_tensor(tokens, c(1L, 64L))
    nn_require(replay_max_abs_diff(deployed$predict(one), single) == 0, "legal out-of-length token substitution invariant")
    one <- sentence_rows(test, 0L)
    tokens <- sentence_ints(one$tokens)
    tokens[seq_len(length_0)] <- 1L
    one$tokens <- sentence_int_tensor(tokens, c(1L, 64L))
    sensitivity <- nn$sum(nn$abs(nn$sub(nn$cast(single, nn$DType$Float64), nn$cast(deployed$predict(one), nn$DType$Float64))))$item()
    nn_require(sensitivity > 1e-6, "prediction depends on observed word IDs")
    nn_require(replay_same_bits(before, deployed$snapshot()), "inference leaves all nine learned tensors unchanged")
    head <- deployed$parameters$at("classifier/W")$values()
    for (c in 0:1) {
        norm <- 0
        for (i in seq_len(3L * config$filters) - 1L) norm <- fma(head[2L * i + c + 1L], head[2L * i + c + 1L], norm)
        nn_require(norm <= 9 + 1e-5, "classifier column norm bounded by3")
    }
    nb <- sentence_nb(fit, test, config$vocabulary)
    majority <- nn$zeros(c(test$size, 2L))
    sentence_metrics("trained CNN-rand", test$labels, logits)
    sentence_metrics("untrained", test$labels, untrained)
    sentence_metrics("fit unigram multinomial NB", test$labels, nb)
    sentence_metrics("majority/tie class0", test$labels, majority)
    sentence_print_confusion("trained CNN-rand test", sentence_confusion(test$labels, sentence_positive(logits), 2L), 2L)
    z <- matrix(logits$values(), ncol = 2L, byrow = TRUE)
    writeLines(c("source_id\tlabel\tprediction\tnb_prediction\tnegative_logit\tpositive_logit",
                 sprintf("%d\t%d\t%d\t%d\t%s\t%s", sentence_ints(test$source_ids), sentence_ints(test$labels),
                         sentence_positive(logits), sentence_positive(nb), fmt(z[, 1L]), fmt(z[, 2L]))),
               file.path(directory, "predictions.tsv"))
    nn$save_npz(file.path(directory, "test_outputs.npz"),
                list(tokens = test$tokens, lengths = test$lengths, labels = test$labels, source_ids = test$source_ids,
                     logits = logits, untrained = untrained, nb_logits = nb))
    sentence_copy_if_exists(file.path(root, "manifest.json"), file.path(directory, "data_manifest.json"))
    cat(sprintf(paste0(" selected=%d,fixed fitting CE %s -> %s,word substitution logit change=%s;",
                       "restore/padding/single-batch/determinism/immutable/maxnorm passed\nArtifacts:%s\n",
                       "Custom MR holdout,CNN-rand,not pretrained or published cross-validation accuracy.\n"),
                as.integer(selected), fmt(initial), fmt(learned), fmt(sensitivity), sentence_quoted(directory)))
}

# ---- independent fixtures -------------------------------------------------------------------

host_ids <- list(c(1L, 2L, 3L, 6L, 5L), c(2L, 4L, 1L, 5L, 3L))
host_lengths <- c(3L, 5L)

# Wide (pad 4) 3/4/5 convolutions, ReLU through max(0, .), max-over-time and head in double.
kim_host <- function(w) {
    result <- numeric(4L)
    for (n in 1:2) {
        pooled <- numeric(6L)
        for (branch in 0:2) {
            kernel <- branch + 3L
            for (f in 0:1) {
                maximum <- 0
                for (t in 0:(5L + 8L - kernel)) {
                    value <- w[[3L + 2L * branch]][f + 1L]
                    for (k in 0:(kernel - 1L)) {
                        position <- t + k - 4L
                        if (position < 0L || position >= host_lengths[n]) next
                        for (e in 0:1)
                            value <- fma(w[[1L]][(host_ids[[n]][position + 1L] - 1L) * 2L + e + 1L],
                                         w[[2L + 2L * branch]][(k * 2L + e) * 2L + f + 1L], value)
                    }
                    maximum <- max(maximum, value)
                }
                pooled[2L * branch + f + 1L] <- maximum
            }
        }
        for (c in 0:1) {
            index <- 2L * (n - 1L) + c + 1L
            result[index] <- w[[9L]][c + 1L]
            for (f in 0:5) result[index] <- fma(pooled[f + 1L], w[[8L]][2L * f + c + 1L], result[index])
        }
    }
    result
}

kim_objective <- function(w) {
    z <- kim_host(w)
    fma(.4, z[4L], fma(-.1, z[3L], fma(.3, z[1L], -(.2 * z[2L]))))
}

host_checks <- function() {
    p <- nn$ParameterStore(337)
    config <- kim_cnn_config(7L, 5L, 2L, 2L)
    tokens <- sentence_int_tensor(c(1, 2, 3, 6, 5, 2, 4, 1, 5, 3), c(2L, 5L))
    lengths <- sentence_int_tensor(c(3, 5), 2L)
    weight <- nn$from_vector(c(.3, -.2, -.1, .4), c(2L, 2L))
    kim_cnn_forward(p, tokens, lengths, config)
    weights <- sentence_fix_parameters(p, function(v, i, shape) {
        if (length(shape) == 1L && v != 8L) f32(fma(f32(.03), f32(sin(i + v)), f32(.17)))
        else f32(.23 * sin(.7 * i + 1.3 * v + .2))
    })
    net <- kim_cnn_forward(p, tokens, lengths, config)
    gradients <- nn$grad(nn$sum(nn$mul(net$logits, weight)), p$parameters())
    expected <- kim_host(weights)
    z <- net$logits$values()
    value_error <- max(abs(expected - z))
    for (e in nn$slice_axis(net$embedding, 0L, 0L, 1L)$values()) nn_near(e, 0, 0, "structural PAD embedding exactly zero")
    fd <- sentence_host_gradient_error(weights, gradients, kim_objective)
    nn_require(fd$coordinates == 80L && p$size() == 9, "all80 tiny parameter coordinates covered")
    nn_near(value_error, 0, 1e-7, "independent host wide convolution/ReLU/max/concat/head")
    nn_near(fd$error, 0, 2e-6, "EVERY embedding/filter/bias/head derivative versus host FD")
    t <- sentence_ints(tokens)
    t[4L] <- 4L
    t[5L] <- 1L
    out <- kim_cnn_forward(p, sentence_int_tensor(t, c(2L, 5L)), lengths, config)$logits$values()
    for (c in 1:2) nn_near(out[c], expected[c], 1e-7, "padding values cannot affect sentence features")
    t[c(6L, 10L)] <- t[c(10L, 6L)]
    out <- kim_cnn_forward(p, sentence_int_tensor(t, c(2L, 5L)), lengths, config)$logits$values()
    difference <- 0
    for (c in 1:2) difference <- difference + abs(out[2L + c] - expected[2L + c])
    nn_require(difference > 1e-5, "window features depend on word order")
    cat(sprintf(" independent wide3/4/5 conv/ReLU/max/head error=%s,ALL80 host finite-difference derivative error=%s;zeroPAD/padding/order passed\n",
                fmt(value_error), fmt(fd$error)))
}

optimizer_checks <- function() {
    p <- nn$ParameterStore(347)
    p$get("classifier/W", c(2L, 2L), nn$init$zeros())
    p$set("classifier/W", nn$from_vector(c(3, .1, 4, .2), c(2L, 2L)))
    step <- kim_cnn_adadelta(p)
    host <- c(3, .1, 4, .2)
    eg <- numeric(4L)
    ed <- numeric(4L)
    error <- 0
    for (t in 0:2) {
        gradient <- numeric(4L)
        for (i in 1:4) {
            value <- f32(f32(i * f32(.2)) * (if (t == 1L) -1 else 1))
            gradient[i] <- value
            eg[i] <- fma(.95, eg[i], .05 * value * value)
            delta <- sqrt(ed[i] + 1e-6) / sqrt(eg[i] + 1e-6) * value
            ed[i] <- fma(.95, ed[i], .05 * delta * delta)
            host[i] <- host[i] - delta
        }
        for (c in 1:2) {
            scale <- max(1, sqrt(fma(host[c], host[c], host[2L + c] * host[2L + c])) / 3)
            host[c] <- host[c] / scale
            host[2L + c] <- host[2L + c] / scale
        }
        step$step(list(nn$from_vector(gradient, c(2L, 2L))))
        w <- p$at("classifier/W")$values()
        g <- step$slots$at("gradient0")$values()
        d <- step$slots$at("update0")$values()
        error <- max(error, abs(host - w), abs(eg - g), abs(ed - d))
    }
    nn_near(error, 0, 1e-6, "independent three-step Adadelta/both-slot/head-column projection")
    cat(sprintf(" three signed Adadelta steps/all slots/maxnorm3 host error=%s\n", fmt(error)))
}

dropout_checks <- function() {
    gen <- nn$Generator(359)
    x <- nn$full(4096L, 1)
    x$requires_grad_(TRUE)
    training <- nn$dropout(x, .5, TRUE, gen)
    inference <- nn$dropout(x, 0, TRUE, gen)
    gradient <- nn$grad(nn$sum(training), list(x))[[1L]]$values()
    values <- training$values()
    identity <- inference$values()
    nn_require(all(values == 0 | values == 2), "dropout.5 mask has only0 or2")
    nn_require(identical(gradient, values), "dropout derivative uses SAME sampled mask")
    nn_require(all(identity == 1), "keep1 dropout identity")
    retained <- sum(values == 2)
    nn_require(retained > 1800 && retained < 2300, "seeded half-drop mask retention")
    cat(sprintf(" dropout.5 values/shared-mask derivatives and keep1 identity passed,retained=%d/4096\n", as.integer(retained)))
}

loss_checks <- function() {
    z <- nn$from_vector(c(1000, 999, -2, 3), c(2L, 2L))
    z$requires_grad_(TRUE)
    loss <- sentence_ce(z, sentence_int_tensor(c(1, 0), 2L))
    g <- nn$grad(loss, list(z))[[1L]]$values()
    logits <- z$values()
    expected <- 0
    for (n in 0:1) {
        a <- logits[2L * n + 1L]
        b <- logits[2L * n + 2L]
        m <- max(a, b)
        total <- exp(a - m) + exp(b - m)
        label <- 1L - n
        expected <- expected + (m + log(total) - (if (label == 1L) b else a)) / 2
        for (c in 0:1)
            nn_near(g[2L * n + c + 1L], (exp((if (c == 1L) b else a) - m) / total - (c == label)) / 2, 1e-7,
                    "all stable mean CE logit derivatives")
    }
    nn_near(loss$item(), expected, 2e-7, "independent stable mean binary CE")
    cat(" extreme-logit stable CE/batch mean/all logit derivatives passed\n")
}

full_checks <- function() {
    p <- nn$ParameterStore(353)
    ids <- 2 + ((0:127) * 17) %% 7997
    model <- kim_cnn_forward(p, sentence_int_tensor(ids, c(2L, 64L)), sentence_int_tensor(c(64, 37), 2L))
    norms <- sentence_squared_norms(nn$grad(nn$sum(nn$square(model$logits)), p$parameters()))
    nn_require(p$count() == 2760602 && p$size() == 9, "independent full CNN-rand parameter formula")
    for (i in 1:3) nn_require(identical(sentence_shape(model$maps[[i]]), c(2, 71 - i, 100)), "window3/4/5 map shape")
    nn_require(identical(sentence_shape(model$features), c(2, 300)) && identical(sentence_shape(model$logits), c(2, 2)),
               "pooled concatenation and binary output")
    nn_require(all(is.finite(norms) & norms > 0), "all9 full parameter tensor gradients finite/nonzero")
    cat(" full CNN-rand2760602 parameters/9 tensors,E300,F100,map70/69/68,concat300;every learned tensor gradient passed\n")
}

main <- function(arguments = commandArgs(TRUE)) {
    args <- nn_options(arguments, list(epochs = 3L, data = nn_dataset("sentence"), checks_only = 0L, threads = 4L,
                                       full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    nn_require(args$checks_only %in% c(0L, 1L), "expected epochs=N data=PATH checks_only=0|1 threads=N")
    nn$set_num_threads(args$threads)
    host_checks()
    optimizer_checks()
    dropout_checks()
    loss_checks()
    full_checks()
    if (args$checks_only == 0L) train_sentence(args$data, args$epochs)
    0L
}

if (sys.nframe() == 0L) quit(status = main())
