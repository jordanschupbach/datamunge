# Model container for tflow's small explicit-unroll recurrent models
# (examples/nn/common/recurrent_model.hpp).
#
# A model is an environment holding a ParameterStore, a forward function
# forward(store, inputs, lengths, initial_states) -> named list incl. "logits",
# a loss loss_fn(outputs, labels) -> list(loss, chosen) and the optimizer.
# Gradients, updates and evaluations are written out in each example script
# (forward, loss, nn$grad, optimizer$step); this file only holds the components
# and the tensor comparison utilities the examples share.

if (!exists("nn_require", inherits = TRUE)) nn_helper("checks")

# One batch fed to a model: inputs, int32 lengths and labels.
recurrent_model_feed <- function(inputs, lengths, labels) list(inputs = inputs, lengths = lengths, labels = labels)

# A gradient probe: d target(outputs) / d (input when wrt == 0, else initial state wrt - 1).
recurrent_model_probe <- function(name, target, wrt = 0L) list(name = name, target = target, wrt = wrt)

# Two-class softmax cross-entropy with int32 labels and the summed target logit (tflow's loss and chosen).
recurrent_model_softmax_loss <- function(outputs, labels) {
    logits <- outputs$logits
    shape <- logits$shape()
    one_hot <- nn$one_hot(labels, shape[length(shape)])
    loss <- nn$neg(nn$mean(nn$sum(nn$mul(one_hot, nn$log_softmax(logits, -1L)), 1L), 0L))
    list(loss = loss, chosen = nn$sum(nn$mul(logits, one_hot)))
}

# Bit-exact equality of two tensors, or of two named tensor lists (names, order and values).
recurrent_model_same_bits <- function(a, b) {
    if (is.list(a)) {
        return(length(a) == length(b) && identical(names(a), names(b)) &&
            all(vapply(seq_along(a), function(i) recurrent_model_same_bits(a[[i]], b[[i]]), logical(1))))
    }
    a$dtype() == b$dtype() && identical(as.numeric(a$shape()), as.numeric(b$shape())) && identical(a$values(), b$values())
}

# Outputs list -> the same list in the sorted key order of the C++ std::map (C locale).
recurrent_model_named <- function(outputs) {
    outputs[sort(names(outputs), method = "radix")]
}

recurrent_model_find <- function(arrays, name) {
    if (is.null(arrays[[name]])) stop("missing array ", name, call. = FALSE)
    arrays[[name]]
}

recurrent_model_max_abs_diff <- function(a, b) {
    x <- a$values()
    y <- b$values()
    if (length(x) != length(y)) return(Inf)
    if (length(x)) max(abs(x - y)) else 0
}

# A gradient from nn$grad (detached), or zeros like `like` when the target did not use it.
recurrent_model_defined_gradient <- function(gradient, like) {
    if (gradient$defined()) gradient$detach() else nn$zeros_like(like$detach())
}

recurrent_model_new <- function(seed, forward, input_shape, initial_shapes, initial_gradient_names, fields) {
    model <- new.env()
    model$store <- nn$ParameterStore(seed)
    model$forward <- forward
    model$loss_fn <- recurrent_model_softmax_loss
    model$probability <- function(logits) nn$softmax(logits, -1L)
    model$input_shape <- input_shape                        # per-example input shape
    model$initial_shapes <- initial_shapes                  # per-example zero initial states
    model$initial_gradient_names <- initial_gradient_names  # "" = not reported
    model$fields <- fields                                  # forward fields returned by predict
    model$evaluate_probes <- list()
    model$side_inputs <- list()
    model$optimizer <- NULL
    model
}

recurrent_model_initials <- function(model, n) lapply(model$initial_shapes, function(shape) nn$zeros(c(n, shape)))

# Creates every parameter with one no-grad example forward; with `training`, adds the optimizer.
recurrent_model_build <- function(model, training, learning_rate = 0.01, kind = nn$OptimizerKind$Adam) {
    nn$no_grad(function() model$forward(model$store, nn$zeros(c(1, model$input_shape)), nn$zeros(1, nn$DType$Int32),
        recurrent_model_initials(model, 1)))
    if (training) {
        config <- nn$OptimizerConfig()
        config$kind <- kind
        model$optimizer <- nn$Optimizer(model$store, learning_rate, config)
    }
    model
}

# Sequence and lengths only, no labels or gradients (tflow's Predict).
recurrent_model_predict <- function(model, feed) {
    nn$no_grad(function() {
        out <- model$forward(model$store, feed$inputs, feed$lengths, recurrent_model_initials(model, feed$inputs$shape()[1]))
        setNames(lapply(model$fields, function(f) if (f == "probability") model$probability(out$logits) else out[[f]]),
            model$fields)
    })
}

# Copies of every parameter, by name, in creation order.
recurrent_model_arrays <- function(model) nn_arrays(model$store)

recurrent_model_save <- function(model, path) model$store$save(path)
recurrent_model_restore <- function(model, path) model$store$load(path, TRUE)

# Overwrites every parameter from an .npz written by tflow (initial_arrays.npz).
recurrent_model_load_arrays <- function(model, path) {
    arrays <- nn$load_npz(path)
    nn_require(length(arrays) == model$store$size(), "init arrays cover exactly the model's parameters")
    for (name in names(arrays)) {
        nn_require(model$store$contains(name), paste0("init array ", name, " is a model parameter"))
        model$store$set(name, nn$cast(arrays[[name]], model$store$at(name)$dtype()))
    }
}
