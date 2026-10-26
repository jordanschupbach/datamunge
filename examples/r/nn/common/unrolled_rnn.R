# Explicit twelve-step shared-weight tanh recurrence with valid-length state
# masking (examples/nn/common/unrolled_rnn.hpp; 206 parameters).
if (!exists("recurrent_model_new", inherits = TRUE)) nn_helper("recurrent_model")
if (!exists("rnn_blend", inherits = TRUE)) nn_helper("rnn")

unrolled_rnn <- function(store, inputs, lengths, initial) {
    steps <- rnn_steps(inputs)
    h <- initial[[1]]
    states <- list(h)
    for (t in 0:11) {
        h <- rnn_blend(rnn_step_mask(lengths, t), nn$rnn_step(store, "cell", steps[[t + 1]], h), h)
        states[[length(states) + 1L]] <- h
    }
    list(logits = rnn_head(store, "head", h, 12, 2), states = rnn_stack_time(states), final_state = h)
}

# tflow's ml::UnrolledRnnModel (vanilla_rnn): 206 parameters, Adam .01, seed 26003.
unrolled_rnn_model <- function(training = TRUE, seed = 26003) {
    model <- recurrent_model_new(seed, unrolled_rnn, c(12, 2), list(12), "initial_gradient",
        c("logits", "probability", "states", "final_state"))
    recurrent_model_build(model, training)
}
