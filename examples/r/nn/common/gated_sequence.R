# Explicit masked sequence unroll around the handwritten LSTM, reset-before
# GRU, peephole LSTM and layer-norm LSTM cells (examples/nn/common/gated_sequence.hpp;
# tflow's common/nn/gated_sequence.h and common/ml/gated_sequence.h: 12 hidden
# units, 12 steps, 12->2 head, Adam .01, Params seed 27003).
if (!exists("recurrent_model_new", inherits = TRUE)) nn_helper("recurrent_model")
if (!exists("layer_norm_lstm_step", inherits = TRUE)) nn_helper("layer_norm_lstm")

# GatedCell kinds: "lstm", "gru", "peephole", "layer_norm".

# Plain LSTM step (nn$lstm_step's parameters and arithmetic) that also returns its gates.
gated_sequence_lstm_step <- function(store, name, x, incoming, units, h, c, forget_bias = 1) {
    wx <- store$get(paste0(name, "/Wx"), c(incoming, 4 * units), nn$init$uniform(sqrt(6 / (incoming + 4 * units))))
    wh <- store$get(paste0(name, "/Wh"), c(units, 4 * units), nn$init$uniform(sqrt(6 / (5 * units))))
    b <- peephole_lstm_bias(store, paste0(name, "/b"), units, forget_bias)
    parts <- rnn_split(nn$bias_add(nn$add(nn$matmul(x, wx), nn$matmul(h, wh)), b), 4L, 1L)
    i <- nn$sigmoid(parts[[1]])
    f <- nn$sigmoid(parts[[2]])
    g <- nn$tanh(parts[[3]])
    o <- nn$sigmoid(parts[[4]])
    cell <- nn$add(nn$mul(f, c), nn$mul(i, g))
    peephole_lstm_gated_step(nn$mul(o, nn$tanh(cell)), cell, rnn_concat(list(i, f, g, o), 1L))
}

# Reset-before GRU step (nn$gru_step's parameters and arithmetic) that also returns concat(z, r, candidate).
gated_sequence_gru_step <- function(store, name, x, incoming, u, h, update_bias = 0) {
    wx <- store$get(paste0(name, "/Wx_zr"), c(incoming, 2 * u), nn$init$uniform(sqrt(6 / (incoming + 2 * u))))
    wh <- store$get(paste0(name, "/Wh_zr"), c(u, 2 * u), nn$init$uniform(sqrt(6 / (3 * u))))
    dtype <- store$dtype()
    b <- rnn_fixed_parameter(store, paste0(name, "/b_zr"),
        function() rnn_concat(list(nn$full(u, update_bias, dtype), nn$zeros(u, dtype)), 0L))
    wxn <- store$get(paste0(name, "/Wx_n"), c(incoming, u), nn$init$uniform(sqrt(6 / (incoming + u))))
    whn <- store$get(paste0(name, "/Wh_n"), c(u, u), nn$init$uniform(sqrt(6 / (2 * u))))
    bn <- store$get(paste0(name, "/b_n"), u, nn$init$zeros())
    zr <- rnn_split(nn$sigmoid(nn$bias_add(nn$add(nn$matmul(x, wx), nn$matmul(h, wh)), b)), 2L, 1L)
    n <- nn$tanh(nn$bias_add(nn$add(nn$matmul(x, wxn), nn$matmul(nn$mul(zr[[2]], h), whn)), bn))
    peephole_lstm_gated_step(nn$add(nn$mul(nn$sub(1, zr[[1]]), n), nn$mul(zr[[1]], h)), NULL,
        rnn_concat(list(zr[[1]], zr[[2]], n), 1L))
}

gated_sequence <- function(store, inputs, lengths, initial, kind) {
    xs <- rnn_steps(inputs)
    h <- initial[[1]]
    c <- if (kind != "gru") initial[[2]] else nn$mul(initial[[1]], 0)
    states <- list(h)
    cells <- list(c)
    gates <- list()
    normed_gates <- list()
    normed_cells <- list()
    for (t in 0:11) {
        mask <- rnn_step_mask(lengths, t)
        valid <- nn$greater(mask, 0)
        x <- xs[[t + 1]]
        if (kind == "lstm") {
            step <- gated_sequence_lstm_step(store, "cell", x, 2, 12, h, c, 1)
        } else if (kind == "peephole") {
            step <- peephole_lstm_step(store, "cell", x, 2, 12, h, c)
        } else if (kind == "layer_norm") {
            step <- layer_norm_lstm_step(store, "cell", x, 2, 12, h, c)
            normed_gates[[t + 1]] <- rnn_masked_diagnostics(valid, step$normalized_gate_affine)
            normed_cells[[t + 1]] <- rnn_masked_diagnostics(valid, step$normalized_cell)
        } else {
            step <- gated_sequence_gru_step(store, "cell", x, 2, 12, h, 0)
            step$c <- c
        }
        h <- rnn_blend(mask, step$h, h)
        c <- rnn_blend(mask, step$c, c)
        states[[t + 2]] <- h
        cells[[t + 2]] <- c
        gates[[t + 1]] <- rnn_masked_diagnostics(valid, step$gates)
    }
    out <- list(logits = rnn_head(store, "head", h, 12, 2), states = rnn_stack_time(states),
        cell_states = rnn_stack_time(cells), final_state = h, final_cell = c, gates = rnn_stack_time(gates))
    if (length(normed_gates)) {
        out$normalized_gate_affine <- rnn_stack_time(normed_gates)
        out$normalized_cell <- rnn_stack_time(normed_cells)
    }
    out
}

gated_sequence_model <- function(kind, training = TRUE) {
    fields <- c("logits", "probability", "states", "final_state", "cell_states", "final_cell", "gates")
    if (kind == "layer_norm") fields <- c(fields, "normalized_gate_affine", "normalized_cell")
    model <- recurrent_model_new(27003, function(p, x, l, s) gated_sequence(p, x, l, s, kind), c(12, 2), list(12, 12),
        c("initial_gradient", if (kind != "gru") "initial_cell_gradient" else ""), fields)
    recurrent_model_build(model, training)
}
