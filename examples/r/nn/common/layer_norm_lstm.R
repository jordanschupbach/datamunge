# Per-example, per-gate layer normalization; the raw cell is carried and the
# normalized cell is used ONLY for the hidden readout
# (examples/nn/common/layer_norm_lstm.hpp; tflow's common/nn/layer_norm_lstm.h).
if (!exists("peephole_lstm_step", inherits = TRUE)) nn_helper("peephole_lstm")

# (x - mean) * rsqrt(var + 1e-5) along `axis` (no gain or bias).
layer_norm_lstm_normalize <- function(x, axis) {
    centered <- nn$sub(x, nn$mean(x, axis, TRUE))
    variance <- nn$mean(nn$square(centered), axis, TRUE)
    nn$mul(centered, nn$rsqrt(nn$add(variance, rnn_float32(1e-5))))
}

layer_norm_lstm_step <- function(store, name, x, incoming, units, h, c) {
    wx <- store$get(paste0(name, "/Wx"), c(incoming, 4 * units), nn$init$uniform(rnn_float32_sqrt_ratio(6, incoming + 4 * units)))
    wh <- store$get(paste0(name, "/Wh"), c(units, 4 * units), nn$init$uniform(rnn_float32_sqrt_ratio(6, 5 * units)))
    bias <- peephole_lstm_bias(store, paste0(name, "/b"), units)
    gate_gamma <- store$get(paste0(name, "/gate_gamma"), 4 * units, nn$init$ones())
    cell_gamma <- store$get(paste0(name, "/cell_gamma"), units, nn$init$ones())
    cell_beta <- store$get(paste0(name, "/cell_beta"), units, nn$init$zeros())
    affine <- nn$add(nn$matmul(x, wx), nn$matmul(h, wh))
    normalized <- nn$reshape(layer_norm_lstm_normalize(nn$reshape(affine, c(-1, 4, units)), 2L), c(-1, 4 * units))
    parts <- rnn_split(nn$add(nn$mul(normalized, gate_gamma), bias), 4L, 1L)
    i <- nn$sigmoid(parts[[1]])
    f <- nn$sigmoid(parts[[2]])
    g <- nn$tanh(parts[[3]])
    o <- nn$sigmoid(parts[[4]])
    cell <- nn$add(nn$mul(f, c), nn$mul(i, g))
    normalized_cell <- nn$add(nn$mul(layer_norm_lstm_normalize(cell, 1L), cell_gamma), cell_beta)
    peephole_lstm_gated_step(nn$mul(o, nn$tanh(normalized_cell)), cell, rnn_concat(list(i, f, g, o), 1L), normalized,
        normalized_cell)
}
