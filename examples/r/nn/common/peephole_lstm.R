# Diagonal cell-to-gate peepholes: old cell for input/forget, NEW cell for
# output (examples/nn/common/peephole_lstm.hpp; tflow's common/nn/peephole_lstm.h).
if (!exists("rnn_blend", inherits = TRUE)) nn_helper("rnn")

# One cell step: the next h and c, concat(i, f, g, o) (GRU: concat(z, r,
# candidate)) and, for the layer-norm LSTM, its normalized diagnostics.
peephole_lstm_gated_step <- function(h, c, gates, normalized_gate_affine = NULL, normalized_cell = NULL)
    list(h = h, c = c, gates = gates, normalized_gate_affine = normalized_gate_affine, normalized_cell = normalized_cell)

# The LSTM gate bias: zeros except the forget block, which starts at forget_bias.
peephole_lstm_bias_value <- function(units, forget_bias = 1, dtype = nn$DType$Float32)
    rnn_concat(list(nn$zeros(units, dtype), nn$full(units, forget_bias, dtype), nn$zeros(2 * units, dtype)), 0L)

# The "<name>" LSTM bias parameter (C++ LstmBias initializer).
peephole_lstm_bias <- function(store, name, units, forget_bias = 1)
    rnn_fixed_parameter(store, name, function() peephole_lstm_bias_value(units, forget_bias, store$dtype()))

peephole_lstm_step <- function(store, name, x, incoming, units, h, c) {
    wx <- store$get(paste0(name, "/Wx"), c(incoming, 4 * units), nn$init$uniform(rnn_float32_sqrt_ratio(6, incoming + 4 * units)))
    wh <- store$get(paste0(name, "/Wh"), c(units, 4 * units), nn$init$uniform(rnn_float32_sqrt_ratio(6, 5 * units)))
    b <- peephole_lstm_bias(store, paste0(name, "/b"), units)
    pi <- store$get(paste0(name, "/peephole_i"), units, nn$init$normal(rnn_float32(0.05)))
    pf <- store$get(paste0(name, "/peephole_f"), units, nn$init$normal(rnn_float32(0.05)))
    po <- store$get(paste0(name, "/peephole_o"), units, nn$init$normal(rnn_float32(0.05)))
    parts <- rnn_split(nn$bias_add(nn$add(nn$matmul(x, wx), nn$matmul(h, wh)), b), 4L, 1L)
    i <- nn$sigmoid(nn$add(parts[[1]], nn$mul(c, pi)))
    f <- nn$sigmoid(nn$add(parts[[2]], nn$mul(c, pf)))
    g <- nn$tanh(parts[[3]])
    cell <- nn$add(nn$mul(f, c), nn$mul(i, g))
    o <- nn$sigmoid(nn$add(parts[[4]], nn$mul(cell, po)))
    peephole_lstm_gated_step(nn$mul(o, nn$tanh(cell)), cell, rnn_concat(list(i, f, g, o), 1L))
}
