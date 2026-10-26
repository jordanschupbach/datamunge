"""Per-example, per-gate layer normalization; the raw cell is carried and the
normalized cell is used ONLY for the hidden readout
(examples/nn/common/layer_norm_lstm.hpp; tflow's common/nn/layer_norm_lstm.h)."""
from pydatamunge import nn
from pydatamunge.nn_workflows import helper

rnn = helper("rnn")
peephole_lstm = helper("peephole_lstm")


def lstm_layer_normalize(x, axis):
    """(x - mean) * rsqrt(var + 1e-5) along `axis` (no gain or bias)."""
    centered = nn.sub(x, nn.mean(x, [axis], True))
    variance = nn.mean(nn.square(centered), [axis], True)
    return nn.mul(centered, nn.rsqrt(nn.add(variance, rnn.float32(1e-5))))


def layer_norm_lstm_step(store, name, x, incoming, units, h, c):
    wx = store.get(name + "/Wx", [incoming, 4 * units], nn.init.uniform(rnn.float32_sqrt_ratio(6, incoming + 4 * units)))
    wh = store.get(name + "/Wh", [units, 4 * units], nn.init.uniform(rnn.float32_sqrt_ratio(6, 5 * units)))
    bias = peephole_lstm.lstm_bias(store, name + "/b", units)
    gate_gamma = store.get(name + "/gate_gamma", [4 * units], nn.init.ones())
    cell_gamma = store.get(name + "/cell_gamma", [units], nn.init.ones())
    cell_beta = store.get(name + "/cell_beta", [units], nn.init.zeros())
    affine = nn.add(nn.matmul(x, wx), nn.matmul(h, wh))
    normalized = nn.reshape(lstm_layer_normalize(nn.reshape(affine, [-1, 4, units]), 2), [-1, 4 * units])
    parts = nn.split(nn.add(nn.mul(normalized, gate_gamma), bias), 4, 1)
    i, f, g, o = nn.sigmoid(parts[0]), nn.sigmoid(parts[1]), nn.tanh(parts[2]), nn.sigmoid(parts[3])
    cell = nn.add(nn.mul(f, c), nn.mul(i, g))
    normalized_cell = nn.add(nn.mul(lstm_layer_normalize(cell, 1), cell_gamma), cell_beta)
    return peephole_lstm.GatedStep(nn.mul(o, nn.tanh(normalized_cell)), cell, nn.concat([i, f, g, o], 1),
                                   normalized, normalized_cell)
