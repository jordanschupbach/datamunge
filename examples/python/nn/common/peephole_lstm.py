"""Diagonal cell-to-gate peepholes: old cell for input/forget, NEW cell for
output (examples/nn/common/peephole_lstm.hpp; tflow's common/nn/peephole_lstm.h)."""
from pydatamunge import nn
from pydatamunge.nn_workflows import helper

rnn = helper("rnn")


class GatedStep:
    """One cell step: the next (h, c), concat(i, f, g, o) (GRU: concat(z, r,
    candidate)) and, for the layer-norm LSTM, its normalized diagnostics."""

    def __init__(self, h, c, gates, normalized_gate_affine=None, normalized_cell=None):
        self.h, self.c, self.gates = h, c, gates
        self.normalized_gate_affine, self.normalized_cell = normalized_gate_affine, normalized_cell


def lstm_bias_value(units, forget_bias=1.0, dtype=nn.DType.Float32):
    """The LSTM gate bias: zeros except the forget block, which starts at forget_bias."""
    return nn.concat([nn.zeros([units], dtype), nn.full([units], forget_bias, dtype), nn.zeros([2 * units], dtype)], 0)


def lstm_bias(store, name, units, forget_bias=1.0):
    """The "<name>" LSTM bias parameter (C++ LstmBias initializer)."""
    return rnn.fixed_parameter(store, name, lambda: lstm_bias_value(units, forget_bias, store.dtype()))


def peephole_lstm_step(store, name, x, incoming, units, h, c):
    wx = store.get(name + "/Wx", [incoming, 4 * units], nn.init.uniform(rnn.float32_sqrt_ratio(6, incoming + 4 * units)))
    wh = store.get(name + "/Wh", [units, 4 * units], nn.init.uniform(rnn.float32_sqrt_ratio(6, 5 * units)))
    b = lstm_bias(store, name + "/b", units)
    pi = store.get(name + "/peephole_i", [units], nn.init.normal(rnn.float32(0.05)))
    pf = store.get(name + "/peephole_f", [units], nn.init.normal(rnn.float32(0.05)))
    po = store.get(name + "/peephole_o", [units], nn.init.normal(rnn.float32(0.05)))
    parts = nn.split(nn.bias_add(nn.add(nn.matmul(x, wx), nn.matmul(h, wh)), b), 4, 1)
    i = nn.sigmoid(nn.add(parts[0], nn.mul(c, pi)))
    f = nn.sigmoid(nn.add(parts[1], nn.mul(c, pf)))
    g = nn.tanh(parts[2])
    cell = nn.add(nn.mul(f, c), nn.mul(i, g))
    o = nn.sigmoid(nn.add(parts[3], nn.mul(cell, po)))
    return GatedStep(nn.mul(o, nn.tanh(cell)), cell, nn.concat([i, f, g, o], 1))
