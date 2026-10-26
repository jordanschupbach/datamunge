"""Explicit masked sequence unroll around the handwritten LSTM, reset-before
GRU, peephole LSTM and layer-norm LSTM cells (examples/nn/common/gated_sequence.hpp;
tflow's common/nn/gated_sequence.h and common/ml/gated_sequence.h: 12 hidden
units, 12 steps, 12->2 head, Adam .01, Params seed 27003)."""
import math

from pydatamunge import nn
from pydatamunge.nn_workflows import helper

recurrent_model = helper("recurrent_model")
rnn = helper("rnn")
peephole_lstm = helper("peephole_lstm")
layer_norm_lstm = helper("layer_norm_lstm")

# GatedCell kinds.
LSTM, GRU, PEEPHOLE, LAYER_NORM = "lstm", "gru", "peephole", "layer_norm"
GatedStep = peephole_lstm.GatedStep


def lstm_gated_step(store, name, x, incoming, units, h, c, forget_bias=1.0):
    """Plain LSTM step (nn.lstm_step's parameters and arithmetic) that also returns its gates."""
    wx = store.get(name + "/Wx", [incoming, 4 * units], nn.init.uniform(math.sqrt(6.0 / (incoming + 4 * units))))
    wh = store.get(name + "/Wh", [units, 4 * units], nn.init.uniform(math.sqrt(6.0 / (5 * units))))
    b = peephole_lstm.lstm_bias(store, name + "/b", units, forget_bias)
    parts = nn.split(nn.bias_add(nn.add(nn.matmul(x, wx), nn.matmul(h, wh)), b), 4, 1)
    i, f, g, o = nn.sigmoid(parts[0]), nn.sigmoid(parts[1]), nn.tanh(parts[2]), nn.sigmoid(parts[3])
    cell = nn.add(nn.mul(f, c), nn.mul(i, g))
    return GatedStep(nn.mul(o, nn.tanh(cell)), cell, nn.concat([i, f, g, o], 1))


def gru_gated_step(store, name, x, incoming, units, h, update_bias=0.0):
    """Reset-before GRU step (nn.gru_step's parameters and arithmetic) that also returns concat(z, r, candidate)."""
    u = units
    wx = store.get(name + "/Wx_zr", [incoming, 2 * u], nn.init.uniform(math.sqrt(6.0 / (incoming + 2 * u))))
    wh = store.get(name + "/Wh_zr", [u, 2 * u], nn.init.uniform(math.sqrt(6.0 / (3 * u))))
    dtype = store.dtype()
    b = rnn.fixed_parameter(store, name + "/b_zr",
                            lambda: nn.concat([nn.full([u], update_bias, dtype), nn.zeros([u], dtype)], 0))
    wxn = store.get(name + "/Wx_n", [incoming, u], nn.init.uniform(math.sqrt(6.0 / (incoming + u))))
    whn = store.get(name + "/Wh_n", [u, u], nn.init.uniform(math.sqrt(6.0 / (2 * u))))
    bn = store.get(name + "/b_n", [u], nn.init.zeros())
    zr = nn.split(nn.sigmoid(nn.bias_add(nn.add(nn.matmul(x, wx), nn.matmul(h, wh)), b)), 2, 1)
    n = nn.tanh(nn.bias_add(nn.add(nn.matmul(x, wxn), nn.matmul(nn.mul(zr[1], h), whn)), bn))
    return GatedStep(nn.add(nn.mul(nn.sub(1.0, zr[0]), n), nn.mul(zr[0], h)), None, nn.concat([zr[0], zr[1], n], 1))


def gated_sequence(store, inputs, lengths, initial, kind):
    xs = nn.time_steps(inputs)
    h = initial[0]
    c = initial[1] if kind != GRU else nn.mul(initial[0], 0.0)
    states, cells, gates, normed_gates, normed_cells = [h], [c], [], [], []
    for t in range(12):
        mask = rnn.step_mask(lengths, t)
        valid = nn.greater(mask, 0.0)
        if kind == LSTM:
            step = lstm_gated_step(store, "cell", xs[t], 2, 12, h, c, 1.0)
        elif kind == PEEPHOLE:
            step = peephole_lstm.peephole_lstm_step(store, "cell", xs[t], 2, 12, h, c)
        elif kind == LAYER_NORM:
            step = layer_norm_lstm.layer_norm_lstm_step(store, "cell", xs[t], 2, 12, h, c)
            normed_gates.append(rnn.masked_diagnostics(valid, step.normalized_gate_affine))
            normed_cells.append(rnn.masked_diagnostics(valid, step.normalized_cell))
        else:
            step = gru_gated_step(store, "cell", xs[t], 2, 12, h, 0.0)
            step.c = c
        h = rnn.blend(mask, step.h, h)
        c = rnn.blend(mask, step.c, c)
        states.append(h)
        cells.append(c)
        gates.append(rnn.masked_diagnostics(valid, step.gates))
    out = {"logits": rnn.head(store, "head", h, 12, 2), "states": nn.stack_time(states),
           "cell_states": nn.stack_time(cells), "final_state": h, "final_cell": c, "gates": nn.stack_time(gates)}
    if normed_gates:
        out["normalized_gate_affine"] = nn.stack_time(normed_gates)
        out["normalized_cell"] = nn.stack_time(normed_cells)
    return out


def gated_sequence_model(kind, training=True):
    fields = ["logits", "probability", "states", "final_state", "cell_states", "final_cell", "gates"]
    if kind == LAYER_NORM:
        fields += ["normalized_gate_affine", "normalized_cell"]
    model = recurrent_model.RecurrentModel(
        27003, lambda p, x, l, s: gated_sequence(p, x, l, s, kind), [12, 2], [[12], [12]],
        ["initial_gradient", "initial_cell_gradient" if kind != GRU else ""], fields)
    return model.build(training)
