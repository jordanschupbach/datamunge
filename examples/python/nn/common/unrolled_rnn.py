"""Explicit twelve-step shared-weight tanh recurrence with valid-length state
masking (examples/nn/common/unrolled_rnn.hpp; 206 parameters)."""
from pydatamunge import nn
from pydatamunge.nn_workflows import helper

recurrent_model = helper("recurrent_model")
rnn = helper("rnn")


def unrolled_rnn(store, inputs, lengths, initial):
    steps = nn.time_steps(inputs)
    h = initial[0]
    states = [h]
    for t in range(12):
        h = rnn.blend(rnn.step_mask(lengths, t), nn.rnn_step(store, "cell", steps[t], h), h)
        states.append(h)
    return {"logits": rnn.head(store, "head", h, 12, 2), "states": nn.stack_time(states), "final_state": h}


def unrolled_rnn_model(training=True, seed=26003):
    """tflow's ml::UnrolledRnnModel (vanilla_rnn): 206 parameters, Adam .01, seed 26003."""
    model = recurrent_model.RecurrentModel(seed, unrolled_rnn, [12, 2], [[12]], ["initial_gradient"],
                                           ["logits", "probability", "states", "final_state"])
    return model.build(training)
