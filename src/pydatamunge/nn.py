"""Functional SWIG bindings to datamunge::nn; models and loops live in Python."""
from contextlib import contextmanager
from types import SimpleNamespace

from .nn_native import *
from . import nn_native as _native

DType = SimpleNamespace(**{name: getattr(_native, 'DType_' + name) for name in
                          ('Bool', 'Int32', 'Int64', 'Float32', 'Float64')})
Activation = SimpleNamespace(**{name: getattr(_native, 'Activation_' + name) for name in
    ('Relu', 'Relu6', 'LeakyRelu', 'Elu', 'Selu', 'Gelu', 'GeluTanh', 'Silu', 'Mish',
     'Softplus', 'Softsign', 'Sigmoid', 'Tanh')}, Identity=_native.Activation__None)
Padding = SimpleNamespace(**{name: getattr(_native, 'Padding_' + name) for name in ('Same', 'Valid', 'Explicit')})
OptimizerKind = SimpleNamespace(**{name: getattr(_native, 'OptimizerKind_' + name) for name in
    ('Sgd', 'Momentum', 'Nesterov', 'Adagrad', 'RMSProp', 'Adam', 'AdamW', 'Nadam',
     'AMSGrad', 'Adamax', 'Lamb', 'Lars', 'Lion', 'Adafactor', 'Adadelta')})
init = SimpleNamespace(**{name[5:]: getattr(_native, name) for name in dir(_native) if name.startswith('init_')})
schedule = SimpleNamespace(**{name[9:]: getattr(_native, name) for name in dir(_native) if name.startswith('schedule_')})
AttentionImpl = SimpleNamespace(**{name: getattr(_native, 'AttentionImpl_' + name)
                                  for name in ('Auto', 'Plain', 'Fused')})
RotaryLayout = SimpleNamespace(**{name: getattr(_native, 'RotaryLayout_' + name)
                                for name in ('Interleaved', 'HalfSplit')})


def _enum(prefix):
    return SimpleNamespace(**{name[len(prefix) + 1:]: getattr(_native, name) for name in dir(_native)
                              if name.startswith(prefix + '_') and not callable(getattr(_native, name))})


for _kind in ('PadMode', 'ResizeMethod', 'DepthToSpaceMode', 'GridSampleMode', 'GridSamplePadding', 'WavFormat',
              'Reduction'):
    if _kind not in globals() or not isinstance(globals()[_kind], SimpleNamespace):
        globals()[_kind] = _enum(_kind)
augment = SimpleNamespace(**{name[8:]: getattr(_native, name) for name in dir(_native) if name.startswith('augment_')})


@contextmanager
def no_grad():
    previous = is_grad_enabled()
    set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(previous)
