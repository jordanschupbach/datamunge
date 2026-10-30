"""Shared NN example components (examples/nn/common/*.hpp) through SWIG.

The C++ examples own their training loops; the shared headers supply model
forward functions, data readers, metrics and checks. This module regroups the
flattened ``<namespace>__Name`` symbols of ``nn_native`` by C++ namespace, so
``image_training::ImageBatch`` is ``image_training.ImageBatch`` here and the
Python examples can run the same loop as the C++ ones over the same code.
Callback-typed C++ functions are not bound; see examples/bindings/nn_components.i.
"""
import functools
import sys
import types
from types import SimpleNamespace

from . import nn_native as _native

NAMESPACES = ("example_runtime", "paths", "dataset_assets", "check", "image_training", "resnet", "resnet_demo", "architecture",
              "mobile_checks", "nn_components", "grouped", "lrn", "inception", "normalized_conv", "deconvnet", "alexnet",
              "googlenet", "inception23", "inception4", "nin", "vgg", "xception", "zfnet", "preact", "densenet", "dpn",
              "pyramidnet", "res2net", "resnext", "revnet", "stochastic_depth", "wide_resnet", "mobilenet", "mbconv",
              "condensenet", "ghostnet", "regnet", "shufflenet", "squeezenet", "convnext", "deformable", "non_local",
              "octconv", "repvgg", "attention_cnn", "mnist", "cnn_baselines", "alexnet_checks", "xception_checks", "densenet_checks", "ghostnet_checks", "mobilenet_v2_checks", "mnasnet_checks", "convnext_checks", "efficientnet_checks", "efficientnet_v2_checks", "preactivation_resnet_checks", "pyramidnet_checks", "res2net_checks", "resnext_checks", "revnet_checks", "cbam_checks", "non_local_checks", "octconv_checks", "repvgg_checks", "sknet_checks", "ae", "memory_components",
              "seg", "fcn", "unet", "segnet", "hrnet", "pspnet", "deeplab", "sentence", "replay", "series", "sensor", "charcnn", "tcn", "wavestack", "sal", "dream", "featurevis", "cam", "occlusion", "ig", "smooth", "guided", "detector", "fpn", "retinanet", "fcos", "ssd", "centernet", "detr", "hungarian", "efficientdet", "bifpn", "yolo", "wordtree", "maskrcnn", "instance", "panoptic", "yolact", "detection_evaluation", "detection", "highway", "stn", "capsnet", "im2col", "receptive")


def _flushing(function):
    """Flush Python's buffered stdout before a C++ call so std::cout output stays in order."""
    @functools.wraps(function)
    def call(*arguments, **keywords):
        sys.stdout.flush()
        return function(*arguments, **keywords)
    return call


def _namespace(prefix):
    marker = prefix + "__"
    members = {}
    for name in dir(_native):
        if not name.startswith(marker) or name.endswith("_swigregister"):
            continue
        value = getattr(_native, name)
        members[name[len(marker):]] = _flushing(value) if isinstance(value, types.FunctionType) else value
    if not members:
        raise ImportError("the installed pydatamunge binding has no " + prefix + " components; rebuild it")
    return SimpleNamespace(**members)


for _name in NAMESPACES:
    globals()[_name] = _namespace(_name)

__all__ = list(NAMESPACES)
