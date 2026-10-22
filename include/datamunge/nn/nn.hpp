#pragma once

/// @file nn.hpp
/// @brief Umbrella header for `datamunge::nn`: tensors, autograd, ops,
///        convolution, layers, losses, optimizers, training utilities,
///        image ops and augmentation, and data I/O.

#include "datamunge/nn/augment.hpp"
#include "datamunge/nn/autograd.hpp"
#include "datamunge/nn/conv.hpp"
#include "datamunge/nn/convert.hpp"
#include "datamunge/nn/datasets.hpp"
#include "datamunge/nn/dtype.hpp"
#include "datamunge/nn/gradcheck.hpp"
#include "datamunge/nn/image_io.hpp"
#include "datamunge/nn/image_ops.hpp"
#include "datamunge/nn/io.hpp"
#include "datamunge/nn/layers.hpp"
#include "datamunge/nn/losses.hpp"
#include "datamunge/nn/metrics.hpp"
#include "datamunge/nn/ops.hpp"
#include "datamunge/nn/optim.hpp"
#include "datamunge/nn/parallel.hpp"
#include "datamunge/nn/params.hpp"
#include "datamunge/nn/random.hpp"
#include "datamunge/nn/schedules.hpp"
#include "datamunge/nn/tensor.hpp"
#include "datamunge/nn/training.hpp"
