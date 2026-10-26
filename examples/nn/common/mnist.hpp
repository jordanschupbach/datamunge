// MNIST evaluation helpers for the example-local classifier workflows: a port of
// tflow's common/ml/mnist.h (RunMnist) and the parts of common/ml/
// classification.h it uses.
//
// The official 60,000 training rows are split once with seed 2026 (the same
// std::mt19937_64 shuffle as tflow, so the split is identical): 55,000 for
// fitting, 5,000 for validation. The official 10,000 test rows are evaluated
// once, after selecting the smallest validation cross entropy. No test labels
// influence updates or checkpoint selection. Pixels map from uint8 to [0,1].
// Every row is used, including short final batches; metrics weight batch
// means by row count. Mini-batches come from nn::BatchOrder(55000, batch, 29),
// which reproduces tflow's batch order exactly.
//
// The model is Adam on mean sparse cross-entropy plus 0.5 * l2 * sum(W^2)
// over parameters of rank >= 2 (tflow Classifier, l2 = 1e-4). Parameter
// initialisation uses datamunge's generator (seed 17), so trajectories match
// tflow only statistically. Checkpoints are .npz files of the parameter
// store (nn::ParameterStore::save/load) instead of TF SaveV2 bundles.
// A custom LogitsBuilder (e.g. a CNN) receives flattened [batch, 784] pixels
// and reshapes them itself, as tflow's architecture lambdas do. threads=N
// sets nn's intra-op thread count (default: all cores).
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <datamunge/nn/nn.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "common/checks.hpp"
#include "common/paths.hpp"

namespace mnist {

namespace nn = datamunge::nn;

/// Builds [batch, 10] logits from flattened [batch, 784] pixels.
using LogitsBuilder = std::function<nn::Tensor(nn::ParameterStore&, const nn::Tensor& x, bool training)>;

struct Metrics {
  nn::WeightedMean    loss, accuracy;
  nn::ConfusionMatrix confusion{10};
};

/// Cross-entropy, accuracy and confusion of fixed parameters over `rows`, in batches of 512.
inline Metrics Evaluate(nn::ParameterStore& store, const LogitsBuilder& logits_of, const nn::Tensor& images,
                        const nn::Tensor& labels, const std::vector<std::int64_t>& rows) {
  nn::NoGradGuard guard;
  Metrics         result;
  for (std::size_t start = 0; start < rows.size(); start += 512) {
    const std::size_t               end = std::min(rows.size(), start + 512);
    const std::vector<std::int64_t> ids(rows.begin() + static_cast<std::ptrdiff_t>(start),
                                        rows.begin() + static_cast<std::ptrdiff_t>(end));
    const nn::Tensor x = nn::take_rows(images, ids), y = nn::take_rows(labels, ids);
    const nn::Tensor logits     = logits_of(store, x, false);
    const nn::Tensor prediction = nn::argmax(logits, 1);
    result.loss.add(nn::sparse_softmax_cross_entropy(logits, y).item(), end - start);
    result.accuracy.add(nn::accuracy(prediction, y), end - start);
    result.confusion.add(y, prediction);
  }
  return result;
}

}  // namespace mnist
