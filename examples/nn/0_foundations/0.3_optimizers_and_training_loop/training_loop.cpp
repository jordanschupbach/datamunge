// CMD: just run-nn training_loop
// Epochs, reproducible shuffling, short batches and sample-weighted logging.
//
// Port of tflow's training_loop. Train a two-feature affine regression model
// on 103 examples with batches of 16. Every epoch visits every row exactly
// once, including its short final batch. Take the loss used by Adam in the
// same step, then weight batch means by their sample counts when reporting
// the epoch's online training loss. This online metric uses changing weights;
// final evaluation uses fixed ones.
//
// Save a sampler midway through an epoch and restore into a new instance.
// Verify the remaining rows and three future shuffles match exactly. Model
// and optimizer checkpoints are separate (see early_stopping_checkpoints);
// save both between updates when resuming a complete training job.
//
// Differences from TF: no graph or session; the loss is computed eagerly and
// Optimizer::minimize applies the update. nn::BatchOrder uses the same
// mt19937_64 stream as tflow's, so every batch (and the trained weights) match
// TF. Row indices are int64 and nn::take_rows gathers them.

#include <cstdint>
#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <set>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::Tensor;
using Rows = std::vector<std::int64_t>;

namespace {

void BatchingChecks() {
  nn::BatchOrder original(11, 4, 7), restored(11, 4, 999);
  original.begin_epoch();
  const auto first = original.next();
  restored.restore_state(original.save_state());
  std::set<std::int64_t> seen(first.begin(), first.end());
  for (;;) {
    const auto batch = original.next();
    check::Require(batch == restored.next(), "restored remainder must match");
    if (batch.empty()) break;
    seen.insert(batch.begin(), batch.end());
  }
  check::Require(seen.size() == 11, "each epoch covers every sample");
  for (int epoch = 0; epoch < 3; ++epoch) {
    original.begin_epoch();
    restored.begin_epoch();
    do {
      const auto batch = original.next();
      check::Require(batch == restored.next(), "future shuffles must match");
      if (batch.empty()) break;
    } while (true);
  }
  const auto state    = restored.save_state();
  bool       rejected = false;
  try {
    restored.restore_state(state + " bad");
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check::Require(rejected && restored.save_state() == state, "invalid state must not change sampler");
  nn::BatchOrder drop(10, 4, 0, false, true);
  drop.begin_epoch();
  check::Require(drop.next() == Rows({0, 1, 2, 3}) && drop.next() == Rows({4, 5, 6, 7}) && drop.next().empty(),
                 "drop_last semantics");
  nn::BatchOrder singleton(1, 16, 0);
  singleton.begin_epoch();
  check::Require(singleton.next().size() == 1 && singleton.next().empty(), "dataset smaller than batch");
  const Tensor source   = nn::arange(0, 6, 1, nn::DType::Int32).reshape({3, 2});
  const auto   gathered = nn::take_rows(source, {2, 0, 2}).to_vector<int>();
  check::Require(gathered == std::vector<int>({4, 5, 0, 1, 4, 5}), "row gathering preserves values");
  rejected = false;
  try {
    (void)nn::take_rows(source, {3});
  } catch (const std::exception&) {
    rejected = true;
  }
  check::Require(rejected, "out-of-range row must fail");
  nn::WeightedMean mean;
  mean.add(2, 4);
  mean.add(10, 1);
  check::Near(mean.value(), 3.6, 1e-12, "short batches need sample-weighted means");
  std::cout << "  exact mid-epoch resume and three future shuffles passed\n"
               "  coverage, drop_last, singleton, row gathering and invalid-state checks passed\n"
               "  batch means 2 (4 samples), 10 (1 sample): epoch mean = 3.6\n";
}

void Train(int epochs) {
  std::vector<float> xv(103 * 2), yv(103);
  for (int i = 0; i < 103; ++i) {
    const float a = std::sin(i * .7f), b = std::cos(i * .4f);
    xv[2 * i]     = a;
    xv[2 * i + 1] = b;
    yv[i]         = 1.5f * a - .7f * b + .2f;
  }
  const Tensor inputs = nn::from_vector(xv, {103, 2}), labels = nn::from_vector(yv, {103, 1});
  Tensor       w = nn::zeros({2, 1}).requires_grad_(), b = nn::zeros({1}).requires_grad_();
  auto         loss_of = [&](const Tensor& x, const Tensor& y) {
    return nn::mean(nn::square(nn::sub(nn::bias_add(nn::matmul(x, w), b), y)));
  };
  nn::OptimizerConfig cfg;
  cfg.kind = nn::OptimizerKind::Adam;
  nn::Optimizer  optimizer({w, b}, .03, cfg);
  nn::BatchOrder sampler(103, 16, 42);
  const double initial_loss = loss_of(inputs, labels).item();
  for (int epoch = 0; epoch < epochs; ++epoch) {
    sampler.begin_epoch();
    nn::WeightedMean       epoch_loss;
    std::set<std::int64_t> visited;
    std::size_t            batches = 0;
    for (auto rows = sampler.next(); !rows.empty(); rows = sampler.next()) {
      for (auto index : rows) check::Require(visited.insert(index).second, "duplicate row within epoch");
      const Tensor loss = loss_of(nn::take_rows(inputs, rows), nn::take_rows(labels, rows));
      optimizer.minimize(loss);
      epoch_loss.add(loss.item(), rows.size());
      ++batches;
    }
    check::Require(visited.size() == 103 && epoch_loss.count() == 103 && batches == 7,
                   "complete epoch including final 7 rows");
    if (epoch == 0 || (epoch + 1) % 10 == 0 || epoch + 1 == epochs) nn::log_epoch(std::cout, epoch + 1, batches, epoch_loss);
  }
  const double final_loss = loss_of(inputs, labels).item();
  const auto   wv = w.to_vector<double>();
  const double bv = b.item();
  check::Require(std::isfinite(final_loss) && final_loss < initial_loss, "minibatch regression must reduce loss");
  if (epochs >= 50) {
    check::Require(final_loss < 1e-9, "minibatch regression must converge");
    check::Near(wv[0], 1.5, 1e-4, "first slope");
    check::Near(wv[1], -.7, 1e-4, "second slope");
    check::Near(bv, .2, 1e-4, "intercept");
  }
  check::Require(optimizer.iterations() == 7 * epochs, "one optimizer update per batch");
  std::cout << "  final fixed-weight MSE=" << static_cast<float>(final_loss) << ", weights=(" << static_cast<float>(wv[0])
            << ", " << static_cast<float>(wv[1]) << "), bias=" << static_cast<float>(bv)
            << ", updates=" << optimizer.iterations() << '\n';
}

}  // namespace

int main(int argc, char** argv) {
  try {
    int epochs = 50;
    for (int i = 1; i < argc; ++i) {
      const std::string argument(argv[i]);
      if (argument.rfind("epochs=", 0) == 0) epochs = std::stoi(argument.substr(7));
    }
    check::Require(epochs > 0, "positive training budget required");
    nn::set_num_threads(2);
    std::cout << "== Reproducible batch ordering and weighted metrics\n";
    BatchingChecks();
    std::cout << "\n== Training 103 samples in batches of 16\n";
    Train(epochs);
    std::cout << "All training-loop checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
