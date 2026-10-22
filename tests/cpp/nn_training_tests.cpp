// Tests for datamunge::nn learning-rate schedules, training helpers and metrics.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/nn/nn.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <vector>

namespace nn       = datamunge::nn;
namespace schedule = datamunge::nn::schedule;
using nn::DType;
using nn::Tensor;

namespace {

/// tflow's BatchOrder reference: iota then std::shuffle with mt19937_64 each epoch.
std::vector<std::vector<std::size_t>> reference_epochs(std::size_t n, std::uint64_t seed, int epochs) {
  std::mt19937_64                       rng(seed);
  std::vector<std::vector<std::size_t>> out;
  std::vector<std::size_t>              order(n);
  for (int e = 0; e < epochs; ++e) {
    std::iota(order.begin(), order.end(), std::size_t{0});
    std::shuffle(order.begin(), order.end(), rng);
    out.push_back(order);
  }
  return out;
}

std::vector<std::int64_t> concat_epoch(nn::BatchOrder& order) {
  std::vector<std::int64_t> all;
  for (auto b = order.next(); !b.empty(); b = order.next()) all.insert(all.end(), b.begin(), b.end());
  return all;
}

}  // namespace

// ---- schedules -----------------------------------------------------------------------

TEST(NnSchedules, StepAndExponential) {
  EXPECT_DOUBLE_EQ(schedule::step_decay(1.0, 0.5, 10, 0), 1.0);
  EXPECT_DOUBLE_EQ(schedule::step_decay(1.0, 0.5, 10, 9), 1.0);
  EXPECT_DOUBLE_EQ(schedule::step_decay(1.0, 0.5, 10, 10), 0.5);
  EXPECT_DOUBLE_EQ(schedule::step_decay(1.0, 0.5, 10, 25), 0.25);
  EXPECT_DOUBLE_EQ(schedule::exponential(1.0, 0.5, 10, 10), 0.5);
  EXPECT_NEAR(schedule::exponential(1.0, 0.5, 10, 5), std::sqrt(0.5), 1e-15);
  EXPECT_NEAR(schedule::exponential(2.0, 0.5, 10, 25), 2.0 * std::pow(0.5, 2.5), 1e-15);
}

TEST(NnSchedules, CosineAndWarmup) {
  EXPECT_DOUBLE_EQ(schedule::cosine(1.0, 0.0, 10, 0), 1.0);
  EXPECT_NEAR(schedule::cosine(1.0, 0.0, 10, 5), 0.5, 1e-15);
  EXPECT_NEAR(schedule::cosine(1.0, 0.2, 10, 10), 0.2, 1e-15);
  EXPECT_NEAR(schedule::cosine(1.0, 0.2, 10, 50), 0.2, 1e-15);  // constant after the end
  EXPECT_DOUBLE_EQ(schedule::warmup(2.0, 4, 1), 0.5);
  EXPECT_DOUBLE_EQ(schedule::warmup(2.0, 4, 8), 2.0);
  EXPECT_DOUBLE_EQ(schedule::warmup(2.0, 0, 0), 2.0);
  EXPECT_DOUBLE_EQ(schedule::warmup_cosine(1.0, 0.0, 10, 30, 5), 0.5);
  EXPECT_DOUBLE_EQ(schedule::warmup_cosine(1.0, 0.0, 10, 30, 10), 1.0);
  EXPECT_NEAR(schedule::warmup_cosine(1.0, 0.0, 10, 30, 20), 0.5, 1e-15);
}

TEST(NnSchedules, OneCycleAndCyclical) {
  EXPECT_DOUBLE_EQ(schedule::one_cycle(0.1, 1.0, 0.01, 30, 100, 0), 0.1);
  EXPECT_DOUBLE_EQ(schedule::one_cycle(0.1, 1.0, 0.01, 30, 100, 30), 1.0);  // peak at the end of the rise
  EXPECT_NEAR(schedule::one_cycle(0.1, 1.0, 0.01, 30, 100, 15), 0.55, 1e-12);
  EXPECT_NEAR(schedule::one_cycle(0.1, 1.0, 0.01, 30, 100, 100), 0.01, 1e-12);
  EXPECT_DOUBLE_EQ(schedule::cyclical(0.0, 1.0, 10, 0), 0.0);
  EXPECT_DOUBLE_EQ(schedule::cyclical(0.0, 1.0, 10, 5), 0.5);
  EXPECT_DOUBLE_EQ(schedule::cyclical(0.0, 1.0, 10, 10), 1.0);
  EXPECT_DOUBLE_EQ(schedule::cyclical(0.0, 1.0, 10, 15), 0.5);
  EXPECT_DOUBLE_EQ(schedule::cyclical(0.0, 1.0, 10, 20), 0.0);
  EXPECT_DOUBLE_EQ(schedule::cyclical(0.1, 0.5, 10, 10), 0.5);
  // Amplitude decays by the factor once per full cycle.
  EXPECT_DOUBLE_EQ(schedule::cyclical(0.0, 1.0, 10, 30, 0.5), 0.5);
  EXPECT_DOUBLE_EQ(schedule::cyclical(0.0, 1.0, 10, 50, 0.5), 0.25);
}

TEST(NnSchedules, Validation) {
  EXPECT_THROW(schedule::step_decay(1, 0.5, 10, -1), std::invalid_argument);
  EXPECT_THROW(schedule::step_decay(-1, 0.5, 10, 0), std::invalid_argument);
  EXPECT_THROW(schedule::step_decay(1, 0, 10, 0), std::invalid_argument);
  EXPECT_THROW(schedule::step_decay(1, 1.5, 10, 0), std::invalid_argument);
  EXPECT_THROW(schedule::step_decay(1, 0.5, 0, 0), std::invalid_argument);
  EXPECT_THROW(schedule::exponential(1, 0.5, 0, 0), std::invalid_argument);
  EXPECT_THROW(schedule::exponential(NAN, 0.5, 1, 0), std::invalid_argument);
  EXPECT_THROW(schedule::cosine(1, 0, 0, 0), std::invalid_argument);
  EXPECT_THROW(schedule::cosine(1, -1, 10, 0), std::invalid_argument);
  EXPECT_THROW(schedule::warmup(1, -1, 0), std::invalid_argument);
  EXPECT_THROW(schedule::warmup_cosine(1, 0, 10, 10, 0), std::invalid_argument);
  EXPECT_THROW(schedule::warmup_cosine(1, 0, -1, 10, 0), std::invalid_argument);
  EXPECT_THROW(schedule::one_cycle(0.1, 1, 0.01, 0, 100, 0), std::invalid_argument);
  EXPECT_THROW(schedule::one_cycle(0.1, 1, 0.01, 100, 100, 0), std::invalid_argument);
  EXPECT_THROW(schedule::one_cycle(0.1, 0.05, 0.01, 10, 100, 0), std::invalid_argument);
  EXPECT_THROW(schedule::cyclical(1, 0.5, 10, 0), std::invalid_argument);
  EXPECT_THROW(schedule::cyclical(0, 1, 0, 0), std::invalid_argument);
  EXPECT_THROW(schedule::cyclical(0, 1, 10, 0, 0), std::invalid_argument);
  EXPECT_THROW(schedule::cyclical(0, 1, 10, 0, 1.5), std::invalid_argument);
}

// ---- training helpers ------------------------------------------------------------------

TEST(NnTraining, BatchOrderMatchesReferenceShuffle) {
  const std::size_t n     = 23;
  const auto        ref   = reference_epochs(n, 42, 3);
  nn::BatchOrder    order(n, 5, 42);
  EXPECT_EQ(order.batches_per_epoch(), 5u);
  for (int e = 0; e < 3; ++e) {
    order.begin_epoch();
    EXPECT_EQ(order.epoch(), e);
    std::vector<std::int64_t> sizes;
    std::vector<std::int64_t> all;
    for (auto b = order.next(); !b.empty(); b = order.next()) {
      sizes.push_back(static_cast<std::int64_t>(b.size()));
      all.insert(all.end(), b.begin(), b.end());
    }
    EXPECT_EQ(sizes, (std::vector<std::int64_t>{5, 5, 5, 5, 3}));
    ASSERT_EQ(all.size(), n);
    for (std::size_t i = 0; i < n; ++i) EXPECT_EQ(static_cast<std::size_t>(all[i]), ref[e][i]);
    EXPECT_TRUE(order.next().empty());
  }
}

TEST(NnTraining, BatchOrderNoShuffleAndDropLast) {
  nn::BatchOrder plain(7, 3, 1, false);
  plain.begin_epoch();
  EXPECT_EQ(plain.next(), (std::vector<std::int64_t>{0, 1, 2}));
  EXPECT_EQ(plain.next(), (std::vector<std::int64_t>{3, 4, 5}));
  EXPECT_EQ(plain.next(), (std::vector<std::int64_t>{6}));
  EXPECT_TRUE(plain.next().empty());

  nn::BatchOrder drop(7, 3, 1, false, true);
  EXPECT_EQ(drop.batches_per_epoch(), 2u);
  drop.begin_epoch();
  EXPECT_EQ(drop.next().size(), 3u);
  EXPECT_EQ(drop.next().size(), 3u);
  EXPECT_TRUE(drop.next().empty());
  EXPECT_EQ(drop.cursor(), 7u);

  nn::BatchOrder fresh(4, 2, 1);
  EXPECT_THROW(fresh.next(), std::logic_error);
  EXPECT_THROW(nn::BatchOrder(0, 2, 1), std::invalid_argument);
  EXPECT_THROW(nn::BatchOrder(4, 0, 1), std::invalid_argument);
}

TEST(NnTraining, BatchOrderStateRoundTrip) {
  nn::BatchOrder a(17, 4, 9);
  a.begin_epoch();
  a.next();
  a.next();
  const std::string state = a.save_state();
  nn::BatchOrder    b(17, 4, 123);
  b.restore_state(state);
  EXPECT_EQ(b.epoch(), a.epoch());
  EXPECT_EQ(b.cursor(), a.cursor());
  EXPECT_EQ(concat_epoch(b), concat_epoch(a));
  // The RNG state carries over, so the next epoch is identical too.
  a.begin_epoch();
  b.begin_epoch();
  EXPECT_EQ(concat_epoch(b), concat_epoch(a));

  nn::BatchOrder other(18, 4, 9);
  EXPECT_THROW(other.restore_state(state), std::invalid_argument);
  EXPECT_THROW(b.restore_state("garbage"), std::invalid_argument);
  std::string bad = state;
  bad.replace(bad.rfind(' ', bad.size() - 2) + 1, std::string::npos, "999 ");
  EXPECT_THROW(b.restore_state(bad), std::invalid_argument);
}

TEST(NnTraining, TakeRowsIsDifferentiable) {
  Tensor x = nn::tensor({{1, 2}, {3, 4}, {5, 6}}, DType::Float64).requires_grad_();
  Tensor r = nn::take_rows(x, {2, 0, 2});
  EXPECT_EQ(r.shape(), (nn::Shape{3, 2}));
  EXPECT_EQ(r.to_vector<double>(), (std::vector<double>{5, 6, 1, 2, 5, 6}));
  const Tensor g = nn::grad(nn::sum(r), {x})[0];
  EXPECT_EQ(g.to_vector<double>(), (std::vector<double>{1, 1, 0, 0, 2, 2}));
}

TEST(NnTraining, EarlyStopping) {
  nn::EarlyStopping es(2, 0.1);
  EXPECT_THROW((void)es.best(), std::logic_error);
  auto d = es.observe(1.0, 0);
  EXPECT_TRUE(d.improved);
  EXPECT_FALSE(d.stop);
  d = es.observe(0.95, 1);  // within min_delta: no improvement
  EXPECT_FALSE(d.improved);
  EXPECT_FALSE(d.stop);
  EXPECT_EQ(es.misses(), 1);
  d = es.observe(0.8, 2);
  EXPECT_TRUE(d.improved);
  EXPECT_EQ(es.misses(), 0);
  EXPECT_DOUBLE_EQ(es.best(), 0.8);
  EXPECT_EQ(es.best_epoch(), 2);
  EXPECT_FALSE(es.observe(0.9, 3).stop);
  d = es.observe(0.85, 4);
  EXPECT_FALSE(d.improved);
  EXPECT_TRUE(d.stop);
  EXPECT_THROW(es.observe(0.1, 4), std::invalid_argument);  // epochs must increase
  EXPECT_THROW(es.observe(NAN, 5), std::invalid_argument);

  nn::EarlyStopping up(0, 0.0, true);
  EXPECT_TRUE(up.observe(0.5, 0).improved);
  EXPECT_TRUE(up.observe(0.6, 1).improved);
  d = up.observe(0.6, 2);
  EXPECT_FALSE(d.improved);
  EXPECT_TRUE(d.stop);  // zero patience stops at the first miss
  EXPECT_THROW(nn::EarlyStopping(-1), std::invalid_argument);
  EXPECT_THROW(nn::EarlyStopping(1, -0.1), std::invalid_argument);
}

TEST(NnTraining, WeightedMeanAndLog) {
  nn::WeightedMean m;
  EXPECT_THROW((void)m.value(), std::logic_error);
  m.add(1.0, 2);
  m.add(4.0, 1);
  EXPECT_DOUBLE_EQ(m.value(), 2.0);
  EXPECT_EQ(m.count(), 3u);
  EXPECT_THROW(m.add(1.0, 0), std::invalid_argument);
  EXPECT_THROW(m.add(INFINITY, 1), std::invalid_argument);
  std::ostringstream out;
  nn::log_epoch(out, 3, 7, m);
  EXPECT_EQ(out.str(), "  epoch 3: batches=7, samples=3, mean loss=2\n");
  m.reset();
  EXPECT_EQ(m.count(), 0u);
}

// ---- metrics -----------------------------------------------------------------------------

TEST(NnMetrics, ConfusionMatrix) {
  nn::ConfusionMatrix cm(3);
  // truth -> prediction
  cm.add(nn::from_vector(std::vector<std::int32_t>{0, 0, 1, 1, 1, 2}),
         nn::from_vector(std::vector<std::int64_t>{0, 1, 1, 1, 0, 1}));
  EXPECT_EQ(cm.cells()[0], (std::vector<std::uint64_t>{1, 1, 0}));
  EXPECT_EQ(cm.cells()[1], (std::vector<std::uint64_t>{1, 2, 0}));
  EXPECT_EQ(cm.cells()[2], (std::vector<std::uint64_t>{0, 1, 0}));
  EXPECT_DOUBLE_EQ(cm.accuracy(), 0.5);
  const auto c1 = cm.per_class(1);
  EXPECT_DOUBLE_EQ(c1.precision, 0.5);
  EXPECT_DOUBLE_EQ(c1.recall, 2.0 / 3.0);
  EXPECT_DOUBLE_EQ(c1.f1, 4.0 / 7.0);
  EXPECT_DOUBLE_EQ(c1.iou, 0.4);
  EXPECT_EQ(c1.support, 3u);
  const auto c2 = cm.per_class(2);  // never predicted, truth once
  EXPECT_DOUBLE_EQ(c2.precision, 0.0);
  EXPECT_DOUBLE_EQ(c2.recall, 0.0);
  EXPECT_DOUBLE_EQ(c2.iou, 0.0);
  const auto macro = cm.macro();
  EXPECT_NEAR(macro.precision, (0.5 + 0.5 + 0) / 3, 1e-15);
  EXPECT_NEAR(macro.recall, (0.5 + 2.0 / 3 + 0) / 3, 1e-15);
  EXPECT_EQ(macro.support, 6u);
  EXPECT_EQ(cm.to_tensor().dtype(), DType::Int64);
  EXPECT_EQ(cm.to_tensor().to_vector<int>(), (std::vector<int>{1, 1, 0, 1, 2, 0, 0, 1, 0}));

  nn::ConfusionMatrix empty(2);
  EXPECT_TRUE(std::isnan(empty.per_class(0).iou));
  EXPECT_THROW((void)empty.accuracy(), std::logic_error);
  EXPECT_THROW(empty.add(2, 0), std::out_of_range);
  EXPECT_THROW(empty.add(nn::tensor({0.0}), nn::tensor({0.0})), std::invalid_argument);
  EXPECT_THROW(nn::ConfusionMatrix(0), std::invalid_argument);
}

TEST(NnMetrics, AccuracyAndTopK) {
  const Tensor scores = nn::tensor({{0.1, 0.7, 0.2}, {0.5, 0.3, 0.2}, {0.2, 0.3, 0.5}, {0.4, 0.4, 0.2}});
  const Tensor labels = nn::from_vector(std::vector<std::int64_t>{1, 1, 2, 1});
  // argmax picks the first of the tied maxima in row 3 (class 0).
  EXPECT_DOUBLE_EQ(nn::accuracy(scores, labels), 0.5);
  EXPECT_DOUBLE_EQ(nn::accuracy(nn::from_vector(std::vector<std::int64_t>{1, 0, 2, 1}), labels), 0.75);
  EXPECT_DOUBLE_EQ(nn::top_k_accuracy(scores, labels, 1), 0.75);  // ties are not "strictly higher"
  EXPECT_DOUBLE_EQ(nn::top_k_accuracy(scores, labels, 2), 1.0);
  EXPECT_THROW(nn::top_k_accuracy(scores, labels, 4), std::invalid_argument);
  EXPECT_THROW(nn::top_k_accuracy(scores, labels, 0), std::invalid_argument);
  EXPECT_THROW(nn::top_k_accuracy(nn::tensor({{NAN, 1.0}}), nn::from_vector(std::vector<std::int64_t>{0}), 1),
               std::invalid_argument);
  EXPECT_THROW(nn::top_k_accuracy(scores, nn::from_vector(std::vector<std::int64_t>{1, 1, 3, 1}), 1),
               std::invalid_argument);
}

TEST(NnMetrics, RocAucAndAveragePrecision) {
  // sklearn: roc_auc_score([0,0,1,1],[.1,.4,.35,.8]) = 0.75, average_precision_score = 0.8333...
  EXPECT_DOUBLE_EQ(nn::roc_auc({0, 0, 1, 1}, {0.1, 0.4, 0.35, 0.8}), 0.75);
  EXPECT_NEAR(nn::average_precision({0, 0, 1, 1}, {0.1, 0.4, 0.35, 0.8}), 5.0 / 6.0, 1e-15);
  // Ties count half: roc_auc_score([0,1,0,1],[.5,.5,.5,.9]) = 0.75.
  EXPECT_DOUBLE_EQ(nn::roc_auc({0, 1, 0, 1}, {0.5, 0.5, 0.5, 0.9}), 0.75);
  EXPECT_DOUBLE_EQ(nn::roc_auc({0, 1}, {0.2, 0.9}), 1.0);
  EXPECT_DOUBLE_EQ(nn::roc_auc({1, 0}, {0.2, 0.9}), 0.0);
  EXPECT_DOUBLE_EQ(nn::average_precision({0, 0}, {0.2, 0.9}), 0.0);
  EXPECT_THROW(nn::roc_auc({1, 1}, {0.2, 0.9}), std::invalid_argument);
  EXPECT_THROW(nn::roc_auc({0, 2}, {0.2, 0.9}), std::invalid_argument);
  EXPECT_THROW(nn::roc_auc({0, 1}, {0.2}), std::invalid_argument);
}
