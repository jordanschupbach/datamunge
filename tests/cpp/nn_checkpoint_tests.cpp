/// @file nn_checkpoint_tests.cpp
/// @brief Tests for training_state(), restore_training_state() and the checkpoint files.

#include <gtest/gtest.h>

#include <datamunge/nn/nn.hpp>

#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <vector>

namespace nn = datamunge::nn;
using nn::Tensor;

namespace {
const auto F64 = nn::DType::Float64;

/// A small regression model with an optimizer and a batch order over 40 rows.
struct Trainer {
  nn::ParameterStore store;
  nn::Optimizer opt;
  nn::BatchOrder order;
  Tensor X, y;
  explicit Trainer(nn::OptimizerKind kind, double fill = 0.0)
      : store(7, F64), opt(make(store, fill), 0.05, nn::OptimizerConfig{kind}), order(40, 8, 3) {
    std::vector<double> xv(40 * 3), yv(40);
    for (int i = 0; i < 40; ++i) {
      for (int j = 0; j < 3; ++j) xv[i * 3 + j] = std::sin(1.0 + i * (j + 1));
      yv[i] = xv[i * 3] - 2 * xv[i * 3 + 1] + 0.5;
    }
    X = nn::from_vector(xv, {40, 3}), y = nn::from_vector(yv, {40, 1});
    order.begin_epoch();
  }
  static nn::ParameterStore& make(nn::ParameterStore& s, double fill) {
    s.get("W", {3, 4}, nn::init::constant(fill == 0.0 ? 0.1 : fill));
    s.get("b", {4}, nn::init::constant(fill));
    s.get("V", {4, 1}, nn::init::constant(fill == 0.0 ? -0.2 : fill));
    return s;
  }
  void train(int steps) {
    for (int t = 0; t < steps; ++t) {
      auto idx = order.next();
      if (idx.empty()) order.begin_epoch(), idx = order.next();
      const Tensor xb = nn::take_rows(X, idx), yb = nn::take_rows(y, idx);
      const Tensor h = nn::tanh(nn::matmul(xb, store.at("W")) + store.at("b"));
      opt.minimize(nn::mean(nn::square(nn::matmul(h, store.at("V")) - yb)));
    }
  }
};

void expect_same(const nn::NamedTensors& a, const nn::NamedTensors& b) {
  ASSERT_EQ(a.size(), b.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    EXPECT_EQ(a[i].first, b[i].first);
    EXPECT_EQ(nn::cast(a[i].second, F64).to_vector<double>(), nn::cast(b[i].second, F64).to_vector<double>()) << a[i].first;
  }
}
}  // namespace

TEST(NnCheckpoint, ResumeReproducesUninterruptedRunExactly) {
  for (auto kind : {nn::OptimizerKind::Adam, nn::OptimizerKind::Adafactor, nn::OptimizerKind::Momentum}) {
    Trainer a(kind);
    a.train(17);
    const auto path = (std::filesystem::temp_directory_path() / "dm_nn_checkpoint_test.npz").string();
    nn::save_training_checkpoint(path, a.store, &a.opt, &a.order);
    a.train(23);
    // A fresh run whose weights are NaN: the checkpoint must supply everything.
    Trainer b(kind, std::numeric_limits<double>::quiet_NaN());
    nn::load_training_checkpoint(path, b.store, &b.opt, &b.order);
    b.train(23);
    expect_same(nn::training_state(a.store, &a.opt, &a.order), nn::training_state(b.store, &b.opt, &b.order));
    EXPECT_EQ(b.opt.iterations(), 40);
    std::filesystem::remove(path);
  }
}

TEST(NnCheckpoint, FailedRestoreChangesNothing) {
  Trainer a(nn::OptimizerKind::Adam);
  a.train(5);
  const auto good = nn::training_state(a.store, &a.opt, &a.order);
  Trainer b(nn::OptimizerKind::Adam);
  b.train(2);
  const auto before = nn::training_state(b.store, &b.opt, &b.order);
  auto modified = [&](auto change) {
    auto s = good;
    change(s);
    return s;
  };
  const std::vector<nn::NamedTensors> bad{
      modified([](auto& s) { s.erase(s.begin()); }),                                                  // missing parameter
      modified([](auto& s) { s[0].second = nn::zeros({4, 3}, F64); }),                                 // wrong parameter shape
      modified([](auto& s) { s.emplace_back("params/extra", nn::zeros({1}, F64)); }),                  // extra parameter
      modified([](auto& s) { s.emplace_back("optimizer/m/9", nn::zeros({1}, F64)); }),                 // index out of range
      modified([](auto& s) { s.emplace_back("optimizer/q/0", nn::zeros({3, 4}, F64)); }),              // unknown slot kind
      modified([](auto& s) { s.emplace_back("optimizer/m/01", nn::zeros({3, 4}, F64)); }),             // non-canonical index
      modified([](auto& s) { for (auto& [n, v] : s) if (n == "optimizer/v/1") v = nn::zeros({2}, F64); }),  // wrong slot shape
      modified([](auto& s) { std::erase_if(s, [](const auto& e) { return e.first.starts_with("optimizer/"); }); }),
      modified([](auto& s) { std::erase_if(s, [](const auto& e) { return e.first == "batch_order/state"; }); }),
      modified([](auto& s) { s.emplace_back("junk", nn::zeros({1}, F64)); }),
  };
  for (std::size_t i = 0; i < bad.size(); ++i) {
    EXPECT_THROW(nn::restore_training_state(bad[i], b.store, &b.opt, &b.order), std::invalid_argument) << "case " << i;
    expect_same(nn::training_state(b.store, &b.opt, &b.order), before);
  }
  // A batch order with other settings is rejected before anything changes.
  nn::BatchOrder other(40, 5, 3);
  EXPECT_THROW(nn::restore_training_state(good, b.store, &b.opt, &other), std::invalid_argument);
  expect_same(nn::training_state(b.store, &b.opt, &b.order), before);
  EXPECT_THROW(nn::load_training_checkpoint("/nonexistent/dir/x.npz", b.store, &b.opt), std::exception);
  expect_same(nn::training_state(b.store, &b.opt, &b.order), before);
}

TEST(NnCheckpoint, WeightsOnlyRestoreKeepsOptimizerState) {
  Trainer a(nn::OptimizerKind::Adam);
  a.train(9);
  const auto good = nn::training_state(a.store, &a.opt, &a.order);
  Trainer b(nn::OptimizerKind::Adam);
  b.train(3);
  const auto opt_before = b.opt.state();
  nn::restore_training_state(good, b.store);
  expect_same(nn::training_state(b.store), nn::training_state(a.store));
  ASSERT_EQ(b.opt.state().size(), opt_before.size());
  EXPECT_EQ(b.opt.iterations(), 3);
  // The optimizer's parameter handles alias the store, so the restored weights are what it updates next.
  EXPECT_EQ(b.opt.parameters()[0].to_vector<double>(), a.store.at("W").to_vector<double>());
}
