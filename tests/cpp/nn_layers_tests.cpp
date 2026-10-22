// Tests for datamunge::nn parameters, initializers, layers, losses and optimizers.

#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <datamunge/nn/nn.hpp>
#include <vector>

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

using Inputs = std::vector<Tensor>;

std::vector<double> vals(const Tensor& t) { return t.to_vector<double>(); }

double mean_of(const Tensor& t) { return nn::mean(nn::cast(t.detach(), DType::Float64)).item(); }
double std_of(const Tensor& t) { return nn::std_dev(nn::cast(t.detach(), DType::Float64)).item(); }

void expect_close(const Tensor& a, const std::vector<double>& b, double tol) {
  const auto v = vals(a);
  ASSERT_EQ(v.size(), b.size());
  for (std::size_t i = 0; i < v.size(); ++i) EXPECT_NEAR(v[i], b[i], tol) << "at " << i;
}

}  // namespace

// ---- initializers and the parameter store ---------------------------------------------

TEST(NnParams, Fans) {
  EXPECT_EQ(nn::init::fans({3, 5}), std::make_pair(3.0, 5.0));
  EXPECT_EQ(nn::init::fans({3, 3, 4, 8}), std::make_pair(36.0, 72.0));
  EXPECT_EQ(nn::init::fans({7}), std::make_pair(7.0, 7.0));
}

TEST(NnParams, InitializerScales) {
  nn::Generator gen(3);
  const Tensor  g = nn::init::glorot_uniform()({200, 300}, DType::Float32, gen);
  const double  limit = std::sqrt(6.0 / 500);
  EXPECT_LE(nn::max(g).item(), limit);
  EXPECT_GE(nn::min(g).item(), -limit);
  EXPECT_NEAR(std_of(g), limit / std::sqrt(3.0), 0.01 * limit);
  EXPECT_NEAR(std_of(nn::init::he_normal()({400, 300}, DType::Float64, gen)), std::sqrt(2.0 / 400), 2e-3);
  EXPECT_NEAR(std_of(nn::init::glorot_normal()({400, 300}, DType::Float64, gen)), std::sqrt(2.0 / 700), 2e-3);
  EXPECT_NEAR(std_of(nn::init::lecun_normal()({400, 300}, DType::Float64, gen)), std::sqrt(1.0 / 400), 2e-3);
  const Tensor tn = nn::init::truncated_normal(0.5)({1000}, DType::Float64, gen);
  EXPECT_LE(nn::max(nn::abs(tn)).item(), 1.0);
  EXPECT_EQ(vals(nn::init::constant(0.25)({2}, DType::Float32, gen)), (std::vector<double>{0.25, 0.25}));
}

TEST(NnParams, OrthogonalInitializer) {
  nn::Generator gen(5);
  const Tensor  tall = nn::init::orthogonal()({6, 3}, DType::Float64, gen);
  expect_close(nn::matmul(tall, tall, true, false), vals(nn::eye(3, DType::Float64)), 1e-12);
  const Tensor wide = nn::init::orthogonal(2.0)({3, 6}, DType::Float64, gen);
  expect_close(nn::matmul(wide, wide, false, true), vals(nn::eye(3, DType::Float64) * 4.0), 1e-12);
  const Tensor conv = nn::init::orthogonal()({2, 2, 3, 4}, DType::Float32, gen);
  EXPECT_EQ(conv.shape(), (nn::Shape{2, 2, 3, 4}));
}

TEST(NnParams, StoreCreatesOnceAndReuses) {
  nn::ParameterStore store(7);
  Tensor             a = store.get("layer/W", {2, 3}, nn::init::glorot_uniform());
  Tensor             b = store.get("layer/W", {2, 3}, nn::init::zeros());
  EXPECT_TRUE(a.is_same(b));
  EXPECT_TRUE(a.requires_grad());
  EXPECT_TRUE(a.is_leaf());
  EXPECT_THROW(store.get("layer/W", {3, 2}, nn::init::zeros()), std::invalid_argument);
  store.get("layer/b", {3}, nn::init::constant(0.5));
  EXPECT_EQ(store.size(), 2u);
  EXPECT_EQ(store.count(), 9);
  EXPECT_EQ(store.names(), (std::vector<std::string>{"layer/W", "layer/b"}));
  EXPECT_THROW((void)store.at("missing"), std::out_of_range);
  // Same seed, same values.
  nn::ParameterStore again(7);
  EXPECT_EQ(vals(again.get("layer/W", {2, 3}, nn::init::glorot_uniform())), vals(a));
}

TEST(NnParams, FlattenCopyCloneSet) {
  nn::ParameterStore store(1);
  store.get("a", {2, 2}, nn::init::normal(1.0));
  store.get("b", {3}, nn::init::normal(1.0));
  const auto flat = store.flatten();
  ASSERT_EQ(flat.size(), 7u);
  std::vector<double> doubled;
  for (double v : flat) doubled.push_back(2 * v);
  store.unflatten(doubled);
  for (std::size_t i = 0; i < 7; ++i) EXPECT_NEAR(store.flatten()[i], doubled[i], 1e-6);
  EXPECT_THROW(store.unflatten({1.0}), std::invalid_argument);

  nn::ParameterStore copy = store.clone();
  EXPECT_FALSE(copy.at("a").is_same(store.at("a")));
  EXPECT_EQ(copy.flatten(), store.flatten());

  nn::ParameterStore target(2);
  target.get("a", {2, 2}, nn::init::zeros());
  target.get("b", {3}, nn::init::zeros());
  target.copy_from(store, 0.25);
  for (std::size_t i = 0; i < 7; ++i) EXPECT_NEAR(target.flatten()[i], 0.25 * store.flatten()[i], 1e-6);
  target.copy_from(store);
  EXPECT_EQ(target.flatten(), store.flatten());

  store.set("b", nn::tensor({1, 2, 3}));
  EXPECT_EQ(vals(store.at("b")), (std::vector<double>{1, 2, 3}));
  store.set("c", nn::tensor({9}));
  EXPECT_TRUE(store.contains("c"));
  EXPECT_THROW(store.set("b", nn::tensor({1})), std::invalid_argument);
}

// ---- layers ---------------------------------------------------------------------------

TEST(NnLayers, DenseValuesShapesAndReuse) {
  nn::ParameterStore store(0);
  const Tensor       x = nn::tensor({{1, 2}, {3, 4}});
  store.set("fc/W", nn::tensor({{1, 0, -1}, {2, 1, 0}}));
  store.set("fc/b", nn::tensor({0.5, 0, 0}));
  const Tensor y = nn::dense(store, "fc", x, 3);
  EXPECT_EQ(vals(y), (std::vector<double>{5.5, 2, -1, 11.5, 4, -3}));
  EXPECT_EQ(vals(nn::dense(store, "fc", x, 3, nn::Activation::Relu)), (std::vector<double>{5.5, 2, 0, 11.5, 4, 0}));
  EXPECT_EQ(store.size(), 2u);  // reused, not recreated
  // Leading axes are kept.
  const Tensor seq = nn::dense(store, "fc", nn::ones({4, 5, 2}), 3);
  EXPECT_EQ(seq.shape(), (nn::Shape{4, 5, 3}));
  // No bias.
  nn::DenseOptions no_bias;
  no_bias.bias = false;
  nn::dense(store, "nb", x, 2, no_bias);
  EXPECT_FALSE(store.contains("nb/b"));
  EXPECT_EQ(nn::mlp(store, "mlp", x, {4, 3}).shape(), (nn::Shape{2, 3}));
  EXPECT_TRUE(store.contains("mlp/dense1/W"));
}

TEST(NnLayers, ActivationNamesAndValues) {
  const Tensor x = nn::tensor({-1.0, 0.5});
  EXPECT_EQ(nn::activation_name(nn::Activation::GeluTanh), "gelu_tanh");
  EXPECT_EQ(vals(nn::activate(x, nn::Activation::LeakyRelu, 0.1)), (std::vector<double>{-0.1f, 0.5}));
  EXPECT_EQ(vals(nn::activate(x, nn::Activation::None)), vals(x));
  for (auto a : {nn::Activation::Relu, nn::Activation::Relu6, nn::Activation::Elu, nn::Activation::Selu,
                 nn::Activation::Gelu, nn::Activation::Silu, nn::Activation::Mish, nn::Activation::Softplus,
                 nn::Activation::Softsign, nn::Activation::Sigmoid, nn::Activation::Tanh})
    EXPECT_EQ(nn::activate(x, a).shape(), x.shape()) << nn::activation_name(a);
}

TEST(NnLayers, DenseGradcheck) {
  nn::ParameterStore store(4, DType::Float64);
  const Tensor       x = nn::normal({3, 4}, 0, 1, store.generator(), DType::Float64);
  nn::dense(store, "fc", x, 2, nn::Activation::Tanh);
  const auto r = nn::gradcheck(
      [&](const Inputs& in) {
        store.set("fc/W", in[0]);
        return nn::tanh(nn::bias_add(nn::matmul(x, in[0]), in[1]));
      },
      {store.at("fc/W").detach().clone().requires_grad_(), store.at("fc/b").detach().clone().requires_grad_()});
  EXPECT_TRUE(r.ok) << r.message;
  // The layer's own parameters receive gradients.
  const auto g = nn::grad(nn::sum(nn::dense(store, "fc", x, 2, nn::Activation::Tanh)), store.parameters());
  EXPECT_EQ(g[0].shape(), (nn::Shape{4, 2}));
}

TEST(NnLayers, Dropout) {
  nn::Generator gen(9);
  const Tensor  x = nn::ones({100, 100});
  EXPECT_TRUE(nn::dropout(x, 0.3, false, gen).is_same(x));
  const Tensor y     = nn::dropout(x, 0.3, true, gen);
  const double kept  = nn::mean(nn::cast(nn::greater(y, 0.0), DType::Float64)).item();
  EXPECT_NEAR(kept, 0.7, 0.02);
  EXPECT_NEAR(mean_of(y), 1.0, 0.03);
  EXPECT_NEAR(nn::max(y).item(), 1.0 / 0.7, 1e-6);
  EXPECT_THROW(nn::dropout(x, 1.0, true, gen), std::invalid_argument);
}

TEST(NnLayers, EmbeddingAndMaskedMean) {
  nn::ParameterStore store(2);
  const Tensor       ids = nn::from_vector(std::vector<std::int32_t>{1, 1, 3, 0}, {2, 2});
  const Tensor       e   = nn::embedding(store, "emb", ids, 5, 4);
  EXPECT_EQ(e.shape(), (nn::Shape{2, 2, 4}));
  EXPECT_LE(nn::max(nn::abs(store.at("emb/table"))).item(), 0.05);
  // Repeated ids accumulate gradient.
  const Tensor g = nn::grad(nn::sum(e), {store.at("emb/table")})[0];
  EXPECT_EQ(vals(nn::sum(g, {1})), (std::vector<double>{4, 8, 0, 4, 0}));

  const Tensor x    = nn::tensor({1, 2, 3, 10, 20, 30, 5, 6, 7, 0, 0, 0}, {2, 2, 3});
  const Tensor mask = nn::tensor({1, 0, 1, 1}, {2, 2});
  EXPECT_EQ(vals(nn::masked_mean(x, mask)), (std::vector<double>{1, 2, 3, 2.5, 3, 3.5}));
}

TEST(NnLayers, Normalization) {
  nn::ParameterStore store(3, DType::Float64);
  const Tensor       x = nn::normal({4, 6}, 2, 3, store.generator(), DType::Float64);
  const Tensor       y = nn::layer_norm(store, "ln", x);
  for (double m : vals(nn::mean(y, {1}))) EXPECT_NEAR(m, 0, 1e-12);
  for (double v : vals(nn::variance(y, {1}))) EXPECT_NEAR(v, 1, 1e-4);
  const Tensor r = nn::rms_norm(store, "rms", x);
  for (double v : vals(nn::mean(nn::square(r), {1}))) EXPECT_NEAR(v, 1, 1e-4);

  const auto check = nn::gradcheck(
      [&](const Inputs& in) {
        nn::ParameterStore local(0, DType::Float64);
        return nn::layer_norm(local, "ln", in[0]) * nn::arange(0, 6, 1, DType::Float64);
      },
      {x.detach().requires_grad_()});
  EXPECT_TRUE(check.ok) << check.message;
  const auto check_rms = nn::gradcheck(
      [&](const Inputs& in) {
        nn::ParameterStore local(0, DType::Float64);
        return nn::rms_norm(local, "rms", in[0]) * nn::arange(0, 6, 1, DType::Float64);
      },
      {x.detach().requires_grad_()});
  EXPECT_TRUE(check_rms.ok) << check_rms.message;
}

TEST(NnLayers, RecurrentCells) {
  nn::ParameterStore store(6, DType::Float64);
  const Tensor       x = nn::normal({2, 3}, 0, 1, store.generator(), DType::Float64);
  const auto         s = nn::lstm_step(store, "lstm", x, nn::lstm_zero_state(2, 4, DType::Float64));
  EXPECT_EQ(s.h.shape(), (nn::Shape{2, 4}));
  EXPECT_EQ(vals(nn::slice_axis(store.at("lstm/b"), 0, 4, 8)), std::vector<double>(4, 1.0));  // forget bias
  EXPECT_EQ(nn::gru_step(store, "gru", x, nn::zeros({2, 5}, DType::Float64)).shape(), (nn::Shape{2, 5}));
  EXPECT_EQ(nn::rnn_step(store, "rnn", x, nn::zeros({2, 5}, DType::Float64)).shape(), (nn::Shape{2, 5}));
  const Tensor seq = nn::normal({2, 4, 3}, 0, 1, store.generator(), DType::Float64);
  for (auto kind : {nn::RnnKind::Simple, nn::RnnKind::Lstm, nn::RnnKind::Gru})
    EXPECT_EQ(nn::recurrent(store, "seq" + std::to_string(static_cast<int>(kind)), seq, 3, kind).shape(),
              (nn::Shape{2, 4, 3}));

  // Backpropagation through time matches finite differences.
  const auto r = nn::gradcheck(
      [&](const Inputs& in) {
        nn::ParameterStore local(11, DType::Float64);
        return nn::recurrent(local, "lstm", in[0], 3, nn::RnnKind::Lstm);
      },
      {seq.detach().requires_grad_()});
  EXPECT_TRUE(r.ok) << r.message;
  const auto r2 = nn::gradcheck(
      [&](const Inputs& in) {
        nn::ParameterStore local(12, DType::Float64);
        return nn::recurrent(local, "gru", in[0], 3, nn::RnnKind::Gru);
      },
      {seq.detach().requires_grad_()});
  EXPECT_TRUE(r2.ok) << r2.message;
}

// ---- losses -----------------------------------------------------------------------------

TEST(NnLosses, SoftmaxCrossEntropy) {
  const Tensor logits = nn::tensor({{2, 1, 0}, {0, 0, 3}}, DType::Float64);
  const Tensor labels = nn::from_vector(std::vector<std::int64_t>{0, 2});
  const auto   ls     = vals(nn::log_softmax(logits, 1));
  const double expected = -(ls[0] + ls[5]) / 2;
  EXPECT_NEAR(nn::sparse_softmax_cross_entropy(logits, labels).item(), expected, 1e-12);
  EXPECT_NEAR(nn::softmax_cross_entropy(logits, nn::one_hot(labels, 3, 1, 0, DType::Float64)).item(), expected, 1e-12);
  expect_close(nn::sparse_softmax_cross_entropy(logits, labels, nn::Reduction::None), {-ls[0], -ls[5]}, 1e-12);
  EXPECT_NEAR(nn::sparse_softmax_cross_entropy(logits, labels, nn::Reduction::Sum).item(), 2 * expected, 1e-12);
  // Label smoothing 0.3 over 3 classes: targets 0.8 / 0.1 / 0.1.
  const double smooth = -(0.8 * ls[0] + 0.1 * ls[1] + 0.1 * ls[2] + 0.1 * ls[3] + 0.1 * ls[4] + 0.8 * ls[5]) / 2;
  EXPECT_NEAR(nn::sparse_softmax_cross_entropy(logits, labels, nn::Reduction::Mean, 0.3).item(), smooth, 1e-12);
  // Gradient is (softmax - onehot) / N.
  Tensor     z = logits.detach().requires_grad_();
  const auto g = nn::grad(nn::sparse_softmax_cross_entropy(z, labels), {z})[0];
  expect_close(g, vals((nn::softmax(logits, 1) - nn::one_hot(labels, 3, 1, 0, DType::Float64)) / 2.0), 1e-12);
  // Stable at large logits.
  EXPECT_NEAR(nn::sparse_softmax_cross_entropy(nn::tensor({{1000, 0}}), nn::from_vector(std::vector<std::int64_t>{1})).item(),
              1000, 1e-3);
  EXPECT_THROW(nn::sparse_softmax_cross_entropy(logits, logits), std::invalid_argument);
}

TEST(NnLosses, SigmoidCrossEntropyIsStable) {
  Tensor       z = nn::tensor({-1000, -1000, 1000, 1000, 0, 0}).requires_grad_();
  const Tensor y = nn::tensor({0, 1, 0, 1, 0, 1});
  const Tensor l = nn::sigmoid_cross_entropy(z, y, nn::Reduction::None);
  expect_close(l, {0, 1000, 1000, 0, std::log(2.0), std::log(2.0)}, 1e-6);
  expect_close(nn::grad(nn::sum(l), {z})[0], {0, -1, 1, 0, 0.5, -0.5}, 1e-6);
}

TEST(NnLosses, RegressionLosses) {
  const Tensor p = nn::tensor({1, 2, 5}), t = nn::tensor({1, 0, 1});
  EXPECT_NEAR(nn::mse(p, t).item(), 20.0 / 3, 1e-6);
  EXPECT_NEAR(nn::mae(p, t).item(), 2.0, 1e-6);
  expect_close(nn::huber(nn::tensor({-3, -0.5, 0, 0.5, 3})), {2.5, 0.125, 0, 0.125, 2.5}, 1e-7);
  expect_close(nn::huber(nn::tensor({3}), 2.0), {4}, 1e-7);
  EXPECT_NEAR(nn::huber_loss(p, t, 1.0, nn::Reduction::Sum).item(), 0 + 1.5 + 3.5, 1e-6);
  Tensor e = nn::tensor({-3, 0.5, 3}).requires_grad_();
  expect_close(nn::grad(nn::sum(nn::huber(e)), {e})[0], {-1, 0.5, 1}, 1e-7);
  EXPECT_THROW(nn::huber(e, 0.0), std::invalid_argument);
  EXPECT_NEAR(nn::hinge(nn::tensor({0.5, 2, -1}), nn::tensor({1, 1, -1}), nn::Reduction::Sum).item(), 0.5, 1e-7);
}

TEST(NnLosses, KlDivergence) {
  const Tensor mu = nn::tensor({0.5, -1}, {2}, DType::Float64), lv = nn::tensor({0.2, -0.4}, {2}, DType::Float64);
  const auto   kl = vals(nn::kl_standard_normal(mu, lv));
  for (int i = 0; i < 2; ++i) {
    const double m = vals(mu)[i], v = vals(lv)[i];
    EXPECT_NEAR(kl[i], 0.5 * (std::exp(v) + m * m - 1 - v), 1e-12);
  }
  expect_close(nn::kl_gaussian(mu, lv, nn::zeros({2}, DType::Float64), nn::zeros({2}, DType::Float64)), kl, 1e-12);
  expect_close(nn::kl_gaussian(mu, lv, mu, lv), {0, 0}, 1e-12);
  const auto r = nn::gradcheck([](const Inputs& in) { return nn::kl_gaussian(in[0], in[1], in[2], in[3]); },
                               {mu.detach().requires_grad_(), lv.detach().requires_grad_(),
                                nn::tensor({0.1, 0.3}, {2}, DType::Float64).requires_grad_(),
                                nn::tensor({-0.2, 0.5}, {2}, DType::Float64).requires_grad_()});
  EXPECT_TRUE(r.ok) << r.message;
}

// ---- optimizers ---------------------------------------------------------------------------

namespace {

/// One step of `kind` from w = [1, -2] with gradient [0.5, -1] (lr 0.1).
std::vector<double> first_step(nn::OptimizerKind kind, nn::OptimizerConfig config = {}) {
  config.kind = kind;
  Tensor        w = nn::tensor({1, -2}, {2}, DType::Float64).requires_grad_();
  nn::Optimizer opt({w}, 0.1, config);
  opt.step({nn::tensor({0.5, -1}, {2}, DType::Float64)});
  return vals(w);
}

}  // namespace

TEST(NnOptim, FirstStepFormulas) {
  const double eps = 1e-8;
  auto         near = [](const std::vector<double>& a, const std::vector<double>& b) {
    ASSERT_EQ(a.size(), b.size());
    for (std::size_t i = 0; i < a.size(); ++i) EXPECT_NEAR(a[i], b[i], 1e-9) << i;
  };
  near(first_step(nn::OptimizerKind::Sgd), {0.95, -1.9});
  near(first_step(nn::OptimizerKind::Momentum), {0.95, -1.9});              // v = g
  near(first_step(nn::OptimizerKind::Nesterov), {1 - 0.1 * 0.95, -2 + 0.1 * 1.9});  // g + 0.9 g
  // Adagrad: a = 0.1 + g^2.
  near(first_step(nn::OptimizerKind::Adagrad),
       {1 - 0.1 * 0.5 / (std::sqrt(0.35) + eps), -2 + 0.1 * 1 / (std::sqrt(1.1) + eps)});
  // RMSProp: v = 0.1 g^2.
  near(first_step(nn::OptimizerKind::RMSProp),
       {1 - 0.1 * 0.5 / std::sqrt(0.025 + eps), -2 + 0.1 * 1 / std::sqrt(0.1 + eps)});
  // Adam family: m_hat = g, v_hat = g^2, so the first step is lr * g / (|g| + eps).
  for (auto kind : {nn::OptimizerKind::Adam, nn::OptimizerKind::AMSGrad, nn::OptimizerKind::Adamax})
    near(first_step(kind), {1 - 0.1 * 0.5 / (0.5 + eps), -2 + 0.1 * 1 / (1 + eps)});
  nn::OptimizerConfig wd;
  wd.weight_decay = 0.01;
  near(first_step(nn::OptimizerKind::AdamW, wd), {1 - 0.1 * (0.5 / (0.5 + eps) + 0.01), -2 + 0.1 * (1 / (1 + eps) + 0.02)});
  near(first_step(nn::OptimizerKind::Lion), {1 - 0.1, -2 + 0.1});
  // Nadam with the TF momentum schedule.
  const double mu = 0.9 * (1 - 0.5 * std::pow(0.96, 1)), mu2 = 0.9 * (1 - 0.5 * std::pow(0.96, 2));
  auto         nadam = [&](double g) {
    const double m = 0.1 * g, v = 0.001 * g * g;
    return (mu2 * m / (1 - mu * mu2) + (1 - mu) * g / (1 - mu)) / (std::sqrt(v / 0.001) + eps);
  };
  near(first_step(nn::OptimizerKind::Nadam), {1 - 0.1 * nadam(0.5), -2 - 0.1 * nadam(-1)});
  // LARS: trust ratio ||w|| / ||g|| scaled by the trust coefficient.
  const double wn = std::sqrt(5.0), gn = std::sqrt(1.25), local = 0.001 * wn / gn;
  near(first_step(nn::OptimizerKind::Lars), {1 - 0.1 * local * 0.5, -2 + 0.1 * local});
  // LAMB: the Adam direction rescaled to the norm of w.
  const double dn = std::sqrt(2.0) * (1 / (1 + eps));
  near(first_step(nn::OptimizerKind::Lamb), {1 - 0.1 * (wn / dn) * 0.5 / (0.5 + eps), -2 + 0.1 * (wn / dn) / (1 + eps)});
}

TEST(NnOptim, EveryKindDecreasesAQuadratic) {
  for (auto kind : {nn::OptimizerKind::Sgd, nn::OptimizerKind::Momentum, nn::OptimizerKind::Nesterov,
                    nn::OptimizerKind::Adagrad, nn::OptimizerKind::RMSProp, nn::OptimizerKind::Adam,
                    nn::OptimizerKind::AdamW, nn::OptimizerKind::Nadam, nn::OptimizerKind::AMSGrad,
                    nn::OptimizerKind::Adamax, nn::OptimizerKind::Lamb, nn::OptimizerKind::Lars,
                    nn::OptimizerKind::Lion, nn::OptimizerKind::Adafactor}) {
    nn::ParameterStore store(1);
    Tensor             W = store.get("W", {3, 2}, nn::init::normal(1.0));
    Tensor             b = store.get("b", {2}, nn::init::constant(1.0));
    auto               loss = [&] { return nn::sum(nn::square(W - 0.5)) + nn::sum(nn::square(b + 0.25)); };
    nn::OptimizerConfig config;
    config.kind      = kind;
    const double lr  = kind == nn::OptimizerKind::Lars      ? 50.0
                      : kind == nn::OptimizerKind::Lion    ? 0.01
                      : kind == nn::OptimizerKind::Adagrad ? 0.5
                                                           : 0.05;
    nn::Optimizer opt(store, lr, config);
    const double  start = loss().item();
    for (int i = 0; i < 200; ++i) opt.minimize(loss());
    EXPECT_LT(loss().item(), 0.1 * start) << nn::optimizer_name(kind);
    EXPECT_EQ(opt.iterations(), 200);
  }
}

TEST(NnOptim, ValidationAndState) {
  Tensor w = nn::tensor({1.0, 2.0}).requires_grad_();
  nn::OptimizerConfig bad;
  bad.weight_decay = 0.1;  // plain Adam has no decoupled decay
  bad.kind         = nn::OptimizerKind::Adam;
  EXPECT_THROW(nn::Optimizer({w}, 0.1, bad), std::invalid_argument);
  EXPECT_THROW(nn::Optimizer({w * 2.0}, 0.1), std::invalid_argument);  // not a leaf
  nn::OptimizerConfig adam;
  adam.kind = nn::OptimizerKind::Adam;
  nn::Optimizer opt({w}, 0.1, adam);
  EXPECT_THROW(opt.step({}), std::invalid_argument);
  EXPECT_THROW(opt.set_learning_rate(-1), std::invalid_argument);

  // Saving and restoring the state continues the same trajectory.
  auto grad_of = [](const Tensor& p) { return nn::grad(nn::sum(nn::pow(p, 4.0)), {p}); };
  for (int i = 0; i < 3; ++i) opt.step(grad_of(w));
  const auto state = opt.state();
  Tensor     w2    = w.detach().clone().requires_grad_();
  nn::Optimizer resumed({w2}, 0.1, adam);
  resumed.load_state(state);
  EXPECT_EQ(resumed.iterations(), 3);
  for (int i = 0; i < 3; ++i) {
    opt.step(grad_of(w));
    resumed.step(grad_of(w2));
  }
  EXPECT_EQ(vals(w), vals(w2));
  // Undefined gradients skip a parameter; grads from .grad() work too.
  const auto before = vals(w);
  opt.step({Tensor()});
  EXPECT_EQ(vals(w), before);
  nn::sum(w).backward();
  opt.step();
  EXPECT_NE(vals(w), before);
  opt.zero_grad();
  EXPECT_FALSE(w.grad().defined());
}

TEST(NnOptim, ClippingAndAccumulation) {
  const std::vector<Tensor> g = {nn::tensor({3, 4}), nn::tensor({0, 12})};
  EXPECT_NEAR(nn::global_norm(g), 13, 1e-9);
  const auto [clipped, norm] = nn::clip_by_global_norm(g, 6.5);
  EXPECT_NEAR(norm, 13, 1e-9);
  expect_close(clipped[0], {1.5, 2}, 1e-6);
  expect_close(clipped[1], {0, 6}, 1e-6);
  EXPECT_TRUE(nn::clip_by_global_norm(g, 20).first[0].is_same(g[0]));
  expect_close(nn::clip_by_norm(g[0], 1), {0.6, 0.8}, 1e-6);
  expect_close(nn::clip_by_value(g, -1, 3.5)[1], {0, 3.5}, 1e-6);

  nn::GradientAccumulator acc;
  EXPECT_THROW((void)acc.mean(), std::logic_error);
  acc.add(g);
  acc.add({nn::tensor({1, 0}), nn::tensor({2, 2})});
  EXPECT_EQ(acc.count(), 2);
  expect_close(acc.sum()[0], {4, 4}, 1e-6);
  expect_close(acc.mean()[1], {1, 7}, 1e-6);
  acc.reset();
  EXPECT_EQ(acc.count(), 0);
}

TEST(NnParams, SaveAndLoadNpz) {
  const std::string path = (std::filesystem::temp_directory_path() / "nn_params_roundtrip.npz").string();
  nn::ParameterStore store(3);
  nn::mlp(store, "mlp", nn::ones({2, 4}), {5, 3});
  store.save(path);
  nn::ParameterStore other(99);
  nn::mlp(other, "mlp", nn::ones({2, 4}), {5, 3});
  EXPECT_NE(other.flatten(), store.flatten());
  other.load(path);
  EXPECT_EQ(other.flatten(), store.flatten());
  nn::ParameterStore empty(0);
  EXPECT_THROW(empty.load(path), std::invalid_argument);  // strict: names must match
  empty.load(path, false);
  EXPECT_EQ(empty.names(), store.names());
  std::filesystem::remove(path);
}
