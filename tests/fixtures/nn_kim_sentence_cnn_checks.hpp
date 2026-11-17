#pragma once
// Independent original host formula and complete-graph regression checks.
#include "nn_sequence_check_support.hpp"
namespace {
std::array<double, 4> Host(const Weights& weights) {
  const int             ids[2][5] = {{1, 2, 3, 6, 5}, {2, 4, 1, 5, 3}}, length[2] = {3, 5};
  std::array<double, 4> result{};
  for (int n = 0; n < 2; ++n) {
    std::array<double, 6> pooled{};
    for (int branch = 0; branch < 3; ++branch) {
      const int kernel = branch + 3;
      for (int f = 0; f < 2; ++f) {
        double maximum = 0;
        for (int t = 0; t < 5 + 8 - kernel + 1; ++t) {
          double value = weights[2 + 2 * branch][f];
          for (int k = 0; k < kernel; ++k) {
            const int position = t + k - 4;
            if (position < 0 || position >= length[n])
              continue;
            for (int e = 0; e < 2; ++e)
              value += weights[0][(ids[n][position] - 1) * 2 + e] *
                       weights[1 + 2 * branch][(k * 2 + e) * 2 + f];
          }
          maximum = std::max(maximum, value);
        }
        pooled[2 * branch + f] = maximum;
      }
    }
    for (int c = 0; c < 2; ++c) {
      result[2 * n + c] = weights[8][c];
      for (int f = 0; f < 6; ++f)
        result[2 * n + c] += pooled[f] * weights[7][2 * f + c];
    }
  }
  return result;
}
double Objective(const Weights& weights) {
  const auto z = Host(weights);
  return .3 * z[0] - .2 * z[1] - .1 * z[2] + .4 * z[3];
}

void HostChecks() {
  nn::ParameterStore        p(337);
  const sentence::KimConfig config{7, 5, 2, 2};
  Tensor       tokens  = nn::from_vector(std::vector<int>{1, 2, 3, 6, 5, 2, 4, 1, 5, 3}, {2, 5});
  const Tensor lengths = nn::from_vector(std::vector<int>{3, 5}, {2});
  const Tensor weight  = nn::from_vector(std::vector<float>{.3f, -.2f, -.1f, .4f}, {2, 2});
  sentence::KimCnn(p, tokens, lengths, config);
  const auto weights =
      sentence::FixParameters(p, [&](std::size_t v, std::int64_t i, const nn::Shape& shape) {
        return shape.size() == 1 && v != 8
                   ? .17f + .03f * static_cast<float>(std::sin(i + v))
                   : static_cast<float>(.23 * std::sin(.7 * i + 1.3 * v + .2));
      });
  const auto net         = sentence::KimCnn(p, tokens, lengths, config);
  const auto gradients   = nn::grad(nn::sum(nn::mul(net.logits, weight)), p.parameters());
  const auto expected    = Host(weights);
  const auto z           = net.logits.to_vector<double>();
  double     value_error = 0;
  for (int i = 0; i < 4; ++i)
    value_error = std::max(value_error, std::abs(expected[i] - z[i]));
  const auto pad = nn::slice_axis(net.embedding, 0, 0, 1).to_vector<double>();
  for (double e : pad)
    check::Near(e, 0, 0, "structural PAD embedding exactly zero");
  int          coordinates = 0;
  const double gradient_error =
      sentence::HostGradientError(weights, gradients, Objective, coordinates);
  check::Require(coordinates == 80 && p.size() == 9, "all80 tiny parameter coordinates covered");
  check::Near(value_error, 0, 1e-7, "independent host wide convolution/ReLU/max/concat/head");
  check::Near(
      gradient_error, 0, 2e-6, "EVERY embedding/filter/bias/head derivative versus host FD");
  auto t   = tokens.to_vector<int>();
  t[3]     = 4;
  t[4]     = 1;
  tokens   = nn::from_vector(t, {2, 5});
  auto out = sentence::KimCnn(p, tokens, lengths, config).logits.to_vector<double>();
  for (int c = 0; c < 2; ++c)
    check::Near(out[c], expected[c], 1e-7, "padding values cannot affect sentence features");
  std::swap(t[5], t[9]);
  tokens            = nn::from_vector(t, {2, 5});
  out               = sentence::KimCnn(p, tokens, lengths, config).logits.to_vector<double>();
  double difference = 0;
  for (int c = 0; c < 2; ++c)
    difference += std::abs(out[2 + c] - expected[2 + c]);
  check::Require(difference > 1e-5, "window features depend on word order");
  std::cout << " independent wide3/4/5 conv/ReLU/max/head error=" << value_error
            << ",ALL80 host finite-difference derivative error=" << gradient_error
            << ";zeroPAD/padding/order passed\n";
}

void OptimizerChecks() {
  nn::ParameterStore p(347);
  p.get("classifier/W", {2, 2}, nn::init::zeros());
  p.set("classifier/W", nn::from_vector(std::vector<float>{3.f, .1f, 4.f, .2f}, {2, 2}));
  nn::OptimizerConfig options;
  options.kind    = nn::OptimizerKind::Adadelta;
  options.rho     = .95;
  options.epsilon = 1e-6;
  nn::Optimizer         step(p, 1., options);
  std::array<double, 4> host  = {3, .1, 4, .2}, eg{}, ed{};
  double                error = 0;
  for (int t = 0; t < 3; ++t) {
    std::vector<float> gradient(4);
    for (int i = 0; i < 4; ++i) {
      const float value  = (i + 1) * .2f * (t == 1 ? -1 : 1);
      gradient[i]        = value;
      eg[i]              = .95 * eg[i] + .05 * value * value;
      const double delta = std::sqrt(ed[i] + 1e-6) / std::sqrt(eg[i] + 1e-6) * value;
      ed[i]              = .95 * ed[i] + .05 * delta * delta;
      host[i] -= delta;
    }
    for (int c = 0; c < 2; ++c) {
      const double scale =
          std::max(1., std::sqrt(host[c] * host[c] + host[2 + c] * host[2 + c]) / 3);
      host[c] /= scale;
      host[2 + c] /= scale;
    }
    step.step({nn::from_vector(gradient, {2, 2})});
    {
      nn::NoGradGuard guard;
      p.at("classifier/W").copy_(nn::models::KimMaxNorm(p.at("classifier/W")));
    }
    std::map<std::string, nn::Tensor> slots;
    for (const auto& [key, value] : step.state())
      slots[key] = value;
    const auto w = p.at("classifier/W").to_vector<double>(),
               g = slots.at("accum/0").to_vector<double>(),
               d = slots.at("accum_update/0").to_vector<double>();
    for (int i = 0; i < 4; ++i)
      error = std::max(
          {error, std::abs(host[i] - w[i]), std::abs(eg[i] - g[i]), std::abs(ed[i] - d[i])});
  }
  check::Near(error, 0, 1e-6, "independent three-step Adadelta/both-slot/head-column projection");
  std::cout << " three signed Adadelta steps/all slots/maxnorm3 host error=" << error << std::endl;
}

void DropoutChecks() {
  nn::Generator gen(359);
  Tensor        x = nn::full({4096}, 1.0);
  x.requires_grad_(true);
  const Tensor training  = nn::dropout(x, .5, true, gen);
  const Tensor inference = nn::dropout(x, 0, true, gen);
  const auto   gradient  = nn::grad(nn::sum(training), {x})[0].to_vector<float>();
  const auto   values = training.to_vector<float>(), identity = inference.to_vector<float>();
  int          retained = 0;
  for (int i = 0; i < 4096; ++i) {
    check::Require(values[i] == 0 || values[i] == 2, "dropout.5 mask has only0 or2");
    retained += values[i] == 2;
    check::Near(gradient[i], values[i], 0, "dropout derivative uses SAME sampled mask");
    check::Near(identity[i], 1, 0, "keep1 dropout identity");
  }
  check::Require(retained > 1800 && retained < 2300, "seeded half-drop mask retention");
  std::cout << " dropout.5 values/shared-mask derivatives and keep1 identity passed,retained="
            << retained << "/4096\n";
}

void LossChecks() {
  Tensor z = nn::from_vector(std::vector<float>{1000, 999, -2, 3}, {2, 2});
  z.requires_grad_(true);
  const Tensor loss     = sentence::SentenceCE(z, nn::from_vector(std::vector<int>{1, 0}, {2}));
  const auto   g        = nn::grad(loss, {z})[0].to_vector<double>();
  const auto   logits   = z.to_vector<double>();
  double       expected = 0;
  for (int n = 0; n < 2; ++n) {
    const double a = logits[2 * n], b = logits[2 * n + 1], m = std::max(a, b),
                 sum = std::exp(a - m) + std::exp(b - m);
    const int label  = 1 - n;
    expected += (m + std::log(sum) - (label ? b : a)) / 2;
    for (int c = 0; c < 2; ++c)
      check::Near(
          g[2 * n + c],
          (std::exp((c ? b : a) - m) / sum - (c == label)) / 2,
          1e-7,
          "all stable mean CE logit derivatives");
  }
  check::Near(loss.item<double>(), expected, 2e-7, "independent stable mean binary CE");
  std::cout << " extreme-logit stable CE/batch mean/all logit derivatives passed\n";
}

void FullChecks() {
  nn::ParameterStore p(353);
  std::vector<int>   ids(128);
  for (int i = 0; i < 128; ++i)
    ids[i] = 2 + (i * 17) % 7997;
  const auto model = sentence::KimCnn(
      p, nn::from_vector(ids, {2, 64}), nn::from_vector(std::vector<int>{64, 37}, {2}));
  const auto norms =
      sentence::SquaredNorms(nn::grad(nn::sum(nn::square(model.logits)), p.parameters()));
  check::Require(
      p.count() == 2760602 && p.size() == 9, "independent full CNN-rand parameter formula");
  for (int i = 0; i < 3; ++i)
    check::Require(model.maps[i].shape() == nn::Shape{2, 70 - i, 100}, "window3/4/5 map shape");
  check::Require(
      model.features.shape() == nn::Shape{2, 300} && model.logits.shape() == nn::Shape{2, 2},
      "pooled concatenation and binary output");
  for (double n : norms)
    check::Require(
        std::isfinite(n) && n > 0, "all9 full parameter tensor gradients finite/nonzero");
  std::cout << " full CNN-rand2760602 parameters/9 tensors,E300,F100,map70/69/68,concat300;every "
               "learned tensor gradient passed\n";
}

} // namespace
