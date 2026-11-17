// Independent numeric/architecture checks; no experiment driver.
#pragma once
#include "nn_interpretation_check_support.hpp"
namespace flow {
using namespace dm::nn::models;
}
namespace pose {
using namespace dm::nn::models;
}
namespace sr {
using namespace dm::nn::models;
}
namespace vqa {
using namespace dm::nn::models;
}
namespace dream {
using namespace dm::nn::models;
}
namespace featurevis {
using namespace dm::nn::models;
}
namespace cam {
using namespace dm::nn::models;
}
namespace guided {
using namespace dm::nn::models;
}
namespace ig {
using namespace dm::nn::models;
}
namespace nn = datamunge::nn;
using nn::Tensor;
using namespace vqa;

namespace {

struct HostLstm {
  std::array<double, 2>              h{}, c{};
  std::vector<std::array<double, 2>> steps;
};
HostLstm Host(
    const std::vector<double>& table,
    const std::vector<double>& wx,
    const std::vector<double>& wh,
    const std::vector<double>& bias,
    const std::array<int, 4>&  tokens,
    int                        length) {
  HostLstm result;
  for (int time = 0; time < 4; ++time) {
    if (time < length) {
      std::array<double, 8> gates{};
      for (int j = 0; j < 8; ++j) {
        gates[j] = bias[j];
        for (int k = 0; k < 2; ++k)
          gates[j] += table[2 * tokens[time] + k] * wx[k * 8 + j] + result.h[k] * wh[k * 8 + j];
      }
      for (int k = 0; k < 2; ++k) {
        const double i = 1 / (1 + std::exp(-gates[k])), f = 1 / (1 + std::exp(-gates[2 + k])),
                     g = std::tanh(gates[4 + k]), o = 1 / (1 + std::exp(-gates[6 + k]));
        result.c[k] = f * result.c[k] + i * g;
        result.h[k] = o * std::tanh(result.c[k]);
      }
    }
    result.steps.push_back(result.h);
  }
  return result;
}

Tensor Ints(const std::vector<int>& values, nn::Shape shape) {
  return nn::from_vector(std::vector<std::int32_t>(values.begin(), values.end()), shape);
}

void LstmChecks() {
  nn::ParameterStore p(283);
  VqaConfig          config;
  config.vocabulary = 6, config.embedding = 2, config.hidden = 2, config.time = 4;
  {
    nn::NoGradGuard guard;
    VqaLstm(p, Ints({0, 0, 0, 0, 0, 0, 0, 0}, {2, 4}), Ints({1, 1}, {2}), config);
  }
  std::vector<double> table(12), wx(16), wh(16), bias = {.1, -.2, 1, .7, .05, -.1, .2, -.3};
  for (int i = 0; i < 12; ++i)
    table[i] = (i % 7 - 3) * .13;
  for (int i = 0; i < 16; ++i)
    wx[i] = (i % 5 - 2) * .11, wh[i] = (i % 7 - 3) * .07;
  const std::vector<std::string> variable_names = {
      "question/embedding/table", "question/lstm/Wx", "question/lstm/Wh", "question/lstm/b"};
  const std::vector<std::vector<double>> data = {table, wx, wh, bias};
  check::Require(p.names() == variable_names, "embedding/LSTM variable creation order");
  for (int v = 0; v < 4; ++v) {
    std::vector<float> value(data[v].begin(), data[v].end());
    p.set(variable_names[v], nn::from_vector(value, p.parameters()[std::size_t(v)].shape()));
  }
  const Tensor weights = nn::tensor({.3, -.4, .2, .1}, {2, 2});
  auto         run     = [&](const std::vector<int>& ids, const std::vector<int>& size) {
    return VqaLstm(p, Ints(ids, {2, 4}), Ints(size, {2}), config);
  };
  const auto result      = run({2, 3, 0, 0, 0, 0, 0, 0}, {2, 0});
  const auto gradients   = nn::grad(nn::sum(result.h * weights), p.parameters());
  const auto expected    = Host(table, wx, wh, bias, {2, 3, 0, 0}, 2);
  double     value_error = 0;
  for (int time = 0; time < 4; ++time) {
    const auto step = result.steps[std::size_t(time)].to_vector<double>();
    for (int c = 0; c < 2; ++c) {
      value_error = std::max(
          value_error,
          std::abs(expected.steps[std::size_t(time)][std::size_t(c)] - step[std::size_t(c)]));
      check::Near(step[std::size_t(2 + c)], 0, 0, "zero-length question state remainszero");
    }
  }
  const auto cell = result.c.to_vector<double>();
  for (int c = 0; c < 2; ++c)
    check::Near(
        cell[std::size_t(c)],
        expected.c[std::size_t(c)],
        1e-7,
        "cell memory freezes after last valid token");
  check::Near(value_error, 0, 1e-7, "independent signed LSTM gate/state values");
  double       gradient_error = 0;
  const double epsilon        = 1e-3;
  for (int variable = 0; variable < 4; ++variable) {
    const auto g = gradients[std::size_t(variable)].to_vector<double>();
    for (int coordinate = 0; coordinate < int(data[std::size_t(variable)].size()); ++coordinate) {
      auto plus = data, minus = data;
      plus[std::size_t(variable)][std::size_t(coordinate)] += epsilon;
      minus[std::size_t(variable)][std::size_t(coordinate)] -= epsilon;
      const auto a            = Host(plus[0], plus[1], plus[2], plus[3], {2, 3, 0, 0}, 2),
                 b            = Host(minus[0], minus[1], minus[2], minus[3], {2, 3, 0, 0}, 2);
      const double derivative = (.3 * (a.h[0] - b.h[0]) - .4 * (a.h[1] - b.h[1])) / (2 * epsilon);
      gradient_error = std::max(gradient_error, std::abs(derivative - g[std::size_t(coordinate)]));
    }
  }
  check::Near(
      gradient_error,
      0,
      2e-6,
      "EVERY embedding/input/recurrent/bias coordinate host finite difference");
  const auto table_gradient = gradients[0].to_vector<double>();
  for (int coordinate = 0; coordinate < 2; ++coordinate)
    check::Near(
        table_gradient[std::size_t(coordinate)], 0, 0, "padding embedding receives zero gradient");
  nn::NoGradGuard guard;
  const auto      padded = run({2, 3, 5, 4, 1, 2, 4, 5}, {2, 0}).h.to_vector<double>();
  for (int c = 0; c < 2; ++c)
    check::Near(
        padded[std::size_t(c)],
        expected.h[std::size_t(c)],
        1e-7,
        "padding values cannot change question state");
  const auto reordered = run({3, 2, 0, 0, 0, 0, 0, 0}, {2, 0}).h.to_vector<double>();
  double     change    = 0;
  for (int c = 0; c < 2; ++c)
    change += std::abs(reordered[std::size_t(c)] - expected.h[std::size_t(c)]);
  check::Require(change > 1e-5, "question encoder is sequence-sensitive,not bag averaging");
  std::cout << " signed masked LSTM every-state error=" << value_error
            << ",ALL52 parameter-coordinate host FD error=" << gradient_error
            << ",zero-length/cell freeze/padding/zero-PAD-gradient/order checks passed\n";
}

void LossScoreChecks() {
  const std::vector<float> logits = {1000, 999, 998, -2, 1, .5},
                           target = {.6f, .3f, .1f, 0, .2f, .8f};
  Tensor     z                    = nn::from_vector(logits, {2, 3}).requires_grad_();
  const auto loss                 = VqaSoftCE(z, nn::from_vector(target, {2, 3}));
  const auto g                    = nn::grad(loss, {z})[0].to_vector<double>();
  double     expected             = 0;
  for (int n = 0; n < 2; ++n) {
    double maximum = -INFINITY, sum = 0, total = 0;
    for (int c = 0; c < 3; ++c)
      maximum = std::max(maximum, double(logits[n * 3 + c])), total += target[n * 3 + c];
    for (int c = 0; c < 3; ++c)
      sum += std::exp(logits[n * 3 + c] - maximum);
    for (int c = 0; c < 3; ++c) {
      const double probability = std::exp(logits[n * 3 + c] - maximum) / sum;
      expected += target[n * 3 + c] * (std::log(sum) + maximum - logits[n * 3 + c]) / 2;
      check::Near(
          g[std::size_t(n * 3 + c)],
          (probability * total - target[n * 3 + c]) / 2,
          1e-7,
          "EVERY soft-target CE derivative including observed targetmass");
    }
  }
  check::Near(loss.item(), expected, 2e-7, "stable ten-answer target mean CE");
  const double known[] = {0, .3, .6, .9, 1, 1, 1, 1, 1, 1, 1};
  for (int matching = 0; matching <= 10; ++matching)
    check::Near(
        VqaConsensusCount(matching),
        known[matching],
        1e-12,
        "official10-way leave-one-out consensus is NOTmin(k/3,1)");
  VqaData data;
  data.target            = nn::tensor({0, 1, 0, 1, 0, 0}, {2, 3});
  data.score             = nn::tensor({0, .6, .3, 0, 0, 0}, {2, 3});
  data.hard              = Ints({1, 0}, {2});
  data.types             = Ints({0, 2}, {2});
  const auto probability = nn::tensor({.1, .8, .1, .8, .1, .1}, {2, 3});
  const auto score       = ScoreVqa(probability, data);
  check::Near(score.consensus, .3, 2e-8, "known score lookup and UNK-answer alwayszero");
  check::Near(score.exact, .5, 0, "unknown-majority never counted as correct unknownprediction");
  check::Near(score.coverage, .5, 0, "held-out answer-vocabulary coverage");
  check::Near(score.oracle, .3, 2e-8, "closed-vocabulary oracle score cap");
  std::cout << " every stable soft-answer CE value/gradient,official0/.3/.6/.9/1 "
               "consensus,UNK-zero/coverage/oracle/type fixtures passed\n";
}

std::int64_t Census(const VqaConfig& c) {
  std::int64_t total = 0;
  int          in    = 3;
  for (int block = 0; block < 3; ++block) {
    const int out = c.base * (1 << block);
    for (int layer = 0; layer < 2; ++layer)
      total += 9LL * in * out + out, in = out;
  }
  total += std::int64_t(in) * c.fusion + c.fusion;
  total += std::int64_t(c.vocabulary) * c.embedding +
           std::int64_t(c.embedding + c.hidden) * 4 * c.hidden + 4 * c.hidden +
           std::int64_t(c.hidden) * c.fusion + c.fusion;
  total += 3LL * c.fusion * c.fusion + c.fusion + std::int64_t(c.fusion) * c.answers + c.answers;
  return total;
}

void ModelChecks() {
  for (bool compact : {false, true}) {
    nn::ParameterStore p(277);
    VqaConfig          config;
    if (compact)
      config.base = 8, config.embedding = 32, config.hidden = 32, config.fusion = 64;
    std::vector<float> rgb(2 * 64 * 64 * 3);
    for (std::size_t i = 0; i < rgb.size(); ++i)
      rgb[i] = float(.5 + .4 * std::sin(i * .13));
    std::vector<int> ids(48, 0);
    for (int t = 0; t < 4; ++t)
      ids[std::size_t(t)] = t + 2, ids[std::size_t(24 + t)] = 6 - t;
    Tensor     image     = nn::from_vector(rgb, {2, 64, 64, 3}).requires_grad_();
    const auto net       = VqaBaseline(p, image, Ints(ids, {2, 24}), Ints({4, 3}, {2}), config);
    const auto objective = nn::mean(nn::square(net.logits));
    auto       inputs    = p.parameters();
    inputs.push_back(image);
    inputs.push_back(net.question_h);
    const auto gradients = nn::grad(objective, inputs);
    check::Require(
        p.count() == Census(config) && p.size() == 24 && net.stages.size() == 14 &&
            net.question_steps.size() == 24,
        "independent VQA parameter/stage/shared-LSTM census");
    std::cout << " " << (compact ? "compact" : "full") << " joint VQA:" << p.count()
              << " parameters/24 derivatives (executed),14fusion stages/24 shared LSTM steps\n";
    if (!compact)
      continue;
    std::vector<nn::Shape> expected;
    int                    side = 64;
    for (int block = 0; block < 3; ++block) {
      for (int layer = 0; layer < 2; ++layer)
        expected.push_back({2, side, side, 8 << block});
      side /= 2;
      expected.push_back({2, side, side, 8 << block});
    }
    for (std::int64_t w : {64, 64, 192, 64, 64})
      expected.push_back({2, w});
    for (int i = 0; i < 14; ++i)
      check::Require(
          net.stages[std::size_t(i)].shape() == expected[std::size_t(i)],
          "all exact image/fusion/head geometries");
    for (const auto& h : net.question_steps)
      check::Require(h.shape() == nn::Shape({2, 32}), "all24 recurrent step state shapes");
    double minimum = INFINITY, maximum = 0;
    for (const auto& g : gradients) {
      double norm = 0;
      for (float v : g.to_vector<float>()) {
        check::Require(std::isfinite(v), "finite VQA tensor/modality gradients");
        norm += double(v) * v;
      }
      norm = std::sqrt(norm);
      check::Require(
          norm > 0, "ALL24parameter tensors and BOTH modalities receive nonzero gradients");
      minimum = std::min(minimum, norm), maximum = std::max(maximum, norm);
    }
    std::cout << " compact all24 learned and image/question gradients finite/nonzero,norm range["
              << minimum << "," << maximum << "],14CNN/fusion and24recurrent shapes passed\n";
  }
}

} // namespace
