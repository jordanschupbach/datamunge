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
namespace {
void GateChecks() {
  std::vector<float> values(9), incoming(9);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) {
      values[i * 3 + j]   = std::array<float, 3>{-2, 0, 2}[i];
      incoming[i * 3 + j] = std::array<float, 3>{-3, 0, 4}[j];
    }
  const Tensor up   = sal::FromF(incoming, {9});
  auto objective_of = [&](const Tensor& x) { return nn::sum(nn::mul(guided::GuidedRelu(x), up)); };
  Tensor x          = sal::FromF(values, {9});
  x.requires_grad_(true);
  const auto guided_out = sal::F(guided::GuidedRelu(x)), ordinary_out = sal::F(nn::relu(x));
  const auto g      = sal::F(nn::grad(objective_of(x), {x})[0]);
  const auto v      = sal::F(nn::grad(nn::sum(nn::mul(nn::relu(x), up)), {x})[0]);
  const auto deconv = sal::F(nn::mul(up, nn::cast(nn::greater(up, 0.0), nn::DType::Float32)));
  auto       rule   = [](float a, float b) {
    return sal::F(guided::GuidedReluBackward(sal::FromF({a}, {1}), sal::FromF({b}, {1})))[0];
  };
  const float a = rule(2, -3), b = rule(2, 4), sum = rule(2, 1);
  for (int i = 0; i < 9; ++i) {
    const double forward = std::max(0.f, values[i]), u = incoming[i];
    check::Near(guided_out[i], forward, 0, "guided forward exactly ReLU");
    check::Near(ordinary_out[i], forward, 0, "ordinary forward exactly ReLU");
    check::Near(
        g[i], values[i] > 0 && u > 0 ? u : 0, 0, "ALL9 guided gate combinations including zeros");
    check::Near(v[i], values[i] > 0 ? u : 0, 0, "ALL9 ordinary gates unchanged");
    check::Near(deconv[i], u > 0 ? u : 0, 0, "ALL9 incoming-only deconv gates");
  }
  check::Near(g[6], 0, 0, "guided blocks negative upstream at active unit");
  check::Near(v[6], -3, 0, "ordinary gradient retains negative upstream");
  check::Near(g[2], 0, 0, "guided blocks inactive forward unit");
  check::Near(deconv[2], 4, 0, "incoming-only rule differs from guided");
  check::Near(a + b, 4, 0, "modified backward separate signals");
  check::Near(sum, 1, 0, "modified backward sum differs,NOT a linear VJP");
  auto plus = values, minus = values;
  plus[6] += .001f;
  minus[6] -= .001f;
  nn::NoGradGuard guard;
  const double    fd = (objective_of(sal::FromF(plus, {9})).item<double>() -
                     objective_of(sal::FromF(minus, {9})).item<double>()) /
                    .002;
  check::Near(
      fd,
      -3,
      .001,
      "true derivative -3 while guided signal0;do NOT validate guided as true gradient");
  std::cout << "guidedReLU ALL9sign/zero gates exact;ordinary/deconv distinctions and "
               "unchangedforward passed;nonlinear backward4!=1,trueFD="
            << fd << " vs guided0 explicitly verified\n";
}
} // namespace
