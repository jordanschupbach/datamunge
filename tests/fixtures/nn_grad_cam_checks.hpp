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
void FormulaChecks() {
  std::vector<float> av(16), gv(16);
  for (int i = 0; i < 16; ++i) {
    av[i] = .1f + .03f * i;
    gv[i] = std::array<float, 8>{.3f, -.2f, 0.f, .4f, -.1f, .2f, .5f, 0.f}[i % 8];
  }
  const auto maps =
      cam::ClassActivationMaps(sal::FromF(av, {2, 2, 2, 2}), sal::FromF(gv, {2, 2, 2, 2}));
  const auto w = sal::F(maps.grad_weights), sgn = sal::F(maps.grad_signed),
             gm = sal::F(maps.grad_map), alpha_t = sal::F(maps.positive_alpha),
             pw_t = sal::F(maps.plus_weights), pm = sal::F(maps.plus_map),
             zero = sal::F(cam::NormalizeCam(nn::zeros({2, 2, 2})));
  auto   A = [&](int b, int y, int x, int c) { return double(av[((b * 2 + y) * 2 + x) * 2 + c]); };
  auto   G = [&](int b, int y, int x, int c) { return double(gv[((b * 2 + y) * 2 + x) * 2 + c]); };
  double error = 0;
  for (int b = 0; b < 2; ++b) {
    double sums[2] = {}, cw[2] = {}, pw[2] = {};
    for (int c = 0; c < 2; ++c)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
          sums[c] += A(b, y, x, c);
          cw[c] += G(b, y, x, c) / 4.;
        }
    for (int c = 0; c < 2; ++c)
      for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 2; ++x) {
          const double v     = G(b, y, x, c);
          const double alpha = v > 0 ? v * v / (2 * v * v + sums[c] * v * v * v) : 0;
          error = std::max(error, std::abs(alpha - alpha_t[((b * 2 + y) * 2 + x) * 2 + c]));
          pw[c] += alpha * std::max(v, 0.);
        }
    for (int c = 0; c < 2; ++c) {
      check::Near(w[b * 2 + c], cw[c], 1e-7, "CAM averages signed gradients");
      check::Near(pw_t[b * 2 + c], pw[c], 1e-7, "CAM++ positive derivative-weight sum");
    }
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x) {
        double classic = 0, plus = 0;
        for (int c = 0; c < 2; ++c) {
          classic += cw[c] * A(b, y, x, c);
          plus += pw[c] * A(b, y, x, c);
        }
        const int i = (b * 2 + y) * 2 + x;
        check::Near(sgn[i], classic, 1e-7, "all signed CAM pixels");
        check::Near(gm[i], std::max(classic, 0.), 1e-7, "all rectified CAM pixels");
        check::Near(pm[i], std::max(plus, 0.), 1e-7, "all CAM++ pixels");
      }
  }
  check::Near(error, 0, 1e-7, "all16positivealpha paper unreduced g2/g3 ratios");
  for (float v : zero)
    check::Near(v, 0, 0, "zeroCAM normalizes tozero,notNaN");
  // True higher derivatives of exp(affine score),not higher derivatives of rawlogit.
  double          higher_error = 0;
  nn::GradOptions create;
  create.create_graph = true;
  for (int i = 0; i < 8; ++i) {
    const float coefficient = .2f * std::sin(.7f * i + .2f);
    Tensor      x           = nn::scalar(.3f);
    x.requires_grad_(true);
    const Tensor exp_score = nn::exp(nn::add(nn::mul(x, double(coefficient)), double(.1f)));
    const Tensor d1 = nn::grad(exp_score, {x}, create)[0], d2 = nn::grad(d1, {x}, create)[0],
                 d3            = nn::grad(d2, {x}, create)[0];
    const double e             = std::exp(double(coefficient) * .3f + .1f);
    const Tensor derivatives[] = {exp_score, d1, d2, d3};
    for (int order = 1; order <= 3; ++order)
      higher_error = std::max(
          higher_error,
          std::abs(e * std::pow(coefficient, order) - derivatives[order].item<double>()));
  }
  check::Near(higher_error, 0, 2e-7, "ALL24 higher-order identities expS*g^order");
  std::cout << "two-batch CAM/CAM++ allweights/alpha/maps/zero normalization passed;alphaerror="
            << error << ",ALL24 actual1st/2nd/3rd exp-affine derivatives error=" << higher_error
            << ";raw-logit g2/g3 areNOT mislabeled as raw higher derivatives\n";
}
} // namespace
