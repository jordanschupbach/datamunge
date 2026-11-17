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
void Checks() {
  auto evaluate = [](const Tensor& point) {
    Tensor x = point.detach().clone();
    x.requires_grad_(true);
    const Tensor objective = nn::neg(nn::mean(nn::square(nn::sub(x, double(.7f)))));
    return dream::DreamEvaluation{
        objective.item<double>(), nn::grad(objective, {x})[0].detach().contiguous()};
  };
  Tensor     point = nn::full({1, 2, 2, 1}, double(.6f));
  const auto step  = dream::DreamAscent(point, 2., evaluate);
  check::Require(
      step.backtracks == 4 && step.after > step.before,
      "overshooting ascent backtracks to genuine increase");
  const auto g = sal::F(step.gradient), im = sal::F(step.image);
  for (int i = 0; i < 4; ++i) {
    check::Near(g[i], (.7f - .6f) / 2, 1e-8, "all analytic quadratic gradients");
    check::Near(im[i], .725, 1e-7, "all RMS normalized accepted pixels");
  }
  const auto zero = dream::DreamAscent(point, 0, evaluate);
  check::Require(
      sal::Bytes(zero.image) == sal::Bytes(point) && zero.step == 0,
      "zero rate preserves every input bit");
  point                 = nn::full({1, 2, 2, 1}, double(.7f));
  const auto stationary = dream::DreamAscent(point, .02, evaluate);
  check::Require(
      stationary.rms == 0 && sal::Bytes(stationary.image) == sal::Bytes(point),
      "zero gradient safe/no division by zero");
  const Tensor corners    = sal::FromF({0, 1, .5f, .25f}, {1, 2, 2, 1});
  const auto   resized    = sal::F(dream::DreamResize(corners, 3, 3));
  const double expected[] = {0, .5, 1, .25, .4375, .625, .5, .375, .25};
  for (int i = 0; i < 9; ++i)
    check::Near(resized[i], expected[i], 0, "all half-pixel bilinear octave pixels");
  check::Require(
      sal::Bytes(dream::DreamResize(corners, 2, 2)) == sal::Bytes(corners),
      "identity resize exact");
  check::Throws([&] { dream::DreamAscent(point, -1, evaluate); }, "negative input rate rejected");
  std::cout << "analytic quadratic gradients/RMS "
               "pixels/4backtracks/zero-rate/zero-gradient/all9half-pixel "
               "resize/identity/invalidrate passed\n";
}
} // namespace
