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
void AxiomChecks() {
  auto first = [](const Tensor& x) {
    const Tensor a = nn::slice_axis(x, 1, 0, 1), b = nn::slice_axis(x, 1, 1, 2);
    return nn::add(nn::sub(nn::relu(a), nn::relu(nn::sub(a, 1.0))), nn::mul(b, 2.0));
  };
  auto second = [](const Tensor& x) {
    const Tensor a = nn::slice_axis(x, 1, 0, 1), b = nn::slice_axis(x, 1, 1, 2);
    return nn::add(nn::sub(1.0, nn::relu(nn::sub(1.0, nn::relu(a)))), nn::mul(b, 2.0));
  };
  auto callback = [](std::function<Tensor(const Tensor&)> f) {
    return [f](const Tensor& points, const Tensor&) {
      Tensor x = points.detach().clone();
      x.requires_grad_(true);
      return nn::grad(nn::sum(f(x)), {x})[0].detach().contiguous();
    };
  };
  const Tensor input = sal::FromF({2, 3, 9, .5f, -1, 4}, {2, 3}), baseline = nn::zeros({2, 3}),
               classes = sal::FromI({0, 0});
  const auto ig        = ig::IntegratedGradients(input, baseline, classes, 64, 7, callback(first)),
             equivalent =
                 ig::IntegratedGradients(input, baseline, classes, 64, 1, callback(second)),
             zero = ig::IntegratedGradients(input, input, classes, 64, 5, callback(first));
  const auto a = sal::F(ig.attribution), e = sal::F(equivalent.attribution),
             z                       = sal::F(zero.attribution);
  const std::vector<double> expected = {1, 6, 0, .5, -2, 0};
  for (int i = 0; i < 6; ++i) {
    check::Near(a[i], expected[i], 1e-7, "ALL6 analytic saturated/linear/dummy IG entries");
    check::Near(
        e[i], expected[i], 1e-7, "equivalent implementation and different batching give same IG");
    check::Near(z[i], 0, 0, "input equals baseline gives exactzero attribution");
  }
  const auto endpoint = sal::F(callback(first)(input, classes));
  check::Near(endpoint[0], 0, 0, "saturated endpoint derivative0");
  check::Near(a[0], 1, 0, "nonzero IG despite saturated endpoint gradient0");
  check::Near(a[0] + a[1], 7, 1e-7, "positive completeness exact");
  check::Near(a[3] + a[4], -1.5, 1e-7, "negative completeness exact");
  std::cout << "IG analytic "
               "saturation/sensitivity/linear/dummy/zero-delta/equivalent-implementation/batching "
               "tests passed;ALL6 "
               "exactattributions,completeness7/-1.5,endpointgradient0butIG1\n";
}
} // namespace
