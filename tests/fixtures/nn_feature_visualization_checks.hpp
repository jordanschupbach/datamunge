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
  const Tensor noise = featurevis::FeatureNoise(1, 1433);
  Tensor       image = noise.clone();
  image.requires_grad_(true);
  const Tensor features = nn::concat({image, nn::mul(image, 2.0)}, 3);
  const auto   objective =
      featurevis::VisualizeFeature(image, features, sal::FromI({1}), 2, .3f, .7f);
  const auto          gradient_out = sal::F(nn::grad(objective.mean, {image})[0]);
  const auto          x            = sal::F(noise);
  double              activation = 0, l2 = 0, tv = 0;
  std::vector<double> gradient(784, 2. / 784);
  for (int y = 0; y < 28; ++y)
    for (int a = 0; a < 28; ++a) {
      const int    i = y * 28 + a;
      const double v = x[i];
      activation += 2 * v / 784;
      l2 += (v - .5) * (v - .5) / 784;
      gradient[i] -= 2 * double(.3f) * (v - .5) / 784;
      for (int j : {y < 27 ? i + 28 : -1, a < 27 ? i + 1 : -1})
        if (j >= 0) {
          const double difference = v - x[j];
          tv += difference * difference / 756;
          gradient[i] -= 2 * double(.7f) * difference / 756;
          gradient[j] += 2 * double(.7f) * difference / 756;
        }
    }
  check::Near(
      objective.mean.item<double>(),
      activation - double(.3f) * l2 - double(.7f) * tv,
      3e-7,
      "known full objective includes separate normalized priors");
  check::Near(
      sal::F(objective.feature)[0],
      activation,
      2e-7,
      "selected channel rather than all-channel average");
  check::Near(sal::F(objective.l2)[0], l2, 2e-8, "all centered L2 pixels");
  check::Near(sal::F(objective.tv)[0], tv, 1e-7, "all1512squared TV edges");
  for (int i = 0; i < 784; ++i)
    check::Near(
        gradient_out[i],
        gradient[i],
        2e-9,
        "ALL784analytic selected-feature/L2/TV derivatives including boundaries");
  check::Require(
      sal::Bytes(noise) == sal::Bytes(featurevis::FeatureNoise(1, 1433)) &&
          sal::Bytes(noise) != sal::Bytes(featurevis::FeatureNoise(1, 1434)),
      "seeded noise exact repeat and secondseed difference");
  std::cout << "known selectedchannel objective/L2/1512TVedges/ALL784gradient boundaries and "
               "seeded noise fixtures passed\n";
}
} // namespace
