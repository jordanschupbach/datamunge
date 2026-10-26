// CMD: just run-nn data_augmentation
// Flip, crop, color jitter, cutout, mixup and CutMix on NHWC batches.
//
// Port of tflow's data_augmentation. Check exact pixel positions, unchanged
// evaluation inputs, clipping, soft labels and area-correct CutMix labels on
// small known NHWC batches. Crop coordinates come from a host RNG that is
// reproducible for a fixed seed and toolchain. Batch-shared transforms keep
// this example compact.
//
// Mixup blends images and probability-vector labels by the same Beta-sampled
// weight (https://arxiv.org/abs/1710.09412). CutMix replaces a rectangle and
// weights labels by its realized clipped area (https://arxiv.org/abs/1905.04899).
// Evaluation bypasses augmentation. This example checks transformations and
// label consistency, not a claim about classification accuracy improvements.
//
// Differences from TF: the transforms are nn::augment functions called
// eagerly (no graph re-run with fed parameters; "one graph reused" becomes
// "one function called with new arguments"). Invalid parameters throw
// std::invalid_argument at the call instead of failing a graph Assert.
// Beta weights are drawn from nn::Generator (seed 7), so the 20 sampled
// weights differ from TF's mt19937_64 draws; the crop coordinates still use
// mt19937_64(17) and match TF.

#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <random>
#include <vector>

#include "common/checks.hpp"

namespace nn  = datamunge::nn;
namespace aug = datamunge::nn::augment;
using nn::Tensor;

namespace {

/// [N, H, W, C] element accessor for a contiguous float vector.
struct View {
  std::vector<float> v;
  int                h, w, c;
  float              operator()(int n, int y, int x, int ch) const { return v[((n * h + y) * w + x) * c + ch]; }
};
View Of(const Tensor& t) {
  return {t.to_vector<float>(), int(t.size(1)), int(t.size(2)), int(t.size(3))};
}

void Transformations() {
  std::vector<float> pv(96);
  for (int i = 0; i < 96; ++i) pv[i] = i / 120.f;
  const Tensor input = nn::from_vector(pv, {2, 4, 4, 3});
  const View   pixels = Of(input);
  const auto   flip   = [](const Tensor& x, bool enabled) { return enabled ? aug::flip_left_right(x) : x; };
  const View   flipped = Of(flip(input, true)), twice = Of(flip(flip(input, true), true)), bypass = Of(flip(input, false));
  const View   crop     = Of(aug::crop(input, 1, 1, 2, 2));
  const View   identity = Of(aug::color_jitter(input, 0, 1, 1, 0));
  const auto   bright   = aug::color_jitter(nn::full({1, 2, 2, 3}, .8), .3, 1, 1, 0).to_vector<float>();
  const auto   green    = aug::color_jitter(nn::tensor({1, 0, 0}, {1, 1, 1, 3}), 0, 1, 1, 1.0 / 3).to_vector<float>();
  const View   cutout   = Of(aug::cutout(input, {1, 1, 2, 2}));
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        for (int c = 0; c < 3; ++c) {
          const auto original = pixels(n, y, x, c);
          check::Near(flipped(n, y, x, c), pixels(n, y, 3 - x, c), 0, "horizontal flip axis");
          check::Near(twice(n, y, x, c), original, 0, "double flip identity");
          check::Near(bypass(n, y, x, c), original, 0, "evaluation bypass");
          check::Near(identity(n, y, x, c), original, 2e-7, "identity color jitter");
          const bool erased = y >= 1 && y < 3 && x >= 1 && x < 3;
          check::Near(cutout(n, y, x, c), erased ? 0 : original, 0, "cutout region");
        }
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
        for (int c = 0; c < 3; ++c) check::Near(crop(n, y, x, c), pixels(n, y + 1, x + 1, c), 0, "crop offset");
  for (float v : bright) check::Near(v, 1, 0, "brightness clamp");
  for (int c = 0; c < 3; ++c) check::Near(green[c], c == 1 ? 1 : 0, 2e-7, "hue red to green");
  std::mt19937_64                    rng(17), repeat(17);
  std::uniform_int_distribution<int> coordinate(0, 2);
  for (int trial = 0; trial < 12; ++trial) {
    const int y = coordinate(rng), x = coordinate(rng);
    check::Require(y == coordinate(repeat) && x == coordinate(repeat), "seeded crop coordinates");
    const View out = Of(aug::crop(input, y, x, 2, 2));
    check::Near(out(0, 0, 0, 0), pixels(0, y, x, 0), 0, "runtime crop coordinates");
  }
  std::cout << "  flip, evaluation bypass, crop coordinates, color jitter and cutout passed\n"
               "  12 seeded random crops reused one crop function\n";
}

void Mixing() {
  std::vector<float> pv(96);
  for (int i = 0; i < 96; ++i) pv[i] = i < 48 ? .2f : .8f;
  const Tensor                    input = nn::from_vector(pv, {2, 4, 4, 3}), labels = nn::tensor({{1, 0}, {0, 1}});
  const std::vector<std::int64_t> partner{1, 0};
  const auto                      mixup  = aug::mixup(input, labels, partner, .25);
  const aug::Rect                 box{-1, 2, 3, 4};
  const auto                      cutmix = aug::cutmix(input, labels, partner, box);
  const auto mixed_images = mixup.images.to_vector<float>(), mixed_labels = mixup.labels.to_vector<float>(),
             cut_labels = cutmix.labels.to_vector<float>();
  check::Near(mixed_images[0], .65, 1e-7, "mixup image interpolation");
  check::Near(mixed_labels[0], .25, 0, "mixup same label coefficient");
  check::Near(mixed_labels[1], .75, 0, "mixup partner label");
  check::Near(cutmix.weight, .75, 0, "CutMix clipped area is 4/16, not nominal 12/16");
  check::Near(cut_labels[0], .75, 0, "area-correct CutMix label");
  const View pasted_images = Of(cutmix.images);
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        for (int c = 0; c < 3; ++c) {
          const bool  pasted   = y < 2 && x >= 2;
          const float expected = (pasted ? 1 - n : n) == 0 ? .2f : .8f;
          check::Near(pasted_images(n, y, x, c), expected, 0, "CutMix paste location");
        }
  nn::Generator rng(7), repeat(7);
  for (int trial = 0; trial < 20; ++trial) {
    const double lambda = aug::beta_mix_weight(rng, .4);
    check::Near(lambda, aug::beta_mix_weight(repeat, .4), 0, "reproducible beta mixing weights");
    const auto sampled = aug::mixup(input, labels, partner, lambda);
    const auto si = sampled.images.to_vector<float>(), sl = sampled.labels.to_vector<float>();
    check::Near(si[0], .2 * lambda + .8 * (1 - lambda), 1e-7, "runtime mixup weight");
    for (int row = 0; row < 2; ++row) check::Near(sl[2 * row] + sl[2 * row + 1], 1, 1e-7, "soft label probability mass");
  }
  check::Throws([&] { aug::mixup(input, labels, partner, 1.5); }, "mixup out-of-range coefficient must fail");
  check::Throws([&] { aug::cutmix(input, labels, partner, {-1, 2, -1, 4}); }, "negative rectangle size must fail");
  bool rejected = false;
  try {
    aug::mixup(input, labels, {1, 1}, .5);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  check::Require(rejected, "duplicate mixing permutation rejected");
  std::cout << "  mixup weight .25: image .65, labels=(.25,.75)\n"
               "  CutMix clipped area 4/16: labels=(.75,.25), every pasted pixel verified\n"
               "  20 seeded Beta(.4,.4) draws, label mass and invalid-parameter checks passed\n";
}

}  // namespace

int main() {
  try {
    nn::set_num_threads(2);
    std::cout << "== Spatial and color augmentation\n";
    Transformations();
    std::cout << "\n== Image/label mixing\n";
    Mixing();
    std::cout << "All augmentation checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
