// Independent numeric and model-geometry checks; experiment driver omitted.
#pragma once
#include "nn_interpretation_check_support.hpp"
namespace inpaint {
using namespace dm::nn::models;
}
namespace denoise {
using namespace dm::nn::models;
}
namespace flow {
using namespace dm::nn::models;
}
namespace color {
using namespace dm::nn::models;
}
namespace depth {
using namespace dm::nn::models;
}
namespace pose {
using namespace dm::nn::models;
}
namespace style {
using namespace dm::nn::models;
}
namespace crnn {
using namespace dm::nn::models;
}
namespace ocr {
using namespace dm::nn::models;
}
namespace cifar {
using namespace dm::nn::models;
}

#include <cmath>
#include <iostream>
namespace nn = datamunge::nn;
using nn::Tensor;
using namespace pose;

namespace {

/// Mutable host fixture: rgb zeros, joints [B,2,18,3] zeros, boxes 8, counts `people`.
struct Fixture {
  int                batch;
  std::vector<float> joints, boxes;
  std::vector<int>   counts;
  explicit Fixture(int b = 1, int people = 2)
      : batch(b),
        joints(std::size_t(b) * 2 * 18 * 3, 0.f),
        boxes(std::size_t(b) * 2 * 4, 8.f),
        counts(std::size_t(b), people) {}
  float& J(int n, int p, int j, int c) {
    return joints[std::size_t(((n * 2 + p) * 18 + j) * 3 + c)];
  }
  float&                B(int n, int p, int c) { return boxes[std::size_t((n * 2 + p) * 4 + c)]; }
  [[nodiscard]] PafData Data() const {
    return {
        nn::zeros({batch, 128, 128, 3}),
        nn::from_vector(joints, {batch, 2, 18, 3}),
        nn::from_vector(boxes, {batch, 2, 4}),
        nn::from_vector(counts, {batch}),
        nn::zeros({batch}, nn::DType::Int32)};
  }
};

float At(
    const std::vector<float>& v,
    const Tensor&             t,
    std::int64_t              n,
    std::int64_t              y,
    std::int64_t              x,
    std::int64_t              c) {
  return v[std::size_t(((n * t.size(1) + y) * t.size(2) + x) * t.size(3) + c)];
}

void TargetChecks() {
  std::vector<float> coco(17 * 3, NAN);
  for (int j = 0; j < 17; ++j)
    coco[std::size_t(j * 3 + 2)] = 0;
  for (int j : {5, 6}) {
    coco[std::size_t(j * 3)]     = j == 5 ? 2 : 6;
    coco[std::size_t(j * 3 + 1)] = 4;
    coco[std::size_t(j * 3 + 2)] = j == 5 ? 1 : 2;
  }
  auto neck = Coco18(nn::from_vector(coco, {1, 1, 17, 3})).to_vector<float>();
  check::Near(neck[1 * 3], 4, 0, "neck is shoulder midpoint");
  check::Near(neck[1 * 3 + 2], 1, 0, "neck visibility is min labeled shoulder flag");
  check::Near(neck[2 * 3], 6, 0, "right shoulder channel mapping");
  check::Near(neck[5 * 3], 2, 0, "left shoulder channel mapping");
  coco[6 * 3 + 2] = 0;
  neck            = Coco18(nn::from_vector(coco, {1, 1, 17, 3})).to_vector<float>();
  check::Near(neck[1 * 3 + 2], 0, 0, "unlabeled shoulder prevents invented neck");
  for (float v : neck)
    check::Require(std::isfinite(v), "unknown source NaNs not copied to new joints");
  Fixture data;
  for (int person = 0; person < 2; ++person) {
    for (int c = 0; c < 2; ++c)
      data.B(0, person, c) = 3.5;
    for (int joint : {1, 2}) {
      data.J(0, person, joint, 0) = ((person == 0) == (joint == 1)) ? 1 : 5;
      data.J(0, person, joint, 1) = 2;
      data.J(0, person, joint, 2) = person ? 1 : 2;
    }
  }
  data.J(0, 0, 3, 0) = 5, data.J(0, 0, 3, 1) = 5, data.J(0, 0, 3, 2) = 2;
  data.J(0, 1, 0, 0) = NAN, data.J(0, 1, 0, 1) = NAN;
  const auto target = PreparePafTargets(data.Data(), 8, 8, .5, 1);
  const auto hm = target.heatmap.to_vector<float>(), hk = target.heatmask.to_vector<float>(),
             pf = target.paf.to_vector<float>(), pk = target.pafmask.to_vector<float>();
  for (int y = 0; y < 8; ++y)
    for (int x = 0; x < 8; ++x) {
      double maximum = 0;
      for (int j = 0; j < 18; ++j) {
        double expected = 0;
        if (j == 1 || j == 2)
          for (int center : {1, 5})
            if (std::abs(x - center) <= 1.5 && std::abs(y - 2) <= 1.5)
              expected = std::max(
                  expected, std::exp(-((x - center) * (x - center) + (y - 2) * (y - 2)) / .5));
        if (j == 3 && std::abs(x - 5) <= 1.5 && std::abs(y - 5) <= 1.5)
          expected = std::exp(-((x - 5) * (x - 5) + (y - 5) * (y - 5)) / .5);
        maximum = std::max(maximum, expected);
        check::Near(
            At(hm, target.heatmap, 0, y, x, j),
            expected,
            2e-8,
            "all multiperson Gaussian pixels use max not sum");
        check::Near(
            At(hk, target.heatmask, 0, y, x, j),
            j == 1 || j == 2 ? 1 : 0,
            0,
            "every conservative missing-person joint mask");
      }
      check::Near(
          At(hm, target.heatmap, 0, y, x, 18),
          1 - maximum,
          3e-8,
          "background is one minus maximum over all body parts");
      check::Near(
          At(hk, target.heatmask, 0, y, x, 18),
          0,
          0,
          "unknown joints suppress background negatives");
      for (int e = 0; e < 19; ++e)
        for (int c = 0; c < 2; ++c) {
          const double expected = e == 2 && c == 1 && y >= 2 && y <= 5 && x >= 4 && x <= 6 ? 1 : 0;
          check::Near(
              At(pf, target.paf, 0, y, x, 2 * e + c),
              expected,
              0,
              "every directed limb-strip pixel;opposite overlapping people average to zero");
          check::Near(
              At(pk, target.pafmask, 0, y, x, 2 * e + c),
              e == 0 ? 1 : 0,
              0,
              "unknown endpoint masks both PAF components");
        }
    }
  // A non-cancelling crossing distinguishes averaging from summation.
  for (int j : {1, 2})
    data.J(0, 1, j, 0) = 3, data.J(0, 1, j, 1) = j == 1 ? 0 : 4;
  const auto crossing = PreparePafTargets(data.Data(), 8, 8, .5, 1);
  const auto cp       = crossing.paf.to_vector<float>();
  for (int c = 0; c < 2; ++c)
    check::Near(
        At(cp, crossing.paf, 0, 2, 3, c),
        .5,
        0,
        "perpendicular overlap is mean(.5,.5),not sum(1,1)");
  data.J(0, 0, 1, 0) = 1, data.J(0, 0, 1, 1) = 1, data.J(0, 0, 2, 0) = 4, data.J(0, 0, 2, 1) = 5;
  const auto diagonal = PreparePafTargets(data.Data(), 8, 8, .5, 1);
  const auto dp       = diagonal.paf.to_vector<float>();
  check::Near(At(dp, diagonal.paf, 0, 1, 1, 0), .6, 3e-8, "known3/4/5 limb dx normalized to.6");
  check::Near(At(dp, diagonal.paf, 0, 1, 1, 1), .8, 3e-8, "known3/4/5 limb dy normalized to.8");
  // Zero-length annotated limbs never create a division by zero.
  data.J(0, 0, 2, 0) = 1, data.J(0, 0, 2, 1) = 1;
  for (float v : PreparePafTargets(data.Data(), 8, 8, .5, 1).paf.to_vector<float>())
    check::Require(std::isfinite(v), "degenerate limb finite field");
  std::cout << " COCO17->18 neck/mapping/unknown-NaN,all multiperson "
               "Gaussian/background/strip/mask values,opposite/perpendicular overlap "
               "averaging,3/4/5 unit vector,zero-length limb checks passed\n";
}

void LossChecks() {
  std::vector<float> prediction(24, NAN), truth(24, NAN), mask(24, 0.f);
  for (int n = 0; n < 2; ++n)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
        for (int c = 0; c < (n ? 1 : 2); ++c) {
          const std::size_t i = std::size_t(((n * 2 + y) * 2 + x) * 2 + c);
          mask[i] = 1, prediction[i] = n ? 3 : 2, truth[i] = 0;
        }
  Tensor     p = nn::from_vector(prediction, {3, 2, 2, 2}).requires_grad_();
  const auto loss =
      PafMaskedMSE(p, nn::from_vector(truth, {3, 2, 2, 2}), nn::from_vector(mask, {3, 2, 2, 2}));
  const auto g = nn::grad(loss, {p})[0].to_vector<float>();
  check::Near(loss.item(), 6.5, 0, "equal nonempty image weighting");
  for (int n = 0; n < 3; ++n)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
        for (int c = 0; c < 2; ++c)
          check::Near(
              g[std::size_t(((n * 2 + y) * 2 + x) * 2 + c)],
              n == 0             ? .25
              : n == 1 && c == 0 ? .75
                                 : 0,
              0,
              "every signed masked field derivative");
  std::cout << " independently known per-image field loss/every coordinate derivative,unequal "
               "counts/empty images/NaN masks passed\n";
}

void DecoderChecks() {
  Fixture data;
  for (int person = 0; person < 2; ++person) {
    data.B(0, person, 0) = 63.5, data.B(0, person, 1) = 63.5, data.B(0, person, 2) = 128,
                      data.B(0, person, 3) = 128;
    for (int j : {1, 2, 3}) {
      const float gx          = j == 1   ? 2
                                : j == 2 ? 6
                                         : 8,
                  gy          = float((person ? 11 : 3) + (j == 3 ? 2 : 0));
      data.J(0, person, j, 0) = (gx + .5f) * 8 - .5f;
      data.J(0, person, j, 1) = (gy + .5f) * 8 - .5f;
      data.J(0, person, j, 2) = 2;
    }
  }
  const auto fixture = data.Data();
  const auto target  = PreparePafTargets(fixture, 16, 128, .5, .75);
  const auto decoded = DecodePafs(target.heatmap, target.paf, 0);
  check::Require(
      decoded.peaks.size() == 6 && decoded.links.size() == 4 && decoded.people.size() == 2,
      "two real disconnected skeletons survive local NMS/directed line integration/one-to-one "
      "matching/group assembly");
  for (const auto& person : decoded.people) {
    check::Require(
        person.joints[1] >= 0 && person.joints[2] >= 0 && person.joints[3] >= 0,
        "both chains retain all3 joints");
    check::Near(
        decoded.peaks[std::size_t(person.joints[1])].y,
        decoded.peaks[std::size_t(person.joints[2])].y,
        0,
        "no cross-person horizontal-to-diagonal splice");
  }
  const auto score = ScorePafs({decoded}, fixture);
  check::Require(
      score.correct == 4 && score.annotated == 4 && score.predicted == 4 && score.people == 2,
      "perfect custom grouped annotated-joint metric excludes derived neck");
  const auto wrong = DecodePafs(target.heatmap, -target.paf, 0);
  check::Require(
      wrong.links.empty() && wrong.people.empty(),
      "reversing directed limb vectors rejects association");
  PafTensorField empty(nn::zeros(target.heatmap.shape()));
  check::Require(
      DecodePafs(empty, PafTensorField(target.paf), 0).people.empty(),
      "no peaks yields no invented people");
  auto plateau                          = empty.values.to_vector<float>();
  plateau[((0 * 16 + 1) * 16 + 1) * 19] = 1;
  plateau[((0 * 16 + 1) * 16 + 2) * 19] = 1;
  empty.values                          = nn::from_vector(plateau, target.heatmap.shape());
  const auto peaks                      = PafPeaks(empty, 0);
  check::Require(
      peaks.size() == 1 && peaks[0].x == 1, "NMS plateau ties choose first spatial pixel");
  std::vector<float> linear(4 * 4 * 38);
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 4; ++x)
      for (int c = 0; c < 38; ++c)
        linear[std::size_t((y * 4 + x) * 38 + c)] = float(x + 2 * y + c);
  const PafTensorField lf(nn::from_vector(linear, {1, 4, 4, 38}));
  check::Near(
      PafSample(lf, 0, 1.25, 2.5, 0),
      6.25,
      0,
      "bilinear line sample on independently affine field");
  check::Near(PafSample(lf, 0, -3, 9, 0), 6, 0, "line sample clamps borders");
  const std::vector<PafPeak> conflict = {
      {1, 0, 0, 1}, {2, 1, 0, 1}, {1, 0, 1, 1}, {2, 1, 1, 1}, {3, 2, 1, 1}};
  const std::vector<PafLink> links = {{0, 1, 0, 1}, {2, 3, 0, 1}, {3, 4, 2, 1}, {1, 4, 2, .9f}};
  check::Require(
      AssemblePafPeople(conflict, links, 2).size() == 2,
      "conflicting same-type peaks forbid merging two person subsets");
  // Two candidates sharing a start peak cannot both be selected for one limb.
  std::vector<float> unit(16 * 16 * 38, 0.f);
  for (std::size_t q = 0; q < 256; ++q)
    unit[q * 38] = 1;
  const auto matched = MatchPafPeaks(
      {{1, 2, 3, 1}, {2, 5, 3, 1}, {2, 6, 3, 1}},
      PafTensorField(nn::from_vector(unit, {1, 16, 16, 38})),
      0);
  check::Require(matched.size() == 1, "greedy limb matching enforces one-to-one endpoint use");
  std::cout << " two-person oracle decode,peak NMS/tie,bilinear samples,directed "
               "support,one-to-one matches,conflict-safe grouping and "
               "perfect/empty metrics passed\n";
}

std::int64_t Census(int divisor) {
  std::int64_t count = 0;
  auto         conv  = [&](int in, int out, int kernel) {
    count += std::int64_t(in) * out * kernel * kernel + out;
  };
  int       in       = 3;
  const int widths[] = {64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 256, 128};
  for (int width : widths) {
    conv(in, width / divisor, 3);
    in = width / divisor;
  }
  for (int stage = 0; stage < 6; ++stage)
    for (int branch = 0; branch < 2; ++branch) {
      in               = 128 / divisor + (stage ? 57 : 0);
      const int layers = stage ? 7 : 5;
      for (int layer = 0; layer < layers; ++layer) {
        const int out    = layer + 1 == layers ? (branch ? 19 : 38)
                                               : (stage == 0 && layer == 3 ? 512 : 128) / divisor;
        const int kernel = stage ? (layer < 5 ? 7 : 1) : (layer < 3 ? 3 : 1);
        conv(in, out, kernel);
        in = out;
      }
    }
  return count;
}

void ModelChecks() {
  for (int divisor : {1, 8}) {
    nn::ParameterStore p(197);
    std::vector<float> values(std::size_t(2) * 128 * 128 * 3);
    for (std::size_t i = 0; i < values.size(); ++i)
      values[i] = float(.5 + .4 * std::sin(double(i) * .13));
    Tensor              image = nn::from_vector(values, {2, 128, 128, 3}).requires_grad_();
    const auto          net   = OpenPose(p, image, divisor);
    std::vector<Tensor> terms;
    for (std::size_t stage = 0; stage < 6; ++stage) {
      terms.push_back(nn::mean(nn::square(net.heatmaps[stage])));
      terms.push_back(nn::mean(nn::square(net.pafs[stage])));
    }
    Tensor objective = terms[0];
    for (std::size_t i = 1; i < terms.size(); ++i)
      objective = objective + terms[i];
    auto inputs = p.parameters();
    inputs.push_back(image);
    nn::GradOptions retain;
    retain.retain_graph  = true;
    const auto gradients = nn::grad(objective, inputs, retain);
    const auto feedback  = nn::grad(terms[10] + terms[11], {net.heatmaps[0], net.pafs[0]});
    check::Require(
        p.count() == Census(divisor) && p.size() == 184 && net.stages.size() == 100,
        "independent original OpenPose parameter count/all92 convs and100 stages");
    std::cout << " " << (divisor == 1 ? "full" : "compact") << " OpenPose:" << p.count()
              << " parameters/184 derivatives,100 stages,BOTH initial branches reach final stage "
                 "(executed)\n";
    if (divisor == 1)
      continue;
    std::vector<nn::Shape> expected;
    std::int64_t           side       = 128;
    const int              channels[] = {64, 64, 128, 128, 256, 256, 256, 256, 512, 512, 256, 128};
    for (int layer = 0; layer < 12; ++layer) {
      expected.push_back({2, side, side, channels[layer] / 8});
      if (layer == 1 || layer == 3 || layer == 7) {
        side /= 2;
        expected.push_back({2, side, side, channels[layer] / 8});
      }
    }
    for (int stage = 0; stage < 6; ++stage) {
      if (stage)
        expected.push_back({2, 16, 16, 73});
      for (int branch = 0; branch < 2; ++branch) {
        const int layers = stage ? 7 : 5;
        for (int layer = 0; layer < layers; ++layer)
          expected.push_back(
              {2,
               16,
               16,
               layer + 1 == layers ? (branch ? 19 : 38) : (stage == 0 && layer == 3 ? 64 : 16)});
      }
    }
    for (std::size_t i = 0; i < 100; ++i)
      check::Require(
          net.stages[i].shape() == expected[i],
          "every exact backbone/pool/concat/refinement/branch head geometry");
    double              minimum = INFINITY, maximum = 0;
    std::vector<Tensor> all = gradients;
    all.insert(all.end(), feedback.begin(), feedback.end());
    for (const auto& g : all) {
      double norm = 0;
      for (float v : g.to_vector<float>()) {
        check::Require(std::isfinite(v), "finite every model/feedback/RGB derivative");
        norm += double(v) * v;
      }
      norm = std::sqrt(norm);
      check::Require(
          norm > 0, "all184 parameter tensors,BOTH branch feedbacks and RGB receive gradients");
      minimum = std::min(minimum, norm), maximum = std::max(maximum, norm);
    }
    std::cout
        << " compact all184 parameter+2branch-feedback+RGB gradients finite/nonzero,norm range["
        << minimum << "," << maximum << "],all100 exact stage geometries passed\n";
  }
}

} // namespace
