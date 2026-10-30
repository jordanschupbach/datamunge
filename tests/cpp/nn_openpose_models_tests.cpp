
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#define main original_openpose_main
#include "../../examples/cpp/source/nn/1_convolutional_neural_networks/1.8_other_vision_tasks/openpose_pafs.cpp"
#undef main
#include "../fixtures/nn_paf_reference.hpp"
TEST(OpenPose, SourceGaussianStripsOverlapUnknownMasks) {
  TargetChecks();
}
TEST(OpenPose, SourceMaskedLossAndAllAdjoints) {
  LossChecks();
}
TEST(OpenPose, SourceDirectedAssociationAndConflictSafeAssembly) {
  DecoderChecks();
}
TEST(OpenPose, FullCompactAllStagesAndEveryLearnedFeedbackDerivative) {
  nn::set_num_threads(2);
  ModelChecks();
}
TEST(OpenPose, EveryRealTargetAndIndependentSourceDecoder) {
  namespace m = dm::nn::models;
  namespace r = paf_reference;
  nn::set_num_threads(2);
  const auto split = m::LoadPafImages("build/cnn-datasets/human_pose");
  for (const auto &data : {split.fit, split.validation, split.test}) {
    const r::PafData source{data.rgb, data.joints, data.boxes, data.counts, data.image_ids};
    for (int side : {7, 16}) {
      const auto a = m::PreparePafTargets(data, side);
      const auto b = r::PreparePafTargets(source, side);
      for (const auto &pair :
           std::vector<std::pair<nn::Tensor, nn::Tensor>>{{a.heatmap, b.heatmap},
                                                          {a.paf, b.paf},
                                                          {a.heatmask, b.heatmask},
                                                          {a.pafmask, b.pafmask}})
        EXPECT_LE(nn::max(nn::abs(pair.first - pair.second)).item(), 2e-7);
      std::vector<m::PafDecoded> decoded;
      std::vector<r::PafDecoded> reference;
      for (int row = 0; row < data.rgb.size(0); ++row) {
        const auto x = m::DecodePafs(a.heatmap, a.paf, row);
        const auto y = r::DecodePafs(a.heatmap, a.paf, row);
        ASSERT_EQ(x.peaks.size(), y.peaks.size());
        ASSERT_EQ(x.links.size(), y.links.size());
        ASSERT_EQ(x.people.size(), y.people.size());
        for (std::size_t k = 0; k < x.peaks.size(); ++k) {
          EXPECT_EQ(x.peaks[k].joint, y.peaks[k].joint);
          EXPECT_FLOAT_EQ(x.peaks[k].x, y.peaks[k].x);
          EXPECT_FLOAT_EQ(x.peaks[k].y, y.peaks[k].y);
          EXPECT_FLOAT_EQ(x.peaks[k].score, y.peaks[k].score);
        }
        for (std::size_t k = 0; k < x.links.size(); ++k) {
          EXPECT_EQ(x.links[k].a, y.links[k].a);
          EXPECT_EQ(x.links[k].b, y.links[k].b);
          EXPECT_EQ(x.links[k].edge, y.links[k].edge);
          EXPECT_NEAR(x.links[k].score, y.links[k].score, 2e-7);
        }
        for (std::size_t k = 0; k < x.people.size(); ++k) {
          EXPECT_EQ(x.people[k].joints, y.people[k].joints);
          EXPECT_EQ(x.people[k].links, y.people[k].links);
        }
        decoded.push_back(x);
        reference.push_back(y);
      }
      const auto x = m::ScorePafs(decoded, data, side);
      const auto y = r::ScorePafs(reference, source, side);
      EXPECT_EQ(x.correct, y.correct);
      EXPECT_EQ(x.predicted, y.predicted);
      EXPECT_EQ(x.annotated, y.annotated);
      EXPECT_EQ(x.people, y.people);
      EXPECT_EQ(x.true_people, y.true_people);
    }
  }
}
TEST(OpenPose, TensorReferenceSgdAndMissingNaNSafeMetrics) {
  namespace m = dm::nn::models;
  const auto f = nn::tensor({1., NAN, 3., NAN}, {2, 1, 1, 2}),
             mask = nn::tensor({1., 0., 1., 0.}, {2, 1, 1, 2});
  nn::ParameterStore p(3, nn::DType::Float64);
  const auto field = p.get("field", {1, 1, 1, 2}, nn::init::zeros());
  nn::OptimizerConfig c;
  c.kind = nn::OptimizerKind::Sgd;
  nn::Optimizer optimizer(p, .5, c);
  const auto active = nn::greater(mask, .5),
             count = nn::sum(active.to(nn::DType::Float64), {0}, true),
             diff = nn::where(active, field, 0.) -
                    nn::where(active, f.to(nn::DType::Float64), 0.);
  const auto loss =
      nn::sum(nn::sum(nn::square(diff), {0}, true) / nn::maximum(count, nn::ones_like(count)));
  optimizer.step(nn::grad(loss, p.parameters()));
  EXPECT_EQ(
      nn::max(nn::abs(field.to(nn::DType::Float32) - m::PafMeanField(f, mask, 1))).item(), 0.);
  EXPECT_EQ(m::PafFieldMSE(nn::ones_like(f), f, mask).item(), 2.);
  EXPECT_EQ(m::PafFieldMSE(f, f, nn::zeros_like(mask)).item(), 0.);
}
TEST(OpenPose, EveryTensorPreviewPixelMatchesIndependentGeometry) {
  namespace m = dm::nn::models;
  nn::set_num_threads(2);
  const auto d = m::LoadPafImages("build/cnn-datasets/human_pose").test;
  const auto target = m::PreparePafTargets(d);
  std::vector<m::PafDecoded> dec;
  for (int n = 0; n < 4; ++n)
    dec.push_back(m::DecodePafs(target.heatmap, target.paf, n));
  const auto tiles = m::PafPreviewTiles(d, dec, target.heatmap, target.paf).to_vector<float>(),
             rgb = d.rgb.to_vector<float>(), j = d.joints.to_vector<float>(),
             heat = target.heatmap.to_vector<float>(), paf = target.paf.to_vector<float>();
  const auto counts = d.counts.to_vector<int>();
  auto near = [](double x, double y, double ax, double ay, double bx, double by) {
    double dx = bx - ax, dy = by - ay,
           t = std::clamp(((x - ax) * dx + (y - ay) * dy) / std::max(dx * dx + dy * dy, 1e-30),
                          0., 1.);
    return std::pow(x - ax - t * dx, 2) + std::pow(y - ay - t * dy, 2) <= 1.;
  };
  for (int n = 0; n < 4; ++n) {
    std::vector<std::array<double, 4>> gt, pred;
    for (int p = 0; p < counts[n]; ++p)
      for (const auto &e : m::kPoseEdges) {
        const int a = ((n * 8 + p) * 18 + e[0]) * 3, b = ((n * 8 + p) * 18 + e[1]) * 3;
        if (j[a + 2] > 0 && j[b + 2] > 0)
          gt.push_back({j[a], j[a + 1], j[b], j[b + 1]});
      }
    for (const auto &person : dec[n].people)
      for (int id : person.links) {
        const auto &link = dec[n].links[id];
        const auto &a = dec[n].peaks[link.a];
        const auto &b = dec[n].peaks[link.b];
        pred.push_back({(double(a.x) + .5) * 8 - .5, (double(a.y) + .5) * 8 - .5,
                        (double(b.x) + .5) * 8 - .5, (double(b.y) + .5) * 8 - .5});
      }
    for (int y = 0; y < 128; ++y)
      for (int x = 0; x < 128; ++x) {
        bool gm = false, pm = false;
        for (auto v : gt)
          gm |= near(x, y, v[0], v[1], v[2], v[3]);
        for (auto v : pred)
          pm |= near(x, y, v[0], v[1], v[2], v[3]);
        const int q = (n * 16 + y / 8) * 16 + x / 8;
        double h = 0, mag = 0;
        for (int k = 0; k < 18; ++k)
          h = std::max(h, double(heat[q * 19 + k]));
        for (int k = 0; k < 19; ++k)
          mag = std::max(
              mag, std::hypot(double(paf[q * 38 + 2 * k]), double(paf[q * 38 + 2 * k + 1])));
        for (int c = 0; c < 3; ++c) {
          const float raw = float(
              std::floor(std::clamp(double(rgb[((n * 128 + y) * 128 + x) * 3 + c]), 0., 1.) *
                             255 +
                         .5) /
              255.);
          const float values[] = {
              raw, gm ? float(c == 1) : raw, pm ? float(c == 0) : raw,
              float(c == 0 ? std::floor(std::clamp(h, 0., 1.) * 255 + .5) / 255. : 0.),
              float(c == 2 ? std::floor(std::clamp(mag, 0., 1.) * 255 + .5) / 255. : 0.)};
          for (int panel = 0; panel < 5; ++panel)
            EXPECT_FLOAT_EQ(tiles[((n * 128 + y) * 640 + panel * 128 + x) * 3 + c],
                            values[panel]);
        }
      }
  }
}
