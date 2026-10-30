#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace nn = dm::nn;
namespace m = nn::models;
using nn::Tensor;
using Box = nn::Box;
namespace check {
template <typename A, typename B> void Near(A a, B b, double t, const std::string &msg) {
  EXPECT_NEAR(double(a), double(b), t) << msg;
}
inline void Require(bool ok, const std::string &msg) {
  if (!ok)
    throw std::runtime_error(msg);
}
} // namespace check
namespace {
m::PanopticConfig Compact() {
  m::PanopticConfig c;
  c.size = 64;
  c.depth = 18;
  c.divisor = 8;
  c.width = 32;
  c.semantic_width = 16;
  c.groups = 8;
  c.semantic_classes = 3;
  c.grayscale = true;
  return c;
}

void RoutingChecks() {
  std::vector<Tensor> features;
  for (int i = 0; i < 4; ++i) {
    const int cells = 16 >> i;
    std::vector<float> values(std::size_t(2 * cells * cells));
    for (int n = 0; n < 2; ++n)
      for (int p = 0; p < cells * cells; ++p)
        values[std::size_t(n * cells * cells + p)] = float(10 * n + i + 1);
    features.push_back(nn::from_vector(values, {2, cells, cells, 1}));
    features.back().requires_grad_(true);
  }
  const std::vector<Box> boxes = {
      {0, 0, 8, 8}, {0, 0, 16, 16}, {0, 0, 32, 32}, {0, 0, 64, 64}};
  const Tensor pooled =
      m::PanopticRoiAlign(features, nn::boxes_from_vector(boxes),
                          nn::from_vector(std::vector<std::int64_t>{0, 0, 1, 1}), 64, 3, 32);
  const auto grads = nn::grad(nn::sum(pooled), features);
  const auto out = pooled.to_vector<float>();
  const double expected[] = {1, 2, 13, 14};
  for (int i = 0; i < 4; ++i)
    for (int pixel = 0; pixel < 9; ++pixel)
      check::Near(out[std::size_t(i * 9 + pixel)], expected[i], 2e-6,
                  "hand-picked scales route to P2/P3/P4/P5");
  for (int i = 0; i < 4; ++i) {
    const int cells = 16 >> i;
    const auto g = grads[std::size_t(i)].to_vector<float>();
    for (int n = 0; n < 2; ++n) {
      double total = 0;
      for (int pixel = 0; pixel < cells * cells; ++pixel)
        total += g[std::size_t(n * cells * cells + pixel)];
      check::Near(total, n == (i < 2 ? 0 : 1) ? 9 : 0, 3e-5,
                  "aligned Jacobian conserves pooled weight only for selected image/level");
    }
  }
  std::cout
      << "  four pyramid RoI levels, image indices, border-aligned values and conserved/zero "
         "feature Jacobians passed"
      << std::endl;
}

/// One forward pass: pyramid/semantic features and the RPN; box and mask heads on demand.
struct Model {
  nn::ParameterStore &p;
  m::PanopticConfig cfg;
  bool compact;
  m::PanopticFeatures features;
  m::Rpn rpn;
  Model(nn::ParameterStore &params, nn::ParameterStore &state, const Tensor &images,
        bool compact_model, bool training, bool update_stats = true)
      : p(params), cfg(compact_model ? Compact() : m::PanopticConfig{}),
        compact(compact_model) {
    features = m::PanopticBackbone(p, state, images, cfg, training, update_stats);
    std::vector<m::RpnLevel> levels;
    for (int i = 0; i < 5; ++i)
      levels.push_back({features.pyramid[std::size_t(i)], cfg.width, cfg.size / (4 << i), 3});
    rpn = m::BuildRpn(p, levels, compact ? 32 : 256);
  }
  [[nodiscard]] int Classes() const {
    return compact ? 10 : 80;
  }
  m::DetectionRoiOutputs Head(const std::vector<Box> &rois,
                                const std::vector<int> &indices) const {
    return m::PooledDetectionRoiHead(
        p,
        m::PanopticRoiAlign(
            features.pyramid, nn::boxes_from_vector(rois),
            nn::from_vector(std::vector<std::int64_t>(indices.begin(), indices.end())),
            cfg.size, 7, compact ? 32.f : 224.f),
        Classes() + 1, compact ? 128 : 1024);
  }
  Tensor MaskLogits(const std::vector<Box> &boxes, const std::vector<int> &indices) const {
    return m::MaskRcnnHead(p,
                           m::PanopticRoiAlign(features.pyramid, nn::boxes_from_vector(boxes),
                                               nn::from_vector(std::vector<std::int64_t>(
                                                   indices.begin(), indices.end())),
                                               cfg.size, 14, compact ? 32.f : 224.f),
                           compact ? 16 : 256, Classes());
  }
};

void ModelChecks() {
  for (bool compact : {false, true}) {
    nn::ParameterStore p(17), state(18);
    const int side = compact ? 64 : 256, batch = compact ? 2 : 1;
    std::vector<float> pixels(std::size_t(batch * side * side * (compact ? 1 : 3)));
    for (std::size_t i = 0; i < pixels.size(); ++i)
      pixels[i] = std::sin(.13f * float(i));
    const Model net(p, state, nn::from_vector(pixels, {batch, side, side, compact ? 1 : 3}),
                    compact, true);
    const float s = compact ? 1.f : 4.f;
    const std::vector<Box> boxes = {{0, 0, 8 * s, 8 * s},
                                    {0, 0, 16 * s, 16 * s},
                                    {0, 0, 32 * s, 32 * s},
                                    {0, 0, 64 * s, 64 * s}};
    const std::vector<int> indices =
        compact ? std::vector<int>{0, 0, 1, 1} : std::vector<int>{0, 0, 0, 0};
    const auto head = net.Head(boxes, indices);
    const Tensor masks = net.MaskLogits(boxes, indices);
    check::Require(p.count() == (compact ? 489751 : 46027110),
                   "independent Panoptic FPN backbone/semantic/RPN/box/mask parameter census");
    Tensor objective = nn::scalar(0.0);
    for (const auto &output : {net.features.semantic, net.rpn.objectness, net.rpn.deltas,
                               head.probabilities, head.deltas, masks})
      objective = objective + nn::mean(nn::square(output));
    const auto grads = nn::grad(objective, p.parameters());
    check::Require(grads.size() == std::size_t(compact ? 125 : 224),
                   "Panoptic FPN independent parameter tensor count");
    double low = INFINITY, high = 0;
    for (std::size_t k = 0; k < grads.size(); ++k) {
      const double norm =
          std::sqrt(nn::sum(nn::square(grads[k].to(nn::DType::Float64))).item<double>());
      check::Require(std::isfinite(norm) && (!compact || norm > 0),
                     "finite nonzero panoptic derivative: " + p.names()[k]);
      low = std::min(low, norm);
      high = std::max(high, norm);
    }
    if (!compact) {
      std::cout << "  full ResNet50/FPN/GN32/RPN/box/mask:" << p.count() << " parameters, all"
                << grads.size() << " derivatives (executed on one 256x256 image)" << std::endl;
      continue;
    }
    check::Require(net.features.semantic.shape() == nn::Shape({2, 64, 64, 3}) &&
                       net.rpn.objectness.shape() == nn::Shape({2, 1023}) &&
                       masks.shape() == nn::Shape({4, 28, 28, 10}),
                   "compact semantic/RPN/instance shape alignment");
    std::cout << "  compact Panoptic FPN:" << p.count() << " parameters, all" << grads.size()
              << " derivatives evaluated; norms [" << low << ',' << high << ']' << std::endl;
  }
}

m::TensorPanopticMap Map(std::initializer_list<std::int64_t> ids,
                         std::map<int, m::PanopticSegment> metadata) {
  return {nn::from_vector(std::vector<std::int64_t>(ids)), metadata};
}
} // namespace
TEST(NnPanoptic, RoutedFourLevelValuesImageIsolationAndEveryFeatureJacobian) {
  RoutingChecks();
}
TEST(NnPanoptic, CompleteFull224GradientsCompact125NonzeroGradientsAndCensus) {
  nn::set_num_threads(2);
  ModelChecks();
}
TEST(NnPanoptic, OverlapEqualityTrimRejectionStuffAreaAndScoreOrder) {
  const auto semantic =
      nn::from_vector(std::vector<std::int64_t>{1, 1, 1, 2, 1, 0, 0, 2, 1, 0, 2, 2}, {3, 4});
  std::vector<int> masks(48);
  for (int pixel : {0, 1, 4, 5})
    masks[pixel] = 1;
  for (int pixel : {1, 2, 5, 6})
    masks[12 + pixel] = 1;
  for (int pixel : {0, 1, 4})
    masks[24 + pixel] = 1;
  masks[47] = 1;
  m::TensorMaskDetections d{{nn::zeros({4, 4}), nn::tensor({.9, .8, .7, .49}),
                             nn::from_vector(std::vector<std::int64_t>{3, 4, 5, 6}),
                             nn::zeros({4}, nn::DType::Int64)},
                            nn::not_equal(nn::from_vector(masks, {4, 3, 4}), 0.)};
  const auto result = m::MergeTensorPanoptic(semantic, d, {-1, 10, 11}, .5, .5, 2);
  EXPECT_EQ(result.ids.to_vector<int>(),
            (std::vector<int>{1, 1, 2, 3, 1, 1, 2, 3, 0, 0, 3, 3}));
  EXPECT_EQ(result.segments.size(), 3);
  EXPECT_EQ(result.segments.at(1).label, 3);
  EXPECT_EQ(result.segments.at(2).label, 4);
  EXPECT_EQ(result.segments.at(3).label, 11);
  d.objects.scores = nn::tensor({.9, .8, .7, .5});
  const auto equality = m::MergeTensorPanoptic(semantic, d, {-1, 10, 11}, .5, .5, 2);
  const auto id = equality.ids.to_vector<int>()[11];
  EXPECT_TRUE(equality.segments.at(id).thing);
}
TEST(NnPanoptic, HandCountedPqStrictIouVoidCrowdAndCategoryMacroAverage) {
  const auto things = nn::from_vector(std::vector<bool>{true, true, false});
  auto truth = Map({1, 1, 1, 2, 2, 2}, {{1, {0, true, false}}, {2, {0, true, false}}}),
       pred = Map({3, 3, 3, 4, 0, 0}, {{3, {0, true, false}}, {4, {0, true, false}}});
  auto stats = m::PanopticStatistics(truth, pred, 3);
  auto q = m::PanopticQuality(stats, things);
  EXPECT_EQ(stats.to_vector<double>(),
            (std::vector<double>{1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0}));
  EXPECT_DOUBLE_EQ(q.pq.item(), .5);
  EXPECT_DOUBLE_EQ(q.sq.item(), 1);
  EXPECT_DOUBLE_EQ(q.rq.item(), .5);
  truth = Map({1, 1}, {{1, {0, true, false}}});
  pred = Map({2, 0}, {{2, {0, true, false}}});
  EXPECT_EQ(m::PanopticStatistics(truth, pred, 3).to_vector<double>()[1], 0);
  truth = Map({1, 1, 0}, {{1, {0, true, false}}});
  pred = Map({7, 7, 7}, {{7, {0, true, false}}});
  EXPECT_DOUBLE_EQ(m::PanopticQuality(m::PanopticStatistics(truth, pred, 3), things).pq.item(),
                   1);
  truth = Map({1, 0, 0}, {{1, {0, true, false}}});
  pred = Map({0, 7, 7}, {{7, {1, true, false}}});
  stats = m::PanopticStatistics(truth, pred, 3);
  EXPECT_EQ(stats.to_vector<double>()[3], 1);
  EXPECT_EQ(stats.to_vector<double>()[6], 0);
  truth = Map({1, 1, 2, 2}, {{1, {0, true, true}}, {2, {2, false, false}}});
  pred = Map({7, 7, 8, 8}, {{7, {0, true, false}}, {8, {2, false, false}}});
  stats = m::PanopticStatistics(truth, pred, 3);
  EXPECT_DOUBLE_EQ(m::PanopticQuality(stats, things).pq.item(), 1);
  EXPECT_EQ(m::PanopticQuality(stats, things, 1).classes.item<std::int64_t>(), 0);
  truth = Map({1, 1, 2, 2}, {{1, {0, true, false}}, {2, {2, false, false}}});
  pred = Map({7, 7, 0, 0}, {{7, {0, true, false}}});
  stats = m::PanopticStatistics(truth, pred, 3);
  EXPECT_DOUBLE_EQ(m::PanopticQuality(stats, things).pq.item(), .5);
  EXPECT_DOUBLE_EQ(m::PanopticQuality(stats + stats, things).pq.item(), .5);
}
TEST(NnPanoptic, AllVoidEmptyPredictionAndInvalidMetadata) {
  const auto empty = Map({0, 0}, {}), truth = Map({1, 1}, {{1, {0, true, false}}});
  const auto things = nn::ones({1}, nn::DType::Bool);
  EXPECT_EQ(m::PanopticStatistics(truth, empty, 1).to_vector<double>(),
            (std::vector<double>{0, 0, 0, 1}));
  EXPECT_EQ(m::PanopticQuality(m::PanopticStatistics(empty, empty, 1), things)
                .classes.item<std::int64_t>(),
            0);
  const auto pred = Map({2, 2}, {{2, {0, true, false}}});
  EXPECT_EQ(m::PanopticStatistics(empty, pred, 1).to_vector<double>(),
            (std::vector<double>{0, 0, 0, 0}));
  EXPECT_THROW(m::PanopticAreas(Map({9}, {})), std::invalid_argument);
  EXPECT_THROW(m::PanopticAreas(Map({0}, {{1, {0, true, false}}})), std::invalid_argument);
}
TEST(NnPanoptic, RetainedSupportsPhotometryMaterialTargetsAndReorderedSemanticRows) {
  const auto iy = nn::arange(0, 28).reshape({28, 1}), ix = nn::arange(0, 28).reshape({1, 28});
  const auto support =
      nn::logical_and(nn::logical_and(nn::greater_equal(iy, 6.), nn::less(iy, 22.)),
                      nn::logical_and(nn::greater_equal(ix, 6.), nn::less(ix, 22.)));
  const auto digits = support.to(nn::DType::Float32).reshape({1, 28, 28, 1}) * (200. / 255.);
  const auto data =
      m::MakeTensorPanopticScenes(digits, nn::zeros({1}, nn::DType::Int64), 4, 11);
  m::DetectionSceneConfig cfg;
  cfg.scenes = 4;
  cfg.seed = 11;
  cfg.max_overlap = 0;
  const auto original =
      m::MakeTensorInstanceScenes(digits, nn::zeros({1}, nn::DType::Int64), cfg);
  const auto pixels = data.instances.images.to_vector<float>(),
             old = original.images.to_vector<float>();
  const auto semantic = data.semantic.to_vector<int>();
  for (int row = 0; row < 4; ++row) {
    const auto masks = data.instances.masks[row].to_vector<bool>();
    const auto ids = data.panoptic[row].ids.to_vector<int>();
    for (int pixel = 0; pixel < 4096; ++pixel) {
      int object = -1;
      for (std::int64_t k = 0; k < data.instances.labels[row].numel(); ++k)
        if (masks[k * 4096 + pixel])
          object = int(k);
      const auto &metadata = data.panoptic[row].segments.at(ids[pixel]);
      if (object >= 0) {
        EXPECT_EQ(semantic[row * 4096 + pixel], 0);
        EXPECT_TRUE(metadata.thing);
        EXPECT_FLOAT_EQ(pixels[row * 4096 + pixel], old[row * 4096 + pixel]);
      } else {
        const int label = semantic[row * 4096 + pixel], base = label == 1 ? 40 : 100;
        EXPECT_TRUE(label == 1 || label == 2);
        EXPECT_FALSE(metadata.thing);
        EXPECT_EQ(metadata.label, label + 9);
        EXPECT_NEAR(pixels[row * 4096 + pixel] * 255,
                    base + (13 * (pixel / 64) + 7 * (pixel % 64) + 3 * row) % 11, 2e-5);
      }
    }
  }
  const auto rows = nn::from_vector(std::vector<std::int64_t>{3, 1});
  EXPECT_EQ(nn::gather(data.semantic, rows).shape(), nn::Shape({2, 64, 64}));
}
