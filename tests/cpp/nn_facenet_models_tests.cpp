#include "common/facenet.hpp"
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace nn = dm::nn;
using nn::Tensor;
namespace {
void NormalizationChecks() {
  const std::vector<float> v = {3, 4, 0, -1, 2, 2, 0, 0, 0, .001f, -.002f, .001f};
  std::vector<float> w(12);
  for (int i = 0; i < 12; ++i)
    w[i] = .3f + .1f * i;
  Tensor x = nn::from_vector(v, {4, 3}).requires_grad_();
  const Tensor weights = nn::from_vector(w, {4, 3});
  const Tensor normalized = metric::UnitEmbedding(x, 1e-4f);
  const Tensor gradient = nn::grad(nn::sum(normalized * weights), {x})[0];
  const std::vector<float> m = {1, 3, 4, 2, -2, -1, 5, 5, 0, -2, -1, -3},
                           g = {1, -2, .5, 3, 4, -1};
  Tensor maxin = nn::from_vector(m, {2, 6}).requires_grad_();
  const Tensor maxout = metric::MaxoutPairs(maxin, 3);
  const Tensor maxgrad = nn::grad(nn::sum(maxout * nn::from_vector(g, {2, 3})), {maxin})[0];
  const auto nv = normalized.to_vector<double>(), gv = gradient.to_vector<double>();
  double error = 0;
  for (int i = 0; i < 4; ++i) {
    double squared = 0, dot = 0;
    for (int j = 0; j < 3; ++j)
      squared += double(v[3 * i + j]) * v[3 * i + j];
    const double norm = std::sqrt(std::max(squared, double(1e-4f)));
    for (int j = 0; j < 3; ++j)
      dot += (v[3 * i + j] / norm) * w[3 * i + j];
    for (int j = 0; j < 3; ++j) {
      const double y = v[3 * i + j] / norm,
                   derivative = (w[3 * i + j] - (squared > 1e-4f ? y * dot : 0)) / norm;
      error = std::max(error, std::abs(nv[3 * i + j] - y));
      error = std::max(error, std::abs(gv[3 * i + j] - derivative));
    }
  }
  check::Near(error, 0, 2e-5,
              "every normalized value and derivative including zero/floored norm");
  const double expected[] = {0, 1, -2, 0, 0, .5, 1.5, 1.5, 4, 0, -1, 0};
  const auto mg = maxgrad.to_vector<double>();
  for (int i = 0; i < 12; ++i)
    check::Near(mg[i], expected[i], 0, "maxout unique/tied selected derivative");
  const float selected[] = {3, 4, -1, 5, 0, -1};
  const auto mo = maxout.to_vector<double>();
  for (int i = 0; i < 6; ++i)
    check::Near(mo[i], selected[i], 0, "maxout pair values");
  double squared_distance = 0, dot = 0;
  for (int j = 0; j < 3; ++j) {
    const double a = nv[j], b = nv[3 + j];
    squared_distance += (a - b) * (a - b);
    dot += a * b;
  }
  check::Near(squared_distance, 2 - 2 * dot, 2e-7,
              "unit embedding cosine/distance equivalence");
  std::cout << "  L2 output/every-gradient host error=" << error
            << " including zero/floor; maxout unique/tied Jacobians and sphere distance "
               "identity passed"
            << std::endl;
}

void TripletChecks() {
  const std::vector<float> v = {0, 0, 1, 0, .5, 0, 2, 0, 0, 1, 0, 1};
  Tensor x = nn::from_vector(v, {6, 2}).requires_grad_();
  const std::vector<metric::TripletIndex> triplets = {
      {0, 1, 2}, {0, 1, 3}, {4, 5, 0}, {1, 4, 2}};
  const Tensor loss = metric::MeanTriplet(x, triplets),
               per = metric::TripletHinge(x, triplets);
  const Tensor empty = metric::MeanTriplet(x, {}),
               boundary = metric::MeanTriplet(x, {{0, 2, 1}}, .75f);
  const auto grad = nn::grad(loss, {x})[0].to_vector<double>(),
             emptygrad = nn::grad(empty, {x})[0].to_vector<double>(),
             boundarygrad = nn::grad(boundary, {x})[0].to_vector<double>();
  const auto pv = per.to_vector<double>();
  double host = 0, expected[12] = {}, error = 0;
  int i = 0;
  for (const auto &t : triplets) {
    double ap = 0, an = 0;
    for (int c = 0; c < 2; ++c) {
      ap += std::pow(double(v[2 * t[0] + c]) - v[2 * t[1] + c], 2);
      an += std::pow(double(v[2 * t[0] + c]) - v[2 * t[2] + c], 2);
    }
    const double value = std::max(0., ap - an + double(.2f));
    check::Near(pv[i++], value, 1e-7, "each squared-distance hinge");
    host += value / triplets.size();
    if (value > 0)
      for (int c = 0; c < 2; ++c) {
        expected[2 * t[0] + c] += 2 * (v[2 * t[2] + c] - v[2 * t[1] + c]) / triplets.size();
        expected[2 * t[1] + c] += 2 * (v[2 * t[1] + c] - v[2 * t[0] + c]) / triplets.size();
        expected[2 * t[2] + c] += 2 * (v[2 * t[0] + c] - v[2 * t[2] + c]) / triplets.size();
      }
  }
  check::Near(loss.item(), host, 1e-7, "mean triplet host value");
  for (int j = 0; j < 12; ++j) {
    error = std::max(error, std::abs(grad[j] - expected[j]));
    check::Near(emptygrad[j], 0, 0, "empty derivative");
    check::Near(boundarygrad[j], 0, 0, "margin-boundary ReLU subgradient");
  }
  check::Near(error, 0, 1e-7, "all active/inactive/repeated-index triplet derivatives");
  check::Near(empty.item(), 0, 0, "empty triplet loss");
  check::Near(boundary.item(), 0, 0, "margin boundary loss");
  std::cout << "  squared-L2 triplet values/every-coordinate Jacobian error=" << error
            << "; inactive/repeated-index/empty/exact-margin cases passed" << std::endl;
}

void MiningChecks() {
  const std::vector<float> values = {0, 1, .5, 1.5, 2, -1, 1.5, 4};
  const std::vector<int> labels = {0, 0, 1, 1, 2, 2, 3, 3};
  const auto mined = metric::MineSemiHard(nn::from_vector(values, {8, 1}), labels, 2.f);
  std::vector<metric::TripletIndex> expected;
  int pairs = 0;
  for (int a = 0; a < 8; ++a)
    for (int p = 0; p < 8; ++p)
      if (a != p && labels[a] == labels[p]) {
        ++pairs;
        const double ap = std::pow(values[a] - values[p], 2);
        std::vector<std::pair<double, int>> eligible;
        for (int n = 0; n < 8; ++n)
          if (labels[n] != labels[a]) {
            const double an = std::pow(values[a] - values[n], 2);
            if (ap < an && an < ap + 2)
              eligible.push_back({an, n});
          }
        std::sort(eligible.begin(), eligible.end());
        if (!eligible.empty())
          expected.push_back({a, p, eligible.front().second});
      }
  check::Require(mined.triplets == expected && mined.positive_pairs == pairs &&
                     mined.without_semihard == pairs - int(expected.size()),
                 "exhaustive sorted semi-hard reference, tie/index/missing accounting");
  check::Require(std::find(mined.triplets.begin(), mined.triplets.end(),
                           metric::TripletIndex{0, 1, 3}) != mined.triplets.end(),
                 "closest eligible negative deterministic tie");
  const Tensor equal = nn::zeros({4, 2});
  check::Require(metric::MineSemiHard(equal, {0, 0, 1, 1}).triplets.empty(),
                 "collapsed equal distances have no strict semi-hard fallback");
  check::Require(metric::MineSemiHard(equal, {0, 0, 0, 0}).triplets.empty() &&
                     metric::MineSemiHard(equal, {0, 1, 2, 3}).positive_pairs == 0,
                 "no negative or positive cases");
  const auto exact = metric::MineSemiHard(nn::tensor({0, 1, 2}, {3, 1}), {0, 0, 1}, 3.f);
  check::Require(std::find(exact.triplets.begin(), exact.triplets.end(),
                           metric::TripletIndex{0, 1, 2}) == exact.triplets.end(),
                 "strict upper margin equality excluded");
  const std::vector<metric::VerificationPair> p = {{0, 1, 1}, {0, 2, 1}, {0, 3, 0}, {0, 4, 0}};
  const std::vector<double> d = {.1, .2, .8, .9};
  const double threshold = metric::CalibrateMetricThreshold(d, p);
  const auto score = metric::ScoreMetricPairs(d, p, threshold);
  check::Near(threshold, .5, 1e-12, "validation-only optimal threshold midpoint");
  check::Near(score.accuracy, 1, 0, "known verification accuracy");
  check::Near(score.auc, 1, 0, "known verification AUC");
  check::Near(metric::ScoreMetricPairs({1, 1, 1, 1}, p, 1).auc, .5, 0, "tied AUC half credit");
  nn::ImageDataset data;
  data.images = nn::zeros({50, 28, 28, 1});
  std::vector<std::int64_t> l(50);
  for (int i = 0; i < 50; ++i)
    l[i] = i / 10;
  data.labels = nn::from_vector(l);
  const auto buckets = metric::SiameseBuckets(data, {0, 1, 2, 3, 4}, 10);
  const auto batch = metric::MetricBatchRows(buckets, 8, 11);
  check::Require(batch == metric::MetricBatchRows(buckets, 8, 11) &&
                     batch != metric::MetricBatchRows(buckets, 8, 12),
                 "P-by-K batch reproducibility");
  std::set<std::size_t> unique(batch.begin(), batch.end());
  int counts[5] = {};
  std::vector<int> y;
  for (std::size_t row : batch) {
    ++counts[l[row]];
    y.push_back(int(l[row]));
  }
  check::Require(unique.size() == 40, "40 distinct batch rows");
  for (int c : counts)
    check::Require(c == 8, "exact P5K8 class balance");
  for (const auto &t : metric::FixedMetricTriplets(y, 100, 7))
    check::Require(t[0] != t[1] && y[t[0]] == y[t[1]] && y[t[0]] != y[t[2]],
                   "fixed probe triplet identities/labels");
  std::cout << "  exhaustive strict semi-hard mining/ties/equalities/no-fallback; P5K8 "
               "uniqueness; calibration and AUC fixtures passed"
            << std::endl;
}

void ModelChecks(bool full) {
  {
    int tensors = 0;
    check::Require(metric::FaceNetCensus({}, &tensors) == 140025664 && tensors == 28,
                   "full NN1 independent parameter census");
    if (full) {
      nn::ParameterStore p(3);
      const auto net = metric::FaceNetNN1(p, nn::uniform({1, 220, 220, 3}), {});
      check::Require(p.count() == 140025664 && p.size() == 28,
                     "full NN1 executed parameter census");
      const auto gradients = nn::grad(nn::sum(net.embedding), p.parameters());
      check::Require(gradients.size() == 28, "all full NN1 parameters connected");
      std::cout << "  full NN1:140025664 parameters,28 connected derivatives; executed on one "
                   "220px image"
                << std::endl;
    } else {
      std::cout << "  full NN1:140025664 parameters,28 tensors from the layer shapes (TF: "
                   "graph-only; full=1 executes it)"
                << std::endl;
    }
  }
  nn::ParameterStore p(17);
  std::vector<float> w(256), in(2 * 28 * 28);
  for (int i = 0; i < 256; ++i)
    w[i] = std::sin(i * .17f);
  for (std::size_t i = 0; i < in.size(); ++i)
    in[i] = .5 + .45 * std::sin(i * .113f);
  const auto net = metric::FaceNetNN1(p, nn::from_vector(in, {2, 28, 28, 1}),
                                      metric::FaceNetConfig::Compact());
  const auto gradients =
      nn::grad(nn::sum(net.embedding * nn::from_vector(w, {2, 128})), p.parameters());
  check::Require(p.count() == 108040 && p.size() == 28 &&
                     metric::FaceNetCensus(metric::FaceNetConfig::Compact()) == 108040,
                 "compact NN1 independent parameter census");
  check::Require(net.embedding.shape() == nn::Shape({2, 128}), "128D compact embedding");
  const int sides[] = {14, 7, 7, 4, 4, 2, 2, 2, 2, 2, 2},
            channels[] = {8, 8, 24, 24, 48, 48, 32, 32, 32, 32, 32};
  for (int i = 0; i < 11; ++i)
    check::Require(net.stages[i].shape() == nn::Shape({2, sides[i], sides[i], channels[i]}),
                   "all NN1 convolution stage shapes");
  for (int i = 11; i < 13; ++i)
    check::Require(net.stages[i].shape() == nn::Shape({2, 128}), "two maxout output shapes");
  const auto e = net.embedding.to_vector<double>();
  for (int i = 0; i < 2; ++i) {
    double norm = 0;
    for (int j = 0; j < 128; ++j) {
      check::Require(std::isfinite(e[i * 128 + j]), "finite sphere embedding");
      norm += e[i * 128 + j] * e[i * 128 + j];
    }
    check::Near(norm, 1, 2e-6, "unit sphere compact output");
  }
  double low = INFINITY, high = 0;
  for (std::size_t i = 0; i < gradients.size(); ++i) {
    const double n = nn::norm(gradients[i]).item();
    check::Require(std::isfinite(n) && n > 0,
                   "every compact NN1 gradient nonzero: " + p.names()[i]);
    low = std::min(low, n), high = std::max(high, n);
  }
  std::cout << "  compact NN1:108040 parameters; all13 stages/28 numerical "
               "derivatives/unit128D outputs passed; norms["
            << low << ',' << high << ']' << std::endl;
}

} // namespace
TEST(FaceNetModels, NormalizationAndMaxoutIndependentJacobians) {
  NormalizationChecks();
}
TEST(FaceNetModels, TripletIndependentEveryCoordinateDerivatives) {
  TripletChecks();
}
TEST(FaceNetModels, StrictMiningCalibrationAndSamplingOracles) {
  MiningChecks();
}
TEST(FaceNetModels, FullCensusCompactEveryStageAndGradient) {
  nn::set_num_threads(2);
  ModelChecks(false);
}
TEST(FaceNetModels, EmptyHingeRetainsInputGraph) {
  auto x = nn::zeros({0, 3}).requires_grad_();
  const auto loss = nn::models::MeanTriplet(x, {});
  const auto gradient = nn::grad(loss, {x})[0];
  EXPECT_EQ(gradient.shape(), x.shape());
  EXPECT_DOUBLE_EQ(loss.item(), 0);
}
TEST(FaceNetModels, InvalidConfigurationAndDistanceContracts) {
  EXPECT_THROW(nn::models::FaceNetCensus({28, 1, 0, 128}), std::invalid_argument);
  EXPECT_THROW(nn::models::FaceNetCensus({28, 1, 3, 128}), std::invalid_argument);
  EXPECT_THROW(nn::models::UnitEmbedding(nn::ones({2, 3}), 0), std::invalid_argument);
  EXPECT_THROW(nn::models::MineSemiHard(nn::zeros({2, 3}), {0}, .2f), std::invalid_argument);
  EXPECT_THROW(nn::models::MetricDistanceMetrics(nn::from_vector(std::vector<double>{0, 1}),
                                                 nn::from_vector(std::vector<double>{1, 1}),
                                                 .5),
               std::invalid_argument);
}
