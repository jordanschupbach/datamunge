#include "common/retrieval.hpp"
#include <dm/dm.hpp>
#include <gtest/gtest.h>
namespace nn = dm::nn;
using nn::Tensor;
using namespace metric;
namespace {
void PoolingChecks() {
  std::vector<float> input(24), up = {.3f, -.7f, .2f, .5f};
  for (int i = 0; i < 24; ++i)
    input[std::size_t(i)] = i % 7 == 0    ? -.2f
                            : i % 11 == 0 ? 1e-7f
                                          : float(.1 + (i % 9) * .17);
  const float power = 2.3f;
  Tensor x = nn::from_vector(input, {2, 2, 3, 2}).requires_grad_(),
         exponent = nn::from_vector(std::vector<float>{power}, {}).requires_grad_();
  const Tensor pooled = GeneralizedMean(x, exponent);
  const auto g = nn::grad(nn::sum(pooled * nn::from_vector(up, {2, 2})), {x, exponent});
  const auto pv = pooled.to_vector<float>(), gx = g[0].to_vector<float>();
  double value_error = 0, gradient_error = 0, dp = 0;
  for (int n = 0; n < 2; ++n)
    for (int c = 0; c < 2; ++c) {
      double sum = 0, logsum = 0;
      for (int i = 0; i < 6; ++i) {
        const double a =
            std::max(double(input[std::size_t((n * 6 + i) * 2 + c)]), double(1e-6f));
        sum += std::pow(a, double(power));
        logsum += std::pow(a, double(power)) * std::log(a);
      }
      const double mean = sum / 6, gm = std::pow(mean, 1 / double(power)),
                   weight = up[std::size_t(n * 2 + c)];
      value_error = std::max(value_error, std::abs(gm - pv[std::size_t(n * 2 + c)]));
      dp += weight * gm *
            (-std::log(mean) / (double(power) * power) + logsum / (double(power) * sum));
      for (int i = 0; i < 6; ++i) {
        const int j = (n * 6 + i) * 2 + c;
        const double a = input[std::size_t(j)],
                     expected = a > 1e-6 ? weight * std::pow(a, power - 1) *
                                               std::pow(gm, 1 - power) / 6
                                         : 0;
        gradient_error = std::max(gradient_error, std::abs(expected - gx[std::size_t(j)]));
      }
    }
  check::Near(value_error, 0, 2e-7, "every GeM value");
  check::Near(gradient_error, 0, 2e-7, "every signed/clamped GeM activation derivative");
  check::Near(g[1].item(), dp, 2e-7, "independent analytic learned exponent derivative");
  const auto mean1 = GeneralizedMean(x.detach(), nn::from_vector(std::vector<float>{1.f}, {}))
                         .to_vector<float>();
  for (int n = 0; n < 2; ++n)
    for (int c = 0; c < 2; ++c) {
      double sum = 0;
      for (int i = 0; i < 6; ++i)
        sum += std::max(input[std::size_t((n * 6 + i) * 2 + c)], 1e-6f);
      check::Near(mean1[std::size_t(n * 2 + c)], sum / 6, 1e-7, "GeM p1 mean-pooling limit");
    }
  const std::vector<float> values = {3, 4, 0, -2, 1, 2, 0, 0, 0},
                           weights = {.3f, -.7f, .2f, .3f, -.7f, .2f, .3f, -.7f, .2f};
  Tensor v = nn::from_vector(values, {3, 3}).requires_grad_();
  const Tensor normalized = DescriptorUnit(v);
  const auto nv = normalized.to_vector<float>(),
             ng = nn::grad(nn::sum(normalized * nn::from_vector(weights, {3, 3})), {v})[0]
                      .to_vector<float>();
  for (int n = 0; n < 3; ++n) {
    double square = 0, dot = 0;
    for (int c = 0; c < 3; ++c) {
      const double a = values[std::size_t(n * 3 + c)];
      square += a * a;
      dot += a * weights[std::size_t(n * 3 + c)];
    }
    const double norm = std::sqrt(std::max(square, 1e-12));
    for (int c = 0; c < 3; ++c) {
      const double a = values[std::size_t(n * 3 + c)],
                   expected = weights[std::size_t(n * 3 + c)] / norm -
                              (square > 1e-12 ? a * dot / std::pow(norm, 3) : 0);
      check::Near(nv[std::size_t(n * 3 + c)], a / norm, 1e-7,
                  "signed unit or zero descriptor values");
      check::Near(ng[std::size_t(n * 3 + c)], expected, n == 2 ? .1 : 1e-7,
                  "all unit-normalization input derivatives including zero-vector epsilon");
    }
  }
  std::cout
      << " GeM every value/input derivative max errors=" << value_error << "/"
      << gradient_error
      << ",analytic p gradient,p1 mean,signed/zero normalization/every derivative passed\n";
}

void MiningRankingChecks() {
  const std::vector<float> f = {0, .3f, .1f, .9f};
  const Tensor features = nn::from_vector(f, {4, 1});
  const std::vector<int> labels = {0, 0, 1, 1};
  const auto triples = RetrievalHardTriplets(features, labels);
  const std::vector<TripletIndex> expected = {{0, 1, 2}, {1, 0, 2}, {2, 3, 0}, {3, 2, 1}};
  check::Require(triples == expected,
                 "independent hardest-positive/nearest-negative selection and self exclusion");
  Tensor x = features.clone().requires_grad_();
  const Tensor loss = MeanTriplet(x, triples, .2f);
  const auto g = nn::grad(loss, {x})[0].to_vector<float>();
  double value = 0;
  std::array<double, 4> gradient{};
  for (const auto &t : expected) {
    const double a = f[std::size_t(t[0])], p = f[std::size_t(t[1])], n = f[std::size_t(t[2])],
                 hinge = (a - p) * (a - p) - (a - n) * (a - n) + .2f;
    value += std::max(hinge, 0.) / 4;
    if (hinge > 0) {
      gradient[std::size_t(t[0])] += 2 * (n - p) / 4;
      gradient[std::size_t(t[1])] += 2 * (p - a) / 4;
      gradient[std::size_t(t[2])] += 2 * (a - n) / 4;
    }
  }
  check::Near(loss.item(), value, 1e-7, "known frozen-hard-triplet hinge mean");
  for (int i = 0; i < 4; ++i)
    check::Near(g[std::size_t(i)], gradient[std::size_t(i)], 2e-7,
                "every mined triplet descriptor derivative");
  const Tensor gallery = nn::from_vector(std::vector<float>{1, 0, 0, 1, -1, 0, 1, 0}, {4, 2});
  const HostEmbeddings query(nn::from_vector(std::vector<float>{1, 0, 1, 0}, {2, 2}));
  DescriptorIndex index(gallery, {9, 3, 7, 2});
  const auto ranking = index.Rank(query, 0, 9);
  check::Require(ranking.size() == 3 && ranking[0].id == 2 && ranking[1].id == 3 &&
                     ranking[2].id == 7,
                 "exact squared distances,deterministic source-ID ties and self exclusion");
  check::Near(ranking[0].distance, 0, 0, "zero distance");
  check::Near(ranking[1].distance, 2, 0, "orthogonal unit squared distance");
  check::Near(ranking[2].distance, 4, 0, "opposite unit squared distance");
  RetrievalPool queries;
  queries.labels = {1, 99};
  queries.ids = {9, 90};
  const auto score = ScoreRetrieval(query, queries, index, {1, 0, 1, 1});
  check::Near(score.map, 5. / 6, 1e-12,
              "known AP positives at ranks1 and3,exclude self from denominator");
  check::Require(score.eligible == 1 && score.without_positive == 1 && score.hit1 == 1 &&
                     score.hit5 == 1,
                 "queries without positives excluded with explicit count");
  const auto tied = index.Rank(query, 0, -1);
  check::Require(tied[0].id == 2 && tied[1].id == 9,
                 "equal distances ordered by source ID,not gallery insertion");
  check::Throws(
      [&] {
        DescriptorIndex invalid(gallery, {1, 1, 2, 3});
      },
      "duplicate source IDs rejected");
  std::cout << " exact hard mining/every triplet gradient,known AP5/6,unit cosine-distance "
               "relation,ties,self exclusion,no-positive queries and duplicate-ID rejection "
               "passed\n";
}

std::int64_t Census(int divisor, int dimensions) {
  const int widths[] = {64, 128, 256, 512, 512}, counts[] = {2, 2, 3, 3, 3};
  int in = 3;
  std::int64_t total = 1;
  for (int block = 0; block < 5; ++block)
    for (int layer = 0; layer < counts[block]; ++layer) {
      const int out = widths[block] / divisor;
      total += 9LL * in * out + out;
      in = out;
    }
  return total + std::int64_t(in) * dimensions;
}

void ModelChecks() {
  for (int divisor : {1, 8}) {
    const int dimensions = divisor == 1 ? 128 : 32;
    nn::ParameterStore p(227);
    std::vector<float> input(2 * 3072);
    for (std::size_t i = 0; i < input.size(); ++i)
      input[i] = float(.5 + .4 * std::sin(i * .17));
    Tensor rgb = nn::from_vector(input, {2, 32, 32, 3}).requires_grad_();
    const auto net = RetrievalEncoder(p, rgb, divisor, dimensions);
    std::vector<float> up(std::size_t(2 * dimensions));
    for (std::size_t i = 0; i < up.size(); ++i)
      up[i] = float((int(i) % 13 - 6) * .07f);
    const Tensor objective = nn::sum(net.descriptor * nn::from_vector(up, {2, dimensions}));
    auto wrt = p.parameters();
    wrt.push_back(rgb);
    const auto gradients = nn::grad(objective, wrt);
    check::Require(p.count() == Census(divisor, dimensions) && p.size() == 28 &&
                       net.stages.size() == 19,
                   "independent full/compact parameter and stage census");
    std::cout << " " << (divisor == 1 ? "full" : "compact") << " retrieval:" << p.count()
              << " parameters/28 derivatives,19 stages (executed)\n";
    if (divisor == 1)
      continue;
    std::vector<nn::Shape> shapes;
    const int widths[] = {8, 16, 32, 64, 64}, counts[] = {2, 2, 3, 3, 3};
    std::int64_t side = 32;
    for (int block = 0; block < 5; ++block) {
      for (int layer = 0; layer < counts[block]; ++layer)
        shapes.push_back({2, side, side, widths[block]});
      if (block < 4) {
        side /= 2;
        shapes.push_back({2, side, side, widths[block]});
      }
    }
    shapes.push_back({2, 64});
    shapes.push_back({2, 32});
    for (int i = 0; i < 19; ++i)
      check::Require(net.stages[std::size_t(i)].shape() == shapes[std::size_t(i)],
                     "all convolution/pool/GeM/projection shapes");
    check::Near(net.exponent.item(), 3, 3e-7, "learnable exponent initialized3");
    double minimum = INFINITY, maximum = 0;
    for (const auto &g : gradients) {
      double norm = 0;
      for (float value : g.to_vector<float>()) {
        check::Require(std::isfinite(value), "finite descriptor model/image gradient");
        norm += double(value) * value;
      }
      norm = std::sqrt(norm);
      check::Require(norm > 0,
                     "all28 model tensors including rho/projection and RGB receive gradients");
      minimum = std::min(minimum, norm), maximum = std::max(maximum, norm);
    }
    std::cout << " compact all28 parameter and RGB gradients finite/nonzero,norm range["
              << minimum << "," << maximum << "],all19 shapes passed\n";
  }
}

RetrievalSplits ReferenceLoadRetrievalData(const std::filesystem::path &root) {
  std::vector<std::filesystem::path> paths;
  for (int i = 1; i <= 5; ++i)
    paths.push_back(root / ("data_batch_" + std::to_string(i) + ".bin"));
  const auto development = ReadCifarRecords(paths),
             test = ReadCifarRecords({root / "test_batch.bin"});
  check::Require(development.labels.size() == 50000 && test.labels.size() == 10000,
                 "official CIFAR-10 source sizes");
  std::array<std::vector<std::size_t>, 10> dev_rows, test_rows;
  for (std::size_t i = 0; i < development.labels.size(); ++i)
    dev_rows[std::size_t(development.labels[i])].push_back(i);
  for (std::size_t i = 0; i < test.labels.size(); ++i)
    test_rows[std::size_t(test.labels[i])].push_back(i);
  std::mt19937 random(211);
  for (auto &rows : dev_rows)
    std::shuffle(rows.begin(), rows.end(), random);
  for (auto &rows : test_rows)
    std::shuffle(rows.begin(), rows.end(), random);
  auto make =
      [](const CifarRecords &source, const std::array<std::vector<std::size_t>, 10> &buckets,
         const std::vector<int> &classes, int begin, int each, std::int64_t id_offset) {
        RetrievalPool out;
        out.pixels.resize(classes.size() * std::size_t(each) * 3072);
        std::size_t row = 0;
        for (int label : classes)
          for (int i = begin; i < begin + each; ++i, ++row) {
            const std::size_t index = buckets[std::size_t(label)].at(std::size_t(i));
            out.labels.push_back(label);
            out.ids.push_back(id_offset + std::int64_t(index));
            const std::uint8_t *record = source.Record(index) + 1;
            for (int pixel = 0; pixel < 1024; ++pixel)
              for (int c = 0; c < 3; ++c)
                out.pixels[row * 3072 + std::size_t(pixel * 3 + c)] =
                    record[c * 1024 + pixel] / 255.f;
          }
        out.images = nn::from_vector(out.pixels, {std::int64_t(row), 32, 32, 3});
        return out;
      };
  RetrievalSplits s{make(development, dev_rows, {0, 1, 2, 3, 4}, 0, 32, 0),
                    make(development, dev_rows, {5, 6}, 0, 8, 0),
                    make(development, dev_rows, {5, 6}, 8, 16, 0),
                    make(test, test_rows, {7, 8, 9}, 0, 16, 100000),
                    make(test, test_rows, {7, 8, 9}, 16, 32, 100000)};
  std::set<std::int64_t> used;
  for (const auto *pool :
       {&s.fit, &s.validation_query, &s.validation_gallery, &s.test_query, &s.test_gallery})
    for (std::int64_t id : pool->ids)
      check::Require(used.insert(id).second,
                     "no repeated source images across any fitting/query/gallery pools");
  return s;
}

inline std::vector<std::vector<std::size_t>>
ReferenceRetrievalBatches(const RetrievalPool &fitting, int epoch) {
  std::array<std::vector<std::size_t>, 5> buckets;
  for (std::size_t i = 0; i < fitting.labels.size(); ++i) {
    check::Require(fitting.labels[i] >= 0 && fitting.labels[i] < 5,
                   "fit-only category labels");
    buckets[std::size_t(fitting.labels[i])].push_back(i);
  }
  std::mt19937 random(std::uint32_t(223 + epoch));
  for (auto &rows : buckets) {
    check::Require(rows.size() == 32, "32 fitting images per class");
    std::shuffle(rows.begin(), rows.end(), random);
  }
  std::vector<std::vector<std::size_t>> result;
  for (int batch = 0; batch < 8; ++batch) {
    std::vector<std::size_t> rows;
    for (const auto &bucket : buckets)
      rows.insert(rows.end(), bucket.begin() + 4 * batch, bucket.begin() + 4 * batch + 4);
    std::shuffle(rows.begin(), rows.end(), random);
    result.push_back(rows);
  }
  return result;
}

} // namespace
TEST(RetrievalModels, IndependentGemInputPowerAndUnitJacobians) {
  PoolingChecks();
}
TEST(RetrievalModels, IndependentHardMiningTripletAndExactRankingOracles) {
  MiningRankingChecks();
}
TEST(RetrievalModels, FullAndCompactAllStagesAndGradients) {
  nn::set_num_threads(2);
  ModelChecks();
}
TEST(RetrievalData, OriginalEveryPoolIdPixelAndSeededEpochBatch) {
  std::filesystem::path cache;
  if (const auto *p = std::getenv("DATAMUNGE_DATA_DIR"))
    cache = p;
  else if (const auto *p = std::getenv("XDG_CACHE_HOME"))
    cache = std::filesystem::path(p) / "datamunge/datasets";
  else
    cache = std::filesystem::path(std::getenv("HOME")) / ".cache/datamunge/datasets";
  if (!std::filesystem::exists(cache / "cifar10/data_batch_1.bin") ||
      !std::filesystem::exists(cache / "cifar10/test_batch.bin"))
    GTEST_SKIP() << "Optional official CIFAR cache is absent.";
  const auto reference = ReferenceLoadRetrievalData(cache / "cifar10");
  const auto data = nn::models::LoadTensorRetrievalData(cache / "cifar10");
  const std::vector<const RetrievalPool *> expected{
      &reference.fit, &reference.validation_query, &reference.validation_gallery,
      &reference.test_query, &reference.test_gallery};
  const std::vector<const nn::models::TensorRetrievalPool *> actual{
      &data.fit, &data.validation_query, &data.validation_gallery, &data.test_query,
      &data.test_gallery};
  for (std::size_t i = 0; i < actual.size(); ++i) {
    EXPECT_EQ(actual[i]->ids, expected[i]->ids);
    EXPECT_EQ(actual[i]->labels, expected[i]->labels);
    EXPECT_EQ(actual[i]->images.to_vector<float>(), expected[i]->pixels);
  }
  for (int epoch : {-1, 0, 1, 3})
    EXPECT_EQ(nn::models::RetrievalBatches(data.fit, epoch),
              ReferenceRetrievalBatches(reference.fit, epoch));
  auto pool = data.fit;
  pool.images = pool.images.clone().requires_grad_();
  const auto selected = nn::models::TensorRetrievalRows(pool, {2, 2, 0});
  const auto grad = nn::grad(nn::sum(selected.images), {pool.images})[0];
  EXPECT_DOUBLE_EQ(nn::sum(nn::slice_axis(grad, 0, 2, 3)).item(), 2 * 3072);
  EXPECT_DOUBLE_EQ(nn::sum(nn::slice_axis(grad, 0, 0, 1)).item(), 3072);
}
TEST(RetrievalIndex, LargeSignedIdsEmptyCasesAndOwnedSnapshot) {
  const std::int64_t base = 9007199254740992LL;
  auto features = nn::ones({3, 2});
  nn::models::TensorDescriptorIndex index(features, {base + 2, base + 1, -7});
  const auto query = nn::ones({1, 2});
  const auto ranking = index.Rank(query, 0);
  EXPECT_EQ(ranking.ids.to_vector<std::int64_t>(),
            (std::vector<std::int64_t>{-7, base + 1, base + 2}));
  features.copy_(nn::zeros_like(features));
  EXPECT_DOUBLE_EQ(nn::sum(index.Rank(query, 0).distances).item(), 0);
  nn::models::TensorDescriptorIndex empty(nn::zeros({0, 2}), {});
  EXPECT_EQ(empty.Rank(query, 0).rows.numel(), 0);
  const auto score = nn::models::ScoreTensorRetrieval(query, {0}, {1}, empty, {});
  EXPECT_DOUBLE_EQ(score.map.item(), 0);
  EXPECT_EQ(score.without_positive.item<std::int64_t>(), 1);
  const auto noquery =
      nn::models::ScoreTensorRetrieval(nn::zeros({0, 2}), {}, {}, index, {0, 0, 1});
  EXPECT_EQ(noquery.eligible.item<std::int64_t>(), 0);
  EXPECT_TRUE(nn::models::RetrievalHardTriplets(nn::zeros({0, 2}), {}).empty());
}
TEST(RetrievalData, IndependentBlockMeanPixelReference) {
  nn::models::TensorRetrievalPool data;
  data.labels = {0, 1};
  data.ids = {10, 11};
  std::vector<float> pixels(2 * 3072);
  for (std::size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = float((i * 7) % 256) / 255.f;
  data.images = nn::from_vector(pixels, {2, 32, 32, 3});
  for (int block : {4, 32}) {
    const int side = 32 / block, dimensions = side * side * 3;
    std::vector<float> expected(2 * std::size_t(dimensions));
    for (int n = 0; n < 2; ++n) {
      double norm = 0;
      for (int y = 0; y < side; ++y)
        for (int x = 0; x < side; ++x)
          for (int c = 0; c < 3; ++c) {
            double sum = 0;
            for (int dy = 0; dy < block; ++dy)
              for (int dx = 0; dx < block; ++dx)
                sum += double(pixels[std::size_t(
                           n * 3072 + ((y * block + dy) * 32 + x * block + dx) * 3 + c)]) -
                       .5;
            const float value = float(sum / (block * block));
            expected[std::size_t(n * dimensions + (y * side + x) * 3 + c)] = value;
            norm += double(value) * value;
          }
      norm = std::sqrt(std::max(norm, 1e-12));
      for (int j = 0; j < dimensions; ++j)
        expected[std::size_t(n * dimensions + j)] =
            float(expected[std::size_t(n * dimensions + j)] / norm);
    }
    const auto observed =
        nn::models::RetrievalPixelDescriptors(data, block).to_vector<float>();
    ASSERT_EQ(observed.size(), expected.size());
    for (std::size_t i = 0; i < observed.size(); ++i)
      EXPECT_NEAR(observed[i], expected[i], 1e-7);
  }
}
TEST(RetrievalData, EveryRankedPreviewPixel) {
  nn::models::TensorRetrievalPool data;
  data.labels = {0, 0, 1, 1, 2, 2};
  data.ids = {10, 11, 12, 13, 14, 15};
  std::vector<float> pixels(6 * 3072);
  for (std::size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = float((i * 11) % 256) / 255.f;
  data.images = nn::from_vector(pixels, {6, 32, 32, 3});
  const auto features = nn::tensor({0., .1, .2, .3, .4, .5}, {6, 1});
  nn::models::TensorDescriptorIndex index(features, data.ids);
  const std::vector<std::size_t> rows{0, 2};
  const auto tiles = nn::models::RetrievalPreviewTiles(data, data, features, index, rows, 3);
  ASSERT_EQ(tiles.shape(), nn::Shape({2, 64, 256, 3}));
  const auto values = tiles.to_vector<float>();
  for (int n = 0; n < 2; ++n) {
    const auto row = rows[std::size_t(n)];
    const auto ranked =
        index.Rank(features, row, data.ids[row]).rows.to_vector<std::int64_t>();
    for (int col = 0; col < 4; ++col) {
      const auto source = col ? std::size_t(ranked[std::size_t(col - 1)]) : row;
      const int colour = col ? (data.labels[source] == data.labels[row] ? 1 : 0) : 2;
      for (int y = 0; y < 64; ++y)
        for (int x = 0; x < 64; ++x)
          for (int c = 0; c < 3; ++c) {
            const bool border = y < 2 || x < 2 || y >= 62 || x >= 62;
            const float expected =
                border
                    ? (c == colour ? 1 : 0)
                    : float(std::lround(
                          pixels[source * 3072 + std::size_t((y / 2 * 32 + x / 2) * 3 + c)] *
                          255)) /
                          255.f;
            EXPECT_FLOAT_EQ(values[std::size_t(((n * 64 + y) * 256 + col * 64 + x) * 3 + c)],
                            expected);
          }
    }
  }
}
