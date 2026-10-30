#include <cmath>
#include <dm/dm.hpp>
#include <gtest/gtest.h>
#include <iostream>
#include <set>
namespace nn = dm::nn;
using nn::Tensor;
using namespace nn::models;
namespace check {
void Require(bool x, const std::string &what) {
  if (!x)
    throw std::runtime_error(what);
}
void Near(double x, double y, double tol, const std::string &what) {
  if (!std::isfinite(x) || std::abs(x - y) > tol)
    throw std::runtime_error(what);
}
} // namespace check
namespace {
auto BinaryCrossEntropy(const Tensor &z, const Tensor &y) {
  return SiameseBinaryCrossEntropy(z, y);
}
int BestOneShot(const std::vector<double> &s) {
  return int(nn::argmax(nn::from_vector(s), 0).item<std::int64_t>());
}
double MaxAbs(const std::vector<double> &a, const std::vector<double> &b) {
  double e = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    e = std::max(e, std::abs(a[i] - b[i]));
  return e;
}

void HeadChecks() {
  nn::ParameterStore p(0);
  const float av[] = {.1f, .8f, -.2f, .4f, -.5f, .7f}, bv[] = {-.4f, .2f, .9f, .4f, .6f, -.3f};
  Tensor a = nn::from_vector(std::vector<float>(av, av + 6), {2, 3}).requires_grad_(),
         b = nn::from_vector(std::vector<float>(bv, bv + 6), {2, 3}).requires_grad_();
  const Tensor t = nn::from_vector(std::vector<float>{1, 0});
  {
    nn::NoGradGuard guard;
    SiameseLogit(p, a, b, 3);
  }
  const std::vector<float> weights = {-.7f, .4f, 1.2f};
  const float bias = -.2f;
  p.set("similarity/W", nn::from_vector(weights, {3, 1}));
  p.set("similarity/b", nn::from_vector(std::vector<float>{bias}));
  const Tensor logit = SiameseLogit(p, a, b, 3), reverse = SiameseLogit(p, b, a, 3),
               identity = SiameseLogit(p, a, a, 3);
  const Tensor loss = nn::mean(BinaryCrossEntropy(logit, t));
  const auto gradient = nn::grad(loss, {a, b, p.at("similarity/W"), p.at("similarity/b")});
  const auto equal = nn::grad(nn::mean(identity), {a})[0].to_vector<double>();
  Tensor extreme =
      nn::from_vector(std::vector<float>{1000.f, -1000.f, 1000.f, -1000.f, 0.f, 0.f})
          .requires_grad_();
  const Tensor stable = BinaryCrossEntropy(
      extreme, nn::from_vector(std::vector<float>{1.f, 0.f, 0.f, 1.f, 0.f, 1.f}));
  const auto sg = nn::grad(nn::sum(stable), {extreme})[0].to_vector<double>();
  const auto lv = logit.to_vector<double>(), rv = reverse.to_vector<double>(),
             iv = identity.to_vector<double>(), ga = gradient[0].to_vector<double>(),
             gb = gradient[1].to_vector<double>(), gw = gradient[2].to_vector<double>(),
             gbias = gradient[3].to_vector<double>(), sv = stable.to_vector<double>();
  double host_loss = 0, error = 0, dw[3] = {}, db = 0;
  const double tv[] = {1, 0};
  for (int i = 0; i < 2; ++i) {
    double z = bias;
    for (int j = 0; j < 3; ++j)
      z += weights[j] * std::abs(double(av[i * 3 + j]) - bv[i * 3 + j]);
    host_loss += (std::max(z, 0.) - z * tv[i] + std::log1p(std::exp(-std::abs(z)))) / 2;
    check::Near(lv[i], z, 2e-7, "weighted L1 logit");
    check::Near(rv[i], z, 2e-7, "swap symmetry");
    check::Near(iv[i], -.2, 1e-7, "identical pair bias");
    const double delta = (1 / (1 + std::exp(-z)) - tv[i]) / 2;
    db += delta;
    for (int j = 0; j < 3; ++j) {
      const double diff = double(av[i * 3 + j]) - bv[i * 3 + j],
                   sign = (diff > 0) - (diff < 0), da = delta * weights[j] * sign;
      error = std::max(error, std::abs(ga[i * 3 + j] - da));
      error = std::max(error, std::abs(gb[i * 3 + j] + da));
      dw[j] += delta * std::abs(diff);
    }
  }
  check::Near(loss.item(), host_loss, 1e-7, "host mean BCE");
  for (int j = 0; j < 3; ++j)
    error = std::max(error, std::abs(gw[j] - dw[j]));
  error = std::max(error, std::abs(gbias[0] - db));
  for (int i = 0; i < 6; ++i)
    check::Near(equal[i], 0, 0, "identical input subgradient");
  const double extreme_loss[] = {0, 0, 1000, 1000, std::log(2.), std::log(2.)},
               extreme_gradient[] = {0, 0, 1, -1, .5, -.5};
  for (int i = 0; i < 6; ++i) {
    check::Near(sv[i], extreme_loss[i], 1e-7, "extreme BCE stable");
    check::Near(sg[i], extreme_gradient[i], 1e-7, "extreme and zero BCE derivative");
  }
  check::Near(error, 0, 2e-7, "every L1/head/BCE derivative");
  check::Require(p.count() == 4 && p.size() == 2,
                 "same/reverse/identical head reuses exact variables");
  std::cout << "  weighted L1 logits/symmetry/identity; every input/weight/bias BCE "
               "derivative error="
            << error << "; stable +/-1000 and zero logits passed" << std::endl;
}

void ModelChecks() {
  {
    nn::ParameterStore p(0);
    const SiameseConfig c;
    std::vector<float> x(2 * 105 * 105), y(x.size());
    for (std::size_t i = 0; i < x.size(); ++i)
      x[i] = .5f + .45f * std::sin(i * .173f), y[i] = .5f + .45f * std::cos(i * .097f);
    const auto a = SiameseEncoder(p, nn::from_vector(x, {2, 105, 105, 1}), c),
               b = SiameseEncoder(p, nn::from_vector(y, {2, 105, 105, 1}), c);
    const auto score = SiameseLogit(p, a.embedding, b.embedding, c.embedding);
    check::Require(p.count() == 38951745 && p.size() == 12, "full shared independent census");
    const auto g = nn::grad(nn::sum(score), p.parameters());
    check::Require(g.size() == 12, "all full derivatives");
    for (const auto &v : g)
      for (float f : v.to_vector<float>())
        check::Require(std::isfinite(f), "finite full twin derivatives");
    std::cout << "  full105px shared Siamese:38951745 parameters/12 derivatives (executed "
                 "once on a 2-pair batch)"
              << std::endl;
  }
  nn::ParameterStore p(17);
  const auto c = SiameseConfig::Compact();
  std::vector<float> xv(2 * 48 * 48), yv(xv.size());
  for (std::size_t i = 0; i < xv.size(); ++i)
    xv[i] = .5f + .45f * std::sin(i * .173f), yv[i] = .5f + .45f * std::cos(i * .097f);
  const Tensor input = nn::from_vector(xv, {2, 48, 48, 1}),
               other = nn::from_vector(yv, {2, 48, 48, 1});
  const auto a = SiameseEncoder(p, input, c), b = SiameseEncoder(p, other, c);
  const Tensor score = SiameseLogit(p, a.embedding, b.embedding, c.embedding),
               reverse = SiameseLogit(p, b.embedding, a.embedding, c.embedding),
               identity = SiameseLogit(p, a.embedding, a.embedding, c.embedding);
  const Tensor left_score =
      SiameseLogit(p, a.embedding, nn::stop_gradient(b.embedding), c.embedding);
  const Tensor right_score =
      SiameseLogit(p, nn::stop_gradient(a.embedding), b.embedding, c.embedding);
  nn::GradOptions keep;
  keep.retain_graph = true;
  const auto g = nn::grad(nn::sum(score), p.parameters(), keep);
  const auto ga = nn::grad(nn::sum(left_score), p.parameters(), keep);
  const auto gb = nn::grad(nn::sum(right_score), p.parameters(), keep);
  check::Require(p.count() == 24337 && p.size() == 12,
                 "compact two twins share single encoder/head variables");
  check::Require(a.embedding.shape() == nn::Shape({2, 64}) &&
                     b.embedding.shape() == a.embedding.shape(),
                 "shared embedding shapes");
  for (float v : a.embedding.to_vector<float>())
    check::Require(v > 0 && v < 1, "sigmoid embedding finite range");
  const int sides[] = {44, 20, 8, 3}, channels[] = {8, 16, 16, 32};
  for (int i = 0; i < 4; ++i)
    check::Require(a.stages[std::size_t(i)].shape() ==
                       nn::Shape({2, sides[i], sides[i], channels[i]}),
                   "compact convolution stage shape");
  const auto sv = score.to_vector<double>(), rv = reverse.to_vector<double>(),
             iv = identity.to_vector<double>();
  for (int i = 0; i < 2; ++i) {
    check::Near(sv[i], rv[i], 1e-7, "whole CNN swap symmetry");
    check::Near(iv[i], .1, 1e-7, "whole CNN identical scores");
  }
  double low = INFINITY, high = 0, error = 0;
  for (std::size_t i = 0; i < g.size(); ++i) {
    const auto gv = g[i].to_vector<double>();
    double n = 0;
    for (double v : gv)
      n += v * v;
    n = std::sqrt(n);
    check::Require(std::isfinite(n) && n > 0,
                   "every Siamese parameter receives gradient: " + p.names()[i]);
    low = std::min(low, n), high = std::max(high, n);
    if (p.names()[i].rfind("encoder/", 0) == 0) {
      const auto av = ga[i].to_vector<double>(), bv = gb[i].to_vector<double>();
      std::vector<double> sum(av.size());
      for (std::size_t j = 0; j < av.size(); ++j)
        sum[j] = av[j] + bv[j];
      error = std::max(error, MaxAbs(gv, sum));
    }
  }
  check::Near(error, 0, 2e-6, "shared encoder gradient is sum of twin contributions");
  std::cout
      << "  compact48px:24337 shared parameters; all stages/12 derivatives passed, norms["
      << low << ',' << high << "], additive twin-gradient error=" << error << std::endl;
}

void DataChecks() {
  std::vector<float> images(30 * 784);
  std::vector<std::int64_t> labels(30);
  for (int i = 0; i < 30; ++i) {
    labels[std::size_t(i)] = i / 3;
    for (int j = 0; j < 784; ++j)
      images[std::size_t(i * 784 + j)] = float((i * 7 + j) % 256) / 255.f;
  }
  nn::ImageDataset d{nn::from_vector(images, {30, 28, 28, 1}), nn::from_vector(labels, {30})};
  const auto fit = SiameseBuckets(d, {0, 1, 2, 3, 4}), val = SiameseBuckets(d, {5, 6}),
             test = SiameseBuckets(d, {7, 8, 9});
  const auto pairs = SiamesePairs(fit, 100, 11);
  check::Require(pairs == SiamesePairs(fit, 100, 11) && pairs != SiamesePairs(fit, 100, 12),
                 "pair reproducibility and independent seeds");
  int positive = 0;
  for (const auto &p : pairs) {
    const int a = int(labels[p.left]), b = int(labels[p.right]);
    check::Require(p.left != p.right && a < 5 && b < 5 && p.same == float(a == b),
                   "pair identity/class/label alignment");
    positive += int(p.same);
  }
  check::Require(positive == 50, "exact balanced labels");
  const auto batch = MakeSiameseBatch(d, {pairs[7], pairs[2], pairs[0]});
  const std::vector<int> order = {7, 2, 0};
  const auto bl = batch.labels.to_vector<float>(), left = batch.left.to_vector<float>(),
             right = batch.right.to_vector<float>();
  for (int i = 0; i < 3; ++i) {
    const auto &p = pairs[std::size_t(order[std::size_t(i)])];
    check::Near(bl[std::size_t(i)], p.same, 0, "pair label gather");
    for (int j = 0; j < 784; ++j) {
      check::Near(left[std::size_t(i * 784 + j)], float((int(p.left) * 7 + j) % 256) / 255.f,
                  0, "left pixel gather");
      check::Near(right[std::size_t(i * 784 + j)], float((int(p.right) * 7 + j) % 256) / 255.f,
                  0, "right pixel gather");
    }
  }
  const auto tasks = SiameseTasks(test, 100, 17);
  check::Require(tasks == SiameseTasks(test, 100, 17) && tasks != SiameseTasks(test, 100, 18),
                 "task reproducibility and independent seeds");
  for (const auto &task : tasks) {
    check::Require(task.support.size() == 3 && task.classes.size() == 3,
                   "exactly one support per unseen class");
    std::set<int> seen;
    for (std::size_t i = 0; i < 3; ++i) {
      const int label = int(labels[task.support[i]]);
      check::Require(task.support[i] != task.query && label == task.classes[i] && label >= 7,
                     "support/query distinct and labels aligned");
      seen.insert(label);
    }
    check::Require(seen.size() == 3 &&
                       labels[task.query] == task.classes[std::size_t(task.correct)],
                   "unique support labels and known correct index");
  }
  check::Require(SiameseClasses(fit) == std::vector<int>({0, 1, 2, 3, 4}) &&
                     SiameseClasses(val) == std::vector<int>({5, 6}) &&
                     SiameseClasses(test) == std::vector<int>({7, 8, 9}),
                 "strict train/validation/test class disjointness");
  check::Require(BestOneShot({-.4, .8, .1}) == 1 && BestOneShot({.8, .1, -.4}) == 0 &&
                     BestOneShot({0, 0, 0}) == 0,
                 "one-shot argmax, support permutation and deterministic tie");
  bool rejected = false;
  try {
    SiamesePairs(fit, 3, 1);
  } catch (const std::invalid_argument &) {
    rejected = true;
  }
  check::Require(rejected, "reject unbalanced odd pair count");
  std::cout << "  class-disjoint pair/task sampling, exact balance, nonidentity positives, "
               "reordered pixels/labels, one support/class and "
               "argmax/tie fixtures passed"
            << std::endl;
}

} // namespace
TEST(SiameseModels, IndependentHeadAndBceDerivatives) {
  nn::set_num_threads(2);
  HeadChecks();
}
TEST(SiameseModels, FullAndCompactSharedEncoderGradients) {
  nn::set_num_threads(2);
  ModelChecks();
}
TEST(SiameseModels, ClassDisjointMetadataAndPixelGather) {
  DataChecks();
}
TEST(SiameseModels, VerificationAndFirstMaximum) {
  const auto z = nn::from_vector(std::vector<float>{2, 0, -1, -2});
  const auto y = nn::from_vector(std::vector<float>{1, 0, 1, 0});
  const auto m = SiameseVerificationMetrics(z, y).to_vector<double>();
  EXPECT_DOUBLE_EQ(m[1], .5);
  for (int j = 4; j < 8; ++j)
    EXPECT_DOUBLE_EQ(m[j], 1);
  const auto scores = nn::zeros({2, 3});
  EXPECT_DOUBLE_EQ(
      SiameseOneShotAccuracy(scores, nn::from_vector(std::vector<std::int64_t>{0, 1})).item(),
      .5);
  EXPECT_THROW(SiameseVerificationMetrics(z, y + .5), std::invalid_argument);
}
TEST(SiameseModels, ByteBaselineAndInputGradients) {
  std::vector<float> pixels(6 * 784);
  for (std::size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = float((i * 13) % 256) / 255.f;
  nn::ImageDataset d{nn::from_vector(pixels, {6, 28, 28, 1}).requires_grad_(),
                     nn::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
  const auto tasks = SiameseTasks(SiameseBuckets(d, {0, 1}), 4, 17);
  const auto scores = SiamesePixelScores(d, tasks).to_vector<double>();
  for (std::size_t i = 0; i < tasks.size(); ++i)
    for (std::size_t j = 0; j < 2; ++j) {
      double sum = 0;
      for (std::size_t k = 0; k < 784; ++k) {
        const double delta = std::lround(pixels[tasks[i].query * 784 + k] * 255.) -
                             std::lround(pixels[tasks[i].support[j] * 784 + k] * 255.);
        sum += delta * delta;
      }
      EXPECT_DOUBLE_EQ(scores[i * 2 + j], -sum);
    }
  const auto gathered = SiameseImages(d, {2, 2, 0});
  const auto gradient = nn::grad(nn::sum(gathered), {d.images})[0];
  EXPECT_DOUBLE_EQ(nn::sum(nn::slice_axis(gradient, 0, 2, 3)).item(), 2 * 784);
  EXPECT_DOUBLE_EQ(nn::sum(nn::slice_axis(gradient, 0, 0, 1)).item(), 784);
}

TEST(SiameseModels, EpisodeGalleryIndependentPixelOracle) {
  std::vector<float> pixels(6 * 784);
  for (std::size_t i = 0; i < pixels.size(); ++i)
    pixels[i] = float((i * 7) % 256) / 255.f;
  nn::ImageDataset data{nn::from_vector(pixels, {6, 28, 28, 1}),
                        nn::from_vector(std::vector<std::int64_t>{0, 0, 0, 1, 1, 1})};
  const auto tasks = SiameseTasks(SiameseBuckets(data, {0, 1}), 2, 17);
  std::vector<float> scores(4, 0);
  scores[std::size_t(tasks[0].correct)] = 1;
  scores[2 + std::size_t(1 - tasks[1].correct)] = 1;
  const auto gallery = SiameseEpisodeGallery(data, tasks, nn::from_vector(scores, {2, 2}));
  ASSERT_EQ(gallery.shape(), nn::Shape({2, 60, 180, 3}));
  const auto values = gallery.to_vector<float>();
  for (int row = 0; row < 2; ++row) {
    const auto &task = tasks[std::size_t(row)];
    const int selected = row == 0 ? task.correct : 1 - task.correct;
    for (int col = 0; col < 3; ++col) {
      const auto source = col == 0 ? task.query : task.support[std::size_t(col - 1)];
      for (int y = 0; y < 60; ++y)
        for (int x = 0; x < 60; ++x)
          for (int c = 0; c < 3; ++c) {
            float expected = 25.f / 255;
            if (y >= 2 && y < 58 && x >= 2 && x < 58)
              expected = pixels[source * 784 + std::size_t((y - 2) / 2 * 28 + (x - 2) / 2)];
            else if (col && col - 1 == selected)
              expected = c == (row == 0 ? 1 : 0) ? 1 : 30.f / 255;
            EXPECT_NEAR(values[std::size_t(((row * 60 + y) * 180 + col * 60 + x) * 3 + c)],
                        expected, 1e-7);
          }
    }
  }
}
