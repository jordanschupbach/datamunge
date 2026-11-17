#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <dm/dm.hpp>
#include <filesystem>
namespace n = dm::nn;
namespace m = n::models;
TEST(AutoRec, MaskedValuesEveryDerivativeHiddenNaNsAndLiveHessian) {
  auto p = n::from_vector(std::vector<double>{.2, .3, .4, .5, .6, .7}, {2, 3}).requires_grad_();
  const auto t    = n::from_vector(std::vector<double>{.1, NAN, .5, NAN, .2, NAN}, {2, 3}),
             mask = n::from_vector(std::vector<double>{1, 0, 1, 0, 1, 0}, {2, 3});
  const auto loss = m::ObservedRatingMse(p, t, mask);
  EXPECT_NEAR(loss.item(), .06, 1e-15);
  const auto g  = n::grad(loss, {p}, {.create_graph = true})[0];
  const auto h  = n::grad(n::sum(g), {p})[0];
  const auto gv = g.to_vector<double>(), hv = h.to_vector<double>();
  for (int i = 0; i < 6; ++i) {
    EXPECT_NEAR(
        gv[i], i % 2 ? 0 : 2 * (p.to_vector<double>()[i] - t.to_vector<double>()[i]) / 3, 1e-15);
    EXPECT_NEAR(hv[i], i % 2 ? 0 : 2. / 3, 1e-15);
  }
  auto bad = p.detach().clone();
  {
    n::NoGradGuard guard;
    bad.copy_(n::from_vector(std::vector<double>{.2, NAN, .4, NAN, .6, NAN}, {2, 3}));
  }
  EXPECT_NEAR(m::ObservedRatingMse(bad, t, mask).item(), .06, 1e-15);
  EXPECT_THROW(m::ObservedRatingMse(p, t, n::zeros_like(mask)), std::invalid_argument);
  EXPECT_THROW(m::ObservedRatingMse(p, t, n::full(mask.shape(), .5)), std::invalid_argument);
  EXPECT_THROW(m::ObservedRatingMse(p, n::ones({6}), mask), std::invalid_argument);
  EXPECT_THROW(
      m::ObservedRatingMse(p, t, n::ones(mask.shape(), n::DType::Int32)), std::invalid_argument);
  EXPECT_THROW(m::ObservedRatingMse(p, n::full(t.shape(), NAN), mask), std::invalid_argument);
}
TEST(AutoRec, EveryForwardAndEveryParameterAndInputAdjointMatchesIndependentHost) {
  n::ParameterStore store(4603, n::DType::Float64);
  m::AutoRec(store, n::zeros({2, 256}, n::DType::Float64));
  ASSERT_EQ(store.count(), 16672);
  ASSERT_EQ(store.size(), 4);
  std::vector<double> ew(8192), eb(32), dw(8192), db(256), x(512), up(512);
  for (int i = 0; i < 8192; ++i) {
    ew[i] = .015 * std::sin(i * .13);
    dw[i] = .012 * std::cos(i * .17);
  }
  for (int j = 0; j < 32; ++j)
    eb[j] = .02 * std::sin(j * .31);
  for (int j = 0; j < 256; ++j)
    db[j] = .01 * std::cos(j * .23);
  for (int i = 0; i < 512; ++i) {
    x[i]  = .5 + .2 * std::sin(i * .19);
    up[i] = .1 * std::cos(i * .29);
  }
  {
    n::NoGradGuard guard;
    store.at("encoder/W").copy_(n::from_vector(ew, {256, 32}));
    store.at("encoder/b").copy_(n::from_vector(eb));
    store.at("decoder/W").copy_(n::from_vector(dw, {32, 256}));
    store.at("decoder/b").copy_(n::from_vector(db));
  }
  auto       input     = n::from_vector(x, {2, 256}).requires_grad_();
  const auto output    = m::AutoRec(store, input);
  const auto objective = n::sum(output.prediction * n::from_vector(up, {2, 256}));
  auto       leaves    = store.parameters();
  leaves.push_back(input);
  const auto          gradients = n::grad(objective, leaves);
  std::vector<double> code(64), prediction(512), ge(8192), geb(32), gd(8192), gdb(256), gx(512);
  for (int b = 0; b < 2; ++b) {
    for (int h = 0; h < 32; ++h) {
      double z = eb[h];
      for (int i = 0; i < 256; ++i)
        z += x[b * 256 + i] * ew[i * 32 + h];
      code[b * 32 + h] = 1 / (1 + std::exp(-z));
    }
    for (int j = 0; j < 256; ++j) {
      double y = db[j];
      for (int h = 0; h < 32; ++h)
        y += code[b * 32 + h] * dw[h * 256 + j];
      prediction[b * 256 + j] = y;
      gdb[j] += up[b * 256 + j];
    }
    for (int h = 0; h < 32; ++h) {
      double dy = 0;
      for (int j = 0; j < 256; ++j) {
        gd[h * 256 + j] += code[b * 32 + h] * up[b * 256 + j];
        dy += dw[h * 256 + j] * up[b * 256 + j];
      }
      const double dz = dy * code[b * 32 + h] * (1 - code[b * 32 + h]);
      geb[h] += dz;
      for (int i = 0; i < 256; ++i) {
        ge[i * 32 + h] += x[b * 256 + i] * dz;
        gx[b * 256 + i] += ew[i * 32 + h] * dz;
      }
    }
  }
  const auto actual_code       = output.code.to_vector<double>(),
             actual_prediction = output.prediction.to_vector<double>();
  for (int i = 0; i < 64; ++i)
    EXPECT_NEAR(actual_code[i], code[i], 2e-14);
  for (int i = 0; i < 512; ++i)
    EXPECT_NEAR(actual_prediction[i], prediction[i], 2e-14);
  const std::map<std::string, std::vector<double>> expected{
      {"encoder/W", ge}, {"encoder/b", geb}, {"decoder/W", gd}, {"decoder/b", gdb}};
  for (std::size_t k = 0; k < store.size(); ++k) {
    const auto  actual = gradients[k].to_vector<double>();
    const auto& host   = expected.at(store.names()[k]);
    for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], host[i], 3e-14) << store.names()[k] << ":" << i;
  }
  const auto input_gradient = gradients.back().to_vector<double>();
  for (int i = 0; i < 512; ++i)
    EXPECT_NEAR(input_gradient[i], gx[i], 3e-14);
  const auto before = store.flatten();
  {
    n::NoGradGuard guard;
    const auto     first = m::AutoRec(store, n::take_rows(input, {0}));
    EXPECT_LE(
        n::max(n::abs(first.prediction - n::slice_axis(output.prediction, 0, 0, 1))).item(), 2e-14);
  }
  EXPECT_EQ(store.flatten(), before);
  EXPECT_THROW(m::AutoRec(store, n::zeros({1, 255})), std::invalid_argument);
  EXPECT_THROW(m::AutoRec(store, n::zeros({1, 256}, n::DType::Int32)), std::invalid_argument);
}
TEST(AutoRec, ExplicitStarUnitsClippingAndHeldoutIsolation) {
  const auto p     = n::from_vector(std::vector<double>{0., 6., NAN}),
             truth = n::from_vector(std::vector<double>{2., 4., NAN}),
             mask  = n::from_vector(std::vector<double>{1, 1, 0});
  const auto raw = m::RatingErrors(p, truth, mask), clipped = m::RatingErrors(p, truth, mask, true);
  EXPECT_NEAR(raw.to_vector<double>()[0], 2, 1e-15);
  EXPECT_NEAR(raw.to_vector<double>()[1], 2, 1e-15);
  EXPECT_NEAR(clipped.to_vector<double>()[0], 1, 1e-15);
  EXPECT_NEAR(clipped.to_vector<double>()[1], 1, 1e-15);
  const auto f = m::RatingErrors(n::tensor({2.f, 4.f}), n::tensor({3.f, 4.f}), n::ones({2}));
  EXPECT_NEAR(f.to_vector<double>()[0], std::sqrt(.5), 1e-15);
}
TEST(AutoRec, OriginalMovieLensRecordsWarmIdsAndEveryMaskVerified) {
  const auto root = dm::datasets::prepare_example_data("autorec");
  const auto data = m::LoadAutoRec(root);
  ASSERT_EQ(data.input.shape(), (n::Shape{199, 256}));
  EXPECT_EQ(n::sum(data.train).item(), 8079);
  EXPECT_EQ(n::sum(data.validation).item(), 898);
  EXPECT_EQ(n::sum(data.test).item(), 898);
  EXPECT_EQ(n::sum(n::abs(data.input) * (data.validation + data.test)).item(), 0);
  EXPECT_TRUE(n::all(n::greater(n::sum(data.train, {0}), 0.)).item<bool>());
  const auto folder = std::filesystem::temp_directory_path() / "dm-autorec-invalid-mask";
  std::filesystem::remove_all(folder);
  std::filesystem::create_directories(folder);
  std::filesystem::copy_file(std::filesystem::path(root) / "u.data", folder / "u.data");
  auto entries = n::load_npz((std::filesystem::path(root) / "ratings.npz").string());
  for (auto& [name, value] : entries)
    if (name == "train_mask")
      value = n::zeros_like(value);
  n::save_npz((folder / "ratings.npz").string(), entries);
  EXPECT_THROW(m::LoadAutoRec(folder), std::invalid_argument);
  auto original=n::load_npz((std::filesystem::path(root)/"ratings.npz").string());n::save_npz((folder/"ratings.npz").string(),original);{std::ofstream append(folder/"u.data",std::ios::app);append<<"trailing malformed record\n";}EXPECT_THROW(m::LoadAutoRec(folder),std::invalid_argument);
  std::filesystem::remove_all(folder);
}
