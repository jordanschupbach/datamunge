#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <datamunge/algorithms/floyd_steinberg.hpp>
#include <datamunge/plot/grayscale_images.hpp>
#include <limits>
#include <random>
#include <stdexcept>
using namespace datamunge::algorithms;
using Grid = std::vector<std::vector<double>>;
namespace {
auto bits(double x) {
  return std::bit_cast<std::uint64_t>(x);
}
// Independent exact fixed-point gather: no mutable updates to future source cells.
std::vector<std::vector<int>> rational(const Grid& g) {
  constexpr std::int64_t                 scale = std::int64_t(1) << 40;
  const auto                             rows = g.size(), cols = g[0].size();
  std::vector<std::vector<std::int64_t>> errors(rows, std::vector<std::int64_t>(cols));
  std::vector<std::vector<int>>          out(rows, std::vector<int>(cols));
  for (std::size_t r = 0; r < rows; ++r)
    for (std::size_t c = 0; c < cols; ++c) {
      auto v = std::int64_t(g[r][c] * scale);
      if (r && c)
        v += errors[r - 1][c - 1] / 16;
      if (r)
        v += errors[r - 1][c] * 5 / 16;
      if (r && c + 1 < cols)
        v += errors[r - 1][c + 1] * 3 / 16;
      if (c)
        v += errors[r][c - 1] * 7 / 16;
      out[r][c]    = v < 128 * scale ? 0 : 255;
      errors[r][c] = v - out[r][c] * scale;
    }
  return out;
}
void replay(const FloydSteinbergTrace& t) {
  auto              scratch = t.input;
  std::size_t       white = 0, updates = 0, missing = 0;
  long double       loss = 0;
  const std::size_t rows = t.work.rows, cols = t.work.cols;
  ASSERT_EQ(t.steps.size(), rows * cols);
  for (std::size_t k = 0; k < t.steps.size(); ++k) {
    const auto& s = t.steps[k];
    EXPECT_EQ(s.row, k / cols);
    EXPECT_EQ(s.col, k % cols);
    EXPECT_EQ(bits(s.value_before), bits(scratch[s.row][s.col]));
    EXPECT_EQ(s.quantized, s.value_before < 128 ? 0 : 255);
    EXPECT_EQ(bits(s.error), bits(s.value_before - s.quantized));
    EXPECT_EQ(t.output[s.row][s.col], s.quantized);
    white += s.quantized == 255;
    EXPECT_EQ(s.update_begin, updates);
    const int   dr[]{0, 1, 1, 1}, dc[]{1, -1, 0, 1}, weight[]{7, 3, 5, 1};
    std::size_t discarded = 0;
    for (int d = 0; d < 4; ++d) {
      const auto rr = std::int64_t(s.row) + dr[d], cc = std::int64_t(s.col) + dc[d];
      if (rr < 0 || cc < 0 || rr >= std::int64_t(rows) || cc >= std::int64_t(cols)) {
        discarded += weight[d];
        ++missing;
        continue;
      }
      ASSERT_LT(updates, t.updates.size());
      const auto& u = t.updates[updates++];
      EXPECT_EQ(u.direction, d);
      EXPECT_EQ(u.row, rr);
      EXPECT_EQ(u.col, cc);
      EXPECT_GT(u.row * cols + u.col, k);
      EXPECT_EQ(bits(u.value_before), bits(scratch[rr][cc]));
      EXPECT_EQ(bits(u.transfer), bits(s.error * double(weight[d]) / 16.));
      scratch[rr][cc] += u.transfer;
      EXPECT_EQ(bits(u.value_after), bits(scratch[rr][cc]));
    }
    EXPECT_EQ(s.update_end, updates);
    EXPECT_EQ(s.discarded_numerator, discarded);
    EXPECT_EQ(s.boundary_loss, static_cast<long double>(s.error) * discarded / 16.L);
    loss += s.boundary_loss;
  }
  EXPECT_EQ(updates, t.work.updates);
  EXPECT_EQ(updates, t.updates.size());
  EXPECT_EQ(missing, t.work.discarded_slots);
  EXPECT_EQ(white, t.work.white);
  EXPECT_EQ(t.work.black + white, t.work.pixels);
  EXPECT_EQ(loss, t.work.boundary_loss);
  EXPECT_EQ(updates + missing, t.work.neighbor_slots);
  EXPECT_NEAR(double(t.work.balance_residual), 0., 1e-8);
}
} // namespace
TEST(FloydSteinbergTrace, ExactRationalGatherForAllTernarySmallGridsAndDyadicInputs) {
  for (unsigned code = 0; code < 729; ++code) {
    Grid g(2, std::vector<double>(3));
    auto z = code;
    for (auto& row : g)
      for (auto& v : row) {
        v = 127.5 * (z % 3);
        z /= 3;
      }
    EXPECT_EQ(floyd_steinberg_dither(g), rational(g));
  }
  for (int level = 0; level < 256; ++level) {
    const Grid g(2, std::vector<double>(3, level));
    EXPECT_EQ(floyd_steinberg_dither(g), rational(g));
  }
  std::mt19937_64 rng(117);
  for (int i = 0; i < 4096; ++i) {
    Grid g(2, std::vector<double>(3));
    for (auto& row : g)
      for (auto& v : row)
        v = double(rng() % 65281) / 256.;
    EXPECT_EQ(floyd_steinberg_dither(g), rational(g));
  }
}
TEST(FloydSteinbergTrace, IndependentLongDoubleGatherAndActualRandomReplay) {
  std::mt19937_64 rng(117);
  std::size_t     conditioned = 0;
  for (int fixture = 0; fixture < 500; ++fixture) {
    const std::size_t rows = 1 + rng() % 15, cols = 1 + rng() % 17;
    Grid              g(rows, std::vector<double>(cols));
    for (auto& row : g)
      for (auto& v : row)
        v = double(rng() % 1000001) * 255. / 1000000.;
    std::vector<std::vector<long double>> err(rows, std::vector<long double>(cols));
    std::vector<std::vector<int>>         expected(rows, std::vector<int>(cols));
    long double                           margin = 1000;
    for (std::size_t r = 0; r < rows; ++r)
      for (std::size_t c = 0; c < cols; ++c) {
        long double v = g[r][c];
        if (r && c)
          v += err[r - 1][c - 1] / 16;
        if (r)
          v += err[r - 1][c] * 5 / 16;
        if (r && c + 1 < cols)
          v += err[r - 1][c + 1] * 3 / 16;
        if (c)
          v += err[r][c - 1] * 7 / 16;
        expected[r][c] = v < 128 ? 0 : 255;
        err[r][c]      = v - expected[r][c];
        margin         = std::min(margin, std::abs(v - 128));
      }
    const auto t = floyd_steinberg_trace(g);
    replay(t);
    EXPECT_EQ(t.input, g);
    EXPECT_EQ(t.output, floyd_steinberg_work(g).output);
    if (margin > 1e-8L) {
      ++conditioned;
      EXPECT_EQ(t.output, expected);
    }
  }
  EXPECT_GE(conditioned, 490);
}
TEST(FloydSteinbergTrace, TinyActualUpdatesAndSignedBoundaryBalance) {
  const auto t = floyd_steinberg_trace({{128., 128.}, {128., 128.}});
  EXPECT_EQ(t.output, (std::vector<std::vector<int>>{{255, 0}, {0, 255}}));
  ASSERT_EQ(t.steps.size(), 4);
  ASSERT_EQ(t.updates.size(), 6);
  EXPECT_EQ(t.steps[0].error, -127);
  EXPECT_EQ(t.steps[0].boundary_loss, -23.8125L);
  EXPECT_EQ(t.updates[0].transfer, -55.5625);
  EXPECT_EQ(t.steps[1].value_before, 72.4375);
  EXPECT_EQ(t.steps[2].value_before, 101.89453125);
  EXPECT_EQ(t.steps[3].value_before, 187.278076171875);
  EXPECT_EQ(t.work.input_sum, 512);
  EXPECT_EQ(t.work.output_sum, 510);
  EXPECT_EQ(t.work.boundary_loss, 2);
  EXPECT_EQ(t.work.balance_residual, 0);
  replay(t);
  const auto single = floyd_steinberg_work({{127.5}});
  EXPECT_EQ(single.output[0][0], 0);
  EXPECT_EQ(single.work.boundary_loss, 127.5L);
}
TEST(FloydSteinbergTrace, FullShapeFiniteChecksAndLegacyFractionalCutoff) {
  const auto nan = std::numeric_limits<double>::quiet_NaN(),
             inf = std::numeric_limits<double>::infinity();
  for (const Grid g : {Grid{{128, 128}, {128}}, Grid{{}, {128}}}) {
    EXPECT_THROW(floyd_steinberg_dither(g), std::invalid_argument);
    EXPECT_THROW(floyd_steinberg_trace(g, 1), std::invalid_argument);
  }
  for (double v : {nan, inf, -inf})
    for (const Grid g : {Grid{{0, v}}, Grid{{v, 0}}}) {
      EXPECT_THROW(floyd_steinberg_dither(g), std::invalid_argument);
      EXPECT_THROW(floyd_steinberg_trace(g, 1), std::invalid_argument);
      EXPECT_THROW(threshold_dither(g), std::invalid_argument);
    }
  for (double v : {127.5, 127.75, std::nextafter(128., 0.)})
    EXPECT_EQ(floyd_steinberg_dither({{v}})[0][0], 0);
  EXPECT_EQ(floyd_steinberg_dither({{128.}})[0][0], 255);
  EXPECT_EQ(floyd_steinberg_dither({{-1., 256.}}), (std::vector<std::vector<int>>{{0, 255}}));
  const auto maximum = std::numeric_limits<double>::max();
  EXPECT_EQ(floyd_steinberg_dither({{maximum}})[0][0], 255);
  EXPECT_EQ(floyd_steinberg_dither({{-maximum}})[0][0], 0);
  EXPECT_THROW(floyd_steinberg_dither({{maximum, 0.}}), std::overflow_error);
  EXPECT_THROW(floyd_steinberg_trace({{maximum, 0.}}), std::overflow_error);
  const auto tiny = std::numeric_limits<double>::denorm_min();
  const auto t    = floyd_steinberg_trace({{-0., tiny}, {tiny, tiny}});
  EXPECT_EQ(bits(t.steps[0].value_before), bits(-0.));
  replay(t);
}
TEST(FloydSteinbergTrace, CompleteUpperQuotaEmptyShapesAndOwnedStorage) {
  const Grid g{{128., 128.}, {128., 128.}};
  EXPECT_THROW(floyd_steinberg_trace(g, 31), std::length_error);
  EXPECT_EQ(floyd_steinberg_trace(g, 32).steps.size(), 4);
  EXPECT_EQ(floyd_steinberg_trace(g, 0).steps.size(), 4);
  EXPECT_TRUE(floyd_steinberg_trace({}, 0).input.empty());
  EXPECT_THROW(floyd_steinberg_trace({{}, {}}, 3), std::length_error);
  const auto empty = floyd_steinberg_trace({{}, {}}, 4);
  EXPECT_EQ(empty.input.size(), 2);
  EXPECT_EQ(empty.output.size(), 2);
  EXPECT_TRUE(empty.steps.empty());
  auto source    = g;
  auto t         = floyd_steinberg_trace(source);
  source[0][0]   = 7;
  t.output[0][0] = 0;
  EXPECT_EQ(t.input, g);
  EXPECT_EQ(t.steps[0].quantized, 255);
  using Legacy = std::vector<std::vector<int>> (*)(Grid);
  Legacy fn    = &floyd_steinberg_dither;
  EXPECT_EQ(fn(g), floyd_steinberg_work(g).output);
}
TEST(FloydSteinbergTrace, FullWidthStencilMatchesIndependentSignedGeometry) {
  const int dr[]{0, 1, 1, 1}, dc[]{1, -1, 0, 1}, weights[]{7, 3, 5, 1};
  for (auto rows : std::vector<std::size_t>{0, 1, 2, 17, SIZE_MAX})
    for (auto cols : std::vector<std::size_t>{0, 1, 2, 17, SIZE_MAX})
      for (auto r : std::vector<std::size_t>{0, 1, rows / 2, rows ? rows - 1 : 0, SIZE_MAX})
        for (auto c : std::vector<std::size_t>{0, 1, cols / 2, cols ? cols - 1 : 0, SIZE_MAX}) {
          const auto s = floyd_steinberg_stencil(rows, cols, r, c);
          if (r >= rows || c >= cols) {
            EXPECT_EQ(s.count, 0);
            continue;
          }
          ASSERT_EQ(s.count, 4);
          for (int k = 0; k < 4; ++k) {
            const __int128 a = __int128(r) + dr[k], b = __int128(c) + dc[k];
            const bool     valid = a >= 0 && b >= 0 && a < __int128(rows) && b < __int128(cols);
            const auto&    n     = s.candidates[k];
            EXPECT_EQ(n.direction, k);
            EXPECT_EQ(n.numerator, weights[k]);
            EXPECT_EQ(n.in_bounds, valid);
            EXPECT_EQ(n.row, valid ? std::size_t(a) : 0);
            EXPECT_EQ(n.col, valid ? std::size_t(b) : 0);
          }
        }
}
TEST(FloydSteinbergTrace, OriginalGradientAndColumnBandCountsAreExact) {
  const auto g = dither_gradient(40, 64);
  for (std::size_t r = 0; r < 40; ++r)
    for (std::size_t c = 0; c < 64; ++c)
      EXPECT_EQ(bits(g[r][c]), bits(255. * c / 63.));
  const auto d = floyd_steinberg_work(g);
  for (auto [first, last, white] : std::vector<std::tuple<std::size_t, std::size_t, std::size_t>>{
           {0, 21, 132}, {21, 42, 415}, {42, 64, 738}}) {
    const auto s = dither_region_stats(g, d.output, first, last);
    EXPECT_EQ(s.pixels, 40 * (last - first));
    EXPECT_EQ(s.white, white);
    EXPECT_EQ(s.black + s.white, s.pixels);
    EXPECT_EQ(s.white_fraction, static_cast<long double>(white) / s.pixels);
  }
  EXPECT_EQ(d.work.white, 1285);
  EXPECT_EQ(d.work.updates, 40 * 63 + 39 * 64 + 2 * 39 * 63);
  EXPECT_EQ(d.work.neighbor_slots, 4 * 40 * 64);
  EXPECT_THROW(dither_gradient(0, 3), std::invalid_argument);
  EXPECT_THROW(dither_gradient(3, 0), std::invalid_argument);
  EXPECT_THROW(dither_gradient(SIZE_MAX, 2, 0), std::length_error);
  EXPECT_THROW(dither_gradient(2, 3, 5), std::length_error);
  EXPECT_EQ(dither_gradient(3, 1), (Grid{{0}, {0}, {0}}));
  EXPECT_EQ(dither_gradient(2, 3, 6)[0][1], 127.5);
}
TEST(FloydSteinbergTrace, RegionStatsValidateAllCellsBeforeRangeAndSummarizeControls) {
  EXPECT_THROW(dither_region_stats({{0., 0.}}, {{0, 1}}, 0, 0), std::invalid_argument);
  EXPECT_THROW(dither_region_stats({{0.}}, {{0, 255}}, 0, 1), std::invalid_argument);
  EXPECT_THROW(dither_region_stats({{0.}}, {{0}}, 1, 0), std::invalid_argument);
  EXPECT_THROW(dither_region_stats({{0.}}, {{0}}, 0, 2), std::invalid_argument);
  EXPECT_EQ(dither_region_stats({}, {}, 0, 0).pixels, 0);
  EXPECT_EQ(dither_region_stats({{}, {}}, {{}, {}}, 0, 0).white_fraction, 0);
  const auto out = threshold_dither({{127.75, 128., 255., 0.}});
  EXPECT_EQ(out, (std::vector<std::vector<int>>{{0, 255, 255, 0}}));
  const auto s = dither_region_stats({{127.75, 128., 255., 0.}}, out, 1, 3);
  EXPECT_EQ(s.white, 2);
  EXPECT_EQ(s.output_mean, 255);
  EXPECT_EQ(s.input_mean, 191.5L);
  EXPECT_EQ(s.mean_error, 63.5L);
}
TEST(GrayscaleImages, WholeValidationCapAndBlackWhiteOwnedPixels) {
  using namespace datamunge::graphics;
  EXPECT_THROW(diagram_grayscale_images({}), std::invalid_argument);
  EXPECT_THROW(
      diagram_grayscale_images(std::vector<GrayscaleImagePanel>(3)), std::invalid_argument);
  EXPECT_THROW(
      diagram_grayscale_images({{{{0., 0.}}, "First"}, {{{0.}, {}}, "Ragged"}}, false, 1),
      std::invalid_argument);
  for (double v :
       {-1.,
        256.,
        std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(
        diagram_grayscale_images({{{{0., v}}, "Invalid"}}, false, 1), std::invalid_argument);
  EXPECT_THROW(diagram_grayscale_images({{{{0., 255.}}, "Binary"}}, false, 1), std::length_error);
  EXPECT_NO_THROW(diagram_grayscale_images({{{{0., 255.}}, "Binary"}}, false, 2));
  EXPECT_NO_THROW(diagram_grayscale_images({{{}, "Empty"}}));
  std::vector<GrayscaleImagePanel> panels{{{{0., 127.5, 255.}}, "Gray"}};
  const auto                       d   = diagram_grayscale_images(panels);
  const auto                       svg = d.to_svg();
  panels[0].values[0][0]               = 255;
  EXPECT_EQ(d.to_svg(), svg);
  EXPECT_NE(svg.find("fill=\"rgb(0,0,0)\""), std::string::npos);
  EXPECT_NE(svg.find("fill=\"rgb(128,128,128)\""), std::string::npos);
  EXPECT_NE(svg.find("fill=\"rgb(255,255,255)\""), std::string::npos);
  EXPECT_FALSE(diagram_grayscale_images(panels, true, 0).to_svg().empty());
}
