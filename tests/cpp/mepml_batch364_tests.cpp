#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/slerp.hpp>
#include <datamunge/algorithms/slr_parser.hpp>
#include <datamunge/algorithms/smith_waterman.hpp>
#include <datamunge/algorithms/sobel.hpp>
#include <datamunge/algorithms/sor.hpp>
#include <datamunge/algorithms/soundex.hpp>
#include <datamunge/algorithms/spectral_layout.hpp>
#include <datamunge/algorithms/spigot.hpp>
#include <datamunge/algorithms/sqrt_digits.hpp>
#include <datamunge/optim/moead.hpp>
#include <datamunge/optim/spea2.hpp>
#include <limits>
#include <numbers>
#include <numeric>
#include <random>
#include <stdexcept>

#include "../../src/datamunge/algorithms/detail/decimal_uint.hpp"
namespace a = datamunge::algorithms;
namespace o = datamunge::optim;
TEST(Mepml364Slerp, ActualBranchesAndConstantRate) {
  const auto target = a::quat_z_rotation(std::numbers::pi / 2);
  for (int i = 0; i <= 20; ++i) {
    const double t = i / 20.;
    auto         r = a::slerp_trace({1, 0, 0, 0}, target, t);
    EXPECT_FALSE(r.normalized_linear);
    EXPECT_NEAR(a::quat_norm(r.value), 1, 1e-14);
    EXPECT_NEAR(a::quat_angle_between(r.start, r.value), t * std::numbers::pi / 2, 1e-14);
  }
  auto r = a::slerp_run({1, 0, 0, 0}, {-1, 0, 0, 0}, .5);
  EXPECT_TRUE(r.sign_flipped);
  EXPECT_TRUE(r.normalized_linear);
  EXPECT_EQ(r.value[0], 1);
  EXPECT_TRUE(a::slerp_run({1, 0, 0, 0}, a::quat_z_rotation(.01), .25).normalized_linear);
}
TEST(Mepml364Slerp, ExtremeNormalizationAndInvalidInputs) {
  EXPECT_EQ(a::quat_normalize({1e308, 0, 0, 0})[0], 1);
  EXPECT_EQ(a::quat_normalize({1e-308, 0, 0, 0})[0], 1);
  EXPECT_THROW(a::quat_normalize({0, 0, 0, 0}), std::invalid_argument);
  EXPECT_THROW(a::slerp({1, 0, 0, 0}, {NAN, 0, 0, 0}, .5), std::invalid_argument);
  EXPECT_THROW(a::slerp({1, 0, 0, 0}, {1, 0, 0, 0}, 1.1), std::invalid_argument);
  EXPECT_THROW(a::quat_dot({1e308, 0, 0, 0}, {1e308, 0, 0, 0}), std::overflow_error);
  EXPECT_NEAR(a::quat_angle_between({1, 0, 0, 0}, a::quat_z_rotation(1e-12)), 1e-12, 1e-26);
}
TEST(Mepml364SLR, ActualConstructionAndLeftRecursion) {
  a::ContextFreeGrammar g{
      {{"E", {"E", "+", "T"}},
       {"E", {"T"}},
       {"T", {"T", "*", "F"}},
       {"T", {"F"}},
       {"F", {"(", "E", ")"}},
       {"F", {"id"}}},
      "E"};
  a::SLRParser parser(g);
  EXPECT_TRUE(parser.is_slr());
  EXPECT_EQ(parser.state_count(), 12);
  auto build = parser.inspection();
  EXPECT_EQ(build.productions.size(), 7);
  EXPECT_TRUE(build.follow.at("E").contains("$"));
  EXPECT_TRUE(build.first.at("E").contains("id"));
  auto r = parser.parse_trace({"id", "+", "id", "*", "id"});
  EXPECT_TRUE(r.accepted);
  EXPECT_EQ(r.consumed, 5);
  EXPECT_EQ(r.steps.back().action, "accept");
  EXPECT_EQ(r.steps.size(), r.actions);
  EXPECT_EQ(r.depths, parser.stack_depth_trace({"id", "+", "id", "*", "id"}));
  EXPECT_FALSE(parser.recognize({"id", "id"}));
  EXPECT_FALSE(parser.recognize({"id", "$", "id"}));
}
TEST(Mepml364SLR, EpsilonConflictsAndLimits) {
  a::ContextFreeGrammar g{{{"S", {"a", "S", "b"}}, {"S", {}}}, "S"};
  a::SLRParser          parser(g);
  EXPECT_TRUE(parser.recognize({}));
  EXPECT_TRUE(parser.recognize({"a", "a", "b", "b"}));
  EXPECT_FALSE(parser.recognize({"a", "b", "b"}));
  a::SLRParser bad({{{"S", {"S", "S"}}, {"S", {"a"}}}, "S"});
  EXPECT_FALSE(bad.is_slr());
  EXPECT_THROW(bad.parse_run({"a"}), std::logic_error);
  EXPECT_THROW(a::SLRParser((a::ContextFreeGrammar{{{"S", {"$"}}}, "S"})), std::invalid_argument);
  a::SLROptions opt;
  opt.max_states = 1;
  EXPECT_THROW(a::SLRParser(g, opt), std::length_error);
  opt             = {};
  opt.max_actions = 1;
  a::SLRParser capped(g, opt);
  EXPECT_THROW(capped.parse_run({"a", "b"}), std::length_error);
  opt                    = {};
  opt.max_history_values = 1;
  a::SLRParser history(g, opt);
  EXPECT_THROW(history.parse_trace({}), std::length_error);
}
TEST(Mepml364Smith, ActualTableSpansAndMotifs) {
  auto r = a::smith_waterman_trace("AAGT", "AGT");
  EXPECT_EQ(r.alignment.score, 6);
  EXPECT_EQ(r.alignment.a_aligned, "AGT");
  EXPECT_EQ(r.alignment.b_aligned, "AGT");
  EXPECT_EQ(r.begin_a, 1);
  EXPECT_EQ(r.end_a, 4);
  EXPECT_EQ(r.begin_b, 0);
  EXPECT_EQ(r.end_b, 3);
  EXPECT_EQ(r.evaluated_cells, 12);
  EXPECT_EQ(r.traceback_steps, 3);
  EXPECT_EQ(r.table[4][3], 6);
  EXPECT_EQ(r.traceback.front().row, 4);
  EXPECT_EQ(a::smith_waterman("TTTGATTACAGGG", "CCGATTACAAA").score, 14);
  EXPECT_EQ(a::smith_waterman("", "A").score, 0);
  EXPECT_EQ(a::smith_waterman("A", "T").score, 0);
}
TEST(Mepml364Smith, OverflowAndCaptureGeometry) {
  EXPECT_THROW(a::smith_waterman("A", "A", {2, -1, 1}), std::invalid_argument);
  EXPECT_THROW(
      a::smith_waterman("AA", "AA", {std::numeric_limits<int>::max(), -1, -2}),
      std::overflow_error);
  a::SmithWatermanOptions opt;
  opt.max_cells = 3;
  EXPECT_THROW(a::smith_waterman_run("A", "A", {}, opt), std::length_error);
  opt                 = {};
  opt.max_trace_steps = 1;
  EXPECT_THROW(a::smith_waterman_trace("AA", "AA", {}, opt), std::length_error);
  auto r = a::smith_waterman_trace("AA", "A");
  EXPECT_EQ(r.begin_a, 0);
  EXPECT_EQ(r.end_a, 1);
}
TEST(Mepml364Sobel, ActualRampContributionsAndOriginalSquare) {
  auto r = a::sobel_trace({{0, 1, 2}, {0, 1, 2}, {0, 1, 2}});
  ASSERT_EQ(r.history.size(), 1);
  EXPECT_EQ(r.result.gx[1][1], 8);
  EXPECT_EQ(r.result.gy[1][1], 0);
  EXPECT_EQ(r.history[0].horizontal_terms, (std::array<double, 9>{0, 0, 2, 0, 0, 4, 0, 0, 2}));
  EXPECT_EQ(r.sample_visits, 9);
  EXPECT_EQ(r.result.magnitude[0][1], 0);
  auto square = a::sobel_run(a::sobel_square_example());
  EXPECT_EQ(square.interior_pixels, 900);
  EXPECT_EQ(square.sample_visits, 8100);
  EXPECT_NEAR(square.maximum_magnitude, std::hypot(765., 765.), 1e-12);
  auto edges = a::sobel_edges(square.result);
  EXPECT_FALSE(edges.pixels.empty());
  for (auto p : edges.pixels)
    EXPECT_GT(square.result.magnitude[p[0]][p[1]], edges.threshold);
  EXPECT_TRUE(a::sobel_edges(square.result, 1).pixels.empty());
}
TEST(Mepml364Sobel, ShapesFiniteAndPolicies) {
  EXPECT_TRUE(a::sobel({}).gx.empty());
  EXPECT_EQ(a::sobel({{1}, {1}}).magnitude[0][0], 0);
  EXPECT_THROW(a::sobel({{1, 2}, {1}}), std::invalid_argument);
  EXPECT_THROW(a::sobel({{NAN}}), std::invalid_argument);
  a::SobelOptions opt;
  opt.max_history_values = 18;
  EXPECT_THROW(a::sobel_trace({{0, 1, 2}, {0, 1, 2}, {0, 1, 2}}, opt), std::length_error);
  EXPECT_THROW(a::sobel_edges(a::sobel({{1}}), 1.1), std::invalid_argument);
  EXPECT_THROW(
      a::sobel({{0, 1e308, 1e308}, {0, 1e308, 1e308}, {0, 1e308, 1e308}}), std::overflow_error);
}
TEST(Mepml364SOR, ActualRelaxedSweepsAndResidual) {
  a::SOROptions opt;
  opt.omega          = 1.2;
  opt.max_iterations = 3;
  opt.tolerance      = 1e-12;
  auto r             = a::sor_trace({{2, -1}, {-1, 2}}, {1, 1}, opt);
  ASSERT_EQ(r.sweeps.size(), 3);
  EXPECT_NEAR(r.sweeps[0].x[0], .6, 1e-15);
  EXPECT_NEAR(r.sweeps[0].x[1], .96, 1e-15);
  for (const auto& s : r.sweeps)
    EXPECT_NEAR(s.residual, std::hypot(2 * s.x[0] - s.x[1] - 1, 2 * s.x[1] - s.x[0] - 1), 1e-14);
  EXPECT_EQ(r.coordinate_updates, 6);
  EXPECT_EQ(r.off_diagonal_visits, 6);
  EXPECT_EQ(r.residual_products, 16);
  EXPECT_FALSE(r.solution.converged);
  EXPECT_TRUE(a::sor({{2, -1}, {-1, 2}}, {1, 1}, 1.2).converged);
}
TEST(Mepml364SOR, InitialResidualAndMalformedControls) {
  a::SOROptions opt;
  opt.max_iterations = 0;
  auto r             = a::sor_run({{2}}, {4}, opt);
  EXPECT_EQ(r.solution.residual, 4);
  EXPECT_FALSE(r.solution.converged);
  EXPECT_EQ(r.solution.iterations, 0);
  EXPECT_TRUE(a::sor_run({{2}}, {0}, opt).solution.converged);
  EXPECT_TRUE(a::sor_run({}, {}).solution.converged);
  EXPECT_THROW(a::sor({{0}}, {1}, 1), std::invalid_argument);
  EXPECT_THROW(a::sor({{1, 2}}, {1}, 1), std::invalid_argument);
  EXPECT_THROW(a::sor({{1}}, {1}, 2), std::invalid_argument);
  opt                        = {};
  opt.max_coefficient_visits = 1;
  EXPECT_THROW(a::sor_run({{2, -1}, {-1, 2}}, {1, 1}, opt), std::length_error);
}
TEST(Mepml364Soundex, ActualBranchesAndOriginalGroups) {
  auto r = a::soundex_trace("Ashcraft");
  EXPECT_EQ(r.code, "A261");
  bool transparent = false, suppressed = false;
  for (const auto& s : r.history) {
    transparent |= s.action == "transparent";
    suppressed |= s.action == "suppress";
  }
  EXPECT_TRUE(transparent);
  EXPECT_TRUE(suppressed);
  EXPECT_EQ(a::soundex("Robert"), a::soundex("Rupert"));
  EXPECT_EQ(a::soundex("BHB"), "B000");
  EXPECT_EQ(a::soundex("BAB"), "B100");
  EXPECT_EQ(a::soundex_groups(a::soundex_example_names()).size(), 12);
  EXPECT_EQ(a::soundex("123Bob"), "1110");
}
TEST(Mepml364Soundex, CompletePolicyAndRawByteRules) {
  EXPECT_TRUE(a::soundex("").empty());
  EXPECT_EQ(a::soundex("robert"), "R163");
  const auto name = "BRTMN" + std::string(10000, 'a');
  EXPECT_EQ(a::soundex_run(name).scanned_bytes, 4);
  a::SoundexOptions opt;
  opt.max_bytes = 5;
  EXPECT_THROW(a::soundex_run(name, opt), std::length_error);
  opt                    = {};
  opt.max_history_values = 1;
  EXPECT_THROW(a::soundex_trace("A", opt), std::length_error);
  EXPECT_EQ(a::soundex(std::string(1, '\0')).size(), 4);
}
TEST(Mepml364SPEA2, PublishedNeighborRankAndActualSelection) {
  auto r = o::spea2_select_trace({{0}, {1}, {3}, {10}}, 2);
  EXPECT_EQ(r.pool[0].strength, 3);
  EXPECT_DOUBLE_EQ(r.pool[0].density, .2);
  EXPECT_DOUBLE_EQ(r.pool[1].raw, 3);
  EXPECT_EQ(r.selected, (std::vector<std::size_t>{0, 1}));
  EXPECT_EQ(r.dominance_tests, 12);
  auto crowd = o::spea2_select_trace({{0, 1}, {.5, .5}, {1, 0}, {.5, .5}}, 2);
  EXPECT_EQ(crowd.selected.size(), 2);
  EXPECT_EQ(crowd.truncated.size(), 2);
  EXPECT_TRUE(o::spea2_select_trace({}, 2).selected.empty());
  EXPECT_THROW(o::spea2_select_trace({{1}, {1, 2}}, 2), std::invalid_argument);
}
TEST(Mepml364SPEA2, ActualSeededGenerationsAndControls) {
  o::ZDT1Objective function(3);
  o::SPEA2Options  opt;
  opt.population_size = 8;
  opt.archive_size    = 4;
  opt.max_generations = 4;
  opt.seed            = 1;
  auto r              = o::SPEA2(opt).optimize_trace(function, {0, 0, 0}, {1, 1, 1});
  EXPECT_EQ(r.evaluations, 32);
  EXPECT_EQ(r.generations.size(), 4);
  EXPECT_EQ(r.generations.front().pool.size(), 8);
  EXPECT_EQ(r.generations.back().pool.size(), 12);
  for (const auto& gen : r.generations)
    for (auto index : gen.selected)
      EXPECT_LT(index, gen.pool.size());
  for (const auto& p : r.front)
    for (double x : p.coordinates)
      EXPECT_TRUE(x >= 0 && x <= 1);
  auto again = o::SPEA2(opt).optimize_run(function, {0, 0, 0}, {1, 1, 1});
  ASSERT_EQ(r.front.size(), again.front.size());
  for (std::size_t i = 0; i < r.front.size(); ++i)
    EXPECT_EQ(r.front[i].objectives, again.front[i].objectives);
  opt.max_generations = 0;
  EXPECT_EQ(o::SPEA2(opt).optimize_run(function, {0, 0, 0}, {1, 1, 1}).evaluations, 8);
  opt.crossover_rate = NAN;
  EXPECT_THROW(o::SPEA2(opt).optimize(function, {0, 0, 0}, {1, 1, 1}), std::invalid_argument);
  opt                 = {};
  opt.max_evaluations = 1;
  EXPECT_THROW(o::SPEA2(opt).optimize(function, {0, 0, 0}, {1, 1, 1}), std::length_error);
}
TEST(Mepml364Spectral, ActualPathModesAndJacobiResiduals) {
  auto r = a::spectral_layout_trace(4, a::spectral_path_edges(4));
  ASSERT_TRUE(r.converged);
  EXPECT_EQ(r.rotations, r.history.size());
  for (std::size_t k = 0; k < 4; ++k) {
    EXPECT_NEAR(r.eigenvalues[k], 2 - 2 * std::cos(std::numbers::pi * k / 4), 1e-12);
    for (std::size_t i = 0; i < 4; ++i) {
      double value = 0;
      for (std::size_t j = 0; j < 4; ++j)
        value += r.laplacian[i][j] * r.eigenvectors[j][k];
      EXPECT_NEAR(value, r.eigenvalues[k] * r.eigenvectors[i][k], 1e-11);
    }
  }
  double mx = 0, my = 0, radius = 0;
  for (const auto& p : r.positions) {
    mx += p.x;
    my += p.y;
    radius = std::max(radius, std::hypot(p.x, p.y));
  }
  EXPECT_NEAR(mx, 0, 1e-14);
  EXPECT_NEAR(my, 0, 1e-14);
  EXPECT_NEAR(radius, 1, 1e-14);
}
TEST(Mepml364Spectral, DegenerateAndExplicitLimits) {
  EXPECT_TRUE(a::spectral_layout_run(0, {}).converged);
  EXPECT_EQ(a::spectral_layout(1, {}).front().x, 0);
  auto r = a::spectral_layout_run(4, {{0, 1}, {2, 3}});
  EXPECT_NEAR(r.eigenvalues[1], 0, 1e-12);
  a::SpectralLayoutOptions opt;
  opt.max_rotations = 1;
  EXPECT_FALSE(a::spectral_layout_run(4, a::spectral_path_edges(4), opt).converged);
  opt                 = {};
  opt.max_scan_visits = 1;
  EXPECT_THROW(a::spectral_layout_run(4, a::spectral_path_edges(4), opt), std::length_error);
  EXPECT_THROW(a::spectral_layout(2, {{0, 2}}), std::invalid_argument);
}
TEST(Mepml364Spigot, ActualCertifiedDigitsAndWork) {
  auto r = a::spigot_e_trace(5);
  EXPECT_EQ(r.digits, "2.71828");
  EXPECT_TRUE(r.certified);
  EXPECT_EQ(r.column_updates, 120);
  EXPECT_EQ(r.history.size(), 5);
  for (std::size_t i = 0; i < r.history.size(); ++i) {
    const auto& s = r.history[i];
    EXPECT_EQ(s.digit, r.digits[i + 2] - '0');
    for (std::size_t j = 2; j < s.remainders.size(); ++j)
      EXPECT_TRUE(s.remainders[j] >= 0 && std::size_t(s.remainders[j]) < j);
  }
  EXPECT_EQ(a::spigot_e(40), "2.7182818284590452353602874713526624977572");
  auto original = a::spigot_e_run(400);
  EXPECT_EQ(
      std::accumulate(original.frequencies.begin(), original.frequencies.end(), std::size_t{0}),
      401);
  EXPECT_TRUE(original.certified);
}
TEST(Mepml364Spigot, ZeroDigitsAndPolicies) {
  EXPECT_EQ(a::spigot_e(0), "2.");
  EXPECT_THROW(a::spigot_e(-1), std::invalid_argument);
  a::SpigotOptions opt;
  opt.max_column_updates = 1;
  EXPECT_THROW(a::spigot_e_run(5, opt), std::length_error);
  opt                    = {};
  opt.max_history_values = 1;
  EXPECT_THROW(a::spigot_e_trace(5, opt), std::length_error);
}
TEST(Mepml364Sqrt, ExactPrefixInequalitiesFullRange) {
  using cpp_int = datamunge::algorithms::detail::DecimalUInt;
  for (std::uint64_t n :
       {std::uint64_t{0},
        std::uint64_t{2},
        std::uint64_t{152},
        std::uint64_t{144},
        std::numeric_limits<std::uint64_t>::max()}) {
    auto    r     = a::sqrt_digits_trace(n, 30);
    cpp_int scale = 1;
    for (int i = 0; i < 30; ++i)
      scale *= 10;
    auto digits = r.digits;
    digits.erase(digits.find('.'), 1);
    cpp_int root = 0;
    for (char c : digits)
      root = root * 10 + c - '0';
    const cpp_int target = cpp_int(n) * scale * scale;
    EXPECT_TRUE(root * root <= target);
    EXPECT_TRUE((root + 1) * (root + 1) > target);
    for (const auto& s : r.history)
      EXPECT_TRUE(cpp_int(s.remainder) >= 0);
  }
  EXPECT_EQ(a::digit_by_digit_sqrt(144, 4), "12.0000");
  EXPECT_EQ(a::digit_by_digit_sqrt(2, 12), "1.414213562373");
}
TEST(Mepml364Sqrt, ActualStepsAndCopiedCharacterPolicies) {
  auto r = a::sqrt_digits_trace(152, 2);
  EXPECT_EQ(r.digits, "12.32");
  ASSERT_EQ(r.history.size(), 4);
  EXPECT_EQ(r.history[0].pair, 1);
  EXPECT_EQ(r.history[1].pair, 52);
  EXPECT_EQ(r.history[1].prefix, "12");
  EXPECT_LE(r.candidate_tests, 40);
  EXPECT_EQ(a::digit_by_digit_sqrt(2, 0), "1.");
  EXPECT_THROW(a::digit_by_digit_sqrt(2, -1), std::invalid_argument);
  a::SqrtDigitsOptions opt;
  opt.max_history_characters = 1;
  EXPECT_THROW(a::sqrt_digits_trace(2, 2, opt), std::length_error);
}
