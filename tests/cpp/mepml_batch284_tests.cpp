#include <gtest/gtest.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cmath>
#include <complex>
#include <datamunge/algorithms/discrete_log.hpp>
#include <datamunge/algorithms/phase_estimation.hpp>
#include <datamunge/algorithms/pigeonhole_sort.hpp>
#include <datamunge/algorithms/pollard_kangaroo.hpp>
#include <datamunge/algorithms/pollard_p_minus_one.hpp>
#include <datamunge/algorithms/pollard_rho.hpp>
#include <datamunge/geometry/point_in_polygon.hpp>
#include <datamunge/geometry/polygon_triangulation.hpp>
#include <limits>
#include <map>
#include <numbers>
#include <numeric>
#include <random>
#include <set>
namespace a = datamunge::algorithms;
namespace g = datamunge::geometry;
namespace {
using U = std::uint64_t;
using W = unsigned __int128;
// Independent test arithmetic; no native modular-power helper participates.
U power(U base, U exponent, U modulus) {
  U r = 1 % modulus;
  base %= modulus;
  while (exponent) {
    if (exponent & 1)
      r = static_cast<U>(W(r) * base % modulus);
    exponent >>= 1;
    if (exponent)
      base = static_cast<U>(W(base) * base % modulus);
  }
  return r;
}
bool prime(U n) {
  if (n < 2)
    return false;
  for (U d = 2; d <= n / d; ++d)
    if (n % d == 0)
      return false;
  return true;
}
long long integer_cross(const g::Point2D& a, const g::Point2D& b, const g::Point2D& c) {
  return static_cast<long long>((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
}
void invariant(const a::RhoLogStep& s, U base, U target, U p, U order) {
  EXPECT_EQ(
      s.before.value,
      static_cast<U>(W(power(base, s.before.a, p)) * power(target, s.before.b, p) % p));
  EXPECT_EQ(
      s.after.value,
      static_cast<U>(W(power(base, s.after.a, p)) * power(target, s.after.b, p) % p));
  EXPECT_LT(s.before.a, order);
  EXPECT_LT(s.after.a, order);
  EXPECT_LT(s.before.b, order);
  EXPECT_LT(s.after.b, order);
  EXPECT_EQ(s.branch, s.before.value % 3);
}
} // namespace
TEST(PhaseActual, IndependentLongDoubleDFTAndEveryCapturedStage) {
  for (int bits = 0; bits <= 7; ++bits)
    for (double phi : {0., .125, .3, .375, .99}) {
      auto       t = a::phase_estimation_trace(bits, phi);
      const auto n = std::size_t{1} << bits;
      ASSERT_EQ(t.distribution.size(), n);
      long double total = 0;
      for (std::size_t k = 0; k < n; ++k) {
        const auto uniform = 1 / std::sqrt(static_cast<long double>(n));
        EXPECT_NEAR(t.after_hadamards[k].real(), uniform, 1e-13);
        EXPECT_NEAR(t.after_hadamards[k].imag(), 0, 1e-13);
        auto kicked = std::polar(uniform, 2 * std::numbers::pi_v<long double> * phi * k);
        EXPECT_NEAR(t.after_kickback[k].real(), kicked.real(), 1e-12);
        EXPECT_NEAR(t.after_kickback[k].imag(), kicked.imag(), 1e-12);
        std::complex<long double> sum{};
        for (std::size_t j = 0; j < n; ++j)
          sum += std::polar(
              1 / static_cast<long double>(n),
              2 * std::numbers::pi_v<long double> * j *
                  (static_cast<long double>(phi) - static_cast<long double>(k) / n));
        EXPECT_NEAR(t.after_inverse_qft[k].real(), sum.real(), 2e-12);
        EXPECT_NEAR(t.after_inverse_qft[k].imag(), sum.imag(), 2e-12);
        EXPECT_NEAR(t.distribution[k], std::norm(sum), 2e-12);
        total += t.distribution[k];
      }
      EXPECT_NEAR(total, 1, 2e-12);
      EXPECT_EQ(
          t.result.measured,
          std::max_element(t.distribution.begin(), t.distribution.end()) - t.distribution.begin());
      EXPECT_EQ(t.result.probability, t.distribution[t.result.measured]);
      EXPECT_EQ(t.hadamards, bits);
      EXPECT_EQ(t.phase_gates, bits);
      auto plain = a::phase_estimation(bits, phi);
      EXPECT_EQ(plain.measured, t.result.measured);
      EXPECT_EQ(plain.probability, t.result.probability);
    }
}
TEST(PhaseActual, OriginalModesCompleteGuardsAndExactAdmission) {
  EXPECT_EQ(a::phase_estimation(3, .375).measured, 3);
  EXPECT_EQ(a::phase_estimation(5, .3).measured, 10);
  EXPECT_THROW(a::phase_estimation(-1, .3), std::invalid_argument);
  EXPECT_THROW(a::phase_estimation(64, .3), std::invalid_argument);
  EXPECT_THROW(a::phase_estimation(3, 1), std::invalid_argument);
  EXPECT_THROW(a::phase_estimation(0, std::nan("")), std::invalid_argument);
  EXPECT_THROW(a::phase_estimation_trace(3, .3, {7, 0}), std::length_error);
  EXPECT_THROW(a::phase_estimation_trace(3, .3, {8, 31}), std::length_error);
  EXPECT_EQ(a::phase_estimation_trace(3, .3, {8, 32}).distribution.size(), 8);
}
TEST(PigeonholeActual, IndependentSortAndEveryPlacementAcrossSignedExtremes) {
  std::mt19937 rng(2);
  for (int repeat = 0; repeat < 1000; ++repeat) {
    std::vector<int> v(2 + rng() % 70);
    for (int& x : v)
      x = std::numeric_limits<int>::min() + int(rng() % 17);
    auto t        = a::pigeonhole_sort_trace(v);
    auto expected = v;
    std::sort(expected.begin(), expected.end());
    EXPECT_EQ(t.run.sorted, expected);
    EXPECT_EQ(a::pigeonhole_sort(v), expected);
    std::map<int, std::size_t> counts;
    for (auto& r : t.placements) {
      EXPECT_EQ(r.value, v[r.input_index]);
      EXPECT_EQ(r.count_after, ++counts[r.value]);
      EXPECT_EQ(
          r.hole, static_cast<std::size_t>(static_cast<std::int64_t>(r.value) - t.run.minimum));
    }
    for (std::size_t i = 0; i < t.occupancies.size(); ++i)
      EXPECT_EQ(
          t.occupancies[i],
          counts[static_cast<int>(
              static_cast<std::int64_t>(t.run.minimum) + static_cast<std::int64_t>(i))]);
    EXPECT_EQ(t.run.placements, v.size());
    EXPECT_EQ(t.run.emitted, v.size());
    EXPECT_EQ(t.run.scanned_holes, t.run.holes);
  }
}
TEST(PigeonholeActual, NoBucketEarlyReturnDenseRangeGuardAndExactHistoryCap) {
  EXPECT_FALSE(a::pigeonhole_sort_trace({}).run.has_range);
  EXPECT_FALSE(a::pigeonhole_sort_trace({1}).run.has_range);
  EXPECT_THROW(a::pigeonhole_sort({INT_MIN, INT_MAX}), std::length_error);
  EXPECT_THROW(a::pigeonhole_sort_trace({-1, 1}, {2, 0}), std::length_error);
  EXPECT_THROW(a::pigeonhole_sort_trace({-1, 1}, {3, 4}), std::length_error);
  EXPECT_EQ(a::pigeonhole_sort_trace({-1, 1}, {3, 5}).run.sorted, (std::vector<int>{-1, 1}));
}
TEST(PohligActual, IndependentCompleteSmallFieldLogsDigitsAndCRT) {
  for (U p : {2, 3, 5, 7, 11, 13, 17, 19, 23, 31, 47, 97}) {
    U base = 1;
    for (U candidate = 1; candidate < p; ++candidate) {
      std::set<U> values;
      for (U x = 0; x < p - 1; ++x)
        values.insert(power(candidate, x, p));
      if (values.size() == p - 1) {
        base = candidate;
        break;
      }
    }
    for (U secret = 0; secret < p - 1; ++secret) {
      auto t = a::pohlig_hellman_trace(base, power(base, secret, p), p, p - 1);
      ASSERT_TRUE(t.run.found);
      EXPECT_EQ(t.run.exponent, secret);
      EXPECT_EQ(a::pohlig_hellman_log(base, power(base, secret, p), p, p - 1), secret);
      U           modulus = 1;
      std::size_t tests   = 0;
      for (auto& f : t.factors) {
        EXPECT_TRUE(prime(f.prime));
        U qe = 1;
        for (unsigned k = 0; k < f.exponent; ++k)
          qe *= f.prime;
        EXPECT_EQ(qe, f.power);
        modulus *= f.power;
      }
      EXPECT_EQ(modulus, p - 1);
      for (auto& d : t.digits) {
        EXPECT_GE(d.digit, 0);
        EXPECT_LT(d.digit, d.prime);
        EXPECT_EQ(d.after, d.before + static_cast<U>(d.digit) * d.prefix_power);
        tests += d.candidate_tests;
      }
      EXPECT_EQ(tests, t.run.work.small_log_tests);
      for (auto& m : t.merges) {
        EXPECT_EQ(m.after % m.modulus, m.residue % m.modulus);
        EXPECT_EQ(m.after % m.previous_modulus, m.previous_value);
        EXPECT_LT(m.after, m.combined_modulus);
      }
    }
  }
}
TEST(PohligActual, OriginalFixtureExactOrderValidationAndPolicyBoundaries) {
  auto t = a::pohlig_hellman_trace(11, power(11, 777, 1009), 1009, 1008);
  EXPECT_EQ(t.run.exponent, 777);
  EXPECT_THROW(a::pohlig_hellman_log(4, 4, 13, 12), std::invalid_argument);
  EXPECT_THROW(a::pohlig_hellman_log(2, 2, 0, 1), std::invalid_argument);
  EXPECT_THROW(a::pohlig_hellman_log(2, 2, 9, 8), std::invalid_argument);
  EXPECT_THROW(a::pohlig_hellman_log(2, 2, 13, 0), std::invalid_argument);
  EXPECT_THROW(a::pohlig_hellman_log(4, 2, 13, 6), std::invalid_argument);
  a::DiscreteLogOptions o;
  o.max_operations = t.run.work.operations;
  EXPECT_EQ(a::pohlig_hellman_run(11, power(11, 777, 1009), 1009, 1008, o).exponent, 777);
  --o.max_operations;
  EXPECT_THROW(a::pohlig_hellman_run(11, power(11, 777, 1009), 1009, 1008, o), std::length_error);
  o.max_operations    = 0;
  o.max_trace_records = t.digits.size() + t.merges.size();
  EXPECT_EQ(a::pohlig_hellman_trace(11, power(11, 777, 1009), 1009, 1008, o).run.exponent, 777);
  --o.max_trace_records;
  EXPECT_THROW(a::pohlig_hellman_trace(11, power(11, 777, 1009), 1009, 1008, o), std::length_error);
}
TEST(RhoLogActual, IndependentGroupInvariantsVerifiedCandidatesAndTrueIterationCounts) {
  for (U secret = 1; secret < 40; ++secret) {
    U    p = 1019, base = 2, target = power(base, secret, p);
    auto t = a::pollard_rho_log_trace(base, target, p, p - 1);
    if (t.run.found)
      EXPECT_EQ(power(base, static_cast<U>(t.run.exponent), p), target);
    else
      EXPECT_EQ(t.run.exponent, -1);
    EXPECT_EQ(t.steps.size(), t.run.work.walk_updates);
    EXPECT_EQ(t.collisions.size(), t.run.work.collisions);
    for (auto& s : t.steps)
      invariant(s, base, target, p, p - 1);
    auto first = a::pollard_rho_log_first_collision(base, target, p, p - 1, 1, 1, 100000);
    EXPECT_EQ(first.steps.size(), 3 * first.iterations);
    EXPECT_TRUE(first.collided);
    EXPECT_EQ(first.tortoise.value, first.hare.value);
    for (auto& s : first.steps)
      invariant(s, base, target, p, p - 1);
  }
}
TEST(RhoLogActual, AnnihilatingOrderNonuniqueOriginalAndCompleteGuards) {
  const U p = 104729, target = power(6, 31337, p);
  auto    r = a::pollard_rho_log_run(6, target, p, p - 1);
  ASSERT_TRUE(r.found);
  EXPECT_EQ(power(6, r.exponent, p), target);
  EXPECT_EQ(power(6, 3777, p), target);
  a::DiscreteLogOptions o;
  o.max_attempts = 0;
  EXPECT_FALSE(a::pollard_rho_log_run(2, power(2, 17, 1019), 1019, 1018, o).found);
  EXPECT_EQ(a::pollard_rho_log_run(2, 1, 1019, 1018, o).exponent, 0);
  EXPECT_THROW(a::pollard_rho_log(2, 2, 0, 0), std::invalid_argument);
  EXPECT_THROW(a::pollard_rho_log(0, 1, 13, 12), std::invalid_argument);
  EXPECT_THROW(a::pollard_rho_log(2, 2, 13, 5), std::invalid_argument);
  EXPECT_THROW(a::pollard_rho_log(2, 2, 13, std::numeric_limits<U>::max()), std::invalid_argument);
  auto t              = a::pollard_rho_log_trace(2, power(2, 17, 1019), 1019, 1018);
  o.max_attempts      = 40;
  o.max_trace_records = t.steps.size() + t.collisions.size();
  EXPECT_TRUE(a::pollard_rho_log_trace(2, power(2, 17, 1019), 1019, 1018, o).run.found);
  --o.max_trace_records;
  EXPECT_THROW(a::pollard_rho_log_trace(2, power(2, 17, 1019), 1019, 1018, o), std::length_error);
  auto study = a::pollard_rho_log_collision_study();
  ASSERT_EQ(study.size(), 6);
  for (auto& s : study) {
    EXPECT_EQ(s.trials, 40);
    EXPECT_EQ(s.collided, 40);
    EXPECT_NEAR(s.mean_iterations, s.mean_legacy_index + 1, 1e-12);
  }
}
TEST(PolygonRayActual, IndependentIntegerWindingAndBoundaryOwnership) {
  const std::vector<g::Point2D> p{{0, 0}, {6, 0}, {6, 4}, {3, 2}, {0, 4}};
  for (int ix = -2; ix <= 14; ++ix)
    for (int iy = -2; iy <= 10; ++iy) {
      g::Point2D q{ix * .5, iy * .5};
      int        winding  = 0;
      bool       boundary = false;
      for (std::size_t i = 0; i < p.size(); ++i) {
        auto         a = p[i], b = p[(i + 1) % p.size()];
        const double c = (b.x - a.x) * (q.y - a.y) - (b.y - a.y) * (q.x - a.x);
        boundary |= c == 0 && q.x >= std::min(a.x, b.x) && q.x <= std::max(a.x, b.x) &&
                    q.y >= std::min(a.y, b.y) && q.y <= std::max(a.y, b.y);
        if (a.y <= q.y && b.y > q.y && c > 0)
          ++winding;
        if (a.y > q.y && b.y <= q.y && c < 0)
          --winding;
      }
      auto t = g::point_in_polygon_trace(q, p);
      EXPECT_EQ(t.location == g::PolygonLocation::Boundary, boundary);
      if (!boundary)
        EXPECT_EQ(t.inside, winding != 0);
      bool parity = false;
      for (auto& e : t.edges) {
        if (e.toggled)
          parity = !parity;
        EXPECT_EQ(e.inside_after, parity);
      }
      EXPECT_EQ(parity, t.inside);
    }
  const std::vector<g::Point2D> s{{0, 0}, {2, 0}, {2, 2}, {0, 2}};
  EXPECT_TRUE(g::point_in_polygon({0, 1}, s));
  EXPECT_FALSE(g::point_in_polygon({2, 1}, s));
  EXPECT_EQ(g::classify_point_in_polygon({0, 1}, s), g::PolygonLocation::Boundary);
  EXPECT_EQ(g::classify_point_in_polygon({2, 1}, s), g::PolygonLocation::Boundary);
}
TEST(PolygonRayActual, ExactNeighboringValuesFullExponentRangeAndGuards) {
  const std::vector<g::Point2D> s{{0, 0}, {1, 0}, {1, 1}, {0, 1}};
  EXPECT_EQ(
      g::classify_point_in_polygon({std::nextafter(1., 0.), .5}, s), g::PolygonLocation::Inside);
  EXPECT_EQ(
      g::classify_point_in_polygon({std::nextafter(1., 2.), .5}, s), g::PolygonLocation::Outside);
  const double m = std::numeric_limits<double>::max();
  EXPECT_TRUE(g::point_in_polygon({0, 0}, {{-m, -m}, {m, -m}, {m, m}, {-m, m}}));
  EXPECT_EQ(g::classify_point_in_polygon({1, 1}, {{1, 1}}), g::PolygonLocation::Boundary);
  EXPECT_EQ(g::classify_point_in_polygon({0, 0}, {}), g::PolygonLocation::Outside);
  EXPECT_THROW(g::point_in_polygon({std::nan(""), 0}, {}), std::invalid_argument);
  EXPECT_THROW(g::point_in_polygon({0, 0}, {{0, 0}, {std::nan(""), 0}}), std::invalid_argument);
  EXPECT_THROW(g::point_in_polygon_trace({.5, .5}, s, 3), std::length_error);
  EXPECT_EQ(g::point_in_polygon_trace({.5, .5}, s, 4).edges.size(), 4);
}
TEST(EarActual, IndependentIntegerAreaEdgeIncidenceAndPostErasureRings) {
  std::vector<std::vector<g::Point2D>> cases{
      {{0, 0}, {4, 0}, {4, 4}, {0, 4}},
      {{0, 0}, {4, 2}, {8, 0}, {6, 4}, {8, 8}, {4, 6}, {0, 8}},
      {{0, 0}, {1, 0}, {2, 0}, {2, 2}, {0, 2}},
      {{0, 0}, {4, 0}, {4, 2}, {2, 2}, {2, 4}, {0, 4}}};
  for (auto polygon : cases)
    for (int reverse = 0; reverse < 2; ++reverse) {
      if (reverse)
        std::reverse(polygon.begin(), polygon.end());
      auto t = g::triangulate_polygon_trace(polygon);
      ASSERT_EQ(t.run.triangles.size(), polygon.size() - 2);
      long long                                          sum = 0, area = 0;
      std::map<std::pair<std::size_t, std::size_t>, int> edges;
      for (std::size_t i = 0; i < polygon.size(); ++i)
        area += static_cast<long long>(
            polygon[i].x * polygon[(i + 1) % polygon.size()].y -
            polygon[i].y * polygon[(i + 1) % polygon.size()].x);
      for (auto& q : t.run.triangles) {
        ASSERT_LT(q.a, polygon.size());
        ASSERT_LT(q.b, polygon.size());
        ASSERT_LT(q.c, polygon.size());
        auto twice = integer_cross(polygon[q.a], polygon[q.b], polygon[q.c]);
        EXPECT_GT(twice, 0);
        sum += twice;
        for (auto e : {std::pair(q.a, q.b), std::pair(q.b, q.c), std::pair(q.c, q.a)})
          ++edges[std::minmax(e.first, e.second)];
      }
      EXPECT_EQ(sum, std::abs(area));
      for (std::size_t i = 0; i < polygon.size(); ++i) {
        std::pair<std::size_t, std::size_t> key = std::minmax(i, (i + 1) % polygon.size());
        EXPECT_EQ(edges[key], 1);
        edges.erase(key);
      }
      for (auto [edge, count] : edges)
        EXPECT_EQ(count, 2);
      auto ring = t.initial_ring;
      for (auto& clip : t.clips) {
        auto at = std::find(ring.begin(), ring.end(), clip.triangle.b);
        ASSERT_NE(at, ring.end());
        ring.erase(at);
        EXPECT_EQ(ring, clip.remaining);
      }
      EXPECT_EQ(ring.size(), 3);
    }
}
TEST(EarActual, CompleteSimplicityGuardsExactPoliciesAndFullRangePredicates) {
  EXPECT_THROW(g::triangulate_polygon({{0, 0}, {2, 2}, {0, 2}, {2, 0}}), std::invalid_argument);
  EXPECT_THROW(g::triangulate_polygon({{0, 0}, {2, 0}, {0, 2}, {0, 0}}), std::invalid_argument);
  EXPECT_THROW(g::triangulate_polygon({{0, 0}, {1, 0}, {2, 0}}), std::invalid_argument);
  EXPECT_THROW(g::triangulate_polygon({{std::nan(""), 0}}), std::invalid_argument);
  const std::vector<g::Point2D>  s{{0, 0}, {2, 0}, {2, 2}, {0, 2}};
  auto                           t = g::triangulate_polygon_trace(s);
  g::PolygonTriangulationOptions o;
  o.max_predicates = t.run.orientations;
  EXPECT_EQ(g::triangulate_polygon_run(s, o).triangles.size(), 2);
  --o.max_predicates;
  EXPECT_THROW(g::triangulate_polygon_run(s, o), std::length_error);
  o.max_predicates    = 0;
  o.max_trace_entries = t.initial_ring.size() + t.candidates.size() + t.clips.size();
  for (auto& clip : t.clips)
    o.max_trace_entries += clip.remaining.size();
  EXPECT_EQ(g::triangulate_polygon_trace(s, o).clips.size(), 1);
  --o.max_trace_entries;
  EXPECT_THROW(g::triangulate_polygon_trace(s, o), std::length_error);
  const double m = std::numeric_limits<double>::max();
  EXPECT_EQ(g::triangulate_polygon({{-m, -m}, {m, -m}, {m, m}, {-m, m}}).size(), 2);
}
TEST(PMinusActual, IndependentFactorialPowersAndFirstGcdDecisions) {
  for (U n = 3; n < 300; n += 2)
    for (U bound = 0; bound <= 12; ++bound) {
      auto        t         = a::pollard_p_minus_one_trace(n, bound);
      U           factorial = 1, divisor = 0;
      std::size_t count = 0;
      for (U j = 2; j <= bound; ++j) {
        factorial *= j;
        const auto residue = power(2, factorial, n), d = std::gcd(residue ? residue - 1 : n, n);
        ASSERT_LT(count, t.stages.size());
        auto& s = t.stages[count++];
        EXPECT_EQ(s.j, j);
        EXPECT_EQ(s.after, residue);
        EXPECT_EQ(s.gcd, d);
        if (d > 1 && d < n) {
          divisor = d;
          break;
        }
        if (d == n)
          break;
      }
      EXPECT_EQ(t.run.divisor, divisor);
      EXPECT_EQ(t.run.stages, count);
      EXPECT_EQ(a::pollard_p_minus_1(n, bound), divisor);
    }
}
TEST(PMinusActual, FactorialMultiplicityCollapseSmallPrimeAndPolicies) {
  EXPECT_EQ(a::pollard_p_minus_1(391, 3), 0);
  EXPECT_EQ(a::pollard_p_minus_1(391, 4), 17);
  EXPECT_EQ(a::pollard_p_minus_one_run(85, 4).stop, a::PMinusOneStop::Collapsed);
  EXPECT_EQ(a::pollard_p_minus_1(2, 20), 0);
  EXPECT_EQ(a::pollard_p_minus_1(14, 0), 2);
  EXPECT_THROW(a::pollard_p_minus_1(0, 1), std::invalid_argument);
  EXPECT_THROW(a::pollard_p_minus_one_trace(391, 4, {2, 0}), std::length_error);
  EXPECT_THROW(a::pollard_p_minus_one_trace(391, 4, {3, 2}), std::length_error);
  EXPECT_EQ(a::pollard_p_minus_one_trace(391, 4, {3, 3}).run.divisor, 17);
  EXPECT_EQ(a::pollard_p_minus_1(391, std::numeric_limits<U>::max()), 17);
}
TEST(KangarooActual, IndependentEveryJumpTableDistanceAndVerifiedCollision) {
  for (U secret : {7, 15, 23}) {
    const U p = 1019, lo = secret - 3, hi = secret + 3;
    auto    t = a::pollard_kangaroo_trace(2, power(2, secret, p), p, lo, hi);
    EXPECT_EQ(t.run.result.jumps, t.jumps.size());
    EXPECT_EQ(t.table.size(), 32 * t.run.attempts);
    std::map<std::pair<std::size_t, unsigned>, a::KangarooJumpSize> table;
    for (auto& r : t.table) {
      EXPECT_EQ(r.power, power(2, r.step, p));
      table[{r.attempt, r.index}] = r;
    }
    std::map<std::pair<std::size_t, bool>, W> distance;
    for (auto& r : t.jumps) {
      auto e = table.at({r.attempt, r.slot});
      EXPECT_EQ(r.slot, r.before % 32);
      EXPECT_EQ(r.after, static_cast<U>(W(r.before) * e.power % p));
      distance[{r.attempt, r.wild}] += r.step;
      EXPECT_TRUE((r.distance_after == distance[{r.attempt, r.wild}]));
      if (!r.wild)
        EXPECT_TRUE(r.trail_exponent_after == W(hi) + r.distance_after);
    }
    for (auto& c : t.collisions)
      if (c.verified) {
        EXPECT_TRUE(c.in_interval);
        EXPECT_GE(c.candidate, lo);
        EXPECT_LE(c.candidate, hi);
        EXPECT_EQ(power(2, c.candidate, p), power(2, secret, p));
      }
    if (t.run.result.found)
      EXPECT_EQ(t.run.result.x, secret);
  }
}
TEST(KangarooActual, OriginalSecretsFullWidthTrailAndPreflightGuards) {
  for (U secret : {123456, 654321, 999001}) {
    auto r =
        a::pollard_kangaroo(2, power(2, secret, 1000003), 1000003, secret - 20000, secret + 20000);
    ASSERT_TRUE(r.found);
    EXPECT_EQ(r.x, secret);
  }
  const U top = std::numeric_limits<U>::max();
  auto    t   = a::pollard_kangaroo_trace(2, power(2, top - 10, 1019), 1019, top - 20, top);
  ASSERT_FALSE(t.jumps.empty());
  EXPECT_TRUE(t.jumps[0].trail_exponent_after > W(top));
  ASSERT_TRUE(t.run.result.found);
  EXPECT_EQ(t.run.result.x, top - 10);
  EXPECT_THROW(a::pollard_kangaroo(2, 2, 0, 0, 1), std::invalid_argument);
  EXPECT_THROW(a::pollard_kangaroo(6, 2, 15, 0, 1), std::invalid_argument);
  EXPECT_THROW(a::pollard_kangaroo_run(2, 2, 1019, 0, top), std::length_error);
  a::KangarooOptions o;
  o.max_attempts = 0;
  EXPECT_FALSE(a::pollard_kangaroo_run(2, power(2, 7, 1019), 1019, 5, 9, 0, o).result.found);
  auto endpoint = a::pollard_kangaroo(2, power(2, 7, 1019), 1019, 7, 9);
  EXPECT_TRUE(endpoint.found);
  EXPECT_EQ(endpoint.jumps, 0);
}
TEST(BrentActual, IndependentTrialFactorsEveryPolynomialAndActualGcd) {
  for (U n = 1; n < 2000; ++n) {
    auto factors = a::factorize(n);
    U    product = 1;
    for (U p : factors) {
      EXPECT_TRUE(prime(p));
      product *= p;
    }
    EXPECT_EQ(product, n);
    EXPECT_TRUE(std::is_sorted(factors.begin(), factors.end()));
  }
  for (U n : {8051ULL, 600851475143ULL, 18446744073709551615ULL}) {
    auto t       = a::factorize_trace(n);
    W    product = 1;
    for (U p : t.factors)
      product *= p;
    EXPECT_TRUE(product == W(n));
    EXPECT_EQ(t.history.evaluations.size(), t.work.evaluations);
    EXPECT_EQ(t.history.gcds.size(), t.work.gcd_calls);
    for (auto& e : t.history.evaluations) {
      const U expected = static_cast<U>((W(e.before) * e.before + e.constant) % e.modulus);
      EXPECT_EQ(e.after, expected);
    }
    for (auto& e : t.history.gcds)
      EXPECT_EQ(e.divisor, std::gcd(e.product, e.modulus));
    for (auto& s : t.history.splits) {
      EXPECT_GT(s.divisor, 1);
      EXPECT_LT(s.divisor, s.composite);
      EXPECT_EQ(s.composite % s.divisor, 0);
    }
  }
}
TEST(BrentActual, FullWidthAdditionSharedOrbitAndExactHistoryEvaluationAdmission) {
  const U top = std::numeric_limits<U>::max();
  for (U n : std::vector<U>{top, top - 58}) {
    auto w = a::rho_polynomial_walk(n, n - 1, n - 100, 2);
    EXPECT_EQ(w.states[1], 9999);
  }
  auto orbit = a::rho_polynomial_walk(8051, 1, 2, 16, 83);
  EXPECT_TRUE(orbit.repeated);
  EXPECT_EQ(orbit.period, 5);
  for (std::size_t i = 1; i < orbit.states.size(); ++i)
    EXPECT_EQ(
        orbit.states[i], static_cast<U>((W(orbit.states[i - 1]) * orbit.states[i - 1] + 1) % 8051));
  auto                t = a::pollard_rho_factor_trace(8051);
  a::RhoFactorOptions o;
  o.max_evaluations = t.work.evaluations;
  EXPECT_EQ(a::pollard_rho_factor_run(8051, o).factor, t.factor);
  --o.max_evaluations;
  EXPECT_THROW(a::pollard_rho_factor_run(8051, o), std::length_error);
  o.max_evaluations   = 0;
  o.max_trace_records = t.history.evaluations.size() + t.history.gcds.size();
  EXPECT_EQ(a::pollard_rho_factor_trace(8051, o).factor, t.factor);
  --o.max_trace_records;
  EXPECT_THROW(a::pollard_rho_factor_trace(8051, o), std::length_error);
  EXPECT_THROW(a::factorize(0), std::invalid_argument);
  EXPECT_THROW(a::rho_polynomial_walk(8051, 1, 2, 2, 17), std::invalid_argument);
  EXPECT_THROW(a::rho_polynomial_walk(8051, 1, 2, 2, 83, 3), std::length_error);
}
