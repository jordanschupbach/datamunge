#include <gtest/gtest.h>

#include <cmath>
#include <datamunge/algorithms/gram_schmidt_analysis.hpp>
#include <limits>
#include <random>
using namespace datamunge::algorithms;
namespace {
using Matrix = std::vector<std::vector<double>>;
double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double x = 0;
  for (std::size_t i = 0; i < a.size(); ++i)
    x += a[i] * b[i];
  return x;
}
void certificate(const Matrix& v, unsigned passes) {
  GramSchmidtOptions o;
  o.passes     = passes;
  auto       t = gram_schmidt_trace(v, o);
  const auto a = gram_schmidt_run(v, o);
  EXPECT_EQ(t.run.basis, a.basis);
  EXPECT_EQ(t.run.input_rows, a.input_rows);
  if (passes == 1)
    EXPECT_EQ(gram_schmidt(v), a.basis);
  ASSERT_EQ(t.rows.size(), v.size());
  std::size_t k = 0, accepted = 0, entries = 0;
  for (std::size_t row = 0; row < v.size(); ++row) {
    auto u = v[row];
    for (unsigned pass = 0; pass < passes; ++pass)
      for (std::size_t j = 0; j < accepted; ++j) {
        ASSERT_LT(k, t.projections.size());
        const auto& p = t.projections[k++];
        EXPECT_EQ(p.input_row, row);
        EXPECT_EQ(p.basis_row, j);
        EXPECT_EQ(p.pass, pass);
        EXPECT_EQ(p.before, u);
        const double coefficient = dot(u, a.basis[j]);
        EXPECT_EQ(p.coefficient, coefficient);
        for (std::size_t i = 0; i < u.size(); ++i)
          u[i] -= coefficient * a.basis[j][i];
        EXPECT_EQ(p.after, u);
        entries += 1 + 2 * u.size();
      }
    const auto& r = t.rows[row];
    EXPECT_EQ(r.residual, u);
    EXPECT_EQ(r.norm, std::sqrt(dot(u, u)));
    EXPECT_EQ(r.accepted, r.norm > o.tolerance);
    entries += 1 + u.size();
    if (r.accepted) {
      for (double& x : u)
        x /= r.norm;
      EXPECT_EQ(a.basis[accepted], u);
      EXPECT_EQ(a.input_rows[accepted], row);
      ++accepted;
    }
    EXPECT_EQ(r.accepted_rows, accepted);
  }
  EXPECT_EQ(k, t.projections.size());
  EXPECT_EQ(entries, t.captured_entries);
  EXPECT_EQ(a.work.projections, k);
  EXPECT_EQ(a.work.dot_products, k + v.size());
  const auto d = v.empty() ? 0 : v[0].size();
  EXPECT_EQ(a.work.dot_terms, d * (k + v.size()));
  EXPECT_EQ(a.work.update_terms, d * k);
  EXPECT_EQ(a.work.divisions, d * accepted);
  if (entries) {
    o.max_trace_entries = entries;
    EXPECT_NO_THROW(gram_schmidt_trace(v, o));
    if (entries > 1) {
      o.max_trace_entries = entries - 1;
      EXPECT_THROW(gram_schmidt_trace(v, o), std::length_error);
    }
  }
}
} // namespace
TEST(GramSchmidtTrace, ExactStepsAndOriginalPair) {
  auto t = gram_schmidt_trace({{1, 0}, {1, 1}});
  ASSERT_EQ(t.projections.size(), 1);
  EXPECT_EQ(t.projections[0].coefficient, 1);
  EXPECT_EQ(t.projections[0].after, (std::vector<double>{0, 1}));
  EXPECT_EQ(t.captured_entries, 11);
  certificate({{2, 1}, {1, 3}}, 1);
  certificate({{1, 0, 0}, {2, 0, 0}, {0, 1, 0}}, 2);
  const auto q = gram_schmidt({{2, 1}, {1, 3}});
  EXPECT_NEAR(q[0][0], 2 / std::sqrt(5.), 1e-15);
  EXPECT_NEAR(q[1][0], -1 / std::sqrt(5.), 1e-15);
}
TEST(GramSchmidtTrace, RandomActualStatesAndBothPasses) {
  std::mt19937 rng(717);
  for (int i = 0; i < 500; ++i) {
    Matrix v(4, std::vector<double>(6));
    for (auto& r : v)
      for (auto& x : r)
        x = (int(rng() % 2001) - 1000) / 100.;
    certificate(v, 1);
    certificate(v, 2);
  }
}
TEST(GramSchmidtTrace, ThresholdAndReorthogonalizationLimits) {
  EXPECT_EQ(gram_schmidt({{1e-13, 0}, {0, 1e-13}}).size(), 0);
  EXPECT_EQ(gram_schmidt({{1, 0}, {0, 1}}).size(), 2);
  GramSchmidtOptions o;
  o.passes     = 2;
  const auto v = gram_schmidt_near_parallel(1e-8);
  auto       a = gram_schmidt_run(v), b = gram_schmidt_run(v, o);
  EXPECT_EQ(a.basis.size(), 3);
  EXPECT_EQ(b.basis.size(), 3);
  const double olderror =
      gram_schmidt_quality(v, a.basis).get_column<double>("max_gram_error").values()[0];
  const double newerror =
      gram_schmidt_quality(v, b.basis).get_column<double>("max_gram_error").values()[0];
  EXPECT_GT(olderror, 1e-10);
  EXPECT_LT(newerror, 1e-14);
  EXPECT_THROW(gram_schmidt({{1e308, 0}}), std::overflow_error);
  EXPECT_TRUE(gram_schmidt({{1e-200, 0}}).empty());
  EXPECT_TRUE(gram_schmidt({{}, {}}).empty());
  certificate({{}, {}}, 1);
  certificate({}, 2);
}
TEST(GramSchmidtTrace, WholeInputCaptureValidationAndOwnership) {
  GramSchmidtOptions o;
  o.max_input_entries = 5;
  o.max_trace_entries = 1;
  EXPECT_THROW(gram_schmidt_run({{1, 0}, {0, 1}}, o), std::length_error);
  EXPECT_THROW(gram_schmidt_trace({{1, 0}, {0}}, o), std::invalid_argument);
  EXPECT_THROW(
      gram_schmidt_trace({{1, 0}, {std::numeric_limits<double>::quiet_NaN(), 0}}, o),
      std::invalid_argument);
  o.max_input_entries = 6;
  EXPECT_NO_THROW(gram_schmidt_run({{1, 0}, {0, 1}}, o));
  EXPECT_THROW(gram_schmidt_trace({{1, 0}, {0, 1}}, o), std::length_error);
  o.max_trace_entries = 0;
  EXPECT_NO_THROW(gram_schmidt_trace({{1, 0}, {0, 1}}, o));
  for (unsigned pass : {0u, 3u}) {
    o.passes = pass;
    EXPECT_THROW(gram_schmidt_run({}, o), std::invalid_argument);
  }
  o.passes = 1;
  for (double tol :
       {-1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    o.tolerance = tol;
    EXPECT_THROW(gram_schmidt_run({}, o), std::invalid_argument);
  }
  auto t                        = gram_schmidt_trace({{1, 0}, {1, 1}});
  auto copy                     = t;
  copy.projections[0].before[0] = 9;
  copy.rows[0].residual[0]      = 9;
  copy.run.basis[0][0]          = 9;
  EXPECT_EQ(t.projections[0].before[0], 1);
  EXPECT_EQ(t.rows[0].residual[0], 1);
  EXPECT_EQ(t.run.basis[0][0], 1);
}
TEST(GramSchmidtTrace, DataframeAndDiagnosticBoundaries) {
  datamunge::dstruct::DataFrame d;
  d.add_column("x", std::vector<double>{1, 3});
  d.add_column("y", std::vector<double>{3, 1});
  EXPECT_EQ(gram_schmidt_columns(d, {"x", "y"}), (Matrix{{-1, 1}, {1, -1}}));
  d.set_null("x", 0);
  EXPECT_THROW(gram_schmidt_columns(d, {"x"}), std::invalid_argument);
  EXPECT_THROW(gram_schmidt_columns(d, {"missing"}), std::out_of_range);
  EXPECT_THROW(gram_schmidt_quality({{1, 0}}, {{1}}), std::invalid_argument);
  const auto quality = gram_schmidt_quality({{1, 0}}, {});
  EXPECT_EQ(quality.get_column<double>("relative_projection_residual").values()[0], 1);
  EXPECT_THROW(gram_schmidt_scores_plot({}, {}), std::invalid_argument);
}
