/// @file fuzzy_c_means.cpp
/// @brief Shared ordered Euclidean fuzzy c-means kernel and bounded actual diagnostics.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/fuzzy_c_means.hpp>
#include <limits>
#include <random>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private checked shapes, numerical updates and history admission.
namespace fcm_detail {
/// @brief Add two entry counts.
/// @param a First count.
/// @param b Second count.
/// @return Checked sum.
/// @throws std::length_error Overflow.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > SIZE_MAX - a)
    throw std::length_error("FCM count overflow");
  return a + b;
}
/// @brief Multiply two entry counts.
/// @param a First count.
/// @param b Second count.
/// @return Checked product.
/// @throws std::length_error Overflow.
std::size_t mul(std::size_t a, std::size_t b) {
  if (a && b > SIZE_MAX / a)
    throw std::length_error("FCM count overflow");
  return a * b;
}
/// @brief Reject nonfinite actual arithmetic.
/// @param x Actual computed value.
/// @return Nothing.
/// @throws std::overflow_error Nonfinite value.
void finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("FCM nonfinite arithmetic");
}
/// @brief Validate rectangular matrix shape without inspecting values.
/// @param x Input rows.
/// @return Width,0 for empty outer vector.
/// @throws std::invalid_argument Ragged rows.
std::size_t width(const FuzzyCMeansMatrix& x) {
  const auto d = x.empty() ? 0 : x[0].size();
  for (const auto& row : x)
    if (row.size() != d)
      throw std::invalid_argument("FCM ragged matrix");
  return d;
}
/// @brief Validate every finite matrix value.
/// @param x Rectangular input.
/// @return Nothing.
/// @throws std::invalid_argument Nonfinite entry.
void values(const FuzzyCMeansMatrix& x) {
  for (const auto& row : x)
    for (double v : row)
      if (!std::isfinite(v))
        throw std::invalid_argument("FCM finite entries required");
}
/// @brief Validate the exponent and optional stopping tolerance.
/// @param m Fuzzifier.
/// @param tol Nonnegative stopping tolerance.
/// @return Nothing.
/// @throws std::invalid_argument Invalid finite parameter range.
void parameters(double m, double tol = 0) {
  if (!std::isfinite(m) || m <= 1 || !std::isfinite(tol) || tol < 0)
    throw std::invalid_argument("FCM requires finite m>1 and tol>=0");
}
/// @brief Validate membership row count and rectangular width.
/// @param u Membership rows.
/// @param n Expected point count.
/// @param c Expected cluster count.
/// @return Nothing.
/// @throws std::invalid_argument Inconsistent shape.
void membership_shape(const FuzzyCMeansMatrix& u, std::size_t n, std::size_t c) {
  const auto d = width(u);
  if (u.size() != n || (!u.empty() && d != c))
    throw std::invalid_argument("FCM membership shape");
}
/// @brief Validate finite unit-interval rows and their long-double unit sums.
/// @param u Rectangular memberships.
/// @return Nothing; inputs are never normalized.
/// @throws std::invalid_argument Invalid value or row sum.
void simplex(const FuzzyCMeansMatrix& u) {
  for (const auto& row : u) {
    long double sum = 0;
    for (double x : row) {
      if (!std::isfinite(x) || x < 0 || x > 1)
        throw std::invalid_argument("FCM membership value");
      sum += x;
    }
    if (std::fabs(sum - 1.L) > 1e-12L)
      throw std::invalid_argument("FCM membership row sum");
  }
}
/// @brief Validate data/center dimensions before any value checks.
/// @param data Query rows.
/// @param centers Center rows.
/// @return Shared coordinate width.
/// @throws std::invalid_argument Empty centers or incompatible/ragged shapes.
std::size_t query_shape(const FuzzyCMeansMatrix& data, const FuzzyCMeansMatrix& centers) {
  const auto d = width(data), cd = width(centers);
  if (centers.empty() || (!data.empty() && d != cd))
    throw std::invalid_argument("FCM point/center dimensions");
  return cd;
}
/// @brief Bound all actual work counters before fitting.
/// @param n Points.
/// @param d Coordinates.
/// @param c Clusters.
/// @param t Requested maximum updates.
/// @param capture Whether diagnostic objectives are evaluated.
/// @return Nothing.
/// @throws std::length_error Counter upper bound overflow.
void work_bounds(std::size_t n, std::size_t d, std::size_t c, std::size_t t, bool capture) {
  const auto nc = mul(n, c), steps = mul(nc, t);
  mul(steps, d);
  mul(steps, c);
  mul(mul(c, d), t);
  if (capture)
    mul(nc, add(mul(t, 2), 1));
}
/// @brief Compute the ordered checked squared Euclidean distance.
/// @param a Point coordinates.
/// @param b Same-width center coordinates.
/// @return Finite squared distance.
/// @throws std::overflow_error Actual nonfinite arithmetic.
double sq_dist(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    s += d * d;
    finite(s);
  }
  return s;
}
/// @brief Evaluate one ordered objective on validated matrices.
/// @param data Points.
/// @param centers Centers.
/// @param u Memberships.
/// @param m Exponent.
/// @param pairs Actual pair counter to increment.
/// @return Finite ordered double objective.
/// @throws std::overflow_error Actual nonfinite arithmetic.
double objective(
    const FuzzyCMeansMatrix& data,
    const FuzzyCMeansMatrix& centers,
    const FuzzyCMeansMatrix& u,
    double                   m,
    std::size_t&             pairs) {
  double J = 0;
  for (std::size_t i = 0; i < data.size(); ++i)
    for (std::size_t j = 0; j < centers.size(); ++j) {
      J += std::pow(u[i][j], m) * sq_dist(data[i], centers[j]);
      ++pairs;
      finite(J);
    }
  return J;
}
/// @brief Assign memberships in the original distance-ratio order.
/// @param data Validated points.
/// @param centers Validated centers.
/// @param u Mutable current memberships; owns rows.
/// @param m Exponent.
/// @param w Actual work counters.
/// @param captured Optional owned actual distances to fill.
/// @return Largest actual membership change.
/// @throws std::overflow_error Actual nonfinite arithmetic.
double update(
    const FuzzyCMeansMatrix& data,
    const FuzzyCMeansMatrix& centers,
    FuzzyCMeansMatrix&       u,
    double                   m,
    FuzzyCMeansWork&         w,
    FuzzyCMeansMatrix*       captured = nullptr) {
  const auto   c          = centers.size();
  const double p          = 2.0 / (m - 1.0);
  double       max_change = 0;
  if (captured)
    captured->resize(data.size());
  for (std::size_t i = 0; i < data.size(); ++i) {
    std::vector<double> dist(c, 0.0);
    bool                coincident   = false;
    std::size_t         coincident_j = 0;
    for (std::size_t j = 0; j < c; ++j) {
      dist[j] = std::sqrt(sq_dist(data[i], centers[j]));
      ++w.membership_distances;
      if (dist[j] < 1e-12) {
        coincident   = true;
        coincident_j = j;
      }
    }
    if (captured)
      (*captured)[i] = dist;
    for (std::size_t j = 0; j < c; ++j) {
      double newu;
      if (coincident)
        newu = j == coincident_j ? 1.0 : 0.0;
      else {
        double sum = 0;
        for (std::size_t l = 0; l < c; ++l) {
          sum += std::pow(dist[j] / dist[l], p);
          ++w.ratio_powers;
          finite(sum);
        }
        newu = 1.0 / sum;
        finite(newu);
      }
      max_change = std::max(max_change, std::abs(newu - u[i][j]));
      u[i][j]    = newu;
      ++w.membership_updates;
    }
  }
  return max_change;
}
/// @brief Execute one validated/admitted fit, optionally capturing actual updates.
/// @param data Input points.
/// @param c Cluster count.
/// @param m Fuzzifier.
/// @param t Maximum updates.
/// @param tol Strict stopping tolerance.
/// @param seed Random initialization seed.
/// @param capture Whether owned history and phase objectives are needed.
/// @param cap Optional whole-entry quota.
/// @param supplied Optional validated-format initial membership input.
/// @return Owned fit and optional history.
/// @throws std::invalid_argument Invalid shapes, values or parameters.
/// @throws std::length_error Counts/container/quota limits.
/// @throws std::overflow_error Nonfinite actual arithmetic.
/// @throws std::underflow_error Positive cluster weights all underflow.
FuzzyCMeansTrace run(
    const FuzzyCMeansMatrix& data,
    std::size_t              c,
    double                   m,
    std::size_t              t,
    double                   tol,
    std::uint64_t            seed,
    bool                     capture,
    std::size_t              cap,
    const FuzzyCMeansMatrix* supplied = nullptr) {
  const auto n = data.size(), d = width(data);
  if (supplied)
    membership_shape(*supplied, n, c);
  values(data);
  if (supplied)
    simplex(*supplied);
  parameters(m, tol);
  if (!c || c > n)
    throw std::invalid_argument("FCM needs 0<c<=n");
  work_bounds(n, d, c, t, capture);
  if (capture) {
    const auto count = fuzzy_c_means_trace_entries(n, d, c, t);
    if (cap && count > cap)
      throw std::length_error("FCM trace admission quota");
  }
  FuzzyCMeansTrace  out;
  auto&             w = out.work;
  FuzzyCMeansMatrix u(n, std::vector<double>(c, 0.0));
  if (supplied)
    u = *supplied;
  else {
    std::mt19937_64                        rng(seed);
    std::uniform_real_distribution<double> unit(.1, 1.0);
    for (std::size_t i = 0; i < n; ++i) {
      double sum = 0;
      for (std::size_t j = 0; j < c; ++j) {
        u[i][j] = unit(rng);
        sum += u[i][j];
        ++w.initial_draws;
      }
      for (std::size_t j = 0; j < c; ++j)
        u[i][j] /= sum;
    }
  }
  FuzzyCMeansMatrix centers(c, std::vector<double>(d, 0.0));
  double            previous = 0;
  if (capture) {
    out.data               = data;
    out.initial_membership = u;
    out.initial_centers    = centers;
    out.initial_objective  = objective(data, centers, u, m, w.diagnostic_objective_pairs);
    previous               = out.initial_objective;
  }
  for (std::size_t iter = 0; iter < t; ++iter) {
    FuzzyCMeansFrame f;
    f.iteration        = iter;
    f.objective_before = previous;
    if (capture) {
      f.numerators.resize(c);
      f.denominators.resize(c);
    }
    for (std::size_t j = 0; j < c; ++j) {
      std::vector<double> num(d, 0.0);
      double              den      = 0;
      bool                positive = false;
      for (std::size_t i = 0; i < n; ++i) {
        const double weight = std::pow(u[i][j], m);
        ++w.center_weights;
        finite(weight);
        positive = positive || u[i][j] > 0;
        for (std::size_t k = 0; k < d; ++k) {
          num[k] += weight * data[i][k];
          ++w.weighted_coordinates;
          finite(num[k]);
        }
        den += weight;
        finite(den);
      }
      if (capture) {
        f.numerators[j]   = num;
        f.denominators[j] = den;
      }
      if (den > 0)
        for (std::size_t k = 0; k < d; ++k) {
          centers[j][k] = num[k] / den;
          finite(centers[j][k]);
          ++w.center_divisions;
        }
      else if (positive)
        throw std::underflow_error("FCM all positive cluster weights underflow");
    }
    if (capture)
      f.objective_centers = objective(data, centers, u, m, w.diagnostic_objective_pairs);
    out.max_change        = update(data, centers, u, m, w, capture ? &f.distances : nullptr);
    out.result.iterations = iter + 1;
    if (capture) {
      f.centers              = centers;
      f.membership           = u;
      f.max_change           = out.max_change;
      f.objective_membership = objective(data, centers, u, m, w.diagnostic_objective_pairs);
      previous               = f.objective_membership;
      out.frames.push_back(std::move(f));
    }
    if (out.max_change < tol) {
      out.stopped_by_tolerance = true;
      break;
    }
  }
  out.result.objective  = objective(data, centers, u, m, w.final_objective_pairs);
  out.result.centers    = std::move(centers);
  out.result.membership = std::move(u);
  return out;
}
} // namespace fcm_detail
FuzzyCMeansResult fuzzy_c_means(
    const FuzzyCMeansMatrix& data,
    std::size_t              c,
    double                   m,
    std::size_t              t,
    double                   tol,
    std::uint64_t            seed) {
  return fcm_detail::run(data, c, m, t, tol, seed, false, 0).result;
}
FuzzyCMeansRun fuzzy_c_means_work(
    const FuzzyCMeansMatrix& data,
    std::size_t              c,
    double                   m,
    std::size_t              t,
    double                   tol,
    std::uint64_t            seed) {
  auto out = fcm_detail::run(data, c, m, t, tol, seed, false, 0);
  return std::move(static_cast<FuzzyCMeansRun&>(out));
}
std::size_t
fuzzy_c_means_trace_entries(std::size_t n, std::size_t d, std::size_t c, std::size_t t) {
  using fcm_detail::add;
  using fcm_detail::mul;
  if (!c || c > n)
    throw std::invalid_argument("FCM needs 0<c<=n");
  if (n > FuzzyCMeansMatrix().max_size() || c > FuzzyCMeansMatrix().max_size() ||
      d > std::vector<double>().max_size() || c > std::vector<double>().max_size() ||
      t > std::vector<FuzzyCMeansFrame>().max_size())
    throw std::length_error("FCM trace vector capacity");
  fcm_detail::work_bounds(n, d, c, t, true);
  const auto nd = mul(n, d), nc = mul(n, c), cd = mul(c, d);
  const auto base  = add(add(add(add(mul(2, n), c), nd), nc), cd);
  const auto frame = add(add(add(add(1, mul(3, c)), mul(2, n)), mul(2, cd)), mul(2, nc));
  return add(base, mul(t, frame));
}
FuzzyCMeansTrace fuzzy_c_means_trace(
    const FuzzyCMeansMatrix& data,
    std::size_t              c,
    double                   m,
    std::size_t              t,
    double                   tol,
    std::uint64_t            seed,
    std::size_t              cap) {
  return fcm_detail::run(data, c, m, t, tol, seed, true, cap);
}
FuzzyCMeansTrace fuzzy_c_means_trace_from(
    const FuzzyCMeansMatrix& data,
    const FuzzyCMeansMatrix& u,
    double                   m,
    std::size_t              t,
    double                   tol,
    std::size_t              cap) {
  return fcm_detail::run(data, u.empty() ? 0 : u[0].size(), m, t, tol, 0, true, cap, &u);
}
FuzzyCMeansMatrix fuzzy_c_means_membership(
    const FuzzyCMeansMatrix& data,
    const FuzzyCMeansMatrix& centers,
    double                   m) {
  fcm_detail::query_shape(data, centers);
  fcm_detail::values(data);
  fcm_detail::values(centers);
  fcm_detail::parameters(m);
  fcm_detail::work_bounds(data.size(), centers[0].size(), centers.size(), 1, false);
  FuzzyCMeansMatrix u(data.size(), std::vector<double>(centers.size()));
  FuzzyCMeansWork   w;
  fcm_detail::update(data, centers, u, m, w);
  return u;
}
double fuzzy_c_means_objective(
    const FuzzyCMeansMatrix& data,
    const FuzzyCMeansMatrix& centers,
    const FuzzyCMeansMatrix& u,
    double                   m) {
  fcm_detail::query_shape(data, centers);
  fcm_detail::membership_shape(u, data.size(), centers.size());
  fcm_detail::values(data);
  fcm_detail::values(centers);
  fcm_detail::simplex(u);
  fcm_detail::parameters(m);
  fcm_detail::mul(data.size(), centers.size());
  std::size_t count = 0;
  return fcm_detail::objective(data, centers, u, m, count);
}
FuzzyCMeansPartition fuzzy_c_means_partition(const FuzzyCMeansMatrix& u) {
  const auto c = fcm_detail::width(u);
  if (u.empty() || !c)
    throw std::invalid_argument("FCM partition needs nonempty simplex rows");
  fcm_detail::simplex(u);
  FuzzyCMeansPartition out;
  for (const auto& row : u)
    for (double x : row) {
      out.coefficient += x * x;
      if (x > 0)
        out.entropy_nats -= x * std::log(x);
    }
  out.coefficient /= static_cast<double>(u.size());
  out.entropy_nats /= static_cast<double>(u.size());
  return out;
}
FuzzyCMeansMatrix fuzzy_c_means_example_data() {
  std::mt19937_64                  rng(21);
  std::normal_distribution<double> g0(0, 2.2), g1(10, 2.2);
  FuzzyCMeansMatrix                data;
  for (int i = 0; i < 80; ++i)
    data.push_back({g0(rng)});
  for (int i = 0; i < 80; ++i)
    data.push_back({g1(rng)});
  return data;
}
} // namespace datamunge::algorithms
