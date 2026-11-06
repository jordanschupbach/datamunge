/// @file adaptive_pspline.cpp
/// @brief Exact hierarchical-space transfers, common-grid penalties and greedy local-MSE search.
#include <algorithm>
#include <cmath>
#include <datamunge/fda/adaptive_pspline.hpp>
#include <datamunge/linalg/cholesky.hpp>
#include <datamunge/plot/plot.hpp>
#include <limits>
#include <stdexcept>

namespace datamunge::fda {
/// @brief Private checked data, exact coordinate transfers and dense fitting helpers.
namespace {
using Matrix = linalg::DenseMatrix<double>; ///< Dense native numerical matrix.

/// @brief Test improvement with the documented relative/absolute tie tolerance.
bool improves(double a, double b) {
  return a < b - 1e-12 * std::max({1., std::abs(a), std::abs(b)});
}
/// @brief Reject malformed domain, size, penalty and bounded-search controls.
void controls(const AdaptiveSplineOptions& o) {
  if (o.degree < 1 || o.degree > 5 || o.root_basis <= o.degree || o.root_basis > 1024 ||
      o.max_depth < 0 || o.max_depth > 9 || o.max_splits < 0 || o.max_splits > 63 ||
      o.min_child_observations < 1 || !std::isfinite(o.lambda) || o.lambda < 0 ||
      !std::isfinite(o.lower) || !std::isfinite(o.upper) || !(o.lower < o.upper) ||
      !std::isfinite(o.upper - o.lower) ||
      (o.root_basis - o.degree) * (1 << o.max_depth) + o.degree > 1024)
    throw std::invalid_argument("adaptive spline: invalid controls or finest basis exceeds 1024");
}
/// @brief Validate finite paired observations; training coordinates must lie in the domain.
void data(
    const std::vector<double>&   x,
    const std::vector<double>&   y,
    const AdaptiveSplineOptions& o,
    bool                         training) {
  if (x.empty() || x.size() != y.size())
    throw std::invalid_argument("adaptive spline: paired nonempty data required");
  for (std::size_t i = 0; i < x.size(); ++i)
    if (!std::isfinite(x[i]) || !std::isfinite(y[i]) ||
        (training && (x[i] < o.lower || x[i] > o.upper)))
      throw std::invalid_argument("adaptive spline: nonfinite or out-of-domain training data");
}
/// @brief Mean squared error of aligned nonempty vectors.
double mse(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double r = a[i] - b[i];
    s += r * r;
  }
  return s / a.size();
}
/// @brief Native exact representation of a tree and its HB/THB-to-finest coordinate map.
struct Space {
  BSplineBasis fine;     ///< Common finest-grid basis.
  Matrix       transfer; ///< Finest coefficients by active hierarchical coefficients.
  std::vector<AdaptiveSplineCell> cells; ///< Full binary tree in creation order.
};
/// @brief Construct active functions and transfer them to the common uniform finest grid.
Space space(const AdaptiveSplineOptions& o, const std::vector<int>& splits) {
  auto root = BSplineBasis::open_uniform(o.degree, o.root_basis, o.lower, o.upper);
  std::vector<BSplineBasis>      levels{root};
  std::vector<BSplineRefinement> maps;
  for (int l = 0; l < o.max_depth; ++l) {
    maps.push_back(dyadic_refine(levels.back()));
    levels.push_back(maps.back().basis);
  }
  TruncatedHierarchicalBSpline    tree(TensorBSpline(root, std::vector<double>(o.root_basis, 0.)));
  std::vector<AdaptiveSplineCell> cells{{0, -1, 0, o.lower, o.upper, false}};
  for (int id : splits) {
    if (id < 0 || id >= int(cells.size()) || cells[id].split || cells[id].depth >= o.max_depth)
      throw std::invalid_argument("adaptive spline: invalid leaf split sequence");
    const auto   parent = cells[id];
    const auto   ids    = tree.subdivide(id);
    const double mid    = (parent.lower + parent.upper) / 2;
    cells[id].split     = true;
    cells.push_back({ids[0], id, parent.depth + 1, parent.lower, mid, false});
    cells.push_back({ids[1], id, parent.depth + 1, mid, parent.upper, false});
  }
  /// @brief One active hierarchical basis function.
  struct Key {
    int level; ///< Dyadic basis level.
    int index; ///< One-dimensional full-grid basis index.
  };
  std::vector<Key> keys;
  for (int l = 0; l < tree.level_count(); ++l)
    for (const auto& idx : tree.active_indices(l))
      keys.push_back({l, idx[0]});
  Matrix R(levels.back().basis_size(), keys.size(), 0.);
  for (std::size_t j = 0; j < keys.size(); ++j) {
    const auto          key = keys[j];
    std::vector<double> c;
    int                 first;
    if (o.truncated) {
      auto unit = tree;
      unit.set_coefficient(key.level, {key.index}, 1.);
      first = unit.level_count() - 1;
      c     = unit.level(first).coefficients();
    } else {
      first = key.level;
      c.assign(levels[first].basis_size(), 0.);
      c[key.index] = 1.;
    }
    for (int l = first; l < o.max_depth; ++l)
      c = maps[l].apply(c);
    for (std::size_t i = 0; i < c.size(); ++i)
      R(i, j) = c[i];
  }
  return {levels.back(), std::move(R), std::move(cells)};
}
/// @brief Mesh-dependent sufficient statistics reused while tuning smoothing strength.
struct PreparedSpline {
  Space                            space;   ///< Exact hierarchical transfer and common basis.
  linalg::DenseMatrix<long double> normal;  ///< Restricted training Gram, lower triangle.
  linalg::DenseMatrix<long double> penalty; ///< Restricted difference Gram, lower triangle.
  linalg::DenseMatrix<long double> delta;   ///< Difference operator applied to transfer columns.
  std::vector<long double>         rhs;     ///< Restricted response cross-product.
  double                           weight;  ///< Finest-spacing inverse cube.
};
/// @brief Assemble mesh-dependent matrices once, optionally preparing a nonzero penalty.
PreparedSpline prepare(
    const std::vector<double>&   x,
    const std::vector<double>&   y,
    const AdaptiveSplineOptions& o,
    const std::vector<int>&      splits,
    bool                         penalized) {
  auto              s = space(o, splits);
  const auto&       R = s.transfer;
  const std::size_t m = R.cols(), nf = R.rows();
  // Assemble the sparse finest-grid data Gram once, then restrict the space.
  Matrix              gram(nf, nf, 0.);
  std::vector<double> fine_rhs(nf, 0.);
  for (std::size_t row = 0; row < x.size(); ++row) {
    const auto local = s.fine.evaluate_point(x[row]);
    for (std::size_t a = 0; a < local.indices.size(); ++a) {
      const auto i = local.indices[a];
      fine_rhs[i] += local.values[a] * y[row] / x.size();
      for (std::size_t b = 0; b < local.indices.size(); ++b)
        gram(i, local.indices[b]) += local.values[a] * local.values[b] / x.size();
    }
  }
  Matrix gr(nf, m, 0.);
  for (std::size_t i = 0; i < nf; ++i)
    for (std::size_t k = 0; k < nf; ++k)
      if (gram(i, k) != 0.)
        for (std::size_t j = 0; j < m; ++j)
          gr(i, j) += gram(i, k) * R(k, j);
  linalg::DenseMatrix<long double> normal(m, m, 0.);
  std::vector<long double>         rhs(m, 0.);
  for (std::size_t j = 0; j < m; ++j) {
    std::vector<std::size_t> support;
    for (std::size_t i = 0; i < nf; ++i)
      if (R(i, j) != 0.) {
        support.push_back(i);
        rhs[j] += static_cast<long double>(R(i, j)) * fine_rhs[i];
      }
    for (std::size_t k = 0; k <= j; ++k)
      for (std::size_t i : support)
        normal(j, k) += static_cast<long double>(R(i, j)) * gr(i, k);
  }
  const double        h = (o.upper - o.lower) / (nf - o.degree), weight = 1 / (h * h * h);
  std::vector<double> greville(nf, 0.);
  for (std::size_t i = 0; i < nf; ++i)
    for (int q = 1; q <= o.degree; ++q)
      greville[i] += s.fine.knots()[i + q] / o.degree;
  linalg::DenseMatrix<long double> delta(nf - 2, m, 0.);
  for (std::size_t i = 0; i + 2 < nf; ++i) {
    const long double left  = greville[i + 1] - greville[i],
                      right = greville[i + 2] - greville[i + 1];
    const long double scale = 2 * static_cast<long double>(h) * h / (left + right);
    for (std::size_t j = 0; j < m; ++j)
      delta(i, j) = scale * ((R(i + 2, j) - R(i + 1, j)) / right - (R(i + 1, j) - R(i, j)) / left);
  }
  linalg::DenseMatrix<long double> penalty(m, m, 0.);
  if (penalized)
    for (std::size_t j = 0; j < m; ++j) {
      std::vector<std::size_t> support;
      for (std::size_t i = 0; i < delta.rows(); ++i)
        if (delta(i, j) != 0.)
          support.push_back(i);
      for (std::size_t k = 0; k <= j; ++k)
        for (std::size_t i : support)
          penalty(j, k) += delta(i, j) * delta(i, k);
    }
  return {
      std::move(s),
      std::move(normal),
      std::move(penalty),
      std::move(delta),
      std::move(rhs),
      weight};
}
/// @brief Solve one prepared space at a multiplier without reconstructing its transfer/Gram.
AdaptiveSplineFit solve_prepared(
    const PreparedSpline&        prepared,
    const std::vector<double>&   x,
    const std::vector<double>&   y,
    const AdaptiveSplineOptions& o,
    const std::vector<int>&      splits) {
  const auto&       s      = prepared.space;
  const auto&       R      = s.transfer;
  const auto&       delta  = prepared.delta;
  const auto&       rhs    = prepared.rhs;
  const auto        weight = prepared.weight;
  const std::size_t m = R.cols(), nf = R.rows();
  auto              normal = prepared.normal;
  if (o.lambda > 0)
    for (std::size_t j = 0; j < m; ++j)
      for (std::size_t k = 0; k <= j; ++k)
        normal(j, k) += static_cast<long double>(o.lambda) * weight * prepared.penalty(j, k);
  for (std::size_t j = 0; j < m; ++j)
    for (std::size_t k = 0; k < j; ++k)
      normal(k, j) = normal(j, k);
  auto chol = linalg::cholesky(normal);
  if (!chol.ok)
    throw std::runtime_error("adaptive spline: normal equations not positive definite");
  const auto          beta = chol.solve(rhs);
  std::vector<double> c(nf, 0.);
  for (std::size_t i = 0; i < nf; ++i) {
    long double v = 0;
    for (std::size_t j = 0; j < m; ++j)
      v += R(i, j) * beta[j];
    c[i] = double(v);
  }
  double rough = 0;
  for (std::size_t i = 0; i < delta.rows(); ++i) {
    long double d = 0;
    for (std::size_t j = 0; j < m; ++j)
      d += delta(i, j) * beta[j];
    rough += weight * double(d * d);
  }
  AdaptiveSplineFit result;
  result.basis          = s.fine;
  result.coefficients   = std::move(c);
  result.options        = o;
  result.splits         = splits;
  result.cells          = s.cells;
  result.parameters     = int(m);
  result.roughness      = rough;
  result.training_mse   = mse(y, result.predict(x));
  result.validation_mse = std::numeric_limits<double>::quiet_NaN();
  if (!std::isfinite(result.training_mse) || !std::isfinite(rough))
    throw std::runtime_error("adaptive spline: nonfinite fit");
  return result;
}
/// @brief Fit a single multiplier while preserving the fixed-space API's arithmetic.
AdaptiveSplineFit fit(
    const std::vector<double>&   x,
    const std::vector<double>&   y,
    const AdaptiveSplineOptions& o,
    const std::vector<int>&      splits) {
  const auto prepared = prepare(x, y, o, splits, o.lambda > 0);
  return solve_prepared(prepared, x, y, o, splits);
}
/// @brief Training score on one half-open parent interval (including the global maximum).
double local_mse(
    const AdaptiveSplineFit&   fit,
    const AdaptiveSplineCell&  cell,
    const std::vector<double>& x,
    const std::vector<double>& y,
    int&                       count) {
  const auto pred = fit.predict(x);
  double     sum  = 0;
  count           = 0;
  for (std::size_t i = 0; i < x.size(); ++i)
    if (x[i] >= cell.lower && (x[i] < cell.upper || cell.upper == fit.options.upper)) {
      const double r = y[i] - pred[i];
      sum += r * r;
      ++count;
    }
  return count ? sum / count : std::numeric_limits<double>::infinity();
}
/// @brief Exact derivative coefficients from repeated B-spline differentiation.
std::pair<BSplineBasis, std::vector<double>>
differentiated(BSplineBasis basis, std::vector<double> c, int order) {
  if (order < 0 || order > int(basis.degree()) || c.size() != basis.basis_size())
    throw std::invalid_argument("adaptive spline: invalid derivative order or coefficients");
  for (int q = 0; q < order; ++q) {
    const auto&         t = basis.knots();
    const int           p = int(basis.degree());
    std::vector<double> d(c.size() - 1);
    for (std::size_t i = 0; i < d.size(); ++i) {
      const double width = t[i + p + 1] - t[i + 1];
      d[i]               = width > 0 ? p * (c[i + 1] - c[i]) / width : 0.;
    }
    basis = BSplineBasis(p - 1, std::vector<double>(t.begin() + 1, t.end() - 1));
    c     = std::move(d);
  }
  return {std::move(basis), std::move(c)};
}
/// @brief Evaluate checked coefficients in a basis at a finite clamped coordinate.
double evaluate(const BSplineBasis& basis, const std::vector<double>& c, double x) {
  if (!std::isfinite(x))
    throw std::invalid_argument("adaptive spline: nonfinite coordinate");
  const auto [lo, hi] = basis.domain();
  const auto local    = basis.evaluate_point(std::clamp(x, lo, hi));
  double     value    = 0;
  for (std::size_t j = 0; j < local.indices.size(); ++j)
    value += local.values[j] * c[local.indices[j]];
  return value;
}
/// @brief Spanwise exact integral of squared second derivative divided by cell width.
double curvature(const AdaptiveSplineFit& fit, const AdaptiveSplineCell& cell) {
  if (fit.basis.degree() < 2)
    return 0.;
  const auto [basis, c] = differentiated(fit.basis, fit.coefficients, 2);
  constexpr double nodes[]{
      -.8611363115940526, -.3399810435848563, .3399810435848563, .8611363115940526};
  constexpr double weights[]{
      .3478548451374539, .6521451548625461, .6521451548625461, .3478548451374539};
  const auto& knots    = fit.basis.knots();
  double      integral = 0;
  for (std::size_t i = 0; i + 1 < knots.size(); ++i) {
    const double lo = std::max(cell.lower, knots[i]), hi = std::min(cell.upper, knots[i + 1]);
    if (hi <= lo)
      continue;
    for (int j = 0; j < 4; ++j) {
      const double v = evaluate(basis, c, (lo + hi) / 2 + (hi - lo) * nodes[j] / 2);
      integral += (hi - lo) * weights[j] * v * v / 2;
    }
  }
  return integral / (cell.upper - cell.lower);
}
/// @brief Tune one accepted tree, recording every solve and interpolating neighboring multipliers.
AdaptiveSplineFit tune_tree(
    const std::vector<double>&         x,
    const std::vector<double>&         y,
    const std::vector<double>&         vx,
    const std::vector<double>&         vy,
    const JointAdaptiveSplineOptions&  search,
    const std::vector<int>&            splits,
    double                             previous,
    std::vector<AdaptiveSplineTuning>& audit,
    int&                               attempts,
    int&                               failures) {
  const auto prepared = prepare(
      x,
      y,
      search.spline,
      splits,
      std::any_of(
          search.lambdas.begin(), search.lambdas.end(), [](double value) { return value > 0; }) ||
          previous > 0);
  std::vector<double> seeds = search.lambdas;
  if (std::isfinite(previous))
    seeds.push_back(previous);
  std::sort(seeds.begin(), seeds.end());
  seeds.erase(std::unique(seeds.begin(), seeds.end()), seeds.end());
  std::vector<double> evaluated;
  AdaptiveSplineFit   best;
  double              loss           = std::numeric_limits<double>::infinity();
  std::size_t         winning_record = 0;
  /// @brief Fit one new multiplier; deterministic ties prefer smaller lambda.
  auto trial = [&](double lambda) {
    if (std::find(evaluated.begin(), evaluated.end(), lambda) != evaluated.end())
      return;
    evaluated.push_back(lambda);
    AdaptiveSplineTuning entry{
        int(splits.size()), lambda, std::numeric_limits<double>::infinity(), false, false};
    ++attempts;
    try {
      auto o                   = search.spline;
      o.lambda                 = lambda;
      auto candidate           = solve_prepared(prepared, x, y, o, splits);
      candidate.validation_mse = mse(vy, candidate.predict(vx));
      if (!std::isfinite(candidate.validation_mse))
        throw std::runtime_error("adaptive spline: nonfinite tuning loss");
      entry.fitted         = true;
      entry.validation_mse = candidate.validation_mse;
      if (!std::isfinite(loss) || improves(entry.validation_mse, loss) ||
          (!improves(loss, entry.validation_mse) && lambda < best.options.lambda)) {
        best           = std::move(candidate);
        loss           = entry.validation_mse;
        winning_record = audit.size();
      }
    } catch (const std::runtime_error&) {
      ++failures;
    }
    audit.push_back(entry);
  };
  for (double lambda : seeds)
    trial(lambda);
  if (!std::isfinite(loss))
    throw std::runtime_error("adaptive spline: no identified root/tuning fit");
  for (int round = 0; round < search.lambda_refinements; ++round) {
    std::sort(evaluated.begin(), evaluated.end());
    const auto          it    = std::find(evaluated.begin(), evaluated.end(), best.options.lambda);
    const auto          index = std::size_t(it - evaluated.begin());
    std::vector<double> probes;
    /// @brief Overflow-safe geometric mean, with a linear midpoint next to zero.
    auto midpoint = [](double a, double b) {
      return a == 0 ? b / 2 : std::exp((std::log(a) + std::log(b)) / 2);
    };
    if (index > 0)
      probes.push_back(midpoint(evaluated[index - 1], evaluated[index]));
    if (index + 1 < evaluated.size())
      probes.push_back(midpoint(evaluated[index], evaluated[index + 1]));
    for (double lambda : probes)
      trial(lambda);
  }
  audit[winning_record].selected = true;
  return best;
}
/// @brief Snapshot one path score, without interpreting parameter count as edf.
AdaptiveSplineStep step(const AdaptiveSplineFit& fit, int cell) {
  return {
      int(fit.splits.size()),
      fit.parameters,
      fit.training_mse,
      fit.validation_mse,
      fit.roughness,
      fit.training_mse + fit.options.lambda * fit.roughness,
      cell,
      fit.options.lambda};
}
} // namespace

std::vector<double> AdaptiveSplineFit::predict(const std::vector<double>& x) const {
  if (coefficients.size() != basis.basis_size())
    throw std::invalid_argument("adaptive spline: coefficient size mismatch");
  std::vector<double> result;
  result.reserve(x.size());
  const auto domain = basis.domain();
  for (double t : x) {
    if (!std::isfinite(t))
      throw std::invalid_argument("adaptive spline: nonfinite prediction coordinate");
    const auto local = basis.evaluate_point(std::clamp(t, domain.first, domain.second));
    double     f     = 0;
    for (std::size_t j = 0; j < local.indices.size(); ++j)
      f += local.values[j] * coefficients[local.indices[j]];
    result.push_back(f);
  }
  return result;
}
std::vector<double> AdaptiveSplineFit::derivative(const std::vector<double>& x, int order) const {
  const auto [dbasis, c] = differentiated(basis, coefficients, order);
  std::vector<double> result;
  result.reserve(x.size());
  for (double t : x)
    result.push_back(evaluate(dbasis, c, t));
  return result;
}
void AdaptiveSplineFit::save_mesh_svg(const std::string& path, const std::string& title) const {
  auto p     = graphics::RPlot::create();
  int  depth = 0;
  for (const auto& c : cells) {
    depth = std::max(depth, c.depth);
    p.segments(
        {c.lower},
        {-double(c.depth)},
        {c.upper},
        {-double(c.depth)},
        c.split ? graphics::RGB("#94a3b8") : graphics::RGB("#2563eb"),
        c.split ? 2. : 5.);
    p.segments(
        {c.lower, c.upper},
        {-double(c.depth) - .12, -double(c.depth) - .12},
        {c.lower, c.upper},
        {-double(c.depth) + .12, -double(c.depth) + .12},
        graphics::RGB("#334155"),
        1.);
  }
  std::vector<double>      pos;
  std::vector<std::string> labels;
  for (int d = 0; d <= depth; ++d) {
    pos.push_back(-double(d));
    labels.push_back(std::to_string(d));
  }
  p.title(title)
      .subtitle("Grey: split parents; blue: selected leaf intervals")
      .x_label("x")
      .y_label("Refinement depth")
      .xlim(options.lower, options.upper)
      .ylim(-depth - .5, .5)
      .y_ticks(pos, labels)
      .show_grid(false)
      .size(1000, 330)
      .save_svg(path);
}
AdaptiveSplineFit fit_hierarchical_pspline(
    const std::vector<double>&   x,
    const std::vector<double>&   y,
    const AdaptiveSplineOptions& o,
    const std::vector<int>&      splits) {
  controls(o);
  data(x, y, o, true);
  auto result = fit(x, y, o, splits);
  result.path.push_back(step(result, -1));
  return result;
}
AdaptiveSplineFit fit_adaptive_pspline(
    const std::vector<double>&   x,
    const std::vector<double>&   y,
    const std::vector<double>&   vx,
    const std::vector<double>&   vy,
    const AdaptiveSplineOptions& o) {
  controls(o);
  data(x, y, o, true);
  data(vx, vy, o, false);
  auto current                              = fit(x, y, o, {});
  current.validation_mse                    = mse(vy, current.predict(vx));
  auto                                 best = current;
  std::vector<AdaptiveSplineStep>      path{step(current, -1)};
  std::vector<AdaptiveSplineCandidate> audit;
  int                                  failures = 0;
  for (int iteration = 1; iteration <= o.max_splits; ++iteration) {
    double            max_gain = 0;
    int               chosen   = -1;
    AdaptiveSplineFit winner;
    std::size_t       record = 0;
    for (const auto& cell : current.cells) {
      if (cell.split || cell.depth >= o.max_depth)
        continue;
      const double mid = (cell.lower + cell.upper) / 2;
      int          nl = 0, nr = 0;
      for (double t : x)
        if (t >= cell.lower && (t < cell.upper || cell.upper == o.upper)) {
          if (t < mid)
            ++nl;
          else
            ++nr;
        }
      if (nl < o.min_child_observations || nr < o.min_child_observations)
        continue;
      int                     count;
      const double            before = local_mse(current, cell, x, y, count);
      AdaptiveSplineCandidate entry{
          iteration,
          cell.id,
          count,
          before,
          std::numeric_limits<double>::quiet_NaN(),
          -std::numeric_limits<double>::infinity(),
          false,
          false};
      entry.lambda  = o.lambda;
      auto proposed = current.splits;
      proposed.push_back(cell.id);
      try {
        auto candidate = fit(x, y, o, proposed);
        if (candidate.parameters <= current.parameters) {
          entry.mse_after = local_mse(candidate, cell, x, y, count);
          entry.gain      = before - entry.mse_after;
          entry.fitted    = true;
          audit.push_back(entry);
          continue;
        }
        entry.mse_after = local_mse(candidate, cell, x, y, count);
        entry.gain      = before - entry.mse_after;
        entry.priority  = entry.gain;
        entry.fitted    = true;
        if (improves(-entry.gain, -max_gain)) {
          max_gain = entry.gain;
          chosen   = cell.id;
          winner   = std::move(candidate);
          record   = audit.size();
        }
      } catch (const std::runtime_error&) {
        ++failures;
      }
      audit.push_back(entry);
    }
    if (chosen < 0)
      break;
    audit[record].selected = true;
    current                = std::move(winner);
    current.validation_mse = mse(vy, current.predict(vx));
    path.push_back(step(current, chosen));
    if (improves(current.validation_mse, best.validation_mse))
      best = current;
  }
  best.path              = std::move(path);
  best.candidates        = std::move(audit);
  best.failed_candidates = failures;
  best.fit_evaluations   = 1 + int(best.candidates.size());
  return best;
}
AdaptiveSplineFit fit_joint_adaptive_pspline(
    const std::vector<double>&        x,
    const std::vector<double>&        y,
    const std::vector<double>&        vx,
    const std::vector<double>&        vy,
    const JointAdaptiveSplineOptions& search) {
  auto o   = search.spline;
  o.lambda = 0; // The scalar option is deliberately superseded by the search seeds.
  controls(o);
  data(x, y, o, true);
  data(vx, vy, o, false);
  if (search.lambdas.empty() || search.lambdas.size() > 64 || search.lambda_refinements < 0 ||
      search.lambda_refinements > 8 || (search.curvature_priority && o.degree < 2))
    throw std::invalid_argument("adaptive spline: invalid joint search controls");
  for (double lambda : search.lambdas)
    if (!std::isfinite(lambda) || lambda < 0)
      throw std::invalid_argument("adaptive spline: invalid lambda seed");
  int                               attempts = 0, failures = 0;
  std::vector<AdaptiveSplineTuning> tuning;
  auto                              current = tune_tree(
      x,
      y,
      vx,
      vy,
      search,
      {},
      std::numeric_limits<double>::quiet_NaN(),
      tuning,
      attempts,
      failures);
  auto                                 best = current;
  std::vector<AdaptiveSplineStep>      path{step(current, -1)};
  std::vector<AdaptiveSplineCandidate> audit;
  for (int iteration = 1; iteration <= o.max_splits; ++iteration) {
    int               chosen  = -1;
    double            highest = -std::numeric_limits<double>::infinity();
    std::size_t       record  = 0;
    AdaptiveSplineFit winner;
    for (const auto& cell : current.cells) {
      if (cell.split || cell.depth >= o.max_depth)
        continue;
      int          nl = 0, nr = 0;
      const double mid = (cell.lower + cell.upper) / 2;
      for (double t : x)
        if (t >= cell.lower && (t < cell.upper || cell.upper == o.upper))
          (t < mid ? nl : nr)++;
      if (nl < o.min_child_observations || nr < o.min_child_observations)
        continue;
      int                     count = 0;
      AdaptiveSplineCandidate entry;
      entry.step                   = iteration;
      entry.cell                   = cell.id;
      entry.observations           = nl + nr;
      entry.lambda                 = current.options.lambda;
      entry.mse_before             = local_mse(current, cell, x, y, count);
      entry.mse_after              = std::numeric_limits<double>::quiet_NaN();
      entry.gain                   = -std::numeric_limits<double>::infinity();
      entry.mean_curvature_squared = curvature(current, cell);
      auto splits                  = current.splits;
      splits.push_back(cell.id);
      ++attempts;
      try {
        auto candidate  = fit(x, y, current.options, splits);
        entry.mse_after = local_mse(candidate, cell, x, y, count);
        entry.gain      = entry.mse_before - entry.mse_after;
        entry.fitted    = true;
        entry.priority  = search.curvature_priority
                              ? std::pow(cell.upper - cell.lower, 4) * entry.mean_curvature_squared
                              : entry.gain;
        if (candidate.parameters > current.parameters && improves(-entry.gain, 0) &&
            (chosen < 0 || improves(-entry.priority, -highest))) {
          highest = entry.priority;
          chosen  = cell.id;
          record  = audit.size();
          winner  = std::move(candidate);
        }
      } catch (const std::runtime_error&) {
        ++failures;
      }
      audit.push_back(entry);
    }
    if (chosen < 0)
      break;
    audit[record].selected = true;
    current                = tune_tree(
        x, y, vx, vy, search, winner.splits, current.options.lambda, tuning, attempts, failures);
    path.push_back(step(current, chosen));
    if (improves(current.validation_mse, best.validation_mse))
      best = current;
  }
  best.path              = std::move(path);
  best.candidates        = std::move(audit);
  best.tuning            = std::move(tuning);
  best.fit_evaluations   = attempts;
  best.failed_candidates = failures;
  return best;
}
} // namespace datamunge::fda
