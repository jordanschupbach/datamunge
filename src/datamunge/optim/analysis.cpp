/// @file analysis.cpp
/// @brief Pure optimizer fixtures, callback histories and explicit post-run plotting diagnostics.
#include <algorithm>
#include <datamunge/linalg/eigen.hpp>
#include <datamunge/optim/analysis.hpp>
#include <datamunge/stats/formula.hpp>
#include <format>
#include <map>
#include <numbers>
#include <numeric>
#include <random>
namespace datamunge::optim {
/// @brief Internal dimension checks and stable scalar links.
namespace {
/// @brief Require an exact finite vector shape.
/// @param x Input vector, borrowed.
/// @param n Required positive length.
/// @throws std::invalid_argument Shape mismatch or nonfinite entry.
void check_point(const std::vector<double>& x, std::size_t n) {
  if (x.size() != n || !n)
    throw std::invalid_argument("optimization fixture: coordinate shape");
  for (double v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument(
          "optimization fixture: nonfinite coordinates");
}
/// @brief Numerically stable logistic inverse link.
/// @param z Finite linear predictor.
/// @return Probability in [0,1].
double logistic(double z) {
  return z >= 0 ? 1 / (1 + std::exp(-z)) : std::exp(z) / (1 + std::exp(z));
}
}  // namespace
namespace detail {
double analysis_norm(const std::vector<double>& x) {
  double n = 0;
  for (double v : x) n = std::hypot(n, v);
  return n;
}
double analysis_half_sse(const std::vector<double>& x) {
  long double n = 0;
  for (double v : x) n += .5L * v * v;
  return static_cast<double>(n);
}
}  // namespace detail
void OptimizationTrace::record(const std::string& kind,
                               const std::vector<double>& x, double value,
                               double norm, std::size_t terms,
                               double component) {
  if (!points_.empty() && x.size() != points_[0].size())
    throw std::invalid_argument("trace coordinate dimensions differ");
  kinds_.push_back(kind);
  points_.push_back(x);
  values_.push_back(value);
  norms_.push_back(norm);
  components_.push_back(component);
  term_cost_.push_back((term_cost_.empty() ? 0 : term_cost_.back()) + terms);
}
dstruct::DataFrame OptimizationTrace::sample(ArbitraryFunction& f,
                                             const std::string& kind) const {
  std::vector<double> events, values, norms, cost, components;
  std::vector<std::vector<double>> points;
  for (std::size_t i = 0; i < kinds_.size(); ++i)
    if (kinds_[i] == kind) {
      events.push_back(i + 1);
      values.push_back(f.evaluate(points_[i]));
      norms.push_back(norms_[i]);
      cost.push_back(term_cost_[i]);
      components.push_back(components_[i]);
      points.push_back(points_[i]);
    }
  dstruct::DataFrame out;
  out.add_column("event", events);
  out.add_column("value", values);
  out.add_column("gradient_norm", norms);
  out.add_column("term_gradients", cost);
  out.add_column("component", components);
  if (!points.empty())
    for (std::size_t j = 0; j < points[0].size(); ++j) {
      std::vector<double> v;
      for (const auto& point : points) v.push_back(point[j]);
      out.add_column("x" + std::to_string(j), v);
    }
  return out;
}
dstruct::DataFrame OptimizationTrace::values() const {
  std::vector<double> events, values, norms, cost;
  std::vector<std::vector<double>> points;
  for (std::size_t i = 0; i < kinds_.size(); ++i)
    if (kinds_[i] == "value" || kinds_[i] == "residuals") {
      events.push_back(i + 1);
      values.push_back(values_[i]);
      norms.push_back(norms_[i]);
      cost.push_back(term_cost_[i]);
      points.push_back(points_[i]);
    }
  dstruct::DataFrame out;
  out.add_column("event", events);
  out.add_column("value", values);
  out.add_column("gradient_norm", norms);
  out.add_column("term_gradients", cost);
  if (!points.empty())
    for (std::size_t j = 0; j < points[0].size(); ++j) {
      std::vector<double> v;
      for (const auto& point : points) v.push_back(point[j]);
      out.add_column("x" + std::to_string(j), v);
    }
  return out;
}
dstruct::DataFrame OptimizationTrace::requests() const {
  dstruct::DataFrame out;
  std::vector<double> events;
  for (std::size_t i = 0; i < calls(); ++i) events.push_back(i + 1);
  out.add_column("event", events);
  out.add_column("kind", kinds_);
  out.add_column("value", values_);
  out.add_column("gradient_norm", norms_);
  out.add_column("term_gradients", term_cost_);
  out.add_column("component", components_);
  if (!points_.empty())
    for (std::size_t j = 0; j < points_[0].size(); ++j) {
      std::vector<double> v;
      for (const auto& point : points_)
        v.push_back(j < point.size() ? point[j] : NAN);
      out.add_column("x" + std::to_string(j), v);
    }
  return out;
}
dstruct::DataFrame OptimizationTrace::counts() const {
  std::map<std::string, double> counts;
  for (const auto& kind : kinds_) ++counts[kind];
  std::vector<std::string> names;
  std::vector<double> values;
  for (const auto& [k, v] : counts) {
    names.push_back(k);
    values.push_back(v);
  }
  names.push_back("term_gradient_equivalents");
  values.push_back(term_cost_.empty() ? 0 : term_cost_.back());
  dstruct::DataFrame out;
  out.add_column("kind", names);
  out.add_column("calls", values);
  return out;
}
dstruct::DataFrame OptimizationRun::summary() const {
  dstruct::DataFrame out;
  out.add_column("value",
                 std::vector<std::string>{std::format("{:.6g}", value)});
  out.add_column("checked_value", std::vector<std::string>{
                                      std::format("{:.6g}", checked_value)});
  out.add_column("gradient_norm", std::vector<std::string>{
                                      std::format("{:.6g}", gradient_norm)});
  out.add_column("callbacks", std::vector<double>{double(trace.calls())});
  out.add_column("exception", std::vector<std::string>{exception});
  for (std::size_t j = 0; j < coordinates.size(); ++j)
    out.add_column(
        "x" + std::to_string(j),
        std::vector<std::string>{std::format("{:.6g}", coordinates[j])});
  return out;
}
QuadraticObjective::QuadraticObjective(std::vector<double> d,
                                       std::vector<double> centre,
                                       double coupling, double angle)
    : centre_(std::move(centre)) {
  check_point(d, d.size());
  check_point(centre_, d.size());
  if (!std::isfinite(coupling) || coupling < 0 || !std::isfinite(angle))
    throw std::invalid_argument("quadratic coupling/angle");
  for (double v : d)
    if (v <= 0)
      throw std::invalid_argument("quadratic curvature must be positive");
  const auto n = d.size();
  matrix_.assign(n, std::vector<double>(n, coupling));
  for (std::size_t i = 0; i < n; ++i) matrix_[i][i] += d[i];
  if (n >= 2 && angle != 0) {
    auto rotation = std::vector<std::vector<double>>(n, std::vector<double>(n));
    for (std::size_t i = 0; i < n; ++i) rotation[i][i] = 1;
    rotation[0][0] = rotation[1][1] = std::cos(angle);
    rotation[0][1] = -std::sin(angle);
    rotation[1][0] = std::sin(angle);
    auto old = matrix_;
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j < n; ++j) {
        long double v = 0;
        for (std::size_t a = 0; a < n; ++a)
          for (std::size_t b = 0; b < n; ++b)
            v += rotation[i][a] * old[a][b] * rotation[j][b];
        matrix_[i][j] = v;
      }
  }
}
double QuadraticObjective::evaluate(const std::vector<double>& x) {
  const auto g = gradient(x);
  long double v = 0;
  for (std::size_t i = 0; i < x.size(); ++i)
    v += .5L * (x[i] - centre_[i]) * g[i];
  return v;
}
std::vector<double> QuadraticObjective::gradient(const std::vector<double>& x) {
  check_point(x, centre_.size());
  std::vector<double> g(x.size());
  for (std::size_t i = 0; i < x.size(); ++i) {
    long double v = 0;
    for (std::size_t j = 0; j < x.size(); ++j)
      v += matrix_[i][j] * (static_cast<long double>(x[j]) - centre_[j]);
    g[i] = v;
  }
  return g;
}
std::vector<std::vector<double>> QuadraticObjective::hessian(
    const std::vector<double>& x) {
  check_point(x, centre_.size());
  return matrix_;
}
SmoothBenchmark::SmoothBenchmark(std::string name) : name_(std::move(name)) {
  if (name_ != "rosenbrock" && name_ != "rastrigin" && name_ != "bumpy")
    throw std::invalid_argument("unknown smooth benchmark");
}
double SmoothBenchmark::evaluate(const std::vector<double>& x) {
  check_point(x, 2);
  if (name_ == "bumpy")
    return std::pow(x[0] - 3, 2) + std::pow(x[1] + 1, 2) +
           5 * std::sin(x[0]) * std::cos(x[1]);
  if (name_ == "rosenbrock")
    return std::pow(1 - x[0], 2) + 100 * std::pow(x[1] - x[0] * x[0], 2);
  return 20 + x[0] * x[0] + x[1] * x[1] -
         10 * std::cos(2 * std::numbers::pi * x[0]) -
         10 * std::cos(2 * std::numbers::pi * x[1]);
}
std::vector<double> SmoothBenchmark::gradient(const std::vector<double>& x) {
  check_point(x, 2);
  if (name_ == "bumpy")
    return {2 * (x[0] - 3) + 5 * std::cos(x[0]) * std::cos(x[1]),
            2 * (x[1] + 1) - 5 * std::sin(x[0]) * std::sin(x[1])};
  if (name_ == "rosenbrock")
    return {-2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0] * x[0]),
            200 * (x[1] - x[0] * x[0])};
  return {
      2 * x[0] + 20 * std::numbers::pi * std::sin(2 * std::numbers::pi * x[0]),
      2 * x[1] + 20 * std::numbers::pi * std::sin(2 * std::numbers::pi * x[1])};
}
std::vector<std::vector<double>> SmoothBenchmark::hessian(
    const std::vector<double>& x) {
  check_point(x, 2);
  if (name_ == "bumpy") {
    double d = 2 - 5 * std::sin(x[0]) * std::cos(x[1]),
           off = -5 * std::cos(x[0]) * std::sin(x[1]);
    return {{d, off}, {off, d}};
  }
  if (name_ == "rosenbrock")
    return {{2 - 400 * x[1] + 1200 * x[0] * x[0], -400 * x[0]},
            {-400 * x[0], 200}};
  return {{2 + 40 * std::numbers::pi * std::numbers::pi *
                   std::cos(2 * std::numbers::pi * x[0]),
           0},
          {0, 2 + 40 * std::numbers::pi * std::numbers::pi *
                      std::cos(2 * std::numbers::pi * x[1])}};
}
EqualityQuadratic::EqualityQuadratic(std::vector<double> target, double total)
    : target_(std::move(target)), total_(total) {
  check_point(target_, target_.size());
  if (!std::isfinite(total))
    throw std::invalid_argument("nonfinite equality total");
}
double EqualityQuadratic::evaluate(const std::vector<double>& x) {
  return detail::analysis_half_sse(gradient(x));
}
std::vector<double> EqualityQuadratic::gradient(const std::vector<double>& x) {
  check_point(x, target_.size());
  auto g = x;
  for (std::size_t j = 0; j < x.size(); ++j) g[j] -= target_[j];
  return g;
}
std::vector<double> EqualityQuadratic::constraints(
    const std::vector<double>& x) {
  check_point(x, target_.size());
  return {std::accumulate(x.begin(), x.end(), 0.) - total_};
}
std::vector<std::vector<double>> EqualityQuadratic::constraint_jacobian(
    const std::vector<double>& x) {
  check_point(x, target_.size());
  return {std::vector<double>(x.size(), 1)};
}
NonnegativeQuadratic::NonnegativeQuadratic(std::vector<double> target)
    : target_(std::move(target)) {
  check_point(target_, target_.size());
}
double NonnegativeQuadratic::evaluate(const std::vector<double>& x) {
  return detail::analysis_half_sse(gradient(x));
}
std::vector<double> NonnegativeQuadratic::gradient(
    const std::vector<double>& x) {
  check_point(x, target_.size());
  auto g = x;
  for (std::size_t j = 0; j < x.size(); ++j) g[j] -= target_[j];
  return g;
}
std::vector<double> NonnegativeQuadratic::inequalities(
    const std::vector<double>& x) {
  check_point(x, target_.size());
  auto g = x;
  for (auto& v : g) v = -v;
  return g;
}
std::vector<std::vector<double>> NonnegativeQuadratic::inequality_jacobian(
    const std::vector<double>& x) {
  check_point(x, target_.size());
  std::vector<std::vector<double>> j(x.size(), std::vector<double>(x.size()));
  for (std::size_t i = 0; i < x.size(); ++i) j[i][i] = -1;
  return j;
}
LeastSquaresObjective::LeastSquaresObjective(
    std::vector<std::vector<double>> design, std::vector<double> response)
    : design_(std::move(design)), response_(std::move(response)) {
  check_point(response_, design_.size());
  if (design_.empty()) throw std::invalid_argument("empty least squares data");
  for (const auto& row : design_) check_point(row, design_[0].size());
}
LeastSquaresObjective LeastSquaresObjective::from_data(
    const dstruct::DataFrame& data, const std::string& formula) {
  const auto info =
      stats::Formula(formula).resolve(data, stats::ResponseKind::Numeric);
  const auto matrix = info.build_matrix(data, true);
  std::vector<std::vector<double>> x;
  for (std::size_t i = 0; i < matrix.X.rows(); ++i)
    x.push_back(matrix.X.row(i));
  return LeastSquaresObjective(x, matrix.y);
}
double LeastSquaresObjective::evaluate(const std::vector<double>& x) {
  long double v = 0;
  for (std::size_t i = 0; i < response_.size(); ++i) v += evaluate_term(x, i);
  return v;
}
std::vector<double> LeastSquaresObjective::gradient(
    const std::vector<double>& x) {
  check_point(x, design_[0].size());
  std::vector<double> g(x.size());
  for (std::size_t i = 0; i < response_.size(); ++i) {
    auto term = gradient_term(x, i);
    for (std::size_t j = 0; j < x.size(); ++j) g[j] += term[j];
  }
  return g;
}
double LeastSquaresObjective::evaluate_term(const std::vector<double>& x,
                                            std::size_t i) {
  check_point(x, design_[0].size());
  if (i >= response_.size())
    throw std::invalid_argument("least squares component index");
  long double r = -response_[i];
  for (std::size_t j = 0; j < x.size(); ++j) r += design_[i][j] * x[j];
  return .5 * r * r;
}
std::vector<double> LeastSquaresObjective::gradient_term(
    const std::vector<double>& x, std::size_t i) {
  check_point(x, design_[0].size());
  if (i >= response_.size())
    throw std::invalid_argument("least squares component index");
  long double r = -response_[i];
  for (std::size_t j = 0; j < x.size(); ++j) r += design_[i][j] * x[j];
  auto g = design_[i];
  for (auto& v : g) v *= r;
  return g;
}
double LeastSquaresObjective::mean_loss_lipschitz() const {
  const auto p = design_[0].size();
  linalg::DenseMatrix<double> gram(p, p, 0.);
  for (std::size_t j = 0; j < p; ++j)
    for (std::size_t k = j; k < p; ++k) {
      long double v = 0;
      for (const auto& row : design_)
        v += static_cast<long double>(row[j]) * row[k];
      const double entry = v / design_.size();
      if (!std::isfinite(entry))
        throw std::overflow_error("nonfinite mean-loss Gram matrix");
      gram(j, k) = gram(k, j) = entry;
    }
  const double result = linalg::symmetric_eigen(gram).eigenvalues[0];
  if (!std::isfinite(result))
    throw std::overflow_error("nonfinite mean-loss Lipschitz constant");
  return std::max(0., result);
}
L1LeastSquaresObjective::L1LeastSquaresObjective(
    std::vector<std::vector<double>> x, std::vector<double> y, double lambda,
    bool first)
    : least_squares_(std::move(x), std::move(y)),
      lambda_(lambda),
      penalize_first_(first) {
  if (!std::isfinite(lambda) || lambda < 0)
    throw std::invalid_argument("invalid L1 penalty");
}
double L1LeastSquaresObjective::evaluate(const std::vector<double>& x) {
  long double v = least_squares_.evaluate(x) / least_squares_.num_functions();
  for (std::size_t j = penalize_first_ ? 0 : 1; j < x.size(); ++j)
    v += lambda_ * std::abs(x[j]);
  return v;
}
std::vector<double> L1LeastSquaresObjective::gradient(
    const std::vector<double>& x) {
  auto g = least_squares_.gradient(x);
  for (auto& v : g) v /= least_squares_.num_functions();
  return g;
}
std::vector<double> L1LeastSquaresObjective::proximal(
    const std::vector<double>& x, double step) {
  check_point(x, least_squares_.design()[0].size());
  if (!std::isfinite(step) || step <= 0)
    throw std::invalid_argument("positive finite proximal step required");
  auto out = x;
  for (std::size_t j = penalize_first_ ? 0 : 1; j < x.size(); ++j)
    out[j] = std::copysign(std::max(0., std::abs(x[j]) - step * lambda_), x[j]);
  return out;
}
CurveResidual::CurveResidual(const dstruct::DataFrame& data, std::string model)
    : model_(std::move(model)) {
  if (model_ != "exponential" && model_ != "exponential_offset" &&
      model_ != "logistic")
    throw std::invalid_argument("unknown curve family");
  for (std::size_t i = 0; i < data.nrows(); ++i) {
    if (data.is_null("x", i) || data.is_null("y", i))
      throw std::invalid_argument("missing curve data");
    x_.push_back(data.double_at("x", i));
    y_.push_back(data.double_at("y", i));
  }
  check_point(x_, data.nrows());
  check_point(y_, data.nrows());
}
std::vector<double> CurveResidual::predict(const std::vector<double>& x) {
  check_point(x, model_ == "exponential" ? 2 : 3);
  std::vector<double> out;
  for (double t : x_)
    out.push_back(model_ == "logistic"
                      ? x[0] * logistic(x[1] * (t - x[2]))
                      : x[0] * std::exp(-x[1] * t) +
                            (model_ == "exponential_offset" ? x[2] : 0));
  return out;
}
std::vector<double> CurveResidual::residuals(const std::vector<double>& x) {
  auto r = predict(x);
  for (std::size_t i = 0; i < r.size(); ++i) r[i] -= y_[i];
  return r;
}
std::vector<std::vector<double>> CurveResidual::jacobian(
    const std::vector<double>& x) {
  check_point(x, model_ == "exponential" ? 2 : 3);
  std::vector<std::vector<double>> out;
  for (double t : x_)
    if (model_ != "logistic") {
      const auto e = std::exp(-x[1] * t);
      std::vector<double> row{e, -x[0] * t * e};
      if (model_ == "exponential_offset") row.push_back(1);
      out.push_back(row);
    } else {
      const auto p = logistic(x[1] * (t - x[2]));
      out.push_back(
          {p, x[0] * (t - x[2]) * p * (1 - p), -x[0] * x[1] * p * (1 - p)});
    }
  return out;
}
dstruct::DataFrame optimization_curve_data(const std::string& model,
                                           std::size_t n, std::uint64_t seed) {
  if (n < 3 || (model != "exponential" && model != "exponential_offset" &&
                model != "logistic"))
    throw std::invalid_argument("invalid curve recipe");
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> noise(
      0, model == "exponential" ? .05
                                : (model == "exponential_offset" ? .1 : .15));
  std::vector<double> x, y, truth;
  for (std::size_t i = 0; i < n; ++i) {
    double t = model == "exponential"
                   ? 5. * i / (n - 1)
                   : i * (model == "exponential_offset" ? .25 : .2);
    double m = model == "exponential" ? 2.5 * std::exp(-.7 * t)
                                      : (model == "exponential_offset"
                                             ? 5 * std::exp(-.7 * t) + 1
                                             : 10 * logistic(1.5 * (t - 5)));
    x.push_back(t);
    truth.push_back(m);
    y.push_back(m + noise(rng));
  }
  dstruct::DataFrame out;
  out.add_column("x", x);
  out.add_column("y", y);
  out.add_column("truth", truth);
  return out;
}
graphics::RPlot optimization_comparison(
    ArbitraryFunction& objective, const std::vector<OptimizationRun>& runs,
    const std::vector<std::string>& labels, const std::string& kind,
    double reference, bool best_observed, bool term_axis) {
  if (runs.size() != labels.size())
    throw std::invalid_argument("optimization labels mismatch");
  const std::vector<graphics::RGB> colors{{40, 100, 190}, {190, 100, 35},
                                          {70, 145, 90},  {140, 75, 160},
                                          {70, 155, 170}, {170, 80, 90}};
  auto plot = graphics::RPlot::create();
  double lo = INFINITY, hi = -INFINITY;
  bool any = false;
  for (std::size_t i = 0; i < runs.size(); ++i) {
    const auto data = runs[i].trace.sample(objective, kind);
    std::vector<double> x, y;
    double best = INFINITY;
    for (std::size_t j = 0; j < data.nrows(); ++j) {
      double v = data.double_at("value", j);
      if (!std::isfinite(v)) continue;
      best = std::min(best, v);
      v = best_observed ? best : v;
      double z = std::log10(std::max(1e-14, v - reference));
      x.push_back(data.double_at(term_axis ? "term_gradients" : "event", j));
      y.push_back(z);
      lo = std::min(lo, z);
      hi = std::max(hi, z);
    }
    if (!x.empty()) {
      any = true;
      plot.line(x, y, labels[i], colors[i % colors.size()]);
    }
  }
  if (!any) throw std::invalid_argument("no finite optimization samples");
  std::vector<graphics::RGB> palette;
  for (std::size_t i = 0; i < labels.size(); ++i)
    palette.push_back(colors[i % colors.size()]);
  plot.title(best_observed ? "Best evaluated trial"
                           : "Objective at requested points")
      .x_label(term_axis ? "nominal component gradients"
                         : "callback request index")
      .y_label("log10 objective gap (floor 1e-14)")
      .y_limits(lo - .1, hi + std::max(1., .5 * (hi - lo)))
      .legend(labels, palette);
  return plot;
}

std::vector<double> EqualityQuadratic::optimum() const {
  const double excess =
      (std::accumulate(target_.begin(), target_.end(), 0.) - total_) /
      target_.size();
  auto x = target_;
  for (auto& v : x) v -= excess;
  return x;
}
dstruct::DataFrame EqualityQuadratic::diagnostics(
    const std::vector<double>& x) {
  auto g = gradient(x), truth = optimum();
  const auto mu =
      (std::accumulate(target_.begin(), target_.end(), 0.) - total_) /
      target_.size();
  std::vector<double> distance = x;
  for (std::size_t j = 0; j < x.size(); ++j) {
    g[j] += mu;
    distance[j] -= truth[j];
  }
  dstruct::DataFrame out;
  out.add_column("feasibility",
                 std::vector<double>{std::abs(constraints(x)[0])});
  out.add_column("stationarity", std::vector<double>{detail::analysis_norm(g)});
  out.add_column("distance",
                 std::vector<double>{detail::analysis_norm(distance)});
  out.add_column("multiplier", std::vector<double>{mu});
  return out;
}
std::vector<double> NonnegativeQuadratic::optimum() const {
  auto x = target_;
  for (auto& v : x) v = std::max(0., v);
  return x;
}
std::vector<double> NonnegativeQuadratic::barrier_optimum(double mu) const {
  if (!std::isfinite(mu) || mu <= 0)
    throw std::invalid_argument("invalid barrier weight");
  auto x = target_;
  for (auto& t : x) {
    const double root = std::hypot(t, 2 * std::sqrt(mu));
    t = t >= 0 ? .5 * t + .5 * root : 2 * mu / (root - t);
  }
  return x;
}
dstruct::DataFrame NonnegativeQuadratic::diagnostics(
    const std::vector<double>& x) {
  auto g = gradient(x), truth = optimum();
  std::vector<double> stationarity = g, distance = x;
  double feasibility = 0, complementarity = 0;
  for (std::size_t j = 0; j < x.size(); ++j) {
    double lambda = std::max(0., g[j]);
    stationarity[j] -= lambda;
    distance[j] -= truth[j];
    feasibility = std::max(feasibility, -x[j]);
    complementarity = std::max(complementarity, std::abs(lambda * x[j]));
  }
  dstruct::DataFrame out;
  out.add_column("feasibility", std::vector<double>{feasibility});
  out.add_column("stationarity",
                 std::vector<double>{detail::analysis_norm(stationarity)});
  out.add_column("complementarity", std::vector<double>{complementarity});
  out.add_column("distance",
                 std::vector<double>{detail::analysis_norm(distance)});
  return out;
}
double proximal_mapping_norm(ProximalFunction& f, const std::vector<double>& x,
                             double step) {
  if (!std::isfinite(step) || step <= 0)
    throw std::invalid_argument("positive mapping step required");
  auto g = f.gradient(x);
  if (g.size() != x.size())
    throw std::invalid_argument("mapping gradient dimension");
  auto z = x;
  for (std::size_t j = 0; j < x.size(); ++j) z[j] -= step * g[j];
  auto p = f.proximal(z, step);
  if (p.size() != x.size())
    throw std::invalid_argument("mapping proximal dimension");
  for (std::size_t j = 0; j < x.size(); ++j) p[j] = (x[j] - p[j]) / step;
  return detail::analysis_norm(p);
}
dstruct::DataFrame check_residual_jacobian(ResidualFunction& f,
                                           const std::vector<double>& x,
                                           double step) {
  if (!std::isfinite(step) || step <= 0 || x.empty())
    throw std::invalid_argument("invalid derivative check step");
  auto j = f.jacobian(x);
  std::vector<double> parameter, error;
  for (std::size_t k = 0; k < x.size(); ++k) {
    const double h = step * std::max(1., std::abs(x[k]));
    auto a = x, b = x;
    a[k] += h;
    b[k] -= h;
    auto ra = f.residuals(a), rb = f.residuals(b);
    if (ra.size() != j.size() || rb.size() != j.size())
      throw std::invalid_argument("residual Jacobian rows");
    double e = 0;
    for (std::size_t i = 0; i < j.size(); ++i) {
      if (j[i].size() != x.size())
        throw std::invalid_argument("residual Jacobian columns");
      e = std::max(e, std::abs(j[i][k] - (ra[i] - rb[i]) / (2 * h)));
    }
    parameter.push_back(k);
    error.push_back(e);
  }
  dstruct::DataFrame out;
  out.add_column("parameter", parameter);
  out.add_column("max_absolute_error", error);
  return out;
}
double residual_gradient_norm(ResidualFunction& f,
                              const std::vector<double>& x) {
  const auto r = f.residuals(x);
  const auto j = f.jacobian(x);
  if (j.size() != r.size())
    throw std::invalid_argument("residual gradient rows");
  std::vector<double> g(x.size());
  for (std::size_t i = 0; i < r.size(); ++i) {
    if (j[i].size() != x.size())
      throw std::invalid_argument("residual gradient columns");
    for (std::size_t k = 0; k < x.size(); ++k) g[k] += j[i][k] * r[i];
  }
  return detail::analysis_norm(g);
}
}  // namespace datamunge::optim
