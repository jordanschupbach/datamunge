/// @file structured_svm.cpp
/// @brief One checked loss-augmented/shrink/subgradient producer with real objective observations.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/structured_svm.hpp>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Translation-unit-private checked producing kernels; no shared mutable state.
namespace {
/// @brief Check actual floating arithmetic.
/// @param x Actual value.
/// @return Same finite value.
double finite(double x) {
  if (!std::isfinite(x))
    throw std::overflow_error("multiclass SVM: arithmetic");
  return x;
}
/// @brief Validate the complete manually constructible model.
/// @param m Borrowed model.
void model(const StructuredSVMModel& m) {
  if (!m.num_classes || m.weights.size() != m.num_classes ||
      m.dim == std::numeric_limits<std::size_t>::max())
    throw std::invalid_argument("multiclass SVM: fitted model");
  for (const auto& row : m.weights) {
    if (row.size() != m.dim + 1)
      throw std::invalid_argument("multiclass SVM: weight shape");
    for (double x : row)
      if (!std::isfinite(x))
        throw std::invalid_argument("multiclass SVM: finite weights");
  }
}
/// @brief Validate complete finite features and labels.
/// @param X Borrowed rows.
/// @param y Borrowed labels.
/// @param d Complete expected width.
/// @param k Complete class count.
void data(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>&                 y,
    std::size_t                             d,
    std::size_t                             k) {
  if (X.size() != y.size())
    throw std::invalid_argument("multiclass SVM: row/label shape");
  for (std::size_t i = 0; i < X.size(); ++i) {
    if (X[i].size() != d || y[i] < 0 || static_cast<std::size_t>(y[i]) >= k)
      throw std::invalid_argument("multiclass SVM: feature/label shape");
    for (double x : X[i])
      if (!std::isfinite(x))
        throw std::invalid_argument("multiclass SVM: finite features");
  }
}
/// @brief Actual local selected-operation/copy accounting.
struct Work {
  StructuredSVMRun&           r;            ///< Borrowed actual result counters.
  const StructuredSVMOptions& o;            ///< Borrowed call policies.
  std::size_t                 operations{}; ///< Actual total selected scalar operations.
  std::size_t                 retained{};   ///< Actual retained epoch/step/weight units.

  /// @brief Count one actual selected scalar operation before overflow.
  /// @param counter Actual category count.
  void count(std::size_t& counter) {
    const auto max = std::numeric_limits<std::size_t>::max();
    if (counter == max || operations == max ||
        (o.max_selected_operations && operations >= o.max_selected_operations))
      throw std::length_error("multiclass SVM: actual work");
    ++counter;
    ++operations;
  }
  /// @brief Bound actual retained records/scalars before copying.
  /// @param n New retained units.
  void keep(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() - retained)
      throw std::length_error("multiclass SVM: history capacity");
    retained += n;
    if (o.max_history_values && retained > o.max_history_values)
      throw std::length_error("multiclass SVM: history");
  }
};
/// @brief Score one actual validated row with optional work accounting.
/// @param m Validated model.
/// @param c Actual class.
/// @param x Validated features.
/// @param work Nullable actual selected-operation accumulator.
/// @return Actual finite score.
double score(const StructuredSVMModel& m, std::size_t c, const std::vector<double>& x, Work* work) {
  double value = m.weights[c][m.dim];
  if (work)
    work->count(work->r.score_terms);
  for (std::size_t j = 0; j < m.dim; ++j) {
    value = finite(value + m.weights[c][j] * x[j]);
    if (work)
      work->count(work->r.score_terms);
  }
  return value;
}
/// @brief Evaluate actual complete current objective without changing the model.
/// @param m Validated model.
/// @param X Validated rows.
/// @param y Validated labels.
/// @param lambda Positive regularization.
/// @param work Actual selected-operation accumulator.
/// @return Actual objective and classification count.
StructuredSVMObjective objective(
    const StructuredSVMModel&               m,
    const std::vector<std::vector<double>>& X,
    const std::vector<int>&                 y,
    double                                  lambda,
    Work&                                   work) {
  StructuredSVMObjective out;
  double                 norm = 0;
  for (const auto& row : m.weights)
    for (double weight : row) {
      norm = finite(norm + weight * weight);
      work.count(work.r.objective_norm_terms);
    }
  out.regularizer = finite(.5 * lambda * norm);
  double hinge    = 0;
  for (std::size_t i = 0; i < X.size(); ++i) {
    double      best = -std::numeric_limits<double>::infinity(), augmented = best, truth = 0;
    std::size_t predicted = 0;
    for (std::size_t c = 0; c < m.num_classes; ++c) {
      const double value = score(m, c, X[i], &work);
      if (value > best) {
        best      = value;
        predicted = c;
      }
      augmented =
          std::max(augmented, finite(value + (c == static_cast<std::size_t>(y[i]) ? 0. : 1.)));
      if (c == static_cast<std::size_t>(y[i]))
        truth = value;
    }
    hinge = finite(hinge + finite(augmented - truth));
    if (predicted == static_cast<std::size_t>(y[i]))
      ++out.correct;
  }
  out.mean_hinge = X.empty() ? 0 : hinge / X.size();
  out.value      = finite(out.regularizer + out.mean_hinge);
  return out;
}
/// @brief Execute actual locally sampled updates and full-data epoch observations.
/// @param X Complete borrowed rows.
/// @param y Complete borrowed labels.
/// @param k Complete classes.
/// @param p Training controls.
/// @param o Policies.
/// @param capture Retain actual sampled weight states.
/// @return Actual owned fit.
StructuredSVMRun run(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>&                 y,
    std::size_t                             k,
    const StructuredSVMParameters&          p,
    const StructuredSVMOptions&             o,
    bool                                    capture) {
  if (!std::isfinite(p.lambda) || p.lambda <= 0 || X.size() != y.size())
    throw std::invalid_argument("multiclass SVM: lambda/data shape");
  StructuredSVMRun r;
  if (X.empty())
    return r;
  if (!k || k > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw std::invalid_argument("multiclass SVM: classes");
  const auto n = X.size(), d = X.front().size(), limit = std::numeric_limits<std::size_t>::max();
  data(X, y, d, k);
  if (d == limit || k > limit / (d + 1) || (p.epochs && n > (limit - 1) / p.epochs))
    throw std::length_error("multiclass SVM: complete geometry");
  const auto values = k * (d + 1), updates = n * p.epochs;
  if ((o.max_model_values && values > o.max_model_values) ||
      (o.max_updates && updates > o.max_updates))
    throw std::length_error("multiclass SVM: model/update policies");
  r.model.num_classes = k;
  r.model.dim         = d;
  r.model.weights.assign(k, std::vector<double>(d + 1));
  Work work{r, o};
  work.keep(1);
  r.epochs.push_back({0, 0, objective(r.model, X, y, p.lambda, work)});
  std::mt19937_64                            rng(p.seed);
  std::uniform_int_distribution<std::size_t> pick(0, n - 1);
  std::size_t                                t = 1;
  for (std::size_t epoch = 0; epoch < p.epochs; ++epoch) {
    for (std::size_t step = 0; step < n; ++step, ++t) {
      const auto   i = pick(rng), truth = static_cast<std::size_t>(y[i]);
      const auto&  x           = X[i];
      const double denominator = finite(p.lambda * static_cast<double>(t)),
                   rate        = finite(1 / denominator);
      if (rate == 0)
        throw std::overflow_error("multiclass SVM: lost learning rate");
      std::size_t violator = 0;
      double      best = -std::numeric_limits<double>::infinity(), true_score = 0;
      for (std::size_t c = 0; c < k; ++c) {
        const double raw = score(r.model, c, x, &work), aug = finite(raw + (c == truth ? 0. : 1.));
        if (c == truth)
          true_score = raw;
        if (aug > best) {
          best     = aug;
          violator = c;
        }
      }
      const double shrink = finite(1 - rate * p.lambda), hinge = finite(best - true_score);
      for (auto& row : r.model.weights)
        for (double& weight : row) {
          weight = finite(weight * shrink);
          work.count(r.shrink_assignments);
        }
      if (violator != truth) {
        for (std::size_t j = 0; j < d; ++j) {
          r.model.weights[truth][j]    = finite(r.model.weights[truth][j] + rate * x[j]);
          r.model.weights[violator][j] = finite(r.model.weights[violator][j] - rate * x[j]);
          work.count(r.hinge_assignments);
          work.count(r.hinge_assignments);
        }
        r.model.weights[truth][d]    = finite(r.model.weights[truth][d] + rate);
        r.model.weights[violator][d] = finite(r.model.weights[violator][d] - rate);
        work.count(r.hinge_assignments);
        work.count(r.hinge_assignments);
      }
      ++r.updates;
      if (capture) {
        work.keep(1);
        work.keep(values);
        r.steps.push_back({t, i, truth, violator, rate, shrink, hinge, r.model.weights});
      }
    }
    work.keep(1);
    r.epochs.push_back({epoch + 1, r.updates, objective(r.model, X, y, p.lambda, work)});
  }
  return r;
}
} // namespace
double StructuredSVMModel::score(std::size_t c, const std::vector<double>& x) const {
  model(*this);
  if (c >= num_classes || x.size() != dim)
    throw std::invalid_argument("multiclass SVM: class/features");
  for (double v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument("multiclass SVM: finite features");
  return datamunge::algorithms::score(*this, c, x, nullptr);
}
StructuredSVMRun structured_svm_run(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>&                 y,
    std::size_t                             k,
    const StructuredSVMParameters&          p,
    const StructuredSVMOptions&             o) {
  return run(X, y, k, p, o, false);
}
StructuredSVMRun structured_svm_trace(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>&                 y,
    std::size_t                             k,
    const StructuredSVMParameters&          p,
    const StructuredSVMOptions&             o) {
  return run(X, y, k, p, o, true);
}
StructuredSVMModel structured_svm_train(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>&                 y,
    std::size_t                             k,
    const StructuredSVMParameters&          p) {
  return structured_svm_run(X, y, k, p).model;
}
std::size_t structured_svm_predict(const StructuredSVMModel& m, const std::vector<double>& x) {
  model(m);
  if (x.size() != m.dim)
    throw std::invalid_argument("multiclass SVM: feature width");
  for (double v : x)
    if (!std::isfinite(v))
      throw std::invalid_argument("multiclass SVM: finite features");
  std::size_t best  = 0;
  double      value = -std::numeric_limits<double>::infinity();
  for (std::size_t c = 0; c < m.num_classes; ++c) {
    const double s = score(m, c, x, nullptr);
    if (s > value) {
      value = s;
      best  = c;
    }
  }
  return best;
}
StructuredSVMObjective structured_svm_objective(
    const StructuredSVMModel&               m,
    const std::vector<std::vector<double>>& X,
    const std::vector<int>&                 y,
    double                                  lambda) {
  model(m);
  if (!std::isfinite(lambda) || lambda <= 0)
    throw std::invalid_argument("multiclass SVM: lambda");
  data(X, y, m.dim, m.num_classes);
  StructuredSVMRun     r;
  StructuredSVMOptions o;
  Work                 work{r, o};
  return objective(m, X, y, lambda, work);
}
} // namespace datamunge::algorithms
