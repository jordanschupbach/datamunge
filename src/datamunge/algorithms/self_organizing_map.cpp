/// @file self_organizing_map.cpp
/// @brief Checked deterministic plan and single actual scalar competitive reference producer.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/self_organizing_map.hpp>
#include <limits>
#include <random>
#include <stdexcept>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked helpers.
namespace {
/// @brief Admit complete size geometry without arithmetic overflow.
/// @param a First dimension.
/// @param b Second dimension.
/// @param cap Zero or enabled limit.
/// @return Checked product.
std::size_t geometry(std::size_t a, std::size_t b, std::size_t cap) {
  if (b && a > std::numeric_limits<std::size_t>::max() / b)
    throw std::length_error("SOM: geometry capacity");
  const auto n = a * b;
  if (cap && n > cap)
    throw std::length_error("SOM: complete geometry policy");
  return n;
}
/// @brief Validate complete finite rectangular input geometry.
/// @param x Borrowed rows.
/// @param dim Required positive dimension.
/// @param o Complete input bounds.
/// @param empty Whether no observations is allowed.
void data_shape(
    const std::vector<std::vector<double>>& x,
    std::size_t                             dim,
    const SOMOptions&                       o,
    bool                                    empty) {
  if (!dim || (!empty && x.empty()))
    throw std::invalid_argument("SOM: positive nonempty data geometry");
  geometry(x.size(), 1, o.max_samples);
  geometry(x.size(), dim, o.max_elements);
  for (const auto& row : x) {
    if (row.size() != dim)
      throw std::invalid_argument("SOM: ragged input");
    for (double v : row)
      if (!std::isfinite(v))
        throw std::invalid_argument("SOM: finite data required");
  }
}
/// @brief Validate the complete owned public codebook before indexed use.
/// @param m Borrowed map.
/// @param o Complete geometry bounds.
void model_shape(const SelfOrganizingMap& m, const SOMOptions& o) {
  if (!m.rows || !m.cols || !m.dim)
    throw std::invalid_argument("SOM: positive model geometry");
  const auto units = geometry(m.rows, m.cols, o.max_units);
  geometry(units, m.dim, o.max_elements);
  if (m.weights.size() != units)
    throw std::invalid_argument("SOM: codebook units");
  for (const auto& row : m.weights) {
    if (row.size() != m.dim)
      throw std::invalid_argument("SOM: codebook dimension");
    for (double v : row)
      if (!std::isfinite(v))
        throw std::invalid_argument("SOM: finite codebook required");
  }
}
/// @brief Compute actual direct squared distance, using widened fallback only on range failure.
/// @param a Validated first vector.
/// @param b Validated equal-length second vector.
/// @return Actual finite double squared distance.
double distance(const std::vector<double>& a, const std::vector<double>& b) {
  double sum = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const auto d = a[i] - b[i];
    sum += d * d;
  }
  if (std::isfinite(sum))
    return sum;
  long double wide = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const auto d = static_cast<long double>(a[i]) - b[i];
    wide += d * d;
  }
  if (!std::isfinite(wide) || wide > std::numeric_limits<double>::max())
    throw std::overflow_error("SOM: squared-distance range");
  return static_cast<double>(wide);
}
/// @brief Execute one actual lowest-index winner scan on validated finite geometry.
/// @param m Borrowed current finite codebook.
/// @param x Borrowed current input.
/// @return Actual winning flat unit id.
std::size_t winner(const SelfOrganizingMap& m, const std::vector<double>& x) {
  std::size_t best    = 0;
  double      minimum = std::numeric_limits<double>::infinity();
  for (std::size_t k = 0; k < m.weights.size(); ++k) {
    const auto d = distance(m.weights[k], x);
    if (d < minimum) {
      minimum = d;
      best    = k;
    }
  }
  return best;
}
/// @brief Execute the checked scalar reference with optional same-execution full-state capture.
/// @param data Borrowed complete training data.
/// @param p Complete original controls.
/// @param o Geometry/work/history policies.
/// @param capture Retain actual updates.
/// @return Owned actual fitted scalar reference output.
SOMRun train(
    const std::vector<std::vector<double>>& data,
    const SOMParameters&                    p,
    const SOMOptions&                       o,
    bool                                    capture) {
  auto       plan  = som_training_plan(data, p, o);
  const auto units = plan.initial.weights.size(), dim = plan.initial.dim,
             entries = geometry(units, dim, 0);
  if (capture) {
    if (entries > (std::numeric_limits<std::size_t>::max() - 1 - units) / 2)
      throw std::length_error("SOM: history geometry");
    geometry(p.iterations, 1 + units + 2 * entries, o.max_trace_entries);
  }
  SOMRun out;
  out.model = std::move(plan.initial);
  for (std::size_t t = 0; t < p.iterations; ++t) {
    const auto& x    = data[plan.sample_indices[t]];
    const auto  best = winner(out.model, x);
    out.distance_terms += entries;
    const auto br = best / p.cols, bc = best % p.cols;
    const auto alpha = plan.learning_rates[t], sigma = plan.radii[t];
    SOMStep    step;
    if (capture) {
      step.iteration     = t;
      step.sample        = plan.sample_indices[t];
      step.winner        = best;
      step.learning_rate = alpha;
      step.radius        = sigma;
      step.before        = out.model.weights;
      step.influence.resize(units);
    }
    for (std::size_t r = 0; r < p.rows; ++r)
      for (std::size_t c = 0; c < p.cols; ++c) {
        const auto   k  = r * p.cols + c;
        const double dr = static_cast<double>(r) - br, dc = static_cast<double>(c) - bc;
        const double square = 2 * sigma * sigma;
        double       influence =
            square > 0 && std::isfinite(square)
                      ? std::exp(-(dr * dr + dc * dc) / square)
                      : std::exp(-.5 * ((dr / sigma) * (dr / sigma) + (dc / sigma) * (dc / sigma)));
        if (influence < 1e-6)
          influence = 0;
        if (capture)
          step.influence[k] = influence;
        if (!influence)
          continue;
        const auto rate = alpha * influence;
        auto&      w    = out.model.weights[k];
        for (std::size_t i = 0; i < dim; ++i) {
          auto value = w[i] + rate * (x[i] - w[i]);
          if (!std::isfinite(value)) {
            const auto wide = std::lerp(
                static_cast<long double>(w[i]),
                static_cast<long double>(x[i]),
                static_cast<long double>(rate));
            if (!std::isfinite(wide) || std::abs(wide) > std::numeric_limits<double>::max())
              throw std::overflow_error("SOM: update range");
            value = static_cast<double>(wide);
          }
          w[i] = value;
          ++out.coordinate_updates;
        }
      }
    ++out.updates;
    if (capture) {
      step.after = out.model.weights;
      out.steps.push_back(std::move(step));
    }
  }
  return out;
}
} // namespace
std::size_t SelfOrganizingMap::index(std::size_t r, std::size_t c) const {
  if (r >= rows || c >= cols)
    throw std::out_of_range("SOM: grid address");
  geometry(rows, cols, 0);
  return r * cols + c;
}
double som_sq_dist(const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size() != b.size())
    throw std::invalid_argument("SOM: distance dimensions");
  for (double v : a)
    if (!std::isfinite(v))
      throw std::invalid_argument("SOM: finite distance input");
  for (double v : b)
    if (!std::isfinite(v))
      throw std::invalid_argument("SOM: finite distance input");
  return distance(a, b);
}
std::pair<std::size_t, std::size_t> som_bmu(
    const SelfOrganizingMap&   m,
    const std::vector<double>& x) {
  model_shape(m, {});
  data_shape({x}, m.dim, {}, false);
  const auto k = winner(m, x);
  return {k / m.cols, k % m.cols};
}
SOMPlan som_training_plan(
    const std::vector<std::vector<double>>& data,
    const SOMParameters&                    p,
    const SOMOptions&                       o) {
  if (data.empty() || data.front().empty() || !p.rows || !p.cols ||
      !std::isfinite(p.learning_rate) || p.learning_rate <= 0 || p.learning_rate > 1 ||
      !std::isfinite(p.radius))
    throw std::invalid_argument("SOM: data/controls");
  data_shape(data, data.front().size(), o, false);
  const auto units   = geometry(p.rows, p.cols, o.max_units),
             entries = geometry(units, data.front().size(), o.max_elements);
  geometry(p.iterations, 1, o.max_iterations);
  geometry(p.iterations, entries, o.max_scalar_updates);
  if (units > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      data.front().size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()) ||
      data.size() > static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))
    throw std::length_error("SOM: tensor-compatible geometry");
  SOMPlan out;
  out.initial = {p.rows, p.cols, data.front().size(), {}};
  out.initial.weights.reserve(units);
  out.initialization_indices.reserve(units);
  std::mt19937_64                            rng(p.seed);
  std::uniform_int_distribution<std::size_t> pick(0, data.size() - 1);
  for (std::size_t k = 0; k < units; ++k) {
    const auto id = pick(rng);
    out.initialization_indices.push_back(id);
    out.initial.weights.push_back(data[id]);
  }
  const double sigma = p.radius > 0 ? p.radius : .5 * static_cast<double>(std::max(p.rows, p.cols));
  out.sample_indices.reserve(p.iterations);
  out.learning_rates.reserve(p.iterations);
  out.radii.reserve(p.iterations);
  for (std::size_t t = 0; t < p.iterations; ++t) {
    const auto fraction = static_cast<double>(t) / p.iterations,
               alpha    = p.learning_rate * std::exp(-fraction),
               radius   = sigma * std::exp(-fraction * std::log1p(sigma));
    if (!std::isfinite(alpha) || alpha <= 0 || !std::isfinite(radius) || radius <= 0)
      throw std::overflow_error("SOM: positive schedule range");
    out.sample_indices.push_back(pick(rng));
    out.learning_rates.push_back(alpha);
    out.radii.push_back(radius);
  }
  return out;
}
SOMRun som_train_run(
    const std::vector<std::vector<double>>& d,
    const SOMParameters&                    p,
    const SOMOptions&                       o) {
  return train(d, p, o, false);
}
SOMRun som_train_trace(
    const std::vector<std::vector<double>>& d,
    const SOMParameters&                    p,
    const SOMOptions&                       o) {
  return train(d, p, o, true);
}
SelfOrganizingMap som_train(const std::vector<std::vector<double>>& d, const SOMParameters& p) {
  return som_train_run(d, p).model;
}
double som_quantization_error(
    const SelfOrganizingMap&                m,
    const std::vector<std::vector<double>>& data) {
  const SOMOptions o;
  model_shape(m, o);
  data_shape(data, m.dim, o, true);
  geometry(data.size(), geometry(m.weights.size(), m.dim, 0), o.max_scalar_updates);
  long double total = 0;
  for (const auto& x : data)
    total += std::sqrt(distance(m.weights[winner(m, x)], x));
  const auto value = data.empty() ? 0 : total / data.size();
  if (!std::isfinite(value) || value > std::numeric_limits<double>::max())
    throw std::overflow_error("SOM: quantization summary range");
  return static_cast<double>(value);
}
SOMExample som_three_blob_example() {
  SOMExample out;
  out.centers = {{0, 0}, {10, 1}, {4, 9}};
  std::mt19937_64                  rng(7);
  std::normal_distribution<double> jitter(0, .4);
  for (std::size_t b = 0; b < out.centers.size(); ++b)
    for (int i = 0; i < 80; ++i) {
      out.inputs.push_back({out.centers[b][0] + jitter(rng), out.centers[b][1] + jitter(rng)});
      out.labels.push_back(static_cast<int>(b));
    }
  return out;
}
} // namespace datamunge::algorithms
