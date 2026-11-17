/// @file moead.cpp
/// @brief Shared decomposition optimizer with actual offspring/replacement capture.
#include <algorithm>
#include <cmath>
#include <datamunge/optim/moead.hpp>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace datamunge::optim {

/// @brief Private checked weight generation and bound validation.
namespace {

/// @brief Validate finite matching strict bounds before any callback.
/// @param lower Nonempty finite lower vector.
/// @param upper Matching finite higher vector with representable widths.
/// @throws std::invalid_argument If shape or geometry is invalid.
void validate_bounds(const std::vector<double>& lower, const std::vector<double>& upper) {
  const std::size_t n = lower.size();
  if (n == 0)
    throw std::invalid_argument("MOEAD::optimize: bounds must not be empty");
  if (upper.size() != n)
    throw std::invalid_argument("MOEAD::optimize: lower/upper bound sizes must match");
  for (std::size_t j = 0; j < n; ++j)
    if (!std::isfinite(lower[j]) || !std::isfinite(upper[j]) ||
        !std::isfinite(upper[j] - lower[j]) || !(lower[j] < upper[j]))
      throw std::invalid_argument(
          "MOEAD::optimize: lower_bound must be < upper_bound in every dimension");
}

/// @brief Count lattice rows with exact bounded integer binomial arithmetic.
/// @param H Positive lattice divisions.
/// @param m At least two objectives.
/// @return Exact count, saturating at size_t max.
std::size_t lattice_count(std::size_t H, std::size_t m) {
  const std::size_t k = std::min(m - 1, H);
  __uint128_t       c = 1;
  for (std::size_t i = 0; i < k; ++i) {
    c = c * (H + m - 1 - i) / (i + 1);
    if (c > std::numeric_limits<std::size_t>::max())
      return std::numeric_limits<std::size_t>::max();
  }
  return static_cast<std::size_t>(c);
}

/// @brief Enumerate the simplex lattice in the original recursive lexicographic order using an explicit stack.
/// @param left Complete divisions at entry.
/// @param depth Entry coordinate index, zero for the full lattice.
/// @param m Objective count.
/// @param H Positive denominator.
/// @param partial Shared coordinate scratch.
/// @param out Owned actual lattice rows to append.
void das_dennis(
    std::size_t                       left,
    std::size_t                       depth,
    std::size_t                       m,
    std::size_t                       H,
    std::vector<std::size_t>&         partial,
    std::vector<std::vector<double>>& out) {
  /// @brief One actual suspended lattice enumeration frame.
  struct Frame {
    std::size_t left;
    std::size_t depth;
    std::size_t next;
  };
  std::vector<Frame> stack{{left, depth, 0}};
  while (!stack.empty()) {
    auto& frame = stack.back();
    if (frame.depth == m - 1) {
      partial[frame.depth] = frame.left;
      std::vector<double> w(m);
      for (std::size_t i = 0; i < m; ++i)
        w[i] = double(partial[i]) / double(H);
      out.push_back(std::move(w));
      stack.pop_back();
      continue;
    }
    if (frame.next > frame.left) {
      stack.pop_back();
      continue;
    }
    const auto value = frame.next++, next_depth = frame.depth + 1, remaining = frame.left - value;
    partial[frame.depth] = value;
    stack.push_back({remaining, next_depth, 0});
  }
}

/// @brief Build the largest lattice fitting the effective population request.
/// @param requested Validated requested population,2..1000000.
/// @param m Validated objective count, at least two.
/// @return Owned actual weight rows in legacy order.
std::vector<std::vector<double>> make_weights(std::size_t requested, std::size_t m) {
  // Choose the largest H whose lattice size does not exceed the requested population.
  std::size_t H = 1;
  while (lattice_count(H + 1, m) <= std::max<std::size_t>(requested, m))
    ++H;
  std::vector<std::vector<double>> weights;
  std::vector<std::size_t>         partial(m, 0);
  das_dennis(H, 0, m, H, partial, weights);
  return weights;
}

} // namespace

MOEAD::MOEAD(MOEADOptions options) : options_(std::move(options)) {}

std::vector<ParetoPoint> MOEAD::run(
    MultiObjectiveFunction&    function,
    const std::vector<double>& lower_bound,
    const std::vector<double>& upper_bound,
    MOEADTrace*                out,
    std::size_t                cap) const {
  validate_bounds(lower_bound, upper_bound);
  const bool weighted_sum = options_.decomposition == "weighted_sum";
  if (!weighted_sum && options_.decomposition != "tchebycheff")
    throw std::invalid_argument(
        "MOEAD::optimize: decomposition must be 'tchebycheff' or 'weighted_sum'");

  if (options_.population_size < 2)
    throw std::invalid_argument("MOEAD: population must be at least two");
  if (options_.population_size > 1000000)
    throw std::length_error("MOEAD: requested population cap exceeded");
  if (!std::isfinite(options_.crossover_rate) || options_.crossover_rate < 0 ||
      options_.crossover_rate > 1 || !std::isfinite(options_.mutation_rate) ||
      (options_.mutation_rate >= 0 && options_.mutation_rate > 1) ||
      !std::isfinite(options_.eta_crossover) || options_.eta_crossover < 0 ||
      !std::isfinite(options_.eta_mutation) || options_.eta_mutation < 0)
    throw std::invalid_argument("MOEAD: invalid variation options");
  /// @brief Admit actual owned capture units before storing the next record.
  auto admit = [&](std::size_t amount) {
    if (!out)
      return;
    if (amount > std::numeric_limits<std::size_t>::max() - out->captured_entries ||
        (cap && (out->captured_entries > cap || amount > cap - out->captured_entries)))
      throw std::length_error("MOEAD: capture quota exceeded");
    out->captured_entries += amount;
  };
  if (out) {
    admit(lower_bound.size());
    admit(upper_bound.size());
    out->lower   = lower_bound;
    out->upper   = upper_bound;
    out->options = options_;
  }
  const std::size_t n = lower_bound.size();
  const double      mut_rate =
      options_.mutation_rate < 0.0 ? 1.0 / static_cast<double>(n) : options_.mutation_rate;

  std::mt19937_64                        rng(options_.seed);
  std::uniform_real_distribution<double> u(0.0, 1.0);
  auto                                   random_point = [&]() {
    std::vector<double> x(n);
    for (std::size_t j = 0; j < n; ++j)
      x[j] = lower_bound[j] + u(rng) * (upper_bound[j] - lower_bound[j]);
    return x;
  };

  // ---- Decision variables: weight vectors, neighborhoods, initial population, ideal point ----
  auto                probe = random_point();
  std::vector<double> f0    = function.evaluate(probe);
  if (out)
    ++out->evaluations;
  const std::size_t m = f0.size();
  if (m < 2)
    throw std::invalid_argument("MOEAD: at least two objectives required");
  if (m > 1000000)
    throw std::length_error("MOEAD: objective geometry cap exceeded");
  for (auto f : f0)
    if (!std::isfinite(f))
      throw std::domain_error("MOEAD: nonfinite objective");
  if (out) {
    admit(1);
    admit(n);
    admit(m);
    out->probe = {probe, f0};
  }
  /// @brief Evaluate and validate actual child/initial objective shape and finiteness.
  auto evaluate = [&](const std::vector<double>& x) {
    for (std::size_t j = 0; j < n; ++j)
      if (!std::isfinite(x[j]) || x[j] < lower_bound[j] || x[j] > upper_bound[j])
        throw std::overflow_error("MOEAD: variation left finite bounds");
    auto f = function.evaluate(x);
    if (out)
      ++out->evaluations;
    if (f.size() != m)
      throw std::invalid_argument("MOEAD: objective dimension changed");
    for (auto v : f)
      if (!std::isfinite(v))
        throw std::domain_error("MOEAD: nonfinite objective");
    return f;
  };
  const std::vector<std::vector<double>> weights = make_weights(options_.population_size, m);
  const std::size_t                      Npop    = weights.size();
  if (options_.max_generations > (std::numeric_limits<std::size_t>::max() - Npop - 1) / Npop)
    throw std::length_error("MOEAD: evaluation count overflow");
  if (out) {
    for (auto& w : weights) {
      admit(1);
      admit(w.size());
    }
    out->weights = weights;
  }
  const std::size_t T = std::min(std::max<std::size_t>(options_.neighborhood_size, 2), Npop);

  // Neighborhoods: the T nearest weight vectors (Euclidean) of each subproblem.
  std::vector<std::vector<std::size_t>> neighbors(Npop);
  for (std::size_t i = 0; i < Npop; ++i) {
    std::vector<std::size_t> order(Npop);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
      double da = 0.0, db = 0.0;
      for (std::size_t k = 0; k < m; ++k) {
        da += (weights[i][k] - weights[a][k]) * (weights[i][k] - weights[a][k]);
        db += (weights[i][k] - weights[b][k]) * (weights[i][k] - weights[b][k]);
      }
      return da < db;
    });
    neighbors[i].assign(order.begin(), order.begin() + static_cast<std::ptrdiff_t>(T));
  }

  if (out) {
    for (auto& row : neighbors) {
      admit(1);
      admit(row.size());
    }
    out->neighbors = neighbors;
  }
  std::vector<std::vector<double>> X(Npop), F(Npop);
  std::vector<double>              ideal(m, std::numeric_limits<double>::infinity());
  for (std::size_t i = 0; i < Npop; ++i) {
    X[i] = random_point();
    F[i] = evaluate(X[i]);
    for (std::size_t k = 0; k < m; ++k)
      ideal[k] = std::min(ideal[k], F[i][k]);
  }

  if (out) {
    for (std::size_t i = 0; i < Npop; ++i) {
      admit(1);
      admit(n);
      admit(m);
      out->initial_population.push_back({X[i], F[i]});
    }
    admit(m);
    out->initial_ideal = ideal;
  }
  /// @brief Evaluate the actual current ideal-anchored scalar score.
  auto scalarize = [&](const std::vector<double>& f, const std::vector<double>& w) {
    if (weighted_sum) {
      double s = 0.0;
      for (std::size_t k = 0; k < m; ++k)
        s += w[k] * (f[k] - ideal[k]);
      if (!std::isfinite(s))
        throw std::overflow_error("MOEAD: scalarization overflow");
      return s;
    }
    double g = 0.0; // Tchebycheff: max_k w_k * |f_k - z*_k|, with a weight floor for extremes.
    for (std::size_t k = 0; k < m; ++k)
      g = std::max(g, std::max(w[k], 1e-6) * std::fabs(f[k] - ideal[k]));
    if (!std::isfinite(g))
      throw std::overflow_error("MOEAD: scalarization overflow");
    return g;
  };

  for (std::size_t gen = 0; gen < options_.max_generations; ++gen) {
    for (std::size_t i = 0; i < Npop; ++i) {
      // Reproduction: two parents drawn with replacement from subproblem i's neighborhood.
      std::uniform_int_distribution<std::size_t> pick(0, T - 1);
      const std::size_t                          a = neighbors[i][pick(rng)];
      std::size_t                                b = neighbors[i][pick(rng)];
      std::vector<double>                        c1, c2;
      detail::sbx_crossover(
          X[a],
          X[b],
          c1,
          c2,
          lower_bound,
          upper_bound,
          options_.crossover_rate,
          options_.eta_crossover,
          rng);
      detail::polynomial_mutation(
          c1, lower_bound, upper_bound, mut_rate, options_.eta_mutation, rng);
      const std::vector<double> fy = evaluate(c1);

      // Update the ideal point, then let the child replace any neighbor it improves.
      for (std::size_t k = 0; k < m; ++k)
        ideal[k] = std::min(ideal[k], fy[k]);
      MOEADStep record;
      if (out) {
        admit(1);
        admit(n);
        admit(m);
        admit(m);
        record.generation = gen;
        record.subproblem = i;
        record.parent_a   = a;
        record.parent_b   = b;
        record.child      = {c1, fy};
        record.ideal      = ideal;
      }
      for (const std::size_t j : neighbors[i]) {
        const auto child_score     = scalarize(fy, weights[j]),
                   incumbent_score = scalarize(F[j], weights[j]);
        const bool replace         = child_score <= incumbent_score;
        if (out) {
          admit(1);
          record.replacements.push_back({j, incumbent_score, child_score, replace});
        }
        if (replace) {
          X[j] = c1;
          F[j] = fy;
        }
      }
      if (out)
        out->steps.push_back(std::move(record));
    }
  }

  std::vector<ParetoPoint> points;
  points.reserve(Npop);
  for (std::size_t i = 0; i < Npop; ++i)
    points.push_back(ParetoPoint{X[i], F[i]});
  if (out) {
    for (auto& p : points) {
      admit(1);
      admit(p.coordinates.size());
      admit(p.objectives.size());
    }
    out->final_population = points;
    admit(m);
    out->final_ideal = ideal;
  }
  auto front = non_dominated_front(points);
  if (out) {
    for (auto& p : front) {
      admit(1);
      admit(p.coordinates.size());
      admit(p.objectives.size());
    }
  }
  return front;
}

std::vector<ParetoPoint> MOEAD::optimize(
    MultiObjectiveFunction&    f,
    const std::vector<double>& l,
    const std::vector<double>& u) const {
  return run(f, l, u, nullptr, 0);
}
MOEADTrace MOEAD::trace(
    MultiObjectiveFunction&    f,
    const std::vector<double>& l,
    const std::vector<double>& u,
    std::size_t                cap) const {
  MOEADTrace t;
  t.front = run(f, l, u, &t, cap);
  return t;
}
} // namespace datamunge::optim
