/// @file sss_star.cpp
/// @brief Actual fail-soft memory search with checked complete input and explicit probe intervals.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/sss_star.hpp>
#include <limits>
#include <random>
#include <stdexcept>
#include <unordered_map>
namespace datamunge::algorithms {
/// @brief Translation-unit-private checked producing kernels; no shared mutable state.
namespace {
/// @brief Actual transposition interval for one immutable node.
struct Bounds {
  double lower; ///< Actual known lower value bound.
  double upper; ///< Actual known upper value bound.
};
/// @brief Invocation-local memory, controls and actual counters.
struct State {
  const SSSOptions& o;       ///< Borrowed controls for this invocation.
  SSSRun&           r;       ///< Borrowed actual result/counter accumulator.
  bool              capture; ///< Retain actual writes/probes.
  std::unordered_map<const GameNode*, std::size_t>
      ids; ///< Actual preorder node identities, valid only during call.
  std::unordered_map<const GameNode*, Bounds>
              table;     ///< Actual memory bounds, no iterator kept across recursion.
  std::size_t records{}; ///< Actual retained records.

  /// @brief Bound an actual fixed-size record before capture.
  void record() {
    if (records == std::numeric_limits<std::size_t>::max() ||
        (o.max_records && records >= o.max_records))
      throw std::length_error("SSS: records");
    ++records;
  }
};
/// @brief Validate actual complete reachable tree and collect all leaf values.
/// @param node Borrowed current node.
/// @param depth Actual root-relative depth.
/// @param state Invocation state.
/// @param leaves Actual finite leaf values read during preprocessing.
void validate(const GameNode& node, std::size_t depth, State& state, std::vector<double>& leaves) {
  if (depth > 1024 || (state.o.max_depth && depth > state.o.max_depth))
    throw std::length_error("SSS: depth");
  if (state.r.preflight_nodes == std::numeric_limits<std::size_t>::max() ||
      (state.o.max_nodes && state.r.preflight_nodes >= state.o.max_nodes))
    throw std::length_error("SSS: nodes");
  state.ids.emplace(&node, state.r.preflight_nodes++);
  if (node.leaf) {
    if (!node.children.empty() || !std::isfinite(node.value))
      throw std::invalid_argument("SSS: finite terminal leaf");
    leaves.push_back(node.value);
    ++state.r.preflight_leaf_reads;
  } else {
    if (node.children.empty())
      throw std::invalid_argument("SSS: empty internal node");
    for (const auto& child : node.children)
      validate(child, depth + 1, state, leaves);
  }
}
/// @brief Execute one actual fail-soft alpha-beta-with-memory call.
/// @param node Actual immutable node.
/// @param alpha Actual lower window endpoint.
/// @param beta Actual upper window endpoint.
/// @param maximizing Actual node player.
/// @param state Invocation-local memory/work.
/// @return Actual fail-soft value/bound.
double search(const GameNode& node, double alpha, double beta, bool maximizing, State& state) {
  auto&      r     = state.r;
  const auto limit = std::numeric_limits<long long>::max();
  if (r.result.nodes_visited == limit ||
      (state.o.max_search_visits &&
       static_cast<std::size_t>(r.result.nodes_visited) >= state.o.max_search_visits))
    throw std::length_error("SSS: visits");
  ++r.result.nodes_visited;
  const double inf = std::numeric_limits<double>::infinity();
  if (const auto found = state.table.find(&node); found != state.table.end()) {
    ++r.table_hits;
    if (found->second.lower >= beta) {
      ++r.result.cutoffs;
      return found->second.lower;
    }
    if (found->second.upper <= alpha) {
      ++r.result.cutoffs;
      return found->second.upper;
    }
    alpha = std::max(alpha, found->second.lower);
    beta  = std::min(beta, found->second.upper);
  }
  double value;
  if (node.leaf) {
    if (r.result.leaves_evaluated == limit)
      throw std::length_error("SSS: leaf counter");
    ++r.result.leaves_evaluated;
    value = node.value;
  } else if (maximizing) {
    value    = -inf;
    double a = alpha;
    for (const auto& child : node.children) {
      if (value >= beta) {
        ++r.result.cutoffs;
        break;
      }
      value = std::max(value, search(child, a, beta, false, state));
      a     = std::max(a, value);
    }
  } else {
    value    = inf;
    double b = beta;
    for (const auto& child : node.children) {
      if (value <= alpha) {
        ++r.result.cutoffs;
        break;
      }
      value = std::min(value, search(child, alpha, b, true, state));
      b     = std::min(b, value);
    }
  }
  Bounds bounds{-inf, inf};
  if (value <= alpha)
    bounds.upper = value;
  else if (value >= beta)
    bounds.lower = value;
  else
    bounds = {value, value};
  const auto prior = state.table.find(&node);
  if (prior != state.table.end()) {
    bounds.lower  = std::max(bounds.lower, prior->second.lower);
    bounds.upper  = std::min(bounds.upper, prior->second.upper);
    prior->second = bounds;
  } else
    state.table.emplace(&node, bounds);
  if (bounds.lower > bounds.upper)
    throw std::logic_error("SSS: crossed memory interval");
  if (state.capture) {
    state.record();
    r.bounds.push_back({state.ids.at(&node), bounds.lower, bounds.upper});
  }
  return value;
}
/// @brief Drive actual upper-bound null-window probes.
/// @param root Immutable complete input.
/// @param maximizing Root player.
/// @param resolution Compatible finite positive width.
/// @param o Policies.
/// @param capture Retain actual probe/memory records.
/// @return Actual owned result.
SSSRun
run(const GameNode& root, bool maximizing, double resolution, const SSSOptions& o, bool capture) {
  if (!std::isfinite(resolution) || resolution <= 0)
    throw std::invalid_argument("SSS: resolution");
  SSSRun              r;
  State               state{o, r, capture};
  std::vector<double> leaves;
  validate(root, 0, state, leaves);
  std::sort(leaves.begin(), leaves.end());
  for (std::size_t i = 1; i < leaves.size(); ++i)
    if (leaves[i] != leaves[i - 1] &&
        static_cast<long double>(leaves[i]) - leaves[i - 1] < resolution)
      throw std::invalid_argument("SSS: leaf gaps below resolution");
  const double maximum = leaves.back(), inf = std::numeric_limits<double>::infinity();
  double       guess = maximum + resolution;
  if (!std::isfinite(guess) || guess <= maximum)
    guess = maximum;
  r.lower        = -inf;
  r.upper        = inf;
  r.result.value = guess;
  while (r.lower < r.upper && r.probes < o.max_probes) {
    double beta = guess;
    if (guess == r.lower) {
      beta = guess + resolution;
      if (beta <= guess)
        beta = std::nextafter(guess, inf);
    }
    double alpha = std::isinf(beta) ? guess : beta - resolution;
    if (alpha >= beta)
      alpha = std::nextafter(beta, -inf);
    guess = search(root, alpha, beta, maximizing, state);
    if (guess < beta)
      r.upper = guess;
    else
      r.lower = guess;
    r.result.value = guess;
    if (capture) {
      state.record();
      r.history.push_back(
          {r.probes,
           alpha,
           beta,
           guess,
           r.lower,
           r.upper,
           r.result.leaves_evaluated,
           r.result.nodes_visited,
           state.table.size()});
    }
    ++r.probes;
  }
  r.converged = r.lower >= r.upper;
  return r;
}
/// @brief Recursively generate the original bounded actual RNG fixture.
/// @param rng Mutable local historical engine.
/// @param depth Remaining depth.
/// @param nodes Actual generated count.
/// @param cap Zero or generated-node policy.
/// @return Owned actual subtree.
GameNode tree(std::mt19937& rng, int depth, std::size_t& nodes, std::size_t cap) {
  if (nodes == std::numeric_limits<std::size_t>::max() || (cap && nodes >= cap))
    throw std::length_error("SSS: fixture nodes");
  ++nodes;
  GameNode node;
  if (!depth) {
    node.leaf  = true;
    node.value = static_cast<double>(rng() % 21) - 10;
  } else {
    const int count = 2 + static_cast<int>(rng() % 3);
    for (int i = 0; i < count; ++i)
      node.children.push_back(tree(rng, depth - 1, nodes, cap));
  }
  return node;
}
} // namespace
SSSRun sss_star_run(const GameNode& root, bool max, double resolution, const SSSOptions& o) {
  return run(root, max, resolution, o, false);
}
SSSRun sss_star_trace(const GameNode& root, bool max, double resolution, const SSSOptions& o) {
  return run(root, max, resolution, o, true);
}
GameResult sss_star(const GameNode& root, bool max, double resolution) {
  auto r = sss_star_run(root, max, resolution);
  if (!r.converged)
    throw std::runtime_error("SSS: unresolved probe cap");
  return r.result;
}
GameNode sss_example_tree(int depth, std::uint32_t seed, std::size_t cap) {
  if (depth < 0 || depth > 10)
    throw std::invalid_argument("SSS: fixture depth");
  std::mt19937 rng(seed);
  std::size_t  nodes = 0;
  return tree(rng, depth, nodes, cap);
}
} // namespace datamunge::algorithms
