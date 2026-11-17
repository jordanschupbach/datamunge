/// @file maximum_parsimony.cpp
/// @brief Canonical maximum parsimony API, producing histories and checked native contracts.

#include <algorithm>
#include <bit>
#include <datamunge/algorithms/maximum_parsimony.hpp>
#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Implementation-only helpers for canonical maximum parsimony production and capture.
namespace {
/// @brief Validate every structural field and derive deterministic left-before-right postorder.
/// @param tree Complete original tree.
/// @return Every node once in actual scoring dependency order.
/// @throws std::invalid_argument On malformed root/children, parent counts or disconnected nodes.
/// @throws std::length_error If ids/counts cannot be represented.
std::vector<int> validate(const ParsimonyTree& tree) {
  const auto n = tree.children.size();
  if (n > static_cast<std::size_t>(INT32_MAX) || n > SIZE_MAX / 7)
    throw std::length_error("Fitch node geometry");
  if (!n) {
    if (tree.root != -1)
      throw std::invalid_argument("Fitch empty root");
    return {};
  }
  if (tree.root < 0 || static_cast<std::size_t>(tree.root) >= n)
    throw std::invalid_argument("Fitch root");
  std::vector<unsigned char> parents(n);
  for (auto [left, right] : tree.children) {
    if (left == -1 && right == -1)
      continue;
    if (left < 0 || right < 0 || static_cast<std::size_t>(left) >= n ||
        static_cast<std::size_t>(right) >= n || left == right)
      throw std::invalid_argument("Fitch full binary children");
    for (auto child : {left, right}) {
      if (++parents[child] > 1)
        throw std::invalid_argument("Fitch shared child");
    }
  }
  for (std::size_t i = 0; i < n; ++i)
    if (parents[i] != (static_cast<int>(i) == tree.root ? 0 : 1))
      throw std::invalid_argument("Fitch parent count");
  std::vector<int> preorder, stack{tree.root};
  preorder.reserve(n);
  while (!stack.empty()) {
    int node = stack.back();
    stack.pop_back();
    preorder.push_back(node);
    auto [left, right] = tree.children[node];
    if (left != -1) {
      stack.push_back(left);
      stack.push_back(right);
    }
  }
  if (preorder.size() != n)
    throw std::invalid_argument("Fitch disconnected cycle");
  std::reverse(preorder.begin(), preorder.end());
  return preorder;
}
/// @brief Validate complete column length and every leaf state before any shift.
/// @param tree Checked complete tree.
/// @param states Original node-indexed column.
/// @throws std::invalid_argument On malformed length or leaf state outside0..31.
void validate_states(const ParsimonyTree& tree, const std::vector<int>& states) {
  if (states.size() != tree.children.size())
    throw std::invalid_argument("Fitch column length");
  for (std::size_t i = 0; i < states.size(); ++i)
    if (tree.children[i].first == -1 && (states[i] < 0 || states[i] > 31))
      throw std::invalid_argument("Fitch leaf state");
}
/// @brief Canonical actual postorder scoring and deterministic minimum-change witness assignment.
/// @param tree Checked original full binary tree.
/// @param states Checked node-indexed column.
/// @param order Validated actual postorder dependency schedule.
/// @param trace Optional producing capture target.
/// @return Actual masks, minimum score, chosen witness and work.
FitchAnalysis produce(
    const ParsimonyTree&    tree,
    const std::vector<int>& states,
    const std::vector<int>& order,
    FitchTrace*             trace) {
  FitchAnalysis out;
  auto          n = tree.children.size();
  out.masks.resize(n);
  out.states.resize(n);
  if (!n)
    return out;
  for (int node : order) {
    auto [left, right]   = tree.children[node];
    int           before = out.score;
    std::uint32_t a = 0, b = 0, intersection = 0, mask;
    if (left == -1) {
      mask = std::uint32_t{1} << states[node];
      ++out.work.leaves;
    } else {
      a            = out.masks[left];
      b            = out.masks[right];
      intersection = a & b;
      if (intersection) {
        mask = intersection;
        ++out.work.intersections;
      } else {
        mask = a | b;
        ++out.score;
        ++out.work.unions;
      }
    }
    out.masks[node] = mask;
    ++out.work.nodes;
    if (trace)
      trace->steps.push_back({node, left, right, a, b, intersection, mask, before, out.score});
  }
  out.states[tree.root] = std::countr_zero(out.masks[tree.root]);
  std::vector<int> stack{tree.root};
  while (!stack.empty()) {
    int node = stack.back();
    stack.pop_back();
    auto [left, right] = tree.children[node];
    if (left == -1)
      continue;
    for (int child : {left, right}) {
      auto parent       = out.states[node];
      auto mask         = out.masks[child];
      int  chosen       = (mask & (std::uint32_t{1} << parent)) ? parent : std::countr_zero(mask);
      out.states[child] = chosen;
      bool changed      = parent != chosen;
      ++out.work.edges;
      out.work.changes += changed;
      if (trace)
        trace->edges.push_back({node, child, parent, chosen, changed});
    }
    stack.push_back(right);
    stack.push_back(left);
  }
  if (out.work.changes != static_cast<std::size_t>(out.score))
    throw std::logic_error("Fitch witness invariant");
  return out;
}
} // namespace
FitchAnalysis fitch_analyze(const ParsimonyTree& tree, const std::vector<int>& states) {
  auto order = validate(tree);
  validate_states(tree, states);
  return produce(tree, states, order, nullptr);
}
FitchTrace
fitch_trace(const ParsimonyTree& tree, const std::vector<int>& states, std::size_t limit) {
  auto order = validate(tree);
  validate_states(tree, states);
  auto n     = tree.children.size();
  auto count = n ? 7 * n : 1;
  if (limit && count > limit)
    throw std::length_error("Fitch complete capture quota");
  FitchTrace out;
  out.captured_entries = count;
  out.tree             = tree;
  out.leaf_states      = states;
  out.steps.reserve(n);
  out.edges.reserve(n ? n - 1 : 0);
  out.analysis = produce(tree, states, order, &out);
  return out;
}
int fitch_small_parsimony(const ParsimonyTree& tree, const std::vector<int>& states) {
  return fitch_analyze(tree, states).score;
}
std::vector<int> parsimony_site_scores(
    const ParsimonyTree&                 tree,
    const std::vector<std::vector<int>>& states) {
  auto order = validate(tree);
  for (auto& column : states)
    validate_states(tree, column);
  std::vector<int> out;
  out.reserve(states.size());
  for (auto& column : states)
    out.push_back(produce(tree, column, order, nullptr).score);
  return out;
}
int parsimony_score(const ParsimonyTree& tree, const std::vector<std::vector<int>>& states) {
  auto scores = parsimony_site_scores(tree, states);
  int  total  = 0;
  for (int score : scores) {
    if (score > std::numeric_limits<int>::max() - total)
      throw std::overflow_error("Fitch summed score");
    total += score;
  }
  return total;
}
ParsimonyTree parsimony_four_taxon_tree(int a, int b, int c, int d) {
  std::array<int, 4> ids{a, b, c, d};
  std::sort(ids.begin(), ids.end());
  if (ids != std::array<int, 4>{0, 1, 2, 3})
    throw std::invalid_argument("Fitch four-taxon permutation");
  ParsimonyTree out;
  out.root = 6;
  out.children.assign(7, {-1, -1});
  out.children[4] = {a, b};
  out.children[5] = {c, d};
  out.children[6] = {4, 5};
  return out;
}
std::vector<int> parsimony_four_taxon_column(std::array<int, 4> states) {
  std::vector<int> out(7, -1);
  for (std::size_t i = 0; i < 4; ++i) {
    if (states[i] < 0 || states[i] > 31)
      throw std::invalid_argument("Fitch four-taxon state");
    out[i] = states[i];
  }
  return out;
}
std::vector<std::vector<int>> parsimony_demo_columns() {
  return {
      parsimony_four_taxon_column({0, 0, 1, 1}),
      parsimony_four_taxon_column({0, 0, 1, 1}),
      parsimony_four_taxon_column({1, 1, 0, 0}),
      parsimony_four_taxon_column({0, 1, 0, 0}),
      parsimony_four_taxon_column({0, 0, 0, 1})};
}
} // namespace datamunge::algorithms
