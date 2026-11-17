/// @file package_merge.cpp
/// @brief Canonical checked explicit-member package-merge with exact128-bit cost and actual list histories.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/integer_arithmetic.hpp>
#include <datamunge/algorithms/package_merge.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Private exact-cost arithmetic, checked counters and one actual coin-list path.
namespace {
/// @brief Native fixed-width exact unsigned arithmetic used only within implementation.
using Wide = unsigned __int128;
/// @brief Checked nonnegative scalar count addition.
/// @param a Existing count.
/// @param b Added count.
/// @return Checked sum.
/// @throws std::length_error If sum is unrepresentable.
std::size_t add(std::size_t a, std::size_t b) {
  if (b > std::numeric_limits<std::size_t>::max() - a)
    throw std::length_error("package merge: count geometry");
  return a + b;
}
/// @brief Checked exact128-bit cost addition.
/// @param a Existing exact cost.
/// @param b Added exact cost.
/// @return Exact sum.
/// @throws std::overflow_error If sum exceeds128bits.
Wide sum(Wide a, Wide b) {
  if (b > ~Wide(0) - a)
    throw std::overflow_error("package merge: exact128-bit cost exceeded");
  return a + b;
}
/// @brief Store exact native128bits in public high/low layout.
/// @param value Complete exact cost.
/// @return Owned exact cost representation.
PackageMergeCost packed(Wide value) {
  return {std::uint64_t(value >> 64), std::uint64_t(value)};
}
/// @brief Recover exact native128bits from public high/low layout.
/// @param value Complete public cost.
/// @return Exact native unsigned value.
Wide unpacked(PackageMergeCost value) {
  return (Wide(value.high) << 64) | value.low;
}
/// @brief Actual mutable working coin and original membership occurrences.
struct Coin {
  Wide                     weight{0}; ///< Actual exact cost, checked before every package sum.
  std::vector<std::size_t> members;   ///< Actual owned symbol occurrences.
  std::size_t              id{0}; ///< Unique construction identity, retained through list copies.
};
/// @brief Execute actual sorted-list packaging and final membership selection.
/// @param weights Complete positive original frequencies.
/// @param max_len Strictly positive requested cap.
/// @param options Complete original policies.
/// @param history Optional owned actual-history sink.
/// @return Actual optimal lengths, exact cost, diagnostics and selected work.
/// @throws std::invalid_argument If complete weights/cap feasibility fail.
/// @throws std::length_error If complete/actual geometry or selected policies fail.
/// @throws std::overflow_error If exact costs exceed128bits.
PackageMergeRun run(
    const std::vector<std::uint64_t>& weights,
    int                               max_len,
    PackageMergeOptions               options,
    PackageMergeTrace*                history) {
  auto n = weights.size();
  if (max_len < 1)
    throw std::invalid_argument("package merge: positive maximum length required");
  for (auto weight : weights)
    if (!weight)
      throw std::invalid_argument("package merge: complete positive weights required");
  std::size_t bits = 0;
  for (auto remaining = n ? n - 1 : 0; remaining; remaining >>= 1)
    ++bits;
  if (n > 1 && std::size_t(max_len) < bits)
    throw std::invalid_argument("package merge: infeasible symbol/length capacity");
  if (options.max_symbols && n > options.max_symbols)
    throw std::length_error("package merge: complete symbol quota");
  if (n > std::vector<int>().max_size() || n > std::vector<Coin>().max_size())
    throw std::length_error("package merge: complete vector geometry");
  auto            take = n > 1 ? add(n - 1, n - 1) : 0;
  PackageMergeRun actual;
  actual.requested_max_length = max_len;
  actual.effective_levels     = n > 1 ? std::min<std::size_t>(max_len, n - 1) : n;
  auto& w                     = actual.work;
  /// @brief Admit actual owned history units before each corresponding copy/append.
  /// @param units Actual next record contribution.
  /// @throws std::length_error If count geometry or selected history quota fails.
  auto charge = [&](std::size_t units) {
    if (!history)
      return;
    auto next = add(history->captured_entries, units);
    if (options.max_trace_entries && next > options.max_trace_entries)
      throw std::length_error("package merge: actual trace quota");
    history->captured_entries = next;
  };
  /// @brief Admit cumulative actual working membership copies before copying elements.
  /// @param entries Actual count of destination member elements about to be written.
  /// @throws std::length_error If count geometry or selected cumulative work quota fails.
  auto members = [&](std::size_t entries) {
    auto next = add(w.member_copies, entries);
    if (options.max_member_copies && next > options.max_member_copies)
      throw std::length_error("package merge: actual membership-copy quota");
    w.member_copies = next;
  };
  if (history) {
    charge(add(2, add(n, n)));
    history->weights = weights;
  }
  actual.lengths.assign(n, 0);
  if (!n)
    return actual;
  if (n == 1) {
    actual.lengths[0]        = 1;
    actual.actual_max_length = 1;
    actual.kraft_sum         = .5L;
    actual.weighted_cost     = {0, weights[0]};
    w.levels                 = 1;
    return actual;
  }
  std::vector<Coin> singletons;
  singletons.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    members(1);
    singletons.push_back({Wide(weights[i]), {i}, i});
    if (history) {
      charge(2);
      history->coins.push_back({i, {0, weights[i]}, {i}, false, 0, 0, 0});
    }
  }
  std::sort(singletons.begin(), singletons.end(), [](const Coin& a, const Coin& b) {
    return a.weight < b.weight;
  });
  members(n);
  std::vector<Coin> current = singletons;
  std::size_t       next_id = n;
  /// @brief Capture one actual already-sorted retained list without rebuilding packages.
  /// @param level Actual zero-based level.
  /// @param unpaired Actual dropped old-list final coin count.
  /// @param pruned Actual removed sorted-tail coin count.
  auto level_record = [&](std::size_t level, std::size_t unpaired, std::size_t pruned) {
    w.levels = add(w.levels, 1);
    if (history) {
      charge(add(1, current.size()));
      PackageMergeLevel row;
      row.level              = level;
      row.discarded_unpaired = unpaired;
      row.pruned             = pruned;
      row.coins.reserve(current.size());
      for (auto& coin : current)
        row.coins.push_back(coin.id);
      history->levels.push_back(std::move(row));
    }
  };
  level_record(0, 0, 0);
  for (std::size_t level = 1; level < actual.effective_levels; ++level) {
    auto              unpaired = current.size() % 2;
    std::vector<Coin> packages;
    packages.reserve(current.size() / 2);
    for (std::size_t j = 0; j + 1 < current.size(); j += 2) {
      auto count = add(current[j].members.size(), current[j + 1].members.size());
      members(count);
      Coin package;
      package.weight = sum(current[j].weight, current[j + 1].weight);
      package.id     = next_id;
      next_id        = add(next_id, 1);
      package.members.reserve(count);
      package.members = current[j].members;
      package.members.insert(
          package.members.end(), current[j + 1].members.begin(), current[j + 1].members.end());
      if (history) {
        charge(add(1, count));
        history->coins.push_back(
            {package.id,
             packed(package.weight),
             package.members,
             true,
             current[j].id,
             current[j + 1].id,
             level});
      }
      packages.push_back(std::move(package));
      w.packages = add(w.packages, 1);
    }
    current.clear();
    current.reserve(add(n, packages.size()));
    members(n);
    current.insert(current.end(), singletons.begin(), singletons.end());
    for (auto& package : packages) {
      members(package.members.size());
      current.push_back(package);
    }
    std::stable_sort(current.begin(), current.end(), [](const Coin& a, const Coin& b) {
      return a.weight < b.weight;
    });
    auto pruned = current.size() > take ? current.size() - take : 0;
    if (pruned)
      current.resize(take);
    w.pruned_coins = add(w.pruned_coins, pruned);
    level_record(level, unpaired, pruned);
  }
  if (current.size() < take)
    throw std::logic_error("package merge: internal feasible-list invariant");
  if (history) {
    charge(take);
    history->selected.reserve(take);
  }
  for (std::size_t i = 0; i < take; ++i) {
    if (history)
      history->selected.push_back(current[i].id);
    for (auto symbol : current[i].members) {
      if (actual.lengths[symbol] == std::numeric_limits<int>::max())
        throw std::length_error("package merge: code length geometry");
      ++actual.lengths[symbol];
      w.selected_member_visits = add(w.selected_member_visits, 1);
    }
  }
  Wide cost = 0;
  for (std::size_t i = 0; i < n; ++i) {
    auto length = actual.lengths[i];
    if (length < 1 || length > max_len)
      throw std::logic_error("package merge: internal length invariant");
    cost                     = sum(cost, Wide(weights[i]) * std::size_t(length));
    actual.actual_max_length = std::max(actual.actual_max_length, length);
    actual.kraft_sum += std::ldexp(1.L, -length);
  }
  actual.weighted_cost = packed(cost);
  std::vector<std::size_t> counts(std::size_t(actual.actual_max_length) + 1);
  for (int length : actual.lengths)
    ++counts[length];
  std::size_t slots = 1;
  for (std::size_t depth = 1; depth < counts.size(); ++depth) {
    slots = add(slots, slots);
    if (counts[depth] > slots)
      throw std::logic_error("package merge: internal oversubscribed Kraft slots");
    slots -= counts[depth];
  }
  actual.kraft_equality = slots == 0;
  if (!actual.kraft_equality)
    throw std::logic_error("package merge: internal incomplete Kraft slots");
  return actual;
}
} // namespace
std::string package_merge_cost_decimal(PackageMergeCost value) {
  return uint128_text(unpacked(value));
}
double package_merge_cost_approximation(PackageMergeCost value) {
  return static_cast<double>(unpacked(value));
}
std::vector<int> package_merge(const std::vector<std::uint64_t>& weights, int max_len) {
  return run(weights, max_len, {}, nullptr).lengths;
}
PackageMergeRun package_merge_run(
    const std::vector<std::uint64_t>& weights,
    int                               max_len,
    PackageMergeOptions               options) {
  return run(weights, max_len, options, nullptr);
}
PackageMergeTrace package_merge_trace(
    const std::vector<std::uint64_t>& weights,
    int                               max_len,
    PackageMergeOptions               options) {
  PackageMergeTrace t;
  t.run = run(weights, max_len, options, &t);
  return t;
}
std::vector<std::uint64_t> package_merge_demo_weights() {
  return {20, 17, 6, 3, 2, 2, 2, 1, 1};
}
} // namespace datamunge::algorithms
