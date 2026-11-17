/// @file todd_coxeter.cpp
/// @brief Actual checked HLT scans and iterative inverse-aware congruence closure.
#include <algorithm>
#include <datamunge/algorithms/todd_coxeter.hpp>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Private invocation-owned coset enumeration state.
namespace {
/// @brief Internal non-error resource stop, converted to explicit unresolved output.
struct Limit {
  const char* reason; ///< Static actual resource-stop description.
};
/// @brief Actual table/union producer, without recursion or borrowed output storage.
struct Enumerator {
  int                columns{};  ///< Actual2*ngen symbol columns.
  const ToddOptions& o;          ///< Borrowed controls for this call.
  ToddRun&           r;          ///< Actual output/counter accumulator.
  bool               capture{};  ///< Retain actual event records.
  std::vector<int>   parent;     ///< Actual coincidence forest, lower root chosen.
  std::size_t        alive{};    ///< Actual current representative count.
  std::size_t        revision{}; ///< Actual changed links/definitions/merges.
  std::size_t        retained{}; ///< Actual retained record/word-symbol units.

  /// @brief Count one actual selected kernel event before policy exhaustion.
  void tick() {
    if (r.operations == std::numeric_limits<std::size_t>::max())
      throw std::length_error("Todd counter capacity");
    if (o.max_operations && r.operations >= o.max_operations)
      throw Limit{"operation limit"};
    ++r.operations;
  }
  /// @brief Capture an actual producer event after its mutation.
  void record(
      const char*             action,
      int                     a,
      int                     b      = -1,
      int                     symbol = -1,
      const std::vector<int>& word   = {}) {
    if (!capture)
      return;
    auto cap = std::numeric_limits<std::size_t>::max();
    if (word.size() == cap || retained > cap - word.size() - 1 ||
        (o.max_history_values &&
         (retained > o.max_history_values || word.size() + 1 > o.max_history_values - retained)))
      throw std::length_error("Todd history");
    retained += word.size() + 1;
    r.steps.push_back({action, a, b, symbol, r.table.size(), alive, word});
  }
  /// @brief Follow actual coincidence parents and compress by halving.
  int rep(int a) {
    tick();
    while (parent[a] != a) {
      tick();
      parent[a] = parent[parent[a]];
      a         = parent[a];
    }
    return a;
  }
  /// @brief Resolve an actual representative without consuming capped algorithm work during output finalization.
  int final_rep(int a) const {
    while (parent[a] != a)
      a = parent[a];
    return a;
  }
  /// @brief Define one actual coset row under total-definition and table-cell policies.
  int define() {
    tick();
    auto n = r.table.size();
    if (n >= o.max_cosets)
      throw Limit{"coset limit"};
    if (n >= std::size_t(std::numeric_limits<int>::max()))
      throw std::length_error("Todd int rows");
    auto cap = std::numeric_limits<std::size_t>::max();
    if (columns && n + 1 > cap / std::size_t(columns))
      throw std::length_error("Todd table capacity");
    if (o.max_table_cells && (n + 1) * std::size_t(columns) > o.max_table_cells)
      throw Limit{"table limit"};
    r.table.emplace_back(columns, -1);
    parent.push_back(int(n));
    ++alive;
    ++revision;
    ++r.definitions;
    record("define", int(n));
    return int(n);
  }
  /// @brief Merge actual roots and cascade functional congruences across both directions.
  void merge(int a, int b) {
    std::vector<std::pair<int, int>> queue{{a, b}};
    while (!queue.empty()) {
      tick();
      auto [x, y] = queue.back();
      queue.pop_back();
      x = rep(x);
      y = rep(y);
      if (x == y)
        continue;
      if (x > y)
        std::swap(x, y);
      parent[y] = x;
      --alive;
      ++revision;
      ++r.coincidences;
      record("merge", x, y);
      for (int symbol = 0; symbol < columns; ++symbol) {
        tick();
        int from = r.table[y][symbol], to = r.table[x][symbol];
        if (from < 0)
          continue;
        if (to < 0) {
          r.table[x][symbol] = rep(from);
          ++revision;
          record("link", x, r.table[x][symbol], symbol);
        } else {
          auto u = rep(to), v = rep(from);
          if (u != v)
            queue.emplace_back(u, v);
        }
      }
    }
  }
  /// @brief Define/deduce an actual two-way generator link, merging existing conflicting images first.
  void link(int a, int symbol, int b) {
    tick();
    a = rep(a);
    b = rep(b);
    if (r.table[a][symbol] >= 0)
      merge(r.table[a][symbol], b);
    a = rep(a);
    b = rep(b);
    if (r.table[b][symbol ^ 1] >= 0)
      merge(r.table[b][symbol ^ 1], a);
    a = rep(a);
    b = rep(b);
    if (r.table[a][symbol] != b || r.table[b][symbol ^ 1] != a) {
      r.table[a][symbol]     = b;
      r.table[b][symbol ^ 1] = a;
      ++revision;
      record("link", a, b, symbol);
    }
  }
  /// @brief Read an actual normalized row image, preserving undefined gaps.
  int image(int a, int symbol) {
    a     = rep(a);
    int v = r.table[a][symbol];
    return v < 0 ? -1 : rep(v);
  }
  /// @brief Execute an actual HLT relator/subgroup scan with gap deduction/definition.
  void scan(int start, const std::vector<int>& word, const char* kind) {
    tick();
    ++r.word_scans;
    start = rep(start);
    record(kind, start, -1, -1, word);
    if (word.empty())
      return;
    int         f = start, b = start;
    std::size_t i = 0, j = word.size();
    while (true) {
      tick();
      while (i < j) {
        auto next = image(f, word[i]);
        if (next < 0)
          break;
        f = next;
        ++i;
        ++r.symbol_visits;
      }
      if (i == j) {
        merge(f, rep(start));
        return;
      }
      while (j > i) {
        auto next = image(b, word[j - 1] ^ 1);
        if (next < 0)
          break;
        b = next;
        --j;
        ++r.symbol_visits;
      }
      if (i == j) {
        merge(f, b);
        return;
      }
      if (j - i == 1) {
        link(f, word[i], b);
        return;
      }
      auto fresh = define();
      link(f, word[i], fresh);
      f = rep(f);
      b = rep(b);
    }
  }
  /// @brief Own normalized raw links and the complete compact action on final representatives.
  void finalize() {
    r.representatives.resize(parent.size());
    std::vector<int> compact(parent.size(), -1);
    for (std::size_t i = 0; i < parent.size(); ++i) {
      r.representatives[i] = final_rep(int(i));
      if (r.representatives[i] == int(i)) {
        compact[i] = int(r.live.size());
        r.live.push_back(int(i));
      }
    }
    for (auto& row : r.table)
      for (auto& v : row)
        if (v >= 0)
          v = final_rep(v);
    if (r.closed) {
      r.index = static_cast<long long>(r.live.size());
      for (auto id : r.live) {
        std::vector<int> row;
        for (int v : r.table[id]) {
          if (v < 0)
            throw std::logic_error("Todd closed row has gap");
          row.push_back(compact[v]);
        }
        r.compact_table.push_back(std::move(row));
      }
    }
  }
};
/// @brief Validate all signed words before allocating coset state and convert them to internal symbols.
std::vector<std::vector<int>> symbols(
    const std::vector<std::vector<int>>& words,
    int                                  ngen,
    std::size_t&                         total,
    const ToddOptions&                   o) {
  std::vector<std::vector<int>> out;
  for (const auto& word : words) {
    if (word.size() > std::numeric_limits<std::size_t>::max() - total)
      throw std::length_error("Todd words capacity");
    total += word.size();
    if (o.max_word_symbols && total > o.max_word_symbols)
      throw std::length_error("Todd complete words");
    std::vector<int> converted;
    for (int v : word) {
      long long magnitude = v < 0 ? -static_cast<long long>(v) : v;
      if (!magnitude || magnitude > ngen)
        throw std::invalid_argument("Todd signed generator");
      converted.push_back(2 * int(magnitude - 1) + (v < 0));
    }
    out.push_back(std::move(converted));
  }
  return out;
}
/// @brief Validate complete presentation then run actual scan/fill passes until stable or capped.
ToddRun execute(
    int                                  ngen,
    const std::vector<std::vector<int>>& relators,
    const std::vector<std::vector<int>>& subgroup,
    const ToddOptions&                   o,
    bool                                 capture) {
  if (ngen < 0 || ngen > std::numeric_limits<int>::max() / 2 || !o.max_cosets)
    throw std::invalid_argument("Todd generator/coset controls");
  if (o.max_cosets > std::size_t(std::numeric_limits<int>::max()))
    throw std::length_error("Todd int coset capacity");
  std::size_t total = 0;
  auto        rel = symbols(relators, ngen, total, o), sub = symbols(subgroup, ngen, total, o);
  ToddRun     r;
  Enumerator  e{2 * ngen, o, r, capture};
  try {
    e.define();
    for (;;) {
      auto before = e.revision;
      for (const auto& w : sub)
        e.scan(0, w, "subgroup");
      for (std::size_t c = 0; c < r.table.size(); ++c) {
        if (e.rep(int(c)) != int(c))
          continue;
        for (const auto& w : rel)
          e.scan(int(c), w, "relator");
        if (e.rep(int(c)) != int(c))
          continue;
        for (int x = 0; x < e.columns; ++x) {
          if (e.rep(int(c)) != int(c))
            break;
          if (e.image(int(c), x) < 0) {
            auto fresh = e.define();
            e.link(int(c), x, fresh);
            for (const auto& w : rel)
              e.scan(int(c), w, "relator");
          }
        }
      }
      if (e.revision == before) {
        r.closed = true;
        r.stop   = "complete";
        break;
      }
    }
  } catch (const Limit& limit) {
    r.stop = limit.reason;
  }
  e.finalize();
  return r;
}
} // namespace
ToddRun todd_coxeter_run(
    int                                  ngen,
    const std::vector<std::vector<int>>& relators,
    const std::vector<std::vector<int>>& subgroup_gens,
    const ToddOptions&                   options) {
  return execute(ngen, relators, subgroup_gens, options, false);
}
ToddRun todd_coxeter_trace(
    int                                  ngen,
    const std::vector<std::vector<int>>& relators,
    const std::vector<std::vector<int>>& subgroup_gens,
    const ToddOptions&                   options) {
  return execute(ngen, relators, subgroup_gens, options, true);
}
long long todd_coxeter_index(
    int                                  ngen,
    const std::vector<std::vector<int>>& relators,
    const std::vector<std::vector<int>>& subgroup_gens,
    long long                            max_cosets) {
  if (max_cosets <= 0)
    throw std::invalid_argument("Todd positive coset cap");
  if (max_cosets > std::numeric_limits<int>::max())
    throw std::length_error("Todd int coset cap");
  ToddOptions o;
  o.max_cosets = static_cast<std::size_t>(max_cosets);
  return todd_coxeter_run(ngen, relators, subgroup_gens, o).index;
}
} // namespace datamunge::algorithms
