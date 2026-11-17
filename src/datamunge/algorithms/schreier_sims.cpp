/// @file schreier_sims.cpp
/// @brief Checked native closure with invocation-owned actual-work counters.
#include <algorithm>
#include <datamunge/algorithms/schreier_sims.hpp>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
/// @brief Canonical checked native algorithm producers and owned outputs.
namespace datamunge::algorithms {
/// @brief Private invocation-local checked algorithm helpers.
namespace {
/// @brief Validate a complete zero-based image-vector bijection before any indexed use.
/// @param p Borrowed complete permutation.
/// @param n Required nonnegative degree.
/// @return Whether all images are unique and in range.
bool valid(const std::vector<int>& p, int n) {
  if (n < 0 || p.size() != static_cast<std::size_t>(n))
    return false;
  std::vector<char> seen(p.size());
  for (int x : p) {
    if (x < 0 || x >= n || seen[x])
      return false;
    seen[x] = 1;
  }
  return true;
}
/// @brief Owned zero-based image vector.
using Perm=std::vector<int>;
/// @brief Invocation-owned stabilizer level.
struct Level {
  int                           point;       ///< Actual stabilizer base point.
  std::vector<Perm>             gens;        ///< Actual generators fixing earlier bases.
  std::unordered_map<int, Perm> transversal; ///< Actual orbit representatives.
};

/// @brief One independent invocation's checked permutation algebra and closure.
struct Worker {
  const SchreierSimsOptions& options;    ///< Borrowed policies.
  bool                       capture;    ///< Retain actual events.
  SchreierSimsRun            out;        ///< Owned actual outputs.
  std::size_t                retained{}; ///< Actual history units admitted.

  /// @brief Admit one actual counted permutation-loop point operation.
  void point() {
    if (out.point_operations == std::numeric_limits<std::size_t>::max() ||
        (options.max_point_operations && out.point_operations >= options.max_point_operations))
      throw std::length_error("Schreier-Sims: work policy");
    ++out.point_operations;
  }
  /// @brief Admit complete next captured event before copying.
  /// @param n Actual scalar/record units.
  void admit(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() - retained ||
        (options.max_trace_entries &&
         (retained > options.max_trace_entries || n > options.max_trace_entries - retained)))
      throw std::length_error("Schreier-Sims: capture policy");
    retained += n;
  }

  /// @brief Construct an actual identity while counting initialized points.
/// @param n Validated degree.
/// @return Owned identity permutation.
Perm identity(int n) {
    Perm p(n);
    for (int i = 0; i < n; ++i) {
      point();
      p[i] = i;
    }
    return p;
  }

  // (a . b)[i] = a[b[i]]  -- apply b, then a.
  /// @brief Compose actual validated permutations and count each output point.
/// @param a Outer permutation.
/// @param b Inner permutation.
/// @return Owned a composed with b.
Perm compose(const Perm& a, const Perm& b) {
    ++out.compositions;
    Perm r(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) {
      point();
      r[i] = a[b[i]];
    }
    return r;
  }

  /// @brief Invert a validated permutation and count actual point assignments.
/// @param a Borrowed bijective image vector.
/// @return Owned inverse.
Perm inverse(const Perm& a) {
    ++out.inversions;
    Perm r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) {
      point();
      r[a[i]] = static_cast<int>(i);
    }
    return r;
  }

  /// @brief Test actual identity, counting visited points until first mismatch.
/// @param a Borrowed validated permutation.
/// @return Whether every image fixes its point.
bool is_identity(const Perm& a) {
    for (std::size_t i = 0; i < a.size(); ++i) {
      point();
      if (a[i] != static_cast<int>(i))
        return false;
    }
    return true;
  }

  // (Re)build the orbit of level.point under level.gens, with a transversal.
  /// @brief Rebuild an actual orbit with breadth-first representatives.
/// @param level Mutable current level.
/// @param n Validated degree.
/// @param level_index Actual history level identity.
void build_transversal(Level& level, int n, std::size_t level_index) {
    ++out.orbit_builds;
    level.transversal.clear();
    level.transversal[level.point] = identity(n);
    std::vector<int> queue{level.point};
    for (std::size_t qi = 0; qi < queue.size(); ++qi) {
      const int   b  = queue[qi];
      const Perm& ub = level.transversal[b];
      for (std::size_t gi = 0; gi < level.gens.size(); ++gi) {
        const auto& s = level.gens[gi];
        point();
        const int g = s[b];
        if (level.transversal.find(g) == level.transversal.end()) {
          level.transversal[g] = compose(s, ub); // maps point -> g
          queue.push_back(g);
          if (capture) {
            admit(1 + static_cast<std::size_t>(n));
            out.discoveries.push_back(
                {level_index, gi, level.point, b, g, level.transversal.at(g)});
          }
        }
      }
    }
  }

  // Strip g through levels [start, end): returns residue and the level index where
  // it fell out (== chain size if it stripped through every level).
  /// @brief Sift one actual permutation through the current chain and retain its residue.
/// @param chain Borrowed completed current transversals.
/// @param g Owned current candidate.
/// @param start First stabilizer level.
/// @return Actual residue and first missing level, or chain size.
std::pair<Perm, std::size_t> strip(const std::vector<Level>& chain, Perm g, std::size_t start) {
    ++out.sifts;
    Perm original;
    if (capture) {
      admit(1 + 2 * g.size());
      original = g;
    }
    for (std::size_t i = start; i < chain.size(); ++i) {
      point();
      const int b  = g[chain[i].point];
      auto      it = chain[i].transversal.find(b);
      if (it == chain[i].transversal.end()) {
        if (capture)
          out.sift_steps.push_back({start, i, std::move(original), g});
        return {g, i};
      }
      g = compose(inverse(it->second), g); // now fixes chain[i].point
    }
    if (capture)
      out.sift_steps.push_back({start, chain.size(), std::move(original), g});
    return {g, chain.size()};
  }

  /// @brief Locate the actual first nonfixed point.
/// @param g Borrowed validated permutation.
/// @return First moved point, or minus one for identity.
int first_moved(const Perm& g) {
    for (std::size_t i = 0; i < g.size(); ++i) {
      point();
      if (g[i] != static_cast<int>(i))
        return static_cast<int>(i);
    }
    return -1;
  }

  /// @brief Execute the original stabilizer closure once with deterministic sorted snapshots.
  /// @param n Validated degree.
  /// @param generators Borrowed validated permutations.
  /// @return Owned actual chain, work and histories.
  SchreierSimsRun run(int n, const std::vector<Perm>& generators) {
    std::vector<Level> chain;

    // Seed the chain: sift each generator, adding base points / generators.
    /// @brief Sift and insert an actual nonidentity input residue.
    /// @param g Borrowed validated input permutation.
    auto add_perm = [&](const Perm& g) {
      auto [residue, level] = strip(chain, g, 0);
      if (is_identity(residue))
        return;
      if (level == chain.size()) {
        Level lv;
        lv.point = first_moved(residue);
        chain.push_back(lv);
      }
      for (std::size_t j = 0; j <= level && j < chain.size(); ++j) {
        chain[j].gens.push_back(residue);
        build_transversal(chain[j], n, j);
      }
    };
    for (const auto& g : generators)
      if (!is_identity(g))
        add_perm(g);

    // Main loop: verify every Schreier generator strips to the identity; if one
    // does not, its residue is a new strong generator -- add it and restart from
    // that level.
    std::size_t i = chain.empty() ? 0 : chain.size();
    while (i-- > 0) {
      bool restart = false;
      // snapshot orbit points (transversal may change on restart)
      std::vector<int> orbit;
      orbit.reserve(chain[i].transversal.size());
      for (const auto& kv : chain[i].transversal)
        orbit.push_back(kv.first);

      std::sort(orbit.begin(), orbit.end());
      const auto generators_snapshot = chain[i].gens;
      for (int beta : orbit) {
        const Perm ub = chain[i].transversal[beta];
        for (const Perm& s : generators_snapshot) {
          // Schreier generator: u_{s[beta]}^{-1} . s . u_beta
          const Perm sub    = compose(s, ub);
          const int  target = sub[chain[i].point]; // = s[beta]
          ++out.schreier_generators;
          const Perm schreier   = compose(inverse(chain[i].transversal[target]), sub);
          auto [residue, level] = strip(chain, schreier, i + 1);
          if (!is_identity(residue)) {
            if (level == chain.size()) {
              Level lv;
              lv.point = first_moved(residue);
              chain.push_back(lv);
            }
            for (std::size_t j = i + 1; j <= level && j < chain.size(); ++j) {
              chain[j].gens.push_back(residue);
              build_transversal(chain[j], n, j);
            }
            i       = level; // restart from the level that changed
            restart = true;
            break;
          }
        }
        if (restart)
          break;
      }
      if (restart)
        ++i; // counteract the loop's i-- so we redo this level
    }

    BSGS result;
    result.n     = n;
    result.order = 1;
    for (const auto& lv : chain) {
      result.base.push_back(lv.point);
      result.transversal.push_back(lv.transversal);
      const auto count = static_cast<unsigned long long>(lv.transversal.size());
      if (count && result.order > std::numeric_limits<unsigned long long>::max() / count)
        throw std::overflow_error("Schreier-Sims: order overflow");
      result.order *= count;
    }
    out.group = std::move(result);
    return std::move(out);
  }
};
/// @brief Validate all input permutations before indexed worker execution.
/// @param n Degree.
/// @param generators Borrowed generators.
/// @param o Input/work/history policies.
/// @param capture Retain actual events.
/// @return Owned actual run.
SchreierSimsRun produce(
    int                                  n,
    const std::vector<std::vector<int>>& generators,
    const SchreierSimsOptions&           o,
    bool                                 capture) {
  if (n < 0)
    throw std::invalid_argument("Schreier-Sims: negative degree");
  if ((o.max_degree && static_cast<std::size_t>(n) > o.max_degree) ||
      (o.max_generators && generators.size() > o.max_generators))
    throw std::length_error("Schreier-Sims: input policy");
  for (const auto& p : generators)
    if (!valid(p, n))
      throw std::invalid_argument("Schreier-Sims: generator is not a bijection");
  Worker w{o, capture};
  return w.run(n, generators);
}
} // namespace
SchreierSimsRun
schreier_sims_run(int n, const std::vector<std::vector<int>>& g, const SchreierSimsOptions& o) {
  return produce(n, g, o, false);
}
SchreierSimsRun
schreier_sims_trace(int n, const std::vector<std::vector<int>>& g, const SchreierSimsOptions& o) {
  return produce(n, g, o, true);
}
BSGS schreier_sims(int n, const std::vector<std::vector<int>>& g) {
  return schreier_sims_run(n, g).group;
}
std::vector<std::vector<int>> symmetric_group_generators(int n) {
  if (n < 2 || n > 64)
    throw std::invalid_argument("symmetric group: degree [2,64]");
  std::vector<int> swap(n), cycle(n);
  std::iota(swap.begin(), swap.end(), 0);
  std::swap(swap[0], swap[1]);
  for (int i = 0; i < n; ++i)
    cycle[i] = (i + 1) % n;
  return {swap, cycle};
}
bool bsgs_contains(const BSGS& g, const std::vector<int>& p) {
  if (g.n < 0 || g.base.size() != g.transversal.size() ||
      g.base.size() > static_cast<std::size_t>(g.n))
    throw std::invalid_argument("BSGS: shape");
  std::vector<char>  seen(static_cast<std::size_t>(g.n));
  unsigned long long order = 1;
  for (std::size_t i = 0; i < g.base.size(); ++i) {
    int base = g.base[i];
    if (base < 0 || base >= g.n || seen[base] || g.transversal[i].empty())
      throw std::invalid_argument("BSGS: base/orbit");
    seen[base]      = 1;
    const auto size = g.transversal[i].size();
    if (size > static_cast<std::size_t>(g.n) ||
        order > std::numeric_limits<unsigned long long>::max() / size)
      throw std::invalid_argument("BSGS: order capacity");
    order *= size;
    if (!g.transversal[i].contains(base))
      throw std::invalid_argument("BSGS: missing base representative");
    for (const auto& [key, rep] : g.transversal[i]) {
      if (key < 0 || key >= g.n || !valid(rep, g.n) || rep[base] != key)
        throw std::invalid_argument("BSGS: representative");
      for (std::size_t j = 0; j < i; ++j)
        if (rep[g.base[j]] != g.base[j])
          throw std::invalid_argument("BSGS: stabilizer");
    }
  }
  if (order != g.order)
    throw std::invalid_argument("BSGS: order metadata");
  if (!valid(p, g.n))
    return false;
  auto h = p;
  for (std::size_t i = 0; i < g.base.size(); ++i) {
    const auto it = g.transversal[i].find(h[g.base[i]]);
    if (it == g.transversal[i].end())
      return false;
    std::vector<int> inv(p.size()), next(p.size());
    for (std::size_t j = 0; j < p.size(); ++j)
      inv[it->second[j]] = static_cast<int>(j);
    for (std::size_t j = 0; j < p.size(); ++j)
      next[j] = inv[h[j]];
    h = std::move(next);
  }
  for (std::size_t j = 0; j < h.size(); ++j)
    if (h[j] != static_cast<int>(j))
      return false;
  return true;
}
} // namespace datamunge::algorithms
