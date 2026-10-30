/// @file arc_consistency.cpp
/// @brief The AC-3 worklist algorithm with work counters.

#include <datamunge/algorithms/arc_consistency.hpp>

#include <algorithm>
#include <cstddef>
#include <deque>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

Ac3Result ac3(std::vector<std::vector<int>>              domains,
              const std::vector<std::pair<int, int>>&    arcs,
              const std::function<bool(int, int, int, int)>& compatible) {
    Ac3Result result;
    long long total = 0;
    for (const auto& d : domains) total += static_cast<long long>(d.size());
    result.candidates.push_back(total);

    // incoming[v] = indices of the arcs (k, v) pointing into v.
    std::vector<std::vector<std::size_t>> incoming(domains.size());
    for (std::size_t a = 0; a < arcs.size(); ++a)
        incoming[static_cast<std::size_t>(arcs[a].second)].push_back(a);

    // Revise arc (i,j): drop every value a in domain[i] with no support b in domain[j].
    auto revise = [&](int i, int j) {
        auto&            di = domains[static_cast<std::size_t>(i)];
        const auto&      dj = domains[static_cast<std::size_t>(j)];
        std::vector<int> kept;
        kept.reserve(di.size());
        for (int a : di) {
            bool supported = false;
            for (int b : dj) {
                ++result.checks;
                if (compatible(i, a, j, b)) { supported = true; break; }
            }
            if (supported) kept.push_back(a);
        }
        const auto removed = static_cast<long long>(di.size() - kept.size());
        di = std::move(kept);
        return removed;
    };

    std::deque<std::size_t> work;
    std::vector<char>       queued(arcs.size(), 1);
    for (std::size_t a = 0; a < arcs.size(); ++a) work.push_back(a);
    while (!work.empty()) {
        const std::size_t a = work.front();
        work.pop_front();
        queued[a] = 0;
        const auto [i, j] = arcs[a];
        ++result.revisions;
        const long long removed = revise(i, j);
        if (removed == 0) continue;
        result.removals += removed;
        total -= removed;
        result.candidates.push_back(total);
        if (domains[static_cast<std::size_t>(i)].empty()) {
            result.consistent = false;
            break;
        }
        // A value left domain[i]; any arc (k,i) may have lost support -> re-enqueue it.
        for (std::size_t b : incoming[static_cast<std::size_t>(i)])
            if (!queued[b] && arcs[b].first != j) {
                queued[b] = 1;
                work.push_back(b);
            }
    }
    result.domains = std::move(domains);
    return result;
}

} // namespace datamunge::algorithms
