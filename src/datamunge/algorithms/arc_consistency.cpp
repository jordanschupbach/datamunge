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
    // Revise arc (i,j): drop every value a in domain[i] with no support b in domain[j].
    auto revise = [&](int i, int j) {
        bool             removed = false;
        std::vector<int> kept;
        for (int a : domains[static_cast<std::size_t>(i)]) {
            bool supported = false;
            for (int b : domains[static_cast<std::size_t>(j)])
                if (compatible(i, a, j, b)) { supported = true; break; }
            if (supported)
                kept.push_back(a);
            else
                removed = true;
        }
        domains[static_cast<std::size_t>(i)] = std::move(kept);
        return removed;
    };

    std::deque<std::pair<int, int>> work(arcs.begin(), arcs.end());
    while (!work.empty()) {
        const auto [i, j] = work.front();
        work.pop_front();
        if (revise(i, j)) {
            if (domains[static_cast<std::size_t>(i)].empty()) return Ac3Result{std::move(domains), false};
            // A value left domain[i]; any arc (k,i) may have lost support -> re-enqueue it.
            for (const auto& arc : arcs)
                if (arc.second == i && arc.first != j) work.push_back(arc);
        }
    }
    return Ac3Result{std::move(domains), true};
}

} // namespace datamunge::algorithms
