#include <datamunge/algorithms/todd_coxeter.hpp>

#include <algorithm>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

namespace {

// Coset enumeration state using the HLT (scan-and-fill) strategy with a
// union-find for coincidences.
struct Enumerator {
    int                           ncols;   // 2 * ngen (a symbol per generator and inverse)
    long long                     maxc;
    bool                          overflow{false};
    std::vector<std::vector<int>> table;   // table[c][sym] = coset or -1
    std::vector<int>              parent;  // union-find for coincidences
    std::vector<char>            alive;
    std::vector<std::vector<int>> relators_sym;
    std::vector<std::vector<int>> subgroup_sym;

    static int inv(int sym) { return sym ^ 1; }

    int rep(int c) {
        while (parent[c] != c) { parent[c] = parent[parent[c]]; c = parent[c]; }
        return c;
    }

    int new_coset() {
        const int c = static_cast<int>(table.size());
        table.emplace_back(ncols, -1);
        parent.push_back(c);
        alive.push_back(1);
        return c;
    }

    // Set table[a][x] = b and the inverse link.
    void set_edge(int a, int x, int b) {
        table[a][x]      = b;
        table[b][inv(x)] = a;
    }

    // Merge cosets b into a (a < b), cascading further coincidences.
    void coincidence(int a, int b, std::vector<std::pair<int, int>>& queue) {
        a = rep(a);
        b = rep(b);
        if (a == b) return;
        if (a > b) std::swap(a, b);
        parent[b] = a;
        alive[b]  = 0;
        for (int x = 0; x < ncols; ++x) {
            const int tb = table[b][x];
            if (tb < 0) continue;
            const int tbr = rep(tb);
            // remove back-link from tb to b (will be re-added to a)
            const int ta = table[a][x] < 0 ? -1 : rep(table[a][x]);
            if (ta < 0) {
                set_edge(a, x, tbr);
            } else if (ta != tbr) {
                queue.emplace_back(ta, tbr);
            }
            // ensure inverse consistency for tbr
            if (table[tbr][inv(x)] < 0 || rep(table[tbr][inv(x)]) == b) table[tbr][inv(x)] = a;
        }
    }

    void process_coincidence(int a, int b) {
        std::vector<std::pair<int, int>> queue{{a, b}};
        while (!queue.empty()) {
            auto [x, y] = queue.back();
            queue.pop_back();
            coincidence(x, y, queue);
        }
    }

    // Scan `word` cyclically starting/ending at coset `c`, filling forward and
    // backward, defining new cosets to make progress. Registers coincidences.
    void scan(int start, const std::vector<int>& word) {
        if (word.empty()) return;
        int f = rep(start);
        int b = rep(start);
        int i = 0;
        int j = static_cast<int>(word.size()) - 1;
        while (true) {
            // forward as far as defined
            while (i <= j && table[f][word[i]] >= 0) { f = rep(table[f][word[i]]); ++i; }
            if (i > j) { // whole word scanned forward; must return to start
                if (f != rep(start)) process_coincidence(f, rep(start));
                return;
            }
            // backward as far as defined
            while (j >= i && table[b][inv(word[j])] >= 0) { b = rep(table[b][inv(word[j])]); --j; }
            if (j < i) { // forward and backward met: coincidence
                process_coincidence(f, b);
                return;
            }
            if (i == j) { // exactly one gap: deduce the edge
                set_edge(f, word[i], b);
                return;
            }
            // gap wider than one: define a new coset and continue
            if (static_cast<long long>(table.size()) >= maxc) { overflow = true; return; }
            const int nc = new_coset();
            set_edge(f, word[i], nc);
        }
    }
};

// Convert a signed word (+（i+1)/-(i+1)) to internal symbols (2i / 2i+1).
std::vector<int> to_symbols(const std::vector<int>& word) {
    std::vector<int> s;
    s.reserve(word.size());
    for (int t : word) {
        const int g = (t > 0 ? t : -t) - 1;
        s.push_back(t > 0 ? 2 * g : 2 * g + 1);
    }
    return s;
}

} // namespace

long long todd_coxeter_index(int ngen, const std::vector<std::vector<int>>& relators,
                             const std::vector<std::vector<int>>& subgroup_gens, long long max_cosets) {
    Enumerator e;
    e.ncols = 2 * ngen;
    e.maxc  = max_cosets;
    for (const auto& r : relators) e.relators_sym.push_back(to_symbols(r));
    for (const auto& s : subgroup_gens) e.subgroup_sym.push_back(to_symbols(s));

    e.new_coset(); // coset 0 = the subgroup H

    // Subgroup generators fix coset 0.
    for (const auto& w : e.subgroup_sym) e.scan(0, w);
    if (e.overflow) return -1;

    // Main HLT loop: for each live coset in order, scan every relator, then fill
    // any still-undefined generator image by defining a new coset.
    int c = 0;
    while (c < static_cast<int>(e.table.size())) {
        if (e.rep(c) != c) { ++c; continue; } // dead / merged
        for (const auto& r : e.relators_sym) e.scan(c, r);
        if (e.overflow) return -1;
        if (e.rep(c) != c) { ++c; continue; }
        for (int x = 0; x < e.ncols; ++x) {
            if (e.rep(c) != c) break;
            if (e.table[c][x] < 0) {
                if (static_cast<long long>(e.table.size()) >= max_cosets) return -1;
                const int nc = e.new_coset();
                e.set_edge(c, x, nc);
                for (const auto& r : e.relators_sym) e.scan(c, r);
                if (e.overflow) return -1;
            }
        }
        ++c;
    }

    // Count live cosets.
    long long live = 0;
    for (int i = 0; i < static_cast<int>(e.table.size()); ++i)
        if (e.rep(i) == i) ++live;
    return live;
}

} // namespace datamunge::algorithms
