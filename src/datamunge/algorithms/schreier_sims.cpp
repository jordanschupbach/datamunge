#include <datamunge/algorithms/schreier_sims.hpp>

#include <unordered_map>
#include <vector>

namespace datamunge::algorithms {

namespace {

using Perm = std::vector<int>;

Perm identity(int n) {
    Perm p(n);
    for (int i = 0; i < n; ++i) p[i] = i;
    return p;
}

// (a . b)[i] = a[b[i]]  -- apply b, then a.
Perm compose(const Perm& a, const Perm& b) {
    Perm r(b.size());
    for (std::size_t i = 0; i < b.size(); ++i) r[i] = a[b[i]];
    return r;
}

Perm inverse(const Perm& a) {
    Perm r(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) r[a[i]] = static_cast<int>(i);
    return r;
}

bool is_identity(const Perm& a) {
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i] != static_cast<int>(i)) return false;
    return true;
}

// Stabilizer chain level.
struct Level {
    int                                point;       // base point beta_i
    std::vector<Perm>                  gens;        // generators fixing base[0..i-1]
    std::unordered_map<int, Perm>      transversal; // orbit point -> u mapping point -> that orbit point
};

// (Re)build the orbit of level.point under level.gens, with a transversal.
void build_transversal(Level& level, int n) {
    level.transversal.clear();
    level.transversal[level.point] = identity(n);
    std::vector<int> queue{level.point};
    for (std::size_t qi = 0; qi < queue.size(); ++qi) {
        const int         b = queue[qi];
        const Perm&       ub = level.transversal[b];
        for (const Perm& s : level.gens) {
            const int g = s[b];
            if (level.transversal.find(g) == level.transversal.end()) {
                level.transversal[g] = compose(s, ub); // maps point -> g
                queue.push_back(g);
            }
        }
    }
}

// Strip g through levels [start, end): returns residue and the level index where
// it fell out (== chain size if it stripped through every level).
std::pair<Perm, std::size_t> strip(const std::vector<Level>& chain, Perm g, std::size_t start) {
    for (std::size_t i = start; i < chain.size(); ++i) {
        const int b  = g[chain[i].point];
        auto      it = chain[i].transversal.find(b);
        if (it == chain[i].transversal.end()) return {g, i};
        g = compose(inverse(it->second), g); // now fixes chain[i].point
    }
    return {g, chain.size()};
}

int first_moved(const Perm& g) {
    for (std::size_t i = 0; i < g.size(); ++i)
        if (g[i] != static_cast<int>(i)) return static_cast<int>(i);
    return -1;
}

} // namespace

BSGS schreier_sims(int n, const std::vector<std::vector<int>>& generators) {
    std::vector<Level> chain;

    // Seed the chain: sift each generator, adding base points / generators.
    auto add_perm = [&](const Perm& g) {
        auto [residue, level] = strip(chain, g, 0);
        if (is_identity(residue)) return;
        if (level == chain.size()) {
            Level lv;
            lv.point = first_moved(residue);
            chain.push_back(lv);
        }
        for (std::size_t j = 0; j <= level && j < chain.size(); ++j) {
            chain[j].gens.push_back(residue);
            build_transversal(chain[j], n);
        }
    };
    for (const auto& g : generators)
        if (!is_identity(g)) add_perm(g);

    // Main loop: verify every Schreier generator strips to the identity; if one
    // does not, its residue is a new strong generator -- add it and restart from
    // that level.
    std::size_t i = chain.empty() ? 0 : chain.size();
    while (i-- > 0) {
        bool restart = false;
        // snapshot orbit points (transversal may change on restart)
        std::vector<int> orbit;
        orbit.reserve(chain[i].transversal.size());
        for (const auto& kv : chain[i].transversal) orbit.push_back(kv.first);

        for (int beta : orbit) {
            const Perm ub = chain[i].transversal[beta];
            for (const Perm& s : chain[i].gens) {
                // Schreier generator: u_{s[beta]}^{-1} . s . u_beta
                const Perm sub    = compose(s, ub);
                const int  target = sub[chain[i].point]; // = s[beta]
                const Perm schreier = compose(inverse(chain[i].transversal[target]), sub);
                auto [residue, level] = strip(chain, schreier, i + 1);
                if (!is_identity(residue)) {
                    if (level == chain.size()) {
                        Level lv;
                        lv.point = first_moved(residue);
                        chain.push_back(lv);
                    }
                    for (std::size_t j = i + 1; j <= level && j < chain.size(); ++j) {
                        chain[j].gens.push_back(residue);
                        build_transversal(chain[j], n);
                    }
                    i       = level; // restart from the level that changed
                    restart = true;
                    break;
                }
            }
            if (restart) break;
        }
        if (restart) ++i; // counteract the loop's i-- so we redo this level
    }

    BSGS result;
    result.n     = n;
    result.order = 1;
    for (const auto& lv : chain) {
        result.base.push_back(lv.point);
        result.transversal.push_back(lv.transversal);
        result.order *= static_cast<unsigned long long>(lv.transversal.size());
    }
    return result;
}

bool bsgs_contains(const BSGS& g, const std::vector<int>& perm) {
    if (static_cast<int>(perm.size()) != g.n) return false;
    Perm h = perm;
    for (std::size_t i = 0; i < g.base.size(); ++i) {
        const int b  = h[g.base[i]];
        auto      it = g.transversal[i].find(b);
        if (it == g.transversal[i].end()) return false;
        h = compose(inverse(it->second), h);
    }
    return is_identity(h);
}

} // namespace datamunge::algorithms
