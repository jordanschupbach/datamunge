#include <datamunge/algorithms/knuth_bendix.hpp>

#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

namespace {

// Shortlex: a > b if a is longer, or same length and lexicographically greater.
bool shortlex_gt(const std::string& a, const std::string& b) {
    if (a.size() != b.size()) return a.size() > b.size();
    return a > b;
}

// Rewrite the first occurrence of any rule's lhs; returns true if a rewrite
// happened. Rules with empty lhs are skipped (never valid).
bool rewrite_once(std::string& w, const std::vector<RewriteRule>& rules) {
    for (const auto& r : rules) {
        if (r.lhs.empty()) continue;
        const auto pos = w.find(r.lhs);
        if (pos != std::string::npos) {
            w.replace(pos, r.lhs.size(), r.rhs);
            return true;
        }
    }
    return false;
}

std::string normal_form(std::string w, const std::vector<RewriteRule>& rules) {
    while (rewrite_once(w, rules)) {}
    return w;
}

bool rule_exists(const std::vector<RewriteRule>& rules, const std::string& l, const std::string& r) {
    for (const auto& x : rules)
        if (x.lhs == l && x.rhs == r) return true;
    return false;
}

// Add the oriented rule for the equation u = v (larger side -> smaller). Returns
// true if a new nontrivial rule was added.
bool add_equation(std::vector<RewriteRule>& rules, const std::string& u, const std::string& v) {
    if (u == v) return false;
    const std::string& big   = shortlex_gt(u, v) ? u : v;
    const std::string& small = shortlex_gt(u, v) ? v : u;
    if (rule_exists(rules, big, small)) return false;
    rules.push_back({big, small});
    return true;
}

// Emit the two reducts of every overlap between rules r1 and r2 into `pairs`.
void critical_pairs(const RewriteRule& r1, const RewriteRule& r2,
                    std::vector<std::pair<std::string, std::string>>& pairs) {
    const std::string& L1 = r1.lhs;
    const std::string& L2 = r2.lhs;

    // Overlap: a proper suffix of L1 equals a proper prefix of L2.
    const std::size_t maxk = L1.size() < L2.size() ? L1.size() : L2.size();
    for (std::size_t k = 1; k <= maxk; ++k) {
        if (k == L1.size() && k == L2.size()) continue; // identical, no genuine overlap
        if (L1.compare(L1.size() - k, k, L2, 0, k) == 0) {
            // overlap word w = L1 + tail(L2)
            const std::string u = r1.rhs + L2.substr(k);          // rewrite L1 prefix
            const std::string v = L1.substr(0, L1.size() - k) + r2.rhs; // rewrite L2 suffix
            pairs.emplace_back(u, v);
        }
    }

    // Containment: L2 occurs inside L1 as L1 = a L2 b.
    if (L2.size() <= L1.size()) {
        for (std::size_t p = 0; p + L2.size() <= L1.size(); ++p) {
            if (L1.compare(p, L2.size(), L2) == 0) {
                const std::string u = r1.rhs;
                const std::string v = L1.substr(0, p) + r2.rhs + L1.substr(p + L2.size());
                pairs.emplace_back(u, v);
            }
        }
    }
}

// Remove rules whose lhs is rewritable by another rule (redundant), and reduce
// every rhs to normal form under the others -- keeps the system canonical.
void interreduce(std::vector<RewriteRule>& rules) {
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t i = 0; i < rules.size();) {
            std::vector<RewriteRule> others;
            for (std::size_t j = 0; j < rules.size(); ++j)
                if (j != i) others.push_back(rules[j]);
            const std::string lhsNF = normal_form(rules[i].lhs, others);
            if (lhsNF != rules[i].lhs) {
                // lhs reducible by others: this rule is redundant; re-add as an equation.
                const std::string rhsNF = normal_form(rules[i].rhs, others);
                rules.erase(rules.begin() + i);
                if (add_equation(rules, lhsNF, rhsNF)) changed = true;
                changed = true;
                continue;
            }
            const std::string rhsNF = normal_form(rules[i].rhs, others);
            if (rhsNF != rules[i].rhs) { rules[i].rhs = rhsNF; changed = true; }
            ++i;
        }
    }
}

} // namespace

KnuthBendixResult knuth_bendix(const std::vector<std::pair<std::string, std::string>>& equations, int max_rules) {
    std::vector<RewriteRule> rules;
    for (const auto& [l, r] : equations) add_equation(rules, l, r);

    int  guard    = 0;
    bool complete = false;
    while (guard++ < 2000) {
        interreduce(rules);
        bool added = false;
        // Snapshot to iterate over a stable set this round.
        const std::vector<RewriteRule> snap = rules;
        std::vector<std::pair<std::string, std::string>> pairs;
        for (const auto& a : snap)
            for (const auto& b : snap) critical_pairs(a, b, pairs);

        for (const auto& [u, v] : pairs) {
            const std::string nu = normal_form(u, rules);
            const std::string nv = normal_form(v, rules);
            if (nu != nv) {
                if (add_equation(rules, nu, nv)) added = true;
                if (static_cast<int>(rules.size()) > max_rules) return {rules, false};
            }
        }
        if (!added) { complete = true; break; }
    }
    interreduce(rules);
    return {rules, complete};
}

std::string kb_normal_form(const std::string& word, const std::vector<RewriteRule>& rules) {
    return normal_form(word, rules);
}

} // namespace datamunge::algorithms
