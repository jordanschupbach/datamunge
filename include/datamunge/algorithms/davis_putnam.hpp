#pragma once

// The Davis-Putnam procedure (1960): the original decision method for the
// satisfiability of a propositional formula in CNF. Unlike its famous successor
// DPLL -- which decides satisfiability by *splitting* on a variable and
// backtracking -- the original DP *eliminates* variables one at a time by
// resolution, shrinking the variable set until the formula is trivially
// satisfiable (no clauses left) or unsatisfiable (the empty clause appears). It
// interleaves three rules:
//
//   * unit propagation   -- a one-literal clause forces that literal;
//   * pure-literal rule   -- a literal whose negation never occurs can be set true;
//   * resolution          -- to eliminate variable x, replace all clauses on x by
//                            every non-tautological resolvent of an x-clause with a
//                            (not-x)-clause.
//
// Resolution can blow the clause set up exponentially (its weakness versus DPLL),
// but as a decision procedure it is complete and elegantly variable-driven.

#include <algorithm>
#include <cstdlib>
#include <set>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

using DpClause = std::set<int>;

inline bool dp_tautology(const DpClause& c) {
    for (int lit : c)
        if (c.count(-lit)) return true;
    return false;
}

} // namespace detail

// Returns true iff the CNF (variables 1..num_vars; a literal is +v or -v) is
// satisfiable, deciding it by the Davis-Putnam resolution procedure.
inline bool davis_putnam(int num_vars, const std::vector<std::vector<int>>& clauses_in) {
    using detail::DpClause;
    std::vector<DpClause> clauses;
    for (const auto& c : clauses_in) {
        DpClause s(c.begin(), c.end());
        if (!detail::dp_tautology(s)) clauses.push_back(std::move(s));
    }

    auto has_empty = [&] {
        for (const auto& c : clauses) if (c.empty()) return true;
        return false;
    };

    // Simplify by unit propagation then pure-literal elimination, to a fixpoint.
    auto simplify = [&]() -> void {
        bool changed = true;
        while (changed) {
            changed = false;
            // Unit propagation.
            int unit = 0;
            for (const auto& c : clauses)
                if (c.size() == 1) { unit = *c.begin(); break; }
            if (unit != 0) {
                std::vector<DpClause> next;
                for (auto& c : clauses) {
                    if (c.count(unit)) continue; // satisfied
                    if (c.count(-unit)) { DpClause d = c; d.erase(-unit); next.push_back(std::move(d)); }
                    else next.push_back(std::move(c));
                }
                clauses.swap(next);
                changed = true;
                continue;
            }
            // Pure-literal elimination.
            std::set<int> lits;
            for (const auto& c : clauses)
                for (int l : c) lits.insert(l);
            for (int l : lits)
                if (!lits.count(-l)) {
                    std::vector<DpClause> next;
                    for (auto& c : clauses)
                        if (!c.count(l)) next.push_back(std::move(c));
                    clauses.swap(next);
                    changed = true;
                    break;
                }
        }
    };

    // Each resolution step eliminates one variable, so at most num_vars steps.
    for (int step = 0; step <= num_vars + 1; ++step) {
        simplify();
        if (has_empty()) return false;
        if (clauses.empty()) return true;

        // Pick a variable that still occurs and eliminate it by resolution.
        int x = std::abs(*clauses.front().begin());
        std::vector<DpClause> withx, withnotx, rest;
        for (auto& c : clauses) {
            if (c.count(x)) withx.push_back(c);
            else if (c.count(-x)) withnotx.push_back(c);
            else rest.push_back(std::move(c));
        }
        for (const auto& a : withx)
            for (const auto& b : withnotx) {
                DpClause res = a;
                res.erase(x);
                for (int l : b) if (l != -x) res.insert(l);
                if (!detail::dp_tautology(res)) rest.push_back(std::move(res));
            }
        clauses.swap(rest);
        if (has_empty()) return false;
    }
    return !has_empty();
}

} // namespace datamunge::algorithms
