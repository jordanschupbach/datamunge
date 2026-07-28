#include <datamunge/algorithms/dpll.hpp>

#include <cstddef>
#include <cstdlib>
#include <vector>

namespace datamunge::algorithms {

namespace {

// assign[v] in {0 unassigned, +1 true, -1 false}, indexed 1..num_vars.
using Assign = std::vector<int>;

// Evaluate the clause set under a partial assignment.
//   returns  1 if all clauses satisfied, -1 if some clause is falsified (conflict), 0 otherwise.
int status(const std::vector<std::vector<int>>& clauses, const Assign& a) {
    bool all_sat = true;
    for (const auto& clause : clauses) {
        bool sat        = false;
        bool has_unassigned = false;
        for (int lit : clause) {
            const int v   = std::abs(lit);
            const int val = a[v];
            if (val == 0) {
                has_unassigned = true;
            } else if ((lit > 0 && val == 1) || (lit < 0 && val == -1)) {
                sat = true;
                break;
            }
        }
        if (sat) continue;
        if (!has_unassigned) return -1; // every literal false -> conflict
        all_sat = false;                // unsatisfied but still has room
    }
    return all_sat ? 1 : 0;
}

// Find a unit clause (exactly one unassigned literal, none satisfied): returns that literal, or 0.
int find_unit(const std::vector<std::vector<int>>& clauses, const Assign& a) {
    for (const auto& clause : clauses) {
        int  unassigned_lit = 0;
        int  unassigned_cnt = 0;
        bool sat            = false;
        for (int lit : clause) {
            const int v   = std::abs(lit);
            const int val = a[v];
            if ((lit > 0 && val == 1) || (lit < 0 && val == -1)) {
                sat = true;
                break;
            }
            if (val == 0) {
                ++unassigned_cnt;
                unassigned_lit = lit;
            }
        }
        if (!sat && unassigned_cnt == 1) return unassigned_lit;
    }
    return 0;
}

// Find a pure literal (a variable appearing with only one polarity among unsatisfied clauses).
int find_pure(const std::vector<std::vector<int>>& clauses, const Assign& a, int num_vars) {
    std::vector<int> pos(num_vars + 1, 0), neg(num_vars + 1, 0);
    for (const auto& clause : clauses) {
        bool sat = false;
        for (int lit : clause) {
            const int v = std::abs(lit);
            if ((lit > 0 && a[v] == 1) || (lit < 0 && a[v] == -1)) { sat = true; break; }
        }
        if (sat) continue;
        for (int lit : clause) {
            const int v = std::abs(lit);
            if (a[v] != 0) continue;
            if (lit > 0) ++pos[v]; else ++neg[v];
        }
    }
    for (int v = 1; v <= num_vars; ++v) {
        if (a[v] != 0) continue;
        if (pos[v] > 0 && neg[v] == 0) return v;
        if (neg[v] > 0 && pos[v] == 0) return -v;
    }
    return 0;
}

bool dpll(const std::vector<std::vector<int>>& clauses, Assign& a, int num_vars) {
    // Unit propagation and pure-literal elimination to fixpoint.
    for (;;) {
        const int st = status(clauses, a);
        if (st == 1) return true;  // satisfied
        if (st == -1) return false; // conflict

        if (const int u = find_unit(clauses, a)) {
            a[std::abs(u)] = u > 0 ? 1 : -1;
            continue;
        }
        if (const int p = find_pure(clauses, a, num_vars)) {
            a[std::abs(p)] = p > 0 ? 1 : -1;
            continue;
        }
        break;
    }

    // Branch on the first unassigned variable.
    int var = 0;
    for (int v = 1; v <= num_vars; ++v)
        if (a[v] == 0) { var = v; break; }
    if (var == 0) return status(clauses, a) == 1;

    for (int val : {1, -1}) {
        Assign next = a;
        next[var]   = val;
        if (dpll(clauses, next, num_vars)) {
            a = next;
            return true;
        }
    }
    return false;
}

} // namespace

SatResult dpll_sat(int num_vars, const std::vector<std::vector<int>>& clauses) {
    SatResult result;
    Assign    a(num_vars + 1, 0);
    if (dpll(clauses, a, num_vars)) {
        // Fill any still-unassigned variables (don't-cares) with true for a total assignment.
        for (int v = 1; v <= num_vars; ++v)
            if (a[v] == 0) a[v] = 1;
        result.satisfiable = true;
        result.assignment  = a;
    }
    return result;
}

} // namespace datamunge::algorithms
