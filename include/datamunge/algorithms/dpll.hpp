#pragma once

#include <vector>

namespace datamunge::algorithms {

/// @brief The result of a SAT solve: whether the formula is @ref satisfiable and, if so, a satisfying
///        @ref assignment. @c assignment has size @c num_vars+1; @c assignment[v] is @c +1 (true) or
///        @c -1 (false) for variable @c v in @c 1..num_vars (index 0 is unused).
struct SatResult {
    bool             satisfiable{false};
    std::vector<int> assignment; ///< assignment[v] in {+1,-1} for v=1..num_vars (valid iff satisfiable).
};

/// @brief The *DPLL algorithm* (Davis-Putnam-Logemann-Loveland, 1962): decides the satisfiability of a
///        Boolean formula in *conjunctive normal form* (an AND of OR-clauses). It is backtracking
///        search supercharged by two rules that shrink the problem without guessing:
///        *unit propagation* (a clause with a single unassigned literal forces that literal) and
///        *pure-literal elimination* (a variable that appears with only one polarity is set to satisfy
///        all its clauses). Only when neither applies does it *branch* on a variable, trying true then
///        false. DPLL is the backbone of every modern CDCL SAT solver.
///
/// @param num_vars the number of variables (named @c 1..num_vars).
/// @param clauses the CNF clauses; each clause is a list of literals, a literal being @c +v (variable
///        @c v true) or @c -v (variable @c v false).
/// @return the @ref SatResult; if satisfiable, its @ref SatResult::assignment satisfies every clause.
SatResult dpll_sat(int num_vars, const std::vector<std::vector<int>>& clauses);

} // namespace datamunge::algorithms
