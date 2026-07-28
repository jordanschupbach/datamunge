#pragma once

// Knuth-Bendix completion for string rewriting systems (monoid presentations).
// Given a set of equations over an alphabet, it tries to produce a confluent,
// terminating rewriting system in which every word has a unique normal form --
// solving the word problem: two words are equal in the monoid iff they reduce to
// the same normal form. Words are std::strings; the reduction order is shortlex
// (shorter first, then lexicographic).

#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

struct RewriteRule {
    std::string lhs;
    std::string rhs;
};

struct KnuthBendixResult {
    std::vector<RewriteRule> rules;
    bool                     complete{false}; // false if the rule/iteration cap was hit
};

// Complete `equations` (each a pair {left, right}) into a rewriting system.
KnuthBendixResult knuth_bendix(const std::vector<std::pair<std::string, std::string>>& equations,
                               int max_rules = 400);

// Reduce `word` to normal form under `rules` (repeatedly rewrite the first
// applicable left-hand side until none applies).
std::string kb_normal_form(const std::string& word, const std::vector<RewriteRule>& rules);

} // namespace datamunge::algorithms
