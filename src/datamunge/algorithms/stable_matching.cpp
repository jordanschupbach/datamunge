#include <datamunge/algorithms/stable_matching.hpp>

#include <queue>
#include <stdexcept>

namespace datamunge::algorithms {

namespace {

// rank[a][b] = position of b in a's preference list (0 = most preferred). Validates that each
// list is a permutation of 0..n-1.
std::vector<std::vector<std::size_t>> rank_table(const std::vector<std::vector<std::size_t>>& prefs,
                                                 std::size_t n, const char* who) {
    std::vector<std::vector<std::size_t>> rank(n, std::vector<std::size_t>(n, kUnmatched));
    for (std::size_t a = 0; a < n; ++a) {
        if (prefs[a].size() != n)
            throw std::invalid_argument(std::string("gale_shapley: each ") + who + " must rank all n of the other side");
        for (std::size_t pos = 0; pos < n; ++pos) {
            const std::size_t b = prefs[a][pos];
            if (b >= n || rank[a][b] != kUnmatched)
                throw std::invalid_argument(std::string("gale_shapley: ") + who +
                                            " preferences must be a permutation of 0..n-1");
            rank[a][b] = pos;
        }
    }
    return rank;
}

} // namespace

StableMatchingResult gale_shapley(const std::vector<std::vector<std::size_t>>& proposer_preferences,
                                  const std::vector<std::vector<std::size_t>>& reviewer_preferences) {
    const std::size_t n = proposer_preferences.size();
    if (reviewer_preferences.size() != n)
        throw std::invalid_argument("gale_shapley: the two sides must have equal size");

    const auto reviewer_rank = rank_table(reviewer_preferences, n, "reviewer");
    (void)rank_table(proposer_preferences, n, "proposer"); // validate proposer lists too

    StableMatchingResult r;
    r.proposer_match.assign(n, kUnmatched);
    r.reviewer_match.assign(n, kUnmatched);
    std::vector<std::size_t> next_choice(n, 0); // index of proposer i's next reviewer to try

    std::queue<std::size_t> free_proposers;
    for (std::size_t i = 0; i < n; ++i) free_proposers.push(i);

    while (!free_proposers.empty()) {
        const std::size_t i = free_proposers.front();
        free_proposers.pop();

        // Propose to the next reviewer on i's list who has not yet rejected i.
        const std::size_t j = proposer_preferences[i][next_choice[i]++];
        ++r.proposals;

        const std::size_t held = r.reviewer_match[j];
        if (held == kUnmatched) { // reviewer free: tentatively accept
            r.reviewer_match[j] = i;
            r.proposer_match[i] = j;
        } else if (reviewer_rank[j][i] < reviewer_rank[j][held]) { // reviewer prefers i to its current
            r.proposer_match[held] = kUnmatched; // dump the old partner
            free_proposers.push(held);
            r.reviewer_match[j] = i;
            r.proposer_match[i] = j;
        } else { // reviewer rejects i; i remains free and will try its next choice
            free_proposers.push(i);
        }
    }
    return r;
}

bool is_stable(const StableMatchingResult& matching,
               const std::vector<std::vector<std::size_t>>& proposer_preferences,
               const std::vector<std::vector<std::size_t>>& reviewer_preferences) {
    const std::size_t n = proposer_preferences.size();
    const auto p_rank = rank_table(proposer_preferences, n, "proposer");
    const auto r_rank = rank_table(reviewer_preferences, n, "reviewer");

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t partner_i = matching.proposer_match[i];
        for (std::size_t j = 0; j < n; ++j) {
            // Does proposer i prefer reviewer j to its current partner (or is it unmatched)?
            const bool i_prefers_j = (partner_i == kUnmatched) || (p_rank[i][j] < p_rank[i][partner_i]);
            if (!i_prefers_j) continue;
            const std::size_t partner_j = matching.reviewer_match[j];
            // Does reviewer j prefer proposer i to its current partner (or is it unmatched)?
            const bool j_prefers_i = (partner_j == kUnmatched) || (r_rank[j][i] < r_rank[j][partner_j]);
            if (j_prefers_i) return false; // (i, j) is a blocking pair
        }
    }
    return true;
}

} // namespace datamunge::algorithms
