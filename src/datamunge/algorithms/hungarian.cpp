#include <datamunge/algorithms/hungarian.hpp>

#include <limits>
#include <stdexcept>

namespace datamunge::algorithms {

AssignmentResult hungarian(const std::vector<std::vector<double>>& cost) {
    const std::size_t n = cost.size();
    if (n == 0)
        throw std::invalid_argument("hungarian: cost matrix must be non-empty");
    for (const auto& row : cost)
        if (row.size() != n)
            throw std::invalid_argument("hungarian: cost matrix must be square (n x n)");

    constexpr double kInf = std::numeric_limits<double>::infinity();

    // Jonker-Volgenant-style primal-dual solve on 1-based dual arrays (index 0 is a virtual
    // "free row" sentinel). u[i]/v[j] are the row/column potentials; p[j] is the row currently
    // matched to column j; way[j] records the predecessor column on the augmenting path.
    std::vector<double> u(n + 1, 0.0);
    std::vector<double> v(n + 1, 0.0);
    std::vector<std::size_t> p(n + 1, 0);
    std::vector<std::size_t> way(n + 1, 0);

    for (std::size_t i = 1; i <= n; ++i) {
        p[0] = i;                     // column 0 is the entry point for augmenting from row i
        std::size_t j0 = 0;           // current column on the alternating path
        std::vector<double> minv(n + 1, kInf); // minv[j] = smallest slack to reach column j
        std::vector<char> used(n + 1, 0);      // columns already on the alternating tree

        // Grow the shortest augmenting path of tight edges, adjusting potentials by the minimum
        // slack whenever the equality subgraph offers no further move, until an unmatched column
        // (p[j0] == 0) is reached.
        do {
            used[j0] = 1;
            const std::size_t i0 = p[j0];
            double delta = kInf;
            std::size_t j1 = 0;
            for (std::size_t j = 1; j <= n; ++j) {
                if (used[j]) continue;
                const double cur = cost[i0 - 1][j - 1] - u[i0] - v[j]; // slack of edge (i0, j)
                if (cur < minv[j]) {
                    minv[j] = cur;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            // Re-price: tighten all tree columns by delta (keeping u_i + v_j <= c_ij feasible)
            // and reduce the outstanding slacks accordingly.
            for (std::size_t j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);

        // Flip the matching along the augmenting path back to the entry point.
        do {
            const std::size_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }

    AssignmentResult result;
    result.assignment.assign(n, 0);
    for (std::size_t j = 1; j <= n; ++j)
        result.assignment[p[j] - 1] = j - 1; // column j went to row p[j]

    double total = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        total += cost[i][result.assignment[i]];
    result.cost = total;
    return result;
}

} // namespace datamunge::algorithms
