#include <datamunge/algorithms/sequence_alignment.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

namespace {

// Score of aligning two characters under the scheme.
inline int pair_score(char x, char y, const AlignmentScoring& s) {
    return x == y ? s.match : s.mismatch;
}

// The last row of the Needleman-Wunsch score matrix for aligning `a` against `b`, in O(|b|) space.
// This is the classic "NWScore" helper used by Hirschberg: element j is the best global-alignment
// score of the whole of `a` against the length-j prefix of `b`.
std::vector<int> nw_last_row(const std::string& a, const std::string& b, const AlignmentScoring& s) {
    const std::size_t m = b.size();
    std::vector<int>  prev(m + 1), curr(m + 1);
    for (std::size_t j = 0; j <= m; ++j) prev[j] = static_cast<int>(j) * s.gap;
    for (std::size_t i = 1; i <= a.size(); ++i) {
        curr[0] = static_cast<int>(i) * s.gap;
        for (std::size_t j = 1; j <= m; ++j) {
            const int diag = prev[j - 1] + pair_score(a[i - 1], b[j - 1], s);
            const int del  = prev[j] + s.gap;
            const int ins  = curr[j - 1] + s.gap;
            curr[j]        = std::max({diag, del, ins});
        }
        std::swap(prev, curr);
    }
    return prev;
}

} // namespace

Alignment needleman_wunsch(const std::string& a, const std::string& b, AlignmentScoring scoring) {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    std::vector<std::vector<int>> H(n + 1, std::vector<int>(m + 1, 0));
    for (std::size_t i = 0; i <= n; ++i) H[i][0] = static_cast<int>(i) * scoring.gap;
    for (std::size_t j = 0; j <= m; ++j) H[0][j] = static_cast<int>(j) * scoring.gap;
    for (std::size_t i = 1; i <= n; ++i)
        for (std::size_t j = 1; j <= m; ++j) {
            const int diag = H[i - 1][j - 1] + pair_score(a[i - 1], b[j - 1], scoring);
            const int del  = H[i - 1][j] + scoring.gap;
            const int ins  = H[i][j - 1] + scoring.gap;
            H[i][j]        = std::max({diag, del, ins});
        }

    // Traceback from the bottom-right corner, preferring diagonal, then up (gap in b), then left.
    Alignment   out;
    out.score = H[n][m];
    std::size_t i = n, j = m;
    while (i > 0 || j > 0) {
        if (i > 0 && j > 0 &&
            H[i][j] == H[i - 1][j - 1] + pair_score(a[i - 1], b[j - 1], scoring)) {
            out.a_aligned.push_back(a[i - 1]);
            out.b_aligned.push_back(b[j - 1]);
            --i;
            --j;
        } else if (i > 0 && H[i][j] == H[i - 1][j] + scoring.gap) {
            out.a_aligned.push_back(a[i - 1]);
            out.b_aligned.push_back('-');
            --i;
        } else {
            out.a_aligned.push_back('-');
            out.b_aligned.push_back(b[j - 1]);
            --j;
        }
    }
    std::reverse(out.a_aligned.begin(), out.a_aligned.end());
    std::reverse(out.b_aligned.begin(), out.b_aligned.end());
    return out;
}

Alignment smith_waterman(const std::string& a, const std::string& b, AlignmentScoring scoring) {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    std::vector<std::vector<int>> H(n + 1, std::vector<int>(m + 1, 0)); // borders stay 0
    int         best = 0;
    std::size_t bi = 0, bj = 0;
    for (std::size_t i = 1; i <= n; ++i)
        for (std::size_t j = 1; j <= m; ++j) {
            const int diag = H[i - 1][j - 1] + pair_score(a[i - 1], b[j - 1], scoring);
            const int del  = H[i - 1][j] + scoring.gap;
            const int ins  = H[i][j - 1] + scoring.gap;
            H[i][j]        = std::max({0, diag, del, ins});
            if (H[i][j] > best) {
                best = H[i][j];
                bi   = i;
                bj   = j;
            }
        }

    Alignment out;
    out.score = best;
    // Traceback from the best cell until reaching a zero cell.
    std::size_t i = bi, j = bj;
    while (i > 0 && j > 0 && H[i][j] > 0) {
        if (H[i][j] == H[i - 1][j - 1] + pair_score(a[i - 1], b[j - 1], scoring)) {
            out.a_aligned.push_back(a[i - 1]);
            out.b_aligned.push_back(b[j - 1]);
            --i;
            --j;
        } else if (H[i][j] == H[i - 1][j] + scoring.gap) {
            out.a_aligned.push_back(a[i - 1]);
            out.b_aligned.push_back('-');
            --i;
        } else {
            out.a_aligned.push_back('-');
            out.b_aligned.push_back(b[j - 1]);
            --j;
        }
    }
    std::reverse(out.a_aligned.begin(), out.a_aligned.end());
    std::reverse(out.b_aligned.begin(), out.b_aligned.end());
    return out;
}

Alignment hirschberg(const std::string& a, const std::string& b, AlignmentScoring scoring) {
    const std::size_t n = a.size();
    const std::size_t m = b.size();

    Alignment out;
    // Base cases: an empty string aligns entirely against gaps; a single row is cheap by full NW.
    if (n == 0) {
        out.a_aligned = std::string(m, '-');
        out.b_aligned = b;
        out.score     = static_cast<int>(m) * scoring.gap;
        return out;
    }
    if (m == 0) {
        out.a_aligned = a;
        out.b_aligned = std::string(n, '-');
        out.score     = static_cast<int>(n) * scoring.gap;
        return out;
    }
    if (n == 1 || m == 1) return needleman_wunsch(a, b, scoring);

    // Divide: split `a` at its midpoint and find the column of `b` where an optimal alignment
    // crosses, by combining a forward score over a[0:mid] and a reverse score over a[mid:].
    const std::size_t xmid = n / 2;
    const std::string a_left  = a.substr(0, xmid);
    const std::string a_right = a.substr(xmid);

    const std::vector<int> score_l = nw_last_row(a_left, b, scoring);
    std::string            a_right_rev(a_right.rbegin(), a_right.rend());
    std::string            b_rev(b.rbegin(), b.rend());
    std::vector<int>       score_r = nw_last_row(a_right_rev, b_rev, scoring);
    std::reverse(score_r.begin(), score_r.end()); // now score_r[j] scores b[j:] against a_right

    std::size_t ymid = 0;
    int         best = std::numeric_limits<int>::min();
    for (std::size_t j = 0; j <= m; ++j) {
        const int total = score_l[j] + score_r[j];
        if (total > best) {
            best = total;
            ymid = j;
        }
    }

    // Conquer: recurse on the two halves and concatenate.
    const Alignment left  = hirschberg(a_left, b.substr(0, ymid), scoring);
    const Alignment right = hirschberg(a_right, b.substr(ymid), scoring);
    out.a_aligned = left.a_aligned + right.a_aligned;
    out.b_aligned = left.b_aligned + right.b_aligned;
    out.score     = left.score + right.score;
    return out;
}

} // namespace datamunge::algorithms
