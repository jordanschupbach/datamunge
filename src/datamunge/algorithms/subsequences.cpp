#include <datamunge/algorithms/subsequences.hpp>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

namespace {

// The LCS-length dynamic-programming table C, where C[i][j] is the LCS length of a[0..i) and b[0..j).
std::vector<std::vector<int>> lcs_table(const std::vector<int>& a, const std::vector<int>& b) {
    const std::size_t n = a.size();
    const std::size_t m = b.size();
    std::vector<std::vector<int>> c(n + 1, std::vector<int>(m + 1, 0));
    for (std::size_t i = 1; i <= n; ++i)
        for (std::size_t j = 1; j <= m; ++j)
            c[i][j] = (a[i - 1] == b[j - 1]) ? c[i - 1][j - 1] + 1
                                             : std::max(c[i - 1][j], c[i][j - 1]);
    return c;
}

} // namespace

std::vector<int> longest_common_subsequence(const std::vector<int>& a, const std::vector<int>& b) {
    const auto  c = lcs_table(a, b);
    std::vector<int> out;
    std::size_t i = a.size();
    std::size_t j = b.size();
    while (i > 0 && j > 0) {
        if (a[i - 1] == b[j - 1]) {
            out.push_back(a[i - 1]);
            --i;
            --j;
        } else if (c[i - 1][j] >= c[i][j - 1]) {
            --i;
        } else {
            --j;
        }
    }
    std::reverse(out.begin(), out.end());
    return out;
}

std::vector<int> longest_increasing_subsequence(const std::vector<int>& a) {
    const std::size_t n = a.size();
    std::vector<std::size_t> tail_index; // tail_index[k] = index in a of the smallest tail of an
                                         // increasing subsequence of length k+1
    std::vector<std::size_t> prev(n, static_cast<std::size_t>(-1)); // predecessor links
    for (std::size_t i = 0; i < n; ++i) {
        // Binary search for the first length-class whose tail value is >= a[i] (strict increase).
        std::size_t lo = 0;
        std::size_t hi = tail_index.size();
        while (lo < hi) {
            const std::size_t mid = lo + (hi - lo) / 2;
            if (a[tail_index[mid]] < a[i])
                lo = mid + 1;
            else
                hi = mid;
        }
        if (lo > 0) prev[i] = tail_index[lo - 1];
        if (lo == tail_index.size())
            tail_index.push_back(i);
        else
            tail_index[lo] = i;
    }

    std::vector<int> out;
    if (tail_index.empty()) return out;
    for (std::size_t k = tail_index.back(); k != static_cast<std::size_t>(-1); k = prev[k])
        out.push_back(a[k]);
    std::reverse(out.begin(), out.end());
    return out;
}

std::vector<int> shortest_common_supersequence(const std::vector<int>& a, const std::vector<int>& b) {
    const auto  c = lcs_table(a, b);
    std::vector<int> out;
    std::size_t i = a.size();
    std::size_t j = b.size();
    // Walk the LCS table from the corner: emit shared elements once, private elements of each side.
    while (i > 0 && j > 0) {
        if (a[i - 1] == b[j - 1]) {
            out.push_back(a[i - 1]);
            --i;
            --j;
        } else if (c[i - 1][j] >= c[i][j - 1]) {
            out.push_back(a[i - 1]); // a's private element
            --i;
        } else {
            out.push_back(b[j - 1]); // b's private element
            --j;
        }
    }
    while (i > 0) out.push_back(a[--i]);
    while (j > 0) out.push_back(b[--j]);
    std::reverse(out.begin(), out.end());
    return out;
}

std::vector<ScoringSegment> ruzzo_tompa(const std::vector<double>& scores) {
    // Each candidate segment tracks its index range and the cumulative prefix sums immediately
    // before it (l) and at its end (r); its score is r - l.
    struct Seg {
        std::size_t begin;
        std::size_t end;
        double      l;
        double      r;
    };
    std::vector<Seg> list;
    double           cum = 0.0;
    for (std::size_t i = 0; i < scores.size(); ++i) {
        const double before = cum;
        cum += scores[i];
        if (scores[i] <= 0.0) continue; // only positive scores can seed a maximal segment
        Seg seg{i, i + 1, before, cum};
        while (true) {
            // Find the rightmost existing segment j whose left sum is below this one's.
            std::size_t j    = list.size();
            bool        have = false;
            for (std::size_t k = list.size(); k-- > 0;) {
                if (list[k].l < seg.l) {
                    j    = k;
                    have = true;
                    break;
                }
            }
            if (!have) break;               // nothing to merge with: keep as its own segment
            if (list[j].r >= seg.r) break;   // predecessor already scores at least as high: stop
            // Otherwise absorb segments j..end and continue (the extended segment may merge again).
            seg.begin = list[j].begin;
            seg.l     = list[j].l;
            list.resize(j);
        }
        list.push_back(seg);
    }

    std::vector<ScoringSegment> out;
    out.reserve(list.size());
    for (const Seg& s : list) out.push_back(ScoringSegment{s.begin, s.end, s.r - s.l});
    return out;
}

} // namespace datamunge::algorithms
