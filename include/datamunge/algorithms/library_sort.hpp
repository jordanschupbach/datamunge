#pragma once

// Library sort (Bender, Farach-Colton & Mosteller, 2006), also called gapped
// insertion sort. Ordinary insertion sort is simple but each insertion shifts, on
// average, half the array -- O(n^2) total. Library sort keeps the sorted elements
// spread out in a larger array with *empty gaps* between them (like a librarian
// leaving space on a shelf for new books), so a new element usually drops into a
// nearby gap with only a tiny local shift. When the array fills up it is
// rebalanced -- the elements re-spread with fresh gaps. With gaps proportional to
// the size, insertions cost O(log n) amortized and the whole sort runs in
// O(n log n) expected time, keeping insertion sort's simplicity.

#include <cstddef>
#include <vector>

namespace datamunge::algorithms {

// Sort `input` ascending using the gapped-insertion (library sort) method.
inline std::vector<int> library_sort(std::vector<int> input) {
    const int n = static_cast<int>(input.size());
    if (n <= 1) return input;

    int               cap = 2 * n + 2;
    std::vector<long long> arr(cap);
    std::vector<char>      occ(cap, 0);
    int                    m = 0; // number of occupied slots

    // Re-spread the m occupied values evenly across a fresh array of `newcap`.
    auto rebalance = [&](int newcap) {
        std::vector<long long> vals;
        vals.reserve(m);
        for (int i = 0; i < cap; ++i)
            if (occ[i]) vals.push_back(arr[i]);
        arr.assign(newcap, 0);
        occ.assign(newcap, 0);
        cap = newcap;
        const int k = static_cast<int>(vals.size());
        if (k == 0) return;
        const double step = static_cast<double>(cap) / (k + 1);
        for (int i = 0; i < k; ++i) {
            int idx = static_cast<int>((i + 1) * step);
            if (idx >= cap) idx = cap - 1;
            while (occ[idx]) --idx;
            arr[idx] = vals[i];
            occ[idx] = 1;
        }
    };

    // Rightmost occupied slot whose value <= x, or -1 if none.
    auto find_le = [&](long long x) {
        int lo = 0, hi = cap - 1, ans = -1;
        while (lo <= hi) {
            const int mid = (lo + hi) / 2;
            int       p   = mid;
            while (p >= lo && !occ[p]) --p; // nearest occupied at/left of mid
            if (p < lo) { lo = mid + 1; continue; }
            if (arr[p] <= x) { ans = p; lo = p + 1; }
            else hi = p - 1;
        }
        return ans;
    };

    arr[cap / 2] = input[0];
    occ[cap / 2] = 1;
    m            = 1;

    for (int e = 1; e < n; ++e) {
        if ((m + 1) * 2 > cap) rebalance(2 * (m + 1) + 2);
        const long long x   = input[e];
        const int       ans = find_le(x);
        int             start = ans + 1; // desired insertion index (just after the LE value)

        if (start < cap && !occ[start]) {
            arr[start] = x; occ[start] = 1;
        } else {
            // find nearest empty slot to the right and shift the run right
            int j = start;
            while (j < cap && occ[j]) ++j;
            if (j < cap) {
                for (int k = j; k > start; --k) { arr[k] = arr[k - 1]; occ[k] = occ[k - 1]; }
                arr[start] = x; occ[start] = 1;
            } else {
                // else shift a run left into a gap on the left of `ans`
                int i = ans;
                while (i >= 0 && occ[i]) --i;
                for (int k = i; k < ans; ++k) { arr[k] = arr[k + 1]; occ[k] = occ[k + 1]; }
                arr[ans] = x; occ[ans] = 1;
            }
        }
        ++m;
    }

    std::vector<int> out;
    out.reserve(n);
    for (int i = 0; i < cap; ++i)
        if (occ[i]) out.push_back(static_cast<int>(arr[i]));
    return out;
}

} // namespace datamunge::algorithms
