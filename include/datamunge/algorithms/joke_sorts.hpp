#pragma once

// "Humorous or ineffective" sorts, included for completeness. Neither is useful in
// practice -- they are teaching curiosities that illustrate what NOT to do.
//
//   * Bogosort ("stupid sort"): shuffle the array at random and check if it happens
//     to be sorted; repeat. Expected time is O(n * n!) -- astronomically slow --
//     so it is only ever run on tiny inputs (and is capped here to stay finite).
//   * Slowsort ("multiply and surrender", Broder & Stolfi 1986): a deliberately
//     pessimal divide-and-conquer -- recursively sort both halves, bubble the
//     larger of the two maxima into place, then recursively sort all but the last
//     element. Its running time is superpolynomial, T(n) = 2 T(n/2) + T(n-1) + 1.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

namespace detail {
struct JokeRng {
    std::uint64_t s;
    explicit JokeRng(std::uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ULL) {}
    std::uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    std::size_t   below(std::size_t n) { return static_cast<std::size_t>(next() % n); }
};

template <class T>
bool is_sorted_vec(const std::vector<T>& v) {
    for (std::size_t i = 1; i < v.size(); ++i)
        if (v[i] < v[i - 1]) return false;
    return true;
}

template <class T>
void slowsort_rec(std::vector<T>& a, std::size_t i, std::size_t j) {
    if (i >= j) return;
    const std::size_t m = (i + j) / 2;
    slowsort_rec(a, i, m);
    slowsort_rec(a, m + 1, j);
    if (a[j] < a[m]) std::swap(a[j], a[m]);
    slowsort_rec(a, i, j - 1);
}
} // namespace detail

// Bogosort: shuffle until sorted, up to `max_shuffles` attempts (a safety cap so
// it terminates on inputs too large for it -- it is only correct in practice for
// very small n). Returns the (possibly still-unsorted, if the cap is hit) result.
template <class T>
std::vector<T> bogosort(std::vector<T> v, std::uint64_t seed = 1, long max_shuffles = 1000000) {
    detail::JokeRng rng(seed);
    long            tries = 0;
    while (!detail::is_sorted_vec(v) && tries++ < max_shuffles) {
        for (std::size_t i = v.size(); i > 1; --i) std::swap(v[i - 1], v[rng.below(i)]); // Fisher-Yates
    }
    return v;
}

// Slowsort: the "multiply and surrender" algorithm. Correct, just gloriously slow.
template <class T>
std::vector<T> slowsort(std::vector<T> v) {
    if (v.size() > 1) detail::slowsort_rec(v, 0, v.size() - 1);
    return v;
}

} // namespace datamunge::algorithms
