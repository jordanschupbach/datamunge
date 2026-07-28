#include <datamunge/algorithms/elementary_more.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace datamunge::algorithms {

namespace {

// Binary splitting for e = sum 1/k!: returns (P, Q) for the partial sum over [a, b).
std::pair<std::uint64_t, std::uint64_t> bs_e(int a, int b) {
    if (b - a == 1) return {1, static_cast<std::uint64_t>(a == 0 ? 1 : a)};
    const int m = (a + b) / 2;
    const auto [p1, q1] = bs_e(a, m);
    const auto [p2, q2] = bs_e(m, b);
    return {p1 * q2 + p2, q1 * q2};
}

} // namespace

BinSplitResult binary_splitting_e(int terms) {
    if (terms < 1) terms = 1;
    const auto [p, q] = bs_e(0, terms);
    return BinSplitResult{p, q, static_cast<double>(p) / static_cast<double>(q)};
}

std::string spigot_e(int n_digits) {
    const int        m = n_digits + 20; // guard columns
    std::vector<int> a(m + 1, 1);       // a[2..m] = 1 (factorial-base digits of e-2)
    std::string      out = "2.";
    for (int j = 0; j < n_digits; ++j) {
        int carry = 0;
        for (int i = m; i >= 2; --i) {
            const int x = a[i] * 10 + carry;
            a[i]        = x % i;
            carry       = x / i;
        }
        out += static_cast<char>('0' + carry);
    }
    return out;
}

std::string digit_by_digit_sqrt(std::uint64_t n, int frac_digits) {
    std::string s = std::to_string(n);
    if (s.size() % 2 == 1) s = "0" + s; // pad to an even number of digits
    const int intpairs = static_cast<int>(s.size()) / 2;

    std::uint64_t root = 0, rem = 0;
    std::string   result;
    const int     total = intpairs + frac_digits;
    for (int step = 0; step < total; ++step) {
        const int pair = (step < intpairs) ? (s[2 * step] - '0') * 10 + (s[2 * step + 1] - '0') : 0;
        rem            = rem * 100 + static_cast<std::uint64_t>(pair);
        int d = 0;
        for (int cand = 9; cand >= 0; --cand)
            if ((20 * root + cand) * static_cast<std::uint64_t>(cand) <= rem) { d = cand; break; }
        rem -= (20 * root + d) * static_cast<std::uint64_t>(d);
        root = root * 10 + d;
        result += static_cast<char>('0' + d);
        if (step == intpairs - 1) result += '.';
    }
    return result;
}

DivResult long_division(std::uint64_t dividend, std::uint64_t divisor) {
    std::uint64_t q = 0, r = 0;
    int           iters = 0;
    for (int i = 63; i >= 0; --i) {
        r = (r << 1) | ((dividend >> i) & 1ULL);
        ++iters;
        if (r >= divisor) {
            r -= divisor;
            q |= (1ULL << i);
        }
    }
    return DivResult{q, r, iters};
}

} // namespace datamunge::algorithms
