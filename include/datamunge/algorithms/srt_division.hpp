#pragma once

// Radix-2 SRT division (Sweeney, Robertson, Tocher). A restoring/non-restoring
// divider produces one nonnegative quotient bit per step; SRT instead allows a
// *redundant* quotient digit set {-1, 0, 1}, so the digit can be chosen from
// just a few leading bits of the shifted partial remainder without a full
// comparison against the divisor. That redundancy is what lets real hardware
// dividers (and radix-4/8 variants) run fast.
//
// For a normalized divisor d in [1/2, 1) and a proper-fraction dividend
// 0 <= n < d, the partial remainder recurrence is
//
//   P_0 = n,   q_j = select(2 P_{j-1}),   P_j = 2 P_{j-1} - q_j d,
//
// with the radix-2 selection q = +1 if 2P >= 1/2, -1 if 2P <= -1/2, else 0. The
// quotient is Q = sum_j q_j 2^{-j}, converging to n/d = N/D with error below
// 2^{-bits}. Inputs are normalized internally, so any 0 <= N < D works.

#include <cmath>
#include <vector>

namespace datamunge::algorithms {

namespace detail {

// Scale D into [1/2, 1) by a power of two; return the exponent e with D*2^e in range.
inline int srt_normalize_exp(double D) {
    int e = 0;
    while (D < 0.5) { D *= 2.0; ++e; }
    while (D >= 1.0) { D *= 0.5; --e; }
    return e;
}

} // namespace detail

// The redundant quotient digits q_1..q_bits (each in {-1,0,1}) for N/D,
// requiring 0 <= N < D and D > 0.
inline std::vector<int> srt_quotient_digits(double N, double D, int bits) {
    const int    e = detail::srt_normalize_exp(D);
    const double d = std::ldexp(D, e); // in [1/2, 1)
    double       P = std::ldexp(N, e); // n = N*2^e, with 0 <= n < d
    std::vector<int> digits;
    digits.reserve(bits);
    for (int j = 0; j < bits; ++j) {
        const double w = 2.0 * P;
        const int    q = w >= 0.5 ? 1 : (w <= -0.5 ? -1 : 0);
        P              = w - q * d;
        digits.push_back(q);
    }
    return digits;
}

// Reconstruct Q = sum q_j 2^{-j} from redundant digits.
inline double srt_digits_to_value(const std::vector<int>& digits) {
    double Q = 0.0, w = 1.0;
    for (int q : digits) { w *= 0.5; Q += q * w; }
    return Q;
}

// Quotient N/D to `bits` bits of precision by radix-2 SRT division.
// Preconditions: 0 <= N < D, D > 0.
inline double srt_divide(double N, double D, int bits = 53) {
    return srt_digits_to_value(srt_quotient_digits(N, D, bits));
}

} // namespace datamunge::algorithms
