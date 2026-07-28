#include <datamunge/algorithms/division.hpp>

#include <cstdint>

namespace datamunge::algorithms {

DivResultEx restoring_divide(std::uint64_t dividend, std::uint64_t divisor) {
    if (divisor == 0) return {0, 0, 0};
    // Partial remainder R and quotient Q built one bit at a time, MSB first.
    // At each step: bring down the next dividend bit, trial-subtract D; if R
    // went negative, restore it (add D back) and record a 0 bit.
    std::uint64_t Q = 0, R = 0;
    int           iters = 0;
    for (int i = 63; i >= 0; --i) {
        R = (R << 1) | ((dividend >> i) & 1ULL);
        ++iters;
        // Trial subtraction; compare instead of letting R wrap (unsigned).
        if (R >= divisor) {
            R -= divisor;           // subtraction stands
            Q |= (1ULL << i);       // quotient bit 1
        }
        // else: "restore" -- R is left unchanged, quotient bit 0.
    }
    return {Q, R, iters};
}

DivResultEx non_restoring_divide(std::uint64_t dividend, std::uint64_t divisor) {
    if (divisor == 0) return {0, 0, 0};
    // Signed partial remainder A over the (A, Q) register pair. Q starts as the
    // dividend and ends as the quotient; A ends as the remainder.
    __int128       A = 0;
    std::uint64_t  Q = dividend;
    const __int128 D = static_cast<__int128>(divisor);
    int            iters = 0;
    for (int i = 0; i < 64; ++i) {
        // Shift the (A, Q) pair left by one, bringing Q's MSB into A.
        A = (A << 1) | static_cast<__int128>((Q >> 63) & 1ULL);
        Q <<= 1;
        if (A >= 0)
            A -= D; // last step left a non-negative remainder: subtract
        else
            A += D; // last step overshot: add instead of restoring
        ++iters;
        if (A >= 0) Q |= 1ULL; // quotient bit from the sign of the new remainder
    }
    if (A < 0) A += D; // single final correction
    return {Q, static_cast<std::uint64_t>(A), iters};
}

DivResultEx newton_raphson_divide(std::uint64_t dividend, std::uint64_t divisor) {
    if (divisor == 0) return {0, 0, 0};
    if (dividend < divisor) return {0, dividend, 0};

    // Multiplicative division: instead of subtracting, approximate the reciprocal
    // r = 1/divisor with the quadratically convergent Newton map r <- r*(2 - d*r)
    // (each step doubles the number of correct bits), then form the quotient as
    // dividend*r and finish with an exact integer correction.
    const double d = static_cast<double>(divisor);
    double       r = 1.0 / d; // seed; a hardware unit would start from a table
    int          iters = 0;
    for (int k = 0; k < 5; ++k) { // quadratic convergence saturates fast
        r = r * (2.0 - d * r);
        ++iters;
    }

    std::uint64_t quotient = static_cast<std::uint64_t>(static_cast<double>(dividend) * r);

    // The reciprocal is only accurate to double precision (~53 bits), so the
    // estimate can be off by a bounded amount; correct it exactly in 128-bit.
    while (static_cast<__int128>(quotient) * divisor > static_cast<__int128>(dividend)) --quotient;
    while (static_cast<__int128>(quotient + 1) * divisor <= static_cast<__int128>(dividend)) ++quotient;

    return {quotient, dividend - quotient * divisor, iters};
}

} // namespace datamunge::algorithms
