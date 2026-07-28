#include <datamunge/algorithms/multiplication.hpp>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace datamunge::algorithms {

namespace {

// ---- base-10 digit-vector helpers (little-endian: index 0 is the ones place) ----

using Coeffs = std::vector<std::int64_t>;

Coeffs from_string(const std::string& s) {
    Coeffs d;
    d.reserve(s.size());
    for (auto it = s.rbegin(); it != s.rend(); ++it) d.push_back(*it - '0');
    if (d.empty()) d.push_back(0);
    return d;
}

// Normalise a convolution of base-10 coefficients (may exceed 9) into a decimal
// string by propagating carries.
std::string to_string(Coeffs c) {
    std::int64_t carry = 0;
    for (std::size_t i = 0; i < c.size(); ++i) {
        const std::int64_t cur = c[i] + carry;
        c[i]                   = ((cur % 10) + 10) % 10; // keep non-negative
        carry                  = (cur - c[i]) / 10;
    }
    while (carry > 0) {
        c.push_back(carry % 10);
        carry /= 10;
    }
    std::size_t hi = c.size();
    while (hi > 1 && c[hi - 1] == 0) --hi;
    std::string out;
    out.reserve(hi);
    for (std::size_t i = hi; i-- > 0;) out.push_back(static_cast<char>('0' + c[i]));
    return out;
}

Coeffs schoolbook(const Coeffs& a, const Coeffs& b) {
    Coeffs r(a.size() + b.size(), 0);
    for (std::size_t i = 0; i < a.size(); ++i)
        for (std::size_t j = 0; j < b.size(); ++j) r[i + j] += a[i] * b[j];
    return r;
}

Coeffs add(const Coeffs& a, const Coeffs& b) {
    Coeffs r(std::max(a.size(), b.size()), 0);
    for (std::size_t i = 0; i < a.size(); ++i) r[i] += a[i];
    for (std::size_t i = 0; i < b.size(); ++i) r[i] += b[i];
    return r;
}

void trim(Coeffs& c) {
    while (c.size() > 1 && c.back() == 0) c.pop_back();
}

// ---- Karatsuba ----

Coeffs karatsuba(const Coeffs& a, const Coeffs& b) {
    const std::size_t n = std::max(a.size(), b.size());
    if (n <= 32) return schoolbook(a, b);
    const std::size_t half = n / 2;

    Coeffs a0(a.begin(), a.begin() + std::min(half, a.size()));
    Coeffs a1(a.size() > half ? a.begin() + half : a.end(), a.end());
    Coeffs b0(b.begin(), b.begin() + std::min(half, b.size()));
    Coeffs b1(b.size() > half ? b.begin() + half : b.end(), b.end());
    if (a0.empty()) a0.push_back(0);
    if (a1.empty()) a1.push_back(0);
    if (b0.empty()) b0.push_back(0);
    if (b1.empty()) b1.push_back(0);

    const Coeffs z0 = karatsuba(a0, b0);
    const Coeffs z2 = karatsuba(a1, b1);
    Coeffs       z1 = karatsuba(add(a0, a1), add(b0, b1));
    for (std::size_t i = 0; i < z0.size(); ++i) z1[i] -= z0[i];
    for (std::size_t i = 0; i < z2.size(); ++i) z1[i] -= z2[i];

    // Size the result to the true maximum write index: the z2 term lands at
    // offset 2*half, and with very unequal operand lengths that can exceed
    // a.size()+b.size(), so a+b+1 is not a safe bound.
    const std::size_t need =
        std::max({z0.size(), z1.size() + half, z2.size() + 2 * half});
    Coeffs r(need, 0);
    for (std::size_t i = 0; i < z0.size(); ++i) r[i] += z0[i];
    for (std::size_t i = 0; i < z1.size(); ++i) r[i + half] += z1[i];
    for (std::size_t i = 0; i < z2.size(); ++i) r[i + 2 * half] += z2[i];
    trim(r);
    return r;
}

// ---- Toom-3 ----
// Splits each operand into three base-B parts (B = 10^k). Evaluates both at the
// five points 0, 1, -1, -2, infinity, multiplies pointwise, then interpolates.

Coeffs slice(const Coeffs& a, std::size_t lo, std::size_t k) {
    Coeffs r(a.begin() + std::min(lo, a.size()), a.begin() + std::min(lo + k, a.size()));
    if (r.empty()) r.push_back(0);
    return r;
}

Coeffs sub(const Coeffs& a, const Coeffs& b) {
    Coeffs r = a;
    r.resize(std::max(a.size(), b.size()), 0);
    for (std::size_t i = 0; i < b.size(); ++i) r[i] -= b[i];
    return r;
}

Coeffs scale(const Coeffs& a, std::int64_t s) {
    Coeffs r = a;
    for (auto& x : r) x *= s;
    return r;
}

// exact division of every coefficient by a small constant (interpolation step)
Coeffs divexact(const Coeffs& a, std::int64_t s) {
    Coeffs r = a;
    for (auto& x : r) x /= s;
    return r;
}

Coeffs shift(const Coeffs& a, std::size_t k) { // multiply by B^k (k base-B limbs)
    Coeffs r(a.size() + k, 0);
    for (std::size_t i = 0; i < a.size(); ++i) r[i + k] = a[i];
    return r;
}

Coeffs toom3(const Coeffs& A, const Coeffs& B) {
    const std::size_t n = std::max(A.size(), B.size());
    if (n <= 48) return karatsuba(A, B);
    const std::size_t k = (n + 2) / 3; // limbs per part

    const Coeffs a0 = slice(A, 0, k), a1 = slice(A, k, k), a2 = slice(A, 2 * k, k);
    const Coeffs b0 = slice(B, 0, k), b1 = slice(B, k, k), b2 = slice(B, 2 * k, k);

    // Evaluate p(x)=a0+a1 x+a2 x^2 and q similarly at x = 0, 1, -1, 2, inf.
    const Coeffs p1 = add(add(a0, a1), a2);              // p(1)
    const Coeffs pm1 = add(sub(a0, a1), a2);             // p(-1)
    const Coeffs p2 = add(add(a0, scale(a1, 2)), scale(a2, 4)); // p(2)
    const Coeffs q1 = add(add(b0, b1), b2);
    const Coeffs qm1 = add(sub(b0, b1), b2);
    const Coeffs q2 = add(add(b0, scale(b1, 2)), scale(b2, 4));

    // Pointwise products w = p q at the five evaluation points.
    const Coeffs w0 = toom3(a0, b0);   // r(0)
    const Coeffs w1 = toom3(p1, q1);   // r(1)
    const Coeffs w2 = toom3(pm1, qm1); // r(-1)
    const Coeffs w3 = toom3(p2, q2);   // r(2)
    const Coeffs w4 = toom3(a2, b2);   // r(inf) = leading coeff

    // Interpolate c0..c4 of r(x) = sum c_i x^i from the five samples.
    //   c0 = w0,  c4 = w4
    //   c2 = (w1 + w2)/2 - c0 - c4
    //   s  = (w1 - w2)/2 = c1 + c3
    //   t  = (w3 - c0 - 4 c2 - 16 c4)/2 = c1 + 4 c3
    //   c3 = (t - s)/3,  c1 = s - c3
    const Coeffs c0 = w0;
    const Coeffs c4 = w4;
    const Coeffs c2 = sub(sub(divexact(add(w1, w2), 2), c0), c4);
    const Coeffs s  = divexact(sub(w1, w2), 2);
    const Coeffs t  = divexact(sub(sub(sub(w3, c0), scale(c2, 4)), scale(c4, 16)), 2);
    const Coeffs c3 = divexact(sub(t, s), 3);
    const Coeffs c1 = sub(s, c3);

    // Recompose r(B) = c0 + c1 B + c2 B^2 + c3 B^3 + c4 B^4.
    Coeffs res = c0;
    res        = add(res, shift(c1, k));
    res        = add(res, shift(c2, 2 * k));
    res        = add(res, shift(c3, 3 * k));
    res        = add(res, shift(c4, 4 * k));
    return res;
}

// ---- Number-theoretic transform (Schoenhage-Strassen representative) ----

constexpr std::int64_t kMod  = 998244353;    // 119*2^23 + 1
constexpr std::int64_t kRoot = 3;            // primitive root

std::int64_t powmod(std::int64_t a, std::int64_t e, std::int64_t m) {
    std::int64_t r = 1 % m;
    a %= m;
    while (e > 0) {
        if (e & 1) r = static_cast<__int128>(r) * a % m;
        a = static_cast<__int128>(a) * a % m;
        e >>= 1;
    }
    return r;
}

void ntt(std::vector<std::int64_t>& a, bool invert) {
    const std::size_t n = a.size();
    for (std::size_t i = 1, j = 0; i < n; ++i) {
        std::size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (std::size_t len = 2; len <= n; len <<= 1) {
        const std::int64_t w = invert ? powmod(powmod(kRoot, kMod - 2, kMod), (kMod - 1) / len, kMod)
                                       : powmod(kRoot, (kMod - 1) / len, kMod);
        for (std::size_t i = 0; i < n; i += len) {
            std::int64_t wn = 1;
            for (std::size_t k = 0; k < len / 2; ++k) {
                const std::int64_t u = a[i + k];
                const std::int64_t v = static_cast<__int128>(a[i + k + len / 2]) * wn % kMod;
                a[i + k]             = (u + v) % kMod;
                a[i + k + len / 2]   = (u - v % kMod + kMod) % kMod;
                wn                   = static_cast<__int128>(wn) * w % kMod;
            }
        }
    }
    if (invert) {
        const std::int64_t ninv = powmod(static_cast<std::int64_t>(n), kMod - 2, kMod);
        for (auto& x : a) x = static_cast<__int128>(x) * ninv % kMod;
    }
}

Coeffs ntt_convolve(const Coeffs& a, const Coeffs& b) {
    std::size_t sz = 1;
    while (sz < a.size() + b.size()) sz <<= 1;
    std::vector<std::int64_t> fa(sz, 0), fb(sz, 0);
    for (std::size_t i = 0; i < a.size(); ++i) fa[i] = a[i];
    for (std::size_t i = 0; i < b.size(); ++i) fb[i] = b[i];
    ntt(fa, false);
    ntt(fb, false);
    for (std::size_t i = 0; i < sz; ++i) fa[i] = static_cast<__int128>(fa[i]) * fb[i] % kMod;
    ntt(fa, true);
    return Coeffs(fa.begin(), fa.end());
}

} // namespace

std::string toom3_mul(const std::string& a, const std::string& b) {
    Coeffs r = toom3(from_string(a), from_string(b));
    return to_string(r);
}

std::string ntt_mul(const std::string& a, const std::string& b) {
    Coeffs r = ntt_convolve(from_string(a), from_string(b));
    return to_string(r);
}

BoothResult booth_multiply(std::int32_t multiplicand, std::int32_t multiplier) {
    // Radix-2 Booth: scan the multiplier's bits with an implicit -1 bit below
    // the LSB, acting on each 0->1 (subtract) / 1->0 (add) transition of the
    // running (b_i, b_{i-1}) pair. The accumulator is the exact product.
    const std::int64_t m = multiplicand;
    std::int64_t       acc = 0;
    BoothResult        out;
    int                prev = 0; // b_{-1}
    for (int i = 0; i < 32; ++i) {
        const int cur = (static_cast<std::uint32_t>(multiplier) >> i) & 1u;
        if (prev == 1 && cur == 0) { // 10 : add multiplicand * 2^i
            acc += m << i;
            ++out.additions;
        } else if (prev == 0 && cur == 1) { // 01 : subtract multiplicand * 2^i
            acc -= m << i;
            ++out.subtractions;
        }
        prev = cur;
    }
    out.product = acc;
    return out;
}

} // namespace datamunge::algorithms
