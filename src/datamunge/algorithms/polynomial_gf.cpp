#include <datamunge/algorithms/polynomial_gf.hpp>

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

namespace {

std::int64_t mod(std::int64_t a, std::int64_t p) { return ((a % p) + p) % p; }

std::int64_t mulmod(std::int64_t a, std::int64_t b, std::int64_t p) {
    return static_cast<std::int64_t>(static_cast<__int128>(a) * b % p);
}

std::int64_t powmod(std::int64_t a, std::uint64_t e, std::int64_t p) {
    std::int64_t r = 1 % p;
    a = mod(a, p);
    while (e > 0) {
        if (e & 1) r = mulmod(r, a, p);
        a = mulmod(a, a, p);
        e >>= 1;
    }
    return r;
}

std::int64_t invmod(std::int64_t a, std::int64_t p) { return powmod(mod(a, p), static_cast<std::uint64_t>(p - 2), p); }

void trim(GFPoly& f) {
    while (!f.empty() && f.back() == 0) f.pop_back();
}

int degree(const GFPoly& f) { return static_cast<int>(f.size()) - 1; } // -1 for zero

GFPoly reduce(GFPoly f, std::int64_t p) {
    for (auto& c : f) c = mod(c, p);
    trim(f);
    return f;
}

GFPoly poly_sub(const GFPoly& a, const GFPoly& b, std::int64_t p) {
    GFPoly r(a.size() > b.size() ? a.size() : b.size(), 0);
    for (std::size_t i = 0; i < a.size(); ++i) r[i] = a[i];
    for (std::size_t i = 0; i < b.size(); ++i) r[i] = mod(r[i] - b[i], p);
    trim(r);
    return r;
}

GFPoly poly_mul(const GFPoly& a, const GFPoly& b, std::int64_t p) {
    if (a.empty() || b.empty()) return {};
    GFPoly r(a.size() + b.size() - 1, 0);
    for (std::size_t i = 0; i < a.size(); ++i)
        if (a[i])
            for (std::size_t j = 0; j < b.size(); ++j)
                r[i + j] = mod(r[i + j] + mulmod(a[i], b[j], p), p);
    trim(r);
    return r;
}

PolyDivMod divmod(const GFPoly& a_in, const GFPoly& b_in, std::int64_t p) {
    GFPoly a = reduce(a_in, p);
    GFPoly b = reduce(b_in, p);
    GFPoly q(a.size() >= b.size() ? a.size() - b.size() + 1 : 0, 0);
    const std::int64_t inv_lead = invmod(b.back(), p);
    while (degree(a) >= degree(b) && !a.empty()) {
        const int         shift = degree(a) - degree(b);
        const std::int64_t coef = mulmod(a.back(), inv_lead, p);
        q[shift]                = coef;
        for (std::size_t i = 0; i < b.size(); ++i)
            a[i + shift] = mod(a[i + shift] - mulmod(coef, b[i], p), p);
        trim(a);
    }
    trim(q);
    return {q, a};
}

GFPoly poly_mod(const GFPoly& a, const GFPoly& b, std::int64_t p) { return divmod(a, b, p).remainder; }

GFPoly make_monic(GFPoly f, std::int64_t p) {
    f = reduce(f, p);
    if (f.empty()) return f;
    const std::int64_t inv = invmod(f.back(), p);
    for (auto& c : f) c = mulmod(c, inv, p);
    return f;
}

GFPoly poly_gcd(GFPoly a, GFPoly b, std::int64_t p) {
    a = reduce(a, p);
    b = reduce(b, p);
    while (!b.empty()) {
        GFPoly r = poly_mod(a, b, p);
        a        = b;
        b        = r;
    }
    return make_monic(a, p);
}

// base^e mod m, all polynomials over GF(p).
GFPoly poly_powmod(GFPoly base, std::uint64_t e, const GFPoly& m, std::int64_t p) {
    GFPoly r{1};
    base = poly_mod(base, m, p);
    while (e > 0) {
        if (e & 1) r = poly_mod(poly_mul(r, base, p), m, p);
        base = poly_mod(poly_mul(base, base, p), m, p);
        e >>= 1;
    }
    return reduce(r, p);
}

// Smallest primitive root of the prime p (trial, adequate for the sizes here).
std::int64_t primitive_root(std::int64_t p) {
    if (p == 2) return 1;
    // Factor p-1.
    std::int64_t              n = p - 1;
    std::vector<std::int64_t> primes;
    for (std::int64_t f = 2; f * f <= n; ++f)
        if (n % f == 0) {
            primes.push_back(f);
            while (n % f == 0) n /= f;
        }
    if (n > 1) primes.push_back(n);
    for (std::int64_t g = 2; g < p; ++g) {
        bool ok = true;
        for (std::int64_t q : primes)
            if (powmod(g, static_cast<std::uint64_t>((p - 1) / q), p) == 1) { ok = false; break; }
        if (ok) return g;
    }
    return 1;
}

struct XorShift {
    std::uint64_t s;
    explicit XorShift(std::uint64_t seed) : s(seed ? seed : 0x1234567ULL) {}
    std::uint64_t next() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
};

// Equal-degree factorisation: split f, a product of distinct irreducibles each
// of degree d, into those irreducibles (Cantor-Zassenhaus core, odd p).
void edf(const GFPoly& f, int d, std::int64_t p, XorShift& rng, std::vector<GFPoly>& out) {
    if (f.size() <= 1) return;
    if (degree(f) == d) { out.push_back(make_monic(f, p)); return; }

    // e = (p^d - 1) / 2
    std::uint64_t pe = 1;
    for (int i = 0; i < d; ++i) pe *= static_cast<std::uint64_t>(p);
    const std::uint64_t e = (pe - 1) / 2;

    while (true) {
        // Random polynomial h of degree < deg(f).
        GFPoly h(f.size() - 1, 0);
        for (auto& c : h) c = static_cast<std::int64_t>(rng.next() % static_cast<std::uint64_t>(p));
        h = reduce(h, p);
        if (h.size() <= 1) continue;

        GFPoly g = poly_gcd(h, f, p);
        if (g.size() <= 1 || degree(g) == degree(f)) {
            // h^e - 1 mod f
            GFPoly hp = poly_powmod(h, e, f, p);
            hp        = poly_sub(hp, GFPoly{1}, p);
            if (hp.empty()) continue;
            g = poly_gcd(hp, f, p);
        }
        if (g.size() > 1 && degree(g) < degree(f)) {
            edf(g, d, p, rng, out);
            edf(divmod(f, g, p).quotient, d, p, rng, out);
            return;
        }
    }
}

} // namespace

PolyDivMod poly_divmod_gf(const GFPoly& a, const GFPoly& b, std::int64_t p) { return divmod(a, b, p); }

std::vector<std::int64_t> chien_search(const GFPoly& f_in, std::int64_t p) {
    const GFPoly              f = reduce(f_in, p);
    std::vector<std::int64_t> roots;
    if (f.empty()) return roots; // zero polynomial: every element is a root; report none
    // r = 0 handled directly.
    if (mod(f.empty() ? 0 : f[0], p) == 0) roots.push_back(0);

    // Walk the multiplicative group as successive powers of a primitive root g.
    // term[j] tracks c_j * (g^i)^j; each step multiplies term[j] by g^j so the
    // running sum is f(g^i) -- Chien's incremental evaluation.
    const std::int64_t g = primitive_root(p);
    std::vector<std::int64_t> term(f.size());
    std::vector<std::int64_t> gj(f.size());
    for (std::size_t j = 0; j < f.size(); ++j) {
        term[j] = f[j];                                            // (g^0)^j = 1
        gj[j]   = powmod(g, static_cast<std::uint64_t>(j), p);     // g^j
    }
    for (std::int64_t i = 0; i < p - 1; ++i) {
        std::int64_t sum = 0;
        for (std::size_t j = 0; j < term.size(); ++j) sum = mod(sum + term[j], p);
        if (sum == 0) roots.push_back(powmod(g, static_cast<std::uint64_t>(i), p));
        for (std::size_t j = 0; j < term.size(); ++j) term[j] = mulmod(term[j], gj[j], p);
    }
    return roots;
}

std::vector<GFPoly> cantor_zassenhaus(const GFPoly& f_in, std::int64_t p, std::uint64_t seed) {
    GFPoly f = make_monic(f_in, p);
    std::vector<GFPoly> factors;
    if (f.size() <= 1) return factors;

    XorShift rng(seed);
    // Distinct-degree factorisation: peel off the product of irreducibles of
    // each degree d via gcd(f, x^(p^d) - x).
    GFPoly fstar = f;
    int    d = 1;
    while (degree(fstar) >= 2 * d) {
        // h = x^(p^d) mod fstar, built by applying x -> x^p d times.
        GFPoly h{0, 1};
        for (int k = 0; k < d; ++k) h = poly_powmod(h, static_cast<std::uint64_t>(p), fstar, p);
        GFPoly hx = poly_sub(h, GFPoly{0, 1}, p);   // x^(p^d) - x
        GFPoly g  = poly_gcd(hx, fstar, p);
        if (g.size() > 1) {
            edf(g, d, p, rng, factors);              // g = product of degree-d irreducibles
            fstar = divmod(fstar, g, p).quotient;
        }
        ++d;
    }
    if (fstar.size() > 1) factors.push_back(make_monic(fstar, p)); // remaining irreducible
    return factors;
}

} // namespace datamunge::algorithms
