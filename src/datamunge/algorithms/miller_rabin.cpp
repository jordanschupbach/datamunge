#include <datamunge/algorithms/miller_rabin.hpp>

namespace datamunge::algorithms {

namespace {

// Modular multiplication via a 128-bit intermediate. The product of two values below 2^64 can be as
// large as ~2^128, which overflows any 64-bit type; unsigned __int128 holds it exactly, so (a*b)%m
// is correct for every modulus up to 2^64 - 1 -- the overflow-safety the whole test rests on.
inline std::uint64_t mulmod(std::uint64_t a, std::uint64_t b, std::uint64_t m) {
    return static_cast<std::uint64_t>((static_cast<unsigned __int128>(a) * b) % m);
}

// Square-and-multiply modular exponentiation: base^exp mod m in O(log exp) multiplications, each
// reduced with the overflow-safe mulmod above. Self-contained on purpose -- Miller-Rabin depends on
// no other header for its arithmetic.
std::uint64_t mod_pow(std::uint64_t base, std::uint64_t exp, std::uint64_t m) {
    std::uint64_t result = 1 % m; // a^0 (== 0 when m == 1, though callers always pass m >= 3)
    base %= m;
    while (exp > 0) {
        if ((exp & 1u) != 0u) result = mulmod(result, base, m); // fold in a^(2^i) where the bit is set
        base = mulmod(base, base, m);                           // square to a^(2^(i+1))
        exp >>= 1;                                              // advance to the next binary digit
    }
    return result;
}

} // namespace

bool is_strong_probable_prime_base(std::uint64_t n, std::uint64_t a) {
    if (n < 2) return false;
    if (n % 2 == 0) return n == 2; // among even numbers only 2 is prime
    a %= n;
    if (a == 0) return true; // base is a multiple of n: conventionally a vacuous pass (never a witness)

    // Factor out the 2s: n - 1 = d * 2^s with d odd.
    std::uint64_t d = n - 1;
    int s = 0;
    while ((d & 1u) == 0u) {
        d >>= 1;
        ++s;
    }

    // x = a^d mod n. If x is 1 or n-1 the round already passes (the sequence starts at, or opens onto,
    // a square root +-1 of 1). Otherwise square up to s-1 times looking for n-1; if we never see it,
    // some a^(2^r d) is a nontrivial square root of 1 -- impossible modulo a prime -- so a is a
    // witness and n is composite.
    std::uint64_t x = mod_pow(a, d, n);
    if (x == 1 || x == n - 1) return true;
    for (int r = 1; r < s; ++r) {
        x = mulmod(x, x, n);
        if (x == n - 1) return true;
    }
    return false;
}

bool is_probable_prime(std::uint64_t n) {
    if (n < 2) return false;

    // The witnesses double as a trial-division screen: n equal to one of them is prime, n divisible
    // by one (and larger) is composite. This disposes of every n <= 37 (and all even n), so past the
    // loop n is odd, exceeds 37, and is coprime to all twelve bases -- exactly the regime the strong
    // pseudoprime rounds below need.
    static constexpr std::uint64_t witnesses[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37};
    for (const std::uint64_t p : witnesses) {
        if (n == p) return true;
        if (n % p == 0) return false;
    }

    // Deterministic for all 64-bit n: this fixed base set witnesses every composite below 3.3e24.
    for (const std::uint64_t a : witnesses)
        if (!is_strong_probable_prime_base(n, a)) return false;
    return true;
}

} // namespace datamunge::algorithms
