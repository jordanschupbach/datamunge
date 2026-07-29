#include <datamunge/algorithms/modular_exponentiation.hpp>

#include <stdexcept>

namespace datamunge::algorithms {

std::uint64_t mod_pow(std::uint64_t base, std::uint64_t exponent, std::uint64_t modulus) {
    if (modulus == 0)
        throw std::invalid_argument("mod_pow: modulus must be nonzero");
    if (modulus == 1)
        return 0; // everything is congruent to 0 mod 1 (this also covers exponent == 0)

    std::uint64_t result = 1;          // a^0
    std::uint64_t running = base % modulus; // a^(2^i) for the current bit i
    while (exponent > 0) {
        if ((exponent & 1u) != 0u)     // bit set: fold a^(2^i) into the product
            result = static_cast<std::uint64_t>(
                (static_cast<unsigned __int128>(result) * running) % modulus);
        running = static_cast<std::uint64_t>(
            (static_cast<unsigned __int128>(running) * running) % modulus); // square to a^(2^(i+1))
        exponent >>= 1;                // advance to the next binary digit
    }
    return result;
}

} // namespace datamunge::algorithms
