#pragma once

// Lenstra-Lenstra-Lovasz (LLL) lattice basis reduction.
//
// Given a basis of an integer lattice (rows are basis vectors), returns an
// equivalent basis (same lattice) that is "LLL-reduced": nearly orthogonal and
// short. Uses floating-point Gram-Schmidt with integer size-reduction, so the
// returned basis stays integral.

#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

// Reduce `basis` (each inner vector a lattice basis vector, all the same length)
// with reduction parameter delta in (1/4, 1). Returns the reduced basis.
std::vector<std::vector<long long>> lll_reduce(std::vector<std::vector<long long>> basis,
                                               double delta = 0.75);

} // namespace datamunge::algorithms
