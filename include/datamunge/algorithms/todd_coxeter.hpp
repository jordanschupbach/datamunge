#pragma once

// Todd-Coxeter coset enumeration.
//
// Given a finite presentation of a group by generators and relators, together
// with generators of a subgroup H, enumerates the cosets of H and returns the
// index [G : H]. With no subgroup generators the index is the group order.
//
// Words are sequences of signed integers: +（i+1) is generator i, -(i+1) is its
// inverse (generators are numbered 0..ngen-1). A relator is a word equal to the
// identity; a subgroup generator is a word representing an element of H.

#include <vector>

namespace datamunge::algorithms {

// Returns [G : H], or -1 if enumeration exceeds `max_cosets` (does not close).
long long todd_coxeter_index(int ngen,
                             const std::vector<std::vector<int>>& relators,
                             const std::vector<std::vector<int>>& subgroup_gens,
                             long long                            max_cosets = 200000);

} // namespace datamunge::algorithms
