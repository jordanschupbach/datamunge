#pragma once

/// @file dixon.hpp
/// @brief Deterministic Dixon square relations, dynamic GF(2) elimination and owned evidence.

#include <cstddef>
#include <cstdint>
#include <vector>

namespace datamunge::algorithms {

/// @brief One positive factor-base-smooth square residue from the actual scan.
/// Owns exponents in the collection's prime order. The empty product is one;
/// residue zero is excluded. No references into solver storage survive return.
/// @par Example
/// @code
/// const auto c = dm::algorithms::dixon_collect(77, 3, 1);
/// std::print("{} {}\n", c.relations[0].x, c.relations[0].residue); // 9 4
/// @endcode
/// @see DixonCollection, dixon_collect()
struct DixonRelation {
    std::uint64_t x = 0; ///< Scanned integer, before reduction modulo n.
    std::uint64_t residue = 0; ///< Positive x*x modulo n, computed without 64-bit product overflow.
    std::size_t offset = 0; ///< Zero-based offset from collection.start.
    std::vector<unsigned> exponents; ///< Exact nonnegative exponents in collection.base order.
};

/// @brief Owned positive smooth relations and actual scan progress for a fixed modulus.
/// Input candidates are consecutive integers from floor(sqrt(n))+1, even for
/// perfect squares. Only accepted relations are retained; zero residues are
/// counted separately. This is a deterministic scan, not uniform random sampling.
/// @par Example
/// @code
/// const auto c = dm::algorithms::dixon_collect(77, 3, 10);
/// std::print("{} {} {}\n", c.start, c.scanned, c.relations.size()); // 9 10 4
/// @endcode
/// @par References
/// - J. D. Dixon (1981), Asymptotically Fast Factorization of Integers,
///   Mathematics of Computation 36(153),255-260. https://doi.org/10.1090/S0025-5718-1981-0595059-1
/// @see dixon_collect(), DixonTrace
struct DixonCollection {
    std::uint64_t n = 0; ///< Original modulus, at least two.
    std::uint64_t start = 0; ///< floor(sqrt(n))+1, computed using exact integer comparisons.
    std::vector<std::uint64_t> base; ///< First base_size primes, including primes dividing n.
    std::size_t scanned = 0; ///< Number of candidates actually examined.
    std::size_t zero_residues = 0; ///< Excluded candidates with x*x modulo n equal to zero.
    std::vector<DixonRelation> relations; ///< Accepted relations, in ascending x order.
};

/// @brief An actual zero row of the eliminated parity matrix and its square congruence.
/// Owns original relation indices and full exponent sums, all even. Both GCDs
/// are retained; the solver selects a proper minus factor first, then plus.
/// A factor need not itself be prime. No combinations of different zero rows
/// are searched, and a trivial congruence can therefore leave the run unsuccessful.
/// @par Example
/// @code
/// const auto t = dm::algorithms::dixon_factor_trace(77, 3, 10);
/// std::print("{} {}\n", t.dependencies[0].X, t.dependencies[0].Y); // 9 2
/// @endcode
/// @par References
/// - Dixon (1981), https://doi.org/10.1090/S0025-5718-1981-0595059-1
/// @see DixonTrace, dixon_factor_trace()
struct DixonDependency {
    std::vector<std::size_t> relation_indices; ///< Original accepted rows selected by this dependency.
    std::vector<std::uint64_t> exponent_sums; ///< Full even exponents, in factor-base order.
    std::uint64_t X = 0; ///< Product of selected x values modulo n.
    std::uint64_t Y = 0; ///< Product of primes to half their exponent sums modulo n.
    std::uint64_t gcd_minus = 0; ///< gcd(abs(X-Y),n), with no unsigned subtraction wrap.
    std::uint64_t gcd_plus = 0; ///< gcd((X+Y) modulo n,n), with no unsigned addition wrap.
};

/// @brief Owned actual collection, tested dependencies and logical elimination work.
/// No replay: the same kernel implements dixon_factor(). Collection stops at
/// base_size+1 relations or the candidate budget. All-row GF(2) elimination uses
/// dynamic 64-bit word vectors, so bases and relation sets can exceed 64 bits.
/// Only dependencies visited up to the first proper factor are captured.
/// Captured scalar cells equal base.size()+sum(exponents.size()+3) over relations
/// +sum(relation_indices.size()+exponent_sums.size()+4) over dependencies.
/// The optional cap excludes vector objects and transient elimination storage.
/// Scratch space is O(k*k), in addition to captured output; no shared mutation.
/// @par Example
/// @code
/// const auto t = dm::algorithms::dixon_factor_trace(77, 3, 10);
/// std::print("{} {} {}\n", t.factor, t.rank, t.collection.scanned); // 7 1 10
/// @endcode
/// @par References
/// - Dixon (1981), https://doi.org/10.1090/S0025-5718-1981-0595059-1
/// @see dixon_factor(), dixon_factor_trace(), DixonDependency
struct DixonTrace {
    DixonCollection collection; ///< Accepted input relations, kept independently of parity reduction.
    std::vector<DixonDependency> dependencies; ///< Tested zero rows in final eliminated row order.
    std::uint64_t factor = 0; ///< Proper divisor, or zero for unsuccessful bounded search.
    std::size_t rank = 0; ///< Rank of the full collected parity matrix, even if factoring stops early.
    std::size_t row_xors = 0; ///< Whole parity-row XOR operations; each has a matching combination XOR.
    std::size_t successful_divisions = 0; ///< Successful prime divisions during collection, excluding tests.
    std::size_t captured_cells = 0; ///< Logical scalar-cell count described above.
};

/// @brief Collect a fixed window of positive smooth square residues for Dixon analysis.
/// Scans exactly candidates integers unless the capture cap or an allocation
/// fails. Supports all uint64_t moduli at least two, including primes/even inputs;
/// no primality classification or factorisation is promised. Zero residues are
/// skipped before trial division, so division of zero cannot loop forever.
/// Each candidate costs O(k+log(n)) trial/modular operations, with k=base_size;
/// base construction uses trial division. Returned vectors own their storage.
/// @param n Modulus, at least two; unchanged.
/// @param base_size Number of first primes, in [1,4096].
/// @param candidates Number of consecutive x values; zero gives an empty scan.
/// @param max_cells Optional captured scalar-cell cap, default 1000000; zero disables only this cap.
/// @return Owned collection, including excluded zero-residue count and accepted offsets.
/// @throws std::invalid_argument If n<2 or base_size is outside [1,4096].
/// @throws std::overflow_error If the last candidate would exceed uint64_t.
/// @throws std::length_error If a capture/count/container bound is exceeded; checked before capture.
/// @throws std::bad_alloc If storage allocation fails.
/// @par Example
/// @code
/// const auto c = dm::algorithms::dixon_collect(92434447, 30, 4000);
/// std::print("{} {}\n", c.scanned, c.relations.size()); // 4000 25
/// @endcode
/// @par References
/// - Dixon (1981), https://doi.org/10.1090/S0025-5718-1981-0595059-1
/// @see dixon_factor_trace()
DixonCollection dixon_collect(std::uint64_t n, int base_size, std::size_t candidates,
                              std::size_t max_cells = 1000000);

/// @brief Factor by deterministic smooth squares with owned actual GF(2) evidence.
/// Same scan, relation target, pivot/row order and arithmetic as dixon_factor(),
/// with a configurable candidate budget. Computes both GCDs for each visited
/// dependency. A zero result means only that this bounded run did not find a
/// factor; neither primality nor irreducibility follows. Rank-nullity guarantees
/// a dependency for k+1 rows but does not guarantee a nontrivial factor.
/// Uses 128-bit products for exact uint64_t modular arithmetic. Search is a
/// sequential teaching variant; Dixon's random-sampling expected-time theorem
/// does not establish a subexponential bound for this scan. For R<=k+1 rows,
/// elimination costs O(k*R*ceil((k+R)/64)) word operations and O(k*R) storage.
/// @param n Modulus, at least two; unchanged.
/// @param base_size Number of first primes in [1,4096].
/// @param max_candidates Candidate budget, default 2000000; zero scans nothing.
/// @param max_cells Captured scalar-cell cap as DixonTrace, default 1000000; zero disables only cap.
/// @return Owned trace, with proper divisor or zero and all actually tested dependencies.
/// @throws std::invalid_argument If n<2 or base_size is outside [1,4096].
/// @throws std::overflow_error If the requested scan endpoint exceeds uint64_t.
/// @throws std::length_error If a capture/count/container bound is exceeded, before the affected capture.
/// @throws std::bad_alloc If storage allocation fails.
/// @par Example
/// @code
/// const auto t = dm::algorithms::dixon_factor_trace(77, 3, 10);
/// std::print("{} {}\n", t.factor, t.dependencies[0].gcd_plus); // 7 11
/// @endcode
/// @par References
/// - J. D. Dixon (1981), Asymptotically Fast Factorization of Integers,
///   Mathematics of Computation 36(153),255-260. https://doi.org/10.1090/S0025-5718-1981-0595059-1
/// @see dixon_factor(), dixon_collect()
DixonTrace dixon_factor_trace(std::uint64_t n, int base_size,
                              std::size_t max_candidates = 2000000,
                              std::size_t max_cells = 1000000);

/// @brief Return a proper divisor from the shared deterministic Dixon kernel.
/// Preserves the original two-argument API and first k+1-relation batch strategy.
/// Tests at most 2000000 consecutive candidates, from floor(sqrt(n))+1. Dynamic
/// parity and combination vectors support bases beyond the former machine-word
/// boundary; singleton dependencies are eligible. See dixon_factor_trace() for
/// ordering, complexity and the limits of this bounded deterministic method.
/// Retains no dependency history and imposes no optional capture cap.
/// @param n Modulus, at least two; unchanged.
/// @param base_size Number of first primes, in [1,4096].
/// @return Proper divisor (possibly composite), or zero if this search is unsuccessful.
/// @throws std::invalid_argument If n<2 or base_size is outside [1,4096].
/// @throws std::length_error If a count/container bound is exceeded.
/// @throws std::bad_alloc If storage allocation fails.
/// @par Example
/// @code
/// std::print("{}\n", dm::algorithms::dixon_factor(8051, 30)); // 83
/// @endcode
/// @par References
/// - Dixon (1981), https://doi.org/10.1090/S0025-5718-1981-0595059-1
/// @see dixon_factor_trace(), dixon_collect()
std::uint64_t dixon_factor(std::uint64_t n, int base_size);

} // namespace datamunge::algorithms
