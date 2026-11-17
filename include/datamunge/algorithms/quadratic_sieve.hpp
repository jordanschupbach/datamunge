#pragma once
/// @file quadratic_sieve.hpp
/// @brief Bounded word-sized quadratic sieve with actual smooth relations and square dependencies.
#include <cstddef>
#include <cstdint>
#include <vector>
/// @brief Canonical factorization and owned producer records.
namespace datamunge::algorithms {
/// @brief Search/allocation policies for the word-sized educational sieve.
/// @par Example
/// @code
/// const dm::algorithms::QuadraticSieveOptions x{};
/// std::print("{}\n",x.max_windows); // 120
/// @endcode
/// @see quadratic_sieve_run()
struct QuadraticSieveOptions {
  unsigned factor_base_bound{
      0}; ///< Zero selects historical heuristic clamped60..6000; otherwise2..6000.
  std::size_t window_size{0}; ///< Zero selects100*base_size+3000; explicit size1..1000000.
  std::size_t max_windows{
      120}; ///< Maximum actual sieve windows, including zero; hard maximum1000000.
  std::size_t max_sieve_values{
      10000000}; ///< Cumulative actual allocated/sieved Q values; zero disables.
  std::size_t max_trace_entries{
      1000000}; ///< Stored relation exponents/dependency members plus headers; zero disables.
};
/// @brief One actual accepted positive smooth relation.
/// @par Example
/// @code
/// const dm::algorithms::QuadraticSieveRelation x{};
/// std::print("{}\n",x.x); // 0
/// @endcode
/// @see quadratic_sieve_run()
struct QuadraticSieveRelation {
  std::uint64_t
      x{}; ///< Actual x with positive Q=x*x-n, computed internally in signed128 arithmetic.
  std::vector<unsigned> exponents; ///< Full actual exponents over the result's factor base.
  int pivot_column{-1}; ///< New GF2 pivot column, or-1 when this relation completes a dependency.
};
/// @brief One actual zero-parity dependency and its attempted factor extraction.
/// @par Example
/// @code
/// const dm::algorithms::QuadraticSieveDependency x{};
/// std::print("{}\n",x.gcd); // 0
/// @endcode
/// @see quadratic_sieve_run()
struct QuadraticSieveDependency {
  std::vector<std::size_t> members; ///< Actual relation indices including the current relation.
  std::uint64_t            X{};     ///< Actual modular product of member x values.
  std::uint64_t            Y{};     ///< Actual modular product of half-exponent prime powers.
  std::uint64_t            gcd{};   ///< Actual gcd(abs(X-Y),n), possibly trivial1 or n.
};
/// @brief Owned result and actual selected work from the sieve producer.
/// @par Example
/// @code
/// const dm::algorithms::QuadraticSieveResult x{};
/// std::print("{}\n",x.found); // false
/// @endcode
/// @see quadratic_sieve_run()
struct QuadraticSieveResult {
  std::uint64_t factor{}; ///< Nontrivial factor when found; otherwise n.
  bool          found{};  ///< Whether1<factor<n and it divides n.
  std::vector<std::uint64_t>
      factor_base; ///< Actual retained base, empty for even/square shortcuts.
  std::vector<QuadraticSieveRelation>
      relations; ///< Actual smooth relations, populated only by the trace.
  std::vector<QuadraticSieveDependency>
              dependencies;   ///< Actual square dependencies, only for the trace.
  std::size_t windows{};      ///< Actual sieve windows allocated.
  std::size_t sieve_values{}; ///< Actual Q values allocated/sieved.
  std::size_t
      smooth_relations{}; ///< Actual accepted full-smooth relations, including a successful final one.
  std::size_t divisions{}; ///< Actual exact prime divisions in sieving and exponent reconstruction.
  std::size_t xor_words{}; ///< Actual parity/composition word XOR updates during elimination.
  std::size_t dependency_tests{}; ///< Actual modular square dependency gcd attempts.
};
/// @brief Attempt word-sized factorization with the shared bounded quadratic sieve.
/// Requires n>=2; handles even inputs and perfect squares directly, then seeks primes up to a base bound for which n is a quadratic residue. Sieves positive x*x-n, reconstructs full smooth exponents and eliminates their parity over GF2. A dependency yields X^2=Y^2 mod n and tests gcd(abs(X-Y),n). Returns n on budget exhaustion or trivial extraction, which does not prove primality. Keeps the historical automatic bounds and120-window budget. Signed128 squares are safe under hard window/base bounds; modular multiplication uses unsigned128. This bounded uint64 educational implementation is not a general arbitrary-precision/subexponential performance guarantee. Search cost depends on actual windows, base, divisions and elimination; caller input unchanged, all outputs owned.
/// @param n Integer at least2.
/// @param options Validated finite window/base, cumulative sieve allocation and history policies.
/// @return Attempted factor, found flag, actual base and selected producer work.
/// @throws std::invalid_argument n<2 or invalid hard window/base policy.
/// @throws std::length_error Cumulative sieve or representable history capacity exceeded.
/// @par Example
/// @code
/// dm::algorithms::QuadraticSieveOptions o; o.factor_base_bound=3;
/// auto r=dm::algorithms::quadratic_sieve_run(77,o);
/// std::print("{}\n",r.factor); // 7
/// @endcode
/// @par References
/// - C. Pomerance (1982), Analysis and comparison of some integer factoring algorithms, https://math.dartmouth.edu/~carlp/PDF/analysiscomparison.pdf
/// @see quadratic_sieve_trace()
QuadraticSieveResult quadratic_sieve_run(
    std::uint64_t                n,
    const QuadraticSieveOptions& options = {});
/// @brief Capture actual full-smooth relations, pivot columns and square dependencies.
/// Same single producer as quadratic_sieve_run(). A relation costs exponent count+2 entries; a dependency costs member count+4 entries, admitted before copying. Histories include the successful final relation and all actual trivial extraction attempts. Storage is independently bounded; zero disables the retained-entry policy.
/// @param n Integer at least2.
/// @param options Base/search/allocation/history policies.
/// @return Attempted factor and actual relation/dependency evidence.
/// @throws std::invalid_argument Invalid integer or hard policy.
/// @throws std::length_error Sieve/history policy exceeded.
/// @par Example
/// @code
/// dm::algorithms::QuadraticSieveOptions o; o.factor_base_bound=3;
/// auto r=dm::algorithms::quadratic_sieve_trace(77,o);
/// std::print("{} {}\n",r.dependencies[0].X,r.dependencies[0].Y); // 9 2
/// @endcode
/// @see quadratic_sieve_run()
QuadraticSieveResult quadratic_sieve_trace(
    std::uint64_t                n,
    const QuadraticSieveOptions& options = {});
/// @brief Return an attempted factor with historical automatic search parameters.
/// Signature preserved. n<2 now throws rather than entering invalid logarithmic arithmetic. A returned n means no nontrivial factor was found; it is not a primality certificate. Uses checked default policies.
/// @param n Integer at least2.
/// @return Nontrivial factor, or n if none was found.
/// @throws std::invalid_argument n<2.
/// @throws std::length_error Default cumulative sieve policy exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::quadratic_sieve(49)); // 7
/// @endcode
/// @see quadratic_sieve_run()
std::uint64_t quadratic_sieve(std::uint64_t n);
} // namespace datamunge::algorithms
