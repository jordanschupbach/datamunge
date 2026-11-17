#pragma once
/// @file shamir_secret_sharing.hpp
/// @brief Checked exact prime-field sharing and actual polynomial/interpolation histories.
#include <cstddef>
#include <cstdint>
#include <vector>
/// @brief Canonical checked native producers and owned algorithm values.
namespace datamunge::algorithms {
/// @brief Owned nonzero field abscissa and its polynomial value.
/// @par Example
/// @code
/// const dm::algorithms::SecretShare value{};
/// std::print("{}\n",value.x); // 0
/// @endcode
struct SecretShare {
  std::uint64_t x{}; ///< Evaluation coordinate in [1,prime).
  std::uint64_t y{}; ///< Actual evaluation residue in [0,prime).
};
/// @brief Complete input and actual arithmetic/history bounds; zero disables a size cap.
/// @par Example
/// @code
/// const dm::algorithms::ShamirOptions value{};
/// std::print("{}\n",value.max_coefficients); // 1024
/// @endcode
struct ShamirOptions {
  std::size_t max_shares{1000000};    ///< Complete split output or combine input count.
  std::size_t max_coefficients{1024}; ///< Complete supplied nonconstant coefficient count.
  std::size_t max_products{
      20000000}; ///< Actual modular products, including delegated inverse powering.
  std::size_t max_trace_entries{
      1000000}; ///< Actual captured coefficient scalars and polynomial/interpolation records.
};
/// @brief One actual ascending-power polynomial accumulation.
/// @par Example
/// @code
/// const dm::algorithms::ShamirEvaluation value{};
/// std::print("{}\n",value.degree); // 0
/// @endcode
struct ShamirEvaluation {
  std::size_t   share_index{}; ///< Actual zero-based output share.
  std::size_t   degree{};      ///< Actual positive polynomial degree.
  std::uint64_t power{};       ///< Actual x^degree modulo prime.
  std::uint64_t coefficient{}; ///< Actual supplied field coefficient.
  std::uint64_t before{};      ///< Actual evaluation before adding this term.
  std::uint64_t term{};        ///< Actual coefficient times power modulo prime.
  std::uint64_t after{};       ///< Actual modular sum.
};
/// @brief Owned actual evaluated shares and selected arithmetic/history.
/// @par Example
/// @code
/// const dm::algorithms::ShamirSplitRun value{};
/// std::print("{}\n",value.modular_products); // 0
/// @endcode
struct ShamirSplitRun {
  std::vector<SecretShare> shares; ///< Actual evaluations at x=1..n.
  std::size_t modular_products{}; ///< Actual two modular products per accumulated nonconstant term.
  std::vector<std::uint64_t>    coefficients; ///< Optional actual copied nonconstant coefficients.
  std::vector<ShamirEvaluation> steps;        ///< Optional actual chronological evaluation records.
};
/// @brief One actual Lagrange-at-zero contribution from a supplied share.
/// @par Example
/// @code
/// const dm::algorithms::ShamirTerm value{};
/// std::print("{}\n",value.index); // 0
/// @endcode
struct ShamirTerm {
  std::size_t   index{};       ///< Actual supplied share index.
  std::uint64_t numerator{};   ///< Actual product of other x coordinates.
  std::uint64_t denominator{}; ///< Actual product of other x minus this x.
  std::uint64_t inverse{};     ///< Actual delegated Fermat inverse of denominator.
  std::uint64_t basis{};       ///< Actual numerator times inverse modulo prime.
  std::uint64_t term{};        ///< Actual y times Lagrange basis.
  std::uint64_t partial{};     ///< Actual accumulated interpolation value.
};
/// @brief Owned actual interpolation at zero, without authenticating a declared threshold.
/// @par Example
/// @code
/// const dm::algorithms::ShamirCombineRun value{};
/// std::print("{}\n",value.secret); // 0
/// @endcode
struct ShamirCombineRun {
  std::uint64_t secret{}; ///< Actual P(0) of the unique interpolant through supplied points.
  std::size_t   modular_products{}; ///< Actual modular products including inverse powering.
  std::size_t inverse_products{}; ///< Actual subset reported by canonical modular-power inversions.
  std::vector<ShamirTerm> terms;  ///< Optional actual contribution records.
};
/// @brief Evaluate checked prime-field secret shares.
/// Validates prime exactly using the canonical uint64 Miller-Rabin driver; requires secret and every supplied coefficient in [0,prime), coefficient_count+1<=n<prime. Coefficients are supplied unchanged, may include zero and are not sampled here. Evaluates ascending powers as the historical implementation, with exact modular products and overflow-safe widened modular addition. Input borrowed unchanged, outputs own storage. O(n*coefficient_count) arithmetic plus bounded primality work. Complete input and trace geometry checked before execution; actual product budget checked before work. No constant-time guarantee. Threshold privacy requires independently uniformly sampled secret coefficients; deterministic public worked coefficients do not provide that security guarantee.
/// @param secret Field secret in [0,prime).
/// @param n Complete share count, at least coefficient_count+1 and below prime.
/// @param coeffs Borrowed nonconstant field coefficients, a1 onward.
/// @param prime Exactly prime uint64 modulus.
/// @param options Complete input and actual work/history bounds.
/// @return Owned actual shares, product work and optional coefficient/term histories.
/// @throws std::invalid_argument Composite modulus, invalid field values or threshold/share geometry.
/// @throws std::length_error Complete input/work/history capacity or policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shamir_split_run(3,3,{2},17);
/// std::print("{} {}\n",r.shares[0].y,r.shares[2].y); // 5 9
/// @endcode
/// @par References
/// - Shamir (1979), How to Share a Secret, Communications of the ACM22,612-613. doi:10.1145/359168.359176.
/// @see shamir_combine_run()
ShamirSplitRun shamir_split_run(
    std::uint64_t                     secret,
    unsigned                          n,
    const std::vector<std::uint64_t>& coeffs,
    std::uint64_t                     prime,
    const ShamirOptions&              options = {});
/// @brief Evaluate checked prime-field secret shares retaining actual polynomial terms.
/// Validates prime exactly using the canonical uint64 Miller-Rabin driver; requires secret and every supplied coefficient in [0,prime), coefficient_count+1<=n<prime. Coefficients are supplied unchanged, may include zero and are not sampled here. Evaluates ascending powers as the historical implementation, with exact modular products and overflow-safe widened modular addition. Input borrowed unchanged, outputs own storage. O(n*coefficient_count) arithmetic plus bounded primality work. Complete input and trace geometry checked before execution; actual product budget checked before work. No constant-time guarantee. Threshold privacy requires independently uniformly sampled secret coefficients; deterministic public worked coefficients do not provide that security guarantee.
/// @param secret Field secret in [0,prime).
/// @param n Complete share count, at least coefficient_count+1 and below prime.
/// @param coeffs Borrowed nonconstant field coefficients, a1 onward.
/// @param prime Exactly prime uint64 modulus.
/// @param options Complete input and actual work/history bounds.
/// @return Owned actual shares, product work and optional coefficient/term histories.
/// @throws std::invalid_argument Composite modulus, invalid field values or threshold/share geometry.
/// @throws std::length_error Complete input/work/history capacity or policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shamir_split_trace(3,3,{2},17);
/// std::print("{} {}\n",r.shares[0].y,r.shares[2].y); // 5 9
/// @endcode
/// @par References
/// - Shamir (1979), How to Share a Secret, Communications of the ACM22,612-613. doi:10.1145/359168.359176.
/// @see shamir_combine_run()
ShamirSplitRun shamir_split_trace(
    std::uint64_t                     secret,
    unsigned                          n,
    const std::vector<std::uint64_t>& coeffs,
    std::uint64_t                     prime,
    const ShamirOptions&              options = {});
/// @brief Interpolate checked distinct nonzero prime-field points at zero.
/// Validates exact primality, nonempty points, each x in [1,prime), y in [0,prime) and distinct x before indexed arithmetic. Builds actual numerator product x_j and denominator product (x_j-x_i), then delegates ONE canonical mod_pow_trace(den,prime-2,prime) per inverse. Subtractions/additions avoid uint64 wrap at large primes. O(k^2+k*log prime) word products plus primality work; O(k) validation/result history. This routine does not know the original threshold or authenticate shares: insufficient points still define a different interpolant, and a returned value alone cannot certify successful reconstruction. Inputs borrowed unchanged, output owned; no shared state or constant-time guarantee.
/// @param shares Borrowed nonempty distinct nonzero field points.
/// @param prime Exactly prime uint64 modulus used for sharing.
/// @param options Complete input and actual product/history bounds.
/// @return Owned actual interpolation value, product counts and optional actual terms.
/// @throws std::invalid_argument Nonprime modulus, empty/duplicate points or out-of-field coordinate/value.
/// @throws std::length_error Complete input/work/history capacity or policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shamir_combine_run({{1,5},{3,9}},17);
/// std::print("{}\n",r.secret); // 3
/// @endcode
/// @par References
/// - Shamir (1979), How to Share a Secret, Communications of the ACM22,612-613. doi:10.1145/359168.359176.
/// @see shamir_split_run()
ShamirCombineRun shamir_combine_run(
    const std::vector<SecretShare>& shares,
    std::uint64_t                   prime,
    const ShamirOptions&            options = {});
/// @brief Interpolate checked distinct nonzero prime-field points at zero with actual Lagrange contributions.
/// Validates exact primality, nonempty points, each x in [1,prime), y in [0,prime) and distinct x before indexed arithmetic. Builds actual numerator product x_j and denominator product (x_j-x_i), then delegates ONE canonical mod_pow_trace(den,prime-2,prime) per inverse. Subtractions/additions avoid uint64 wrap at large primes. O(k^2+k*log prime) word products plus primality work; O(k) validation/result history. This routine does not know the original threshold or authenticate shares: insufficient points still define a different interpolant, and a returned value alone cannot certify successful reconstruction. Inputs borrowed unchanged, output owned; no shared state or constant-time guarantee.
/// @param shares Borrowed nonempty distinct nonzero field points.
/// @param prime Exactly prime uint64 modulus used for sharing.
/// @param options Complete input and actual product/history bounds.
/// @return Owned actual interpolation value, product counts and optional actual terms.
/// @throws std::invalid_argument Nonprime modulus, empty/duplicate points or out-of-field coordinate/value.
/// @throws std::length_error Complete input/work/history capacity or policy exceeded.
/// @par Example
/// @code
/// auto r=dm::algorithms::shamir_combine_trace({{1,5},{3,9}},17);
/// std::print("{}\n",r.secret); // 3
/// @endcode
/// @par References
/// - Shamir (1979), How to Share a Secret, Communications of the ACM22,612-613. doi:10.1145/359168.359176.
/// @see shamir_split_run()
ShamirCombineRun shamir_combine_trace(
    const std::vector<SecretShare>& shares,
    std::uint64_t                   prime,
    const ShamirOptions&            options = {});
/// @brief Return checked evaluated shares through the preserved legacy signature.
/// Same producer/default bounds without optional history.
/// @param secret Field secret.
/// @param n Complete share count.
/// @param coeffs Nonconstant field coefficients.
/// @param prime Exactly prime modulus.
/// @return Owned field shares.
/// @throws std::invalid_argument Invalid prime, field values or threshold geometry.
/// @throws std::length_error Default policy/capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shamir_split(3,3,{2},17)[1].y); // 7
/// @endcode
/// @par References
/// - Shamir (1979), How to Share a Secret, Communications of the ACM22,612-613. doi:10.1145/359168.359176.
/// @see shamir_split_run()
std::vector<SecretShare> shamir_split(
    std::uint64_t                     secret,
    unsigned                          n,
    const std::vector<std::uint64_t>& coeffs,
    std::uint64_t                     prime);
/// @brief Return checked interpolation at zero through the preserved signature.
/// Same producer/default bounds; does not validate an unknown original threshold.
/// @param shares Nonempty distinct nonzero field points.
/// @param prime Exactly prime modulus.
/// @return Actual zero-coordinate interpolation residue.
/// @throws std::invalid_argument Invalid prime or points.
/// @throws std::length_error Default policy/capacity exceeded.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::shamir_combine({{1,5},{3,9}},17)); // 3
/// @endcode
/// @par References
/// - Shamir (1979), How to Share a Secret, Communications of the ACM22,612-613. doi:10.1145/359168.359176.
/// @see shamir_combine_run()
std::uint64_t shamir_combine(const std::vector<SecretShare>& shares, std::uint64_t prime);
/// @brief Complete two observed shares to a quadratic with any specified candidate secret.
/// Returns the unique a1,a2 such that candidate+a1*x+a2*x^2 agrees with both nonzero distinct field points. This constructive ambiguity control shows every candidate can fit two observations; the information-theoretic privacy proof additionally assumes uniform independent hidden coefficients. Uses exact canonical modular products and powers; no random sampling or shared state.
/// @param observed Exactly two distinct nonzero valid field shares.
/// @param candidate Any field candidate for the constant term.
/// @param prime Exactly prime modulus.
/// @return Two owned field coefficients a1,a2.
/// @throws std::invalid_argument Invalid field, candidate or observed points.
/// @par Example
/// @code
/// auto c=dm::algorithms::shamir_quadratic_completion({{1,5},{2,9}},3,17);
/// std::print("{}\n",c); // [1, 1]
/// @endcode
/// @par References
/// - Shamir (1979), How to Share a Secret, Communications of the ACM22,612-613. doi:10.1145/359168.359176.
/// @see shamir_split_run()
std::vector<std::uint64_t> shamir_quadratic_completion(
    const std::vector<SecretShare>& observed,
    std::uint64_t                   candidate,
    std::uint64_t                   prime);
} // namespace datamunge::algorithms
