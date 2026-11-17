#pragma once
/// @file index_calculus.hpp
/// @brief Checked finite-field index calculus with reusable tables and actual relations.
#include <cstddef>
#include <cstdint>
#include <vector>
namespace datamunge::algorithms {
/// @brief Verified discrete logarithm or an explicit unsuccessful bounded search.
/// Legacy aggregate order/types preserved. Success means g^x=h modulo p;
/// failure is not proof of nonexistence. A valid generator has a logarithm for
/// every nonzero target; bounded relation/rank/query work may still fail.
/// @par Example
/// @code
/// const auto r=dm::algorithms::index_calculus(2,4,5);
/// std::print("{} {}\n",r.x,r.found); // 2 true
/// @endcode
/// @see index_calculus()
struct DiscreteLogResult {
  std::uint64_t x{0};         ///< Canonical exponent in [0,p-2] on success; zero on failure.
  bool          found{false}; ///< Whether exponentiation verified the returned exponent.
};
/// @brief Actual bounded precomputation outcome.
/// Rank deficiency or a relation limit is not a discrete-log nonexistence proof.
/// The complete generator domain is validated before collection.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::IndexCalculusStatus::ready==dm::algorithms::IndexCalculusStatus::ready); // true
/// @endcode
/// @see IndexCalculusTable
enum class IndexCalculusStatus {
  ready,          ///< Full prime-power/CRT solution and every factor-base logarithm verified.
  relation_limit, ///< Too few accepted relations for the factor-base variable count.
  rank_deficient, ///< At least one prime-field system lacked a unique full-column-rank solution.
  verification_failed ///< Exact reconstruction or exponent verification failed; table must not be used.
};
/// @brief Finite work budgets and complete allocation/history admission policies.
/// Attempt budgets are exact counts, zero means no attempts. Optional allocation
/// caps use zero to disable. A trace relation/stage costs 1+K logical entries,
/// one record plus the complete K-dimensional slot budget, also reserved for
/// a failed stage with an empty solution. No rejected trial is retained.
/// History cap excludes ordinary matrices, table/output arrays, seed/scalar
/// metadata, vector capacity and byte/peak RAM. Untraced preparation ignores it.
/// @par Example
/// @code
/// const dm::algorithms::IndexCalculusOptions o;
/// std::print("{} {}\n",o.relation_attempts,o.relation_surplus); // 4000000 20
/// @endcode
/// @see index_calculus_prepare_trace()
struct IndexCalculusOptions {
  std::uint64_t factor_base_bound{
      0}; ///< Prime bound B, zero for the legacy adaptive bound clamped to [30,4000], then below p.
  std::size_t relation_attempts{
      4000000}; ///< Maximum actual randomized g-power trials; zero disables collection.
  std::size_t relation_surplus{
      20}; ///< Requested accepted rows beyond K; collection stops at K+surplus or budget.
  std::size_t max_sieve_entries{
      1000000}; ///< Complete B+1 Boolean sieve positions, zero disables cap.
  std::size_t max_matrix_cells{
      1000000}; ///< Complete (K+surplus)*K matrix cells, before collection; zero disables.
  std::size_t max_history_entries{
      1000000}; ///< Actual relation/stage logical entries admitted before append, zero disables.
};
/// @brief Selected actual precomputation operations, excluding validation/factorization.
/// Modular exponentiation internals, RNG shifts, sieve work, allocations/copies,
/// comparisons and final exponent checks are outside these selected totals.
/// Counter overflow throws. They are operation observations, not CPU time.
/// @par Example
/// @code
/// const dm::algorithms::IndexCalculusWork w;std::print("{}\n",w.relation_attempts); // 0
/// @endcode
/// @see index_calculus_prepare()
struct IndexCalculusWork {
  std::size_t relation_attempts{0}; ///< Actual generated exponents/residue smoothness trials.
  std::size_t accepted_relations{
      0}; ///< Actual fully factor-base-smooth residues retained in the matrix.
  std::size_t divisibility_tests{
      0}; ///< Actual remainder tests during complete factor-base trial division.
  std::size_t field_pivots{
      0}; ///< Actual nonzero prime-field pivots, including repeated lift solves.
  std::size_t row_eliminations{
      0}; ///< Actual other-row reductions with a nonzero pivot-column coefficient.
};
/// @brief Reusable verified factor-base logarithms and deterministic query-start state.
/// Public value storage owns all arrays. Query validates group-factor data,
/// factor-base ordering/primality, dimensions and every supplied logarithm before
/// use, so editing values does not bypass domain checks. Validation has its own
/// cost and is not counted as target search. No hidden cache or shared state.
/// @par Example
/// @code
/// const auto t=dm::algorithms::index_calculus_prepare(2,5);
/// std::print("{} {}\n",t.factor_base.size(),t.logarithms.size()); // 2 2
/// @endcode
/// @see index_calculus_prepare(), index_calculus_query()
struct IndexCalculusTable {
  std::uint64_t p{0};     ///< Validated prime modulus after preparation.
  std::uint64_t g{0};     ///< Validated normalized primitive generator.
  std::uint64_t bound{0}; ///< Actual complete prime bound B<p.
  std::vector<std::uint64_t>
      order_factors; ///< Complete sorted prime factors of p-1 with multiplicity.
  std::vector<std::uint64_t> factor_base; ///< Sorted primes <=B and strictly below p.
  std::vector<std::uint64_t>
                logarithms; ///< Canonical verified base-g logs, populated only for ready tables.
  std::uint64_t query_state{
      0}; ///< Actual xorshift state after relation collection; each query starts a copy here.
  IndexCalculusStatus status{IndexCalculusStatus::relation_limit}; ///< Actual preparation outcome.
  IndexCalculusWork
      work; ///< Actual selected producing work, excluding group validation/factorization.
};
/// @brief One accepted smooth relation captured at its actual collection boundary.
/// g^exponent modulo p equals residue and the ordinary integer product of
/// factor-base primes raised to powers. Failed nonsmooth trials are not stored.
/// @par Example
/// @code
/// const auto t=dm::algorithms::index_calculus_prepare_trace(2,5);
/// std::print("{}\n",t.relations.empty()); // false
/// @endcode
/// @see index_calculus_prepare_trace()
struct IndexCalculusRelation {
  std::size_t                attempt{0};  ///< One-based actual attempted-trial position.
  std::uint64_t              exponent{0}; ///< Actual sampled exponent in [1,p-2], for p>2.
  std::uint64_t              residue{0};  ///< Actual normalized nonzero g-power residue.
  std::vector<std::uint64_t> powers;      ///< Complete ordinary nonnegative factor-base exponents.
};
/// @brief Actual prime-field or lifted prime-power solution boundary.
/// Each stage solves modulo the prime field; modulus names the resulting
/// q^j solution modulus after a successful correction lift. A failed rank stage
/// has empty solution. No elimination is performed by dividing nonunits modulo p-1.
/// @par Example
/// @code
/// const auto t=dm::algorithms::index_calculus_prepare_trace(2,5);
/// std::print("{}\n",t.stages.back().modulus); // 4
/// @endcode
/// @see index_calculus_prepare_trace()
struct IndexCalculusStage {
  std::uint64_t prime{0};   ///< Prime q defining the field solve.
  std::uint64_t modulus{0}; ///< Resulting q^j modulus for this actual base/lift stage.
  std::size_t   rank{0};    ///< Actual prime-field column rank of the solved matrix.
  std::vector<std::uint64_t>
      solution; ///< Actual canonical solution modulo modulus, or empty on failed rank.
};
/// @brief Owned table plus actual accepted relation and linear-stage histories.
/// Records are captured inside the producing precomputation, not by replay.
/// Additional history storage is O((accepted rows+stages)*K); logical caps and
/// exclusions are documented by IndexCalculusOptions.
/// @par Example
/// @code
/// const auto t=dm::algorithms::index_calculus_prepare_trace(2,5);
/// std::print("{}\n",t.table.status==dm::algorithms::IndexCalculusStatus::ready); // true
/// @endcode
/// @see index_calculus_prepare_trace()
struct IndexCalculusPreparationTrace {
  IndexCalculusTable                 table;     ///< Final table from this same execution.
  std::vector<IndexCalculusRelation> relations; ///< Actual accepted rows in collection order.
  std::vector<IndexCalculusStage> stages; ///< Actual prime-field and prime-power lift boundaries.
};
/// @brief Actual bounded target-search result and its successful smooth relation.
/// Trivial h=1 or h=g uses no random trial, leaves relation fields empty/zero.
/// An unready valid table gives failure for other targets. Each call copies the
/// stored query RNG state; queries do not mutate a table or each other.
/// @par Example
/// @code
/// const auto t=dm::algorithms::index_calculus_prepare(2,5);
/// const auto q=dm::algorithms::index_calculus_query(t,4);
/// std::print("{} {}\n",q.result.x,q.result.found); // 2 true
/// @endcode
/// @see index_calculus_query()
struct IndexCalculusQuery {
  DiscreteLogResult result;                ///< Exponent verified against the normalized target.
  std::size_t       attempts{0};           ///< Actual shifted-target trials performed.
  std::size_t       divisibility_tests{0}; ///< Actual target-search trial-division remainder tests.
  std::uint64_t     shift{
      0}; ///< Successful actual shift s, zero if no captured successful nontrivial relation.
  std::uint64_t residue{0}; ///< Successful actual h*g^s mod p, zero for trivial/failure.
  std::vector<std::uint64_t>
      powers; ///< Successful smooth residue's factor-base exponents; empty otherwise.
};
/// @brief Prepare reusable finite-field factor-base logarithms.
/// Requires prime p>=2 and a nonzero primitive generator g modulo p. Complete
/// group-order factorization certifies generator order; it is additional work,
/// not part of a claimed index-calculus runtime. Sieve primes are strictly below
/// p. Collect at most K+surplus rows within the attempt budget, solve modulo each
/// prime factor, lift to its full prime power, combine by CRT and exponent-check
/// every logarithm. Full column rank is required in every prime field. No generic
/// discrete-log fallback. Bounded failure is an explicit status, not nonexistence.
/// Memory includes O(B+(K+surplus)*K) ordinary sieve/matrix and transient field
/// copies; tracing adds actual retained records. No mutation, shared state or
/// retained input references. The fixed B cap/batch/budgets do not implement an
/// asymptotically tuned subexponential or production cryptographic solver.
/// @param g Generator, normalized modulo p after primality validation.
/// @param p Prime uint64 modulus, at least two.
/// @param seed Xorshift seed, default 0x1D2C3B4A59687766; zero substitutes 0x9E3779B97F4A7C15.
/// @param options Bound/budgets/admission policies; default reproduces the legacy adaptive rule and trial count.
/// @return Owned verified or explicitly failed table.
/// @throws std::invalid_argument On nonprime p, zero or nongenerator g, or explicit B<2 for p>2.
/// @throws std::length_error On complete sieve/matrix/history cap, count overflow or capacity exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::index_calculus_prepare(2,5);
/// std::print("{}\n",t.status==dm::algorithms::IndexCalculusStatus::ready); // true
/// @endcode
/// @par References
/// - A. J. Menezes, P. C. van Oorschot and S. A. Vanstone (1996), Handbook of
///   Applied Cryptography, Algorithm 3.68; https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
///   This uint64 finite-field variant uses prime-power lifting and CRT for the
///   composite group order, a bounded fixed relation batch and xorshift trials.
/// @see IndexCalculusOptions, index_calculus_query()
IndexCalculusTable index_calculus_prepare(
    std::uint64_t        g,
    std::uint64_t        p,
    std::uint64_t        seed    = 0x1D2C3B4A59687766ULL,
    IndexCalculusOptions options = {});
/// @brief Capture actual preparation of finite-field factor-base logarithms.
/// Requires prime p>=2 and a nonzero primitive generator g modulo p. Complete
/// group-order factorization certifies generator order; it is additional work,
/// not part of a claimed index-calculus runtime. Sieve primes are strictly below
/// p. Collect at most K+surplus rows within the attempt budget, solve modulo each
/// prime factor, lift to its full prime power, combine by CRT and exponent-check
/// every logarithm. Full column rank is required in every prime field. No generic
/// discrete-log fallback. Bounded failure is an explicit status, not nonexistence.
/// Memory includes O(B+(K+surplus)*K) ordinary sieve/matrix and transient field
/// copies; tracing adds actual retained records. No mutation, shared state or
/// retained input references. The fixed B cap/batch/budgets do not implement an
/// asymptotically tuned subexponential or production cryptographic solver.
/// @param g Generator, normalized modulo p after primality validation.
/// @param p Prime uint64 modulus, at least two.
/// @param seed Xorshift seed, default 0x1D2C3B4A59687766; zero substitutes 0x9E3779B97F4A7C15.
/// @param options Bound/budgets/admission policies; default reproduces the legacy adaptive rule and trial count.
/// @return Owned actual relations/stages and final table.
/// @throws std::invalid_argument On nonprime p, zero or nongenerator g, or explicit B<2 for p>2.
/// @throws std::length_error On complete sieve/matrix/history cap, count overflow or capacity exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::index_calculus_prepare_trace(2,5);
/// std::print("{}\n",t.table.status==dm::algorithms::IndexCalculusStatus::ready); // true
/// @endcode
/// @par References
/// - A. J. Menezes, P. C. van Oorschot and S. A. Vanstone (1996), Handbook of
///   Applied Cryptography, Algorithm 3.68; https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
///   This uint64 finite-field variant uses prime-power lifting and CRT for the
///   composite group order, a bounded fixed relation batch and xorshift trials.
/// @see IndexCalculusOptions, index_calculus_query()
IndexCalculusPreparationTrace index_calculus_prepare_trace(
    std::uint64_t        g,
    std::uint64_t        p,
    std::uint64_t        seed    = 0x1D2C3B4A59687766ULL,
    IndexCalculusOptions options = {});
/// @brief Search for one target logarithm using a validated reusable table.
/// Normalizes nonzero h after complete table/group checks. Trivial h=1 or g is
/// solved even for an unready valid table, without trials. Other targets need a
/// ready table and a successful smooth shift within max_attempts. The complete
/// table is checked on every public query, including each supplied logarithm;
/// this validation cost is separate from attempts. No refactoring of the group
/// order or rebuilding of the relation matrix is performed. Search costs
/// O(attempts*(K divisions plus log p modular powering)) and O(K) work/output
/// storage, apart from table checks. No mutation/retained references/shared state.
/// @param table Borrowed complete ready or bounded-failure precomputation value.
/// @param h Nonzero target modulo table.p, normalized before search.
/// @param max_attempts Actual target trial budget, default four million; zero means no trials.
/// @return Owned verified result, actual trial counts and successful relation.
/// @throws std::invalid_argument On zero target or malformed/unverified table/group/logarithms.
/// @throws std::length_error On counter overflow or capacity exhaustion.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto t=dm::algorithms::index_calculus_prepare(2,5);
/// std::print("{}\n",dm::algorithms::index_calculus_query(t,4).result.x); // 2
/// @endcode
/// @par References
/// - A. J. Menezes, P. C. van Oorschot and S. A. Vanstone (1996), Handbook of
///   Applied Cryptography, Algorithm 3.68; https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
///   This uint64 finite-field variant uses prime-power lifting and CRT for the
///   composite group order, a bounded fixed relation batch and xorshift trials.
/// @see IndexCalculusQuery, index_calculus_prepare()
IndexCalculusQuery index_calculus_query(
    const IndexCalculusTable& table,
    std::uint64_t             h,
    std::size_t               max_attempts = 4000000);
/// @brief Solve one finite-field logarithm with the default bounded index-calculus policy.
/// Legacy signature/result layout and RNG stream retained for ordinary inputs.
/// Validates prime p, primitive nonzero generator and nonzero target before any
/// shortcut. Trivial h=1 or g returns immediately after group validation. Other
/// targets use the same preparation/query kernels; no brute-force/BSGS fallback.
/// Corrected factor bases exclude primes >=p and exact prime-power/CRT solves
/// avoid invalid composite-modulus field assumptions. Bounded failure remains
/// possible; found=false is not a proof that the logarithm does not exist.
/// No mutation, retained references or shared state. See preparation/query for
/// costs; this wrapper performs fresh precomputation for each nontrivial target.
/// @param g Generator normalized modulo p after prime validation.
/// @param h Nonzero target normalized modulo p.
/// @param p Prime uint64 modulus, at least two.
/// @param seed Deterministic xorshift seed, default 0x1D2C3B4A59687766; zero substitutes fixed nonzero seed.
/// @return Owned canonical verified exponent or bounded failure.
/// @throws std::invalid_argument On invalid prime/generator/target domain.
/// @throws std::length_error On default complete admission caps, count overflow or exhausted capacity.
/// @throws std::bad_alloc On allocation failure.
/// @par Example
/// @code
/// const auto r=dm::algorithms::index_calculus(2,4,5);
/// std::print("{} {}\n",r.x,r.found); // 2 true
/// @endcode
/// @par References
/// - A. J. Menezes, P. C. van Oorschot and S. A. Vanstone (1996), Handbook of
///   Applied Cryptography, Algorithm 3.68; https://cacr.uwaterloo.ca/hac/about/chap3.pdf.
///   This uint64 finite-field variant uses prime-power lifting and CRT for the
///   composite group order, a bounded fixed relation batch and xorshift trials.
/// @see index_calculus_prepare(), index_calculus_query()
DiscreteLogResult index_calculus(
    std::uint64_t g,
    std::uint64_t h,
    std::uint64_t p,
    std::uint64_t seed = 0x1D2C3B4A59687766ULL);
} // namespace datamunge::algorithms
