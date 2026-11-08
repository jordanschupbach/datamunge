#pragma once
/// @file freivalds.hpp
/// @brief Checked floating-point Freivalds verification, actual traces and bounded experiments.
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned row-major double matrix used by the verifier.
/// An empty outer vector represents0x0; nonempty zero-width rows represent m x0. Public algorithms validate every row and finite value.
/// @par Example
/// @code
/// dm::algorithms::FreivaldsMatrix a{{1}};
/// std::cout << a.size(); // 1
/// @endcode
/// @see freivalds_trace()
using FreivaldsMatrix = std::vector<std::vector<double>>;
/// @brief Absolute residual threshold retained by the numerical verifier.
/// Acceptance tests fabs(A(Br)[i]-Cr[i]) <=1e-6. This is not an exact equality certificate or a scale-relative tolerance.
/// @par Example
/// @code
/// std::cout << dm::algorithms::kFreivaldsTolerance; // 1e-06
/// @endcode
/// @see freivalds_trace()
inline constexpr double kFreivaldsTolerance = 1e-6;
/// @brief Actual kernel work, excluding validation, allocation and capture.
/// A term means one multiply and accumulate, not a hardware instruction. All three matvecs finish before row comparisons. Failed rounds count as completed.
/// @par Example
/// @code
/// dm::algorithms::FreivaldsCounts x;
/// std::cout << x.random_words; // 0
/// @endcode
/// @see freivalds_trace()
struct FreivaldsCounts {
    std::size_t kernel_calls{}; ///< Compatible kernel invocations, including zero-round calls.
    std::size_t rounds_completed{}; ///< Rounds actually performed, including a failed round.
    std::size_t random_words{}; ///< SplitMix64 words drawn, one per generated challenge bit.
    std::size_t provided_bits{}; ///< Supplied challenge bits consumed; no RNG words.
    std::size_t br_terms{}; ///< B times r multiply-accumulate terms.
    std::size_t abr_terms{}; ///< A times Br multiply-accumulate terms.
    std::size_t cr_terms{}; ///< C times r multiply-accumulate terms.
    std::size_t compared_rows{}; ///< Residual comparisons actually performed.
};
/// @brief Actual counts and input/RNG metadata for one call.
/// State advances modulo2^64. Supplied probes set seeded=false and states/seed=0; incompatible calls draw nothing.
/// @par Example
/// @code
/// dm::algorithms::FreivaldsWork x;
/// std::cout << x.rounds_completed; // 0
/// @endcode
/// @see freivalds_trace()
struct FreivaldsWork : public FreivaldsCounts {
    std::size_t rows{}; ///< Left and claimed-product row count m.
    std::size_t inner{}; ///< Left column count k.
    std::size_t columns{}; ///< Right column count n.
    int requested_rounds{}; ///< Requested rounds, including negative values.
    bool seeded{true}; ///< Whether challenges use SplitMix64.
    std::uint64_t seed{}; ///< Initial state, zero for supplied challenges.
    std::uint64_t final_state{}; ///< Actual state after the last draw.
};
/// @brief A numerical verdict with actual work and first failure.
/// SIZE_MAX failure indices mean no failed round. Compatible nonpositive-round calls accept vacuously; incompatible calls reject without rounds.
/// @par Example
/// @code
/// dm::algorithms::FreivaldsResult x;
/// std::cout << x.accepted; // 0
/// @endcode
/// @see freivalds_trace()
struct FreivaldsResult {
    bool compatible{}; ///< Whether all three rectangular dimensions agree.
    bool accepted{}; ///< Whether the numerical residual test accepted.
    std::size_t failed_round{std::numeric_limits<std::size_t>::max()}; ///< Zero-based first failed round, otherwise SIZE_MAX.
    std::size_t failed_row{std::numeric_limits<std::size_t>::max()}; ///< Zero-based first failed row, otherwise SIZE_MAX.
    FreivaldsWork work{}; ///< Actual work and metadata.
};
/// @brief Owned values from one actual performed round.
/// Differences contain only the tested row prefix, as absolute values; all matvec output rows exist even after a row0 mismatch. No replay or fabricated unchecked differences.
/// @par Example
/// @code
/// dm::algorithms::FreivaldsRound x;
/// std::cout << x.rv.size(); // 0
/// @endcode
/// @see freivalds_trace()
struct FreivaldsRound {
    std::size_t iteration{}; ///< Zero-based actual round.
    std::uint64_t state_before{}; ///< RNG state before this round; zero for supplied probe.
    std::uint64_t state_after{}; ///< RNG state after challenge generation; zero for supplied probe.
    std::vector<double> rv{}; ///< Actual binary challenge.
    std::vector<double> br{}; ///< Actual B times r.
    std::vector<double> abr{}; ///< Actual A times Br.
    std::vector<double> cr{}; ///< Actual C times r.
    std::vector<double> difference{}; ///< Absolute differences for compared rows only.
    bool accepted{true}; ///< Whether this round passed.
    std::size_t failing_row{std::numeric_limits<std::size_t>::max()}; ///< First mismatch row or SIZE_MAX.
};
/// @brief Owned inputs and actual round frames from one kernel execution.
/// All input matrices and frame vectors own storage independently. No partial result escapes an exception. Entry admission is conservative for all requested rounds, before input copies and RNG draws.
/// @par Example
/// @code
/// dm::algorithms::FreivaldsTrace x;
/// std::cout << x.frames.size(); // 0
/// @endcode
/// @see freivalds_trace()
struct FreivaldsTrace : public FreivaldsResult {
    FreivaldsMatrix a{}; ///< Owned a input.
    FreivaldsMatrix b{}; ///< Owned b input.
    FreivaldsMatrix c{}; ///< Owned c input.
    std::vector<FreivaldsRound> frames{}; ///< Actual frames, stopping after first mismatch.
};
/// @brief One requested prefix length and its deterministic seed count.
/// Rows retain caller order and duplicates. Trials0 has accepted0; no undefined fraction is stored.
/// @par Example
/// @code
/// dm::algorithms::FreivaldsCurveRow x;
/// std::cout << x.trials; // 0
/// @endcode
/// @see freivalds_trace()
struct FreivaldsCurveRow {
    int rounds{}; ///< Caller requested prefix length.
    std::size_t trials{}; ///< Number of seeds requested.
    std::size_t accepted{}; ///< Numerically accepted prefixes.
};
/// @brief Actual maximum-prefix outcome for one seed.
/// Each seed executes once at the largest nonnegative requested prefix. A requested prefix K passes iff compatible and K<=0, or no failure, or K<=failed_round.
/// @par Example
/// @code
/// dm::algorithms::FreivaldsSeedTrial x;
/// std::cout << x.rounds_completed; // 0
/// @endcode
/// @see freivalds_trace()
struct FreivaldsSeedTrial {
    std::uint64_t seed{}; ///< Seed used.
    bool accepted{}; ///< Maximum requested prefix verdict.
    std::size_t failed_round{std::numeric_limits<std::size_t>::max()}; ///< Zero-based first failed round, otherwise SIZE_MAX.
    std::size_t failed_row{std::numeric_limits<std::size_t>::max()}; ///< First failed row or SIZE_MAX.
    std::size_t rounds_completed{}; ///< Actual performed rounds, including failure.
};
/// @brief Deterministic once-per-seed experiment and actual aggregate counts.
/// These fixed pseudorandom seeds are not an IID sample or a statistical confidence interval. Empty round requests produce no rows or seed executions. Incompatible requests retain rejecting seed records with no kernel work.
/// @par Example
/// @code
/// dm::algorithms::FreivaldsSeedSweep x;
/// std::cout << x.seeds.size(); // 0
/// @endcode
/// @see freivalds_trace()
struct FreivaldsSeedSweep {
    bool compatible{}; ///< Dimension compatibility.
    std::vector<FreivaldsCurveRow> rows{}; ///< Requested lengths in caller order.
    std::vector<FreivaldsSeedTrial> seeds{}; ///< Maximum-prefix outcomes in increasing seed order.
    FreivaldsCounts work{}; ///< Sum of actual kernel counts, not independent rerun costs.
};
/// @brief Exhaustive supplied-binary-vector counts for the numerical verifier.
/// Enumeration uses column0 as the low bit. This audits the double tolerance test, not exact algebraic equality. Incompatible matrices have vectors0 and no kernel work.
/// @par Example
/// @code
/// dm::algorithms::FreivaldsBinaryAudit x;
/// std::cout << x.vectors; // 0
/// @endcode
/// @see freivalds_trace()
struct FreivaldsBinaryAudit {
    bool compatible{}; ///< Dimension compatibility.
    std::size_t vectors{}; ///< Number of vectors tested.
    std::size_t accepted{}; ///< Number accepted by the numerical test.
    std::size_t rejected{}; ///< Number rejected by the numerical test.
    FreivaldsCounts work{}; ///< Actual aggregate work, using supplied bits and no RNG.
};
/// @brief Original40x40 integer fixture and its exact representable product.
/// A and B use mt19937 seed3 and uniform integers[-9,9], all A then all B in row order. Product reuses strassen_multiply(); bounded integer arithmetic is exact for this fixture. Generator distribution mapping follows the C++ standard library implementation.
/// @par Example
/// @code
/// const auto x = dm::algorithms::freivalds_example_matrices();
/// std::cout << x.a.size(); // 40
/// @endcode
/// @see freivalds_trace()
struct FreivaldsExampleMatrices {
    FreivaldsMatrix a{}; ///< Left40x40 integer matrix.
    FreivaldsMatrix b{}; ///< Right40x40 integer matrix.
    FreivaldsMatrix product{}; ///< Exact40x40 integer product for this bounded fixture.
};
/// @brief Verify a claimed product using seeded double binary projections.
/// Every matrix must be rectangular with finite entries; all shapes then all values are validated before compatibility, nonpositive rounds or optional quotas. Wellformed incompatible dimensions return a rejecting result. No caller mutation or shared state. Actual nonfinite matvec accumulation or residual throws; no unused AB product or norms are evaluated. Empty outer matrices represent0x0. Compatible rounds<=0 accept without draws. Ordinary arithmetic order and fixed absolute tolerance are retained; small wrong products can pass and cancellation can reject an exactly correct product. The exact-arithmetic IID-bit theorem is not a general guarantee for this double/SplitMix implementation.
/// @param a Left m x k matrix.
/// @param b Right k x n matrix.
/// @param c Claimed m x n product.
/// @param rounds Requested number of rounds, negative values mean no checks. Default 20.
/// @param seed Initial wrapping SplitMix64 state; one word per bit. Default 0xD1B54A32D192ED03ULL.
/// @return Whether dimensions agree and all performed residual tests pass.
/// @throws std::invalid_argument Ragged/nonfinite input or malformed supplied challenge.
/// @throws std::length_error Count/container overflow or exceeded optional admission quota.
/// @throws std::overflow_error Actual nonfinite numerical arithmetic.
/// @par Example
/// @code
/// std::cout << dm::algorithms::freivalds_verify({{2}}, {{3}}, {{6}}); // 1
/// @endcode
/// @par References
/// - R. Freivalds (1979), Fast probabilistic algorithms, MFCS, pp57-69. doi:10.1007/3-540-09526-8_5. The paper uses two-point sign probes; this implementation uses binary probes and an absolute floating tolerance.
/// @see freivalds_verify(), freivalds_trace()
bool freivalds_verify(const FreivaldsMatrix& a, const FreivaldsMatrix& b, const FreivaldsMatrix& c, int rounds = 20, std::uint64_t seed = 0xD1B54A32D192ED03ULL);
/// @brief Verify and count actual numerical kernel work.
/// Every matrix must be rectangular with finite entries; all shapes then all values are validated before compatibility, nonpositive rounds or optional quotas. Wellformed incompatible dimensions return a rejecting result. No caller mutation or shared state. Actual nonfinite matvec accumulation or residual throws; no unused AB product or norms are evaluated. Empty outer matrices represent0x0. Compatible rounds<=0 accept without draws. Ordinary arithmetic order and fixed absolute tolerance are retained; small wrong products can pass and cancellation can reject an exactly correct product. The exact-arithmetic IID-bit theorem is not a general guarantee for this double/SplitMix implementation.
/// @param a Left m x k matrix.
/// @param b Right k x n matrix.
/// @param c Claimed m x n product.
/// @param rounds Requested number of rounds, negative values mean no checks. Default 20.
/// @param seed Initial wrapping SplitMix64 state; one word per bit. Default 0xD1B54A32D192ED03ULL.
/// @return Verdict, first failure and actual work, without captured inputs/frames.
/// @throws std::invalid_argument Ragged/nonfinite input or malformed supplied challenge.
/// @throws std::length_error Count/container overflow or exceeded optional admission quota.
/// @throws std::overflow_error Actual nonfinite numerical arithmetic.
/// @par Example
/// @code
/// const auto x = dm::algorithms::freivalds_verify_work({{2}}, {{3}}, {{6}}, 2);
/// std::cout << x.work.br_terms; // 2
/// @endcode
/// @par References
/// - R. Freivalds (1979), Fast probabilistic algorithms, MFCS, pp57-69. doi:10.1007/3-540-09526-8_5. The paper uses two-point sign probes; this implementation uses binary probes and an absolute floating tolerance.
/// @see freivalds_verify(), freivalds_trace()
FreivaldsResult freivalds_verify_work(const FreivaldsMatrix& a, const FreivaldsMatrix& b, const FreivaldsMatrix& c, int rounds = 20, std::uint64_t seed = 0xD1B54A32D192ED03ULL);
/// @brief Forecast whole requested trace admission without RNG or copies.
/// Every matrix must be rectangular with finite entries; all shapes then all values are validated before compatibility, nonpositive rounds or optional quotas. Wellformed incompatible dimensions return a rejecting result. No caller mutation or shared state. Actual nonfinite matvec accumulation or residual throws; no unused AB product or norms are evaluated. Empty outer matrices represent0x0. Compatible rounds<=0 accept without draws. Ordinary arithmetic order and fixed absolute tolerance are retained; small wrong products can pass and cancellation can reject an exactly correct product. The exact-arithmetic IID-bit theorem is not a general guarantee for this double/SplitMix implementation. Entries count input rows+cells, plus T*(1+n+k+3m) for compatible T=max(rounds,0); otherwise T0. Frame1 includes scalar metadata, vectors count elements. No allocation/counter metadata, temporary buffers or allocator bytes are charged. Checked arithmetic and vector capacities always apply; early rejection does not reduce admission.
/// @param a Left m x k matrix.
/// @param b Right k x n matrix.
/// @param c Claimed m x n product.
/// @param rounds Requested number of rounds, negative values mean no checks. Default 20.
/// @return Conservative retained entry upper bound, not bytes or peak RAM.
/// @throws std::invalid_argument Ragged/nonfinite input or malformed supplied challenge.
/// @throws std::length_error Count/container overflow or exceeded optional admission quota.
/// @throws std::overflow_error Actual nonfinite numerical arithmetic.
/// @par Example
/// @code
/// std::cout << dm::algorithms::freivalds_trace_entries({{1}}, {{1}}, {{2}}, 1); // 12
/// @endcode
/// @par References
/// - R. Freivalds (1979), Fast probabilistic algorithms, MFCS, pp57-69. doi:10.1007/3-540-09526-8_5. The paper uses two-point sign probes; this implementation uses binary probes and an absolute floating tolerance.
/// @see freivalds_verify(), freivalds_trace()
std::size_t freivalds_trace_entries(const FreivaldsMatrix& a, const FreivaldsMatrix& b, const FreivaldsMatrix& c, int rounds = 20);
/// @brief Capture actual projections with whole-trace admission before RNG and copies.
/// Every matrix must be rectangular with finite entries; all shapes then all values are validated before compatibility, nonpositive rounds or optional quotas. Wellformed incompatible dimensions return a rejecting result. No caller mutation or shared state. Actual nonfinite matvec accumulation or residual throws; no unused AB product or norms are evaluated. Empty outer matrices represent0x0. Compatible rounds<=0 accept without draws. Ordinary arithmetic order and fixed absolute tolerance are retained; small wrong products can pass and cancellation can reject an exactly correct product. The exact-arithmetic IID-bit theorem is not a general guarantee for this double/SplitMix implementation. Uses freivalds_trace_entries() for conservative whole-request admission. Cap0 disables only the optional quota. No partial trace escapes.
/// @param a Left m x k matrix.
/// @param b Right k x n matrix.
/// @param c Claimed m x n product.
/// @param rounds Requested number of rounds, negative values mean no checks. Default 20.
/// @param seed Initial wrapping SplitMix64 state; one word per bit. Default 0xD1B54A32D192ED03ULL.
/// @param max_entries Whole conservative trace entry cap;0 disables only this cap. Default 1000000.
/// @return Owned actual execution, including input copies and only performed frames.
/// @throws std::invalid_argument Ragged/nonfinite input or malformed supplied challenge.
/// @throws std::length_error Count/container overflow or exceeded optional admission quota.
/// @throws std::overflow_error Actual nonfinite numerical arithmetic.
/// @par Example
/// @code
/// const auto x = dm::algorithms::freivalds_trace({{1}}, {{1}}, {{1}}, 2);
/// std::cout << x.frames.size(); // 2
/// @endcode
/// @par References
/// - R. Freivalds (1979), Fast probabilistic algorithms, MFCS, pp57-69. doi:10.1007/3-540-09526-8_5. The paper uses two-point sign probes; this implementation uses binary probes and an absolute floating tolerance.
/// @see freivalds_verify(), freivalds_trace()
FreivaldsTrace freivalds_trace(const FreivaldsMatrix& a, const FreivaldsMatrix& b, const FreivaldsMatrix& c, int rounds = 20, std::uint64_t seed = 0xD1B54A32D192ED03ULL, std::size_t max_entries = 1000000);
/// @brief Inspect one supplied binary challenge without RNG draws.
/// Every matrix must be rectangular with finite entries; all shapes then all values are validated before compatibility, nonpositive rounds or optional quotas. Wellformed incompatible dimensions return a rejecting result. No caller mutation or shared state. Actual nonfinite matvec accumulation or residual throws; no unused AB product or norms are evaluated. Empty outer matrices represent0x0. Compatible rounds<=0 accept without draws. Ordinary arithmetic order and fixed absolute tolerance are retained; small wrong products can pass and cancellation can reject an exactly correct product. The exact-arithmetic IID-bit theorem is not a general guarantee for this double/SplitMix implementation. Validate every challenge bit (0 or1) and length n before admission even for incompatible input. Uses the one-round whole-trace bound; states are0 and seeded=false.
/// @param a Left m x k matrix.
/// @param b Right k x n matrix.
/// @param c Claimed m x n product.
/// @param challenge Exactly n binary bits, column order.
/// @param max_entries Whole conservative trace entry cap;0 disables only this cap. Default 1000000.
/// @return Owned one-round supplied-challenge trace, or no frames when incompatible.
/// @throws std::invalid_argument Ragged/nonfinite input or malformed supplied challenge.
/// @throws std::length_error Count/container overflow or exceeded optional admission quota.
/// @throws std::overflow_error Actual nonfinite numerical arithmetic.
/// @par Example
/// @code
/// const auto x = dm::algorithms::freivalds_probe({{1}}, {{1}}, {{2}}, {1});
/// std::cout << x.accepted; // 0
/// @endcode
/// @par References
/// - R. Freivalds (1979), Fast probabilistic algorithms, MFCS, pp57-69. doi:10.1007/3-540-09526-8_5. The paper uses two-point sign probes; this implementation uses binary probes and an absolute floating tolerance.
/// @see freivalds_verify(), freivalds_trace()
FreivaldsTrace freivalds_probe(const FreivaldsMatrix& a, const FreivaldsMatrix& b, const FreivaldsMatrix& c, const std::vector<std::uint8_t>& challenge, std::size_t max_entries = 1000000);
/// @brief Count all binary vectors with checked integer exponentiation.
/// Reject columns at least the size_t bit width immediately; no allocation or enumeration.
/// @param columns Number of binary coordinates.
/// @return Exactly2^columns, with0 columns yielding1.
/// @throws std::length_error The power of two cannot fit size_t.
/// @par Example
/// @code
/// std::cout << dm::algorithms::freivalds_binary_vectors(2); // 4
/// @endcode
/// @par References
/// - R. Freivalds (1979), Fast probabilistic algorithms, MFCS, pp57-69. doi:10.1007/3-540-09526-8_5. The paper uses two-point sign probes; this implementation uses binary probes and an absolute floating tolerance.
/// @see freivalds_verify(), freivalds_trace()
std::size_t freivalds_binary_vectors(std::size_t columns);
/// @brief Exhaustively audit every binary challenge under a bounded vector count.
/// Every matrix must be rectangular with finite entries; all shapes then all values are validated before compatibility, nonpositive rounds or optional quotas. Wellformed incompatible dimensions return a rejecting result. No caller mutation or shared state. Actual nonfinite matvec accumulation or residual throws; no unused AB product or norms are evaluated. Empty outer matrices represent0x0. Compatible rounds<=0 accept without draws. Ordinary arithmetic order and fixed absolute tolerance are retained; small wrong products can pass and cancellation can reject an exactly correct product. The exact-arithmetic IID-bit theorem is not a general guarantee for this double/SplitMix implementation. Validation/count/work bounds and quota admission precede enumeration. No vectors or frames are retained. Incompatible matrices return no vectors without counting2^n. Cap0 disables only optional quota. A compatible zero-column matrix has one empty challenge.
/// @param a Left m x k matrix.
/// @param b Right k x n matrix.
/// @param c Claimed m x n product.
/// @param max_vectors Maximum enumerated vectors;0 disables only this cap. Default 1000000.
/// @return Counts of all supplied binary vectors under the numerical tolerance policy.
/// @throws std::invalid_argument Ragged/nonfinite input or malformed supplied challenge.
/// @throws std::length_error Count/container overflow or exceeded optional admission quota.
/// @throws std::overflow_error Actual nonfinite numerical arithmetic.
/// @par Example
/// @code
/// const auto x = dm::algorithms::freivalds_binary_audit({{1}}, {{1}}, {{2}});
/// std::cout << x.accepted; // 1
/// @endcode
/// @par References
/// - R. Freivalds (1979), Fast probabilistic algorithms, MFCS, pp57-69. doi:10.1007/3-540-09526-8_5. The paper uses two-point sign probes; this implementation uses binary probes and an absolute floating tolerance.
/// @see freivalds_verify(), freivalds_trace()
FreivaldsBinaryAudit freivalds_binary_audit(const FreivaldsMatrix& a, const FreivaldsMatrix& b, const FreivaldsMatrix& c, std::size_t max_vectors = 1000000);
/// @brief Evaluate requested round prefixes once per seed, stopping at first mismatch.
/// Every matrix must be rectangular with finite entries; all shapes then all values are validated before compatibility, nonpositive rounds or optional quotas. Wellformed incompatible dimensions return a rejecting result. No caller mutation or shared state. Actual nonfinite matvec accumulation or residual throws; no unused AB product or norms are evaluated. Empty outer matrices represent0x0. Compatible rounds<=0 accept without draws. Ordinary arithmetic order and fixed absolute tolerance are retained; small wrong products can pass and cancellation can reject an exactly correct product. The exact-arithmetic IID-bit theorem is not a general guarantee for this double/SplitMix implementation. Each seed runs once at max(0,requested lengths); derive shorter verdicts from its actual first failure. No replay. Validate nonwrapping consecutive seed range and whole aggregate counter bounds before execution. Zero trials produce zero counts; empty lengths perform no calls. Numerical failure in the longest requested run throws the whole experiment without partial output.
/// @param a Left m x k matrix.
/// @param b Right k x n matrix.
/// @param c Claimed m x n product.
/// @param rounds Requested prefix lengths; order, duplicates and negative values retained.
/// @param trials Number of consecutive seeds. Default 4000.
/// @param first_seed First seed; range wrapping is rejected. Default 1000.
/// @return Caller-order prefix counts, per-seed first failure and actual aggregate work.
/// @throws std::invalid_argument Ragged/nonfinite input or malformed supplied challenge.
/// @throws std::length_error Count/container overflow or exceeded optional admission quota.
/// @throws std::overflow_error Actual nonfinite numerical arithmetic.
/// @par Example
/// @code
/// const auto x = dm::algorithms::freivalds_seed_sweep({{1}}, {{1}}, {{1}}, {0, 2}, 3);
/// std::cout << x.rows[1].accepted; // 3
/// @endcode
/// @par References
/// - R. Freivalds (1979), Fast probabilistic algorithms, MFCS, pp57-69. doi:10.1007/3-540-09526-8_5. The paper uses two-point sign probes; this implementation uses binary probes and an absolute floating tolerance.
/// @see freivalds_verify(), freivalds_trace()
FreivaldsSeedSweep freivalds_seed_sweep(const FreivaldsMatrix& a, const FreivaldsMatrix& b, const FreivaldsMatrix& c, const std::vector<int>& rounds, std::size_t trials = 4000, std::uint64_t first_seed = 1000);
/// @brief Construct the original40x40 seed3 verification fixture.
/// No input/shared state. Fills all A then all B; bounded integer product reuses the canonical Strassen implementation. This is synthetic test data, not an external dataset.
/// @return Owned original bounded integer fixture and its product.
/// @par Example
/// @code
/// const auto x = dm::algorithms::freivalds_example_matrices();
/// std::cout << x.product.size(); // 40
/// @endcode
/// @par References
/// - R. Freivalds (1979), Fast probabilistic algorithms, MFCS, pp57-69. doi:10.1007/3-540-09526-8_5. The paper uses two-point sign probes; this implementation uses binary probes and an absolute floating tolerance.
/// @see freivalds_verify(), freivalds_trace()
FreivaldsExampleMatrices freivalds_example_matrices();
} // namespace datamunge::algorithms
