#pragma once
/// @file forward_backward.hpp
/// @brief Validated discrete HMM smoothing, actual scaled tables and bounded path enumeration.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned finite-alphabet hidden Markov model parameters in state-major order.
/// Initial has N entries, transition N rows of N entries, emission N equal rows
/// of M>0 entries. Forward-backward and path-query APIs validate all shapes first,
/// then finite weights in[0,1] and each distribution's long-double sum within
/// absolute1e-12 of1. They never normalize caller parameters. The entirely empty
/// model is accepted with empty observations. Other legacy consumers may assume
/// valid parameters. State and symbol IDs are zero based; arrays own their storage.
/// @par Example
/// @code
/// const auto h=dm::algorithms::forward_backward_example_model();
/// std::cout << h.num_states(); // 2
/// @endcode
/// @par References
/// - L.R. Rabiner (1989), A tutorial on hidden Markov models and selected applications in speech recognition, Proceedings of the IEEE77(2),257–286, doi:10.1109/5.18626; https://www.cs.ubc.ca/~murphyk/Software/HMM/rabiner.pdf
/// @see forward_backward(), hmm_path_probability()
struct HiddenMarkovModel {
    std::vector<double> initial; ///< Initial-state probabilities, length N.
    std::vector<std::vector<double>> transition; ///< A[i][j]=P(next j given current i), N by N.
    std::vector<std::vector<double>> emission; ///< B[i][k]=P(symbol k given state i), N by M.

    /// @brief Read the initial-vector length without validating the model.
    /// @return Number of entries in initial, possibly zero.
    std::size_t num_states() const { return initial.size(); }
    /// @brief Read the first emission-row length without validating remaining rows.
    /// @return First emission-row length, or zero when there are no rows.
    std::size_t num_symbols() const { return emission.empty()?0:emission.front().size(); }
};
/// @brief Owned actual scaled forward/backward tables and smoothed state marginals.
/// Historical field layout retained. For positive representable evidence, alpha rows are filtering distributions; beta uses terminal beta[T-1]=scale[T-1] and is not a probability distribution. In exact arithmetic sum(alpha[t]*beta[t])=scale[t], so gamma is explicitly normalized. Impossible sequences have log_likelihood=-infinity and zero gamma placeholders, not defined posteriors. Empty observations have empty arrays and log likelihood0.
/// @par Example
/// @code
/// dm::algorithms::ForwardBackwardResult x{};
/// std::cout << x.log_likelihood; // 0
/// @endcode
/// @see forward_backward(), hmm_enumerate_paths()
struct ForwardBackwardResult {
    std::vector<std::vector<double>> alpha{}; ///< Actual scaled forward rows, T by N; valid prefixes retained before impossibility.
    std::vector<std::vector<double>> beta{}; ///< Actual scaled backward rows, T by N; all zero for impossible evidence.
    std::vector<std::vector<double>> gamma{}; ///< Smoothed marginals, T by N; zero placeholders for impossible evidence.
    std::vector<double> scale{}; ///< Reciprocal predictive masses c[t], length T; zero at/after impossibility.
    double log_likelihood{}; ///< Negative sum log(c[t]) for possible evidence; -infinity if impossible,0 when empty.
};
/// @brief Actual once-only inference counts, excluding validation and allocations.
/// Transition terms count pair visits: one product plus accumulation in the forward pass, two products plus accumulation in the backward pass. Support probes run only on the first zero predictive mass to distinguish a mathematically impossible prefix from numeric underflow. No replay or diagnostic arithmetic is included.
/// @par Example
/// @code
/// dm::algorithms::ForwardBackwardWork x{};
/// std::cout << x.states; // 0
/// @endcode
/// @see forward_backward(), hmm_enumerate_paths()
struct ForwardBackwardWork {
    std::size_t states{}; ///< N hidden states.
    std::size_t symbols{}; ///< M emission symbols.
    std::size_t observations{}; ///< T observation records.
    std::size_t initial_products{}; ///< Initial pi times emission products.
    std::size_t forward_transition_terms{}; ///< Actual forward predecessor-pair terms.
    std::size_t forward_emission_products{}; ///< Actual emission products after the first row.
    std::size_t forward_scalings{}; ///< Actual alpha times c operations.
    std::size_t backward_initializations{}; ///< Terminal beta assignments.
    std::size_t backward_transition_terms{}; ///< Actual backward successor-pair terms.
    std::size_t backward_scalings{}; ///< Actual backward accumulator times c products.
    std::size_t posterior_products{}; ///< Actual alpha times beta products.
    std::size_t posterior_divisions{}; ///< Actual gamma normalization divisions; zero for impossible evidence.
    std::size_t scale_logs{}; ///< Logarithms of positive scale factors actually evaluated.
    std::size_t completed_forward_steps{}; ///< Forward rows completed.
    std::size_t support_initial_tests{}; ///< Initial-state visits in a zero-mass Boolean support probe.
    std::size_t support_pair_visits{}; ///< State-pair visits in that exceptional Boolean support probe.
};
/// @brief Owned inference tables and actual work from the same kernel.
/// All arrays own storage; caller edits are independent. There is no additional phase-history copy because the result already contains the actual tables.
/// @par Example
/// @code
/// dm::algorithms::ForwardBackwardOutcome x{};
/// std::cout << x.work.states; // 0
/// @endcode
/// @see forward_backward(), hmm_enumerate_paths()
struct ForwardBackwardOutcome {
    ForwardBackwardResult result{}; ///< Owned actual inference result.
    ForwardBackwardWork work{}; ///< Actual counts from that execution.
};
/// @brief Owned widened path sum, state marginals and best joint path.
/// A bounded small-model reference, not a fast inference method or exact arithmetic. Time0 is the least significant base-N digit; strict best-weight updates keep the first enumerated tie. Impossible evidence yields probability0, log_likelihood=-infinity, zero marginal placeholders and no best path. Empty observations enumerate one empty path of weight1. Long-double positive product underflow throws.
/// @par Example
/// @code
/// dm::algorithms::HmmPathEnumeration x{};
/// std::cout << x.paths; // 0
/// @endcode
/// @see forward_backward(), hmm_enumerate_paths()
struct HmmPathEnumeration {
    long double probability{}; ///< Sum of all enumerated joint path masses.
    long double log_likelihood{}; ///< Log of total mass, or -infinity for zero mass.
    std::vector<std::vector<long double>> gamma{}; ///< T by N widened state marginals, or zero placeholders for impossible evidence.
    std::vector<std::size_t> best_states{}; ///< Owned first maximum-weight path; empty for empty or impossible observations.
    long double best_probability{}; ///< Maximum joint path weight;1 for empty observations,0 if impossible.
    std::size_t paths{}; ///< Actual enumerated path count, including zero-weight paths.
};
/// @brief Compute validated scaled filtering, smoothing and log likelihood.
/// Fully validate model shapes/finite stochastic rows and every observation before
/// allocation, quotas or empty-input shortcuts. No caller mutation, RNG or shared
/// state. The original double operation order and scaling convention are retained
/// for representable positive evidence. Impossible evidence returns log likelihood
/// -infinity and zero undefined-posterior placeholders. A zero predictive mass
/// with a positive-probability support path throws underflow_error; nonfinite
/// actual arithmetic or reciprocal scales throw overflow_error. Individual tiny
/// terms can still round away: scaling is not an arbitrary-length/precision
/// guarantee. Empty observations have likelihood1 and empty tables.
/// Time O(N²+NM+T+TN²), owned storage O(TN+T), auxiliary scratch O(N) only for
/// the exceptional support probe. Returned arrays already expose actual states.
/// The historical two-argument call has no optional admission cap; representability checks remain.
/// @param hmm Finite discrete stochastic model, not mutated.
/// @param observations Zero-based symbols, possibly empty.
/// @return Owned historical result layout with actual scaled tables.
/// @throws std::invalid_argument Invalid model shape, probability or row sum.
/// @throws std::out_of_range Observation symbol outside the model alphabet.
/// @throws std::length_error Unrepresentable counts/container capacities or exceeded optional cap.
/// @throws std::overflow_error Nonfinite actual arithmetic or reciprocal.
/// @throws std::underflow_error Positive support lost entirely to floating underflow.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::forward_backward(dm::algorithms::forward_backward_example_model(),{});
/// std::cout << r.log_likelihood; // 0
/// @endcode
/// @par References
/// - L.R. Rabiner (1989), A tutorial on hidden Markov models and selected applications in speech recognition, Proceedings of the IEEE77(2),257–286, doi:10.1109/5.18626; https://www.cs.ubc.ca/~murphyk/Software/HMM/rabiner.pdf; terminal beta equals the final scale factor.
/// @see forward_backward_entries(), hmm_enumerate_paths()
ForwardBackwardResult forward_backward(const HiddenMarkovModel& hmm, const std::vector<std::size_t>& observations);
/// @brief Measure actual smoothing work with optional whole-result admission.
/// Fully validate model shapes/finite stochastic rows and every observation before
/// allocation, quotas or empty-input shortcuts. No caller mutation, RNG or shared
/// state. The original double operation order and scaling convention are retained
/// for representable positive evidence. Impossible evidence returns log likelihood
/// -infinity and zero undefined-posterior placeholders. A zero predictive mass
/// with a positive-probability support path throws underflow_error; nonfinite
/// actual arithmetic or reciprocal scales throw overflow_error. Individual tiny
/// terms can still round away: scaling is not an arbitrary-length/precision
/// guarantee. Empty observations have likelihood1 and empty tables.
/// Time O(N²+NM+T+TN²), owned storage O(TN+T), auxiliary scratch O(N) only for
/// the exceptional support probe. Returned arrays already expose actual states.
/// Admission uses forward_backward_entries() before allocation. Entry units are not bytes or peak RAM.
/// @param hmm Finite stochastic model, validated in full first.
/// @param observations Zero-based symbols.
/// @param max_entries Whole result entry cap, default1million; zero disables only this cap.
/// @return Owned actual result and work without replay.
/// @throws std::invalid_argument Invalid model shape, probability or row sum.
/// @throws std::out_of_range Observation symbol outside the model alphabet.
/// @throws std::length_error Unrepresentable counts/container capacities or exceeded optional cap.
/// @throws std::overflow_error Nonfinite actual arithmetic or reciprocal.
/// @throws std::underflow_error Positive support lost entirely to floating underflow.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::forward_backward_work(dm::algorithms::forward_backward_example_model(),{0,1});
/// std::cout << r.work.forward_transition_terms; // 4
/// @endcode
/// @par References
/// - L.R. Rabiner (1989), A tutorial on hidden Markov models and selected applications in speech recognition, Proceedings of the IEEE77(2),257–286, doi:10.1109/5.18626; https://www.cs.ubc.ca/~murphyk/Software/HMM/rabiner.pdf; terminal beta equals the final scale factor.
/// @see forward_backward_entries(), hmm_enumerate_paths()
ForwardBackwardOutcome forward_backward_work(const HiddenMarkovModel& hmm, const std::vector<std::size_t>& observations, std::size_t max_entries=1000000);
/// @brief Forecast whole owned inference entry records without allocation.
/// For positive N,T, counts4T+3TN: three row records and one scale scalar per
/// time plus three N-scalar rows. Otherwise0. Units are entry records, not bytes
/// or peak RAM; container objects and exceptional support scratch are additional.
/// Shape and probability validity cannot be checked from counts alone.
/// @param states N hidden states.
/// @param observations T observation records.
/// @return Whole result entry count.
/// @throws std::length_error Count or required container capacity is unrepresentable.
/// @par Example
/// @code
/// std::cout << dm::algorithms::forward_backward_entries(2,2); // 20
/// @endcode
/// @see forward_backward_work()
std::size_t forward_backward_entries(std::size_t states, std::size_t observations);
/// @brief Forecast N to the T power with checked integer arithmetic, not floating pow.
/// The empty path count is1 even with N=0. For positive T, N=0 gives0 and
/// N=1 gives1 immediately, including SIZE_MAX times. No allocation or mutation.
/// @param states N possible states.
/// @param observations T time points.
/// @return Number of possible state paths.
/// @throws std::length_error Exact count cannot fit size_t.
/// @par Example
/// @code
/// std::cout << dm::algorithms::hmm_enumeration_paths(2,4); // 16
/// @endcode
/// @see hmm_enumerate_paths()
std::size_t hmm_enumeration_paths(std::size_t states, std::size_t observations);
/// @brief Evaluate one validated joint hidden-path and observation mass.
/// Validates the full model, symbols, path length and all state IDs before products. Time O(N²+NM+T), O(1) auxiliary space, no caller mutation or shared state.
/// @param hmm Finite stochastic discrete model.
/// @param observations Zero-based symbols.
/// @param states Exactly one valid state ID per observation.
/// @return Widened joint probability; empty path/observations has mass1.
/// @throws std::invalid_argument Invalid model or mismatched path length.
/// @throws std::out_of_range Invalid symbol or supplied state ID.
/// @throws std::overflow_error Nonfinite widened arithmetic.
/// @throws std::underflow_error A positive joint prefix product rounds to zero.
/// @par Example
/// @code
/// const auto h=dm::algorithms::forward_backward_example_model();
/// std::cout << dm::algorithms::hmm_path_probability(h,{0},{0}); // 0.3
/// @endcode
/// @par References
/// - L.R. Rabiner (1989), A tutorial on hidden Markov models and selected applications in speech recognition, Proceedings of the IEEE77(2),257–286, doi:10.1109/5.18626; https://www.cs.ubc.ca/~murphyk/Software/HMM/rabiner.pdf; direct path definition, equations12–17.
/// @see forward_backward(), hmm_enumeration_paths()
long double hmm_path_probability(const HiddenMarkovModel& hmm, const std::vector<std::size_t>& observations, const std::vector<std::size_t>& states);
/// @brief Sum joint paths and state marginals by bounded widened enumeration.
/// Full model/symbol validation precedes path admission and copies. Checked integer N^T and T*N^T work bounds remain with cap0. This cap counts paths, separately from inference entry records. Time O(N²+NM+T*N^T), output O(TN+T), scratch O(T), no caller mutation or RNG. Long-double products/sums are rounded; positive product underflow throws. No partial enumeration result.
/// @param hmm Finite stochastic discrete model.
/// @param observations Zero-based symbols.
/// @param max_paths Path-count cap, default1million; zero disables only this cap.
/// @return Owned widened path sum, marginals, best joint path and actual count.
/// @throws std::invalid_argument Invalid model shape/probabilities/row sums.
/// @throws std::out_of_range Invalid observation symbol.
/// @throws std::length_error Count/capacity overflow or exceeded enumeration cap.
/// @throws std::overflow_error Nonfinite widened arithmetic.
/// @throws std::underflow_error A positive joint product rounds to zero.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// const auto r=dm::algorithms::hmm_enumerate_paths(dm::algorithms::forward_backward_example_model(),{0,1,2,2});
/// std::cout << r.paths; // 16
/// @endcode
/// @par References
/// - L.R. Rabiner (1989), A tutorial on hidden Markov models and selected applications in speech recognition, Proceedings of the IEEE77(2),257–286, doi:10.1109/5.18626; https://www.cs.ubc.ca/~murphyk/Software/HMM/rabiner.pdf; direct path definition, equations12–17.
/// @see forward_backward(), hmm_enumeration_paths()
HmmPathEnumeration hmm_enumerate_paths(const HiddenMarkovModel& hmm, const std::vector<std::size_t>& observations, std::size_t max_paths=1000000);
/// @brief Return the original two-state, three-symbol illustrative model.
/// Deterministic fixture, no RNG or shared state. Labels0/1 name Healthy/Fever;
/// symbols0/1/2 name normal/cold/dizzy in the original synthetic example.
/// @return Owned pi(.6,.4), transitions(.7,.3)/(.4,.6), emissions(.5,.4,.1)/(.1,.3,.6).
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// std::cout << dm::algorithms::forward_backward_example_model().num_symbols(); // 3
/// @endcode
/// @see forward_backward()
HiddenMarkovModel forward_backward_example_model();
/// @brief Return the original fourteen-symbol illustrative observation stream.
/// Deterministic fixture, no RNG or shared state. Labels0/1 name Healthy/Fever;
/// symbols0/1/2 name normal/cold/dizzy in the original synthetic example.
/// @return Owned symbols0,0,1,2,2,2,1,0,1,2,2,1,0,0.
/// @throws std::bad_alloc Allocation fails.
/// @par Example
/// @code
/// std::cout << dm::algorithms::forward_backward_example_observations().size(); // 14
/// @endcode
/// @see forward_backward()
std::vector<std::size_t> forward_backward_example_observations();
} // namespace datamunge::algorithms
