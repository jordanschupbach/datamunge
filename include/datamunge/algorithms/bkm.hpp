#pragma once
/// @file bkm.hpp
/// @brief Checked real positive-digit BKM-style shift/add logarithms, exponentials and traces.
#include <array>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace datamunge::algorithms {
/// @brief Supported unreduced input intervals for the 53-term real greedy variant.
/// Logarithms accept [1,max_argument]; exponentials accept [0,max_exponent].
/// Limits are the actual double product/sum of all available factors, not rounded
/// decimal estimates. Values are owned, immutable with respect to library state.
/// @par Example
/// @code
/// const auto domain=dm::algorithms::bkm_domain();
/// std::print("{} {}\n",domain.terms,domain.max_argument>4.7); // 53 true
/// @endcode
/// @see bkm_domain(), bkm_ln(), bkm_exp()
struct BkmDomain {
    int terms{53}; ///< Available shifts, numbered zero through 52.
    double max_argument{1}; ///< Greatest supported logarithm argument, approximately 4.76846.
    double max_exponent{0}; ///< Greatest supported exponential argument, approximately 1.56202.
};

/// @brief One attempted real greedy factor and the resulting accepted state.
/// Both candidate fields are computed from the preceding state. A rejected step
/// keeps product/log_sum unchanged; a true accepted flag includes equality.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::bkm_ln_trace(3,2);
/// std::print("{} {}\n",trace.steps[1].shift,trace.steps[1].product); // 1 3
/// @endcode
/// @see BkmTrace
struct BkmStep {
    int shift{0}; ///< Factor index k in 1+2^-k.
    double candidate_product{1}; ///< product + ldexp(product,-k) before the decision.
    double candidate_log{0}; ///< log_sum + log1p(2^-k) before the decision.
    bool accepted{false}; ///< Whether the candidate satisfies the selected input constraint.
    double product{1}; ///< Running product after this attempt.
    double log_sum{0}; ///< Running logarithm sum after this attempt.
};

/// @brief Owned result and attempted factors from the same loop as the scalar API.
/// Initial state is product=1, log_sum=0. There is one event per requested term,
/// even after reaching a floating-point plateau. No shared state is modified.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::bkm_exp_trace(1,8);
/// std::print("{} {}\n",trace.steps.size(),trace.logarithm); // 8 false
/// @endcode
/// @see bkm_ln_trace(), bkm_exp_trace()
struct BkmTrace {
    double input{0}; ///< Validated original argument.
    bool logarithm{true}; ///< True for product-constrained logarithm mode; false for sum-constrained exponential mode.
    double value{0}; ///< Final log_sum in logarithm mode or product in exponential mode.
    std::vector<BkmStep> steps; ///< Chronological accepted and rejected attempts.
};

/// @brief Implementation of the fixed logarithm table and shared greedy iteration.
namespace detail {
/// @brief Lazily initialise the 53 positive log1p(2^-k) entries once, thread-safely.
/// @return Process-lifetime read-only table, independent of calls and input values.
inline const std::array<double,53>& bkm_table() {
    /// @brief Runtime-generated double table; initialisation calls the platform log1p.
    static const std::array<double,53> table=[] {
        std::array<double,53> values{};
        for (int k=0;k<53;++k) values[k]=std::log1p(std::ldexp(1.,-k));
        return values;
    }();
    return table;
}
} // namespace detail

/// @brief Inspect the actual finite domain of the real 53-term greedy implementation.
/// Returns the accumulated double product of all 1+2^-k factors and the sum of
/// their log1p entries. This is a restricted positive-digit variant, without
/// range reduction or complex/signed-digit BKM. First-call initialisation is
/// thread-safe; subsequent calls copy three scalars and allocate no storage.
/// @return Owned term count and inclusive upper bounds; lower bounds are 1 and 0.
/// @par Example
/// @code
/// const auto domain=dm::algorithms::bkm_domain();
/// std::print("{}\n",dm::algorithms::bkm_ln(domain.max_argument)==domain.max_exponent); // true
/// @endcode
/// @see BkmDomain, bkm_ln(), bkm_exp()
inline BkmDomain bkm_domain() {
    /// @brief All-factors endpoint computed with the same additions as evaluation.
    static const BkmDomain domain=[] {
        BkmDomain result;
        const auto& table=detail::bkm_table();
        for (int k=0;k<result.terms;++k) {
            result.max_argument+=std::ldexp(result.max_argument,-k);
            result.max_exponent+=table[k];
        }
        return result;
    }();
    return domain;
}

namespace detail {
/// @brief Execute one checked canonical positive-digit product/log-sum iteration.
/// @tparam Observe Callable receiving an attempted BkmStep; may propagate exceptions.
/// @param input Original finite argument in the full 53-term domain.
/// @param logarithm Select product-constrained logarithm rather than sum-constrained exponential.
/// @param terms Number of attempted shifts, between one and 53 inclusive.
/// @param observe Consumer of post-decision states.
/// @return Final product and log_sum; scalar calls need O(1) auxiliary storage.
/// @throws std::invalid_argument If terms is outside [1,53].
/// @throws std::domain_error If input is nonfinite or outside the selected domain.
template<class Observe>
std::pair<double,double> bkm_run(double input,bool logarithm,int terms,Observe observe) {
    const auto domain=bkm_domain();
    if (terms<1 || terms>domain.terms) throw std::invalid_argument("BKM terms must be in [1,53]");
    if (!std::isfinite(input) || input<(logarithm?1.:0.) ||
        input>(logarithm?domain.max_argument:domain.max_exponent))
        throw std::domain_error("BKM input outside the unreduced positive-digit domain");
    const auto& table=bkm_table();
    double product=1,log_sum=0;
    for (int k=0;k<terms;++k) {
        const double candidate_product=product+std::ldexp(product,-k);
        const double candidate_log=log_sum+table[k];
        const bool accepted=logarithm?candidate_product<=input:candidate_log<=input;
        if (accepted) {product=candidate_product;log_sum=candidate_log;}
        observe(BkmStep{k,candidate_product,candidate_log,accepted,product,log_sum});
    }
    return {product,log_sum};
}
} // namespace detail

/// @brief Approximate ln(v) with 53 checked positive-digit shift/add attempts.
/// Starts with product=1 and log_sum=0; accepts product+ldexp(product,-k)<=v
/// and accumulates log1p(2^-k). No automatic range reduction is performed.
/// This double implementation is not correctly rounded: table generation uses
/// platform log1p and arithmetic can plateau at roundoff. O(53) work, O(1)
/// auxiliary storage, no input mutation, thread-safe shared read-only table.
/// @param v Finite value in [1,bkm_domain().max_argument], inclusive.
/// @return Approximate natural logarithm, possibly slightly above std::log(v) due to rounding.
/// @throws std::domain_error If v is nonfinite or outside the supported interval.
/// @par Example
/// @code
/// std::print("{:.6f}\n",dm::algorithms::bkm_ln(3)); // 1.098612
/// @endcode
/// @par References
/// - J.-C. Bajard, S. Kla and J.-M. Muller (1994), BKM: A New Hardware Algorithm
///   for Complex Elementary Functions, IEEE Transactions on Computers 43(8),
///   955-963, doi:10.1109/12.295857. This is a restricted real {0,1}-digit variant.
/// @see bkm_ln_trace(), bkm_exp(), bkm_domain()
inline double bkm_ln(double v) {
    return detail::bkm_run(v,true,53,[](const BkmStep&){}).second;
}

/// @brief Approximate exp(y) with 53 checked positive-digit shift/add attempts.
/// Accepts log_sum+log1p(2^-k)<=y and mirrors accepted digits in the product.
/// No range reduction, complex arithmetic or signed-digit hardware is provided.
/// Double rounding/table error remains; the result is not guaranteed correctly
/// rounded or below exp(y). O(53) work, O(1) auxiliary storage, thread-safe.
/// @param y Finite value in [0,bkm_domain().max_exponent], inclusive; -0 is accepted.
/// @return Approximate exponential from the final product.
/// @throws std::domain_error If y is nonfinite or outside the supported interval.
/// @par Example
/// @code
/// std::print("{:.6f}\n",dm::algorithms::bkm_exp(1)); // 2.718282
/// @endcode
/// @par References
/// - J.-C. Bajard, S. Kla and J.-M. Muller (1994), BKM: A New Hardware Algorithm
///   for Complex Elementary Functions, doi:10.1109/12.295857; restricted real variant here.
/// @see bkm_exp_trace(), bkm_ln(), bkm_domain()
inline double bkm_exp(double y) {
    return detail::bkm_run(y,false,53,[](const BkmStep&){}).first;
}

/// @brief Record the canonical product-constrained logarithm iteration.
/// Uses exactly the scalar loop, with a configurable prefix of the same table.
/// The input domain is always the full 53-term domain, even for a short prefix.
/// O(terms) work and owned recording storage; no input mutation or shared writes.
/// Each term is an attempt, not a guaranteed additional correct bit.
/// @param v Finite logarithm argument in [1,bkm_domain().max_argument].
/// @param terms Attempted shifts, default 53, allowed [1,53].
/// @return Owned argument, final log_sum and all accepted/rejected candidates.
/// @throws std::invalid_argument If terms is outside [1,53].
/// @throws std::domain_error If v is nonfinite or outside the supported interval.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::bkm_ln_trace(3,2);
/// std::print("{} {:.6f}\n",trace.steps.back().product,trace.value); // 3 1.098612
/// @endcode
/// @par References
/// - J.-C. Bajard, S. Kla and J.-M. Muller (1994), BKM: A New Hardware Algorithm
///   for Complex Elementary Functions, doi:10.1109/12.295857; restricted real variant here.
/// @see bkm_ln(), BkmStep, bkm_exp_trace()
inline BkmTrace bkm_ln_trace(double v,int terms=53) {
    BkmTrace result;result.input=v;
    result.value=detail::bkm_run(v,true,terms,[&](const BkmStep& step){result.steps.push_back(step);}).second;
    return result;
}

/// @brief Record the canonical sum-constrained exponential iteration.
/// Uses the scalar loop with a configurable table prefix; every attempted digit
/// is recorded. The input domain remains the full 53-term domain. O(terms) work
/// and owned storage; no caller input or shared state is modified.
/// @param y Finite exponential argument in [0,bkm_domain().max_exponent].
/// @param terms Attempted shifts, default 53, allowed [1,53].
/// @return Owned argument, final product and accepted/rejected candidates.
/// @throws std::invalid_argument If terms is outside [1,53].
/// @throws std::domain_error If y is nonfinite or outside the supported interval.
/// @par Example
/// @code
/// const auto trace=dm::algorithms::bkm_exp_trace(1);
/// std::print("{} {:.6f}\n",trace.steps.size(),trace.value); // 53 2.718282
/// @endcode
/// @par References
/// - J.-C. Bajard, S. Kla and J.-M. Muller (1994), BKM: A New Hardware Algorithm
///   for Complex Elementary Functions, doi:10.1109/12.295857; restricted real variant here.
/// @see bkm_exp(), BkmStep, bkm_ln_trace()
inline BkmTrace bkm_exp_trace(double y,int terms=53) {
    BkmTrace result;result.input=y;result.logarithm=false;
    result.value=detail::bkm_run(y,false,terms,[&](const BkmStep& step){result.steps.push_back(step);}).first;
    return result;
}
} // namespace datamunge::algorithms
