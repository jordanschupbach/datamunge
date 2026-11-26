#pragma once

/// @file acceptance_sampling.hpp
/// @brief Single sampling plans by attributes: operating characteristic, plan design from producer and
/// consumer risk points, and rectifying-inspection performance (AOQ, AOQL, ATI).

#include <cstddef>

namespace datamunge::stats {

/// @brief Distribution of the number of nonconforming items (or defects) in the sample.
/// @par Example
/// @code
/// const auto m = dm::stats::SamplingModel::poisson;
/// std::println("{}", m == dm::stats::SamplingModel::poisson); // true
/// @endcode
enum class SamplingModel {
    binomial,       ///< Large (or continuing) lots: Binomial(n, p).
    hypergeometric, ///< Finite isolated lot of size N with round(pN) nonconforming items.
    poisson,        ///< Defects per unit (or the Poisson approximation): Poisson(np).
};

/// @brief A single sampling plan: inspect n items and accept the lot if at most c are nonconforming.
/// @par Example
/// @code
/// dm::stats::AttributePlan p{.n = 50, .c = 1};
/// std::println("{} {}", p.n, p.c); // 50 1
/// @endcode
struct AttributePlan {
    std::size_t n{}; ///< Sample size.
    std::size_t c{}; ///< Acceptance number (reject at c + 1).
};

/// @brief Model options for the sampling functions.
/// @par Example
/// @code
/// dm::stats::SamplingOptions o{.model = dm::stats::SamplingModel::hypergeometric, .lot_size = 500};
/// std::println("{}", o.lot_size); // 500
/// @endcode
struct SamplingOptions {
    SamplingModel model{SamplingModel::binomial}; ///< Distribution.
    std::size_t lot_size{};                       ///< Lot size N (required for hypergeometric).
    std::size_t max_sample{1000000};              ///< Search limit for find_attribute_plan().
};

/// @brief Operating characteristic: probability of accepting a lot of quality p.
/// @param plan Plan.
/// @param p Fraction nonconforming (or mean defects per unit for the Poisson model), at least 0.
/// @param options Model.
/// @return \f$P(X\le c)\f$.
/// @throws std::invalid_argument For p outside [0, 1] (binomial, hypergeometric), p < 0, a missing or too
/// small lot size, or n = 0.
/// @par Example
/// @code
/// std::println("{:.4f}", dm::stats::acceptance_probability({.n = 50, .c = 1}, 0.02)); // 0.7358
/// @endcode
double acceptance_probability(const AttributePlan& plan, double p, const SamplingOptions& options = {});

/// @brief Smallest single sampling plan meeting a producer and a consumer risk point.
///
/// Finds the plan with the smallest n (and, for that n, the smallest c) such that
/// \f$P_a(\text{AQL})\ge1-\alpha\f$ and \f$P_a(\text{RQL})\le\beta\f$, by the search of
/// `AcceptanceSampling::find.plan`: increase n while the consumer point fails, and c while the producer point
/// fails.
/// @param aql Acceptable quality level (producer's risk point quality).
/// @param producer_risk alpha, the largest acceptable rejection probability at the AQL.
/// @param rql Rejectable quality level / LTPD (consumer's risk point quality), above the AQL.
/// @param consumer_risk beta, the largest acceptable acceptance probability at the RQL.
/// @param options Model.
/// @return The plan.
/// @throws std::invalid_argument For invalid qualities or risks, or no plan within max_sample (or the lot).
/// @par Example
/// @code
/// const auto p = dm::stats::find_attribute_plan(0.01, 0.05, 0.06, 0.10);
/// std::println("{} {}", p.n, p.c); // 110 3
/// @endcode
AttributePlan find_attribute_plan(double aql, double producer_risk, double rql, double consumer_risk,
                                  const SamplingOptions& options = {});

/// @brief Average outgoing quality and inspection under rectifying inspection at one quality level.
/// @par Example
/// @code
/// dm::stats::RectifyingPerformance r{.aoq = 0.01};
/// std::println("{}", r.aoq); // 0.01
/// @endcode
struct RectifyingPerformance {
    double acceptance{};  ///< P(accept).
    double aoq{};         ///< Average outgoing quality \f$pP_a(N-n)/N\f$ (rejected lots are screened).
    double ati{};         ///< Average total inspection \f$n+(1-P_a)(N-n)\f$.
};

/// @brief Rectifying-inspection performance: rejected lots are 100% inspected and nonconforming items replaced.
/// @param plan Plan.
/// @param p Incoming quality.
/// @param lot_size N (at least n).
/// @param options Model (its lot_size is replaced by `lot_size`).
/// @return Performance.
/// @throws std::invalid_argument As for acceptance_probability(), or N < n.
/// @par Example
/// @code
/// const auto r = dm::stats::rectifying_performance({.n = 50, .c = 1}, 0.02, 1000);
/// std::println("{:.5f}", r.aoq); // 0.01398
/// @endcode
RectifyingPerformance rectifying_performance(const AttributePlan& plan, double p, std::size_t lot_size,
                                             SamplingOptions options = {});

/// @brief Average outgoing quality limit: the worst AOQ over all incoming qualities.
/// @par Example
/// @code
/// dm::stats::Aoql a{.aoql = 0.02};
/// std::println("{}", a.aoql); // 0.02
/// @endcode
struct Aoql {
    double aoql{};    ///< Maximum of AOQ(p).
    double quality{}; ///< Incoming quality p at which it occurs.
};

/// @brief Maximizes AOQ(p) over p (grid search refined by golden section).
/// @param plan Plan.
/// @param lot_size N.
/// @param options Model.
/// @return AOQL and its location.
/// @throws std::invalid_argument As for rectifying_performance().
/// @par Example
/// @code
/// const auto a = dm::stats::average_outgoing_quality_limit({.n = 50, .c = 1}, 1000);
/// std::println("{:.4f}", a.aoql); // 0.0159
/// @endcode
Aoql average_outgoing_quality_limit(const AttributePlan& plan, std::size_t lot_size, SamplingOptions options = {});

} // namespace datamunge::stats
