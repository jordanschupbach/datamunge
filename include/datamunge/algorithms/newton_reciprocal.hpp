#pragma once
/// @file newton_reciprocal.hpp
/// @brief Shared actual double reciprocal-Newton updates and fixed-seed educational histories.
#include <cstddef>
#include <vector>
namespace datamunge::algorithms {
/// @brief Actual multiply/subtract update and separately evaluated fused residual diagnostics.
/// @par Example
/// @code
/// auto s=dm::algorithms::newton_reciprocal_step(2,.25);std::print("{}\n",s.after); // 0.375
/// @endcode
/// @see newton_reciprocal_step()
struct NewtonReciprocalStep {
  double before{0};  ///< Actual double reciprocal before the update.
  double product{0}; ///< Actual rounded denominator*before used by the producer.
  double factor{0};  ///< Actual rounded2-product used by the producer.
  double after{0};   ///< Actual rounded before*factor assigned by the producer.
  double residual_before{
      0}; ///< Separately evaluated fma(-denominator,before,1), not an update operand.
  double residual_after{
      0}; ///< Separately evaluated fma(-denominator,after,1), not a certified real error bound.
};
/// @brief Execute one actual double update x*(2-d*x), without using a division to initialize x.
/// The supplied seed is caller data; this primitive does not claim that another
/// producer's seed was division-free. Fused residual diagnostics reduce product
/// cancellation but do not remove update rounding. No input mutation/shared state.
/// @param denominator Finite strictly positive d.
/// @param reciprocal Finite caller-provided x, including zero/negative values.
/// @return Actual product, factor, next x and separately computed fused residuals.
/// @throws std::invalid_argument If complete arguments are invalid.
/// @throws std::overflow_error If a produced double intermediate/diagnostic is nonfinite.
/// @par Example
/// @code
/// auto s=dm::algorithms::newton_reciprocal_step(2,.25);std::print("{} {}\n",s.product,s.factor); // 0.5 1.5
/// @endcode
/// @see NewtonReciprocalStep
NewtonReciprocalStep newton_reciprocal_step(double denominator, double reciprocal);
/// @brief Owned fixed-update history from an explicit supplied seed, with no hidden convergence stop.
/// @par Example
/// @code
/// auto t=dm::algorithms::newton_reciprocal_trace(2,.25,1);std::print("{}\n",t.final); // 0.375
/// @endcode
/// @see newton_reciprocal_trace()
struct NewtonReciprocalTrace {
  double denominator{0};      ///< Original finite positive d.
  double seed{0};             ///< Original supplied finite reciprocal seed.
  double initial_residual{0}; ///< Initial fused residual, fma(-d,seed,1).
  double final{0}; ///< Actual final reciprocal after all requested updates, seed if none.
  std::vector<NewtonReciprocalStep>
              steps;               ///< Actual sequential updates, including stationary ones.
  std::size_t captured_entries{0}; ///< Complete logical units4+requested updates.
};
/// @brief Execute exactly the requested number of shared actual reciprocal-Newton updates.
/// Exact arithmetic obeys e_next=e squared for e=1-d*x, but floating states
/// may stagnate or oscillate. Arbitrary seeds need not converge. O(K) work and
/// owned records; fixed metadata4 plus one logical update record each is known
/// and pre-admitted. No partial history or report-side replay.
/// @param denominator Finite strictly positive d.
/// @param seed Finite explicit x0.
/// @param updates Number K of actual updates, default6, zero valid.
/// @param max_entries Complete4+K history cap, default1000000, zero disables.
/// @return Owned original seed and complete actual update sequence.
/// @throws std::invalid_argument If complete denominator/seed controls fail.
/// @throws std::length_error If count/container geometry or complete quota is exceeded.
/// @throws std::overflow_error If an actual intermediate/diagnostic is nonfinite.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::newton_reciprocal_trace(2,.25,0).captured_entries); // 4
/// @endcode
/// @see newton_reciprocal_step()
NewtonReciprocalTrace newton_reciprocal_trace(
    double      denominator,
    double      seed,
    std::size_t updates     = 6,
    std::size_t max_entries = 1000000);
} // namespace datamunge::algorithms
