#pragma once
/// @file velocity_verlet.hpp
/// @brief Checked native velocity verlet and actual producing diagnostics.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <vector>
namespace datamunge::algorithms {
/// @brief Owned sampled velocity-Verlet trajectory.
/// @par Example
/// @code
/// dm::algorithms::VerletSolution r;
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct VerletSolution {
  std::vector<double>              t; ///< Actual time labels, steps+1.
  std::vector<std::vector<double>> x; ///< Actual positions aligned to time.
  std::vector<std::vector<double>> v; ///< Actual velocities aligned to time.
};
/// @brief One actual drift and kick, including acceleration operands.
/// @par Example
/// @code
/// dm::algorithms::VerletStep r;
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct VerletStep {
  std::vector<double> before_x;         ///< Actual position before drift.
  std::vector<double> before_v;         ///< Actual velocity before kick.
  std::vector<double> old_acceleration; ///< Actual acceleration at old position.
  std::vector<double> new_acceleration; ///< Actual acceleration at new position.
  std::vector<double> after_x;          ///< Actual drifted position.
  std::vector<double> after_v;          ///< Actual kicked velocity.
};
/// @brief Actual integrator output and selected callback history.
/// @par Example
/// @code
/// dm::algorithms::VerletWork r;
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
struct VerletWork {
  VerletSolution          solution; ///< Actual complete output trajectory.
  std::vector<VerletStep> steps; ///< Actual selected drift/kick operands, bounded by max_records.
  std::size_t acceleration_calls{0}; ///< Actual callback invocations, steps+1 except zero interval.
  double      step_size{0};          ///< Computed constant integration step h.
};
/// @brief Integrate x double-prime=a(x) using checked velocity Verlet.
/// At most2000000 trajectory scalars and1000000 captured scalar operands. Fixed h=(t1-t0)/steps; final time label equals t1. Equal endpoints copy state without callback calls. O(steps*dimension) arithmetic/storage plus callback cost. Calls independent; callback exceptions propagate. Symplectic/energy statements require a suitable conservative acceleration and stable step.
/// @param accel Borrowed position-only callback returning finite matching-size acceleration.
/// @param x0 Owned finite initial position, nonempty.
/// @param v0 Owned finite initial velocity, same size.
/// @param t0 Finite initial time.
/// @param t1 Finite final time; backward integration allowed.
/// @param steps Positive steps<=1000000.
/// @param max_records Number of complete actual drift/kick snapshots to retain, default0.
/// @return Actual trajectory, h, callback count and optional snapshots.
/// @throws std::invalid_argument Invalid shapes, callback, times, step count or nonfinite callback result.
/// @throws std::length_error Output/history budget exceeded.
/// @throws std::overflow_error Nonfinite h or state arithmetic.
/// @par Example
/// @code
/// auto r=dm::algorithms::velocity_verlet_work([](const std::vector<double>& x){return std::vector<double>{-x[0]};},{1},{0},0,.1,1,1);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
VerletWork velocity_verlet_work(
    const std::function<std::vector<double>(const std::vector<double>&)>& accel,
    std::vector<double>                                                   x0,
    std::vector<double>                                                   v0,
    double                                                                t0,
    double                                                                t1,
    int                                                                   steps,
    std::size_t                                                           max_records = 0);
/// @brief Return the same checked velocity-Verlet producer trajectory.
/// @param accel Matching finite position-only acceleration.
/// @param x0 Finite initial position.
/// @param v0 Same-sized finite velocity.
/// @param t0 Finite start.
/// @param t1 Finite end.
/// @param steps Positive grid steps.
/// @return Actual steps+1 trajectory.
/// @throws std::invalid_argument Invalid producer inputs.
/// @throws std::length_error Output budget exceeded.
/// @throws std::overflow_error Nonfinite arithmetic.
/// @par Example
/// @code
/// auto r=dm::algorithms::velocity_verlet([](const std::vector<double>&){return std::vector<double>{0};},{0},{1},0,1,2);
/// @endcode
/// @par References
/// - The accompanying report derives the implemented recurrence.
/// @see Other operations in this header.
VerletSolution velocity_verlet(
    const std::function<std::vector<double>(const std::vector<double>&)>& accel,
    std::vector<double>                                                   x0,
    std::vector<double>                                                   v0,
    double                                                                t0,
    double                                                                t1,
    int                                                                   steps);

} // namespace datamunge::algorithms
