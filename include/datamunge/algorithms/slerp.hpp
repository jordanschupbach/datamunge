#pragma once
/// @file slerp.hpp
/// @brief Checked scale-safe quaternion normalization and actual interpolation branches.
#include <array>
/// @brief Canonical checked native producers and owned values.
namespace datamunge::algorithms {
/// @brief Store a scalar-first quaternion as four owned doubles.
/// Order{w,x,y,z}; zero is a valid raw value but is not a rotation and cannot be normalized. Finite nonzero scaled rotations can be normalized. No shared state.
/// @par Example
/// @code
/// const dm::algorithms::Quaternion q{1,0,0,0};
/// std::print("{}\n",q[0]); // 1
/// @endcode
/// @see quat_normalize()
using Quaternion = std::array<double, 4>;
/// @brief SlerpRun owns actual producer values or explicit invocation policies; fields state units and defaults.
/// @par Example
/// @code
/// const dm::algorithms::SlerpRun value{};
/// std::print("{}\n",value.dot); // 0
/// @endcode
struct SlerpRun {
  Quaternion start{};         ///< Actual normalized starting representative.
  Quaternion target{};        ///< Actual normalized target after optional sign flip.
  Quaternion value{};         ///< Actual normalized interpolated representative.
  double     dot{};           ///< Actual nonnegative clamped dot of normalized chosen endpoints.
  double     omega{};         ///< Actual half rotation angle in quaternion space, radians.
  double     first_weight{};  ///< Actual coefficient of start before final normalization.
  double     second_weight{}; ///< Actual coefficient of target before final normalization.
  bool       sign_flipped{};  ///< Actual original normalized target dot was negative.
  bool       normalized_linear{}; ///< Actual dot>0.9995 branch used normalized chord interpolation.
};
/// @brief Return a finite raw quaternion dot product.
/// Does not normalize. Sum/products use double; nonrepresentable result throws. Constant work, no mutation or shared state.
/// @param a Complete finite raw quaternion.
/// @param b Complete finite raw quaternion.
/// @return Finite scalar dot.
/// @throws std::invalid_argument Nonfinite component.
/// @throws std::overflow_error Dot not representable.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::quat_dot({1,0,0,0},{1,0,0,0})); // 1
/// @endcode
/// @see slerp_run()
double quat_dot(const Quaternion& a, const Quaternion& b);
/// @brief Return the scale-safe Euclidean quaternion norm.
/// Uses nested hypot; zero norm is allowed. Constant work, no shared state.
/// @param q Complete finite raw quaternion.
/// @return Nonnegative finite norm.
/// @throws std::invalid_argument Nonfinite component.
/// @throws std::overflow_error Norm exceeds double range.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::quat_norm({3,4,0,0})); // 5
/// @endcode
/// @see slerp_run()
double quat_norm(const Quaternion& q);
/// @brief Normalize a finite nonzero quaternion using scaled components.
/// Divides by largest magnitude before computing the norm, allowing near-maximum and subnormal components without avoidable overflow/underflow. Returns owned values, no shared state.
/// @param q Complete finite nonzero raw quaternion.
/// @return Owned normalized quaternion.
/// @throws std::invalid_argument Nonfinite component or all-zero quaternion.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::quat_normalize({2,0,0,0})[0]); // 1
/// @endcode
/// @see slerp_run()
Quaternion quat_normalize(const Quaternion& q);
/// @brief Return the shorter rotation angle in radians.
/// Normalizes both and aligns their signs, then uses four times atan2(norm(a-b),norm(a+b)) for stable small-angle measurement. Range[0,pi]; q and -q have angle zero. Constant work, no mutation/shared state.
/// @param a Finite nonzero rotation representative.
/// @param b Finite nonzero rotation representative.
/// @return Shorter full rotation angle in radians.
/// @throws std::invalid_argument Nonfinite or all-zero representative.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::quat_angle_between({1,0,0,0},{-1,0,0,0})); // 0
/// @endcode
/// @see slerp_run()
double quat_angle_between(const Quaternion& a, const Quaternion& b);
/// @brief Construct a rotation about the positive z axis.
/// Returns{cos(angle/2),0,0,sin(angle/2)}. Constant work and owned values, no shared state.
/// @param radians Finite full rotation angle in radians.
/// @return Owned unit rotation representative.
/// @throws std::invalid_argument Nonfinite angle.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::quat_z_rotation(0)[0]); // 1
/// @endcode
/// @see slerp_run()
Quaternion quat_z_rotation(double radians);
/// @brief Interpolate a checked shortest quaternion arc and expose its actual branch.
/// Both run/trace return identical constant-size actual state; no replay. Normalize endpoints safely; negative dot flips target sign because q and -q represent the same rotation. Dot>0.9995 uses normalized linear interpolation to avoid small sine denominators. Otherwise sine weights traverse the quaternion great circle at constant angular rate. The normalized-linear branch approximates this and is not exactly constant-rate. At t=1 the representative may be -q1 but the rotation matches; at exactly180degrees both shorter arcs are valid and the supplied representative selects one. Constant time/storage, owned output, no shared state.
/// @param q0 Finite nonzero starting rotation representative.
/// @param q1 Finite nonzero final rotation representative.
/// @param t Finite interpolation fraction in[0,1].
/// @return Owned normalized endpoints/result and actual branch/weights.
/// @throws std::invalid_argument Nonfinite/zero representative or t outside[0,1].
/// @par Example
/// @code
/// auto r=dm::algorithms::slerp_run({1,0,0,0},{1,0,0,0},.5);
/// std::print("{} {}\n",r.value[0],r.normalized_linear); // 1 true
/// @endcode
/// @par References
/// - K. Shoemake, Animating rotation with quaternion curves, SIGGRAPH1985, DOI10.1145/325334.325242.
/// @see slerp()
SlerpRun slerp_run(Quaternion q0, Quaternion q1, double t);
/// @brief Interpolate a checked shortest quaternion arc and expose its actual branch.
/// Both run/trace return identical constant-size actual state; no replay. Normalize endpoints safely; negative dot flips target sign because q and -q represent the same rotation. Dot>0.9995 uses normalized linear interpolation to avoid small sine denominators. Otherwise sine weights traverse the quaternion great circle at constant angular rate. The normalized-linear branch approximates this and is not exactly constant-rate. At t=1 the representative may be -q1 but the rotation matches; at exactly180degrees both shorter arcs are valid and the supplied representative selects one. Constant time/storage, owned output, no shared state.
/// @param q0 Finite nonzero starting rotation representative.
/// @param q1 Finite nonzero final rotation representative.
/// @param t Finite interpolation fraction in[0,1].
/// @return Owned normalized endpoints/result and actual branch/weights.
/// @throws std::invalid_argument Nonfinite/zero representative or t outside[0,1].
/// @par Example
/// @code
/// auto r=dm::algorithms::slerp_trace({1,0,0,0},{1,0,0,0},.5);
/// std::print("{} {}\n",r.value[0],r.normalized_linear); // 1 true
/// @endcode
/// @par References
/// - K. Shoemake, Animating rotation with quaternion curves, SIGGRAPH1985, DOI10.1145/325334.325242.
/// @see slerp()
SlerpRun slerp_trace(Quaternion q0, Quaternion q1, double t);
/// @brief Return the checked shortest-arc quaternion interpolation.
/// Same actual producer as slerp_run, including normalized-linear near-parallel branch and t validation.
/// @param q0 Finite nonzero start representative.
/// @param q1 Finite nonzero target representative.
/// @param t Finite interpolation fraction in[0,1].
/// @return Owned normalized representative.
/// @throws std::invalid_argument Malformed rotation or fraction.
/// @par Example
/// @code
/// std::print("{}\n",dm::algorithms::slerp({1,0,0,0},{-1,0,0,0},.5)[0]); // 1
/// @endcode
/// @see slerp_run()
Quaternion slerp(Quaternion q0, Quaternion q1, double t);
} // namespace datamunge::algorithms
