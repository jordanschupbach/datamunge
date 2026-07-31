#pragma once

/// \file slerp.hpp
/// \brief Spherical linear interpolation (Slerp) of unit quaternions.
///
/// A rotation in 3D is a unit quaternion, a point on the unit 3-sphere. To animate smoothly
/// from orientation \f$q_0\f$ to \f$q_1\f$, we want to move along the *great-circle arc*
/// between them at *constant angular velocity* -- not along the chord (which would speed up in
/// the middle). *Slerp* (Shoemake, 1985) does exactly this:
/// \f[
///   \mathrm{slerp}(q_0, q_1; t) = \frac{\sin((1-t)\Omega)}{\sin\Omega}\,q_0
///     + \frac{\sin(t\Omega)}{\sin\Omega}\,q_1,\qquad \cos\Omega = q_0\cdot q_1,
/// \f]
/// where \f$\Omega\f$ is the angle between the quaternions. The result stays on the unit
/// sphere and sweeps equal angles in equal time. Two practical details: because \f$q\f$ and
/// \f$-q\f$ are the same rotation, we flip \f$q_1\f$'s sign when the dot product is negative
/// to take the *shorter* arc; and when the quaternions are nearly parallel we fall back to
/// normalized linear interpolation to avoid dividing by \f$\sin\Omega\approx 0\f$.
///
/// Slerp is the standard tool for keyframe rotation animation, camera interpolation, and
/// blending orientations.

#include <array>
#include <cmath>

namespace datamunge::algorithms {

/// \brief A quaternion \f$w + xi + yj + zk\f$ stored as {w, x, y, z}.
using Quaternion = std::array<double, 4>;

/// \brief Dot product of two quaternions.
inline double quat_dot(const Quaternion& a, const Quaternion& b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3];
}

/// \brief Euclidean norm of a quaternion.
inline double quat_norm(const Quaternion& q) { return std::sqrt(quat_dot(q, q)); }

/// \brief Return \p q scaled to unit length.
inline Quaternion quat_normalize(const Quaternion& q) {
    double n = quat_norm(q);
    return {q[0] / n, q[1] / n, q[2] / n, q[3] / n};
}

/// \brief Spherical linear interpolation between unit quaternions \p q0 and \p q1 at \p t.
/// \param t interpolation parameter, 0 gives q0 and 1 gives q1.
/// \returns a unit quaternion on the shorter great-circle arc.
inline Quaternion slerp(Quaternion q0, Quaternion q1, double t) {
    q0 = quat_normalize(q0);
    q1 = quat_normalize(q1);
    double dot = quat_dot(q0, q1);

    // Take the shorter arc: q and -q represent the same rotation.
    if (dot < 0.0) {
        for (double& c : q1) c = -c;
        dot = -dot;
    }

    // Nearly parallel: fall back to normalized linear interpolation.
    constexpr double kParallelThreshold = 0.9995;
    if (dot > kParallelThreshold) {
        Quaternion r = {q0[0] + t * (q1[0] - q0[0]), q0[1] + t * (q1[1] - q0[1]),
                        q0[2] + t * (q1[2] - q0[2]), q0[3] + t * (q1[3] - q0[3])};
        return quat_normalize(r);
    }

    double omega = std::acos(dot);
    double sin_omega = std::sin(omega);
    double a = std::sin((1.0 - t) * omega) / sin_omega;
    double b = std::sin(t * omega) / sin_omega;
    return {a * q0[0] + b * q1[0], a * q0[1] + b * q1[1],
            a * q0[2] + b * q1[2], a * q0[3] + b * q1[3]};
}

/// \brief Angle (radians) of the shorter rotation between two unit quaternions.
inline double quat_angle_between(const Quaternion& a, const Quaternion& b) {
    double d = std::fabs(quat_dot(quat_normalize(a), quat_normalize(b)));
    if (d > 1.0) d = 1.0;
    return 2.0 * std::acos(d);  // full rotation angle
}

}  // namespace datamunge::algorithms
