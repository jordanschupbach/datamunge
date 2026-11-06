/// @file world.cpp
/// @brief Native convex rigid dynamics, sequential constraints and exact solver-state rewind.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <datamunge/sim/world.hpp>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "collision.hpp"

namespace datamunge::sim {
namespace detail {
/// @brief Mutable simulation values for one body; no renderer objects are retained.
struct RuntimeBody {
  BodyId       id;              ///< Stable world-scoped identifier.
  Body         value;           ///< Current pose, velocities and physical parameters.
  Tensor3      inverse_inertia; ///< Principal inverse inertias in local coordinates.
  Vec3         force{};         ///< World force accumulated until a successful outer step.
  Vec3         torque{};        ///< World torque accumulated until a successful outer step.
  double       quiet_time{0};   ///< Time below the sleep threshold.
  bool         sleeping{false}; ///< Whether integration is currently suspended.
  mutable bool transform_valid{
      false}; ///< Whether world COM/inverse-inertia caches match the full key.
  mutable Vec3 cache_position{},
      cache_com{}; ///< Reference pose and local COM used by the transform cache.
  mutable Quaternion cache_rotation{}; ///< Orientation used by the transform cache.
  mutable Vec3       world_center{};   ///< Cached world COM for point Jacobians.
};
/// @brief A compiled joint's authored frames and persistent actuator state.
struct RuntimeJoint {
  Joint                 value;            ///< Validated public settings.
  Vec3                  child_anchor;     ///< Local child pivot.
  Vec3                  parent_anchor;    ///< Parent-local pivot, or world pivot.
  Vec3                  parent_axis;      ///< Parent-local generalized axis, or world direction.
  Vec3                  child_axis;       ///< Child-local generalized axis.
  Quaternion            reference;        ///< Authored child-relative-to-parent orientation.
  double                force{0};         ///< Accumulated generalized actuator force.
  std::array<double, 6> impulses{};       ///< Cached bilateral constraint row impulses.
  double                limit_impulse{0}; ///< Signed accumulated unilateral limit impulse.
  std::size_t           drive_cache{0};   ///< Pose-local projected-drive response index.
  double motor_mass{0};        ///< Bilaterally projected inverse drive inertia, rebuilt with pose.
  double armature_rate{0};     ///< Auxiliary generalized rotor velocity.
  double dry_impulse{0};       ///< Current-substep dry friction impulse.
  double motor_target{0};      ///< Requested relative velocity.
  double motor_gain{0};        ///< Implicit viscous servo gain.
  double motor_limit{0};       ///< Absolute effort bound.
  double motor_feedforward{0}; ///< Additive generalized effort.
  double motor_impulse{0};     ///< Current-substep bounded motor impulse.
  double damping_impulse{0};   ///< Current-substep passive damping impulse.
  double previous_delta{0};    ///< Last warm-start substep duration.
};
/// @brief Coupled scalar coordinates with an ideal transmission ratio.
struct RuntimeGear {
  BodyId master; ///< Driving joint child.
  BodyId slave;  ///< Driven joint child.
  double ratio;  ///< Signed generalized-coordinate multiplier.
  double offset; ///< Coordinate offset in radians/metres.
};
/// @brief One solver contact including tangential warm-start impulse.
struct SolverContact {
  Contact     value;             ///< Public contact observation and normal impulse.
  Vec3        angular{};         ///< Combined rolling/spinning contact impulse.
  Vec3        friction{};        ///< Tangential impulse in world coordinates.
  double      bounce{0};         ///< Restitution velocity selected at contact creation.
  std::size_t manifold_count{1}; ///< Points sharing this body's contact pair.
};
/// @brief Complete rewindable native state, tied to one world and topology generation.
struct WorldState {
  std::uint64_t              owner{0};    ///< Identity of the originating world.
  std::uint64_t              topology{0}; ///< Topology/configuration revision.
  std::vector<RuntimeBody>   bodies;      ///< Poses, velocities, forces and sleep timers.
  std::vector<RuntimeGear>   gears;       ///< Dynamic mimic/gear constraints.
  std::vector<RuntimeJoint>  joints;      ///< Joint reference frames and actuator accumulators.
  std::vector<SolverContact> contacts;    ///< Contact manifold and warm-start impulses.
  double                     previous_delta{0}; ///< Previous solver substep for warm-start scaling.
};
/// @brief Sweep-sort proxy with a body index and its conservative bounds.
struct Proxy {
  std::size_t index; ///< Index in live body storage.
  Bounds      box;   ///< Current or swept bounds.
};
/// @brief Compute exact principal inverse inertia of uniform boxes, spheres and capsules.
inline Tensor3 inverse_inertia(const Body& b) {
  return b.motion == Motion::Dynamic ? invert_tensor(body_tensor(b)) : Tensor3{};
}
/// @brief Multiply by the projected inverse mass matrix.
inline Vec3 inverse_mass(const RuntimeBody& b, Vec3 v) {
  if (b.value.motion != Motion::Dynamic)
    return {};
  for (int i = 0; i < 3; ++i)
    v[i] = b.value.allowed_translation[i] ? v[i] / b.value.mass : 0;
  return v;
}
/// @brief Cache a body's exact COM transform once per pose/local-COM state.
/// @param b Body whose mutable derived cache is refreshed; physical state is unchanged.
inline void body_transform(const RuntimeBody& b) {
  if (b.transform_valid && b.cache_position == b.value.position &&
      b.cache_rotation == b.value.rotation && b.cache_com == b.value.center_of_mass)
    return;
  b.world_center    = detail::center(b.value);
  b.cache_position  = b.value.position;
  b.cache_rotation  = b.value.rotation;
  b.cache_com       = b.value.center_of_mass;
  b.transform_valid = true;
}
/// @brief Read the cached COM of one runtime body.
/// @param b Live body.
/// @return World COM, metres.
inline Vec3 center(const RuntimeBody& b) {
  body_transform(b);
  return b.world_center;
}
/// @brief Multiply by the full world inverse inertia including rotational locks.
/// @param b Live body with authored local inverse inertia.
/// @param v World torque or angular impulse.
/// @return World angular acceleration or velocity increment.
inline Vec3 inertia(const RuntimeBody& b, Vec3 v) {
  if (b.value.motion != Motion::Dynamic)
    return {};
  for (int i = 0; i < 3; ++i)
    if (!b.value.allowed_rotation[i])
      v[i] = 0;
  v = rotate(conjugate(b.value.rotation), v);
  v = apply_tensor(b.inverse_inertia, v);
  v = rotate(b.value.rotation, v);
  for (int i = 0; i < 3; ++i)
    if (!b.value.allowed_rotation[i])
      v[i] = 0;
  return v;
}
/// @brief Velocity at one world point, including angular motion.
inline Vec3 velocity(const RuntimeBody* b, const Vec3& point) {
  return b ? add(b->value.linear_velocity, cross(b->value.angular_velocity, sub(point, center(*b))))
           : Vec3{};
}
/// @brief Apply a point impulse to a dynamic body, waking it for nonzero impulse.
inline void point_impulse(RuntimeBody* b, const Vec3& impulse, const Vec3& point) {
  if (!b || b->value.motion != Motion::Dynamic)
    return;
  auto dv = inverse_mass(*b, impulse), dw = inertia(*b, cross(sub(point, center(*b)), impulse));
  b->value.linear_velocity  = add(b->value.linear_velocity, dv);
  b->value.angular_velocity = add(b->value.angular_velocity, dw);
  if (length(dv) + length(dw) > 1e-8) {
    if (b->sleeping)
      b->quiet_time = 0;
    b->sleeping = false;
  }
}
/// @brief Effective inverse mass along a point-constraint direction.
inline double point_mass(const RuntimeBody* b, const Vec3& point, const Vec3& axis) {
  if (!b)
    return 0;
  auto arm = sub(point, center(*b)), r = cross(arm, axis);
  return dot(axis, inverse_mass(*b, axis)) + dot(r, inertia(*b, r));
}
/// @brief Apply a pure angular impulse to a dynamic body.
inline void angular_impulse(RuntimeBody* b, const Vec3& impulse) {
  if (!b || b->value.motion != Motion::Dynamic)
    return;
  auto change               = inertia(*b, impulse);
  b->value.angular_velocity = add(b->value.angular_velocity, change);
  if (length(change) > 1e-8) {
    if (b->sleeping)
      b->quiet_time = 0;
    b->sleeping = false;
  }
}
/// @brief Translate and rotate a copied body by constant world velocities.
inline Body predicted(Body body, double delta, bool gyro = false) {
  if (body.motion != Motion::Static) {
    integrate_pose(body, delta, gyro);
  }
  return body;
}
/// @brief Conservative advancement from disjoint convex shapes; never silently drops an unresolved
/// sweep.
inline double
impact_time(const Body& a, const Body& b, double delta, double slop, bool gyro = false) {
  double speed = length(sub(a.linear_velocity, b.linear_velocity)) +
                 rotation_radius(a) * angular_bound(a, gyro) +
                 rotation_radius(b) * angular_bound(b, gyro);
  if (speed < 1e-10)
    return delta;
  double time    = 0;
  auto   initial = distance(a, b);
  if (initial.value <= slop)
    return delta;
  // Stop inside the manifold margin so a swept stop produces response contacts.
  const double target = .5 * slop;
  for (int i = 0; i < 128; ++i) {
    auto current = distance(predicted(a, time, gyro), predicted(b, time, gyro));
    if (current.value <= target + 1e-8)
      return std::max(1e-9, time);
    double step = .9 * (current.value - target) / speed;
    if (time + step >= delta)
      return delta;
    time += std::max(step, 1e-10);
  }
  // Consume this proved-safe interval and continue sweeping in the outer event
  // loop. Never advance an unresolved remainder without its distance bound.
  return std::max(1e-9, time);
}
} // namespace detail
using namespace detail;
/// @brief Translation-unit-local validation and construction helpers.
namespace {
/// @brief Unique world identity for snapshot compatibility; no shared physics state.
std::atomic<std::uint64_t> next_world{1};
/// @brief Reject a nonfinite or unreasonably large force/impulse input.
void check_force(const Vec3& force) {
  for (double v : force)
    if (!std::isfinite(v) || std::abs(v) > 100000)
      throw std::invalid_argument("invalid simulation force or impulse");
}
} // namespace
/// @brief Exclusive bounded native solver state and reusable working memory.
struct World::Impl {
  WorldOptions       options; ///< Validated solver controls.
  compute::Context   context; ///< Explicit execution owner; currently native CPU world execution.
  detail::WorldState state;   ///< Live simulation state.
  detail::WorldState backup;  ///< Capacity-reserved failure rollback, reused each outer step.
  BodyId             next_id{1}; ///< Never-reused body ID counter.
  std::unordered_map<BodyId, std::size_t>
                     indices; ///< Stable-ID lookup, changed only on topology edits.
  std::vector<Proxy> proxies; ///< Reusable sweep-sort storage.
  std::vector<std::array<std::size_t, 2>> pairs;    ///< Reusable broadphase candidates.
  std::vector<SolverContact>              previous; ///< Previous manifold for warm starting.
  std::vector<Contact>                    visible;  ///< Public final contact observations.

  std::vector<std::pair<std::size_t, Body>>
      replacements; ///< Reused subtree teleport proposal storage.

  /// @brief One transient joint velocity row; no pointers are retained in snapshots.
  struct JointRow {
    Vec3   axis;         ///< Unit world-space constraint direction.
    Vec3   child_point;  ///< Child anchor for point rows.
    Vec3   parent_point; ///< Parent anchor for point rows.
    double bias;         ///< Position-error velocity in m/s or rad/s.
    bool   angular;      ///< Pure angular rather than point constraint.
  };

  /// @brief Cached bilateral Jacobian row and its two rigid-body wrenches.
  struct BlockRow {
    RuntimeJoint* joint;      ///< Borrowed live incoming joint.
    JointRow      row;        ///< Cached world-space constraint direction and anchors.
    std::size_t   slot;       ///< Bilateral row index, or six for an auxiliary rotor.
    RuntimeBody*  body[2];    ///< Borrowed child and optional parent.
    Vec3          linear[2];  ///< Signed world linear Jacobians.
    Vec3          angular[2]; ///< Signed world angular Jacobians.
  };
  /// @brief Frozen-pose response of every mechanism coordinate to one joint impulse.
  struct DriveResponse {
    std::vector<Vec3>   linear;    ///< Projected world linear response per body, m/s per N s.
    std::vector<Vec3>   angular;   ///< Projected world angular response per body, rad/s per N s.
    std::vector<double> reactions; ///< Opposing bilateral impulses per unit drive impulse.
  };
  std::vector<DriveResponse>
       drive_responses; ///< Rebuilt once per pose; reused in all motor sweeps.
  bool block_valid{
      false}; ///< Whether the factor/drive responses match the borrowed body topology and pose.
  double        block_delta{0};    ///< Substep interval used by cached drift rows.
  std::uint64_t block_topology{0}; ///< Topology version used by borrowed cached row pointers.
  std::vector<std::array<double, 7>>
                        block_poses; ///< Pose keys; forces and velocities do not change the factor.
  std::vector<BlockRow> block_rows;  ///< Pose-local bilateral rows.
  std::vector<double>   block_factor; ///< Reused pose-local dense Cholesky factor.
  std::vector<double>   block_rhs;    ///< Reused projection/Schur-complement solve vector.

  /// @brief Reserve bounded working storage; optional coupled solve grows its dense workspace.
  /// @param settings Validated capacities and solver controls.
  /// @param owner Explicit execution context, currently CPU.
  Impl(WorldOptions settings, compute::Context owner)
      : options(settings), context(std::move(owner)) {
    state.owner = next_world.fetch_add(1);
    state.bodies.reserve(options.body_capacity);
    backup.bodies.reserve(options.body_capacity);
    state.joints.reserve(options.body_capacity);
    backup.joints.reserve(options.body_capacity);
    state.contacts.reserve(options.contact_capacity);
    backup.contacts.reserve(options.contact_capacity);
    proxies.reserve(options.body_capacity);
    pairs.reserve(options.pair_capacity);
    previous.reserve(options.contact_capacity);
    visible.reserve(options.contact_capacity);
    replacements.reserve(options.body_capacity);
  }
  /// @brief Resolve a live body or reject a foreign/stale identifier.
  RuntimeBody& get(BodyId id) {
    auto it = indices.find(id);
    if (it == indices.end())
      throw std::out_of_range("unknown simulation body");
    return state.bodies[it->second];
  }
  /// @brief Resolve an optional parent; ID zero denotes the fixed world.
  RuntimeBody* parent(BodyId id) { return id ? &get(id) : nullptr; }
  /// @brief Resolve the one constraint attached to a child.
  RuntimeJoint& joint(BodyId id) {
    auto it = std::find_if(state.joints.begin(), state.joints.end(), [&](const RuntimeJoint& j) {
      return j.value.child == id;
    });
    if (it == state.joints.end())
      throw std::out_of_range("body has no joint");
    return *it;
  }
  /// @brief Clear contact caches after external edits, preserving body IDs.
  void invalidate_contacts() {
    block_valid = false;
    state.contacts.clear();
    visible.clear();
    state.previous_delta = 0;
    for (auto& joint : state.joints) {
      joint.impulses       = {};
      joint.limit_impulse  = 0;
      joint.previous_delta = 0;
    }
  }
  /// @brief Test the direct-joint collision suppression rule.
  bool connected(BodyId a, BodyId b) const {
    for (auto g : state.gears)
      if ((g.master == a && g.slave == b) || (g.master == b && g.slave == a))
        return true;
    for (const auto& j : state.joints)
      if ((j.value.child == a && j.value.parent == b) ||
          (j.value.child == b && j.value.parent == a))
        return true;
    return false;
  }
  /// @brief Generate stable sweep-and-prune pairs; throw rather than truncate capacity overflow.
  void broadphase(double swept) {
    proxies.clear();
    pairs.clear();
    for (std::size_t i = 0; i < state.bodies.size(); ++i)
      proxies.push_back(
          {i,
           swept > 0 ? swept_bounds(state.bodies[i].value, swept, options.gyroscopic)
                     : bounds(state.bodies[i].value)});
    std::sort(proxies.begin(), proxies.end(), [&](const Proxy& a, const Proxy& b) {
      return a.box.lo[0] != b.box.lo[0] ? a.box.lo[0] < b.box.lo[0]
                                        : state.bodies[a.index].id < state.bodies[b.index].id;
    });
    for (std::size_t i = 0; i < proxies.size(); ++i)
      for (std::size_t j = i + 1; j < proxies.size(); ++j) {
        if (proxies[j].box.lo[0] > proxies[i].box.hi[0] + options.contact_slop)
          break;
        auto  ai = proxies[i].index, bi = proxies[j].index;
        auto &a = state.bodies[ai], &b = state.bodies[bi];
        if (a.value.motion == Motion::Static && b.value.motion == Motion::Static &&
            !a.value.sensor && !b.value.sensor)
          continue;
        if (!(a.value.collision_group & b.value.collision_mask) ||
            !(b.value.collision_group & a.value.collision_mask) || connected(a.id, b.id) ||
            !overlap(proxies[i].box, proxies[j].box, options.contact_slop))
          continue;
        if (pairs.size() == options.pair_capacity)
          throw std::runtime_error("physics broadphase pair capacity exhausted");
        if (a.id > b.id)
          std::swap(ai, bi);
        pairs.push_back({ai, bi});
      }
    std::sort(pairs.begin(), pairs.end(), [&](const auto& a, const auto& b) {
      auto a0 = state.bodies[a[0]].id, b0 = state.bodies[b[0]].id;
      return a0 != b0 ? a0 < b0 : state.bodies[a[1]].id < state.bodies[b[1]].id;
    });
  }
  /// @brief Rebuild exact manifolds and optionally apply cached response impulses.
  void build_contacts(double delta, bool warm) {
    previous = state.contacts;
    state.contacts.clear();
    broadphase(0);
    for (auto pair : pairs) {
      auto &a = state.bodies[pair[0]], &b = state.bodies[pair[1]];
      for (std::size_t shape_a = 0; shape_a < component_count(a.value); ++shape_a)
        for (std::size_t shape_b = 0; shape_b < component_count(b.value); ++shape_b) {
          auto av = component(a.value, shape_a), bv = component(b.value, shape_b);
          if (!overlap(bounds(av), bounds(bv), options.contact_slop))
            continue;
          auto       manifold  = collide(av, bv, options.contact_slop);
          const auto first_old = std::lower_bound(
              previous.begin(),
              previous.end(),
              std::array<BodyId, 2>{a.id, b.id},
              [](const SolverContact& old, const auto& ids) {
                return old.value.first < ids[0] ||
                       (old.value.first == ids[0] && old.value.second < ids[1]);
              });
          for (std::size_t i = 0; i < manifold.count; ++i) {
            if (state.contacts.size() == options.contact_capacity)
              throw std::runtime_error("physics contact capacity exhausted");
            auto&         p = manifold.points[i];
            SolverContact c{
                {a.id,
                 b.id,
                 p.point,
                 p.normal,
                 p.separation,
                 0,
                 a.value.sensor || b.value.sensor,
                 shape_a,
                 shape_b},
                {},
                {},
                0};
            c.manifold_count = manifold.count;
            double vn        = dot(sub(velocity(&b, p.point), velocity(&a, p.point)), p.normal);
            c.bounce = vn < -1 ? -std::max(a.value.restitution, b.value.restitution) * vn : 0;
            if ((warm || options.persistent_contacts) && !c.value.sensor) {
              double               nearest = .02;
              const SolverContact* match   = nullptr;
              for (auto it = first_old;
                   it != previous.end() && it->value.first == a.id && it->value.second == b.id;
                   ++it) {
                const auto& old = *it;
                if (old.value.first_shape == shape_a && old.value.second_shape == shape_b &&
                    dot(old.value.normal, p.normal) > .95) {
                  double d = length(sub(old.value.point, p.point));
                  if (d < nearest) {
                    nearest = d;
                    match   = &old;
                  }
                }
              }
              if (match) {
                double scale    = warm && state.previous_delta > 0
                                      ? std::min(2., delta / state.previous_delta)
                                      : 1;
                c.value.impulse = match->value.impulse * scale;
                c.friction      = mul(match->friction, scale);
                c.angular       = mul(match->angular, scale);
                auto impulse    = add(mul(p.normal, c.value.impulse), c.friction);
                if (warm) {
                  point_impulse(&a, mul(impulse, -1), p.point);
                  point_impulse(&b, impulse, p.point);
                  angular_impulse(&a, mul(c.angular, -1));
                  angular_impulse(&b, c.angular);
                }
              }
            }
            state.contacts.push_back(c);
          }
        }
    }
    state.previous_delta = delta;
  }
  /// @brief Apply one incremental row impulse with equal/opposite parent response.
  void apply_row(RuntimeBody& c, RuntimeBody* p, const JointRow& row, double value) {
    const auto impulse = mul(row.axis, value);
    if (row.angular) {
      angular_impulse(&c, impulse);
      angular_impulse(p, mul(impulse, -1));
    } else {
      point_impulse(&c, impulse, row.child_point);
      point_impulse(p, mul(impulse, -1), row.parent_point);
    }
  }
  /// @brief Evaluate effective inverse mass for an angular or point row.
  double row_mass(const RuntimeBody& c, const RuntimeBody* p, const JointRow& row) {
    if (!row.angular)
      return point_mass(&c, row.child_point, row.axis) + point_mass(p, row.parent_point, row.axis);
    return dot(row.axis, inertia(c, row.axis)) + (p ? dot(row.axis, inertia(*p, row.axis)) : 0);
  }
  /// @brief Evaluate relative row velocity before bias correction.
  double row_rate(const RuntimeBody& c, const RuntimeBody* p, const JointRow& row) {
    if (!row.angular)
      return dot(sub(velocity(&c, row.child_point), velocity(p, row.parent_point)), row.axis);
    return dot(sub(c.value.angular_velocity, p ? p->value.angular_velocity : Vec3{}), row.axis);
  }
  /// @brief Build five bilateral rows from the current authored joint frames.
  std::array<JointRow, 6> joint_rows(const RuntimeJoint& j, double delta) {
    auto&      c    = get(j.value.child);
    auto*      p    = parent(j.value.parent);
    const auto pq   = p ? p->value.rotation : Quaternion{1, 0, 0, 0};
    const auto cp   = add(c.value.position, rotate(c.value.rotation, j.child_anchor));
    const auto pp   = p ? add(p->value.position, rotate(pq, j.parent_anchor)) : j.parent_anchor;
    const auto axis = unit(rotate(pq, j.parent_axis));
    const auto u    = unit(cross(axis, std::abs(axis[0]) < .8 ? Vec3{1, 0, 0} : Vec3{0, 1, 0})),
               v = cross(axis, u), error = sub(cp, pp);
    std::array<JointRow, 6> rows{};
    if (j.value.type == JointType::Hinge) {
      for (int k = 0; k < 3; ++k)
        rows[k] = {basis(k), cp, pp, .2 * error[k] / delta, false};
      const auto er = cross(axis, rotate(c.value.rotation, j.child_axis));
      rows[3]       = {u, cp, pp, .2 * dot(er, u) / delta, true};
      rows[4]       = {v, cp, pp, .2 * dot(er, v) / delta, true};
    } else if (j.value.type == JointType::Fixed) {
      auto er = rotation_error(product(pq, j.reference), c.value.rotation);
      for (int k = 0; k < 3; ++k) {
        rows[k]     = {basis(k), cp, pp, .2 * error[k] / delta, false};
        rows[k + 3] = {basis(k), cp, pp, -.2 * er[k] / delta, true};
      }
    } else {
      rows[0]       = {u, cp, pp, .2 * dot(error, u) / delta, false};
      rows[1]       = {v, cp, pp, .2 * dot(error, v) / delta, false};
      const auto er = rotation_error(product(pq, j.reference), c.value.rotation);
      for (int k = 0; k < 3; ++k)
        rows[k + 2] = {basis(k), cp, pp, -.2 * er[k] / delta, true};
    }
    return rows;
  }
  /// @brief Construct a generalized coordinate row for a unilateral hard limit.
  JointRow limit_row(const RuntimeJoint& j) {
    auto&      c  = get(j.value.child);
    auto*      p  = parent(j.value.parent);
    const auto pq = p ? p->value.rotation : Quaternion{1, 0, 0, 0};
    return {
        unit(rotate(pq, j.parent_axis)),
        add(c.value.position, rotate(c.value.rotation, j.child_anchor)),
        p ? add(p->value.position, rotate(pq, j.parent_anchor)) : j.parent_anchor,
        0,
        j.value.type == JointType::Hinge};
  }
  /// @brief Apply cached bilateral/limit impulses once at the start of a substep.
  void warm_joints(double delta) {
    for (auto& j : state.joints) {
      auto&        c     = get(j.value.child);
      auto*        p     = parent(j.value.parent);
      const auto   rows  = joint_rows(j, delta);
      const double scale = j.previous_delta > 0 ? std::min(2., delta / j.previous_delta) : 0;
      for (std::size_t k = 0; !options.block_joints && k < 6; ++k) {
        j.impulses[k] *= scale;
        if (k < 5 || j.value.type == JointType::Fixed)
          apply_row(c, p, rows[k], j.impulses[k]);
      }
      j.limit_impulse *= scale;
      if (j.value.limited && j.value.type != JointType::Fixed)
        apply_row(c, p, limit_row(j), j.limit_impulse);
      j.previous_delta = delta;
    }
  }
  /// @brief Evaluate the generalized coordinate against the original authored frames.
  JointState coordinate(const RuntimeJoint& j) {
    if (j.value.type == JointType::Fixed)
      return {};
    auto& c    = get(j.value.child);
    auto* p    = parent(j.value.parent);
    auto  pq   = p ? p->value.rotation : Quaternion{1, 0, 0, 0};
    auto  pa   = p ? add(p->value.position, rotate(pq, j.parent_anchor)) : j.parent_anchor;
    auto  ca   = add(c.value.position, rotate(c.value.rotation, j.child_anchor));
    auto  axis = rotate(pq, j.parent_axis);
    if (j.value.type == JointType::Slider)
      return {dot(sub(ca, pa), axis), dot(sub(velocity(&c, ca), velocity(p, pa)), axis)};
    auto   relative = product(conjugate(pq), c.value.rotation);
    auto   q        = product(relative, conjugate(j.reference));
    double angle =
        2 * std::atan2(
                q[1] * j.parent_axis[0] + q[2] * j.parent_axis[1] + q[3] * j.parent_axis[2], q[0]);
    angle = std::remainder(angle, 2 * std::acos(-1.));
    return {
        angle, dot(sub(c.value.angular_velocity, p ? p->value.angular_velocity : Vec3{}), axis)};
  }
  /// @brief Solve implicit passive damping and effort-clipped velocity feedback.
  /// @param j Live drive with passive reflected inertia and finite effort bound.
  /// @param dt Positive substep interval, seconds.
  void solve_motor(RuntimeJoint& j, double dt) {
    if (j.value.type == JointType::Fixed)
      return;
    auto&  c   = get(j.value.child);
    auto*  p   = parent(j.value.parent);
    auto   row = limit_row(j);
    double eff = options.block_joints ? j.motor_mass : row_mass(c, p, row);
    if (j.value.armature > 0 && !options.block_joints) {
      double change = -(row_rate(c, p, row) - j.armature_rate) / (eff + 1 / j.value.armature);
      apply_row(c, p, row, change);
      j.armature_rate -= change / j.value.armature;
    }
    if (j.value.friction > 0) {
      double old    = j.dry_impulse;
      j.dry_impulse = std::clamp(
          old - row_rate(c, p, row) / std::max(eff, 1e-20),
          -j.value.friction * dt,
          j.value.friction * dt);
      drive_impulse(j, row, j.dry_impulse - old);
    }
    if (j.value.damping > 0) {
      double gamma  = 1 / (j.value.damping * dt);
      double change = -(row_rate(c, p, row) + gamma * j.damping_impulse) / (eff + gamma);
      j.damping_impulse += change;
      drive_impulse(j, row, change);
    }
    if (j.motor_gain > 0 && j.motor_limit > 0) {
      double gamma = 1 / (j.motor_gain * dt), old = j.motor_impulse;
      double next = old + (j.motor_target + j.motor_feedforward / j.motor_gain -
                           row_rate(c, p, row) - gamma * old) /
                              (eff + gamma);
      j.motor_impulse = std::clamp(next, -j.motor_limit * dt, j.motor_limit * dt);
      drive_impulse(j, row, j.motor_impulse - old);
    }
  }
  /// @brief Solve gear constraints using combined wrenches, including common-parent cross terms.
  /// @param dt Positive substep interval for coordinate drift bias.
  void solve_gears(double dt) {
    for (auto g : state.gears) {
      auto& a  = joint(g.master);
      auto& b  = joint(g.slave);
      auto  ar = limit_row(a), br = limit_row(b);
      struct W {
        RuntimeBody* body;
        Vec3         linear{}, angular{};
      };
      std::array<W, 4> rows{};
      int              count = 0;
      auto             add   = [&](RuntimeBody* body, const JointRow& r, double sign, bool child) {
        if (!body)
          return;
        int i = 0;
        while (i < count && rows[i].body != body)
          ++i;
        if (i == count)
          rows[count++] = {body, {}, {}};
        if (r.angular)
          rows[i].angular = detail::add(rows[i].angular, mul(r.axis, sign));
        else {
          auto f          = mul(r.axis, sign);
          rows[i].linear  = detail::add(rows[i].linear, f);
          rows[i].angular = detail::add(
              rows[i].angular,
              cross(sub(child ? r.child_point : r.parent_point, center(*body)), f));
        }
      };
      add(&get(b.value.child), br, 1, true);
      add(parent(b.value.parent), br, -1, false);
      add(&get(a.value.child), ar, -g.ratio, true);
      add(parent(a.value.parent), ar, g.ratio, false);
      double mass = 0;
      for (int i = 0; i < count; ++i)
        mass += dot(rows[i].linear, inverse_mass(*rows[i].body, rows[i].linear)) +
                dot(rows[i].angular, inertia(*rows[i].body, rows[i].angular));
      if (options.block_joints) {
        // Eliminate bilateral/rotor reactions before computing the gear impulse.
        // Shared-parent cross terms are already combined in rows above.
        const auto n          = block_rows.size();
        double     correction = 0;
        for (std::size_t i = 0; i < n; ++i) {
          double value = 0;
          auto&  r     = block_rows[i];
          for (int side = 0; side < 2; ++side)
            for (int k = 0; k < count; ++k)
              if (r.body[side] && r.body[side] == rows[k].body)
                value += dot(r.linear[side], inverse_mass(*rows[k].body, rows[k].linear)) +
                         dot(r.angular[side], inertia(*rows[k].body, rows[k].angular));
          for (std::size_t k = 0; k < i; ++k)
            value -= block_factor[i * n + k] * block_rhs[k];
          block_rhs[i] = value / block_factor[i * n + i];
          correction += block_rhs[i] * block_rhs[i];
        }
        mass = std::max(0., mass - correction);
      }
      if (mass < 1e-15)
        continue;
      auto   qa = coordinate(a), qb = coordinate(b);
      double lambda = -(qb.velocity - g.ratio * qa.velocity +
                        .2 * (qb.position - g.ratio * qa.position - g.offset) / dt) /
                      mass;
      for (int i = 0; i < count; ++i) {
        auto& w = rows[i];
        point_impulse(w.body, mul(w.linear, lambda), center(*w.body));
        angular_impulse(w.body, mul(w.angular, lambda));
      }
      if (options.block_joints)
        solve_block();
    }
  }
  /// @brief Factor the full bilateral effective-mass matrix once per fixed-pose solve.
  /// @param dt Positive internal timestep used for drift bias.
  void prepare_block(double dt) {
    if (!options.block_joints) {
      block_rows.clear();
      return;
    }
    bool same = block_valid && block_delta == dt && block_topology == state.topology &&
                block_poses.size() == state.bodies.size();
    if (same)
      for (std::size_t i = 0; i < state.bodies.size(); ++i) {
        const auto&           b = state.bodies[i].value;
        std::array<double, 7> pose{
            b.position[0],
            b.position[1],
            b.position[2],
            b.rotation[0],
            b.rotation[1],
            b.rotation[2],
            b.rotation[3]};
        if (pose != block_poses[i]) {
          same = false;
          break;
        }
      }
    if (same)
      return;
    block_valid = false;
    block_rows.clear();
    for (auto& j : state.joints) {
      auto rows = joint_rows(j, dt);
      for (std::size_t k = 0; k < (j.value.armature > 0 && j.value.type != JointType::Fixed ? 7
                                   : j.value.type == JointType::Fixed                       ? 6
                                                                                            : 5);
           ++k) {
        if (k == 5 && j.value.type != JointType::Fixed)
          continue;
        auto  r = k == 6 ? limit_row(j) : rows[k];
        auto* c = &get(j.value.child);
        auto* p = parent(j.value.parent);
        if (row_mass(*c, p, r) < 1e-15)
          continue;
        BlockRow b{&j, r, k, {c, p}, {}, {}};
        for (int side = 0; side < 2; ++side) {
          if (!b.body[side])
            continue;
          double sign = side ? -1 : 1;
          if (r.angular)
            b.angular[side] = mul(r.axis, sign);
          else {
            b.linear[side]  = mul(r.axis, sign);
            b.angular[side] = cross(
                sub(side ? r.parent_point : r.child_point, center(*b.body[side])), b.linear[side]);
          }
        }
        block_rows.push_back(b);
      }
    }
    drive_responses.resize(state.joints.size());
    auto n = block_rows.size();
    if (n > 256)
      throw std::length_error("native coupled joint solver supports at most 256 bilateral rows");
    block_factor.assign(n * n, 0);
    block_rhs.resize(n);
    for (std::size_t i = 0; i < n; ++i)
      for (std::size_t j = 0; j <= i; ++j) {
        double value = 0;
        auto&  a     = block_rows[i];
        auto&  b     = block_rows[j];
        for (int x = 0; x < 2; ++x)
          for (int y = 0; y < 2; ++y)
            if (a.body[x] && a.body[x] == b.body[y])
              value += dot(a.linear[x], inverse_mass(*a.body[x], b.linear[y])) +
                       dot(a.angular[x], inertia(*a.body[x], b.angular[y]));
        if (i == j && a.slot == 6)
          value += 1 / a.joint->value.armature;
        for (std::size_t k = 0; k < j; ++k)
          value -= block_factor[i * n + k] * block_factor[j * n + k];
        if (i == j) {
          if (value <= 1e-14)
            throw std::runtime_error(
                "singular native bilateral mechanism; disable block_joints for redundant "
                "constraints");
          block_factor[i * n + j] = std::sqrt(value);
        } else
          block_factor[i * n + j] = value / block_factor[j * n + j];
      }
    for (auto& joint : state.joints) {
      if (joint.value.type == JointType::Fixed)
        continue;
      auto         row = limit_row(joint);
      auto*        c   = &get(joint.value.child);
      auto*        p   = parent(joint.value.parent);
      Vec3         linear[2]{}, angular[2]{};
      RuntimeBody* bodies[2]{c, p};
      for (int side = 0; side < 2; ++side)
        if (bodies[side]) {
          double sign = side ? -1 : 1;
          if (row.angular)
            angular[side] = mul(row.axis, sign);
          else {
            linear[side]  = mul(row.axis, sign);
            angular[side] = cross(
                sub(side ? row.parent_point : row.child_point, center(*bodies[side])),
                linear[side]);
          }
        }
      double correction = 0;
      for (std::size_t i = 0; i < n; ++i) {
        double value = 0;
        auto&  b     = block_rows[i];
        for (int x = 0; x < 2; ++x)
          for (int y = 0; y < 2; ++y)
            if (b.body[x] && b.body[x] == bodies[y])
              value += dot(b.linear[x], inverse_mass(*b.body[x], linear[y])) +
                       dot(b.angular[x], inertia(*b.body[x], angular[y]));
        for (std::size_t k = 0; k < i; ++k)
          value -= block_factor[i * n + k] * block_rhs[k];
        block_rhs[i] = value / block_factor[i * n + i];
        correction += block_rhs[i] * block_rhs[i];
      }
      joint.motor_mass  = std::max(0., row_mass(*c, p, row) - correction);
      joint.drive_cache = std::size_t(&joint - state.joints.data());
      auto& response    = drive_responses[joint.drive_cache];
      response.linear.assign(state.bodies.size(), {});
      response.angular.assign(state.bodies.size(), {});
      for (int side = 0; side < 2; ++side)
        if (bodies[side]) {
          auto index              = std::size_t(bodies[side] - state.bodies.data());
          response.linear[index]  = inverse_mass(*bodies[side], linear[side]);
          response.angular[index] = inertia(*bodies[side], angular[side]);
        }
      for (std::size_t i = n; i-- > 0;) {
        for (std::size_t k = i + 1; k < n; ++k)
          block_rhs[i] -= block_factor[k * n + i] * block_rhs[k];
        block_rhs[i] /= block_factor[i * n + i];
      }
      response.reactions.resize(n);
      for (std::size_t i = 0; i < n; ++i) {
        response.reactions[i] = -block_rhs[i];
        auto& r               = block_rows[i];
        for (int side = 0; side < 2; ++side)
          if (r.body[side]) {
            auto index = std::size_t(r.body[side] - state.bodies.data());
            response.linear[index] =
                sub(response.linear[index],
                    mul(inverse_mass(*r.body[side], r.linear[side]), block_rhs[i]));
            response.angular[index] =
                sub(response.angular[index],
                    mul(inertia(*r.body[side], r.angular[side]), block_rhs[i]));
          }
      }
    }
    block_poses.resize(state.bodies.size());
    for (std::size_t i = 0; i < state.bodies.size(); ++i) {
      const auto& b  = state.bodies[i].value;
      block_poses[i] = {
          b.position[0],
          b.position[1],
          b.position[2],
          b.rotation[0],
          b.rotation[1],
          b.rotation[2],
          b.rotation[3]};
    }
    block_delta    = dt;
    block_topology = state.topology;
    block_valid    = true;
  }
  /// @brief Apply a cached constrained drive response without refactoring/projecting every motor
  /// row.
  /// @param joint Live hinge/slider whose response was prepared at this pose.
  /// @param row Unconstrained drive row, used by the default sequential solver only.
  /// @param impulse Signed generalized impulse, N s or N m s.
  void drive_impulse(RuntimeJoint& joint, const JointRow& row, double impulse) {
    if (!options.block_joints) {
      apply_row(get(joint.value.child), parent(joint.value.parent), row, impulse);
      return;
    }
    if (impulse == 0)
      return;
    const auto& response = drive_responses[joint.drive_cache];
    for (std::size_t i = 0; i < state.bodies.size(); ++i) {
      auto& body = state.bodies[i];
      if (body.value.motion != Motion::Dynamic)
        continue;
      auto linear = mul(response.linear[i], impulse), angular = mul(response.angular[i], impulse);
      body.value.linear_velocity  = add(body.value.linear_velocity, linear);
      body.value.angular_velocity = add(body.value.angular_velocity, angular);
      if (length(linear) > 1e-8 || length(angular) > 1e-8) {
        body.sleeping   = false;
        body.quiet_time = 0;
      }
    }
    for (std::size_t i = 0; i < block_rows.size(); ++i) {
      auto&  r        = block_rows[i];
      double reaction = response.reactions[i] * impulse;
      if (r.slot == 6)
        r.joint->armature_rate -= reaction / r.joint->value.armature;
      else
        r.joint->impulses[r.slot] += reaction;
    }
  }
  /// @brief Project all coupled bilateral velocities in one exact linear solve.
  void solve_block() {
    auto n = block_rows.size();
    for (std::size_t i = 0; i < n; ++i) {
      auto&  b = block_rows[i];
      double value =
          -(row_rate(*b.body[0], b.body[1], b.row) + b.row.bias -
            (b.slot == 6 ? b.joint->armature_rate : 0));
      for (std::size_t k = 0; k < i; ++k)
        value -= block_factor[i * n + k] * block_rhs[k];
      block_rhs[i] = value / block_factor[i * n + i];
    }
    for (std::size_t i = n; i-- > 0;) {
      for (std::size_t k = i + 1; k < n; ++k)
        block_rhs[i] -= block_factor[k * n + i] * block_rhs[k];
      block_rhs[i] /= block_factor[i * n + i];
    }
    for (std::size_t i = 0; i < n; ++i) {
      auto& b = block_rows[i];
      if (b.slot == 6)
        b.joint->armature_rate -= block_rhs[i] / b.joint->value.armature;
      else
        b.joint->impulses[b.slot] += block_rhs[i];
      apply_row(*b.body[0], b.body[1], b.row, block_rhs[i]);
    }
  }
  /// @brief Solve cached bilateral rows and clamped unilateral/speculative joint limits.
  void solve_joints(double delta) {
    if (options.block_joints)
      solve_block();
    for (auto& j : state.joints) {
      auto&      c    = get(j.value.child);
      auto*      p    = parent(j.value.parent);
      const auto rows = joint_rows(j, delta);
      for (std::size_t k = 0; !options.block_joints && k < 6; ++k) {
        const double mass = row_mass(c, p, rows[k]);
        if (mass < 1e-15)
          continue;
        const double change = -(row_rate(c, p, rows[k]) + rows[k].bias) / mass;
        j.impulses[k] += change;
        apply_row(c, p, rows[k], change);
      }
      solve_motor(j, delta);
      if (j.value.limited && j.value.type != JointType::Fixed) {
        const auto   q    = coordinate(j);
        auto         row  = limit_row(j);
        const double mass = options.block_joints ? j.motor_mass : row_mass(c, p, row);
        if (mass < 1e-15)
          continue;
        int    side = 0;
        double gap  = 0;
        if (q.position < j.value.limit_min || q.position + q.velocity * delta < j.value.limit_min) {
          side = 1;
          gap  = q.position - j.value.limit_min;
        } else if (
            q.position > j.value.limit_max || q.position + q.velocity * delta > j.value.limit_max) {
          side = -1;
          gap  = q.position - j.value.limit_max;
        }
        const double old = j.limit_impulse;
        if (!side)
          j.limit_impulse = 0;
        else {
          const double bias   = (side * gap < 0 ? .2 : 1) * gap / delta;
          const double target = old - (row_rate(c, p, row) + bias) / mass;
          j.limit_impulse     = side > 0 ? std::max(0., target) : std::min(0., target);
        }
        drive_impulse(j, row, j.limit_impulse - old);
      }
    }
  }
  /// @brief Iterative normal/friction solve, followed by bounded positional correction.
  void solve_contacts(double delta) {
    prepare_block(delta);
    for (std::size_t iteration = 0; iteration < options.solver_iterations; ++iteration) {
      solve_joints(delta);
      solve_gears(delta);
      for (auto& c : state.contacts) {
        auto& v = c.value;
        if (v.sensor)
          continue;
        auto & a = get(v.first), &b = get(v.second);
        double mass = point_mass(&a, v.point, v.normal) + point_mass(&b, v.point, v.normal);
        if (mass <= 1e-15)
          continue;
        double rate  = dot(sub(velocity(&b, v.point), velocity(&a, v.point)), v.normal);
        double old   = v.impulse;
        double bias  = options.contact_stabilization
                           ? std::max(0., -v.separation - options.contact_slop) *
                                options.position_correction / delta
                           : 0;
        v.impulse    = std::max(0., old + (std::max(c.bounce, bias) - rate) / mass);
        auto impulse = mul(v.normal, v.impulse - old);
        point_impulse(&a, mul(impulse, -1), v.point);
        point_impulse(&b, impulse, v.point);
        auto relative =
            sub(add(velocity(&b, v.point), b.value.surface_velocity),
                add(velocity(&a, v.point), a.value.surface_velocity));
        auto   tangent = sub(relative, mul(v.normal, dot(relative, v.normal)));
        double speed   = length(tangent);
        if (speed > 1e-10) {
          auto   direction = mul(tangent, 1 / speed);
          double effective =
              point_mass(&a, v.point, direction) + point_mass(&b, v.point, direction);
          if (effective > 1e-15) {
            auto next   = sub(c.friction, mul(direction, speed / effective));
            next        = sub(next, mul(v.normal, dot(next, v.normal)));
            next        = capped(next, std::sqrt(a.value.friction * b.value.friction) * v.impulse);
            auto change = sub(next, c.friction);
            c.friction  = next;
            point_impulse(&a, mul(change, -1), v.point);
            point_impulse(&b, change, v.point);
          }
        }
        // Coulomb rolling/spinning torques are limited by the same solved normal load.
        auto relative_w = sub(b.value.angular_velocity, a.value.angular_velocity);
        Vec3 u = unit(cross(v.normal, std::abs(v.normal[0]) < .8 ? Vec3{1, 0, 0} : Vec3{0, 1, 0})),
             w = cross(v.normal, u);
        for (auto axis : std::array<Vec3, 3>{v.normal, u, w}) {
          bool   spin        = dot(axis, v.normal) > .99;
          double coefficient = spin ? std::max(a.value.spinning_friction, b.value.spinning_friction)
                                    : std::max(a.value.rolling_friction, b.value.rolling_friction);
          double effective   = dot(axis, add(inertia(a, axis), inertia(b, axis)));
          if (coefficient <= 0 || effective < 1e-15)
            continue;
          double old_angular = dot(c.angular, axis),
                 next        = std::clamp(
                     old_angular -
                         dot(sub(b.value.angular_velocity, a.value.angular_velocity), axis) /
                             effective,
                     -coefficient * v.impulse,
                     coefficient * v.impulse);
          auto change = mul(axis, next - old_angular);
          c.angular   = add(c.angular, change);
          angular_impulse(&a, mul(change, -1));
          angular_impulse(&b, change);
        }
      }
    }
    if (options.block_joints)
      solve_block();
    if (options.contact_stabilization)
      return;
    for (auto& c : state.contacts) {
      auto& v = c.value;
      if (v.sensor || v.separation >= -options.contact_slop)
        continue;
      auto & a = get(v.first), &b = get(v.second);
      auto   ia = inverse_mass(a, v.normal), ib = inverse_mass(b, v.normal);
      double weight = dot(v.normal, add(ia, ib));
      if (weight <= 1e-15)
        continue;
      double correction =
          options.position_correction * (-v.separation - options.contact_slop) / weight;
      correction /= c.manifold_count;
      a.value.position = sub(a.value.position, mul(ia, correction));
      b.value.position = add(b.value.position, mul(ib, correction));
    }
  }
  /// @brief Cap post-constraint dynamic speeds before sweep integration and state validation.
  /// Joint/contact impulses can exceed the force-integration limits; apply the same
  /// rotation-invariant 500 m/s and 100 rad/s bounds after every solve as well.
  void limit_velocities() {
    for (auto& body : state.bodies)
      if (body.value.motion == Motion::Dynamic) {
        if (!options.clamp_velocities &&
            (length(body.value.linear_velocity) > 500 ||
             length(body.value.angular_velocity) > body.value.angular_speed_limit))
          throw std::runtime_error("native velocity bound exceeded");
        if (options.clamp_velocities) {
          body.value.linear_velocity = capped(body.value.linear_velocity, 500);
          body.value.angular_velocity =
              capped(body.value.angular_velocity, body.value.angular_speed_limit);
        }
      }
  }
  /// @brief Publish plain contact observations without exposing mutable solver impulses.
  void publish() {
    visible.clear();
    for (const auto& c : state.contacts)
      visible.push_back(c.value);
  }
  /// @brief Advance a fixed substep, conservatively stopping at possible impact times.
  void substep(double delta) {
    for (auto& j : state.joints) {
      auto&  c          = get(j.value.child);
      auto*  p          = parent(j.value.parent);
      auto   pq         = p ? p->value.rotation : Quaternion{1, 0, 0, 0};
      auto   axis       = rotate(pq, j.parent_axis);
      double f          = j.force;
      j.motor_impulse   = 0;
      j.damping_impulse = 0;
      j.dry_impulse     = 0;
      if (j.value.type == JointType::Hinge) {
        angular_impulse(&c, mul(axis, f * delta));
        angular_impulse(p, mul(axis, -f * delta));
      } else {
        apply_row(c, p, limit_row(j), f * delta);
      }
    }
    for (auto& b : state.bodies) {
      if (b.value.motion != Motion::Dynamic || b.sleeping)
        continue;
      b.value.linear_velocity = add(
          b.value.linear_velocity,
          mul(add(inverse_mass(b, b.force), mul(options.gravity, b.value.gravity_scale)), delta));
      b.value.angular_velocity = add(b.value.angular_velocity, mul(inertia(b, b.torque), delta));
      b.value.linear_velocity =
          mul(b.value.linear_velocity, std::max(0., 1 - b.value.linear_damping * delta));
      b.value.angular_velocity =
          mul(b.value.angular_velocity, std::max(0., 1 - b.value.angular_damping * delta));
      for (int k = 0; k < 3; ++k) {
        if (!b.value.allowed_translation[k])
          b.value.linear_velocity[k] = 0;
        if (!b.value.allowed_rotation[k])
          b.value.angular_velocity[k] = 0;
      }
      if (options.clamp_velocities)
        b.value.linear_velocity = capped(b.value.linear_velocity, 500);
      if (options.clamp_velocities)
        b.value.angular_velocity = capped(b.value.angular_velocity, b.value.angular_speed_limit);
    }
    warm_joints(delta);
    build_contacts(delta, true);
    solve_contacts(delta);
    limit_velocities();
    double remaining = delta;
    for (int event = 0; remaining > 1e-12; ++event) {
      if (event >= 128)
        throw std::runtime_error("continuous collision event capacity exhausted");
      double advance_delta = remaining;
      if (options.continuous) {
        broadphase(remaining);
        for (auto pair : pairs) {
          auto &a = state.bodies[pair[0]], &b = state.bodies[pair[1]];
          if (a.value.sensor || b.value.sensor ||
              ((a.value.motion != Motion::Dynamic) && (b.value.motion != Motion::Dynamic)))
            continue;
          advance_delta = std::min(
              advance_delta,
              impact_time(a.value, b.value, remaining, options.contact_slop, options.gyroscopic));
        }
      }
      for (auto& b : state.bodies)
        if (b.value.motion != Motion::Static && !b.sleeping) {
          integrate_pose(b.value, advance_delta, options.gyroscopic);
        }
      remaining -= advance_delta;
      build_contacts(delta, false);
      solve_contacts(delta);
      limit_velocities();
    }
    for (auto& b : state.bodies) {
      if (options.sleeping && b.value.motion == Motion::Dynamic) {
        if (length(b.value.linear_velocity) < .02 && length(b.value.angular_velocity) < .02) {
          b.quiet_time += delta;
          if (b.quiet_time > .5) {
            b.sleeping               = true;
            b.value.linear_velocity  = {};
            b.value.angular_velocity = {};
          }
        } else
          b.quiet_time = 0;
      }
      double qnorm = 0;
      for (double q : b.value.rotation) {
        if (!std::isfinite(q))
          throw std::runtime_error("nonfinite native rotation");
        qnorm += q * q;
      }
      if (std::abs(qnorm - 1) > 1e-6)
        throw std::runtime_error("native rotation escaped unit sphere");
      for (double x : b.value.position)
        if (!std::isfinite(x) || std::abs(x) > 100000)
          throw std::runtime_error("native pose escaped bounds");
      for (double x : b.value.angular_velocity)
        if (!std::isfinite(x) || std::abs(x) > b.value.angular_speed_limit)
          throw std::runtime_error("native angular velocity escaped bounds");
      for (double x : b.value.linear_velocity)
        if (!std::isfinite(x) || std::abs(x) > 500)
          throw std::runtime_error("native velocity escaped bounds");
    }
  }
};

void WorldOptions::validate() const {
  for (double x : gravity)
    if (!std::isfinite(x))
      throw std::invalid_argument("nonfinite gravity");
  if (!body_capacity || !pair_capacity || !contact_capacity || !solver_iterations || !substeps ||
      substeps > 1024 || !std::isfinite(contact_slop) || contact_slop < 0 ||
      !std::isfinite(position_correction) || position_correction <= 0 || position_correction > 1)
    throw std::invalid_argument("invalid physics controls");
}
World::World(WorldOptions options, compute::Context context) {
  options.validate();
  if (context.device().backend != compute::Backend::CPU)
    throw std::runtime_error("requested native physics backend is not implemented");
  impl_ = std::make_unique<Impl>(options, std::move(context));
}
World::~World() = default;
BodyId World::add_body(const Body& body) {
  body.validate();
  auto& s = impl_->state;
  if (s.bodies.size() == impl_->options.body_capacity)
    throw std::length_error("physics body capacity exhausted");
  if (impl_->next_id == 0 ||
      impl_->next_id > std::uint64_t(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("physics body identifiers exhausted");
  auto id = impl_->next_id;
  impl_->indices.emplace(id, s.bodies.size());
  try {
    s.bodies.push_back({id, body, inverse_inertia(body)});
  } catch (...) {
    impl_->indices.erase(id);
    throw;
  }
  ++impl_->next_id;
  ++s.topology;
  impl_->invalidate_contacts();
  return id;
}
void World::remove_body(BodyId id) {
  impl_->get(id);
  for (const auto& j : impl_->state.joints)
    if (j.value.child == id || j.value.parent == id)
      throw std::invalid_argument("remove attached joints before a body");
  auto  index                      = impl_->indices.at(id);
  auto& bodies                     = impl_->state.bodies;
  bodies[index]                    = bodies.back();
  impl_->indices[bodies[index].id] = index;
  bodies.pop_back();
  impl_->indices.erase(id);
  ++impl_->state.topology;
  impl_->invalidate_contacts();
}
Body World::body(BodyId id) const {
  return impl_->get(id).value;
}
void World::set_body(BodyId id, const Body& value) {
  value.validate();
  auto& b = impl_->get(id);
  for (const auto& j : impl_->state.joints)
    if (j.value.child == id && value.motion != Motion::Dynamic)
      throw std::invalid_argument("joint child must remain dynamic");
  // A sleeping payload has no gravity-generated normal impulse. Wake nearby
  // dynamic bodies when a tread starts/stops so friction can respond again.
  if (b.value.surface_velocity != value.surface_velocity) {
    const auto support = bounds(value);
    for (auto& other : impl_->state.bodies)
      if (other.value.motion == Motion::Dynamic &&
          overlap(support, bounds(other.value), 2 * impl_->options.contact_slop + 1e-6)) {
        other.sleeping   = false;
        other.quiet_time = 0;
      }
  }
  auto old             = b.value;
  old.position         = value.position;
  old.rotation         = value.rotation;
  old.linear_velocity  = value.linear_velocity;
  old.angular_velocity = value.angular_velocity;
  old.surface_velocity = value.surface_velocity;
  bool config =
      old.motion != value.motion || old.shape != value.shape ||
      old.dimensions != value.dimensions || old.mass != value.mass ||
      old.inertia_tensor != value.inertia_tensor || old.center_of_mass != value.center_of_mass ||
      old.colliders != value.colliders || old.vertices != value.vertices ||
      old.collision_group != value.collision_group || old.collision_mask != value.collision_mask ||
      old.rolling_friction != value.rolling_friction ||
      old.spinning_friction != value.spinning_friction ||
      old.angular_speed_limit != value.angular_speed_limit || old.friction != value.friction ||
      old.restitution != value.restitution || old.gravity_scale != value.gravity_scale ||
      old.linear_damping != value.linear_damping || old.angular_damping != value.angular_damping ||
      old.sensor != value.sensor || old.allowed_translation != value.allowed_translation ||
      old.allowed_rotation != value.allowed_rotation;
  b.value           = value;
  b.inverse_inertia = inverse_inertia(value);
  b.sleeping        = false;
  b.quiet_time      = 0;
  if (config)
    ++impl_->state.topology;
  impl_->invalidate_contacts();
}
std::vector<BodyId> World::bodies() const {
  std::vector<BodyId> ids;
  ids.reserve(body_count());
  for (const auto& b : impl_->state.bodies)
    ids.push_back(b.id);
  std::sort(ids.begin(), ids.end());
  return ids;
}
std::size_t World::body_count() const noexcept {
  return impl_->state.bodies.size();
}
void World::apply_impulse(BodyId id, const Vec3& impulse) {
  check_force(impulse);
  auto& b = impl_->get(id);
  if (b.value.motion != Motion::Dynamic)
    throw std::invalid_argument("impulse needs a dynamic body");
  auto next = add(b.value.linear_velocity, inverse_mass(b, impulse));
  if (length(next) > 500)
    throw std::invalid_argument("impulse exceeds body velocity limit");
  b.value.linear_velocity = next;
  b.sleeping              = false;
  b.quiet_time            = 0;
}
void World::apply_force(BodyId id, const Vec3& force) {
  check_force(force);
  auto& b = impl_->get(id);
  if (b.value.motion != Motion::Dynamic)
    throw std::invalid_argument("force needs a dynamic body");
  auto next = add(b.force, force);
  check_force(next);
  b.force      = next;
  b.sleeping   = false;
  b.quiet_time = 0;
}
void World::add_joint(const Joint& joint) {
  joint.validate();
  auto& c = impl_->get(joint.child);
  if (c.value.motion != Motion::Dynamic)
    throw std::invalid_argument("joint child must be dynamic");
  auto* p = impl_->parent(joint.parent);
  for (const auto& j : impl_->state.joints)
    if (j.value.child == joint.child)
      throw std::invalid_argument("duplicate child joint");
  BodyId ancestor = joint.parent;
  while (ancestor) {
    if (ancestor == joint.child)
      throw std::invalid_argument("joint parent cycle");
    auto it =
        std::find_if(impl_->state.joints.begin(), impl_->state.joints.end(), [&](const auto& j) {
          return j.value.child == ancestor;
        });
    ancestor = it == impl_->state.joints.end() ? 0 : it->value.parent;
  }
  auto pivot = add(c.value.position, rotate(c.value.rotation, joint.anchor)),
       axis  = unit(rotate(c.value.rotation, joint.axis));
  auto pq    = p ? p->value.rotation : Quaternion{1, 0, 0, 0};
  impl_->state.joints.push_back(
      {joint,
       joint.anchor,
       p ? rotate(conjugate(pq), sub(pivot, p->value.position)) : pivot,
       rotate(conjugate(pq), axis),
       unit(joint.axis),
       product(conjugate(pq), c.value.rotation),
       0});
  ++impl_->state.topology;
  impl_->invalidate_contacts();
}
void World::remove_joint(BodyId child) {
  impl_->joint(child);
  for (const auto& g : impl_->state.gears)
    if (g.master == child || g.slave == child)
      throw std::invalid_argument("remove gears before their joints");
  auto& js = impl_->state.joints;
  js.erase(
      std::remove_if(js.begin(), js.end(), [&](const auto& j) { return j.value.child == child; }),
      js.end());
  ++impl_->state.topology;
  impl_->invalidate_contacts();
}
std::size_t World::joint_count() const noexcept {
  return impl_->state.joints.size();
}
JointState World::joint_state(BodyId child) const {
  return impl_->coordinate(impl_->joint(child));
}
void World::apply_joint_force(BodyId child, double force) {
  if (!std::isfinite(force) || std::abs(force) > 100000)
    throw std::invalid_argument("invalid generalized force");
  auto&  j    = impl_->joint(child);
  double next = j.force + force;
  if (!std::isfinite(next) || std::abs(next) > 100000)
    throw std::invalid_argument("accumulated joint force limit");
  j.force      = next;
  auto& c      = impl_->get(child);
  c.sleeping   = false;
  c.quiet_time = 0;
  if (auto* p = impl_->parent(j.value.parent)) {
    p->sleeping   = false;
    p->quiet_time = 0;
  }
}
void World::set_joint_motor(
    BodyId child,
    double velocity,
    double gain,
    double effort,
    double feedforward) {
  if (!std::isfinite(velocity) || std::abs(velocity) > 100000 || !std::isfinite(gain) || gain < 0 ||
      gain > 1e9 || !std::isfinite(effort) || effort < 0 || effort > 100000 ||
      !std::isfinite(feedforward) || std::abs(feedforward) > 100000)
    throw std::invalid_argument("invalid native servo parameters");
  auto& j = impl_->joint(child);
  if (j.value.type == JointType::Fixed || effort > j.value.effort_limit ||
      std::abs(velocity) > j.value.velocity_limit + 1e-12)
    throw std::invalid_argument("motor exceeds authored joint limits");
  j.motor_target             = velocity;
  j.motor_gain               = gain;
  j.motor_limit              = effort;
  j.motor_feedforward        = feedforward;
  impl_->get(child).sleeping = false;
  if (auto* p = impl_->parent(j.value.parent))
    p->sleeping = false;
}
double World::joint_motor_effort(BodyId child) const {
  auto& j = impl_->joint(child);
  return j.previous_delta > 0 ? j.motor_impulse / j.previous_delta : 0;
}
void World::add_gear(BodyId master, BodyId slave, double ratio, double offset) {
  auto& a = impl_->joint(master);
  auto& b = impl_->joint(slave);
  if (master == slave || !std::isfinite(ratio) || ratio == 0 || std::abs(ratio) > 1000 ||
      !std::isfinite(offset) || std::abs(offset) > 1000 || a.value.type == JointType::Fixed ||
      b.value.type == JointType::Fixed)
    throw std::invalid_argument("invalid native gear");
  for (auto g : impl_->state.gears)
    if (g.slave == slave)
      throw std::invalid_argument("duplicate native gear slave");
  BodyId node = master;
  for (std::size_t i = 0; i <= impl_->state.gears.size(); ++i) {
    if (node == slave)
      throw std::invalid_argument("native gear cycle");
    auto g = std::find_if(impl_->state.gears.begin(), impl_->state.gears.end(), [&](auto g) {
      return g.slave == node;
    });
    if (g == impl_->state.gears.end())
      break;
    node = g->master;
  }
  impl_->state.gears.push_back({master, slave, ratio, offset});
  ++impl_->state.topology;
  impl_->invalidate_contacts();
}
void World::remove_gear(BodyId slave) {
  auto& g = impl_->state.gears;
  auto  q = std::find_if(g.begin(), g.end(), [&](auto x) { return x.slave == slave; });
  if (q == g.end())
    throw std::out_of_range("unknown native gear");
  g.erase(q);
  ++impl_->state.topology;
  impl_->invalidate_contacts();
}
double World::joint_gravity_compensation(BodyId child) const {
  auto& j = impl_->joint(child);
  if (j.value.type == JointType::Fixed)
    return 0;
  auto   row   = impl_->limit_row(j);
  double total = 0;
  for (auto& b : impl_->state.bodies) {
    BodyId node   = b.id;
    bool   inside = false;
    while (node) {
      if (node == child) {
        inside = true;
        break;
      }
      auto q = std::find_if(impl_->state.joints.begin(), impl_->state.joints.end(), [&](auto& q) {
        return q.value.child == node;
      });
      node   = q == impl_->state.joints.end() ? 0 : q->value.parent;
    }
    if (!inside || b.value.motion != Motion::Dynamic)
      continue;
    auto f = mul(impl_->options.gravity, -b.value.mass * b.value.gravity_scale);
    total += j.value.type == JointType::Hinge
                 ? dot(row.axis, cross(sub(center(b), row.child_point), f))
                 : dot(row.axis, f);
  }
  return total;
}
void World::set_joint_state(BodyId child, double position, double velocity) {
  auto& j = impl_->joint(child);
  if (!std::isfinite(position) || !std::isfinite(velocity) ||
      (j.value.limited && (position < j.value.limit_min || position > j.value.limit_max)))
    throw std::invalid_argument("invalid joint target");
  auto old = impl_->coordinate(j);

  auto* p         = impl_->parent(j.value.parent);
  auto  pq        = p ? p->value.rotation : Quaternion{1, 0, 0, 0};
  auto  axis      = rotate(pq, j.parent_axis),
       pivot      = p ? add(p->value.position, rotate(pq, j.parent_anchor)) : j.parent_anchor;
  auto   rotation = axis_angle(axis, position - old.position);
  double delta = position - old.position, rate = velocity - old.velocity;
  auto&  replacements = impl_->replacements;
  replacements.clear();
  for (std::size_t i = 0; i < impl_->state.bodies.size(); ++i) {
    auto&  b          = impl_->state.bodies[i];
    BodyId node       = b.id;
    bool   descendant = false;
    while (node) {
      if (node == child) {
        descendant = true;
        break;
      }
      auto it =
          std::find_if(impl_->state.joints.begin(), impl_->state.joints.end(), [&](const auto& x) {
            return x.value.child == node;
          });
      node = it == impl_->state.joints.end() ? 0 : it->value.parent;
    }
    if (!descendant)
      continue;
    auto       next            = b.value;
    const auto parent_rate     = p ? p->value.angular_velocity : Vec3{};
    const auto parent_velocity = detail::velocity(p, center(b));
    if (j.value.type == JointType::Slider) {
      next.position = add(next.position, mul(axis, delta));
      next.linear_velocity =
          add(add(next.linear_velocity, cross(parent_rate, mul(axis, delta))), mul(axis, rate));
    } else {
      next.position = add(pivot, rotate(rotation, sub(next.position, pivot)));
      next.rotation = normalized(product(rotation, next.rotation));
      next.linear_velocity =
          add(detail::velocity(p, center(next)),
              rotate(rotation, sub(next.linear_velocity, parent_velocity)));
      next.angular_velocity =
          add(parent_rate, rotate(rotation, sub(next.angular_velocity, parent_rate)));
      next.linear_velocity =
          add(next.linear_velocity, cross(mul(axis, rate), sub(center(next), pivot)));
      next.angular_velocity = add(next.angular_velocity, mul(axis, rate));
    }
    next.validate();
    replacements.emplace_back(i, next);
  }
  j.armature_rate = velocity;
  for (const auto& r : replacements) {
    auto& b      = impl_->state.bodies[r.first];
    b.value      = r.second;
    b.sleeping   = false;
    b.quiet_time = 0;
  }
  impl_->invalidate_contacts();
}
void World::step(double delta) {
  if (!std::isfinite(delta) || delta <= 0 || delta > 1. / 30)
    throw std::invalid_argument("physics delta must be in (0,1/30]");
  auto& s       = impl_->state;
  impl_->backup = s;
  try {
    for (std::size_t i = 0; i < impl_->options.substeps; ++i)
      impl_->substep(delta / impl_->options.substeps);
    for (auto& b : s.bodies) {
      b.force  = {};
      b.torque = {};
    }
    for (auto& j : s.joints)
      j.force = 0;
    impl_->publish();
  } catch (...) {
    s                  = impl_->backup;
    impl_->block_valid = false;
    impl_->publish();
    throw;
  }
}
bool World::touching(BodyId first, BodyId second, double tolerance) const {
  if (first == second || !std::isfinite(tolerance) || tolerance < 0)
    throw std::invalid_argument("invalid proximity query");
  return distance(impl_->get(first).value, impl_->get(second).value).value <= tolerance;
}
bool World::grounded(BodyId id, double tolerance) const {
  if (!std::isfinite(tolerance) || tolerance < 0)
    throw std::invalid_argument("invalid ground probe distance");
  const auto& body    = impl_->get(id).value;
  double      maximum = body.position[1] - bounds(body).lo[1] + tolerance;
  for (const auto& other : impl_->state.bodies) {
    if (other.id == id || other.value.sensor)
      continue;
    double time;
    Vec3   normal;
    if (ray_hit(other.value, body.position, {0, -1, 0}, maximum, time, normal) && normal[1] > .6)
      return true;
  }
  return false;
}
const std::vector<Contact>& World::contacts() const noexcept {
  return impl_->visible;
}
PhysicsState World::save_state() const {
  PhysicsState result;
  result.state_ = std::make_shared<WorldState>(impl_->state);
  return result;
}
void World::validate_state(const PhysicsState& state) const {
  if (!state.state_ || state.state_->owner != impl_->state.owner ||
      state.state_->topology != impl_->state.topology)
    throw std::invalid_argument("physics snapshot owner/topology mismatch");
}
void World::restore_state(const PhysicsState& state) {
  validate_state(state);
  impl_->state       = *state.state_;
  impl_->block_valid = false;
  impl_->publish();
}
void World::rewind_step() noexcept {
  impl_->state       = impl_->backup;
  impl_->block_valid = false;
  impl_->publish();
}
std::unique_ptr<World> World::fork() const {
  auto       result          = std::make_unique<World>(impl_->options, impl_->context);
  const auto owner           = result->impl_->state.owner;
  result->impl_->state       = impl_->state;
  result->impl_->state.owner = owner;
  result->impl_->indices     = impl_->indices;
  result->impl_->next_id     = impl_->next_id;
  result->impl_->publish();
  return result;
}
} // namespace datamunge::sim
