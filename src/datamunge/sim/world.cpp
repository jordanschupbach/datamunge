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
  BodyId id;              ///< Stable world-scoped identifier.
  Body   value;           ///< Current pose, velocities and physical parameters.
  Vec3   inverse_inertia; ///< Principal inverse inertias in local coordinates.
  Vec3   force{};         ///< World force accumulated until a successful outer step.
  Vec3   torque{};        ///< World torque accumulated until a successful outer step.
  double quiet_time{0};   ///< Time below the sleep threshold.
  bool   sleeping{false}; ///< Whether integration is currently suspended.
};
/// @brief A compiled joint's authored frames and persistent actuator state.
struct RuntimeJoint {
  Joint                 value;             ///< Validated public settings.
  Vec3                  child_anchor;      ///< Local child pivot.
  Vec3                  parent_anchor;     ///< Parent-local pivot, or world pivot.
  Vec3                  parent_axis;       ///< Parent-local generalized axis, or world direction.
  Vec3                  child_axis;        ///< Child-local generalized axis.
  Quaternion            reference;         ///< Authored child-relative-to-parent orientation.
  double                force{0};          ///< Accumulated generalized actuator force.
  std::array<double, 5> impulses{};        ///< Cached bilateral constraint row impulses.
  double                limit_impulse{0};  ///< Signed accumulated unilateral limit impulse.
  double                previous_delta{0}; ///< Last warm-start substep duration.
};
/// @brief One solver contact including tangential warm-start impulse.
struct SolverContact {
  Contact     value;             ///< Public contact observation and normal impulse.
  Vec3        friction{};        ///< Tangential impulse in world coordinates.
  double      bounce{0};         ///< Restitution velocity selected at contact creation.
  std::size_t manifold_count{1}; ///< Points sharing this body's contact pair.
};
/// @brief Complete rewindable native state, tied to one world and topology generation.
struct WorldState {
  std::uint64_t              owner{0};    ///< Identity of the originating world.
  std::uint64_t              topology{0}; ///< Topology/configuration revision.
  std::vector<RuntimeBody>   bodies;      ///< Poses, velocities, forces and sleep timers.
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
inline Vec3 inverse_inertia(const Body& b) {
  if (b.motion != Motion::Dynamic)
    return {};
  double m = b.mass, r = .5 * b.dimensions[0];
  Vec3   inertia;
  if (b.shape == Shape::Box) {
    for (int i = 0; i < 3; ++i)
      inertia[i] = m *
                   (b.dimensions[(i + 1) % 3] * b.dimensions[(i + 1) % 3] +
                    b.dimensions[(i + 2) % 3] * b.dimensions[(i + 2) % 3]) /
                   12;
  } else if (b.shape == Shape::Sphere)
    inertia = {.4 * m * r * r, .4 * m * r * r, .4 * m * r * r};
  else {
    double l = b.dimensions[1] - 2 * r, mc = m * l / (l + 4 * r / 3), ms = m - mc;
    double across = mc * (3 * r * r + l * l) / 12 + ms * (.4 * r * r + .375 * l * r + .25 * l * l);
    inertia       = {across, .5 * mc * r * r + .4 * ms * r * r, across};
  }
  for (auto& v : inertia)
    v = 1 / v;
  return inertia;
}
/// @brief Multiply by the projected inverse mass matrix.
inline Vec3 inverse_mass(const RuntimeBody& b, Vec3 v) {
  if (b.value.motion != Motion::Dynamic)
    return {};
  for (int i = 0; i < 3; ++i)
    v[i] = b.value.allowed_translation[i] ? v[i] / b.value.mass : 0;
  return v;
}
/// @brief Multiply by the symmetric world inverse inertia including rotational locks.
inline Vec3 inertia(const RuntimeBody& b, Vec3 v) {
  if (b.value.motion != Motion::Dynamic)
    return {};
  for (int i = 0; i < 3; ++i)
    if (!b.value.allowed_rotation[i])
      v[i] = 0;
  v = rotate(conjugate(b.value.rotation), v);
  for (int i = 0; i < 3; ++i)
    v[i] *= b.inverse_inertia[i];
  v = rotate(b.value.rotation, v);
  for (int i = 0; i < 3; ++i)
    if (!b.value.allowed_rotation[i])
      v[i] = 0;
  return v;
}
/// @brief Velocity at one world point, including angular motion.
inline Vec3 velocity(const RuntimeBody* b, const Vec3& point) {
  return b ? add(b->value.linear_velocity,
                 cross(b->value.angular_velocity, sub(point, b->value.position)))
           : Vec3{};
}
/// @brief Apply a point impulse to a dynamic body, waking it for nonzero impulse.
inline void point_impulse(RuntimeBody* b, const Vec3& impulse, const Vec3& point) {
  if (!b || b->value.motion != Motion::Dynamic)
    return;
  auto dv                   = inverse_mass(*b, impulse),
       dw                   = inertia(*b, cross(sub(point, b->value.position), impulse));
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
  auto arm = sub(point, b->value.position), r = cross(arm, axis);
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
inline Body predicted(Body body, double delta) {
  if (body.motion != Motion::Static) {
    body.position = add(body.position, mul(body.linear_velocity, delta));
    body.rotation = advance(body.rotation, body.angular_velocity, delta);
  }
  return body;
}
/// @brief Conservative advancement from disjoint convex shapes; never silently drops an unresolved
/// sweep.
inline double impact_time(const Body& a, const Body& b, double delta, double slop) {
  double speed = length(sub(a.linear_velocity, b.linear_velocity)) +
                 .5 * length(a.dimensions) * length(a.angular_velocity) +
                 .5 * length(b.dimensions) * length(b.angular_velocity);
  if (speed < 1e-10)
    return delta;
  double time    = 0;
  auto   initial = distance(a, b);
  if (initial.value <= slop)
    return delta;
  // Stop inside the manifold margin so a swept stop produces response contacts.
  const double target = .5 * slop;
  for (int i = 0; i < 128; ++i) {
    auto current = distance(predicted(a, time), predicted(b, time));
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

  /// @brief Reserve bounded working storage without per-step allocation.
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
           swept > 0 ? swept_bounds(state.bodies[i].value, swept) : bounds(state.bodies[i].value)});
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
        if (connected(a.id, b.id) || !overlap(proxies[i].box, proxies[j].box, options.contact_slop))
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
      auto &     a = state.bodies[pair[0]], &b = state.bodies[pair[1]];
      auto       manifold  = collide(a.value, b.value, options.contact_slop);
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
            {a.id, b.id, p.point, p.normal, p.separation, 0, a.value.sensor || b.value.sensor},
            {},
            0};
        c.manifold_count = manifold.count;
        double vn        = dot(sub(velocity(&b, p.point), velocity(&a, p.point)), p.normal);
        c.bounce         = vn < -1 ? -std::max(a.value.restitution, b.value.restitution) * vn : 0;
        if (warm && !c.value.sensor) {
          double               nearest = .02;
          const SolverContact* match   = nullptr;
          for (auto it = first_old;
               it != previous.end() && it->value.first == a.id && it->value.second == b.id;
               ++it) {
            const auto& old = *it;
            if (dot(old.value.normal, p.normal) > .95) {
              double d = length(sub(old.value.point, p.point));
              if (d < nearest) {
                nearest = d;
                match   = &old;
              }
            }
          }
          if (match) {
            double scale =
                state.previous_delta > 0 ? std::min(2., delta / state.previous_delta) : 1;
            c.value.impulse = match->value.impulse * scale;
            c.friction      = mul(match->friction, scale);
            auto impulse    = add(mul(p.normal, c.value.impulse), c.friction);
            point_impulse(&a, mul(impulse, -1), p.point);
            point_impulse(&b, impulse, p.point);
          }
        }
        state.contacts.push_back(c);
      }
    }
    state.previous_delta = delta;
  }
  /// @brief One transient joint velocity row; no pointers are retained in snapshots.
  struct JointRow {
    Vec3   axis;         ///< Unit world-space constraint direction.
    Vec3   child_point;  ///< Child anchor for point rows.
    Vec3   parent_point; ///< Parent anchor for point rows.
    double bias;         ///< Position-error velocity in m/s or rad/s.
    bool   angular;      ///< Pure angular rather than point constraint.
  };
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
  std::array<JointRow, 5> joint_rows(const RuntimeJoint& j, double delta) {
    auto&      c    = get(j.value.child);
    auto*      p    = parent(j.value.parent);
    const auto pq   = p ? p->value.rotation : Quaternion{1, 0, 0, 0};
    const auto cp   = add(c.value.position, rotate(c.value.rotation, j.child_anchor));
    const auto pp   = p ? add(p->value.position, rotate(pq, j.parent_anchor)) : j.parent_anchor;
    const auto axis = unit(rotate(pq, j.parent_axis));
    const auto u    = unit(cross(axis, std::abs(axis[0]) < .8 ? Vec3{1, 0, 0} : Vec3{0, 1, 0})),
               v = cross(axis, u), error = sub(cp, pp);
    std::array<JointRow, 5> rows{};
    if (j.value.type == JointType::Hinge) {
      for (int k = 0; k < 3; ++k)
        rows[k] = {basis(k), cp, pp, .2 * error[k] / delta, false};
      const auto er = cross(axis, rotate(c.value.rotation, j.child_axis));
      rows[3]       = {u, cp, pp, .2 * dot(er, u) / delta, true};
      rows[4]       = {v, cp, pp, .2 * dot(er, v) / delta, true};
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
      for (std::size_t k = 0; k < 5; ++k) {
        j.impulses[k] *= scale;
        apply_row(c, p, rows[k], j.impulses[k]);
      }
      j.limit_impulse *= scale;
      if (j.value.limited)
        apply_row(c, p, limit_row(j), j.limit_impulse);
      j.previous_delta = delta;
    }
  }
  /// @brief Evaluate the generalized coordinate against the original authored frames.
  JointState coordinate(const RuntimeJoint& j) {
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
  /// @brief Solve cached bilateral rows and clamped unilateral/speculative joint limits.
  void solve_joints(double delta) {
    for (auto& j : state.joints) {
      auto&      c    = get(j.value.child);
      auto*      p    = parent(j.value.parent);
      const auto rows = joint_rows(j, delta);
      for (std::size_t k = 0; k < 5; ++k) {
        const double mass = row_mass(c, p, rows[k]);
        if (mass < 1e-15)
          continue;
        const double change = -(row_rate(c, p, rows[k]) + rows[k].bias) / mass;
        j.impulses[k] += change;
        apply_row(c, p, rows[k], change);
      }
      if (j.value.limited) {
        const auto   q    = coordinate(j);
        auto         row  = limit_row(j);
        const double mass = row_mass(c, p, row);
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
        apply_row(c, p, row, j.limit_impulse - old);
      }
    }
  }
  /// @brief Iterative normal/friction solve, followed by bounded positional correction.
  void solve_contacts(double delta) {
    for (std::size_t iteration = 0; iteration < options.solver_iterations; ++iteration) {
      solve_joints(delta);
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
        v.impulse    = std::max(0., old + (c.bounce - rate) / mass);
        auto impulse = mul(v.normal, v.impulse - old);
        point_impulse(&a, mul(impulse, -1), v.point);
        point_impulse(&b, impulse, v.point);
        auto   relative = sub(velocity(&b, v.point), velocity(&a, v.point));
        auto   tangent  = sub(relative, mul(v.normal, dot(relative, v.normal)));
        double speed    = length(tangent);
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
      }
    }
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
        body.value.linear_velocity  = capped(body.value.linear_velocity, 500);
        body.value.angular_velocity = capped(body.value.angular_velocity, 100);
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
      auto&  c    = get(j.value.child);
      auto*  p    = parent(j.value.parent);
      auto   pq   = p ? p->value.rotation : Quaternion{1, 0, 0, 0};
      auto   axis = rotate(pq, j.parent_axis);
      double f    = j.force - j.value.damping * coordinate(j).velocity;
      if (j.value.type == JointType::Hinge) {
        angular_impulse(&c, mul(axis, f * delta));
        angular_impulse(p, mul(axis, -f * delta));
      } else {
        point_impulse(&c, mul(axis, f * delta), c.value.position);
        if (p)
          point_impulse(p, mul(axis, -f * delta), p->value.position);
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
      b.value.linear_velocity  = capped(b.value.linear_velocity, 500);
      b.value.angular_velocity = capped(b.value.angular_velocity, 100);
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
              advance_delta, impact_time(a.value, b.value, remaining, options.contact_slop));
        }
      }
      for (auto& b : state.bodies)
        if (b.value.motion != Motion::Static && !b.sleeping) {
          b.value.position = add(b.value.position, mul(b.value.linear_velocity, advance_delta));
          b.value.rotation = advance(b.value.rotation, b.value.angular_velocity, advance_delta);
        }
      remaining -= advance_delta;
      build_contacts(delta, false);
      solve_contacts(delta);
      limit_velocities();
    }
    for (auto& b : state.bodies) {
      if (b.value.motion == Motion::Dynamic) {
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
      b.value.validate();
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
  auto old             = b.value;
  old.position         = value.position;
  old.rotation         = value.rotation;
  old.linear_velocity  = value.linear_velocity;
  old.angular_velocity = value.angular_velocity;
  bool config          = old.motion != value.motion || old.shape != value.shape ||
                old.dimensions != value.dimensions || old.mass != value.mass ||
                old.friction != value.friction || old.restitution != value.restitution ||
                old.gravity_scale != value.gravity_scale ||
                old.linear_damping != value.linear_damping ||
                old.angular_damping != value.angular_damping || old.sensor != value.sensor ||
                old.allowed_translation != value.allowed_translation ||
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
void World::set_joint_state(BodyId child, double position, double velocity) {
  auto& j = impl_->joint(child);
  if (!std::isfinite(position) || !std::isfinite(velocity) ||
      (j.value.limited && (position < j.value.limit_min || position > j.value.limit_max)))
    throw std::invalid_argument("invalid joint target");
  auto  old       = impl_->coordinate(j);
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
    const auto parent_velocity = detail::velocity(p, b.value.position);
    if (j.value.type == JointType::Slider) {
      next.position = add(next.position, mul(axis, delta));
      next.linear_velocity =
          add(add(next.linear_velocity, cross(parent_rate, mul(axis, delta))), mul(axis, rate));
    } else {
      next.position = add(pivot, rotate(rotation, sub(next.position, pivot)));
      next.rotation = normalized(product(rotation, next.rotation));
      next.linear_velocity =
          add(detail::velocity(p, next.position),
              rotate(rotation, sub(next.linear_velocity, parent_velocity)));
      next.angular_velocity =
          add(parent_rate, rotate(rotation, sub(next.angular_velocity, parent_rate)));
      next.linear_velocity =
          add(next.linear_velocity, cross(mul(axis, rate), sub(next.position, pivot)));
      next.angular_velocity = add(next.angular_velocity, mul(axis, rate));
    }
    next.validate();
    replacements.emplace_back(i, next);
  }
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
    s = impl_->backup;
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
  impl_->state = *state.state_;
  impl_->publish();
}
void World::rewind_step() noexcept {
  impl_->state = impl_->backup;
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
