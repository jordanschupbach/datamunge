/// @file native_stream.cpp
/// @brief Native finite-effort URDF robot, physical jaw contacts and free compound fruit.
#include "native_stream.hpp"

#include <datamunge/sim/articulation.hpp>
#include <map>
#include <set>

#include "collision.hpp"
#include "rigid_math.hpp"
namespace datamunge::sim::detail {
/// @brief Coordinate conversion and explicit nominal workcell settings.
namespace {
/// @brief Swap public Y-up linear axes into URDF Z-up (involutive).
/// @param v Linear vector in either convention.
/// @return Linear vector in the other convention.
Vec3 swap_axes(Vec3 v) {
  return {v[0], v[2], v[1]};
}
/// @brief Conjugate a rotation by the public/URDF reflection (involutive).
/// @param q Unit WXYZ rotation in either convention.
/// @return Unit rotation in the other convention.
Quaternion swap_rotation(Quaternion q) {
  return {q[0], -q[1], -q[3], -q[2]};
}
/// @brief Orient the local capsule Y axis onto a nonzero direction.
/// @param d Nonzero segment direction.
/// @return Unit WXYZ capsule orientation.
Quaternion along_y(Vec3 d) {
  d = unit(d);
  if (d[1] < -.999999)
    return {0, 1, 0, 0};
  return normalized({1 + d[1], d[2], 0, -d[0]});
}
/// @brief Configure SI constraints with explicit discrete contact and conservative timestep.
/// @return Native coupled-solver controls; no external engine.
WorldOptions native_options() {
  WorldOptions o;
  o.gravity               = {0, 0, -9.81};
  o.body_capacity         = 128;
  o.pair_capacity         = 2048;
  o.contact_capacity      = 4096;
  o.substeps              = 1;
  o.solver_iterations     = 12;
  o.block_joints          = true;
  o.contact_slop          = .0002;
  o.position_correction   = .15;
  o.continuous            = false;
  o.persistent_contacts   = true;
  o.contact_stabilization = true;
  o.gyroscopic            = true;
  o.sleeping              = false;
  o.clamp_velocities      = false;
  return o;
}
/// @brief Configure explicit nominal drive/finger parameters missing from the public URDF.
/// @return Nominal fallback inertias, drive inertia and finger collision slabs.
UrdfPhysicsOptions import_options() {
  UrdfPhysicsOptions o;
  o.missing_mass    = .04;
  o.hinge_armature  = .002;
  o.slider_armature = 1;
  o.joint_damping   = .1;
  o.slider_damping  = .5;
  for (auto name : {"finger_left_link", "finger_right_link"}) {
    o.mesh_x_slices[name].push_back({-10, -.055});
    for (int i = 0; i < 5; ++i)
      o.mesh_x_slices[name].push_back({-.055 + .011 * i, -.055 + .011 * (i + 1)});
  }
  return o;
}
} // namespace
/// @brief Owned independent native physical workcell and measured diagnostics.
struct NativeStreamDynamics::Impl {
  UrdfModel                model;      ///< Immutable imported physical tree shared across resets.
  std::unique_ptr<World>   world;      ///< Independent native solver, never MuJoCo.
  std::vector<Body>        fixtures;   ///< Converted stationary workcell.
  Articulation             robot;      ///< Live robot body/joint identifiers.
  std::array<BodyId, 8>    joints{};   ///< Six hinges followed by opposing sliders.
  std::set<BodyId>         floors;     ///< Actual bin support bodies.
  std::map<BodyId, BodyId> fruit;      ///< Public payload IDs to native dynamic bodies.
  std::array<double, 8>    previous{}; ///< Previous 240 Hz reference.
  double                   dt;         ///< Internal physics interval.
  std::size_t              capacity;   ///< Maximum active payloads.
  double                   ratio{0};   ///< Last tick actuator effort fraction.
  double                   peak{0};    ///< Peak bounded effort fraction since reset.
  double                   penetration{0};  ///< Maximum active contact penetration, metres.
  double                   joint_speed{0};  ///< Peak measured hinge rate, rad/s.
  double                   finger_speed{0}; ///< Peak measured slider rate, m/s.

  /// @brief Import immutable dynamics for later native resets.
  /// @param path Existing physical URDF path.
  /// @param d Internal timestep in seconds.
  /// @param n Active payload capacity.
  Impl(const std::string& path, double d, std::size_t n)
      : model(UrdfModel::load(path, import_options())), dt(d), capacity(n) {}
};
NativeStreamDynamics::NativeStreamDynamics(
    const std::string& path,
    const World&       source,
    double             dt,
    std::size_t        capacity)
    : impl_(std::make_unique<Impl>(path, dt, capacity)) {
  if (!std::isfinite(dt) || dt <= 0 || dt > 1. / 240 ||
      std::abs(std::round((1. / 240) / dt) * dt - 1. / 240) > 1e-10 || !capacity || capacity > 32)
    throw std::invalid_argument("invalid native workcell timestep/capacity");
  for (auto id : source.bodies()) {
    auto b = source.body(id);
    if (b.motion != Motion::Static || b.sensor)
      continue;
    b.position          = swap_axes(b.position);
    b.rotation          = swap_rotation(b.rotation);
    b.dimensions        = swap_axes(b.dimensions);
    b.center_of_mass    = {};
    b.inertia_tensor    = {};
    b.collision_group   = 1;
    b.collision_mask    = 6;
    b.restitution       = 0;
    b.spinning_friction = .002;
    b.rolling_friction  = .001;
    // Boxes retain identity in the source workcell; swapped dimensions preserve their geometry.
    if (std::abs(b.position[2] - .68) < .02 && b.dimensions[0] > 1)
      b.surface_velocity = {.22, 0, 0};
    else
      b.surface_velocity = {};
    impl_->fixtures.push_back(b);
  }
}
NativeStreamDynamics::~NativeStreamDynamics() = default;
void NativeStreamDynamics::reset(const std::array<double, 6>& q, double finger) {
  auto& p     = *impl_;
  auto  world = std::make_unique<World>(native_options());
  p.floors.clear();
  p.fruit.clear();
  for (auto b : p.fixtures) {
    auto id = world->add_body(b);
    if (std::abs(b.position[2] - .43) < .001)
      p.floors.insert(id);
  }
  p.robot = p.model.instantiate(*world, {.62, 0, .72}, {0, 0, 0, 1});
  for (auto [name, id] : p.robot.links) {
    auto b           = world->body(id);
    b.collision_mask = 7;
    if (b.collision_group)
      b.collision_group = 2;
    b.friction          = .8;
    b.restitution       = 0;
    b.spinning_friction = .002;
    b.rolling_friction  = .001;
    if (name == "finger_left_link" || name == "finger_right_link")
      b.center_of_mass = {-.025, 0, 0};
    world->set_body(id, b);
  }
  for (int i = 0; i < 8; ++i) {
    p.joints[i] = p.robot.joints.at(
        i < 6    ? "joint" + std::to_string(i + 1)
        : i == 6 ? "finger_left"
                 : "finger_right");
    p.previous[i] = i < 6 ? q[i] : i == 6 ? finger : -finger;
    world->set_joint_state(p.joints[i], p.previous[i], 0);
  }
  p.world = std::move(world);
  p.ratio = p.peak = p.penetration = p.joint_speed = p.finger_speed = 0;
}
void NativeStreamDynamics::spawn(BodyId id, const ConveyorObservation& s) {
  auto& p = *impl_;
  if (p.fruit.count(id) || p.fruit.size() >= p.capacity)
    throw std::runtime_error("native payload capacity exhausted/duplicate");
  Body b;
  b.motion              = Motion::Dynamic;
  b.position            = swap_axes(s.position);
  b.rotation            = swap_rotation(s.rotation);
  b.angular_velocity    = mul(swap_axes(s.angular_velocity), -1);
  b.linear_velocity     = swap_axes(s.velocity);
  b.angular_speed_limit = 10000;
  b.linear_damping = b.angular_damping = 0;
  b.restitution                        = 0;
  b.friction                           = .8;
  b.spinning_friction                  = .002;
  b.rolling_friction                   = .001;
  b.collision_group                    = 4;
  b.collision_mask                     = 7;
  int prototype                        = s.appearance.prototype;
  if (prototype < 0 || prototype >= 6)
    throw std::invalid_argument("invalid fruit prototype");
  const double girth[6]{1, .80, 1.10, 1.02, .9, .94}, lengths[6]{1, 1.15, .86, 1.02, 1.08, .84},
      curves[6]{.028, .018, .045, .05, .035, .055}, thickness[6]{1, .82, 1.16, .9, 1.04, 1.12};
  double scale = s.appearance.scale, r = .039 * scale;
  b.mass = (s.kind == FruitKind::Apple ? .16 : .12) * scale * scale * scale;
  if (s.kind == FruitKind::Apple) {
    b.shape          = Shape::Ellipsoid;
    b.dimensions     = {2 * r * girth[prototype], 2 * r * girth[prototype], 2 * r};
    b.inertia_tensor = body_tensor(b);
  } else {
    std::vector<Collider> parts;
    double                total = 0;
    std::array<double, 5> volumes{};
    for (int k = 0; k < 5; ++k) {
      double u = -1 + 2. * k / 5, v = -1 + 2. * (k + 1) / 5;
      Vec3   a{0, -curves[prototype] * u * u * scale, .076 * lengths[prototype] * u * scale},
          z{0, -curves[prototype] * v * v * scale, .076 * lengths[prototype] * v * scale};
      Collider c;
      c.shape    = Shape::Capsule;
      c.position = mul(add(a, z), .5);
      c.rotation = along_y(sub(z, a));
      double rr  = .023 * thickness[prototype] * scale *
                  (.12 + .88 * std::sqrt(std::sin(std::acos(-1.) * (k + .5) / 5)));
      c.dimensions = {2 * rr, length(sub(z, a)) + 2 * rr, 2 * rr};
      volumes[k]   = std::acos(-1.) * rr * rr * (c.dimensions[1] - 2 * rr + 4 * rr / 3);
      total += volumes[k];
      b.center_of_mass = add(b.center_of_mass, mul(c.position, volumes[k]));
      parts.push_back(c);
    }
    b.center_of_mass = mul(b.center_of_mass, 1 / total);
    for (int k = 0; k < 5; ++k) {
      Body part;
      part.shape      = Shape::Capsule;
      part.dimensions = parts[k].dimensions;
      part.mass       = b.mass * volumes[k] / total;
      auto I          = body_tensor(part);
      auto d          = sub(parts[k].position, b.center_of_mass);
      for (int j = 0; j < 3; ++j) {
        auto col = rotate(
            parts[k].rotation, apply_tensor(I, rotate(conjugate(parts[k].rotation), basis(j))));
        for (int i = 0; i < 3; ++i)
          b.inertia_tensor[3 * i + j] +=
              col[i] + part.mass * ((i == j ? dot(d, d) : 0) - d[i] * d[j]);
      }
    }
    b.colliders  = std::make_shared<const std::vector<Collider>>(std::move(parts));
    b.dimensions = {.1, .2, .1};
  }
  b.linear_velocity =
      add(b.linear_velocity, cross(b.angular_velocity, rotate(b.rotation, b.center_of_mass)));
  for (std::size_t pass = 0; pass <= p.fruit.size(); ++pass) {
    bool raised = false;
    for (auto [_, other] : p.fruit) {
      auto a = bounds(b), c = bounds(p.world->body(other));
      if (a.lo[0] < c.hi[0] + .002 && a.hi[0] > c.lo[0] - .002 && a.lo[1] < c.hi[1] + .002 &&
          a.hi[1] > c.lo[1] - .002 && a.lo[2] < c.hi[2] + .002 && a.hi[2] > c.lo[2] - .002) {
        b.position[2] += c.hi[2] + .002 - a.lo[2];
        raised = true;
      }
    }
    if (!raised)
      break;
  }
  p.fruit.emplace(id, p.world->add_body(b));
}
void NativeStreamDynamics::remove(BodyId id) {
  auto& p = *impl_;
  p.world->remove_body(p.fruit.at(id));
  p.fruit.erase(id);
}
void NativeStreamDynamics::step(const std::array<double, 6>& target, double finger) {
  auto& p   = *impl_;
  p.ratio   = 0;
  int count = int(std::lround((1. / 240) / p.dt));
  for (int k = 0; k < count; ++k) {
    for (int i = 0; i < 8; ++i) {
      double goal = i < 6    ? target[i]
                    : i == 6 ? finger
                             : -finger,
             vmax = i < 6 ? 1.5 : .08, ff = std::clamp((goal - p.previous[i]) * 240., -vmax, vmax),
             kp       = i < 3   ? 9
                        : i < 6 ? 12.5
                                : 10,
             effort   = i < 3   ? 27
                        : i < 6 ? 7
                                : 8,
             gain     = i < 3   ? 20
                        : i < 6 ? 4
                                : 80;
      auto   state    = p.world->joint_state(p.joints[i]);
      double velocity = std::clamp(kp * (goal - state.position) + ff, -vmax, vmax);
      p.world->set_joint_motor(
          p.joints[i],
          velocity,
          gain,
          effort,
          i < 6 ? p.world->joint_gravity_compensation(p.joints[i]) : 0);
    }
    p.world->step(p.dt);
    for (int i = 0; i < 8; ++i) {
      double effort = i < 3 ? 27 : i < 6 ? 7 : 8;
      p.ratio     = std::max(p.ratio, std::abs(p.world->joint_motor_effort(p.joints[i])) / effort);
      auto& speed = i < 6 ? p.joint_speed : p.finger_speed;
      speed       = std::max(speed, std::abs(p.world->joint_state(p.joints[i]).velocity));
    }
    for (auto& c : p.world->contacts())
      p.penetration = std::max(p.penetration, -c.separation);
  }
  for (int i = 0; i < 8; ++i)
    p.previous[i] = i < 6 ? target[i] : i == 6 ? finger : -finger;
  p.peak = std::max(p.peak, p.ratio);
}
void NativeStreamDynamics::observe(PickObservation& a) const {
  auto& p           = *impl_;
  a.measured_joints = true;
  for (int i = 0; i < 6; ++i)
    a.joints[i] = p.world->joint_state(p.joints[i]).position;
  a.finger = .5 * (p.world->joint_state(p.joints[6]).position -
                   p.world->joint_state(p.joints[7]).position);
}
cad::Vec3d NativeStreamDynamics::tool_point() const {
  auto b = impl_->world->body(impl_->robot.links.at("end_link"));
  auto p = add(b.position, rotate(b.rotation, {-.02, 0, 0}));
  return {p[0], p[1], p[2]};
}
Body NativeStreamDynamics::body(BodyId id, Body out) const {
  auto b       = impl_->world->body(impl_->fruit.at(id));
  out.position = swap_axes(b.position);
  out.rotation = swap_rotation(b.rotation);
  auto v = sub(b.linear_velocity, cross(b.angular_velocity, rotate(b.rotation, b.center_of_mass)));
  out.linear_velocity     = swap_axes(v);
  out.angular_velocity    = mul(swap_axes(b.angular_velocity), -1);
  out.motion              = Motion::Dynamic;
  out.allowed_rotation    = {true, true, true};
  out.mass                = b.mass;
  out.angular_speed_limit = b.angular_speed_limit;
  out.friction            = b.friction;
  return out;
}
double NativeStreamDynamics::grip_force(BodyId id) const {
  auto&  p       = *impl_;
  auto   payload = p.fruit.at(id);
  double force[2]{};
  for (auto& c : p.world->contacts())
    for (int i = 0; i < 2; ++i)
      if ((c.first == payload && c.second == p.joints[6 + i]) ||
          (c.second == payload && c.first == p.joints[6 + i]))
        force[i] += c.impulse / p.dt;
  return std::min(force[0], force[1]);
}
bool NativeStreamDynamics::gripped(BodyId id) const {
  return grip_force(id) > .05;
}
bool NativeStreamDynamics::bin_supported(BodyId id) const {
  auto& p       = *impl_;
  auto  payload = p.fruit.at(id);
  for (auto& c : p.world->contacts())
    if (((c.first == payload && p.floors.count(c.second) && -c.normal[2] > .5) ||
         (c.second == payload && p.floors.count(c.first) && c.normal[2] > .5)) &&
        c.impulse / p.dt > .05)
      return true;
  return false;
}
double NativeStreamDynamics::effort_ratio() const {
  return impl_->ratio;
}
std::array<double, 4> NativeStreamDynamics::diagnostics() const {
  return {impl_->penetration, impl_->peak, impl_->joint_speed, impl_->finger_speed};
}
} // namespace datamunge::sim::detail
