/// @file devarm_pick.cpp
/// @brief Real-URDF Cartesian servos, ideal stabilized grasps and native Q-learning.
#include <algorithm>
#include <cmath>
#include <datamunge/algorithms/tabular_rl.hpp>
#include <datamunge/sim/devarm_pick.hpp>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>

#include "math.hpp"
#include "visual_assets.hpp"

namespace datamunge::sim {
/// @brief Private lattice, inverse-kinematics and scene helpers.
namespace {
/// @brief Six independent revolute joint coordinates, in URDF order.
using Coordinates = std::array<double, 6>;
/// @brief Fixed B601-DM arm joint names; the finger is controlled separately.
const std::array<std::string, 6>
    joint_names{"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
/// @brief Reachable lattice dimensions: X=-2..2, Y=-16..16, Z=0..8.
constexpr int nx = 5;  ///< Longitudinal lattice size.
constexpr int ny = 33; ///< Lateral lattice size.
constexpr int nz = 9;  ///< Vertical lattice size.

/// @brief Number of distinct Cartesian servo targets, before adding grasp state.
constexpr std::size_t cells = nx * ny * nz;
/// @brief Position reference and exploration-independent IK seed.
const Coordinates home{0, -2.65, -2.90, 1.4, 0, 0};
/// @brief Requested goal's release point; used for rewards, never for selecting actions.
cad::Vec3d release_point(PickBin goal) {
  return {.10, goal == PickBin::Red ? -.42 : .42, .82};
}
/// @brief Convert Y-up physics to Z-up rendering/agent coordinates.
cad::Vec3d cad_point(const Vec3& p) {
  return {p[0], p[2], p[1]};
}
/// @brief Convert Z-up agent coordinates to Y-up physics.
Vec3 physics_point(const cad::Vec3d& p) {
  return {p.x, p.z, p.y};
}
/// @brief Set validated real URDF coordinates.
void set_joints(cad::Robot& robot, const Coordinates& q) {
  for (int i = 0; i < 6; ++i)
    robot.set_joint(joint_names[i], q[i]);
}
/// @brief Place authored or physically measured recorded coordinates without hiding compliance.
/// @param robot Owned visual model copy.
/// @param frame Recorded joints and aperture.
void recorded_joints(cad::Robot& robot,const PickObservation& frame){
 if(frame.measured_joints){std::vector<std::pair<std::string,double>> values;for(int i=0;i<6;++i)values.emplace_back(joint_names[i],frame.joints[i]);values.emplace_back("finger_left",frame.finger);robot.set_measured_joints(values);}
 else{set_joints(robot,frame.joints);robot.set_joint("finger_left",frame.finger);}
}
/// @brief Encode a valid Cartesian target as a contiguous table row.
std::size_t cell(int x, int y, int z) {
  return ((x + 2) * ny + (y + 16)) * nz + z;
}
/// @brief Lattice target in CAD workcell metres.
cad::Vec3d target(int x, int y, int z) {
  return {.10 + .03 * x, .03 * y, .76 + .03 * z};
}
/// @brief Grasp point in robot-local metres, and softly downward approach direction.
std::array<double, 6> feature(const cad::Robot& robot) {
  auto t = robot.link_transform("end_link");
  auto p = t.TransformPoint({-.02, 0, 0});
  auto d = t.TransformVector({1, 0, 0});
  return {p.x, p.y, p.z, .015 * d.x, .015 * d.y, .015 * d.z};
}
/// @brief Solve a regularized six-by-six normal system with partial pivoting.
Coordinates solve(double m[6][7]) {
  for (int a = 0; a < 6; ++a) {
    int pivot = a;
    for (int b = a + 1; b < 6; ++b)
      if (std::abs(m[b][a]) > std::abs(m[pivot][a]))
        pivot = b;
    for (int c = a; c < 7; ++c)
      std::swap(m[a][c], m[pivot][c]);
    double v = m[a][a];
    for (int c = a; c < 7; ++c)
      m[a][c] /= v;
    for (int b = 0; b < 6; ++b)
      if (b != a) {
        v = m[b][a];
        for (int c = a; c < 7; ++c)
          m[b][c] -= v * m[a][c];
      }
  }
  Coordinates q{};
  for (int i = 0; i < 6; ++i)
    q[i] = m[i][6];
  return q;
}
/// @brief Cache entry for one real-URDF IK target; unreachable targets remain blocked.
struct Pose {
  bool        known{false};     ///< Whether IK has been attempted.
  bool        reachable{false}; ///< Grasp-point position error is at most 12 mm.
  Coordinates q{};              ///< Clamped joint target.
};
/// @brief Add a real robot scene at the workcell's end mounting plate.
void append_arm(cad::Scene& scene, cad::Scene arm) {
  int mesh_base = int(scene.meshes.size()), texture_base = int(scene.textures.size());
  for (auto& m : arm.meshes) {
    for (std::size_t i = 0; i < m.positions.size(); i += 3) {
      m.positions[i] = .62f - m.positions[i];
      m.positions[i + 1] *= -1;
      m.positions[i + 2] += .72f;
    }
    for (std::size_t i = 0; i < m.normals.size(); i += 3) {
      m.normals[i] *= -1;
      m.normals[i + 1] *= -1;
    }
    scene.meshes.push_back(std::move(m));
  }
  for (auto& t : arm.textures)
    scene.textures.push_back(std::move(t));
  for (auto o : arm.objects) {
    o.mesh_index += mesh_base;
    o.parent = -1;
    for (int* slot :
         {&o.texture_index, &o.normal_map_index, &o.roughness_map_index, &o.metallic_map_index})
      if (*slot >= 0)
        *slot += texture_base;
    scene.AddObject(o);
  }
}
/// @brief Require a complete finite checkpoint before serializing it.
void validate_policy(const PickPolicy& p) {
  if (p.q.size() != DevArmPickEnvironment::state_count() && p.q.size() != cells * 6 + 1)
    throw std::invalid_argument("invalid pick policy states");
  for (const auto& row : p.q) {
    if (row.size() != 9)
      throw std::invalid_argument("invalid pick policy actions");
    for (double x : row)
      if (!std::isfinite(x))
        throw std::invalid_argument("nonfinite pick policy value");
  }
}
} // namespace

/// @brief Owned live robot, stable pose cache, native physics and task status.
struct DevArmPickEnvironment::Impl {
  cad::Robot          robot;    ///< Parsed real visual model and actual servo positions.
  Coordinates         lower{};  ///< Lower limits read from the real URDF.
  Coordinates         upper{};  ///< Upper limits read from the real URDF.
  ConveyorEnvironment conveyor; ///< Native fruit/belt/bin world.
  std::vector<Pose>   cache =
      std::vector<Pose>(cells); ///< Deterministic IK solutions, reused across resets.
  PickBin destination{
      PickBin::Red};     ///< Goal preserved across resets, independent of fruit identity.
  PickObservation state; ///< Current task observation, including actual joint coordinates.
  int             x{0};  ///< Last accepted longitudinal Cartesian command.
  int             y{0};  ///< Last accepted lateral Cartesian command.
  int             z{4};  ///< Last accepted vertical Cartesian command.
  FruitVariation  variation{FruitVariation::Basic}; ///< Domain and state abstraction.
  std::size_t     horizon;                          ///< Positive finite decision horizon.
  cad::Vec3d      grasp_offset; ///< Stabilized world-space fruit offset from the tool at capture.
  std::function<void(const PickObservation&)> recorder; ///< Optional video trajectory sink.
  std::uint64_t ticks{0}; ///< Physics ticks, used for exact 24 fps frame sampling.

  /// @brief Own the actual URDF and record its six bounded joint limits.
  Impl(const std::string& path, std::size_t limit, FruitVariation domain)
      : robot(cad::Robot::load_urdf(path)), conveyor(domain), variation(domain), horizon(limit) {
    if (!limit)
      throw std::invalid_argument("pick horizon must be positive");
    auto joints = robot.joints();
    for (int i = 0; i < 6; ++i) {
      auto it = std::find_if(
          joints.begin(), joints.end(), [&](const auto& j) { return j.name == joint_names[i]; });
      if (it == joints.end() || it->type != cad::JointType::Revolute)
        throw std::invalid_argument("expected DevArm revolute joints");
      lower[i] = it->lower;
      upper[i] = it->upper;
    }
    robot.link_transform("end_link");
    robot.set_joint("finger_left", .05);
  }
  /// @brief Convert the actual FK grasp point into world coordinates.
  cad::Vec3d tool() const {
    auto p = robot.link_transform("end_link").TransformPoint({-.02, 0, 0});
    return {.62 - p.x, -p.y, .72 + p.z};
  }
  /// @brief Refresh measured body/joint values and the policy's discrete row.
  void observe() {
    auto fruit             = conveyor.observation();
    state.kind             = fruit.kind;
    state.variation        = fruit.variation;
    state.appearance       = fruit.appearance;
    state.fruit_rotation   = fruit.rotation;
    state.fruit_dimensions = fruit.dimensions;
    state.feed_stage       = fruit.stage;
    state.belt_travel      = fruit.belt_travel;
    state.destination      = destination;
    state.fruit            = cad_point(fruit.position);
    state.tool             = tool();
    for (int i = 0; i < 6; ++i)
      state.joints[i] = robot.joint_position(joint_names[i]);
    state.finger        = robot.joint_position("finger_left");
    const auto terminal = variation == FruitVariation::Diverse ? cells * 6 : cells * 2;
    int        bucket   = std::clamp(int(std::lround(state.fruit.y / .03)), -2, 2) + 2;
    state.state         = state.terminated || state.truncated
                              ? terminal
                              : cell(x, y, z) + (variation == FruitVariation::Basic
                                                     ? (state.grasped ? cells : 0)
                                                     : (state.grasped ? cells * 5 : cells * bucket));
    state.time          = ticks / 240.;
  }
  /// @brief Reward potential in the two grasp phases; absorbing potential is zero.
  double potential() const {
    if (state.terminated || state.truncated)
      return 0;
    return state.grasped ? 1 - 3 * (state.fruit - release_point(destination)).Length()
                         : -1 - 3 * (state.tool - state.fruit).Length();
  }
  /// @brief Damped finite-difference IK with soft wrist-down preference and bounded steps.
  Pose inverse(int ix, int iy, int iz) {
    auto& cached = cache.at(cell(ix, iy, iz));
    if (cached.known)
      return cached;
    auto robot_copy = robot;
    auto goal       = target(ix, iy, iz);
    goal            = {.62 - goal.x, -goal.y, goal.z - .72};
    Pose result;
    result.known = true;
    double best  = std::numeric_limits<double>::infinity();
    for (int attempt = 0; attempt < 2; ++attempt) {
      Coordinates q = home;
      if (attempt) {
        q[1] = -2.9;
        q[2] = -2.6;
        q[3] = 1.2;
      }
      q[0] = std::atan2(goal.y, goal.x);
      q[5] = 0;
      for (int iteration = 0; iteration < 180; ++iteration) {
        set_joints(robot_copy, q);
        auto   f = feature(robot_copy);
        double e[6]{goal.x - f[0], goal.y - f[1], goal.z - f[2], -f[3], -f[4], -.015 - f[5]};
        double cost = 0;
        for (double v : e)
          cost += v * v;
        if (cost < best) {
          best     = cost;
          result.q = q;
        }
        if (cost < 1e-8)
          break;
        double j[6][6]{};
        for (int a = 0; a < 6; ++a) {
          auto   q1 = q;
          double h  = q[a] + 1e-5 <= upper[a] ? 1e-5 : -1e-5;
          q1[a] += h;
          set_joints(robot_copy, q1);
          auto f1 = feature(robot_copy);
          for (int b = 0; b < 6; ++b)
            j[b][a] = (f1[b] - f[b]) / h;
        }
        double normal[6][7]{};
        for (int a = 0; a < 6; ++a) {
          for (int b = 0; b < 6; ++b)
            for (int c = 0; c < 6; ++c)
              normal[a][b] += j[c][a] * j[c][b];
          normal[a][a] += 1e-5;
          for (int c = 0; c < 6; ++c)
            normal[a][6] += j[c][a] * e[c];
        }
        auto   dq    = solve(normal);
        double scale = 1;
        for (double v : dq)
          scale = std::max(scale, std::abs(v) / .16);
        for (int a = 0; a < 6; ++a)
          q[a] = std::clamp(q[a] + dq[a] / scale, lower[a], upper[a]);
        q[5] = 0;
      }
      set_joints(robot_copy, result.q);
      auto p = feature(robot_copy);
      if ((cad::Vec3d{p[0], p[1], p[2]} - goal).Length() < .003)
        break;
    }
    set_joints(robot_copy, result.q);
    auto p           = feature(robot_copy);
    result.reachable = (cad::Vec3d{p[0], p[1], p[2]} - goal).Length() < .012;
    cached           = result;
    return result;
  }
  /// @brief Conservative tool sphere and carried-payload AABB against static workcell boxes.
  bool blocked(const cad::Vec3d& at) const {
    const auto tool_at = physics_point(at);
    auto       f       = conveyor.world().body(conveyor.observation().fruit);
    Vec3       half{f.dimensions[0] / 2, f.dimensions[0] / 2, f.dimensions[0] / 2};
    if (f.shape == Shape::Capsule) {
      auto   axis = detail::rotate(f.rotation, {0, 1, 0});
      double line = std::max(0., f.dimensions[1] - f.dimensions[0]) / 2;
      for (int i = 0; i < 3; ++i)
        half[i] += line * std::abs(axis[i]);
    }
    auto fruit_at = physics_point(at + grasp_offset);
    for (auto id : conveyor.world().bodies()) {
      auto b = conveyor.world().body(id);
      if (b.motion != Motion::Static || b.sensor)
        continue;
      double squared       = 0;
      bool   fruit_overlap = state.grasped;
      for (int a = 0; a < 3; ++a) {
        double distance = std::abs(tool_at[a] - b.position[a]) - b.dimensions[a] / 2;
        squared += std::pow(std::max(0., distance), 2);
        if (std::abs(fruit_at[a] - b.position[a]) >= half[a] + b.dimensions[a] / 2 - .001)
          fruit_overlap = false;
      }
      if (squared < .011 * .011 || fruit_overlap)
        return true;
    }
    return false;
  }
  /// @brief Align the jaw's horizontal closing direction to the payload's short axis.
  Coordinates align_wrist(Coordinates q) const {
    if (variation != FruitVariation::Diverse)
      return q;
    auto body = conveyor.world().body(conveyor.observation().fruit);
    if (body.shape != Shape::Capsule)
      return q;
    auto       axis = detail::rotate(body.rotation, {0, 1, 0});
    cad::Vec3d desired{-axis[2], axis[0], 0};
    auto       model = robot;
    set_joints(model, q);
    auto   jaw     = model.link_transform("end_link").TransformVector({0, 1, 0});
    double current = std::atan2(-jaw.y, -jaw.x), goal = std::atan2(desired.y, desired.x);
    auto   shifted = q;
    shifted[5]     = std::clamp(q[5] + .001, lower[5], upper[5]);
    set_joints(model, shifted);
    auto   test       = model.link_transform("end_link").TransformVector({0, 1, 0});
    double change     = std::remainder(std::atan2(-test.y, -test.x) - current, 2 * std::acos(-1.));
    double difference = std::remainder(goal - current, std::acos(-1.));
    if (std::abs(change) > 1e-8)
      q[5] = std::clamp(q[5] + difference * (shifted[5] - q[5]) / change, lower[5], upper[5]);
    return q;
  }
  /// @brief Integrate a rate-limited joint/finger servo, and record native frames at 24 Hz.
  void animate(const Coordinates& destination, double finger, double minimum = .15) {
    auto   initial      = state.joints;
    double start_finger = state.finger, duration = minimum;
    for (int i = 0; i < 6; ++i)
      duration = std::max(duration, std::abs(destination[i] - initial[i]) / 1.5);
    duration  = std::max(duration, std::abs(finger - start_finger) / .08);
    int count = int(std::ceil(duration * 240));
    for (int tick = 1; tick <= count; ++tick) {
      double      t = double(tick) / count;
      Coordinates q{};
      for (int i = 0; i < 6; ++i)
        q[i] = initial[i] + t * (destination[i] - initial[i]);
      set_joints(robot, q);
      robot.set_joint("finger_left", start_finger + t * (finger - start_finger));
      if (state.grasped) {
        auto f            = conveyor.world().body(conveyor.observation().fruit);
        f.position        = physics_point(tool() + grasp_offset);
        f.linear_velocity = {0, 0, 0};
        conveyor.world().set_body(conveyor.observation().fruit, f);
      }
      conveyor.step(1. / 240);
      ++ticks;
      if (recorder && ticks % 10 == 0) {
        observe();
        recorder(state);
      }
    }
    observe();
  }
  /// @brief Reinitialize the arm and sampled payload, optionally at its conveyor entry.
  void restart(std::uint64_t seed, bool ready) {
    if (ready)
      conveyor.reset_for_pick(seed);
    else
      conveyor.reset(seed);
    state        = {};
    x            = 0;
    y            = 0;
    z            = 4;
    ticks        = 0;
    state.joints = home;
    auto pose    = inverse(0, 0, 4);
    if (!pose.reachable)
      throw std::runtime_error("DevArm initial grasp-point IK is unreachable");
    set_joints(robot, pose.q);
    robot.set_joint("finger_left", .05);
    observe();
    if (recorder)
      recorder(state);
  }
  /// @brief Test a complete joint interpolation before changing the live servo coordinates.
  bool path_clear(const Coordinates& destination) const {
    auto test_robot = robot;
    for (int k = 1; k <= 12; ++k) {
      Coordinates q{};
      for (int i = 0; i < 6; ++i)
        q[i] = state.joints[i] + double(k) / 12 * (destination[i] - state.joints[i]);
      set_joints(test_robot, q);
      auto p = test_robot.link_transform("end_link").TransformPoint({-.02, 0, 0});
      if (blocked({.62 - p.x, -p.y, .72 + p.z}))
        return false;
    }
    return true;
  }
};
DevArmPickEnvironment::DevArmPickEnvironment(
    const std::string& urdf,
    std::size_t        horizon,
    FruitVariation     variation)
    : impl_(std::make_unique<Impl>(urdf, horizon, variation)) {
  reset(0);
}
DevArmPickEnvironment::~DevArmPickEnvironment() = default;
PickObservation DevArmPickEnvironment::reset(std::uint64_t seed) {
  impl_->restart(seed, true);
  return impl_->state;
}
PickObservation DevArmPickEnvironment::reset_feed(std::uint64_t seed) {
  impl_->restart(seed, false);
  return impl_->state;
}
PickObservation DevArmPickEnvironment::advance_feed(double seconds) {
  if (!std::isfinite(seconds) || seconds <= 0 || seconds > 1. / 30)
    throw std::invalid_argument("invalid feed delta");
  auto& p = *impl_;
  if (p.state.decisions || p.state.terminated || p.state.truncated)
    throw std::logic_error("feeding is before agent actions");
  const int count = int(std::ceil(seconds * 240));
  for (int i = 0; i < count; ++i) {
    p.conveyor.step(seconds / count);
    ++p.ticks;
    p.observe();
    if (p.recorder && p.ticks % 10 == 0)
      p.recorder(p.state);
  }
  if (p.state.feed_stage == ConveyorStage::Lost) {
    p.state.terminated = true;
    p.observe();
  }
  return p.state;
}
void DevArmPickEnvironment::set_destination(PickBin destination) {
  if (destination != PickBin::Red && destination != PickBin::Yellow)
    throw std::invalid_argument("unknown pick destination");
  auto& p = *impl_;
  if (p.state.decisions || p.state.terminated || p.state.truncated)
    throw std::logic_error("select destination before the first action");
  p.destination       = destination;
  p.state.destination = destination;
}
PickObservation DevArmPickEnvironment::step(PickAction action) {
  int a = static_cast<int>(action);
  if (a < 0 || a >= 9)
    throw std::invalid_argument("invalid pick action");
  auto& p = *impl_;
  if (p.state.terminated || p.state.truncated)
    throw std::logic_error("reset ended pick episode");
  if (p.state.feed_stage == ConveyorStage::Travelling)
    throw std::logic_error("wait for the pick sensor before controlling the arm");
  double before     = p.potential();
  p.state.collision = false;
  p.state.reward    = 0;
  ++p.state.decisions;
  if (a < 6) {
    int x = p.x, y = p.y, z = p.z;
    int direction = a % 2 ? -1 : 1;
    if (a < 2)
      x += direction;
    else if (a < 4)
      y += direction;
    else
      z += direction;
    bool accepted = false;
    if (x >= -2 && x <= 2 && y >= -16 && y <= 16 && z >= 0 && z <= 8) {
      auto pose = p.inverse(x, y, z);
      if (p.state.grasped)
        pose.q = p.align_wrist(pose.q);
      if (pose.reachable && p.path_clear(pose.q)) {
        p.animate(pose.q, p.state.finger);
        p.x      = x;
        p.y      = y;
        p.z      = z;
        accepted = true;
      }
    }
    if (!accepted) {
      p.state.collision = true;
      p.animate(p.state.joints, p.state.finger);
    }
  } else if (action == PickAction::Close && !p.state.grasped) {
    bool capture = (p.state.tool - p.state.fruit).Length() < .032;
    if (capture) {
      double aperture = p.state.kind == FruitKind::Apple ? .040 : .024;
      auto   aligned  = p.align_wrist(p.state.joints);
      if (p.variation == FruitVariation::Diverse) {
        aperture = std::min(.049, p.state.fruit_dimensions[0] / 2 + .001);
        p.animate(aligned, .05);
      }
      p.animate(aligned, aperture);
      p.state.grasped   = true;
      p.grasp_offset    = p.state.fruit - p.state.tool;
      auto f            = p.conveyor.world().body(p.conveyor.observation().fruit);
      f.motion          = Motion::Kinematic;
      f.linear_velocity = {0, 0, 0};
      p.conveyor.world().set_body(p.conveyor.observation().fruit, f);
    } else {
      p.animate(p.state.joints, .006);
      p.animate(p.state.joints, .05);
    }
  } else if (action == PickAction::Open && p.state.grasped) {
    auto f            = p.conveyor.world().body(p.conveyor.observation().fruit);
    f.motion          = Motion::Dynamic;
    f.linear_velocity = {0, 0, 0};
    p.conveyor.world().set_body(p.conveyor.observation().fruit, f);
    p.state.grasped  = false;
    p.state.released = true;
    p.animate(p.state.joints, .05, .7);
    auto   body = p.conveyor.world().body(p.conveyor.observation().fruit);
    double v2   = 0;
    for (double v : body.linear_velocity)
      v2 += v * v;
    p.state.success = std::abs(p.state.fruit.x - .1) < .13 &&
                      std::abs(p.state.fruit.y - release_point(p.destination).y) < .105 &&
                      p.state.fruit.z > .44 && p.state.fruit.z < .56 && v2 < .0009 &&
                      p.conveyor.world().grounded(p.conveyor.observation().fruit, .006);
    p.state.terminated = true;
  } else
    p.animate(p.state.joints, p.state.finger);
  if (p.state.fruit.z < .1 || std::abs(p.state.fruit.x) > 1.5 || std::abs(p.state.fruit.y) > 1.2)
    p.state.terminated = true;
  if (!p.state.terminated && p.state.decisions >= p.horizon)
    p.state.truncated = true;
  p.observe();
  p.state.reward = -.05 - (p.state.collision ? .15 : 0) +
                   (p.state.terminated ? (p.state.success ? 10 : -3) : 0) + .98 * p.potential() -
                   before;
  return p.state;
}
PickObservation DevArmPickEnvironment::observation() const {
  return impl_->state;
}
void DevArmPickEnvironment::record(std::function<void(const PickObservation&)> recorder) {
  impl_->recorder = std::move(recorder);
}
std::array<double, 6> DevArmPickEnvironment::joint_target(int x, int y, int z) {
  if (x < -2 || x > 2 || y < -16 || y > 16 || z < 0 || z > 8)
    throw std::invalid_argument("invalid DevArm lattice target");
  const auto pose = impl_->inverse(x, y, z);
  if (!pose.reachable)
    throw std::runtime_error("unreachable DevArm lattice target");
  return pose.q;
}
std::size_t DevArmPickEnvironment::state_count() noexcept {
  return cells * 2 + 1;
}
std::size_t DevArmPickEnvironment::state_space_size() const noexcept {
  return impl_->variation == FruitVariation::Diverse ? cells * 6 + 1 : cells * 2 + 1;
}
void DevArmPickEnvironment::set_visual_style(WorkcellVisualStyle style) {
  impl_->conveyor.set_visual_style(style);
}
WorkcellVisualStyle DevArmPickEnvironment::visual_style() const noexcept {
  return impl_->conveyor.visual_style();
}
cad::Scene DevArmPickEnvironment::static_scene() const {
  auto scene = impl_->conveyor.static_scene();
  scene.objects.erase(
      std::remove_if(
          scene.objects.begin(),
          scene.objects.end(),
          [](const auto& o) {
            return o.name == "apple bin symbol" || o.name == "banana bin symbol";
          }),
      scene.objects.end());
  return scene;
}
cad::Scene DevArmPickEnvironment::fruit_scene(const PickObservation& frame) const {
  const auto& p = *impl_;
  if (frame.kind != p.state.kind)
    throw std::invalid_argument("frame belongs to another fruit reset");
  auto recorded        = p.conveyor.observation();
  recorded.position    = physics_point(frame.fruit);
  recorded.rotation    = frame.fruit_rotation;
  recorded.dimensions  = frame.fruit_dimensions;
  recorded.appearance  = frame.appearance;
  recorded.variation   = frame.variation;
  recorded.stage       = frame.feed_stage;
  recorded.belt_travel = frame.belt_travel;
  auto scene           = p.conveyor.dynamic_scene(recorded);
  return scene;
}
cad::Scene DevArmPickEnvironment::robot_scene(const PickObservation& frame) const {
  auto robot = impl_->robot;
  recorded_joints(robot,frame);
  cad::Scene scene;
  append_arm(scene, robot.scene());
  if (visual_style() == WorkcellVisualStyle::Realistic)
    detail::visuals::robot_materials(scene);
  return scene;
}
cad::Mat4d DevArmPickEnvironment::link_pose(const PickObservation& frame, const std::string& name)
    const {
  auto robot = impl_->robot;
  recorded_joints(robot,frame);
  return cad::Mat4d::Translation({.62, 0, .72}) * cad::Mat4d::Rotation({0, 0, 1}, std::acos(-1.)) *
         robot.link_transform(name);
}
cad::Scene DevArmPickEnvironment::dynamic_scene(const PickObservation& frame) const {
  auto scene = fruit_scene(frame);
  auto robot = impl_->robot;
  recorded_joints(robot,frame);
  append_arm(scene, robot.scene());
  if (visual_style() == WorkcellVisualStyle::Realistic)
    detail::visuals::robot_materials(scene);
  return scene;
}
PickAction PickPolicy::action(std::size_t state) const {
  const auto& row = q.at(state);
  if (row.size() != 9)
    throw std::invalid_argument("malformed Q row");
  for (double value : row)
    if (!std::isfinite(value))
      throw std::invalid_argument("nonfinite Q row");
  return static_cast<PickAction>(std::max_element(row.begin(), row.end()) - row.begin());
}
void PickPolicy::save(const std::string& path) const {
  validate_policy(*this);
  std::ofstream out(path);
  out << "DATAMUNGE_DEVARM_Q 1 " << q.size() << " 9\n" << std::setprecision(17);
  for (const auto& row : q) {
    for (double x : row)
      out << x << ' ';
    out << '\n';
  }
  out.close();
  if (!out)
    throw std::runtime_error("could not write pick checkpoint");
}
PickPolicy PickPolicy::load(const std::string& path) {
  std::ifstream in(path);
  std::string   magic;
  int           version = 0;
  std::size_t   states = 0, actions = 0;
  if (!(in >> magic >> version >> states >> actions) || magic != "DATAMUNGE_DEVARM_Q" ||
      version != 1 || (states != DevArmPickEnvironment::state_count() && states != cells * 6 + 1) ||
      actions != 9)
    throw std::runtime_error("invalid pick checkpoint header");
  PickPolicy result;
  result.q.assign(states, std::vector<double>(9));
  for (auto& row : result.q)
    for (double& x : row)
      if (!(in >> x) || !std::isfinite(x))
        throw std::runtime_error("invalid pick checkpoint values");
  in >> std::ws;
  if (!in.eof())
    throw std::runtime_error("trailing pick checkpoint data");
  return result;
}
PickTrainingResult
train_red_bin(DevArmPickEnvironment& environment, std::size_t episodes, std::uint64_t seed) {
  return train_pick_bin(environment, PickBin::Red, episodes, seed);
}
PickTrainingResult train_pick_bin(
    DevArmPickEnvironment& environment,
    PickBin                goal,
    std::size_t            episodes,
    std::uint64_t          seed) {
  if (goal != PickBin::Red && goal != PickBin::Yellow)
    throw std::invalid_argument("unknown pick destination");
  if (!episodes)
    throw std::invalid_argument("pick training episodes must be positive");
  environment.record({});
  PickTrainingResult result;
  result.destination                     = goal;
  std::size_t                    episode = 0;
  algorithms::TabularEnvironment task;
  task.num_states  = environment.state_space_size();
  task.num_actions = 9;
  task.reset       = [&]() {
    if (episode)
      result.successes.push_back(environment.observation().success);
    environment.reset(seed + episode++);
    environment.set_destination(goal);
    return environment.observation().state;
  };
  task.step = [&](std::size_t, std::size_t action) {
    auto s = environment.step(static_cast<PickAction>(action));
    return algorithms::StepResult{s.state, s.reward, s.terminated || s.truncated};
  };
  algorithms::RLParameters parameters;
  parameters.episodes      = episodes;
  parameters.max_steps     = 100000;
  parameters.learning_rate = .3;
  parameters.discount      = .98;
  parameters.epsilon       = .25;
  parameters.seed          = seed;
  auto trained             = algorithms::q_learning(task, parameters);
  result.successes.push_back(environment.observation().success);
  result.policy.q = std::move(trained.q);
  result.returns  = std::move(trained.episode_returns);
  return result;
}
} // namespace datamunge::sim
