/// @file fruit_stream.cpp
/// @brief Continuous multi-payload rigid-body workcell and semi-Markov scheduler learning.
#include <algorithm>
#include <cmath>
#include <datamunge/sim/devarm_camera.hpp>
#include <datamunge/sim/fruit_stream.hpp>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <set>
#include <stdexcept>
#include <tuple>

#include "math.hpp"
#include "stream_dynamics.hpp"

namespace datamunge::sim {
/// @brief Private geometry, randomization and checkpoint helpers.
namespace {
/// @brief Independent full-width deterministic seed mixer.
std::uint64_t mix(std::uint64_t v) {
  v += 0x9e3779b97f4a7c15ULL;
  v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ULL;
  v = (v ^ (v >> 27)) * 0x94d049bb133111ebULL;
  return v ^ (v >> 31);
}
/// @brief Actual revolute joint names in URDF order.
const std::array<std::string, 6> names{"joint1", "joint2", "joint3", "joint4", "joint5", "joint6"};
/// @brief Convert physics Y-up to CAD Z-up.
cad::Vec3d cad_point(Vec3 p) {
  return {p[0], p[2], p[1]};
}
/// @brief Convert CAD Z-up to physics Y-up.
Vec3 physical(cad::Vec3d p) {
  return {p.x, p.z, p.y};
}
/// @brief Validate a Poisson intensity before changing a stream.
void rate_ok(double v) {
  if (!std::isfinite(v) || v < 0 || v > 5)
    throw std::invalid_argument("stream rate must be 0..5 fruit/second");
}
/// @brief Append world-baked actor meshes, textures and objects with rebased indices.
void append(cad::Scene& scene, cad::Scene extra) {
  int base = int(scene.meshes.size()), tex = int(scene.textures.size());
  for (auto& m : extra.meshes)
    scene.meshes.push_back(std::move(m));
  for (auto& t : extra.textures)
    scene.textures.push_back(std::move(t));
  for (auto o : extra.objects) {
    o.mesh_index += base;
    o.parent = -1;
    for (int* p :
         {&o.texture_index, &o.normal_map_index, &o.roughness_map_index, &o.metallic_map_index})
      if (*p >= 0)
        *p += tex;
    scene.AddObject(o);
  }
}
/// @brief Check the exact fixed action-table shape and finite values.
void policy_ok(const StreamPolicy& p) {
  if (p.q.size() != FruitStreamEnvironment::state_count())
    throw std::invalid_argument("invalid stream Q rows");
  for (const auto& r : p.q)
    for (double v : r)
      if (!std::isfinite(v))
        throw std::invalid_argument("nonfinite stream Q value");
}
} // namespace
/// @brief Exclusive continuing clock, independent randomness, payloads and rate-limited servo
/// state.
struct FruitStreamEnvironment::Impl {
  std::array<double, 6> watch_from{};    ///< Coordinated return-segment initial joints.
  double                watch_finger{0}; ///< Coordinated return-segment initial aperture.
  int watch_segment_ticks{0};   ///< Validated segment duration, or zero before preparation.
  int watch_segment_elapsed{0}; ///< Completed ticks of the coordinated return segment.
  FruitStreamOptions
      options; ///< Validated intensity, capacity and explicit training error channel.
  ConveyorEnvironment workcell{FruitVariation::Diverse}; ///< Canonical static geometry and native
                                                         ///< world; its stop sensor is bypassed.
  ConveyorEnvironment sampler{
      FruitVariation::Diverse}; ///< Independent seeded payload geometry/proxy factory.
  DevArmPickEnvironment model;  ///< Real-URDF cached IK and recorded-pose visual owner.
  cad::Robot            robot;  ///< Lightweight FK state sharing immutable parsed visual geometry.
  std::vector<Body>
      fixtures; ///< Immutable static collision boxes for allocation-free route guards.
  std::set<std::tuple<int, int, int>>
      unreachable; ///< Lattice IK failures independent of live dynamics.
  std::unique_ptr<detail::StreamDynamics>
                        dynamics;    ///< Optional articulated solver; no ideal latch.
  std::array<double, 6> commanded{}; ///< Desired servo coordinates retained between movements.
  double                commanded_finger{.05}; ///< Desired physical half-aperture.
  double                lost_contact_time{0};  ///< Consecutive unsupported grasp duration, seconds.
  std::array<double, 6> lower{};               ///< Lower real-URDF revolute-joint limits.
  std::array<double, 6> upper{};               ///< Upper real-URDF revolute-joint limits.
  StreamFrame           frame;                 ///< Current actual servo state and counters.

  /// @brief One retained physical payload and its asynchronous outcome bookkeeping.
  struct Payload {
    StreamFruit fruit;             ///< Replay appearance, actual pose and predicted class.
    BodyId      body;              ///< World-owned live collision-body identifier.
    double      outcome{-1};       ///< Release/outcome time, negative before either event.
    Vec3        release_origin{};  ///< Position when the dynamic release begins.
    double      settled_since{-1}; ///< Start of continuously supported low-speed bin dwell.
    bool        rejected{false};   ///< Unknown perception excludes further attempts on this fruit.
  };
  std::vector<Payload> payloads;        ///< Bounded records, cleared two seconds after outcomes.
  std::mt19937_64      arrivals;        ///< Arrival RNG, never consumed by policy/vision decisions.
  std::mt19937_64      vision;          ///< Independent synthetic-perception RNG.
  std::uint64_t        seed{0};         ///< Initial stream reproducibility seed.
  std::uint64_t        next_id{1};      ///< Monotonic stream ID for the next Poisson arrival.
  std::uint64_t        ticks{0};        ///< Exact elapsed 240 Hz physics-clock ticks.
  double               next_arrival{0}; ///< Hidden exact time of the next Poisson arrival.
  double               rate{0};         ///< Selected Poisson intensity, fruit per second.
  std::uint64_t        held{0};         ///< Current latched stream ID, zero without a grasp.
  cad::Vec3d           offset;          ///< Stabilized world-space tool-to-fruit offset.
  std::function<StreamPrediction(const StreamFrame&, std::uint64_t)>
      perception; ///< Actual RGB callback, or empty synthetic training channel.
  std::function<void(const StreamFrame&)> recorder; ///< Optional exact-rate frame sink.
  int                   stride{10}; ///< Physics ticks per recorded frame (24 Hz by default).
  bool                  failed_attempt{false};    ///< Per-option penalty flag.
  std::array<double, 6> watch_joints{};           ///< Cached camera-aimed entry observation pose.
  std::vector<std::array<double, 6>> watch_route; ///< Guarded idle-only return waypoints.
  std::size_t   watch_waypoint{0}; ///< Next idle waypoint; cleared by an active interception.
  std::uint64_t mask_tick{std::numeric_limits<std::uint64_t>::max()}; ///< Cached feasibility clock.
  std::array<bool, 5>           mask_cache{};      ///< Owned action feasibility at mask_tick.
  mutable std::array<double, 6> observed_joints{}; ///< Cached actual joints for observation FK.
  mutable cad::Vec3d            observed_point;    ///< Grasp point at observed_joints.
  mutable bool observed_valid{false}; ///< Whether observation FK has a valid cached placement.

  /// @brief Own the real model and validate all independent domain controls.
  Impl(const std::string& path, FruitStreamOptions o)
      : options(o), model(path, 100, FruitVariation::Diverse), robot(cad::Robot::load_urdf(path)) {
    workcell.set_visual_style(o.visual_style);
    model.set_visual_style(o.visual_style);
    rate_ok(o.arrival_rate);
    if (o.articulated_physics && o.training_proxy)
      throw std::invalid_argument("articulated training cannot use the analytic proxy");
    bool valid_dt = false;
    for (int divisor : {240, 480, 960, 1920})
      valid_dt |= std::abs(o.physics_timestep - 1. / divisor) < 1e-12;
    if (!valid_dt)
      throw std::invalid_argument("unsupported articulated physics timestep");
    if (!o.active_capacity || o.active_capacity > 32)
      throw std::invalid_argument("stream capacity must be 1..32");
    for (auto v : o.training_error)
      if (!std::isfinite(v) || v < 0 || v > 1)
        throw std::invalid_argument("invalid synthetic classification error");
    for (double v : {o.queued_wait_cost, o.miss_cost, o.wrong_cost})
      if (!std::isfinite(v) || v < 0 || v > 100)
        throw std::invalid_argument("stream reward costs must be finite in 0..100");
    if (!std::isfinite(o.grasp_alignment_tolerance) || o.grasp_alignment_tolerance < .005 ||
        o.grasp_alignment_tolerance > .05)
      throw std::invalid_argument("grasp alignment tolerance must be finite in .005..05 m");
    for (int k = 0; k < 6; ++k)
      for (const auto& j : robot.joints())
        if (j.name == names[k]) {
          lower[k] = j.lower;
          upper[k] = j.upper;
        }
    for (auto id : workcell.world().bodies()) {
      auto b = workcell.world().body(id);
      if (b.motion == Motion::Static && !b.sensor)
        fixtures.push_back(b);
    }
    if (options.watch_infeed)
      watch_joints = entry_view();
    if (options.articulated_physics)
      dynamics = std::make_unique<detail::StreamDynamics>(
          path, workcell.world(), o.physics_timestep, o.active_capacity, o.physics_backend);
  }
  /// @brief Fit a fixed upstream camera view using the actual rigid-mount extrinsics and URDF
  /// limits. Deterministic bounded coordinate descent balances tool clearance, optical aim and
  /// image roll.
  /// @return Joint coordinates looking at CAD (-.92,0,.86), covering randomized belt entry.
  /// @throws std::runtime_error The model cannot produce the required upstream view.
  std::array<double, 6> entry_view() {
    DevArmWristCamera camera(model, 16, 95);
    PickObservation   f;
    f.finger = .05;
    /// @brief Score camera direction, elevated grasp point and roll without accessing payloads.
    /// @param q Candidate URDF coordinates.
    /// @return Nonnegative deterministic view-placement error.
    auto cost = [&](const std::array<double, 6>& q) {
      f.joints       = q;
      auto view      = camera.pose(f);
      auto direction = (view.target - view.position).Normalized();
      auto aim       = (cad::Vec3d{-.92, 0, .86} - view.position).Normalized();
      auto off       = tool(q) - cad::Vec3d{.12, 0, 1.03};
      auto error     = direction - aim;
      return off.Dot(off) + .5 * error.Dot(error) + .02 * std::pow(1 - view.up.z, 2);
    };
    auto   q    = model.joint_target(0, 0, 4);
    double best = cost(q);
    for (double step : {.3, .15, .075, .03, .01, .003})
      for (int iteration = 0; iteration < 100; ++iteration) {
        bool changed = false;
        for (int i = 0; i < 6; ++i)
          for (double sign : {-1., 1.}) {
            auto next    = q;
            next[i]      = std::clamp(q[i] + sign * step, lower[i], upper[i]);
            double value = cost(next);
            if (value < best) {
              best    = value;
              q       = next;
              changed = true;
            }
          }
        if (!changed)
          break;
      }
    f.joints  = q;
    auto view = camera.pose(f);
    if ((view.target - view.position)
                .Normalized()
                .Dot((cad::Vec3d{-.92, 0, .86} - view.position).Normalized()) < .97 ||
        tool(q).z < .95 || blocked(tool(q)))
      throw std::runtime_error("DevArm cannot safely watch the conveyor entry");
    return q;
  }
  /// @brief FK grasp point from actual six-joint positions, in world CAD metres.
  cad::Vec3d tool(const std::array<double, 6>& q) const {
    auto r = robot;
    if (options.articulated_physics) {
      std::vector<std::pair<std::string, double>> values;
      values.reserve(6);
      for (int i = 0; i < 6; ++i)
        values.emplace_back(names[i], q[i]);
      r.set_measured_joints(values);
    } else
      for (int i = 0; i < 6; ++i)
        r.set_joint(names[i], q[i]);
    auto p = r.link_transform("end_link").TransformPoint({-.02, 0, 0});
    return {.62 - p.x, -p.y, .72 + p.z};
  }
  /// @brief Refine a nearby Cartesian grasp command without changing actual robot state.
  /// Uses a damped 3D FK Jacobian; preserves the nearby wrist configuration.
  /// @param q Nearby bounded IK seed.
  /// @param goal Measured target point, CAD metres.
  /// @return Bounded desired joint coordinates.
  std::array<double, 6> fit_tool(std::array<double, 6> q, cad::Vec3d goal) const {
    for (int iteration = 0; iteration < 16; ++iteration) {
      auto at = tool(q), error = goal - at;
      if (error.Length() < .0004)
        break;
      double j[3][6]{};
      for (int k = 0; k < 6; ++k) {
        auto   test = q;
        double h    = q[k] + .0001 < upper[k] ? .0001 : -.0001;
        test[k] += h;
        auto v  = (tool(test) - at) / h;
        j[0][k] = v.x;
        j[1][k] = v.y;
        j[2][k] = v.z;
      }
      double a[3][4]{};
      for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
          for (int k = 0; k < 6; ++k)
            a[r][c] += j[r][k] * j[c][k];
          if (r == c)
            a[r][c] += .0001;
        }
        a[r][3] = r == 0 ? error.x : r == 1 ? error.y : error.z;
      }
      for (int r = 0; r < 3; ++r) {
        int pivot = r;
        for (int k = r + 1; k < 3; ++k)
          if (std::abs(a[k][r]) > std::abs(a[pivot][r]))
            pivot = k;
        for (int c = 0; c < 4; ++c)
          std::swap(a[r][c], a[pivot][c]);
        double scale = a[r][r];
        for (int c = r; c < 4; ++c)
          a[r][c] /= scale;
        for (int k = 0; k < 3; ++k)
          if (k != r) {
            double v = a[k][r];
            for (int c = r; c < 4; ++c)
              a[k][c] -= v * a[r][c];
          }
      }
      for (int k = 0; k < 6; ++k) {
        double delta = 0;
        for (int r = 0; r < 3; ++r)
          delta += j[r][k] * a[r][3];
        q[k] =
            std::clamp(q[k] + std::clamp(delta, -.12, .12), lower[k] + .00001, upper[k] - .00001);
      }
    }
    return q;
  }
  /// @brief Find a retained payload without exposing a future arrival.
  Payload* find(std::uint64_t id) {
    auto it = std::find_if(
        payloads.begin(), payloads.end(), [&](const auto& p) { return p.fruit.id == id; });
    return it == payloads.end() ? nullptr : &*it;
  }
  /// @brief Read actual articulated payload state, or legacy native dynamics state.
  /// @param id Live native payload identifier.
  /// @return Measured state; the native world is metadata-only in articulated mode.
  Body payload_body(BodyId id) const {
    auto base = workcell.world().body(id);
    return dynamics ? dynamics->body(id, base) : base;
  }
  /// @brief Sort currently observed unhandled payloads by actual leading position.
  std::vector<std::uint64_t> queue() const {
    std::vector<const Payload*> p;
    for (const auto& x : payloads)
      if (x.fruit.status == StreamFruitStatus::Belt && !x.rejected)
        p.push_back(&x);
    std::sort(p.begin(), p.end(), [](auto a, auto b) {
      return a->fruit.body.position[0] > b->fruit.body.position[0];
    });
    std::vector<std::uint64_t> ids;
    for (auto x : p)
      ids.push_back(x->fruit.id);
    return ids;
  }
  /// @brief Refresh bounded snapshots and the observable scheduler abstraction.
  StreamFrame observe() const {
    auto f                = frame;
    f.training_proxy      = options.training_proxy;
    f.articulated_physics = bool(dynamics);
    f.physics_backend     = options.physics_backend;
    if (dynamics) {
      auto diagnostic           = dynamics->diagnostics();
      f.maximum_penetration     = diagnostic[0];
      f.maximum_effort_fraction = diagnostic[1];
      f.maximum_joint_speed     = diagnostic[2];
      f.maximum_finger_speed    = diagnostic[3];
    }
    f.time            = ticks / 240.;
    f.arm.time        = f.time;
    f.arm.belt_travel = .22 * f.time;
    if (!observed_valid || observed_joints != f.arm.joints) {
      observed_point  = tool(f.arm.joints);
      observed_joints = f.arm.joints;
      observed_valid  = true;
    }
    f.arm.tool             = dynamics ? dynamics->tool_point() : observed_point;
    f.arm.grasped          = held != 0;
    f.synthetic_perception = !perception;
    f.fruit.clear();
    for (const auto& p : payloads) {
      auto fruit                  = p.fruit;
      auto b                      = payload_body(p.body);
      fruit.body.position         = b.position;
      fruit.body.velocity         = b.linear_velocity;
      fruit.body.angular_velocity = b.angular_velocity;
      fruit.body.mass             = b.mass;
      fruit.body.rotation         = b.rotation;
      fruit.body.dimensions       = b.dimensions;
      fruit.body.time             = f.time;
      fruit.body.belt_travel      = .22 * f.time;
      f.fruit.push_back(fruit);
    }
    const auto ids   = queue();
    int        count = std::min<std::size_t>(3, ids.size()), front = 0, gap = 4;
    if (!ids.empty()) {
      auto a = std::find_if(
          f.fruit.begin(), f.fruit.end(), [&](const auto& p) { return p.id == ids[0]; });
      front = std::clamp(int(std::floor((a->body.position[0] + 1.) / .1)), 0, 13);
      if (ids.size() > 1) {
        auto b = std::find_if(
            f.fruit.begin(), f.fruit.end(), [&](const auto& p) { return p.id == ids[1]; });
        double d = a->body.position[0] - b->body.position[0];
        gap      = d < .1 ? 0 : d < .25 ? 1 : d < .5 ? 2 : 3;
      }
    }
    int side = f.arm.tool.y < -.2 ? 1 : f.arm.tool.y > .2 ? 2 : 0, pending = 0;
    for (const auto& p : payloads)
      pending |= p.fruit.status == StreamFruitStatus::Released;
    f.state = ((((count * 14) + front) * 5 + gap) * 3 + side) * 2 + pending;
    return f;
  }
  /// @brief Sample the next exact exponential gap without consuming any policy RNG draws.
  double gap() {
    return rate > 0 ? std::exponential_distribution<double>(rate)(arrivals)
                    : std::numeric_limits<double>::infinity();
  }
  /// @brief Insert every scheduled Poisson arrival, counting capacity loss rather than deferring
  /// it.
  void spawn() {
    const auto id = next_id++;
    ++frame.counts.arrivals;
    if (payloads.size() >= options.active_capacity) {
      ++frame.counts.missed;
      ++frame.counts.overflow;
      return;
    }
    auto fruit_seed = mix(seed ^ mix(id));
    sampler.reset(fruit_seed);
    auto s = sampler.observation();
    auto b = sampler.world().body(s.fruit);
    if (dynamics) {
      b.allowed_rotation = {true, true, true};
      s.physical_origin  = true;
    }
    auto body = workcell.world().add_body(b);
    if (dynamics)
      dynamics->spawn(body, s);
    s.fruit       = body;
    s.time        = ticks / 240.;
    s.belt_travel = .22 * s.time;
    payloads.push_back(
        {StreamFruit{id, fruit_seed, next_arrival, s, StreamFruitStatus::Belt, -1, 0},
         body,
         -1,
         {},
         -1,
         false});
  }
  /// @brief Integrate one real physics tick, all asynchronous outcomes and scheduled births.
  /// @param command Adopt the newly supplied servo reference when true.
  void tick(bool command = false) {
    const double now = ticks / 240.;
    while (next_arrival <= now + 1e-12) {
      spawn();
      next_arrival += gap();
    }
    if (held && !dynamics) {
      auto* p           = find(held);
      auto  b           = payload_body(p->body);
      b.position        = physical(tool(frame.arm.joints) + offset);
      b.linear_velocity = {0, 0, 0};
      workcell.world().set_body(p->body, b);
    }
    if (dynamics) {
      if (command) {
        commanded        = frame.arm.joints;
        commanded_finger = frame.arm.finger;
      }
      dynamics->step(commanded, commanded_finger);
      dynamics->observe(frame.arm);
      if (held) {
        auto* p           = find(held);
        lost_contact_time = dynamics->gripped(p->body) ? 0 : lost_contact_time + 1. / 240;
        if (lost_contact_time > .10) {
          p->fruit.status   = StreamFruitStatus::Released;
          p->outcome        = (ticks + 1) / 240.;
          p->release_origin = payload_body(p->body).position;
          held              = 0;
          failed_attempt    = true;
        }
      }
    } else if (!payloads.empty())
      workcell.world().step(1. / 240);
    ++ticks;
    for (auto& p : payloads) {
      auto b                = payload_body(p.body);
      p.fruit.body.position = b.position;
      p.fruit.body.velocity = b.linear_velocity;
      if (p.fruit.status == StreamFruitStatus::Belt &&
          (b.position[0] > .30 || b.position[1] < .15 || std::abs(b.position[2]) > .8)) {
        p.fruit.status = StreamFruitStatus::Missed;
        p.outcome      = ticks / 240.;
        ++frame.counts.missed;
      } else if (p.fruit.status == StreamFruitStatus::Released) {
        double v2 = 0;
        for (double v : b.linear_velocity)
          v2 += v * v;
        bool bin =
            std::abs(b.position[0] - .1) < .13 && std::abs(std::abs(b.position[2]) - .42) < .105 &&
            b.position[1] > .44 && b.position[1] < .57 && v2 < .0009 &&
            (dynamics ? dynamics->bin_supported(p.body) : workcell.world().grounded(p.body, .006));
        if (dynamics) {
          double spin = 0;
          for (double w : b.angular_velocity)
            spin += w * w;
          bin &= spin < .04;
          if (bin) {
            if (p.settled_since < 0)
              p.settled_since = ticks / 240.;
            bin = ticks / 240. - p.settled_since >= .2;
          } else
            p.settled_since = -1;
        }
        if (bin) {
          bool correct   = (b.position[2] < 0) == (p.fruit.body.kind == FruitKind::Apple);
          p.fruit.status = correct ? StreamFruitStatus::Correct : StreamFruitStatus::Wrong;
          p.outcome      = ticks / 240.;
          if (correct)
            ++frame.counts.correct;
          else
            ++frame.counts.wrong;
        } else if (
            b.position[1] < .15 ||
            (dynamics && b.position[0] > .30 && b.position[1] > .65 &&
             std::abs(b.position[2]) < .2) ||
            ticks / 240. - p.outcome > 3) {
          p.fruit.status = StreamFruitStatus::Missed;
          p.outcome      = ticks / 240.;
          ++frame.counts.missed;
        }
      }
    }
    for (auto it = payloads.begin(); it != payloads.end();)
      if ((it->fruit.status == StreamFruitStatus::Correct ||
           it->fruit.status == StreamFruitStatus::Wrong ||
           it->fruit.status == StreamFruitStatus::Missed) &&
          ticks / 240. - it->outcome >= 2) {
        if (dynamics)
          dynamics->remove(it->body);
        workcell.world().remove_body(it->body);
        it = payloads.erase(it);
      } else
        ++it;
    if (recorder && ticks % std::uint64_t(stride) == 0)
      recorder(observe());
  }
  /// @brief Analytic vertical support extent for the optional, explicitly approximate trainer.
  double support(const Body& b) const {
    double r = b.dimensions[0] / 2;
    if (b.shape == Shape::Capsule)
      r += .5 * (b.dimensions[1] - b.dimensions[0]) *
           std::abs(detail::rotate(b.rotation, {0, 1, 0})[1]);
    return r;
  }
  /// @brief Event-driven proxy: constant belt transport, fixed settling delay, no fruit contacts.
  void advance_fast(std::uint64_t count) {
    const auto end = ticks + count;
    while (ticks < end) {
      const double now = ticks / 240.;
      while (next_arrival <= now + 1e-12) {
        spawn();
        next_arrival += gap();
      }
      auto stop = end;
      if (std::isfinite(next_arrival) && next_arrival <= end / 240.)
        stop = std::min(stop, std::max(ticks + 1, std::uint64_t(std::ceil(next_arrival * 240))));
      for (auto& p : payloads) {
        double event = std::numeric_limits<double>::infinity();
        if (p.fruit.status == StreamFruitStatus::Belt) {
          auto b = payload_body(p.body);
          event  = now + std::max(0., (.30 - b.position[0]) / .22) + 1. / 240;
        } else if (p.fruit.status == StreamFruitStatus::Released)
          event = p.outcome + .4;
        else if (p.outcome >= 0)
          event = p.outcome + 2;
        if (std::isfinite(event))
          stop = std::min(stop, std::max(ticks + 1, std::uint64_t(std::ceil(event * 240))));
      }
      const double dt = (stop - ticks) / 240., at = stop / 240.;
      for (auto& p : payloads) {
        auto b = payload_body(p.body);
        if (p.fruit.status == StreamFruitStatus::Belt) {
          b.position[0] += .22 * dt;
          b.position[1]     = .72 + support(b);
          b.linear_velocity = {.22, 0, 0};
          if (b.position[0] > .30) {
            p.fruit.status = StreamFruitStatus::Missed;
            p.outcome      = at;
            ++frame.counts.missed;
          }
        } else if (p.fruit.status == StreamFruitStatus::Held) {
          b.position        = physical(tool(frame.arm.joints) + offset);
          b.linear_velocity = {0, 0, 0};
        } else if (p.fruit.status == StreamFruitStatus::Released) {
          double age = at - p.outcome;
          bool   bin = std::abs(p.release_origin[0] - .1) < .13 &&
                     std::abs(std::abs(p.release_origin[2]) - .42) < .105;
          b.position = p.release_origin;
          b.position[1] =
              std::max((bin ? .4425 : 0) + support(b), p.release_origin[1] - 4.905 * age * age);
          b.linear_velocity = {0, 0, 0};
          if (age >= .4 - 1e-9) {
            if (bin) {
              bool correct   = (b.position[2] < 0) == (p.fruit.body.kind == FruitKind::Apple);
              p.fruit.status = correct ? StreamFruitStatus::Correct : StreamFruitStatus::Wrong;
              if (correct)
                ++frame.counts.correct;
              else
                ++frame.counts.wrong;
            } else {
              p.fruit.status = StreamFruitStatus::Missed;
              ++frame.counts.missed;
            }
            p.outcome = at;
          }
        }
        workcell.world().set_body(p.body, b);
        p.fruit.body.position = b.position;
        p.fruit.body.velocity = b.linear_velocity;
      }
      ticks = stop;
      for (auto it = payloads.begin(); it != payloads.end();)
        if ((it->fruit.status == StreamFruitStatus::Correct ||
             it->fruit.status == StreamFruitStatus::Wrong ||
             it->fruit.status == StreamFruitStatus::Missed) &&
            at - it->outcome >= 2 - 1e-9) {
          if (dynamics)
            dynamics->remove(it->body);
          workcell.world().remove_body(it->body);
          it = payloads.erase(it);
        } else
          ++it;
    }
  }
  /// @brief Advance a tick-quantized duration, preserving the one continuing world clock.
  void advance(double seconds) {
    int n = std::max(1, int(std::ceil(seconds * 240)));
    if (options.training_proxy && !recorder) {
      advance_fast(n);
      return;
    }
    for (int i = 0; i < n; ++i)
      tick();
  }
  /// @brief Real servo time at 1.5 rad/s and .08 m/s, with a .12 s settling floor.
  double duration(const std::array<double, 6>& q, double finger) const {
    return travel(frame.arm.joints, frame.arm.finger, q, finger);
  }
  /// @brief Same bounded servo duration between two explicit poses without mutating the frame.
  /// @param from Initial revolute coordinates.
  /// @param aperture Initial finger aperture, metres.
  /// @param q Destination revolute coordinates.
  /// @param finger Destination aperture, metres.
  /// @return Maximum bounded joint/finger travel time with the existing .12-second floor.
  double travel(
      const std::array<double, 6>& from,
      double                       aperture,
      const std::array<double, 6>& q,
      double                       finger) const {
    double t = .12;
    for (int i = 0; i < 6; ++i)
      t = std::max(t, std::abs(q[i] - from[i]) / 1.5);
    return std::max(t, std::abs(finger - aperture) / .08);
  }
  /// @brief Conservative tool sphere and rotated capsule/sphere payload bounds against fixtures.
  bool blocked(cad::Vec3d at) const {
    auto tool_at = physical(at);
    Vec3 half{};
    Vec3 fruit_at{};
    if (held) {
      auto it = std::find_if(
          payloads.begin(), payloads.end(), [&](const auto& p) { return p.fruit.id == held; });
      auto b = payload_body(it->body);
      half   = {b.dimensions[0] / 2, b.dimensions[0] / 2, b.dimensions[0] / 2};
      if (b.shape == Shape::Capsule) {
        auto axis = detail::rotate(b.rotation, {0, 1, 0});
        for (int i = 0; i < 3; ++i)
          half[i] += .5 * (b.dimensions[1] - b.dimensions[0]) * std::abs(axis[i]);
      }
      fruit_at = physical(at + offset);
    }
    for (const auto& b : fixtures) {
      double squared = 0;
      bool   overlap = held != 0 && !dynamics;
      for (int i = 0; i < 3; ++i) {
        double d = std::abs(tool_at[i] - b.position[i]) - b.dimensions[i] / 2;
        squared += std::max(0., d) * std::max(0., d);
        if (std::abs(fruit_at[i] - b.position[i]) >= half[i] + b.dimensions[i] / 2 - .001)
          overlap = false;
      }
      if (squared < .011 * .011 || overlap)
        return true;
    }
    return false;
  }
  /// @brief Test the existing sixteen-point tool/payload fixture guard along joint interpolation.
  /// @param from Start coordinates.
  /// @param q Destination coordinates.
  /// @return True when every sampled placement clears static fixtures.
  bool clear_path(const std::array<double, 6>& from, const std::array<double, 6>& q) const {
    for (int k = 1; k <= 16; ++k) {
      auto test = from;
      for (int i = 0; i < 6; ++i)
        test[i] += double(k) / 16 * (q[i] - from[i]);
      if (blocked(tool(test)))
        return false;
    }
    return true;
  }
  /// @brief Rate-limited interpolation, rejecting fixture intersections before moving anything.
  bool move(const std::array<double, 6>& q, double finger) {
    auto from = frame.arm.joints;
    if (!clear_path(from, q))
      return false;
    const bool carrying = dynamics && held;
    if (carrying)
      finger = 0;
    const double start = frame.arm.finger;
    // Quintic carrying references start/end with zero velocity and acceleration.
    // Its peak normalized derivative is 1.875; doubling the travel interval
    // keeps the existing joint/finger reference speed limits while reducing jerk.
    const bool smooth = dynamics && held && options.synchronized_grasp;
    const int  n      = int(std::ceil(duration(q, finger) * 240 * (smooth ? 2. : 1.)));
    if (options.training_proxy && !recorder) {
      frame.arm.joints = q;
      frame.arm.finger = finger;
      advance_fast(n);
      return true;
    }
    for (int k = 1; k <= n; ++k) {
      double t = double(k) / n;
      if (smooth)
        t = t * t * t * (10 + t * (-15 + 6 * t));
      for (int i = 0; i < 6; ++i)
        frame.arm.joints[i] = from[i] + t * (q[i] - from[i]);
      // The measured gap is not a new opening command. Preserve the existing
      // zero-aperture force request throughout a carried move, from its first tick.
      frame.arm.finger = carrying ? 0 : start + t * (finger - start);
      tick(true);
    }
    return true;
  }
  /// @brief Spend exactly one wait budget looking upstream, yielding at the next scheduler
  /// decision. Builds a direct route or elevated lateral waypoints; failed guards leave the arm
  /// stationary. Full physics and the analytic trainer share servo endpoints and elapsed duration.
  /// @param empty_queue True only when no eligible belt fruit is currently tracked.
  void wait(bool empty_queue) {
    if (!options.watch_infeed || !empty_queue || held) {
      advance(.125);
      return;
    }
    if (watch_route.empty()) {
      auto previous = frame.arm.joints;
      watch_route.push_back(watch_joints);
      if (!clear_path(previous, watch_joints)) {
        watch_route.clear();
        int side = std::clamp(int(std::lround(tool(previous).y / .03)), -16, 16);
        watch_route.push_back(pose(-2, side, 6));
        int direction = side > 0 ? -1 : 1;
        for (int y = side; y != 0; y += direction)
          watch_route.push_back(pose(-2, y + direction, 6));
        watch_route.push_back(watch_joints);
      }
      for (const auto& q : watch_route) {
        if (!clear_path(previous, q)) {
          watch_route.clear();
          advance(.125);
          return;
        }
        previous = q;
      }
      watch_waypoint      = 0;
      watch_segment_ticks = 0;
    }
    if (watch_waypoint == watch_route.size()) {
      advance(.125);
      return;
    }
    if (options.synchronized_grasp) {
      // Full physics and the proxy follow the same coordinated, prevalidated
      // segment. Only the proxy batches ticks; no held payload is present.
      int remaining = 30;
      while (remaining && watch_waypoint < watch_route.size()) {
        const auto& goal = watch_route[watch_waypoint];
        if (!watch_segment_ticks) {
          watch_from     = frame.arm.joints;
          watch_finger   = frame.arm.finger;
          double seconds = std::abs(.05 - watch_finger) / .08;
          for (int i = 0; i < 6; ++i)
            seconds = std::max(seconds, std::abs(goal[i] - watch_from[i]) / 1.5);
          watch_segment_ticks   = std::max(1, int(std::ceil(seconds * 240)));
          watch_segment_elapsed = 0;
        }
        const int count = std::min(remaining, watch_segment_ticks - watch_segment_elapsed);
        /// @brief Place the real coordinates at an exact tick of the guarded segment.
        /// @param elapsed Completed segment ticks, at most watch_segment_ticks.
        auto place = [&](int elapsed) {
          double t = double(elapsed) / watch_segment_ticks;
          for (int i = 0; i < 6; ++i)
            frame.arm.joints[i] = elapsed == watch_segment_ticks
                                      ? goal[i]
                                      : watch_from[i] + t * (goal[i] - watch_from[i]);
          frame.arm.finger = watch_finger + t * (.05 - watch_finger);
        };
        if (options.training_proxy && !recorder) {
          watch_segment_elapsed += count;
          place(watch_segment_elapsed);
          advance_fast(count);
        } else
          for (int k = 0; k < count; ++k) {
            place(++watch_segment_elapsed);
            tick(true);
          }
        remaining -= count;
        if (watch_segment_elapsed == watch_segment_ticks) {
          ++watch_waypoint;
          watch_segment_ticks = 0;
        }
      }
      if (remaining) {
        if (options.training_proxy && !recorder)
          advance_fast(remaining);
        else
          for (int k = 0; k < remaining; ++k)
            tick();
      }
      return;
    }
    for (int k = 0; k < 30; ++k) {
      if (watch_waypoint < watch_route.size()) {
        const auto& goal    = watch_route[watch_waypoint];
        auto        next    = frame.arm.joints;
        bool        reached = true;
        for (int i = 0; i < 6; ++i) {
          const double delta = goal[i] - next[i];
          next[i] =
              std::abs(delta) <= 1.5 / 240 ? goal[i] : next[i] + std::copysign(1.5 / 240, delta);
          reached &= next[i] == goal[i];
        }
        if (clear_path(frame.arm.joints, next)) {
          frame.arm.joints = next;
          frame.arm.finger += std::clamp(.05 - frame.arm.finger, -.08 / 240, .08 / 240);
          if (reached)
            ++watch_waypoint;
        }
      }
      if (options.training_proxy && !recorder)
        advance_fast(1);
      else
        tick(true);
    }
  }
  /// @brief Resolve a canonical IK lattice waypoint and preserve a held fruit's jaw alignment.
  std::array<double, 6> pose(int x, int y, int z, bool align = false) {
    std::array<double, 6> q;
    try {
      if (dynamics && unreachable.count({x, y, z}))
        throw std::runtime_error("unreachable DevArm lattice target");
      q = model.joint_target(x, y, z);
    } catch (const std::runtime_error& error) {
      if (!dynamics || std::string(error.what()) != "unreachable DevArm lattice target")
        throw;
      unreachable.emplace(x, y, z);
      q = frame.arm.joints;
      for (int i = 0; i < 6; ++i)
        q[i] = std::clamp(q[i], lower[i] + .00001, upper[i] - .00001);
      q = fit_tool(q, {.1 + .03 * x, .03 * y, .76 + .03 * z});
    }
    if (!align || !held)
      return q;
    auto* p = find(held);
    auto  b = payload_body(p->body);
    return align_jaws(q, b);
  }
  /// @brief Align the actual sixth joint to a capsule's horizontal short axis.
  std::array<double, 6> align_jaws(std::array<double, 6> q, const Body& b) {
    if (b.shape != Shape::Capsule)
      return q;
    auto       axis = detail::rotate(b.rotation, {0, 1, 0});
    cad::Vec3d desired{-axis[2], axis[0], 0};
    auto       f   = frame.arm;
    f.joints       = q;
    auto   jaw     = model.link_pose(f).TransformVector({0, 1, 0});
    double current = std::atan2(jaw.y, jaw.x), goal = std::atan2(desired.y, desired.x);
    f.joints[5]   = std::clamp(q[5] + .001, lower[5], upper[5]);
    auto   test   = model.link_pose(f).TransformVector({0, 1, 0});
    double change = std::remainder(std::atan2(test.y, test.x) - current, 2 * std::acos(-1.));
    if (std::abs(change) > 1e-8) {
      double base =
          q[5] + std::remainder(goal - current, std::acos(-1.)) * (f.joints[5] - q[5]) / change;
      double best = std::numeric_limits<double>::infinity();
      for (int k = -3; k <= 3; ++k) {
        double candidate = base + k * std::acos(-1.);
        if (candidate < lower[5] || candidate > upper[5])
          continue;
        double distance = std::abs(candidate - frame.arm.joints[5]);
        if (distance < best) {
          best = distance;
          q[5] = candidate;
        }
      }
    }
    return q;
  }
  /// @brief Shared deadline/fixture plan for feasibility queries and actual timed interception.
  struct InterceptPlan {
    std::array<double, 6>              down{}; ///< Jaw-aligned grasp pose.
    std::vector<std::array<double, 6>> route; ///< Guarded approach, ending at the inspection hover.
    double approach{0};                       ///< Sum of rate-limited approach durations, seconds.
    double closure{0};      ///< Estimated jaw-closing time before two-sided contact.
    double descent{0};      ///< Hover-to-grasp duration, seconds.
    double gate{0};         ///< Requested longitudinal capture position in CAD metres.
    int    y{0};            ///< Observed lateral lattice bin.
    bool   feasible{false}; ///< Target, path and deadline all permit this attempt.
    bool   clear{false};    ///< Geometry-only feasibility, independent of longitudinal belt travel.
  };
  /// @brief Bounded geometry-plan cache; live deadlines are reevaluated on every query.
  struct PlanCache {
    bool                  valid{false}; ///< Whether the remaining key and plan are initialized.
    std::array<double, 6> joints{};     ///< Exact actual arm coordinates.
    double                finger{0};    ///< Exact actual finger aperture.
    std::uint64_t         id{0};        ///< Payload identity within the current reset.
    std::array<double, 4> rotation{};   ///< Actual stabilized payload quaternion.
    int                   y{0};         ///< Current lateral target lattice bin.
    InterceptPlan         value;        ///< Owned geometry route, capped at three waypoints.
  };
  std::array<PlanCache, 4> plans; ///< Four direct-mapped identity/gate geometry entries.
  /// @brief Plan without advancing physics, reading labels or looking at scheduled births.
  /// @param id Currently eligible payload.
  /// @param x Longitudinal lattice gate, -2 or 2.
  /// @return Owned approach/descent plan and feasibility flag.
  InterceptPlan plan(std::uint64_t id, int x) {
    InterceptPlan result;
    auto*         p = find(id);
    if (!p)
      return result;
    result.y    = std::clamp(int(std::lround(p->fruit.body.position[2] / .03)), -2, 2);
    result.gate = .1 + .03 * x;

    auto& cached = plans[(id % 2) * 2 + (x == 2)];
    if (cached.valid && cached.id == id && cached.joints == frame.arm.joints &&
        cached.finger == frame.arm.finger && cached.y == result.y &&
        cached.rotation == payload_body(p->body).rotation) {
      result          = cached.value;
      result.feasible = result.clear && (result.gate - p->fruit.body.position[0]) / .22 + .08 >=
                                            result.approach + result.descent + result.closure + .05;
      return result;
    }
    /// @brief Save only geometry, then apply the current deadline to the owned result.
    /// @return Complete current plan without reading a future arrival.
    auto finish = [&]() {
      cached.valid    = true;
      cached.id       = id;
      cached.joints   = frame.arm.joints;
      cached.finger   = frame.arm.finger;
      cached.y        = result.y;
      cached.rotation = payload_body(p->body).rotation;
      cached.value    = result;
      result.feasible = result.clear && (result.gate - p->fruit.body.position[0]) / .22 + .08 >=
                                            result.approach + result.descent + result.closure + .05;
      return result;
    };
    auto hover  = pose(x, result.y, x == 2 ? 3 : 4);
    result.down = align_jaws(pose(x, result.y, 0), payload_body(p->body));
    if (dynamics)
      result.down = fit_tool(
          result.down,
          {result.gate,
           p->fruit.body.position[2],
           std::clamp(p->fruit.body.position[1], .745, .795)});
    if (options.synchronized_grasp)
      hover[5] = result.down[5];
    result.descent      = travel(hover, .05, result.down, .05);
    result.gate         = .1 + .03 * x;
    const double direct = duration(hover, .05);
    if (clear_path(frame.arm.joints, hover)) {
      result.route    = {hover};
      result.approach = direct;
      result.clear    = !options.synchronized_grasp || clear_path(hover, result.down);
      return finish();
    }
    if (!options.synchronized_grasp)
      return finish();
    std::vector<std::vector<std::array<double, 6>>> candidates{{hover}};
    if (options.synchronized_grasp) {
      int  side = std::clamp(int(std::lround(tool(frame.arm.joints).y / .03)), -16, 16);
      auto high = pose(x, side, 6);
      high[5]   = hover[5];
      candidates.push_back({high, hover});
      auto over = pose(-2, result.y, 6);
      over[5]   = hover[5];
      candidates.push_back({high, over, hover});
    }
    double best = std::numeric_limits<double>::infinity();
    for (const auto& route : candidates) {
      auto   from   = frame.arm.joints;
      double finger = frame.arm.finger, seconds = 0;
      bool   clear = true;
      for (const auto& q : route) {
        clear &= clear_path(from, q);
        seconds += travel(from, finger, q, .05);
        from   = q;
        finger = .05;
      }
      if (clear && seconds < best) {
        best         = seconds;
        result.route = route;
      }
    }
    result.approach = best;
    result.clear    = std::isfinite(best) && clear_path(hover, result.down);
    return finish();
  }
  /// @brief Cache current mask once per physical tick; no future events or true labels are read.
  /// @return Owned wait/front/next gate flags.
  std::array<bool, 5> action_mask() {
    if (mask_tick == ticks)
      return mask_cache;
    mask_cache = {true, false, false, false, false};
    if (!held) {
      const auto ids = queue();
      for (int a = 1; a < 5; ++a) {
        std::size_t index = a >= 3 ? 1 : 0;
        if (index < ids.size())
          mask_cache[a] = plan(ids[index], a % 2 ? -2 : 2).feasible;
      }
    }
    mask_tick = ticks;
    return mask_cache;
  }
  /// @brief Attempt one timed, camera-routed pick without stopping the belt or resetting joints.
  void intercept(std::uint64_t id, int x) {
    watch_route.clear();
    auto* p = find(id);
    if (!p)
      return;
    frame.target = id;
    ++frame.counts.attempts;
    const auto target = plan(id, x);
    if (!target.feasible) {
      failed_attempt = true;
      advance(.125);
      frame.target = 0;
      return;
    }
    const auto&  down    = target.down;
    const double descent = target.descent, gate = target.gate;
    const int    y = target.y;
    for (const auto& q : target.route)
      if (!move(q, .05)) {
        failed_attempt = true;
        frame.target   = 0;
        return;
      }
    p = find(id);
    if (!p || p->fruit.status != StreamFruitStatus::Belt) {
      failed_attempt = true;
      frame.target   = 0;
      return;
    }
    double wait = (gate - payload_body(p->body).position[0]) / .22 - descent - target.closure - .05;
    if (wait > 0)
      advance(wait);
    advance(.05); // Explicit simulated inference latency.
    p = find(id);
    if (!p || p->fruit.status != StreamFruitStatus::Belt) {
      failed_attempt = true;
      frame.target   = 0;
      return;
    }
    StreamPrediction prediction;
    if (perception)
      prediction = perception(observe(), id);
    else {
      int  truth = p->fruit.body.kind == FruitKind::Apple ? 0 : 1;
      bool wrong = std::generate_canonical<double, 53>(vision) < options.training_error[truth];
      prediction = {wrong ? 1 - truth : truth, 1 - options.training_error[truth]};
    }
    if (prediction.label < -1 || prediction.label > 1 || !std::isfinite(prediction.confidence) ||
        prediction.confidence < 0 || prediction.confidence > 1)
      throw std::invalid_argument("invalid stream perception prediction");
    p                   = find(id);
    p->fruit.prediction = prediction.label;
    p->fruit.confidence = prediction.confidence;
    if (prediction.label < 0) {
      ++frame.counts.unknown;
      p->rejected    = true;
      mask_tick      = std::numeric_limits<std::uint64_t>::max();
      failed_attempt = true;
      frame.target   = 0;
      return;
    }
    ++frame.counts.inspections;
    frame.counts.classification_correct +=
        prediction.label == (p->fruit.body.kind == FruitKind::Apple ? 0 : 1);
    if (!move(down, .05)) {
      failed_attempt = true;
      frame.target   = 0;
      return;
    }
    p = find(id);
    if (!p || p->fruit.status != StreamFruitStatus::Belt) {
      failed_attempt = true;
      frame.target   = 0;
      return;
    }
    auto b  = payload_body(p->body);
    auto at = tool(frame.arm.joints);
    if ((cad_point(b.position) - at).Length() > (dynamics ? .20 : .032)) {
      failed_attempt = true;
      frame.target   = 0;
      return;
    }
    if (dynamics) {
      // Contact forces, not a body attachment, determine whether the jaws carry fruit.
      const auto body_id   = p->body;
      double     aperture  = .05;
      auto       desired   = down;
      bool       contact   = false;
      int        supported = 0;
      bool       aligned   = false;
      for (int k = 1; k <= 360; ++k) {
        p = find(id);
        if (!p || p->fruit.status != StreamFruitStatus::Belt)
          break;
        auto measured = payload_body(body_id);
        auto centre   = cad_point(measured.position);
        auto actual   = tool(frame.arm.joints);
        const bool touching = options.hold_grasp_target && dynamics->grip_force(body_id) > .5;
        if (!touching && k % 8 == 1)
          desired = fit_tool(
              desired,
              {centre.x + std::max(0., measured.linear_velocity[0]) * .025,
               centre.y,
               std::clamp(centre.z, .745, .795)});
        aligned |= std::hypot(actual.x - centre.x, actual.y - centre.y) <
                       options.grasp_alignment_tolerance &&
                   std::abs(actual.z - centre.z) < .025;
        auto next = commanded;
        for (int j = 0; j < 6; ++j)
          next[j] += std::clamp(desired[j] - next[j], -1.5 / 240, 1.5 / 240);
        frame.arm.joints = next;
        if (aligned)
          aperture = std::max(0., aperture - .08 / 240);
        frame.arm.finger = aperture;
        tick(true);
        supported = dynamics->grip_force(body_id) >= 2 ? supported + 1 : 0;
        if (supported >= 24) {
          contact = true;
          break;
        }
      }
      p = find(id);
      if (!contact || !p || p->fruit.status != StreamFruitStatus::Belt) {
        failed_attempt = true;
        move(frame.arm.joints, .05);
        frame.target = 0;
        return;
      }
      // Keep a closing force command while the actual fingers stop at the object.
      commanded_finger  = 0;
      b                 = payload_body(p->body);
      held              = id;
      lost_contact_time = 0;
      offset            = cad_point(b.position) - tool(frame.arm.joints);
      p->fruit.status   = StreamFruitStatus::Held;
      ++frame.counts.grasped;
    } else {
      held              = id;
      offset            = cad_point(b.position) - at;
      b.motion          = Motion::Kinematic;
      b.linear_velocity = {0, 0, 0};
      workcell.world().set_body(p->body, b);
      p->fruit.status = StreamFruitStatus::Held;
      ++frame.counts.grasped;
      move(frame.arm.joints, std::min(.049, b.dimensions[0] / 2 + .001));
    }
    int  side = prediction.label == 0 ? -13 : 13;
    bool safe = false;
    if (dynamics && options.synchronized_grasp) {
      // Clear the optical post with the whole payload: first lift at the capture
      // point, then cross downstream of the post before approaching the bin.
      // Nearby Cartesian IK seeds avoid changing wrist roll to a lattice branch
      // while a curved object is clamped between the physical finger pads.
      auto move_tool = [&](cad::Vec3d goal) {
        if (held != id)
          return false;
        auto target = fit_tool(frame.arm.joints, goal);
        if ((tool(target) - goal).Length() > .012)
          return false;
        return move(target, frame.arm.finger) && held == id;
      };
      auto   capture = tool(frame.arm.joints);
      double bin_y   = prediction.label == 0 ? -.42 : .42;
      safe           = move_tool({capture.x, capture.y, .94}) && move_tool({.29, capture.y, .94}) &&
             move_tool({.29, bin_y, .94}) && move_tool({.16, bin_y, .86});
    } else
      safe = move(pose(x, y, 3, true), frame.arm.finger) &&
             move(pose(0, side, 4, true), frame.arm.finger);
    if (dynamics && held != id) {
      failed_attempt = true;
      move(frame.arm.joints, .05);
      frame.target = 0;
      return;
    }
    if (!safe) {
      // The optical sensor post reaches .88 m: route over it at .94 m,
      // using small Cartesian waypoints to keep the interpolated FK path clear.
      safe = move(pose(-2, y, 6, true), frame.arm.finger);
      if (safe && !move(pose(-2, side, 6, true), frame.arm.finger)) {
        int direction = side > y ? 1 : -1;
        for (int yy = y + direction; safe && (direction > 0 ? yy <= side : yy >= side);
             yy += direction)
          safe = move(pose(-2, yy, 6, true), frame.arm.finger);
      }
      if (safe)
        safe = move(pose(0, side, 4, true), frame.arm.finger);
    }
    if (dynamics && held != id) {
      failed_attempt = true;
      move(frame.arm.joints, .05);
      frame.target = 0;
      return;
    }
    p        = find(id);
    b        = payload_body(p->body);
    b.motion = Motion::Dynamic;
    if (!dynamics)
      b.linear_velocity = {0, 0, 0};
    if (!dynamics)
      workcell.world().set_body(p->body, b);
    p->fruit.status   = StreamFruitStatus::Released;
    p->outcome        = ticks / 240.;
    p->release_origin = b.position;
    held              = 0;
    if (!safe)
      failed_attempt = true;
    move(frame.arm.joints, .05);
    frame.target = 0;
  }
};
FruitStreamEnvironment::FruitStreamEnvironment(const std::string& path, FruitStreamOptions options)
    : impl_(std::make_unique<Impl>(path, options)) {
  reset(0);
}
FruitStreamEnvironment::~FruitStreamEnvironment() = default;
StreamFrame FruitStreamEnvironment::reset(std::uint64_t seed) {
  return reset(seed, impl_->options.arrival_rate);
}
StreamFrame FruitStreamEnvironment::reset(std::uint64_t seed, double rate) {
  rate_ok(rate);
  auto& p = *impl_;
  p.workcell.reset(seed);
  auto initial = p.workcell.observation();
  p.workcell.world().remove_body(initial.fruit);
  p.payloads.clear();
  p.frame                = {};
  p.frame.arm.joints     = p.options.watch_infeed ? p.watch_joints : p.model.joint_target(0, 0, 4);
  p.frame.arm.finger     = .05;
  p.frame.arm.variation  = FruitVariation::Diverse;
  p.frame.arm.feed_stage = ConveyorStage::Travelling;
  p.commanded            = p.frame.arm.joints;
  p.commanded_finger     = p.frame.arm.finger;
  p.lost_contact_time    = 0;
  if (p.dynamics) {
    p.dynamics->reset(p.frame.arm.joints, p.frame.arm.finger);
    p.dynamics->observe(p.frame.arm);
  }
  p.seed         = seed;
  p.next_id      = 1;
  p.ticks        = 0;
  p.rate         = rate;
  p.held         = 0;
  p.frame.target = 0;
  p.watch_route.clear();
  p.watch_waypoint = 0;
  p.mask_tick      = std::numeric_limits<std::uint64_t>::max();
  p.observed_valid = false;
  for (auto& cache : p.plans)
    cache.valid = false;
  p.arrivals.seed(mix(seed ^ 0x4152524956414cULL));
  p.vision.seed(mix(seed ^ 0x564953494f4eULL));
  p.next_arrival = p.gap();
  auto f         = p.observe();
  if (p.recorder)
    p.recorder(f);
  return f;
}
StreamTransition FruitStreamEnvironment::step(StreamAction action) {
  int a = int(action);
  if (a < 0 || a > 4)
    throw std::invalid_argument("invalid stream action");
  auto&      p      = *impl_;
  const auto before = p.frame.counts;
  double     start  = p.ticks / 240.;
  p.failed_attempt  = false;
  auto ids          = p.queue();
  if (a == 0)
    p.wait(ids.empty());
  else {
    std::size_t index = a >= 3 ? 1 : 0;
    if (index >= ids.size()) {
      p.failed_attempt = true;
      p.wait(ids.empty());
    } else
      p.intercept(ids[index], a % 2 == 1 ? -2 : 2);
  }
  if (p.ticks / 240. - start < 1e-9)
    p.advance(1. / 240);
  double dt     = p.ticks / 240. - start;
  auto   after  = p.frame.counts;
  double reward = 5. * (after.correct - before.correct) -
                  p.options.wrong_cost * (after.wrong - before.wrong) -
                  p.options.miss_cost * (after.missed - before.missed) - .02 * dt -
                  (p.failed_attempt ? .25 : 0.) -
                  (a == 0 && !ids.empty() ? p.options.queued_wait_cost * dt : 0.);
  return {p.observe(), reward, dt};
}
StreamFrame FruitStreamEnvironment::observation() const {
  return impl_->observe();
}
std::array<bool, 5> FruitStreamEnvironment::action_mask() const {
  return impl_->action_mask();
}
rl::DiscreteObservation FruitStreamEnvironment::learning_observation() const {
  auto                    f    = impl_->observe();
  auto                    ids  = impl_->queue();
  auto                    mask = impl_->action_mask();
  rl::DiscreteObservation out;
  out.state = f.state;
  out.allowed.assign(mask.begin(), mask.end());
  /// @brief Append a normalized scalar with a finite bounded numerical range.
  /// @param value Normalized scalar.
  auto add = [&](double value) { out.features.push_back(float(std::clamp(value, -1., 1.))); };
  add(std::min<std::size_t>(3, ids.size()) / 3.);
  add(std::min<std::size_t>(
          3,
          std::count_if(
              f.fruit.begin(),
              f.fruit.end(),
              [](const auto& p) { return p.status == StreamFruitStatus::Released; })) /
      3.);
  for (double q : f.arm.joints)
    add(q / std::acos(-1.));
  add(f.arm.finger / .05);
  add(f.arm.tool.x / .62);
  add(f.arm.tool.y / .5);
  add((f.arm.tool.z - .72) / .6);
  for (std::size_t i = 0; i < 2; ++i) {
    if (i >= ids.size()) {
      for (int k = 0; k < 4; ++k)
        add(0);
      continue;
    }
    auto p =
        std::find_if(f.fruit.begin(), f.fruit.end(), [&](const auto& p) { return p.id == ids[i]; });
    add(1);
    add(p->body.position[0]);
    add(p->body.position[2] / .1);
    add((f.time - p->arrival) / 8);
  }
  for (bool allowed : mask)
    add(allowed ? 1 : 0);
  return out;
}
rl::DiscreteEnvironment
stream_learning_environment(FruitStreamEnvironment& env, double lo, double hi) {
  rate_ok(lo);
  rate_ok(hi);
  if (lo > hi)
    throw std::invalid_argument("inverted stream curriculum");
  return {
      [&env, lo, hi](std::uint64_t seed) {
        double u = double(mix(seed ^ 0x435552524943554cULL) >> 11) * 0x1.0p-53;
        env.reset(seed, lo + (hi - lo) * u);
        return env.learning_observation();
      },
      [&env](std::size_t action) {
        if (action > 4)
          throw std::invalid_argument("invalid stream learning action");
        auto t = env.step(StreamAction(action));
        return rl::DiscreteTransition{
            env.learning_observation(), t.reward, t.seconds, false, false};
      }};
}
void FruitStreamEnvironment::set_perception(
    std::function<StreamPrediction(const StreamFrame&, std::uint64_t)> classifier) {
  impl_->perception = std::move(classifier);
}
void FruitStreamEnvironment::record(std::function<void(const StreamFrame&)> recorder, int fps) {
  if (fps <= 0 || fps > 60 || 240 % fps)
    throw std::invalid_argument("stream fps must divide 240 and be <=60");
  if (recorder && impl_->options.training_proxy)
    throw std::logic_error("recording requires the full physics environment");
  impl_->stride   = 240 / fps;
  impl_->recorder = std::move(recorder);
}
DevArmPickEnvironment& FruitStreamEnvironment::arm_model() {
  return impl_->model;
}
cad::Scene FruitStreamEnvironment::fruit_scene(const StreamFrame& frame) const {
  ConveyorObservation empty;
  empty.variation   = FruitVariation::Diverse;
  empty.dimensions  = {.078, .078, .078};
  empty.rotation    = {1, 0, 0, 0};
  empty.stage       = ConveyorStage::Travelling;
  empty.position    = {0, -100, 0};
  empty.belt_travel = .22 * frame.time;
  auto scene        = impl_->workcell.dynamic_scene(empty);
  scene.objects.erase(
      std::remove_if(
          scene.objects.begin(),
          scene.objects.end(),
          [](const auto& o) {
            return o.name.starts_with("apple") || o.name.starts_with("banana");
          }),
      scene.objects.end());
  for (const auto& fruit : frame.fruit) {
    auto s        = fruit.body;
    s.belt_travel = .22 * frame.time;
    s.stage       = ConveyorStage::Travelling;
    auto actors   = impl_->workcell.dynamic_scene(s);
    actors.objects.erase(
        std::remove_if(
            actors.objects.begin(),
            actors.objects.end(),
            [](const auto& o) {
              return !(o.name.starts_with("apple") || o.name.starts_with("banana"));
            }),
        actors.objects.end());
    append(scene, std::move(actors));
  }
  return scene;
}
StreamAction StreamPolicy::action(std::size_t state) const {
  const auto& r = q.at(state);
  return StreamAction(std::max_element(r.begin(), r.end()) - r.begin());
}
StreamAction StreamPolicy::action(std::size_t state, const std::array<bool, 5>& allowed) const {
  const auto& row  = q.at(state);
  int         best = -1;
  for (int a = 0; a < 5; ++a)
    if (allowed[a] && (best < 0 || row[a] > row[best]))
      best = a;
  if (best < 0)
    throw std::invalid_argument("no feasible stream action");
  return StreamAction(best);
}
void StreamPolicy::save(const std::string& path) const {
  policy_ok(*this);
  std::ofstream f(path);
  f << "DATAMUNGE_FRUIT_STREAM_Q 1 " << q.size() << " 5\n" << std::setprecision(17);
  for (const auto& row : q) {
    for (double v : row)
      f << v << ' ';
    f << '\n';
  }
  f.close();
  if (!f)
    throw std::runtime_error("could not save stream policy");
}
StreamPolicy StreamPolicy::load(const std::string& path) {
  std::ifstream f(path);
  std::string   magic;
  int           version = 0, actions = 0;
  std::size_t   rows = 0;
  if (!(f >> magic >> version >> rows >> actions) || magic != "DATAMUNGE_FRUIT_STREAM_Q" ||
      version != 1 || rows != FruitStreamEnvironment::state_count() || actions != 5)
    throw std::runtime_error("invalid stream policy header");
  StreamPolicy p;
  p.q.resize(rows);
  for (auto& row : p.q)
    for (auto& v : row)
      if (!(f >> v) || !std::isfinite(v))
        throw std::runtime_error("invalid stream Q values");
  f >> std::ws;
  if (!f.eof())
    throw std::runtime_error("trailing stream checkpoint content");
  return p;
}
StreamTrainingResult train_fruit_stream(
    FruitStreamEnvironment& env,
    StreamTrainingOptions   options) {
  rate_ok(options.rate_min);
  rate_ok(options.rate_max);
  if (!options.rollouts || !std::isfinite(options.seconds) || options.seconds <= 0 ||
      options.rate_min > options.rate_max || !std::isfinite(options.learning_rate) ||
      options.learning_rate <= 0 || options.learning_rate > 1 ||
      !std::isfinite(options.discount_per_second) || options.discount_per_second <= 0 ||
      options.discount_per_second > 1 || !std::isfinite(options.epsilon_start) ||
      options.epsilon_start < 0 || options.epsilon_start > 1 ||
      !std::isfinite(options.epsilon_end) || options.epsilon_end < 0 || options.epsilon_end > 1 ||
      !std::isfinite(options.trace_decay) || options.trace_decay < 0 || options.trace_decay > 1)
    throw std::invalid_argument("invalid continuing Q-learning settings");
  env.record({});
  StreamTrainingResult result;
  result.policy.q.resize(FruitStreamEnvironment::state_count());
  std::mt19937_64                        random(options.seed);
  std::uniform_real_distribution<double> uniform(0, 1), rate(options.rate_min, options.rate_max);
  std::uniform_int_distribution<int>     action(0, 4);
  std::vector<double>                    eligibility(FruitStreamEnvironment::state_count() * 5, 0);
  std::vector<std::size_t>               active;
  for (std::size_t window = 0; window < options.rollouts; ++window) {
    auto   s       = env.reset(mix(options.seed + window), rate(random));
    double total   = 0;
    double epsilon = options.epsilon_start + (options.epsilon_end - options.epsilon_start) *
                                                 double(window) /
                                                 std::max<std::size_t>(1, options.rollouts - 1);
    for (auto i : active)
      eligibility[i] = 0;
    active.clear();
    int carried_action = -1;
    /// @brief Sample epsilon-greedy behavior using only the allowed actions.
    /// @param state Current scheduler row.
    /// @param allowed Owned current feasibility mask.
    /// @return Allowed action index, consuming the independent training RNG.
    auto choose = [&](std::size_t state, const std::array<bool, 5>& allowed) {
      int selected = int(result.policy.action(state, allowed));
      if (uniform(random) < epsilon) {
        if (options.mask_actions) {
          std::array<int, 5> legal{};
          int                count = 0;
          for (int k = 0; k < 5; ++k)
            if (allowed[k])
              legal[count++] = k;
          selected = legal[std::uniform_int_distribution<int>(0, count - 1)(random)];
        } else
          selected = action(random);
      }
      return selected;
    };
    while (s.time < options.seconds) {
      auto        allowed      = options.mask_actions ? env.action_mask()
                                                      : std::array<bool, 5>{true, true, true, true, true};
      int         a            = carried_action < 0 ? choose(s.state, allowed) : carried_action;
      auto        t            = env.step(StreamAction(a));
      auto&       q            = result.policy.q[s.state][a];
      const auto& next         = result.policy.q[t.frame.state];
      auto        next_allowed = options.mask_actions ? env.action_mask()
                                                      : std::array<bool, 5>{true, true, true, true, true};
      double      discount     = std::pow(options.discount_per_second, t.seconds);
      double      target =
          t.reward + discount * next[int(result.policy.action(t.frame.state, next_allowed))];
      bool continue_trace = true;
      if (options.trace_decay > 0) {
        // Select next behavior before updating Q, as in Watkins's Q(lambda).
        carried_action = choose(t.frame.state, next_allowed);
        continue_trace =
            next[carried_action] >= next[int(result.policy.action(t.frame.state, next_allowed))];
      }
      if (options.trace_decay == 0)
        q += options.learning_rate * (target - q);
      else {
        std::size_t current = s.state * 5 + a;
        if (eligibility[current] == 0)
          active.push_back(current);
        eligibility[current] = 1;
        const double delta   = options.learning_rate * (target - q);
        for (auto it = active.begin(); it != active.end();) {
          const auto index = *it;
          result.policy.q[index / 5][index % 5] += delta * eligibility[index];
          eligibility[index] *= continue_trace ? discount * options.trace_decay : 0;
          if (eligibility[index] < 1e-8) {
            eligibility[index] = 0;
            *it                = active.back();
            active.pop_back();
          } else
            ++it;
        }
      }
      total += t.reward;
      s = std::move(t.frame);
    }
    result.counts.push_back(s.counts);
    result.seconds.push_back(s.time);
    result.returns.push_back(total);
    const auto complete = window + 1;
    if ((complete >= 128 && (complete & (complete - 1)) == 0) || complete == options.rollouts) {
      result.snapshots.push_back(result.policy);
      result.snapshot_windows.push_back(complete);
    }
  }
  return result;
}
} // namespace datamunge::sim
