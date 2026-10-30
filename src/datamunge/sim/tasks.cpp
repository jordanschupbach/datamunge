/// @file tasks.cpp
/// @brief Native articulated control task definitions and batched rollout transitions.
#include <algorithm>
#include <cmath>
#include <datamunge/sim/tasks.hpp>
#include <stdexcept>
namespace datamunge::sim {
/// @brief Translation-unit-local validation and construction helpers.
namespace {
/// @brief Configure bounded capacities for small articulated task worlds.
/// @return Explicit small-world solver controls.
WorldOptions limits() {
  WorldOptions o;
  o.body_capacity     = 8;
  o.pair_capacity     = 32;
  o.contact_capacity  = 64;
  o.solver_iterations = 24;
  return o;
}
/// @brief Construct one validated planar body with rotation about Z.
/// @param position World centre of mass.
/// @param dimensions Primitive full dimensions.
/// @param mass Positive dynamic mass.
/// @return Validated dynamic box body.
Body planar(Vec3 position, Vec3 dimensions, double mass) {
  Body b;
  b.motion              = Motion::Dynamic;
  b.position            = position;
  b.dimensions          = dimensions;
  b.mass                = mass;
  b.allowed_translation = {true, true, false};
  b.allowed_rotation    = {false, false, true};
  b.linear_damping      = 0;
  b.angular_damping     = 0;
  b.restitution         = 0;
  return b;
}
/// @brief Build authored native task geometry and joint zero frames.
/// @param task Supported task.
/// @return Owned prototype whose IDs are stable across forks.
std::unique_ptr<World> prototype(Task task) {
  auto w = std::make_unique<World>(limits());
  if (task == Task::InvertedPendulum) {
    auto cart                = planar({0, 0, 0}, {.4, .2, .2}, 1);
    cart.allowed_translation = {true, false, false};
    cart.allowed_rotation    = {false, false, false};
    auto  cart_id            = w->add_body(cart);
    Joint rail;
    rail.type  = JointType::Slider;
    rail.child = cart_id;
    rail.axis  = {1, 0, 0};
    w->add_joint(rail);
    auto  pole    = planar({0, .6, 0}, {.06, 1.2, .06}, .1);
    auto  pole_id = w->add_body(pole);
    Joint hinge;
    hinge.child  = pole_id;
    hinge.parent = cart_id;
    hinge.anchor = {0, -.6, 0};
    w->add_joint(hinge);
  } else {
    auto torso  = planar({0, 1.335, 0}, {.2, .6, .2}, 2);
    torso.shape = Shape::Capsule;
    w->add_body(torso);
    auto thigh  = planar({0, .81, 0}, {.12, .45, .12}, 1);
    thigh.shape = Shape::Capsule;
    w->add_body(thigh);
    auto shin  = planar({0, .36, 0}, {.12, .45, .12}, .8);
    shin.shape = Shape::Capsule;
    w->add_body(shin);
    w->add_body(planar({.08, .075, 0}, {.4, .15, .15}, .5));
    for (BodyId id = 2; id <= 4; ++id) {
      Joint j;
      j.child     = id;
      j.parent    = id - 1;
      j.anchor    = id == 4 ? Vec3{-.08, .06, 0} : Vec3{0, .225, 0};
      j.limited   = true;
      j.limit_min = id == 3 ? -2.5 : -1.5;
      j.limit_max = id == 3 ? .1 : 1.5;
      j.damping   = .1;
      w->add_joint(j);
    }
    Body floor;
    floor.dimensions  = {1000, .2, 10};
    floor.position    = {0, -.1, 0};
    floor.friction    = .8;
    floor.restitution = 0;
    w->add_body(floor);
  }
  return w;
}
/// @brief Read planar Z rotation in radians, ignoring quaternion sign.
/// @param body Native planar body.
/// @return Principal signed angle.
double pitch(const Body& body) {
  return std::remainder(2 * std::atan2(body.rotation[3], body.rotation[0]), 2 * std::acos(-1.));
}
} // namespace
TaskBatch::TaskBatch(
    Task             task,
    std::size_t      count,
    compute::Context context,
    std::uint64_t    seed,
    std::uint64_t    time_limit)
    : task_(task), time_limit_(time_limit) {
  if ((task != Task::InvertedPendulum && task != Task::Hopper) || !time_limit)
    throw std::invalid_argument("invalid native task or time limit");
  auto p  = prototype(task);
  worlds_ = std::make_unique<BatchWorld>(*p, count, std::move(context), seed);
  if (count > output_.observations.max_size() / observation_size())
    throw std::length_error("task output size overflow");
  output_.observations.resize(count * observation_size());
  output_.final_observations.resize(count * observation_size());
  output_.rewards.resize(count);
  output_.terminated.resize(count);
  output_.truncated.resize(count);
  output_.final_valid.resize(count);
  reset();
}
void TaskBatch::validate_mask(const std::vector<std::uint8_t>& mask) const {
  if ((!mask.empty() && mask.size() != size()) ||
      std::any_of(mask.begin(), mask.end(), [](auto v) { return v > 1; }))
    throw std::invalid_argument("task mask must be empty or one binary byte per environment");
}
void TaskBatch::observe(std::size_t index) {
  const auto& w    = worlds_->world(index);
  auto*       out  = output_.observations.data() + index * observation_size();
  const auto  root = w.body(1);
  if (task_ == Task::InvertedPendulum) {
    const auto pole = w.joint_state(2);
    out[0]          = root.position[0];
    out[1]          = pole.position;
    out[2]          = root.linear_velocity[0];
    out[3]          = pole.velocity;
  } else {
    out[0] = root.position[1];
    out[1] = pitch(root);
    for (int j = 0; j < 3; ++j) {
      const auto q = w.joint_state(j + 2);
      out[2 + j]   = q.position;
      out[8 + j]   = q.velocity;
    }
    out[5] = root.linear_velocity[0];
    out[6] = root.linear_velocity[1];
    out[7] = root.angular_velocity[2];
  }
}
const TaskOutput& TaskBatch::reset(const std::vector<std::uint8_t>& mask) {
  validate_mask(mask);
  worlds_->reset(mask);
  for (std::size_t i = 0; i < size(); ++i)
    if (mask.empty() || mask[i]) {
      auto& w = worlds_->world(i);
      if (task_ == Task::InvertedPendulum) {
        const double x = .02 * worlds_->uniform(i) - .01, v = .02 * worlds_->uniform(i) - .01;
        w.set_joint_state(1, x, v);
        const double angle = .02 * worlds_->uniform(i) - .01,
                     rate  = .02 * worlds_->uniform(i) - .01;
        w.set_joint_state(2, angle, rate);
      } else
        for (BodyId j = 2; j <= 4; ++j) {
          const double angle = .01 * worlds_->uniform(i) - .005,
                       rate  = .01 * worlds_->uniform(i) - .005;
          w.set_joint_state(j, angle, rate);
        }
      output_.rewards[i]    = 0;
      output_.terminated[i] = 0;
      output_.truncated[i]  = 0;
      observe(i);
    }
  return output_;
}
const TaskOutput& TaskBatch::step(
    const std::vector<double>&       actions,
    const std::vector<std::uint8_t>& mask) {
  validate_mask(mask);
  if (actions.size() != size() * action_size())
    throw std::invalid_argument("native task action shape mismatch");
  for (std::size_t i = 0; i < size(); ++i)
    if (mask.empty() || mask[i]) {
      if (output_.terminated[i] || output_.truncated[i])
        throw std::logic_error("reset completed task before stepping");
      for (std::size_t j = 0; j < action_size(); ++j)
        if (!std::isfinite(actions[i * action_size() + j]) ||
            std::abs(actions[i * action_size() + j]) > (task_ == Task::InvertedPendulum ? 3 : 1))
          throw std::invalid_argument("native task action outside finite action box");
    }
  for (std::size_t i = 0; i < size(); ++i)
    if (mask.empty() || mask[i]) {
      auto& w = worlds_->world(i);
      if (task_ == Task::InvertedPendulum)
        w.apply_joint_force(1, 100 * actions[i]);
      else
        for (std::size_t j = 0; j < 3; ++j)
          w.apply_joint_force(j + 2, 200 * actions[i * 3 + j]);
    }
  worlds_->step(task_ == Task::InvertedPendulum ? .02 : .032, mask);
  for (std::size_t i = 0; i < size(); ++i)
    if (mask.empty() || mask[i]) {
      observe(i);
      const auto* row = output_.observations.data() + i * observation_size();
      bool        terminal;
      double      reward = 1;
      if (task_ == Task::InvertedPendulum)
        terminal = std::abs(row[0]) > 2.4 || std::abs(row[1]) > .2;
      else {
        terminal = row[0] < .65 || std::abs(row[1]) > .5;
        reward += row[5];
        for (int j = 0; j < 3; ++j)
          reward -= .001 * actions[i * 3 + j] * actions[i * 3 + j];
      }
      output_.rewards[i]    = reward;
      output_.terminated[i] = terminal;
      output_.truncated[i]  = !terminal && worlds_->steps(i) >= time_limit_;
      if (output_.terminated[i] || output_.truncated[i]) {
        std::copy_n(
            row, observation_size(), output_.final_observations.data() + i * observation_size());
        output_.final_valid[i] = 1;
      }
    }
  return output_;
}
} // namespace datamunge::sim
