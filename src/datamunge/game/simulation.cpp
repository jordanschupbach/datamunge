/// @file simulation.cpp
/// @brief Native stateful behavior lifecycle and transactional scene/world reconciliation.
#include <algorithm>
#include <cmath>
#include <datamunge/game/simulation.hpp>
#include <stdexcept>

#include "text.hpp"
namespace datamunge::game {
/// @brief Translation-unit-local validation and construction helpers.
namespace {
/// @brief Compare physical settings, excluding mutable pose and velocities.
bool same_configuration(const sim::Body& a, const sim::Body& b) {
  return a.motion == b.motion && a.shape == b.shape && a.dimensions == b.dimensions &&
         a.mass == b.mass && a.friction == b.friction && a.restitution == b.restitution &&
         a.gravity_scale == b.gravity_scale && a.linear_damping == b.linear_damping &&
         a.angular_damping == b.angular_damping && a.sensor == b.sensor &&
         a.allowed_translation == b.allowed_translation && a.allowed_rotation == b.allowed_rotation;
}
/// @brief Compare complete physical state without padding-byte comparisons.
bool same_body(const sim::Body& a, const sim::Body& b) {
  return same_configuration(a, b) && a.position == b.position && a.rotation == b.rotation &&
         a.linear_velocity == b.linear_velocity && a.angular_velocity == b.angular_velocity;
}
/// @brief Compare authored constraint settings using scene identifiers.
bool same_joint(const std::optional<sim::Joint>& a, const std::optional<sim::Joint>& b) {
  if (bool(a) != bool(b))
    return false;
  if (!a)
    return true;
  return a->type == b->type && a->parent == b->parent && a->anchor == b->anchor &&
         a->axis == b->axis && a->limited == b->limited && a->limit_min == b->limit_min &&
         a->limit_max == b->limit_max && a->damping == b->damping;
}
/// @brief Scope a tick-recursion guard and restore it even after callback failure.
struct TickGuard {
  bool& active; ///< Borrowed simulation recursion flag.

  /// @brief Mark a tick active, rejecting recursion before changing the flag.
  /// @param value Runtime recursion flag.
  explicit TickGuard(bool& value) : active(value) {
    if (active)
      throw std::logic_error("recursive simulation tick");
    active = true;
  }
  /// @brief Clear the active tick flag on all exit paths.
  ~TickGuard() { active = false; }
};
} // namespace
void BehaviorRegistry::add(std::string name, Factory factory) {
  detail::validate_name(name);
  if (name.empty() || name.find('\0') != std::string::npos || !factory)
    throw std::invalid_argument("invalid native behavior factory");
  factories_.insert_or_assign(std::move(name), std::move(factory));
}
bool BehaviorRegistry::contains(const std::string& name) const noexcept {
  return factories_.find(name) != factories_.end();
}
std::unique_ptr<Behavior> BehaviorRegistry::create(const std::string& name) const {
  auto it = factories_.find(name);
  if (it == factories_.end())
    throw std::out_of_range("unknown native behavior: " + name);
  auto result = it->second();
  if (!result)
    throw std::runtime_error("native behavior factory returned null: " + name);
  return result;
}
/// @brief Owned scene, native world mapping and stateful behavior instances.
struct Simulation::Impl {
  /// @brief One independently owned behavior and whether initialization has completed.
  struct Instance {
    std::string               name;           ///< Registered factory name.
    std::unique_ptr<Behavior> behavior;       ///< Exclusive native callback object.
    bool                      started{false}; ///< Whether on_start has completed successfully.
  };
  Scene                       scene; ///< Authoritative owned components and current spatial state.
  BehaviorRegistry            registry;    ///< Explicit factory ownership.
  std::unique_ptr<sim::World> world;       ///< Native physics independent of behavior callbacks.
  std::map<EntityId, sim::BodyId> mapping; ///< Scene IDs to native body IDs.
  std::map<EntityId, std::shared_ptr<Instance>>
      instances; ///< Shared only during an active callback to survive self-destruction.
  std::vector<EntityId> visits;         ///< Reused ascending-ID tick snapshot.
  InputFrame            input;          ///< Validated controls for the current tick, never latched.
  bool                  ticking{false}; ///< Reject recursive tick submission.

  /// @brief Compile a starting scene and validate all behavior names.
  Impl(Scene source, BehaviorRegistry behaviors, sim::WorldOptions options)
      : scene(std::move(source)),
        registry(std::move(behaviors)),
        world(std::make_unique<sim::World>(options)) {
    visits.reserve(10000);
    for (auto id : scene.entities()) {
      const auto& e = scene.get(id);
      if (!e.behavior.empty() && !registry.contains(e.behavior))
        throw std::invalid_argument("unregistered native behavior: " + e.behavior);
      if (e.physics_enabled)
        mapping.emplace(id, world->add_body(e.spatial));
    }
    for (auto id : scene.entities())
      if (scene.get(id).joint) {
        auto j   = *scene.get(id).joint;
        j.child  = mapping.at(id);
        j.parent = j.parent ? mapping.at(j.parent) : 0;
        world->add_joint(j);
      }
  }
  /// @brief Resolve a physical entity to native body ownership.
  sim::BodyId body_id(EntityId id) const {
    scene.get(id);
    auto it = mapping.find(id);
    if (it == mapping.end())
      throw std::invalid_argument("entity has no rigid body");
    return it->second;
  }
  /// @brief Validate and commit a complete scene edit using a private world fork when physics
  /// changes.
  void replace(Scene candidate) {
    bool physics_changed = false;
    for (auto id : candidate.entities()) {
      const auto& e = candidate.get(id);
      if (!e.behavior.empty() && !registry.contains(e.behavior))
        throw std::invalid_argument("unregistered native behavior: " + e.behavior);
      if (!scene.alive(id)) {
        physics_changed |= e.physics_enabled;
        continue;
      }
      const auto& old = scene.get(id);
      physics_changed |= old.physics_enabled != e.physics_enabled ||
                         !same_joint(old.joint, e.joint) ||
                         (e.physics_enabled && !same_body(old.spatial, e.spatial));
    }
    for (auto id : scene.entities())
      if (!candidate.alive(id) && scene.get(id).physics_enabled)
        physics_changed = true;
    if (!physics_changed) {
      scene = std::move(candidate);
      return;
    }
    auto next         = world->fork();
    auto next_mapping = mapping;
    for (auto id : scene.entities()) {
      const auto& old = scene.get(id);
      if (old.joint && (!candidate.alive(id) || !same_joint(old.joint, candidate.get(id).joint) ||
                        !candidate.get(id).physics_enabled))
        next->remove_joint(next_mapping.at(id));
    }
    for (auto id : scene.entities())
      if (scene.get(id).physics_enabled &&
          (!candidate.alive(id) || !candidate.get(id).physics_enabled)) {
        next->remove_body(next_mapping.at(id));
        next_mapping.erase(id);
      }
    for (auto id : candidate.entities()) {
      const auto& e = candidate.get(id);
      if (!e.physics_enabled)
        continue;
      if (!next_mapping.count(id))
        next_mapping.emplace(id, next->add_body(e.spatial));
      else if (!same_body(next->body(next_mapping.at(id)), e.spatial))
        next->set_body(next_mapping.at(id), e.spatial);
    }
    for (auto id : candidate.entities()) {
      const auto& e = candidate.get(id);
      if (e.joint && (!scene.alive(id) || !same_joint(scene.get(id).joint, e.joint) ||
                      !scene.get(id).physics_enabled)) {
        auto j   = *e.joint;
        j.child  = next_mapping.at(id);
        j.parent = j.parent ? next_mapping.at(j.parent) : 0;
        next->add_joint(j);
      }
    }
    scene   = std::move(candidate);
    world   = std::move(next);
    mapping = std::move(next_mapping);
  }
};
Simulation::Simulation(Scene scene, BehaviorRegistry behaviors, sim::WorldOptions physics)
    : impl_(std::make_unique<Impl>(std::move(scene), std::move(behaviors), physics)) {}
Simulation::~Simulation() = default;
const Scene& Simulation::scene() const noexcept {
  return impl_->scene;
}
void Simulation::tick(double delta, const InputFrame& input) {
  if (!std::isfinite(delta) || delta <= 0 || delta > 1. / 30)
    throw std::invalid_argument("game delta must be in (0,1/30]");
  if (input.size() > 32)
    throw std::invalid_argument("at most 32 named inputs are supported");
  for (const auto& item : input)
    if (item.first.empty() || item.first.size() > 64 ||
        item.first.find('\0') != std::string::npos || !std::isfinite(item.second) ||
        item.second < -1 || item.second > 1)
      throw std::invalid_argument("invalid normalized input frame");
  TickGuard guard(impl_->ticking);
  impl_->input = input;
  impl_->visits.clear();
  for (const auto& item : impl_->scene.entities_)
    impl_->visits.push_back(item.first);
  for (auto id : impl_->visits) {
    if (!impl_->scene.alive(id))
      continue;
    auto name = impl_->scene.get(id).behavior;
    if (name.empty()) {
      impl_->instances.erase(id);
      continue;
    }
    auto it = impl_->instances.find(id);
    if (it == impl_->instances.end() || it->second->name != name) {
      auto instance      = std::make_shared<Impl::Instance>();
      instance->name     = name;
      instance->behavior = impl_->registry.create(name);
      impl_->instances.insert_or_assign(id, std::move(instance));
    }
    auto instance = impl_->instances.at(id);
    try {
      if (!instance->started) {
        instance->behavior->on_start(*this, id, delta);
        instance->started = true;
      }
      if (impl_->scene.alive(id))
        instance->behavior->on_update(*this, id, delta);
    } catch (const std::exception& e) {
      throw std::runtime_error("native behavior entity " + std::to_string(id) + ": " + e.what());
    }
  }
  for (auto it = impl_->instances.begin(); it != impl_->instances.end();)
    if (!impl_->scene.alive(it->first) || impl_->scene.get(it->first).behavior.empty())
      it = impl_->instances.erase(it);
    else
      ++it;
  impl_->world->step(delta);
  for (const auto& item : impl_->mapping)
    impl_->scene.entities_.at(item.first).spatial = impl_->world->body(item.second);
  impl_->scene.touch();
}
double Simulation::input(const std::string& name) const noexcept {
  auto it = impl_->input.find(name);
  return it == impl_->input.end() ? 0 : it->second;
}
EntityId Simulation::create(Entity entity) {
  Scene next = impl_->scene;
  auto  id   = next.create(std::move(entity));
  impl_->replace(std::move(next));
  return id;
}
void Simulation::destroy(EntityId id) {
  Scene next = impl_->scene;
  next.destroy(id);
  impl_->replace(std::move(next));
}
void Simulation::set(EntityId id, Entity entity) {
  Scene next = impl_->scene;
  next.set(id, std::move(entity));
  impl_->replace(std::move(next));
}
EntityId Simulation::spawn(const std::string& name) {
  Scene next = impl_->scene;
  auto  id   = next.spawn(name);
  impl_->replace(std::move(next));
  return id;
}
void Simulation::set_motion(EntityId id, const sim::Body& spatial) {
  const auto& e    = impl_->scene.get(id);
  auto        next = e;
  next.spatial     = spatial;
  next.validate();
  if (!same_configuration(e.spatial, spatial))
    throw std::invalid_argument("set_motion cannot change physical settings");
  if (e.physics_enabled)
    impl_->world->set_body(impl_->body_id(id), spatial);
  impl_->scene.entities_.at(id).spatial = spatial;
  impl_->scene.touch();
}
void Simulation::apply_impulse(EntityId id, const sim::Vec3& impulse) {
  auto body_id = impl_->body_id(id);
  impl_->world->apply_impulse(body_id, impulse);
  impl_->scene.entities_.at(id).spatial = impl_->world->body(body_id);
  impl_->scene.touch();
}
bool Simulation::touching(EntityId first, EntityId second) const {
  return impl_->world->touching(impl_->body_id(first), impl_->body_id(second));
}
bool Simulation::grounded(EntityId id) const {
  return impl_->world->grounded(impl_->body_id(id));
}
} // namespace datamunge::game
