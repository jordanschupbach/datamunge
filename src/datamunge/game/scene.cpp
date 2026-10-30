/// @file scene.cpp
/// @brief Native component validation, parent-cycle checks and transactional scene edits.
#include <atomic>
#include <cmath>
#include <datamunge/game/scene.hpp>
#include <limits>
#include <stdexcept>

#include "text.hpp"
namespace datamunge::game {
/// @brief Translation-unit-local validation and construction helpers.
namespace {
/// @brief Process-wide unique revision source; scene contents themselves remain independent.
std::atomic<std::uint64_t> next_revision{1};
/// @brief Require a finite scalar in a closed interval.
void range(double value, double lo, double hi) {
  if (!std::isfinite(value) || value < lo || value > hi)
    throw std::invalid_argument("invalid game component value");
}
} // namespace
void Mesh::validate() const {
  if (primitive != Primitive::Box && primitive != Primitive::Sphere &&
      primitive != Primitive::Crystal)
    throw std::invalid_argument("unknown mesh primitive");
  for (double v : color)
    range(v, 0, 1);
  range(metallic, 0, 1);
  range(roughness, .04, 1);
  range(emissive, 0, 10);
}
void Camera::validate() const {
  if (!std::isfinite(vertical_fov) || vertical_fov <= 0 || vertical_fov >= 180 ||
      !std::isfinite(near_plane) || near_plane <= 0 || !std::isfinite(far_plane) ||
      far_plane <= near_plane)
    throw std::invalid_argument("invalid camera projection");
}
void Entity::validate() const {
  detail::validate_name(name);
  detail::validate_name(behavior, true);
  if (physics_enabled)
    spatial.validate();
  else {
    for (double p : spatial.position)
      range(p, -100000, 100000);
    for (double v : spatial.dimensions)
      range(v, 1e-6, 1000000);
    double qnorm = 0;
    for (double q : spatial.rotation) {
      range(q, -1, 1);
      qnorm += q * q;
    }
    if (std::abs(qnorm - 1) > 1e-6)
      throw std::invalid_argument("entity quaternion must be unit length");
  }
  if (mesh)
    mesh->validate();
  if (camera)
    camera->validate();
  if (joint && !physics_enabled)
    throw std::invalid_argument("joint requires a rigid body");
}
Scene::Scene() {
  touch();
}
void Scene::touch() noexcept {
  revision_ = next_revision.fetch_add(1, std::memory_order_relaxed);
}
bool Scene::alive(EntityId id) const noexcept {
  return entities_.find(id) != entities_.end();
}
const Entity& Scene::get(EntityId id) const {
  auto it = entities_.find(id);
  if (it == entities_.end())
    throw std::out_of_range("unknown scene entity");
  return it->second;
}
void Scene::validate_graph(EntityId id, const Entity& proposed) const {
  proposed.validate();
  /// @brief Resolve one proposed graph node, rejecting missing parents before mutation.
  auto lookup = [&](EntityId candidate) -> const Entity& {
    if (candidate == id)
      return proposed;
    const auto it = entities_.find(candidate);
    if (it == entities_.end())
      throw std::invalid_argument("joint references a missing scene parent");
    return it->second;
  };
  if (proposed.joint) {
    auto joint  = *proposed.joint;
    joint.child = id;
    joint.validate();
    if (proposed.spatial.motion != sim::Motion::Dynamic)
      throw std::invalid_argument("joint child must be dynamic");
  }
  for (const auto& item : entities_) {
    const auto& e = lookup(item.first);
    if (e.joint && e.joint->parent) {
      const auto& p = lookup(e.joint->parent);
      if (!p.physics_enabled)
        throw std::invalid_argument("joint parent needs a rigid body");
    }
  }
  EntityId ancestor = proposed.joint ? proposed.joint->parent : 0;
  for (std::size_t n = 0; ancestor; ++n) {
    if (ancestor == id || n > entities_.size())
      throw std::invalid_argument("joint parent cycle");
    const auto& p = lookup(ancestor);
    if (!p.physics_enabled)
      throw std::invalid_argument("joint parent needs a rigid body");
    ancestor = p.joint ? p.joint->parent : 0;
  }
}
EntityId Scene::create(Entity entity) {
  if (entities_.size() == 10000)
    throw std::length_error("scene entity capacity exhausted");
  if (next_id_ >= std::uint64_t(std::numeric_limits<std::int64_t>::max()))
    throw std::overflow_error("scene identifiers exhausted");
  validate_graph(next_id_, entity);
  if (entity.joint)
    entity.joint->child = next_id_;
  auto id = next_id_;
  entities_.emplace(id, std::move(entity));
  ++next_id_;
  touch();
  return id;
}
void Scene::destroy(EntityId id) {
  get(id);
  for (const auto& item : entities_)
    if (item.first != id && item.second.joint && item.second.joint->parent == id)
      throw std::invalid_argument("entity remains a joint parent");
  entities_.erase(id);
  touch();
}
void Scene::set(EntityId id, Entity entity) {
  get(id);
  validate_graph(id, entity);
  if (entity.joint)
    entity.joint->child = id;
  entities_.at(id) = std::move(entity);
  touch();
}
std::vector<EntityId> Scene::entities() const {
  std::vector<EntityId> ids;
  ids.reserve(entities_.size());
  for (const auto& item : entities_)
    ids.push_back(item.first);
  return ids;
}
void Scene::define_prefab(std::string name, Entity entity) {
  entity.validate();
  detail::validate_name(name);
  if (prefabs_.find(name) == prefabs_.end() && prefabs_.size() >= 10000)
    throw std::length_error("scene prefab capacity exhausted");
  if (name.empty() || name.find('\0') != std::string::npos || entity.joint)
    throw std::invalid_argument("invalid prefab name or snapshot-scoped joint");
  prefabs_.insert_or_assign(std::move(name), std::move(entity));
  touch();
}
EntityId Scene::spawn(const std::string& name) {
  auto it = prefabs_.find(name);
  if (it == prefabs_.end())
    throw std::out_of_range("unknown scene prefab");
  return create(it->second);
}
} // namespace datamunge::game
