/// @file io.cpp
/// @brief Native version-one scene codecs with strict bounded parsing and explicit script
/// replacement.
#include <array>
#include <datamunge/game/io.hpp>
#include <fstream>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>

#include "json.hpp"
namespace datamunge::game {
/// @brief Translation-unit-local schema conversion and deterministic serialization helpers.
namespace {
using detail::Json; ///< Owned bounded syntax-tree value.

/// @brief Require a string-valued schema member.
/// @param value JSON member.
/// @return Borrowed scalar UTF-8 text.
const std::string& text(const Json& value) {
  value.require(Json::Kind::String);
  return value.text;
}
/// @brief Require a boolean-valued schema member.
/// @param value JSON member.
/// @return Boolean payload.
bool boolean(const Json& value) {
  value.require(Json::Kind::Boolean);
  return value.boolean;
}
/// @brief Convert an optional numeric object member into an existing default.
/// @param value Schema object.
/// @param key Optional member name.
/// @param target Existing default overwritten only when present.
void number(const Json& value, const char* key, double& target) {
  if (const auto* v = value.find(key))
    target = v->number();
}
/// @brief Parse a fixed-width numeric array without partial target mutation.
/// @tparam N Exact component count.
/// @param value JSON numeric array.
/// @return Owned numeric components.
template<std::size_t N>
std::array<double, N> vector(const Json& value) {
  value.require(Json::Kind::Array);
  if (value.array.size() != N)
    throw std::invalid_argument("scene vector width mismatch");
  std::array<double, N> result{};
  for (std::size_t i = 0; i < N; ++i)
    result[i] = value.array[i].number();
  return result;
}
/// @brief Parse three explicit world-axis motion booleans.
/// @param value JSON boolean array of length three.
/// @return Validated axis mask.
std::array<bool, 3> mask(const Json& value) {
  value.require(Json::Kind::Array);
  if (value.array.size() != 3)
    throw std::invalid_argument("scene motion mask width mismatch");
  return {boolean(value.array[0]), boolean(value.array[1]), boolean(value.array[2])};
}
/// @brief Resolve a strict string enum from ordered alternatives.
/// @param value JSON string enum token.
/// @param alternatives Supported exact strings.
/// @return Zero-based enum alternative.
std::size_t choice(const Json& value, std::initializer_list<std::string_view> alternatives) {
  std::size_t i    = 0;
  const auto& name = text(value);
  for (auto candidate : alternatives) {
    if (candidate == name)
      return i;
    ++i;
  }
  throw std::invalid_argument("unknown scene enum: " + name);
}
/// @brief Decode owned components and require an explicit replacement for nonempty Lua source.
/// @param value Component object.
/// @param replacement Optional explicit native factory name for this entity/prefab.
/// @return Validated standalone components; graph references validate after all entities load.
Entity entity(const Json& value, const std::string* replacement) {
  value.keys({"name", "transform", "camera", "mesh", "rigid_body", "joint", "script", "behavior"});
  Entity e;
  if (const auto* v = value.find("name"))
    e.name = text(*v);
  if (const auto* v = value.find("behavior"))
    e.behavior = text(*v);
  if (const auto* t = value.find("transform")) {
    t->keys({"position", "rotation", "scale"});
    if (const auto* v = t->find("position"))
      e.spatial.position = vector<3>(*v);
    if (const auto* v = t->find("rotation"))
      e.spatial.rotation = vector<4>(*v);
    if (const auto* v = t->find("scale"))
      e.spatial.dimensions = vector<3>(*v);
  }
  if (const auto* c = value.find("camera")) {
    c->keys({"vertical_fov", "near_plane", "far_plane"});
    e.camera = Camera{};
    number(*c, "vertical_fov", e.camera->vertical_fov);
    number(*c, "near_plane", e.camera->near_plane);
    number(*c, "far_plane", e.camera->far_plane);
  }
  if (const auto* m = value.find("mesh")) {
    m->keys({"primitive", "color", "metallic", "roughness", "emissive"});
    e.mesh = Mesh{};
    if (const auto* v = m->find("primitive"))
      e.mesh->primitive = static_cast<Primitive>(choice(*v, {"box", "sphere", "crystal"}));
    if (const auto* v = m->find("color"))
      e.mesh->color = vector<3>(*v);
    number(*m, "metallic", e.mesh->metallic);
    number(*m, "roughness", e.mesh->roughness);
    number(*m, "emissive", e.mesh->emissive);
  }
  if (const auto* b = value.find("rigid_body")) {
    b->keys(
        {"motion",
         "shape",
         "mass",
         "friction",
         "restitution",
         "gravity_scale",
         "sensor",
         "lock_rotation",
         "allowed_translation",
         "allowed_rotation",
         "linear_velocity",
         "angular_velocity",
         "linear_damping",
         "angular_damping"});
    e.physics_enabled = true;
    auto& body        = e.spatial;
    if (const auto* v = b->find("motion"))
      body.motion = static_cast<sim::Motion>(choice(*v, {"static", "dynamic", "kinematic"}));
    if (const auto* v = b->find("shape"))
      body.shape = static_cast<sim::Shape>(choice(*v, {"box", "sphere", "capsule"}));
    number(*b, "mass", body.mass);
    number(*b, "friction", body.friction);
    number(*b, "restitution", body.restitution);
    number(*b, "gravity_scale", body.gravity_scale);
    number(*b, "linear_damping", body.linear_damping);
    number(*b, "angular_damping", body.angular_damping);
    if (const auto* v = b->find("sensor"))
      body.sensor = boolean(*v);
    if (const auto* v = b->find("allowed_translation"))
      body.allowed_translation = mask(*v);
    if (const auto* v = b->find("allowed_rotation"))
      body.allowed_rotation = mask(*v);
    if (const auto* v = b->find("lock_rotation"))
      if (boolean(*v))
        body.allowed_rotation = {false, false, false};
    if (const auto* v = b->find("linear_velocity"))
      body.linear_velocity = vector<3>(*v);
    if (const auto* v = b->find("angular_velocity"))
      body.angular_velocity = vector<3>(*v);
    auto& d = body.dimensions;
    if (body.shape == sim::Shape::Sphere && std::abs(d[1] - d[0]) <= 1e-5 &&
        std::abs(d[2] - d[0]) <= 1e-5)
      d[1] = d[2] = d[0];
    if (body.shape == sim::Shape::Capsule && std::abs(d[2] - d[0]) <= 1e-5)
      d[2] = d[0];
  }
  if (const auto* j = value.find("joint")) {
    j->keys({"type", "parent", "anchor", "axis", "limited", "limit_min", "limit_max", "damping"});
    e.joint = sim::Joint{};
    if (const auto* v = j->find("type"))
      e.joint->type = static_cast<sim::JointType>(choice(*v, {"hinge", "slider"}));
    if (const auto* v = j->find("parent"))
      e.joint->parent = v->integer();
    if (const auto* v = j->find("anchor"))
      e.joint->anchor = vector<3>(*v);
    if (const auto* v = j->find("axis"))
      e.joint->axis = vector<3>(*v);
    if (const auto* v = j->find("limited"))
      e.joint->limited = boolean(*v);
    number(*j, "limit_min", e.joint->limit_min);
    number(*j, "limit_max", e.joint->limit_max);
    number(*j, "damping", e.joint->damping);
  }
  bool scripted = false;
  if (const auto* script = value.find("script")) {
    script->keys({"source"});
    const auto& source = text(script->at("source"));
    if (source.size() > 1024 * 1024)
      throw std::length_error("legacy script exceeds 1 MiB");
    scripted = !source.empty();
  }
  if (scripted) {
    if (!replacement || replacement->empty() || !e.behavior.empty())
      throw std::invalid_argument("legacy script requires one explicit native replacement");
    e.behavior = *replacement;
  } else if (replacement)
    throw std::invalid_argument("unused legacy behavior replacement");
  e.validate();
  return e;
}
/// @brief Emit one finite scalar with deterministic round-trip precision.
/// @param out Locale-independent output stream.
/// @param value Finite scalar.
void scalar(std::ostream& out, double value) {
  if (!std::isfinite(value))
    throw std::invalid_argument("cannot encode nonfinite scene value");
  out << value;
}
/// @brief Emit one fixed-width numeric component vector.
/// @tparam N Exact component count.
/// @param out Output stream.
/// @param value Components to encode.
template<std::size_t N>
void array(std::ostream& out, const std::array<double, N>& value) {
  out << '[';
  for (std::size_t i = 0; i < N; ++i) {
    if (i)
      out << ',';
    scalar(out, value[i]);
  }
  out << ']';
}
/// @brief Emit one three-axis boolean motion mask.
/// @param out Output stream.
/// @param value Motion mask.
void axes(std::ostream& out, const std::array<bool, 3>& value) {
  out << '[';
  for (int i = 0; i < 3; ++i) {
    if (i)
      out << ',';
    out << (value[i] ? "true" : "false");
  }
  out << ']';
}
/// @brief Serialize validated components using stable schema key order.
/// @param out Output stream.
/// @param e Valid native entity or prefab.
void emit(std::ostream& out, const Entity& e) {
  e.validate();
  const auto& b = e.spatial;
  out << "{\"name\":" << detail::quote(e.name) << ",\"transform\":{\"position\":";
  array(out, b.position);
  out << ",\"rotation\":";
  array(out, b.rotation);
  out << ",\"scale\":";
  array(out, b.dimensions);
  out << '}';
  if (!e.behavior.empty())
    out << ",\"behavior\":" << detail::quote(e.behavior);
  if (e.camera) {
    const auto& c = *e.camera;
    out << ",\"camera\":{\"vertical_fov\":" << c.vertical_fov << ",\"near_plane\":" << c.near_plane
        << ",\"far_plane\":" << c.far_plane << '}';
  }
  if (e.mesh) {
    const auto& m         = *e.mesh;
    const char* primitive = m.primitive == Primitive::Box      ? "box"
                            : m.primitive == Primitive::Sphere ? "sphere"
                                                               : "crystal";
    out << ",\"mesh\":{\"primitive\":" << detail::quote(primitive) << ",\"color\":";
    array(out, m.color);
    out << ",\"metallic\":" << m.metallic << ",\"roughness\":" << m.roughness
        << ",\"emissive\":" << m.emissive << '}';
  }
  if (e.physics_enabled) {
    const char* motion = b.motion == sim::Motion::Static    ? "static"
                         : b.motion == sim::Motion::Dynamic ? "dynamic"
                                                            : "kinematic";
    const char* shape  = b.shape == sim::Shape::Box      ? "box"
                         : b.shape == sim::Shape::Sphere ? "sphere"
                                                         : "capsule";
    out << ",\"rigid_body\":{\"motion\":" << detail::quote(motion)
        << ",\"shape\":" << detail::quote(shape) << ",\"mass\":" << b.mass
        << ",\"friction\":" << b.friction << ",\"restitution\":" << b.restitution
        << ",\"gravity_scale\":" << b.gravity_scale << ",\"linear_damping\":" << b.linear_damping
        << ",\"angular_damping\":" << b.angular_damping
        << ",\"sensor\":" << (b.sensor ? "true" : "false") << ",\"allowed_translation\":";
    axes(out, b.allowed_translation);
    out << ",\"allowed_rotation\":";
    axes(out, b.allowed_rotation);
    out << ",\"linear_velocity\":";
    array(out, b.linear_velocity);
    out << ",\"angular_velocity\":";
    array(out, b.angular_velocity);
    out << '}';
  }
  if (e.joint) {
    const auto& j = *e.joint;
    out << ",\"joint\":{\"type\":"
        << detail::quote(j.type == sim::JointType::Hinge ? "hinge" : "slider")
        << ",\"parent\":" << j.parent << ",\"anchor\":";
    array(out, j.anchor);
    out << ",\"axis\":";
    array(out, j.axis);
    out << ",\"limited\":" << (j.limited ? "true" : "false") << ",\"limit_min\":" << j.limit_min
        << ",\"limit_max\":" << j.limit_max << ",\"damping\":" << j.damping << '}';
  }
  out << '}';
}
} // namespace
Scene parse_scene(const std::string& document, const SceneImport& mapping) {
  const auto root = detail::JsonParser(document).parse();
  root.keys({"version", "next_id", "entities", "prefabs"});
  if (root.at("version").integer() != 1)
    throw std::invalid_argument("unsupported scene version");
  const auto& items = root.at("entities");
  items.require(Json::Kind::Array);
  if (items.array.size() > 10000)
    throw std::length_error("scene entity limit exceeded");
  Scene          scene;
  EntityId       highest = 0;
  std::size_t    used    = 0;
  constexpr auto max     = std::uint64_t(std::numeric_limits<std::int64_t>::max());
  for (const auto& item : items.array) {
    item.keys({"id", "components"});
    const auto id = item.at("id").integer();
    if (!id || id >= max)
      throw std::invalid_argument("invalid scene entity ID");
    const auto replacement = mapping.entities.find(id);
    if (replacement != mapping.entities.end())
      ++used;
    auto e = entity(
        item.at("components"),
        replacement == mapping.entities.end() ? nullptr : &replacement->second);
    if (e.joint)
      e.joint->child = id;
    if (!scene.entities_.emplace(id, std::move(e)).second)
      throw std::invalid_argument("duplicate scene entity ID");
    highest = std::max(highest, id);
  }
  if (used != mapping.entities.size())
    throw std::invalid_argument("legacy mapping names an unknown entity ID");
  scene.next_id_ = root.find("next_id") ? root.at("next_id").integer() : highest + 1;
  if (scene.next_id_ <= highest || scene.next_id_ > max)
    throw std::invalid_argument("next_id must exceed live IDs and fit signed 64-bit range");
  for (const auto& item : scene.entities_)
    scene.validate_graph(item.first, item.second);
  std::size_t used_prefabs = 0;
  if (const auto* prefabs = root.find("prefabs")) {
    prefabs->require(Json::Kind::Object);
    if (prefabs->object.size() > 10000)
      throw std::length_error("scene prefab limit exceeded");
    for (const auto& item : prefabs->object) {
      const auto replacement = mapping.prefabs.find(item.first);
      if (replacement != mapping.prefabs.end())
        ++used_prefabs;
      scene.define_prefab(
          item.first,
          entity(
              item.second, replacement == mapping.prefabs.end() ? nullptr : &replacement->second));
    }
  }
  if (used_prefabs != mapping.prefabs.size())
    throw std::invalid_argument("legacy mapping names an unknown prefab");
  return scene;
}
std::string encode_scene(const Scene& scene) {
  std::ostringstream out;
  out.imbue(std::locale::classic());
  out << std::setprecision(std::numeric_limits<double>::max_digits10);
  out << "{\"version\":1,\"next_id\":" << scene.next_id_ << ",\"entities\":[";
  bool first = true;
  for (const auto& item : scene.entities_) {
    if (!first)
      out << ',';
    first = false;
    out << "{\"id\":" << item.first << ",\"components\":";
    emit(out, item.second);
    out << '}';
    if(out.tellp()>16*1024*1024)throw std::length_error("encoded scene exceeds 16 MiB");
  }
  out << "],\"prefabs\":{";
  first = true;
  for (const auto& item : scene.prefabs_) {
    if (!first)
      out << ',';
    first = false;
    out << detail::quote(item.first) << ':';
    emit(out, item.second);
    if(out.tellp()>16*1024*1024)throw std::length_error("encoded scene exceeds 16 MiB");
  }
  out << "}}\n";
  auto result = out.str();
  if (result.size() > 16 * 1024 * 1024)
    throw std::length_error("encoded scene exceeds 16 MiB");
  return result;
}
Scene load_scene(const std::string& path, const SceneImport& mapping) {
  if (path.empty() || path.find('\0') != std::string::npos)
    throw std::invalid_argument("invalid scene path");
  std::ifstream input(path, std::ios::binary);
  if (!input)
    throw std::runtime_error("cannot open scene: " + path);
  std::string            document;
  std::array<char, 8192> buffer{};
  while (input.read(buffer.data(), buffer.size()) || input.gcount()) {
    if (std::size_t(input.gcount()) > 16 * 1024 * 1024 - document.size())
      throw std::length_error("scene file exceeds 16 MiB");
    document.append(buffer.data(), std::size_t(input.gcount()));
  }
  if (input.bad())
    throw std::runtime_error("cannot read scene: " + path);
  return parse_scene(document, mapping);
}
} // namespace datamunge::game
