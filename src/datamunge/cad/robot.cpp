#include <algorithm>
#include <cmath>
#include <datamunge/cad/robot.hpp>
#include <datamunge/cad/solid.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>
#include <stdexcept>

#include "detail/xml.hpp"

namespace datamunge::cad {
namespace {
using detail::xml::xml_node;
void ensure(bool b, const std::string& error) {
  if (!b)
    throw std::invalid_argument("URDF: " + error);
}
std::vector<double> numbers(const std::string& text, std::size_t count) {
  std::istringstream  s(text);
  std::vector<double> out(count);
  for (auto& v : out)
    ensure(bool(s >> v) && std::isfinite(v), "invalid numeric vector: " + text);
  std::string extra;
  ensure(!(s >> extra), "excess numeric components: " + text);
  return out;
}
double number(const std::string& text) {
  return numbers(text, 1).front();
}
Vec3d vec(const std::string& text) {
  auto v = numbers(text, 3);
  return {v[0], v[1], v[2]};
}
Mat4d pose(xml_node node) {
  if (!node)
    return {};
  auto p = vec(node.attribute("xyz").as_string("0 0 0"));
  auto r = vec(node.attribute("rpy").as_string("0 0 0"));
  return Mat4d::Translation(p) * Mat4d::Rotation({0, 0, 1}, r.z) * Mat4d::Rotation({0, 1, 0}, r.y) *
         Mat4d::Rotation({1, 0, 0}, r.x);
}
Mat4d object_pose(const Object3D& o) {
  return Mat4d::Translation({o.position.x, o.position.y, o.position.z}) *
         Mat4d::Rotation({0, 0, 1}, DegToRad(o.rotation_deg.z)) *
         Mat4d::Rotation({0, 1, 0}, DegToRad(o.rotation_deg.y)) *
         Mat4d::Rotation({1, 0, 0}, DegToRad(o.rotation_deg.x)) *
         Mat4d::Scaling({o.scale.x, o.scale.y, o.scale.z});
}
std::filesystem::path
asset(const std::string& name, const std::filesystem::path& base, const std::string& package_root) {
  ensure(!name.empty(), "empty asset filename");
  if (name.starts_with("package://")) {
    auto part  = name.substr(10);
    auto slash = part.find('/');
    ensure(slash != std::string::npos, "invalid package URI");
    auto package = part.substr(0, slash), relative = part.substr(slash + 1);
    if (!package_root.empty())
      return std::filesystem::path(package_root) / package / relative;
    for (auto p = base; !p.empty(); p = p.parent_path()) {
      if (p.filename() == package)
        return p / relative;
      if (std::filesystem::is_directory(p / package))
        return p / package / relative;
      if (p == p.parent_path())
        break;
    }
    throw std::invalid_argument("URDF: unresolved package URI " + name);
  }
  if (name.starts_with("file://"))
    return name.substr(7);
  auto p = std::filesystem::path(name);
  return p.is_absolute() ? p : base / p;
}
struct Material {
  RgbaColorF  color{.8f, .8f, .8f, 1};
  std::string texture;
};
Material material(xml_node n, Material fallback = {}) {
  if (auto c = n.child("color")) {
    auto v = numbers(c.attribute("rgba").as_string(), 4);
    for (double x : v)
      ensure(x >= 0 && x <= 1, "material color outside [0,1]");
    fallback.color = {float(v[0]), float(v[1]), float(v[2]), float(v[3])};
  }
  if (auto t = n.child("texture"))
    fallback.texture = t.attribute("filename").as_string();
  return fallback;
}
void append_scene(
    Scene&             dest,
    const Scene&       local,
    const Mat4d&       transform,
    const std::string& prefix) {
  const int texture_base = static_cast<int>(dest.textures.size());
  dest.textures.insert(dest.textures.end(), local.textures.begin(), local.textures.end());
  for (const auto& o : local.objects) {
    if (o.mesh_index < 0)
      continue;
    auto m  = local.meshes.at(o.mesh_index);
    auto tr = transform * object_pose(o);
    for (std::size_t i = 0; i < m.positions.size(); i += 3) {
      auto p         = tr.TransformPoint({m.positions[i], m.positions[i + 1], m.positions[i + 2]});
      m.positions[i] = float(p.x);
      m.positions[i + 1] = float(p.y);
      m.positions[i + 2] = float(p.z);
    }
    for (std::size_t i = 0; i < m.normals.size(); i += 3) {
      auto n = tr.TransformNormal({m.normals[i], m.normals[i + 1], m.normals[i + 2]}).Normalized();
      m.normals[i]     = float(n.x);
      m.normals[i + 1] = float(n.y);
      m.normals[i + 2] = float(n.z);
    }
    Object3D output     = o;
    output.name         = prefix + "/" + o.name;
    output.position     = {};
    output.rotation_deg = {};
    output.scale        = {1, 1, 1};
    output.parent       = -1;
    output.mesh_index   = static_cast<int>(dest.meshes.size());
    for (int* index :
         {&output.texture_index,
          &output.normal_map_index,
          &output.roughness_map_index,
          &output.metallic_map_index})
      if (*index >= 0)
        *index += texture_base;
    dest.meshes.push_back(std::move(m));
    dest.AddObject(output);
  }
}
} // namespace
struct Robot::Impl {
  struct Joint {
    RobotJoint info;
    Mat4d      origin;
    Vec3d      axis{1, 0, 0};
  };
  struct Visual {
    std::string link;
    Mat4d       origin;
    Scene       local;
  };
  std::string                        path, root;
  std::set<std::string>              links;
  std::vector<Joint>                 joint_list;
  std::map<std::string, std::size_t> by_name;
  std::map<std::string, std::size_t> parent_joint;
  std::vector<Visual>                visuals;
  /// @brief Resolve one finite coordinate and mimic chain, optionally enforcing authored limits.
  /// @param name Existing joint name.
  /// @param positions Independent coordinate map.
  /// @param active Cycle-detection stack, restored after this call.
  /// @param enforce True for strict commands; false for finite measured encoder states.
  /// @return Effective radians/metres.
  /// @throws std::invalid_argument Unknown/cyclic/nonfinite/out-of-range coordinates.
  double                             coordinate(
                                  const std::string&                   name,
                                  const std::map<std::string, double>& positions,
                                  std::set<std::string>&               active, bool enforce=true) const {
    auto found = by_name.find(name);
    ensure(found != by_name.end(), "unknown joint " + name);
    const auto& j = joint_list[found->second].info;
    if (j.type == JointType::Fixed)
      return 0;
    ensure(active.insert(name).second, "cyclic mimic joints");
    double value = 0;
    if (!j.mimic.empty())
      value = coordinate(j.mimic, positions, active,enforce) * j.multiplier + j.offset;
    else if (auto p = positions.find(name); p != positions.end())
      value = p->second;
    active.erase(name);
    ensure(std::isfinite(value), "nonfinite joint " + name);
    ensure(
        !enforce || j.type == JointType::Continuous || (value >= j.lower - 1e-12 && value <= j.upper + 1e-12),
        "joint outside limits: " + name);
    return value;
  }
  /// @brief Resolve a coordinate with a private cycle-detection stack.
  /// @param name Existing joint.
  /// @param positions Independent coordinate map.
  /// @param enforce Enforce authored motion limits when true.
  /// @return Effective finite coordinate.
  /// @throws std::invalid_argument Invalid coordinate or mimic chain.
  double coordinate(const std::string& name, const std::map<std::string, double>& positions,bool enforce=true) const {
    std::set<std::string> active;
    return coordinate(name, positions, active,enforce);
  }
  /// @brief Evaluate the URDF transform using commanded or measured coordinates.
  /// @param link Existing link name.
  /// @param positions Independent coordinate map.
  /// @param enforce Whether authored limits are enforced.
  /// @return Link-to-root rigid transform.
  /// @throws std::invalid_argument Invalid link or coordinates.
  Mat4d world(const std::string& link, const std::map<std::string, double>& positions,bool enforce=true) const {
    ensure(links.contains(link), "unknown link " + link);
    if (link == root)
      return {};
    const auto& j = joint_list.at(parent_joint.at(link));
    double      q = coordinate(j.info.name, positions,enforce);
    Mat4d       motion;
    if (j.info.type == JointType::Prismatic)
      motion = Mat4d::Translation(j.axis * q);
    else if (j.info.type != JointType::Fixed)
      motion = Mat4d::Rotation(j.axis, q);
    return world(j.info.parent, positions,enforce) * j.origin * motion;
  }
};
Robot::Robot(std::shared_ptr<const Impl> p) : impl_(std::move(p)) {}
Robot Robot::load_urdf(const std::string& path, const std::string& package_root) {
  std::ifstream file(path, std::ios::binary);
  if (!file)
    throw std::runtime_error("Cannot open URDF: " + path);
  std::string               text{std::istreambuf_iterator<char>(file), {}};
  detail::xml::xml_document doc;
  auto                      parsed = doc.load_buffer(text.data(), text.size());
  if (!parsed)
    throw std::runtime_error(std::string("URDF XML: ") + parsed.description());
  auto root = doc.child("robot");
  ensure(bool(root), "no robot root");
  auto data                            = std::make_shared<Impl>();
  data->path                           = path;
  auto                            base = std::filesystem::absolute(path).parent_path();
  std::map<std::string, Material> materials;
  for (auto m : root.children("material")) {
    auto name = std::string(m.attribute("name").as_string());
    ensure(!name.empty() && !materials.contains(name), "duplicate/empty material " + name);
    materials[name] = material(m);
  }
  for (auto link : root.children("link")) {
    auto name = std::string(link.attribute("name").as_string());
    ensure(!name.empty() && data->links.insert(name).second, "duplicate/empty link " + name);
    for (auto visual : link.children("visual")) {
      auto geometry = visual.child("geometry");
      ensure(bool(geometry), "visual has no geometry");
      Scene local;
      Mat4d scale;
      if (auto mesh = geometry.child("mesh")) {
        auto source = asset(mesh.attribute("filename").as_string(), base, package_root);
        local       = load_scene(source.string());
        auto s      = vec(mesh.attribute("scale").as_string("1 1 1"));
        ensure(s.x != 0 && s.y != 0 && s.z != 0, "zero visual mesh scale");
        scale = Mat4d::Scaling(s);
      } else if (auto box = geometry.child("box")) {
        auto s = vec(box.attribute("size").as_string());
        add_solid(local, Solid::box(s.x, s.y, s.z).translated(-s.x / 2, -s.y / 2, -s.z / 2));
      } else if (auto sphere = geometry.child("sphere"))
        add_solid(local, Solid::sphere(number(sphere.attribute("radius").as_string())));
      else if (auto cylinder = geometry.child("cylinder")) {
        double h = number(cylinder.attribute("length").as_string());
        add_solid(
            local,
            Solid::cylinder(number(cylinder.attribute("radius").as_string()), h)
                .translated(0, 0, -h / 2));
      } else
        throw std::invalid_argument("URDF: unsupported visual geometry in " + name);
      auto     mat = visual.child("material");
      Material value;
      auto     found = materials.find(mat.attribute("name").as_string());
      if (found != materials.end())
        value = found->second;
      value       = material(mat, value);
      int texture = -1;
      if (!value.texture.empty()) {
        std::string error;
        texture =
            LoadTextureIntoScene(&local, asset(value.texture, base, package_root).string(), &error);
        if (texture < 0)
          throw std::runtime_error(error);
      }
      for (auto& o : local.objects) {
        o.color = value.color;
        if (texture >= 0)
          o.texture_index = texture;
      }
      data->visuals.push_back({name, pose(visual.child("origin")) * scale, std::move(local)});
    }
  }
  ensure(!data->links.empty(), "no links");
  for (auto node : root.children("joint")) {
    Impl::Joint joint;
    auto&       j = joint.info;
    j.name        = node.attribute("name").as_string();
    j.parent      = node.child("parent").attribute("link").as_string();
    j.child       = node.child("child").attribute("link").as_string();
    auto type     = std::string(node.attribute("type").as_string());
    ensure(!j.name.empty() && !data->by_name.contains(j.name), "duplicate/empty joint " + j.name);
    ensure(
        data->links.contains(j.parent) && data->links.contains(j.child) && j.parent != j.child,
        "invalid links in joint " + j.name);
    ensure(!data->parent_joint.contains(j.child), "link has multiple parents: " + j.child);
    if (type == "fixed")
      j.type = JointType::Fixed;
    else if (type == "revolute")
      j.type = JointType::Revolute;
    else if (type == "continuous")
      j.type = JointType::Continuous;
    else if (type == "prismatic")
      j.type = JointType::Prismatic;
    else
      throw std::invalid_argument("URDF: unsupported joint type " + type);
    joint.origin = pose(node.child("origin"));
    joint.axis   = vec(node.child("axis").attribute("xyz").as_string("1 0 0"));
    ensure(j.type == JointType::Fixed || joint.axis.Length() > 0, "zero joint axis");
    joint.axis = joint.axis.Normalized();
    if (j.type == JointType::Revolute || j.type == JointType::Prismatic) {
      auto limit = node.child("limit");
      ensure(bool(limit), "missing limits for " + j.name);
      j.lower = number(limit.attribute("lower").as_string());
      j.upper = number(limit.attribute("upper").as_string());
      ensure(j.lower <= j.upper, "inverted joint limits");
    }
    if (auto mimic = node.child("mimic")) {
      j.mimic = mimic.attribute("joint").as_string();
      ensure(!j.mimic.empty(), "empty mimic joint");
      j.multiplier = number(mimic.attribute("multiplier").as_string("1"));
      j.offset     = number(mimic.attribute("offset").as_string("0"));
    }
    data->by_name[j.name]       = data->joint_list.size();
    data->parent_joint[j.child] = data->joint_list.size();
    data->joint_list.push_back(joint);
  }
  for (const auto& l : data->links)
    if (!data->parent_joint.contains(l)) {
      ensure(data->root.empty(), "multiple root links");
      data->root = l;
    }
  ensure(!data->root.empty(), "cyclic link graph");
  for (const auto& l : data->links) {
    std::set<std::string> active;
    auto                  current = l;
    while (current != data->root) {
      ensure(active.insert(current).second, "cyclic link graph");
      auto p = data->parent_joint.find(current);
      ensure(p != data->parent_joint.end(), "disconnected link graph");
      current = data->joint_list[p->second].info.parent;
    }
  }
  Robot robot(data);
  for (const auto& j : data->joint_list) {
    if (j.info.type != JointType::Fixed && j.info.mimic.empty())
      robot.positions_[j.info.name] =
          j.info.type == JointType::Continuous ? 0 : std::clamp(0., j.info.lower, j.info.upper);
  }
  for (const auto& j : data->joint_list)
    data->coordinate(j.info.name, robot.positions_);
  return robot;
}
void Robot::set_joint(const std::string& name, double value) {
  auto i = impl_->by_name.find(name);
  ensure(i != impl_->by_name.end(), "unknown joint " + name);
  const auto& j = impl_->joint_list[i->second].info;
  ensure(
      j.type != JointType::Fixed && j.mimic.empty(),
      "joint is fixed or mimics another joint: " + name);
  ensure(std::isfinite(value), "nonfinite joint value");
  auto next  = positions_;
  next[name] = value;
  for (const auto& v : impl_->joint_list)
    impl_->coordinate(v.info.name, next);
  positions_ = std::move(next);
}
void Robot::set_measured_joint(const std::string& name,double value){set_measured_joints({{name,value}});}
void Robot::set_measured_joints(const std::vector<std::pair<std::string,double>>& values){
 if(values.empty())return;auto next=positions_;std::set<std::string> seen;
 for(const auto& item:values){const auto& name=item.first;double value=item.second;auto i=impl_->by_name.find(name);ensure(i!=impl_->by_name.end(),"unknown joint "+name);const auto& j=impl_->joint_list[i->second].info;
 ensure(seen.insert(name).second,"duplicate measured joint "+name);ensure(j.type!=JointType::Fixed&&j.mimic.empty(),"joint is fixed or mimics another joint: "+name);ensure(std::isfinite(value)&&std::abs(value)<=1000,"invalid measured coordinate");next[name]=value;}
 for(const auto& v:impl_->joint_list)impl_->coordinate(v.info.name,next,false);positions_=std::move(next);measured_=true;
}
double Robot::joint_position(const std::string& name) const {
  return impl_->coordinate(name, positions_,!measured_);
}
std::vector<RobotJoint> Robot::joints() const {
  std::vector<RobotJoint> r;
  for (const auto& j : impl_->joint_list)
    r.push_back(j.info);
  return r;
}
std::vector<std::string> Robot::joint_names() const {
  std::vector<std::string> r;
  for (const auto& j : impl_->joint_list)
    if (j.info.type != JointType::Fixed && j.info.mimic.empty())
      r.push_back(j.info.name);
  return r;
}
Mat4d Robot::link_transform(const std::string& name) const {
  return impl_->world(name, positions_,!measured_);
}
Scene Robot::scene() const {
  Scene result;
  result.source_path = impl_->path;
  for (const auto& v : impl_->visuals)
    append_scene(result, v.local, impl_->world(v.link, positions_,!measured_) * v.origin, v.link);
  return result;
}
} // namespace datamunge::cad
