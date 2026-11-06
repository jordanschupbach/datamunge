/// @file articulation.cpp
/// @brief Native URDF inertia/collision import and transactional rigid-tree instantiation.
#include <cstring>
#include <datamunge/sim/articulation.hpp>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
#include <sstream>

#include "../cad/detail/xml.hpp"
#include "rigid_math.hpp"
namespace datamunge::sim {
/// @brief URDF attribute and immutable collision-asset helpers.
namespace {
namespace x = cad::detail::xml; ///< Non-network XML parser already owned by the core.
using namespace detail;
/// @brief Parse a finite scalar, preserving an explicit fallback for absent attributes.
/// @param a Optional scalar XML attribute.
/// @param fallback Value when absent, default zero.
/// @return Finite scalar.
/// @throws std::invalid_argument Malformed or nonfinite attribute.
/// @throws std::out_of_range Numeric value outside double range.
double scalar(x::xml_attribute a, double fallback = 0) {
  std::string s = a.as_string();
  if (s.empty())
    return fallback;
  std::size_t end = 0;
  double      v   = std::stod(s, &end);
  if (end != s.size() || !std::isfinite(v))
    throw std::invalid_argument("invalid URDF scalar");
  return v;
}
/// @brief Parse an optional finite three-vector.
/// @param s Three whitespace-separated components, or empty.
/// @param fallback Value when absent, default zero.
/// @return Finite vector.
/// @throws std::invalid_argument Malformed vector.
Vec3 vector(const std::string& s, Vec3 fallback = {}) {
  if (s.empty())
    return fallback;
  std::istringstream in(s);
  Vec3               v;
  for (auto& a : v)
    if (!(in >> a) || !std::isfinite(a))
      throw std::invalid_argument("invalid URDF vector");
  std::string tail;
  if (in >> tail)
    throw std::invalid_argument("extra URDF vector fields");
  return v;
}
/// @brief Fixed-axis URDF RPY quaternion.
/// @param node Optional origin node.
/// @return Unit WXYZ rotation; identity if absent.
Quaternion orientation(x::xml_node node) {
  auto a = vector(node.attribute("rpy").as_string());
  return product(
      axis_angle({0, 0, 1}, a[2]),
      product(axis_angle({0, 1, 0}, a[1]), axis_angle({1, 0, 0}, a[0])));
}
/// @brief Binary or ASCII STL triangle vertices; rejects malformed/nonfinite data.
/// @param path Existing collision mesh file.
/// @return Triangle vertices in source coordinates.
/// @throws std::runtime_error Asset I/O failure.
/// @throws std::invalid_argument Invalid mesh records.
std::vector<Vec3> stl(const std::filesystem::path& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("cannot load collision STL: " + path.string());
  std::string       bytes{std::istreambuf_iterator<char>(f), {}};
  std::vector<Vec3> out;
  auto              u32 = [&](std::size_t i) {
    return std::uint32_t((unsigned char)bytes[i]) |
           (std::uint32_t((unsigned char)bytes[i + 1]) << 8) |
           (std::uint32_t((unsigned char)bytes[i + 2]) << 16) |
           (std::uint32_t((unsigned char)bytes[i + 3]) << 24);
  };
  if (bytes.size() >= 84 && bytes.size() == 84 + 50ULL * u32(80)) {
    for (std::size_t k = 0; k < u32(80); ++k)
      for (int p = 0; p < 3; ++p) {
        Vec3 v;
        for (int j = 0; j < 3; ++j) {
          auto  raw = u32(84 + 50 * k + 12 + 12 * p + 4 * j);
          float x;
          std::memcpy(&x, &raw, 4);
          if (!std::isfinite(x))
            throw std::invalid_argument("nonfinite collision STL");
          v[j] = x;
        }
        out.push_back(v);
      }
  } else {
    std::istringstream in(bytes);
    std::string        token;
    while (in >> token)
      if (token == "vertex") {
        Vec3 v;
        for (auto& a : v)
          if (!(in >> a) || !std::isfinite(a))
            throw std::invalid_argument("malformed ASCII collision STL");
        out.push_back(v);
      }
  }
  if (out.size() < 12 || out.size() % 3)
    throw std::invalid_argument("empty/malformed collision STL");
  return out;
}
/// @brief Clip triangle edges into a convex slab, then deterministically sample extreme vertices.
/// @param triangles Source triangle vertices.
/// @param lo Inclusive mesh-local X lower bound.
/// @param hi Inclusive mesh-local X upper bound.
/// @param directions Support sampling count; zero retains all clipped vertices.
/// @return Shared immutable hull vertex cloud.
/// @throws std::invalid_argument Empty or degenerate slab.
std::shared_ptr<const std::vector<Vec3>>
hull(const std::vector<Vec3>& triangles, double lo, double hi, std::size_t directions) {
  std::set<std::array<long long, 3>> unique;
  auto                               put = [&](Vec3 p) {
    unique.insert({std::llround(p[0] * 1e7), std::llround(p[1] * 1e7), std::llround(p[2] * 1e7)});
  };
  for (std::size_t i = 0; i < triangles.size(); i += 3)
    for (int j = 0; j < 3; ++j) {
      auto a = triangles[i + j], b = triangles[i + (j + 1) % 3];
      if (a[0] >= lo && a[0] <= hi)
        put(a);
      for (double cut : {lo, hi})
        if ((a[0] < cut && b[0] > cut) || (b[0] < cut && a[0] > cut))
          put(add(a, mul(sub(b, a), (cut - a[0]) / (b[0] - a[0]))));
    }
  std::vector<Vec3> vertices;
  for (auto p : unique)
    vertices.push_back({p[0] / 1e7, p[1] / 1e7, p[2] / 1e7});
  if (vertices.size() < 4)
    throw std::invalid_argument("degenerate collision slab");
  if (directions && vertices.size() > directions) {
    std::set<std::size_t> ids;
    auto                  extreme = [&](Vec3 d) {
      std::size_t best = 0;
      for (std::size_t i = 1; i < vertices.size(); ++i)
        if (dot(vertices[i], d) > dot(vertices[best], d))
          best = i;
      ids.insert(best);
    };
    for (int i = 0; i < 3; ++i) {
      extreme(basis(i));
      extreme(mul(basis(i), -1));
    }
    for (std::size_t i = 0; i < directions; ++i) {
      double z = 1 - 2 * (i + .5) / directions, r = std::sqrt(std::max(0., 1 - z * z)),
             phi = 2.399963229728653 * i;
      extreme({r * std::cos(phi), r * std::sin(phi), z});
    }
    std::vector<Vec3> reduced;
    for (auto i : ids)
      reduced.push_back(vertices[i]);
    vertices = std::move(reduced);
  }
  return std::make_shared<const std::vector<Vec3>>(std::move(vertices));
}
/// @brief Rotate the complete authored COM tensor into the body frame.
/// @param inertial URDF inertial element.
/// @return Full body-frame COM inertia, kg m².
Tensor3 tensor(x::xml_node inertial) {
  auto        node = inertial.child("inertia");
  const char* keys[9]{"ixx", "ixy", "ixz", "ixy", "iyy", "iyz", "ixz", "iyz", "izz"};
  Tensor3     I{};
  for (int k = 0; k < 9; ++k)
    I[k] = scalar(node.attribute(keys[k]));
  auto    q = orientation(inertial.child("origin"));
  Tensor3 result{};
  for (int j = 0; j < 3; ++j) {
    auto col = rotate(q, apply_tensor(I, rotate(conjugate(q), basis(j))));
    for (int i = 0; i < 3; ++i)
      result[3 * i + j] = col[i];
  }
  return result;
}
} // namespace
/// @brief Authored link frame, mass properties and incoming joint.
struct UrdfModel::Impl {
  struct Link {
    std::string name;                 ///< Source link name.
    std::string parent;               ///< Parent link name, empty at root.
    std::string joint;                ///< Incoming joint name.
    std::string mimic;                ///< Master joint name, empty without gearing.
    Body        body;                 ///< Reference-frame rigid mass and collision definition.
    Joint       constraint;           ///< Authored incoming constraint and passive drive data.
    Vec3        translation{};        ///< Joint origin in parent link coordinates.
    Quaternion  rotation{1, 0, 0, 0}; ///< Joint frame orientation in parent coordinates.
    double      ratio{1};             ///< Mimic multiplier.
    double      offset{0};            ///< Mimic offset in radians/metres.
  }; ///< Immutable tree node.
  std::vector<Link> links; ///< Parent-before-child order, starting with the fixed root.
};
UrdfModel UrdfModel::load(const std::string& path, UrdfPhysicsOptions options) {
  if (!std::isfinite(options.missing_mass) || options.missing_mass < 0 ||
      !std::isfinite(options.joint_damping) || options.joint_damping < 0 ||
      !std::isfinite(options.slider_damping) || options.slider_damping < 0 ||
      options.hull_directions > 4096 || (options.hull_directions && options.hull_directions < 16))
    throw std::invalid_argument("invalid native URDF options");
  std::ifstream   f(path);
  std::string     bytes{std::istreambuf_iterator<char>(f), {}};
  x::xml_document doc;
  if (!f || !doc.load_buffer(bytes.data(), bytes.size()))
    throw std::runtime_error("cannot read physical URDF");
  auto                                            root = doc.child("robot");
  std::map<std::string, x::xml_node>              links, incoming;
  std::map<std::string, std::vector<std::string>> children;
  std::set<std::string>                           names;
  for (auto node : root.children("link")) {
    std::string n = node.attribute("name").as_string();
    if (n.empty() || !links.emplace(n, node).second)
      throw std::invalid_argument("duplicate/empty URDF link");
  }
  for (auto node : root.children("joint")) {
    std::string n = node.attribute("name").as_string(),
                p = node.child("parent").attribute("link").as_string(),
                c = node.child("child").attribute("link").as_string();
    if (n.empty() || !names.insert(n).second || !links.count(p) || !links.count(c) ||
        !incoming.emplace(c, node).second)
      throw std::invalid_argument("invalid URDF joint graph");
    children[p].push_back(c);
  }
  std::string base;
  for (auto [n, node] : links)
    if (!incoming.count(n)) {
      if (!base.empty())
        throw std::invalid_argument("multiple URDF roots");
      base = n;
    }
  if (base.empty())
    throw std::invalid_argument("URDF has no root");
  auto                             model = std::make_shared<Impl>();
  std::set<std::string>            visited;
  std::function<void(std::string)> visit = [&](std::string name) {
    if (!visited.insert(name).second)
      throw std::invalid_argument("URDF cycle");
    auto node = links.at(name), joint = incoming.count(name) ? incoming.at(name) : x::xml_node{};
    Impl::Link link;
    link.name                = name;
    link.body.motion         = name == base ? Motion::Static : Motion::Dynamic;
    link.body.dimensions     = {.1, .1, .1};
    link.body.linear_damping = link.body.angular_damping = 0;
    link.body.angular_speed_limit                        = 10000;
    if (name != base) {
      link.parent      = joint.child("parent").attribute("link").as_string();
      link.joint       = joint.attribute("name").as_string();
      link.translation = vector(joint.child("origin").attribute("xyz").as_string());
      link.rotation    = orientation(joint.child("origin"));
      auto type        = std::string(joint.attribute("type").as_string());
      if (type == "fixed")
        link.constraint.type = JointType::Fixed;
      else if (type == "prismatic")
        link.constraint.type = JointType::Slider;
      else if (type == "revolute" || type == "continuous")
        link.constraint.type = JointType::Hinge;
      else
        throw std::invalid_argument("unsupported native URDF joint type");
      if (type != "fixed") {
        link.constraint.axis = vector(joint.child("axis").attribute("xyz").as_string(), {1, 0, 0});
        link.constraint.limited        = type != "continuous";
        link.constraint.effort_limit   = scalar(joint.child("limit").attribute("effort"), 100000);
        link.constraint.velocity_limit = scalar(joint.child("limit").attribute("velocity"), 100000);
        link.constraint.limit_min      = scalar(joint.child("limit").attribute("lower"));
        link.constraint.limit_max      = scalar(joint.child("limit").attribute("upper"));
        link.constraint.damping        = scalar(
            joint.child("dynamics").attribute("damping"),
            type == "prismatic" ? options.slider_damping : options.joint_damping);
        link.constraint.friction = scalar(joint.child("dynamics").attribute("friction"));
        link.constraint.armature =
            type == "prismatic" ? options.slider_armature : options.hinge_armature;
      }
      auto mimic = joint.child("mimic");
      if (mimic) {
        link.mimic  = mimic.attribute("joint").as_string();
        link.ratio  = scalar(mimic.attribute("multiplier"), 1);
        link.offset = scalar(mimic.attribute("offset"));
        if (!names.count(link.mimic) || link.ratio == 0)
          throw std::invalid_argument("invalid URDF mimic");
      }
      link.constraint.child = 1;
      link.constraint.validate();
      link.constraint.child = 0;
    }
    auto inertial = node.child("inertial");
    if (inertial) {
      link.body.mass           = scalar(inertial.child("mass").attribute("value"));
      link.body.center_of_mass = vector(inertial.child("origin").attribute("xyz").as_string());
      link.body.inertia_tensor = tensor(inertial);
      if (link.body.inertia_tensor == detail::Tensor3{})
        throw std::invalid_argument("zero authored URDF inertia");
    } else if (name != base) {
      if (options.missing_mass <= 0)
        throw std::invalid_argument("missing URDF inertia: " + name);
      link.body.mass = options.missing_mass;
      for (int i = 0; i < 3; ++i)
        link.body.inertia_tensor[4 * i] = options.missing_inertia[i];
    }
    std::vector<Collider> colliders;
    for (auto collision : node.children("collision")) {
      Collider c;
      c.position = vector(collision.child("origin").attribute("xyz").as_string());
      c.rotation = orientation(collision.child("origin"));
      auto geom  = collision.child("geometry");
      if (auto box = geom.child("box")) {
        c.dimensions = vector(box.attribute("size").as_string());
        colliders.push_back(c);
      } else if (auto sphere = geom.child("sphere")) {
        c.shape      = Shape::Sphere;
        double d     = 2 * scalar(sphere.attribute("radius"));
        c.dimensions = {d, d, d};
        colliders.push_back(c);
      } else if (auto cylinder = geom.child("cylinder")) {
        std::vector<Vec3> v;
        double            r = scalar(cylinder.attribute("radius")),
               h            = scalar(cylinder.attribute("length")) / 2;
        for (int k = 0; k < 32; ++k)
          for (double z : {-h, h})
            v.push_back(
                {r * std::cos(2 * std::acos(-1.) * k / 32),
                 r * std::sin(2 * std::acos(-1.) * k / 32),
                 z});
        c.shape    = Shape::ConvexHull;
        c.vertices = std::make_shared<const std::vector<Vec3>>(std::move(v));
        colliders.push_back(c);
      } else if (auto mesh = geom.child("mesh")) {
        std::string filename = mesh.attribute("filename").as_string();
        if (filename.starts_with("package://"))
          throw std::invalid_argument(
              "expand package:// collision paths before native URDF import");
        auto vertices =
            stl((std::filesystem::path(path).parent_path() / filename).lexically_normal());
        auto scale = vector(mesh.attribute("scale").as_string(), {1, 1, 1});
        for (auto& p : vertices)
          for (int i = 0; i < 3; ++i)
            p[i] *= scale[i];
        auto slices = options.mesh_x_slices.count(name)
                          ? options.mesh_x_slices.at(name)
                          : std::vector<std::array<double, 2>>{{-1000, 1000}};
        for (auto slice : slices) {
          if (!std::isfinite(slice[0]) || !std::isfinite(slice[1]) || slice[0] >= slice[1])
            throw std::invalid_argument("invalid mesh slab");
          c.shape    = Shape::ConvexHull;
          c.vertices = hull(vertices, slice[0], slice[1], options.hull_directions);
          colliders.push_back(c);
        }
      } else
        throw std::invalid_argument("unsupported URDF collision geometry");
    }
    if (!colliders.empty())
      link.body.colliders = std::make_shared<const std::vector<Collider>>(std::move(colliders));
    else
      link.body.collision_group = 0;
    link.body.validate();
    model->links.push_back(std::move(link));
    for (auto child : children[name])
      visit(child);
  };
  visit(base);
  if (visited.size() != links.size())
    throw std::invalid_argument("disconnected URDF tree");
  UrdfModel out;
  out.impl_ = std::move(model);
  return out;
}
Articulation UrdfModel::instantiate(World& world, Vec3 position, Quaternion rotation) const {
  if (!impl_)
    throw std::invalid_argument("empty physical URDF model");
  Articulation        result;
  std::vector<BodyId> added, joints, gears;
  try {
    for (auto& link : impl_->links) {
      auto b = link.body;
      if (link.parent.empty()) {
        b.position = position;
        b.rotation = rotation;
      } else {
        auto parent = world.body(result.links.at(link.parent));
        b.position =
            detail::add(parent.position, detail::rotate(parent.rotation, link.translation));
        b.rotation = detail::normalized(detail::product(parent.rotation, link.rotation));
      }
      auto id = world.add_body(b);
      added.push_back(id);
      result.links[link.name] = id;
      if (!link.parent.empty()) {
        auto j   = link.constraint;
        j.child  = id;
        j.parent = result.links.at(link.parent);
        world.add_joint(j);
        joints.push_back(id);
        result.joints[link.joint] = id;
      }
    }
    for (auto& link : impl_->links)
      if (!link.mimic.empty()) {
        world.add_gear(
            result.joints.at(link.mimic), result.joints.at(link.joint), link.ratio, link.offset);
        gears.push_back(result.joints.at(link.joint));
      }
  } catch (...) {
    for (auto id : gears)
      world.remove_gear(id);
    for (auto i = joints.rbegin(); i != joints.rend(); ++i)
      world.remove_joint(*i);
    for (auto i = added.rbegin(); i != added.rend(); ++i)
      world.remove_body(*i);
    throw;
  }
  return result;
}
} // namespace datamunge::sim
