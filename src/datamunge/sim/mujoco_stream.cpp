/// @file mujoco_stream.cpp
/// @brief MuJoCo C API adapter: URDF dynamics, physical jaws and compound fruit.
#include "mujoco_stream.hpp"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>

#include "../cad/detail/xml.hpp"
#include "math.hpp"
#ifdef DATAMUNGE_ENABLE_MUJOCO
#include <mujoco/mujoco.h>
#endif
namespace datamunge::sim::detail {
#ifdef DATAMUNGE_ENABLE_MUJOCO
/// @brief Local MJCF serialization and coordinate conversion helpers.
namespace {
std::mutex compilation_mutex;   ///< Serializes MuJoCo XML compiler global last-model bookkeeping.
namespace x = cad::detail::xml; ///< Existing non-network XML DOM.
/// @brief XML-escape an owned name or absolute asset filename.
/// @param s Input text.
/// @return Attribute-safe XML text.
std::string escape(std::string s) {
  std::string out;
  for (char c : s) {
    if (c == '&')
      out += "&amp;";
    else if (c == '\"')
      out += "&quot;";
    else if (c == '<')
      out += "&lt;";
    else
      out += c;
  }
  return out;
}
/// @brief Read three finite numeric components, with an explicit default.
/// @param s Authored attribute.
/// @param fallback Empty-attribute value.
/// @return Finite vector.
/// @throws std::runtime_error Malformed authored vector.
Vec3 vector(const std::string& s, Vec3 fallback = {}) {
  if (s.empty())
    return fallback;
  std::istringstream in(s);
  Vec3               v;
  for (auto& a : v)
    if (!(in >> a) || !std::isfinite(a))
      throw std::runtime_error("invalid dynamics vector");
  return v;
}
/// @brief Serialize three finite components at reproducible full precision.
/// @param v Vector.
/// @return Space-separated components.
std::string text(Vec3 v) {
  std::ostringstream s;
  s << std::setprecision(17) << v[0] << ' ' << v[1] << ' ' << v[2];
  return s.str();
}
/// @brief URDF fixed-axis RPY quaternion, radians, in MuJoCo Z-up coordinates.
/// @param node Origin element, possibly absent.
/// @return WXYZ quaternion.
Quaternion rotation(x::xml_node node) {
  auto a = vector(node.attribute("rpy").as_string());
  return product(
      axis_angle({0, 0, 1}, a[2]),
      product(axis_angle({0, 1, 0}, a[1]), axis_angle({1, 0, 0}, a[0])));
}
/// @brief Serialize WXYZ rotation.
/// @param q Quaternion.
/// @return Space-separated components.
std::string text(Quaternion q) {
  std::ostringstream s;
  s << std::setprecision(17) << q[0] << ' ' << q[1] << ' ' << q[2] << ' ' << q[3];
  return s.str();
}
/// @brief Reflection-conjugate public Y-up rotation into right-handed Z-up.
/// @param q Public or MuJoCo WXYZ rotation.
/// @return Converted rotation; operation is its own inverse.
Quaternion converted(Quaternion q) {
  return {q[0], -q[1], -q[3], -q[2]};
}
/// @brief Construct a quaternion carrying local Z onto an authored segment.
/// @param d Nonzero segment direction.
/// @return WXYZ rotation.
Quaternion along(Vec3 d) {
  d = unit(d);
  if (d[2] < -.999999)
    return {0, 1, 0, 0};
  return normalized({1 + d[2], -d[1], d[0], 0});
}
/// @brief Split a finger into its carriage and five clipped distal-pad collision hulls.
/// Prevents one convex hull from filling the open space below the rear carriage.
/// @param file Official binary STL collision file.
/// @param part Zero for the carriage, 1..5 for successive clipped distal-pad sections.
/// @return Reproducible inline convex-hull vertex list in authored link coordinates.
/// @throws std::runtime_error Malformed or empty collision mesh.
std::string finger_vertices(const std::filesystem::path& file, int part) {
  std::ifstream in(file, std::ios::binary);
  std::string   b{std::istreambuf_iterator<char>(in), {}};
  if (b.size() < 84)
    throw std::runtime_error("truncated finger collision STL");
  /// @brief Decode one checked binary-STL little-endian word.
  /// @param p Byte offset within the verified record layout.
  /// @return Unsigned word.
  auto u32 = [&](std::size_t p) {
    return std::uint32_t((unsigned char)b[p]) | (std::uint32_t((unsigned char)b[p + 1]) << 8) |
           (std::uint32_t((unsigned char)b[p + 2]) << 16) |
           (std::uint32_t((unsigned char)b[p + 3]) << 24);
  };
  auto n = u32(80);
  if (b.size() != 84 + 50ULL * n)
    throw std::runtime_error("invalid finger collision STL");
  std::set<std::array<int, 3>> points;
  double                       lo = part == 0 ? -1 : -.055 + .011 * (part - 1),
         hi                       = part == 0 ? -.055 : -.055 + .011 * part;
  auto add_point                  = [&](Vec3 p) {
    points.insert(
        {int(std::lround(p[0] * 100000)),
         int(std::lround(p[1] * 100000)),
         int(std::lround(p[2] * 100000))});
  };
  for (std::size_t i = 0; i < n; ++i) {
    Vec3 triangle[3];
    for (int v = 0; v < 3; ++v)
      for (int j = 0; j < 3; ++j) {
        auto  raw = u32(84 + 50 * i + 12 + 12 * v + 4 * j);
        float f;
        std::memcpy(&f, &raw, 4);
        triangle[v][j] = f;
      }
    for (int v = 0; v < 3; ++v) {
      auto p = triangle[v], q = triangle[(v + 1) % 3];
      if (p[0] >= lo && p[0] <= hi)
        add_point(p);
      for (double bound : {lo, hi})
        if ((p[0] < bound && q[0] > bound) || (q[0] < bound && p[0] > bound)) {
          double t = (bound - p[0]) / (q[0] - p[0]);
          add_point(add(p, mul(sub(q, p), t)));
        }
    }
  }
  if (points.size() < 4)
    throw std::runtime_error("empty finger hull");
  std::ostringstream out;
  out << std::setprecision(12);
  for (auto p : points)
    out << p[0] / 100000. << ' ' << p[1] / 100000. << ' ' << p[2] / 100000. << ' ';
  return out.str();
}
/// @brief Add conveyor material speed to tangential contact constraints, never body motion.
/// @param m Independent compiled model.
/// @param d Current position/velocity constraint references after forward evaluation.
/// @param belt Stationary belt-support geom identifiers.
void conveyor_reference(const mjModel* m, mjData* d, const std::vector<int>& belt) {
  for (int c = 0; c < d->ncon; ++c) {
    auto& contact = d->contact[c];
    int   sign    = std::find(belt.begin(), belt.end(), contact.geom[0]) != belt.end()   ? 1
                    : std::find(belt.begin(), belt.end(), contact.geom[1]) != belt.end() ? -1
                                                                                         : 0;
    if (!sign || contact.efc_address < 0)
      continue;
    for (int t = 1; t < 3 && t < contact.dim; ++t) {
      int    row  = contact.efc_address + t;
      double slip = sign * .22 * contact.frame[3 * t];
      d->efc_vel[row] -= slip;
      d->efc_aref[row] += d->efc_KBIP[4 * row + 1] * slip;
    }
  }
}
/// @brief Conservative world bounds of a compound fruit from actual primitive transforms.
/// @param m Compiled model with runtime fruit geometry.
/// @param d Forwarded live state.
/// @param body Fruit body containing ellipsoids or capsules.
/// @return Minimum XYZ followed by maximum XYZ, metres.
std::array<double, 6> fruit_bounds(const mjModel* m, const mjData* d, int body) {
  std::array<double, 6> bounds{INFINITY, INFINITY, INFINITY, -INFINITY, -INFINITY, -INFINITY};
  for (int k = 0; k < m->body_geomnum[body]; ++k) {
    int g = m->body_geomadr[body] + k;
    if (!m->geom_contype[g])
      continue;
    auto* r    = d->geom_xmat + 9 * g;
    auto* size = m->geom_size + 3 * g;
    for (int j = 0; j < 3; ++j) {
      double extent = 0;
      if (m->geom_type[g] == mjGEOM_ELLIPSOID) {
        for (int a = 0; a < 3; ++a)
          extent += std::pow(r[3 * j + a] * size[a], 2);
        extent = std::sqrt(extent);
      } else
        extent = size[0] + std::abs(r[3 * j + 2]) * size[1];
      bounds[j]     = std::min(bounds[j], d->geom_xpos[3 * g + j] - extent);
      bounds[j + 3] = std::max(bounds[j + 3], d->geom_xpos[3 * g + j] + extent);
    }
  }
  return bounds;
}
/// @brief Rotate authored COM-frame inertia into its link frame before MJCF compilation.
/// @param inertial URDF inertial element.
/// @return Full symmetric tensor in MJCF ordering, kg m squared.
std::string inertia_tensor(x::xml_node inertial) {
  auto        n = inertial.child("inertia");
  double      tensor[3][3]{};
  const char* names[3][3]{{"ixx", "ixy", "ixz"}, {"ixy", "iyy", "iyz"}, {"ixz", "iyz", "izz"}};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      tensor[i][j] = std::stod(n.attribute(names[i][j]).as_string("0"));
  auto   q = rotation(inertial.child("origin"));
  Vec3   columns[3]{rotate(q, {1, 0, 0}), rotate(q, {0, 1, 0}), rotate(q, {0, 0, 1})};
  double result[3][3]{};
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
          result[i][j] += columns[a][i] * tensor[a][b] * columns[b][j];
  std::ostringstream out;
  out << std::setprecision(17) << result[0][0] << ' ' << result[1][1] << ' ' << result[2][2] << ' '
      << result[0][1] << ' ' << result[0][2] << ' ' << result[1][2];
  return out.str();
}
/// @brief Compile a complete DevArm MJCF, with authored inertias and collision hulls.
/// @param path Existing URDF.
/// @param world Static workcell geometry.
/// @param dt Internal timestep.
/// @param capacity Free payload slots.
/// @return Standalone XML with absolute collision mesh paths.
std::string
model_xml(const std::string& path, const World& world, double dt, std::size_t capacity) {
  std::ifstream   source(path, std::ios::binary);
  std::string     bytes{std::istreambuf_iterator<char>(source), {}};
  x::xml_document doc;
  if (!source || !doc.load_buffer(bytes.data(), bytes.size()))
    throw std::runtime_error("cannot read dynamics URDF");
  auto                                            root = doc.child("robot");
  std::map<std::string, x::xml_node>              links, joints;
  std::map<std::string, std::vector<x::xml_node>> children;
  for (auto l : root.children("link"))
    links[l.attribute("name").as_string()] = l;
  for (auto j : root.children("joint")) {
    joints[j.attribute("name").as_string()] = j;
    children[j.child("parent").attribute("link").as_string()].push_back(j);
  }
  std::ostringstream assets, bodies, actuators;
  assets << std::setprecision(17);
  bodies << std::setprecision(17);
  actuators << std::setprecision(17);
  /// @brief Emit a validated link and all children without altering live simulation data.
  /// @param name Link name.
  /// @param joint Authored incoming joint, empty for the fixed base.
  std::function<void(std::string, x::xml_node)> emit = [&](std::string name, x::xml_node joint) {
    auto link = links.at(name), origin = joint.child("origin");
    bool base = name == "base_link";
    bodies << "<body name=\"" << escape(name) << "\" pos=\""
           << (base ? "0.62 0 0.72" : text(vector(origin.attribute("xyz").as_string())))
           << "\" quat=\"" << (base ? "0 0 0 1" : text(rotation(origin))) << "\">";
    auto inertia = link.child("inertial");
    if (inertia && !base) {
      auto i = inertia.child("inertia");
      auto o = inertia.child("origin");
      bodies << "<inertial pos=\"" << escape(o.attribute("xyz").as_string("0 0 0")) << "\" mass=\""
             << escape(inertia.child("mass").attribute("value").as_string()) << "\" fullinertia=\"";
      bodies << inertia_tensor(inertia) << "\"/>";
    } else if (name.find("finger_") == 0)
      bodies
          << "<inertial pos=\"-0.025 0 0\" mass=\"0.04\" diaginertia=\"0.00002 0.00002 0.00002\"/>";
    std::string type = joint.attribute("type").as_string();
    if (type == "revolute" || type == "prismatic") {
      std::string jn     = joint.attribute("name").as_string();
      auto        lim    = joint.child("limit");
      double      effort = std::stod(lim.attribute("effort").as_string("8"));
      bodies << "<joint name=\"" << escape(jn) << "\" type=\""
             << (type == "prismatic" ? "slide" : "hinge") << "\" axis=\""
             << escape(joint.child("axis").attribute("xyz").as_string("1 0 0")) << "\" range=\""
             << escape(lim.attribute("lower").as_string()) << ' '
             << escape(lim.attribute("upper").as_string()) << "\" damping=\""
             << (type == "prismatic" ? .5 : .1) << "\" armature=\""
             << (type == "prismatic" ? 1.0 : .002) << "\"/>";
      int          order = jn.rfind("joint", 0) == 0 ? std::stoi(jn.substr(5)) : 7;
      const double kv    = order <= 3 ? 20 : order <= 6 ? 4 : 80;
      const double vmax  = type == "prismatic" ? .08 : 1.5;
      actuators << "<general name=\"servo_" << escape(jn) << "\" joint=\"" << escape(jn)
                << "\" gaintype=\"fixed\" biastype=\"affine\" gainprm=\"" << kv
                << "\" biasprm=\"0 0 " << -kv << "\" forcelimited=\"true\" forcerange=\"" << -effort
                << ' ' << effort << "\" ctrllimited=\"true\" ctrlrange=\"" << -vmax << ' ' << vmax
                << "\"/>";
    }
    int index = 0;
    for (auto collision : link.children("collision")) {
      auto mesh = collision.child("geometry").child("mesh");
      if (!mesh)
        throw std::runtime_error("DevArm dynamics needs authored mesh collisions");
      auto file =
          (std::filesystem::path(path).parent_path() / mesh.attribute("filename").as_string())
              .lexically_normal();
      if (!std::filesystem::is_regular_file(file))
        throw std::runtime_error("missing collision mesh: " + file.string());
      for (int part = 0; part < (name.find("finger_") == 0 ? 6 : 1); ++part) {
        std::string mn = name + "_collision_" + std::to_string(index++);
        if (name.find("finger_") == 0)
          assets << "<mesh name=\"" << escape(mn) << "\" vertex=\"" << finger_vertices(file, part)
                 << "\" maxhullvert=\"64\"/>";
        else
          assets << "<mesh name=\"" << escape(mn) << "\" file=\""
                 << escape(std::filesystem::absolute(file).string()) << "\" maxhullvert=\"64\"/>";
        bodies << "<geom name=\"" << escape(mn) << "\" type=\"mesh\" mesh=\"" << escape(mn)
               << "\" pos=\""
               << escape(collision.child("origin").attribute("xyz").as_string("0 0 0"))
               << "\" quat=\"" << text(rotation(collision.child("origin")))
               << "\" mass=\"0\" contype=\"2\" conaffinity=\"7\" friction=\"0.8 0.002 0.001\"/>";
      }
    }
    for (auto j : children[name])
      emit(j.child("child").attribute("link").as_string(), j);
    bodies << "</body>";
  };
  emit("base_link", {});
  std::ostringstream out;
  out << std::setprecision(17)
      << "<mujoco model=\"DevArm contact sorting\"><compiler angle=\"radian\" "
         "inertiafromgeom=\"false\"/><option timestep=\""
      << dt
      << "\" gravity=\"0 0 -9.81\" integrator=\"implicit\" cone=\"elliptic\" "
         "jacobian=\"sparse\" solver=\"Newton\" "
         "iterations=\"80\" tolerance=\"1e-9\"><flag "
         "autoreset=\"disable\"/></option><default><geom condim=\"6\" friction=\"0.8 0.002 "
         "0.004\" solref=\"0.004 1\" solimp=\"0.95 0.99 0.001\" "
         "margin=\"0.0001\"/></default><asset>"
      << assets.str() << "</asset><worldbody>";
  int k = 0;
  for (auto id : world.bodies()) {
    auto b = world.body(id);
    if (b.motion != Motion::Static || b.sensor)
      continue;
    out << "<geom name=\"fixture_" << id << "\" type=\"box\" pos=\""
        << text(Vec3{b.position[0], b.position[2], b.position[1]}) << "\" size=\""
        << text(Vec3{b.dimensions[0] / 2, b.dimensions[2] / 2, b.dimensions[1] / 2})
        << "\" contype=\"1\" conaffinity=\"6\"/>";
    ++k;
  }
  out << bodies.str();
  for (std::size_t i = 0; i < capacity; ++i) {
    out << "<body name=\"fruit_" << i << "\" pos=\"" << 10 + i
        << " 0 -10\" gravcomp=\"1\"><freejoint align=\"false\"/><inertial pos=\"0.001 0.002 "
           "0.003\" quat=\"0.9999995 0.000999999833 0 0\" mass=\"0.16\" diaginertia=\"0.0001 "
           "0.0001 0.0001\"/>";
    for (int g = 0; g < 5; ++g)
      out << "<geom name=\"fruit_" << i << '_' << g
          << "\" type=\"capsule\" size=\"0.15 0.15\" contype=\"4\" conaffinity=\"7\" mass=\"0\"/>";
    out << "</body>";
  }
  out << "</worldbody><contact>";
  for (const auto& item : joints) {
    auto j = item.second;
    out << "<exclude body1=\"" << escape(j.child("parent").attribute("link").as_string())
        << "\" body2=\"" << escape(j.child("child").attribute("link").as_string()) << "\"/>";
  }
  out << "<exclude body1=\"finger_left_link\" body2=\"finger_right_link\"/></contact><actuator>"
      << actuators.str()
      << "</actuator><equality><joint joint1=\"finger_right\" joint2=\"finger_left\" polycoef=\"0 "
         "-1 0 0 0\" solref=\"0.004 1\"/></equality></mujoco>";
  return out.str();
}
} // namespace
/// @brief Independent MuJoCo storage, fixed payload slots and contact identities.
struct MujocoStreamDynamics::Impl {
  std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model{
      nullptr,
      mj_deleteModel}; ///< Compiled dynamics.
  std::unique_ptr<mjData, decltype(&mj_deleteData)> data{nullptr, mj_deleteData}; ///< Solver state.
  std::unique_ptr<mjData, decltype(&mj_deleteData)> constants{
      nullptr,
      mj_deleteData};                                  ///< Non-live mass-property scratch.
  std::map<BodyId, int> slots;                         ///< Native identifier to free-joint slot.
  std::vector<int> bodies, ///< Slot body identifiers.
      addresses, ///< Free-joint position addresses.
      velocities; ///< Free-joint velocity addresses.
  std::array<int, 8> joints{}, ///< Six arm and two finger joint identifiers.
      actuators{}, ///< Velocity-servo actuator identifiers.
      dofs{}, ///< Generalized velocity coordinates.
      positions{}; ///< Generalized position coordinates.
  std::array<int, 2> fingers{}; ///< Contact body identifiers.
  std::vector<int>   floors;    ///< Bin floor geom identifiers.
  std::vector<int>   belt;      ///< Conveyor top contact geoms.
  std::array<double, 8>
         previous_target{}; ///< Last outer-tick servo references for bounded velocity feedforward.
  double peak_ratio{0}, ///< Maximum actuator effort fraction since reset.
      penetration{0}, ///< Maximum contact penetration, metres.
      joint_speed{0}, ///< Maximum measured arm joint speed, rad/s.
      finger_speed{0}; ///< Maximum measured finger speed, m/s.
  double ratio{0};     ///< Last commanded torque fraction.

  /// @brief Find a required named model element or reject malformed dynamics.
  /// @param kind MuJoCo object kind.
  /// @param name Required exact name.
  /// @return Nonnegative element index.
  int find(mjtObj kind, const std::string& name) {
    int id = mj_name2id(model.get(), kind, name.c_str());
    if (id < 0)
      throw std::runtime_error("missing dynamics element: " + name);
    return id;
  }
  /// @brief Return a retained body's slot or reject stale bookkeeping.
  /// @param id Native payload ID.
  /// @return Nonnegative slot.
  int slot(BodyId id) const {
    auto it = slots.find(id);
    if (it == slots.end())
      throw std::out_of_range("unknown dynamics payload");
    return it->second;
  }
};
MujocoStreamDynamics::MujocoStreamDynamics(
    const std::string& path,
    const World&       world,
    double             dt,
    std::size_t        capacity)
    : impl_(std::make_unique<Impl>()) {
  if (mj_version() != mjVERSION_HEADER)
    throw std::runtime_error("MuJoCo header/runtime version mismatch");
  std::lock_guard<std::mutex> compile_guard(compilation_mutex);
  auto                        xml = model_xml(path, world, dt, capacity);
  mjVFS                       vfs;
  mj_defaultVFS(&vfs);
  mj_addBufferVFS(&vfs, "workcell.xml", xml.data(), xml.size());
  char error[2048]{};
  auto raw = mj_loadXML("workcell.xml", &vfs, error, sizeof(error));
  mj_deleteVFS(&vfs);
  if (!raw)
    throw std::runtime_error(std::string("invalid articulated model: ") + error);
  impl_->model.reset(raw);
  impl_->data.reset(mj_makeData(raw));
  if (!impl_->data)
    throw std::runtime_error("cannot allocate articulated world");
  auto& p = *impl_;
  for (std::size_t i = 0; i < capacity; ++i) {
    int body = p.find(mjOBJ_BODY, "fruit_" + std::to_string(i)), j = raw->body_jntadr[body];
    p.bodies.push_back(body);
    p.addresses.push_back(raw->jnt_qposadr[j]);
    p.velocities.push_back(raw->jnt_dofadr[j]);
  }
  for (int i = 0; i < 8; ++i) {
    std::string name = i < 6    ? "joint" + std::to_string(i + 1)
                       : i == 6 ? "finger_left"
                                : "finger_right";
    p.joints[i]      = p.find(mjOBJ_JOINT, name);
    p.actuators[i]   = p.find(mjOBJ_ACTUATOR, "servo_" + name);
    p.positions[i]   = raw->jnt_qposadr[p.joints[i]];
    p.dofs[i]        = raw->jnt_dofadr[p.joints[i]];
  }
  p.fingers = {p.find(mjOBJ_BODY, "finger_left_link"), p.find(mjOBJ_BODY, "finger_right_link")};
  for (auto id : world.bodies()) {
    auto b = world.body(id);
    if (b.motion != Motion::Static || b.sensor)
      continue;
    int g = p.find(mjOBJ_GEOM, "fixture_" + std::to_string(id));
    if (std::abs(b.position[1] - .43) < .001)
      p.floors.push_back(g);
    if (std::abs(b.position[1] - .68) < .02 && b.dimensions[0] > 1)
      p.belt.push_back(g);
  }
}
MujocoStreamDynamics::~MujocoStreamDynamics() = default;
void MujocoStreamDynamics::reset(const std::array<double, 6>& q, double finger) {
  auto& p = *impl_;
  mj_resetData(p.model.get(), p.data.get());
  p.slots.clear();
  p.ratio = p.peak_ratio = p.penetration = p.joint_speed = p.finger_speed = 0;
  for (std::size_t s = 0; s < p.bodies.size(); ++s) {
    int b                     = p.bodies[s];
    p.model->body_gravcomp[b] = 1;
    p.model->body_contype[b] = p.model->body_conaffinity[b] = 0;
    for (int k = 0; k < 5; ++k) {
      int g                    = p.model->body_geomadr[b] + k;
      p.model->geom_contype[g] = p.model->geom_conaffinity[g] = 0;
    }
  }
  for (int i = 0; i < 8; ++i) {
    p.previous_target[i]         = i < 6 ? q[i] : i == 6 ? finger : -finger;
    p.data->qpos[p.positions[i]] = p.previous_target[i];
  }
  mj_forward(p.model.get(), p.data.get());
}
void MujocoStreamDynamics::spawn(BodyId id, const ConveyorObservation& s) {
  auto& p    = *impl_;
  int   slot = -1;
  for (std::size_t i = 0; i < p.bodies.size(); ++i) {
    bool used = false;
    for (auto kv : p.slots)
      used |= kv.second == int(i);
    if (!used) {
      slot = i;
      break;
    }
  }
  if (slot < 0)
    throw std::runtime_error("articulated payload capacity exhausted");
  p.slots[id] = slot;
  auto* m     = p.model.get();
  auto* d     = p.data.get();
  int   body = p.bodies[slot], at = p.addresses[slot], dof = p.velocities[slot];
  m->body_gravcomp[body]    = 0;
  m->body_sameframe[body]   = mjSAMEFRAME_NONE;
  m->body_contype[body]     = 4;
  m->body_conaffinity[body] = 7;
  auto q                    = converted(s.rotation);
  d->qpos[at]               = s.position[0];
  d->qpos[at + 1]           = s.position[2];
  d->qpos[at + 2]           = s.position[1];
  for (int k = 0; k < 4; ++k)
    d->qpos[at + 3 + k] = q[k];
  for (int k = 0; k < 6; ++k)
    d->qvel[dof + k] = 0;
  d->qvel[dof]     = s.velocity[0];
  d->qvel[dof + 1] = s.velocity[2];
  d->qvel[dof + 2] = s.velocity[1];
  auto omega       = rotate(
      conjugate(q), {-s.angular_velocity[0], -s.angular_velocity[2], -s.angular_velocity[1]});
  for (int k = 0; k < 3; ++k)
    d->qvel[dof + 3 + k] = omega[k];
  double mass        = (s.kind == FruitKind::Apple ? .16 : .12) * std::pow(s.appearance.scale, 3);
  m->body_mass[body] = mass;
  const int    prototype = s.appearance.prototype;
  const double girth[6]{1, .80, 1.10, 1.02, .9, .94}, lengths[6]{1, 1.15, .86, 1.02, 1.08, .84},
      curves[6]{.028, .018, .045, .05, .035, .055}, thickness[6]{1, .82, 1.16, .9, 1.04, 1.12};
  double scale = s.appearance.scale, r = .039 * scale;
  for (int k = 0; k < 5; ++k) {
    int g                  = m->body_geomadr[body] + k;
    m->geom_contype[g]     = k == 0 || s.kind == FruitKind::Banana ? 4 : 0;
    m->geom_conaffinity[g] = m->geom_contype[g] ? 7 : 0;
    for (int j = 0; j < 3; ++j)
      m->geom_pos[3 * g + j] = 0;
    for (int j = 0; j < 4; ++j)
      m->geom_quat[4 * g + j] = j == 0 ? 1 : 0;
    if (s.kind == FruitKind::Apple) {
      m->geom_sameframe[g] = mjSAMEFRAME_NONE;
      m->geom_type[g]      = mjGEOM_ELLIPSOID;
      m->geom_size[3 * g] = m->geom_size[3 * g + 1] = r * girth[prototype];
      m->geom_size[3 * g + 2]                       = r;
    } else {
      double u = -1 + 2. * k / 5, v = -1 + 2. * (k + 1) / 5;
      Vec3   a{0, -curves[prototype] * u * u * scale, .076 * lengths[prototype] * u * scale},
          b{0, -curves[prototype] * v * v * scale, .076 * lengths[prototype] * v * scale};
      auto delta = sub(b, a);
      auto qq    = along(delta);
      for (int j = 0; j < 3; ++j)
        m->geom_pos[3 * g + j] = (a[j] + b[j]) / 2;
      for (int j = 0; j < 4; ++j)
        m->geom_quat[4 * g + j] = qq[j];
      m->geom_sameframe[g] = mjSAMEFRAME_NONE;
      m->geom_type[g]      = mjGEOM_CAPSULE;
      m->geom_size[3 * g] =
          .023 * thickness[prototype] * scale *
          (.12 + .88 * std::sqrt(std::max(0., std::sin(3.141592653589793 * (k + .5) / 5))));
      m->geom_size[3 * g + 1] = length(delta) / 2;
      m->geom_size[3 * g + 2] = 0;
    }
    m->geom_rbound[g] =
        s.kind == FruitKind::Apple ? r * std::max(1., girth[prototype]) : .12 * scale;
    for (int j = 0; j < 3; ++j) {
      m->geom_aabb[6 * g + j]     = 0;
      m->geom_aabb[6 * g + 3 + j] = m->geom_rbound[g];
    }
  }
  if (s.kind == FruitKind::Apple) {
    double a                  = r * girth[prototype];
    m->body_inertia[3 * body] = m->body_inertia[3 * body + 1] = mass * (a * a + r * r) / 5;
    m->body_inertia[3 * body + 2]                             = 2 * mass * a * a / 5;
    for (int j = 0; j < 3; ++j)
      m->body_ipos[3 * body + j] = 0;
    for (int j = 0; j < 4; ++j)
      m->body_iquat[4 * body + j] = j == 0 ? 1 : 0;
  } else {
    double volumes[5]{}, total = 0;
    Vec3   com{};
    for (int k = 0; k < 5; ++k) {
      int    g = m->body_geomadr[body] + k;
      double r = m->geom_size[3 * g], l = 2 * m->geom_size[3 * g + 1];
      volumes[k] = 3.141592653589793 * r * r * (l + 4 * r / 3);
      total += volumes[k];
      for (int j = 0; j < 3; ++j)
        com[j] += volumes[k] * m->geom_pos[3 * g + j];
    }
    com = mul(com, 1 / total);
    mjtNum tensor[9]{};
    for (int k = 0; k < 5; ++k) {
      int    g = m->body_geomadr[body] + k;
      double r = m->geom_size[3 * g], l = 2 * m->geom_size[3 * g + 1],
             part = mass * volumes[k] / total, mc = part * l / (l + 4 * r / 3), ms = part - mc,
             along_ = mc * r * r / 2 + .4 * ms * r * r,
             across =
                 mc * (3 * r * r + l * l) / 12 + ms * (.4 * r * r + .375 * l * r + .25 * l * l);
      auto axis = rotate(
          {m->geom_quat[4 * g],
           m->geom_quat[4 * g + 1],
           m->geom_quat[4 * g + 2],
           m->geom_quat[4 * g + 3]},
          {0, 0, 1});
      Vec3 offset{
          m->geom_pos[3 * g] - com[0],
          m->geom_pos[3 * g + 1] - com[1],
          m->geom_pos[3 * g + 2] - com[2]};
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          tensor[3 * i + j] += (i == j ? across : 0) + (along_ - across) * axis[i] * axis[j] +
                               part * ((i == j ? dot(offset, offset) : 0) - offset[i] * offset[j]);
    }
    mjtNum values[3], vectors[9], orientation[4];
    mju_eig3(values, vectors, orientation, tensor);
    for (int j = 0; j < 3; ++j) {
      m->body_inertia[3 * body + j] = values[j];
      m->body_ipos[3 * body + j]    = com[j];
    }
    for (int j = 0; j < 4; ++j)
      m->body_iquat[4 * body + j] = orientation[j];
  }

  if (!p.constants)
    p.constants.reset(mj_makeData(m));
  if (!p.constants)
    throw std::runtime_error("cannot allocate dynamics constants");
  mj_setConst(m, p.constants.get());
  mj_forward(m, d);
  // Initialize exogenous arrivals above existing fruit instead of creating
  // penetrating bodies and artificial collision energy. Arrival timing and
  // horizontal sampling remain independent of the robot and queue.
  for (std::size_t pass = 0; pass < p.slots.size(); ++pass) {
    bool raised = false;
    for (auto [other, other_slot] : p.slots) {
      if (other == id)
        continue;
      auto a = fruit_bounds(m, d, body), b = fruit_bounds(m, d, p.bodies[other_slot]);
      if (a[0] < b[3] + .002 && a[3] > b[0] - .002 && a[1] < b[4] + .002 && a[4] > b[1] - .002 &&
          a[2] < b[5] + .002 && a[5] > b[2] - .002) {
        d->qpos[at + 2] += b[5] + .002 - a[2];
        mj_forward(m, d);
        raised = true;
      }
    }
    if (!raised)
      break;
  }
}
void MujocoStreamDynamics::remove(BodyId id) {
  auto& p    = *impl_;
  int   slot = p.slot(id), body = p.bodies[slot];
  p.model->body_gravcomp[body] = 1;
  p.model->body_contype[body] = p.model->body_conaffinity[body] = 0;
  for (int k = 0; k < 5; ++k) {
    int g                    = p.model->body_geomadr[body] + k;
    p.model->geom_contype[g] = p.model->geom_conaffinity[g] = 0;
  }
  int at = p.addresses[slot], v = p.velocities[slot];
  p.data->qpos[at]     = 10 + slot;
  p.data->qpos[at + 1] = 0;
  p.data->qpos[at + 2] = -10;
  for (int k = 0; k < 6; ++k)
    p.data->qvel[v + k] = 0;
  p.slots.erase(id);
  mj_forward(p.model.get(), p.data.get());
}
void MujocoStreamDynamics::step(const std::array<double, 6>& target, double finger) {
  auto& p   = *impl_;
  auto* m   = p.model.get();
  auto* d   = p.data.get();
  p.ratio   = 0;
  int count = int(std::lround((1. / 240) / m->opt.timestep));
  for (int k = 0; k < count; ++k) {
    mj_step1(m, d);
    for (int c = 0; c < d->ncon; ++c)
      if (d->contact[c].efc_address >= 0)
        p.penetration = std::max(p.penetration, -d->contact[c].dist);
    conveyor_reference(m, d, p.belt);
    for (int i = 0; i < 8; ++i) {
      double goal = i < 6 ? target[i] : i == 6 ? finger : -finger;
      int    v = p.dofs[i], a = p.actuators[i];
      double velocity =
          std::clamp((goal - p.previous_target[i]) * 240., i < 6 ? -1.5 : -.08, i < 6 ? 1.5 : .08);
      m->actuator_biasprm[mjNBIAS * a] = i < 6 ? d->qfrc_bias[v] : 0;
      const double position_gain       = i < 3 ? 9 : i < 6 ? 12.5 : 10;
      d->ctrl[a]                       = std::clamp(
          position_gain * (goal - d->qpos[p.positions[i]]) + velocity,
          i < 6 ? -1.5 : -.08,
          i < 6 ? 1.5 : .08);
    }
    mj_step2(m, d);
    for (int i = 0; i < 8; ++i) {
      auto& maximum = i < 6 ? p.joint_speed : p.finger_speed;
      maximum       = std::max(maximum, std::abs(d->qvel[p.dofs[i]]));
    }
    for (int i = 0; i < 8; ++i) {
      int a = p.actuators[i];
      p.ratio =
          std::max(p.ratio, std::abs(d->actuator_force[a]) / m->actuator_forcerange[2 * a + 1]);
    }
    for (int warning = 0; warning < mjNWARNING; ++warning)
      if (d->warning[warning].number)
        throw std::runtime_error(
            "MuJoCo warning " + std::to_string(warning) + " at t=" + std::to_string(d->time) +
            " detail=" + std::to_string(d->warning[warning].lastinfo));
  }
  for (int i = 0; i < 8; ++i)
    p.previous_target[i] = i < 6 ? target[i] : i == 6 ? finger : -finger;
  mj_forward(m, d);
  conveyor_reference(m, d, p.belt);
  mj_fwdConstraint(m, d);
  p.peak_ratio = std::max(p.peak_ratio, p.ratio);
  for (int c = 0; c < d->ncon; ++c)
    if (d->contact[c].efc_address >= 0)
      p.penetration = std::max(p.penetration, -d->contact[c].dist);
}
void MujocoStreamDynamics::observe(PickObservation& a) const {
  a.measured_joints = true;
  auto& p           = *impl_;
  for (int i = 0; i < 6; ++i)
    a.joints[i] = p.data->qpos[p.positions[i]];
  a.finger = .5 * (p.data->qpos[p.positions[6]] - p.data->qpos[p.positions[7]]);
}
cad::Vec3d MujocoStreamDynamics::tool_point() const {
  auto& p  = *impl_;
  int   id = mj_name2id(p.model.get(), mjOBJ_BODY, "end_link");
  auto* at = p.data->xpos + 3 * id;
  auto* r  = p.data->xmat + 9 * id;
  return {at[0] - .02 * r[0], at[1] - .02 * r[3], at[2] - .02 * r[6]};
}
Body MujocoStreamDynamics::body(BodyId id, Body b) const {
  auto& p = *impl_;
  int   s = p.slot(id), a = p.addresses[s];
  auto* d    = p.data.get();
  b.position = {d->qpos[a], d->qpos[a + 2], d->qpos[a + 1]};
  b.rotation = converted({d->qpos[a + 3], d->qpos[a + 4], d->qpos[a + 5], d->qpos[a + 6]});
  mjtNum v[6];
  mj_objectVelocity(p.model.get(), d, mjOBJ_BODY, p.bodies[s], v, 0);
  b.linear_velocity = {
      d->qvel[p.velocities[s]], d->qvel[p.velocities[s] + 2], d->qvel[p.velocities[s] + 1]};
  b.angular_velocity = {-v[0], -v[2], -v[1]};
  b.motion           = Motion::Dynamic;
  b.allowed_rotation = {true, true, true};
  b.mass             = p.model->body_mass[p.bodies[s]];
  b.friction         = .8;
  return b;
}
double MujocoStreamDynamics::grip_force(BodyId id) const {
  auto&  p    = *impl_;
  int    body = p.bodies[p.slot(id)];
  double sides[2]{};
  for (int c = 0; c < p.data->ncon; ++c) {
    const auto& co = p.data->contact[c];
    int         a = p.model->geom_bodyid[co.geom[0]], b = p.model->geom_bodyid[co.geom[1]];
    for (int i = 0; i < 2; ++i)
      if ((a == body && b == p.fingers[i]) || (b == body && a == p.fingers[i])) {
        mjtNum force[6];
        mj_contactForce(p.model.get(), p.data.get(), c, force);
        sides[i] += std::max(0., force[0]);
      }
  }
  return std::min(sides[0], sides[1]);
}
bool MujocoStreamDynamics::gripped(BodyId id) const {
  return grip_force(id) > .05;
}
bool MujocoStreamDynamics::bin_supported(BodyId id) const {
  auto& p    = *impl_;
  int   body = p.bodies[p.slot(id)];
  for (int c = 0; c < p.data->ncon; ++c) {
    const auto& co = p.data->contact[c];
    for (int k = 0; k < 2; ++k)
      if (p.model->geom_bodyid[co.geom[k]] == body &&
          std::find(p.floors.begin(), p.floors.end(), co.geom[1 - k]) != p.floors.end() &&
          (k == 1 ? co.frame[2] : -co.frame[2]) > .5) {
        mjtNum force[6];
        mj_contactForce(p.model.get(), p.data.get(), c, force);
        if (force[0] > .05)
          return true;
      }
  }
  return false;
}
double MujocoStreamDynamics::effort_ratio() const {
  return impl_->ratio;
}
std::array<double, 4> MujocoStreamDynamics::diagnostics() const {
  return {impl_->penetration, impl_->peak_ratio, impl_->joint_speed, impl_->finger_speed};
}
#else
/// @brief Empty storage keeps unsupported builds link-compatible and fail closed.
struct MujocoStreamDynamics::Impl {};
MujocoStreamDynamics::MujocoStreamDynamics(const std::string&, const World&, double, std::size_t) {
  throw std::runtime_error("articulated workcell requires DATAMUNGE_ENABLE_MUJOCO=ON");
}
MujocoStreamDynamics::~MujocoStreamDynamics() = default;
void       MujocoStreamDynamics::reset(const std::array<double, 6>&, double) {}
void       MujocoStreamDynamics::spawn(BodyId, const ConveyorObservation&) {}
void       MujocoStreamDynamics::remove(BodyId) {}
void       MujocoStreamDynamics::step(const std::array<double, 6>&, double) {}
void       MujocoStreamDynamics::observe(PickObservation&) const {}
cad::Vec3d MujocoStreamDynamics::tool_point() const {
  return {};
}
Body MujocoStreamDynamics::body(BodyId, Body b) const {
  return b;
}
bool MujocoStreamDynamics::gripped(BodyId) const {
  return false;
}
double MujocoStreamDynamics::grip_force(BodyId) const {
  return 0;
}
bool MujocoStreamDynamics::bin_supported(BodyId) const {
  return false;
}
double MujocoStreamDynamics::effort_ratio() const {
  return 0;
}
std::array<double, 4> MujocoStreamDynamics::diagnostics() const {
  return {};
}
#endif
} // namespace datamunge::sim::detail

namespace datamunge::sim {
std::string articulated_physics_version() {
#ifdef DATAMUNGE_ENABLE_MUJOCO
  return mj_versionString();
#else
  return {};
#endif
}
} // namespace datamunge::sim
