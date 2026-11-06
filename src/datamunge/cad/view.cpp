#include <cmath>
#include <datamunge/cad/view.hpp>
#include <stdexcept>
namespace datamunge::cad {
namespace {
RgbaColorF hex(const std::string& s) {
  if ((s.size() != 7 && s.size() != 9) || s[0] != '#')
    throw std::invalid_argument("Expected #RRGGBB or #RRGGBBAA");
  auto digit = [](char c) {
    if (c >= '0' && c <= '9')
      return c - '0';
    if (c >= 'a' && c <= 'f')
      return c - 'a' + 10;
    if (c >= 'A' && c <= 'F')
      return c - 'A' + 10;
    throw std::invalid_argument("Invalid hex color");
  };
  RgbaColorF r;
  float*     values[] = {&r.r, &r.g, &r.b, &r.a};
  for (std::size_t i = 0; i < (s.size() - 1) / 2; ++i)
    *values[i] = (digit(s[1 + 2 * i]) * 16 + digit(s[2 + 2 * i])) / 255.f;
  return r;
}
} // namespace
View View::from_file(const std::string& path) {
  View v;
  v.scene_ = load_scene(path);
  return v;
}
View View::from_urdf(const std::string& path) {
  View v;
  v.robot_ = Robot::load_urdf(path);
  v.scene_ = v.robot_->scene();
  return v;
}
void View::set_joint(const std::string& name, double value) {
  if (!robot_)
    throw std::invalid_argument("View has no URDF model");
  auto next = *robot_;
  next.set_joint(name, value);
  auto s = next.scene();
  robot_ = std::move(next);
  scene_ = std::move(s);
}
std::vector<std::string> View::joint_names() const {
  return robot_ ? robot_->joint_names() : std::vector<std::string>{};
}
int View::add(const Solid& solid, const std::string& color, const std::string& name) {
  return add_solid(scene_, solid, hex(color), name);
}
int View::primitive(const std::string& kind) {
  PrimitiveKind k;
  if (kind == "cube")
    k = PrimitiveKind::Cube;
  else if (kind == "sphere")
    k = PrimitiveKind::Sphere;
  else if (kind == "cylinder")
    k = PrimitiveKind::Cylinder;
  else if (kind == "cone")
    k = PrimitiveKind::Cone;
  else if (kind == "plane")
    k = PrimitiveKind::Plane;
  else if (kind == "torus")
    k = PrimitiveKind::Torus;
  else if (kind == "wedge")
    k = PrimitiveKind::Wedge;
  else
    throw std::invalid_argument("Unknown primitive: " + kind);
  return AddPrimitiveToScene(&scene_, k);
}
void View::size(int w, int h) {
  if (w < 1 || h < 1 || w > 8192 || h > 8192)
    throw std::invalid_argument("Invalid render size");
  options_.width  = w;
  options_.height = h;
}
void View::samples(int n) {
  if (n < 1 || n > 4)
    throw std::invalid_argument("Samples must be 1..4");
  options_.samples = n;
}
void View::style(const std::string& name) {
  if (name == "shaded")
    options_.style = RenderStyle::Shaded;
  else if (name == "unlit")
    options_.style = RenderStyle::Unlit;
  else if (name == "wireframe")
    options_.style = RenderStyle::Wireframe;
  else if (name == "shaded-wireframe")
    options_.style = RenderStyle::ShadedWireframe;
  else
    throw std::invalid_argument("Unknown render style");
}
void View::shadows(bool enabled) {
  options_.shadows = enabled;
}
void View::camera(double x, double y, double z, double tx, double ty, double tz) {
  Camera c;
  c.position = {x, y, z};
  c.target   = {tx, ty, tz};
  c.ray(0, 0, 1, 1);
  camera_   = c;
  auto_fit_ = false;
}
void View::fit() {
  auto_fit_ = true;
}
Camera View::current_camera() const {
  return auto_fit_ && !scene_.objects.empty()
             ? fit_camera(scene_, {1, -1, .75}, {0, 0, 1}, double(options_.width) / options_.height)
             : camera_;
}
int View::light(double x, double y, double z, const std::string& color, double intensity) {
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || Vec3d{x, y, z}.Length() == 0 ||
      !std::isfinite(intensity) || intensity < 0)
    throw std::invalid_argument("Invalid light");
  Light l;
  l.direction = {float(x), float(y), float(z)};
  l.color     = hex(color);
  l.intensity = float(intensity);
  if (!std::isfinite(l.direction.x) || !std::isfinite(l.direction.y) ||
      !std::isfinite(l.direction.z) || !std::isfinite(l.intensity))
    throw std::invalid_argument("Light exceeds float range");
  return scene_.AddLight(l);
}
void View::render(const std::string& path) const {
  datamunge::cad::render(scene_, current_camera(), options_).save(path);
}
void View::save_gltf(const std::string& path) const {
  std::string error;
  if (!SaveModel3DGltf(scene_, path, &error))
    throw std::runtime_error(error);
}
int View::pick_object(double x, double y) const {
  auto c = current_camera();
  return datamunge::cad::pick(
             scene_, c.ray(x, y, options_.width, options_.height), c.near_clip, c.far_clip)
      .object_id;
}
int View::object_count() const {
  return static_cast<int>(scene_.objects.size());
}
int View::triangle_count() const {
  return scene_.TotalTriangleCount();
}
} // namespace datamunge::cad
