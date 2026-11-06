#include <algorithm>
#include <cctype>
#include <cmath>
#include <datamunge/cad/blend.hpp>
#include <datamunge/cad/doc.hpp>
#include <datamunge/cad/pcurve.hpp>
#include <datamunge/cad/solid.hpp>
#include <datamunge/cad/validate.hpp>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <stdexcept>

namespace datamunge::cad {
namespace {
void positive(double v, const char* name) {
  if (!std::isfinite(v) || v <= 0)
    throw std::invalid_argument(std::string(name) + " must be positive and finite");
}
std::string extension(const std::string& path) {
  auto e = std::filesystem::path(path).extension().string();
  std::transform(e.begin(), e.end(), e.begin(), [](unsigned char c) { return std::tolower(c); });
  return e;
}
std::string read_bytes(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f)
    throw std::runtime_error("Cannot open CAD input: " + path);
  std::string bytes{std::istreambuf_iterator<char>(f), {}};
  if (f.bad())
    throw std::runtime_error("Cannot read CAD input: " + path);
  return bytes;
}
void write_bytes(const std::string& path, const std::string& bytes) {
  std::ofstream f(path, std::ios::binary);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  f.close();
  if (!f)
    throw std::runtime_error("Cannot write CAD output: " + path);
}
Solid finish(Model model, EntityId body) {
  std::string error;
  if (!BuildAllPCurves(&model, {}, &error))
    throw std::runtime_error(error);
  return Solid(std::move(model), body);
}
} // namespace
Solid::Solid(Model model, EntityId body) : model_(std::move(model)), body_(body) {
  if (!model_.GetBody(body_))
    throw std::invalid_argument("Solid requires an existing body");
}
Solid Solid::box(double w, double d, double h) {
  positive(w, "width");
  positive(d, "depth");
  positive(h, "height");
  Model    m;
  EntityId b;
  if (!MakeBox({0, 0, 0}, {w, d, h}, &m, &b))
    throw std::runtime_error("Box construction failed");
  return finish(std::move(m), b);
}
Solid Solid::cylinder(double r, double h) {
  positive(r, "radius");
  positive(h, "height");
  Model    m;
  EntityId b;
  if (!MakeCylinder({0, 0, 0}, {0, 0, 1}, r, h, &m, &b))
    throw std::runtime_error("Cylinder construction failed");
  return finish(std::move(m), b);
}
Solid Solid::sphere(double r) {
  positive(r, "radius");
  Model    m;
  EntityId b;
  if (!MakeSphere({0, 0, 0}, r, &m, &b))
    throw std::runtime_error("Sphere construction failed");
  return finish(std::move(m), b);
}
Solid Solid::torus(double r, double t) {
  positive(r, "major_radius");
  positive(t, "minor_radius");
  if (t >= r)
    throw std::invalid_argument("Tube radius must be smaller than major radius");
  Model    m;
  EntityId b;
  if (!MakeTorus({0, 0, 0}, {0, 0, 1}, r, t, &m, &b))
    throw std::runtime_error("Torus construction failed");
  return finish(std::move(m), b);
}
Solid Solid::boolean_with(const Solid& other, BooleanOp op) const {
  Model         m;
  BooleanReport report;
  if (!BooleanOperation(model_, body_, other.model_, other.body_, op, &m, &report))
    throw std::runtime_error(report.error);
  if (report.body == kNoEntity)
    throw std::runtime_error("Boolean result contains no solid");
  return finish(std::move(m), report.body);
}
Solid Solid::united(const Solid& other) const {
  return boolean_with(other, BooleanOp::Union);
}
Solid Solid::subtracted(const Solid& other) const {
  return boolean_with(other, BooleanOp::Difference);
}
Solid Solid::intersected(const Solid& other) const {
  return boolean_with(other, BooleanOp::Intersection);
}
Solid Solid::transformed(const Mat4d& transform) const {
  for (const auto& row : transform.m)
    for (double value : row)
      if (!std::isfinite(value))
        throw std::invalid_argument("Nonfinite transform");
  Model       m;
  EntityId    b;
  std::string error;
  if (!TransformBody(model_, body_, transform, &m, &b, &error))
    throw std::runtime_error(error);
  return finish(std::move(m), b);
}
Solid Solid::translated(double x, double y, double z) const {
  return transformed(Mat4d::Translation({x, y, z}));
}
Solid Solid::rotated(double x, double y, double z, double radians) const {
  Vec3d axis{x, y, z};
  if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) || !std::isfinite(radians) ||
      axis.Length() == 0)
    throw std::invalid_argument("Rotation requires finite nonzero axis and finite angle");
  return transformed(Mat4d::Rotation(axis, radians));
}
Solid Solid::filleted(const std::vector<int>& edges, double radius) const {
  positive(radius, "radius");
  if (edges.empty())
    return *this;
  BlendOptions o;
  o.radius = radius;
  Model       m;
  EntityId    b;
  std::string error;
  if (!BlendEdges(model_, body_, edges, o, &m, &b, &error))
    throw std::runtime_error(error);
  return finish(std::move(m), b);
}
Solid Solid::chamfered(const std::vector<int>& edges, double setback) const {
  positive(setback, "setback");
  if (edges.empty())
    return *this;
  BlendOptions o;
  o.radius = setback;
  o.kind   = BlendKind::Chamfer;
  Model       m;
  EntityId    b;
  std::string error;
  if (!BlendEdges(model_, body_, edges, o, &m, &b, &error))
    throw std::runtime_error(error);
  return finish(std::move(m), b);
}
Solid Solid::shelled(double thickness, const std::vector<int>& open_faces) const {
  positive(thickness, "thickness");
  Model       m;
  EntityId    b;
  std::string error;
  if (!ShellBody(model_, body_, open_faces, thickness, {}, &m, &b, &error))
    throw std::runtime_error(error);
  return finish(std::move(m), b);
}
TessellationMesh Solid::tessellate(double chord) const {
  positive(chord, "chord");
  TessellationOptions o;
  o.chord_tolerance = chord;
  TessellationMesh mesh;
  std::string      error;
  if (!TessellateBody(model_, body_, o, &mesh, &error))
    throw std::runtime_error(error);
  return mesh;
}
double Solid::volume(double chord) const {
  return tessellate(chord).SignedVolume();
}
bool Solid::valid() const {
  ValidationReport r;
  return ValidateBody(model_, body_, &r);
}
std::vector<int> Solid::faces() const {
  std::vector<int> r;
  for (auto shell : model_.GetBody(body_)->shells) {
    auto* s = model_.GetShell(shell);
    r.insert(r.end(), s->faces.begin(), s->faces.end());
  }
  return r;
}
std::vector<int> Solid::edges() const {
  std::set<int> r;
  for (int face : faces())
    for (int edge : model_.EdgesOfFace(face))
      r.insert(edge);
  return {r.begin(), r.end()};
}
void Solid::save(const std::string& path, double chord) const {
  positive(chord, "chord");
  auto              ext = extension(path);
  std::string       bytes, error;
  bool              ok = false;
  MeshExportOptions mesh;
  mesh.tessellation.chord_tolerance = chord;
  mesh.name                         = "datamunge";
  if (ext == ".step" || ext == ".stp") {
    StepWriteOptions o;
    o.author      = "datamunge";
    o.description = "CAD export";
    o.product     = "Part";
    ok            = WriteStepShapes(model_, {body_}, &bytes, &error, o);
  } else if (ext == ".stl")
    ok = WriteStl(model_, {body_}, &bytes, &error, mesh);
  else if (ext == ".obj")
    ok = WriteObj(model_, {body_}, &bytes, &error, mesh);
  else if (ext == ".gltf")
    ok = WriteGltf(model_, {body_}, &bytes, &error, mesh);
  else if (ext == ".svg") {
    DrawingView        view;
    DrawingViewOptions o;
    o.direction    = {1, -1, -.7};
    o.up           = {0, 0, 1};
    o.tessellation = mesh.tessellation;
    ok             = MakeDrawingView(model_, {body_}, o, &view, &error);
    if (ok)
      bytes = DrawingViewToSvg(view, 180);
  } else
    throw std::invalid_argument("Unsupported solid output extension: " + ext);
  if (!ok)
    throw std::runtime_error(error);
  write_bytes(path, bytes);
}
int add_solid(
    Scene&             scene,
    const Solid&       solid,
    RgbaColorF         color,
    const std::string& name,
    double             chord) {
  for (float c : {color.r, color.g, color.b, color.a})
    if (!std::isfinite(c) || c < 0 || c > 1)
      throw std::invalid_argument("Color must be RGBA in [0,1]");
  auto     tess = solid.tessellate(chord);
  MeshData mesh;
  mesh.name = name;
  for (auto p : tess.positions) {
    mesh.positions.push_back(static_cast<float>(p.x));
    mesh.positions.push_back(static_cast<float>(p.y));
    mesh.positions.push_back(static_cast<float>(p.z));
  }
  for (auto p : tess.normals) {
    mesh.normals.push_back(static_cast<float>(p.x));
    mesh.normals.push_back(static_cast<float>(p.y));
    mesh.normals.push_back(static_cast<float>(p.z));
  }
  mesh.indices.assign(tess.indices.begin(), tess.indices.end());
  Object3D object;
  object.name       = name;
  object.kind       = PrimitiveKind::Imported;
  object.color      = color;
  object.mesh_index = static_cast<int>(scene.meshes.size());
  scene.meshes.push_back(std::move(mesh));
  return scene.AddObject(object);
}
Scene load_scene(const std::string& path, double chord) {
  positive(chord, "chord");
  const auto            ext = extension(path);
  Scene                 scene;
  Model                 model;
  std::vector<EntityId> bodies;
  std::string           error;
  if (ext == ".step" || ext == ".stp") {
    StepReadReport r;
    if (!ReadStepText(read_bytes(path), &model, &r) || !r.unsupported.empty() ||
        !r.warnings.empty())
      throw std::runtime_error(
          "STEP import incomplete: " + r.error + (r.warnings.empty() ? "" : r.warnings.front()));
    bodies = r.bodies;
  } else if (ext == ".iges" || ext == ".igs") {
    IgesReadReport r;
    if (!ReadIgesText(read_bytes(path), &model, &r) || !r.unsupported.empty() ||
        !r.warnings.empty())
      throw std::runtime_error(
          "IGES import incomplete: " + r.error + (r.warnings.empty() ? "" : r.warnings.front()));
    bodies = r.bodies;
  } else if (ext == ".mepcad") {
    CadDocument d;
    if (!ReadCadDocument(read_bytes(path), &d, &error) || !d.tree.Rebuild(&error))
      throw std::runtime_error(error);
    model  = d.tree.Result();
    bodies = d.tree.Bodies();
  } else if (ext == ".stl") {
    TessellationMesh t;
    if (!ReadStl(read_bytes(path), &t, &error))
      throw std::runtime_error(error);
    MeshData mesh;
    mesh.name = std::filesystem::path(path).stem().string();
    for (auto p : t.positions) {
      mesh.positions.push_back(static_cast<float>(p.x));
      mesh.positions.push_back(static_cast<float>(p.y));
      mesh.positions.push_back(static_cast<float>(p.z));
    }
    mesh.indices.assign(t.indices.begin(), t.indices.end());
    mesh.RecalculateNormals();
    scene.meshes.push_back(std::move(mesh));
    Object3D obj;
    obj.mesh_index = 0;
    obj.kind       = PrimitiveKind::Imported;
    obj.name       = scene.meshes.front().name;
    scene.AddObject(obj);
  } else if (ext == ".blend") {
    if (!LoadBlendFile(path, &scene, &error))
      throw std::runtime_error(error);
  } else if (IsModel3DPath(path)) {
    if (!LoadModel3DFile(path, &scene, &error))
      throw std::runtime_error(error);
  } else
    throw std::invalid_argument("Unsupported scene input extension: " + ext);
  for (auto b : bodies)
    add_solid(scene, Solid(model, b), {.25f, .5f, .85f, 1}, model.GetBody(b)->name, chord);
  scene.source_path = path;
  return scene;
}
} // namespace datamunge::cad
