#include <cmath>
#include <datamunge/cad/cad.hpp>
#include <datamunge/cad/view.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
using namespace datamunge::cad;
namespace {
int  checks = 0;
void check(bool value, const char* expression, int line) {
  ++checks;
  if (!value)
    throw std::runtime_error(std::string(expression) + " at " + std::to_string(line));
}
#define CHECK(x) check(bool(x), #x, __LINE__)
bool near(double a, double b, double tolerance = 1e-6) {
  return std::abs(a - b) < tolerance;
}
template<class F>
bool fails(F f) {
  try {
    f();
    return false;
  } catch (const std::exception&) {
    return true;
  }
}
void write(const std::string& path, const std::string& text) {
  std::ofstream(path) << text;
}
} // namespace
int main() {
  std::filesystem::create_directories("cad-api-artifacts");
  const auto box = Solid::box(2, 3, 4);
  CHECK(box.valid());
  CHECK(near(box.volume(), 24));
  CHECK(box.faces().size() == 6);
  CHECK(box.edges().size() == 12);
  BodyMass    mass;
  bool        closed = false;
  std::string mass_error;
  CHECK(ComputeBodyMass(box.model(), box.body(), .01, &mass, &closed, &mass_error));
  CHECK(closed);
  CHECK(near(mass.volume, 24));
  CHECK(near(mass.area, 52));
  CHECK(near(mass.centroid.x, 1));
  CHECK(near(mass.centroid.y, 1.5));
  CHECK(near(mass.centroid.z, 2));
  CHECK(near(mass.inertia[0][0], 50));
  CHECK(near(mass.inertia[1][1], 40));
  CHECK(near(mass.inertia[2][2], 26));
  auto moved = box.translated(.5, .5, .5);
  CHECK(near(box.united(moved).volume(), 34.875));
  CHECK(near(box.intersected(moved).volume(), 13.125));
  CHECK(near(box.subtracted(moved).volume(), 10.875));
  CHECK(near(box.volume(), 24));
  CHECK(near(box.transformed(Mat4d::Scaling({2, 3, 4})).volume(), 576));
  CHECK(near(box.rotated(0, 0, 1, kHalfPi).volume(), 24));
  CHECK(Solid::sphere(1).valid());
  CHECK(near(Solid::sphere(1).volume(.005), 4 * kPi / 3, .06));
  CHECK(Solid::cylinder(1, 2).valid());
  CHECK(near(Solid::cylinder(1, 2).volume(.005), 2 * kPi, .06));
  CHECK(Solid::torus(2, .5).valid());
  auto bevel = box.chamfered({box.edges().front()}, .1);
  CHECK(bevel.valid());
  CHECK(bevel.volume() < 24);
  auto fillet = box.filleted({box.edges().front()}, .1);
  CHECK(fillet.valid());
  CHECK(fillet.volume() < 24);
  auto shell = Solid::box(4, 4, 4).shelled(.2);
  CHECK(shell.valid());
  CHECK(near(shell.volume(), 64 - 3.6 * 3.6 * 3.6, 1e-6));
  CHECK(fails([] { Solid::box(0, 1, 1); }));
  CHECK(fails([] { Solid::sphere(NAN); }));
  CHECK(fails([] { Solid::torus(1, 2); }));
  CHECK(fails([&] { box.tessellate(-1); }));
  CHECK(fails([&] { box.rotated(0, 0, 0, 1); }));
  for (auto extension : {"step", "stl", "obj", "gltf"}) {
    auto path = "cad-api-artifacts/box." + std::string(extension);
    box.save(path);
    auto scene = load_scene(path);
    CHECK(scene.objects.size() == 1);
    CHECK(scene.TotalTriangleCount() > 0);
    CHECK(std::isfinite(scene.meshes[0].positions[0]));
  }
  box.save("cad-api-artifacts/box.svg");
  CHECK(std::filesystem::file_size("cad-api-artifacts/box.svg") > 100);
  Scene    scene;
  MeshData mesh;
  mesh.positions = {-1, -1, 0, 1, -1, 0, 0, 1, 0};
  mesh.indices   = {0, 1, 2};
  mesh.normals   = {0, 0, 1, 0, 0, 1, 0, 0, 1};
  mesh.texcoords = {0, 0, 1, 0, .5f, 1};
  scene.meshes.push_back(mesh);
  Object3D object;
  object.mesh_index = 0;
  object.color      = {1, 0, 0, 1};
  int    id         = scene.AddObject(object);
  Camera camera;
  camera.position = {0, 0, 3};
  camera.target   = {0, 0, 0};
  camera.up       = {0, 1, 0};
  RenderOptions options;
  options.width   = 64;
  options.height  = 64;
  options.style   = RenderStyle::Unlit;
  options.threads = 1;
  auto image      = render(scene, camera, options);
  auto centre     = 32 * 64 + 32;
  CHECK(image.rgba[centre * 4] == 255);
  CHECK(image.rgba[centre * 4 + 1] == 0);
  CHECK(image.object_ids[centre] == id);
  CHECK(near(image.depth[centre], 3, .01));
  CHECK(image.object_ids[0] == -1);
  CHECK(std::isinf(image.depth[0]));
  options.threads = 4;
  auto parallel   = render(scene, camera, options);
  CHECK(parallel.rgba == image.rgba);
  CHECK(parallel.depth == image.depth);
  CHECK(parallel.object_ids == image.object_ids);
  auto hit = pick(scene, {{0, 0, 3}, {0, 0, -2}});
  CHECK(hit.hit && hit.object_id == id);
  CHECK(near(hit.distance, 3));
  CHECK(near(hit.normal.z, 1));
  scene.textures.push_back({"green", 1, 1, {0, 255, 0, 255}});
  scene.objects[0].color         = {1, 1, 1, 1};
  scene.objects[0].texture_index = 0;
  auto textured                  = render(scene, camera, options);
  CHECK(textured.rgba[centre * 4] == 0);
  CHECK(textured.rgba[centre * 4 + 1] == 255);
  scene.objects[0].texture_index = -1;
  scene.objects[0].color         = {1, 0, 0, .5f};
  auto translucent               = render(scene, camera, options);
  CHECK(translucent.rgba[centre * 4 + 1] > 50);
  CHECK(translucent.rgba[centre * 4] > 200);
  scene.objects[0].color = {1, 0, 0, 0};
  CHECK(render(scene, camera, options).object_ids[centre] == -1);
  for (auto ext : {"png", "jpg", "ppm"}) {
    auto path = "cad-api-artifacts/triangle." + std::string(ext);
    image.save(path);
    CHECK(std::filesystem::file_size(path) > 100);
  }
  auto bad                 = scene;
  bad.objects[0].color.a   = 1;
  bad.meshes[0].indices[0] = 100;
  CHECK(fails([&] { render(bad, camera, options); }));
  bad                        = scene;
  bad.meshes[0].positions[0] = NAN;
  bad.objects[0].color.a     = 1;
  CHECK(fails([&] { render(bad, camera, options); }));
  camera.up = {0, 0, 1};
  CHECK(fails([&] { render(scene, camera, options); }));
  camera.up           = {0, 1, 0};
  camera.projection   = Projection::Orthographic;
  camera.ortho_height = 3;
  CHECK(near(camera.ray(0, 0, 64, 64).direction.z, -1));
  // Imported STEP cones must invert near the apex even over a wide slant domain.
  ConeSurface cone({0, 0, 0}, {1, 0, 0}, {0, 0, 1}, kPi / 4, 0, kTwoPi, 0, 1e5);
  auto        point = cone.Point(.7, 2);
  double      u = 0, v = 0;
  Vec3d       projected;
  CHECK(cone.ClosestPoint(point, &u, &v, &projected));
  CHECK((projected - point).Length() < 1e-7);
  // URDF kinematics oracle: a 90-degree joint rotates a translated box centre
  // from (1,0,0) to (0,1,0), with an independently checked mimic joint.
  write(
      "cad-api-artifacts/robot.urdf",
      R"(<robot name="test"><link name="base"/><link name="arm"><visual><origin xyz="1 0 0"/><geometry><box size=".2 .2 .2"/></geometry><material><color rgba="1 0 0 1"/></material></visual></link><link name="finger"/><joint name="turn" type="revolute"><parent link="base"/><child link="arm"/><axis xyz="0 0 1"/><limit lower="-3.2" upper="3.2"/></joint><joint name="follow" type="revolute"><parent link="arm"/><child link="finger"/><axis xyz="0 0 1"/><limit lower="-3.2" upper="3.2"/><mimic joint="turn" multiplier="-1"/></joint></robot>)");
  auto robot = Robot::load_urdf("cad-api-artifacts/robot.urdf");
  CHECK(robot.joint_names().size() == 1);
  auto copy = robot;
  robot.set_joint("turn", kHalfPi);
  CHECK(near(robot.joint_position("follow"), -kHalfPi));
  CHECK(near(copy.joint_position("turn"), 0));
  auto location = robot.link_transform("arm").TransformPoint({1, 0, 0});
  CHECK(near(location.x, 0) && near(location.y, 1));
  CHECK(robot.scene().objects.size() == 1);
  CHECK(fails([&] { robot.set_joint("turn", 4); }));
  CHECK(near(robot.joint_position("turn"), kHalfPi));
  CHECK(fails([&] { robot.set_joint("follow", 0); }));
  CHECK(fails([&] { robot.link_transform("absent"); }));
  write(
      "cad-api-artifacts/cycle.urdf",
      R"(<robot><link name="a"/><link name="b"/><joint name="x" type="fixed"><parent link="a"/><child link="b"/></joint><joint name="y" type="fixed"><parent link="b"/><child link="a"/></joint></robot>)");
  CHECK(fails([] { Robot::load_urdf("cad-api-artifacts/cycle.urdf"); }));
  View view;
  int  selected = view.add(box);
  view.size(128, 96);
  view.samples(2);
  view.light(1, -2, 3);
  CHECK(view.object_count() == 1 && view.triangle_count() == 12);
  view.render("cad-api-artifacts/view.png");
  CHECK(view.pick_object(64, 48) == selected);
  view.save_gltf("cad-api-artifacts/view.gltf");
  CHECK(View::from_file("cad-api-artifacts/view.gltf").object_count() == 1);
  auto rv = View::from_urdf("cad-api-artifacts/robot.urdf");
  rv.set_joint("turn", 1);
  CHECK(rv.joint_names().size() == 1);
  CHECK(fails([&] { view.set_joint("turn", 0); }));
  std::cout << "CAD API: " << checks << " independent checks passed\n";
}
