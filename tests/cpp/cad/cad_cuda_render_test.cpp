/// @file cad_cuda_render_test.cpp
/// @brief Real CUDA rendering regressions with explicit unavailable-device diagnostics.
#include <cmath>
#include <dm/dm.hpp>
#include <future>
#include <iostream>
#include <stdexcept>
using namespace datamunge::cad;
namespace {
/// @brief Require an invariant in both Release and Debug builds.
void check(bool value, const char* message) {
  if (!value)
    throw std::runtime_error(message);
}
/// @brief Compare bounded image errors and centre-ray geometry against the CPU reference.
void compare(const RenderResult& cpu, const RenderResult& gpu) {
  check(cpu.width == gpu.width && cpu.height == gpu.height, "shape mismatch");
  double      error = 0;
  std::size_t ids   = 0;
  double      depth = 0;
  for (std::size_t i = 0; i < cpu.depth.size(); ++i) {
    ids += cpu.object_ids[i] != gpu.object_ids[i];
    if (cpu.object_ids[i] >= 0 && cpu.object_ids[i] == gpu.object_ids[i])
      depth = std::max(depth, std::abs(cpu.depth[i] - gpu.depth[i]));
    for (int k = 0; k < 3; ++k) {
      double d = int(cpu.rgba[4 * i + k]) - int(gpu.rgba[4 * i + k]);
      error += d * d;
    }
  }
  double rmse = std::sqrt(error / (cpu.depth.size() * 3));
  std::cout << "RGB RMSE=" << rmse << " ID differences=" << ids << " max depth=" << depth << '\n';
  check(rmse < 3, "excessive CUDA color error");
  check(ids < cpu.depth.size() * .002 + 1, "excessive CUDA ID error");
  check(depth < .002, "excessive CUDA depth error");
}
} // namespace
int main() {
  try {
    auto device = dm::compute::probe({dm::compute::Backend::CUDA, 0});
    if (!device.available) {
      bool failed = false;
      try {
        CudaRenderer gpu;
      } catch (const std::runtime_error&) {
        failed = true;
      }
      check(failed, "unavailable CUDA must fail explicitly");
      std::cout << "CUDA rendering unavailable: " << device.diagnostic << '\n';
      return 0;
    }
    std::cout << device.name << '\n';
    CudaRenderer gpu;
    Scene        base;
    int          floor               = AddPrimitiveToScene(&base, PrimitiveKind::Cube);
    base.FindObject(floor)->position = {0, 0, -.7f};
    base.FindObject(floor)->scale    = {6, 6, .2f};
    int cube                         = AddPrimitiveToScene(&base, PrimitiveKind::Cube);
    base.FindObject(cube)->position  = {-1, 0, .1f};
    base.FindObject(cube)->scale     = {-.6f, .8f, 1.2f};
    base.textures.push_back(
        {"color",
         2,
         2,
         {100, 180, 70, 255, 150, 220, 90, 255, 80, 160, 100, 255, 130, 210, 60, 255}});
    base.textures.push_back({"normal", 1, 1, {128, 128, 255, 255}});
    base.textures.push_back({"rough", 1, 1, {150, 150, 150, 255}});
    base.textures.push_back({"metal", 1, 1, {20, 20, 20, 255}});
    auto& m               = *base.FindObject(cube);
    m.texture_index       = 0;
    m.normal_map_index    = 1;
    m.roughness_map_index = 2;
    m.metallic_map_index  = 3;
    Light key;
    key.direction = {.5f, -1, 2};
    base.AddLight(key);
    Light point;
    point.type     = LightType::Point;
    point.position = {2, -3, 4};
    base.AddLight(point);
    Camera camera;
    camera.position = {3, -5, 3};
    camera.target   = {0, 0, 0};
    RenderOptions o;
    o.width                     = 128;
    o.height                    = 96;
    o.threads                   = 4;
    o.physically_based          = true;
    o.shadow_samples            = 3;
    o.light_angular_radius      = .07;
    o.ambient_occlusion_samples = 2;
    PreparedScene prepared(base);
    Scene         actors;
    auto          sphere                = AddPrimitiveToScene(&actors, PrimitiveKind::Sphere);
    actors.FindObject(sphere)->position = {1, 0, 0};
    actors.FindObject(sphere)->scale    = {.6f, .6f, .6f};
    actors.textures.push_back({"actor", 1, 1, {220, 100, 30, 255}});
    actors.FindObject(sphere)->texture_index = 0;
    for (auto style :
         {RenderStyle::Shaded,
          RenderStyle::Unlit,
          RenderStyle::Wireframe,
          RenderStyle::ShadedWireframe}) {
      o.style = style;
      compare(prepared.render(actors, camera, o), gpu.render(prepared, actors, camera, o));
    }
    o.style              = RenderStyle::Shaded;
    o.studio_environment = .7;
    compare(prepared.render(actors, camera, o), gpu.render(prepared, actors, camera, o));
    o.studio_environment = 0;
    o.style              = RenderStyle::Shaded;
    o.samples            = 2;
    compare(prepared.render(actors, camera, o), gpu.render(prepared, actors, camera, o));
    o.samples          = 1;
    o.physically_based = false;
    camera.projection  = Projection::Orthographic;
    compare(prepared.render(actors, camera, o), gpu.render(prepared, actors, camera, o));
    actors.FindObject(sphere)->color.a  = .4f;
    actors.FindObject(sphere)->position = {.3f, .2f, .1f};
    compare(prepared.render(actors, camera, o), gpu.render(prepared, actors, camera, o));
    compare(prepared.render(camera, o), gpu.render(prepared, {}, camera, o));
    auto pointbase = base;
    pointbase.lights.erase(pointbase.lights.begin());
    PreparedScene points(pointbase);
    compare(points.render(camera, o), gpu.render(points, {}, camera, o));
    Scene         empty;
    PreparedScene blank(empty);
    compare(blank.render(camera, o), gpu.render(blank, {}, camera, o));
    auto a =
        std::async(std::launch::async, [&] { return gpu.render(prepared, actors, camera, o); });
    auto b =
        std::async(std::launch::async, [&] { return gpu.render(prepared, actors, camera, o); });
    check(a.get().rgba == b.get().rgba, "serialized concurrent CUDA calls differ");
    auto bad    = o;
    bad.width   = 0;
    bool failed = false;
    try {
      gpu.render(prepared, {}, camera, bad);
    } catch (const std::invalid_argument&) {
      failed = true;
    }
    check(failed, "invalid options accepted");
    auto far       = camera;
    far.position.x = 1e100;
    failed         = false;
    try {
      gpu.render(prepared, {}, far, o);
    } catch (const std::invalid_argument&) {
      failed = true;
    }
    check(failed, "unrepresentable camera accepted");
    compare(
        prepared.render(actors, camera, o),
        gpu.render(prepared, actors, camera, o)); // Failed replacement must not poison cache.
    auto huge                    = base;
    huge.FindObject(cube)->scale = {1e30f, 1e30f, 1e30f};
    PreparedScene overflow(huge);
    failed = false;
    try {
      gpu.render(overflow, {}, camera, o);
    } catch (const std::invalid_argument&) {
      failed = true;
    }
    check(failed, "unrepresentable geometry accepted");
    compare(prepared.render(actors, camera, o), gpu.render(prepared, actors, camera, o));
    std::cout << "CUDA rendering, maps, styles, sampling, transparency, point lights, empty "
                 "scenes, caching, concurrency and validation passed\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
