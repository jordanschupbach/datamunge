/// @file courier_offscreen.cpp
/// @brief Run the frozen native 3D Courier scene and save an optional Vulkan frame.
#include <cstdio>
#include <dm/dm.hpp>
#include <exception>

/// @brief Load explicitly mapped native behaviors, simulate one second and capture owned RGBA.
/// @param argc Must be three, including executable name.
/// @param argv Scene JSON and destination PNG paths; destination directory must exist.
/// @return Zero on success, 77 if Vulkan is unavailable, two for usage, one for other errors.
int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "Usage: %s <3d-courier-scene.json> <capture.png>\n", argv[0]);
    return 2;
  }
  try {
    const auto device = dm::compute::probe({dm::compute::Backend::Vulkan, 0});
    if (!device.available) {
      std::fprintf(stderr, "Vulkan unavailable: %s\n", device.diagnostic.c_str());
      return 77;
    }
    const auto           scene = dm::game::load_scene(argv[1], dm::game::courier_import(true));
    dm::game::Simulation game(scene, dm::game::courier_behaviors());
    for (int step = 0; step < 60; ++step)
      game.tick(1. / 60);
    dm::game::RenderConfig config;
    config.width  = 640;
    config.height = 400;
    dm::game::Renderer renderer(config);
    renderer.set_overlay("Crystal Courier 3D", game.scene().get(2).name, "Native headless capture");
    renderer.draw(game.scene());
    dm::image::write_png(renderer.capture(), argv[2]);
    std::printf(
        "Captured %ux%u after 60 native physics ticks\n", renderer.width(), renderer.height());
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
