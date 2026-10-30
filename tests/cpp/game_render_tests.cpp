#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <datamunge/game/render.hpp>
#include <datamunge/image/png.hpp>
#include <filesystem>
namespace g = datamunge::game;
namespace c = datamunge::compute;
namespace {
g::Scene primitive(g::Primitive kind) {
  g::Scene  scene;
  g::Entity entity;
  entity.mesh               = g::Mesh{};
  entity.mesh->primitive    = kind;
  entity.mesh->color        = {.8, .03, .02};
  entity.spatial.dimensions = {2, 2, 2};
  scene.create(entity);
  return scene;
}
std::size_t red_pixels(const datamunge::image::Image& image) {
  std::size_t count = 0;
  for (int y = 0; y < image.height(); ++y)
    for (int x = 0; x < image.width(); ++x) {
      const auto p = image.get_pixel(x, y);
      count += int(p.r) > int(p.g) + 20;
    }
  return count;
}
} // namespace
TEST(GameRender, CpuValidationAndUnavailableBackendsNeverRasterizeOnHost) {
  g::RenderConfig config;
  config.width = 0;
  EXPECT_THROW(config.validate(), std::invalid_argument);
  EXPECT_THROW({g::Renderer renderer(config);}, std::invalid_argument);
  config.width  = 32;
  config.height = 24;
  EXPECT_THROW(g::Renderer(config, c::Device{}), std::invalid_argument);
  const c::Context cpu;
  EXPECT_THROW(g::Renderer(config, cpu), std::invalid_argument);
  g::RenderView view;
  view.bounds = {.5, 0, .6, 1};
  EXPECT_THROW(view.validate(), std::invalid_argument);
  g::AmbientOcclusion ao;
  ao.bias = ao.radius;
  EXPECT_THROW(ao.validate(), std::invalid_argument);
}
TEST(GameRender, VulkanProceduralGeometryCaptureAndResourceLifetime) {
  const auto info = c::probe({c::Backend::Vulkan, 0});
  if (!info.available)
    GTEST_SKIP() << info.diagnostic;
  c::Context context({c::Backend::Vulkan, 0});
  c::Buffer  retained;
  {
    g::RenderConfig config;
    config.width           = 64;
    config.height          = 64;
    config.object_capacity = 8;
    g::Renderer   renderer(config, context);
    g::RenderView view;
    EXPECT_THROW(renderer.capture(), std::logic_error);
    EXPECT_THROW(renderer.color_buffer(), std::logic_error);
    std::array<std::size_t, 3> counts{};
    for (int kind = 0; kind < 3; ++kind) {
      renderer.draw(primitive(static_cast<g::Primitive>(kind)), &view);
      const auto image = renderer.capture();
      EXPECT_EQ(image.width(), 64);
      EXPECT_EQ(image.channels(), 4);
      EXPECT_EQ(image.get_pixel(32, 32).a, 255);
      EXPECT_GT(image.get_pixel(32, 32).r, image.get_pixel(32, 32).g);
      counts[kind] = red_pixels(image);
    }
    EXPECT_GT(counts[0], counts[1]);
    EXPECT_GT(counts[1], counts[2]);
    EXPECT_GT(counts[2], 20);
    retained = renderer.color_buffer();
    EXPECT_EQ(retained.size(), 64 * 64 * 4);
    EXPECT_THROW(retained.host_data(), std::logic_error);
    renderer.resize(37, 23);
    EXPECT_THROW(renderer.capture(), std::logic_error);
    EXPECT_EQ(retained.size(), 64 * 64 * 4);
    renderer.draw(primitive(g::Primitive::Sphere), &view);
    datamunge::image::Image destination;
    renderer.capture(destination);
    EXPECT_EQ(destination.width(), 37);
    EXPECT_EQ(destination.height(), 23);
    const auto before = destination.data().data();
    renderer.capture(destination);
    EXPECT_EQ(before, destination.data().data());
    EXPECT_THROW(renderer.resize(0, 23), std::invalid_argument);
    EXPECT_EQ(renderer.width(), 37);
    EXPECT_THROW(renderer.draw(g::Scene{}), std::invalid_argument);
    EXPECT_NO_THROW(renderer.capture());
  }
  EXPECT_EQ(context.live_bytes(), 64 * 64 * 4);
  retained = {};
  EXPECT_EQ(context.live_bytes(), 0);
}
TEST(GameRender, VulkanAmbientOcclusionOverlayViewportAndPngCapture) {
  const auto info = c::probe({c::Backend::Vulkan, 0});
  if (!info.available)
    GTEST_SKIP() << info.diagnostic;
  g::RenderConfig config;
  config.width           = 256;
  config.height          = 192;
  config.object_capacity = 8;
  g::Renderer renderer(config);
  auto        scene = primitive(g::Primitive::Box);
  g::Entity   ground;
  ground.mesh               = g::Mesh{};
  ground.mesh->color        = {.4, .4, .4};
  ground.spatial.position   = {0, -1.1, 0};
  ground.spatial.dimensions = {12, .2, 12};
  scene.create(ground);
  g::RenderView view;
  view.position      = {0, 2.5, 5};
  const double pitch = -std::atan2(2.5, 5.);
  view.rotation      = {std::cos(pitch / 2), std::sin(pitch / 2), 0, 0};
  g::AmbientOcclusion settings;
  settings.visualize = true;
  renderer.set_ambient_occlusion(settings);
  renderer.draw(scene, &view);
  const auto  ao   = renderer.capture();
  std::size_t dark = 0;
  for (std::size_t i = 0; i < ao.data().size(); i += 4)
    dark += ao.data()[i] < 250;
  EXPECT_GT(dark, 0);
  settings.enabled = false;
  renderer.set_ambient_occlusion(settings);
  renderer.draw(scene, &view);
  const auto disabled = renderer.capture();
  for (std::size_t i = 0; i < disabled.data().size(); i += 4)
    EXPECT_EQ(disabled.data()[i], 255);
  settings.visualize = false;
  settings.enabled   = true;
  renderer.set_ambient_occlusion(settings);
  renderer.set_overlay("Native Courier", "Ready", "Move / Jump");
  renderer.draw(scene, &view);
  const auto image = renderer.capture();
  EXPECT_EQ(image.get_pixel(16, 16), (datamunge::image::Pixel{8, 204, 128, 255}));
  if (const auto* output = std::getenv("DATAMUNGE_RENDER_TEST_OUTPUT")) {
    const auto file = std::filesystem::path(output) / "native-render.png";
    std::filesystem::create_directories(file.parent_path());
    datamunge::image::write_png(image, file.string());
    EXPECT_EQ(datamunge::image::read_png(file.string()).data(), image.data());
  }
  renderer.set_overlay("");
  view.bounds = {.5, 0, .5, 1};
  renderer.draw(scene, &view);
  const auto half = renderer.capture();
  EXPECT_EQ(half.get_pixel(0, 0), half.get_pixel(100, 100));
  EXPECT_THROW(renderer.set_overlay(std::string(1025, 'a')), std::invalid_argument);
}
