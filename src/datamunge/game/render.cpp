/// @file render.cpp
/// @brief Native Vulkan procedural rendering, deferred presentation and explicit raster readback.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <datamunge/game/render.hpp>
#include <limits>
#include <stdexcept>

#include "font.hpp"
#include "render_shaders.hpp"
#include "text.hpp"
namespace datamunge::game {
/// @brief Private float32 GPU layouts and bounded overlay preparation.
namespace {
/// @brief Reject a nonfinite scalar or value outside a closed range.
/// @param value Input scalar.
/// @param low Inclusive minimum.
/// @param high Inclusive maximum.
void range(double value, double low, double high) {
  if (!std::isfinite(value) || value < low || value > high)
    throw std::invalid_argument("invalid native render control");
}
/// @brief Convert one finite scalar to GPU float32, rejecting overflow.
/// @param value Native scene/control scalar.
/// @return Finite shader scalar.
float gpu_float(double value) {
  const auto result = static_cast<float>(value);
  if (!std::isfinite(value) || !std::isfinite(result))
    throw std::invalid_argument("scene outside finite float32 render range");
  return result;
}
/// @brief Convert a strictly positive shader field and reject float32 underflow.
/// @param value Native positive value.
/// @return Strictly positive finite shader scalar.
float positive_float(double value) {
  const auto result = gpu_float(value);
  if (result <= 0)
    throw std::invalid_argument("positive render field underflows float32");
  return result;
}
/// @brief Exact std430 five-vec4 mesh layout consumed by scene.comp.
struct alignas(16) GpuObject {
  std::array<float, 4> position_kind;  ///< World centre and procedural primitive index.
  std::array<float, 4> rotation;       ///< Unit w,x,y,z rotation.
  std::array<float, 4> scale;          ///< Full local extents, padding zero.
  std::array<float, 4> color_metallic; ///< Linear RGB and metallic fraction.
  std::array<float, 4> material;       ///< Roughness/emission and padding.
};
static_assert(sizeof(GpuObject) == 80, "native scene shader object layout");
/// @brief Exact 128-byte native push-constant contract shared by both compute passes.
struct alignas(16) Parameters {
  std::array<float, 4>         camera_position; ///< World camera origin and padding.
  std::array<float, 4>         camera_rotation; ///< Unit w,x,y,z camera rotation.
  std::array<float, 4>         camera;          ///< Tangent half-FOV, viewport aspect, near/far.
  std::array<std::uint32_t, 4> output_size;     ///< Pixel width, height, object count and padding.
  std::array<float, 4>         bounds;          ///< Normalized viewport left/top/width/height.
  std::array<float, 4>         ao;              ///< Enabled/radius/bias/strength controls.
  std::array<float, 4>         flags;           ///< AO visualization, exposure and padding.
  std::array<float, 4>         background;      ///< Linear background and padding.
};
static_assert(sizeof(Parameters) == 128, "native render push constant layout");
/// @brief Own reusable scene-pass, presentation and CPU-prepared overlay storage.
struct Frame {
  compute::Buffer            direct;       ///< HDR direct lighting/emissive values.
  compute::Buffer            geometry;     ///< Surface world normal and ray depth.
  compute::Buffer            ambient;      ///< Unoccluded HDR ambient values.
  compute::Buffer            overlay;      ///< Optional packed RGBA8 bitmap overlay.
  compute::Buffer            color;        ///< Presented packed RGBA8 GPU image.
  std::vector<std::uint32_t> host_overlay; ///< Explicit host-authored bitmap upload source.
};
/// @brief Pack presentation-order RGBA bytes into a shader-visible uint32.
/// @param r Red byte.
/// @param g Green byte.
/// @param b Blue byte.
/// @param a Alpha byte, default opaque.
/// @return Packed red-low/alpha-high word.
std::uint32_t rgba(unsigned r, unsigned g, unsigned b, unsigned a = 255) {
  return r | (g << 8) | (b << 16) | (a << 24);
}
/// @brief Paint a clipped solid native overlay rectangle in caller-owned host preparation storage.
/// @param target Output words.
/// @param width Pixel width.
/// @param height Pixel height.
/// @param x Left coordinate.
/// @param y Top coordinate.
/// @param w Rectangle width.
/// @param h Rectangle height.
/// @param color Packed RGBA presentation color.
void rectangle(
    std::vector<std::uint32_t>& target,
    int                         width,
    int                         height,
    int                         x,
    int                         y,
    int                         w,
    int                         h,
    std::uint32_t               color) {
  for (int row = std::max(0, y); row < std::min(height, y + h); ++row)
    for (int column = std::max(0, x); column < std::min(width, x + w); ++column)
      target[std::size_t(row) * width + column] = color;
}
/// @brief Prepare bounded original bitmap HUD layout without performing scene shading on the CPU.
/// @param target Preallocated overlay words.
/// @param width Pixel width.
/// @param height Pixel height.
/// @param lines Title, status and control strings.
void overlay(
    std::vector<std::uint32_t>&       target,
    int                               width,
    int                               height,
    const std::array<std::string, 3>& lines) {
  std::fill(target.begin(), target.end(), 0);
  if (width < 200 || height < 180 || lines[0].empty())
    return;
  rectangle(target, width, height, 16, 16, width - 32, 80, rgba(2, 4, 6));
  rectangle(target, width, height, 16, 16, 4, 80, rgba(8, 204, 128));
  rectangle(target, width, height, 16, height - 46, width - 32, 30, rgba(2, 4, 6));
  const std::array<int, 3>           top{28, 64, height - 38}, size{3, 2, 2};
  const std::array<std::uint32_t, 3> colors{
      rgba(230, 240, 255), rgba(28, 230, 153), rgba(166, 184, 217)};
  for (int line = 0; line < 3; ++line) {
    int x = 28;
    for (unsigned char c : lines[line]) {
      if (x + 5 * size[line] > width - 24)
        break;
      const auto& glyph = detail::glyph(c);
      for (int row = 0; row < 7; ++row)
        for (int column = 0; column < 5; ++column)
          if (glyph[row] & (1u << (4 - column)))
            rectangle(
                target,
                width,
                height,
                x + column * size[line],
                top[line] + row * size[line],
                size[line],
                size[line],
                colors[line]);
      x += 6 * size[line];
    }
  }
}
} // namespace
void AmbientOcclusion::validate() const {
  range(radius, .00001, 10);
  range(bias, 0, radius);
  if (bias >= radius)
    throw std::invalid_argument("AO bias must be smaller than radius");
  range(strength, 0, 5);
}
void RenderView::validate() const {
  camera.validate();
  for (auto p : position)
    range(p, -100000, 100000);
  double norm = 0;
  for (auto q : rotation) {
    range(q, -1, 1);
    norm += q * q;
  }
  if (std::abs(norm - 1) > 1e-6)
    throw std::invalid_argument("render view quaternion must be unit length");
  for (auto value : bounds)
    range(value, 0, 1);
  if (bounds[2] <= 0 || bounds[3] <= 0 || bounds[0] + bounds[2] > 1 || bounds[1] + bounds[3] > 1)
    throw std::invalid_argument("invalid normalized render viewport");
}
void RenderConfig::validate() const {
  if (!width || !height || width > 4096 || height > 4096 || !object_capacity ||
      object_capacity > 10000)
    throw std::invalid_argument("invalid bounded render dimensions/capacity");
  ambient_occlusion.validate();
  range(exposure, .001, 100);
  for (auto value : background)
    range(value, 0, 10);
}
/// @brief Exclusive native pipelines and bounded scene/frame resources, retaining a shared context.
struct Renderer::Impl {
  compute::Context           context;       ///< Shared Vulkan device/program/allocation owner.
  RenderConfig               config;        ///< Validated current controls.
  compute::Program           scene_program; ///< Procedural geometry/lighting pipeline.
  compute::Program           post_program;  ///< AO/tonemap/overlay pipeline.
  compute::Buffer            objects;       ///< Reusable std430 mesh storage.
  std::vector<GpuObject>     host_objects;  ///< Capacity-reserved scene upload preparation.
  Frame                      frame;         ///< Current target resources.
  std::array<std::string, 3> lines;         ///< Owned current overlay text.
  bool complete{false};                     ///< Whether a complete draw is available for readback.

  /// @brief Allocate one candidate frame before committing resize state.
  /// @param width Positive bounded pixel width.
  /// @param height Positive bounded pixel height.
  /// @return Complete frame allocation with zeroed overlay.
  Frame make_frame(std::uint32_t width, std::uint32_t height) {
    const auto count = std::size_t(width) * height;
    Frame      result;
    result.direct   = context.allocate(count * 16);
    result.geometry = context.allocate(count * 16);
    result.ambient  = context.allocate(count * 16);
    result.overlay  = context.allocate(count * 4);
    result.color    = context.allocate(count * 4);
    result.host_overlay.resize(count);
    overlay(result.host_overlay, int(width), int(height), lines);
    context.upload(result.overlay, result.host_overlay.data(), count * 4);
    return result;
  }
  /// @brief Compile retained native programs and reserve all target/scene capacities.
  /// @param settings Validated render settings.
  /// @param owner Existing Vulkan compute context.
  Impl(RenderConfig settings, compute::Context owner)
      : context(std::move(owner)), config(settings) {
    if (context.device().backend != compute::Backend::Vulkan)
      throw std::invalid_argument("native renderer requires a Vulkan context");
    compute::KernelSource source;
    source.buffers         = 4;
    source.parameter_bytes = sizeof(Parameters);
    source.local_size      = {8, 8, 1};
    source.spirv.assign(std::begin(detail::scene_spirv), std::end(detail::scene_spirv));
    scene_program  = context.program(source);
    source.buffers = 5;
    source.spirv.assign(std::begin(detail::post_spirv), std::end(detail::post_spirv));
    post_program = context.program(source);
    objects      = context.allocate(config.object_capacity * sizeof(GpuObject));
    host_objects.reserve(config.object_capacity);
    frame = make_frame(config.width, config.height);
  }
};
Renderer::Renderer(RenderConfig config, compute::Device device) {
  config.validate();
  if (device.backend != compute::Backend::Vulkan)
    throw std::invalid_argument("native renderer requires Vulkan");
  impl_ = std::make_unique<Impl>(config, compute::Context(device));
}
Renderer::Renderer(RenderConfig config, const compute::Context& context) {
  config.validate();
  impl_ = std::make_unique<Impl>(config, context);
}
Renderer::~Renderer() = default;
void Renderer::draw(const Scene& scene, const RenderView* view) {
  RenderView selected;
  if (view)
    selected = *view;
  else {
    bool found = false;
    for (auto id : scene.entities()) {
      const auto& entity = scene.get(id);
      if (entity.camera) {
        selected.position = entity.spatial.position;
        selected.rotation = entity.spatial.rotation;
        selected.camera   = *entity.camera;
        found             = true;
        break;
      }
    }
    if (!found)
      throw std::invalid_argument("render scene has no camera; supply RenderView explicitly");
  }
  selected.validate();
  auto& state = *impl_;
  state.host_objects.clear();
  for (auto id : scene.entities()) {
    const auto& entity = scene.get(id);
    if (!entity.mesh)
      continue;
    if (state.host_objects.size() == state.config.object_capacity)
      throw std::length_error("renderer mesh capacity exhausted");
    const auto& mesh = *entity.mesh;
    GpuObject   object{};
    for (int i = 0; i < 3; ++i) {
      object.position_kind[i]  = gpu_float(entity.spatial.position[i]);
      object.scale[i]          = positive_float(entity.spatial.dimensions[i]);
      object.color_metallic[i] = gpu_float(mesh.color[i]);
    }
    object.position_kind[3]  = float(mesh.primitive);
    object.color_metallic[3] = gpu_float(mesh.metallic);
    for (int i = 0; i < 4; ++i)
      object.rotation[i] = gpu_float(entity.spatial.rotation[i]);
    object.material = {gpu_float(mesh.roughness), gpu_float(mesh.emissive), 0, 0};
    state.host_objects.push_back(object);
  }
  Parameters parameters{};
  for (int i = 0; i < 3; ++i) {
    parameters.camera_position[i] = gpu_float(selected.position[i]);
    parameters.background[i]      = gpu_float(state.config.background[i]);
  }
  for (int i = 0; i < 4; ++i) {
    parameters.camera_rotation[i] = gpu_float(selected.rotation[i]);
    parameters.bounds[i] =
        i >= 2 ? positive_float(selected.bounds[i]) : gpu_float(selected.bounds[i]);
  }
  const double pi   = std::acos(-1.);
  parameters.camera = {
      positive_float(std::tan(selected.camera.vertical_fov * pi / 360)),
      positive_float(
          double(state.config.width) * selected.bounds[2] /
          (double(state.config.height) * selected.bounds[3])),
      positive_float(selected.camera.near_plane),
      positive_float(selected.camera.far_plane)};
  parameters.output_size = {
      state.config.width, state.config.height, std::uint32_t(state.host_objects.size()), 0};
  const auto& ao = state.config.ambient_occlusion;
  parameters.ao  = {
      ao.enabled ? 1.f : 0.f, gpu_float(ao.radius), gpu_float(ao.bias), gpu_float(ao.strength)};
  parameters.flags = {ao.visualize ? 1.f : 0.f, gpu_float(state.config.exposure), 0, 0};
  state.complete   = false;
  state.context.upload(
      state.objects, state.host_objects.data(), state.host_objects.size() * sizeof(GpuObject));
  const std::array<std::uint32_t, 3> groups{
      (state.config.width + 7) / 8, (state.config.height + 7) / 8, 1};
  state.context.dispatch(
      state.scene_program,
      {state.objects, state.frame.direct, state.frame.geometry, state.frame.ambient},
      &parameters,
      sizeof(parameters),
      groups);
  state.context.dispatch(
      state.post_program,
      {state.frame.direct,
       state.frame.geometry,
       state.frame.ambient,
       state.frame.overlay,
       state.frame.color},
      &parameters,
      sizeof(parameters),
      groups);
  state.complete = true;
}
void Renderer::resize(std::uint32_t width, std::uint32_t height) {
  auto config   = impl_->config;
  config.width  = width;
  config.height = height;
  config.validate();
  auto frame      = impl_->make_frame(width, height);
  impl_->frame    = std::move(frame);
  impl_->config   = config;
  impl_->complete = false;
}
void Renderer::set_ambient_occlusion(const AmbientOcclusion& settings) {
  settings.validate();
  impl_->config.ambient_occlusion = settings;
}
void Renderer::set_overlay(std::string title, std::string status, std::string controls) {
  std::array<std::string, 3> lines{std::move(title), std::move(status), std::move(controls)};
  for (const auto& line : lines) {
    if (line.size() > 1024 || line.find('\0') != std::string::npos)
      throw std::invalid_argument("invalid bounded overlay text");
    detail::utf8(line);
  }
  overlay(impl_->frame.host_overlay, int(width()), int(height()), lines);
  impl_->context.upload(
      impl_->frame.overlay, impl_->frame.host_overlay.data(), impl_->frame.host_overlay.size() * 4);
  impl_->lines = std::move(lines);
}
image::Image Renderer::capture() const {
  image::Image result;
  capture(result);
  return result;
}
void Renderer::capture(image::Image& destination) const {
  if (!impl_->complete)
    throw std::logic_error("renderer has no complete frame");
  if (destination.width() != int(width()) || destination.height() != int(height()) ||
      destination.mode() != image::ImageMode::RGBA)
    destination = image::Image(int(width()), int(height()), image::ImageMode::RGBA);
  impl_->context.download(destination.data().data(), impl_->frame.color, destination.data().size());
}
compute::Buffer Renderer::color_buffer() const {
  if (!impl_->complete)
    throw std::logic_error("renderer has no complete frame");
  return impl_->frame.color;
}
std::uint32_t Renderer::width() const noexcept {
  return impl_->config.width;
}
std::uint32_t Renderer::height() const noexcept {
  return impl_->config.height;
}
} // namespace datamunge::game
