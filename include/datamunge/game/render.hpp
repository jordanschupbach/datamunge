#pragma once
/// @file render.hpp
/// @brief Optional native Vulkan offscreen rendering and explicit image capture.
#include <datamunge/compute/context.hpp>
#include <datamunge/game/scene.hpp>
#include <datamunge/image/image.hpp>
#include <memory>
namespace datamunge::game {
/// @brief Screen-space ambient occlusion controls in world metres and dimensionless strength.
/// @par Example
/// @code
/// dm::game::AmbientOcclusion ao;ao.radius=.5;ao.validate();
/// @endcode
struct AmbientOcclusion {
  bool   enabled{true};    ///< Enable geometric screen-space occlusion of ambient lighting.
  double radius{1};        ///< Positive world radius in [0.00001,10] metres.
  double bias{.035};       ///< Nonnegative horizon bias smaller than radius, metres.
  double strength{1.5};    ///< Occlusion exponent in [0,5]; zero disables darkening.
  bool   visualize{false}; ///< Present the AO factor as grayscale instead of scene color.

  /// @brief Check finite radius, bias and strength ranges.
  /// @throws std::invalid_argument Invalid settings.
  /// @par Example
  /// @code
  /// const dm::game::AmbientOcclusion settings;settings.validate();
  /// @endcode
  void validate() const;
};
/// @brief One perspective view into a normalized top-left-origin output rectangle.
/// Uses a right-handed local -Z camera, unit w,x,y,z rotation and SI positions.
/// @par Example
/// @code
/// const dm::game::RenderView view;view.validate();
/// @endcode
struct RenderView {
  sim::Vec3       position{0, 0, 5}; ///< World camera position, each component within +/-100000 m.
  sim::Quaternion rotation{1, 0, 0, 0}; ///< Unit w,x,y,z camera orientation.
  Camera          camera;               ///< Validated vertical perspective lens.
  std::array<double, 4> bounds{
      0,
      0,
      1,
      1}; ///< Normalized left,top,width,height; positive size inside [0,1].

  /// @brief Validate the lens, pose and normalized viewport bounds.
  /// @throws std::invalid_argument Invalid values.
  /// @par Example
  /// @code
  /// dm::game::RenderView view;view.bounds={0,.5,1,.5};view.validate();
  /// @endcode
  void validate() const;
};
/// @brief Bounded native offscreen render controls, with no window or GUI state.
/// @par Example
/// @code
/// dm::game::RenderConfig config;config.width=64;config.height=48;config.validate();
/// @endcode
struct RenderConfig {
  std::uint32_t    width{1280};            ///< Positive output pixel width, at most 4096.
  std::uint32_t    height{800};            ///< Positive output pixel height, at most 4096.
  std::size_t      object_capacity{10000}; ///< Maximum mesh components, in [1,10000].
  AmbientOcclusion ambient_occlusion;      ///< World-space SSAO settings.
  double           exposure{1};            ///< Positive HDR exposure multiplier in [.001,100].
  sim::Vec3        background{
      .02,
      .04,
      .06}; ///< Nonnegative linear RGB background, each component at most 10.

  /// @brief Validate bounded image dimensions, mesh capacity and presentation settings.
  /// @throws std::invalid_argument Invalid controls.
  /// @par Example
  /// @code
  /// const dm::game::RenderConfig config;config.validate();
  /// @endcode
  void validate() const;
};
/// @brief Native offscreen Vulkan renderer retaining its own or a shared compute context.
/// Procedural box/sphere/crystal intersections, GGX metallic/roughness lighting,
/// geometric shadow rays, HDR/ACES presentation, screen-space AO and bitmap
/// overlays run without NVRHI or a window. Frame shading/composition runs on the
/// selected Vulkan device; capture is an explicit host readback. Single-writer.
/// Shader arithmetic is float32. Geometry/lens values that overflow or underflow
/// required positive float32 fields are rejected before drawing.
/// @par Example
/// @code
/// if(dm::compute::probe({dm::compute::Backend::Vulkan,0}).available){
///   dm::game::RenderConfig config;config.width=32;config.height=24;
///   dm::game::Renderer renderer(config);
///   dm::game::Scene scene;dm::game::RenderView view;renderer.draw(scene,&view);
///   const auto image=renderer.capture();
/// }
/// @endcode
/// @par References
/// - GGX/material and ACES conventions follow the frozen VSim scene/tonemap shaders.
///   This native implementation uses exact procedural intersections and shadow rays.
/// @see RenderConfig, RenderView, image::write_png()
class Renderer {
 public:
  /// @brief Own a native Vulkan render context and preallocate bounded frame resources.
  /// @param config Validated pixel/capacity/presentation controls, defaults above.
  /// @param device Explicit Vulkan device, index zero by default.
  /// @throws std::invalid_argument Invalid controls or non-Vulkan backend.
  /// @throws std::runtime_error Backend unavailable or native allocation/compilation failure.
  /// @par Example
  /// @code
  /// if(dm::compute::probe({dm::compute::Backend::Vulkan,0}).available){
  ///   dm::game::RenderConfig c;c.width=16;c.height=16;dm::game::Renderer renderer(c);
  /// }
  /// @endcode
  explicit Renderer(
      RenderConfig    config = {},
      compute::Device device = {compute::Backend::Vulkan, 0});
  /// @brief Share an existing Vulkan compute context while owning independent frame/pipeline
  /// resources.
  /// @param config Validated render controls.
  /// @param context Existing Vulkan execution owner; retained by copy.
  /// @throws std::invalid_argument Invalid controls or non-Vulkan context.
  /// @throws std::runtime_error Native allocation/compilation failure.
  /// @par Example
  /// @code
  /// if(dm::compute::probe({dm::compute::Backend::Vulkan,0}).available){
  ///   dm::compute::Context context({dm::compute::Backend::Vulkan,0});
  ///   dm::game::RenderConfig c;c.width=16;c.height=16;dm::game::Renderer renderer(c,context);
  /// }
  /// @endcode
  Renderer(RenderConfig config, const compute::Context& context);
  /// @brief Finish resource ownership and release native pipelines/frame allocations.
  ~Renderer();
  /// @brief Disable implicit copying of mutable frame state.
  Renderer(const Renderer&) = delete;
  /// @brief Disable assignment of mutable frame state.
  Renderer& operator=(const Renderer&) = delete;
  /// @brief Shade and compose one native scene synchronously into the retained GPU image.
  /// @param scene Borrowed scene; optional meshes determine visible objects.
  /// @param view Optional explicit view; null selects the lowest-ID camera entity.
  /// @throws std::invalid_argument No camera, invalid view, float32 overflow or native
  /// storage-binding limits.
  /// @throws std::length_error Mesh capacity exceeded before GPU state changes.
  /// @throws std::runtime_error Native dispatch failure; no complete frame is then available.
  /// @par Example
  /// @code
  /// if(dm::compute::probe({dm::compute::Backend::Vulkan,0}).available){
  ///   dm::game::RenderConfig c;c.width=16;c.height=16;dm::game::Renderer renderer(c);
  ///   const dm::game::Scene scene;const dm::game::RenderView view;renderer.draw(scene,&view);
  /// }
  /// @endcode
  void draw(const Scene& scene, const RenderView* view = nullptr);
  /// @brief Transactionally replace pixel resources; a resized image needs a new draw.
  /// Previously copied color buffers retain their old allocations.
  /// @param width Positive new width at most 4096 pixels.
  /// @param height Positive new height at most 4096 pixels.
  /// @throws std::invalid_argument Invalid dimensions.
  /// @throws std::runtime_error Native allocation failure; old frame resources remain owned.
  /// @par Example
  /// @code
  /// if(dm::compute::probe({dm::compute::Backend::Vulkan,0}).available){
  ///   dm::game::RenderConfig c;c.width=16;c.height=16;dm::game::Renderer
  ///   renderer(c);renderer.resize(32,24);
  /// }
  /// @endcode
  void resize(std::uint32_t width, std::uint32_t height);
  /// @brief Set validated AO controls for the next draw without reallocating frame resources.
  /// @param settings New finite AO settings.
  /// @throws std::invalid_argument Invalid controls.
  /// @par Example
  /// @code
  /// if(dm::compute::probe({dm::compute::Backend::Vulkan,0}).available){
  ///   dm::game::RenderConfig c;c.width=16;c.height=16;dm::game::Renderer renderer(c);
  ///   dm::game::AmbientOcclusion
  ///   settings;settings.enabled=false;renderer.set_ambient_occlusion(settings);
  /// }
  /// @endcode
  void set_ambient_occlusion(const AmbientOcclusion& settings);
  /// @brief Prepare source-compatible title/status/control bitmap overlays for subsequent draws.
  /// Text uses native uppercase 5x7 glyphs; unknown bytes use '?'. Empty title disables
  /// overlays, and targets below 200x180 omit them. Composition remains on the GPU.
  /// @param title UTF-8 title, at most 1024 bytes; empty by default disables the HUD.
  /// @param status UTF-8 status, at most 1024 bytes; may be empty.
  /// @param controls UTF-8 controls, at most 1024 bytes; may be empty.
  /// @throws std::invalid_argument Invalid UTF-8, NUL or excessive text length.
  /// @throws std::runtime_error Native transfer failure.
  /// @par Example
  /// @code
  /// if(dm::compute::probe({dm::compute::Backend::Vulkan,0}).available){
  ///   dm::game::RenderConfig c;c.width=16;c.height=16;dm::game::Renderer
  ///   renderer(c);renderer.set_overlay("Courier","Ready","Move / Jump");
  /// }
  /// @endcode
  void set_overlay(std::string title, std::string status = {}, std::string controls = {});
  /// @brief Explicitly read back the latest complete image as owned top-left-origin RGBA bytes.
  /// @return Independent native raster suitable for image::write_png().
  /// @throws std::logic_error No complete draw since construction/resize or a failed dispatch.
  /// @throws std::runtime_error Native readback failure.
  /// @par Example
  /// @code
  /// if(dm::compute::probe({dm::compute::Backend::Vulkan,0}).available){
  ///   dm::game::RenderConfig c;c.width=16;c.height=16;dm::game::Renderer renderer(c);
  ///   dm::game::Scene scene;dm::game::RenderView view;renderer.draw(scene,&view);const auto
  ///   image=renderer.capture();
  /// }
  /// @endcode
  image::Image capture() const;
  /// @brief Read back a complete frame, reusing caller storage when dimensions/mode already match.
  /// @param image Mutable owned RGBA raster; replaced when incompatible.
  /// @throws std::logic_error No complete frame.
  /// @throws std::runtime_error Native readback failure.
  /// @see capture() for a runnable example.
  void capture(image::Image& image) const;
  /// @brief Share the latest packed RGBA8 Vulkan storage buffer without host transfer.
  /// Subsequent draws update the same allocation; resize preserves previously copied
  /// buffers' ownership. Calling this does not synchronize unrelated contexts.
  /// @return Owning buffer view retaining its native allocation/context.
  /// @throws std::logic_error No complete frame.
  /// @par Example
  /// @code
  /// if(dm::compute::probe({dm::compute::Backend::Vulkan,0}).available){
  ///   dm::game::RenderConfig c;c.width=16;c.height=16;dm::game::Renderer renderer(c);
  ///   dm::game::Scene scene;dm::game::RenderView view;renderer.draw(scene,&view);const auto
  ///   pixels=renderer.color_buffer();
  /// }
  /// @endcode
  compute::Buffer color_buffer() const;
  /// @brief Current output width in pixels.
  /// @return Positive pixel width.
  std::uint32_t width() const noexcept;
  /// @brief Current output height in pixels.
  /// @return Positive pixel height.
  std::uint32_t height() const noexcept;

 private:
  struct Impl; ///< Native pipelines, reusable scene/frame buffers and presentation controls.
  std::unique_ptr<Impl> impl_; ///< Exclusive mutable frame ownership.
};
} // namespace datamunge::game
