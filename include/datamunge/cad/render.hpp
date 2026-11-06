#pragma once
/// @file render.hpp
/// @brief Device-independent CAD scene cameras, rendering, picking and image output.
#include <cstdint>
#include <datamunge/cad/math.hpp>
#include <datamunge/cad/scene.hpp>
#include <limits>
#include <memory>

namespace datamunge::cad {
/// @brief Camera projection for offline rendering.
enum class Projection {
  Perspective, ///< Pinhole projection with a vertical field of view.
  Orthographic ///< Parallel projection with a fixed vertical span.
};
/// @brief Surface shading and optional visible triangle-edge display.
enum class RenderStyle {
  Shaded,         ///< Material, light and optional shadow shading.
  Unlit,          ///< Texture multiplied by material tint, without lighting.
  Wireframe,      ///< Visible triangle edges over the background.
  ShadedWireframe ///< Shaded surfaces with visible triangle edges.
};
/// @brief A world-space ray; rendering/picking normalize its direction.
struct Ray {
  Vec3d origin;              ///< Start in scene units.
  Vec3d direction{0, 0, -1}; ///< Finite nonzero direction.
};
/// @brief A camera value independent of a window or graphics device.
/// Camera units match the scene; up need not be normalized. A valid camera
/// has distinct position/target and an up direction not parallel to its view.
/// @par Example
/// @code
/// dm::cad::Camera camera;
/// camera.position = {6, -8, 5}; camera.target = {1, 1, 1};
/// @endcode
struct Camera {
  Vec3d      position{6, -8, 5};                     ///< World position.
  Vec3d      target{0, 0, 0};                        ///< Point being viewed.
  Vec3d      up{0, 0, 1};                            ///< Sheet-up reference direction.
  Projection projection   = Projection::Perspective; ///< Pinhole or parallel projection.
  double     vertical_fov = 45;   ///< Perspective vertical angle in degrees, strictly (0,179).
  double     ortho_height = 10;   ///< Positive vertical span for parallel projection.
  double     near_clip    = 1e-5; ///< Positive minimum distance along a ray.
  double     far_clip     = 1e9;  ///< Maximum ray distance, greater than near_clip.

  /// @brief Construct a ray through a continuous pixel coordinate.
  /// @param x Pixel X; .5 is the first pixel centre.
  /// @param y Pixel Y from the top; .5 is the first row centre.
  /// @param width Positive image width.
  /// @param height Positive image height.
  /// @return Normalized world ray; orthographic rays have shifted origins.
  /// @throws std::invalid_argument Invalid camera, dimensions or coordinates.
  /// @par Example
  /// @code
  /// auto ray = dm::cad::Camera{}.ray(320.5, 240.5, 640, 480);
  /// @endcode
  Ray ray(double x, double y, int width, int height) const;
};
/// @brief Controls for deterministic, device-independent CPU rendering.
/// The optional material workflow uses GGX reflection and Schlick Fresnel with
/// approximate ambient illumination and a filmic display curve; it is not a
/// path tracer or a subsurface-scattering model. Defaults retain legacy shading.
/// @par Example
/// @code
/// dm::cad::RenderOptions settings;
/// settings.physically_based=true; settings.shadow_samples=6;
/// settings.studio_environment=.5; // optional broad studio reflections
/// @endcode
/// @par References
/// - Walter et al. (2007), Microfacet Models for Refraction through Rough Surfaces.
///   https://doi.org/10.2312/EGWR/EGSR07/195-206
/// - Schlick (1994), An Inexpensive BRDF Model for Physically-based Rendering.
///   https://doi.org/10.1111/1467-8659.1330233
struct RenderOptions {
  int         width   = 800; ///< Positive width, at most 8192 pixels.
  int         height  = 600; ///< Positive height, at most 8192 pixels.
  int         samples = 1;   ///< Square samples per axis: 1 to 4, i.e. 1 to 16 rays/pixel.
  int         threads = 0;   ///< 0 chooses up to 8 workers; 1..256 caps the worker count.
  RenderStyle style   = RenderStyle::Shaded;   ///< Shading/visible mesh-edge mode.
  RgbaColorF  background{.96f, .97f, .99f, 1}; ///< RGBA in [0,1].
  RgbaColorF  wire_color{.08f, .1f, .15f, 1};  ///< Triangle edge tint in [0,1].
  double      wire_width          = 1;         ///< Approximate positive edge width in image pixels.
  bool        shadows             = true;      ///< First visible light casts ray-tested shadows.
  bool        smooth_normals      = true; ///< Interpolate supplied vertex normals when available.
  bool        double_sided        = true; ///< Render both triangle orientations.
  int         transparency_layers = 32;   ///< Maximum composited surface intersections, 1 to 256.
  bool physically_based{false}; ///< Opt-in linear-light GGX/Schlick shading and filmic/sRGB output;
                                ///< legacy shading stays default.
  int    shadow_samples{1};     ///< Deterministic disk samples for the key light, 1..32.
  double light_angular_radius{
      0}; ///< Soft-key angular radius in radians, 0..0.3; zero is a point-direction key.
  int ambient_occlusion_samples{
      0}; ///< Deterministic cosine-hemisphere rays, 0..16; zero disables occlusion.
  double ambient_occlusion_distance{.12}; ///< Positive local occlusion radius in scene units.
  double exposure{1};                     ///< Positive linear exposure for physically_based output.
  double studio_environment{0}; ///< Analytic sky/ground and softbox reflection strength, 0..4.
                                ///< Used by physical surface shading only; zero retains the
                                ///< original ambient term. This is approximate environment
                                ///< illumination, not scene reflections or indirect path tracing.
};
/// @brief A nearest visible triangle hit in scene coordinates.
struct PickResult {
  bool   hit       = false; ///< False for background.
  int    object_id = -1;    ///< Source Scene object ID, or -1.
  int    triangle  = -1;    ///< Triangle within the object's mesh, or -1.
  double distance  = std::numeric_limits<double>::infinity(); ///< Distance along normalized ray.
  Vec3d  position;                                            ///< World intersection when hit.
  Vec3d  normal; ///< Interpolated world normal, oriented against the ray.
  Vec2d  uv;     ///< Interpolated texture coordinates; zero when absent.
};
/// @brief Owned RGBA8 render and centre-ray depth/object buffers.
/// Pixels are top-left row-major. Depth is normalized-ray distance in scene
/// units, infinity for background; object IDs are -1 for background. Samples
/// affect color only. Buffers outlive the scene and may be inspected in code.
struct RenderResult {
  int                        width  = 0; ///< Image width.
  int                        height = 0; ///< Image height.
  std::vector<unsigned char> rgba;       ///< Four bytes per pixel, straight alpha.
  std::vector<double>        depth;      ///< One centre-ray distance per pixel.
  std::vector<int>           object_ids; ///< One centre-ray source object ID per pixel.

  /// @brief Save the owned color image as PNG, JPEG or binary PPM.
  /// @param path Destination path with .png/.jpg/.jpeg/.ppm extension; parent must exist.
  /// @throws std::invalid_argument Invalid buffer shape or unsupported extension.
  /// @throws std::runtime_error File write failure.
  /// @par Example
  /// See render().
  void save(const std::string& path) const;
};

class CudaRenderer;

/// @brief Owned immutable transformed geometry reusable across cameras and frames.
/// Construction validates and prepares a snapshot. Copies share that immutable
/// snapshot; later changes to the source Scene cannot affect it. Concurrent const
/// renders are safe. An optional actor scene is prepared for just that call and
/// participates in all primary, shadow and AO rays. Base lighting is retained.
/// Actor rendering shares validated, transformed base triangles and rebuilds a
/// combined hierarchy in flattened scene order, preserving equal-depth ties.
/// The base-only hierarchy is initialized lazily under a thread-safe once guard.
/// @par Example
/// @code
/// dm::cad::Scene scene;
/// dm::cad::AddPrimitiveToScene(&scene,dm::cad::PrimitiveKind::Cube);
/// auto camera=dm::cad::fit_camera(scene);
/// dm::cad::PreparedScene prepared(std::move(scene));
/// auto first=prepared.render(camera);
/// camera.position.x+=1;
/// auto second=prepared.render(camera);
/// @endcode
/// @see render(), Scene, RenderOptions
class PreparedScene {
 public:
  /// @brief Prepare an owned snapshot once, including validation and transformed triangle data.
  /// @param scene Scene copied from an lvalue or moved from an rvalue; caller ownership is
  /// independent.
  /// @throws std::invalid_argument Malformed or nonfinite geometry, materials or lights.
  /// @par Example
  /// See the class example.
  explicit PreparedScene(Scene scene);
  /// @brief Render a camera using the existing acceleration structure.
  /// @param camera Valid perspective or orthographic camera.
  /// @param options Image, sampling, shading and worker settings; default RenderOptions{}.
  /// @return Owned color/depth/object-ID buffers, independent of this snapshot.
  /// @throws std::invalid_argument Invalid camera or render options.
  /// @par Example
  /// See the class example.
  RenderResult render(const Camera& camera, const RenderOptions& options = {}) const;
  /// @brief Render transient actors together with the prepared base, including mutual shadows.
  /// Actors are borrowed until this call completes, without mutation. Actor
  /// lights are ignored. Actor IDs are assigned in vector order beginning at
  /// the base scene's next_object_id, as with Scene::AddObject(). No pose or
  /// surface changes are hidden behind an implicit cache key.
  /// @param actors World-space meshes/materials/objects; may be empty.
  /// @param camera Valid perspective or orthographic camera.
  /// @param options Image, sampling, shading and worker settings; default RenderOptions{}.
  /// @return Owned image of the complete base-plus-actor world.
  /// @throws std::invalid_argument Invalid actors, camera or render options.
  /// @par Example
  /// @code
  /// dm::cad::Scene base,actors;
  /// dm::cad::AddPrimitiveToScene(&base,dm::cad::PrimitiveKind::Cube);
  /// auto camera=dm::cad::fit_camera(base);
  /// dm::cad::PreparedScene prepared(std::move(base));
  /// auto id=dm::cad::AddPrimitiveToScene(&actors,dm::cad::PrimitiveKind::Sphere);
  /// actors.FindObject(id)->position={2,0,0};
  /// auto image=prepared.render(actors,camera);
  /// @endcode
  RenderResult render(const Scene& actors, const Camera& camera, const RenderOptions& options = {})
      const;

 private:
  /// @brief CUDA rendering borrows the validated snapshot while retaining its ownership.
  friend class CudaRenderer;
  struct Impl; ///< Owned base geometry, materials and immutable acceleration storage.
  std::shared_ptr<const Impl> impl_; ///< Lifetime shared by cheap prepared-scene copies.
};

/// @brief Explicit NVIDIA CUDA offscreen CAD renderer with persistent device resources.
/// Uses the existing compute CUDA kernel backend. Primary, shadow and AO rays,
/// materials/maps, transparency, sampling and wire overlays execute on the GPU.
/// Geometry/BVH preparation stays on the CPU. Float32 device arithmetic and
/// scale-aware ray bias can differ from the CPU reference; this is not a
/// bit-exact replacement. Inputs must be representable in float32. The last
/// base snapshot and its uploaded geometry are retained until replaced or this
/// renderer is destroyed. Calls on one instance serialize; separate instances
/// own separate contexts. No silent fallback is performed. CPU render() remains
/// available in builds without CUDA; select fallback explicitly in the application.
/// @par Example
/// @code
/// if (dm::compute::probe({dm::compute::Backend::CUDA,0}).available) {
///   dm::cad::Scene scene;
///   dm::cad::AddPrimitiveToScene(&scene,dm::cad::PrimitiveKind::Cube);
///   auto camera=dm::cad::fit_camera(scene);
///   dm::cad::PreparedScene prepared(std::move(scene));
///   dm::cad::CudaRenderer gpu;
///   auto image=gpu.render(prepared,{},camera);
/// }
/// @endcode
/// @par References
/// - NVIDIA CUDA C++ Programming Guide, kernel execution and device memory.
///   https://docs.nvidia.com/cuda/archive/12.5.1/cuda-c-programming-guide/index.html
/// @see PreparedScene, RenderOptions, datamunge::compute::Context
class CudaRenderer {
 public:
  /// @brief Create a persistent CUDA context and compile the embedded ray kernel.
  /// @param device Zero-based CUDA device index; default zero.
  /// @throws std::runtime_error CUDA support/driver/device unavailable or kernel compilation
  /// failed.
  /// @par Example See the class example.
  explicit CudaRenderer(std::uint32_t device = 0);
  /// @brief Release cached device resources, the retained scene and native context.
  ~CudaRenderer();
  /// @brief Device/context ownership cannot be copied.
  /// @param other Source renderer; copying is deleted.
  CudaRenderer(const CudaRenderer& other) = delete;
  /// @brief Device/context ownership cannot be assigned.
  /// @param other Source renderer; assignment is deleted.
  /// @return No value; assignment is deleted.
  CudaRenderer& operator=(const CudaRenderer& other) = delete;
  /// @brief Render a prepared base and transient actors with the base's lights on CUDA.
  /// Reuses base uploads for the same snapshot. Moving actors and the combined
  /// hierarchy are uploaded each call. Output is synchronous, owned host RGBA8,
  /// centre-ray depth and source IDs. No window, hardware ray-tracing API, or
  /// external renderer is needed. `options.threads` affects only CPU rendering.
  /// @param base Owned validated base snapshot; unchanged and retained by the cache.
  /// @param actors Transient owned scene; may be empty; validated without mutating it.
  /// @param camera Camera in the same world units as the scene.
  /// @param options Output/material settings; defaults match CPU render options.
  /// @return Owned float32-derived color/depth/ID buffers.
  /// @throws std::invalid_argument Invalid scene, camera/options or float32 overflow/underflow.
  /// @throws std::length_error Device buffer indexing exceeds signed 32-bit limits.
  /// @throws std::runtime_error Native allocation, upload, execution or readback failure.
  /// @par Example See the class example.
  RenderResult render(
      const PreparedScene& base,
      const Scene&         actors,
      const Camera&        camera,
      const RenderOptions& options = {});

 private:
  /// @brief Persistent context, program, buffers and serialized last-snapshot cache.
  struct Impl;
  std::unique_ptr<Impl> impl_; ///< Exclusive device/cache owner.
};
/// @brief Choose a perspective camera framing all visible transformed vertices.
/// @param scene Borrowed scene, unchanged.
/// @param direction Finite nonzero direction from scene centre toward the camera.
/// @param up Finite up vector, not parallel to direction.
/// @param aspect Positive finite image width / height; default 4/3.
/// @return Camera with a 45 degree FOV and 15 percent framing margin.
/// @throws std::invalid_argument Invalid scene arrays, directions/aspect or empty scene.
/// @par Example
/// @code
/// dm::cad::Scene scene;
/// dm::cad::AddPrimitiveToScene(&scene, dm::cad::PrimitiveKind::Cube);
/// auto camera = dm::cad::fit_camera(scene);
/// @endcode
Camera fit_camera(
    const Scene& scene,
    Vec3d        direction = {1, -1, .75},
    Vec3d        up        = {0, 0, 1},
    double       aspect    = 4.0 / 3);
/// @brief Intersect a ray with visible transformed scene meshes using a BVH.
/// @param scene Borrowed scene; arrays/indices must be well formed and finite.
/// @param ray Finite origin and finite nonzero direction; normalized internally.
/// @param near_clip Positive finite minimum distance, default 1e-5.
/// @param far_clip Finite maximum distance greater than near_clip, default 1e9.
/// @return Closest triangle, with texture UVs and transformed normals.
/// @throws std::invalid_argument Invalid geometry, ray or clip range.
/// @par Example
/// @code
/// dm::cad::Scene scene;
/// dm::cad::AddPrimitiveToScene(&scene, dm::cad::PrimitiveKind::Cube);
/// auto hit = dm::cad::pick(scene, {{0,0,3},{0,0,-1}});
/// @endcode
PickResult pick(const Scene& scene, const Ray& ray, double near_clip = 1e-5, double far_clip = 1e9);
/// @brief Draw compact bitmap text onto owned RGBA pixels, for native video/chart overlays.
/// Uses the original built-in 5x7 font: ASCII lowercase is uppercased, unsupported
/// bytes use '?', newline starts an eight-row line. Pixels outside the image are
/// clipped. Alpha blends only RGBA; depth/object buffers remain unchanged.
/// @param image Valid positive-size RGBA8 image, mutated in place.
/// @param text UTF-8 bytes; supported ASCII glyphs are rendered without external font assets.
/// @param x Left pixel coordinate, may be outside the image.
/// @param y Top pixel coordinate, may be outside the image.
/// @param scale Integer glyph pixel scale, 1..16; default 2.
/// @param color Finite RGBA in [0,1]; default opaque white.
/// @throws std::invalid_argument Invalid image, scale or color, before mutation.
/// @par Example
/// @code
/// dm::cad::RenderResult image;image.width=64;image.height=20;image.rgba.resize(64*20*4);
/// dm::cad::draw_text(image,"T 1.0",2,2);
/// image.save("overlay.png");
/// @endcode
void draw_text(
    RenderResult&      image,
    const std::string& text,
    int                x,
    int                y,
    int                scale = 2,
    RgbaColorF         color = {1, 1, 1, 1});

/// @brief Render a scene without a window, GPU context or external executable.
/// Uses the source modeller's ambient/diffuse/specular metallic-roughness
/// formula, directional/point lights, RGBA textures and tangent-space normal
/// maps. Shadows use triangle rays rather than the editor's shadow map.
/// Materials use the source's red-channel roughness/metallic textures.
/// Const scene access is safe across independent concurrent renders.
/// @param scene Borrowed value, unchanged; lights empty selects the source default light.
/// @param camera Valid perspective or orthographic camera.
/// @param options Image, sampling, shading and concurrency controls.
/// @return Owned color/depth/ID image; objects with no opacity are skipped.
/// @throws std::invalid_argument Malformed/nonfinite scene, camera or options.
/// @par Example
/// @code
/// dm::cad::Scene scene;
/// dm::cad::AddPrimitiveToScene(&scene, dm::cad::PrimitiveKind::Cube);
/// auto image = dm::cad::render(scene, dm::cad::fit_camera(scene));
/// image.save("cube.png");
/// @endcode
/// @see add_solid(), load_scene(), Camera, RenderOptions
/// @par References
/// - Physically Based Rendering, 4th ed., section 7.3 (equal-count BVH concept):
///   https://pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies
RenderResult render(const Scene& scene, const Camera& camera, const RenderOptions& options = {});
} // namespace datamunge::cad
