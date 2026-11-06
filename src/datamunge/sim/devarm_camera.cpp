/// @file devarm_camera.cpp
/// @brief Real-URDF camera extrinsics and depth-composited native wrist RGB frames.
#include <algorithm>
#include <cmath>
#include <datamunge/sim/devarm_camera.hpp>
#include <future>
#include <limits>
#include <stdexcept>
#include <thread>

#include "visual_assets.hpp"

namespace datamunge::sim {
/// @brief Private workcell camera geometry helpers.
namespace {
/// @brief Fixed optical centre in the actual end-link coordinate frame, metres.
const cad::Vec3d mount{-.09, -.075, .045};
/// @brief Fixed optical-axis point in the same frame; independent of the payload.
const cad::Vec3d look{.13, 0, 0};
/// @brief Append already world-space geometry without altering placements/materials.
void append(cad::Scene& dest, cad::Scene extra) {
  int base = int(dest.meshes.size()), tex = int(dest.textures.size());
  for (auto& m : extra.meshes)
    dest.meshes.push_back(std::move(m));
  for (auto& t : extra.textures)
    dest.textures.push_back(std::move(t));
  for (auto o : extra.objects) {
    o.mesh_index += base;
    o.parent = -1;
    for (int* s :
         {&o.texture_index, &o.normal_map_index, &o.roughness_map_index, &o.metallic_map_index})
      if (*s >= 0)
        *s += tex;
    dest.AddObject(o);
  }
}
/// @brief Add consistent world lighting to both cached scenery and moving actors.
void lighting(cad::Scene& scene) {
  cad::Light key;
  key.direction = {-1, -2, 3};
  key.intensity = 1;
  scene.AddLight(key);
  cad::Light fill;
  fill.direction = {1, 1, 2};
  fill.intensity = .35f;
  scene.AddLight(fill);
}
/// @brief Add a wrist-local primitive with its vertex geometry baked into world coordinates.
void part(
    cad::Scene&        scene,
    const cad::Mat4d&  wrist,
    cad::PrimitiveKind kind,
    const std::string& name,
    cad::Vec3d         at,
    cad::Vec3d         size,
    cad::RgbaColorF    color,
    bool               realistic = false) {
  auto  id    = cad::AddPrimitiveToScene(&scene, kind);
  auto& o     = *scene.FindObject(id);
  o.name      = name;
  o.color     = color;
  o.roughness = .4f;
  auto& m     = scene.meshes[o.mesh_index];
  if (realistic && kind == cad::PrimitiveKind::Cube)
    m = detail::visuals::rounded_box({1, 1, 1}, .08);
  for (std::size_t i = 0; i < m.positions.size(); i += 3) {
    auto p = wrist.TransformPoint(
        {at.x + size.x * m.positions[i],
         at.y + size.y * m.positions[i + 1],
         at.z + size.z * m.positions[i + 2]});
    m.positions[i]     = float(p.x);
    m.positions[i + 1] = float(p.y);
    m.positions[i + 2] = float(p.z);
  }
  for (std::size_t i = 0; i < m.normals.size(); i += 3) {
    auto normal =
        cad::Vec3d{m.normals[i] / size.x, m.normals[i + 1] / size.y, m.normals[i + 2] / size.z}
            .Normalized();
    auto n           = wrist.TransformVector(normal);
    m.normals[i]     = float(n.x);
    m.normals[i + 1] = float(n.y);
    m.normals[i + 2] = float(n.z);
  }
}
/// @brief Resize a square padded RGB-only foreground crop without annotating the sensor.
void crop_rgb(WristInspection& result, int crop_side) {
  if (result.box.width <= 0)
    return;
  int x0 = result.box.x, y0 = result.box.y, x1 = x0 + result.box.width - 1,
      y1     = y0 + result.box.height - 1;
  int   side = result.rgb.width;
  auto& crop = result.crop;
  crop.width = crop.height = crop_side;
  crop.rgba.resize(crop_side * crop_side * 4);
  const double length = 1.4 * std::max(result.box.width, result.box.height);
  const double cx = (x0 + x1 + 1) * .5, cy = (y0 + y1 + 1) * .5;
  for (int y = 0; y < crop_side; ++y)
    for (int x = 0; x < crop_side; ++x) {
      int  sx = int(std::floor(cx + ((x + .5) / crop_side - .5) * length));
      int  sy = int(std::floor(cy + ((y + .5) / crop_side - .5) * length));
      auto i  = 4 * (y * crop_side + x);
      if (sx >= 0 && sy >= 0 && sx < side && sy < side)
        std::copy_n(result.rgb.rgba.begin() + 4 * (sy * side + sx), 4, crop.rgba.begin() + i);
      else {
        crop.rgba[i]     = 232;
        crop.rgba[i + 1] = 240;
        crop.rgba[i + 2] = 245;
        crop.rgba[i + 3] = 255;
      }
    }
}
} // namespace
/// @brief Camera ownership and a background cache keyed by actual arm/finger pose.
struct DevArmWristCamera::Impl {
  DevArmPickEnvironment& env;          ///< Borrowed fixed workcell/model owner.
  double                 fov;          ///< Vertical field of view in degrees.
  int                    side;         ///< Square image side.
  bool                   valid{false}; ///< Whether a background has been rendered.
  std::array<double, 6>  joints{};     ///< Last background's actual joints.
  double                 finger{0};    ///< Last background's gripper aperture.
  cad::RenderResult      background;   ///< Opaque static workcell plus real robot geometry.
  std::unique_ptr<cad::PreparedScene> prepared; ///< Immutable realistic static/robot acceleration.
  std::array<double, 6>               prepared_joints{}; ///< Independent realistic cache pose key.
  double prepared_finger{0};    ///< Realistic cache aperture; independent of legacy RGB cache.
  int    threads{0};            ///< Explicit render worker budget; zero preserves the default.
  cad::RenderResult base_image; ///< Owned realistic base-only RGB/depth/IDs at the prepared pose.
  bool base_image_valid{false}; ///< Invalidated whenever the prepared snapshot changes.

  /// @brief Refresh immutable realistic geometry only when the exact arm/finger pose changes.
  /// @param frame Recorded URDF joint/finger state.
  /// @throws std::invalid_argument Invalid recorded pose or world geometry.
  void prepare(const PickObservation& frame) {
    if (!prepared || prepared_joints != frame.joints || prepared_finger != frame.finger) {
      auto scene = env.static_scene();
      append(scene, env.robot_scene(frame));
      scene.lights     = workcell_lights(WorkcellVisualStyle::Realistic);
      prepared         = std::make_unique<cad::PreparedScene>(std::move(scene));
      base_image_valid = false;
      prepared_joints  = frame.joints;
      prepared_finger  = frame.finger;
    }
  }

  /// @brief Build the unchanged realistic sensor settings with the selected worker budget.
  /// @return Full-quality square CPU render settings.
  cad::RenderOptions realistic_options() const {
    auto options    = workcell_render_options(WorkcellVisualStyle::Realistic, side, side);
    options.threads = threads;
    return options;
  }
  /// @brief Render the immutable base once per exact prepared joint/finger pose.
  /// @param camera Current wrist extrinsics for the prepared pose.
  /// @param options Fixed realistic sensor options; worker changes do not alter pixels.
  /// @return Borrowed cached buffers, valid until this camera's next base replacement.
  const cad::RenderResult& base_view(const cad::Camera& camera, const cad::RenderOptions& options) {
    if (!base_image_valid) {
      base_image       = prepared->render(camera, options);
      base_image_valid = true;
    }
    return base_image;
  }

  /// @brief Bind a workcell and validate the positive supported sensor resolution.
  Impl(DevArmPickEnvironment& owner, int size, double angle) : env(owner), fov(angle), side(size) {
    if (side < 16 || side > 512 || !std::isfinite(fov) || fov < 30 || fov > 120)
      throw std::invalid_argument("wrist camera side must be 16..512");
  }
};
DevArmWristCamera::DevArmWristCamera(DevArmPickEnvironment& env, int side, double fov)
    : impl_(std::make_unique<Impl>(env, side, fov)) {}
DevArmWristCamera::~DevArmWristCamera() = default;
void DevArmWristCamera::set_render_threads(int threads) {
  if (threads < 0 || threads > 256)
    throw std::invalid_argument("camera render threads must be 0..256");
  impl_->threads = threads;
}
cad::Camera DevArmWristCamera::pose(const PickObservation& frame) const {
  auto        wrist = impl_->env.link_pose(frame);
  cad::Camera c;
  c.position     = wrist.TransformPoint(mount);
  c.target       = wrist.TransformPoint(look);
  c.up           = wrist.TransformVector({0, 0, 1});
  c.vertical_fov = impl_->fov;
  c.near_clip    = .002;
  c.far_clip     = 5;
  return c;
}
cad::RenderResult DevArmWristCamera::capture(const PickObservation& frame) {
  return capture_actors(frame, impl_->env.fruit_scene(frame));
}
cad::RenderResult DevArmWristCamera::capture_actors(
    const PickObservation& frame,
    const cad::Scene&      supplied) {
  auto& p = *impl_;
  if (p.env.visual_style() == WorkcellVisualStyle::Realistic) {
    p.prepare(frame);
    auto options = p.realistic_options();
    auto camera  = pose(frame);
    // With no objects/textures the actor constructor has nothing to validate.
    // Unused meshes/lights were ignored by the original actor render as well.
    if (supplied.objects.empty() && supplied.textures.empty())
      return p.base_view(camera, options);
    return p.prepared->render(supplied, camera, options);
  }
  auto               c = pose(frame);
  cad::RenderOptions options;
  options.width      = p.side;
  options.height     = p.side;
  options.samples    = 1;
  options.threads    = p.threads;
  options.background = {.91f, .94f, .96f, 1};
  if (!p.valid || frame.joints != p.joints || frame.finger != p.finger) {
    auto scene = p.env.static_scene();
    append(scene, p.env.robot_scene(frame));
    lighting(scene);
    p.background = cad::render(scene, c, options);
    p.joints     = frame.joints;
    p.finger     = frame.finger;
    p.valid      = true;
  }
  auto actors = supplied;
  actors.lights.clear();
  lighting(actors);
  auto rgb   = cad::render(actors, c, options);
  auto image = p.background;
  for (std::size_t i = 0; i < image.depth.size(); ++i)
    if (rgb.depth[i] < image.depth[i]) {
      std::copy_n(rgb.rgba.begin() + 4 * i, 4, image.rgba.begin() + 4 * i);
      image.depth[i]      = rgb.depth[i];
      image.object_ids[i] = rgb.object_ids[i];
    }
  return image;
}
WristInspection DevArmWristCamera::inspect(const PickObservation& frame, int crop_side) {
  if (impl_->env.visual_style() == WorkcellVisualStyle::Realistic)
    return inspect_actors(frame, impl_->env.fruit_scene(frame), frame.fruit, crop_side);
  if (crop_side < 16 || crop_side > 256)
    throw std::invalid_argument("crop side must be 16..256");
  WristInspection result;
  result.rgb  = capture(frame);
  auto actors = impl_->env.fruit_scene(frame);
  // Keep every non-payload actor in the registered empty reference.
  actors.objects.erase(
      std::remove_if(
          actors.objects.begin(),
          actors.objects.end(),
          [](const auto& o) {
            return o.name.starts_with("apple") || o.name.starts_with("banana");
          }),
      actors.objects.end());
  lighting(actors);
  cad::RenderOptions options;
  options.width = options.height = impl_->side;
  options.samples                = 1;
  options.threads                = impl_->threads;
  options.background             = {.91f, .94f, .96f, 1};
  const auto marks               = cad::render(actors, pose(frame), options);
  auto       reference           = impl_->background;
  for (std::size_t i = 0; i < reference.depth.size(); ++i)
    if (impl_->env.visual_style() != WorkcellVisualStyle::Realistic &&
        marks.depth[i] < reference.depth[i])
      std::copy_n(marks.rgba.begin() + 4 * i, 4, reference.rgba.begin() + 4 * i);
  int x0 = impl_->side, y0 = impl_->side, x1 = -1, y1 = -1;
  for (int y = 0; y < impl_->side; ++y)
    for (int x = 0; x < impl_->side; ++x) {
      const auto i          = 4 * (y * impl_->side + x);
      int        difference = 0;
      for (int c = 0; c < 3; ++c)
        difference = std::max(
            difference, std::abs(int(result.rgb.rgba[i + c]) - int(reference.rgba[i + c])));
      if (difference <= 12)
        continue;
      ++result.foreground_pixels;
      x0 = std::min(x0, x);
      x1 = std::max(x1, x);
      y0 = std::min(y0, y);
      y1 = std::max(y1, y);
    }
  if (result.foreground_pixels < 9)
    return result;
  result.box = {x0, y0, x1 - x0 + 1, y1 - y0 + 1};
  crop_rgb(result, crop_side);
  return result;
}
WristInspection DevArmWristCamera::inspect_actors(
    const PickObservation& frame,
    const cad::Scene&      actors,
    cad::Vec3d             focus,
    int                    crop_side) {
  if (crop_side < 16 || crop_side > 256 || !std::isfinite(focus.x) || !std::isfinite(focus.y) ||
      !std::isfinite(focus.z))
    throw std::invalid_argument("invalid camera focus or crop");
  WristInspection result;
  const auto      camera  = pose(frame);
  auto            forward = (camera.target - camera.position).Normalized(),
       right = forward.Cross(camera.up).Normalized(), up = right.Cross(forward).Normalized();
  auto   d = focus - camera.position;
  double z = d.Dot(forward), tangent = std::tan(camera.vertical_fov * std::acos(-1.) / 360.);
  if (z <= camera.near_clip) {
    result.rgb = capture_actors(frame, actors);
    return result;
  }
  const int side = impl_->side;
  double    px   = (1 + d.Dot(right) / (z * tangent)) * side / 2,
         py      = (1 - d.Dot(up) / (z * tangent)) * side / 2;
  if (px < 0 || py < 0 || px >= side || py >= side) {
    result.rgb = capture_actors(frame, actors);
    return result;
  }
  auto empty = actors;
  empty.objects.erase(
      std::remove_if(
          empty.objects.begin(),
          empty.objects.end(),
          [](const auto& o) {
            return o.name.starts_with("apple") || o.name.starts_with("banana");
          }),
      empty.objects.end());
  empty.lights.clear();
  lighting(empty);
  cad::RenderOptions options;
  options.width = options.height = impl_->side;
  options.samples                = 1;
  options.background             = {.91f, .94f, .96f, 1};
  options.threads                = impl_->threads;
  cad::RenderResult                 reference;
  const std::vector<unsigned char>* reference_pixels = nullptr;
  if (impl_->env.visual_style() == WorkcellVisualStyle::Realistic) {
    auto& p = *impl_;
    p.prepare(frame);
    auto     full_options = p.realistic_options();
    unsigned budget       = p.threads ? unsigned(p.threads)
                                      : std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
    if (budget < 2 || (empty.objects.empty() && p.base_image_valid)) {
      result.rgb = capture_actors(frame, actors);
      if (empty.objects.empty())
        reference_pixels = &p.base_view(camera, full_options).rgba;
      else
        reference = capture_actors(frame, empty);
    } else {
      auto reference_options    = full_options;
      full_options.threads      = int((budget + 1) / 2);
      reference_options.threads = int(budget / 2);
      // Both views borrow the same immutable snapshot. Worker counts include
      // their callers, so concurrent renders stay within the requested budget.
      auto registered = std::async(std::launch::async, [&p, &empty, camera, reference_options] {
        return empty.objects.empty() ? p.prepared->render(camera, reference_options)
                                     : p.prepared->render(empty, camera, reference_options);
      });
      result.rgb      = p.prepared->render(actors, camera, full_options);
      reference       = registered.get();
      if (empty.objects.empty()) {
        p.base_image       = std::move(reference);
        p.base_image_valid = true;
        reference_pixels   = &p.base_image.rgba;
      }
    }
  } else {
    result.rgb = capture_actors(frame, actors);
    auto marks = cad::render(empty, camera, options);
    reference  = impl_->background;
    for (std::size_t i = 0; i < reference.depth.size(); ++i)
      if (marks.depth[i] < reference.depth[i])
        std::copy_n(marks.rgba.begin() + 4 * i, 4, reference.rgba.begin() + 4 * i);
  }
  if (!reference_pixels)
    reference_pixels = &reference.rgba;
  std::vector<unsigned char> mask(side * side);
  for (int i = 0; i < side * side; ++i) {
    int difference = 0;
    for (int c = 0; c < 3; ++c)
      difference = std::max(
          difference,
          std::abs(int(result.rgb.rgba[4 * i + c]) - int((*reference_pixels)[4 * i + c])));
    mask[i] = difference > 12;
  }
  std::vector<ImageBox>    boxes;
  std::vector<std::size_t> areas;
  for (int root = 0; root < side * side; ++root)
    if (mask[root]) {
      std::vector<int> queue{root};
      mask[root] = 0;
      int x0 = side, x1 = 0, y0 = side, y1 = 0;
      for (std::size_t k = 0; k < queue.size(); ++k) {
        int x = queue[k] % side, y = queue[k] / side;
        x0 = std::min(x0, x);
        x1 = std::max(x1, x);
        y0 = std::min(y0, y);
        y1 = std::max(y1, y);
        for (int dy = -1; dy <= 1; ++dy)
          for (int dx = -1; dx <= 1; ++dx) {
            int xx = x + dx, yy = y + dy;
            if (xx < 0 || yy < 0 || xx >= side || yy >= side)
              continue;
            int at = yy * side + xx;
            if (mask[at]) {
              mask[at] = 0;
              queue.push_back(at);
            }
          }
      }
      if (queue.size() >= 3) {
        boxes.push_back({x0, y0, x1 - x0 + 1, y1 - y0 + 1});
        areas.push_back(queue.size());
      }
    }
  int    chosen = -1;
  double best   = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < boxes.size(); ++i) {
    const auto& b  = boxes[i];
    double      dx = std::max({0., b.x - px, px - b.x - b.width}),
           dy      = std::max({0., b.y - py, py - b.y - b.height});
    double score =
        dx * dx + dy * dy +
        .02 * (std::pow(px - b.x - b.width * .5, 2) + std::pow(py - b.y - b.height * .5, 2));
    if (score < best) {
      best   = score;
      chosen = int(i);
    }
  }
  if (chosen < 0 || best > 256)
    return result;
  result.box               = boxes[chosen];
  result.foreground_pixels = areas[chosen];
  for (std::size_t i = 0; i < boxes.size(); ++i)
    if (int(i) != chosen) {
      auto b = boxes[i], a = result.box;
      int  dx = std::max({0, b.x - a.x - a.width, a.x - b.x - b.width}),
          dy  = std::max({0, b.y - a.y - a.height, a.y - b.y - b.height});
      if (dx <= 3 && dy <= 3) {
        int x0 = std::min(a.x, b.x), y0 = std::min(a.y, b.y);
        result.box = {
            x0,
            y0,
            std::max(a.x + a.width, b.x + b.width) - x0,
            std::max(a.y + a.height, b.y + b.height) - y0};
        result.foreground_pixels += areas[i];
      }
    }
  if (result.foreground_pixels < 9) {
    result.box               = {};
    result.foreground_pixels = 0;
    return result;
  }
  crop_rgb(result, crop_side);
  return result;
}
cad::Scene DevArmWristCamera::housing(const PickObservation& frame) const {
  auto       wrist = impl_->env.link_pose(frame);
  cad::Scene scene;
  part(
      scene,
      wrist,
      cad::PrimitiveKind::Cube,
      "wrist camera blue housing",
      mount,
      {.038, .035, .027},
      {.08f, .32f, .70f, 1},
      impl_->env.visual_style() == WorkcellVisualStyle::Realistic);
  auto axis = (look - mount).Normalized();
  part(
      scene,
      wrist,
      cad::PrimitiveKind::Sphere,
      "wrist camera lens",
      mount + axis * .021,
      {.021, .021, .021},
      {.025f, .04f, .06f, 1});
  part(
      scene,
      wrist,
      cad::PrimitiveKind::Cube,
      "wrist camera bracket",
      {-.09, -.044, .027},
      {.018, .055, .012},
      {.25f, .30f, .34f, 1},
      impl_->env.visual_style() == WorkcellVisualStyle::Realistic);
  return scene;
}
} // namespace datamunge::sim
