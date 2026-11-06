#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <cmath>
#include <datamunge/cad/render.hpp>
#include <datamunge/compute/context.hpp>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "detail/cuda_types.hpp"
#include "detail/jpeg_codec.hpp"
#include "detail/png_codec.hpp"
#include "detail/ray_box.hpp"
#include "detail/render_ptx.hpp"

namespace datamunge::cad {
namespace {
bool finite(Vec3d p) {
  return std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z);
}
void require(bool ok, const char* text) {
  if (!ok)
    throw std::invalid_argument(text);
}
void color_valid(RgbaColorF c) {
  for (double v : {c.r, c.g, c.b, c.a})
    require(std::isfinite(v) && v >= 0 && v <= 1, "RGBA colors must be finite in [0,1]");
}
Vec3d point(Vec3f p) {
  return {p.x, p.y, p.z};
}
Vec3d multiply(Vec3d a, Vec3d b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}
struct Frame {
  Vec3d forward, right, up;
};
Frame frame(const Camera& c) {
  require(finite(c.position) && finite(c.target) && finite(c.up), "Nonfinite camera");
  require((c.target - c.position).Length() > 0, "Camera position equals target");
  Vec3d f = (c.target - c.position).Normalized(), r = f.Cross(c.up).Normalized();
  require(r.Length() > 0, "Camera up is parallel to view");
  require(
      c.projection == Projection::Perspective || c.projection == Projection::Orthographic,
      "Unknown projection");
  require(
      std::isfinite(c.vertical_fov) && c.vertical_fov > 0 && c.vertical_fov < 179,
      "Invalid camera FOV");
  require(std::isfinite(c.ortho_height) && c.ortho_height > 0, "Invalid orthographic height");
  require(
      std::isfinite(c.near_clip) && std::isfinite(c.far_clip) && c.near_clip > 0 &&
          c.far_clip > c.near_clip,
      "Invalid camera clip range");
  return {f, r, r.Cross(f)};
}
Ray camera_ray(const Camera& c, Frame f, double x, double y, int w, int h) {
  double u = 2 * x / w - 1, v = 1 - 2 * y / h, aspect = double(w) / h;
  if (c.projection == Projection::Orthographic)
    return {
        c.position + f.right * (u * aspect * c.ortho_height * .5) +
            f.up * (v * c.ortho_height * .5),
        f.forward};
  double s = std::tan(DegToRad(c.vertical_fov) * .5);
  return {c.position, (f.forward + f.right * (u * aspect * s) + f.up * (v * s)).Normalized()};
}
struct Box {
  Vec3d lo{INFINITY, INFINITY, INFINITY}, hi{-INFINITY, -INFINITY, -INFINITY};
  void  add(Vec3d p) {
    for (int i = 0; i < 3; ++i) {
      lo[i] = std::min(lo[i], p[i]);
      hi[i] = std::max(hi[i], p[i]);
    }
  }
  void add(const Box& b) {
    add(b.lo);
    add(b.hi);
  }
  bool intersects(const detail::BoundsRay& r, double near, double far) const {
    return detail::box_reciprocal(lo, hi, r, near, far);
  }
};
struct Triangle {
  Vec3d           p[3], n[3], tangent;
  Vec2d           uv[3];
  int             object = -1, index = -1;
  const Object3D* material = nullptr;
  const Scene*    source   = nullptr; ///< Owner of this triangle's local texture/material indices.
  double determinant_epsilon = 0;     ///< Original relative parallel-ray threshold, prepared once.
};
struct Intersection {
  int    triangle = -1;
  double distance = INFINITY, u = 0, v = 0;
};
struct Node {
  Box         box;
  std::size_t start = 0, count = 0;
  int         left = -1, right = -1;
};
/// @brief Compact leaf-order intersection data; shading-only normals/UVs stay out of hot traversal.
struct QueryTriangle {
  Vec3d  origin; ///< First world vertex.
  Vec3d  edge1;  ///< Original second-minus-first vertex, prepared without arithmetic reassociation.
  Vec3d  edge2;  ///< Original third-minus-first vertex.
  double epsilon; ///< Original determinant tolerance.
  int    index;   ///< Global triangle index for material/depth/ID lookup.
};
class Acceleration {
 public:
  std::vector<Triangle>    triangles;
  std::vector<std::size_t> order;
  std::vector<Node>        nodes;
  std::vector<Box>         bounds; ///< Local triangle bounds separated from shading arrays.
  std::vector<Vec3d> centers; ///< Local lo+hi values, matching the original sorting comparator.
  std::vector<QueryTriangle> query; ///< Complete leaf-order hot data, including the parent.
  const Acceleration*        parent = nullptr; ///< Borrowed immutable prepared base, or null.
  std::size_t offset = 0; ///< Parent triangle count for globally addressable hit indices.

  /// @brief Prepare local geometry, optionally querying it together with an already prepared base.
  /// @param scene Borrowed scene whose materials/textures must outlive this acceleration.
  /// @param base Optional prepared parent that must outlive this acceleration.
  /// @param first_id Actor ID base; negative retains scene object IDs.
  /// @param build_now Build the hierarchy now, or defer it for a prepared immutable base.
  /// @throws std::invalid_argument Malformed scene geometry or materials.
  explicit Acceleration(
      const Scene&        scene,
      const Acceleration* base      = nullptr,
      int                 first_id  = -1,
      bool                build_now = true)
      : parent(base), offset(base ? base->size() : 0) {
    for (const auto& texture : scene.textures) {
      require(texture.width > 0 && texture.height > 0, "Invalid texture dimensions");
      require(
          static_cast<std::uint64_t>(texture.width) * texture.height * 4 == texture.pixels.size(),
          "Invalid RGBA texture size");
    }
    // Reserve once: a detailed robot otherwise repeatedly copies hundreds of MB
    // while geometrically growing its flattened triangle/node vectors.
    std::size_t capacity = 0;
    for (std::size_t object_index = 0; object_index < scene.objects.size(); ++object_index) {
      const auto& o = scene.objects[object_index];
      if (!o.visible || o.mesh_index == -1 || o.color.a == 0)
        continue;
      require(
          o.mesh_index >= 0 && static_cast<std::size_t>(o.mesh_index) < scene.meshes.size(),
          "Invalid object mesh index");
      capacity += scene.meshes[o.mesh_index].indices.size() / 3;
    }
    triangles.reserve(capacity);
    bounds.reserve(capacity);
    centers.reserve(capacity);
    for (std::size_t object_index = 0; object_index < scene.objects.size(); ++object_index) {
      const auto& o = scene.objects[object_index];
      if (!o.visible || o.mesh_index == -1 || o.color.a == 0)
        continue;
      require(
          o.mesh_index >= 0 && static_cast<std::size_t>(o.mesh_index) < scene.meshes.size(),
          "Invalid object mesh index");
      color_valid(o.color);
      require(
          finite(point(o.position)) && finite(point(o.rotation_deg)) && finite(point(o.scale)),
          "Nonfinite object transform");
      require(o.scale.x != 0 && o.scale.y != 0 && o.scale.z != 0, "Singular object scale");
      require(
          std::isfinite(o.roughness) && std::isfinite(o.metallic) && o.roughness >= 0 &&
              o.roughness <= 1 && o.metallic >= 0 && o.metallic <= 1,
          "Invalid material scalars");
      for (int i :
           {o.texture_index, o.normal_map_index, o.roughness_map_index, o.metallic_map_index})
        require(
            i == -1 || (i >= 0 && static_cast<std::size_t>(i) < scene.textures.size()),
            "Invalid material texture index");
      const auto& m = scene.meshes[o.mesh_index];
      const auto  n = m.positions.size() / 3;
      require(m.positions.size() % 3 == 0 && m.indices.size() % 3 == 0, "Malformed mesh arrays");
      require(
          m.normals.empty() || m.normals.size() == m.positions.size(), "Malformed normal array");
      require(m.texcoords.empty() || m.texcoords.size() == n * 2, "Malformed UV array");
      for (float v : m.positions)
        require(std::isfinite(v), "Nonfinite mesh position");
      for (float v : m.normals)
        require(std::isfinite(v), "Nonfinite mesh normal");
      for (float v : m.texcoords)
        require(std::isfinite(v), "Nonfinite texture coordinate");
      for (auto i : m.indices)
        require(i < n, "Mesh index out of bounds");
      Mat4d tr = Mat4d::Translation(point(o.position)) *
                 Mat4d::Rotation({0, 0, 1}, DegToRad(o.rotation_deg.z)) *
                 Mat4d::Rotation({0, 1, 0}, DegToRad(o.rotation_deg.y)) *
                 Mat4d::Rotation({1, 0, 0}, DegToRad(o.rotation_deg.x)) *
                 Mat4d::Scaling(point(o.scale));
      // Use precisely TransformNormal's arithmetic, including singular fallback,
      // but share its inverse-transpose across all vertices of this object.
      Mat3d      inverse;
      const bool invertible    = tr.LinearPart().Inverse(&inverse);
      const auto normal_matrix = inverse.Transposed();
      for (std::size_t i = 0; i < m.indices.size() / 3; ++i) {
        Triangle t;
        Box      box;
        t.object   = first_id < 0 ? o.id : first_id + int(object_index);
        t.index    = static_cast<int>(i);
        t.material = &o;
        t.source   = &scene;
        for (int j = 0; j < 3; ++j) {
          auto k = m.indices[3 * i + j];
          t.p[j] = tr.TransformPoint(
              {m.positions[3 * k], m.positions[3 * k + 1], m.positions[3 * k + 2]});
          require(finite(t.p[j]), "Transformed mesh position overflow");
          box.add(t.p[j]);
          if (!m.normals.empty()) {
            Vec3d normal{m.normals[3 * k], m.normals[3 * k + 1], m.normals[3 * k + 2]};
            t.n[j] =
                (invertible ? normal_matrix * normal : tr.TransformVector(normal)).Normalized();
          }
          if (!m.texcoords.empty())
            t.uv[j] = {m.texcoords[2 * k], m.texcoords[2 * k + 1]};
        }
        Vec3d normal = (t.p[1] - t.p[0]).Cross(t.p[2] - t.p[0]);
        if (normal.Length() == 0)
          continue;
        t.determinant_epsilon = 1e-15 * (t.p[1] - t.p[0]).Length() * (t.p[2] - t.p[0]).Length();
        for (auto& v : t.n)
          if (v.Length() == 0)
            v = normal.Normalized();
        auto   du = t.uv[1] - t.uv[0], dv = t.uv[2] - t.uv[0];
        double det = du.x * dv.y - du.y * dv.x;
        t.tangent  = std::abs(det) > 1e-20
                         ? ((t.p[1] - t.p[0]) * dv.y - (t.p[2] - t.p[0]) * du.y) / det
                         : normal.AnyPerpendicular();
        triangles.push_back(t);
        bounds.push_back(box);
        centers.push_back(box.lo + box.hi);
      }
    }
    if (!build_now || (parent && triangles.empty()))
      return;
    index();
  }
  /// @brief Build the exact flattened-order hierarchy once over existing transformed triangles.
  /// Called during construction, or under PreparedScene's one-time initialization guard.
  /// @param prepare_queries Prepare CPU leaf records; false is for transient GPU-only hierarchies.
  void index(bool prepare_queries = true) {
    nodes.clear();
    nodes.reserve(size() / 2 + 1);
    // Rebuild only the lightweight combined hierarchy. Its initial global
    // triangle order matches a flattened Scene exactly, including coincident
    // surfaces; the large parent's validated/transformed triangles stay shared.
    order.resize(size());
    std::iota(order.begin(), order.end(), 0);
    if (!order.empty())
      build(0, order.size());
    query.clear();
    if (!prepare_queries)
      return;
    query.reserve(order.size());
    for (auto index : order) {
      const auto& t = triangle(index);
      query.push_back(
          {t.p[0], t.p[1] - t.p[0], t.p[2] - t.p[0], t.determinant_epsilon, int(index)});
    }
  }
  /// @brief Return the number of addressable triangles including an immutable parent.
  /// @return Parent plus local triangle count.
  std::size_t size() const { return offset + triangles.size(); }
  /// @brief Resolve a globally numbered hit into its original owned triangle/material.
  /// @param index Valid hit index returned by closest().
  /// @return Borrowed triangle reference valid for this acceleration's lifetime.
  const Triangle& triangle(std::size_t index) const {
    return index < offset ? parent->triangle(index) : triangles[index - offset];
  }
  /// @brief Resolve compact world bounds without loading a shading triangle.
  /// @param index Global triangle index.
  /// @return Borrowed immutable bounds.
  const Box& bound(std::size_t index) const {
    return index < offset ? parent->bound(index) : bounds[index - offset];
  }
  /// @brief Resolve the precomputed original lo+hi sort key.
  /// @param index Global triangle index.
  /// @return Borrowed three-axis doubled centroid; no averaging/reassociation.
  const Vec3d& center(std::size_t index) const {
    return index < offset ? parent->center(index) : centers[index - offset];
  }
  int build(std::size_t start, std::size_t count) {
    int id = static_cast<int>(nodes.size());
    nodes.emplace_back();
    Box b;
    for (std::size_t i = start; i < start + count; ++i)
      b.add(bound(order[i]));
    nodes[id].box   = b;
    nodes[id].start = start;
    nodes[id].count = count;
    if (count <= 8)
      return id;
    auto span = b.hi - b.lo;
    int  axis = span.y > span.x ? 1 : 0;
    if (span.z > span[axis])
      axis = 2;
    auto mid = start + count / 2;
    std::nth_element(
        order.begin() + start,
        order.begin() + mid,
        order.begin() + start + count,
        [&](auto a, auto c) { return center(a)[axis] < center(c)[axis]; });
    int left = build(start, mid - start), right = build(mid, start + count - mid);
    nodes[id].left  = left;
    nodes[id].right = right;
    nodes[id].count = 0;
    return id;
  }
  /// @brief Query the nearest triangle, or return the first qualifying blocker for shadow rays.
  /// @param ray Normalized world-space ray.
  /// @param near Inclusive minimum ray distance.
  /// @param far Exclusive maximum ray distance.
  /// @param both Whether to intersect back-facing triangles; default true.
  /// @param any Whether a first blocker suffices; false preserves nearest-hit traversal.
  /// @return Nearest hit, arbitrary blocker when any is true, or triangle=-1.
  Intersection closest(const Ray& ray, double near, double far, bool both = true, bool any = false)
      const {
    Intersection result;
    result.distance = far;
    if (nodes.empty())
      return parent ? parent->closest(ray, near, far, both, any) : result;
    const detail::BoundsRay bounds_ray(ray);
    std::array<int, 128>    stack;
    int                     size = 1;
    stack[0]                     = 0;
    while (size) {
      const auto& node = nodes[stack[--size]];
      if (!node.box.intersects(bounds_ray, near, result.distance))
        continue;
      if (!node.count) {
        stack[size++] = node.left;
        stack[size++] = node.right;
        continue;
      }
      for (std::size_t i = node.start; i < node.start + node.count; ++i) {
        const auto& t   = query[i];
        const auto& e1  = t.edge1;
        const auto& e2  = t.edge2;
        Vec3d       p   = ray.direction.Cross(e2);
        double      det = e1.Dot(p);
        if ((!both && det <= 0) || std::abs(det) < t.epsilon)
          continue;
        auto   v = ray.origin - t.origin;
        double u = v.Dot(p) / det;
        if (u < 0 || u > 1)
          continue;
        auto   q = v.Cross(e1);
        double w = ray.direction.Dot(q) / det;
        if (w < 0 || u + w > 1)
          continue;
        double distance = e2.Dot(q) / det;
        if (distance >= near && distance < result.distance) {
          result = {t.index, distance, u, w};
          if (any)
            return result;
        }
      }
    }
    return result;
  }
};
RgbaColorF sample(const Scene& s, int index, Vec2d uv) {
  if (index < 0)
    return {1, 1, 1, 1};
  const auto& t = s.textures[index];
  // Source renderer uses repeating, bilinear texture sampling.
  double x  = (uv.x - std::floor(uv.x)) * t.width - .5,
         y  = (uv.y - std::floor(uv.y)) * t.height - .5;
  int    ix = static_cast<int>(std::floor(x)), iy = static_cast<int>(std::floor(y));
  double fx = x - ix, fy = y - iy;
  double value[4] = {};
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 2; ++i) {
      int px        = ((ix + i) % t.width + t.width) % t.width,
          py        = ((iy + j) % t.height + t.height) % t.height;
      double weight = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
      for (int c = 0; c < 4; ++c)
        value[c] += weight * t.pixels[(static_cast<std::size_t>(py) * t.width + px) * 4 + c] / 255.;
    }
  return {float(value[0]), float(value[1]), float(value[2]), float(value[3])};
}
PickResult hit_value(const Triangle& t, Intersection h, const Ray& r, bool smooth) {
  double a = 1 - h.u - h.v;
  Vec3d  n = smooth ? (t.n[0] * a + t.n[1] * h.u + t.n[2] * h.v).Normalized()
                    : (t.p[1] - t.p[0]).Cross(t.p[2] - t.p[0]).Normalized();
  if (n.Dot(r.direction) > 0)
    n = -n;
  return {
      true,
      t.object,
      t.index,
      h.distance,
      r.origin + r.direction * h.distance,
      n,
      t.uv[0] * a + t.uv[1] * h.u + t.uv[2] * h.v};
}
std::vector<Light> lights(const Scene& scene) {
  std::vector<Light> r;
  for (auto l : scene.lights) {
    if (!l.visible)
      continue;
    color_valid(l.color);
    require(finite(point(l.position)) && finite(point(l.direction)), "Nonfinite light");
    require(
        std::isfinite(l.intensity) && l.intensity >= 0 && std::isfinite(l.range) && l.range > 0,
        "Invalid light intensity/range");
    require(l.type == LightType::Point || l.type == LightType::Directional, "Unknown light type");
    if (l.type == LightType::Directional)
      require(point(l.direction).Length() > 0, "Zero light direction");
    r.push_back(l);
  }
  if (scene.lights.empty())
    r.push_back(Light{});
  return r;
}
/// @brief Decode one sRGB channel to linear radiance for the opt-in material workflow.
double decode_srgb(double v) {
  return v <= .04045 ? v / 12.92 : std::pow((v + .055) / 1.055, 2.4);
}
/// @brief Render-local light data with its constant directional normalization already evaluated.
struct ShadingLight {
  Light light;     ///< Original validated light, preserving scalar/material multiplication order.
  Vec3d direction; ///< The same normalized direction previously recomputed at each hit.
  std::vector<Vec3d> shadow_directions; ///< Pixel-independent disk rays for a directional key.
};
/// @brief Evaluate one original finite-disk ray without changing arithmetic grouping.
/// @param direction Unit light direction.
/// @param u First disk basis.
/// @param v Second disk basis.
/// @param k Sample index.
/// @param o Validated render sampling settings.
/// @return The original normalized soft-shadow ray direction.
Vec3d disk_direction(Vec3d direction, Vec3d u, Vec3d v, int k, const RenderOptions& o) {
  double r     = o.shadow_samples == 1
                     ? 0
                     : std::tan(o.light_angular_radius) * std::sqrt((k + .5) / o.shadow_samples);
  double angle = k * 2.399963229728653;
  return (direction + u * (r * std::cos(angle)) + v * (r * std::sin(angle))).Normalized();
}
/// @brief Validate lights and prepare only camera/pixel-independent direction work.
/// @param scene Borrowed scene; unchanged.
/// @param o Validated render options controlling key-ray samples.
/// @return Owned per-render light records, including the original default when needed.
/// @throws std::invalid_argument Invalid light fields.
std::vector<ShadingLight> shading_lights(const Scene& scene, const RenderOptions& o) {
  std::vector<ShadingLight> prepared;
  for (auto light : lights(scene)) {
    ShadingLight item{light, point(light.direction).Normalized(), {}};
    if (prepared.empty() && o.shadows && light.type == LightType::Directional) {
      auto d = item.direction,
           u = d.Cross(std::abs(d.z) < .9 ? Vec3d{0, 0, 1} : Vec3d{0, 1, 0}).Normalized(),
           v = d.Cross(u);
      for (int k = 0; k < o.shadow_samples; ++k)
        item.shadow_directions.push_back(disk_direction(d, u, v, k, o));
    }
    prepared.push_back(std::move(item));
  }
  return prepared;
}
/// @brief Cache the original cosine-hemisphere coefficients, independent of hit geometry.
/// @param count Validated AO sample count, including zero.
/// @return Owned r*cos(angle), r*sin(angle), sqrt(1-r*r) triples.
std::vector<Vec3d> occlusion_samples(int count) {
  std::vector<Vec3d> samples;
  samples.reserve(count);
  for (int k = 0; k < count; ++k) {
    double r = std::sqrt((k + .5) / count), angle = k * 2.399963229728653;
    samples.push_back({r * std::cos(angle), r * std::sin(angle), std::sqrt(1 - r * r)});
  }
  return samples;
}
/// @brief Apply a bounded filmic display curve and encode one exposed linear channel as sRGB.
double display_channel(double v, double exposure) {
  double x = std::max(0., v * exposure);
  double f = std::clamp(x * (2.51 * x + .03) / (x * (2.43 * x + .59) + .14), 0., 1.);
  return f <= .0031308 ? 12.92 * f : 1.055 * std::pow(f, 1 / 2.4) - .055;
}
/// @brief Deterministic finite-disk key visibility; defaults exactly retain legacy hard shadows.
/// @param a Immutable scene acceleration.
/// @param hit Surface position used for the original scale-dependent ray bias.
/// @param normal Surface normal pointing toward the visible side.
/// @param direction Unit direction toward the key light.
/// @param distance Light distance or the original directional-light maximum.
/// @param o Validated shadow settings.
/// @param cached Directional disk rays, or null to compute point-light rays at this hit.
/// @return Mean visibility, retaining the legacy or physical blocked contribution.
double key_visibility(
    const Acceleration&       a,
    const PickResult&         hit,
    Vec3d                     normal,
    Vec3d                     direction,
    double                    distance,
    const RenderOptions&      o,
    const std::vector<Vec3d>* cached) {
  double bias = std::max(1e-8, hit.position.Length() * 1e-7);
  Vec3d  u, v;
  if (!cached) {
    u = direction.Cross(std::abs(direction.z) < .9 ? Vec3d{0, 0, 1} : Vec3d{0, 1, 0}).Normalized();
    v = direction.Cross(u);
  }
  double sum = 0;
  for (int k = 0; k < o.shadow_samples; ++k) {
    auto d     = cached ? (*cached)[k] : disk_direction(direction, u, v, k, o);
    auto block = a.closest({hit.position + normal * bias, d}, bias, distance - bias, true, true);
    sum += block.triangle >= 0 ? (o.physically_based ? .06 : .35) : 1.;
  }
  return sum / o.shadow_samples;
}
/// @brief Local cosine-hemisphere ambient visibility, using stable low-discrepancy directions.
/// @param a Immutable scene acceleration.
/// @param hit Visible surface position and normal.
/// @param o Validated AO sampling settings.
/// @param samples Per-render hemisphere coefficients in original sample order.
/// @return Original distance-weighted visibility, or one when AO is disabled.
double local_visibility(
    const Acceleration&       a,
    const PickResult&         hit,
    const RenderOptions&      o,
    const std::vector<Vec3d>& samples) {
  if (!o.ambient_occlusion_samples)
    return 1;
  auto n     = hit.normal;
  auto u     = n.Cross(std::abs(n.z) < .9 ? Vec3d{0, 0, 1} : Vec3d{0, 1, 0}).Normalized(),
       v     = n.Cross(u);
  double sum = 0, bias = std::max(1e-7, hit.position.Length() * 1e-6);
  for (int k = 0; k < o.ambient_occlusion_samples; ++k) {
    const auto& sample = samples[k];
    auto        d      = u * sample.x + v * sample.y + n * sample.z;
    auto        block = a.closest({hit.position + n * bias, d}, bias, o.ambient_occlusion_distance);
    sum +=
        block.triangle < 0 ? 1 : std::clamp(block.distance / o.ambient_occlusion_distance, 0., 1.);
  }
  return .22 + .78 * sum / o.ambient_occlusion_samples;
}
/// @brief Approximate a fixed studio sky/ground and two broad softbox reflection lobes.
/// @param direction Unit world reflection direction.
/// @param roughness Clamped material roughness, broadening the softbox lobes.
/// @return Linear HDR radiance; does not trace scene reflections.
Vec3d studio_radiance(Vec3d direction, double roughness) {
  const Vec3d ground{.055, .062, .07}, sky{.55, .64, .76};
  double      t        = std::clamp(.5 + .5 * direction.z, 0., 1.);
  auto        value    = ground * (1 - t) + sky * t;
  double      exponent = 3 + 100 * (1 - roughness) * (1 - roughness);
  auto        warm = Vec3d{-.45, -.5, 1}.Normalized(), cool = Vec3d{.7, .3, .65}.Normalized();
  return value +
         Vec3d{1, .95, .87} * (2.5 * std::pow(std::max(0., direction.Dot(warm)), exponent)) +
         Vec3d{.72, .84, 1} * (1.1 * std::pow(std::max(0., direction.Dot(cool)), exponent));
}
/// @brief Approximate diffuse hemisphere irradiance and rough reflected studio radiance.
/// @param color Linear material RGB.
/// @param spec Normal-incidence reflectance.
/// @param normal Mapped unit surface normal.
/// @param incoming Unit camera ray direction.
/// @param roughness Clamped roughness.
/// @param metal Metallic fraction in [0,1].
/// @return Ambient surface radiance before visibility and user strength.
Vec3d studio_surface(
    Vec3d  color,
    Vec3d  spec,
    Vec3d  normal,
    Vec3d  incoming,
    double roughness,
    double metal) {
  double t = .5 + .25 * normal.z, nv = std::clamp(-normal.Dot(incoming), 0., 1.);
  auto   irradiance = Vec3d{.055, .062, .07} * (1 - t) + Vec3d{.55, .64, .76} * t;
  auto   grazing    = Vec3d{
      std::max(spec.x, 1 - roughness),
      std::max(spec.y, 1 - roughness),
      std::max(spec.z, 1 - roughness)};
  auto fresnel   = spec + (grazing - spec) * std::pow(1 - nv, 5);
  auto reflected = incoming - normal * (2 * normal.Dot(incoming));
  return multiply(multiply(Vec3d{1, 1, 1} - fresnel, color), irradiance) * (1 - metal) +
         multiply(fresnel, studio_radiance(reflected, roughness)) * (1 - .35 * roughness);
}
/// @brief Shade a visible surface with legacy lighting or the opt-in linear microfacet workflow.
Vec3d shade(
    const Scene&                     scene,
    const Acceleration&              a,
    const Triangle&                  t,
    const PickResult&                hit,
    const Ray&                       ray,
    const RenderOptions&             o,
    const std::vector<ShadingLight>& lamps,
    const std::vector<Vec3d>&        ao_samples,
    RgbaColorF                       base) {
  Vec3d color{base.r, base.g, base.b};
  if (o.physically_based)
    color = {decode_srgb(color.x), decode_srgb(color.y), decode_srgb(color.z)};
  if (o.style == RenderStyle::Unlit)
    return color;
  Vec3d n = hit.normal;
  if (t.material->normal_map_index >= 0) {
    auto c       = sample(scene, t.material->normal_map_index, hit.uv);
    auto tangent = (t.tangent - n * n.Dot(t.tangent)).Normalized();
    if (tangent.Length() > 0)
      n = (tangent * (2 * c.r - 1) + n.Cross(tangent) * (2 * c.g - 1) + n * (2 * c.b - 1))
              .Normalized();
  }
  double rough = t.material->roughness_map_index < 0
                     ? t.material->roughness
                     : sample(scene, t.material->roughness_map_index, hit.uv).r;
  double metal = t.material->metallic_map_index < 0
                     ? t.material->metallic
                     : sample(scene, t.material->metallic_map_index, hit.uv).r;
  rough        = std::clamp(rough, .04, 1.);
  Vec3d spec   = Vec3d{.04, .04, .04} * (1 - metal) + color * metal;
  Vec3d lit    = color * ((1 - metal) * .35);
  if (o.physically_based) {
    auto ambient =
        o.studio_environment > 0
            ? studio_surface(color, spec, n, ray.direction, rough, metal) * o.studio_environment
            : color * ((1 - metal) * .24) + spec * (.20 + .35 * rough);
    lit = ambient * local_visibility(a, hit, o, ao_samples);
  } else if (o.ambient_occlusion_samples)
    lit *= local_visibility(a, hit, o, ao_samples);
  for (std::size_t i = 0; i < lamps.size(); ++i) {
    const auto& l           = lamps[i].light;
    Vec3d       direction   = lamps[i].direction;
    double      attenuation = 1, distance = 1e100;
    if (l.type == LightType::Point) {
      auto toward = point(l.position) - hit.position;
      distance    = toward.Length();
      direction   = distance > 1e-9 ? toward / distance : Vec3d{0, 1, 0};
      attenuation = std::clamp(1 - distance / l.range, 0., 1.);
    }
    double ndotl = std::max(0., n.Dot(direction));
    if (ndotl <= 0 || attenuation == 0)
      continue;
    double shadow = 1;
    if (o.shadows && i == 0) {
      shadow = key_visibility(
          a,
          hit,
          hit.normal,
          direction,
          distance,
          o,
          lamps[i].shadow_directions.empty() ? nullptr : &lamps[i].shadow_directions);
    }
    auto half  = (direction - ray.direction).Normalized();
    auto light = Vec3d{l.color.r, l.color.g, l.color.b} * (l.intensity * attenuation * shadow);
    if (o.physically_based) {
      const double pi = 3.14159265358979323846;
      double       nv = std::max(.001, n.Dot(-ray.direction)), nh = std::max(0., n.Dot(half)),
             vh    = std::max(0., (-ray.direction).Dot(half));
      double alpha = rough * rough, a2 = alpha * alpha, den = nh * nh * (a2 - 1) + 1;
      double distribution = a2 / (pi * den * den);
      double k            = (rough + 1) * (rough + 1) / 8;
      double masking      = nv / (nv * (1 - k) + k) * ndotl / (ndotl * (1 - k) + k);
      auto   fresnel      = spec + (Vec3d{1, 1, 1} - spec) * std::pow(1 - vh, 5);
      auto   diffuse      = multiply(Vec3d{1, 1, 1} - fresnel, color) * ((1 - metal) / pi);
      auto   microfacet   = fresnel * (distribution * masking / std::max(4 * nv * ndotl, 1e-8));
      lit += multiply(diffuse + microfacet, light) * ndotl;
    } else {
      // The legacy lobe is unused by GGX; do not evaluate its pow() for PBR pixels.
      double strength = std::pow(std::max(0., n.Dot(half)), 128 + (4 - 128) * rough) * ndotl;
      lit += multiply(color * ((1 - metal) * .65 * ndotl) + spec * strength, light);
    }
  }
  return lit;
}
RgbaColorF trace(
    const Scene&                     scene,
    const Acceleration&              a,
    Ray                              ray,
    const Camera&                    camera,
    const RenderOptions&             o,
    const std::vector<ShadingLight>& lamps,
    const std::vector<Vec3d>&        ao_samples,
    const Intersection*              first = nullptr) {
  Vec3d  color;
  double alpha = 0, remaining = 1, near = camera.near_clip;
  for (int layer = 0; layer < o.transparency_layers && remaining > 1e-5; ++layer) {
    auto h = layer == 0 && first ? *first : a.closest(ray, near, camera.far_clip, o.double_sided);
    if (h.triangle < 0)
      break;
    const auto& t    = a.triangle(h.triangle);
    auto        hit  = hit_value(t, h, ray, o.smooth_normals);
    auto        tex  = sample(*t.source, t.material->texture_index, hit.uv);
    auto        tint = t.material->color;
    RgbaColorF  base{tex.r * tint.r, tex.g * tint.g, tex.b * tint.b, tex.a * tint.a};
    auto        c = shade(*t.source, a, t, hit, ray, o, lamps, ao_samples, base);
    if (o.style == RenderStyle::Wireframe || o.style == RenderStyle::ShadedWireframe) {
      double pixel  = camera.projection == Projection::Perspective
                          ? 2 * h.distance * std::tan(DegToRad(camera.vertical_fov) * .5) / o.height
                          : camera.ortho_height / o.height;
      auto   normal = (t.p[1] - t.p[0]).Cross(t.p[2] - t.p[0]);
      double area   = normal.Length();
      double edge   = std::min(
          {(1 - h.u - h.v) * area / (t.p[2] - t.p[1]).Length(),
             h.u * area / (t.p[2] - t.p[0]).Length(),
             h.v * area / (t.p[1] - t.p[0]).Length()});
      if (edge < o.wire_width * pixel) {
        c      = {o.wire_color.r, o.wire_color.g, o.wire_color.b};
        base.a = o.wire_color.a;
      } else if (o.style == RenderStyle::Wireframe)
        base.a = 0;
    }
    color += c * (remaining * base.a);
    alpha += remaining * base.a;
    remaining *= 1 - base.a;
    near = h.distance + std::max(1e-9, std::abs(h.distance) * 1e-8);
  }
  Vec3d backdrop{o.background.r, o.background.g, o.background.b};
  if (o.physically_based)
    backdrop = {decode_srgb(backdrop.x), decode_srgb(backdrop.y), decode_srgb(backdrop.z)};
  color += backdrop * (remaining * o.background.a);
  alpha += remaining * o.background.a;
  if (alpha > 0)
    color /= alpha;
  if (o.physically_based)
    color = {
        display_channel(color.x, o.exposure),
        display_channel(color.y, o.exposure),
        display_channel(color.z, o.exposure)};
  return {float(color.x), float(color.y), float(color.z), float(alpha)};
}
} // namespace
Ray Camera::ray(double x, double y, int w, int h) const {
  require(
      w > 0 && h > 0 && std::isfinite(x) && std::isfinite(y),
      "Invalid pixel coordinate/dimensions");
  return camera_ray(*this, frame(*this), x, y, w, h);
}
Camera fit_camera(const Scene& scene, Vec3d direction, Vec3d up, double aspect) {
  require(
      finite(direction) && direction.Length() > 0 && finite(up) && std::isfinite(aspect) &&
          aspect > 0,
      "Invalid camera fit arguments");
  Acceleration a(scene);
  require(!a.nodes.empty(), "Cannot fit an empty scene");
  auto   b      = a.nodes[0].box;
  auto   centre = (b.lo + b.hi) * .5;
  double radius = (b.hi - b.lo).Length() * .5;
  Camera c;
  c.target       = centre;
  c.up           = up;
  double angle   = std::atan(std::tan(DegToRad(c.vertical_fov) * .5) * std::min(1., aspect));
  c.position     = centre + direction.Normalized() * (radius / std::sin(angle) * 1.15);
  c.ortho_height = radius * 2.3 / std::min(1., aspect);
  c.near_clip    = std::max(1e-12, radius * 1e-5);
  c.far_clip     = std::max(c.near_clip * 10, (c.position - centre).Length() + radius * 2);
  frame(c);
  return c;
}
PickResult pick(const Scene& scene, const Ray& ray, double near, double far) {
  require(finite(ray.origin) && finite(ray.direction) && ray.direction.Length() > 0, "Invalid ray");
  require(
      std::isfinite(near) && std::isfinite(far) && near > 0 && far > near, "Invalid clip range");
  Acceleration a(scene);
  Ray          normalized{ray.origin, ray.direction.Normalized()};
  auto         h = a.closest(normalized, near, far);
  return h.triangle < 0 ? PickResult{} : hit_value(a.triangle(h.triangle), h, normalized, true);
}
namespace {
/// @brief Share the exact CPU option checks with the explicit CUDA renderer.
/// @param o Borrowed image, sampling, material and worker settings.
/// @throws std::invalid_argument Invalid render settings.
void validate_render_options(const RenderOptions& o) {
  require(
      o.width > 0 && o.height > 0 && o.width <= 8192 && o.height <= 8192,
      "Invalid render dimensions (maximum 8192)");
  require(
      o.samples >= 1 && o.samples <= 4 && o.threads >= 0 && o.threads <= 256,
      "Invalid sample/thread count");
  require(
      o.transparency_layers > 0 && o.transparency_layers <= 256,
      "Invalid transparency layer count");
  require(std::isfinite(o.wire_width) && o.wire_width > 0, "Invalid wire width");
  require(
      o.style == RenderStyle::Shaded || o.style == RenderStyle::Unlit ||
          o.style == RenderStyle::Wireframe || o.style == RenderStyle::ShadedWireframe,
      "Unknown render style");
  require(
      o.shadow_samples >= 1 && o.shadow_samples <= 32 && std::isfinite(o.light_angular_radius) &&
          o.light_angular_radius >= 0 && o.light_angular_radius <= .3,
      "Invalid soft shadow settings");
  require(
      o.ambient_occlusion_samples >= 0 && o.ambient_occlusion_samples <= 16 &&
          std::isfinite(o.ambient_occlusion_distance) && o.ambient_occlusion_distance > 0,
      "Invalid ambient occlusion settings");
  require(std::isfinite(o.exposure) && o.exposure > 0, "Invalid exposure");
  require(
      std::isfinite(o.studio_environment) && o.studio_environment >= 0 && o.studio_environment <= 4,
      "Invalid studio environment strength");
  color_valid(o.background);
  color_valid(o.wire_color);
}
/// @brief Render a borrowed prepared acceleration with base lighting and owned output buffers.
/// @param scene Base scene supplying lights and background/material defaults.
/// @param a Prepared geometry, including any transient actor parent relationship.
/// @param camera Valid view.
/// @param o Checked image/shading/worker settings.
/// @return Owned synchronous color, depth and object IDs.
/// @throws std::invalid_argument Invalid camera, lighting or render options.
RenderResult render_acceleration(
    const Scene&         scene,
    const Acceleration&  a,
    const Camera&        camera,
    const RenderOptions& o) {
  auto f = frame(camera);
  validate_render_options(o);
  auto         lamps      = shading_lights(scene, o);
  auto         ao_samples = occlusion_samples(o.ambient_occlusion_samples);
  RenderResult result;
  result.width      = o.width;
  result.height     = o.height;
  std::size_t count = static_cast<std::size_t>(o.width) * o.height;
  result.rgba.resize(count * 4);
  result.depth.assign(count, INFINITY);
  result.object_ids.assign(count, -1);
  std::atomic<int> row{0};
  auto             work = [&] {
    for (int y = row.fetch_add(1); y < o.height; y = row.fetch_add(1))
      for (int x = 0; x < o.width; ++x) {
        auto index  = static_cast<std::size_t>(y) * o.width + x;
        auto centre = camera_ray(camera, f, x + .5, y + .5, o.width, o.height);
        auto h = a.closest(centre, camera.near_clip, camera.far_clip, o.double_sided);
        if (h.triangle >= 0) {
          result.depth[index]      = h.distance;
          result.object_ids[index] = a.triangle(h.triangle).object;
        }
        double sum[4] = {};
        for (int sy = 0; sy < o.samples; ++sy)
          for (int sx = 0; sx < o.samples; ++sx) {
            auto r = camera_ray(
                camera, f, x + (sx + .5) / o.samples, y + (sy + .5) / o.samples, o.width, o.height);
            auto c =
                trace(scene, a, r, camera, o, lamps, ao_samples, o.samples == 1 ? &h : nullptr);
            sum[0] += c.r * c.a;
            sum[1] += c.g * c.a;
            sum[2] += c.b * c.a;
            sum[3] += c.a;
          }
        double alpha = sum[3] / (o.samples * o.samples);
        for (int i = 0; i < 3; ++i)
          result.rgba[index * 4 + i] = static_cast<unsigned char>(
              std::lround(std::clamp(sum[3] > 0 ? sum[i] / sum[3] : 0., 0., 1.) * 255));
        result.rgba[index * 4 + 3] =
            static_cast<unsigned char>(std::lround(std::clamp(alpha, 0., 1.) * 255));
      }
  };
  unsigned workers = o.threads ? static_cast<unsigned>(o.threads)
                               : std::min(8u, std::max(1u, std::thread::hardware_concurrency()));
  workers          = std::min(workers, static_cast<unsigned>(o.height));
  std::vector<std::jthread> threads;
  for (unsigned i = 1; i < workers; ++i)
    threads.emplace_back(work);
  work();
  for (auto& t : threads)
    t.join();
  return result;
}
} // namespace
/// @brief Immutable owned base scene; member order keeps borrowed material pointers valid.
struct PreparedScene::Impl {
  Scene                  scene;        ///< Owned mesh/material/light snapshot.
  mutable Acceleration   acceleration; ///< Fixed triangles; base-only hierarchy initialized once.
  mutable std::once_flag indexed; ///< Synchronizes the lazy base-only BVH across concurrent views.

  /// @brief Adopt geometry and prepare the shared acceleration once.
  /// @param owned Snapshot transferred into immutable ownership.
  /// @throws std::invalid_argument Invalid geometry, materials or lighting.
  explicit Impl(Scene owned) : scene(std::move(owned)), acceleration(scene, nullptr, -1, false) {
    lights(scene);
  }
};
PreparedScene::PreparedScene(Scene scene) : impl_(std::make_shared<Impl>(std::move(scene))) {}
RenderResult PreparedScene::render(const Camera& camera, const RenderOptions& options) const {
  std::call_once(impl_->indexed, [&] { impl_->acceleration.index(); });
  return render_acceleration(impl_->scene, impl_->acceleration, camera, options);
}
RenderResult PreparedScene::render(
    const Scene&         actors,
    const Camera&        camera,
    const RenderOptions& options) const {
  Acceleration complete(actors, &impl_->acceleration, impl_->scene.next_object_id);
  if (complete.triangles.empty())
    return render(camera, options);
  return render_acceleration(impl_->scene, complete, camera, options);
}
RenderResult render(const Scene& scene, const Camera& camera, const RenderOptions& options) {
  Acceleration acceleration(scene);
  return render_acceleration(scene, acceleration, camera, options);
}
namespace {
/// @brief GPU layout names stay private to this translation unit.
namespace gpu = detail::cuda;
/// @brief Convert finite values without silently overflowing or flushing a nonzero input.
/// @param value World/scalar value validated by the reference renderer.
/// @return Representable float32 value.
/// @throws std::invalid_argument Value overflows or underflows float32.
float device_float(double value) {
  auto f = static_cast<float>(value);
  require(std::isfinite(f) && (value == 0 || f != 0), "CAD CUDA input is outside float32 range");
  return f;
}
/// @brief Convert a world vector to the SDK-free device layout.
/// @param v Validated vector.
/// @return Checked float32 vector.
gpu::V3 device_vector(Vec3d v) {
  return {device_float(v.x), device_float(v.y), device_float(v.z)};
}
/// @brief Convert an already validated color to packed float32 RGBA.
/// @param c Color in [0,1].
/// @return Unchanged channel values.
gpu::V4 device_color(RgbaColorF c) {
  return {c.r, c.g, c.b, c.a};
}
/// @brief Check signed device offsets before narrowing buffer counts.
/// @param count Record/byte count.
/// @return Signed 32-bit count.
/// @throws std::length_error Count exceeds INT_MAX.
int device_count(std::size_t count) {
  if (count > std::size_t(std::numeric_limits<int>::max()))
    throw std::length_error("CAD CUDA index capacity exceeded");
  return static_cast<int>(count);
}
/// @brief Grow a retained device buffer only as needed and upload the used prefix.
/// @param context Creating CUDA context.
/// @param target Mutable retained buffer; capacity can exceed used bytes.
/// @param data Host source, null only for zero bytes.
/// @param bytes Logical prefix length; empty arrays retain a four-byte sentinel.
void device_upload(
    const compute::Context& context,
    compute::Buffer&        target,
    const void*             data,
    std::size_t             bytes) {
  if (!target.defined() || target.size() < std::max(bytes, std::size_t(4)))
    target = context.allocate(std::max(bytes, std::size_t(4)));
  if (bytes)
    context.upload(target, data, bytes);
}
/// @brief Upload one typed array through the checked byte transport.
/// @tparam T Trivially copyable GPU layout record.
/// @param context Creating context.
/// @param target Retained allocation.
/// @param values Owned host staging records.
template<class T>
void device_upload(
    const compute::Context& context,
    compute::Buffer&        target,
    const std::vector<T>&   values) {
  device_upload(context, target, values.data(), values.size() * sizeof(T));
}
/// @brief Persistent base or transient-actor device arrays, all owned by one renderer context.
struct DeviceGeometry {
  compute::Buffer triangles; ///< Local world triangles.
  compute::Buffer materials; ///< Material records in triangle-local index order.
  compute::Buffer textures;  ///< Local texture metadata.
  compute::Buffer pixels;    ///< Packed local RGBA8 texture bytes.
};
/// @brief Pack validated triangles/materials/textures once for one base or actor scene.
/// @param context Creating CUDA context.
/// @param output Device arrays; uploaded in place.
/// @param source Scene owning the local material/texture indices.
/// @param triangles Already transformed local triangles; parent data is uploaded separately.
void upload_geometry(
    const compute::Context&      context,
    DeviceGeometry&              output,
    const Scene&                 source,
    const std::vector<Triangle>& triangles) {
  device_count(triangles.size());
  std::vector<gpu::Triangle> packed;
  packed.reserve(triangles.size());
  std::vector<gpu::Material>               materials;
  std::unordered_map<const Object3D*, int> ids;
  for (const auto& t : triangles) {
    auto [entry, inserted] = ids.emplace(t.material, device_count(materials.size()));
    if (inserted) {
      const auto& m = *t.material;
      materials.push_back(
          {device_color(m.color),
           m.roughness,
           m.metallic,
           m.texture_index,
           m.normal_map_index,
           m.roughness_map_index,
           m.metallic_map_index});
    }
    gpu::Triangle q{};
    q.origin   = device_vector(t.p[0]);
    q.edge1    = device_vector(t.p[1] - t.p[0]);
    q.edge2    = device_vector(t.p[2] - t.p[0]);
    q.tangent  = device_vector(t.tangent);
    q.epsilon  = device_float(t.determinant_epsilon);
    q.object   = t.object;
    q.material = entry->second;
    for (int k = 0; k < 3; ++k) {
      q.normals[k]    = device_vector(t.n[k]);
      q.uv[2 * k]     = device_float(t.uv[k].x);
      q.uv[2 * k + 1] = device_float(t.uv[k].y);
    }
    packed.push_back(q);
  }
  std::vector<gpu::Texture>  textures;
  std::vector<unsigned char> pixels;
  for (const auto& t : source.textures) {
    device_count(pixels.size() + t.pixels.size());
    textures.push_back({t.width, t.height, device_count(pixels.size())});
    pixels.insert(pixels.end(), t.pixels.begin(), t.pixels.end());
  }
  device_upload(context, output.triangles, packed);
  device_upload(context, output.materials, materials);
  device_upload(context, output.textures, textures);
  device_upload(context, output.pixels, pixels);
}
} // namespace
/// @brief Serialized native context/program and bounded last-scene device cache.
struct CudaRenderer::Impl {
  compute::Context context; ///< Exclusive explicit CUDA context, retained by every allocation.
  compute::Program program; ///< JIT-compiled embedded ray tracing kernel.
  std::mutex       mutex;   ///< Serializes cache replacement, uploads, dispatch and readback.
  std::shared_ptr<const PreparedScene::Impl> base; ///< Last successfully uploaded owned snapshot.
  DeviceGeometry  fixed, actors; ///< Retained base and transient actor allocation capacities.
  compute::Buffer nodes, order, lights,
      output; ///< Reused per-call hierarchy/light/readback allocations.

  /// @brief Create a real GPU context and compile the fixed kernel layout, without fallback.
  /// @param device Explicit CUDA ordinal.
  explicit Impl(std::uint32_t device) : context({compute::Backend::CUDA, device}) {
    compute::KernelSource source;
    source.entry           = "cad_render";
    source.ptx             = detail::render_ptx;
    source.buffers         = 12;
    source.parameter_bytes = sizeof(gpu::Parameters);
    source.local_size      = {128, 1, 1};
    program                = context.program(source);
  }
};
CudaRenderer::CudaRenderer(std::uint32_t device) : impl_(std::make_unique<Impl>(device)) {}
CudaRenderer::~CudaRenderer() = default;
RenderResult CudaRenderer::render(
    const PreparedScene& base,
    const Scene&         actors,
    const Camera&        camera,
    const RenderOptions& o) {
  auto f = frame(camera);
  validate_render_options(o);
  auto                        lamps = shading_lights(base.impl_->scene, o);
  auto                        ao    = occlusion_samples(o.ambient_occlusion_samples);
  std::lock_guard<std::mutex> lock(impl_->mutex);
  auto&                       g = *impl_;
  Acceleration        complete(actors, &base.impl_->acceleration, base.impl_->scene.next_object_id);
  const Acceleration* acceleration = &complete;
  if (complete.triangles.empty()) {
    std::call_once(base.impl_->indexed, [&] { base.impl_->acceleration.index(); });
    acceleration = &base.impl_->acceleration;
  }
  device_count(acceleration->size());
  if (g.base != base.impl_) {
    g.base.reset();
    upload_geometry(g.context, g.fixed, base.impl_->scene, base.impl_->acceleration.triangles);
    g.base = base.impl_;
  }
  upload_geometry(g.context, g.actors, actors, complete.triangles);
  std::vector<gpu::Node> nodes;
  nodes.reserve(acceleration->nodes.size());
  for (const auto& n : acceleration->nodes) {
    auto lo = device_vector(n.box.lo), hi = device_vector(n.box.hi);
    lo = {
        std::nextafter(lo.x, -INFINITY),
        std::nextafter(lo.y, -INFINITY),
        std::nextafter(lo.z, -INFINITY)};
    hi = {
        std::nextafter(hi.x, INFINITY),
        std::nextafter(hi.y, INFINITY),
        std::nextafter(hi.z, INFINITY)};
    require(
        std::isfinite(lo.x) && std::isfinite(lo.y) && std::isfinite(lo.z) && std::isfinite(hi.x) &&
            std::isfinite(hi.y) && std::isfinite(hi.z),
        "CAD CUDA bounds overflow float32");
    nodes.push_back({lo, hi, device_count(n.start), device_count(n.count), n.left, n.right});
  }
  std::vector<int> order;
  order.reserve(acceleration->order.size());
  for (auto i : acceleration->order)
    order.push_back(device_count(i));
  std::vector<gpu::Light> lights;
  for (const auto& l : lamps)
    lights.push_back(
        {device_vector(point(l.light.position)),
         device_vector(l.direction),
         {l.light.color.r, l.light.color.g, l.light.color.b},
         l.light.intensity,
         l.light.range,
         l.light.type == LightType::Point});
  device_upload(g.context, g.nodes, nodes);
  device_upload(g.context, g.order, order);
  device_upload(g.context, g.lights, lights);
  gpu::Parameters p{};
  p.origin             = device_vector(camera.position);
  p.forward            = device_vector(f.forward);
  p.right              = device_vector(f.right);
  p.up                 = device_vector(f.up);
  p.scale              = device_float(std::tan(DegToRad(camera.vertical_fov) * .5));
  p.aspect             = float(o.width) / o.height;
  p.ortho_height       = device_float(camera.ortho_height);
  p.near_clip          = device_float(camera.near_clip);
  p.far_clip           = device_float(camera.far_clip);
  p.width              = o.width;
  p.height             = o.height;
  p.projection         = int(camera.projection);
  p.samples            = o.samples;
  p.style              = int(o.style);
  p.smooth             = o.smooth_normals;
  p.both               = o.double_sided;
  p.layers             = o.transparency_layers;
  p.shadows            = o.shadows;
  p.shadow_samples     = o.shadow_samples;
  p.ao_samples         = o.ambient_occlusion_samples;
  p.physical           = o.physically_based;
  p.ao_distance        = device_float(o.ambient_occlusion_distance);
  p.exposure           = device_float(o.exposure);
  p.wire_width         = device_float(o.wire_width);
  p.angular_radius     = device_float(o.light_angular_radius);
  p.studio_environment = device_float(o.studio_environment);
  p.background         = device_color(o.background);
  p.wire_color         = device_color(o.wire_color);
  p.base_count         = device_count(base.impl_->acceleration.triangles.size());
  p.node_count         = device_count(nodes.size());
  p.light_count        = device_count(lights.size());
  if (!lamps.empty())
    for (std::size_t k = 0; k < lamps[0].shadow_directions.size(); ++k)
      p.shadow_directions[k] = device_vector(lamps[0].shadow_directions[k]);
  for (std::size_t k = 0; k < ao.size(); ++k)
    p.ao[k] = device_vector(ao[k]);
  std::size_t             count = std::size_t(o.width) * o.height;
  std::vector<gpu::Pixel> pixels(count);
  if (!g.output.defined() || g.output.size() < count * sizeof(gpu::Pixel))
    g.output = g.context.allocate(count * sizeof(gpu::Pixel));
  g.context.dispatch(
      g.program,
      {g.fixed.triangles,
       g.actors.triangles,
       g.nodes,
       g.order,
       g.fixed.materials,
       g.actors.materials,
       g.fixed.textures,
       g.actors.textures,
       g.fixed.pixels,
       g.actors.pixels,
       g.lights,
       g.output},
      &p,
      sizeof(p),
      {static_cast<std::uint32_t>((count + 127) / 128), 1, 1});
  g.context.download(pixels.data(), g.output, count * sizeof(gpu::Pixel));
  RenderResult result;
  result.width  = o.width;
  result.height = o.height;
  result.rgba.resize(count * 4);
  result.depth.resize(count);
  result.object_ids.resize(count);
  for (std::size_t i = 0; i < count; ++i) {
    std::copy_n(pixels[i].rgba, 4, result.rgba.begin() + 4 * i);
    result.depth[i]      = pixels[i].depth;
    result.object_ids[i] = pixels[i].object;
  }
  return result;
}
void RenderResult::save(const std::string& path) const {
  require(
      width > 0 && height > 0 && static_cast<std::uint64_t>(width) * height * 4 == rgba.size(),
      "Malformed render buffer");
  auto ext = std::filesystem::path(path).extension().string();
  std::transform(
      ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
  std::string bytes;
  if (ext == ".png")
    bytes = detail::png::Encode(width, height, 4, rgba.data(), width * 4);
  else if (ext == ".jpg" || ext == ".jpeg")
    bytes = detail::jpeg::Encode(width, height, 4, rgba.data(), width * 4, 90);
  else if (ext == ".ppm") {
    bytes = "P6\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
    for (std::size_t i = 0; i < rgba.size(); i += 4)
      bytes.append(reinterpret_cast<const char*>(rgba.data() + i), 3);
  } else
    throw std::invalid_argument("Unsupported render output extension: " + ext);
  std::ofstream file(path, std::ios::binary);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  file.close();
  if (!file || bytes.empty())
    throw std::runtime_error("Cannot write render: " + path);
}
} // namespace datamunge::cad
