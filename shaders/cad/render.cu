/// @file render.cu
/// @brief Float32 CUDA CAD rays, mapped materials and offscreen image readback.
// CAD CUDA offscreen ray tracer. Regenerate the embedded PTX with
// just prebuild-nn-cuda-kernels --only render. Host layouts live in cuda_types.hpp.
// Float32 is intentional: CPU rendering remains the reference/sensor path.
#include <math.h>

#include "../../src/datamunge/cad/detail/cuda_types.hpp"
using namespace datamunge::cad::detail::cuda;
/// @brief Add two vector values.
__device__ V3 operator+(V3 a, V3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}
/// @brief Subtract or negate vector values.
__device__ V3 operator-(V3 a, V3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}
/// @brief Subtract or negate vector values.
__device__ V3 operator-(V3 a) {
  return {-a.x, -a.y, -a.z};
}
/// @brief Scale a vector by a scalar.
__device__ V3 operator*(V3 a, float b) {
  return {a.x * b, a.y * b, a.z * b};
}
/// @brief Divide a vector by a nonzero scalar.
__device__ V3 operator/(V3 a, float b) {
  return a * (1.f / b);
}
/// @brief Multiply vector components.
__device__ V3 mul(V3 a, V3 b) {
  return {a.x * b.x, a.y * b.y, a.z * b.z};
}
/// @brief Compute a vector dot product.
__device__ float dot(V3 a, V3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
/// @brief Compute a right-handed vector cross product.
__device__ V3 cross(V3 a, V3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
/// @brief Compute a vector dot product.
/// @brief Compute Euclidean vector length.
__device__ float length(V3 a) {
  return sqrtf(dot(a, a));
}
/// @brief Compute Euclidean vector length.
/// @brief Normalize a vector, retaining zero when its length is zero.
__device__ V3 unit(V3 a) {
  float n = length(a);
  return n > 0 ? a / n : V3{0, 0, 0};
}
/// @brief Clamp a finite scalar to inclusive bounds.
__device__ float clamp(float x, float a = 0, float b = 1) {
  return fminf(b, fmaxf(a, x));
}
/// @brief Read one vector axis.
__device__ float component(V3 a, int i) {
  return i == 0 ? a.x : i == 1 ? a.y : a.z;
}
/// All storage pointers are supplied by Context::dispatch; no host numerical fallback.
/// @brief Borrowed device arrays; all indices refer to the original flattened scene.
struct SceneView {
  const Triangle *           base, *actors;
  const Node*                nodes;
  const int*                 order;
  const Material *           base_materials, *actor_materials;
  const Texture *            base_textures, *actor_textures;
  const unsigned char *      base_pixels, *actor_pixels;
  const Light*               lights;
  int                        base_count, node_count;
  __device__ const Triangle& triangle(int i) const {
    return i < base_count ? base[i] : actors[i - base_count];
  }
};
/// @brief World-space sample ray with normalized direction.
struct Ray {
  V3 origin, direction;
};
/// @brief Closest triangle index, ray distance and barycentric pair.
struct Hit {
  int   triangle;
  float distance, u, v;
};
/// @brief Intersect a conservatively rounded ray slab.
__device__ bool box(const Node& n, Ray r, V3 inverse, float near, float far) {
  for (int axis = 0; axis < 3; ++axis) {
    float d = component(r.direction, axis), o = component(r.origin, axis),
          lo = component(n.lo, axis), hi = component(n.hi, axis);
    if (fabsf(d) < 1e-30f) {
      if (o < lo || o > hi)
        return false;
      continue;
    }
    float a = (lo - o) * component(inverse, axis), b = (hi - o) * component(inverse, axis);
    near = fmaxf(near, fminf(a, b));
    far  = fminf(far, fmaxf(a, b));
    if (far < near)
      return false;
  }
  return true;
}
/// @brief Traverse the balanced BVH for a nearest hit or any shadow blocker.
__device__ Hit
closest(const SceneView& s, Ray ray, float near, float far, bool both = true, bool any = false) {
  Hit result{-1, far, 0, 0};
  if (!s.node_count)
    return result;
  int stack[64], size = 1;
  stack[0] = 0;
  V3 inverse{1 / ray.direction.x, 1 / ray.direction.y, 1 / ray.direction.z};
  while (size) {
    const Node& n = s.nodes[stack[--size]];
    if (!box(n, ray, inverse, near, result.distance))
      continue;
    if (!n.count) {
      stack[size++] = n.left;
      stack[size++] = n.right;
      continue;
    }
    for (int k = n.start; k < n.start + n.count; ++k) {
      int         index = s.order[k];
      const auto& t     = s.triangle(index);
      auto        p     = cross(ray.direction, t.edge2);
      float       det   = dot(t.edge1, p);
      if ((!both && det <= 0) || fabsf(det) < t.epsilon || det == 0)
        continue;
      V3    v = ray.origin - t.origin;
      float u = dot(v, p) / det;
      if (u < 0 || u > 1)
        continue;
      auto  q = cross(v, t.edge1);
      float w = dot(ray.direction, q) / det;
      if (w < 0 || u + w > 1)
        continue;
      float distance = dot(t.edge2, q) / det;
      if (distance >= near && distance < result.distance) {
        result = {index, distance, u, w};
        if (any)
          return result;
      }
    }
  }
  return result;
}
/// @brief Read repeating bilinear RGBA8 with the CPU sampler convention.
__device__ V4 texture(const SceneView& s, bool actor, int index, float u, float v) {
  if (index < 0)
    return {1, 1, 1, 1};
  const auto& t      = (actor ? s.actor_textures : s.base_textures)[index];
  auto        pixels = actor ? s.actor_pixels : s.base_pixels;
  float       x = (u - floorf(u)) * t.width - .5f, y = (v - floorf(v)) * t.height - .5f;
  int         ix = int(floorf(x)), iy = int(floorf(y));
  float       fx = x - ix, fy = y - iy;
  float       values[4] = {};
  for (int j = 0; j < 2; ++j)
    for (int i = 0; i < 2; ++i) {
      int px       = ((ix + i) % t.width + t.width) % t.width,
          py       = ((iy + j) % t.height + t.height) % t.height;
      float weight = (i ? fx : 1 - fx) * (j ? fy : 1 - fy);
      for (int c = 0; c < 4; ++c)
        values[c] += weight * pixels[t.offset + (py * t.width + px) * 4 + c] / 255.f;
    }
  return {values[0], values[1], values[2], values[3]};
}
/// @brief Convert one sRGB channel to linear radiance.
__device__ float decode(float v) {
  return v <= .04045f ? v / 12.92f : powf((v + .055f) / 1.055f, 2.4f);
}
/// @brief Apply exposure, filmic mapping and sRGB encoding.
__device__ float display(float v, float exposure) {
  float x = fmaxf(0, v * exposure),
        f = clamp(x * (2.51f * x + .03f) / (x * (2.43f * x + .59f) + .14f));
  return f <= .0031308f ? 12.92f * f : 1.055f * powf(f, 1 / 2.4f) - .055f;
}
// Float32 origins need a representable offset. This explicitly differs from CPU's double bias.
/// @brief Compute Euclidean vector length.
/// @brief Offset a float32 ray origin by a representable world-scale amount.
__device__ float bias(V3 p) {
  return fmaxf(2e-6f, length(p) * 2e-6f);
}
/// @brief Compute a right-handed vector cross product.
/// @brief Normalize a vector, retaining zero when its length is zero.
/// @brief Construct a stable perpendicular tangent.
__device__ V3 basis(V3 d) {
  return unit(cross(d, fabsf(d.z) < .9f ? V3{0, 0, 1} : V3{0, 1, 0}));
}
/// @brief Trace the original distance-weighted hemisphere sample set.
__device__ float ambient(const SceneView& s, V3 position, V3 normal, const Parameters& p) {
  if (!p.ao_samples)
    return 1;
  auto  u = basis(normal), v = cross(normal, u);
  float sum = 0, b = bias(position);
  for (int k = 0; k < p.ao_samples; ++k) {
    auto a = p.ao[k];
    auto d = u * a.x + v * a.y + normal * a.z;
    auto h = closest(s, {position + normal * b, d}, b, p.ao_distance);
    sum += h.triangle < 0 ? 1 : clamp(h.distance / p.ao_distance);
  }
  return .22f + .78f * sum / p.ao_samples;
}
/// @brief Trace finite-disk key visibility with float32 origin bias.
__device__ float visibility(
    const SceneView&  s,
    V3                position,
    V3                normal,
    V3                d,
    float             distance,
    bool              point,
    const Parameters& p) {
  float sum = 0, b = bias(position);
  V3    u{}, v{};
  if (point) {
    u = basis(d);
    v = cross(d, u);
  }
  for (int k = 0; k < p.shadow_samples; ++k) {
    V3 direction = p.shadow_directions[k];
    if (point) {
      float r     = p.shadow_samples == 1
                        ? 0
                        : tanf(p.angular_radius) * sqrtf((k + .5f) / p.shadow_samples),
            angle = k * 2.399963229728653f;
      direction   = unit(d + u * (r * cosf(angle)) + v * (r * sinf(angle)));
    }
    auto h = closest(s, {position + normal * b, direction}, b, distance - b, true, true);
    sum += h.triangle < 0 ? 1 : p.physical ? .06f : .35f;
  }
  return sum / p.shadow_samples;
}
/// @brief Match the CPU analytic sky/ground and rough softbox reflection lobes.
/// @param d Unit reflection direction.
/// @param rough Clamped roughness.
/// @return Linear studio radiance, without scene-reflection rays.
__device__ V3 studio_radiance(V3 d, float rough) {
  float t = clamp(.5f + .5f * d.z), exponent = 3 + 100 * (1 - rough) * (1 - rough);
  V3    value = V3{.055f, .062f, .07f} * (1 - t) + V3{.55f, .64f, .76f} * t;
  return value +
         V3{1, .95f, .87f} * (2.5f * powf(fmaxf(0, dot(d, unit(V3{-.45f, -.5f, 1}))), exponent)) +
         V3{.72f, .84f, 1} * (1.1f * powf(fmaxf(0, dot(d, unit(V3{.7f, .3f, .65f}))), exponent));
}
/// @brief Approximate diffuse hemisphere irradiance and rough studio specular reflection.
/// @param color Linear material RGB.
/// @param spec Normal-incidence reflectance.
/// @param n Mapped normal.
/// @param incoming Unit camera ray direction.
/// @param rough Clamped roughness.
/// @param metal Metallic fraction.
/// @return Ambient radiance before visibility and strength.
__device__ V3 studio_surface(V3 color, V3 spec, V3 n, V3 incoming, float rough, float metal) {
  float t = .5f + .25f * n.z, nv = clamp(-dot(n, incoming));
  auto  irradiance = V3{.055f, .062f, .07f} * (1 - t) + V3{.55f, .64f, .76f} * t;
  auto  grazing  = V3{fmaxf(spec.x, 1 - rough), fmaxf(spec.y, 1 - rough), fmaxf(spec.z, 1 - rough)};
  auto  fresnel  = spec + (grazing - spec) * powf(1 - nv, 5),
       reflected = incoming - n * (2 * dot(n, incoming));
  return mul(mul(V3{1, 1, 1} - fresnel, color), irradiance) * (1 - metal) +
         mul(fresnel, studio_radiance(reflected, rough)) * (1 - .35f * rough);
}
/// @brief Evaluate mapped legacy or physical surface lighting.
__device__ V3 shade(
    const SceneView&  s,
    const Triangle&   t,
    const Material&   m,
    bool              actor,
    Ray               ray,
    V3                position,
    V3                normal,
    float             u,
    float             v,
    V4                base,
    const Parameters& p) {
  V3 color{base.x, base.y, base.z};
  if (p.physical)
    color = {decode(color.x), decode(color.y), decode(color.z)};
  if (p.style == 1)
    return color;
  auto n = normal;
  if (m.normal_map >= 0) {
    auto c       = texture(s, actor, m.normal_map, u, v);
    auto tangent = unit(t.tangent - n * dot(n, t.tangent));
    if (length(tangent) > 0)
      n = unit(tangent * (2 * c.x - 1) + cross(n, tangent) * (2 * c.y - 1) + n * (2 * c.z - 1));
  }
  float rough = m.roughness_map < 0 ? m.roughness : texture(s, actor, m.roughness_map, u, v).x;
  float metal = m.metallic_map < 0 ? m.metallic : texture(s, actor, m.metallic_map, u, v).x;
  rough       = clamp(rough, .04f, 1);
  V3 spec = V3{.04f, .04f, .04f} * (1 - metal) + color * metal, lit = color * ((1 - metal) * .35f);
  if (p.physical) {
    auto environment =
        p.studio_environment > 0
            ? studio_surface(color, spec, n, ray.direction, rough, metal) * p.studio_environment
            : color * ((1 - metal) * .24f) + spec * (.20f + .35f * rough);
    lit = environment * ambient(s, position, normal, p);
  } else if (p.ao_samples)
    lit = lit * ambient(s, position, normal, p);
  for (int i = 0; i < p.light_count; ++i) {
    const auto& l           = s.lights[i];
    V3          d           = l.direction;
    float       attenuation = 1, distance = 1e30f;
    if (l.point) {
      auto toward = l.position - position;
      distance    = length(toward);
      d           = distance > 1e-9f ? toward / distance : V3{0, 1, 0};
      attenuation = clamp(1 - distance / l.range);
    }
    float nl = fmaxf(0, dot(n, d));
    if (nl <= 0 || attenuation == 0)
      continue;
    float shadow =
        p.shadows && i == 0 ? visibility(s, position, normal, d, distance, l.point, p) : 1;
    V3 half = unit(d - ray.direction), light = l.color * (l.intensity * attenuation * shadow);
    if (p.physical) {
      const float pi = 3.141592653589793f;
      float       nv = fmaxf(.001f, dot(n, -ray.direction)), nh = fmaxf(0, dot(n, half)),
            vh    = fmaxf(0, dot(-ray.direction, half));
      float alpha = rough * rough, a2 = alpha * alpha, den = nh * nh * (a2 - 1) + 1,
            distribution = a2 / (pi * den * den), k = (rough + 1) * (rough + 1) / 8;
      float masking   = nv / (nv * (1 - k) + k) * nl / (nl * (1 - k) + k);
      auto  fresnel   = spec + (V3{1, 1, 1} - spec) * powf(1 - vh, 5);
      auto  diffuse   = mul(V3{1, 1, 1} - fresnel, color) * ((1 - metal) / pi),
           microfacet = fresnel * (distribution * masking / fmaxf(4 * nv * nl, 1e-8f));
      lit             = lit + mul(diffuse + microfacet, light) * nl;
    } else {
      float strength = powf(fmaxf(0, dot(n, half)), 128 + (4 - 128) * rough) * nl;
      lit            = lit + mul(color * ((1 - metal) * .65f * nl) + spec * strength, light);
    }
  }
  return lit;
}
/// @brief Construct a perspective or parallel sample ray.
__device__ Ray camera_ray(float x, float y, const Parameters& p) {
  float u = 2 * x / p.width - 1, v = 1 - 2 * y / p.height;
  if (p.projection == 1)
    return {
        p.origin + p.right * (u * p.aspect * p.ortho_height * .5f) +
            p.up * (v * p.ortho_height * .5f),
        p.forward};
  return {p.origin, unit(p.forward + p.right * (u * p.aspect * p.scale) + p.up * (v * p.scale))};
}
/// @brief Composite mapped transparent layers over the background.
__device__ V4 trace(const SceneView& s, Ray ray, const Parameters& p, Hit first, bool reuse) {
  V3    color{};
  float alpha = 0, remaining = 1, near = p.near_clip;
  for (int layer = 0; layer < p.layers && remaining > 1e-5f; ++layer) {
    auto h = layer == 0 && reuse ? first : closest(s, ray, near, p.far_clip, p.both);
    if (h.triangle < 0)
      break;
    const auto& t     = s.triangle(h.triangle);
    bool        actor = h.triangle >= p.base_count;
    const auto& m     = (actor ? s.actor_materials : s.base_materials)[t.material];
    float       a     = 1 - h.u - h.v;
    V3          n     = p.smooth ? unit(t.normals[0] * a + t.normals[1] * h.u + t.normals[2] * h.v)
                                 : unit(cross(t.edge1, t.edge2));
    if (dot(n, ray.direction) > 0)
      n = -n;
    auto  position = ray.origin + ray.direction * h.distance;
    float u        = t.uv[0] * a + t.uv[2] * h.u + t.uv[4] * h.v,
          v        = t.uv[1] * a + t.uv[3] * h.u + t.uv[5] * h.v;
    auto tex       = texture(s, actor, m.color_map, u, v);
    V4   base{tex.x * m.color.x, tex.y * m.color.y, tex.z * m.color.z, tex.w * m.color.w};
    auto c = shade(s, t, m, actor, ray, position, n, u, v, base, p);
    if (p.style == 2 || p.style == 3) {
      float pixel =
                p.projection == 0 ? 2 * h.distance * p.scale / p.height : p.ortho_height / p.height,
            area = length(cross(t.edge1, t.edge2));
      float edge = fminf(
          a * area / length(t.edge2 - t.edge1),
          fminf(h.u * area / length(t.edge2), h.v * area / length(t.edge1)));
      if (edge < p.wire_width * pixel) {
        c      = {p.wire_color.x, p.wire_color.y, p.wire_color.z};
        base.w = p.wire_color.w;
      } else if (p.style == 2)
        base.w = 0;
    }
    color = color + c * (remaining * base.w);
    alpha += remaining * base.w;
    remaining *= 1 - base.w;
    near = h.distance + fmaxf(2e-6f, fabsf(h.distance) * 2e-6f);
  }
  V3 backdrop{p.background.x, p.background.y, p.background.z};
  if (p.physical)
    backdrop = {decode(backdrop.x), decode(backdrop.y), decode(backdrop.z)};
  color = color + backdrop * (remaining * p.background.w);
  alpha += remaining * p.background.w;
  if (alpha > 0)
    color = color / alpha;
  if (p.physical)
    color = {
        display(color.x, p.exposure), display(color.y, p.exposure), display(color.z, p.exposure)};
  return {color.x, color.y, color.z, alpha};
}
/// @brief Render one pixel per CUDA thread and write RGBA/depth/source ID.
extern "C" __global__ void cad_render(
    const Triangle*      base,
    const Triangle*      actors,
    const Node*          nodes,
    const int*           order,
    const Material*      base_materials,
    const Material*      actor_materials,
    const Texture*       base_textures,
    const Texture*       actor_textures,
    const unsigned char* base_pixels,
    const unsigned char* actor_pixels,
    const Light*         lights,
    Pixel*               output,
    Parameters           p) {
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= unsigned(p.width) * p.height)
    return;
  SceneView s{
      base,
      actors,
      nodes,
      order,
      base_materials,
      actor_materials,
      base_textures,
      actor_textures,
      base_pixels,
      actor_pixels,
      lights,
      p.base_count,
      p.node_count};
  int   x = index % p.width, y = index / p.width;
  auto  ray = camera_ray(x + .5f, y + .5f, p);
  auto  h   = closest(s, ray, p.near_clip, p.far_clip, p.both);
  Pixel pixel{};
  pixel.depth  = h.triangle < 0 ? INFINITY : h.distance;
  pixel.object = h.triangle < 0 ? -1 : s.triangle(h.triangle).object;
  V4 sum{};
  for (int sy = 0; sy < p.samples; ++sy)
    for (int sx = 0; sx < p.samples; ++sx) {
      auto r = camera_ray(x + (sx + .5f) / p.samples, y + (sy + .5f) / p.samples, p);
      auto c = trace(s, r, p, h, p.samples == 1);
      sum.x += c.x * c.w;
      sum.y += c.y * c.w;
      sum.z += c.z * c.w;
      sum.w += c.w;
    }
  pixel.rgba[0] = (unsigned char)floorf(clamp(sum.w > 0 ? sum.x / sum.w : 0) * 255 + .5f);
  pixel.rgba[1] = (unsigned char)floorf(clamp(sum.w > 0 ? sum.y / sum.w : 0) * 255 + .5f);
  pixel.rgba[2] = (unsigned char)floorf(clamp(sum.w > 0 ? sum.z / sum.w : 0) * 255 + .5f);
  pixel.rgba[3] = (unsigned char)floorf(clamp(sum.w / (p.samples * p.samples)) * 255 + .5f);
  output[index] = pixel;
}
