/// @file workcell_visuals.cpp
/// @brief Deterministic procedural PBR surfaces, botanical fruit and industrial fixture details.
#include <algorithm>
#include <cmath>
#include <datamunge/sim/workcell_visuals.hpp>
#include <map>
#include <stdexcept>

#include "../game/font.hpp"
#include "visual_assets.hpp"
namespace datamunge::sim {
/// @brief Internal texture and geometry construction helpers.
namespace {
constexpr double pi = 3.14159265358979323846; ///< Half turn in radians.

/// @brief Reproducible spatial noise; never consumes the environment's random streams.
double noise(int x, int y, std::uint64_t seed) {
  std::uint64_t h = seed ^ (std::uint64_t(x) * 0x9e3779b97f4a7c15ULL) ^
                    (std::uint64_t(y) * 0xbf58476d1ce4e5b9ULL);
  h = (h ^ (h >> 30)) * 0xbf58476d1ce4e5b9ULL;
  h = (h ^ (h >> 27)) * 0x94d049bb133111ebULL;
  return double((h ^ (h >> 31)) >> 11) * 0x1.0p-53;
}
/// @brief Build a deterministic RGBA surface, or matching tangent-normal/roughness map.
cad::TextureData surface(const std::string& kind, std::uint64_t seed = 17, int map = 0) {
  cad::TextureData t;
  t.name   = kind + (map == 1 ? " normal" : map == 2 ? " roughness" : " albedo");
  t.width  = 256;
  t.height = 128;
  t.pixels.resize(t.width * t.height * 4);
  for (int y = 0; y < t.height; ++y)
    for (int x = 0; x < t.width; ++x) {
      double n = noise(x, y, seed), fine = noise(x * 3, y * 3, seed + 1);
      double r = 1, g = 1, b = 1;
      if (map == 1) {
        r = .5 + (n - .5) * .06;
        g = .5 + (fine - .5) * .06;
        b = .999;
      } else if (map == 2) {
        r = g = b = kind == "brushed aluminium" ? .34 + .14 * n
                    : kind == "rubber"          ? .8 + .1 * n
                                                : .55 + .12 * n;
      } else if (kind == "epoxy floor") {
        double grain = .53 + .045 * (n - .5) + .015 * std::sin(x * .071) * std::sin(y * .097);
        r            = grain * 1.06;
        g            = grain * 1.045;
        b            = grain;
        if (x < 1 || y < 1) {
          r *= .74;
          g *= .74;
          b *= .74;
        }
      } else if (kind == "rubber") {
        double weave = ((x % 6 == 0) || (y % 6 == 0)) ? .78 : 1;
        double v     = (.075 + .025 * n) * weave;
        r            = v * .91;
        g            = v * .98;
        b            = v;
      } else if (kind == "brushed aluminium") {
        double v = .66 + .065 * (noise(x / 3, y, seed) - .5) + .012 * (fine - .5);
        r        = v;
        g        = v * 1.02;
        b        = v * 1.035;
      } else if (kind == "plastic") {
        r = g = b = .94 + .06 * n;
        if (n < .005)
          r = g = b = .69;
      } else if (kind == "hazard tape") {
        bool dark = ((x + y) / 24) % 2;
        r         = dark ? .08 : .88;
        g         = dark ? .085 : .65;
        b         = dark ? .09 : .12;
      } else {
        r = g = b = .45 + .08 * n;
      }
      auto at          = 4 * (y * t.width + x);
      t.pixels[at]     = static_cast<unsigned char>(255 * std::clamp(r, 0., 1.));
      t.pixels[at + 1] = static_cast<unsigned char>(255 * std::clamp(g, 0., 1.));
      t.pixels[at + 2] = static_cast<unsigned char>(255 * std::clamp(b, 0., 1.));
      t.pixels[at + 3] = 255;
    }
  return t;
}
/// @brief Attach a matched albedo/normal/roughness set and return its albedo slot.
int maps(cad::Scene& s, const std::string& kind) {
  int base = int(s.textures.size());
  for (int k = 0; k < 3; ++k)
    s.textures.push_back(surface(kind, 17, k));
  return base;
}
/// @brief Add a sized, locally bevelled block with scalar PBR material.
cad::Object3D& block(
    cad::Scene&        s,
    const std::string& name,
    cad::Vec3d         at,
    cad::Vec3d         size,
    cad::RgbaColorF    color,
    double             radius = .002,
    double             metal  = 0,
    double             rough  = .5) {
  cad::Object3D o;
  o.name       = name;
  o.mesh_index = int(s.meshes.size());
  o.position   = {float(at.x), float(at.y), float(at.z)};
  o.color      = color;
  o.metallic   = float(metal);
  o.roughness  = float(rough);
  s.meshes.push_back(detail::visuals::rounded_box(size, radius));
  int id = s.AddObject(o);
  return *s.FindObject(id);
}
/// @brief Add a capped tube between world/local points, with configurable cross-section facets.
void tube(
    cad::Scene&        s,
    const std::string& name,
    cad::Vec3d         a,
    cad::Vec3d         b,
    double             ra,
    double             rb,
    cad::RgbaColorF    color,
    int                sides = 12,
    double             metal = 0) {
  auto axis = (b - a).Normalized(),
       u    = axis.Cross(std::abs(axis.z) < .9 ? cad::Vec3d{0, 0, 1} : cad::Vec3d{0, 1, 0})
               .Normalized(),
       v = axis.Cross(u);
  cad::MeshData m;
  m.name = name;
  for (int row = 0; row < 2; ++row)
    for (int j = 0; j <= sides; ++j) {
      double angle  = 2 * pi * j / sides;
      auto   normal = u * std::cos(angle) + v * std::sin(angle),
           p        = (row ? b : a) + normal * (row ? rb : ra);
      m.positions.insert(m.positions.end(), {float(p.x), float(p.y), float(p.z)});
      m.normals.insert(m.normals.end(), {float(normal.x), float(normal.y), float(normal.z)});
      m.texcoords.insert(m.texcoords.end(), {float(j) / sides, float(row)});
    }
  for (int j = 0; j < sides; ++j) {
    unsigned a0 = j, b0 = j + sides + 1;
    m.indices.insert(m.indices.end(), {a0, b0, a0 + 1, a0 + 1, b0, b0 + 1});
  }
  for (int row = 0; row < 2; ++row) {
    auto     p = row ? b : a, n = axis * (row ? 1. : -1.);
    unsigned centre = m.positions.size() / 3;
    m.positions.insert(m.positions.end(), {float(p.x), float(p.y), float(p.z)});
    m.normals.insert(m.normals.end(), {float(n.x), float(n.y), float(n.z)});
    m.texcoords.insert(m.texcoords.end(), {.5f, .5f});
    for (int j = 0; j < sides; ++j) {
      unsigned first = row * (sides + 1) + j;
      if (row)
        m.indices.insert(m.indices.end(), {centre, first, first + 1});
      else
        m.indices.insert(m.indices.end(), {centre, first + 1, first});
    }
  }
  cad::Object3D o;
  o.name       = name;
  o.mesh_index = int(s.meshes.size());
  o.color      = color;
  o.roughness  = .44f;
  o.metallic   = float(metal);
  s.meshes.push_back(std::move(m));
  s.AddObject(o);
}
/// @brief Render a permanent industrial label into a small RGBA texture, without external fonts.
cad::TextureData label(const std::string& text) {
  cad::TextureData t;
  t.name   = "moulded bin identification";
  t.width  = 128;
  t.height = 48;
  t.pixels.assign(128 * 48 * 4, 235);
  for (int y = 0; y < 48; ++y)
    for (int x = 0; x < 128; ++x)
      t.pixels[4 * (y * 128 + x) + 3] = 255;
  int start = (128 - int(text.size()) * 12) / 2;
  for (std::size_t k = 0; k < text.size(); ++k) {
    const auto& glyph = game::detail::glyph(text[k]);
    for (int y = 0; y < 7; ++y)
      for (int x = 0; x < 5; ++x)
        if (glyph[y] & (1u << (4 - x)))
          for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
              int  xx = start + k * 12 + x * 2 + dx, yy = 17 + y * 2 + dy;
              auto p          = 4 * (yy * 128 + xx);
              t.pixels[p]     = 25;
              t.pixels[p + 1] = 30;
              t.pixels[p + 2] = 32;
            }
  }
  return t;
}
/// @brief Build a non-repeating label quad facing an outward bin side.
void bin_label(cad::Scene& s, double y, const std::string& text) {
  cad::MeshData m;
  m.name      = "printed bin label";
  float yy    = float(y);
  m.positions = {.025f, yy, .515f, .175f, yy, .515f, .175f, yy, .565f, .025f, yy, .565f};
  float n     = y < 0 ? -1.f : 1.f;
  m.normals   = {0, n, 0, 0, n, 0, 0, n, 0, 0, n, 0};
  m.texcoords = y < 0 ? std::vector<float>{0, 1, 1, 1, 1, 0, 0, 0}
                      : std::vector<float>{1, 1, 0, 1, 0, 0, 1, 0};
  m.indices   = {0, 1, 2, 0, 2, 3};
  cad::Object3D o;
  o.name          = "bin identification label";
  o.mesh_index    = s.meshes.size();
  o.texture_index = s.textures.size();
  o.color         = {1, 1, 1, 1};
  o.roughness     = .75f;
  s.meshes.push_back(std::move(m));
  s.textures.push_back(label(text));
  s.AddObject(o);
}
/// @brief Smooth periodic lattice variation, deterministic and independent of domain RNGs.
/// @param u Repeating longitudinal texture coordinate.
/// @param v Vertical texture coordinate.
/// @param seed Stable material realization.
/// @param frequency Number of longitudinal cells per repeat.
/// @return Smooth scalar in [0,1].
double grain(double u, double v, std::uint64_t seed, int frequency = 8) {
  double x = u * frequency, y = v * frequency;
  int    ix = int(std::floor(x)), iy = int(std::floor(y));
  double a = x - ix, b = y - iy;
  a = a * a * (3 - 2 * a);
  b = b * b * (3 - 2 * b);
  /// @brief Read one periodic lattice value without consuming simulation RNGs.
  /// @param i Lattice column.
  /// @param j Lattice row.
  /// @return Stable scalar sample in [0,1].
  auto value = [&](int i, int j) {
    return noise(
        (i % frequency + frequency) % frequency, (j % frequency + frequency) % frequency, seed);
  };
  return (value(ix, iy) * (1 - a) + value(ix + 1, iy) * a) * (1 - b) +
         (value(ix, iy + 1) * (1 - a) + value(ix + 1, iy + 1) * a) * b;
}
/// @brief Refine a material with correlated albedo, restrained bump and broad roughness variation.
/// @param kind Concrete, powder, steel, rubber or plastic finish.
/// @param map Zero albedo, one normal, two roughness.
/// @return Owned repeating 256x128 RGBA8 surface.
cad::TextureData finish_map(const std::string& kind, int map) {
  auto t = surface("plastic", 47, map);
  t.name = "presentation " + kind;
  for (int y = 0; y < t.height; ++y)
    for (int x = 0; x < t.width; ++x) {
      double u = double(x) / t.width, v = double(y) / t.height, n = noise(x, y, 47),
             broad = grain(u, v, 37, 12), fine = grain(u, v, 91, 32);
      double r = .5, g = .5, b = 1;
      if (map == 0) {
        double tone = kind == "concrete" ? .29 + .035 * (broad - .5) + .018 * (fine - .5)
                      : kind == "rubber" ? .065 + .014 * (broad - .5)
                      : kind == "steel"  ? .68 + .035 * (noise(x / 5, y, 9) - .5)
                                         : .94 + .035 * (broad - .5);
        // Subtle roller/trowel variation and occasional rubber scuffs, not grout lines.
        if (kind == "rubber")
          tone +=
              .018 * std::pow(std::max(0., std::sin(2 * pi * (v * 7 + .1 * std::sin(u * 11)))), 20);
        r = tone;
        g = tone * (kind == "concrete" ? 1.025 : 1.01);
        b = tone * (kind == "concrete" ? 1.045 : 1.025);
      } else if (map == 1) {
        double amplitude = kind == "concrete" ? .014 : kind == "steel" ? .012 : .006;
        r                = .5 + amplitude * (grain(u + .01, v, 37, 12) - broad);
        g                = .5 + amplitude * (grain(u, v + .01, 37, 12) - broad);
        b                = .999;
      } else {
        r = g = b = kind == "steel"      ? .24 + .10 * broad
                    : kind == "rubber"   ? .78 + .08 * broad
                    : kind == "concrete" ? .62 + .14 * broad
                                         : .36 + .10 * broad;
      }
      int at           = 4 * (y * t.width + x);
      t.pixels[at]     = static_cast<unsigned char>(255 * std::clamp(r, 0., 1.));
      t.pixels[at + 1] = static_cast<unsigned char>(255 * std::clamp(g, 0., 1.));
      t.pixels[at + 2] = static_cast<unsigned char>(255 * std::clamp(b, 0., 1.));
      t.pixels[at + 3] = 255;
    }
  return t;
}
/// @brief Append one matched exterior material set to an owned scene.
/// @param scene Owned presentation scene.
/// @param kind Material finish identifier.
/// @return Albedo index followed by normal and roughness indices.
int finish_maps(cad::Scene& scene, const std::string& kind) {
  int first = int(scene.textures.size());
  for (int k = 0; k < 3; ++k)
    scene.textures.push_back(finish_map(kind, k));
  return first;
}
/// @brief Generate seeded clustered freckles, pigment variation and wax on an existing fruit.
/// @param scene Owned presentation scene; sensor assets are never passed here.
/// @param object Main apple or banana object with valid original texture maps.
void botanical_finish(cad::Scene& scene, cad::Object3D& object) {
  if (object.texture_index < 0 || std::size_t(object.texture_index) >= scene.textures.size())
    return;
  const auto original = scene.textures[object.texture_index];
  if (original.pixels.empty())
    return;
  std::uint64_t seed = 1469598103934665603ULL;
  double        color[3]{};
  for (std::size_t i = 0; i < original.pixels.size(); ++i) {
    seed = (seed ^ original.pixels[i]) * 1099511628211ULL;
    if (i % 4 < 3)
      color[i % 4] += original.pixels[i] / 255.;
  }
  for (auto& c : color)
    c /= original.pixels.size() / 4;
  bool             apple = object.name == "apple";
  cad::TextureData pigment, normal, rough;
  for (auto* t : {&pigment, &normal, &rough}) {
    t->width  = 512;
    t->height = 256;
    t->pixels.resize(512 * 256 * 4);
  }
  pigment.name = "presentation botanical pigment";
  normal.name  = "presentation peel microstructure";
  rough.name   = "presentation variable wax";
  for (int y = 0; y < 256; ++y)
    for (int x = 0; x < 512; ++x) {
      double u = x / 512., v = y / 256., soft = grain(u, v, seed, 10),
             patch = grain(u, v, seed + 2, 24), micro = noise(x, y, seed + 4);
      double striation = apple ? .025 * std::sin(2 * pi * (u * 17 + .11 * std::sin(v * 9)))
                               : .035 * std::cos(10 * pi * u);
      double shade     = .98 + .13 * (soft - .5) + striation;
      bool   pore      = apple ? micro < .004 : micro < (.001 + .016 * std::max(0., patch - .45));
      double blemish =
          apple ? 1 - .12 * std::max(0., patch - .75) : 1 - .33 * std::max(0., soft - .58);
      if (!apple && (v < .055 || v > .955))
        blemish *= .55;
      int at = 4 * (y * 512 + x);
      for (int c = 0; c < 3; ++c) {
        double value = color[c] * shade * blemish;
        if (pore)
          value = apple ? .78 * value + .22 * std::array<double, 3>{.8, .71, .46}[c] : value * .35;
        pigment.pixels[at + c] = static_cast<unsigned char>(255 * std::clamp(value, 0., 1.));
      }
      normal.pixels[at] =
          static_cast<unsigned char>(128 + (micro - .5) * 2 + (!apple ? std::sin(10 * pi * u) : 0));
      normal.pixels[at + 1] = static_cast<unsigned char>(128 + (noise(x, y, seed + 5) - .5) * 2);
      normal.pixels[at + 2] = 255;
      double r              = apple ? .24 + .045 * soft : .4 + .10 * soft;
      if (pore)
        r += .16;
      for (int c = 0; c < 3; ++c)
        rough.pixels[at + c] = static_cast<unsigned char>(255 * std::clamp(r, 0., 1.));
      pigment.pixels[at + 3] = normal.pixels[at + 3] = rough.pixels[at + 3] = 255;
    }
  int first = int(scene.textures.size());
  scene.textures.push_back(std::move(pigment));
  scene.textures.push_back(std::move(normal));
  scene.textures.push_back(std::move(rough));
  object.texture_index       = first;
  object.normal_map_index    = first + 1;
  object.roughness_map_index = first + 2;
}
/// @brief Sweep a cosmetic flexible harness through supplied URDF-derived anchors.
/// @param scene Owned presentation scene.
/// @param anchors Valid world-space cable clips.
void cable_harness(cad::Scene& scene, const std::vector<cad::Vec3d>& anchors) {
  for (std::size_t i = 1; i < anchors.size(); ++i) {
    auto a = anchors[i - 1], b = anchors[i];
    if ((b - a).Length() < .001)
      continue;
    cad::Vec3d previous = a;
    for (int k = 1; k <= 10; ++k) {
      double t = k / 10.;
      auto   p =
          a * (1 - t) + b * t + cad::Vec3d{0, .018 * std::sin(pi * t), -.012 * std::sin(pi * t)};
      tube(
          scene,
          "presentation flexible servo harness",
          previous,
          p,
          .0032,
          .0032,
          {.027f, .032f, .036f, 1},
          8);
      previous = p;
    }
    // A secondary small service lead adds scale without changing any collision proxy.
    auto offset = cad::Vec3d{0, .005, 0};
    tube(
        scene,
        "presentation service lead",
        a + offset,
        b + offset,
        .0011,
        .0011,
        {.18f, .2f, .21f, 1},
        6);
  }
}
/// @brief Add distant factory context and visibly grounded cosmetic fixture hardware.
/// @param scene Owned static presentation scene in CAD Z-up metres.
void factory_context(cad::Scene& scene) {
  block(
      scene,
      "presentation back wall",
      {0, 2.3, 1.45},
      {24, .08, 2.9},
      {.47f, .5f, .52f, 1},
      .005,
      0,
      .9);
  for (double x : {-6., -4.5, -3., -1.5, 0., 1.5, 3., 4.5, 6.}) {
    block(
        scene,
        "presentation wall pilaster",
        {x, 2.22, 1.45},
        {.045, .06, 2.9},
        {.2f, .24f, .26f, 1},
        .003,
        .35,
        .6);
    block(
        scene,
        "presentation clerestory daylight",
        {x + .48, 2.245, 1.4},
        {.86, .015, .55},
        {.73f, .82f, .87f, 1},
        .002,
        0,
        .18);
  }
  for (double x : {-6., -4.5, -3., -1.5, 0., 1.5, 3., 4.5, 6.}) {
    for (double dx : {.025, .935})
      block(
          scene,
          "presentation window jamb",
          {x + dx, 2.218, 1.4},
          {.028, .045, .61},
          {.16f, .19f, .21f, 1},
          .002,
          .45,
          .34);
    for (double z : {1.095, 1.705})
      block(
          scene,
          "presentation window sill",
          {x + .48, 2.218, z},
          {.94, .045, .025},
          {.16f, .19f, .21f, 1},
          .002,
          .45,
          .34);
  }
  block(
      scene,
      "presentation wall skirting",
      {0, 2.225, .09},
      {24, .04, .18},
      {.16f, .19f, .2f, 1},
      .002,
      .1,
      .72);
  for (double z : {.9, 1.8})
    block(
        scene,
        "presentation wall joint",
        {0, 2.25, z},
        {24, .012, .006},
        {.35f, .38f, .4f, 1},
        .0005,
        0,
        .9);
  block(
      scene,
      "presentation electrical cabinet",
      {1.65, 1.5, .52},
      {.48, .34, 1.04},
      {.56f, .59f, .58f, 1},
      .015,
      .08,
      .48);
  block(
      scene,
      "presentation cabinet door",
      {1.65, 1.322, .57},
      {.43, .016, .88},
      {.62f, .65f, .64f, 1},
      .008,
      .1,
      .44);
  block(
      scene,
      "presentation cabinet handle",
      {1.81, 1.302, .58},
      {.012, .018, .11},
      {.055f, .061f, .065f, 1},
      .004,
      .1,
      .48);
  for (int k = 0; k < 6; ++k)
    block(
        scene,
        "presentation cabinet vent",
        {1.49 + .025 * k, 1.31, .25},
        {.011, .004, .12},
        {.1f, .12f, .13f, 1},
        .001,
        0,
        .7);
  for (double y : {-.42, .42}) {
    block(
        scene,
        "presentation crate stand cross tie",
        {.1, y, .06},
        {.29, .025, .025},
        {.22f, .25f, .27f, 1},
        .003,
        .6,
        .35);
    for (double x : {-.03, .23})
      for (double d : {-.075, .075}) {
        block(
            scene,
            "presentation crate isolation pad",
            {x, y + d, .008},
            {.065, .07, .016},
            {.033f, .037f, .04f, 1},
            .004,
            0,
            .8);
        tube(
            scene,
            "presentation crate adjustable stud",
            {x, y + d, .016},
            {x, y + d, .04},
            .009,
            .009,
            {.42f, .44f, .46f, 1},
            10,
            .85);
      }
  }
  for (double x : {-1.025, .425})
    for (double y : {-.215, .215}) {
      block(
          scene,
          "presentation roller bearing housing",
          {x, y, .684},
          {.045, .026, .05},
          {.22f, .25f, .27f, 1},
          .003,
          .35,
          .45);
      double sign = y < 0 ? -1 : 1;
      tube(
          scene,
          "presentation bearing shaft",
          {x, y, .684},
          {x, y + sign * .018, .684},
          .012,
          .012,
          {.72f, .75f, .78f, 1},
          16,
          .9);
      scene.objects.back().roughness = .26f;
    }
  // Service loops and a compact junction box stay outside the active picking area.
  block(
      scene,
      "presentation pedestal junction box",
      {.77, .04, .38},
      {.055, .14, .1},
      {.25f, .28f, .3f, 1},
      .008,
      .15,
      .5);
  std::vector<cad::Vec3d> route{
      {.8, .02, .4},
      {.86, .02, .18},
      {.85, .05, .025},
      {.5, .22, .018},
      {.1, .3, .018},
      {-.45, .28, .08},
      {-.7, .23, .44}};
  cable_harness(scene, route);
}
} // namespace
std::vector<cad::Light> workcell_lights(WorkcellVisualStyle style) {
  if (style != WorkcellVisualStyle::Simple && style != WorkcellVisualStyle::Realistic)
    throw std::invalid_argument("unknown workcell visual style");
  cad::Light key;
  key.direction = {-1, -2, 3};
  key.intensity = 1;
  cad::Light fill;
  fill.direction = {1, 1, 2};
  fill.intensity = .35f;
  if (style == WorkcellVisualStyle::Simple)
    return {key, fill};
  key.name       = "large warm overhead softbox";
  key.direction  = {-.65f, -.85f, 1.8f};
  key.color      = {1, .94f, .86f, 1};
  key.intensity  = 3.2f;
  fill.name      = "cool factory skylight";
  fill.direction = {1, .6f, 1.2f};
  fill.color     = {.78f, .88f, 1, 1};
  fill.intensity = 1.25f;
  cad::Light rim;
  rim.name      = "rear strip light";
  rim.direction = {.3f, 1.2f, 1.5f};
  rim.color     = {1, .98f, .92f, 1};
  rim.intensity = .9f;
  return {key, fill, rim};
}
cad::RenderOptions workcell_render_options(WorkcellVisualStyle style, int w, int h) {
  if (w <= 0 || h <= 0 || w > 8192 || h > 8192)
    throw std::invalid_argument("invalid workcell render size");
  workcell_lights(style);
  cad::RenderOptions o;
  o.width      = w;
  o.height     = h;
  o.background = {.91f, .94f, .96f, 1};
  if (style == WorkcellVisualStyle::Realistic) {
    o.physically_based           = true;
    o.shadow_samples             = w <= 256 ? 2 : 6;
    o.light_angular_radius       = .09;
    o.ambient_occlusion_samples  = w <= 256 ? 0 : 2;
    o.ambient_occlusion_distance = .11;
    o.exposure                   = 1.05;
    o.background                 = {.64f, .67f, .69f, 1};
  }
  return o;
}
cad::Scene workcell_presentation(
    cad::Scene                     scene,
    bool                           surroundings,
    const std::vector<cad::Vec3d>& cable_anchors) {
  if (!cable_anchors.empty() && (cable_anchors.size() < 2 || cable_anchors.size() > 32))
    throw std::invalid_argument("presentation cables need 2..32 anchor points");
  for (auto p : cable_anchors)
    if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
        std::max({std::abs(p.x), std::abs(p.y), std::abs(p.z)}) > 1e4)
      throw std::invalid_argument("invalid presentation cable anchor");
  for (const auto& t : scene.textures)
    if (t.width <= 0 || t.height <= 0 || std::uint64_t(t.width) * t.height * 4 != t.pixels.size())
      throw std::invalid_argument("invalid presentation RGBA texture");
  for (const auto& o : scene.objects) {
    if (o.mesh_index < -1 ||
        (o.mesh_index >= 0 && std::size_t(o.mesh_index) >= scene.meshes.size()))
      throw std::invalid_argument("invalid presentation mesh index");
    for (int i : {o.texture_index, o.normal_map_index, o.roughness_map_index, o.metallic_map_index})
      if (i < -1 || (i >= 0 && std::size_t(i) >= scene.textures.size()))
        throw std::invalid_argument("invalid presentation texture index");
  }
  std::map<std::string, int> material_slots;
  /// @brief Lazily allocate only finishes actually used by this component.
  /// @param kind Material finish identifier.
  /// @return First texture slot of the matched albedo/normal/roughness set.
  auto finish = [&](const std::string& kind) {
    auto it = material_slots.find(kind);
    if (it != material_slots.end())
      return it->second;
    int i                = finish_maps(scene, kind);
    material_slots[kind] = i;
    return i;
  };
  for (auto& o : scene.objects) {
    if (o.name == "apple" || o.name == "banana") {
      botanical_finish(scene, o);
      continue;
    }
    std::string kind;
    if (o.name == "floor") {
      kind       = "concrete";
      o.color    = {1, 1, 1, 1};
      o.metallic = 0;
    } else if (o.name == "conveyor rubber bed") {
      kind       = "rubber";
      o.color    = {1, 1, 1, 1};
      o.metallic = 0;
    } else if (
        o.name.find("bin") != std::string::npos && o.name.find("stand") == std::string::npos &&
        o.name.find("label") == std::string::npos && o.name.find("symbol") == std::string::npos) {
      kind       = "plastic";
      o.metallic = 0;
    } else if (o.name.find("rail") != std::string::npos) {
      kind       = "steel";
      o.color    = {1, 1, 1, 1};
      o.metallic = .82f;
    } else if (o.name == "arm pedestal" || o.name == "conveyor leg" || o.name == "bin stand") {
      kind       = "powder";
      o.color    = {.32f, .37f, .4f, 1};
      o.metallic = .12f;
    }
    if (!kind.empty()) {
      int first             = finish(kind);
      o.texture_index       = first;
      o.normal_map_index    = first + 1;
      o.roughness_map_index = first + 2;
    }
    // Keep the real imported link meshes and seeded brand colors; improve the
    // distinction between anodized parts, painted housings, hardware and fingers.
    if (o.name.find("/") != std::string::npos) {
      if (o.name.find("silver") != std::string::npos || o.name.find("metal") != std::string::npos ||
          o.name.find("anodized") != std::string::npos) {
        o.metallic  = .78f;
        o.roughness = .24f;
      } else if (o.name.find("finger_black") != std::string::npos) {
        o.metallic  = 0;
        o.roughness = .78f;
      } else if (o.name.find("hardware") != std::string::npos) {
        o.metallic  = .52f;
        o.roughness = .28f;
      } else if (o.name.find("matte") != std::string::npos) {
        o.metallic  = .02f;
        o.roughness = .54f;
      } else {
        o.metallic  = .02f;
        o.roughness = .35f;
      }
    }
  }
  if (surroundings)
    factory_context(scene);
  if (!cable_anchors.empty())
    cable_harness(scene, cable_anchors);
  scene.lights              = workcell_lights(WorkcellVisualStyle::Realistic);
  scene.lights[0].intensity = 2.4f;
  scene.lights[0].direction = {-.8f, -1, 1.6f};
  scene.lights[1].intensity = .65f;
  scene.lights[2].intensity = 1.2f;
  return scene;
}
cad::RenderOptions workcell_presentation_options(int width, int height) {
  auto options           = workcell_render_options(WorkcellVisualStyle::Realistic, width, height);
  options.samples        = 2;
  options.shadow_samples = 6;
  options.ambient_occlusion_samples  = 3;
  options.ambient_occlusion_distance = .18;
  options.light_angular_radius       = .12;
  options.exposure                   = 1;
  options.studio_environment         = .55;
  return options;
}

namespace detail::visuals {
cad::MeshData rounded_box(cad::Vec3d size, double radius) {
  double h[3]{size.x / 2, size.y / 2, size.z / 2};
  radius = std::min({radius, h[0] * .48, h[1] * .48, h[2] * .48});
  cad::MeshData m;
  m.name                  = "rounded manufactured surface";
  constexpr int divisions = 8;
  for (int face = 0; face < 6; ++face) {
    int      normal = face / 2, u = (normal + 1) % 3, v = (normal + 2) % 3;
    double   sign = face % 2 ? 1 : -1;
    unsigned base = m.positions.size() / 3;
    for (int j = 0; j <= divisions; ++j)
      for (int i = 0; i <= divisions; ++i) {
        double p[3]{};
        p[normal] = sign * h[normal];
        /// @brief Concentrate face-grid samples near the manufactured edge fillets.
        auto coordinate = [&](int k, double extent) {
          if (k == 0)
            return -extent;
          if (k == divisions)
            return extent;
          if (k < divisions / 2)
            return -extent + radius * (k / 3.);
          if (k == divisions / 2)
            return 0.;
          return extent - radius * ((divisions - k) / 3.);
        };
        p[u] = coordinate(i, h[u]);
        p[v] = coordinate(j, h[v]);
        double q[3], d[3];
        for (int k = 0; k < 3; ++k) {
          q[k] = std::clamp(p[k], -h[k] + radius, h[k] - radius);
          d[k] = p[k] - q[k];
        }
        double norm = std::sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]);
        for (int k = 0; k < 3; ++k) {
          m.positions.push_back(float(q[k] + radius * d[k] / norm));
          m.normals.push_back(float(d[k] / norm));
        }
        m.texcoords.insert(m.texcoords.end(), {float(p[u] / .12), float(p[v] / .12)});
      }
    for (int j = 0; j < divisions; ++j)
      for (int i = 0; i < divisions; ++i) {
        unsigned a = base + j * (divisions + 1) + i, b = a + divisions + 1;
        if (sign > 0)
          m.indices.insert(m.indices.end(), {a, a + 1, b, a + 1, b + 1, b});
        else
          m.indices.insert(m.indices.end(), {a, b, a + 1, a + 1, b, b + 1});
      }
  }
  return m;
}
void decorate(cad::Scene& s) {
  const int aluminium = maps(s, "brushed aluminium"), rubber = maps(s, "rubber"),
            plastic = maps(s, "plastic"), floor = maps(s, "epoxy floor");
  for (auto& o : s.objects) {
    if (o.kind == cad::PrimitiveKind::Cube) {
      cad::Vec3d size{o.scale.x, o.scale.y, o.scale.z};
      if (o.name == "floor")
        size = {12, 12, .12};
      s.meshes[o.mesh_index] = rounded_box(size, o.name == "floor" ? .001 : .0025);
      o.scale                = {1, 1, 1};
    }
    int texture = -1;
    if (o.name == "floor") {
      texture     = floor;
      o.color     = {1, 1, 1, 1};
      o.roughness = .78f;
      for (auto& uv : s.meshes[o.mesh_index].texcoords)
        uv *= .12f;
    } else if (o.name == "conveyor rubber bed") {
      texture     = rubber;
      o.color     = {1, 1, 1, 1};
      o.roughness = .88f;
    } else if (
        o.name.find("bin") != std::string::npos && o.name.find("stand") == std::string::npos &&
        o.name.find("symbol") == std::string::npos) {
      texture     = plastic;
      o.roughness = .42f;
      o.color     = o.name == "apple bin floor" || o.position.y < 0
                        ? cad::RgbaColorF{.57f, .065f, .045f, 1}
                        : cad::RgbaColorF{.88f, .61f, .055f, 1};
    } else if (o.name == "pick station mark") {
      o.color     = {.54f, .57f, .51f, 1};
      o.roughness = .85f;
    } else if (o.name.find("symbol") != std::string::npos) {
      o.visible = false;
    } else {
      texture     = aluminium;
      o.color     = {.83f, .86f, .88f, 1};
      o.metallic  = .8f;
      o.roughness = .38f;
    }
    if (texture >= 0) {
      o.texture_index       = texture;
      o.normal_map_index    = texture + 1;
      o.roughness_map_index = texture + 2;
    }
  }
  // T-slot rails, assembly screws and vibration-isolating feet.
  for (double y : {-.19, .19}) {
    block(
        s,
        "extrusion slot",
        {-.3, y + (y < 0 ? -.018 : .018), .681},
        {1.52, .0012, .006},
        {.045f, .052f, .055f, 1},
        .0004,
        0,
        .7);
    for (double x : {-.94, -.62, -.3, .02, .34})
      tube(
          s,
          "M5 rail fastener",
          {x, y + (y < 0 ? -.019 : .019), .704},
          {x, y + (y < 0 ? -.023 : .023), .704},
          .005,
          .005,
          {.38f, .4f, .42f, 1},
          6,
          .88);
    for (double x : {-.94, .34}) {
      tube(
          s,
          "adjustable rubber foot",
          {x, y, .001},
          {x, y, .019},
          .047,
          .047,
          {.042f, .046f, .049f, 1},
          24);
      tube(
          s,
          "foot adjustment collar",
          {x, y, .019},
          {x, y, .045},
          .018,
          .018,
          {.45f, .47f, .48f, 1},
          16,
          .85);
    }
  }
  for (double x : {.52, .72})
    for (double y : {-.1, .1})
      tube(
          s,
          "base mounting screw",
          {x, y, .721},
          {x, y, .728},
          .007,
          .007,
          {.3f, .32f, .34f, 1},
          6,
          .9);
  // Drive assembly is at the inlet, away from the manipulation workspace.
  tube(
      s,
      "conveyor gearmotor",
      {-.96, -.21, .62},
      {-.96, -.36, .62},
      .048,
      .048,
      {.12f, .17f, .21f, 1},
      32,
      .45);
  for (int k = 0; k < 6; ++k)
    tube(
        s,
        "motor cooling fin",
        {-.96, -.235 - .018 * k, .62},
        {-.96, -.240 - .018 * k, .62},
        .052,
        .052,
        {.16f, .20f, .23f, 1},
        32,
        .5);
  tube(
      s,
      "motor end cap",
      {-.96, -.36, .62},
      {-.96, -.369, .62},
      .045,
      .04,
      {.38f, .42f, .45f, 1},
      32,
      .8);
  block(
      s,
      "drive controller enclosure",
      {-.78, -.225, .48},
      {.17, .07, .12},
      {.61f, .64f, .63f, 1},
      .006,
      .2,
      .55);
  for (int k = 0; k < 8; ++k)
    block(
        s,
        "controller ventilation",
        {-.83 + .014 * k, -.261, .48},
        {.005, .001, .055},
        {.06f, .075f, .08f, 1},
        .0003);
  // Rounded rolled rims, reinforcing ribs and real printed identification labels.
  for (double y : {-.42, .42}) {
    auto color =
        y < 0 ? cad::RgbaColorF{.62f, .075f, .055f, 1} : cad::RgbaColorF{.90f, .64f, .07f, 1};
    for (double side : {-.14, .14})
      block(s, "moulded bin rim", {.1, y + side, .655}, {.35, .029, .021}, color, .008, 0, .38);
    for (double x : {-.06, .26})
      block(s, "moulded bin rim", {x, y, .655}, {.029, .30, .021}, color, .008, 0, .38);
    for (double side : {-.151, .151})
      for (int k = 0; k < 6; ++k)
        block(
            s,
            "bin reinforcing rib",
            {-.025 + .05 * k, y + side, .535},
            {.009, .009, .17},
            color,
            .003,
            0,
            .48);
    for (double x : {-.071, .271})
      for (double yy : {-.08, 0., .08})
        block(s, "bin end rib", {x, y + yy, .535}, {.008, .009, .17}, color, .0025, 0, .48);
    block(
        s,
        "recessed carry handle",
        {.1, y + (y < 0 ? -.152 : .152), .616},
        {.115, .002, .023},
        {.08f, .075f, .06f, 1},
        .007,
        0,
        .75);
    bin_label(s, y + (y < 0 ? -.158 : .158), y < 0 ? "APPLES" : "BANANAS");
  }
  const int tape = int(s.textures.size());
  s.textures.push_back(surface("hazard tape"));
  for (double y : {-.78, .78}) {
    auto& o =
        block(s, "floor safety tape", {-.15, y, .0015}, {2.25, .026, .001}, {1, 1, 1, 1}, .0002);
    o.texture_index = tape;
  }
  for (double x : {-1.275, .975}) {
    auto& o = block(s, "floor safety tape", {x, 0, .0015}, {.026, 1.56, .001}, {1, 1, 1, 1}, .0002);
    o.texture_index = tape;
  }
}
cad::Scene fruit(FruitKind kind, const FruitAppearance& a) {
  const double girth[]{1, .80, 1.10, 1.02, .9, .94}, lobes[]{.055, .025, .02, .12, .08, .04},
      taper[]{0, .16, -.08, .04, .23, -.12}, lengths[]{1, 1.15, .86, 1.02, 1.08, .84},
      curves[]{.028, .018, .045, .05, .035, .055}, thickness[]{1, .82, 1.16, .9, 1.04, 1.12};
  const auto    k = a.prototype;
  cad::Scene    s;
  cad::MeshData m;
  m.name             = kind == FruitKind::Apple ? "waxy lobed apple" : "longitudinal banana peel";
  constexpr int rows = 48, columns = 64;
  for (int i = 0; i <= rows; ++i)
    for (int j = 0; j <= columns; ++j) {
      double     t = double(i) / rows, angle = 2 * pi * j / columns;
      cad::Vec3d p;
      if (kind == FruitKind::Apple) {
        double lat = pi * t,
               r   = .039 * girth[k] * std::sin(lat) *
                   (1 + lobes[k] * std::cos((k == 3 ? 6 : 5) * angle) * std::sin(lat)) *
                   (1 + taper[k] * std::cos(lat));
        p = {
            r * std::cos(angle),
            r * std::sin(angle),
            .039 * std::cos(lat) - .005 * std::exp(-lat * lat / .08) +
                .002 * std::exp(-(pi - lat) * (pi - lat) / .12)};
      } else {
        double u = 2 * t - 1, r = .023 * thickness[k] *
                                  (.17 + .83 * std::pow(std::max(0., std::sin(pi * t)), .43)) *
                                  (1 + .045 * std::cos(5 * angle));
        p = {.076 * lengths[k] * u, -curves[k] * u * u + r * std::cos(angle), r * std::sin(angle)};
      }
      m.positions.insert(m.positions.end(), {float(p.x), float(p.y), float(p.z)});
      m.texcoords.insert(m.texcoords.end(), {float(j) / columns, float(t)});
    }
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < columns; ++j) {
      unsigned p = i * (columns + 1) + j, q = p + columns + 1;
      m.indices.insert(m.indices.end(), {p, q, p + 1, p + 1, q, q + 1});
    }
  m.RecalculateNormals();
  s.meshes.push_back(std::move(m));
  cad::TextureData albedo, normal, rough;
  albedo.name = "botanical peel pigment";
  normal.name = "peel pores and fibres";
  rough.name  = "variable wax and peel roughness";
  for (auto* t : {&albedo, &normal, &rough}) {
    t->width  = 256;
    t->height = 128;
    t->pixels.resize(256 * 128 * 4);
  }
  // Class-independent natural hues retain green/gold/russet overlap across categories.
  std::array<double, 3> color;
  if (a.skin[1] > a.skin[0] * 1.2)
    color = {.36, .56, .12};
  else if (a.skin[0] > .7 && a.skin[1] > .45)
    color = {.88, .72, .18};
  else if (a.skin[0] > .65 && a.skin[1] < .3)
    color = {.77, .61, .20};
  else
    color = {.63, .46, .18};
  for (int y = 0; y < 128; ++y)
    for (int x = 0; x < 256; ++x) {
      double n = noise(x, y, a.texture_seed), f = noise(x * 5, y * 7, a.texture_seed + 4),
             u = x / 256., v = y / 128.;
      double stripe = kind == FruitKind::Apple
                          ? .055 * std::sin(2 * pi * (u * 19 + .15 * std::sin(v * 11)))
                          : .045 * std::cos(10 * pi * u);
      double shade  = .94 + stripe + .08 * (n - .5) + .035 * std::sin(2 * pi * v);
      bool   pore   = kind == FruitKind::Apple ? n < .018 : n < (.012 + .025 * a.mottling);
      double spot   = pore ? (kind == FruitKind::Apple ? 1.14 : .44) : 1.;
      if (kind == FruitKind::Banana && (v < .055 || v > .955))
        spot *= .45;
      int at = 4 * (y * 256 + x);
      for (int c = 0; c < 3; ++c) {
        double pigment = color[c] * shade * spot;
        if (pore && kind == FruitKind::Apple)
          pigment = .65 * pigment + .35 * std::array<double, 3>{.84, .72, .43}[c];
        albedo.pixels[at + c] = static_cast<unsigned char>(255 * std::clamp(pigment, 0., 1.));
      }
      normal.pixels[at] = static_cast<unsigned char>(
          128 + ((kind == FruitKind::Banana ? std::sin(10 * pi * u) * 1.4 : 0) + (n - .5) * 4));
      normal.pixels[at + 1] = static_cast<unsigned char>(128 + (f - .5) * 4);
      normal.pixels[at + 2] = 255;
      double r              = kind == FruitKind::Apple ? .24 + .12 * a.roughness + .025 * n
                                                       : .43 + .14 * a.roughness + .06 * n;
      if (pore)
        r += .12;
      for (int c = 0; c < 3; ++c)
        rough.pixels[at + c] = static_cast<unsigned char>(255 * std::clamp(r, 0., 1.));
      albedo.pixels[at + 3] = normal.pixels[at + 3] = rough.pixels[at + 3] = 255;
    }
  s.textures = {std::move(albedo), std::move(normal), std::move(rough)};
  cad::Object3D o;
  o.name                = kind == FruitKind::Apple ? "apple" : "banana";
  o.mesh_index          = 0;
  o.color               = {1, 1, 1, 1};
  o.texture_index       = 0;
  o.normal_map_index    = 1;
  o.roughness_map_index = 2;
  s.AddObject(o);
  if (kind == FruitKind::Apple) {
    tube(
        s,
        "apple woody stem",
        {0, 0, .034},
        {.0025, -.001, .049},
        .0028,
        .0017,
        {.24f, .115f, .042f, 1},
        10);
    for (int j = 0; j < 5; ++j) {
      double t = 2 * pi * j / 5;
      tube(
          s,
          "apple calyx",
          {0, 0, -.036},
          {.004 * std::cos(t), .004 * std::sin(t), -.0365},
          .0012,
          .0003,
          {.20f, .12f, .045f, 1},
          5);
    }
    if (a.leaf) {
      cad::MeshData leaf;
      leaf.name = "apple curved leaf";
      for (int i = 0; i <= 16; ++i)
        for (int j = 0; j <= 4; ++j) {
          double t = i / 16.,
                 w = (j / 4. - .5) * .014 * std::sin(pi * t) * (1 + .045 * std::sin(i * pi));
          leaf.positions.insert(
              leaf.positions.end(),
              {float(.002 + .028 * t),
               float(w),
               float(.045 + .008 * std::sin(pi * t) - .004 * std::abs(w) / .007)});
          leaf.texcoords.insert(leaf.texcoords.end(), {float(j) / 4, float(t)});
        }
      for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 4; ++j) {
          unsigned p = i * 5 + j;
          leaf.indices.insert(leaf.indices.end(), {p, p + 1, p + 5, p + 1, p + 6, p + 5});
        }
      leaf.RecalculateNormals();
      cad::Object3D l;
      l.name       = "apple veined leaf";
      l.mesh_index = s.meshes.size();
      l.color      = {.12f, .31f, .035f, 1};
      l.roughness  = .57f;
      s.meshes.push_back(std::move(leaf));
      s.AddObject(l);
      tube(
          s,
          "apple leaf midrib",
          {.002, 0, .0455},
          {.029, 0, .047},
          .00035,
          .00012,
          {.30f, .43f, .085f, 1},
          5);
    }
  } else {
    double length = .076 * lengths[k], curve = curves[k];
    tube(
        s,
        "banana stalk",
        {-length, -curve, 0},
        {-length - .009, -curve - .008, .002},
        .004,
        .003,
        {.29f, .24f, .075f, 1},
        10);
    tube(
        s,
        "banana dry blossom tip",
        {length - .002, -curve, 0},
        {length + .004, -curve - .005, 0},
        .004,
        .002,
        {.20f, .105f, .035f, 1},
        9);
  }
  return s;
}
void robot_materials(cad::Scene& s) {
  for (auto& o : s.objects) {
    double mx = std::max({o.color.r, o.color.g, o.color.b}),
           mn = std::min({o.color.r, o.color.g, o.color.b});
    if (mx < .22) {
      o.roughness = .39f;
      o.metallic  = .12f;
    } else if (mx - mn < .13) {
      o.roughness = .34f;
      o.metallic  = o.name.find("screw") != std::string::npos ? .85f : .05f;
    } else {
      o.roughness = .43f;
      o.metallic  = 0;
      float grey  = (o.color.r + o.color.g + o.color.b) / 3;
      o.color.r   = .84f * o.color.r + .16f * grey;
      o.color.g   = .84f * o.color.g + .16f * grey;
      o.color.b   = .84f * o.color.b + .16f * grey;
    }
  }
}
} // namespace detail::visuals
} // namespace datamunge::sim
