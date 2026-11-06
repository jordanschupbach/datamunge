/// @file conveyor.cpp
/// @brief Native fruit-conveyor task geometry, fixed-step physics and observations.
#include <algorithm>
#include <cmath>
#include <datamunge/cad/math.hpp>
#include <datamunge/sim/conveyor.hpp>
#include <random>
#include <stdexcept>

#include "math.hpp"
#include "visual_assets.hpp"

namespace datamunge::sim {
/// @brief Private geometry and simulation construction helpers.
namespace {
/// @brief Shared SI dimensions; the sensor samples the payload centre.
constexpr double speed  = .22; ///< Moving tread speed, metres per second.
constexpr double pick_x = .10; ///< Centre-based optical trigger, metres along the belt.
constexpr double top    = .72; ///< Belt top above the world floor, metres.
constexpr double pi     = 3.14159265358979323846; ///< Half turn in radians.
/// @brief Neutral steel, belt rubber and payload/bin colors.
const cad::RgbaColorF steel{.48f, .57f, .64f, 1};   ///< Painted metal frame.
const cad::RgbaColorF rubber{.055f, .09f, .12f, 1}; ///< Belt bed.
const cad::RgbaColorF red{.8f, .065f, .075f, 1};    ///< Apple peel.
const cad::RgbaColorF yellow{1, .73f, .06f, 1};     ///< Banana peel.
const cad::RgbaColorF green{.1f, .48f, .18f, 1};    ///< Leaf and running sensor lamp.

/// @brief Convert Y-up physics positions/full extents to Z-up CAD values.
cad::Vec3f z_up(const Vec3& v) {
  return {float(v[0]), float(v[2]), float(v[1])};
}
/// @brief Add a centred primitive, returning a mutable object only for immediate use.
cad::Object3D& primitive(
    cad::Scene&        scene,
    cad::PrimitiveKind kind,
    const std::string& name,
    cad::Vec3f         at,
    cad::Vec3f         size,
    cad::RgbaColorF    color) {
  const auto id     = cad::AddPrimitiveToScene(&scene, kind);
  auto&      object = *scene.FindObject(id);
  object.name       = name;
  object.position   = at;
  object.scale      = size;
  object.color      = color;
  object.roughness  = .65f;
  return object;
}
/// @brief Construct compact solver storage for a single workcell.
WorldOptions options() {
  WorldOptions o;
  o.body_capacity     = 64;
  o.pair_capacity     = 1024;
  o.contact_capacity  = 1024;
  o.substeps          = 1;
  o.contact_slop      = .0002;
  o.solver_iterations = 24;
  return o;
}
/// @brief Build a lobed apple or a curved banana tube, centred at the body origin.
cad::MeshData fruit_mesh(FruitKind kind, const FruitAppearance& appearance, bool varied) {
  const std::size_t k = varied ? appearance.prototype : 0;
  const double      girth[6]{1, .80, 1.10, 1.02, .9, .94};
  const double      lobes[6]{.055, .025, .02, .12, .08, .04};
  const double      taper[6]{0, .16, -.08, .04, .23, -.12};
  const double      lengths[6]{1, 1.15, .86, 1.02, 1.08, .84};
  const double      curves[6]{.028, .018, .045, .05, .035, .055};
  const double      thickness[6]{1, .82, 1.16, .9, 1.04, 1.12};
  cad::MeshData     m;
  m.name             = kind == FruitKind::Apple ? "apple peel" : "banana peel";
  constexpr int rows = 24, columns = 32;
  for (int i = 0; i <= rows; ++i) {
    double t = double(i) / rows;
    for (int j = 0; j <= columns; ++j) {
      double a = 2 * pi * j / columns;
      if (kind == FruitKind::Apple) {
        double lat = pi * t, r = .039 * (varied ? girth[k] : 1) * std::sin(lat) *
                                 (1 + lobes[k] * std::cos((k == 3 ? 6 : 5) * a) * std::sin(lat)) *
                                 (1 + taper[k] * std::cos(lat));
        m.positions.insert(
            m.positions.end(),
            {float(r * std::cos(a)),
             float(r * std::sin(a)),
             float(.039 * std::cos(lat) - .004 * std::exp(-lat * lat / .08))});
      } else {
        double u = 2 * t - 1,
               r = .023 * thickness[k] * (.12 + .88 * std::sqrt(std::max(0., std::sin(pi * t))));
        m.positions.insert(
            m.positions.end(),
            {float(.076 * lengths[k] * u),
             float((varied ? -curves[k] * u * u : .028 * (1 - u * u) - .014) + r * std::cos(a)),
             float(r * std::sin(a))});
      }
      m.texcoords.insert(m.texcoords.end(), {float(j) / columns, float(t)});
    }
  }
  for (int i = 0; i < rows; ++i)
    for (int j = 0; j < columns; ++j) {
      unsigned a = i * (columns + 1) + j, b = a + columns + 1;
      m.indices.insert(m.indices.end(), {a, b, a + 1, a + 1, b, b + 1});
    }
  m.RecalculateNormals();
  return m;
}
/// @brief Shared seeded skin texture; appearance, not semantic class, controls its colors.
cad::TextureData skin_texture(const FruitAppearance& a) {
  cad::TextureData t;
  t.name   = "seeded fruit skin";
  t.width  = 96;
  t.height = 48;
  t.pixels.resize(96 * 48 * 4);
  std::mt19937_64                       rng(a.texture_seed);
  std::uniform_real_distribution<float> uniform(0, 1);
  for (int y = 0; y < t.height; ++y)
    for (int x = 0; x < t.width; ++x) {
      float  u     = uniform(rng);
      double shade = .90 + .12 * std::sin(.3 * x + .15 * y) + .12 * uniform(rng);
      if (u < a.mottling * .15)
        shade *= .35 + .25 * uniform(rng);
      for (int c = 0; c < 3; ++c)
        t.pixels[4 * (y * t.width + x) + c] =
            static_cast<unsigned char>(255 * std::clamp(a.skin[c] * shade, 0., 1.));
      t.pixels[4 * (y * t.width + x) + 3] = 255;
    }
  return t;
}
/// @brief Convert a mesh's world-CAD rotation to a Y-up physics quaternion.
Quaternion appearance_rotation(const FruitAppearance& a) {
  auto cad = detail::product(
      detail::axis_angle({0, 0, 1}, a.yaw),
      detail::product(
          detail::axis_angle({0, 1, 0}, a.pitch), detail::axis_angle({1, 0, 0}, a.roll)));
  return {cad[0], -cad[1], -cad[3], -cad[2]};
}
/// @brief Conservative vertical support extent for the supported fruit proxies.
double support_height(const Body& b) {
  if (b.shape == Shape::Sphere)
    return b.dimensions[0] / 2;
  auto axis = detail::rotate(b.rotation, {0, 1, 0});
  return b.dimensions[0] / 2 + (b.dimensions[1] - b.dimensions[0]) / 2 * std::abs(axis[1]);
}
/// @brief Seed a geometry, shared palette, scale, material and three-dimensional pose.
FruitAppearance sample_appearance(std::mt19937_64& rng) {
  std::uniform_real_distribution<double> u(0, 1);
  FruitAppearance                        a;
  a.prototype = std::uniform_int_distribution<int>(0, 5)(rng);
  a.scale     = .85 + .3 * u(rng);
  const std::array<std::array<float, 3>, 4> palette{
      {{.25f, .58f, .14f}, {.93f, .76f, .16f}, {.75f, .14f, .08f}, {.59f, .36f, .13f}}};
  a.skin = palette[std::uniform_int_distribution<int>(0, 3)(rng)];
  for (float& c : a.skin)
    c = float(std::clamp(c * (.85 + .3 * u(rng)), .02, .98));
  a.mottling     = .35 * u(rng);
  a.roughness    = .25 + .5 * u(rng);
  a.yaw          = 2 * pi * u(rng);
  a.pitch        = .24 * (u(rng) - .5);
  a.roll         = .24 * (u(rng) - .5);
  a.texture_seed = rng();
  a.leaf         = u(rng) > .5;
  return a;
}
} // namespace

/// @brief Exclusive physics state, reproducible fruit sequence and static CAD geometry.
struct ConveyorEnvironment::Impl {
  WorkcellVisualStyle style{
      WorkcellVisualStyle::Simple}; ///< Rendering profile, independent of domain RNGs.
  std::shared_ptr<const cad::Scene>
                 realistic; ///< Immutable static visual cache retained across resets.
  FruitVariation variation{FruitVariation::Basic}; ///< Domain protocol retained across resets.
  std::unique_ptr<World> physics; ///< Rebuilt by reset; external world references then expire.
  std::mt19937_64        random;  ///< Seeded sequence, independent of frame rate/rendering.
  ConveyorObservation    state;   ///< Latched timing, class and completion observations.
  BodyId                 belt{0}; ///< Stationary support with a moving material surface.
  cad::Scene             scenery; ///< Static CAD geometry matching the rigid collision boxes.

  /// @brief Add one static box to both physics and render geometry.
  BodyId box(const std::string& name, Vec3 at, Vec3 size, cad::RgbaColorF color) {
    Body b;
    b.position    = at;
    b.dimensions  = size;
    b.restitution = 0;
    b.friction    = .9;
    auto id       = physics->add_body(b);
    primitive(scenery, cad::PrimitiveKind::Cube, name, z_up(at), z_up(size), color);
    return id;
  }
  /// @brief Start one new seeded payload, preserving elapsed episode time and bins.
  void spawn() {
    state.kind         = (random() & 1) ? FruitKind::Banana : FruitKind::Apple;
    state.stage        = ConveyorStage::Travelling;
    state.arrival_time = -1;
    Body b;
    b.motion     = Motion::Dynamic;
    b.shape      = state.kind == FruitKind::Apple ? Shape::Sphere : Shape::Capsule;
    b.dimensions = state.kind == FruitKind::Apple ? Vec3{.078, .078, .078} : Vec3{.046, .16, .046};
    if (state.kind == FruitKind::Banana)
      b.rotation = {std::sqrt(.5), 0, 0, -std::sqrt(.5)};
    state.variation = variation;
    if (variation == FruitVariation::Diverse) {
      state.appearance = sample_appearance(random);
      auto&        a   = state.appearance;
      const double lengths[6]{1, 1.15, .86, 1.02, 1.08, .84},
          thickness[6]{1, .82, 1.16, .9, 1.04, 1.12};
      double diameter = .078 * a.scale;
      if (state.kind == FruitKind::Apple) {
        const double girth[6]{1, .80, 1.10, 1.02, .9, .94};
        diameter *= std::max(1., girth[a.prototype] * 1.08);
        b.dimensions = {diameter, diameter, diameter};
      } else
        b.dimensions = {
            .046 * thickness[a.prototype] * a.scale,
            .164 * lengths[a.prototype] * a.scale,
            .046 * thickness[a.prototype] * a.scale};
      b.rotation = detail::product(appearance_rotation(a), b.rotation);
      std::uniform_real_distribution<double> entry(-1., -.7), lateral(-.065, .065);
      b.position = {entry(random), top + support_height(b) + .02, lateral(random)};
    } else
      b.position = {-.9, top + (state.kind == FruitKind::Apple ? .039 : .023) + .001, 0};
    b.mass                 = state.kind == FruitKind::Apple ? .16 : .12;
    b.restitution          = 0;
    b.friction             = .9;
    b.linear_damping       = 0;
    b.allowed_rotation     = {false, false, false};
    state.fruit            = physics->add_body(b);
    auto tread             = physics->body(belt);
    tread.surface_velocity = {speed, 0, 0};
    physics->set_body(belt, tread);
  }
  /// @brief Assemble collision boxes and matching workcell scenery in metres.
  void build() {
    box("floor", {-.15, -.06, 0}, {2.9, .12, 1.9}, {.79f, .83f, .85f, 1});
    belt = box("conveyor rubber bed", {-.30, .68, 0}, {1.5, .08, .34}, rubber);
    for (double z : {-.19, .19}) {
      box("conveyor side rail", {-.30, .686, z}, {1.58, .10, .035}, steel);
      for (double x : {-.94, .34})
        box("conveyor leg", {x, .32, z}, {.065, .64, .065}, steel);
    }
    for (double x : {-1.025, .425}) {
      primitive(
          scenery,
          cad::PrimitiveKind::Cylinder,
          "end roller",
          {float(x), -.17f, .68f},
          {.075f, .34f, .075f},
          {.25f, .31f, .36f, 1});
    }
    box("arm pedestal", {.62, .35, 0}, {.24, .70, .24}, steel);
    box("arm mounting plate", {.62, .71, 0}, {.29, .02, .29}, {.16f, .22f, .28f, 1});
    for (double z : {-.42, .42}) {
      auto color =
          z < 0 ? cad::RgbaColorF{.77f, .19f, .21f, 1} : cad::RgbaColorF{.99f, .84f, .05f, 1};
      box(z < 0 ? "apple bin floor" : "banana bin floor",
          {pick_x, .43, z},
          {.34, .025, .30},
          color);
      for (double dx : {-.16, .16})
        box("bin end wall", {pick_x + dx, .54, z}, {.02, .22, .30}, color);
      for (double dz : {-.14, .14})
        box("bin side wall", {pick_x, .54, z + dz}, {.30, .22, .02}, color);
      for (double dx : {-.13, .13})
        box("bin stand", {pick_x + dx, .21, z}, {.035, .42, .22}, steel);
    }
    // A narrow, nonblocking optical sensor; its beam is visual, never a solid wall.
    box("sensor post", {pick_x, .78, .205}, {.025, .20, .025}, steel);
    primitive(
        scenery,
        cad::PrimitiveKind::Cube,
        "pick station mark",
        {float(pick_x), 0, .7205f},
        {.007f, .33f, .001f},
        {.25f, .8f, .7f, 1});
    // Category silhouettes on the outer bin faces remain readable from the camera.
    primitive(
        scenery,
        cad::PrimitiveKind::Sphere,
        "apple bin symbol",
        {.10f, -.572f, .55f},
        {.07f, .006f, .065f},
        {1, .73f, .72f, 1});
    for (int i = 0; i < 6; ++i) {
      float x = .065f + .013f * i, z = .548f + .018f * std::sin(pi * i / 5);
      primitive(
          scenery,
          cad::PrimitiveKind::Sphere,
          "banana bin symbol",
          {x, .572f, z},
          {.024f, .006f, .018f},
          {1, .95f, .66f, 1});
    }
  }
};
ConveyorEnvironment::ConveyorEnvironment() : ConveyorEnvironment(FruitVariation::Basic) {}
ConveyorEnvironment::ConveyorEnvironment(FruitVariation variation)
    : impl_(std::make_unique<Impl>()) {
  if (variation != FruitVariation::Basic && variation != FruitVariation::Diverse)
    throw std::invalid_argument("unknown fruit variation");
  impl_->variation = variation;
  reset(0);
}
ConveyorEnvironment::~ConveyorEnvironment() = default;
void ConveyorEnvironment::set_visual_style(WorkcellVisualStyle style) {
  if (style != WorkcellVisualStyle::Simple && style != WorkcellVisualStyle::Realistic)
    throw std::invalid_argument("unknown workcell style");
  impl_->style = style;
}
WorkcellVisualStyle ConveyorEnvironment::visual_style() const noexcept {
  return impl_->style;
}

void ConveyorEnvironment::reset(std::uint64_t seed) {
  auto fresh       = std::make_unique<Impl>();
  fresh->style     = impl_->style;
  fresh->realistic = impl_->realistic;
  fresh->variation = impl_->variation;
  fresh->physics   = std::make_unique<World>(options());
  fresh->random.seed(seed);
  fresh->state.seed = seed;
  fresh->build();
  fresh->spawn();
  impl_ = std::move(fresh);
}
void ConveyorEnvironment::step(double seconds) {
  if (!std::isfinite(seconds) || seconds <= 0 || seconds > 1. / 30)
    throw std::invalid_argument("conveyor step must be in (0,1/30]");
  auto&        p     = *impl_;
  const int    ticks = int(std::ceil(seconds * 240));
  const double dt    = seconds / ticks;
  for (int i = 0; i < ticks; ++i) {
    p.physics->step(dt);
    p.state.time += dt;
    auto fruit = p.physics->body(p.state.fruit);
    if (p.state.stage == ConveyorStage::Travelling) {
      p.state.belt_travel += speed * dt;
      if (fruit.position[0] >= pick_x) {
        p.state.stage        = ConveyorStage::Waiting;
        p.state.arrival_time = p.state.time;
        auto b               = p.physics->body(p.belt);
        b.surface_velocity   = {0, 0, 0};
        p.physics->set_body(p.belt, b);
      }
    }
    if (p.state.stage != ConveyorStage::Travelling && p.state.stage != ConveyorStage::Waiting)
      continue;
    double v2 = 0;
    for (double v : fruit.linear_velocity)
      v2 += v * v;
    if (std::abs(fruit.position[0] - pick_x) < .14 && fruit.position[1] > .43 &&
        fruit.position[1] < .60 && std::abs(std::abs(fruit.position[2]) - .42) < .12 &&
        v2 < .0009) {
      bool correct  = (fruit.position[2] < 0) == (p.state.kind == FruitKind::Apple);
      p.state.stage = correct ? ConveyorStage::Sorted : ConveyorStage::Missorted;
    } else if (
        fruit.position[1] < .10 || std::abs(fruit.position[0]) > 1.5 ||
        std::abs(fruit.position[2]) > 1.2) {
      p.state.stage = ConveyorStage::Lost;
    }
    if (p.state.stage != ConveyorStage::Travelling) {
      auto b = p.physics->body(p.belt);
      if (b.surface_velocity[0] != 0) {
        b.surface_velocity = {0, 0, 0};
        p.physics->set_body(p.belt, b);
      }
    }
  }
}
void ConveyorEnvironment::reset_for_pick(std::uint64_t seed) {
  reset(seed);
  auto& p            = *impl_;
  auto  b            = p.physics->body(p.belt);
  b.surface_velocity = {0, 0, 0};
  p.physics->set_body(p.belt, b);
  auto                                   f = p.physics->body(p.state.fruit);
  std::uniform_real_distribution<double> jitter(-.008, .008);
  f.position[0] = pick_x + jitter(p.random);
  if (p.variation == FruitVariation::Basic)
    f.position[2] = jitter(p.random);
  f.position[1]     = top + support_height(f) + .001;
  f.linear_velocity = {0, 0, 0};
  p.physics->set_body(p.state.fruit, f);
  p.state.stage        = ConveyorStage::Waiting;
  p.state.arrival_time = 0;
  for (int i = 0; i < 36; ++i)
    step(1. / 240);
  p.state.time = 0;
}
void ConveyorEnvironment::next_fruit() {
  auto& p = *impl_;
  if (p.state.stage == ConveyorStage::Travelling || p.state.stage == ConveyorStage::Waiting)
    throw std::logic_error("pick and place the current fruit before spawning another");
  p.physics->remove_body(p.state.fruit);
  p.spawn();
}
ConveyorObservation ConveyorEnvironment::observation() const {
  auto s       = impl_->state;
  auto b       = impl_->physics->body(s.fruit);
  s.position   = b.position;
  s.velocity   = b.linear_velocity;
  s.angular_velocity=b.angular_velocity;s.mass=b.mass;
  s.rotation   = b.rotation;
  s.dimensions = b.dimensions;
  return s;
}
World& ConveyorEnvironment::world() {
  return *impl_->physics;
}
const World& ConveyorEnvironment::world() const {
  return *impl_->physics;
}
cad::Scene ConveyorEnvironment::static_scene() const {
  if (impl_->style == WorkcellVisualStyle::Realistic) {
    if (!impl_->realistic) {
      auto scene = impl_->scenery;
      detail::visuals::decorate(scene);
      impl_->realistic = std::make_shared<cad::Scene>(std::move(scene));
    }
    return *impl_->realistic;
  }
  return impl_->scenery;
}
cad::Scene ConveyorEnvironment::dynamic_scene() const {
  return dynamic_scene(observation());
}
cad::Scene ConveyorEnvironment::dynamic_scene(const ConveyorObservation& s) const {
  const bool varied = s.variation == FruitVariation::Diverse;
  if (s.appearance.prototype >= 6 || !std::isfinite(s.appearance.scale) || s.appearance.scale <= 0)
    throw std::invalid_argument("invalid recorded fruit appearance");
  if (s.variation != FruitVariation::Basic && s.variation != FruitVariation::Diverse)
    throw std::invalid_argument("unknown recorded fruit variation");
  if (s.kind != FruitKind::Apple && s.kind != FruitKind::Banana)
    throw std::invalid_argument("unknown recorded fruit kind");
  for (auto v : s.position)
    if (!std::isfinite(v))
      throw std::invalid_argument("nonfinite fruit position");
  double norm = 0;
  for (auto v : s.rotation) {
    if (!std::isfinite(v))
      throw std::invalid_argument("nonfinite fruit rotation");
    norm += v * v;
  }
  if (std::abs(norm - 1) > 1e-5)
    throw std::invalid_argument("fruit rotation must be normalized");
  for (auto v : s.dimensions)
    if (!std::isfinite(v) || v <= 0)
      throw std::invalid_argument("invalid fruit dimensions");
  for (auto v : s.appearance.skin)
    if (!std::isfinite(v) || v < 0 || v > 1)
      throw std::invalid_argument("invalid skin color");
  for (auto v : {s.belt_travel, s.appearance.mottling, s.appearance.roughness})
    if (!std::isfinite(v))
      throw std::invalid_argument("invalid fruit material or belt travel");
  cad::Scene scene;
  for (int i = 0; i < 13; ++i) {
    double x = -1.04 + std::fmod(i * .12 + s.belt_travel, 1.48);
    primitive(
        scene,
        cad::PrimitiveKind::Cube,
        "moving tread",
        {float(x), 0, .7215f},
        {.009f, .33f, .0015f},
        {.18f, .24f, .28f, 1});
  }
  primitive(
      scene,
      cad::PrimitiveKind::Sphere,
      "sensor status",
      {float(pick_x), .205f, .89f},
      {.035f, .035f, .035f},
      s.stage == ConveyorStage::Travelling ? green : cad::RgbaColorF{1, .55f, .04f, 1});
  cad::Scene local;
  if (impl_->style == WorkcellVisualStyle::Realistic)
    local = detail::visuals::fruit(s.kind, s.appearance);
  else {
    local.meshes.push_back(fruit_mesh(s.kind, s.appearance, varied));
    cad::Object3D fruit;
    fruit.name       = s.kind == FruitKind::Apple ? "apple" : "banana";
    fruit.mesh_index = 0;
    fruit.color      = s.kind == FruitKind::Apple ? red : yellow;
    fruit.roughness  = float(s.appearance.roughness);
    if (varied) {
      local.textures.push_back(skin_texture(s.appearance));
      fruit.texture_index = 0;
      fruit.color         = {1, 1, 1, 1};
    }
    local.AddObject(fruit);
    if (s.kind == FruitKind::Apple) {
      primitive(
          local,
          cad::PrimitiveKind::Cube,
          "apple stem",
          {0, 0, .043f},
          {.005f, .005f, .017f},
          {.25f, .12f, .035f, 1});
      if (!varied || s.appearance.leaf) {
        auto& leaf = primitive(
            local,
            cad::PrimitiveKind::Sphere,
            "apple leaf",
            {.011f, 0, .044f},
            {.025f, .011f, .004f},
            green);
        leaf.rotation_deg = {0, -20, 20};
      }
    } else {
      const double lengths[6]{1, 1.15, .86, 1.02, 1.08, .84},
          curves[6]{.028, .018, .045, .05, .035, .055};
      auto k = varied ? s.appearance.prototype : 0;
      for (float sign : {-1.f, 1.f})
        primitive(
            local,
            cad::PrimitiveKind::Sphere,
            "banana tip",
            {float(sign * .076 * lengths[k]), float(varied ? -curves[k] : -.014), 0},
            {.011f, .009f, .009f},
            {.29f, .15f, .035f, 1});
    }
  }
  if (impl_->style == WorkcellVisualStyle::Realistic)
    for (auto& o : scene.objects)
      if (o.name == "moving tread") {
        o.scale      = {.0014f, .327f, .00018f};
        o.position.z = .72018f;
        o.color      = {.12f, .135f, .14f, 1};
        o.roughness  = .9f;
      }
  if (impl_->style == WorkcellVisualStyle::Realistic)
    for (auto& o : scene.objects)
      if (o.name == "sensor status") {
        o.scale      = {.012f, .012f, .012f};
        o.position.z = .886f;
        o.roughness  = .18f;
      }
  auto q = s.rotation;
  if (s.kind == FruitKind::Banana)
    q = detail::product(q, detail::conjugate(Quaternion{std::sqrt(.5), 0, 0, -std::sqrt(.5)}));
  cad::Vec3d   axis{-q[1], -q[3], -q[2]};
  auto         rotation = axis.Length() > 1e-12
                              ? cad::Mat4d::Rotation(axis, 2 * std::atan2(axis.Length(), q[0]))
                              : cad::Mat4d::Identity();
  const double scale    = varied ? s.appearance.scale : 1;
  double       lowest   = 1e9;
  for (auto& o : local.objects) {
    const double degree    = pi / 180.;
    auto         transform = rotation * cad::Mat4d::Scaling({scale, scale, scale}) *
                     cad::Mat4d::Translation({o.position.x, o.position.y, o.position.z}) *
                     cad::Mat4d::Rotation({0, 0, 1}, o.rotation_deg.z * degree) *
                     cad::Mat4d::Rotation({0, 1, 0}, o.rotation_deg.y * degree) *
                     cad::Mat4d::Rotation({1, 0, 0}, o.rotation_deg.x * degree) *
                     cad::Mat4d::Scaling({o.scale.x, o.scale.y, o.scale.z});
    auto& m = local.meshes[o.mesh_index];
    for (std::size_t i = 0; i < m.positions.size(); i += 3) {
      auto p = transform.TransformPoint({m.positions[i], m.positions[i + 1], m.positions[i + 2]});
      m.positions[i]     = float(p.x);
      m.positions[i + 1] = float(p.y);
      m.positions[i + 2] = float(p.z);
      lowest             = std::min(lowest, p.z);
    }
    for (std::size_t i = 0; i < m.normals.size(); i += 3) {
      auto n = transform.TransformNormal({m.normals[i], m.normals[i + 1], m.normals[i + 2]})
                   .Normalized();
      m.normals[i]     = float(n.x);
      m.normals[i + 1] = float(n.y);
      m.normals[i + 2] = float(n.z);
    }
    o.position     = {};
    o.scale        = {1, 1, 1};
    o.rotation_deg = {};
  }
  Body proxy;
  proxy.shape         = s.kind == FruitKind::Apple ? Shape::Sphere : Shape::Capsule;
  proxy.dimensions    = s.dimensions;
  proxy.rotation      = s.rotation;
  const double offset = varied && !s.physical_origin ? -support_height(proxy) - lowest : 0;
  for (auto& m : local.meshes)
    for (std::size_t i = 0; i < m.positions.size(); i += 3) {
      m.positions[i] += float(s.position[0]);
      m.positions[i + 1] += float(s.position[2]);
      m.positions[i + 2] += float(s.position[1] + offset);
    }
  int base = int(scene.meshes.size());
  for (auto& m : local.meshes)
    scene.meshes.push_back(std::move(m));
  scene.textures = std::move(local.textures);
  for (auto o : local.objects) {
    o.mesh_index += base;
    scene.AddObject(o);
  }
  return scene;
}
} // namespace datamunge::sim
