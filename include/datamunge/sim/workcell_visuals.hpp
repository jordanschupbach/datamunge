#pragma once
/// @file workcell_visuals.hpp
/// @brief Reproducible visual profiles for the native robot sorting workcell.
#include <datamunge/cad/render.hpp>
namespace datamunge::sim {
/// @brief Rendering-only workcell profile; collision geometry and random arrivals are unchanged.
/// Simple preserves earlier examples. Realistic adds rounded fixtures, hardware,
/// surface maps, botanical fruit detail and studio lighting. Its fruit tints use
/// overlapping natural green/gold/russet palettes rather than saturated debug colors.
/// @par Example
/// @code
/// auto style=dm::sim::WorkcellVisualStyle::Realistic;
/// @endcode
enum class WorkcellVisualStyle {
  Simple,   ///< Original diagnostic materials and geometry.
  Realistic ///< Detailed industrial assets and physically based shading.
};
/// @brief Construct consistent workcell lights for external and wrist views.
/// @param style Simple or Realistic.
/// @return Owned deterministic light list; Realistic uses warm key/cool fill/rim.
/// @throws std::invalid_argument Unknown visual profile.
/// @par Example
/// @code
/// dm::cad::Scene scene;
/// scene.lights=dm::sim::workcell_lights(dm::sim::WorkcellVisualStyle::Realistic);
/// @endcode
std::vector<cad::Light> workcell_lights(WorkcellVisualStyle style);
/// @brief Configure native render quality for a workcell view.
/// Realistic enables linear-light microfacet shading, tone mapping, soft shadows
/// and local occlusion. Sensor-sized views use fewer samples for training speed.
/// @param style Simple or Realistic.
/// @param width Positive image width, at most 8192; default 960.
/// @param height Positive image height, at most 8192; default 640.
/// @return Owned render settings; callers may increase sampling for still images.
/// @throws std::invalid_argument Unknown profile or invalid dimensions.
/// @par Example
/// @code
/// auto settings=dm::sim::workcell_render_options(dm::sim::WorkcellVisualStyle::Realistic);
/// @endcode
cad::RenderOptions
workcell_render_options(WorkcellVisualStyle style, int width = 960, int height = 640);

/// @brief Build an independent exterior presentation copy without changing the calibrated sensor
/// scene. Refines floor/rubber/plastic/metal finishes and seeded fruit surface maps, retaining
/// source meshes/transforms and brand/fruit hues. Optional factory context and a cosmetic flexible
/// servo harness add visual geometry only. Lighting is a warm key, cool fill and rim. Apply once to
/// each fresh component; pass robot link-derived anchors again when constructing a new joint pose.
/// The function is deterministic, consumes no domain RNGs and creates no physics
/// bodies. Keep the original scene for wrist RGB and classifier calibration.
///
/// @param scene Owned source; lvalues are copied, rvalues adopted.
/// @param surroundings Add factory backdrop, cabinet and fixture feet; default false.
/// @param cable_anchors Empty, or 2..32 world points in Z-up metres within +/-10000 m.
/// @return Owned scene with presentation materials, lights and optional cosmetic geometry.
/// @throws std::invalid_argument Invalid texture/mesh indices, RGBA shape or cable anchors.
/// @par Example
/// @code
/// dm::sim::ConveyorEnvironment env;
/// env.set_visual_style(dm::sim::WorkcellVisualStyle::Realistic);
/// auto exterior=dm::sim::workcell_presentation(env.static_scene(),true);
/// auto settings=dm::sim::workcell_presentation_options();
/// auto camera=dm::cad::fit_camera(exterior);
/// auto image=dm::cad::render(exterior,camera,settings);
/// @endcode
/// @see workcell_presentation_options(), WorkcellVisualStyle
cad::Scene workcell_presentation(
    cad::Scene                     scene,
    bool                           surroundings  = false,
    const std::vector<cad::Vec3d>& cable_anchors = {});

/// @brief Full-quality exterior settings with antialiasing and analytic studio reflections.
/// Uses 2x2 primary sampling, six soft-shadow rays and three local occlusion
/// rays per hit, plus a 0.55-strength analytic studio environment. The CPU wrist
/// sensor's workcell_render_options() is unchanged. This approximates studio
/// illumination rather than tracing scene reflections or multiple light bounces.
/// @param width Positive image width, at most 8192; default 960.
/// @param height Positive image height, at most 8192; default 640.
/// @return Owned physically based render settings; CPU and CUDA support the same fields.
/// @throws std::invalid_argument Invalid image dimensions.
/// @par Example See workcell_presentation().
/// @see cad::RenderOptions, workcell_render_options()
cad::RenderOptions workcell_presentation_options(int width = 960, int height = 640);
} // namespace datamunge::sim
