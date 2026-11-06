#pragma once
/// @file visual_assets.hpp
/// @brief Private procedural, code-native sorting-workcell asset builders.
#include <datamunge/sim/conveyor.hpp>
/// @brief Private deterministic workcell asset construction.
namespace datamunge::sim::detail::visuals {
/// @brief Build a locally centred rounded box, with analytic normals and metre-scaled UVs.
/// @param size Positive full XYZ extents, metres.
/// @param radius Edge radius in metres, clamped to fit the smallest dimension.
/// @return Owned indexed surface mesh, with its vertex placements already sized.
cad::MeshData rounded_box(cad::Vec3d size, double radius);
/// @brief Add plausible materials, rounded fixtures and cosmetic industrial hardware.
/// @param scene Canonical static workcell copy, mutated; no physics bodies are changed.
void decorate(cad::Scene& scene);
/// @brief Build an unscaled, local detailed fruit with seeded surface maps and botanical parts.
/// @param kind Apple or banana.
/// @param appearance Valid prototype 0..5 and deterministic material realization.
/// @return Owned local scene, centred like the original collision proxy.
cad::Scene fruit(FruitKind kind, const FruitAppearance& appearance);
/// @brief Assign plausible plastic/metal roughness to the imported arm without altering its
/// geometry.
/// @param scene Owned arm copy, mutated in place.
void robot_materials(cad::Scene& scene);
} // namespace datamunge::sim::detail::visuals
