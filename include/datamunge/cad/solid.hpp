#pragma once
/// @file solid.hpp
/// @brief Owned precise CAD solids, modelling operations and scene/file adapters.

#include <datamunge/cad/boolean.hpp>
#include <datamunge/cad/drawing.hpp>
#include <datamunge/cad/exchange.hpp>
#include <datamunge/cad/feature.hpp>
#include <datamunge/cad/modify.hpp>
#include <datamunge/cad/pattern.hpp>
#include <datamunge/cad/scene.hpp>
#include <datamunge/cad/step.hpp>

namespace datamunge::cad {

/// @brief An owned B-rep solid with value-returning modelling and export operations.
/// Geometry uses doubles in caller-selected consistent units. Copies own topology;
/// immutable surfaces may be shared. Operations return new solids, preserving inputs.
/// Unsupported kernel operations throw std::runtime_error with the kernel diagnostic.
/// Concurrent const access is safe; do not mutate a shared object concurrently.
/// @par Example
/// @code
/// auto block = dm::cad::Solid::box(4, 3, 2);
/// auto hole = dm::cad::Solid::cylinder(.4, 4).translated(2, 1.5, -1);
/// auto part = block.subtracted(hole);
/// part.save("part.step");
/// @endcode
/// @see Model, FeatureTree, load_scene()
class Solid {
 public:
  /// @brief Own a copy/move of one body from an existing kernel model.
  /// @param model Owned topology, copied for an lvalue.
  /// @param body Existing body ID in model.
  /// @throws std::invalid_argument Missing body.
  /// @par Example
  /// See the class example; kernel users may construct Solid(model, body).
  Solid(Model model, EntityId body);

  /// @brief Construct an axis-aligned box with minimum corner at the origin.
  /// @param width Positive finite X extent.
  /// @param depth Positive finite Y extent.
  /// @param height Positive finite Z extent.
  /// @return Owned solid.
  /// @throws std::invalid_argument Nonpositive/nonfinite size.
  /// @par Example
  /// See the class example.
  static Solid box(double width, double depth, double height);
  /// @brief Construct a cylinder with base at the origin and axis +Z.
  /// @param radius Positive finite radius.
  /// @param height Positive finite height.
  /// @return Owned solid.
  /// @throws std::invalid_argument Invalid dimensions.
  /// @par Example
  /// See the class example.
  static Solid cylinder(double radius, double height);
  /// @brief Construct a sphere centred at the origin.
  /// @param radius Positive finite radius.
  /// @return Owned solid.
  /// @throws std::invalid_argument Invalid radius.
  /// @par Example
  /// @code
  /// auto ball = dm::cad::Solid::sphere(2).translated(0, 0, 2);
  /// @endcode
  static Solid sphere(double radius);
  /// @brief Construct a ring torus centred at the origin around +Z.
  /// @param major_radius Positive finite distance from axis to tube centre.
  /// @param minor_radius Positive finite tube radius smaller than major_radius.
  /// @return Owned solid.
  /// @throws std::invalid_argument Invalid radii.
  /// @par Example
  /// @code
  /// auto ring = dm::cad::Solid::torus(3, 1);
  /// @endcode
  static Solid torus(double major_radius, double minor_radius);
  /// @brief Return the union with another solid.
  /// @param other Borrowed operand, in the same units/frame.
  /// @return New solid; inputs unchanged.
  /// @throws std::runtime_error Kernel failure.
  /// @par Example
  /// @code
  /// auto a = dm::cad::Solid::box(2, 2, 2);
  /// auto b = a.translated(.5, .5, .5);
  /// auto joined = a.united(b);
  /// @endcode
  Solid united(const Solid& other) const;
  /// @brief Subtract another solid, preserving cavities and trimmed faces.
  /// @param other Borrowed tool solid.
  /// @return New solid.
  /// @throws std::runtime_error Kernel failure or empty result.
  /// @par Example
  /// See the class example.
  Solid subtracted(const Solid& other) const;
  /// @brief Return the common material of two solids.
  /// @param other Borrowed operand.
  /// @return New solid.
  /// @throws std::runtime_error Kernel failure or empty result.
  /// @par Example
  /// @code
  /// auto a = dm::cad::Solid::box(2, 2, 2);
  /// auto common = a.intersected(a.translated(.5, .5, .5));
  /// @endcode
  Solid intersected(const Solid& other) const;
  /// @brief Return a translated copy.
  /// @param x Finite X displacement in model units.
  /// @param y Finite Y displacement.
  /// @param z Finite Z displacement.
  /// @return New solid.
  /// @throws std::invalid_argument Nonfinite displacement.
  /// @throws std::runtime_error Transform failure.
  /// @par Example
  /// See the class example.
  Solid translated(double x, double y, double z) const;
  /// @brief Rotate about an axis through the origin, using the right-hand rule.
  /// @param x Axis X component.
  /// @param y Axis Y component.
  /// @param z Axis Z component; axis must be finite and nonzero.
  /// @param radians Finite rotation angle.
  /// @return New solid.
  /// @throws std::invalid_argument Invalid axis or angle.
  /// @throws std::runtime_error Kernel failure.
  /// @par Example
  /// @code
  /// auto turned = dm::cad::Solid::box(2, 3, 4).rotated(0, 0, 1, dm::cad::kHalfPi);
  /// @endcode
  Solid rotated(double x, double y, double z, double radians) const;
  /// @brief Apply a nonsingular affine transform, preserving boundary orientation.
  /// @param transform Finite affine placement; see TransformBody for supported surfaces.
  /// @return New solid.
  /// @throws std::runtime_error Unsupported/singular transform.
  /// @par Example
  /// @code
  /// auto a = dm::cad::Solid::box(1, 1, 1);
  /// auto b = a.transformed(dm::cad::Mat4d::Scaling({2, 3, 4}));
  /// @endcode
  Solid transformed(const Mat4d& transform) const;
  /// @brief Fillet selected edge IDs in sequence using persistent face-pair references.
  /// @param edges Edge IDs from this solid's model; empty list leaves a copy unchanged.
  /// @param radius Positive finite rolling-ball radius.
  /// @return New solid. Source kernel supports straight edges between planar faces.
  /// @throws std::invalid_argument Invalid radius.
  /// @throws std::runtime_error Unsupported edge/blend geometry.
  /// @par Example
  /// @code
  /// auto a = dm::cad::Solid::box(4, 4, 4);
  /// auto rounded = a.filleted({a.edges().front()}, .2);
  /// @endcode
  Solid filleted(const std::vector<int>& edges, double radius) const;
  /// @brief Chamfer selected edge IDs by a common setback.
  /// @param edges Source edge IDs; empty list returns a copy.
  /// @param setback Positive finite distance along adjacent faces.
  /// @return New solid.
  /// @throws std::invalid_argument Invalid setback.
  /// @throws std::runtime_error Kernel failure.
  /// @par Example
  /// @code
  /// auto a = dm::cad::Solid::box(4, 4, 4);
  /// auto bevel = a.chamfered({a.edges().front()}, .2);
  /// @endcode
  Solid chamfered(const std::vector<int>& edges, double setback) const;
  /// @brief Hollow a solid, optionally opening specified faces.
  /// @param thickness Positive finite wall thickness.
  /// @param open_faces Face IDs to open; empty creates a sealed cavity.
  /// @return New solid, subject to ShellBody's supported surface combinations.
  /// @throws std::invalid_argument Invalid thickness.
  /// @throws std::runtime_error Kernel failure.
  /// @par Example
  /// @code
  /// auto hollow = dm::cad::Solid::box(4, 4, 4).shelled(.2);
  /// @endcode
  Solid shelled(double thickness, const std::vector<int>& open_faces = {}) const;
  /// @brief Approximate the trimmed boundary by a triangle mesh.
  /// @param chord Positive finite chord tolerance in model units; default .01.
  /// @return Owned mesh, including face IDs and normals.
  /// @throws std::invalid_argument Invalid chord.
  /// @throws std::runtime_error Tessellation failure.
  /// @par Example
  /// @code
  /// auto mesh = dm::cad::Solid::sphere(1).tessellate(.01);
  /// @endcode
  TessellationMesh tessellate(double chord = .01) const;
  /// @brief Measure volume from the trimmed tessellated boundary.
  /// @param chord Positive tessellation tolerance; lower improves curved-face accuracy.
  /// @return Signed volume in cubic model units; honours imported STEP trimming.
  /// @throws std::invalid_argument Invalid tolerance.
  /// @throws std::runtime_error Tessellation failure.
  /// @par Example
  /// @code
  /// auto volume = dm::cad::Solid::box(2, 3, 4).volume(); // 24
  /// @endcode
  double volume(double chord = .01) const;
  /// @brief Validate topology, orientations and geometric boundary consistency.
  /// @return True when the kernel validation report has no errors.
  /// @par Example
  /// @code
  /// bool valid = dm::cad::Solid::box(2, 3, 4).valid(); // true
  /// @endcode
  bool valid() const;
  /// @brief List the source edge IDs usable by filleted() and chamfered().
  /// @return Owned vector of unique edge IDs.
  std::vector<int> edges() const;
  /// @brief List face IDs usable by shelled() and the low-level modifiers.
  /// @return Owned vector of face IDs.
  std::vector<int> faces() const;
  /// @brief Borrow immutable topology for advanced kernel operations.
  /// @return Reference valid until this solid is destroyed or assigned.
  const Model& model() const { return model_; }
  /// @brief Identify the owned body inside model().
  /// @return Stable body ID until assignment.
  EntityId body() const { return body_; }
  /// @brief Save STEP, binary STL, OBJ, glTF or an SVG drawing by extension.
  /// @param path Output path; parent directory must exist; existing files are replaced.
  /// @param chord Positive tessellation tolerance for mesh/drawing output; default .01.
  /// @throws std::invalid_argument Unsupported extension or tolerance.
  /// @throws std::runtime_error Export or file I/O failure.
  /// @par Example
  /// See the class example.
  void save(const std::string& path, double chord = .01) const;

 private:
  Model    model_;            ///< Owned topology with shared immutable geometry values.
  EntityId body_ = kNoEntity; ///< The one body represented by this value.

  /// @brief Delegate a binary modelling operation to the canonical B-rep kernel.
  /// @param other Borrowed tool/operand.
  /// @param operation Union, intersection or difference.
  /// @return Newly owned result solid, preserving both operands.
  /// @throws std::runtime_error Kernel failure or empty result.
  Solid boolean_with(const Solid& other, BooleanOp operation) const;
};

/// @brief Add a tessellated solid as an owned polygon scene object.
/// @param scene Mutable destination; existing objects are retained.
/// @param solid Borrowed source, unchanged.
/// @param color RGBA tint in [0,1]; default blue.
/// @param name Object name, default "Solid".
/// @param chord Positive chord tolerance in model units, default .01.
/// @return Scene object ID; its mesh is independent of solid's lifetime.
/// @throws std::invalid_argument Invalid tolerance or color.
/// @throws std::runtime_error Tessellation failure.
/// @par Example
/// @code
/// dm::cad::Scene scene;
/// dm::cad::add_solid(scene, dm::cad::Solid::box(2, 3, 4));
/// @endcode
int add_solid(
    Scene&             scene,
    const Solid&       solid,
    RgbaColorF         color = {.25f, .5f, .85f, 1},
    const std::string& name  = "Solid",
    double             chord = .01);

/// @brief Load a polygon or CAD document into an owned renderable scene.
/// Supports STEP/STP, IGES/IGS, STL, mepcad, OBJ/MTL, glTF/GLB, IQM, VOX,
/// M3D and native Blender mesh import. Parametric documents are rebuilt first.
/// CAD data are tessellated for rendering; use ReadStepText/ReadIgesText to
/// retain exact topology and their detailed unsupported-entity reports.
/// @param path Existing input file; relative assets resolve beside it.
/// @param chord Positive chord tolerance for CAD imports; default .01.
/// @return Independent scene with source_path set.
/// @throws std::invalid_argument Invalid chord or unsupported extension.
/// @throws std::runtime_error Import, rebuild, unsupported CAD entities or I/O failure.
/// @par Example
/// @code
/// auto scene = dm::cad::load_scene("part.step");
/// @endcode
Scene load_scene(const std::string& path, double chord = .01);
} // namespace datamunge::cad
