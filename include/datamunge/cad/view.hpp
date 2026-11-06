#pragma once
/// @file view.hpp
/// @brief A thin native scene facade for code-driven CAD and generated bindings.
#include <datamunge/cad/render.hpp>
#include <datamunge/cad/robot.hpp>
#include <datamunge/cad/solid.hpp>
#include <optional>

namespace datamunge::cad {
/// @brief A small code-driven scene facade, exposed as CadScene in Python/R.
/// Owns scene/camera/render controls. All operations delegate to the canonical
/// CAD, polygon and renderer APIs; copying retains independent scene state.
/// @par Example
/// @code
/// dm::cad::View view;
/// view.add(dm::cad::Solid::box(2, 3, 4), "#3b82f6", "block");
/// view.size(640, 480); view.render("block.png");
/// @endcode
class View {
 public:
  /// @brief Create an empty scene with automatic camera framing.
  /// @par Example
  /// See the class example.
  View() = default;
  /// @brief Import any format supported by load_scene().
  /// @param path Input filename.
  /// @return Owned view with automatic framing.
  /// @throws std::invalid_argument Unsupported format.
  /// @throws std::runtime_error Import/I/O failure.
  /// @par Example
  /// @code
  /// auto view = dm::cad::View::from_file("part.step");
  /// view.render("part.png");
  /// @endcode
  static View from_file(const std::string& path);
  /// @brief Load an articulated URDF visual model and its asset files.
  /// @param path URDF file; relative meshes resolve beside it.
  /// @return Owned view at the loader's initial joint coordinates.
  /// @throws std::invalid_argument Invalid URDF/joint data.
  /// @throws std::runtime_error Import failure.
  /// @par Example
  /// @code
  /// auto view = dm::cad::View::from_urdf("robot.urdf");
  /// view.set_joint("joint2", -.6); view.render("robot.png");
  /// @endcode
  static View from_urdf(const std::string& path);
  /// @brief Set a movable independent joint on a view loaded from URDF.
  /// @param name Joint name.
  /// @param value Coordinate in radians/metres, within limits.
  /// @throws std::invalid_argument View has no robot or joint value is invalid.
  /// @par Example
  /// See from_urdf().
  void set_joint(const std::string& name, double value);
  /// @brief List independent movable joints of a URDF view.
  /// @return Owned names; empty for ordinary scenes.
  std::vector<std::string> joint_names() const;
  /// @brief Tessellate a solid and append it as a scene object.
  /// @param solid Borrowed source, unchanged.
  /// @param color Hex RGB/RGBA tint, default #4080d9.
  /// @param name Object name, default Solid.
  /// @return New object ID.
  /// @throws std::invalid_argument Invalid color.
  /// @throws std::runtime_error Tessellation failure.
  /// @par Example
  /// See the class example.
  int add(
      const Solid&       solid,
      const std::string& color = "#4080d9",
      const std::string& name  = "Solid");
  /// @brief Append a source polygon modeller primitive.
  /// @param kind cube, sphere, cylinder, cone, plane, torus or wedge (lowercase).
  /// @return New object ID; editable via scene() in C++.
  /// @throws std::invalid_argument Unknown kind.
  /// @par Example
  /// @code
  /// dm::cad::View view; view.primitive("torus"); view.render("ring.png");
  /// @endcode
  int primitive(const std::string& kind);
  /// @brief Set the output dimensions used by render() and picking.
  /// @param width Pixel width, 1..8192.
  /// @param height Pixel height, 1..8192.
  /// @throws std::invalid_argument Invalid dimensions.
  /// @par Example
  /// See the class example.
  void size(int width, int height);
  /// @brief Set regular-grid antialiasing samples per axis.
  /// @param count 1..4, i.e. 1..16 rays/pixel.
  /// @throws std::invalid_argument Invalid count.
  /// @par Example
  /// @code
  /// dm::cad::View view; view.samples(2);
  /// @endcode
  void samples(int count);
  /// @brief Select shaded, unlit, wireframe or shaded-wireframe output.
  /// @param name One of those exact lowercase names.
  /// @throws std::invalid_argument Unknown style.
  /// @par Example
  /// @code
  /// dm::cad::View view; view.style("shaded-wireframe");
  /// @endcode
  void style(const std::string& name);
  /// @brief Enable/disable shadows cast by the first visible light.
  /// @param enabled True enables triangle-ray shadow tests.
  /// @par Example
  /// @code
  /// dm::cad::View view; view.shadows(false);
  /// @endcode
  void shadows(bool enabled);
  /// @brief Set explicit camera position and target, with +Z as up.
  /// @param x Camera X position.
  /// @param y Camera Y position.
  /// @param z Camera Z position.
  /// @param target_x Target X.
  /// @param target_y Target Y.
  /// @param target_z Target Z; all values finite, target distinct and view not parallel to +Z.
  /// @throws std::invalid_argument Invalid camera.
  /// @par Example
  /// @code
  /// dm::cad::View view; view.camera(6,-8,5,0,0,0);
  /// @endcode
  void camera(double x, double y, double z, double target_x, double target_y, double target_z);
  /// @brief Resume automatic perspective framing after an explicit camera.
  /// @par Example
  /// @code
  /// dm::cad::View view; view.fit();
  /// @endcode
  void fit();
  /// @brief Add a directional light; the first explicit light replaces the fallback light.
  /// @param x Direction X from surface toward light.
  /// @param y Direction Y.
  /// @param z Direction Z; finite nonzero vector.
  /// @param color Hex RGB/RGBA light color, default white; alpha unused.
  /// @param intensity Nonnegative finite multiplier, default 1.
  /// @return New light ID.
  /// @throws std::invalid_argument Invalid direction, color or intensity.
  /// @par Example
  /// @code
  /// dm::cad::View view; view.light(1,-2,3,"#ffffff",1.2);
  /// @endcode
  int light(
      double             x,
      double             y,
      double             z,
      const std::string& color     = "#ffffff",
      double             intensity = 1);
  /// @brief Render synchronously and save a PNG/JPEG/PPM color image.
  /// @param path Output filename, parent directory must exist.
  /// @throws std::invalid_argument Invalid scene or extension.
  /// @throws std::runtime_error File I/O failure.
  /// @par Example
  /// See the class example.
  void render(const std::string& path) const;
  /// @brief Save the scene as self-contained glTF with meshes, textures and lights.
  /// @param path Output .gltf filename; parent must exist.
  /// @throws std::runtime_error Export/I/O failure.
  /// @par Example
  /// @code
  /// dm::cad::View view; view.primitive("cube"); view.save_gltf("cube.gltf");
  /// @endcode
  void save_gltf(const std::string& path) const;
  /// @brief Pick a source object using the current camera and image dimensions.
  /// @param x Pixel X from left; continuous coordinates.
  /// @param y Pixel Y from top.
  /// @return Object ID or -1 for background.
  /// @throws std::invalid_argument Invalid ray or geometry.
  /// @par Example
  /// @code
  /// dm::cad::View view; view.primitive("cube"); int id=view.pick_object(400,300);
  /// @endcode
  int pick_object(double x, double y) const;
  /// @brief Count scene objects, including empty groups.
  /// @return Object count.
  int object_count() const;
  /// @brief Count triangles over all object mesh references, including hidden objects.
  /// @return Triangle count.
  int triangle_count() const;
  /// @brief Borrow the mutable scene for advanced C++ editing.
  /// @return Reference owned by this view; URDF set_joint replaces it.
  Scene& scene() { return scene_; }

 private:
  Scene                scene_;           ///< Owned polygon scene.
  Camera               camera_;          ///< Explicit camera; automatic framing ignores this value.
  RenderOptions        options_;         ///< Output/shading controls.
  bool                 auto_fit_ = true; ///< Select automatic camera framing at rendering time.
  std::optional<Robot> robot_;           ///< Source URDF state when loaded from URDF.
  Camera               current_camera() const; ///< Resolve framing at the current output aspect.
};
} // namespace datamunge::cad
