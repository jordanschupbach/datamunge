#pragma once
/// @file devarm_camera.hpp
/// @brief A rigid end-link RGB camera and visible housing for the DevArm workcell.
#include <datamunge/cad/render.hpp>
#include <datamunge/sim/devarm_pick.hpp>

namespace datamunge::sim {
/// @brief Pixel bounds of the visible RGB foreground, with exclusive right/bottom edges.
/// Empty detections have zero width/height. Coordinates refer to the raw camera image.
/// @par Example
/// @code
/// dm::sim::ImageBox box; // empty
/// @endcode
struct ImageBox {
  int x{0};      ///< Leftmost foreground pixel.
  int y{0};      ///< Topmost foreground pixel.
  int width{0};  ///< Horizontal pixel extent, zero when empty.
  int height{0}; ///< Vertical pixel extent, zero when empty.
};
/// @brief Unannotated wrist RGB, a foreground box and a square padded RGB classifier crop.
/// The registered empty-scene RGB reference supplies localization; no depth, IDs,
/// fruit coordinates, semantic label or classifier prediction determines the box.
/// @par Example
/// @code
/// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
/// dm::sim::DevArmWristCamera camera(env,128);
/// auto inspected=camera.inspect(env.observation());
/// inspected.crop.save("crop.png");
/// @endcode
struct WristInspection {
  cad::RenderResult rgb;  ///< Raw sensor pixels, never annotated.
  cad::RenderResult crop; ///< 64x64 RGB by default; empty RGBA when detection is empty.
  ImageBox          box;  ///< Tight visible RGB foreground bounds before crop padding.
  std::size_t       foreground_pixels{0}; ///< Pixels differing from the registered RGB background.
};
/// @brief Fixed wrist camera with native CPU RGB rendering, matching the real URDF pose.
/// The mount is end_link local (-.09,-.075,.045) m, viewing local (.13,0,0),
/// with local +Z as image-up. These extrinsics never track fruit coordinates.
/// Intrinsics are square pixels, configurable vertical field of view, and a 2 mm
/// near plane. capture() includes real arm self-occlusion and static geometry.
/// Simple backgrounds are cached while joint/finger coordinates stay identical.
/// The attached owner's Realistic visual profile renders complete scenes with
/// shared materials and studio lighting, including cross-object shadows.
/// Validated static/robot triangles are retained while joint/finger coordinates
/// are unchanged; moving actors are always prepared from the supplied frame.
/// The realistic base-only RGB/depth/ID image is retained at that exact pose.
/// Empty actor captures return independent copies; registered references reuse
/// those same base pixels when no non-fruit objects remain. Other reference
/// actors are always rendered afresh. Caches do not alter samples or lighting.
/// The borrowed environment must outlive this single-writer camera.
/// @par Example
/// @code
/// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
/// dm::sim::DevArmWristCamera camera(env);
/// camera.capture(env.observation()).save("wrist.png");
/// @endcode
class DevArmWristCamera {
 public:
  /// @brief Attach a fixed optical mount to an existing environment's actual end link.
  /// @param environment Borrowed owner, which must outlive the camera.
  /// @param side Square RGB image size, 16..512 pixels; default 64.
  /// @param vertical_fov Finite vertical angle in degrees, 30..120; default 75.
  /// @throws std::invalid_argument Invalid image side or field of view.
  /// @par Example
  /// See the class example.
  explicit DevArmWristCamera(
      DevArmPickEnvironment& environment,
      int                    side         = 64,
      double                 vertical_fov = 75);
  /// @brief Set the CPU worker budget for subsequent camera and reference renders.
  /// Geometry caches remain reusable; this setting does not alter image samples.
  /// @param threads 0 selects the renderer default (up to eight); 1..256 caps workers.
  /// @throws std::invalid_argument Worker count outside 0..256; prior setting is unchanged.
  /// @par Example
  /// @code
  /// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
  /// dm::sim::DevArmWristCamera camera(env,256,95);
  /// camera.set_render_threads(16);
  /// auto image=camera.capture(env.observation());
  /// @endcode
  void set_render_threads(int threads);
  /// @brief Free the private background cache; does not destroy the borrowed environment.
  ~DevArmWristCamera();
  /// @brief Copying a live cache with a borrowed owner is disabled.
  /// @param other Camera that cannot be copied.
  DevArmWristCamera(const DevArmWristCamera& other) = delete;
  /// @brief Camera cache assignment is disabled.
  /// @param other Camera that cannot be assigned.
  /// @return No value; deleted operation.
  DevArmWristCamera& operator=(const DevArmWristCamera& other) = delete;
  /// @brief Compute camera extrinsics from the recorded URDF joint pose alone.
  /// @param frame Valid recorded arm pose.
  /// @return Perspective camera in CAD Z-up workcell metres.
  /// @throws std::invalid_argument Invalid recorded joint values.
  /// @par Example
  /// See the class example; pose() exposes its exact view for independent checks.
  cad::Camera pose(const PickObservation& frame) const;
  /// @brief Render the arm-mounted RGB view with real geometry and depth occlusion.
  /// Payloads use the recorded appearance, orientation and position from the
  /// current reset's class. Camera orientation depends only on actual arm joints;
  /// neither payload metadata nor classifier predictions aim the camera.
  /// @param frame Recorded pose from the current reset's fruit class.
  /// @return Owned RGBA8 pixels and metre-valued depth; alpha is opaque.
  /// @throws std::invalid_argument Invalid frame or mismatched fruit class.
  /// @throws std::runtime_error Rendering failure.
  /// @par Example
  /// See the class example.
  cad::RenderResult capture(const PickObservation& frame);
  /// @brief Render caller-owned moving actors in the registered real wrist view.
  /// Uses the same static/arm cache, lighting and depth occlusion as capture().
  /// Unlike capture(), this supports multiple fruit from a continuing world.
  /// @param frame Actual recorded URDF joint/finger pose.
  /// @param actors World-space fruit/tread actors, without arm or fixed fixtures.
  /// @return Owned unannotated camera RGB and depth.
  /// @throws std::invalid_argument Invalid recorded joint values or malformed actors/textures.
  /// @throws std::runtime_error Rendering failure.
  /// @par Example
  /// @code
  /// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
  /// dm::sim::DevArmWristCamera camera(env);
  /// auto frame=env.observation();
  /// auto rgb=camera.capture_actors(frame,env.fruit_scene(frame));
  /// @endcode
  cad::RenderResult capture_actors(const PickObservation& frame, const cad::Scene& actors);
  /// @brief Associate an RGB foreground component with a geometrically tracked target.
  /// Actual sensor RGB includes all actors. A registered empty-scene reference
  /// produces connected foreground components; the component nearest the projected
  /// focus is cropped, with nearby fragments merged. Focus association explicitly
  /// uses privileged pose, not a learned detector. Bounds use RGB differences,
  /// never object IDs/depth or class labels. No nearby visible component means
  /// an empty crop; callers must handle unknown perception without truth fallback.
  /// Off-image focus returns the full raw capture without rendering an unused
  /// reference. Realistic full/reference views can execute concurrently within
  /// the configured total worker budget; one worker retains sequential execution.
  /// A reference with no non-fruit objects reuses the exact base-only
  /// image at this pose; moving non-fruit reference objects never use that cache.
  /// @param frame Actual recorded arm pose.
  /// @param actors All world-space fruit/tread actors.
  /// @param focus Tracked target centre in CAD world metres, finite XYZ.
  /// @param crop_side Square RGB crop side, 16..256; default 64.
  /// @return Raw multi-object RGB, associated box and unannotated padded crop.
  /// @throws std::invalid_argument Invalid focus, frame or crop size.
  /// @throws std::runtime_error Rendering failure.
  /// @par Example
  /// @code
  /// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
  /// dm::sim::DevArmWristCamera camera(env,128,95);
  /// auto frame=env.observation();
  /// auto input=camera.inspect_actors(frame,env.fruit_scene(frame),frame.fruit);
  /// @endcode
  WristInspection inspect_actors(
      const PickObservation& frame,
      const cad::Scene&      actors,
      cad::Vec3d             focus,
      int                    crop_side = 64);
  /// @brief Localize visible foreground by RGB difference and crop it for classification.
  /// Compares against a rendered empty workcell at the identical arm pose, including
  /// moving tread/sensor state. Difference threshold is 12 RGB levels; fewer than
  /// nine differing pixels returns an empty box/crop. This is calibrated synthetic
  /// background subtraction, not a learned object detector. A square crop with 20%
  /// padding preserves aspect ratio; off-image samples use neutral background RGB.
  /// @param frame Recorded pose from the current fruit reset.
  /// @param crop_side Square output resolution, 16..256; default 64.
  /// @return Owned raw RGB, visible-pixel box and unannotated resized crop.
  /// @throws std::invalid_argument Invalid frame or crop side, before rendering.
  /// @throws std::runtime_error Native rendering failure.
  /// @par Example
  /// See WristInspection.
  WristInspection inspect(const PickObservation& frame, int crop_side = 64);
  /// @brief Build the visible blue camera housing, black lens and wrist bracket.
  /// @param frame Valid recorded joint pose.
  /// @return Owned world-space CAD mesh; add to the external workcell view.
  /// @throws std::invalid_argument Invalid recorded joint values.
  /// @par Example
  /// @code
  /// dm::sim::DevArmPickEnvironment env("ReBot_Arm_DM.urdf");
  /// dm::sim::DevArmWristCamera camera(env);
  /// auto housing=camera.housing(env.observation());
  /// @endcode
  cad::Scene housing(const PickObservation& frame) const;

 private:
  struct Impl;                 ///< Borrowed owner, resolution and pose-keyed render cache.
  std::unique_ptr<Impl> impl_; ///< Exclusive cache, never shared across writers.
};
} // namespace datamunge::sim
