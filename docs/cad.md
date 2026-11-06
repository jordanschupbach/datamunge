# CAD and 3D modelling in code

The reusable CAD kernel and polygon modeller from mep are now part of the
canonical C++ library under `dm::cad`. Include `<dm/dm.hpp>` or the narrower
`<datamunge/cad/cad.hpp>` and link datamunge normally. Modelling, file imports
and offline rendering need no editor, window or graphics context.

```cpp
#include <dm/dm.hpp>

int main() {
  auto block = dm::cad::Solid::box(4, 3, 2);
  auto hole = dm::cad::Solid::cylinder(.4, 4).translated(2, 1.5, -1);
  auto part = block.subtracted(hole);
  part.save("part.step");

  dm::cad::View view;
  view.add(part, "#4080d9", "Drilled block");
  view.size(800, 600);
  view.samples(2);
  view.render("part.png");
}
```

`Solid` owns a B-rep body. Operations return new values and preserve inputs;
`model()` and `body()` give access to the complete lower-level API. Use
`edges()` / `faces()` to select fillets, chamfers and shell openings. The kernel
also exposes sketches, constraints, feature trees with rebuilds, extrusion,
revolve, loft, sweep, persistent naming, patterns, assembly mating, analytic
geometry, NURBS curves/surfaces, intersections, classification, validation and
mass integration. The source files and the actual imported revision/hashes
are listed in [cad-port-manifest.json](cad-port-manifest.json).

Python uses `from pydatamunge import datamunge as dm` and the native
`dm.CadSolid` / `dm.CadScene` facade. R uses `CadSolid_box(...)`, `CadScene()`
and the usual `$method(...)` calls; `?native-cad` documents their contracts.
The complete precise kernel remains available directly in C++; the language
facades share its algorithms rather than implementing alternate solvers.

## Seeed DevArm working example

The same real arm is available in both episodic and continuing native physics
workcells. Run `just playground` for the continuing learning/video example.
The [episodic report](mepml/examples/robot_fruit_episodic.mepml) preserves the
arrival, red-bin Q-learning and varied-fruit MobileNet examples. The main
[playground](mepml/examples/playground.mepml) now introduces
`dm::sim::FruitStreamEnvironment`: an unbounded-time, multi-fruit workcell with
independent Poisson arrivals and a continuously moving belt. Rate-limited real
URDF servos keep their joint state between picks. Fruit crossing the pickup
window is counted as missed; capacity overflow is counted rather than delayed,
and bins clear completed items to bound retained state.

The installed C++ `dm::rl` backend now provides masked Q(λ), Double DQN,
dueling Double DQN and clipped categorical PPO through
`rl::train_discrete`. All share owned observations, actual elapsed-time discounts,
terminal/time-limit semantics and strict inference NPZ checkpoints. Neural
methods use native tensors, autograd and Adam. `FruitStreamEnvironment::learning_observation()`
supplies 25 bounded geometry/mask features plus the existing tabular state;
`sim::stream_learning_environment()` borrows the workcell as reset/step callbacks
with a deterministically seeded arrival-rate curriculum. The legacy
`train_fruit_stream` and text policies remain available.

The playground first trains each fresh pipeline for 2,048 independent windows
of at least 120 seconds, using matched reset seeds and the same proxy, reward,
motors and frozen MobileNet. Twelve full-physics streams screen checkpoints;
the top two per method receive six separate actual-RGB validation streams.
Validation score (5 correct − 8 wrong − 6 missed) selects the method/checkpoint
before sixteen independent actual-RGB test streams are run. The prior trained
controller is a separately labelled incumbent with a larger historical budget.
This is one deterministic training seed per method and compares tabular versus
continuous-feature representations as well as learning rules; it is not a broad
hyperparameter search or a universal algorithm ranking.

All inference/checkpoint histories, per-stream outcomes, class predictions and
selection hashes are exported by the complete editable C++ report. The same
frozen vision classifier is used for every candidate; no classifier is retrained
by this comparison. A fixed-seed minute supplies the demonstration. The default
video renders a contiguous 20-second excerpt (world time 40–60 seconds), while
all 60 seconds remain simulated and recorded.
The video enables `FruitStreamOptions::watch_infeed`: while no eligible fruit is
waiting, the arm returns its rigid wrist camera toward the randomized belt-entry
area. Guarded waypoints and bounded joint/finger motion share each 0.125-second
wait, so the next pick can interrupt the return. The optional setting defaults
to false for existing callers. `synchronized_grasp` aligns wrist roll before
inspection/descent and selects a guarded short approach from a bin. It also
coordinates idle return segments so the training proxy can batch their ticks
with the same servo endpoints. Motor speed limits and belt speed are unchanged.
`queued_wait_cost`, `miss_cost` and `wrong_cost` configure reward costs in 0..100;
their defaults preserve the earlier criterion. `StreamTrainingOptions` exposes
`trace_decay` and `mask_actions`, and `StreamPolicy::action(state, mask)` applies
the same mask at deployment. Masked policies must also use that overload after
loading; the unchanged five-action checkpoint format does not store a mask or
motor/reward profile. The report records the common configuration separately. Arrival
awareness still uses tracked simulator geometry, rather than a learned detector.
RGB component boxes are associated with a geometrically tracked target; class
predictions alone route to bins. The motor state and box association remain
privileged geometry, and grasps/servos/collision guards remain idealized.
`just playground-preview` reloads frozen checkpoints and regenerates the preview.
Set `DATAMUNGE_VIDEO_START=0 DATAMUNGE_VIDEO_SECONDS=60` before that command to
render the full minute, or choose another integer-second window within it.
The report also links the preserved full-minute rendering.

The current playground enables `WorkcellVisualStyle::Realistic` for both its
external view and its wrist camera. Procedural assets add rounded manufactured
edges, brushed aluminium, textured rubber, moulded bin details, hardware and a
factory floor, plus botanical fruit geometry with natural overlapping skin hues,
pores, fibres and roughness maps. The opt-in CPU material workflow supplies
linear-light microfacet shading, soft key shadows, local occlusion and filmic
output. Whole-scene rendering lets arm/fruit shadows reach their surroundings.
Its perspective overview and 256-pixel wrist images are rebuilt and vision is
adapted on the new appearance; physics and actuator limits are unchanged.
The prior continuing report remains in
[`robot_fruit_continuing_simple.mepml`](mepml/examples/robot_fruit_continuing_simple.mepml).

`dm::cad::PreparedScene` retains an owned immutable snapshot of validated,
transformed geometry. Reuse it for multiple cameras or append a small transient
actor scene to each render. Actor rendering shares the large base triangles and
rebuilds the combined hierarchy in original flattened order, preserving color,
depth, object IDs and equal-distance ties. Base-only views initialize their BVH
once. Camera caches use exact joint/finger values and keep moving actors outside
the cached base; complete-scene shadows remain enabled.

`dm::cad::CudaRenderer` provides explicit CUDA offscreen rendering of a
`PreparedScene` plus transient actors. Build with `DATAMUNGE_ENABLE_CUDA=ON`
or use the existing `datamunge-cuda` Nix package. Normal builds consume embedded
PTX and need no CUDA SDK; execution needs an available NVIDIA driver/device.
The renderer owns a context/program, retains the last base snapshot and device
uploads, and reuses allocation capacities. It serializes concurrent calls on one
instance. Primary rays, mapped materials, transparency, wire overlays, soft
shadows, AO and synchronous RGBA/depth/ID readback run on the GPU. World geometry
and combined BVH construction remain on the CPU. Float32 arithmetic and a larger
self-intersection bias can differ at edges from the double CPU reference.
Explicit CUDA never silently falls back; the application can choose CPU rendering.

Use `just playground-gpu` for a frozen-checkpoint **20-second** video at world
time 40–60 seconds. It builds the CUDA-enabled package and supplies its pkg-config
path to the report. The report uses CUDA for the external view and keeps wrist
RGB/boxes/classification on the CPU reference. Set `DATAMUNGE_VIDEO_START` and
`DATAMUNGE_VIDEO_SECONDS` to override the clip, and
`DATAMUNGE_RENDER_BACKEND=auto|cpu|cuda` to select the external-view path in normal
playground builds. Auto uses CPU if CUDA cannot be probed; explicit CUDA reports
unavailability. `just playground-gpu` explicitly requests CUDA. Backend/device
and kernel/source hashes are recorded with timings. Default CPU builds remain
portable. The CUDA recipe requires a supported Linux/NVIDIA host.

For performance work, use `just bench-fruit-render-gpu --repetitions 1` or the
default three repetitions. It compares **four still poses**, alternating CPU and
CUDA, and records upload/initialization costs plus RGB/depth/ID errors. It does
not encode a video, train models, or render a full minute. Regenerate the embedded
CAD kernel with `just prebuild-cad-cuda-kernels` after changing its source/layout.

The CPU wrist camera retains exact base-only RGB/depth/ID buffers at an unchanged
joint/finger pose and returns owned copies for genuinely empty actors. Inspection
still renders moving non-fruit reference objects afresh. Its two realistic views
can execute concurrently while sharing the configured total worker budget;
`set_render_threads(1)` keeps them sequential. Off-image focus returns the full
raw capture without an unused reference render. Use `just bench-fruit-wrist` for
bounded paired comparisons of raw buffers, crops and boxes.

`dm::cad::VideoWriter` sends RGBA frames directly to a bounded ffmpeg input pipe.
Call `append(frame)` for each frame and `finish()` to publish a completed MP4;
the previous destination is preserved on child failure or abandonment. ffmpeg
with libx264 must be on PATH. The writer uses native process arguments, permits
filenames with spaces, and handles a failed child without changing the caller's
SIGPIPE handler. The Linux implementation is exercised by native tests; the
Windows process branch has not been runtime-tested here.

The playground saves stage timings and input/settings hashes. It normally writes
only the MP4 and representative stills; `DATAMUNGE_STREAM_KEEP_FRAMES=1` also
keeps all PNGs. `DATAMUNGE_RENDER_THREADS=8` overrides its worker budget. The
benchmark and implementation record are in
[`PERFORMANCE.md`](../PERFORMANCE.md#robot-workcell-rendering-2026-11-01).


The [Seeed DevArm report](mepml/examples/cad/seeed_devarm.mepml) contains a complete workflow
in each C++/Python/R tab: download/verify, import, pose, export glTF, reopen it
and render the saved models. Each program obtains both real inputs from
[Seeed-Projects/reBot-DevArm](https://github.com/Seeed-Projects/reBot-DevArm):

- The B601-DM arm, assembled from its URDF and 34 colored STL visual meshes.
  The example poses revolute joints and the gripper, including its mimic
  relation: 430,912 triangles and seven independent movable joints.
- `03_Link1.step`, imported through the precise STEP reader and tessellated
  into 12,125 render triangles. Its render mesh is also exported as glTF.

The native `dm::datasets::fetch("rebot_devarm")` call downloads and verifies
all 38 pinned files, including cached files, by SHA-256 and size. It uses the
usual dataset cache/offline/mirror controls and is exposed as `fetch` in
Python/R. The report passes an explicit repository-local cache directory;
no separate downloader command is needed there. Generated glTF files contain
embedded mesh data; both rendered views come from reopening those exports.

```sh
just cad-assets        # Download and hash-check the pinned assets once.
just cad-rebot         # Compile and run the complete C++ example.
just test-cad          # Windowless native regression and integration tests.
```

The editable standalone program is
[rebot_arm.cpp](../examples/cpp/source/cad/rebot_arm.cpp). It accepts optional
asset-root and output-directory arguments. Outputs go to `build/cad-demo/` by
default. The manifest pins revision `922c5fa3d8c8919db1847b49166f02fc26c8fdad`
and verifies SHA-256 and byte size for every downloaded file; the approximately
56 MB cache remains under ignored `build/cad-assets/rebot`. Verify an existing
cache offline with `python3 scripts/fetch_cad_example.py --check`.

The source hardware is © 2026 Seeed Studio Co., Ltd., CERN-OHL-W-2.0. Its
original license is downloaded beside the files. The rendered arm uses its
published geometry, material colors and URDF placement data. URDF coordinates
use metres/radians; the independent STEP part uses millimetres. They are
rendered in separate scenes, so those units are not accidentally mixed.

## File interfaces

| Format | Read | Write | Representation |
| --- | --- | --- | --- |
| STEP / STP | `ReadStepText`, `load_scene` | `Solid::save`, `WriteStepShapes` | Supported analytic/NURBS B-rep geometry/topology |
| IGES / IGS | `ReadIgesText`, `load_scene` | — | Supported B-rep entities; detailed unsupported report |
| DXF | `ReadDxf` | `WriteDxf` | Sketch geometry, without constraints |
| mepcad / mepsketch | `ReadCadDocument`, `ReadSketch` | `WriteCadDocument`, `WriteSketch` | Parametric features/sketches; rebuild after loading |
| STL | `ReadStl`, `load_scene` | `Solid::save`, `WriteStl` | Binary or ASCII triangle meshes |
| OBJ / MTL | `load_scene` | `Solid::save`, `WriteObj` | Polygon geometry, source OBJ material scope |
| glTF / GLB | `load_scene` | `SaveModel3DGltf`, `View::save_gltf`, `WriteGltf` | Triangles, node placements, materials, textures and lights; glTF output |
| IQM / VOX / M3D | `load_scene` | — | Source modeller's static mesh import |
| Blender | `LoadBlendFile`, `load_scene` | — | Native SDNA mesh/material import; no Blender executable needed |
| SVG drawings | — | `MakeDrawingView`, `DrawingViewToSvg`, `Solid::save` | Sampled visible/hidden edges and silhouettes |
| PNG / JPEG / PPM | Texture codecs where supported | `render(...).save`, `View::render` | Offline color images |
| URDF | `Robot::load_urdf`, `View::from_urdf` | — | Visual link tree and forward kinematics |

`load_scene()` tessellates exact CAD to render meshes. Use the underlying
STEP/IGES readers when you need to retain exact surfaces or inspect partial
import diagnostics. The facade rejects incomplete CAD imports. Native mepcad
serialization is intentionally compatible with mep's versioned document format.
A mesh export does not preserve exact surfaces, sketch constraints or feature
history. Save those through their precise/native formats instead.

## Rendering and code-driven scenes

`Scene`, `MeshData`, `Object3D`, `TextureData` and `Light` retain the source
modeller's owned data model and mesh editing operations, including vertex
removal/merge, normal recalculation, face subdivision/extrusion, dissolve,
face creation, distance welding and normal flipping. Primitive meshes,
lathe surfaces and custom indexed meshes are available directly. Object
transforms use XYZ Euler degrees; CAD transforms use radians. Object `parent`
is source-compatible grouping metadata, while URDF link transforms are
computed explicitly by `Robot`.

`render(scene, camera, options)` returns owned RGBA8 pixels, centre-ray depth
and source object-ID buffers. It uses a triangle BVH, perspective/orthographic
cameras, regular-grid antialiasing, alpha compositing, directional/point
lights, normal maps, roughness/metallic textures, optional triangle edges and
first-light shadows. Material shading retains the source's simplified
ambient/diffuse/specular formula; it is not a physically based path tracer.
Shadows are tested with geometry rays rather than the editor's GPU shadow
map. `pick()` exposes the same geometry queries. Camera and output options
are values; independent const renders may run concurrently.

To animate, change object transforms or `Robot::set_joint`, generate a scene,
and render another frame. `Robot` supports fixed, revolute, continuous,
prismatic and mimic joints, visual origins, mesh scales and named materials.
ROS `package://` assets resolve through an explicit package root or ancestors.
Expand Xacro before importing; this visual API does not evaluate robot
inertial, collision, control or dynamics descriptions.

The robot-sorting example uses `sim::workcell_presentation(scene, surroundings,
cable_anchors)` to create an owned exterior scene with finer fruit peel maps,
concrete, rubber, plastic, bare steel and painted-metal finishes. Optional
surroundings add a factory backdrop, service cabinet, roller bearings and fixture
hardware. World-space robot link anchors drive a cosmetic flexible servo harness.
Apply this function once to each fresh scene component. It preserves source
meshes/transforms, consumes no simulation random numbers and creates no physics
bodies. Keep the original scene for the calibrated wrist camera and classifier.

`sim::workcell_presentation_options()` selects 2x2 primary samples, six shadow
samples, three local-occlusion samples and `studio_environment=0.55`. The optional
`cad::RenderOptions::studio_environment` field accepts strengths from zero to four;
zero preserves existing ambient shading. CPU and CUDA use matching analytic
sky/ground and broad softbox illumination with roughness-dependent reflections.
This approximates studio lighting; it does not trace scene reflections or indirect
light bounces. The appended options field requires rebuilding native consumers.
The playground retains its 20-second default video and uses the CPU wrist sensor
with its original lighting, sample counts and fruit textures.

## Scope inherited from the source kernel

This is a port of mep's implemented modeller, rather than a replacement for a
commercial CAD kernel. It retains its explicit diagnostics and supported
operation sets: general coincident-face booleans and some disconnected
results are refused; straight-edge planar-face fillets/chamfers and the
supported shell/draft surfaces are covered by the source tests. STEP/IGES
readers map the entity sets in their headers, rather than every ISO entity or
assembly representation. The complete robot example therefore uses its
published URDF link assembly, with a separate exact STEP part example.

Curved boundaries are tessellated to a tolerance. Imported, trimmed parts can
have imperfect mesh edge conformity; inspect `ValidateBody` and
`TessellationMesh::IsClosed()` before treating a render mesh as a manufacturing
or simulation solid. `Solid::volume()` integrates the tessellated trimmed
boundary, avoiding the untrimmed-surface mass integration trap documented in
`mass.hpp`. `ComputeBodyMass()` also returns trimmed-boundary area, centroid
and centroidal inertia, together with the mesh conformity flag.
Drawing hidden-line transitions are sampled. Native Blender
import retains the source's static-mesh scope and does not evaluate modifiers,
rigging or animation; its explicit Blender conversion helpers remain available.

## Validation and provenance

Twenty-one native CTest suites pass, including all 19 source CAD/polygon/Blender
suites and new API/format checks for solid ownership, boolean volumes,
IQM/VOX/M3D/GLB decoding, file round trips, render colors/depth/picking, antialiasing, deterministic
threading, textures, transparency, URDF forward kinematics, mimic coordinates
and rejected inputs. The actual DevArm STEP file exposed and verified fixes
for simple B-spline entity attribute offsets and projection near a cone apex
with a wide imported parameter domain. A separate glTF translation oracle
verifies explicit matrix placement. Four API/format/polygon/Blender suites pass
AddressSanitizer and UndefinedBehaviorSanitizer with leak detection disabled.
Python/R binding tests cover owned geometry, errors, import/export, rendering
and URDF joints. An independent CMake consumer builds against the installed
library and verifies mass properties and rendering.

`tests/cad/CMakeLists.txt` runs the actual CAD sources in isolation without
optional data integrations. They also build in the normal shared library.
Floating point contraction is disabled for CAD sources as in mep, preserving
its exact predicates and tolerance choices. The MIT notice ships in source,
native installs and Python/R packages. Blender fixtures retain their source
attribution and license files. Build evidence is under `build/cad-port/`.

## Articulated sorting physics

The playground explicitly selects datamunge's in-house articulated solver for both
training and evaluation; the standard CPU and CUDA packages do not link MuJoCo.
`UrdfModel::load()` imports full COM inertias, fixed/hinge/slide joints, drive
limits, collision primitives/STL hulls and affine mimic gears into `sim::World`.
The native engine now supports compound/ellipsoid/hull contacts, implicit bounded
motors, reflected rotor inertia, full-chain bilateral projection and rolling/spin
friction. Fruit is free throughout contact-driven grasps and releases.

Native and MuJoCo use distinct equations/solvers; hardware parameters remain
nominal. Native contact is discrete at 960 Hz in the sorting example. See the
[native physics notes](native-physics.md) for supported features, approximation
limits and reproducible checks. The earlier MuJoCo results remain labelled in
`robot_fruit_mujoco.mepml`. An optional reference build is available as
`nix build .#datamunge-mujoco` (or `DATAMUNGE_ENABLE_MUJOCO=ON` in CMake).
