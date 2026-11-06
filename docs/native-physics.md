# Native articulated physics

The previous sorting benchmark used MuJoCo. The sorting environment now supports
an independent in-house C++ implementation in `datamunge::sim::World`.
`FruitStreamOptions::physics_backend` defaults to `ArticulatedBackend::Native`
when articulated physics is enabled. The standard Nix CPU/CUDA packages have no
MuJoCo dependency. There is no runtime substitution between engines.

```cpp
#include <dm/dm.hpp>
dm::sim::FruitStreamOptions options;
options.articulated_physics = true;
options.physics_backend = dm::sim::ArticulatedBackend::Native;
options.physics_timestep = 1. / 960;
// Supply the complete downloaded Seeed URDF, including its collision meshes.
dm::sim::FruitStreamEnvironment environment("ReBot_Arm_DM.urdf", options);
auto frame = environment.reset(42);
```

The reusable URDF importer is independent of the sorting task and renderer:

```cpp
auto model = dm::sim::UrdfModel::load("robot.urdf");
dm::sim::WorldOptions controls;
controls.gravity = {0, 0, -9.81}; // URDF model is Z-up; World defaults to Y-up.
controls.gyroscopic = true;
controls.block_joints = true;
controls.sleeping = false;
controls.clamp_velocities = false;
dm::sim::World world(controls);
auto robot = model.instantiate(world);
```

Missing dynamic-link inertias are rejected unless the caller explicitly supplies
nominal fallback parameters in `UrdfPhysicsOptions`. The Seeed adapter explicitly
supplies nominal finger inertias, motor armatures and damping; these are not
measurements inferred from the visual CAD. Position is a body's reference origin;
linear velocity is its world COM velocity. Imported tensors are expressed at the
COM in the link frame, including authored inertial-frame rotations.

| Capability used by the sorting task | Native implementation |
| --- | --- |
| Full inertia/COM and coupled links | Full 3×3 tensors; hinges, sliders and fixed joints |
| Reflected drive inertia | Dynamically coupled auxiliary rotor rates |
| Bounded actuation | Implicit velocity servos, authored effort/requested-rate limits, gravity compensation |
| Coupled gripper | Affine mimic gears, including common-parent inverse-mass terms |
| Robot collision | STL support hulls; 32-sided cylinders; compound primitives/hulls |
| Fruit collision | Ellipsoid apples; five tapered capsule sections per banana |
| Conveyor | Tangential contact surface velocity; no assigned fruit velocity |
| Friction | Coulomb sliding disk, separate bounded spin/rolling torque rows |
| Grasp/release/scoring | Measured opposing jaw loads, free-body release, actual bin-floor support and dwell |
| Free asymmetric rotation | Implicit midpoint Euler equations with matching Cayley rotation |
| Determinism/failure | Fixed steps/iterations, owned worlds, snapshots including controls/gears/rotors, rollback on solver failure |

The coupled bilateral option factors a pose-local dense effective-mass matrix
(up to 256 independent rows), including reflected rotor constraints. Motor and
limit rows use its Schur-complement inverse inertia. Redundant/singular bilateral
systems reject the step in this mode; the default sequential solver remains
available for other worlds. The sorting adapter uses this coupled option and
12 contact/actuator sweeps per solve. Contacts are integrated discretely at 960 Hz
in the playground; general World sweeps also account for compound offsets and
COM-centered rotation when continuous collision is enabled.

This is a distinct maximal-coordinate solver. It does not implement MuJoCo's
reduced-coordinate equations, soft contact law, full implicit integrator, elliptic
six-dimensional friction cones or every MuJoCo feature. Sliding and angular
friction bounds are separate. Convex mesh sampling approximates geometry;
concavities need explicit compounds or clipped slabs. Continuous hinge positions
wrap into one revolution. Public joint limits must contain the authored zero
coordinate. `package://` paths must be resolved before import; collision mesh
input is STL. Bodies and snapshots borrow immutable shared collision geometry.

Rigid contact does not model fruit deformation/damage, soft pads, cable forces,
motor electronics, latency or camera noise. Contact parameters and servo gains
still need hardware calibration. Conservation and numerical checks establish
solver properties, not agreement with a physical arm.

## Verification and examples

`tests/cpp/sim_native_dynamics_tests.cpp` supplies analytic checks for energy and
world momentum at both moderate and rapid asymmetric spin, COM transport, bounded
motors/reflected inertia, loaded coupled chains, mimic load sharing, gravity
compensation, fixed joints, hull witnesses/penetration, compound rays/sweeps,
ellipsoid settling/support force, strict URDF import and rollback, snapshots and
a full native conveyor/bin workcell. They run with MuJoCo disabled. The optional
MuJoCo regression suite remains separate and skips when its SDK is absent.

The `robot_fruit_native_port.mepml` report runs native physics and actual wrist
RGB. It shows that throughput is limited by the fixed grasp skill and pick cycle,
not the scheduler, and enables the opt-in `grasp_alignment_tolerance` (20 mm) and
`hold_grasp_target` options, which raise held-out correct sorts by about 40%.
Its 60-second video uses a greedy scheduler. `playground.mepml` contains a **fresh native
training/comparison** with separate cache/provenance and a default 20-second
selected-policy video. Its previous MuJoCo scores are preserved exclusively in
`robot_fruit_mujoco.mepml`; they are not native results. A completed fresh native
256-window comparison is not claimed by this port.

Build the optional reference with `nix build .#datamunge-mujoco`, or configure
CMake with `-DDATAMUNGE_ENABLE_MUJOCO=ON` and available MuJoCo headers/library.
`articulated_physics_version(ArticulatedBackend::Native)` identifies the native
engine. The legacy no-argument version query retains its optional-MuJoCo meaning.

Algorithm references: [Gilbert, Johnson and Keerthi (1988)](https://doi.org/10.1109/56.2083),
[van den Bergen's EPA paper](https://graphics.stanford.edu/courses/cs468-01-fall/Papers/van-den-bergen.pdf),
and [Simo and Wong (1991)](https://doi.org/10.1002/nme.1620310103). The native
implementation enumerates simplex faces for the GJK closest-point subproblem.
