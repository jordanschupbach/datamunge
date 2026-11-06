# Robot sorting physics audit

The original continuing task was useful for scheduler development, but its
kinematic grasp and analytic training transport could not establish physical
pickup performance. The playground now opts into an articulated MuJoCo backend
for training, validation, test and rendering. It remains a nominal model requiring
hardware identification before results can be interpreted as robot performance.

## Findings and changes

| Previous mechanism | Contact-backed experiment |
| --- | --- |
| Interpolated joint coordinates | Coupled articulated dynamics with finite actuator effort |
| Fruit latched before fingers close | Sustained bilateral finger contact before recording a grasp |
| Payload teleported with the tool | Free rigid body throughout; frictional support can fail |
| Release velocity set to zero | Linear and angular momentum retained at release |
| Fruit rotation locked | Free rotation, compound shape and computed inertia |
| Ray-query support sufficient for scoring | Upward bin-floor contact force and continuous low-speed dwell |
| Analytic conveyor/settling during training | Same articulated contact backend in all control phases |
| Render-only URDF geometry | Authored link masses, centers of mass, full inertia tensors and collision assets |

`sim::World` remains the lightweight native primitive rigid-body engine for its
existing users. It is not silently replaced. `FruitStreamOptions::articulated_physics`
selects the optional backend; incompatible analytic proxy mode or an unavailable
MuJoCo build throws. `articulated_physics_version()` identifies the linked engine.
CMake enables it with `DATAMUNGE_ENABLE_MUJOCO=ON`; the Nix core packages enable it.
Recorded structs and the robot object gained pose/diagnostic storage, so native
consumers and bindings must be rebuilt against matching headers and library.
The Nix and playground recipes build matching artifacts.

## Model and nominal parameters

The asset manifest pins Seeed revision
`922c5fa3d8c8919db1847b49166f02fc26c8fdad`, including ten collision meshes,
file sizes, SHA-256 digests and CERN-OHL-W-2.0 notices. The rigid robot root uses
the same transform as the visible assembly. MuJoCo and CAD forward kinematics
are independently compared in the native tests. Decorative cables and factory
scenery remain visual assets; only the listed hulls and fixture boxes collide.
The fingers render their symmetric measured aperture while dynamics retains both
slide coordinates and a soft gearing constraint. Measured encoder coordinates are
rendered without clipping them to commanded joint limits.

The model reads URDF joint origins, axes, position/effort limits, masses, center
of mass and full inertia tensors, including inertial-frame rotations. Adjacent
links and the geared finger pair are excluded from self-contact; other robot
links, fixtures and fruit can collide. Arm collision meshes are convex hulls.
Each concave finger is decomposed into a carriage and five pad sections. These
are collision approximations, not exact surface contact geometry.

Apples are ellipsoids; bananas are five tapered capsule segments. Prototype,
scale and orientation affect collision shape. Nominal masses are 0.16 kg and
0.12 kg times scale cubed. Ellipsoid inertia is analytic; banana inertia sums
volume-weighted capsule components with parallel-axis terms. Overlapping capsule
components are a mass-distribution approximation. Nominal sliding friction is
0.8, torsional friction 0.002 m and rolling friction 0.004 m (robot geoms use
0.001 m rolling friction). No fruit deformation or damage is modeled.

The task clock remains 240 Hz; the playground uses 960 Hz internal integration.
MuJoCo uses full implicit-in-velocity integration, elliptic six-dimensional contact cones,
sparse Newton constraints, 80 iterations, tolerance 1e-9, contact solref 0.004 s
with damping ratio 1, solimp 0.95/0.99/0.001 and a 0.1 mm contact margin.
New arrivals keep their Poisson timing and horizontal sampling; when their
initial bounds overlap an existing fruit, they start above it with 2 mm
clearance and subsequently fall under gravity. Existing bodies are untouched.
The stationary belt support supplies moving-material tangential contact
references at 0.22 m/s. Fruit is not assigned that speed independently of contact.

A force-limited velocity servo tracks bounded position references, with joint
velocity feedback gains 20/4 and finger gain 80. Gravity/Coriolis compensation
is included within the same actuator effort clip. Position gains are 9/12.5
for arm joints and 10 for fingers. Requested rates are capped at 1.5 rad/s and
0.08 m/s; actual rates can exceed those requests under collision or backdriving.
Effort limits come from the pinned URDF (27/7 Nm arms, 8 N per modeled finger
drive). Both finger slides have drives plus a symmetric gear constraint; the
actual gripper motor/transmission and force allocation have not been identified.
This is a nominal symmetric-jaw model, not an identified single-motor drive. Reflected
inertia is nominal 0.002 kg m² for arm joints and 1 kg for finger slides. Missing
finger inertias use 0.04 kg and diagonal 0.00002 kg m². Damping is nominal
0.1 for arms and 0.5 for fingers. These values are not fitted motor measurements.

A grasp requires at least 2 N normal force on each finger for 0.1 s. The payload
remains free and a sustained loss of bilateral support records a dropped grasp.
A sort needs actual upward bin-floor force, linear speed below 0.03 m/s,
angular speed below 0.2 rad/s, and 0.2 s continuous dwell. Completed fruit clears
after two seconds. The eight-body capacity and bin clearing are benchmark
protocol choices, not an implementation of physical bin emptying.

The local development build used MuJoCo 3.3.7; the installed packages used for
reported RL training, evaluation and rendering use **3.4.0**. Each native build
checks that its linked runtime matches its compile-time header version.

## Verification and interpretation

A separate sweep of 1,024 sampled single-fruit arrivals found zero initial
penetrations into an empty belt; the close-arrival unit check covers clearance
from an already present fruit.

Native tests check analytic free flight, linear momentum, angular motion,
timestep refinement, traction, force-based support, invariant state while
spawning, reset/reuse and agreement between dynamics and rendered link poses.
Fast-spinning asymmetric fruit exposed an instability in implicit-fast
integration. Full implicit-in-velocity integration includes gyroscopic derivatives
and a dedicated tumbling-body regression check covers this failure. Any MuJoCo
warning rejects the run; automatic numerical reset is disabled.
A separate perfect-perception development stream compares 240, 480, 960 and
1920 Hz, exporting transport errors, maximum penetration, effort usage and
actual joint/finger rates. Contact outcomes need not converge monotonically;
matching aggregate outcomes is weaker evidence than trajectory agreement.
MuJoCo's default contact safeguard enforces a time constant at least twice the
timestep, so the coarsest runs can also change effective contact softness. This
is a practical numerical sensitivity study, not a fixed-equation convergence
proof. Contacts are detected at discrete steps; this adapter does not add
continuous collision detection. These checks therefore do not exclude rare
high-speed tunneling, which needs impact-specific validation or smaller steps.

The frozen camera adaptation uses contact-driven RGB from the local 3.3.7
development plant, collected before the final arrival-clearance guard. Its source/model fingerprints are preserved in the audit;
this is an explicitly fixed perception input, not a claim of identically generated
training imagery under every later physics revision. The final executable can
refit vision; `DATAMUNGE_STREAM_REUSE_VISION=1` preserves the cached checkpoint
for the recorded comparison.

A separate two-second torque-free ellipsoid check quantifies numerical damping.
At 960 Hz, energy retention is 99.918% for initial angular velocity (3,4,2)
rad/s, 97.993% for (15,20,10), and only 49.919% for the extreme (150,200,100)
stress case. At 1920 Hz those values are 99.959%, 98.986% and 55.848%.
Full implicit integration stabilizes that extreme case but does not make its
trajectory accurate. Rapid-tumbling impact predictions remain outside the
validated regime; they need a suitable higher-order/energy-preserving treatment
and further timestep validation. Solver stability alone is not accuracy evidence.
The raw sweep is retained in the physics-fidelity benchmark record.

Fresh Q(λ), Double DQN, dueling Double DQN and PPO controllers each train 256
independent 120-second windows on this physical backend. Synthetic perception
errors during training are calibrated on validation camera crops. Independent
actual-RGB streams select checkpoints and methods; separate test streams report
outcomes without reselection. The old proxy-trained policy is a transfer
reference with a different historical budget, eligible for deployment if it wins
independent validation. The report exports per-stream results and
physical diagnostics; a rendered demonstration is illustrative evidence only.
The closed-interval trajectory includes time zero through 60 s (901 states);
its fruit log records physical mass, prototype/scale, position, WXYZ body
quaternion, linear velocity and angular velocity in consistent CAD Z-up SI
coordinates. These post-simulation outputs do not enter the controller.

This establishes what the software simulates, not agreement with reality.
Required hardware work includes identifying the correct arm revision and joint
limits, weighing links/fruit, measuring centers of mass, servo step responses,
force saturation, reflected inertia/backlash, belt traction and fruit/finger
friction, then comparing recorded free falls, belt travel, grasp/slip and drop
trajectories against uncertainty ranges. The official flyer and pinned URDF
have differing limits, so revision agreement must be checked explicitly.
Camera distortion, exposure, blur, communication latency, observation noise,
tracking errors and domain randomization over identified uncertainties are also
needed before physical deployment claims. This benchmark learns discrete
scheduling over fixed IK skills; it does not learn joint torques or detection.
Its 25-feature abstraction omits velocities and detailed contact state, making
the scheduler partially observed even though the simulator evolves full dynamics.

## Recorded results

Each fresh method completed 256 physical windows, approximately 8.64 simulated
hours per method. Eight checkpoints were screened, 54 actual-RGB validation
rollouts covered six common streams, and 80 held-out rollouts covered sixteen
common streams across all five controllers.
Dueling Double DQN, checkpoint 256, was selected before any held-out outcome:
validation score 36, with 50 correct, 2 wrong, 33 missed and 5 pending from 90
arrivals. Its held-out results are 111 correct, 8 wrong, 73 missed and 15 pending
from 207 arrivals, versus 103/7/84/13 for the old proxy-trained controller.
The selected model's held-out maxima are 6.09 mm contact penetration, effort
fraction 1.000, joint speed 5.35 rad/s and finger speed 0.366 m/s. These measured
overshoots and contact tolerances are part of the result, not hidden constraints.

The fixed-minute demonstration has four arrivals, two correct sorts, two misses
and no wrong placements. The final 20 seconds render at 15 fps, 960×640, H.264.
Recorded fruit spin peaks at 21.88 rad/s, and at 2.54 rad/s while marked held;
this example does not exercise the extreme-spin stress case above. Its 901-state
trace passes finite-value, quaternion-norm and physical mass/scale checks.
All 353 recorded scheduler decisions independently reproduce from the selected
NPZ. The MP4 fully decodes; exported media and checkpoints match their originals.

The full training/comparison run, reusing the completed camera checkpoint, took
4523.40 s, including 2795.30 s in the parallel training phase and
659.95 s for held-out RGB evaluation. The final executable replay/render took
191.05 s; its Mep code block compiled and ran in 206.32 s and exported HTML/TeX.
The installed native suite passes 115 tests with eight optional GPU skips;
430 scoped definitions, twelve clangd hovers and four public examples are checked.
Raw results and provenance are retained in
`benchmarks/results/fruit_render/physics_fidelity.json` and the report's artifacts.
This is one training seed per method, fixed tuning and a nominal plant, not a
claim of universal algorithm superiority or validated hardware performance.

## Primary sources

- [MuJoCo 3.4.0 modeling and contact conventions](https://mujoco.readthedocs.io/en/3.4.0/modeling.html).
- [MuJoCo numerical integration and gyroscopic stability](https://mujoco.readthedocs.io/en/3.4.0/computation/index.html).
- [MuJoCo XML reference: solver, inertias and actuators](https://mujoco.readthedocs.io/en/3.4.0/XMLreference.html).
- [MuJoCo API functions and model/data state](https://mujoco.readthedocs.io/en/3.4.0/APIreference/APIfunctions.html).
- [MuJoCo constraint reference implementation](https://raw.githubusercontent.com/google-deepmind/mujoco/3.4.0/src/engine/engine_core_constraint.c).
- [Seeed pinned robot hardware sources](https://github.com/Seeed-Projects/reBot-DevArm/tree/922c5fa3d8c8919db1847b49166f02fc26c8fdad).
- [Seeed POS_VEL control guidance](https://wiki.seeedstudio.com/rebot_arm_b601_dm_pinocchio_meshcat/).
- [Seeed B601 DM specifications](https://files.seeedstudio.com/wiki/robotics/reBot_Arm_B601_DM_Physical_AI_Robotics_Arm.pdf).
