# Native game and simulation library

The current implementation supplies a native CPU rigid-body engine, game scenes
and C++ behaviors, independent batched worlds, physical control tasks, legacy
scene import and optional offscreen Vulkan rendering.
Everything runs as a library; no window or external physics runtime is required.
The full VSim/GPU migration is still in progress: see [the implementation
checklist](../simulation-plan.md).

`datamunge::compute` also provides actual CUDA Driver API and Vulkan storage,
explicit host transfers, and native PTX/SPIR-V dispatch. **Compute backend
availability does not imply GPU physics or full numerical NN support.** Native
tensor storage and transfer primitives and optional CUDA Float32/Float64 matrix
products are available. The remaining numerical NN kernels and GPU physics remain
open. Ordinary builds are CPU-only and require neither GPU SDK.

## Build and verify

```sh
just test-simulation
# Or use the normal native tools directly:
cmake -S tests -B build/simulation -DCMAKE_BUILD_TYPE=Release
cmake --build build/simulation --target run_simulation_tests check_simulation_allocations
ctest --test-dir build/simulation -R '^simulation\.' --output-on-failure
```

The allocation check prints a JSON CPU throughput sample and fails if steady-state
stepping/reset performs a C++ heap allocation. It includes 32 articulated Hopper
worlds, four CPU workers, 6,400 transitions and explicit episode resets. It is a
simulator measurement; it does not measure policy inference or training.

Optional native compute uses explicit CMake options:

```sh
nix develop .#simulation
cmake -S tests -B build/simulation-vulkan -DCMAKE_BUILD_TYPE=Release \
  -DDATAMUNGE_ENABLE_VULKAN=ON -DDATAMUNGE_ENABLE_CUDA=ON
cmake --build build/simulation-vulkan --target run_simulation_tests
ctest --test-dir build/simulation-vulkan -R '^simulation\.' --output-on-failure
```

For CUDA numerical adapters on Linux, the flake supplies a small CUDA 12.8 SDK
(NVCC, runtime/headers, NVRTC and cuBLAS), without installing a host driver:

```sh
just build-cuda
just test-cuda
# Or configure explicitly inside the SDK shell:
nix develop .#cuda
cmake -S . -B build/cuda -DCMAKE_BUILD_TYPE=Release \
  -DDATAMUNGE_ENABLE_CUDA=ON -DDATAMUNGE_ENABLE_CUBLAS=ON
cmake --build build/cuda --target datamunge
# An installed CUDA variant is also available:
nix build .#datamunge-cuda
```

The simulation shell also includes CUDA on supported Linux systems. CPU shells
and packages keep their original dependencies. The cuBLAS flag requires CUDA and
CMake 3.17 or newer; driver/PTX-only builds still need no toolkit. CUDA discovery
tries the loader search path and `/run/opengl-driver/lib/libcuda.so.1` on NixOS.
Set `DATAMUNGE_CUDA_DRIVER_LIBRARY` to an exact driver library path when needed;
an invalid explicit override fails. Never point it at toolkit link-only stubs.
Hardware execution needs NVIDIA device access; a sandbox that hides `/dev/nvidia*`
cannot run CUDA even when the driver loads. `just test-cuda` requires CUDA and
cuBLAS tests to execute, and fails rather than accepting unavailable-device skips.

`compute::Context::matmul` validates bounded, scalar-aligned row-major allocation
views and dispatches synchronous cuBLAS SGEMM/DGEMM. Its output cannot overlap
input ranges. `nn::matmul` uses this adapter when both operands are on the same
CUDA context with the same Float32/Float64 dtype. Matrix transposes, native
strided packing, leading batch broadcasting and zero-sized dimensions work.
TF32 is disabled to preserve the requested Float32 arithmetic. Matching-batch
matrix gradients stay on CUDA. Broadcast-reduction gradients, shared-path
gradient accumulation, other NN numerical operations and GPU training still need
their native adapters; unsupported work raises an error.

```cpp
dm::compute::Context gpu({dm::compute::Backend::CUDA, 0});
auto a = dm::nn::from_vector(std::vector<float>{1, 2, 3, 4}, {2, 2}).to(gpu);
auto b = dm::nn::from_vector(std::vector<float>{5, 6}, {2, 1}).to(gpu);
auto c = dm::nn::matmul(a, b); // Result stays on this CUDA context.
auto host = c.to(dm::compute::Context{}); // Explicit transfer for host consumption.
```

Vulkan requires version 1.3 headers, loader and a compute-capable device. CUDA
currently loads the POSIX NVIDIA driver directly and accepts PTX; NVCC is not a
core build dependency. Windows CUDA loading is not implemented. Unsupported
backends fail with an explicit diagnostic. GPU tests skip unavailable devices;
a skip does not count as execution evidence. A software Vulkan driver can test
native resource and shader correctness, but does not qualify physical GPU speed.
The checked-in compute fixture regenerates with `just prebuild-compute-fixtures`;
render shaders regenerate with `just prebuild-simulation-shaders` (glslc or
glslangValidator, or `--shaderc-library <path>`). Embedded SPIR-V lets consumers
build the library without a shader compiler.

## Physics and game contracts

The C++ API is under `dm::sim` and `dm::game` when including `<dm/dm.hpp>`.
Header examples are executable contracts; run
`python3 scripts/check_simulation_examples.py --library-dir <core-library-dir>`.

`sim::World` owns validated static/dynamic/kinematic boxes, spheres and Y-axis
capsules. Positions, velocities, dimensions, forces and inertia use SI units.
Dimensions are full extents; rotations are unit w,x,y,z quaternions. World-axis
motion masks constrain translation and rotation. Gravity defaults to Y=-9.81.
Each call to `step` advances at most 1/30 second, using four semi-implicit
substeps and sequential impulse constraints by default. Small analytic feature
queries generate contact manifolds; conservative sweeps prevent fast primitive
motion from skipping solid contacts. Sensors report overlap without response.
Solver capacities are explicit and overflow throws instead of discarding pairs.
Dynamic speed magnitudes are capped at 500 m/s and 100 rad/s after forces and
constraint impulses, including before every swept advance.

Hinges and sliders connect a dynamic child to another body or the world. Their
coordinates start at the authored pose; anchors and axes are child-local.
Connected bodies do not collide. Generalized actuation, unilateral limits,
viscous damping, contact/joint warm starts and moving-parent subtree teleports
are native. Snapshots include pending actuation and solver caches, and restore
only to their originating world with unchanged topology/configuration. `fork`
creates independent mutable state with the same numeric body IDs.

`sim::BatchWorld` forks a prototype, retaining bounded working capacities per
world. Use small explicit capacities for small environments. Its persistent CPU
workers step independent worlds; a failed batch step rewinds all successful
worlds. Binary masks select steps/resets. Each environment has an independent
random stream; reset retains its stream and `seed` explicitly restarts streams.

`game::Scene` owns entities, optional mesh/camera/body/joint components and
independent prefabs. `game::FixedClock` bounds catch-up work.
`game::BehaviorRegistry` maps names to native C++ factories; `game::Simulation`
visits behavior instances in ascending ID order before physics. Instances retain
state, initialization runs once, newly created entities first run next tick,
and destroyed entities skip later callbacks. Input values never latch between
ticks. Scene edits validate transactionally; unchanged physics retains live
solver state. C++ callbacks are trusted caller code, and their effects are not
rolled back when a later callback throws. These APIs do not execute Lua.

## Scene import and native Courier

`game::parse_scene`, `load_scene` and `encode_scene` read/write bounded version-one
JSON with exact IDs, prefabs and component round trips. Import rejects malformed
UTF-8, duplicate/unknown keys, invalid parents/cycles and documents above 16 MiB.
Every legacy script needs an explicit `SceneImport` replacement; source is never
executed, translated or retained. A registry supplies the named native factories.

Both frozen Crystal Courier games have native rules, including grid movement,
pickups/traps/locked exits, and 3D acceleration, grounded jump, hazards, camera
orbit and moving lift. For the frozen 3D scene:

```cpp
const auto scene = dm::game::load_scene("scene.json", dm::game::courier_import(true));
dm::game::Simulation game(scene, dm::game::courier_behaviors());
game.tick(1.0 / 60.0); // Supply Input values for player control.
```

`courier_import` maps the known reference IDs; use your own mapping for edited
scenes. Construct a fresh Simulation to restart gameplay and behavior state.

## Offscreen Vulkan rendering

`game::Renderer` is an optional library component, with no window runtime. It
renders procedural Box/Sphere/Crystal meshes with GGX materials, shadow rays,
HDR lighting, screen-space ambient occlusion, ACES presentation and bitmap HUDs.
The default view comes from the lowest-ID camera; pass `RenderView` explicitly
to render a camera-free scene or choose normalized viewport bounds.

```cpp
dm::game::RenderConfig config;
config.width = 640; config.height = 400;
dm::game::Renderer renderer(config); // Requires the Vulkan build and device.
renderer.set_overlay("Crystal Courier", "Ready", "Move / Jump");
renderer.draw(game.scene());
const auto image = renderer.capture();
dm::image::write_png(image, "build/courier.png");
```

Capture explicitly reads back owned top-left-origin RGBA bytes. `color_buffer()`
retains native GPU storage without a host transfer. Resize invalidates the current
frame and preserves ownership of previously copied buffers. Scene preparation
and glyph generation run on the host; shading and presentation run on Vulkan.
Rendering uses float32 and checks required lens/geometry representations. It
currently traces all meshes per pixel, so hardware throughput needs measurement.

## Batched physical tasks

`sim::TaskBatch` implements InvertedPendulum (cart/hinged pole) and Hopper
(planar free torso, three leg hinges and ground). The exact observation/action,
reward, time-step and failure definitions are in `sim/tasks.hpp`. They are
native datamunge tasks, rather than numerical reproductions of Jolt/Gym results.

C++ outputs are preallocated borrowed arrays. Selected terminal rows must be
reset explicitly. `terminated` denotes physical failure; `truncated` denotes a
time limit. Final observations remain available across reset for bootstrapping.
Unselected rows retain all previous output values. Validate actions before
applying forces; after a propagated physics failure reset before retrying.

Python exposes `datamunge.TaskBatch.create(Task_InvertedPendulum, count,
workers=1, seed=0, time_limit=1000)`. R exposes
`TaskBatch_create("InvertedPendulum", count, workers, seed, time_limit)` with the
normal SWIG object methods (`$step`, `$reset`, `$output`, `$seed`). R seeds use
the binding's integer convention; use nonnegative 32-bit seeds. Language results
are owned snapshots and numeric/byte-mask fields become native sequences,
remaining valid after the task is collected. This copies output data; the
zero-allocation guarantee applies to native C++ stepping/reset.

Core SWIG regeneration remains `just prebuild-python` / `just prebuild-r`.
Task adapters are shared in `prebindings/simulation/src/tasks.i`. Tests are
`tests/python/test_simulation.py` and `bindings_tests/r/test-simulation.R`.
The existing Python packaging build currently encounters unrelated unqualified
model types in the separate `_nn_native` wrapper; local task tests used the
successfully compiled generated core extension, not a successful full wheel.
R installation and its task tests run against the installed native package.

Runnable demonstrations now perform actual rollouts:

- C++: `just run-nn vsim_envs`.
- Python: `examples/python/nn/6_deep_reinforcement_learning/6.1_environments/vsim_envs.py`.
- R: `examples/r/simulation_ex.R`.

The `vsim_envs` pendulum feedback controller is a baseline. Native PPO/SAC
training examples are available separately:

```sh
just run-nn ppo_vsim
just run-nn ppo_vsim task=hopper
just run-nn sac_vsim
just run-nn sac_vsim task=hopper
# Longer experiments, with the standard full-run switch:
just run-nn-full sac_vsim task=hopper steps=100000
```

These C++ examples collect actual native transitions and run eager NN gradient
updates. PPO uses batched TaskBatch workers; SAC stores real transitions in replay
and learns twin critics, a squashed Gaussian actor and entropy temperature. Both
bootstrap time limits from actual final observations; PPO cuts GAE traces before
auto-reset states. Requested GPU training fails explicitly while numerical GPU
kernels are incomplete. Defaults are short smoke runs; full training convergence
and physical GPU throughput are not qualified. Python/R currently expose native
task rollouts, with training example ports still pending.

Training logs include native transition counts, optimizer updates, measured loop
time and before/after evaluation returns. Random-policy demonstration returns
differ between C++ and Python because their host example RNGs differ; native
seed/reset behavior is shared.

The complete native 3D Courier capture example is
[`examples/simulation/courier_offscreen.cpp`](../../examples/simulation/courier_offscreen.cpp).
With Vulkan available, build it through the standard C++ examples project:

```sh
cmake -S examples/cpp -B build/courier-example -DBUILD_W_INSTALLED=OFF \
  -DDATAMUNGE_BUILD_NN_EXAMPLES=OFF -DDATAMUNGE_ENABLE_VULKAN=ON
cmake --build build/courier-example --target courier_offscreen
./build/courier-example/courier_offscreen \
  tests/fixtures/game/crystal_courier_3d.json build/courier.png
```

It advances 60 physics ticks and saves a 640x400 PNG. It returns 77 when Vulkan
is unavailable and does not create a window or execute legacy scripts.

## Device-aware NN storage

`nn::Tensor` retains a shared compute allocation and its explicit context. Default
factories still create CPU tensors. The context overloads of `empty`, `zeros` and
`full` allocate native CPU/CUDA/Vulkan storage; `device()` and `context()` report
ownership. Views share that storage and the autograd mutation counter.

```cpp
dm::compute::Context gpu({dm::compute::Backend::Vulkan, 0});
dm::compute::Context cpu;
const auto x = dm::nn::from_vector(std::vector<std::int64_t>{9007199254740993LL, 2});
const auto device_values = x.to(gpu);
const auto host_copy = device_values.to(cpu);
const auto exact_ids = host_copy.to_vector<std::int64_t>();
```

Transfers always produce independent packed storage. GPU clone/contiguous and
scalar fills use native byte kernels, preserving all five built-in dtype bit
patterns without floating-point conversion. Packing handles transpose, reversed,
sliced and broadcast views, including odd byte counts. GPU storage currently
limits each allocation to UINT32_MAX-3 logical bytes; native Vulkan descriptor
limits can impose a smaller limit when packing. Buffers reserve up to three
padding bytes, excluded from NN memory accounting.

`data_ptr`, `at`, `item`, `to_vector` and NPY serialization require CPU storage.
Read back explicitly with `to(cpu)`; missing mathematical GPU kernels raise an
error instead of evaluating on the host. Transfers between distinct GPU contexts
explicitly stage through host memory; same-context packing remains native.
Commands are synchronous. Numerical GPU operation/model parity is unfinished.

Floating transfers record an identity backward into the source context. Gradient
metadata retains device ownership, missing gradients allocate on that context,
and incompatible GPU seeds/returned gradients fail explicitly. This supports
transfer graphs and native fills/packing; arbitrary mathematical GPU backward
operations and repeated GPU gradient accumulation still need numerical kernels.

Run `just test-nn-device` for the tensor/view/autograd/operation/memory/NPY
regression suite, with optional CMake backend flags as for simulation. Regenerate
embedded storage SPIR-V with `just prebuild-nn-storage-shaders`. These new device
APIs are C++-only while broader compute bindings remain on the plan; generated
Python/R CPU APIs and native task bindings retain their current interfaces.

## Remaining migration work

Complete numerical NN GPU dispatch/gradients, GPU physics,
Python/R training ports, broader scene/physics language exposure and physical
GPU qualification remain on the plan. The reference manifest records exact
source hashes and pending NN declarations; it does not mark implementations or
backend parity complete.
