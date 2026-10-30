# Native simulation and GPU implementation

This tracks the implementation authorized on 2026-10-29. The reference is the
implemented VSim library in `~/projects/vsim`, commit `de350c8`; its unimplemented
editor/audio/asset roadmap is not part of this baseline. VSim remains untouched.

The engine is a datamunge library, with native physics/rendering, native game
behaviors, optional graphics, CPU/CUDA/Vulkan execution, and C++/Python/R APIs.
Metal is deferred. Full NN GPU parity is required, not just MLP inference.

## Completion gates

- [ ] Freeze source, operation, dtype and reference-test inventories.
- [ ] Shared compute contexts, resource ownership, transfers and capability checks.
- [ ] Device-aware tensor allocation, views, dispatch and autograd.
  - [x] Retained compute storage, device metadata and explicit packed transfers.
  - [x] Native GPU strided packing/scalar fills and context-preserving transfer gradients.
  - [x] Optional CUDA toolkit/cuBLAS builds and native Float32/Float64 matrix dispatch.
  - [ ] Complete numerical dispatch and GPU mathematical gradient operations.
- [ ] CPU/CUDA/Vulkan parity for every built-in numerical NN operation and model.
- [x] Validated native scenes, components, prefabs, fixed clock and C++ behaviors.
- [x] CPU box/sphere/capsule physics, CCD, contacts, material response and locks.
- [x] CPU hinge/slider joints, forces, limits, subtree teleports and saved solver state.
- [x] CPU batched worlds, masked resets, per-environment RNG and preallocated stepping.
- [ ] CUDA/Vulkan physics with the same complete feature set as CPU.
- [x] Native Courier gameplay and explicit import of legacy script-bearing scenes.
- [x] Optional Vulkan rendering, shadows, HDR, SSAO, overlays and capture.
- [x] Real InvertedPendulum/Hopper environments and native CPU PPO/SAC training.
- [ ] Generated Python/R interfaces and ecosystem-native tests.
- [ ] CPU/GPU numerical and gradient tests, sanitizer and documentation gates.
- [ ] Measured simulator and end-to-end training throughput on physical GPUs.

Each item is checked only with executable evidence. Missing GPU kernels must
raise an error; host evaluation is never reported as GPU support. CPU is the
default. Host-only I/O and arbitrary host callbacks keep explicit device-transfer
boundaries. Cross-backend parity is numerical/behavioral, not bitwise agreement
with Jolt. Ordinary builds must not require GPU SDKs or a renderer.

## Execution order

1. Record immutable reference manifests and introduce the shared compute layer.
2. Preserve CPU tensor behavior while adding explicit device allocation/access.
3. Implement scene/game primitives and native CPU physics with analytic tests.
4. Add CUDA/Vulkan kernels and batch execution, then migrate all NN primitives.
5. Connect rendering, real RL tasks, language bindings and throughput benchmarks.

CPU, software-Vulkan and physical CUDA tests can run locally. The host has an
RTX 5080 Laptop GPU with an NVIDIA 580-series driver. The execution sandbox hides
its device nodes, so physical-device checks require host execution. Skipped
hardware tests are not passes. End-to-end training throughput remains unqualified.

## Evidence

The machine-readable baseline and local verification belong under
`build/simulation-verification/`. Remaining work is recorded here rather than
represented by success-shaped stubs or silently substituted CPU algorithms.

## Implemented CPU and compute foundation (2026-10-30)

- CPU worker contexts, owning buffers/views and explicit transfers are implemented.
  CUDA Driver API and Vulkan 1.3 native allocation/program/dispatch are implemented;
  commands are currently synchronous. GPU physics/NN kernels are not implemented.
- Native scenes, prefabs, camera/material validation, fixed clock, per-entity C++
  behaviors and transactionally reconciled simulation updates are implemented.
- CPU box/sphere/capsule contacts, conservative sweeps, material response, motion
  masks, forces/impulses, sleeping and sensors are implemented. Hinge/slider
  constraints include cached impulses, unilateral limits and subtree teleport
  velocities relative to moving parents. Saved/forked state includes solver caches.
- CPU batch stepping/reset uses independent worlds and RNG streams. Successful
  worlds rewind on batch failure. Native Hopper stepping/reset passed the
  zero-allocation check after warmup.
- Physical InvertedPendulum/Hopper tasks expose explicit terminal/time-limit flags,
  retained final observations and masked reset. C++/Python skipped demonstrations
  now execute native rollouts; R has a native task example. C++ PPO/SAC now train
  on real native transitions; Python/R training ports remain open.
- Task bindings are generated through shared SWIG adapters. Language outputs own
  their copies. Full scene/world/compute bindings remain open.

### Verification evidence

`build/simulation-verification/` contains native build/test logs, an allocation
and throughput JSON sample, header example executions, Doxygen coverage,
clangd hover replies, Python/R tests and rollout logs, GPU compile/configuration
and shader execution logs. The native CPU suite passes 56 executed checks (including the allocation check),
with four optional GPU tests skipped. All three renderer checks pass separately
on software Vulkan, including procedural geometry, AO, overlays, resize/resource
lifetime and PNG capture. The compute shader/buffer checks also pass on software
Vulkan; CUDA explicitly skips because libcuda is unavailable. Nine representative
new definition hovers pass; 137 header examples compile and run against the
GPU-enabled library. Fresh scoped Doxygen coverage is 856/856 definitions.
Address/undefined sanitizers pass 53 CPU tests, including the strict scene parser
and both native Courier games; leak detection was disabled because this ptrace
host does not support LeakSanitizer.

The Debug CPU sample is about 3,200 Hopper transitions/s (32 worlds, four workers,
6,400 transitions), with zero steady-state C++ allocations. This is a local sample,
not a release-build or end-to-end training claim. Source inventory currently
records 55 VSim files and 1,119 public NN declarations, all GPU declarations
pending. This is a provenance snapshot; model/dtype/gradient parity remains open.

The installed R package and its task tests pass. Python task tests pass against
its generated core extension. A complete Python packaging build remains blocked
by existing unqualified model types in the separate NN wrapper; no wheel success
is claimed. No physical GPU execution, graphics UI validation or external Jolt
conformance comparison has been performed.

### Native scene import, Courier and rendering (2026-10-30)

Version-one JSON import preserves exact signed-64-compatible IDs and rejects
invalid UTF-8, duplicate/unknown keys, invalid graphs and unmapped legacy scripts.
Native Courier factories execute both frozen source scenes; a replay solves the
original grid level. Lua is neither executed nor retained.

The optional Vulkan renderer uses datamunge compute resources directly, with
procedural Box/Sphere/Crystal geometry, GGX materials, geometric shadows, HDR/ACES,
SSAO, bitmap overlays, viewport control and explicit owned RGBA/PNG capture.
Embedded shaders regenerate from local GLSL; normal builds need no shader SDK.
Rendering currently traces every mesh per pixel and has no acceleration hierarchy;
software-driver correctness does not establish real-time hardware performance.

### Device-aware tensor foundation (2026-10-30)

Tensor storage now retains compute contexts and native owning buffers. All five
dtypes support bit-preserving explicit transfer, native strided GPU packing and
native scalar fills. Gradient transfers return to source contexts; gradient
metadata and zero materialization retain device ownership. Numerical GPU kernels,
model execution and complete GPU gradient accumulation remain open. Device APIs
are currently C++-only; generated NN Python/R CPU interfaces explicitly omit them
until compute-context bindings are implemented.

The tensor/view/autograd/operation/memory/I/O suite passes 164 executed CPU checks,
with two hardware tests skipped in a default build. Software Vulkan passes the
native storage, exact Int64, odd Boolean, reversed/broadcast view, context lifetime,
two-dimensional dispatch and transfer-gradient checks. CUDA PTX is included,
but driver compilation and CUDA execution remain unverified. Changed storage/compute/scene contracts
pass scoped documentation checks; verification artifacts record current counts.

Final combined verification for this batch: 220 CPU checks pass with six optional
GPU skips; the GPU-enabled suite passes 14 checks with two CUDA skips, including
four actual software-Vulkan execution cases. All 154 selected header examples
compile/run; 12 scoped clangd hovers and 1,236/1,236 scoped definitions pass.
Python task tests (2) and installed R task assertions (16) still pass. Tensor,
transfer-autograd and native backend host code pass Address/Undefined sanitizer
checks, including software-Vulkan storage execution; the remaining numerical core
was linked without sanitizer instrumentation. Leak detection remains unavailable.
The standard source archive includes embedded shaders, GLSL, generators and shared
simulation interfaces. The standard C++ example build produces a 640x400 capture
of the full 3D Courier scene after 60 physics ticks.

Three Release CPU Hopper samples measured about 2,550–2,945 transitions/s with
zero steady-state allocations (32 worlds/four workers/6,400 transitions per run).
The shared host was not performance-isolated; these samples do not establish
compiler/build comparisons, training throughput or hardware GPU speed.

### Native CPU policy training and Hopper speed-bound regression (2026-10-30)

The synthetic sac_vsim replay demonstration is replaced by native InvertedPendulum
and Hopper rollouts with actual SAC optimizer steps and evaluation. New ppo_vsim
uses native batched tasks, clipped Gaussian PPO and GAE that bootstraps time limits
from retained final states while cutting traces before resets. Both programs
reject GPU training requests while numerical kernels remain incomplete. All four
default native training/task combinations execute with finite losses and real
parameter updates. Short runs establish execution, not policy convergence.
The examples print measured CPU training-loop throughput; GPU training and Python/R
training ports remain open.

Full-range Hopper actions exposed constraint impulses exceeding speed bounds after
the pre-solve clamp. Dynamic speeds now retain the same 500 m/s and 100 rad/s
magnitude limits after every constraint solve and before swept motion. A seeded
800-transition full-action Hopper regression verifies bounded state across resets;
adapter tests verify final-state handling and terminal/truncation GAE semantics.
The combined CPU suite now executes 220 passing checks with six optional GPU skips.

Final native-training verification retains 220 executed CPU checks (six optional
GPU skips), 53 CPU engine/adapter sanitizer cases, 14 passes in the GPU-enabled
suite (two CUDA skips), 154 compiled/executed header examples and 12 clangd hovers.
Fresh scoped documentation coverage is 1,236/1,236 definitions. Native training
reports are saved in native-training.json with actual nonzero parameter changes;
short-run returns are not a convergence claim.

### Physical CUDA toolkit and matrix adapters (2026-10-30)

The host's RTX 5080 Laptop GPU is accessible outside the execution sandbox.
CUDA discovery now handles the NixOS driver path and an exact
DATAMUNGE_CUDA_DRIVER_LIBRARY override. Driver exceptions include numeric codes,
and PTX compilation failures include the native JIT error log.

The flake's Linux-only CUDA shell supplies a coherent CUDA 12.8 toolkit: NVCC,
runtime/headers, NVRTC and cuBLAS. The simulation shell includes it on supported
Linux systems. CPU packages/shells keep their existing dependencies, and macOS
outputs omit CUDA. CMake's optional DATAMUNGE_ENABLE_CUBLAS flag requires CUDA;
the driver/PTX-only path still needs no SDK. `just build-cuda`, `just test-cuda`
and the datamunge-cuda package expose the corresponding build paths. CUDA test
mode requires execution and rejects missing CUDA/cuBLAS rather than skipping.

Context::matmul and nn::matmul now dispatch native Float32/Float64 GEMM with
scalar-aligned bounded views, output/input alias validation, transposes, native
strided packing, leading batch broadcasting and empty dimensions. A retained
per-context cuBLAS handle uses prescribed arithmetic with TF32 disabled. Matrix
gradients with matching batch shapes remain on CUDA. General reductions,
gradient accumulation, other numerical operations, complete models and GPU
physics remain unfinished; GPU training is still explicitly unsupported.

Verification in `build/cuda-verification/` includes the realized Nix shell and
its complete Release core build; four physical CUDA test cases pass against that
library. They include both precisions, all transpose/gradient combinations,
matching-batch gradients, partial broadcasting, negative-stride packing,
empty products, invalid ownership/alignment/aliasing/bounds/overflow and bad PTX.
NVCC also compiles sm_120 PTX in the shell, and native execution verifies all 64
outputs. The combined CUDA/software-Vulkan suite passes all 18 checks. The CPU
suite executes 220 passing checks with eight optional GPU skips. All 158 header
examples execute, four representative clangd hovers pass, and all 263 scoped
compute/matrix implementation definitions are documented. Hardware throughput
and full CUDA NN/physics parity are not claimed.
