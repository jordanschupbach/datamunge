# Recorded robot workcell benchmark

From a completed realistic playground run with its optional PNG dump:

```sh
DATAMUNGE_VIDEO_START=0 DATAMUNGE_VIDEO_SECONDS=60 DATAMUNGE_STREAM_KEEP_FRAMES=1 just playground-preview
just bench-fruit-render
just bench-fruit-render --repetitions 1 --output build/fruit-render-smoke.json
```

This builds a Release C++23 driver against the C++ shell's installed core using
pkg-config. It needs the real DevArm asset cache, completed `trajectory.csv`,
`fruit_trajectory.csv`, frozen classifier/policy, and all 900 PNGs produced by the
realistic playground. Missing inputs produce an error; the benchmark does not
download assets, train again, run mep, or overwrite the report/video/checkpoints.
Allow a few minutes for the default three repetitions.

For normal iteration, `just playground-preview` renders only world time 40–60
seconds at the same resolution and sampling quality. Override
`DATAMUNGE_VIDEO_START` and `DATAMUNGE_VIDEO_SECONDS` to choose another window.
Both full trajectory CSVs and the complete 60-second simulation are retained.
Short-window PNG dumps go to `frames_START_SECONDS/`, preserving the benchmark’s
full-minute `frames/` input. Benchmark comparisons always sample the same full
900-frame trajectory, regardless of the preview duration.

The driver also participates in `benchmarks/CMakeLists.txt` as
`fruit_render_bench`. A standalone installed-library build avoids configuring the
other suites or fetching Google Benchmark:

```sh
cmake -S benchmarks/cad -B build/fruit-render-bench -DCMAKE_BUILD_TYPE=Release
cmake --build build/fruit-render-bench --target fruit_render_bench -j4
python3 scripts/benchmark_fruit_render.py --no-build
```

For an existing local Release library, set `PKG_CONFIG_PATH` to its .pc directory.
For example, the measured baseline uses `PKG_CONFIG_PATH="$PWD/build/mepml-pc"`.
That file identifies `build/diagram-tests/_deps/datamunge-build/libdatamunge.so`,
built with GCC 15.2, `-O3 -DNDEBUG`. The local driver also builds with Release
flags. `--reuse-samples` reuses completed driver samples and remeasures encoding.
Keep separate `--build-dir`, `--work-dir` and `--output` paths when comparing
different core builds; do not reuse samples after rebuilding their executable.

Outputs:

- `benchmarks/results/fruit_render/current.json`: raw per-call times, grouped
  statistics, environment, linked core and binary hashes, input hashes, and
  full-sequence encoding times.
- `baseline.json` in the same directory remains the frozen pre-optimization result.
- `build/fruit-render-benchmark/`: raw CSV, simulation outcomes, scene inventory,
  12 sampled frame images and render-only quality controls.
- [The performance plan](../../PERFORMANCE.md#robot-workcell-rendering-2026-11-01)
  records the interpretation, priorities and acceptance gates.

Measurement details:

- Twelve equally spaced temporal strata across the actual 60-second trajectory,
  three serial repetitions, one full warmup, 960x640 overview, 256x256 wrist RGB,
  six overview shadow samples and two AO samples. Replay uses the stored rounded
  joint/body poses, with seeded appearance reconstruction. It never resimulates
  the geometry used for these rendering samples.
  Each sample primes prepared caches with its actual immediate predecessor,
  outside timing. This measures real cache-hit/miss behavior; jumping directly
  between the twelve samples would falsely turn stationary frames into misses.
- Each pipeline sample times moving assets, scene assembly, overview rendering,
  actual wrist capture/inspection, a representative inset/HUD, and PNG output.
  HUD timing omits exact annotation details. Focused samples include registered
  background rendering and localization/cropping. Pipeline timings exclude
  neural inference, simulation, initial loading and encoding.
  The optimized overview reports preparation separately and reuses the exact
  joint/finger-keyed base. This benchmark retains PNG output for comparison;
  production playground video streams to `VideoWriter` and skips that stage.
- Separate render-only controls use the first classified target (frame 611),
  the same prepared full scene, rotated case order, and 1/2/4/8/16 threads.
  Their RGBA buffers must match exactly across thread counts. Quality controls
  change one setting at a time; their pixels intentionally differ and are saved
  for visual comparison. Do not interpret a single-pose speedup as a minute-wide
  speedup, or quietly change the classifier's input distribution.
- A one-pixel, one-thread render without shadows/AO measures scene validation,
  geometry transforms and BVH construction **plus** one pixel's rendering. It
  is a preparation-cost proxy, not an instrumented internal BVH timer. Subtracting
  it from a full render is an estimate, not an exact additive decomposition.
- Actual MobileNet prediction uses one NN thread, a real 64x64 crop and 50 calls
  per timing sample; renderer threads are independent. The full 60-second
  simulation uses the same frozen policy, seed, actual RGB callback and 240 Hz
  physics as the report, and records 901 snapshots through the exact endpoint.
- Encoding repeats the report's full 900-PNG, 15 fps, x264 baseline/CRF19/fast
  command three times, after all native benchmark cases have ended. File caches
  are warm. PNG timings include encoding and normal file writes, not an fsync or
  cold-disk throughput measurement. Cases run without CPU pinning; min/max and
  standard deviations are retained to expose host variability.

The historical 1,642-second replay is retained separately from sampled estimates.
The implementation results and full 900-frame replay timing are recorded in
`PERFORMANCE.md`; full data collection/model-training wall times were not
remeasured during the renderer pass. `--threads 16` matches the new playground
worker budget; the benchmark default remains the renderer's eight-worker cap.

## Bounded CUDA comparisons

GPU work should start with `just bench-fruit-render-gpu --repetitions 1`, then
use three repetitions when accepting a change. The recipe obtains the existing
CUDA-enabled Nix package. `DATAMUNGE_MEPML_PKG_CONFIG_PATH` may instead name a local
CUDA-enabled Release library's pkg-config directory. The wrapper builds
`fruit_render_gpu_bench` and records `benchmarks/results/fruit_render/cuda.json`.
It requires the existing full trajectory CSVs and cached URDF, but no PNG dump,
new training, simulation replay or video encoding.

Four poses cover idle belt, classified pickup, arm motion and late stream.
Alternating CPU/CUDA calls share a prepared snapshot, actors, 960x640 camera,
six shadow samples and two AO samples. Timings include synchronous readback and
all combined CPU hierarchy work; renderer initialization, first base upload/render,
and common base preparation are recorded separately. Four CPU/GPU comparison PNG
pairs and raw timings/quality metrics are saved under the work directory.
The acceptance gates are RGB RMSE below 3/255 and source-ID mismatch below 0.1%;
this float32 backend is not required to reproduce double-precision pixels exactly.
Per-pose depth errors, maximum channel differences and foreground overlap are also
retained. Sparse pose results are not a full-video speedup estimate.

`just playground-gpu` defaults to a 20-second video (40–60 s). For shorter
checks use `DATAMUNGE_VIDEO_SECONDS=5 just playground-gpu`. Keep
performance validation clips within **5–20 seconds** and prefer still poses for
benchmarks. The complete minute of physics/trajectory recording remains, while
only the selected video window is rendered. The wrist view stays on the CPU
reference, so these external-view changes do not change its classifier inputs.
The earlier full-minute CPU protocol above is retained for reproducing historical
results; it is not the default GPU performance loop.

## Bounded CPU wrist comparisons

`just bench-fruit-wrist --repetitions 1` measures four actual poses with a
256x256 wrist camera and a total budget of sixteen render workers. It performs
capture or inspection exactly as the video does, priming each measurement with
the immediate preceding frame. Idle/capture, changed-pose inspection and stable
inspection cases are reported separately; off-image focus is a separate control.
It generates no video, training or new simulation trajectory.

Use `--reference-library /path/to/libdatamunge.so` with an archived native core
for paired reference/candidate runs; each variant uses the same executable and
recorded inputs. Two rounds alternate variant order by default. All 31 binary
RGB/depth/ID/crop/box outputs must match exactly across runs and implementations.
Keep `--repetitions` within 1–3. Results include raw samples, per-case statistics,
reference/candidate library hashes and source/input fingerprints. The default
result is `benchmarks/results/fruit_render/wrist.json`. Equal-weight sparse cases
are not an extrapolation of a complete minute. The realistic fruit-free reference
contains moving tread marks, so it must stay fresh; the implementation overlaps
the two views within the same total worker budget rather than freezing it.
