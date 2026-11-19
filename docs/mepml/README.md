# mepml reports in the docs

All literate reports live directly in this directory, grouped by subject and
topic. Neural-network reports are under [`nn/`](nn/). Each report contains its
editable programs, saved results and local figures.

The [source catalogue](overview/index.mepml) is generated from the current files.
`just docs-report-catalogue` refreshes it without a JSON inventory or conversion
ledger. The website and `just docs-mepml-run` discover `docs/mepml/**/*.mepml`.

The documentation website (`just docs`, `scripts/website/`) shows every report
under **Reports**, links C++ names to the API reference, and lists each API's
example reports. Run a report with `just docs-mepml-run PATH.mepml`, or select a
neural-network report by name with `just run-nn NAME`.

Reports are organized by subject and, where useful, a more specific topic:

| Directory | Topics |
| --- | --- |
| [`optim/`](optim/) | Optimization methods and worked comparisons |
| [`stats/`](stats/) | `bayes`, `regression`, `inference`, `time_series` |
| [`ml/`](ml/) | `classification`, `associative_memory`, `ensembles`, `trees`, `dimensionality_reduction`, `clustering`, `association_rules` |
| [`random/`](random/) | `generators`, `sampling`, `estimation` |
| [`math/`](math/) | `number_theory`, `coding_theory`, `geometry`, `calendar` |
| [`algorithm/`](algorithm/) | `strings`, `automata`, `hashing`, `cryptography`, `compression`, `concurrency`, `distributed`, `search`, `sorting`, `graphics`, `quantum` |
| [`geometry/`](geometry/) | `hulls`, `point_location`, `registration` |
| [`graph/`](graph/) | `search`, `shortest_paths`, `tours`, `link_analysis`, `matching` |
| [`numerics/`](numerics/) | `linear_algebra`, `root_finding`, `interpolation`, `ode`, `transforms`, `approximation` |
| [`nn/`](nn/) | Existing neural-network chapter structure |
| [`cad/`](cad/) | Code-driven CAD import, robot posing, export and rendering |
| [`datasets/`](datasets/) | Complete dataset studies: fitting, convergence, held-out errors and interpretability |

Examples with independent C++, Python and R tabs:

- [Matrix expressions and least squares](numerics/linear_algebra/matrix_expressions.mepml)
- [Computer algebra](math/computer_algebra.mepml)
- [B-spline regression](stats/regression/bspline_regression.mepml)
- [B-splines, P-splines and HB/HP refinement meshes](stats/regression/hierarchical_splines.mepml)
- [Seeed DevArm CAD: download, import, pose, export and render](cad/seeed_devarm.mepml)

The [playground](playground.mepml) currently compares native robot-arm
RL pipelines (Q(λ), Double DQN, dueling Double DQN and PPO), selects on independent
actual-RGB validation streams, then evaluates and renders the selected controller.
The current experiment trains and evaluates with optional MuJoCo articulated
contact dynamics, physical finger grasps and measured-pose rendering. The
[physics audit](../robot-physics-audit.md) records assumptions and calibration
limits; earlier proxy-task results remain historical evidence.
`just playground` trains the comparison; `just playground-preview` replays frozen
checkpoints and `just playground-gpu` renders that replay with CUDA. Standalone
research manuscripts keep their own source,
figures, outcome archives and Nix/just build files separately from the library
examples.

Neural-network programs and their fitting loops are editable in the [`nn/`](nn/)
reports. List them with `python scripts/run_nn_examples.py --list`.
The [diffusion reports](nn/3_generative_models/3.4_diffusion/ddpm.mepml) preserve
the original short fits, frozen-model samplers, baseline shortfalls and complete
editable C++ training loops, using shared native tensor components.
The [auxiliary-classifier GAN report](nn/3_generative_models/3.1_gans/auxiliary_classifier_gan.mepml)
shows native conditional generation, image-only discrimination and all alternating
updates, including the short run's weak samples and explicit diagnostic limits.
The [bidirectional GAN report](nn/3_generative_models/3.1_gans/bidirectional_gan.mepml)
keeps joint pairs, three independent optimizer states and unfitted cycle diagnostics
explicit, including its failed short-run reconstruction.
The [vanilla GAN](nn/3_generative_models/3.1_gans/vanilla_gan.mepml),
[least-squares GAN](nn/3_generative_models/3.1_gans/lsgan.mepml),
[hinge GAN](nn/3_generative_models/3.1_gans/hinge_gan.mepml) and
[conditional GAN](nn/3_generative_models/3.1_gans/conditional_gan.mepml)
reports keep the shared dense networks, distinct source objectives and both-network
class conditions explicit, including complete loops and weak generated images.
The [InfoGAN report](nn/3_generative_models/3.1_gans/infogan.mepml)
keeps generated codes, separate image-only recognition and all three optimizer
phases explicit, including weak images and the negative empirical information score.
The [Spectral GAN report](nn/3_generative_models/3.1_gans/spectral_gan.mepml)
keeps cached spectral directions, differentiable weight normalization and every
state/D/G phase explicit, including the measured approximation error and weak images.
The [Self-Attention GAN report](nn/3_generative_models/3.1_gans/self_attention_gan.mepml)
keeps spatial patch mixing, zero residual gates, two spectral caches and every
state/D/state/G phase explicit, including weak samples and measured nonlocal controls.
The [BigGAN report](nn/3_generative_models/3.1_gans/big_gan.mepml)
keeps hierarchical noise, conditional BN captures, projection scores, rejection
proposals and every spectral/Adam/moving-statistic write explicit, including weak
samples and the original batch-statistics loss convention.
The [DCGAN report](nn/3_generative_models/3.1_gans/dcgan.mepml)
keeps learned transposed maps, joint real/fake BN for D, fresh G through saved-BN
D and all four state/Adam write owners explicit, including checkerboard-like
samples and the favorable coarse score of a constant fitting-mean baseline.
The [Wasserstein GAN report](nn/3_generative_models/3.1_gans/wasserstein_gan.mepml)
keeps clipped RMSProp and gradient-penalty Adam trials separate, with actual input
Jacobians, stopped interpolation paths and every three-critic/fresh-G write explicit.
It preserves the weak diagonal samples and unconverged signed gaps.
The [Progressive GAN report](nn/3_generative_models/3.1_gans/progressive_gan.mepml)
shows both real fade paths, eight caller-owned Adam states and all-active-gradient
capture before selected writes. Its fixed-weight fade gallery is separate from
its three weak epoch galleries.
The [Pix2Pix report](nn/3_generative_models/3.1_gans/pix2pix.mepml)
keeps quantized binary conditions, both U-Net skips and D-then-fresh-G writes
explicit. Its preserved weak result is compared with the stronger fitting-only
mean image, with original and ten-class galleries and blank-condition controls.
The [CycleGAN report](nn/3_generative_models/3.1_gans/cycle_gan.mepml)
keeps the domains disjoint and both generator captures before either write. Its
original and ten-class galleries, wrong-snapshot control and stronger fitting-only
mean references qualify the weak fixed-final result.
The [StyleGAN report](nn/3_generative_models/3.1_gans/style_gan.mepml)
compares native AdaIN and modulated/demodulated paths with supplied spatial noise,
visible optimizer steps and all original weak-result galleries. Its additional
fixed-style control distinguishes truncation from shared-noise conditions.
The [StarGAN report](nn/3_generative_models/3.1_gans/star_gan.mepml)
keeps the target code, dual critic heads, shared-generator round trip and every
optimizer write explicit. Its original and all-digit galleries plus fitting-only
means qualify the weak requested-domain result.
The [SRGAN/ESRGAN report](nn/3_generative_models/3.1_gans/srgan_esrgan.mepml)
keeps frozen VGG content losses, warm-up/cold Adam reset, live relativistic means
and same-capture BN commits explicit. Original RGB previews and all32 per-image
PSNR/SSIM gains qualify the small improvements over bicubic.
The [SinGAN report](nn/3_generative_models/3.1_gans/singan.mepml)
keeps frozen lower-scale draws, explicit shared-RGB noise, fixed amplitudes,
both warm-start groups and cold per-scale Adam clocks visible. Its original
weak samples and training reconstruction remain unchanged.
The website serializes reports sharing a directory so their local figures and
fixtures cannot overwrite one another; separate directories can run concurrently.

Keep each report's `figures/` beside the report in its topic directory; its
complete C++ diagram blocks are the editable sources. The website uses these directories to group its Reports index
and navigation. The `nn/` hierarchy is maintained separately.

| Command | What it does |
| --- | --- |
| `just playground [FORMATS]` | Runs the playground manuscript's blocks, prints their captured results, and exports HTML and TeX with local figures to `build/playground/`. Optional arguments select export formats. |
| `just docs-mepml-run [FILES]` | Runs every code block of every report (or FILES) with mep (`mep-mepml run`, the editor's `C-c C-c` without a window) and writes their results and `file=` figures back into the report. Prints each block's outcome and fails if one failed; a failed block keeps its earlier results. The C++ blocks build against datamunge found by `pkg-config`. |
| `just view-docs` | Opens the unified API and report website in your browser. If the site is missing, it first runs `just docs-preview`, which exports stored results. |
| `just docs` | Runs the reports that changed since their last run (and any that failed), exports every report with mep (`mep-mepml convert`) and builds the website in `build/debug/docs/site/`, with a **Report build status** page. Fails if a code block failed. `just docs-serve` serves it; it also works opened from disk. |

The docs build schedules new or edited reports, reports with missing figures,
and reports whose last run failed. The last run record is
`build/debug/docs/mepml-status/<page>.json`. Commit the results written into
the report and keep generated figures locally.

For a batch chunk whose only required result is its document output, add
`id=descriptive-name, cache=yes` to the code fence. Keep the ID unique within
the report and retain it when moving or editing the chunk. MEpML reuses
successful results when the chunk's source and options match and stores them
in an ignored `*.mep-cache/` directory beside the report. Figure chunks with
`file=`, interactive chunks, and chunks that produce required artifacts run
without a result cache.

After changing an external input such as datamunge, prepared data, or an
environment option, use `mep-mepml run --refresh-cache PATH.mepml`,
`:MepmlRefreshCache` in the editor, or change the chunk's `cache-key=`.
To refresh the entire collection, run `just docs-force`.

| Option | Default | Effect |
| --- | --- | --- |
| `DOCS_MEPML_RUN` | `ON` | Run changed reports before exporting. `OFF` exports the stored results only. |
| `DOCS_MEPML_FORCE` | `OFF` | Rerun every report and refresh cached chunk results. |
| `DOCS_MEPML_ALLOW_FAILURES` | `OFF` | Build the site even when a block failed (the status page and a warning on the report's page still say so). |
| `DOCS_MEPML_TIMEOUT` | `1800` | Seconds a block may run before it is stopped and counted as failed. |
| `DOCS_MEPML_PKG_CONFIG_PATH` | empty | `PKG_CONFIG_PATH` for the blocks. `just` passes `$DATAMUNGE_MEPML_PKG_CONFIG_PATH`; set it to `$PWD/build/mepml-pc` to build against the working tree. |

mep's command line works on any report from any directory:

```sh
mep-mepml run docs/mepml/math/calendar/zeller.mepml          # run its blocks in place
mep-mepml run --report out.json --timeout 600 a.mepml b.mepml    # per-block outcomes as JSON
mep-mepml build --run docs/mepml/math/calendar/zeller.mepml md html -d /tmp/zeller
mep-mepml convert --run x.mepml /anywhere/x.pdf                  # run, then export one file
```

mep comes from the flake, pinned to a commit of its main branch;
`just update-mep` moves it to the latest one. Run reports inside
`nix develop .#cpp`: it supplies both mep and matching installed datamunge headers
and library. `just docs` and `just docs-mepml-run` select this shell automatically,
building its library when necessary. `cpp-tools` supplies tools for local builds
and documentation audits, without installing datamunge.

Before executing C++ reports, both commands check `pkg-config`, then compile,
link and run a small `<dm/dm.hpp>` program. A setup failure stops the build before
report results or statuses are overwritten, including when PCH is disabled.
For a missing-header error, check `pkg-config --cflags --libs datamunge` inside
the selected shell and any package-path override. Custom builds can still use
`DATAMUNGE_MEPML_PKG_CONFIG_PATH` or `DOCS_MEPML_PKG_CONFIG_PATH`; their `.pc` file
must describe matching headers and library. `just docs` refreshes mep discovery
on each configure. Pass `-DMEP_MEPML_EXECUTABLE=/absolute/path` to select another
executable explicitly.

Code blocks execute from their report's directory. Dataset-preparation blocks
locate and enter the source-tree root before running repository helpers; diagram
and figure blocks continue using paths relative to their report. Keep cached
source archives rather than downloading them again on each documentation build.
The installed-library shell also supplies HDF5 C++ and a versioned Python 3.12
for isolated style-reference fixtures. On NixOS its reference-library path is
applied only to that Python subprocess, so pip wheels can find their native
runtime without changing mep's loader environment.

Binding examples can select a language shell per block. In
[`numerics/linear_algebra/matrix_expressions.mepml`](numerics/linear_algebra/matrix_expressions.mepml), the Python fence uses
`nix-shell="git+file:../../../../..#python"` and
`cmd="env -u PYTHONPATH ../../../../../build/venv/pydatamunge-run/bin/python {file}"`;
the R fence uses `nix-shell="git+file:../../../../..#r"`. Shell and interpreter paths are
relative to the report's directory. The explicit `git+file:` scheme selects the
local Git checkout, including tracked working-tree edits, rather than an indirect
registry or GitHub reference, and avoids copying ignored build artifacts into the
Nix store. A remote shell reference instead uses `github:owner/repo#shell`.
Use `cmd` to select the Python interpreter
in mepml; the pinned runner does not apply Org's `python` interpreter option
or its `dir` option. Python imports `pydatamunge` from the local virtualenv;
the R shell hook exposes `datamunger` from `build/r/library`.
Prepare these installations with `just build-python`
and `just build-r`. Executing blocks reuses them; rebuild the affected binding
explicitly when its native code or interface changes. The development shells
provide tools and the core without adding Nix-built binding packages.

To add a report, put `NAME.mepml` in the appropriate `<subject>/<topic>/`
directory (or directly in a subject directory for an overview). Its path under
`` becomes its website path: `stats/bayes/NAME.mepml` becomes
`reports/stats/bayes/NAME.html`, listed on the Reports index. Each runnable block is a complete
program, because mepml has no noweb. A block with `file="figures/X.svg"`
shows the newest `.svg` its program writes under this directory. Programs here
write figures to `build/`, which is ignored by git. `ml/classification/iris_lda.mepml`
is a worked example.

Use an introductory native concept diagram when it helps explain the method.
`dm::graphics::Diagram` draws labelled boxes and circles, boundary-anchored arrows,
multiline UTF-8 labels, polygons, ellipses and paths with fitted SVG bounds and
the plotting library's embedded font. Keep a complete C++ program in the report;
extend the C++ graphics library when another reusable primitive is needed.
Diagram generation uses the same datamunge compiler/linker setup as the numerical
examples. Save all rendered images into the report's ignored `build/` directory,
then copy/reference them with `file="figures/..."` and `\image(...)`. Keep SVG as
a standalone local file: never print SVG XML, inline data URIs or base64 images
into result blocks. Keep reusable algorithms, data preparation and graphics in the canonical library.

Report data preparation runs in C++. Missing sources are downloaded with pinned
size/checksum verification, then extracted and prepared through the library;
reports do not require Python data scripts, NumPy, SciPy, pip or a reference venv.
The shared cache is `DATAMUNGE_DATA_DIR`, then the normal XDG/home dataset cache.
Per-report `DATAMUNGE_*_DIR` overrides still select caller-supplied prepared data.

For example, inside `nix develop .#cpp`:

```sh
datamunge-prepare-data signals "$PWD/build/datasets"
datamunge-prepare-data flowers "$PWD/build/datasets/flowers102"
```

The same operations are available as
`dm::datasets::prepare_example_data("signals", directory)` and
`dm::datasets::prepare_example_data("flowers", directory)`. Signals and text use
a parent directory containing their individual caches. OCR and VQA automatically
prepare the sibling COCO128 cache; two-stream video prepares its KTH source
cache; voxel occupancy prepares the original ModelNet meshes first. Video uses
FFmpeg for codec decoding, and downloads use curl; the flake supplies both.
`DATAMUNGE_DATA_OFFLINE=1` prohibits downloads, and
`DATAMUNGE_DATA_MIRROR/<dataset>/<filename>` can supply checksum-identical files.
A failed transfer or extraction never publishes an incomplete file. Original
source identities, split policies and attribution remain in JSON/TSV manifests.
Independent framework oracle scripts are optional development checks and are
not part of report data preparation.

CMake builds enable these adapters with `DATAMUNGE_ENABLE_DATA_ASSETS=ON`
(default), requiring libarchive, OpenSSL Crypto, nlohmann-json and HDF5 C++.
Minimal consumers can disable the option; data APIs then throw an explicit
feature-unavailable error. Numeric NPY/NPZ reading and writing remain native
and do not depend on this option or a Python installation.

Reports sharing a checkpoint declare its producer with
`//? Requires: vanilla_saliency.mepml` (a path relative to the report).
The website includes prerequisites once, runs them before their dependents,
and skips dependent execution if a prerequisite fails. This lets a fresh
`just docs` build create shared checkpoints through their original C++ reports.
For direct `mep-mepml run` use, run the declared prerequisite reports first.

The native array writers also support explicit Fortran-order/big-endian NPY,
DEFLATE NPZ archives and IDX output through `<datamunge/nn/array_io.hpp>` (also
included by `<dm/dm.hpp>`). This keeps the small format fixtures and fitted-weight
exports in the foundational reports native as well.

## Publish the documentation site

Run `just docs-publish` from the source checkout. It builds the API reference
and exports the saved report results and locally available figures, then pushes
the generated site to `origin/gh-pages`. The publishing command keeps the
current branch and index in place. It uses your normal Git credentials.

Generate missing figures locally with `just docs-mepml-run PATH.mepml` before
publishing the affected reports. Keep the generated images, videos and audio
in your local checkout and on the published branch. A fresh source checkout
can preview the saved text and results, with placeholders for missing figures.

After the first push, set **Settings → Pages → Source** to **Deploy from a
branch**, and select **gh-pages** and **/(root)**. The branch contains the
finished HTML and media. GitHub serves those files as-is. You can also fetch
the branch and open its `index.html` from a local checkout.
