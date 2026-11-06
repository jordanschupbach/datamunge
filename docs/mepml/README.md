# mepml reports in the docs

`examples/**/*.mepml` are literate reports written in mepml (mep's document
format): prose plus small C++ programs built on datamunge, with each program's
output stored under it. The documentation website (`just docs`,
`scripts/website/`) shows every report as a page under **Reports**, its C++
names linked to the API reference, and each API page lists the reports that
use it.

Reports are organized by subject and, where useful, a more specific topic:

| Directory | Topics |
| --- | --- |
| [`examples/optim/`](examples/optim/) | Optimization methods and worked comparisons |
| [`examples/stats/`](examples/stats/) | `bayes`, `regression`, `inference`, `time_series` |
| [`examples/ml/`](examples/ml/) | `classification`, `ensembles`, `trees`, `dimensionality_reduction`, `clustering`, `association_rules` |
| [`examples/random/`](examples/random/) | `generators`, `sampling`, `estimation` |
| [`examples/math/`](examples/math/) | `number_theory`, `coding_theory`, `geometry`, `calendar` |
| [`examples/algorithm/`](examples/algorithm/) | `strings`, `hashing`, `compression`, `concurrency`, `distributed`, `search`, `sorting`, `graphics`, `quantum` |
| [`examples/graph/`](examples/graph/) | `search`, `shortest_paths`, `tours` |
| [`examples/numerics/`](examples/numerics/) | `linear_algebra`, `interpolation`, `ode`, `transforms`, `approximation` |
| [`examples/nn/`](examples/nn/) | Existing neural-network chapter structure |
| [`examples/cad/`](examples/cad/) | Code-driven CAD import, robot posing, export and rendering |
| [`examples/datasets/`](examples/datasets/) | Complete dataset studies: fitting, convergence, held-out errors and interpretability |

Examples with independent C++, Python and R tabs:

- [Matrix expressions and least squares](examples/numerics/linear_algebra/matrix_expressions.mepml)
- [Computer algebra](examples/math/computer_algebra.mepml)
- [B-spline regression](examples/stats/regression/bspline_regression.mepml)
- [B-splines, P-splines and HB/HP refinement meshes](examples/stats/regression/hierarchical_splines.mepml)
- [Seeed DevArm CAD: download, import, pose, export and render](examples/cad/seeed_devarm.mepml)

The [playground](examples/playground.mepml) currently compares native robot-arm
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

Keep each report's `figures/` beside the report in its topic directory; its
complete C++ diagram blocks are the editable sources. The website uses these directories to group its Reports index
and navigation. The `nn/` hierarchy is maintained separately.

| Command | What it does |
| --- | --- |
| `just playground [FORMATS]` | Runs the playground manuscript's blocks, prints their captured results, and exports HTML and TeX with local figures to `build/playground/`. Optional arguments select export formats. |
| `just docs-mepml-run [FILES]` | Runs every code block of every report (or FILES) with mep (`mep-mepml run`, the editor's `C-c C-c` without a window) and writes their results and `file=` figures back into the report. Prints each block's outcome and fails if one failed; a failed block keeps its earlier results. The C++ blocks build against datamunge found by `pkg-config`. |
| `just docs` | Runs the reports that changed since their last run (and any that failed), exports every report with mep (`mep-mepml convert`) and builds the website in `build/debug/docs/site/`, with a **Report build status** page. Fails if a code block failed. `just docs-serve` serves it; it also works opened from disk. |

The docs build runs a report only when its `.mepml` is newer than its last run
record (`build/debug/docs/mepml-status/<page>.json`), so the first build runs
every report and later ones only the edited ones (and those that failed last
time, since the fix may be in the library). The results and figures it
writes go into the report itself, so commit them with it. CMake options
(`just docs -DOPTION=...`):

| Option | Default | Effect |
| --- | --- | --- |
| `DOCS_MEPML_RUN` | `ON` | Run changed reports before exporting. `OFF` exports the stored results only. |
| `DOCS_MEPML_FORCE` | `OFF` | Rerun every report, changed or not (e.g. after changing the library). |
| `DOCS_MEPML_ALLOW_FAILURES` | `OFF` | Build the site even when a block failed (the status page and a warning on the report's page still say so). |
| `DOCS_MEPML_TIMEOUT` | `1800` | Seconds a block may run before it is stopped and counted as failed. |
| `DOCS_MEPML_PKG_CONFIG_PATH` | empty | `PKG_CONFIG_PATH` for the blocks. `just` passes `$DATAMUNGE_MEPML_PKG_CONFIG_PATH`; set it to `$PWD/build/mepml-pc` to build against the working tree. |

mep's command line works on any report from any directory:

```sh
mep-mepml run docs/mepml/examples/math/calendar/zeller_analysis.mepml          # run its blocks in place
mep-mepml run --report out.json --timeout 600 a.mepml b.mepml    # per-block outcomes as JSON
mep-mepml build --run docs/mepml/examples/math/calendar/zeller_analysis.mepml md html -d /tmp/zeller
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
[`examples/numerics/linear_algebra/matrix_expressions.mepml`](examples/numerics/linear_algebra/matrix_expressions.mepml), the Python fence uses
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

To add a report, put `NAME.mepml` in the appropriate `examples/<subject>/<topic>/`
directory (or directly in a subject directory for an overview). Its path under
`examples/` becomes its website path: `examples/stats/bayes/NAME.mepml` becomes
`reports/stats/bayes/NAME.html`, listed on the Reports index. Each runnable block is a complete
program, because mepml has no noweb. A block with `file="figures/X.svg"`
shows the newest `.svg` its program writes under this directory. Programs here
write figures to `build/`, which is ignored by git. `examples/ml/classification/iris_lda.mepml`
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
into result blocks. The conversion requirements live in [conversion-plan.md](conversion-plan.md).

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
