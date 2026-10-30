# mepml reports in the docs

`examples/*.mepml` are literate reports written in mepml (mep's document
format): prose plus small C++ programs built on datamunge, with each program's
output stored under it. The documentation website (`just docs`,
`scripts/website/`) shows every report as a page under **Reports**, its C++
names linked to the API reference, and each API page lists the reports that
use it.

| Command | What it does |
| --- | --- |
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
mep-mepml run docs/mepml/examples/zeller_analysis.mepml          # run its blocks in place
mep-mepml run --report out.json --timeout 600 a.mepml b.mepml    # per-block outcomes as JSON
mep-mepml build --run docs/mepml/examples/zeller_analysis.mepml md html -d /tmp/zeller
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

To add a report, put `NAME.mepml` in `examples/` (or a subdirectory). It becomes
the page `reports/NAME.html`, listed on the Reports index. Each runnable block is a complete
program, because mepml has no noweb. A block with `file="figures/X.svg"`
shows the newest `.svg` its program writes under this directory. Programs here
write figures to `build/`, which is ignored by git. `examples/iris_lda.mepml`
is a worked example.

Use an introductory TikZ concept diagram when it helps explain the method; keep its
editable `.tex`/`.tikz` source under `examples/diagrams/` and a short, repeatable
rendering block. LaTeX with PGF/TikZ and `dvisvgm` are needed only to regenerate such
diagrams. Save all rendered images into the report's ignored `build/` directory,
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
