# SWIG bindings

The layout follows the pattern in `~/projects/charm`: a C++ core, per-language
SWIG interfaces, generated native wrappers and proxy classes, and packages built
with the language's standard tools. Existing public module names are preserved.
Python and R wrap `NnTensor` and `NnModel` from `include/datamunge/nn/api.hpp`;
this facade keeps tensor computation, automatic differentiation, and training in
C++. Values cross the boundary as copied numeric vectors in row-major order.
These are not zero-copy NumPy or R-array views.

## Python

```sh
just prebuild-python  # SWIG -> src/datamunge_python_wrap.* and src/pydatamunge/datamunge.py
just build-python    # build/install a wheel in build/venv/pydatamunge-run
just test-python     # pytest against that installed wheel
just repl-python
```

```python
from pydatamunge import NnTensor
x = NnTensor([1, 2, 3], [3], "float64")
x.set_requires_grad(True)
x.square().sum().backward()
print(list(x.grad().values()))  # [2, 4, 6]
```

Outside Nix, install a C++23 compiler, Python >=3.9, SWIG, setuptools and wheel.
Generate once with `python scripts/prebuild_bindings.py python`, then use
`python -m pip install .`. To distribute, run `python -m build`; install `build`
first. The source archive includes the generated wrapper, all core sources and
headers, so installing the archive does not require SWIG. The default package
compiles its core into the extension and needs no separately installed dylib.
Build parallelism defaults to four jobs; override with
`CMAKE_BUILD_PARALLEL_LEVEL`. macOS uses Accelerate; other platforms currently
use the portable numeric kernels in self-contained builds.

## R

```sh
just prebuild-r       # SWIG -> src/datamunge_r_wrap.* and R/datamunger.R
just build-r          # R CMD INSTALL into build/r/library
just test-r           # testthat against that installed package
just repl-r
```

```r
library(datamunger, lib.loc = "build/r/library")
x <- NnTensor(c(1, 2, 3), as.integer(3), "float64")
x$set_requires_grad(TRUE)
x$square()$sum()$backward()
as.numeric(x$grad()$values()) # 2 4 6
```

Outside Nix, install R with its development tools, a C++23 compiler, Python 3
(for regeneration only), SWIG and testthat. Generate with
`python3 scripts/prebuild_bindings.py r`, then run `R CMD INSTALL .`.
`just package-r` (or `python3 scripts/build_r_source.py`) stages only the R
package inputs and invokes `R CMD build` to create `dist/datamunger_*.tar.gz`;
this avoids following the development checkout’s cyclic Node/direnv symlinks.
The archive contains generated wrappers; recipients do not need SWIG or Python.
`src/Makevars` uses R's normal compile/link rules rather than recursively calling
`R CMD SHLIB`. `src/Makevars.win` supplies the corresponding Rtools build rules.
Self-contained builds avoid mandatory X11; plotting to files remains available.
macOS uses Accelerate; other systems use portable kernels.

## Shared core builds and regeneration

Both packages support explicit `DATAMUNGE_USE_SYSTEM_LIB=1` builds using
`pkg-config datamunge`. Nix packages use this to compile the core once, including
its detected BLAS/LAPACK, Arrow and display support. Those packages depend on the
installed core; the default source packages are self-contained. Unix system-core
builds include an rpath to the reported library directory. System-core builds on
Windows are not covered by this workflow.

Edit the `.i` files and C++ headers, never the generated proxy/wrapper files.
The regeneration script stages output before replacing it, so a SWIG failure
preserves the previous generated files. The repository pins its SWIG fork in
`flake.lock`; the Python/R shells include it. Linux source-package CI uses the system SWIG package. Regenerate with the same SWIG version for reproducible
output. C++ tensor exceptions become Python/R errors; SWIG owns returned tensor
and model objects and releases them when collected.

## Audit and remaining scope

The existing `lib_sources.txt` correctly listed the core sources. The Python
source-distribution manifest omitted `src/datamunge`, so archives could not
reliably build independently. It now includes that tree, while setuptools
collects the same canonical source directory as CMake. R's old build hard-coded
X11 and called `R CMD SHLIB` recursively. The installed-package workflows now
avoid source-tree imports and `testthat::test_local`, which could mask packaging
problems or load a stale installed library. Tests cover imports, operations,
training, gradients, serialization and object lifetimes.

Other language packages and their existing `just prebuild-*`, `build-*` and
`test-*` commands are preserved. The README's existing implemented markers do
not imply tensor facade parity: only Python and R currently expose `nn/api.hpp`.
Extending that facade to other languages, zero-copy array typemaps, public wheel
publishing, and CRAN submission checks remain separate work. The packages have
not been certified for CRAN or tested locally on Windows/Linux in this pass;
Linux CI now builds both from their source archives.

## Validation for this pass

On macOS arm64 with Python 3.13.11, R 4.5.2 and SWIG 4.6.0:

- Built the self-contained Python wheel; installed it into a separate virtual
  environment and passed all 28 pytest tests, including the installed-package checks.
- Built and installed the self-contained R source archive; passed all 91
  testthat assertions with no test warnings or skips.
- Installed both source archives with `DATAMUNGE_USE_SYSTEM_LIB=1` against the
  Nix-built core; the same 28 Python tests and 91 R assertions passed.
- Both source archives contain all 271 core C++ translation units. The R source
  archive contains no compiled libraries or object files.
- Repeated generation preserves both generated bytes and file timestamps;
  unchanged wrappers no longer force an expensive recompilation.

Linux source-archive CI jobs were added but have not run locally. Windows build
rules are supplied but remain unverified. These checks do not constitute a CRAN
submission review or a published PyPI wheel release.

## NN example catalogue

The optional [Python NN example extension](../examples/python/nn/README.md) wraps
all 165 canonical C++ experiments from sections 1 and 2. Build it with
`just build-nn-python`. It is separate from the core tensor/model binding and
shares the C++ architecture and training implementations.

### Optional R neural network examples

`just build-nn-r` installs the standard `datamungerNN` companion package in
`build/r/library`. It mirrors all 165 Python CNN and autoencoder experiments
through the same C++/SWIG entry points. Use `just test-nn-r`,
`just run-nn-r NAME`, `just run-all-nn-r`, and `just package-nn-r`.
See [the R NN examples](../examples/r/nn/README.md) for options and validation.

### Functional NN API and language migration

Python now exposes the C++ functional API as `pydatamunge.nn`; R exposes it as
`datamungerNN::nn`. Bindings are generated from `prebindings/nn/src/nn.i` using
`scripts/prebuild_nn_api.py`. They include named parameter stores, layers,
optimizers, differentiation, batching, tensor I/O and dataset loading. Python
ships this API in its normal core wheel, independently of the optional compiled
reference extension. R's companion package can be built without the reference
bridge using `python3 scripts/build_nn_r_core.py`, or with the separate legacy
reference tooling. `just build-nn-r` now prepares the native package using R/SWIG
and installs it with `R CMD INSTALL`.

Three native-language ports are currently complete: CNN MNIST, explicit im2col
convolution, and the undercomplete autoencoder. Other catalogue entries remain
compiled references; they have not yet been rewritten. Callback-based custom
functions, gradient checkpointing and reversible callbacks are not exposed by
this initial functional interface.

The standard NN R commands now use an R-only toolchain: `just prebuild-nn-r`
runs `scripts/prebuild_nn_r.R`, `build-nn-r` uses `R CMD INSTALL`, and
`run-nn-r` invokes `Rscript` and `datamungerNN::nn_example_run` directly in the
Nix R shell. Only genuine R implementations are listed; legacy compiled
experiment wrappers and the Python R runner are outside this workflow.
