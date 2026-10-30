# Neural-network examples

Experiment orchestration lives in each `.cpp` example. Dataset setup, epoch and
batch loops, optimizer updates, validation, checkpoint selection and result
reporting are visible in that source file. Shared `common/` headers provide
model components, numerical operations, data readers and small reusable helpers;
they no longer own end-to-end `Run`/`Train` harnesses. MNIST examples show the
workflow directly in `main`; examples with multiple experiment variants may
keep their workflow in a function defined in the same file, taking plain
parameters (a depth, a version) and defining the model inside. A generic harness
parameterized by model callbacks does not count, even when pasted into the file:
the R and Python examples must be able to mirror the loop line by line over the
SWIG-bound components (`examples/bindings/nn_components.i`). `just nn-audit`
reports every example's status (see `docs/nn-example-audit.md`).


Sections **1 (129 CNN examples)** and **2 (36 autoencoder examples)** now live
under [`examples/cpp/source/nn`](../cpp/source/nn/README.md). Their matching Python entry points
are under [`examples/python/nn`](../python/nn/README.md). The commands below still
discover both the migrated C++ sources and the remaining sections here. Shared
C++ helpers stay in `examples/nn/common` until the remaining catalogue migrates.

Examples run as **quick demonstrations by default**. Training uses small,
deterministic subsets and a few updates. Numerical fixtures, gradients, finite
losses, and checkpoint round trips still run. Quick results do not claim the
accuracy or rewards of a fully trained model.

```sh
just run-nn pyramidnet
just run-all-nn --with-library --jobs 3
```

`run-nn` builds only the requested example and its checkpoint prerequisites.
NN recipes use the tool-only `.#cpp-tools` shell; the installed-library
`.#cpp` shell remains available for workflows such as `repl-cpp`.

The runner enforces a **180-second deadline per example**, including its child
processes. A timeout is a failure and is never cached as a pass. Override it
with `just run-all-nn NAME --force --timeout SECONDS`. Parallel runs respect checkpoint dependencies and
write individual logs under `build/nn-examples/run-logs/quick/`.

Quick checkpoints go in `output/nn/quick/`. Quick and full runs have separate
caches. Customized `key=value` runs are not cached as default passes.

Full training is an explicit opt-in and can take hours:

```sh
just run-nn-full pyramidnet
just run-all-nn --full --with-library
# Direct binaries use the same mode switch:
DATAMUNGE_EXAMPLES_FULL=1 ./build/nn-examples/bin/pyramidnet
```

The source headers describe the reference/full experiments. Full mode retains
those schedules and convergence thresholds. Its outputs go in
`output/nn/full/`; it has no runtime deadline unless `--timeout` is specified.
`DATAMUNGE_OUTPUT_DIR` overrides either output location.

Datasets use `DATAMUNGE_DATA_DIR`, otherwise `$XDG_CACHE_HOME/datamunge/datasets`
or `~/.cache/datamunge/datasets`. MNIST, Fashion-MNIST and CIFAR are downloaded
and verified on first use. MNIST-derived autoencoder pools and memory copy/QA
tasks are prepared in C++; Python and R use these functions through SWIG.
Cached files are preserved. `DATAMUNGE_DATA_OFFLINE=1` disables network fetches.
The development shells default BLAS/OpenMP to one thread for small examples;
set `OMP_NUM_THREADS` or `OPENBLAS_NUM_THREADS` to override that budget.
Specialized corpora and pretrained-model assets still need preparation as
described in their individual examples. Example builds default to four compiler jobs; set
`DATAMUNGE_EXAMPLE_BUILD_JOBS` to change that limit.
