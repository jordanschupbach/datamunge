# Neural-network examples

Experiment orchestration lives in each `.cpp` example. Dataset setup, epoch and
batch loops, optimizer updates, validation, checkpoint selection and result
reporting are visible in that source file. Shared `common/` headers provide
model components, numerical operations, data readers and small reusable helpers;
they no longer own end-to-end `Run`/`Train` harnesses. MNIST examples show the
workflow directly in `main`; examples with multiple experiment variants may
keep their workflow in a function defined in the same file.


Sections **1 (129 CNN examples)** and **2 (36 autoencoder examples)** now live
under [`examples/cpp/nn`](../cpp/nn/README.md). Their matching Python entry points
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

Datasets still come from `DATAMUNGE_DATA_DIR` or the existing tflow download
cache. Example builds default to four compiler jobs; set
`DATAMUNGE_EXAMPLE_BUILD_JOBS` to change that limit.
