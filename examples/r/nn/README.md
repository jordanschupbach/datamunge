# Neural network examples in R

The R workflow uses `just`, the Nix `.#r` shell, SWIG and the `datamungerNN`
R package. No Python interpreter, Python build helper or Python runner is
invoked by the NN R commands.

The migration covers all 414 canonical NN C++ examples and is in progress.
Run `just nn-language-status` and inspect [the ledger](../../nn/language_status.json)
for actual R implementations and recorded validation. Reference wrappers do not
count as ports. Normal R commands reject unported names; the canonical C++
references remain available through `just run-nn NAME`.

Each R script owns its training/evaluation loop, checkpoint and report ordering.
Tensor operations, parameters, autograd and optimizers use the functional
`datamunge::nn` API through SWIG. Classification examples print a confusion matrix.
Shared example components (`examples/nn/common/*.hpp`) are bound as
`datamungerNN::nn_components$<namespace>` mirroring the C++ namespaces
(`nn_components$image_training$ImageBatch`, `nn_components$resnet_demo$CompactResNet(18L)`
-> `model$call(params, state, x, training)`); `resnet_basic.R` implements the C++
example's loop over them and prints exactly the C++ numbers. `just nn-audit`
lists the headers still unbound per example.

The `nn_components$memory_components$Model` binding supplies the canonical NTM,
DNC, MemNet and key-value models. Convert a named tensor list with
`nn$named_tensors` and returned records with `nn$named_list`. R owns autograd and
Adam updates; the adapter contains no fitting loop. The four native examples own
their full training, evaluation, streaming, checkpoint and diagnostic loops.
All 62 archives and four histories match C++ and Python byte for byte at five epochs.
Pass `epochs=5` for development runs; explicit counts are honored. `--quick`
limits datasets. Direct solvers and no-data demonstrations do not train.

```sh
just r-example cnn_mnist epochs=5
just r-example tensor_basics
just r-example autodiff epochs=5
just r-example linear_regression epochs=5
just r-example polynomial_regression
just r-example principal_components
just r-example logistic_regression epochs=5
just r-example softmax_mnist epochs=5
just r-example mlp_mnist epochs=5
just r-example sessions
just r-example learning_rate_schedules epochs=5
just r-example gradient_clipping_accumulation epochs=5
just r-example adaptive_optimizers epochs=5
just r-example advanced_optimizers epochs=5
just r-example sgd_momentum epochs=5
just r-example training_loop epochs=5
just r-example early_stopping_checkpoints epochs=5
just r-example dense_layer
just r-example activations
just r-example embedding epochs=5
just r-example pooling
just r-example dropout epochs=5
just r-example normalization_layers
just r-example batch_norm epochs=5
just r-example conv_layers
just r-example weight_spectral_norm
just r-example variables_initializers
just r-example k_sparse_autoencoder epochs=5
just r-example winner_take_all_autoencoder epochs=5
just r-example alexnet epochs=5
just r-example xception epochs=5
just r-example densenet epochs=5
just r-example ghostnet epochs=5
just r-example mobilenet_v2 epochs=5
just r-example mnasnet epochs=5
just r-example convnext epochs=5
just r-example efficientnet epochs=5
just r-example efficientnet_v2 epochs=5
just r-example preactivation_resnet epochs=5
just r-example pyramidnet epochs=5
just r-example res2net epochs=5
just r-example resnext epochs=5
just r-example revnet epochs=5
just r-example cbam epochs=5
just r-example non_local epochs=5
just r-example octconv epochs=5
just r-example repvgg epochs=5
just r-example sknet epochs=5
just run-nn-r cnn_mnist epochs=3
just run-nn-r-full cnn_mnist epochs=100
just run-nn-r convolution_im2col
just run-nn-r undercomplete_autoencoder epochs=1
just run-all-nn-r --quick epochs=1
just test-nn-r
just package-nn-r
```

The listed architecture ports own their batching, Adam updates, validation and
checkpoint loops in R, calling bound C++ forward functions and numerical
diagnostics. Their five-epoch checkpoints and histories match Python and C++
byte for byte. `nn$resize_bilinear` and `nn$resize_nearest` accept integer or
numeric `c(height, width)` pairs, or a bound C++ `ImageSize` object.

The competitive autoencoders use the bound C++ winner masks, forward functions
and fixtures under `nn_components$ae`, with explicit R epoch/batch loops.
Missing MNIST pools are prepared by the shared C++ component automatically.
Their manifest records the deterministic local selection and original source indices.
Existing prepared pools are preserved.

`just prebuild-nn-r` uses `Rscript scripts/prebuild_nn_r.R` to generate the SWIG
wrapper and stage a standard R package in `build/nn-r-native/source/datamungerNN`.
`just build-nn-r` installs it using `R CMD INSTALL` into `build/r/library`, linking
the core library provided by the Nix shell. The package source also supports
self-contained compilation without `DATAMUNGE_USE_SYSTEM_LIB=1`.

The run recipes invoke `Rscript` and the package directly. They have no Python
cache, Python subprocess scheduler or imposed training timeout. `run-all-nn-r`
runs the native R catalogue sequentially and forwards the supplied example
options to each script; the old Python runner's `--jobs`, `--with-library`, and
cache options are not used.

```r
library(datamungerNN, lib.loc = "build/r/library")
nn_example_names()
nn_example_run("cnn_mnist", c("--quick", "epochs=3"))
nn_example_run("convolution_im2col")
```

The package reads scripts from the checkout when available, honoring edits to
those scripts. It also ships the R implementations and utility module, so it
works outside the checkout. `DATAMUNGE_EXAMPLES_ROOT` selects an explicit checkout.

`DATAMUNGE_DATA_DIR` selects the dataset cache; otherwise it is
`$XDG_CACHE_HOME/datamunge/datasets` or `~/.cache/datamunge/datasets`. Missing
MNIST, CIFAR and derived MNIST pools are prepared through C++ bindings. Memory
tasks are generated on first use. Cached files are preserved.
`DATAMUNGE_DATA_OFFLINE=1` disables downloads. Outputs default to
`output/nn/r/quick` or `output/nn/r/full`; `DATAMUNGE_OUTPUT_DIR` overrides this.
For `cnn_mnist`, epochs are honored in both modes. Quick mode uses 64 fitting,
32 validation and 32 test images, with batch size at most 16. Full mode uses
55,000 fitting, 5,000 validation and 10,000 test images. The startup message
reports the effective mode and settings.

The native ports were validated against the C++ experiments: 62 checkpoint and
fixture arrays and ten autoencoder gallery images matched byte-for-byte in each
language. Earlier validation of all 165 entries concerned the legacy compiled
reference bridge, not a complete language rewrite.
