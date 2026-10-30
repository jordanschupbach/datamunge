# Native NN example migration

The active goal is genuine Python and R implementations of every canonical NN
C++ example. Each example owns its option parsing, data setup, epoch/batch loop,
forward/loss/update ordering, evaluation, checkpoints and reporting. Shared C++
model functions, data readers and numerical diagnostics are exposed through
SWIG and reused by the matching R and Python loops. Shared high-level experiment
`Run` functions and compiled example adapters do not count as language ports.

## Inventory and validation

`just nn-language-status` refreshes `examples/nn/language_status.json`. It scans
both canonical source roots: sections 1 and 2 under `examples/cpp/source/nn`, and the
remaining numbered sections under `examples/nn`. There are 414 distinct names:
57 foundations, 129 CNN, 36 autoencoder, 53 generative, 36 recurrent,
36 attention and 67 reinforcement-learning examples.

Each entry records C++ source, classification status, matching Python/R paths,
implementation status and validation evidence. `reference_wrapper` and `missing`
are unfinished. A `native` marker means the experiment was rewritten; it also
needs successful execution and numerical checks before it is validated. Source
and API hashes invalidate old validation evidence after changes. A null
classification flag means that example still needs review.

Keep training budgets small during development: normally five epochs or steps.
Do not assert convergence thresholds that require the original long training
budget. Check finite values, loss changes, gradients, shapes and known finite-step
trajectories as appropriate. Keep stronger convergence assertions for sufficiently
long runs. Explicit epoch options must be honored even in quick mode.
Classification examples must print the final confusion matrix with true classes
as rows and predictions as columns, over the entire evaluation set.

Commands for completed ports:

```sh
just python-example NAME epochs=5
just r-example NAME epochs=5
just run-nn NAME epochs=5
```

The first two reject unfinished ports. The R command uses only just, the Nix R
shell, Rscript and the installed datamungerNN package. The Python command builds
the functional API without the optional compiled experiment extension. R scripts
are also bundled in the installed package for use outside the checkout.

## Automatic example data

Native examples prepare missing MNIST and CIFAR-10 files through the C++
`dataset_assets` component, also exposed through SWIG in Python and R. The
component supports Fashion-MNIST and CIFAR-100 for the data-loader demos too.
Archives are checked against their published MD5 values, extracted in C++, and
published from temporary files. Existing supplied files are preserved. Failed
downloads leave no partial final files. Development shells include curl.

All languages use `DATAMUNGE_DATA_DIR` when set, otherwise
`$XDG_CACHE_HOME/datamunge/datasets` or `~/.cache/datamunge/datasets`.
A `data=PATH` option prepares that directory when the example reads it; looking
up a default path causes no download. `DATAMUNGE_DATA_OFFLINE=1` disables network
fetches with an error naming the missing cache. `DATAMUNGE_DATA_MIRROR` optionally
sets a mirror with `<dataset>/<archive>` layout and unchanged checksums.

Missing autoencoder/saliency pools are built in C++ from real MNIST images,
with balanced 1000/200/200 pools, distinct images, disjoint source IDs and a
manifest recording the independent seeded selection. Existing pools are kept.
The four memory examples generate their original synthetic copy/QA tasks on
first use. No preparation in another repository is needed for these workflows.
Specialized C++ examples that still require externally prepared corpora,
annotations, or pretrained model artifacts remain part of the migration work.

Use `just python-example softmax_mnist`, `just r-example softmax_mnist`, or
`just run-nn softmax_mnist`. `just run python-example softmax_mnist` also invokes
the unrelated default `run` recipe; it is not needed for NN examples.

Batch normalization, normalization and pooling now check their sampled inputs
against independent moments, finite differences and pooling-window calculations
instead of requiring hard-coded TensorFlow values from a particular standard-library
normal stream. Batch normalization also checks the moving mean and unbiased
variance recurrence at every step, including the full 300-step budget.

Development shells default `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS` to one
while preserving user overrides. This also handles OpenMP OpenBLAS builds,
which otherwise ignored the OpenBLAS-only limit and used 32 threads on the test host.
Evidence includes `auto-assets-thread-budget.log`, the `softmax-auto-assets-*`
logs, `cifar-auto-assets-cpp-mirror.log`, `idx-auto-assets-cpp.log`,
`saliency-auto-assets-*`, `memory-auto-assets-*`, `portable-fixtures-validation.log`,
and the `auto-assets-all-native-*` sweep logs under `build/nn-language-validation`.

## Current checkpoint

The current audit covers all 414 examples. There are 110 native Python scripts
and 95 native R scripts; these counts describe implementations, not completed
validation. All 414 C++ examples now own their orchestration; no generic
training runner remains in a shared header or pasted into an example. The
audit still flags 117 examples that call bundled per-step methods (including
pointer calls), and 324 without complete component bindings.
Those findings still require work; the overall migration is unfinished.

The final ten generic C++ runners are expanded: `seq2seq`, `luong`,
`translation`, `pointer_generator`, `pointer_network`, `word_rnn_lm`, `memnet`,
`key_value`, `ntm` and `dnc`. Each owns forward/loss/autograd/optimizer ordering,
including optional recorded-run replay. All ten targets build. Five-epoch
before/after comparisons preserve 218 archives containing 2,108 tensors. A
separate three-epoch replay check exercises the original weight/history/greedy
branches and preserves 214 archives. These use clearly labeled synthetic
regression fixtures and original C++ records as the oracle; they are not PTB,
translation-corpus or independent TensorFlow validation.

All ten report held-out confusion counts. Token models print sparse coordinates
with omitted cells explicitly zero, including the 10,000-word vocabulary;
masked sequence and bit-copy tasks count only valid target positions.
The unused bundled optimizer methods are removed from `memory_task.hpp`.

The four memory models are now exposed through
`nn_components.memory_components.Model` in Python and `nn_components$memory_components$Model`
in R. The shared adapter accepts named tensor records and provides parameter,
leaf, forward, loss, one-batch evaluation, prediction and checkpoint components;
it contains no optimizer or fitting loop. R's `nn$named_tensors` and
`nn$named_list` marshal named records while retaining their owners. Binding tests
cover gradients after output collection, host-owned Adam, label-free inference
and restore for all four models. At five epochs, both languages reproduce the
C++ first gradients, first updates and final parameters bit for bit across all
80 updates per model. All four now have native Python and R scripts with
explicit training, evaluation, streaming, diagnostic and checkpoint loops. Their
62 archives (501 tensors) and four histories match C++ byte for byte after five
epochs. Recorded-run replay also passes in both languages against C++ records;
this is regression evidence, not independent TensorFlow validation.

The preceding batch expands sixteen more C++ workflows: PointNet/PointNet++,
FlowNetS/PWC-Net, factorized video/I3D, capsule networks/spatial transformers,
k-sparse/winner-take-all autoencoders, ConvLSTM, entropy exploration, PixelCNN,
PixelRNN, gated PixelCNN and MADE. Every changed target builds. The CNN,
autoencoder, generative and RL sections now own all their loops. This does not
mean their per-step methods, bindings or language ports are complete.

Before/after comparisons preserve 261 archives containing 1,902 tensors on
MNIST and generated ConvLSTM data, plus six archives/370 tensors on a named
synthetic PointNet fixture. Entropy exploration preserves the full seeded
rollout/evaluation trace. Video and flow numerical checks pass; their real
training assets are not installed, so those checks are not training validation.
ConvLSTM also prints the full held-out target-pixel confusion matrix.
Capsule/STN parity uses their original one-epoch quick budget; their expanded
workflows also pass explicit five-epoch runs. The unused bundled update methods
are removed from the shared PointNet, optical-flow and binary-pixel headers.

The two competitive autoencoders now use their canonical C++ forward functions,
discrete winner masks and numerical fixtures through SWIG instead of duplicated
R/Python models. Both pass five epochs in all three languages. All 66 saved
archives (274 tensors) and four nested histories are byte-identical across the
languages. The comparison command now checks variant histories recursively.

`just nn-mnist-assets` downloads checksum-verified original MNIST IDX archives
and prepares balanced, disjoint 1000/200/200 autoencoder pools. Selection is
deterministic and records every original source index in a manifest; it does
not claim to reproduce tflow's selection. Repeated preparation preserves the
NPZ bytes. Set `DATAMUNGE_DATA_DIR="$PWD/build/datasets"` when using these assets.

The earlier batch removes generic CIFAR workflows from 18 examples and expands
25 detection/segmentation workflows into their example files. These workflows
honor an explicit epoch budget in quick mode. Nine marked-sequence recurrent
examples now own their epoch loops and parameter-gradient/optimizer updates.
All 75 affected C++ targets build. At five epochs, the nine recurrent examples
preserve all 270 saved `.npz` archives byte for byte against their previous C++
implementations. They still need component bindings and language ports.

GRU, LSTM, peephole LSTM and layer-normalized LSTM now also own their loops in
`main`, as do the gradient-clipping, recurrent-dropout and truncated-BPTT
comparisons. The remaining bundled peephole update is expanded into explicit
forward/loss/autograd/optimizer calls. All seven targets build and pass at five
epochs, including every policy/window/dropout variant. Their 1,296 archives
match the prior C++ implementations byte for byte, and every held-out binary
classifier prints its full confusion matrix. These seven still need bindings
and native language ports.

Nine normalizing-flow examples now also own their loops in `main`, including
the three distributions in `flow_density_2d` and both planar/radial variants.
All nine targets build; all 314 saved archives match their previous C++ versions
byte for byte at five epochs. Their per-step bundles and language ports remain
unfinished.

`resnet_bottleneck` (all three depths), `senet`, and `eca_net` now implement
their workflows in Python and R over SWIG-bound components. The latest batch
adds `alexnet`, `xception`, `densenet`, `ghostnet`, `mobilenet_v2`, `mnasnet`,
`convnext`, `efficientnet`, and `efficientnet_v2`. A further batch adds
`preactivation_resnet`, `pyramidnet`, `res2net`, `resnext`, `revnet`, `cbam`,
`non_local`, `octconv`, `repvgg`, and `sknet`. All twenty-two examples pass
C++, Python and R validation at five epochs on CIFAR-10, using 32 fit,
32 validation and 32 test rows in quick mode. The twenty-four models' 8,812
parameter/state tensors, 46 checkpoint archives and twenty-four histories match
byte for byte across all three languages. These small runs check implementation
parity; they do not establish full-dataset classification quality.

The architectures share their C++ numerical diagnostics through
`common/cnn_validation.hpp` and `common/residual_validation.hpp`, bound under
`<example>_checks` namespaces. RevNet uses the reversible backward pass during
fitting; the non-local and selective-kernel configuration hooks stay in C++.
Fitting workflows remain explicit in each language's example. The Python/R
resize helpers marshal spatial pairs into the existing bound `ImageSize`
struct, preserving C++ resize values and input derivatives.

SWIG array accessors no longer disappear behind ignored members. R component
enums retain their types, reserved C++ `in` fields generate parseable R proxies,
and numeric scalar arguments accept ordinary R integer values for both float
and double. Validation fingerprints now include shared model headers, language
helpers and component interfaces, so changes invalidate stale evidence.
The shared component changes invalidate the earlier CNN validation
fingerprints. Only records matching the current source/API fingerprint remain
valid; earlier batches need revalidation after shared data-preparation changes.

Evidence lives in `build/nn-language-validation`: `residual-ports-validation.log`,
`preactivation-cpp-retry.log`, `residual-output-comparison.json`,
`residual-parity-counts.json`, `recurrent-cpp-validation.log`,
`gated-parity/summary.json`, `variant-parity/summary.json`,
`recurrent-parity/summary.json`, `flow-parity/summary.json`, and the language/test
logs. The latest evidence also includes `vision-ae-parity/summary.json`,
`binary-parity/summary.json`, `conv-entropy-parity/summary.json`,
`competitive-ports-validation-final.log`, `competitive-final-recheck.log`,
`vision-binary-cpp-validation-final.log`, `binary-cpp-final-recheck.log`,
`point-flow-final.json`, and `competitive-output-comparison.json`.
The recurrent evidence includes `all-final-recurrent-build.log`,
`final-recurrent-parity/summary.json`, `final-recurrent-replay/summary.json`,
`final-recurrent-python-suite.log`, `final-recurrent-r-suite.log`,
`memory-python-oracle.json`, `memory-r-oracle.log`, and the memory
component/package test logs.
Before automatic-data preparation was added, the full Python suite passed
86 tests, including four compiler-oracle checks; the R binding suite passed
333 assertions. The memory evidence includes `memory-port-output-comparison.json`,
`memory-native-recorded-replay.log`, `memory-ports-python-suite.log` and
`memory-ports-r-suite-final.log`. `just test-nn-api-python`
runs these Python API/workflow tests without building the optional experiment
bridge. The validation runner now honors `--timeout` for C++ as well as Python
and R: the pre-activation C++ reference timed out during concurrent testing at
the former fixed 180-second cap, then passed its unchanged five-epoch budget
with the requested longer timeout.

```sh
just nn-mnist-assets
DATAMUNGE_DATA_DIR="$PWD/build/datasets" just validate-nn-ports k_sparse_autoencoder winner_take_all_autoencoder --classify k_sparse_autoencoder=no --classify winner_take_all_autoencoder=no
just nn-compare k_sparse_autoencoder winner_take_all_autoencoder --include-history
DATAMUNGE_DATA_DIR="$PWD/build/datasets" just validate-nn-ports resnet_bottleneck senet eca_net --threads 2
DATAMUNGE_DATA_DIR="$PWD/build/datasets" just validate-nn-ports preactivation_resnet pyramidnet res2net resnext revnet cbam non_local octconv repvgg sknet --threads 2 --timeout 600
just validate-nn-ports gru lstm peephole_lstm layer_norm_lstm gradient_clipping recurrent_dropout truncated_bptt --languages cpp --timeout 600
just test-nn-api-python
just nn-compare resnet50_compact_cifar10 resnet101_compact_cifar10 resnet152_compact_cifar10 se_resnet50_compact_cifar10 eca_resnet50_compact_cifar10 --include-history
just nn-audit --require-complete
```

The CIFAR commands prepare missing official binary batches under
`$DATAMUNGE_DATA_DIR/cifar10`. The last command deliberately fails while any
workflow, binding or validated port remains unfinished; `just nn-audit --check`
only enforces the incremental ownership checks.

## Previously validated batches

At an earlier checkpoint, thirty-six examples had validated native Python and
R implementations: `cnn_mnist`,
`convolution_im2col`, `undercomplete_autoencoder`, `tensor_basics`, `autodiff`,
`linear_regression`, `polynomial_regression`, `sessions`, `variables_initializers`,
`principal_components`, `logistic_regression`, `softmax_mnist`, `mlp_mnist`,
`learning_rate_schedules`, `gradient_clipping_accumulation`, `adaptive_optimizers`
`advanced_optimizers`, `sgd_momentum`, `training_loop` and
`early_stopping_checkpoints`, `dense_layer`, `activations`, `embedding`,
`pooling`, `dropout`, `normalization_layers`, `batch_norm`, `conv_layers` and
`weight_spectral_norm`, `positional_encodings`, `attention`,
`scaled_dot_attention`, `attention_masks`, `attention_visualization` and
`multi_head_attention` and `transformer_positions`.
All thirty-six pairs passed both languages and their C++ references. Consult the
ledger for current pass records and logs; the overall migration remains unfinished.

Validation artifacts live under `build/nn-language-validation`, with numerical
outputs under `output/nn/language-migration/{cpp,python,r}/quick`. Eighty-eight
persisted tensor arrays from MNIST classifiers, the autoencoder and early-stopping
checkpoints match byte for byte in all three languages at five epochs.
Polynomial bias/variance summaries agree between Python/R within
1e-11 and with the six-digit C++ output within 5e-6. PCA checks compare known
variances, principal axes up to sign, reconstruction energy, whitened covariance,
kernel eigenproblem residuals and out-of-sample batching invariance.

Of 247 persisted attention arrays, 246 match byte for byte across all three
languages; the remaining bias array differs by at most 7.45e-9, with identical
outputs and gradients. A further 107 arrays initialized without any TensorFlow
fixtures match exactly. The installed R package can run these scripts outside
the checkout. All 21 C++ attention tests, 46 targeted Python tests and
112 R assertions passed.

The attention ports implement the masks, experiments, diagnostics and finite
differences directly in each language. They cover empty rows, padded queries,
causal cache offsets, shared-input self-attention derivatives, fixed positional
rotations, additive attention, and key-removal interventions. The pointer
classification experiment prints a six-class confusion matrix after five updates.
Public attention operations are bound through SWIG. Bulk standard-library normal
draws preserve the exact scalar random stream, including its cached odd sample.
The core now rejects zero key widths before scaling. Python extension builds
also detect changed immutable library paths and relink cached objects correctly.

The optimizer ports cover schedule endpoints and invalid arguments, seven
scheduled SGD trajectories, unequal microbatch accumulation compared with all
full-batch Adam moments, three independent fused-kernel reference formulas,
seven adaptive methods, trust ratios, Lion's old-momentum sign convention and
Adafactor's factored moments and slot-memory counts. All eleven adaptive/advanced
fit summaries match the C++ results within its printed precision. Forty prescribed
gradient updates in the kernel diagnostics test formulas; the fitting budgets are
five updates. Both older C++ autodiff and linear-regression demonstrations now
accept explicit epoch/step budgets instead of always running 100/180 steps.

The SGD/momentum/Nesterov example preserves the original standard-library shuffle
sequence. The training-loop example tests mid-epoch sampler restore, three future
shuffles, short batches and sample-weighted online metrics. Early stopping tests
monitor decisions separately from its fitting budget: five updates need not reach
the validation minimum or exhaust patience. A two-update prefix and three-update
resume reproduce every model/Adam value exactly. Malformed moment shapes and
missing checkpoint files leave all live model/optimizer values unchanged. The
scalar Adam checkpoint example validates its own slot schema because the general
optimizer deliberately defers shape validation until the next step.

The layer ports preserve all forward-only diagnostics at their original sizes:
24-layer initializer-scale comparisons, all thirteen activation formulas and
derivatives, 200,000-sample statistics, 30-layer SELU propagation, 2-D/3-D pooling
windows and gradients, translation tolerance, sparse embedding gradients and
masked means. Skip-gram and dropout fits use five development updates; their
original long-budget convergence checks remain conditional on sufficient training.
The skip-gram context classifier prints the same 12x12 confusion matrix in all
three languages over all 10,400 synthetic corpus pairs. This is a corpus fit,
not held-out generalization. All 35 reported dense-layer RMS values, 78 activation
statistics, six dropout fits and eleven Monte Carlo uncertainty values match C++
at its printed precision. Evidence is recorded in
`build/nn-language-validation/layers-parity.json` and the per-example ledger.

Layer/RMS/group/instance normalization now includes grouped hand-computed
moments, sequence inputs, independence from other batch rows, group-count limits
and shift/scale invariances. Batch normalization checks biased normalization
versus its reported unbiased variance, moving-state inference, a 24-layer
forward/backward stack and all input/scale/shift derivatives. Its development
budget updates moving statistics five times; it does not optimize model weights.
The 300-batch reference checks remain available at that exact diagnostic budget.
Convolution examples implement independent N-dimensional cross-correlation loops
for ordinary and depthwise kernels, and check transpose gradients, checkerboard
overlaps and dilated kernels with holes. Weight/spectral normalization implements
its reparameterizations directly in each script, including direction/gain
closed-form gradients, persistent power iteration, SVD and Lipschitz comparisons.
Cross-language comparison details are recorded in
`build/nn-language-validation/normalization-parity.json`.

## Example workflow contract and bound components

Every canonical example should have the same shape in all three languages:

1. **The C++ example owns its workflow.** Option parsing, data setup, the
   epoch/batch loop, the per-step forward -> loss -> gradient -> optimizer
   update, evaluation, checkpoint selection and the report are written out in
   the `.cpp` file: in `main`, or in one file-local function taking plain
   parameters when the file runs several variants (depths, versions). A generic
   harness parameterized by callbacks (`LogitsBuilder`, `TrainingBuilder`,
   `MarkedDemo` specs, model factories) hides exactly the loop the ports need,
   whether it lives in a shared header or was pasted into the file. See
   `examples/cpp/source/nn/.../resnet_basic.cpp` (several depths) and
   `cnn_cifar10.cpp` (one variant, loop in `main`).
2. **Shared headers provide components only** (model forward functions and
   configs, data readers, metrics, numerical checks) and are bound through SWIG
   in `examples/bindings/nn_components.i`, which `prebindings/nn/src/nn.i`
   includes. Python gets them as `pydatamunge.nn_components.<namespace>` and R
   as `datamungerNN::nn_components$<namespace>`, mirroring the C++ namespaces
   (`image_training::ImageBatch` is `image_training.ImageBatch` /
   `nn_components$image_training$ImageBatch`). Callback-typed functions are not
   bound (R has no directors); functions returning a builder are wrapped as a
   callable handle (`resnet_demo.CompactResNet(depth)` -> `model(params, state,
   x, training)` in Python, `model$call(...)` in R). Header-only, so the core
   library is unchanged; `setup.py` and `scripts/prebuild_nn_r.R` add
   `examples/nn` to the include path and stage the headers into the R package.
3. **The R and Python examples implement the same loop** over those bindings,
   line for line, and print the same report. `resnet_basic` is the reference:
   its C++, Python and R runs print identical numbers.

`just nn-audit` (`scripts/nn_example_audit.py`) classifies all 414 examples:
where the loop lives (`main`, `local`, `pasted_harness`, `header`, `none`),
whether the per-step logic is hidden in a header model method
(`model.Run/Update/Train`), which shared headers the example uses and which of
them are bound, plus the ledger's Python/R status. It writes
`examples/nn/example_audit.json` and `docs/nn-example-audit.md`;
`tests/python/test_nn_example_workflow_ownership.py` fails when a shared header
owns a training loop or when an example recorded as compliant regresses.

The headers currently bound are listed through `%include` directives and
`bound-header:` annotations in `nn_components.i`; add a header there (with `%ignore` for std::function hooks
and indexed accessors for std::array members, as the existing entries do) when
its examples are ported. Per-example `binding-adapter: <name> <header>`
annotations can identify a facade that covers all required operations while
retaining the transitive unbound-header inventory. SWIG must parse that header;
the loop, step, native-port and validation gates still apply independently.
The four memory examples use this mechanism.

## Binding details for subsequent ports

The SWIG interface is `prebindings/nn/src/nn.i`. Python imports `pydatamunge.nn`;
R accesses `datamungerNN::nn`. R vector-valued tensor arguments can use
`nn$tensor_vector(list(...))`, and tensor-vector results use `nn$tensor_list`.
R autograd accepts output lists and explicit `grad_outputs` for vector-Jacobian
products. Use `nn$no_grad(function() ...)` around parameter mutations.
Learning-rate functions are available as `nn.schedule.*` in Python and
`nn$schedule$...` in R. Global-norm clipping returns `(gradients, norm)` in Python
and a pair with `$first`/`$second` in R. Keep the R pair alive while using its
gradient vector; clone tensors if storing them past their parent's lifetime.

Tensor fluent mutators return owned copies of the reference-counted handle;
returning borrowed pointers breaks temporary receivers. Copies must share the
same tensor implementation. R scalar double arguments must accept ordinary R
integer literals as well as numeric values.

`nn::Generator` preserves the library's existing random algorithm. The separate
header-only `nn::Mt19937` and `nn::Mt19937_64` expose standard-library float/double
normal and uniform streams, shuffle and modulo draws for the teaching datasets.
Neither replaces Generator. Compiler-oracle tests check both engines' mixed
normal/uniform streams and repeated shuffles against standard C++. Keep dataset
algorithms in the language scripts rather than adding experiment-specific C++
generators.

Normal regeneration preserves native scripts. `scripts/prebuild_nn_api.py`
refreshes the canonical Python catalogue through `scripts/nn_migration_status.py`.
Legacy reference-generation tools must not shrink it back to the old 165-example
CNN/autoencoder inventory.
