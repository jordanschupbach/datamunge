# Native NN example migration

The active goal is genuine Python and R implementations of every canonical NN
C++ example. Each example owns its option parsing, data setup, epoch/batch loop,
forward/loss/update ordering, evaluation, checkpoints and reporting. Shared C++
model functions, data readers and numerical diagnostics are exposed through
SWIG and reused by the matching R and Python loops. Shared high-level experiment
`Run` functions and compiled example adapters do not count as language ports.


The compact masked Gaussian conditioner and MAF now use canonical
`nn.MafMasks`, `nn.MafConditioner`, `nn.MafDensity` and `nn.MafInverse` (R uses
`$`). Native Python/R tests verify strict own/future input exclusion, owning flat
fields through collection, higher graphs, sequential inverse recovery and
explicit host-owned Adam with all160 masked kernel coordinates unchanged.
The core operations and complete C++ report program reproduce the source
exactly; the completed literate report replaces its standalone C++ original. These APIs do not
mark a full independent Python/R experiment loop as ported.

The compact affine IAF now uses canonical `nn.IafDensity` and `nn.IafInverse`
(R uses `$`), sharing `MafMasks`/`MafConditioner`. Host tests verify reversed
source warmup, recovered owning stage fields, live higher paths, parallel
inverse recovery and host-owned Adam with every masked coordinate unchanged.
The completed literate report replaces its original executable and unused
legacy IAF/MAF helpers. Full independent Python/R experiment-loop ports remain
unfinished; this source-exact affine density study makes no VAE/ELBO claim.

The rational-quadratic spline flow now uses canonical
`nn.RationalQuadraticSpline`, `nn.SplineCoupling`, `nn.SplineDensity` and
`nn.SplineInverse` (R uses `$`). Host tests cover configurable bins/bounds,
source-rounded Double precision, actual input derivatives, owning bin/stage
fields through collection, higher paths and explicit host-owned Adam. The
full C++ program matches every original record, including fixture input/head
gradients. Its verified 15-block literate report replaces the standalone original
and unused legacy spline helper; full independent Python/R experiment loops
remain unfinished.

The planar/radial maps now use canonical `nn.PlanarFlow`, `nn.RadialFlow`,
`nn.ElementaryFlow`, `nn.PlanarRadialDensity` and `nn.PlanarRadialInverse`
(R uses `$`). Both families' host tests verify owning fields/higher graphs,
explicit Adam and parameter paths through inverses. Planar higher graphs follow
the source stopped Newton correction, not an exact implicit Hessian; radial
center derivatives retain the source finite convention. The complete C++
program is source-exact for both independently fitted variants. The verified
24-block literate report replaces its standalone original and unused legacy
helper; full independent Python/R experiment-loop ports remain unfinished.

The continuous flow now exposes canonical `nn.CnfField`, `nn.CnfIntegrate`,
`nn.CnfDensity` and `nn.CnfInverse` (R uses `$`). Host tests check deterministic
probe contractions, source Float32 times, owning quarter/field graphs, explicit
Adam and reverse-solver parameter gradients. Full unrolled RK4 retains first/
higher paths; no continuous adjoint or exact finite-map inverse/logdet is implied.
Only exact/paired modes approximate likelihood; fixed single-probe modes are
diagnostic integrals. The complete C++ program matches all original outputs and
updates. Its verified 18-block literate report replaces the standalone original
and unused legacy CNF helper; full independent Python/R experiment-loop ports
remain unfinished.

The three-family density study now uses canonical `nn.FlowStudyPoints`,
`nn.FlowStudyLogProbability` and `nn.FlowStudySampleDiagnostics` (R uses `$`),
with existing RealNvpDensity/RealNvpInverse. Native host tests verify all three
explicit transforms, erfc tails, rounded boundary points, half-open checker
support/negative-infinite reference log density, stable mixture tails, inclusive
mode thresholds, owning detached records and invalid/overflow cases. Complete
C++ loops reproduce all three original fits and quality summaries. The verified
14-block report replaces its original executable and last-consumer affine/flow
helpers; full independent Python/R experiment-loop ports remain unfinished.

The compact DDPM operations now use canonical `nn.CosineDiffusionSchedule`,
`nn.DiffusionExtract`, `nn.DiffusionTimeEmbedding`, `nn.DiffusionNoised`,
`nn.DdpmEpsilon` and `nn.DdpmReverse` (R uses `$`). Native host tests verify
owning schedule/predictor/posterior fields, applicable higher graphs, source
Float32 rounding even for Double, explicit Adam, exact last zero-innovation
behavior, clipping/raw estimates, strides, invalid math and warmed RNG/store
immutability. Complete editable C++ loops reproduce every original state and
update. Report narrative/figures and retirement remain pending; full independent
Python/R experiment-loop ports remain unfinished.

## Inventory and validation

`just nn-language-status` refreshes `examples/nn/language_status.json`. It scans
both canonical source roots: sections 1 and 2 under `examples/cpp/source/nn`, and the
remaining numbered sections under `examples/nn`. The ledger covers the remaining
standalone C++ programs; its total decreases as verified literate reports replace
them. The original inventory, replacement paths and retirement evidence are kept
separately in [`mepml/nn-conversion-status.json`](mepml/nn-conversion-status.json).
Native Python/R scripts remain runnable after their C++ counterpart is retired;
the Python catalogue discovers those scripts independently of the live C++ list.

The beta-VAE reusable Gaussian operations and fitting-only latent factor probes
now belong to the canonical C++ API, exposed as `nn.BetaVariationalAutoencoder`,
`nn.GaussianVaeDecode`, `nn.GaussianReparameterize`, `nn.DiagonalGaussianKl` and
`nn.LatentFactorDiagnostics` (R uses `$`). Their native Python/R tests cover
owning results, explicit caller-owned Adam, reductions and fitting-only probe
selection. The full beta-VAE literate C++ report replaces its standalone original;
this does not mark an unfinished Python/R experiment loop as native.

The class-conditioned Gaussian VAE now uses native
`nn.ConditionalVariationalAutoencoder` and `nn.ConditionalVaeDecode` (R uses `$`).
Both posterior and decoder receive the same supplied class; labels are required
conditions, not predictions. Binding tests cover inherited owning endpoints,
Int64 validation, Float64 condition rows, code-only decoding and caller-owned Adam.
Its completed literate report replaces the standalone original and obsolete
reference launchers; no full Python/R experiment loop is marked native by this API addition.

The convolutional Gaussian VAE exposes native
`nn.ConvolutionalVariationalAutoencoder` and `nn.ConvolutionalVaeDecode` (R uses
`$`). The explicit sampled z is the only decoder input; pooled encoder maps are
inspection endpoints. Bindings retain all Gaussian/spatial results and decoder-only
stages through collection, with Float64/non-square geometry, discrete geometry
validation and caller-owned Adam checks. Its verified literate report replaces
the standalone original and reference launchers; a reusable API is not a full
native host-language experiment port.

The total-correlation Gaussian forwards expose native
`nn.FactorVariationalAutoencoder` and `nn.BetaTcVariationalAutoencoder`, with
`nn.EmpiricalTotalCorrelation`, `nn.FactorDiscriminator`,
`nn.FactorDiscriminatorLoss` and `nn.PermuteLatentColumns` (R uses `$`). The first
keeps the z path live through a caller-frozen D; the second re-encodes all supplied
fitting references with live shared weights. Host tests cover explicit alternating
optimizers, detached products, signed ratios, live reference gradients and nested
Gaussian ownership after collection. Its verified report replaces the standalone
original, unused helper and reference launchers. Complete independent Python/R
experiment loops remain unfinished.

The conditional-flow posterior exposes native
`nn.ConditionalAffinePosteriorFlow` and `nn.FlowVariationalAutoencoder` (R uses
`$`), with `nn.GaussianVaeDecode` for final-code-only decoding. Both inverse and
forward use the same leaves and fixed context for the code Jacobian; context
still receives gradients during fitting. Tests cover all eighteen owning named
endpoints through collection, live context, density signs, Float64, explicit host
Adam and prior decoding without posterior inference. The verified literate report
replaces its original and reference launchers. The old shared include was retained
for IWAE and is now removed following that conversion. Complete Python/R
experiment-loop ports remain unfinished.

The categorical VAE exposes native `nn.GumbelNoise`, `nn.GumbelCategorical`,
`nn.CategoricalUniformKl`, `nn.GumbelDecodeSoft`, `nn.GumbelDecodeIds` and
`nn.GumbelCategoricalAutoencoder` (R uses `$`). Explicit local noise banks,
source Float32 rounding and separate soft/hard diagnostics preserve the original
trajectory. The soft reconstruction/discrete-KL sum is a continuous surrogate;
no straight-through gradient or exact discrete-ELBO claim is inserted. Tests
retain all bank/forward fields through collection, exercise live Float64 inputs,
full-uint32 seeds, both target-free decoders and caller-owned Adam. The verified
report replaces the original, unused helper and reference launchers; complete
independent Python/R experiment loops remain unfinished.

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

The [NN report ledger](mepml/nn-conversion-plan.md) now records 95 retired
originals and 224 reports present out of 416 original sources; 192 reports still
need conversion. The [standalone audit](nn-example-audit.md) describes the 321
remaining C++ programs. The 414-source language checkpoint below predates the
verified report retirements and is retained as historical evidence.

The earlier audit covered all 414 examples. There were 110 native Python scripts
and 95 native R scripts; these counts describe implementations, not completed
validation. At that checkpoint, all 414 C++ examples owned their orchestration; no generic
training runner remains in a shared header or pasted into an example. The
earlier audit flagged 117 examples that call bundled per-step methods (including
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

The functional NN bindings now expose the native sentence-VAE components:
`SequenceEncoder`, `SequenceDecoderInputs`, `SequenceTeacher`,
`SequenceVariationalAutoencoder`, `AnnealedSequenceObjective` and
`SequenceGenerate`. All eighteen forward, five teacher and four generation
fields survive garbage collection. PAD-masked encoding, source 22 leaves/3,243
scalars, explicit Gaussian/categorical draws and continuous BPTT/noise paths are
shared across languages. `CollapseBeta` and `CollapseStatistics` supply the
explicit source-rounded schedule and owned host-double finite-mixture summaries;
negative Monte Carlo values and self-inclusion bias remain explicit. There is no
hidden optimizer or selection. Both legacy C++ examples delegate their model and
diagnostics mathematics to these APIs; their native Python/R experiment loops and
report retirement still remain. Focused native/binding checks and the 55-archive
source comparison pass; evidence is under
`build/nn-report-conversion/sequence_vae_core/`.

`just nn-sequence-vae-assets` prepares the local exhaustive sentence grammar
through `dm::datasets::prepare_example_data("sequence-vae-assets")`. The shared
cache gains `sequence_vae/sentences.npz`, `vocabulary.txt` and `manifest.json`:
432 unique sentences, split 288/72/72, with nineteen fixed token IDs and seven
steps. All sixteen arrays are Int32; decoder inputs use BOS followed by the
previous target, and lengths include EOS but exclude PAD. Rows use ascending
source IDs within each compositional split. This is procedural teaching data;
the manifest records its grammar and ordering. Caller-supplied archives and
vocabularies are preserved, with unverified provenance when no manifest exists.

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

The paired spatial hierarchy exposes native `nn.HierarchyPosterior`,
`nn.ConditionalGaussianKl`, `nn.HierarchyNvaeCell`, `nn.HierarchyDecode`,
`nn.HierarchyPrior` and `nn.HierarchicalVariationalAutoencoder` (R uses `$`).
Weights and BN moving buffers live in separate caller-owned stores. Host tests
cover probe versus update passes, optimizer exclusion of state, image-free
ancestral decoding, live Float64 gradients and all 19 owning forward endpoints
through collection. The verified report replaces the original, unused hierarchy
helper and reference launchers. The ladder helper was retained
for its conversion and is now removed; independent Python/R experiment-loop ports remain unfinished.

The importance-weighted Gaussian VAE exposes native `nn.IwaeLogMeanExp`,
`nn.ImportanceWeightedBound` and `nn.ImportanceWeightedAutoencoder` (R uses `$`).
All five bound and eleven forward fields own lifetime. Host tests exercise live
Float64 image/noise/density paths, sample-major layouts, explicit caller-owned
Adam and image-free `GaussianVaeDecode`. The loss uses sampled density ratios
and ordinary pathwise gradients; analytic KL is diagnostic only. Its verified
report replaces the original, unused IWAE/flow helpers and reference launchers.
Independent Python/R experiment-loop ports remain unfinished.

InfoVAE/MMD-VAE exposes native `nn.ImqKernel`, `nn.ImqMmd`,
`nn.InfoVaeObjective`, `nn.InfoVaeDecode` and `nn.InfoVariationalAutoencoder`
(R uses `$`). All seven MMD and sixteen full-forward fields own lifetime. Host
tests exercise live Float64 image/noise/prior paths, signed unequal-count MMD,
image-free tanh decoding and caller-owned Adam. The general training objective
is separate from the diagnostic negative ELBO. Its verified report replaces
the original, unused InfoVAE helper and reference launchers. The WAE kernel
compatibility import remains and delegates to canonical C++ after original-run
parity. Independent Python/R experiment-loop ports remain unfinished.

The Ladder VAE exposes native `nn.LadderPrecisionFusion`, `nn.LadderPrior` and
`nn.LadderVariationalAutoencoder` (R uses `$`), reusing conditional Gaussian KL
and `GaussianVaeDecode`. All fusion, six ancestral and sixteen full-forward
fields own lifetime. Host tests exercise Float64 paths through evidence/prior,
images/noise, lower-first noise interventions, image-free lower-only decoding
and caller-owned Adam. The verified report replaces the original, unused Ladder
helper and reference launchers. Independent Python/R experiment-loop ports
remain unfinished. Native sentence preparation, corpus validation and model APIs are now available;
posterior-collapse and sequence-VAE retirements are verified.

The posterior-collapse report now retires its standalone C++ source, unused
posterior helper and generated reference launchers. Native corpus functions
`SequenceVocabulary`, `LoadSequenceVaeData`, `SequenceDataArrays`, `SequenceText`
and `SequenceGrammarValid` provide the complete checked local grammar and owning
Python/R pool lists; `prepare_example_data` delegates native preparation. Corpus
fields survive collection of both list/vector and backing files. All 105 focused
Python checks and 730 R expectations across 22 files pass. The report preserves
all 35 source archives/2,265,740 values, full TSVs and non-timing JSON values.
At that checkpoint the sequence-VAE example retained its 20-archive baseline
and shared helper; its verified report now retires both source and helper. These
component checks do not claim completion of independent Python/R experiment-loop
ports. Evidence: `build/nn-report-conversion/posterior_report/`.

The sequence-VAE report replaces its standalone C++ source, unused compatibility
helper and generated reference launchers. Native `nn.MidpointUniformNoise`
(R uses `$`) supplies explicit owning Float64 categorical uniforms from a fresh
full-uint32 MT19937 midpoint stream. Scalar/empty shapes and negative/overflow
contracts match C++; Float64 avoids Float32 endpoint rounding. All 106 focused
Python checks and 743 R expectations across 22 files pass. The complete report retains source training,
all 20 archives/1,454,190 values, vocabulary, histories and non-timing metrics
exactly, with independent BPTT/Adam, endpoint, validation and generation checks.
These component checks do not complete the independent Python/R experiment loops.
VAE-GAN now has its verified report and native components below. Evidence:
`build/nn-report-conversion/sequence_report/` and `vae_gan/`.

The VAE-GAN report retires its standalone C++ source, unused helper and generated
reference launchers. Native `nn.VaeGanFeatureNll`, `nn.VaeGanDecode`,
`nn.VaeGanDiscriminator` and `nn.VaeGan` (R uses `$`) provide the detached real
feature target, code-only generation, feature/logit record and all nineteen
owning endpoints. Float64 image/noise/prior paths and unequal prior counts remain
live; the caller owns three separate Adam states and recomputes isolated D/E/G
updates. Pixel NLL is diagnostic, not an optimized image ELBO. All 111 focused
Python checks and 775 R expectations across 23 files pass. All 24 source archives/3,078,235 values and
histories are retained; only the two Float64 summary MSE reductions have roundoff
below 1.1e-14. These component checks do not complete independent Python/R
experiment-loop ports. The vanilla VAE baseline is saved for the next conversion.
Evidence: `build/nn-report-conversion/vae_gan_core/` and `vae_gan_report/`.

The vanilla VAE report now replaces its standalone source and generated
reference launchers. Existing `nn.BetaVariationalAutoencoder` and
`nn.GaussianVaeDecode` (R uses `$`) supply beta1/latent2/hidden64 behavior;
five legacy math functions forward to the canonical Gaussian core and retain
the original eight-field compatibility record. The shared helper remains for
latent diffusion and other consumers. All 112 focused Python checks and 791 R
expectations across 23 files pass, including fields through GC, ten parameter
leaves, input/noise paths and caller-owned Adam. Source arrays and histories
remain exact; one report Float64 MSE reduction differs by 2.998e-15. Both VQ
source runs retain all 29 archives after shared delegation. These component
checks do not complete independent Python/R experiment-loop ports. Next:
canonical VQ components and report conversion. Evidence:
`build/nn-report-conversion/vanilla_vae_report/`.

The spatial VQ-VAE report retires its C++ source, unused legacy model/decoder
adapters and generated launchers. Native `nn.VectorQuantize`, `nn.VqLookup`,
`nn.VqDecoder`, `nn.VectorQuantizedAutoencoder`, `nn.VqUsage` and
`nn.VqMarkovPrior` (R uses `$`) expose discrete assignment, explicit gradient
routes, generic Float32/64 spatial inference and a copied fitting-code prior.
Nested fields survive GC; caller owns Adam and supplies Float64 prior uniforms.
All 116 focused Python checks and 830 R expectations across 24 files pass.
Both source VQ runs preserve all 29 archives/2,691,906 values and non-timing JSON
through shared core delegation. Report array/history parity is exact, with only
two summary MSE reductions differing below 3e-14. At that checkpoint shared compatibility remained
for VQ-VAE-2; its subsequent conversion removes both legacy VQ headers. These component checks
do not complete independent Python/R experiment-loop ports. Evidence:
`build/nn-report-conversion/vq_core/` and `vq_report/`.

The two-scale VQ-VAE-2 report now replaces its standalone source and generated
reference launchers. Native `nn.Vq2TopDecoder`, `nn.Vq2Decoder`,
`nn.VectorQuantizedAutoencoder2` and `nn.Vq2ConditionalPrior` (R uses `$`)
expose coarse-conditioned fine inference, two-grid-only decoding and copied
fitting-code conditional tables. Tests retain eleven fields and nested
quantizers through GC, exercise Float64/non-square geometry, live hierarchy
surrogates, both stopped dictionary paths, explicit uniforms and caller-owned
Adam. All 121 focused Python checks and 865 R expectations across 25 files pass.
All fifteen source archives / 1,932,136 values and TSV histories remain exact;
only two report summary MSE reductions differ below 3e-15. Both unused legacy
VQ headers are removed. These component checks do not complete independent
Python/R experiment loops. That checkpoint preceded the WAE-MMD conversion below.
Evidence: `build/nn-report-conversion/vq2_core/` and `vq2_report/`.

The WAE-MMD report now replaces its source and generated reference launchers.
Native `nn.WaeSquaredPixelCost`, `nn.WaeObjective`, `nn.WaeDecode` and
`nn.WassersteinAutoencoder` (R uses `$`) expose pixel-summed squared cost,
explicit signed discrepancy weighting and deterministic code-only decoding.
Tests retain six Tensor fields and the nested seven-field MMD record through
GC, exercise unequal Float64/strided input and prior counts, prior-only
regularizer dependence and caller-owned Adam. All 126 focused Python checks and
910 R expectations across 26 files pass. All 18 archives / 1,147,656 values and
TSVs remain exact, with only two report MSE reductions differing below 4e-15.
The unused WAE, beta-VAE and discrepancy compatibility headers are removed.
These component checks do not complete independent Python/R experiment loops.
Next: frozen-model restoration, whose zero-update baseline and independent
oracle are pinned. Evidence: `build/nn-report-conversion/wae_core/` and
`wae_report/`.

The frozen restoration report now replaces its standalone source and generated
reference launchers. Native `nn.VisiblePatchValues` and
`nn.CompositePatchValues` (R uses `$`) expose generic target-free selection,
preserving signed zeros and observed bits with selected continuous paths.
They reuse existing native MAE encoding/decoding, mapping and hidden loss.
All 130 focused Python checks and 941 R expectations across 27 files pass.
Four visible prerequisite programs reproduce the three original model files;
restoration itself adds zero updates. All four original archives / 4,704,817
values remain exact; only Float64 metric reductions have small roundoff.
The three unused C++ restoration/denoising/masked helpers are removed. These
component checks do not complete independent Python/R experiment-loop ports.
Next: locate AutoRec's original asset preparation before its numeric baseline.
Evidence: `build/nn-report-conversion/restoration_core/` and `restoration_report/`.

The encoder-pretraining report replaces its source and generated reference
launchers. Native FrozenEncoderFeatures, FitTransferScaling and
FrozenEncoderClassifier expose frozen owning state and caller-owned head
updates in Python and R. All 134 focused Python checks and 971 R
expectations across 28 files pass, including GC/ownership and stopped/live
paths. All 26 source archives / 2,849,964 values remain exact; report probability
NLL reductions differ by at most 3.11e-15. The unused C++ transfer adapter is
removed. Component tests do not finish independent Python/R experiment loops.
AutoRec still awaits its original preparation recipe. Latent arithmetic has a
frozen original baseline for the next independent report.
Evidence: `build/nn-report-conversion/pretraining_core/` and `pretraining_report/`.

Latent arithmetic now has a verified executable report replacing its standalone
source and generated reference launchers. Native LatentLerp, LatentDirection,
LatentClassMeans, LatentFlatten and UndercompleteDecoder expose signed CPU
Float32/64 arithmetic, stopped fitting summaries and code-only decoding.
All 139 focused Python checks and 1011 R expectations across 29 files pass.
Both archives / 1,448,809 values, every JSON value and all five original PNGs
remain exact. The unused C++ arithmetic adapter is removed. These component
checks do not finish independent Python/R experiment loops. AutoRec still
awaits its original preparation recipe; latent visualization has a frozen
baseline and independent PCA/retrieval oracle for the next conversion.
Evidence: `build/nn-report-conversion/latent_core/` and `latent_report/`.

Latent visualization now has a verified executable report replacing its source
and generated reference launchers. Native DoubleSampleRows, SvdPca,
NearestFitRows, NeighborOverlap and NeighborLabelAccuracy expose fitting-only
owning state, live query/inverse paths and label-free retrieval in Python and R.
All 144 focused Python checks and 1047 R expectations across 30 files pass.
Both original archives / 3,833,394 values are exact; report Float64 MSE reductions
differ below 3.5e-14, with timing remeasured and other JSON values exact.
The unused visualization and legacy tensor-PCA helpers are removed. Component
tests do not finish independent Python/R experiment loops. AutoRec awaits its
original preparation recipe; learned image codec is the next independent report.
Evidence: `build/nn-report-conversion/visualization_core/` and `visualization_report/`.

Learned image codec now has a verified executable report replacing its source
and generated reference launchers. Native CodecQuantizer, FitCodecQuantizer,
QuantizeCodec, DequantizeCodec, SymbolTensor, LearnedCodecDecode, CodecPacket,
PackCodec, UnpackCodec and DecodeCodec expose generic explicit Float32 ranges,
exact byte operations, live continuous decoder paths and stopped packet decode.
Nested range exports and Tensor handles survive parent collection. All 149 focused
Python checks and 1087 R expectations across 31 files pass. All three original
archives / 1359472 values, all 110972 bitstream bytes and 210 PNGs remain exact;
report Float64 MSE/PSNR reductions differ below 2e-15/1.73e-13. The complete file
is larger than the measured PNG baseline; payload-only rate is kept separate.
The unused C++ codec helper is removed. Component checks do not finish independent
Python/R experiment loops. AutoRec awaits its original preparation recipe;
reconstruction anomaly is the next independent report.
Evidence: `build/nn-report-conversion/codec_core/` and `codec_report/`.

Reconstruction anomaly now has a verified executable report replacing its source
and generated reference launchers. Native ReconstructionScores, InkScores,
NominalThreshold, ReconstructionRanking, RankAnomalies and AnomalyCorruption expose
owning stopped diagnostics, explicit calibration and strict/tied score evaluation.
All 154 focused Python checks and 1147 R expectations across 32 files pass.
Both original archives/3,198,145 values, every JSON/TSV value and all ten PNG strips
are exact; native canonical ROC/AP are reused without another ranking algorithm.
The report discloses reused validation for encoder selection/calibration, synthetic
paired pools, AP prevalence and all failures. The unused C++ anomaly helper is
removed. Component checks do not finish independent Python/R experiment loops.
AutoRec awaits its original preparation recipe; auxiliary-classifier GAN is next.
Evidence: `build/nn-report-conversion/anomaly_core/` and `anomaly_report/`.

Auxiliary-classifier GAN now has a verified executable report replacing its C++ source.
There were no Python/R launchers for this model. Native LogisticGanObjectives, ConditionalGanGenerate,
AuxiliaryGanDiscriminate, AuxiliaryClassLoss, AuxiliaryClassifierGan and GanCoarsePixels
expose explicit live components in Python and R. All 159 focused Python checks
and 1199 R expectations across 33 files pass, including selected-store
Adam writes and owned Tensor fields after collection. All ten original archives /
919949 values, TSVs, three PNGs and non-timing JSON values are exact. The complete
C++ loop keeps all 192 updates, fixed final epoch3 and the weak grey samples;
initial test inference is an explicit nonselecting control. Component checks do
not finish independent Python/R experiment loops. The unused model helper is
removed; shared GAN helpers remain. AutoRec needs its original ratings recipe;
BiGAN baseline/oracle and native primitive reuse are ready for the next report.
Evidence: `build/nn-report-conversion/acgan_core/` and `acgan_report/`.

Bidirectional GAN now has a verified executable report replacing its C++ source
and unused model helper; no host-language launchers existed. Native GaussianRingPoints,
BiganMap, BiganJointScore and BidirectionalGan expose explicit data and joint paths
in Python/R. All 164 focused Python checks and 1244 R expectations across
34 files pass, including owned fields after collection, unequal pools, selected
D/E/G Adam writes and E-independent G partials. All nine archives/6210 values,
TSVs and non-timing JSON values are exact; core delegation also preserves the
three source PNGs. The report uses annotated native scatter plots with all points,
equal aspect, explicit fixed windows/outside counts and weak outcomes retained.
The complete C++ loop has all 117 updates, fixed finalepoch 3 and unfitted cycle
residuals; no learned-inverse or convergence claim. Component checks do not finish
independent Python/R experiment loops. AutoRec needs its original ratings recipe;
conditional GAN's baseline, oracle and native reuse are ready for the next report.
Evidence: `build/nn-report-conversion/bigan_core/` and `bigan_report/`.

The compact dense GAN components now use native `DenseGan`, required
`DenseGanObjective`, `MlpGanGenerate`, `MlpGanDiscriminate`,
`LeastSquaresGanObjectives`, `HingeGanObjectives`, `ConditionalGan` and
`ConditionalGanDiscriminate` (R uses `$`). Python/R tests cover raw-score algebra,
exact hinge boundaries, owned five/seven-field endpoints, default parameter counts,
both-network class inputs, malformed inputs and host-owned D-then-fresh-G Adam
writes. The four complete C++ reports replace the standalone originals and unused
MlpGanModel class. These component APIs do not mark independent Python/R full
experiment loops complete.

InfoGAN components now use native `InfoPrior`, `InfoGanGenerate`, `InfoGanRecognize`,
`InfoGanRecognitionLoss` and `InfoGan` (R uses `$`). Generated code categories are
not true image labels. Tests cover exact raw words and rejection, final casts,
independent image-only Q, unit-variance loss, negative variational scores, owned
records, continuous gradients and explicit host-owned D-Q-G writes. The complete
C++ report replaces its original and unused helper; these component tests do not
finish independent Python/R full experiment loops.

Spectral GAN components now use native `cached_spectral_power_iteration`,
`cached_spectral_kernel`, `advance_cached_spectral_state`,
`SpectralGanDiscriminate` and `SpectralGan` (R uses `$`). Tests cover owning cache
records, live denominator gradients, stopped columns, signed floors, zero kernels,
initialization census, immutable inference and explicit state-D-fresh-G writes.
The complete C++ report replaces its source; shared SpectralDense remains for
Self-Attention GAN. Component tests do not finish independent Python/R experiments.

Self-Attention GAN components use native `cached_spectral_dense`,
`SpectralSpatialAttention`, `SelfAttentionGanGenerate`,
`SelfAttentionGanDiscriminate` and `SelfAttentionGan` (R uses `$`). The shared
layer also supplies spectral GAN dense math. Tests cover owned records, zero
gates, continuous graphs, initialization census, row probabilities, invalid
geometry and explicit D-state/D/G-state/fresh-G writes. The complete C++ report
replaces its source and both unused helpers. Component coverage does not finish
independent Python/R full experiment loops.

BigGAN components use native `capture_batch_norm`, `batch_norm_moving_values`,
`GaussianRejectionPrior`, `BigGanResize8`, `BigGanGenerate`, `BigGanDiscriminate`,
`ModifiedOrthogonalPenalty`, `BigGanCommitMoments` and `BigGan` (R uses `$`).
Tests cover audited rejection streams, detached moving candidates, owned nested
moments through parent/store collection, live continuous paths and five explicit
phase owners. Both game losses always use batch-statistics fake images even when
saved BN supplies displayed images. The complete C++ report replaces its source
and unused helper. Component coverage does not finish independent Python/R full
experiment loops.

DCGAN components use native `capture_affine_batch_norm`, `DcganGenerate`,
`DcganDiscriminate`, `DcganDiscriminatorPhase`, `DcganGeneratorPhase` and `Dcgan`
(R uses `$`). The fused BN returns detached sample-moment captures with live
output graphs; moving candidates use explicit Double .9 retention. Tests cover
stopped G during joint-BN D, batch-BN G through current saved-BN D, available D
weight paths with caller-selected G updates, flat-record lifetime and four
explicit write owners. The complete C++ report replaces its source and unused
helper section. Component coverage does not finish independent Python/R full
experiment loops.

Wasserstein components use native `MidpointUniform`, `WassersteinGenerate`,
`WassersteinCritic`, `WassersteinGan` and `ClipWassersteinCritic` (R uses `$`).
The eleven-field record retains actual/analytic Jacobians and explicit objectives;
penalty history through critic parameters is optional, while external real/alpha/G
penalty paths stop. Tests cover owning fields, higher derivatives, no-grad input
derivatives, both optimizers, all-bias clipping and three critic writes per fresh G.
The complete C++ report replaces its source and unused helper class. Component
coverage does not finish independent Python/R full experiment loops.

Progressive components use native `pixel_rms_normalize`, `ProgressiveActiveGroups`,
`ProgressiveGanGenerate`, `ProgressiveGanDiscriminate` and `ProgressiveGan` (R uses
`$`). Activity returns native group indices; the flat records expose genuine old/
new images, feature mixtures and full-scaled G loss. Tests cover owning fields,
higher paths, both alpha formats, stage activity, independent Adam counters and
capturing every active group before any selected phase write. The complete C++
report replaces its source and unused helper. This component coverage does not
finish independent Python/R full experiment loops.

Pix2Pix components use native `quantized_sobel_edges`, `Pix2PixGenerate`,
`Pix2PixDiscriminate` and `Pix2Pix` (R uses `$`). Conditions preserve Double-before-
quantization and inclusive zero-padded raw Sobel. Owning generated boundaries
retain both real concatenation skips; flat game fields retain paired L1 and
patch-mean logistic losses. Tests cover higher paths, source-only G, conditional
D, separate explicit Adam writes, exact skip order and invalid inputs before
leaf creation. The C++ report replaces the original and unused Pix2PixModel
adapter; pure shared Pix maps remain for CycleGAN. These component tests do not
finish independent Python/R full experiment loops.

CycleGAN components use native `RotateInvertGrayscale`, `SplitCycleDomains`,
`CycleGanDiscriminate` and `CycleGan` (R uses `$`). The flat domain record preserves
original IDs/untransformed B and live gathers. The19-field game permits unequal
unpaired rows and retains both coupled generator paths, full G adversarial means
and half-scaled critics. Tests capture both G vectors before either explicit Adam
write, verify the changed sequential control, store ownership, higher paths and
source-only translation. The C++ report replaces its original/helper; unused
legacy Pix maps are removed and shared PixConv remains for StyleGAN/StarGAN.
These component tests do not finish independent Python/R experiment loops.

StyleGAN components use native `modulated_conv2d`, `spatial_instance_normalize`,
`StyleGanMap`, `StyleGanSynthesize`, `StyleGanDiscriminate` and `StyleGan` (R uses
`$`). The flat primitive/synthesis/game records own live fields after parent
collection; all applicable higher paths survive. Tests check analytic modulation
partials, both source censuses, zero initial noise with learnable strengths,
style mixing, positive gain normalization, fixed-style/common-noise conditions,
explicit D then fresh G captures and update ownership. The complete C++ report
replaces its original/helper while shared PixConv remains for StarGAN. These
component tests do not finish independent Python/R experiment loops.

StarGAN components use native `UniformCategorical`, `StarDomainImages`,
`SplitStarDomains`, `StarGanGenerate`, `StarGanDiscriminate`, `StarDomainLoss`
and `StarGan` (R uses `$`). Five flat record types own live/detached fields as
specified; higher paths through both shared G uses survive parent collection.
Tests cover disjoint selected/unused identities, rejected RNG words, stable CE,
actual conditioned skips, image-only D, explicit fresh-G scheduling and store
ownership. The complete C++ report replaces its original/helper and retires the
last unused legacy PixConv header. These component checks do not finish full
independent Python/R fitting loops.

SRGAN/ESRGAN components use native `RelativisticAverageGan`, `AdversarialSRGenerate`,
`AdversarialSRDiscriminate`, `AdversarialSRFeatures` and `AdversarialSR` (R uses `$`).
Flat generated/game/loss records own live fields and detached optional BN captures.
Tests check source-rounded coefficients, frozen feature stops, target-independent
two-path G pixels, cold Adam resets, same-capture post-G BN commits, live opposite-
pool means and owning higher paths. The complete C++ report replaces its original
and retires unused legacy model code; only SRConv remains for SinGAN. These
component checks do not finish independent full Python/R fitting loops.

SinGAN components use native `SingleImageNoise`, `SingleImageAmplitude`,
`SingleImageGenerate`, `SingleImageDiscriminate` and `SingleImageGan` (R uses `$`).
Five/eight flat owning record fields preserve applicable higher paths and storage.
Tests cover shared RGB stream layout, detached Double RMS, signed rectangular
residuals, full-G/half-D factors, both G branches, D-then-fresh-G ownership,
cold clocks after weight warm-start and invalid inputs before leaves. Both
unused legacy single-image/adversarial-SR headers are removed after the complete
C++ report replaces the source. Full independent Python/R fitting loops remain
separate work; component checks do not establish them.

FrozenGraph component bindings now load immutable numeric GraphDef models from
files or bytes, expose names/operations/constants and evaluate named CPU feed/
fetch subgraphs without TensorFlow. Detached owning results survive model/feed
collection, declared dtypes/ports/byte bounds are checked, and the existing
ByteVector needs thin Python sequence conversion (R uses integer bytes).
Tests exercise constructor/file lifetime, duplicate outputs, detached storage,
malformed bytes and invalid endpoints. The original Inception graph is verified
against TensorFlow only in an isolated optional development oracle. Native
preparation uses inception-metric, including pinned archive/graph provenance and
rehashed completed caches. The diagnostics report and metric Python/R exports
remain unfinished; this is component infrastructure, not a complete conversion.
