# Neural-network examples as mepml reports

Data preparation policy: executable reports obtain their inputs through the
canonical C++ library (`dm::datasets::fetch()` or
`dm::datasets::prepare_example_data()`). Implement missing download, archive,
metadata and numeric-format support in datamunge; do not introduce Python data
scripts, manual archive placement, pip/venv setup, or calls into another checkout.
Use the shared configurable cache, retain verified source provenance, and keep
independent framework oracle generation in optional development tests.


Working plan for turning the C++ examples under `examples/nn/` (and later
`examples/cpp/source/nn/`) into polished reports under `docs/mepml/examples/nn/`,
one per example, in the same directory layout. The user asked for this on
2026-10-28, starting with `examples/nn/0_foundations`, "until everything in that
folder has its own". The checklist below is the state: the first unchecked
entry is the next report.

## Rules


### Tensor requirement for neural-network reports (2026-10-29)

Everything beneath `docs/mepml/examples/nn/` uses `dm::nn` tensors for model
computation: preprocessing, predictions, objectives, inference and fitting.
DataFrames may load datasets and present results; host containers may hold row
indices or plot values. Do not substitute a `dm::stats` model or a host-array
implementation for a tensor example.

For models with learnable parameters, show the learnable tensor leaves or
ParameterStore, optimizer kind and learning rate, objective, and explicit
iteration/minibatch loop calling the tensor optimizer (`minimize` or `step`).
Do not hide the fitting loop inside a model constructor or a reusable helper.
Closed-form solutions, QR/SVD and EM may be explained or used as independent
checks, but must not replace the demonstrated optimizer-based fit. Explain
constraints/projections and nonsmooth proximal steps explicitly. Methods with
no fitted parameters, such as nearest neighbours, should say so and show their
tensor inference. Formatting must keep statements and control flow readable.

### General conversion rules

- Same standard and procedure as `docs/mepml/conversion-plan.md` (abstract opens
  with the problem; method with derivations; simple hand-checkable examples with
  figures; theory checks; optional dataset; conclusion; references; prose written
  after running and checked against the output).
- Layout is 1-1: `examples/nn/<section>/<group>/<name>.cpp` becomes
  `docs/mepml/examples/nn/<section>/<group>/<name>.mepml`, with its figures in
  `docs/mepml/examples/nn/<section>/<group>/figures/<name>_*.svg`.
- Port reusable logic from the example and from `examples/nn/common/` into
  `datamunge::nn` (or the matching datamunge module), documented and tested.
- Report blocks use `#include <dm/dm.hpp>` and the standard flags.
- Generate executable model architecture figures through `nn::Module::plot` or
  `nn::plot_model`, using inference mode and the reported configuration. Keep
  local SVGs and the complete generating block; label shape probes and grouped
  stages explicitly. Retain explanatory TikZ diagrams when they teach the method.
  Keep architecture graphs left to right with counterclockwise node labels; choose
  Flow, Graph or LeNet style and block/layer/operation detail explicitly. Demonstrate
  relevant plot settings and disclose shape scales, visibility floors and sampled
  sheet/neuron counts. Use shared linear dimensions for proportional views; label
  square-root/logarithmic compression explicitly when readability calls for it.
- For pixel-image contact sheets, prefer tensor `save_image_grid` to a local PNG
  with explicit class order and integer pixel scaling. This avoids large SVG
  rectangle payloads for raster data. Keep scalar plots, heatmaps and concept
  diagrams as local SVGs; both formats are committed and exported.
- The original `.cpp` examples stay for now: they are `just run-nn` targets with
  Python/R mirrors and audit entries (`examples/nn/example_audit.json`). Removing
  them is a separate decision for the user.
- The many TensorFlow-port examples (sessions, graphs, frozen graphs, TFLite,
  TFRecord) describe TF concepts: each report explains the concept and shows how
  `datamunge::nn` (eager, define-by-run) does the same job, with real maths and
  measurements, rather than imitating TF.

## Infrastructure (done 2026-10-28)

- The website (`scripts/website/build.py`) finds reports recursively, gives nested
  ones pages by relative path (`reports/nn/0_foundations/....html`) and groups the
  Reports index and sidebar by directory.
- `just docs-mepml-run` finds reports recursively; `.gitignore` keeps
  `docs/mepml/examples/**/figures/*.svg` and ignores `**/build/`.

## Library additions

- `nn/tape.hpp`: `nn::tape()` (recorded graph as data, `to_text()`/`to_dot()`), `nn::plot_tape()` (autodiff).
- `nn/derivatives.hpp`: `nn::vjp`, `nn::jvp` (double-backward), `nn::jacobian`, `nn::hessian`, `nn::hvp` (autodiff).
  Tests: `tests/cpp/nn_derivatives_tape_tests.cpp`.
- `nn/gradient_tricks.hpp`: `straight_through`, `round_ste`, `stochastic_round_ste`, `fake_quantize` (clipped STE),
  `gradient_reversal`, `scale_gradient`, `clip_gradient` (custom_gradient). Tests: `tests/cpp/nn_gradient_tricks_tests.cpp`.
- `nn/kernel.hpp`: `dispatch_floating`, `map_elements` (parallel elementwise map), `Summation` + `sum_with` (naive/pairwise/Kahan/wide),
  `summation_error_bound` (custom_op_kernel). Tests: `tests/cpp/nn_kernel_tests.cpp`. `plot_tape` rows now barycentre-ordered.
- `nn/build_info.hpp`: `BuildInfo` + `build_info()` (version, dtypes, GEMM/LAPACK backend, threads, compiler, CPU features; `to_markdown`),
  `matmul_flops` (hello_tf). Tests: `tests/cpp/nn_build_info_tests.cpp`.
- `nn/profiler.hpp`: `Profiler` (events, `summary`, `to_markdown`, `to_chrome_trace`), `ProfileScope`, `profiling`, `plot_timeline`;
  the autograd engine records one "backward" event per executed node (sessions). Tests: `tests/cpp/nn_profiler_tests.cpp`.
- `nn/views.hpp`: `view_strides` (NumPy no-copy reshape rule), `can_view`, `sliding_window` (differentiable zero-copy unfold),
  `may_share_memory`, `storage_bytes`; `nn::reshape` (so `squeeze`/`expand_dims`) now returns views of non-contiguous inputs when
  the rule allows (tensor_basics). Tests: `tests/cpp/nn_views_tests.cpp`.
- `nn/variance_scaling.hpp`: `init::variance_scaling` (Keras VarianceScaling: FanIn/FanOut/FanAvg × TruncatedNormal/Normal/Uniform),
  `init::gain` (PyTorch calculate_gain), `init::truncated_normal_std`, `init::signal_propagation` (forward/backward RMS per layer)
  (variables_initializers). Tests: `tests/cpp/nn_variance_scaling_tests.cpp`.
- `nn/activations.hpp`: `glu` (GLU/SwiGLU/GeGLU/ReGLU), `ActivationMoments` + `activation_moments` (Gaussian moments of f and f'
  by integer-aligned midpoint quadrature); `nn::Activation` gained `HardSigmoid`, `HardSwish`, `HardTanh`, `LogSigmoid`
  and `kLastActivation` (activations). Tests: `tests/cpp/nn_activations_tests.cpp`.
- `nn/attention_variants.hpp`: `kernel_attention` (linear attention, causal by prefix sums), `elu_feature_map`,
  `positive_random_features` (FAVOR+), `orthogonal_gaussian_projection`, `local_keep_mask` (sliding windows),
  `attention_entropy` (attention). Tests: `tests/cpp/nn_attention_variants_tests.cpp`.
- `nn/batch_statistics.hpp`: `batch_renorm` (Ioffe 2017 r/d corrections), `BatchStatsAccumulator` (exact "precise BN"
  statistics, Chan et al. merge; `write()` into the stats store), `FoldedLinear` + `fold_batch_norm` (batch_norm).
  Tests: `tests/cpp/nn_batch_statistics_tests.cpp`.
- `nn/conv_analysis.hpp`: `conv2d_matrix` (explicit matrix of a 2-D convolution: strides, dilations, groups, Valid/Same/
  Explicit or circular padding), `ConvGeometry` + `ReceptiveField` + `receptive_fields` (size-and-jump recursion),
  `conv2d_circular_singular_values` (Sedghi et al. 2019 per-frequency SVDs) (conv_layers). Tests: `tests/cpp/nn_conv_analysis_tests.cpp`.
- `nn/differential_privacy.hpp`: `dense_per_example_grad_norms` (Goodfellow 2015 trick; Gram route for sequences),
  `clip_factors`, `rdp_subsampled_gaussian` (integer-order RDP of the Poisson-subsampled Gaussian), `DpEpsilon` +
  `dp_sgd_epsilon` (RDP accountant) (dense_layer). Tests: `tests/cpp/nn_differential_privacy_tests.cpp`.
- `nn/dropout_variants.hpp`: `gaussian_dropout`, `alpha_dropout` (SELU-preserving, Klambauer et al.), `spatial_dropout`
  (per-channel masks), `drop_path` (stochastic depth, per-example masks) (dropout). Tests: `tests/cpp/nn_dropout_variants_tests.cpp`.
- `nn/word_embeddings.hpp`: `embedding_bag` (sum/mean/weighted mean/max over offset-delimited bags), `cooccurrence_matrix`
  (window counts, optional harmonic weighting), `pmi` (shift, positive, context smoothing), `negative_sampling_loss`,
  `sample_negatives` (embedding). Tests: `tests/cpp/nn_word_embeddings_tests.cpp`.
- `nn/normalization_variants.hpp`: `normalize` (functional standardisation over any axes), `layer_norm_jacobian`,
  `rms_norm_jacobian` (closed forms), `scale_norm`, `dynamic_tanh` (DyT), `weight_standardize` (normalization_layers).
  Tests: `tests/cpp/nn_normalization_variants_tests.cpp`.
- `nn/pooling_variants.hpp`: `power_mean_pool2d` (ReLU-safe), `gem_pool2d` (fixed or learnable p), `logsumexp_pool2d`
  (per-window stabilised), `adaptive_avg_pool2d`, `adaptive_max_pool2d`, `blur_pool2d` (anti-aliased, binomial) (pooling).
  Tests: `tests/cpp/nn_pooling_variants_tests.cpp`.
- `nn/positional_variants.hpp`: `RopeScaling`, `rope_inverse_frequencies` (position interpolation, NTK-aware),
  `apply_rotary_at` (real positions), `sinusoidal_encoding_2d`, `fourier_features` (NeRF), `gaussian_fourier_features`
  (positional_encodings). Tests: `tests/cpp/nn_positional_variants_tests.cpp`.
- `nn/recurrent_variants.hpp`: `linear_scan` (Hillis–Steele parallel scan for diagonal linear recurrences),
  `mingru`/`mingru_step`, `minlstm`/`minlstm_step` (Feng et al. 2024), `indrnn_step` (recurrent_cells).
  Tests: `tests/cpp/nn_recurrent_variants_tests.cpp`.
- `nn/weight_reparam.hpp` (ported from `examples/nn/common/reparam.hpp`): `weight_normalize`, `weight_norm_dense`,
  `spectral_norm_power_iteration`, `spectral_norm_dense` (persistent u in a state store), `lipschitz_bound`
  (weight_spectral_norm). Tests: `tests/cpp/nn_weight_reparam_tests.cpp`.
- `nn/adaptive_variants.hpp`: `AdaptiveOptimizer` (RAdam, AdaBelief, Yogi; same calling conventions as `Optimizer`),
  `radam_rho`, `radam_rectification`, `adam_step_bound` (exact Cauchy–Schwarz worst-case Adam step) (adaptive_optimizers).
  Tests: `tests/cpp/nn_adaptive_variants_tests.cpp`.
- `nn/advanced_variants.hpp`: `newton_schulz_orthogonalize` (+ `NewtonSchulzCoefficients::cubic/quintic`), `Muon` (Newton–Schulz
  orthogonalised momentum), `Lookahead` (slow-weight wrapper for any optimizer), `sam_gradients` (sharpness-aware gradients
  for any optimizer) (advanced_optimizers). Tests: `tests/cpp/nn_advanced_variants_tests.cpp`.

- `nn/optimizer_analysis.hpp`: `optimizer_gradient_trace`, `optimizer_quadratic_trace` and `optimizer_mode_radius` expose actual Float64 tensor state updates and fixed-quadratic stability for SGD/Momentum/Nesterov (sgd_momentum). Tests: `tests/cpp/nn_optimizer_analysis_tests.cpp`.

- `nn/mixed_precision.hpp`: binary16/bfloat16 rounding, explicit rounded-VJP storage emulation, `mixed_dense`, `DynamicLossScale` and `unscale_gradients`, ported from the example common header. Float32 storage/arithmetic on CPU; the compatibility header now forwards to the library. Tests: `tests/cpp/nn_mixed_precision_tests.cpp`.

The optimizer/training-loop section is complete (9/9 reports, 2026-10-28). The seven
new reports include retained TikZ sources, local figures, executed results and
Markdown/current-website HTML export checks; original runnable C++ targets remain available.

## Classical ML tensor refactor (2026-10-29)

All 15 reports in `0_foundations/0.6_classical_machine_learning` now use tensor
model computation. Their 37 C++ blocks include 22 explicit optimizer blocks;
every learnable-model report shows its parameter leaves/store, loss, optimizer
configuration and update loop. Nearest neighbours has no fitted parameters and
uses tensor distance selection, exact-match weighting and voting instead.

- Linear/polynomial fits and fold-specific refits use visible SGD; QR checks
  results. Ridge uses SGD; lasso/elastic net show SGD plus explicit soft
  thresholding, preserving the optimizer's parameter leaf with `copy_`.
- Softmax/MLP digits show minibatch Adam, fixed pixel scaling and a seeded
  stratified holdout. Both include representative grayscale digit inputs.
- GLMs and generative naive Bayes show direct likelihood optimisation. HMM and
  Gaussian-mixture reports use Adam rather than hidden EM fits. Probability
  constraints and the absence of an Adam monotonicity guarantee are explicit.
- PCA learns its projection with SGD/Adam and a differentiable orthonormal
  constraint; tensor SVD checks reconstruction error. All former `stats::`
  model constructors have been removed from this section's executable blocks.
- `nn/classical_inference.hpp` supplies differentiable log-space HMM evidence
  and tensor nearest-neighbour inference, with no shared training loop.
- Validation: all 37 C++ and 15 retained TikZ blocks executed; all 15 reports
  export to Markdown and Doxygen; 41 referenced SVGs reviewed with Inkscape.
  Four focused native tests pass, including HMM finite differences and
  neighbour tie/duplicate/target-gradient checks. Three header examples run;
  10/10 definitions documented; two clangd hovers verified.

Artifacts: `build/classical-tensor-validation/`, `build/classical-tensor-export/`,
`build/classical-tensor-coverage.json`, and `build/classical-tensor-hover.json`.
The CNN conversion was paused during this requested refactor, then resumed at the user's request.

## Foundation completion (2026-10-29)

All **58 source examples have a report** in the matching directory (57 C++
examples plus the NumPy Python companion). The final 29 cover data loading (8),
persistence/interoperability (6), and classical machine learning (15).
Original source programs remain available.

- New reports contain executed local fixtures/results, derivations, small worked
  examples and theory checks, with 29 retained TikZ sources and 67 local SVGs.
  Figure review uses Inkscape at 1100 px, including distinct comparison colours
  and explicit limits for the all-negative GLM score bars.
- All 29 new reports run against the rebuilt working-tree library; all 58 export
  to Markdown and nested Doxygen pages. No downloads are required by the new
  report blocks. NumPy is required for the independent interchange examples;
  LaTeX/PGF/TikZ, standalone and dvisvgm regenerate concept diagrams.
- MNIST-named reports explain the original architecture/workflow and execute
  compact **8×8 bundled digits** demonstrations, explicitly distinguished from
  the original 28×28 MNIST benchmarks. SavedModel/GraphDef/TFLite-named reports
  explicitly demonstrate native eager equivalents, without claiming those
  TensorFlow file formats are executed.
- Canonical headers now own the former example-only binary/CIFAR/CSV,
  tokenization/TFRecord, linear/Legendre, folds/validation, log-link GLM,
  Gaussian-mixture and naive-Bayes helpers. `examples/nn/common/` forwards the
  legacy names, preserving existing consumers. Log-link GLM scalar constants
  and proximal thresholds now retain double precision for Float64 probes.
- `nn/artifacts.hpp` adds validated `restore_tensors`, byte-exact
  `encode_strings`/`decode_strings`, reachable eager `inference_parameters` and
  checked `affine_inference`. Integer state remains outside ParameterStore.
- Validation: 22 focused native gtests; 67 compiled and executed header examples;
  175/175 documented definitions across the 12 new headers; three clangd hovers.
  All 11 affected original programs compile and run using the reports' C++23
  toolchain. Their declared example targets still request C++17; the current
  unrelated DataFrame header uses C++20 `unordered_set::contains`, so direct
  C++17 validation is not claimed.
- The report runner compiles blocks concurrently but executes them in document
  order, allowing independent Python/C++ writer-reader checks to reproduce
  reliably. Two Python integration tests (plus 16 existing workflow-ownership checks) cover
  delayed preparation and failure
  without modifying stored report content.

Local validation artifacts live under `build/foundations-validation/` and the
coverage inventory is `build/debug/docs/foundations-coverage.json`.

## Checklist

- [x] 0_foundations/0.1_tensors_graphs_sessions/autodiff
- [x] 0_foundations/0.1_tensors_graphs_sessions/custom_gradient
- [x] 0_foundations/0.1_tensors_graphs_sessions/custom_op_kernel
- [x] 0_foundations/0.1_tensors_graphs_sessions/graph_construction
- [x] 0_foundations/0.1_tensors_graphs_sessions/hello_tf
- [x] 0_foundations/0.1_tensors_graphs_sessions/sessions
- [x] 0_foundations/0.1_tensors_graphs_sessions/tensor_basics
- [x] 0_foundations/0.1_tensors_graphs_sessions/variables_initializers
- [x] 0_foundations/0.2_reusable_layer_library/activations
- [x] 0_foundations/0.2_reusable_layer_library/attention
- [x] 0_foundations/0.2_reusable_layer_library/batch_norm
- [x] 0_foundations/0.2_reusable_layer_library/conv_layers
- [x] 0_foundations/0.2_reusable_layer_library/dense_layer
- [x] 0_foundations/0.2_reusable_layer_library/dropout
- [x] 0_foundations/0.2_reusable_layer_library/embedding
- [x] 0_foundations/0.2_reusable_layer_library/normalization_layers
- [x] 0_foundations/0.2_reusable_layer_library/pooling
- [x] 0_foundations/0.2_reusable_layer_library/positional_encodings
- [x] 0_foundations/0.2_reusable_layer_library/recurrent_cells
- [x] 0_foundations/0.2_reusable_layer_library/weight_spectral_norm
- [x] 0_foundations/0.3_optimizers_and_training_loop/adaptive_optimizers
- [x] 0_foundations/0.3_optimizers_and_training_loop/advanced_optimizers
- [x] 0_foundations/0.3_optimizers_and_training_loop/early_stopping_checkpoints
- [x] 0_foundations/0.3_optimizers_and_training_loop/gradient_clipping_accumulation
- [x] 0_foundations/0.3_optimizers_and_training_loop/learning_rate_schedules
- [x] 0_foundations/0.3_optimizers_and_training_loop/metrics
- [x] 0_foundations/0.3_optimizers_and_training_loop/mixed_precision
- [x] 0_foundations/0.3_optimizers_and_training_loop/sgd_momentum
- [x] 0_foundations/0.3_optimizers_and_training_loop/training_loop
- [x] 0_foundations/0.4_data_loading_and_preprocessing/audio_features
- [x] 0_foundations/0.4_data_loading_and_preprocessing/cifar_loader
- [x] 0_foundations/0.4_data_loading_and_preprocessing/csv_features
- [x] 0_foundations/0.4_data_loading_and_preprocessing/data_augmentation
- [x] 0_foundations/0.4_data_loading_and_preprocessing/idx_loader
- [x] 0_foundations/0.4_data_loading_and_preprocessing/image_folder_loader
- [x] 0_foundations/0.4_data_loading_and_preprocessing/text_tokenization
- [x] 0_foundations/0.4_data_loading_and_preprocessing/tfrecord_io
- [x] 0_foundations/0.5_persistence_and_interoperability/checkpoint_io
- [x] 0_foundations/0.5_persistence_and_interoperability/frozen_graph
- [x] 0_foundations/0.5_persistence_and_interoperability/numpy_interop
- [x] 0_foundations/0.5_persistence_and_interoperability/numpy_weights
- [x] 0_foundations/0.5_persistence_and_interoperability/saved_model_inference
- [x] 0_foundations/0.5_persistence_and_interoperability/tflite_inference
- [x] 0_foundations/0.6_classical_machine_learning/cross_validation
- [x] 0_foundations/0.6_classical_machine_learning/gaussian_mixtures
- [x] 0_foundations/0.6_classical_machine_learning/generalized_linear_models
- [x] 0_foundations/0.6_classical_machine_learning/hidden_markov_models
- [x] 0_foundations/0.6_classical_machine_learning/kmeans_clustering
- [x] 0_foundations/0.6_classical_machine_learning/linear_regression
- [x] 0_foundations/0.6_classical_machine_learning/logistic_regression
- [x] 0_foundations/0.6_classical_machine_learning/mlp_mnist
- [x] 0_foundations/0.6_classical_machine_learning/naive_bayes
- [x] 0_foundations/0.6_classical_machine_learning/nearest_neighbors
- [x] 0_foundations/0.6_classical_machine_learning/polynomial_regression
- [x] 0_foundations/0.6_classical_machine_learning/principal_components
- [x] 0_foundations/0.6_classical_machine_learning/regularized_regression
- [x] 0_foundations/0.6_classical_machine_learning/softmax_mnist
- [x] 0_foundations/0.6_classical_machine_learning/support_vector_machines

## CNN conversion (2026-10-29)

### Neural-network architecture SVGs (2026-10-30)

Added `nn::Module`, `nn::plot_model`, `nn::ModelDiagram` and `nn::model_stage`.
Functional linear, deep and convolutional models can now plot their executed
architecture with `model.plot(example).save_svg("network.svg")`. The free
`plot_model(forward, example)` form works with existing forward functions.
Diagrams show actual batch-inclusive tensor shapes, directed branches/joins,
named layers and parameter counts with shared tensors counted once. Composite
stages can be expanded into primitive operations. SVG output is native and
requires no display server, Graphviz or TeX. The current layouts flow left to
right without folding, with wrapped labels rotated counterclockwise and XML
special characters escaped (see the configurable-views update below).

Tracing executes one forward pass on a cloned floating example input, enables
and restores autograd recording, and leaves input flags/values/gradients intact.
It runs no optimizer or backward pass. Callbacks keep their ordinary lazy
parameter creation and other side effects; use inference mode when plotting.
Diagrams describe the executed differentiable path, with constants and detached
paths unexpanded. Ordinary forwards skip diagram metadata collection.

All 18 reports in CNN `1.1_fundamentals` and `1.2_classic_architectures` now have
executed diagram programs and 22 reviewed local SVGs, covering both Inception
variants/families and VGG-11/16/19. The figures use the report model settings or
explicitly labelled operator probes. Existing fitting loops and pedagogical TikZ
sources remain. Every learned model diagram checks its scalar count against its
parameter store; the inference diagrams explicitly disable dropout and auxiliary
training heads where applicable.

Validation: the focused CMake/CTest target passes 39 tests (nine new plotting
tests and 30 layer/parameter/derivative/tape regressions). All 12 public examples
compile and execute, the header compiles standalone, six clangd hovers pass,
and all 53 new header/implementation definitions are documented. Combined
completed-CNN-header coverage is 3,248/3,248. All 18 reports export to Markdown
and current website HTML with local media present; the full folder's 245 rendered
tables remain consistent. Current CNN totals are 625 executed C++ blocks and
789 reviewed figures (595 SVG, 194 PNG), across the same 129 completed reports.

Repeat the native checks with `just test-nn-model-plot`. Evidence, report
inventories, XML/layout checks, rendered reviews and exports are retained in
`build/nn-model-plot-validation/`; Doxygen XML is in `build/nn-model-plot-docs/`.


### Configurable neural-network architecture views (2026-10-30)

The native SVG API now supports `ModelPlotStyle::{Flow,Graph,LeNet}` and
`ModelPlotDetail::{Blocks,Layers,Operations}`. All dependencies flow left to right;
visible node labels rotate 90 degrees counterclockwise. There is no row folding.
`ModelPlotOptions` exposes fill, edge and funnel opacity, stroke width, category
and canvas/text/edge/outline colors, column/lane/label/margin spacing, and label
content/font size/wrapping. Labels can show full metadata, names, shapes, both,
or nothing; node tooltips retain names and shapes.

`ModelPlotSizing` uses shared, explicit dimension scales. Spatial tensor faces
represent their spatial axes; channels control depth, and vector feature count
controls column height. Batch size is excluded from geometry. Linear scaling is
the default: ratios are preserved above the configurable pixel visibility floor,
and spatial flow volumes are proportional to per-example activation element
counts. Square-root and logarithmic compression are opt-in and disclosed in the
legend. LeNet sheets and graph neuron markers are bounded samples inside the
scaled envelope, with a depth guide retained even for a single sampled sheet.
Printed shapes remain exact. Rank-three sequences and unfolded rank-five spatial
axes have documented mappings; no batch or scalar byte-width scaling is implied.

All 18 reports in `1.1_fundamentals` and `1.2_classic_architectures` demonstrate
explicit presentation settings and linear scales with a one-pixel visibility
floor. Their 22 configurations each have Flow/block, Graph/layer and LeNet/block
views: 66 regenerated SVGs. The website preserves native image size inside a
keyboard-accessible scrolling region. Fitting loops and tensor model math are
unchanged.

Validation: 49 native tests pass (19 plotting tests and 30 adjacent regressions),
as do five website/report-runner tests. All 20 public header examples compile and
execute; the header compiles standalone. Eight clangd hovers and documentation
coverage pass: 186/186 affected header/implementation definitions and 3,311/3,311
completed CNN-header definitions. All 66 figures were rendered with Inkscape and
reviewed; independent SVG checks verify forward edges, rotated labels, positive
shape dimensions, actual rendered label bounds and separation from glyphs.
All 18 reports export to Markdown and current website HTML with local media
present. Chromium's startup fails in this environment, so a browser UI smoke
check is not claimed. Evidence: `build/nn-model-plot-settings/`; fresh Doxygen XML:
`build/nn-model-settings-docs/`.


### CNN folder complete (2026-10-30)

All 129 C++ source examples in `examples/cpp/source/nn/1_convolutional_neural_networks`
now have matching reports in `docs/mepml/examples/nn/1_convolutional_neural_networks`;
no conversions remain. Reports use tensor computations, show explicit tensor
optimizer loops wherever fitting or input optimization occurs, and explain cases
with no fitted parameters. C++ blocks are formatted across readable lines. Data
previews, image galleries and editable TikZ diagrams accompany the methods.

Final validation records 607 executed C++ blocks, 129 executed TikZ blocks,
599 passing native tests, 1,300 compiled and executed public header examples,
129 original-source C++23 syntax checks, 428 clangd hover checks, and documentation
for all 3,220 definitions in the converted CNN headers. All 129 reports freshly
export to standalone Markdown and the current website HTML exporter. The final
artifact audit finds 736 nonempty captured result blocks and 767 referenced figure
files (573 SVG and 194 PNG), with no missing or unreferenced figures. All 245
rendered HTML tables have consistent column counts. Six adjacent-table boundaries
were corrected and their affected programs rerun; regenerated figures exactly
match the prior exports. No C++ code line exceeds 160 characters.

Local validation evidence: `build/cnn-validation-summary.json`,
`build/cnn-combined-coverage.json`, `build/cnn-final-artifact-audit.json`,
`build/cnn-final-table-reexecution.json`, and `build/cnn-final-validation/`
(all 129 fresh exports and `rendered-tables.json`). Per-report validation and
figure-review directories retain the individual checks and independent references.


### Tensor full VGG19filter activations completed (129/129 overall)

Four executed C++ blocks,editable/executed TikZ,fifteen reviewed figures(3SVG/12PNG),
8/8documenteddefinitions,threecompiled/run publicexamples,twoclangdhovers,
threenative tests,standaloneheader,original full frozen-VGGworkflow and
Markdown/currentwebsite exports pass. Canonical visual_features.hpp executes
ALL16original-widthVGG19convs/SAME/ReLU/fourVALIDMAXpools,not styleaveragepools;
32officialKerasImageNetno-toparrays/20024384valuesunchanged. RGB255->BGRminus
[103.939,116.779,123.68]tensorpreprocessing. Fiveactualtaps:sides64/32/16/8/4,
widths64/128/256/512/512. Legacyvisualheaderforwards. No fit/head/optimizer here.
SameoriginalYellowLabrador(Elf,CCBY-SA3.0)/KandinskyCompositionVII(publicdomain)
JPEGs/hashes/URLs/attributions asstyle reports;center-squareintegercrop,Float32
legacybilinear64px/RGB255throughcanonicalLoadStyleImage,no synthetic substitution.

All2944channelmaps acrossbothobservationsshownWITHOUTselection,ten complete
shared/localmosaics;observation-major/channel-minor,16columns,explicitindexmapping,
minimum32pixeldeepmapintegerenlargement. SharedONEmaximumperlayeracrossboth
images/allchannels preservesamplitude;localmaxdiscardsit. Everyrawfeature,
channelmean/positivefraction/maximumretained;all1472perimagefractions plotted.
Lastblock708/1024channelsallzero,blacktilesaregenuinezeros,notmissingdata.
All64actualfirst3x3kernels,signedsharedpeak.650995195,grey0,BGRweightsreversed
ONLYforRGBdisplay. Genuineobservedinputsdisplayed;nooptimized-image/classattributionclaim.
Zero/constantpositive/sharedhalfdisplayfixturesandtensorordering/scales/invalid
contracts pass. OriginalhostALL640firstconvcorner/center/channelprobes max2.57966e-5;
every524288firstReLUvalue/all24576RGB-BGRvalues andeveryfulltap checked.
Repeat0,singlebatchmax.00146484375withinDOCUMENTEDsource.005F32tolerance,
all32/20024384officialconstantsimmutable. IndependentofficialKerasHDF5fullMAXpool
VGG19alltapvalues agree(absmax.0001220703/.001464844/.005859375/.001861572/
.0000610352,declaredF32abs.02+rel2e-5);ALL2944channelstatisticrows agreeNumPy.
ALL1574656original-source/reportobservedfeatures/preprocessing/kernelvalues EXACT.
All15figures individuallyreviewed,includingbothobservationsatscalablelayer-level
shared/localmosaicreviews;shortenedfixturelegendtoavoidclipping.
Evidence:build/cnn-filter_activations-{coverage,hover}.json,validation/,review/.
129/129complete,0remaining;everyCNNsource now has a validated report.


### Tensor feature visualization completed (128/129 overall)

Three executed C++ blocks,editable/executed TikZ,four reviewed figures(2SVG/2PNG),
9/9documenteddefinitions,threecompiled/run publicexamples,twoclangdhovers,
twonative tests,standaloneheader,original full960iteration workflow and
Markdown/currentwebsite exports pass. Canonical feature_visualization.hpp supplies
selectedchannelobjective,explicitcenteredmean-squareandseparatelynormalized
squared-TVpriors,andtensoruniformscalingfromrawmt19937draws. Legacyheaderforwards.
All784analyticobjectivegradientcoordinates/all1512TVedges/boundaries,allobjective
terms,1/Bbatchmeanscaling,exactuniformdraworacle/seedreplay andinvalidprior/noise
contracts pass. Frozen121classifier/actualinitialarrays,zero newclassifierfit;
links96explicitAdamfittingupdates. ExplicitlySYNTHETICuniformnoise starts,
16independentimages/channels0..15,seeds1423/1429,noreal-image/prototypepretence.

ReportsSHOWoneDoubleimageleaf/optimizerconstructionPERchannel,12TensorSGD.05
ordinarynegativeRMSnormalizedderivatives,bounds/Float32candidateevaluation,
reset/rejection/12trialbacktracking,fullobjectiveacceptance. All5variants/all16
channels retained:mainL2.01/TV.2,unregularized,actualinitial,secondseed,zero.
Every960channeliteration/image/gradient/objectiveterm/rate/backtrack retained;
all16selected/controlclassifierarraysunchanged,zero-rate allstartingpixelsEXACT.
Every80finalselectedresponse/objective/priorrow,all16mainobjectivetraces,
ALL256crosschannelresponses(fullnumerictablerowsandheatmap)shown.
Mainobjectivesimprove16/16,buttargetisstrongestonly1/16;offdiagonalresponses
andlackofchannelexclusivity explicitlydisclosed,nocherry-picking/superiorityclaim.
Twoactualuniformbanksandall16×6final/controlgalleryimages reviewed.
Independentoriginal/reportALL1572160images/gradients/objectivetermvalues andevery
rate/backtrack EXACT. TFEVERY960regularized/unregularized/controlimagegradient
max2.79396772e-9. Activationgainnotsemanticprototype/causal/recognitionqualityproof.
Evidence:build/cnn-feature_visualization-{coverage,hover}.json,validation/,review/.
128/129complete,1remaining:filteractivations.


### Tensor DeepDream completed (127/129 overall)

Three executed C++ blocks,editable/executed TikZ,four reviewed figures(3SVG/1PNG),
15/15documenteddefinitions,fivecompiled/run publicexamples,twoclangdhovers,
threenative tests,standaloneheader,original full input-optimization workflow and
Markdown/currentwebsite exports pass. Canonical dream.hpp replaceshosthalf-pixel
resize/RMS/proposals withDoubletensorcoordinate/gather/bilinearinterpolation,
DoubleRMS,oneDoubleimageleaf/plainTensorSGD,clip[0,1]/Float32roundandmonotonic
12trialbacktracking. Originalclassifierarraysremainfrozen. Legacyheaderforwards.
All9analyticresizevalues,identity/rectangular/fullcoordinateoracle,quadratic
.6->.725/4halvings,rms/zero-rate/stationary/invalidrate fixtures andindependent
hostproposalcomparisons pass. ReportsSHOWoptimizerconstruction,negative-normalized
ordinarygradientstep,projection,reset/rejectedtrial/halvingandalliterationloops.

Same10genuinefirst-digitMNISTimages/sourceIDs,121classifier/actualinitial arrays,
zero newclassifierfit. Fiveactualvariantsmainconv2multi20/24/28,conv1,
actualinitial,single28px15steps,zero-rate;75imageiterations(everyaccepted/no-op
andtrialmetadata),globalbatchRMSexplicit. Mainfiveupdatesperoctave.02rate:
20px.419970304->.536893606,24px.511448920->.654157698,
28px.631932259->.779571593. Cross-octaveresamplingdiscontinuities displayed;
monotonicityONLYwithinoneoctave,notfalsefirst/finalcross-scalecomparison.
Zero-rateEXACTresize-only,notoriginalinput;all16classifier/controlarraysimmutable.
All10×6actualimage/controlgalleryrows andeverylocalobjectivetrace reviewed.
IndependentNumPyEVERYDoublehalf-pixelresize/acceptedproposal EXACT;
TFall75ordinaryfeaturegradients max2.32830644e-10. ALL939500source/report
image/gradient/score/rate/backtrackvalues andallfinalimages EXACT.
No Inception/ImageNet benchmark,realism,recognition-improvementorcausalclaim.
Evidence:build/cnn-deepdream-{coverage,hover}.json,validation/,review/.
127/129complete,2remaining:featurevisualization andfilteractivations.


### Tensor occlusion sensitivity completed (126/129 overall)

Three executed C++ blocks,editable/executed TikZ,five reviewed figures(4SVG/1PNG),
25/25documenteddefinitions,fivecompiled/run publicexamples,twoclangdhovers,
threenative tests,standaloneheader,original full frozen-classifier workflow and
Markdown/currentwebsite exports pass. Canonical occlusion.hpp returns alltensor
rectangles/coverage/actualmodifiedinputs/rawpredictions/fixedscores/Double signed
patchdrops/Double overlapmeans. Anchors areindexmetadata,masks/where/replacement/
scoring/maskmatrixprojectionuse tensors. LegacyhostarrayAPIis conversion-onlyglue.
All240fixturemodifiedpixels/12drops/20coverage/40means,rectangularodd-edge,
identity/full/invalidstridecontrols pass;allactualinterventions/projections match
retainedindependenthostimplementation inthreebatchconfigurations.

Restores121verifiedMNISTclassifier/actualinitialarrays,zero newfit,links96visible
Adamupdates/88%heldoutaccuracy24errors. Same10genuinecases/sourceIDs/fixedtop/runner.
All8actualvariants:main7x7stride4/black,gray.5,runner,initial,5x5stride3,
7x7stride7,identity,fullblack. Main49windows,coverage1..9includingappended21px
edges,276positive/154negative/60zero scorechanges. Every490primarydrops and49
rectangles explicitlyprintedwithouttensortruncation;allvariantsactualinputs/scores
retained. All10×7galleryrowsoneabsolutepeak,signedzero grey;coverageandall10case
controlL1changes/signcounts plotted. PatchprojectionexplicitlyNOTadditivepixel
attribution,gradient,segmentationorcausalimportance.
Repeat0,singlebatch3.81469727e-6/identity2.86102295e-6withinDOCUMENTED1e-5F32
batchingtolerance,full-baseline0,all16selected/controlarraysimmutable.
IndependentNumPyallmodifiedpixels/coverage/drops/overlapmeansverify;TFALLactual
counterfactualrawlogits max3.81469727e-6;ALL72source/reportarrays EXACT.
Replacement/patch/stride/target/modeldependence andfloatingidentityresidualexplicit.
Evidence:build/cnn-occlusion_sensitivity-{coverage,hover}.json,validation/,review/.
126/129complete,3remaining:DeepDream,featurevisualization,filteractivations.


### Tensor SmoothGrad completed (125/129 overall)

Three executed C++ blocks,editable/executed TikZ,five reviewed figures(3SVG/2PNG),
12/12documenteddefinitions,threecompiled/run publicexamples,twoclangdhovers,
threenative tests,standaloneheader,original full frozen-classifier workflow and
Markdown/currentwebsite exports pass. Canonical smoothgrad.hpp replaceshostnoise
transforms/perturbedinputs/allstatisticswithtensors. Rawmt19937drawsingressonly,
DoubletensorBoxMuller/midpointuniforms/source1171seed,Doubleinput+sigma*noise,
Float32roundingBEFOREoptionalclip,ordinaryfixedtargetgradients,Doublemoment
reductions/Float32outputs. SignedMEANprimary,meanabs/meansquareexplicitdifferent
summaries;unbiasedvariance/MonteCarloSE,allnoisypoints/derivativesretained.
Linearconstantgradientvariance0/quadraticEXACTfinite-bankexpectation/sigma0/
dummy/batching/seed tests,allhostreferencepoints/derivatives/moments for1/7/32
samples×clipped/unclipped×1/11callbackbatches,invalidconfigcontractspass.
HostsourceGaussianbankagreeswithin1e-14;seedselfreplayEXACT.

Restores121verifiedMNISTclassifier/actualinitialarrays,zero newfit,links96explicit
Adamupdates/88%heldoutaccuracy24errors. Same10genuinecases/sourceIDs/fixedtop/runner.
All8groups:main32/.15/1171NOCLIP,exactprefix8,secondseed1173,sigma.30,clipped,
runner,actualinitial,sigma0with2replicas. PairedcontrolsSAMEbank,repeat0,
sigma0vanillawithin2e-6,all16selected/controlarraysimmutable. Main114211/250880
noisycoordinatesoutside[0,1]=45.5242%,explicitdomainextrapolationdisclosure.
PrimaryvsordinaryL1 132.801926,independentseedchange33.9671936,
meanabs-minus-absmeansum29.6115341,meanprimarySE.00376360933.
All10percase6roughnesscontrols plotted/printed;meanordinary/mainroughness
.909645/.884317descriptive,notuniversalquality/faithfulnessclaim.
All10×10galleryrowsoneabsolutesignalpeak,signedzero grey;first8actualnoisy
replicasdisplay-clippedONLY,modelusesretainedunclippedvalues.
IndependentNumPyMT/BoxMullereverybank/allmomentformulas andTFEVERYnoisyderivative,
TFmax8.9406967e-8;ALL64source/reportarrays EXACT. No causal/qualityclaim.
Evidence:build/cnn-smoothgrad-{coverage,hover}.json,validation/,review/.
125/129complete,4remaining:occlusion,DeepDream,featurevisualization,filteractivations.


### Tensor integrated gradients completed (124/129 overall)

Three executed C++ blocks,editable/executed TikZ,four reviewed figures(3SVG/1PNG),
9/9documenteddefinitions,threecompiled/run publicexamples,twoclangdhovers,
threenative tests,standaloneheader,original full frozen-classifier workflow and
Markdown/currentwebsite exports pass. Canonical integrated_gradients.hpp replaces
hostpath construction/accumulation with Doubletensor nodes/baseline-differences,
Float32callbackevaluation,Doublemean/attribution arithmetic,Float32outputs;
allactualstep-major/case-minor derivatives retained. Legacyheaderforwards.
All6analytic saturation/linear/dummyattributions exact,endpoint0vsIG1,
completeness7/-1.5,equivalentReLUimplementations,batching,zero-deltaandinvalid
contracts pass. Nine steps/batchcombinations compare EVERYpathderivative,
Doublemidpointnode,meanandattribution EXACTwithretainedindependenthostimplementation.
No optimizer needed for frozenmodelnumericalintegration;sharedclassifierexplicit
96Adamfit/actualinitialarrays from121,zero newclassifiertraining.
Same10originalMNISTdigitcases/sourceIDs/fixedtop/runner classes,black0/gray.1
baselines,actualinitialcontrol.64midpoints sufficeall40declaredcompletenessbounds
.01+.01abs(scorechange);adaptive64->512codefailsifstillunmet,notusedtofitweights.
All40actualscorechange/sumattribution/residual/tolerancerows printed/plotted;
coarse32maxprimaryresidual.0401768225,selected64.0196272538,nonzerohonestlyshown.
PrimaryL1vsendpoint-times-input8.06342411,runner83.9826889,gray55.5069809,
initial73.4423523. All10×6galleryrows shareoneabsolutepeak,signedzero grey.
Singlebatch7.4505806e-9within2e-6,all16selected/controlarraysimmutable.
Independentoriginal/reporteverypathgradient/mean/attribution/delta/residual agrees
within1e-12Double rounding. TFALL2007040pathderivatives max1.1920929e-7;
ALL31360attributions max2.98023224e-8. Finitemidpointquadrature,baselinedependence,
nonmonotonicindividualrefinement,andnoncausalinterpretation statedexplicitly.
Evidence:build/cnn-integrated_gradients-{coverage,hover}.json,validation/,review/.
124/129complete,5remaininginterpretabilityreports.


### Tensor Grad-CAM and Grad-CAM++ completed (123/129 overall)

Three executed C++ blocks,editable/executed TikZ,four reviewed figures(3SVG/1PNG),
21/21documenteddefinitions,sixcompiled/run publicexamples,threeclangdhovers,
threenative tests,standaloneheader,original full frozen-classifier workflow and
Markdown/currentwebsite exports pass. Canonical grad_cam.hpp supplies every
classicchannelweight/signedmap/rectifiedmap,positivealpha/plusweight/plusmap,
zero-safe normalization,shared frozenMNISTlastconvderivative/all13intermediates/
halfpixelresize andexplicitfeaturetail. Legacycommonheaderforwards.
Allhostformulaweights/alphas/maps/twobatches,zero normalization,24ACTUAL
first/second/thirdexp-affinederivatives andinvalidnegative/nonfinitecontracts pass.
Grad-CAM++explicitly uses exp(S-c),cFIXEDatcurrentS,downstreamrawSlocallyaffine,
nonnegativepost-ReLU A;g>0alpha1/(2+sum(A)*g),g<=0irrelevantcontribution0.
NEVERmislabels g squared/cubed asrawlogit higher derivatives. No pretrained
benchmark/causal/segmentation/superiority claim.

Restores121shared26826value/eightleafclassifier,zero newfit;links96visibleAdam
updates,88%heldoutaccuracy/24errors. Same10originaldigitcases/sourceIDs/fixed
topandrunnerchoices;true4predicts9,itsmapscorrespond9nottruth4. Actual31360
lastconvcoordinatespercontrol,all13intermediatescheckednativeagainstoriginal.
Selected/initial/headreset3stores24arraysimmutable,logit/repeat0,
singlebatchplusmap1.1920929e-7within2e-6;actualtailFDcoordinate1089,
step.001error.000205405056withdeclaredpiecewise/F32tolerance.
Originalworkflowcheckshiddenpreactivationmargin>1e-8atall320coordinates.
All10×9galleryrows:images,classic/plusselected,runner,initial,headreset.
Per-mapmaxnormalizationdiscardsamplitude;allrawpeaksandintermediatesretained.
All1960selectedmethodnormalizedpixelsL1difference247.787888,zero classiccases0.
Meancosines classicrunner/initial/headreset .688794081/.539342060/.960113744;
plus .998598319/.971336165/.999975561. Near-identicalpluscontrols disclosed,
attractive/sharpnormalizedmapsnotproof oftarget/headdependence/explanationquality.
IndependentNumPyallrawformulas andTFALL125440featurederivativesverify,
TFmax1.1175871e-8,NumPymaxpluserrorbelow2e-6;ALL52source/reportarrays EXACT.
Evidence:build/cnn-grad_cam-{coverage,hover}.json,validation/,review/.
123/129complete,6remaininginterpretabilityreports.


### Tensor guided backpropagation completed (122/129 overall)

Three executed C++ blocks,editable/executed TikZ,five reviewed local figures(3SVG/2PNG),
6/6documenteddefinitions,fivecompiled/run publicexamples,threeclangdhovers,
twonative tests,standaloneheader,original full frozen-classifier workflow and
Markdown/currentwebsite exports pass. Canonical guided_backprop.hpp implements
ordinaryReLUforward withguidedpositiveactivation/positiveincomingcustomreverse,
atomicdiagnosticcount andall3sharedCNNgates;legacyheaderforwards.
Every9activation/incomingsign/zerocombination,ordinary/deconvdistinction,
negativeincomingtrueFD(-3)versusguided0,andnonlinearVJP4!=1 checkedexactly.
ALL2000actualheldoutforwardlogits unchanged,exactly3gatedcallseachforward,
all8learnedarraysimmutable. No new classifier fitting;restores verified121
selected/actualinitialarrays,links96explicitAdamupdates andactual24heldouterrors.
Same10genuinefirst-digitcases/sourceIDs/fixedtop/runnerchoices.
Ordinary/guidedselectedshareonestore;twoindependentfullinitial/headresetstores,
all24arraysunchanged. Repeat0,singlebatch2.98023224e-8within2e-6.
Ordinary/runner/fullinitial/headreset L1changes351.020233/190.699829/
292.196411/34.7883873;meancosines.635562927/.817330223/.758817072/.997573187.
Near-identicalheadreset visualization disclosed;sharp-lookingmapsdonotprove
learnedclassheaddependence/explanationquality. Galleryall10×6rows sharesone
absolutepeak,signedzero grey;all10percasecosines plotted.
IndependentTFcustomguidedrulechecksALL31360guided/runner/initial/headreset
coordinates,max4.47035e-8/2.98023e-8/5.58794e-9/2.98023e-8.
All39200ordinary/guided/controlsource/reportcoordinates EXACT.
GuidedsignalexplicitlyNOTordinaryderivative,linearVJP,causalexplanationor
segmentation. Evidence:build/cnn-guided_backprop-{coverage,hover}.json,
validation/,review/.122/129complete,7remaininginterpretabilityreports.


### Tensor vanilla saliency completed (121/129 overall)

Five executed C++ blocks,editable/executed TikZ,six reviewed local figures(4SVG/2PNG),
21/21documenteddefinitions,elevencompiled/run publicexamples,fourclangdhovers,
fivenative tests,standaloneheader,original three-epoch workflow and Markdown/currentwebsite exports pass.
Canonical saliency_models.hpp supplies shared26826value/eightleafdigitCNN,
conv5SAME8HeUniform/ReLU/avg2,conv3SAME16HeUniform/ReLU/avg2,
flatten784/GlorotFC32ReLU/linear10head,bias0,noBN/dropout;conv-onlyoctaves supported.
Rawfixedclasslogit,SUMbatchordinaryinputderivative,alignedMNISTdataandtensor49cell
memory1NNarecanonical;legacycommonhelpers forward. Everyreal7840selected input
coordinate checkedrepeat0/singlebatch2.98e-8,all16learned/controlarraysimmutable.
Declaredtwo-pixel fixture differentiates[2,-3]vs[-1,4]withoutbatchmean scaling.
Every16independenthostFD,all6networkendpoint/inputgradient shapes,all8realCE
leafgradients,invalidtarget/imagecontracts andall1400coarsefeatures pass native checks.
Independent original IDXgzip reconstruction checks ALL1097600pixels/all1400labels/
sourceIDs,balanced1000/200/200seed1931pools,officialtestoffset60000,no duplicates.
All200actualheldoutimagesgallery;deterministicfirsttrueclass0..9casesIDsretained.

Warmall8before96explicitAdam.001,b32x3(last8per epoch),seed1103/order1109.
Initial8recordgradientprobe updatesnothing. Saveactualinitialandvalidation-selected
weights,separatefrozenreportsrestoreexactlythese. Select3,val.366006,
fixedfitNLL2.350038605->.395236061. Heldout.88/F1.880701/NLL.443415,
recalls.95/1/.9/.85/.7/.85/.9/.85/.8/1,all100confusioncounts and24errors shown.
Initial.105/NLL2.36705,coarse1NN.84/NLL1.62115. Freshrestore/repeat0,
singlebatchlogit2.86102295e-6withinDOCUMENTED5e-6Float32tolerance,all8immutable.
Fixedlearnedtop/runner/actualinitialgradientmaps shareoneglobalabsolutepeak;
signedzero grey,absrow0black/peakwhite. Target/initialweightL1changes
571.481628/449.962189,positive4406/negative3434;realFDcoordinate407,
centralstep.001error.000268697739within declaredpiecewise/F32tolerance.
IndependentTFCPUinference checksall200learned/initiallogits,max4.64916e-6/
4.17233e-7;all23520selected/runner/initialinputgradients max6.70552e-8/
4.47035e-8/7.45058e-9. ALLselectedandinitialarrays EXACTwithoriginalworkflow.
Localrawscore sensitivity,notcausalimportance/objectsegmentation/benchmarkclaim.
Evidence:build/cnn-vanilla_saliency-{coverage,hover}.json,validation/,review/.
121/129complete,8remaininginterpretabilityreports.


### Tensor Two-Stream completed (120/129 overall)

Five executed C++ blocks,editable/executed TikZ,seven reviewed figures(5SVG/2PNG),
33/33documenteddefinitions,fourteencompiled/run publicexamples,fourclangdhovers,
ninenative tests,standaloneheader,original three-epoch workflow and Markdown/currentwebsite exports pass.
Canonical two_stream_models.hpp retains both complete CNN-M branches,5convs/3pools/
2hidden layers each,sourceexplicitborders/negative-infinitysecondpoolborder,
RGBLRNafterconv1/2,flowLRNafterconv1only,four independentkeep.5dropouts.
Compact96px/divisor16/FC64/32/3class actual78748values/32learnedleaves;
full224px/4096/2048/101class181335594census ANALYTICAL,notfullallocatedrun.
Allcompactendpoint/gradientchecks,all16stackhostFDmax5.62e-12,all12probabilityfusion
hostFDmax8.69e-12,explicitpooladjoints,24variationalflowhostFD andclassical2x2
brightnessderivative/borderoracles,invalidcontractspass. Everymodalitydescriptor
andmemory1NNprediction agreesretainedoriginalhost. ALL10948608actualoriginal
resizedRGBpixels agreeDoubleweightoracle/sourceFloat32coordinates max1.19209e-7.

Genuine18KTHAVIs/custom24/6/6clips,people1..4/5/6,scenario1,first2segments.
ElevenCONSECUTIVEactualframescenteredineachsegment;middleRGB+tenadjacentfields,
notpriorvideo16wide-spacedobservations. Originalhashes/frameIDs/person/video/segment
retained. Declared single-scaleHorn-Schunck-stylequadraticbrightness/smoothness,
alpha.05,400explicitTensorSGD.2stepsfromzero u/v,noFarneback/TVL1/groundtruth/
large-motion/publishedflowclaim. Reportreplaysall184320firstclipvalues EXACT,
energy8.28073692->3.00280547. Every36actualclipenergydecreases.
IndependentNumPy400stepreplay verifies ALL6635520displacements,max1.78814e-6,
allmiddleframealignmentsEXACT. All6middleRGBand60flowmagnitudemapsgallery,
sharedsquare-rootcontrast/globalmaximum,noindividualrescaling.

Warm32leaves,36explicitjointAdam.001,b2x3,seed853/drop823/827/829/839/order857,
first2gradientprobeadvancesall4streams. SeparatebranchCE SUM,latefusionAVERAGE
SOFTMAXPROBABILITIES(notlogits),logfloor1e-8. Select2,val1.088412,fixedfused
fitNLL1.122472737->1.074159313. Heldout1/3/F1.266667/NLL1.097883,
recalls0/1/0,allboxing/wavingmissed;flow1NN.5/F1.412698/NLL1.051445.
Ninepredictor/controlmetrics/all6predictions/fullconfusionsretained.
Freshrestore/repeat0,singlebatch1.1920929e-7within1e-6,all32immutable;
zero-flow/neutral-RGB/reverse-orderlogitL1 .212156296/.5244205/.125096917.
Independent original/report ALL32arrays/78748scalars and initial/testlogits EXACT.
Sourcepreparation: scripts/prepare_cnn_two_stream.py,nn_two_stream_prepare.cpp.
Evidence:build/cnn-two_stream-{coverage,hover}.json,validation/,review/;
/tmp/cnn-two_stream-{source,fit}-reference.log.120/129complete,9remaining;
all1.10reportscomplete,remainingnineinterpretabilityreports.


### Tensor VoxNet completed (119/129 overall)

Five executed C++ blocks,editable/executed TikZ,seven reviewed figures(6SVG/1PNG),
29/29documenteddefinitions,elevencompiled/run publicexamples,fourclangdhovers,
sixnative tests,standaloneheader,original three-epoch workflow and Markdown/currentwebsite exports pass.
Canonical voxnet_models.hpp retains original32cube/32filter5stride2/32filter3/pool2/
FC128/head,LeakyReLU.1,drop.2/.3/.4,HeNormalconvs/Normal.01dense,bias0,noBN.
Full40class921736/8 actualforward/allgradonrealoccupancy passes;3class916963/8.
Real42originalModelNet10chair/sofa/table meshes,24/6/12objectsplit aspointreports.
Declared sampled-SURFACE occupancy,not publishedsolid/intersection voxelization:
Doubletensorareas,65536area-weightedfaces/seed1193+ID,barycentric sampling,
originalfullvertexbboxcenter/maxhalfextent,Float32points,floor((xyz+1)*14.5)+1,
clamp1..30,tensorsegmentsum>0,emptyborder. Everyoriginalmesh/hash/face/uniform/
center/radius retained. Independent NumPy/OFF/Xoshiro oracle checks2752512face
choices,5505024uniforms and ALL1376256binaryvoxelvalues EXACT across42objects.
AllcoordinateconvFD,108rotationvalues/27rotationadjoints,8poolvalues/64adjoints,
weight-onlyL2hostFD,actualfull/compactendpoint/census/gradients,invalidcontracts pass.
Every42descriptor and12memory1NNprediction matches retained host oracle.

Warm8leaves,18explicitMomentumSGD.001/.9,b4x3,seed907,drop881/883/887,
order919,quarterturnmt19937seed929. First4rowCEprobe advancesall3streams.
VisibleCEgrad+2lambdaW withlambda.001,biasesunchanged,sourcecoupledL2SUM.
Fourrotated LOGITS average forvalidation/inference,notsoftmaxaveraging.
Select3,val1.092044,fixedC4fitNLL1.106853194->1.065721723.
Heldout7/12accuracy/F1.464646/NLL1.097433,recalls.75/0/1,ALL4sofasmissed;
coarse1NN7/12/F1.555556/NLL.968111,recalls.75/.25/.75. All12predictions,
sixpredictor/controlmetrics andcompleteconfusions retained. Noqualitythreshold.
Freshrestore/repeat0,singlebatch3.7252903e-8,C4orbit7.4505806e-9within1e-6,
singleviewrotationL1 2.10843587,all8leavesimmutable. Independentoriginal/report
ALL8arrays/916963scalars andinitial/testlogits EXACT. All12heldoutmeshXY/XZ
projections andall24fitoccupancycounts displayed;custom42objectexperiment,
notpublishedModelNet40benchmark. Evidence:build/cnn-voxnet-{coverage,hover}.json,
validation/,review/;scripts/prepare_cnn_voxels.py,nn_voxel_prepare.cpp;
/tmp/cnn-voxnet-{source,fit}-reference.log.119/129complete,10remaining.


### Tensor SlowFast completed (118/129 overall)

Six executed C++ blocks,editable/executed TikZ,nine reviewed local figures(8SVG/1PNG),
22/22documenteddefinitions,sevencompiled/run publicexamples,fourclangdhovers,
sixnative tests,standaloneheader,original three-epoch workflow and Markdown/currentwebsite exports pass.
Genuine KTH24/6/6clip pools and all96heldoutframes reuse verified original source preparation.
Canonical slowfast_models.hpp retains two3/4/6/3R50 paths,Fast1/8width,
four temporal5/stride8 Fast-to-Slow projections,110normalizedunits,
332learned/220moving arrays. Compact16x32/base8 actual533524values;
full32x224/base64/400class34480216value census ANALYTICAL,not a full allocated run.
All32expand gammas0initially,96upstream kernels exactlygated;host gamma/beta
and residual identity adjoints pass. Every56fusionoutputs/all192independent
hostFDcoordinates agree,max errors2.78e-17/1.84e-11. Endpoint-inclusive samples
16->{0,15},32->{0,10,20,31},allinputadjoints checked. Everycompact32block/fourfusion
endpoint,allmovingmutationmodes,invalidcontracts and source workflow pass.
Warmalllearned/moving arrays;36explicitAdam.001,b2x3,seed751/state753/dropout743/order647,
first2gradientprobe advancesRNGwithoutmovingupdate. Selectpaired epoch1,val1.101417,
fixedfitinference NLL1.098741081->1.097462405. Heldoutaccuracy1/3/NLL1.099577,
all6handwaving,recalls0/0/1,below unchangedmotion1NN2/3/NLL.884778.
Freshpairedrestore/repeat0,singlebatch5.58793545e-9within1e-6,all552arraysimmutable;
reverse/repeatfirst logitL1.017159964/.016396952. Fast-only intervened frames preserve
rawSlowexactly butfinalSlowbody L1changes4.24053049;zero-lateral logitL1.242395535.
Independent NumPy original/report replay finds ALL552arrays/541110scalars and every
initial/heldoutlogit EXACT. No benchmark/pretraining/quality claim.
Evidence:build/cnn-slowfast-{coverage,hover}.json,validation/,review/;
/tmp/cnn-slowfast-fit-reference.log.118/129complete,11remaining.


### Tensor trained PointNet++SSG completed (117/129 overall)

FourexecutedC++blocks,editable/executedTikZ,eightreviewedSVGs showall256actual
observedpoints oforiginalchair/sofa/table(XY/XZ),bothactual64/16FPS centerlevels,
18visibleAdamupdates,all12originalheldoutobjectpredictions/metrics/confusions
and source-order/zero-cloudcontrols. Samecanonicalpoint_models/data/sharedsix
native/14publicexamples/sixhovers from115reused,notdouble-counted. ONEadditional
native testindependentlycomparesEVERYactualfirst/secondFPScenter,64512neighbor
IDs,3360counts,193536relativeXYZ and10080centercoordinates acrossALL42original
CADobjects with retainedoriginalhostgeometry;EXACTallvalues. Originalspecific
PointNet++three-epochrunpasses,Markdown/currentwebsiteexports resolve8media,
allHTMLtablesconsistent.

Compact93907/46learned+22moving,allradius.2/.4first64/second16centerabstractions,
16/32slots,relativeXYZ/gatheredfeatures/all3sharedMLPabstractions/classifier
retained. SourceGlorot/bias/populationBN eps.001/decay.9,keep.5afterBOTHhidden
classifierlayers. SourceFloat32geometry/DoubletensorFPSdistances,firstpointseed,
lowestindexties,strict-radiusINPUTORDER/first-hitpadding remainexplicit.
Groupingchoicesdiscrete,frozenforXYZ/featurederivatives;noPointNetpermutation
invarianceclaim. SamegenuineModelNet10/42objects/24fit6val12test/threeclasses,
originalmemberhashes/unitball/independentarea-samplingreplay as115.

Warmall68arraysbeforetensorAdam.001,seeds971/977,dropout971/order983,
b4x3,18updates,BNmovingupdateonceperfitforward,NOtransformpenaltyforthisgraph.
Postfitbatch/no-stat-updategradientprobechecksall46finite/46nonzero.
Pairedvalidationcheckpoint,nocalibration/testselectionhidden. Select1,
val1.145227;fixedfitINFERENCE NLL1.100224884->1.137933987 WORSE,explicitlyreported.
Heldoutaccuracy1/3,F1.166667,NLL1.141731,recalls1/0/0,all12predictchair,
confusion4/0/0;4/0/0;4/0/0. Initial/tie1/3,NLL1.102321/1.098612.
All8sofa/tableobjects missed;noobjectrecognitionadvantage/modelbenchmarkclaim.
EVERYsourceID/trueclass/predictionprinted;completeerrors/inference-modegapshown.
FreshXYZ-onlypairedrestore/repeat0,singlebatch1.63912773e-7within2e-6,
all46learned+22movingarraysimmutable,reversemax.0262378752(orderdependent),
zero-cloudchange.168130994. IndependentSOURCEvsREPORT replay EXACTall68selected
arrays/95571scalars and everyheldout/untrainedlogit. No label/ID/optimizer/RNG
at inference. Evidence:build/cnn-pointnet_plus_plus-validation/,review/,
fit-reference.log;sharedproofbuild/cnn-pointchecks-validation/.
117/129complete,12remaining (three3D/video reports and nineinterpretability).


### Tensor trained PointNet completed (116/129 overall)

Four executed C++blocks,editable/executedTikZ,eightreviewedSVGs showall256actual
observedpoints oforiginalchair/sofa/table(XY/XZ),bothEXACTidentityinitial
T-Nets,18visibleAdamupdates,every12heldoutoriginalobjectprediction/metric/
confusion and ordering/zero-cloud controls. Samecanonicalpoint_models/data
headers/sharedsixnative/14publicexamples/sixhovers verified115reused,
notdouble-counted. Originalthree-epochPointNetrunpasses,standaloneMarkdown/
currentwebsiteexports resolve8media,allHTMLtablesconsistent.

OriginalModelNet10source42objects/24fit/6val/12test/classeschair0sofa1table2,
first256ofindependentlyverified1024actualarea-sampledpoints/mesh;unitball
sourcegeometry/provenanceretained. Compact171836/74learned+34moving arrays,
allboth3x3/16x16T-Nets/sharedMLPs/256global/two-layerclassifier retained.
SourceGlorot/bias/populationBN eps.001/decay.9,keep.7afterBOTHclassifierlayers.
FitCE+.0005SUM((AA^T-I)^2)featurepenalty EXPLICIT in everyoptimizerstep,
batchsum ratherthanmean. IdentityzeroT-Netheads/upstreaminitialzeroadjoints
andpreBNbiaszero possibilitiesdisclosed,nottreatedasdisconnection.
Warmall108arraysbeforetensorAdam.001,init971/state977/dropout971/order983,
b4x3,18updates,BNupdateonceperfitforward. Postfitbatch/no-stat-updategradient
probechecksall74finite/74nonzero inthisactualrun. Pairedvalidationcheckpoint,
nohiddenpopulationcalibration/testselection.
Select3,val1.081701;fixedfitINFERENCE NLL1.099125861->1.128235405 WORSE,
reportedalongsidebatchtrainingobjective. Held-outaccuracy2/3,F1.538721,
NLL1.089158,recalls1/0/1,confusion4/0/0;1/0/3;0/0/4:ALLfoursofasmissed.
Initial/tie1/3;initialNLL1.101295,tie1.098612. EverysourceID/label/prediction
printed;12objects/custom3classshortfitnot40classbenchmarkadvantage.
FreshXYZ-onlyrestore/repeat0,singlebatch4.17232513e-7within2e-6,
all74learned+34movingarraysimmutable,reversemax1.0561198e-6within2e-6,
zero-cloudlogitchange.74798882. IndependentSOURCEvsREPORT fit replay matches
ALL108selectedarrays/174876scalars and EVERYheldout/untrainedlogit EXACTLY.
Evidence:build/cnn-pointnet-validation/,review/,fit-reference.log;
sharedheaders/tests/docproof:build/cnn-pointchecks-validation/.
116/129complete,13remaining.


### Tensor complete PointNet architecture checks completed (115/129 overall)

Four executed C++blocks,editable/executedTikZ,sevenreviewedSVGs,42/42documented
definitions,14compiled/runpublicexamples,sixclangdhovers,sixnative tests,
standaloneheaders,originalfullgraph/penalty9FD/real1024pointworkflow and
Markdown/currentwebsite exports pass. Canonicalpoint_models.hpp/point_data.hpp
own bothcompletegraphs,sharedMLP/populationBN/T-Net/classifier,orthogonality
penalty andgenuineobjectingress;legacyreusablelogicpureforward.
FullPointNet3480049/74learned+34moving,compact171836/74+34;
fullPointNet++1475688/46+22,compact93907/46+22,allfull40class/1024point
andcompact3class/256point graphs ACTUALLYexecuteallendpoints/derivatives on
originalCADsurfaces. Full/compactPointNetboth3x3and64/divT-Nets,sharedMLPs,
max/2layerclassifier retained;PointNet++bothFPS/radius.2/.4abstractions plus
allglobalabstractionMLPs/classifier retained. SourceGlorot/bias/populationBN,
eps.001/decay.9/gamma1beta0;movingvariancepopulation,nofusedunbiasedcorrection.
Headkeeps.7/.5 appliedafterBOTHhiddenlayersintraining.

OfficialModelNet10.zip473402300bytesSHA256
9d8679435fc07d1d26f13009878db164a7aa8ea5e7ea3c8880e42794b7307d51.
Classeschair0/sofa1/table2,perclass lexicalfirst8officialtrainfit/next2val/
first4officialtesttest,24/6/12originalobjects. OriginalOFFbytes/memberhashes,
fantriangulationtopology and allsampledraws/centers/radii retained.
DoubleTENSORtriangleareas,1024areaweightedfaces,generator941+sourceID,
sqrtuniformbarycentricdraws,centeredsampledmean/maxL2radius,F32rounding.
IndependentNumPy/OFF/xoshiro replay EXACTall129024XYZvalues,43008areaface
choices,86016barycentricuniforms across42originalmeshes. Noabstractsynthetic
shape substitutes. Source-IDdisjointness doesnotcertifynear-duplicatemeshabsence.
prepare_cnn_meshes.py/nn_mesh_prepare.cpp andcnn_mesh_reference.py repeatable.
All1024samples ofactualchair/sofa/table shownasXY/XZprojections;compactkeeps
first256withoutnewpoints. RPlot2panelsmax,shortTikZlabelsreviewed.

Nativeindependentall9penaltymatrixFD1.48719e-10,all18sharedDense/populationBN/
inputFDcoordinates,fullactuallearnedgradientsfinite,allsourcecompactrows,
explicitmovingNoUpdate/Update/Infercontracts andinvalidinputsguards pass.
BothT-NetheadsinitializeEXACTidentity withzeroW/b;upstream40of74initial
adjoints zeroEXPECTED,reportprintsall74finite/34nonzero vs++46/46. Do not
misreportzeroasdisconnectionorfituntrainedarchitecture. Penalty.0005SUM
((AA^T-I)^2),batchsum ratherthanmean,explicitcallerintrainingreports.
All12realheldoutreverse-order logits:PointNetmax2.04890966e-8within2e-6,
PointNet++max.0130864717 (sourcefirst-seed/input-ordergroupingORDERDEPENDENT).
Bothrepeat0/alllearned+movingimmutable,zerocloudsensitivity.0960100442/
.151500165. No labels/IDsenterforward. ArchitecturechecksareUNTRAINED and
nooptimizerhidden;nexttworeportsshowactualfits. Every4graph table row now
renderswithfinite/nonzerocolumns,narrativelinesdonotbreakthetable.
Evidence:build/cnn-pointchecks-{coverage,hover}.json,validation/,review/,
originalmeshmanifest/samplingarrays,independent-mesh-reference.log.
115/129complete,14remaining.


### Tensor genuine medical CT volumes completed (114/129 overall)

Six executed C++ blocks,editable/executed TikZ,eight reviewed figures(7SVG/1PNG),
18/18documenteddefinitions,eightcompiled/run publicexamples,fiveclangdhovers,
five native tests,standaloneheaders,original646coordinate/three-epochworkflow,
and Markdown/currentwebsite exports pass. Canonicalmedical_volume_models/data
own fullresidualgraph,originalingress,tensor intensityscaling/blockfeatures/
nearestmemory;legacyreusablefunctionspureforward,originalfitmomentsTensorized.

PinnedcorrectedNoduleMNIST3Dv2.1archive29299364bytesSHA256
5dc6f82bdb2b91f300263522aab82f7c034d0b264692f6d5f4c70179ad27f76b,
MD58755a7e9e05a4d9ce80a24c3e7a256f3. Official1158/165/310releasedarrayrows,
sourceIDs offset0/1158/1323. First32fit/8val/32test distinct records perclass,
64/16/64balanced,noidenticalselectedbytecubes,alloriginalsourceidentitiesretained.
CCBY4,originalLIDC-IDRIderivedCT28cube,categoricalsource0benignratings1/2,
source1malignantratings4/5,rating3excluded;labels/sourcepolicyverifiedagainst
creatorINFO. PatientindependenceNOTverifiedfromreleasedrowIDs;notclinical
orfullbenchmarkclaim. prepare_cnn_medical.py/nn_medical_prepare.cpp reproduce
selection/metadataandunchangedoriginalbytes. NativeALL3161088selectedvoxels/
labels/IDs matchofficialarchive,ALLfittingmoments matchindependenthostoracle.
Every64heldoutdepth14slice shown,plusoriginalfittingorthogonalviews;noanatomical
orientationinferredfromarrayaxes. TensorFloat32byte/255andfittingDoublemean/std
roundedF32;onlytwofitnormalizationstates,notoptimizerleaves.

Customthree-stage residual3DCNN (notnamedpaperarchitecture),width4/8/16,
28->14->7->4,SAME3stride2/ReLU+3linear+1stride2projection/ReLU,
Heuniform/b0,GAP,keep.8head,Glorotlinear2head,NoBN. 13782/20learned+2moment
scalars. ALL646independenthostkernel/projection/bias/inputFD max6.48237e-9,
all72signedoutputs max9.97916e-9,odd/evenSAMEalignment,actual20livegradients,
stagecounts/freshintensity-onlyrestore/invalidnormalization/dropoutpass.
CanonicalMedicalFeatures preservesSOURCEF32division343BEFOREDoubleblocksum,
all64blockfeatures/everyfittingrecord and allheldout1NNscores exactsourcehost
parity. Parameter-free64recordmemory,first-rowdistancetie,nohiddenoptimizer.

Warm20leavesbefore48explicitAdam.001updates,b4x3,seeds1049/1051,dropout1031/
order1061. Visiblefirst4fitrecorddropoutgradientprobeadvances sameRNG/checksall20.
Validation-onlypairedcheckpoint,unchangedfittingmomentstate. Select3,
val.670794,fixedfit.872803->.659494. HeldoutCNNaccuracy.609375/F1.551444/
NLL.681749,confusion31/1/24/8,recalls.96875/.25;miss24/32source1records.
Initial.53125/NLL.843172;block1NN.640625/F1.640537/NLL.672637,confusion
21/11/12/20;tie.5/NLL.693147;depthreverse.5625/NLL.686332,collapse.515625/
NLL.688064. ExplicitCNNbelowmemoryreferenceaccuracyandNLL,noqualityclaim.
Every64originalsourceID/label/predictionandcompleteall6predictor-controls
metricsprinted. Independentfreshintensity-onlyrestore/repeat/singlebatch0
within1e-6,all22learned/momentarraysimmutable;depthreverse/collapsechanges
8.547919488/13.473229527. Controlsarenewexplicittransformations,notobserveddata.
Evidence:build/cnn-medical-{coverage,hover}.json,validation/,review/,originalCTmanifest.
114/129complete,15remaining.


### Tensor complete RGB I3D completed (113/129 overall)

Six executed C++ blocks,editable/executed TikZ,nine reviewed figures(8SVG/1PNG),
15/15documenteddefinitions,sixcompiled/run publicexamples,fourclangdhovers,
sixnative tests,standaloneheader,original196+14coordinate/fullwidth/three-epoch
workflow and Markdown/currentwebsite exports pass. All96genuineKTHheldoutframes
and full original identity metadata displayed,shared source24/6/6clips asC3D.
Canonicali3d_models.hpp retains57inflatednormalizedunits,nine4branchMixed
modules,three stemconvs/fourpools,16endpoints,temporal2/finalspatial averagepool,
keep.5head dropout,linear1x1x1head then temporalmean. All57kernels initialize
by inflating RANDOMspatialGlorot filters across time,Tdivisor;NOpretrainedweights.
Sonnet-v1offsetONLY BN(beta,no gamma),eps.001,sourcepopulationvarianceEMAdecay.999,
mean0/variance1. Explicitly distinguished from fused-BNunbiased movingvariance.
Everyinitializedkernelcoordinate temporal-slice parity0. Independent fourvalue
populationupdatefixture givesmean1.25/variance1.125 withdecay.5,everyoutputchecked.

Native ACTUALLY executes full64x224x224/400class graph,12689984learnedscalars/
116leaves+114movingarrays,all16endpoints/independentpublishedbranchinventory/
EVERYlearnedderivative;compactdivisor8/16x32/classes3values200261withsamecounts.
All196inflation/inputhostFD max8.45954e-8,all14unitFD3.61336e-8;
all160original repeated-image values/borders agreehostoracle. Reportactualreal
firstframe explicitlyrepeatedascontrol,all65536output2D/tapfractionparity
4.56348062e-8,source3tap temporaledgefraction2/3,interior1. Tinyerroraxisrescaled
x1e9forreadableticks. Every16actualinitializedsourceclipendpoint plotted/printed.
NativeTrainNoUpdateall114unchanged,TrainUpdatechangesstate,Inferimmutable,
all57initialslicesexact,invaliddtype/shape/decay/dropoutcontractspass.

Warm116learned/114moving arrays before36explicitAdam.001,b2x3,
seed677/state679/dropout673/order647. First2fitclipgradientprobe usesbatch
moments/no stateupdate andadvances samehead RNG;every116gradientlive.
Movingpopulationupdateonceperfitforward,validation-onlyPAIREDcheckpoint.
Select3,val1.098550,fitINFERENCE NLL1.098623920->1.098829636 SLIGHTLYWORSE,
reportedhonestly,nocalibrationorqualitythresholdsubstitution.
Held-outI3D1/3accuracy/F1.166667/NLL1.098270,recalls1/0/0,all6predictboxing;
initial1/3/NLL1.098530,static1NN1/3/NLL1.218111,motion1NN2/3/NLL.884778,
tie1/3/NLL1.098612. Belowmotionreference/noactionrecognitionbenchmarkclaim.
All6predictions/all7predictor-controls printed/completeconfusionsplotted.
FreshRGB-onlypairedrestore/repeat0,singlebatch1.86264515e-9within1e-6,
EVERY116learned+114movingarrayimmutable,reverse/repeatfirstchanges
.004682282/.023916861. No label,source ID,optimizer or RNG at inference.
Evidence:build/cnn-i3d-{coverage,hover}.json,validation/,review/.
113/129complete,16remaining.


### Tensor full-depth factorized video completed (112/129 overall)

Six executed C++ blocks,editable/executed TikZ,nine reviewed figures(8SVG/1PNG),
18/18documenteddefinitions,eightcompiled/run publicexamples,fiveclangdhovers,
five native tests,standaloneheader,original530coordinate/bothfullwidthgraphs/
boththree-epochworkflows and Markdown/currentwebsite exports pass.
Canonical factorized_video_models.hpp retains parameter-balanced factor width,
explicit symmetric stride2borders,source P3D Aserial/Bparallel/C.5shortcutmix,
all2/2/2/2R(2+1)D blocks and3/8/36/3P3D blocks(finalstage spatial-only).
Release coeff1/1(B),.5/.5(C) directly verified in authors original
prototxt/deploy_p3d_resnet_kinetics.prototxt (retained validation copy/hash).
BothactualFULLreferencesizes16x112x112R2/16x160x160P3D execute allendpoints,
inventories and EVERYlearnedgradient. Full31505325/113learned+74moving,
66492240/608learned+404moving;compactbase4/classes3values123457/267567,
retainingall9/51endpoints and113/608learned+74/404moving leaves.
NativeALL530literalhostfactor/kernel/inputFD coordinates max1.70843e-8,
explicitstatepolicy and symmetricstride/invalidcontracts pass.
Independent four-value BNfixture verifies training population-normalized outputs,
fused-BN unbiased movingvariance,mean.25/variance1.0666667. All source units
fan-outKaimingNormal/BN eps1e-5/momentum.9/gamma1/beta0,moving0/1;
headN(0,.01)/b0,RGB*2-1. R2no dropout;P3Dkeep.5head. No pretraining.

SamegenuineKTH24/6/6clips/18videos/sixpeople asC3D,all96heldoutframes printed,
originalmirror revision7076149bdbeff8d70003a68d6291fd080db44683 retained.
Allfittingmotion,all60compactendpoints,sourceA/B/Crouting/adjoints and
explicitBNTrainNoUpdate/TrainUpdate/Infer controls plotted. Allshared1769472
sourcepixels/referencefeatures/scores alreadyindependentlychecked C3D.
TrainNoUpdate movingchange0;TrainUpdatechangesallmovingarrays;inference repeat0.
Warmbothstores beforetensorAdam.001,seeds727/729,dropout719/order647,
visiblefirst2fitclipbatch-moment/no-updategradientprobe advancesP3D RNG.
72explicitAdamupdates,36/model,b2x3;BNmovingupdateonceperfitforward,
pairedlearned/movingvalidationcheckpoint,nocalibrationhidden.
Bothselect2,val1.099605/1.095704;fixedfit1.098353->1.094324 and
10287.326152->1.116260. LargeinitialP3DNLL/activationgrowth explicitly
attributedto uncalibrated0/1movingmoments alongsideweights;improvement is
NOTclaimedasweight-onlyoptimizationorusefulrecognition. Log10NLLplot
retainsinitialheldoutP3D8386.22,tableordinaryvalues,noaxisclipping.
Held-outR2accuracy.5/F1.412698/NLL1.094525,recall1/0/.5,confusion
2/0/0;2/0/0;1/0/1. P3D1/3/F1.166667/NLL1.122472,recalls0/0/1,
all6predicthandwaving. BothmissALLhandclapping andbelowmotion1NN2/3,
NLL.884778. Every6prediction/7predictor-control metrics/confusions printed.
IndependentfreshpairedRGB-onlyrestore/repeat0,singlebatch2.98023224e-8/
4.02331352e-7within1e-6,all187/1012learned+movingarraysimmutable.
Reverse/repeatfirstchangesR2.119245425/.143999757,P3D.710639759/.871223575.
Evidence:build/cnn-factor-{coverage,hover}.json,validation/,review/.
112/129complete,17remaining.


### Tensor C3D and genuine KTH video ingress completed (111/129 overall)

Six executed C++ blocks,editable/executed TikZ,nine reviewed figures(8SVG/1PNG)
show every96held-out frame,all fitting temporal changes,negative ceil-pool
fixture,actual initialized features/stages,all24static/motion reference features,
36visible Adam updates,every held-out prediction/confusion/control/metric.
c3d_models.hpp/video_data.hpp pass27/27documenteddefinitions,ninecompiled/run
publicexamples,sixclangdhovers,standalone includes,original164coordinate/
pool/compact/fullmetadata/three-epoch workflow and six native tests.
Native ACTUALLY executes full79991015-scalar/22leaf Sports1M-shaped graph at
16x112x112,all8conv/5pool/FC shapes and alllearned gradients,not only census.
Compact114963/22,base4/FC64/classes3/32px,all original8SAME3D/ReLUconvs,
fiveCaffe-ceilpools(firsttime1),two dense/ReLU/dropouts,linearhead,NoBN.
SourceHeuniformconv/Glorotdense/RGB*2-1 explicitly disclosed,no pretraining.

Original KTHserver fails502;retained originalAVI/00sequences via publicmirror
AtharvaPote/KTH-Action-Recognition-mirror revision
7076149bdbeff8d70003a68d6291fd080db44683. All18originalAVIs and theirhashes,
original annotations/selectedframeIDs/segmentbounds retained inmanifest.
Boxing0/handclapping1/handwaving2,scenario1,people1..4fit/5val/6test,
first2annotatedsegments/video,16unique evenlyspacedactualframes/segment,
24/6/6balancedclips,person/video-disjoint. Not consecutive-original-frame or
published KTH/Sports1M benchmark claims. FFmpegcodecdecodeonly;TensorF32legacy
bilinear32px and/255. Native independently checks ALL1769472pixels withliteral
hostbilinear and everyidentity;no synthetic motion. Gallery shows all6test
clips x16actualframes. OriginalKTHnoncommercial terms retained,not remirrored.
prepare_cnn_video.py/nn_video_prepare.cpp repeatable;sourceAVI filesremainignored.

CanonicalVideoFeatures usesDouble tensor4x4spatialmean all16colors or15absolute
frame differences thenF32rounding;VideoNearest fittingDoublepopulationstd>=1e-6,
first-row distance ties,retainedmemory,NOlearnedparameters/optimizer.
EVERYfitfeature/everyheldoutneighbor/metric agrees retainedoriginal hostoracle.
Legacyreusablemodel/data/referenceforward;ScoreLogitsforwards canonicaltensor
ScoreSensor metrics. Nativeall164signed3Dweights/bias/inputFDerror1.66264e-8,
negativeodd-edgeceilpool/alladjoints and invaliddropout/inputcontracts pass.

Warm22leaves;36explicitAdam.001updates,b2x3,init643/dropout631/order647,
visiblefirst2fitclipdropoutgradientprobe advances samegenerator/checksall22.
Select1,val1.099830,fixedfit1.106528->1.098891. Held-outlearnedaccuracy1/3,
macroF1.166667,NLL1.096773,recalls0/1/0,all6predicthandclapping,confusion
0/2/0;0/2/0;0/2/0. Motion1NN2/3/NLL.884778/static1NN1/3/NLL1.218111,
initial1/3/NLL1.104972,tie1/3/NLL1.098612;nolearnedaccuracyadvantage.
Fresh RGB-only restore/repeat0,single/batch1.95577741e-8within1e-6,
22leavesimmutable;reverse/repeatfirstchanges.097756668/.229040474.
Everyheldoutpredictionprintedandcompleteconfusionsreviewed. Tensor metrics,
validation-onlyselection,oneheldoutperson/sixdependentsegmentscope explicit.
Markdown/currentwebsiteexports resolve9media,allHTMLtablesconsistent.
Evidence:build/cnn-c3d-{coverage,hover}.json,validation/,review/,KTHmanifest.
111/129complete,18remaining.


### Tensor point geometry completed (110/129 overall)

Four executed C++ blocks,editable/executed TikZ,five reviewed SVGs show all12
source-fixture XYZ points (z=0),both FPS selections,all32neighbor identities,
capped counts,allfirst-cloud feature/XYZ adjoints and36Double finite differences.
Canonical point_geometry.hpp passes13/13documenteddefinitions,sixcompiled/run
publicexamples,fourclangdhovers,standalone inclusion,original full60-coordinate
hostFD fixture and four native tests. Markdown/current website media/tables pass.
Legacy geometry is now pureforward. FPS/radius distances,nearest updates,
ranked input-order selections and batch offsets are ALL tensor computation,
not former host numeric arrays. Source Float32 ingress then Double distances,
first-index seed/ties,collapsed repeats,strict radius,first input-order hits,
first-hit padding and capped unpadded counts retained. Supports slots>N.
Discrete outputs detached;gather/subtraction retain live feature/coordinate
adjoints,duplicate scatter-add and negative centroid terms. Frozen-choice
Jacobian explicitly distinguished from selection discontinuities.
Native retained independent original host source compares EVERY FPS/radius/
count/gather value over12three-batch clouds x9radius/slot settings;original
fixture independently checks64gathered values,96relative values,all60feature/
XYZ hostfinite differences(maxbelow3e-7),ties/strictboundary/translation/errors.
Reportall36Double frozen-coordinate FD error maxbelow4e-12,translation exact,
all2cloud coordinate translation-adjoint sumsbelow3e-7. No fitted parameters,
optimizer or classifier-quality claim;these declared fixtures are not CAD data.
Evidence:build/cnn-geometry-{coverage,hover}.json,validation/,review/.
110/129complete,19remain (video/model reports continue separately).


### Tensor gated causal WaveNet stack completed (109/129 overall)

Six executed C++ blocks,editable/executed TikZ,seven reviewed SVGs,
35/35 documented definitions,13compiled/run public examples,sixclangdhovers,
sevennative tests,standalone headers,originalfull/all33coordinate/three-epoch
workflow and sampled feedback,Markdown/current website exports pass.
Canonical wavenet_stack_models/data.hpp own causal tensor graph,warmupNLL,
validated original ingress,segment count targets,prior scores and tensor metrics;
legacy core functions forward. Production rollout CDF/code/context/codec arithmetic
is tensorized;only categorical indices/RNG draws/presentation remain host.

Pinned official FSDDv1.0.10 archive16419872bytes SHA256
c680098f5d2355ba7cc4bd8b0ca9e0a24c00ebe3b4c552208889a4d28d4fbc2b,
27original8kHzPCM16mono digit0/1/2clips. Custom speaker-disjoint18fit/3val/6test
clips from george/jackson/lucas/nicolas,theo,yweweler;not official take split.
First1025actual samples,no resample/peaknorm/syntheticpad;Double tensor mu-law256,
starts0/256/512/768,strict next-sample shift. 72/12/24windows,scoredt127..255,
129samples/window,3096held-out samples. Manifest retains originalclip hashes,
identities,frames/license;prepare_cnn_audio.py/nn_audio_prepare.cpp repeatable.
Independent PCM/NumPy reference matches ALL55296input/target values and1152prompt
values;native separately checks all27sources/windows/codes/prompts too.

Full124096/76learnedscalars/leaves,R32/S64;compact14512/76,R8/S16.
Initialtwo-tap causal conv,12tanh*sigmoidgates,dilations1..32twice,
1x1skips/allbutlastresiduals,skip sum/ReLU/postReLU/head,Glorot/b0,RF128.
NoBN/dropout/conditioning. All33tinyhostcoordinates/fullall76gradients,
exactRF5Jacobian/allreachablelags/futurezero,allcodec bins/extremeNLL/adjoints pass.
One visible DoubleSGD(.5) fits65792add-oneunigram/transitioncountvalues,
thenlogscores;countparity0,allreference scores/metrics agreehostoracles.
Warm76leaves,27explicitAdam.001updates,b8x3,seed541/order557.
Select1,val5.543614,fit5.551162->5.535600. HeldoutlearnedNLL5.535256,
initial5.550588,unigram4.940436,transition4.603732. LearnedamplitudeMSE.000148357
vs transition.000114889,persistence.000038696795;exactcode.001292vs.094315
transition. Explicit poor comparison/no speech-quality claim.

Fresh previous-code-only restore/repeat0,all24heldoutprefixes1/64/128/256
max3.16649675e-8within2e-6,futureearlier0/later13252.614975,76leavesimmutable.
Three real128code seeds+256sampledcodes,minstd_rand547,DoubletensorCDF,
actualsampledfeedback ONLY,all768draws/logits retained,six48msWAVs.
IndependentNumPyexact768draws/inverseCDFsamples,completefresh-storereplayall
196608step logits fromactualpreceding128seed/sampledcontext exact0.
Generatedamplitudesnear1vsquietseed/nointelligibilityclaim;differentfuture
codes256/256/255. MSEplotrescaledx1e6forreadableticks,shortlegendsreviewed,
actualgates sharedata-derivedsymmetricrange. ExportHTMLtablesconsistent.
Evidence:build/cnn-wave-{coverage,hover}.json,validation/,review/,
scripts/cnn_audio_reference.py,independent-audio.log,feedback.log.
Entire1.9group6/6complete;109/129overall,20remain.


### Tensor raw-character CNN completed (108/129 overall)

Six executed C++ blocks,editable/executed TikZ,eight reviewed SVG figures,
23/23 documented definitions,11 compiled/run public examples,six clangd
hovers,six native tests,original full/all62-coordinate/three-epoch workflow,
standalone headers and Markdown/current website exports pass.
Genuine MR source rows exactly match the word CNN (1000/200/200 balanced),
last256 Latin1bytes reversed/ASCII lowercased,source69slot duplicate-hyphen
alphabet,last match,unknown/space/PAD-1 structural zero. Native independently
checks EVERY1400raw source/length/label/ID,alignment and truncated-byte-group
separation. Word first64tokens vscharacter last256bytes are different views.
Full11335938/18learned scalars/leaves,T1014,C256,dense1024,flat8704;
compact22162/18,T256,C16,dense64,flat80;all six7/7/3/3/3/3valid convs,
three3stride3pools,two dense/ReLU/dropouts,binary head,NoBN/embedding.
All62tiny host finite differences/full every-leaf backward pass.
One visible Double SGD(.5) fits10462character/word/class count values,
normalizes/scores them explicitly,all200score parity0;closedformsoraclesonly.
Warm18leaves,60explicit MomentumSGD.01/.9,b50x3,keep.5twice,seed2/
dropout367/order383. Corrected old invalid2000row probe to first25fit rows
perclass in actual1000pool,source/report match;no seed search. Select1,
probe.699299->.696613;CNNaccuracy.455/NLL.694620/confusion82/18/91/9,
initial.505/.693648,charNB.55/.725349,wordNB.655/.843006,tie0.5/.693147.
Report explicitly records poor CNN outcome/below count references.
Fresh character-ID/length-only restore/repeat/singlebatch0within1e-6,
padding0,18leavesimmutable,observed replacement logit change.0859848.
All original count/reference values and every known byte checked independently;
all figures visually reviewed and exported HTML tables have consistent widths.
Evidence:build/cnn-character-{coverage,hover}.json,validation/,review/.
108/129complete,21remaining.


### Tensor MR word CNN and standard Adadelta completed (107/129 overall)

Six executed C++ blocks,editable/executed TikZ and eight reviewed SVGs show
all1400 genuine word lengths,fitting-word counts,actual initialized wide maps,
structuralPAD0/zero adjoints,ordinary tensor Adadelta histories/separate head
projection,full/compact graph,visible SGD unigram counts,all60 neural updates,
complete held-out confusion/metrics and six original review excerpts.
kim_cnn_models.hpp/sentence_data.hpp pass28/28 documented definitions,
11 compiled/run public examples,seven clangd hovers,standalone inclusion,
original C++23 full/every-weight/three-epoch workflow and eight native tests.

Official Pang/Lee MR v1.0 archive487770bytes SHA256
a05befe52aafda71d458d188a1c54506a998b1308613ba76bbda2e5029409ce9,
MD550c1c2c047b4225e148e97aa7708c34e. New scripts prepare_cnn_text.py/
nn_text_prepare.cpp retain10662original Latin1source lines,ASCII word/digit/
apostrophe first64 tokenization,duplicate-token-group holdout excluding
conflicting labels,std::mt19937seed337 balanced500fit/100val/100test perclass
and final row shuffle. Actual1000/200/200rows,5161fitting vocabulary entries
(PAD0/UNK1,frequency then lexical ties,max8000). Only fitting strings choose
vocabulary. Original source IDs/labels and raw text preserved,UTF8 reviews.tsv
for presentation;same IDs/labels also prepare raw-character tensors,with
cross-split truncated-character duplicate verification. No synthesized review
inputs,pretrained embeddings or published cross-validation score claim.
Native independently reparses EVERY original source review using separate
ASCII regex,all1400token/length/label/ID sequences,all5161ranked vocabulary
entries and original duplicate-group separation. Count features use tensor
segment reduction O(NV+NT),not huge[N,T,V] onehot cubes. Every fitting count
and all200 NB scores match retained original host source.

Full CNN-randV8000/E300/F100 executes2760602 scalars/9leaves,every learned
derivative;actualcompactV5161/E32/F32 has177698/9. Only[V-1,E]embedding
rows learn,U(-.25,.25);PAD0structuralconstant0. Length mask before4zero pads
each side,VALID3/4/5 Glorot/ReLU maps70/69/68,max-over-time,concat3F,
keep.5pooled dropout,linear binary head. Fully padded boundary windows can
carry learned bias,explicit source policy rather than masked-window pooling.
Every80tiny-coordinate host finite difference,value/padding/order check passes;
structuralPADtable row and adjoint0,legal out-of-length substitution invariant.
Dropout every4096values and shared-mask derivative/identity pass.

Tensor OptimizerKind::Adadelta is used DIRECTLY in report loops,rate1,rho.95,
eps1e-6,zero history;explicit post-step KimMaxNorm copies classifier/W under
NoGradGuard,only head columns capped3. Legacy KimAdadelta is now a thin
standard-optimizer/spelling/history adapter,not a second hand-coded update
rule. Three signed update steps/EVERY history/weight coordinate and maxnorm
match independent literal host oracle(max1.35265e-7). Actual optimizer slot
plots distinguish squared-gradient and nearly equal squared-update histories;
source state names accum/accum_update,18historyarrays after neural updates.
Negative-ratio bar bounds corrected during visual review to retain zero axis
and keep bars out of titles;all result/table lines separately rendered.

Visible Double SGD(.5) fits10324add-one token/class count values in one
summed-count-regression update,then explicit normalization/NB scoring;
closed-form helper parity oracle only,every held-out score exact0parity.
Fitting negative/positive word-ratio tails include actual vocabulary labels.
Warm9learned leaves before60explicit Adadelta+projection updates,batch50/
3epochs,seed317/dropout313/order331. Visible50review dropout derivative probe
advances the same generator as source and verifies all9live gradients.
Select epoch2,val.690920,probe.806359->.623513. Held-out200reviews:
CNNaccuracy.565000/NLL.685829/confusion40/60/27/73;initial.500000/.756972;
SGDunigram.655000/.843006/confusion65/35/34/66;class0tie.500000/.693147.
Explicit CNN accuracy BELOW unigram despite lower NLL,negative/positive recall
.40/.73,not useful accuracy/paper advantage. Original review excerpts and every
held-out class count remain visible;not verified movie/author independence.

Independent observed-word/length-only restore/repeat parity0,single/batch0
within printed1e-6,legal padding substitutions0,all9leaves immutable,
observed-word->UNK logit change.425801. Selected head norms1.39202/1.29442,
actual60-step norm curves below3;toy explicitly activates cap,neural run does
not claim cap activation. Inference takes no label,optimizer/history or RNG.

Evidence:build/cnn-kim-{coverage,hover}.json,build/cnn-kim-validation/,
build/cnn-kim-review/,original MR manifests/reviews.tsv. Markdown/current
website resolve8media;rendered tables have consistent cell counts. Also fixed
selective-rerun physical result-line joining in VQA/TCN/sensor reports and
re-executed affected blocks,then verified8/4/7 separate HTML tables with
consistent row widths;no result marker has collapsed physical lines anywhere
in the completed CNN tree. Those report counters are unchanged.
Combined2819/2819 across129CNNheaders;107/129complete,22remain.


### Tensor original ECG/six-channel residual CNNs completed (106/129 overall)

Six executed C++ blocks,editable/executed TikZ and ten reviewed SVGs show all280
original records,genuine ECG/four-class six-channel examples,weighted loss/
every logit adjoint,tie-aware AUROC,full/compact graphs,visible SGD centroid/
count references,all42 Adam updates and complete source-class held-out metrics/
confusions/recalls/channel interventions. sensor_models.hpp,sensor_data.hpp,
sensor_evaluation.hpp pass35/35 documented definitions,12 compiled/run public
examples,seven clangd hovers,independent inclusion,original C++23 full checks/
both three-epoch fits and eight native tests. Thin old adapters retain harness
and optional externally supplied TF artifact oracle;no unavailable TF parity
claim. Source numeric channel-replacement/count-prior operations now use
tensors too;native and original workflow rerun after those changes and pass.

Official ECG200 archive298857bytes SHA256
755937ea37849346ea20033666b19d17345bb039971e8fa33458b16b62c09380;
BasicMotions556567bytes SHA256
f427609dfe343b1da0dc54ee361f61579e7f18c2a007121740a9f0d90286fac3.
prepare_cnn_signals.py/nn_signals_prepare.cpp retain original TS source values,
class order/channels/source IDs,seed407 rounded20% class-stratified train-only
holdout and unchanged official test. ECG96x1:80fit25/55classes,20val6/14,
100test36/64;original -1/+1 become0/1 without guessing clinical code meaning.
Basic100x6:32fit8each,8val2each,40test10each;original Standing/Running/
Walking/Badminton0..3,accelerometerxyz then gyroscopexyz. Explicit original
record separation NOT verified patient/participant holdout or unprocessed mV.
New independent scripts/cnn_signals_reference.py reads DIFFERENT original
TXT/ARFF formats versus prepared TS ingress:every19200 ECG and48000 Basic
sample,all IDs/labels exactly match;also independently verifies ALL30000
GunPoint samples used104/105. Oracle-only NumPy from pinned separate reference
environment;all report numeric preprocessing/fits/metrics/plots use tensors.

Fitting Double per-channel population mean/std and N/(Kcount) weights round
Float32 once,never use held-out inputs. Native retained host loader confirms
every mean/std/weight exactly;every raw ingress sample/ID/label matches literal
source metadata. Separate two state arrays contain mean/std of all channels.
Three He-uniform temporal residual stages:5x1stride2/ReLU,3x1stride1,learned
1x1stride2skip,ReLU,GAP/linear head. TensorFlow SAME odd/even phase preserved,
no BN/dropout. Full ECG506050/20leaves and Basic508484/20 execute every
learned derivative;compact8 ECG8154/20,Basic8460/20. Exact stage lengths
96->48->24->12 and100->50->25->13. Independent EVERY171tiny-coordinate host
finite difference verifies multichannel/stride/projection/GAP/head. Every
input channel has gradients and actual replacement sensitivity. Invalid saved
std,missing source-class support and undefined AUC reject explicitly.

Weighted CE denominator is raw row count,not selected weight sum. Extreme
logits/every derivative and source-class weighting fixtures pass. Canonical
metric reductions preserve stable Double NLL,all confusion cells,equal-class
balanced recall/macroF1 and original class recall. Binary class0AUROC compares
Double logit0-logit1 pairwise,ties half;fixture.625 passes. Multiclass auc0
unused and report labels unavailable. Every real metric/reference output
matches retained independent host source;Float32 nearest subtract/divide then
Double square reduction,first-row ties and Double centroid arithmetic retained.

Visible Double SGD(.5) fits ECG192centroid+2count values and Basic2400+4,
class-balanced per-coordinate MSE plus count-regression target(add-one count).
Closed-form centroid branch parity oracle only,every test score exact0parity.
Retained-observation standardized1NN has no optimized leaves. Warm20learned/
2state before Adam(.001),seed587/state593/order599,batch8/3epochs,30ECG+
12Basic=42explicit updates. Paired learned/normalization validation selects
both epoch3:ECGval.716165,probe3.943199->.637953;Basicval1.318549,
probe3.991817->1.077909. All actual280source records and every class shown.

ECGtest learned accuracy.410000,balanced.539062,macroF1.347273,
NLL.769382,AUROC.784288,recall1/.078125,confusion36/0/59/5. Untrained/count
accuracy.640000,standard1NN.900000,SGDcentroid.750000;ranking AUROC does NOT
rescue poor argmax classification. Basic learned accuracy/balanced.275000,
macroF1.174286,NLL1.256781,recall0/.3/0/.8,all Standing/Walking fail.
Standard1NN.500000,SGDcentroid.425000,untrained/count.250000. Explicit no
useful learned-model/benchmark advantage,full class failures remain visible.

Independent raw-signal-only restored/saved/repeated logits exact0parity,
single/batch0within printed2e-6,all22learned/state arrays immutable. ECG
channel replacement.150435;Basic.056370/.100093/.134089/.0417562/.0221467/
.022233;reversal total changes.996276669/.632113844. These are declared
observed-input interventions,not feature importance/accuracy or new sources.

Evidence:build/cnn-sensor-{coverage,hover}.json,build/cnn-sensor-validation/,
build/cnn-sensor-review/,original signal manifests and independent source
oracle log. Markdown/current website resolve10media. Combined2791/2791
across127CNNheaders;106/129complete,23remain. Genuine MR word/character and
pinned FSDD audio preparations also ready for remaining1D reports.


### Tensor causal TCN completed (105/129 overall)

Six executed C++ blocks,editable/executed TikZ and nine reviewed SVGs show
all200 actual GunPoint signals,exact RF15 input Jacobian/future-zero fixture,
weight-normalization values/adjoints/scale intervention,full/compact graph,
visible SGD centroid reference,all15 actually clipped Adam updates,all150-row
prefix inference and actual held-out future perturbation/metrics/confusions.
tcn_models.hpp passes16/16 documented definitions,six compiled/run public
examples,four clangd hovers,standalone inclusion,original C++23 syntax/full
backward/three-epoch workflow and six native tests. Uses verified canonical
series/sequence helpers from104;all numeric kernels/clip/plots use tensors.

Same pinned GunPoint observations:40fit/10val from official50train,all150test,
seed407 class-stratified20% holdout,exact original values/IDs. Original label
0gun-draw/1pointing and two-actor dependence retained. Fitting-only scalar
population normalization saved separately;no prefix statistic estimation.
Explicit distinction between causal observed-input graph and physically online
raw-sensor/source-preprocessing guarantees,not actor-disjoint benchmark.

Full width64 executes137154 learned scalars/40leaves;compact8 has2362/40,
2normalization leaves,6residual blocks/two convolutions each,dilations1..32,
RF253. v~N(0,.01),g=sqrt(2.f),b0,W=g*v/sqrt(sum(v²)+1e-12),left-only
padding/VALID dilated convolution,ReLU/keep.8dropout,identity/projected skips
and shared per-time binary head/last-time classification retained. Every full
learned gradient and block geometry pass. Independent host causal equations
and EVERY20tiny-coordinate finite differences pass. RF15 positive fixture
has exact-zero input Jacobian outside support,all reachable lags nonzero,
future prefix outputs exactzero and dynamic prefix equality. Native real-data
future gradients/prefixes pass. Bad dropout generator and receptive-field
overflow reject before leaf creation. Actual v*=3 forward error1.1920929e-7
includes epsilon/Float32 rounding;no exact scale-invariance claim.

One visible Double SGD(.5) step refits300class-centroid values,closed-form
helper parity oracle only,all held-out scores exact0parity. Raw1NN is
parameter-free retained observations. Warm40learned/2state;initial/dropout
461/463/457 and batch467,visible full-fitting dropout derivative probe matches
source RNG advancement. All15 explicit Adam(.001) updates,batch8/3epochs,
Tensor Double norm reduction/sourceFloat32 scale min(1,1/(norm+1e-6));
ALL15 actually clipped. Native literal Float32 norm/scale/every-gradient
coordinate parity passes. Log10 norm plot makes cap1 visible. Original
workflow global clipping and future perturbation arithmetic also converted
to tensors,then native+original checks rerun and pass.

Select epoch3,val.541272,probe4.985522->.670643. Actual held-out accuracy/NLL
learned.426667/.714699,untrained.466667/4.193172,raw1NN.906667/.406595,
SGDcentroid.660000/.676829,class0tie.506667/.693147. Learned confusion
11/65/21/53 exposes gun-draw failures. Explicit accuracy BELOW untrained,
majority and both distance references despite better NLL,no useful TCN/
benchmark advantage claim. Independent observed-signal restore/repeat parity0,
whole150-row dynamic prefixes1/17/75/150 max2.38418579e-7,single/batch0,
printed tolerance1e-5,all42learned/state leaves immutable. Actual future-only
perturbation from75 gives exact0earlier change and394.420254 later absolute
logit change;causal sensitivity is not a new independent accuracy sample.

Evidence:build/cnn-tcn-{coverage,hover}.json,build/cnn-tcn-validation/,
build/cnn-tcn-review/,shared original GunPoint manifest. Markdown/current
website resolve9media. Combined2756/2756 across124CNNheaders;
105/129 complete,24remain. MR text/character sources and pinned six-speaker
FSDD audio archive prepared/downloaded for later reports,not yet conversions.


### Tensor temporal FCN/GunPoint completed (104/129 overall)

Six executed C++ blocks,editable/executed TikZ and eight reviewed SVGs show
EVERY200 original raw signals,four actual fitting motions,even8 SAME impulse/
adjoint,explicit sequential fitting BN calibration,full/compact graph,
visible SGD centroid reference,all15 Adam updates,paired normalization/state
restore and held-out accuracy/NLL/confusions. sequence_classification.hpp,
time_series_models.hpp,time_series_data.hpp pass37/37 documented definitions,
13 compiled/run public examples,seven clangd hovers,independent header inclusion,
original C++23 syntax/full checks/three-epoch workflow and six native tests.

Official GunPoint archive435433bytes SHA256
d7513cfe222418fabfdb5a6434ffb21ac3de4923e637971e9388ebc857816803.
New scripts prepare_cnn_signals.py/nn_signals_prepare.cpp retain raw original
numeric strings and exact source IDs. Official50train/150test,T150/C1;
seed407 per-class20% training holdout gives40fit/10val,all150test unchanged.
Source labels1gun-draw/2pointing map0/1,original two-actor dependence explicit,
not actor-disjoint. Every raw sample/ID/label and fitting Double population
mean/std matches independent literal ingress/reductions. Fit normalization
roundsFloat32 once and lives in separate detached checkpoint state. No numeric
augmentation,per-test normalization or fabricated signal observations.

Full128/256/128 FCN264962 scalars/14 leaves;compact16/32/16 has4450/14;
2scalar normalization+6BN leaves. Three SAME8/5/3 BN/ReLU stages/GAP/linear
binary head,no dropout. Independent host finite differences for EVERY40tiny
learned coordinate verify even8 asymmetric3left/4right,BN-affine,normalization,
GAP/head values. Full every-leaf derivative executes. Explicit BN modes frozen,
batch updating,batch readonly and sequential all-fitting population calibration
are retained. Native checks every calibrated channel moment through each layer,
frozen-state immutability/single-batch parity. Generic tensor categorical CE,
Double equal-record NLL/accuracy and exact Int64 confusion/predictions all
match independent every-record literal oracles;no numeric host metric loops.

One visible Double SGD(.5) update fits300 centroid values with class-balanced
per-coordinate MSE;closed-form tensor centroid branch is parity oracle only,
every held-out distance score exact0parity. Raw1NN is parameter-free retained
observations,Float32 differences then Double square sum,first fitting-row tie.
Native explicit tie/known-centroid fixtures pass. Scores are declared raw
pseudo-logits,not calibrated likelihoods. Warm14learned/8state before15explicit
Adam(.001) updates,batch8/3epochs,initial419/state421/order431. All40fitting
records once per epoch;sequential fitting-only calibration before validation
and at initialization. Validation selects paired learned/normalization/BN epoch2,
val.549827,fixed fitting calibrated CE.804196->.632365. Separate raw-signal-only
restore/repeat parity0,single/batch printed tolerance1e-6,all22leaves immutable,
time-reversal logit change.0198425. No target/state recalibration at inference.

Actual held-out150records:learned accuracy.626667,NLL.625393,confusion75/1/55/19;
untrained.506667/.806120;raw1NN.906667/.406595;SGDcentroid.660000/.676829;
class0majoritytie.506667/.693147. Explicit FCN worse than both distance
references/no benchmark improvement claim,pointing-class failure visible.

Signals preparation also pins genuine ECG200298857bytes SHA256
755937ea37849346ea20033666b19d17345bb039971e8fa33458b16b62c09380
and BasicMotions556567bytes SHA256
f427609dfe343b1da0dc54ee361f61579e7f18c2a007121740a9f0d90286fac3.
Those remain prepared assets for later sensor report,not completed conversions.

Evidence:build/cnn-series-{coverage,hover}.json,build/cnn-series-validation/,
build/cnn-series-review/,build/cnn-datasets/time_series/manifest.json.
Markdown/current website resolve8media. Combined2740/2740 across123CNN
headers;104/129 complete,25remain. Original remaining1.9data dependences
are preserved;all reusable numerical FCN/data/reference helpers forward to core.


### Tensor human-question VQA completed (103/129 overall)

Six executed C++ blocks,editable/executed TikZ and eight reviewed figures show
masked recurrent memory/zero-PAD adjoints,stable ten-answer soft CE,official
consensus count formula,full/compact joint/unimodal graphs,genuine photos,
visible SGD count references,all288 Adam updates,coverage/type diagnostics
and all89 actual held-out question-image observations. vqa_models.hpp,
vqa_data.hpp,vqa_visualization.hpp pass64/64 documented definitions,21 compiled/
run public examples,nine clangd hovers,separate header inclusion and original
C++23 syntax/full backward/three-model workflow. Six native tests pass.

Pinned official VQA-v2 train questions7,239,401bytes SHA256
05a64b6e2582d06d7585f5429674a9a33851878be1bff9f8668cdcf792df611e;
train annotations21,708,861bytes SHA256
fb101bcefe91422c543c2bb6d70af11eb3119d0ff745ae283d09acdf66250853.
Exact COCO2014 question image IDs match85 previously pinned genuine COCO128
2017 cached photographs;identity overlap explicit. All415 human questions
retain ten released human answers. Sorted source IDs/Random271 select55fit/
15validation/15test photos,254/72/89 questions,all questions of each photo
stay together. Fitting-only frequency/lexical vocab417tokens(PAD0/UNK1),
64answers(UNK0),time24. Answer strings lowercase/whitespace collapse,
numberword mapping none/zero..ten,article removal;explicitly NOT full official
punctuation/contraction normalization or benchmark evaluation. Scripts
prepare_cnn_vqa.py/nn_vqa_prepare.cpp retain full provenance,all selected
human text/answers and source hashes. Numeric JPEG RGB/255-before-legacy64px
resize,ten-ID onehot frequencies/LOO consensus,UNK-zero and hard firstmax
use tensors. Native independently checks EVERY source RGB pixel for85photos,
all415 numeric question records,answer targets/scores/prefixes/IDs. ExactInt64
question-ID gather regression above2^53 passes. Original host metric/count
oracles verify every real score/prior;every preview pixel matches source.

Full/compact graph and all24 learned/both modality derivatives pass;14joint
stages/24shared LSTM steps. Six-convolution CNN/global average/tanh projection,
shared i/f/g/o LSTM/forgetbias1,frozen h/c after lengths,concat image/question/
product fusion and linear head retained. Independent host every-state and
52-coordinate finite differences verify recurrent gates,cell freeze,zero
length,padding-token zero gradients and order sensitivity. Large1000-logit
soft CE and every adjoint pass. Official ten-annotator LOO0/.3/.6/.9/1 sequence
verified;no min(k/3,1) substitution. Full normalization contract/data leakage
checks use tensors,host sets manage IDs only. Original text adapter now reads
actual3column question metadata and maps hard-majority labels correctly.

One visible Double SGD(.5) update fits16,320 global/prefix count values,
add-one global normalization/prefix smoothing explicit,closed-form helper
oracle only. Separate joint/question/image stores warm before96 explicit
Adam(.001) updates EACH,288total,batch8/3epochs,seed277/order281. Actual
vocab417 gives60,584/24joint,32,096/10question,28,616/18image leaves.
All select epoch3;validation NLL1.888420/2.134730/2.261480;fixed probes
4.381060->2.348700,4.160640->2.730420,4.511800->2.773140.
Held-out mapped consensus joint.308989/question.312360/image.312360,
SGDglobal.312360/SGDprefix.392135. NLL1.700596/2.028432/2.140744/
2.230419/1.500836. Explicit no joint advantage/grounding claim;learnedjoint
number/other0,yes-no.6875. Known-majority coverage.606742,attainable mapped
vocabulary score.676404;questions share15photos,not89independent images.
Separate observed-input restore/repeat parity0,single/batch max1.1920929e-7/
3.27825546e-7/1.78813934e-7 within2e-6,all24/10/18leaves immutable.
Image/question mean sensitivity joint2.32780039e-5/.0106978975,
question0/.00522722257,image.000516392873/0. Inactive modality exactzero.
All89 held-out image/question prediction tiles and first16 original text/
majority/predicted-answer rows shown;genuine photographs and UNK failures visible.

Evidence:build/cnn-vqa-{coverage,hover}.json,build/cnn-vqa-validation/,
build/cnn-vqa-review/,build/cnn-datasets/vqa/manifest.json. Markdown/current
website resolve8media. Combined2703/2703 across120 CNN headers;
103/129 complete,26 remain;all20 other-vision reports complete.


### Tensor OpenPose directed fields completed (102/129 overall)

Six executed C++ blocks,editable/executed TikZ and nine reviewed figures show
directed Gaussian/strip targets,missing-safe MSE/every adjoint,complete six-stage
graph,bilinear decoder,actual full-image fitting inputs,visible SGD prior and
all9 Adam updates. Canonical openpose_models.hpp,paf_data.hpp,paf_decode.hpp,
paf_evaluation.hpp pass77/77 documented definitions,32 compiled/run public
examples,eight clangd hovers,separate header inclusion,and original C++23
syntax/full forward-backward/three-epoch workflow. Seven native tests pass.
Thin legacy adapters preserve mutable host field fixtures while all canonical
numeric operations use tensors;host containers manage topology/IDs only.

Same official pinned COCO8-Pose archive/preparation as hourglass:whole-image
6fit/1validation/4test,IDs36/49/77 plus anatomical mirrors fit,86 validates,
110/113/136/151 held out. Original full RGB128 enters prediction alone,no
person crops or boxes. COCO17->18 derives neck only with both shoulder labels.
Float32 ingress/Double half-pixel Gaussian and strip geometry,Float32 overlap
accumulation,conservative unknown-box masks/background all preserve source.
Independent retained host source verifies EVERY real target/mask coordinate
at7/16 resolution,every local peak/link/group and grouped metric. NaN-safe
loss,every source adjoint,opposite/orthogonal overlaps,zero limbs,plateau ties,
affine samples,direction reversal,one-to-one links/conflict-safe groups pass.
Every RGB/skeleton/confidence/magnitude preview pixel matches independent
literal geometric/half-byte oracle. Continuous distance<=1 overlay is an
explicit display-only change from legacy sampled3x3 dots.

Full128px model executes52,311,446 scalars/184 leaves;compact divisor8
1,217,438/184. All92 convolutions,100 stages,all learned/RGB derivatives and
both initial-head feedbacks pass. BGR255/256-.5,VGG10+CPM2,three VALID pools,
six19/38 linear head pairs and five7x7 refinements are retained,no BN/pretrain.
Explicit Double SGD(.5) fits14,592 spatial reference values in one update;
closed-form tensor mean is a parity oracle only. Warm184 leaves before
9 explicit Adam(.001) updates,batch2,3epochs;all12 intermediate objectives
are visible. Select3,val.312052,probe1.329592->.282178. RGB-only independent
restore/repeat parity0,single/batch8.94069672e-8 within printed2e-6,184 leaves
immutable,sensitivity.003542. All actual6fit and4test photographs are shown.

Held-out learned heat/PAF MSE.044782/.015214 and ZERO retained people;
empty.024235/.015298;SGD.026796/.016192 with4/152/116 correct/pred/GT,
36 groups vs14 people,precision.026316/recall.034483/F1.029851. Explicit poor
outcome and no useful pose/COCO AP/realtime claims. Custom grouped.2box
matching excludes derived neck;one validation photograph and four test images
limit conclusions;rectangular masks are not crowd segmentation.

Evidence:build/cnn-openpose-{coverage,hover}.json,
build/cnn-openpose-validation/,build/cnn-openpose-review/,
build/cnn-datasets/human_pose/manifest.json. Markdown/current website resolve
nine media. Combined2639/2639 across117 CNN headers;102/129 complete,27 remain.


### Tensor stacked-hourglass real-person pose completed (101/129 overall)

Six executed C++ blocks,editable/executed TikZ and nine reviewed figures show
half-pixel/truncated Gaussian targets,NaN-safe missing-channel MSE and every
adjoint,complete recursive graph/feedback,inclusive PCK,actual original person
crops,visible SGD coordinate prior and all18 Adam updates. Canonical
hourglass_models.hpp/pose_data.hpp/pose_visualization.hpp pass57/57 documented
definitions,25 compiled/run public examples,seven asserted clangd hovers,
seven native checks,standalone inclusion and original complete three-epoch
workflow. Legacy reusable APIs forward canonical tensor implementations.

Reproducible scripts/prepare_cnn_pose.py/nn_pose_prepare.cpp verify official
COCO8-Pose341756-byte archive,SHA256
468e06839e06eaf317594386967d238af4bdfde670e33d9a2a287412b6a37de5.
Original train source IDs36/49/77 fit,86 validates;original validation IDs
110/113/136/151 are held out. Six fitting people plus anatomical mirrors
produce12 crops,validation1,test14 people/four photographs. Oracle boxes
expand1.25 independently in each axis,floor/ceil clip to original pixels,
legacy bilinear resize128,RGB/255 and bounded joint/box transforms use tensors.
Manifest retains exact original files,person IDs,crop bounds and mirrors.
Full-image6/1/4 source observations/COCO17 labels are also prepared for
OpenPose,with padded8-person dimensions;no detector boxes feed its images.
Every person-crop RGB pixel matches independent source JPEG/legacy-bilinear
arithmetic. Every labeled coordinate,visibility,box scale and anatomical
left/right mirrored coordinate passes independent literal host formulas.
No source photograph crosses splits;multiple people/mirrors remain dependent.

Full256px width256/stem64 executes complete backward:6571170 learned values,
396 leaves,31488 BN values/192 leaves. Compact128px width32/stem8 has
110450 learned/3936 BN values with identical396/192 leaf structure. Both
stacks/depth4,all14 recursive/stem/head/feedback shapes,every learned/RGB
and first-head feedback gradient pass. Preactivation1x1/3x3/1x1 bottlenecks,
identity/projected skips and no terminal ReLU follow original Torch model.
Explicit symmetric stride phase,VALID pooling,nearest decoder and feature+
prediction feedback remain intact. BN mutation is explicit in separate state.

Tensor target generation preserves source Float32 annotations/sigma constants,
Double half-pixel centers,3sigma square truncation,visibility1/2,missing/invalid
NaN exclusion. Independent every-value odd7/.7-sigma Gaussian and generalized
29px first-maximum decoding fixtures pass. Equal-supported-image loss selects
missing NaNs out BEFORE subtraction;all-empty and Bool-mask derivatives pass.
Tensor metric uses Float32 coordinate ingress/Double distance,joint weighting,
inclusive.2 maximum-original-box normalization and excludes missing NaNs.
Independent3/4/5 and unequal-joint-count fixtures verify PCK/NME/pixel means.
Tensor preview uses rounded3x3 points and continuous distance<=1 skeleton
lines,an explicitly documented display change from legacy sampled lines.
Every RGB/GT/predicted skeleton/heatmap preview pixel matches an independent
literal geometric raster/half-byte oracle;numeric targets/scores are unchanged.

One visible Double SGD(.5) step fits34 coordinate leaves with declared63.5
unsupported fallback;closed-form tensor mean is a parity oracle only. All396
learned/192 BN leaves warm before18 visible Adam(.001) updates,batch2,three
epochs with both masked Gaussian heads supervised. Select epoch3,val1.724545;
fixed training-mode probe2.219143->1.244535. Test116 labeled joints across14
crops/four images:PCK learned.103448,SGD.172414,center.068966;mean crop-pixel
error66.137554/41.032688/46.302833,NME.589964/.365837/.411891. Explicitly
worse than SGD and inconsistent center ranking by diagnostic,no accuracy claim.
All12 actual fitting RGB/GT crops and all14 held-out RGB/GT/prediction/heatmaps
shown. Separate RGB-only restored inference has exact checkpoint/repeat parity,
single/batch max3.57627869e-6 within2e-5 Float32 tolerance,all396+192 leaves
immutable,different-crop heatmap change.757771. No targets,box values,optimizer
or source IDs enter prediction. Oracle boxes/single validation person/tiny
photograph dependence and no COCO AP/MPII benchmark claims are explicit.

Evidence:build/cnn-hourglass-{coverage,hover}.json,
build/cnn-hourglass-validation/,build/cnn-hourglass-review/,
build/cnn-datasets/human_pose/manifest.json. Markdown/current website resolve
nine media. Combined2562/2562 across113 CNN headers;101/129 complete,28 remain.


### Tensor full-objective partial inpainting completed (100/129 overall)

Six executed C++ blocks,editable/executed TikZ and nine reviewed figures
show partial-convolution support renormalization,signed values and adjoints,
mask propagation,dilated-hole L1 TV,full/compact U-Nets,original CIFAR images
with reproducible irregular masks,all48 visible Adam updates and every held-out
reconstruction. Canonical partial_unet_models.hpp/partial_inpainting_losses.hpp/
partial_inpainting_data.hpp pass38/38 documented definitions,22 compiled/run
public examples,six asserted clangd hovers,six native checks,standalone
inclusion and original complete three-epoch workflow. Legacy headers forward
reusable APIs;experiment compatibility remains while report optimization is
fully visible. Bool composite masks correctly use a Double.5 threshold.

Full8enc/8dec width64..512 executes256px backward,32865236 learned/10880 BN
values,60 learned/28 BN tensors,16 feature/mask stages. Compact5enc/5dec
8/16/32/64/64 has183684/592 values,36 learned/16 BN tensors,ten stages.
Exact odd33x35 geometries,binary propagated masks,all finite gradients,
every nonzero kernel derivative and empty/all-known identities pass.
Masks detach at each partial convolution;decoder nearest-upsamples AND
concatenates both feature/mask skips,including original RGB/mask at final
skip. BN except first encoder/final decoder;raw RGB is linear/unbounded.

Irregular masks are now entirely tensor geometry:MT19937(seed+65537*row)
draws rectangle/two-stroke metadata;Double tensor coordinates evaluate all
65-position swept disks,rectangle unions and123..563/1024 hole rejection.
No host pixel loops. Independent saved original literal geometry compares
EVERY mask,input,truth and observed-mean pixel at five seeds including0 and
UINT32_MAX,nontrivial row reordering and duplicates. Native source tests prove
hidden-target and row-order independence. Tensor visible-only means use
Double sums and select hidden values BEFORE aggregation,including hidden NaN
regression. Every five-panel RGB channel/half-byte matches original preview
arithmetic,including source Float32 error division then Double accumulation.

Full objective valid+6hole+.05perceptual+120(style_raw+style_comp)+.1TV uses
full-image pixel normalization. Genuine frozen Keras VGG16 pool1/2/3 features
are divided by255 explicitly;no Torch parity claim. Independent host formulas
check every pixel/feature derivative,dilated-hole neighbor endpoints and all
seven components. Genuine independent Keras feature relative error4.41207e-7,
all-component max2.38419e-7,every batched pixel gradient relative1.50482e-7.
Actual official assets/Keras checks run rather than skip.

Original CIFAR32fit train0..31,8val32..39,16heldout official test192..207;
mask seeds107/109/113. Warm36 learned+16 BN leaves before Adam(.0002),
batch2/three epochs/48 visible updates. All seven objective components and
six weighted contributions are shown. Select epoch3,validation357.555614;
fixed training-mode probe544.770264->176.898010. Paired parameter/BN files
save the same selected epoch;separate image+mask-only prediction contains
no VGG,truth or optimizer. Exact checkpoint/repeat parity,all36+16 values
immutable,single/batch max2.07126141e-6 within printed1e-5 Float32 tolerance,
finite hidden-fill difference0,known composite max error0. All26 frozen VGG
tensors and cached truth features stay unchanged.

Raw hole MSE1.238816,mean hole PSNR.880180dB,pooled PSNR-.930067dB,
full PSNR7.354958. Metric-only clipped hole PSNR7.461787 remains below
observed-mean14.555607dB. Explicitly poor reconstruction and bright hole
artifacts,no quality/benchmark claim. Negative pooled PSNR remains visible;
all16 actual observed/reference/prediction/truth/error tiles are shown.

Evidence:build/cnn-partial-{coverage,hover}.json,build/cnn-partial-validation/,
build/cnn-partial-review/. Markdown/current website resolve nine media.
Combined2505/2505 across110 CNN headers;100/129 complete,29 remain.


### Tensor Johnson generator-only fast style completed (99/129 overall)

Six executed C++ blocks,editable/executed TikZ and nine reviewed figures
show exact reflection/cropped residual adjoints,learned transpose phase,
quadratic TV and derivatives,full/compact transformation and genuine VGG16,
actual original CIFAR/painting data,all36 visible Adam updates and every
held-out prediction. Canonical johnson_models.hpp/johnson_losses.hpp pass
32/32 documented definitions,23 compiled/run public examples,six asserted
clangd hovers,eight native checks,standalone inclusion and original complete
three-epoch workflow. Frozen source-map lifetime is owned. Tensor preparation,
feature objectives,metrics,checkpoint comparisons and preview quantization
replace host numeric loops. Legacy experiment classes keep compatibility;
report training directly owns every optimizer step.

Full base32 executes256x256 backward:1679235 learned/3200 BN values,
62 learned/30 moving-statistic tensors. Compact base8 retains all11 stages,
108771/800 values,exact64/80 geometry and every finite learned derivative.
Five VALID cropped residual blocks have no post-add ReLU. Torch down uses
symmetric pad1 then VALID stride2;transpose uses VALID2H+1 and crops1..2H,
with true9*Cin fan-in initialization. Independent literal host checks verify
all reflection/crop/stride values and complete input/kernel/bias Jacobians.
Quadratic joint-interior TV and channel/spatial-normalized Gram/Frobenius
content/style derivatives pass independent formulas. BN mutation is explicit.

Official pinned VGG16 assets/reference preparation reuse the genuine shared
style cache from Gatys:26 Float32 arrays/14714688 values;max pools,selected
relu1_2,2_2,3_3,4_3 and content relu2_2. Independent Keras feature relative
error4.12842e-7,all objective error5.00838e-7,every batched input gradient
6.36724e-6. Actual asset checks run rather than skip. Native checks cover
source-map lifetime,frozen mutation and invalid requested blocks. Every64x64
prepared pixel matches independent original-source legacy bilinear arithmetic;
every content/painting/prediction half-byte display coordinate passes.
Paired checkpoint restoration validates all counts,unique names,prefixes,
exact leaf names,shapes and dtypes before copying;native malformed checkpoint
checks prove no partial learned/BN mutation on failure.

Original CIFAR24fit train0..23,8val24..31,8heldout original test8..15;
test0..7 development-only. Tensor bytes/255 and legacy32->64 resize;fixed
actual Kandinsky painting. Warm62+30 leaves before Adam(.001),three epochs,
batch2/36 visible updates. Fixed initial content/style/TV denominators
515694.093750/126185922560/.127916. Training updates BN once per optimizer
step;validation uses immutable moving statistics. Select epoch3,val1.763701;
fixed train-mode fit3->1.734083. Held-out objective12.357131->1.783372,
relative content/style/TV.679889/.912451/.191033. Actual bright textured outputs
have weak content preservation,explicitly acknowledged;objective reduction is
not a semantic or subjective quality score. All8 actual contents/painting/
predictions shown. Distinct-content mean RGB difference.139763 is sensitivity
only. Generator-only RGB deployment has exact checkpoint/repeat parity,
single/batch max1.79e-7,all62+30 values immutable. All26 VGG tensors and
cached content/style targets stay unchanged. Three-call64px CPU mean5.959ms
is local timing only,no real-time/paper-quality claim.

Evidence:build/cnn-johnson-{coverage,hover}.json,build/cnn-johnson-validation/,
build/cnn-johnson-review/. Markdown/current website resolve nine media.
Combined2467/2467 across107 CNN headers;99/129 complete,30 remain.


### Tensor Gatys frozen-VGG19 pixel optimization completed (98/129 overall)

Six executed C++ blocks,editable/executed TikZ and ten reviewed figures show
fixed BGR preprocessing and signed adjoints,per-image Gram invariance and
nonsymmetric upstream derivatives,exact content/style coefficients,original
Labrador/Kandinsky source images,full genuine VGG features and all30 visible
projected Adam pixel updates. Canonical perceptual_models.hpp/style_losses.hpp
pass24/24 documented definitions,14 compiled/run public examples,six asserted
clangd hovers,five native checks,standalone inclusion and original complete
30-step workflow. Input map lifetime is owned by the frozen extractor rather
than retaining a dangling reference. Detached cloned constants are outside
optimizer stores;numeric validation,comparison,preprocessing and PNG byte
quantization use tensors. Legacy headers forward canonical APIs.

Reproducible scripts/prepare_cnn_style.py/nn_style_prepare.cpp validate exact
original bytes/SHA256 for official Keras no-top VGG19(80134624 bytes,
bde7da0dfd621cbf86e11ce04e4946a9a42634c2cfd21318bba090d860699e35),
VGG16(58889256,bfe5187d0a272bed55ba430631598124cff8e880b98d38c9e56c8d66032abdc1),
and original TensorFlow tutorial Labrador83281/Kandinsky195196 JPEGs.
HDF5 decodes unmodified Float32 weight ingress only. Canonical report tensor
JPEG center-square/legacy bilinear64/RGB255 processing preserves source policy.
Photo attribution/CC BY-SA3.0 and painting provenance are retained.
The isolated pinned TensorFlow CPU2.20.0/Keras3.15.1 environment is only for
independent test oracles via scripts/cnn_style_reference.py;no report fitting
or numeric data preprocessing uses it. All three VGG19/Johnson/partial feature,
loss and complete input-gradient fixtures reproduce. Future two reports retain
these genuine source assets rather than random or synthetic substitutes.

VGG19 imports32 tensors/20024384 values;12944960 execute through conv5_1.
Average pools are explicit;content conv4_2/style conv1_1..5_1. All six selected
feature arrays,Keras content/style/combined scalars and every32x48 RGB image
derivative pass independently. Feature relative error4.68149e-7,content
2.91661e-7,style/combined0,image-gradient2.25652e-6. Native literal host formulas
check all batched Gram/content/style derivatives,nonsymmetric upstream,batch
isolation and permutation invariance;temporary source-map lifetime and frozen
mutation detection pass. Every half-byte PNG coordinate matches std::lround.
Actual official/Keras asset checks execute rather than skip.

Only12288 RGB pixel scalars are optimizer leaves. Tensor initialization.75
content+.25 seeded MT17 noise;fixed raw denominators263952832/973494272.
Thirty explicit Adam(.03) steps with in-place[0,1] projection;best/final step30,
objective2->.824895,relative content/style.116245/.708650,mean absolute pixel
change.085392,residual gradient norm.051868. All32 frozen weight tensors and
six cached targets stay unchanged;image-only checkpoint restores the chosen
objective. Actual source,initial/final images,selected feature RMS maps,
objectives and pixel-change diagnostics are shown. Low64px/30-step subjective
style quality and no held-out or exact-paper convergence claims are explicit.

Evidence:build/cnn-gatys-{coverage,hover}.json,build/cnn-gatys-validation/,
build/cnn-gatys-review/,build/cnn-datasets/style_transfer/*manifest.json.
Markdown/current website resolve ten media. Combined2435/2435 across105 CNN
headers;98/129 complete,31 remain.


### Five tensor super-resolution architectures completed (97/129 overall)

Seven executed C++ blocks,editable/executed TikZ and14 reviewed figures
(8 SVG/6 PNG) show signed antialiased bicubic,exact CRD permutation,
RCAN channel gates and derivatives,VDSR adjustable clipping,five genuine
full/compact graphs,original CIFAR degradation pairs and240 visible optimizer
updates. Canonical super_resolution_models.hpp/super_resolution_data.hpp
pass77/77 documented definitions,42 substantive compiled/run public examples,
seven asserted clangd hovers,eight native checks,standalone inclusion and
all five original three-epoch workflows. Legacy interfaces forward core APIs;
legacy experiment compatibility retains its optimizer-owned class while the
report shows the full loop explicitly. Numeric preprocessing/metrics/displays
and clipping diagnostics are tensor-based.

Full scalar/leaf counts:SRCNN8129/6,ESPCN21284/6,VDSR665921/40,
EDSR40729603/136,RCAN15444643/1628. Compact8129/6,21284/6,42065/40,
49603/40,45851/60. Full native backward passes and exact rectangular compact
stage inventories pass;only inactive RCAN attention ReLU paths can have0
parameter derivatives. Raw observed-input derivatives are finite/nonzero.
Independent original host oracles verify signed cubic phase/DC/antialias,
CRD indices/every adjoint,attention input/weight derivatives and residual skips.
New oracles compare every nonconstant rectangular/singleton bicubic pixel
and analytic input adjoint;nonconstant Gaussian SSIM agrees with literal
11x11 source moments;every five-model preview channel/half-byte is checked.

Tensor dense interpolation weights use Keys a=-.5,normalized widened support,
half-pixel centers,symmetric duplicated edges. Float64 column sums cast to
Float32 before Float64 row sums/final Float32. Raw overshoot is preserved.
Original CIFAR rows128fit/32val/32test,HR32/LR16 x2;source rows0..127,
128..159 and original test64..95. Training Y is studio-range BT.601 Float32;
metrics use source-equivalent Double Y after raw Float32 ingress. All models
share central20x20 coordinates6..25,metric-only clipping,mean image PSNR,
valid11x11/sigma1.5 SSIM and pooled MSE/PSNR. No hidden HR resize/reference fit.

All leaves warm before each optimizer. Four Adam(.001) fits and VDSR
momentum(.9) SGD(.1) with explicit theta.01/rate derivative clip each execute
48 updates. Y models raw MSE,RGB models raw L1. Every selection is epoch3.
Test PSNR/SSIM:SRCNN22.148973/.590156,ESPCN21.950641/.643553,
VDSR26.724774/.855048,EDSR26.378989/.845764,RCAN25.723505/.818180.
Unchanged bicubic26.711192/.854135. Only VDSR has a tiny.013582dB gain;
no.05dB benchmark claim. VDSR max raw derivative.001174,zero clipped
coordinates;the hand fixture independently activates clipping. Every model
has falling fixed/mean fitting loss. Separate restored input-only stores
match checkpoints exactly,repeat exactly,retain every leaf unchanged,and
single/batch differences<=1.79e-7. All32 actual held-out comparisons are
shown for EACH of five models,with all8 original input/degradation previews.

Evidence:build/cnn-sr-{coverage,hover}.json,build/cnn-sr-validation/,
build/cnn-sr-review/. Markdown/current website resolve all14 media.
Combined2411/2411 across103 CNN headers;97/129 complete,32 remain.


### Tensor RGB-only monocular depth completed (96/129 overall)

Six executed C++ blocks,editable/executed TikZ and ten reviewed figures show
positive heads and derivatives,scale-sensitive/invariant masked log loss,
valid-neighbor structure,full coarse/fine RGB graph,genuine NYU sensor data,
visible independent SGD prior fits and complete Adam training.
Canonical depth_models.hpp/depth_data.hpp pass64/64 documented definitions,
25 compiled/run public examples,six asserted clangd hovers,nine native checks,
standalone inclusion and original complete three-epoch workflow.

Reproducible scripts/prepare_cnn_depth.py and nn_depth_prepare.cpp read the
pinned official2,972,037,809-byte NYU labeled Matlab/HDF5 archive,SHA256
2d724b0c0ab358aa1ce5df855e5bd14a2279ab7202efe22f32a30d612dcb86aa.
Inventory464 canonical rooms,sort/shuffle seed137 and select32/8/16 distinct
rooms and original first frames. All56 room IDs AND source rows are unique.
Only images and aligned rawDepths are read;no inpainted target. Tensor gathers
both modalities at rows/columns5,15,... to48x64;no interpolation or hole filling.
Finite sensor depth>.1f,<=10m remains metric;invalid holes become mask0/depth0.
Manifest retains exact names,source rows,processing and source fingerprint.
Native checks independently verify every sampled full-frame RGB/depth/mask
pixel,NaN/Inf/range boundaries,all prior positions and preview quantization.
Actual56-room cache checks pass rather than skip.

231745 scalars/16 leaves,eleven stages and all learned/RGB gradients pass.
Safe selection precedes logs/metrics;invalid NaNs have zero derivative.
Losses weight supported images equally;neighbor counts require both endpoints.
Metrics preserve raw predictions without GT median scale or numeric clipping;
strict deltas and centered SILog variance are checked. Independent native
formulas verify per-room metrics and spatial-prior unsupported fallback.

Two separate visible Double SGD(.5) updates fit pooled and3072-coordinate
log-depth priors,then verify closed-form tensor oracle parity. All16 leaves
warm before24 visible Adam(.001) updates. Select epoch3,val.211358;
fixed fit1.173153->.254808. Test16 rooms/31808 true sensor samples:
refined RMSE1.388465m/AbsRel.450525;SGD geometric1.304952/.408530;
SGD spatial1.117481/.341685. Explicitly worse than both prior references.
All16 actual RGB/sensor/prediction/error/support tiles are shown.
Independent RGB-only restore has exact checkpoint/repeat parity and all16
leaves immutable;single/batch max difference1.43051147e-6m is printed and
checked within1e-5m Float32 tolerance. Cross-room difference.702084m.

Evidence: build/cnn-depth-{coverage,hover}.json,build/cnn-depth-validation/,
build/cnn-depth-review/. Markdown/current website resolve all ten media.
Combined2334/2334 across101 CNN headers;96/129 complete,33 remain.


### Tensor PWC-Net shared-pyramid flow completed (95/129 overall)

Six executed C++ blocks,editable/executed TikZ and ten reviewed figures show
hand normalized correlation/query gradients,per-level warp unit conversion,
source HSV legend,differentiable dilated context,full/compact shared-pyramid
graphs,actual published pairs,visible SGD constant and complete multiscale Adam.
Canonical pwcnet_models.hpp has3/3 definitions,two compiled/run public examples,
four clangd hovers,seven native checks,standalone inclusion and the original
complete three-epoch workflow passing. Legacy includes preserve the source
portable-dilation test entry point;canonical code calls tensor kernel expansion.

Full9374274/126 and compact410442/126 graphs execute,67 exact stages,
18 feature convolutions/36 leaves shared across both inputs. Native radius4
cost and both feature Jacobians,warp/coverage/image-coordinate adjoints,
transpose derivatives,unit/target/safe-loss/strict-metric boundaries,context
rates1/2/4/8/16 every signed value/input/tap derivative pass. All compact126
learned and both-image gradients are finite/nonzero. Shared deep feature
weight receives exact sum of both branch derivatives;identical inputs produce
identical shared features. Repeated forwards retain126 leaves. Explicit fan-in
initializer metadata validates positivity and preserves16*Cin transpose
scaling,not stored Cout. No BN or joint RGB centering;raw[0,1] features enter.

Reuses reproducible pinned Middlebury32/8/24 custom crops and declared source-
family isolation from FlowNetS:four fitting,one validation,three test pairs.
The report independently fits its constant with one visible SGD(.5) update,
mean parity0,dx/dy-.169839/-.734982px. All126 leaves warm before48 visible
Adam(.001) updates on five smooth endpoint heads,source weights and fixed
input-pixel/20 units. Select epoch1 validation .021812;fixed probe
.050475->.043961. Test EPE learned3.946422,initial4.370310,zero3.916913,
constant4.174930px. Bad3 learned.700714,zero.701772,constant.650614;no broad
quality claim. Urban2/Urban3/Venus2.412559/4.198488/5.228220px. All24 actual
pair/GT/prediction/error/warp tiles shown,with dataset/crop dependence explicit.
Independent pair-only restore matches exactly,all126 leaves immutable,
single/batch agree,second-image sensitivity.005057px. No targets,masks,
optimizer or pretrained weights are inference inputs.

Evidence: build/cnn-pwcnet-{coverage,hover}.json,build/cnn-pwcnet-validation/,
build/cnn-pwcnet-review/. Markdown/current website resolve ten media.
Combined2270/2270 across99 CNN headers;95/129 complete,34 remain.


### Tensor FlowNetS and published optical flow completed (94/129 overall)

Six executed C++ blocks,editable/executed TikZ and ten reviewed figures show
backward-warp coordinates/coverage,source HSV direction legend,explicit field
units,smooth endpoint loss/zoomed exact gradients,full/compact six-scale graphs,
published Middlebury pairs/GT,visible SGD constant and complete multiscale Adam.
Review shortened/rerendered concept labels to remove multiscale/pixels word
hyphenation and zoomed loss gradients near zero. Five new canonical headers
flow_components/losses/data/visualization and flownet_models have73/73 documented
definitions,36 compiled/run public examples,eight hovers,ten native checks,
standalone includes and original full three-epoch workflow passing.

New prepare_cnn_optical_flow.py/nn_optical_flow_prepare.cpp verify official
frame-pair ZIP SHA256dc7afbab15a7da40b52d633f825191bf693e227320bb624546ed9f2f8d668e50
and public GT SHA256e967b729cc47a493b3226515e0172224e503f7b14a54f97660f8d5f745632ab9.
PIEH little-endian raw ingress,unknown-vector masks,RGB normalization,crop
coordinates,all five GT targets,metrics,HSV colors and previews use tensors.
Eight64px crops per scene come from fixed5x5 geometry,row-major first8 unique
origins with>409 valid GT samples,no image/flow resize. Manifest stores every
scene/family/dimension/origin. Fit Dimetrodon/Grove2/Grove3/Hydrangea32 crops,
validation RubberWhale8,test Urban2/Urban3/Venus24. Related Grove/Urban scenes
stay together. Crops overlap;test represents three source pairs,not24 independent
sequences. This is fresh custom preparation,not the old external archive or
benchmark evaluation. Published data include real and rendered imagery.

Independent oracles cover correlation every signed value/both derivatives,
warp/coverage/all image/coordinate derivatives,integer slope adaptation,all
transpose values/input/kernel adjoints,anisotropic grid versus fixed input
units,nonconstant masked pyramid every block,NaN-invalid/empty/zero endpoint
loss and every gradient,strict3px/5% metrics,equal-image rather than pooled
weighting,raw .flo/image ingress,crop origins and every HSV/preview pixel.
Legacy ScoreFlow preserves its old numeric->Float32 presentation conversion.
Full38676506/42 and compact615658/42 learned graphs run,27 exact stages;
compact all42 leaves plus both image gradients finite/nonzero,no BN.

One visible SGD(.5) step fits constant dx/dy-.169839/-.734982px,independent
mean parity0. Forty-eight visible Adam(.001) updates fit all five losses with
fixed input-pixel/20 units. Validation selects epoch1/.023476;fixed probe
.050434->.039880. Test EPE learned4.246750,initial4.205583,zero3.916913,
SGD constant4.174930px. Urban2/Urban3/Venus2.541693/4.478318/5.720239px.
No held-out gain claim;every actual test pair/target/error/warp shown. Independent
pair-only deployment matches exactly,all42 tensors immutable,single/batch agree,
second-image sensitivity.140411px. No target/mask/optimizer inference input.

Evidence: build/cnn-flownet-{coverage,hover}.json,build/cnn-flownet-validation/,
build/cnn-flownet-review/. Markdown/current website resolve ten media.
Combined2267/2267 across98 CNN headers;94/129 complete,35 remain.


### Tensor CRNN/CTC word recognition completed (93/129 overall)

Five executed C++ blocks,editable/executed TikZ and seven reviewed figures show
repeated-character unique paths,every CTC adjoint,three-path marginal mass,
full/compact seven-convolution/two-projected-BLSTM graphs,actual human word
crops,visible Adam and every held-out word prediction. Review shortened and
rerendered concept labels to remove character-word hyphenation. Canonical
crnn_models/ocr_data headers have42/42 documented definitions,16 compiled/run
public examples,six clangd hovers,seven native tests,standalone includes and
original complete three-epoch workflow passing.

New prepare_cnn_ocr.py/nn_ocr_prepare.cpp pin official COCO-Text V2 annotations
SHA2561444893ce7dbcd8419b2ec9be6beb0dba9cf8a43bf36cab4293d5ba6cecb7fb1 and
reuse pinned original COCO128 photos. Human legible lowercase ASCII1..12 filter
finds60 words/16 photos;ascending image/annotation-ID groups produce43/7/6
words on11/2/3 disjoint source photos,dropping excess words rather than leaking
photos. Manifest retains human text,original source splits,IDs,boxes,dimensions
and discarded annotations. This is a fresh custom subset,not the old external
archive or official evaluation. Tensor outward floor/ceil/clipping,Float64
encoded grayscale and Float32 half-pixel32x100 stretch perform all pixel work.
Original dimensions are verified,no character synthesis/rotation rectification.
Known human boxes make it oracle-crop recognition,not text detection.

CTC exhaustive path/posterior adjoints,variable-time mean reductions,padded zero
gradients and all18 finite differences pass. Tensor Int64 edit DP matches49
independent string pairs,including empties/repeats;numeric greedy argmax is
tensor based,class collapse/string serialization metadata only. Native tests
independently check every crop/resize pixel,text-independent observations,
source-image disjointness/all56 annotations and perfect metrics. Full8327205/36
plus2560/6 BN and compact133285/36 plus320/6 BN execute with all learned/image
gradients. Exact7 convolution shapes/T26/asymmetric symmetric-side pooling
pass. Recurrent zero states now match input dtype;a Float64 projected-BLSTM
value/gradient case passes.

Eighteen visible Adam(.001) updates,43 images,batch8,three epochs;warm all36
leaves first. Paired checkpoint selects epoch3 validation CTC33.693634;full
fixed fitting CTC78.655991->36.410206. All six test words decode empty,0/6
exact,CER1(27/27),equal to empty reference;initial CER1.148148(31/27).
No recognition-quality claim. Every real test crop/human word/prediction and
blank time course is shown. Independent image-only restore has exact parity,
all36 learned+6 moving tensors immutable,single/batch agree,image inversion
sensitivity.015552. No test tuning or hidden optimizer helper.

Evidence: build/cnn-crnn-{coverage,hover}.json,build/cnn-crnn-validation/,
build/cnn-crnn-review/. Markdown/current website resolve seven media.
Combined2194/2194 across93 CNN headers;93/129 complete,36 remain.


### Tensor DnCNN residual denoising completed (92/129 overall)

Six executed C++ blocks,editable/executed TikZ,eight reviewed figures cover
signed residual half-MSE/Jacobians,hand reflected filtering,actual source-seeded
Gaussian residual distribution,full/compact17-layer graphs,natural CIFAR
corruption,visible Adam and every held-out reconstruction. Canonical dncnn_models
and denoising_data headers have35/35 documented definitions,19 compiled/run
public examples,six clangd hovers,seven native checks,standalone includes and
original complete three-epoch workflow passing. Legacy data/model/preview APIs
forward to tensor implementations.

Host RNG emits source-order53-bit integers only;uniform scaling,Box-Muller,
trigonometry,grayscale/noise/residual preparation,reflected filtering,raw/clipped
metrics and every preview pixel are tensor based. Independent exact source
oracle checks every odd-length noisy/residual sample and selected grayscale
byte. Native131072-sample standardized moments.001154/.992207/2.943705 pass;
.501335 of black/white corrupted intensities lie outside[0,1],as intended.
Independent reflected values/every derivative include singleton axes;residual
and BN every-coordinate derivatives,unbiased moving variance,all compact
orthogonal kernels,17 exact odd-rectangular stages and unclipped identity pass.
Full556096/48+1920/30 BN and compact35344/48+480/30 BN execute,all learned
and observed-input derivatives finite/nonzero.

CIFAR fit0..255/validation256..287,official-test128..159,grayscale32px,
sigma25/255. Ninety-six visible Adam(.001) updates with fresh5300+epoch
corruption;fixed validation/test43/47. Paired learned/BN epoch3 validation
half-MSE .005227. Train-mode fixed probe .287240->.005109,while moving-BN
probe .004734->.005051 worsens. Test raw PSNR learned19.826817,noisy20.219740,
fixed reflected mean24.034324dB;clipping preserves the ordering. No quality-gain
claim;short optimization and state handling are distinguished from denoising
success. Independent noisy-only deployment matches exactly,all48 learned+30
moving tensors immutable,single/batch agree. All32 held-out errors shown.

Evidence: build/cnn-dncnn-{coverage,hover}.json,build/cnn-dncnn-validation/,
build/cnn-dncnn-review/. Markdown/current website resolve eight media.
Combined2152/2152 across91 CNN headers;92/129 complete,37 remain.


### Tensor lightness-only image colorization completed (91/129 overall)

Six executed/readable C++ blocks,editable/executed TikZ and nine reviewed figures
show six physical RGB/Lab colors,piecewise transfer curves,opposite-color MSE
ambiguity,complete odd-rectangular colorizer stages,actual CIFAR images,visible
SGD constant-chroma fitting and complete-model Adam. New colorization_data/models
headers have48/48 definitions,27 compiled/run public examples,six clangd hovers,
eight native checks,standalone includes and original three-epoch workflow.

Every pixel conversion now uses Float64 tensors before final Float32 cast:
sRGB gamma,scikit-image D65/2degree matrices,Lab knees and inverse gamut values.
Compatibility scalar triples delegate to tensors. Independent known colors,
2197 round trips,both transfer knees,transform finite-difference Jacobians,
every selected raw pixel,paired data,metric normalizations,all half-pixel
interpolation values/Jacobians and every preview pixel/half-byte rounding pass.
Complete140082-scalar/20-leaf graph has ten exact odd-rectangular stages and
all learned/lightness gradients. Zero model has no hidden color shortcut.

CIFAR fit0..255/validation256..287,official-test96..127;original32px,no labels.
One visible SGD(.5) step on sum-of-coordinate MSE fits normalized constant ab,
independent mean parity0. Physical a/b .111270/6.156950. Complete CNN fits96
visible Adam(.001) updates,selects epoch3 validation MSE .010270;fixed fitting
.033218->.019443. Test learned/gray/SGD-constant ab MSE
.011777/.013745/.012040. Gray has higher RGB PSNR26.589655 versus24.258774dB
and lower mean DeltaE76 14.588410 versus14.956496;learned raw gamut fraction
.064972. Ambiguity,desaturation and metric disagreement are explicit,no semantic
color recovery or benchmark claim. Independent lightness-only deployment
matches exactly,all20 leaves immutable,single/batch agree,image sensitivity
.016750. All32 actual held-out colors are shown.

Evidence: build/cnn-colorization-{coverage,hover}.json,
build/cnn-colorization-validation/,build/cnn-colorization-review/. Markdown and
current website resolve nine media. Combined2117/2117 across89 CNN headers;
91/129 complete,38 remain.


### Tensor context-encoder inpainting completed (90/129 overall)

Six executed C++ blocks,editable/executed TikZ,eight visually reviewed figures
cover stable real/fake BCE/gradients,hand3x3 partial convolution,missing-region
PSNR,full/compact generator/adversary graphs,actual CIFAR masks and a visible
alternating fit. Canonical inpainting_models/data headers have52/52 documented
definitions,28 compiled/run public examples,six clangd hovers,six native checks,
standalone includes and the original full three-epoch workflow passing.

CIFAR fit0..127/validation128..159 from original training,official test160..191;
all normalization,masks,patch extraction,visible-context means,hole metrics and
preview rendering now use tensors. Legacy APIs forward to canonical headers.
Independent source oracles cover partial convolution values/count/mask/every
input-kernel-bias Jacobian,strides1/2/empty windows,center alignment/hidden-target
independence,composite derivatives,stable extreme BCE,per-image PSNR and every
nonzero image/preview pixel. Both adversarial phases freeze the other learned
and moving state. Full G7221231/40,D2766529/16 and compact G22523/24,D2745/8
execute forward/backward;all compact reconstruction/adversarial gradients and
10 stage geometries pass. Full/compact G/D BN values4040/1792 and160/32.

Warm both stores before visible Adam G.002/D.0002,beta1.5. Report fits48 D/G
updates each with .999 patch MSE+.001 non-saturating G adversarial objective,
selects paired G/BN epoch3 validation MSE .189386. Fixed fitting MSE
.248021->.187083. All32 test images:learned/visible-mean hole MSE
.052763/.059099,mean hole PSNR13.375851/13.411761dB. Aggregate MSE versus mean
log-PSNR disagreement and full-image6.0206dB inflation are explicit. Generator-
only independent deployment has no clean target/adversary/optimizer,exact
parity,immutable weights/BN and single/batch agreement. Report uses algebraic
sum of separate real/fake mean BCE,so Float32 optimizer trajectory need not
match the original concatenated2*mean expression bitwise.

Evidence: build/cnn-context-{coverage,hover}.json,build/cnn-context-validation/,
build/cnn-context-review/. Markdown/current website exports resolve all eight
media. Combined2069/2069 definitions across87 CNN headers;90/129 complete,
39 remain.


### Tensor multilabel classification completed (89/129 overall)

Six executed C++ blocks,editable/executed TikZ and nine reviewed figures show
weighted stable BCE/gradients,grouped score ties and absent-class AP,full/compact
ResNet18 graphs,actual COCO128 images/support,visible SGD priors,complete Adam
and every held-out bit decision. Canonical multilabel_data/models/metrics headers
have43/43 documented definitions,15 compiled/run public examples,six clangd
hovers,seven native checks,standalone includes and original three-epoch run.

Pinned COCO128 zip SHA256 61e5e3028863d8ffc3b81d6a514603954889f0edd5e4b44c4ce60b2da99aeb8e
feeds reproducible Python metadata/C++ tensor preparation. The manifest records
128 source IDs,80/24/24 disjoint splits,two photos without labels and two orphan
labels. Empty annotations are category-presence targets,not proof of absence.
All resize/normalization/OR-target/count/weight/metric/bitmap arithmetic is tensor
based. Boolean/empty targets,Float64 threshold boundaries,weighted zero/extreme
logit gradients,score ties,no-positive classes,original image ingress and every
preview pixel including half-byte rounding have independent native oracles.

Full11209872/62 and compact180952/62 learned scalars/tensors,9600/1200 moving BN
scalars execute. Fit-only positive factors accompany30 visible Adam(.001)
updates;paired learned/BN checkpoints select epoch3 validation BCE .495729.
Fixed fitting weighted BCE1.113948->.750087. Held-out38 present/42 absent classes,
66 positive bits/1920. CNN/prior mAP .184866/.072368,calibrated microF1
.061538/.311111;fixed-.5 CNN F1 .074074. Weak test decisions,prior strength and
lack of guaranteed calibration transfer are explicit. Separate deployment
matches exactly,all62 learned+40 moving tensors immutable,single/batch agree.

Evidence: build/cnn-multilabel-{coverage,hover}.json,
build/cnn-multilabel-validation/,build/cnn-multilabel-review/. Markdown/current
website exports resolve all nine media. Combined2017/2017 definitions across85
CNN headers;89/129 reports complete,40 remain. Flower preview half-byte rounding
was also fixed and native/public/original checks rerun;refreshed gallery reviewed.
Retrieval optional-cache test guard rerun,seven tests pass.


### Tensor GeM image retrieval completed (88/129 overall)

Five executed/readable C++ blocks,editable/executed TikZ and seven reviewed
figures show GeM values/power gradients,hand exact0/2/4 ranking/AP5/6,
actual CIFAR fitting images,visible complete hard-triplet Adam,baseline
comparisons and actual held-out nearest neighbors. Review shortened/re-rendered
TikZ phrases to remove image/inspect word hyphenation. Canonical
`nn/retrieval_models.hpp` retains13 conv/four VALID pools,shared learned
p=1+softplus(rho),bias-free projection and unit descriptors. Full14780225/28
and compact232617/28 parameter gradients plus RGB execute;compact all nonzero,
19 exact stages. Independent GeM activation/power and zero/signed-normalization
Jacobians pass. Singleton power tensors retain source from_vector compatibility.

`nn/retrieval_data.hpp` preserves source mt19937 metadata draws and selected
CIFAR pixels exactly:raw planar bytes enter integer tensors before channel
transpose/Float32 /255. Native independent source-style loader compares every
selected pixel/label/ID and seeded class-balanced epoch batch.160 fitting
images/classes0–4;16/32 validation query/gallery/classes5–6;48/96 official
test query/gallery/classes7–9;all352 source IDs disjoint. Frozen Float64
tensor hard mining chooses first-tie farthest positive/nearest negative.
`nn/retrieval_index.hpp` owns cloned descriptors,exact Float64 squared
ranking with Int64 source-ID tie order/self exclusion,noninterpolated AP,
no-positive counts,pixel references and entirely tensor-rendered neighbor
previews. Native oracles cover AP5/6,ties,self/empty/no-positive cases,
owned index snapshots,large signed IDs,pooled pixels and every gallery pixel.

Large-ID tests exposed existing `Tensor::to_vector`/`item` conversion through
double. Both now read Int64 storage directly before destination casts,retaining
full-range/strided/offset integer values. Two new regressions and the complete
35-test tensor value suite pass;both changed contracts have compiled/run public
examples and hovers. Focused existing tensor-header coverage69/69 passes
separately from the54 new retrieval definitions. No signatures/layout changed.

Complete compact model fits24 visible Adam(.0005) updates,three epochs/eight
balanced20-image batches,mean hard-triplet hinge.2,no hidden regularization.
Recompute gradients after detached mining;all28 leaves including rho are warmed
first. Validation mAP selects epoch2/.575607;restored score reproduces,fixed
fitting hinge.231091→.200064,selected power2.99553. Test mAP/hit1 learned
.402486/.291667,untrained.383918/.375000,pooled RGB.419990/.395833,mean RGB
.434927/.479167. Actual error gallery and weaker-than-color results are explicit;
no particular-object/paper benchmark claim. Independent restored deployment
matches exactly,all28 cloned leaves remain immutable,single/batch agree,and
all48x96 serialized gallery IDs/distances reproduce exactly.

Three new headers54/54 definitions,20 compiled/run public examples(including
both shared-tensor extraction examples),eight hovers,seven retrieval native
tests plus two new tensor regressions pass. Standalone headers,original complete
three-epoch workflow,Markdown/current website exports and seven media pass.
Evidence: `build/cnn-retrieval-validation/`, `build/cnn-retrieval-coverage.json`,
`build/cnn-retrieval-tensor-coverage.json`, `build/cnn-retrieval-hover.json`,
`build/cnn-retrieval-review/`. Combined1974/1974 across82 CNN headers;
369 counted native checks,plus the unchanged33 tensor value tests also pass.
41 reports remain.


### Tensor bilinear flower classification completed (87/129 overall)

Six executed/readable C++ blocks,editable/executed TikZ and eight reviewed
figures show hand aligned outer-product/both-stream derivatives,smoothed
signed-root behavior,actual official flower photographs,visible SGD color
prototypes,visible complete two-stream Adam,confusion and all48 held-out
prediction tiles. Canonical `nn/bilinear_models.hpp` retains two independent
13-convolution/four-VALID-pool streams,mean local outer products,smoothed
signed root,unit descriptor and dense head. Full102-way56168166/54-gradient
and compact6-way121806/54-gradient graphs execute,plus RGB gradients;
compact all54+RGB derivatives are nonzero and all37 stages agree. Native
independent oracles check every aligned pooling entry/both Jacobians,joint
permutation invariance versus independent permutation sensitivity,signed-root/
normalization including zero,extreme CE,metrics,prototype distances and every
prediction tile/bar pixel. Flower row gather retains repeated-row gradients.

The formerly external Flowers asset prerequisite now has a reproducible local
workflow: `scripts/prepare_cnn_flowers.py` verifies official Oxford image/
label/split MD5 values,parses only integer MAT metadata,selects original
classes1–6/first8 ascending IDs per official split/class,and extracts only
144 needed photographs. `scripts/nn_flowers_prepare.cpp` calls canonical
`nn/flower_data.hpp` for byte decode ingress,Float32 /255,tensor bilinear64px
square stretch and prepared NPZ validation. No crop/antialias/augmentation;
source/checksums/IDs/protocol are saved in the manifest. This documented fresh
selection may differ from an old external flowers.npz. The full346MB official
archive checksum passed after retrying timed-out byte ranges. Assets remain
in ignored `build/cnn-datasets/flowers102`;no user cache was overwritten.

All NN preprocessing,probability metrics,mean-color reference and galleries
use tensors. The report even fits the18 color prototype coordinates through
one visible SGD(.5) step on within-class squared error;it exactly matches the
stable tensor mean-color reference. Complete CNN fits36 Adam(.001) updates,
three epochs,batch4,shuffle239,48 fit/48 validation/48 official test photos.
Epoch3 validation NLL1.778811 selects/restores;fixed fitting1.792877→1.772950.
Test CNN NLL1.780632/accuracy.333333/Brier.829591;untrained accuracy.166667;
color SGD NLL2.289061/accuracy.375000/Brier.781003. The report explicitly
states that this short fit does not beat mean-color accuracy. Independent
RGB-only deployment reproduces outputs exactly,all54 cloned leaves remain
immutable,and single/batch inference agrees. No pretrained/full102-way
benchmark accuracy is claimed. Legacy APIs forward active tensor calculations.

Both headers38/38 definitions,17 compiled/run public examples,six hovers,
seven native tests,standalone includes,original complete three-epoch workflow,
Markdown/current website export and all eight media pass. Evidence:
`build/cnn-bilinear-validation/`, `build/cnn-bilinear-coverage.json`,
`build/cnn-bilinear-hover.json`, `build/cnn-bilinear-review/`.
Combined1920/1920 across79 headers;360 native tests.42 reports remain.


### Tensor angular margins completed (86/129 overall)

Five executed/readable C++ blocks,editable/executed TikZ and eight reviewed
figures show target transforms/derivatives,analytic two-class boundaries,
actual fitting probe images,visible two-model Adam,held-out confusion and
actual unseen retrieval episodes. Review caught a newest-SVG export ordering
issue;the training block now saves its claimed training SVG last,and its
actual curves plus all exports were rerun/reviewed. No mislabeled media remain.

Canonical `nn/angular_margin_models.hpp` documents ArcFace/CosFace policy,
target-only transformed logits,normalized bias-free heads,separate label-free
inference,stable CE,tensor label gather and exact Int64 confusion reductions.
Native independent analytic/central-difference oracles cover every target,
non-target,feature/column-weight derivative,endpoint clips,zero margin,
monotonic tail,equal-boundary decisions,independent positive rescaling and
invalid inputs. Legacy classification and unit-norm evaluation forward tensor
numerics. Both full220pxRGB/1000-way140153664-scalar/29-gradient graphs
execute;compact108680/29 has every gradient nonzero.

Both complete compact models fit120 visible Adam(.001) steps:three epochs,
batch32,1280 seen-class fitting/320 disjoint validation/640 official test
images,digits0–4. Target-margin CE+.00005*kernel squares;ArcFace m.3 radians,
CosFace m.2 cosine units,scale16. Plain label-free validation CE selects epoch3
for both;restored scores reproduce. ArcFace validation.567354,seen test
CE.592132/accuracy.907813;CosFace.313994,seen.340600/.918750. Separate
unseen validation digits5–6 calibrate thresholds1.033885/.974165;512 unseen
7–9 test pairs accuracy.578125/.636719,AUC.666458/.663422.2563-way unseen
tasks trained.453125/.503906 versus untrained.359375. Separate seen/unseen
performance and actual errors are explicit,no face/pretrained accuracy claims.

Header24/24 definitions,nine compiled/run public examples,six hovers,five
native tests,standalone include,original complete two-model three-epoch
workflow,Markdown/current website export and all eight media pass. Evidence:
`build/cnn-angular-validation/`, `build/cnn-angular-coverage.json`,
`build/cnn-angular-hover.json`, `build/cnn-angular-review/`.
Combined1882/1882 across77 headers;353 native tests.43 reports remain.


### Tensor FaceNet-style triplet learning completed (85/129 overall)

Five executed/readable C++ blocks,editable/executed TikZ and six reviewed
figures show hand triplet derivatives,frozen distance/mining fixtures,actual
P5K8 data,visible complete compact Adam and actual held-out retrieval episodes.
Canonical `nn/facenet_models.hpp` provides NN1 eleven-convolution/two-LRN/
four-pool/two-maxout/128D-unit embeddings;full220pxRGB140025664/28 gradients
actually execute,not just a geometry census. Compact108040/28 retains all13
recorded stages with every gradient nonzero. Independent normalization/maxout
Jacobians include zero/floored norms and equal-maximum gradient splitting.

`nn/metric_learning.hpp` provides graph-connected empty triplet hinges,
strict frozen Float64 tensor semi-hard selection,metadata-only P-by-K/fixed
sampling,aligned tensor squared distances,validation-only threshold calibration
and pairwise half-tie AUC. Every ordered positive pair is considered;closest
strict dAP<dAN<dAP+margin chooses first-index ties,with no fallback. Native
exhaustive sorted oracle checks every choice/equality/missing count. Host loops
only serialize discrete selected row metadata;model/loss/mining/calibration/
verification/retrieval/gallery numerics are tensor operations. Legacy entry
points forward those calculations;normalization checks also use tensors.

Official MNIST fit0–4(256/class),validation5–6(128/class),test7–9(128/class).
96 visible Adam(.001) steps,three epochs32 P5K8 batches,recompute the gradient
graph after detached mining,mean hinge+.00005*kernel squares.14426 total
mined triplets;fixed512 validation triplets select epoch3/restored.089677;
fixed fitting.194942→.018558. Validation256-pair threshold.556712;512 unseen
test pairs accuracy.609375/AUC.678619,TP/FP/FN/TN199/143/57/113;256 unseen
3-way tasks trained.511719/untrained.386719. Actual error images and bounded
digit adaptation are explicit;no real-face/LFW/pretrained claims. Fixed probes
use the same float.2 margin as differentiable loss;legacy probe used double.2.

Both headers35/35 definitions,21 compiled/run public examples,six hovers,
six native tests,standalone includes,original complete three-epoch workflow,
Markdown/current website export and six media pass. Evidence:
`build/cnn-facenet-validation/`, `build/cnn-facenet-coverage.json`,
`build/cnn-facenet-hover.json`, `build/cnn-facenet-review/`.
Combined1858/1858 across76 headers;348 native tests.44 reports remain.


### Tensor Siamese verification completed (84/129 overall)

Four executed/readable C++ blocks,editable/executed TikZ and five reviewed
figures show stable BCE derivatives,actual fitting pairs,visible complete
compact Adam and actual held-out query/support episodes. Canonical
`nn/siamese_models.hpp` shares four VALID convolutions/three pools/sigmoid
embedding plus learned unconstrained weighted-absolute-difference head.
Full105 model38951745/12 gradients and compact48 model24337/12 execute;
native checks compare independent head/BCE derivatives,extreme/zero logits,
swap/identity and summed shared encoder gradient contributions.
`nn/metric_learning_data.hpp` keeps host operations to discrete row/class
sampling metadata;image gather,verification reductions,byte-pixel SSD and
nearest-upsampled colored episode galleries use tensors. Native gallery
checks compare every pixel with an independent source-style painting oracle.
Legacy entry points forward all active data/model/evaluation pixel numerics.

Official MNIST first512/class:fit0–4/validation5–6/test7–9.96 visible
Adam(.001) steps,three epochs1024 resampled balanced pairs,batch32,
BCE+.00005*kernel squares.128 fixed2-way validation tasks select epoch2,
ties by256-pair BCE;both restored scores reproduce. Fixed fitting BCE
.694596→.566996;512 test pairs BCE.702042,accuracy.509766,TP/FP/FN/TN
5/0/251/256;2563-way test tasks trained.500000/untrained.601562/
byte-pixel.550781. Poor transfer and threshold-versus-ranking differences
are explicit;no Omniglot20-way or paper-accuracy claim.

Both headers43/43 definitions,23 compiled/run public examples,six hovers,
six native tests,standalone includes,original complete three-epoch workflow,
Markdown/current website export and all five media pass. Evidence:
`build/cnn-siamese-validation/`, `build/cnn-siamese-coverage.json`,
`build/cnn-siamese-hover.json`, `build/cnn-siamese-review/`.
Combined1823/1823 across74 headers;342 native tests.45 reports remain.


### Tensor volumetric segmentation completed (83/129 overall)

Five executed C++ blocks,one editable/executed TikZ and seven reviewed figures
show sparse CE logit gradients,squared-Dice probability derivatives,actual
orthogonal generated-data slices,visible two-model fitting and held-out slices.
Independent native oracles check every3D convolution/transpose input/filter/
bias gradient,stride/padding,crop Jacobian,PReLU zero convention/residual
shortcut,volumetric BN,and sparse CE/squared-Dice perfect/void derivatives.
Tensor-generated ellipsoids retain deterministic independent splits;native
row/axial checks confirm exactly25% annotations. Fit-only class weights count
annotated voxels through exact Int64 tensor reductions with one pseudocount.
The legacy example now forwards this preprocessing to the tensor helper.

Full3D U-Net convolution census19069955/33 gradients,affine-BN variant
19072291/47 gradients plus4672 moving scalars,and full V-Net65175314/89
gradients execute on small SAME8/16-cube volumes. Original VALID anisotropic
132x132x116->44x44x28 geometry is checked separately. Compact base2 models
retain all stages,74828/1019348 learned scalars and every gradient nonzero.
No full-size VALID execution or medical/anatomical accuracy is claimed.

Each complete compact model fits192 visible Adam(.001) updates,eight epochs,
batch1,24 fit/8 validation/8 test16-cube volumes,seeds11/22/33. U-Net uses
25% axial-label weighted CE,V-Net dense squared Dice;explicit kernel L2 and
paired learned/BN checkpoint restoration precede test. U-Net uses source
current-volume BN at inference without updates,so evaluation forwards one
volume at a time. Tensor RNG changes the original host sampler fixed pixels.
U-Net selected epoch8 validation Dice.327564,test Dice.319163/IoU.192295/
accuracy.793549,TP/FP/FN/TN1608/6762/3/24395. V-Net selected epoch8 validation
Dice.989172,test Dice.991557/IoU.983311/accuracy.999115,1596/14/15/31143.
The report explicitly exposes U-Net oversegmentation and forbids interpreting
the unequal annotation/loss protocols as a controlled architecture ranking.

`nn/volumetric_models.hpp` and `nn/volumetric_data.hpp` have41/41 documented
definitions,19 compiled/run public examples,six hovers and nine native tests.
Standalone includes,original C++23 full fixtures/three-epoch learning workflow,
Markdown/current website exports pass;all seven media resolve. Evidence:
`build/cnn-volume-validation/`, `build/cnn-volume-coverage.json`,
`build/cnn-volume-hover.json`, `build/cnn-volume-review/`. Combined coverage is
1780/1780 across72 headers,336 native tests. All13 segmentation reports are
complete;46 reports remain in other vision,1D,3D/video and interpretability.
Compilation scratch now uses `build/cnn-tmp` on the project filesystem after
storage was restored;no source/figures/validation evidence were discarded.


### Tensor Panoptic FPN completed (82/129 overall)

Six executed C++ blocks,one editable/executed TikZ and seven reviewed figures
show overlap resolution,hand-counted PQ/SQ/RQ,routed aligned feature gradients,
actual textured image/semantic/panoptic truth,visible four-task fitting and
held-out merged maps. Mask overlap equality.5 is accepted/trimmed,score equality
.5 retained;heavy/full overlap rejects and stuff needs32 remaining pixels.
PQ matches same-class noncrowd segments at strict IoU>.5,subtracts truth void
from match union,ignores unmatched predictions mostly in void/same-class crowd,
and aggregates dataset counts before category macro averages. Native known
TP1/FP1/FN1/IoU1 yields PQ.5/SQ1/RQ.5;strict-IoU,void,crowd,empty maps and
invalid metadata cases pass. P2..P5 constant maps give1/2/13/14 and selected
image/level gradient mass9,with all unselected-image gradients zero.

Complete full256px ResNet50/FPN256/GN32 semantic/RPN/box/mask graph executes
46027110 learned/53120 BN moving scalars and all224 parameter gradients.
Compact64px ResNet18/div8/FPN32/GN8 semantic width16,three semantic classes,
RPN32,box hidden128 and mask16/ten classes has489751 learned/1200 moving
scalars and all125 gradients nonzero. GN has no moving state. Pixel texture,
retained object support/intensity,semantic and panoptic targets use tensors;
all-pixel native checks preserve original supports while filling only stuff.
Legacy scene/metric/model helpers forward numerical work to canonical APIs.

Every compact branch fits512 visible Adam(.001) updates,32epochs/batch4,
64 fit/16 validation MNIST-source scenes and8 independent test-source scenes.
11399 positive mask regions train. Objective is RPN+box+mask+.5 unit-weight
semantic CE plus explicit kernel L2;paired learned/BN checkpoints restore
selected epoch32 validation box AP50.063308. Held-out box AP50.113274,
integral mask AP50 zero,semantic mIoU.735548,PQ/SQ/RQ
.165495/.165495/.181818,things PQ zero/stuff PQ.910225. The report explicitly
attributes strong stuff results to easy material cues,shows missed objects/
void regions and does not claim successful instance segmentation or a benchmark.

Three new canonical headers `nn/panoptic_fpn_models.hpp`, `nn/panoptic_models.hpp`,
`nn/panoptic_scene_models.hpp` have44/44 documented definitions,19 compiled/run
examples,six hovers and six native tests. Standalone includes,original C++23
full fixtures/quick fit and Markdown/current website exports pass;all seven
media resolve. Original quick probes decrease mask BCE.870167->.724862 and
semantic CE1.45415->.601147 without changing their learning gates. Evidence:
`build/cnn-panoptic-validation/`, `build/cnn-panoptic-coverage.json`,
`build/cnn-panoptic-hover.json`, `build/cnn-panoptic-review/`. Combined coverage
is1739/1739 across70 headers,327 native tests.47 reports remain;volumetric
segmentation is the final unfinished report in1.7.


### Tensor YOLACT completed (81/129 overall)

Five executed C++ blocks,one editable/executed TikZ and seven reviewed figures
show signed prototype assembly/crop derivatives,the all-predecessor suppression
chain,actual image/support pairs,visible whole-model fitting,failed held-out
masks and restored shared prototype maps. Native independent arithmetic checks
every prototype/coefficient derivative,crop normalization,ignored-band matching,
3:1 mining/class derivatives,weighted joint objective,empty positives,reordered
support/crop/semantic target pairs and known decoded prototype supports.
Fast NMS suppresses via every higher-score predecessor,including suppressed
ones;class/image isolation,caps and stable ties retain source conventions.
All numerical matching,target sampling,mining,losses,Fast NMS and mask decoding
are canonical tensor APIs;legacy helpers only adapt containers.

Full ResNet101/FPN256/32-prototype YOLACT executes50157071 learned scalars and
all348 gradients at64px;550px grid69/35/18/9/5 yields19248 priors/138px
prototypes by geometry,not a full550px fitting claim. Compact ResNet18/div8,
FPN32/eight prototypes has293407 learned/1200 BN state scalars,258 anchors,
16px prototypes and every learned branch gradient nonzero. Background-last,
true-ratio priors,pixel-centre targets/discrete crop areas,no one-pixel crop
expansion and32 positive masks/image remain explicit teaching adaptations.
Tensor shuffle draws differ from the old std::mt19937 positive-mask sequence.

Every compact branch fits512 visible Adam(.001) updates,32 epochs/batch4,
64 fit/16 validation MNIST-source scenes and8 independent test-source scenes.
Joint loss is mined CE+1.5 SmoothL1 normalized by positives +6.125 cropped mask
BCE+semantic BCE,with explicit kernel L2 and paired learned/BN restoration.
6080 positive masks train. Epoch18 validation box AP50.038677 selects the
checkpoint;all held-out26-digit box/mask AP metrics are zero. Wrong large
masks and weak prototypes remain visible;the report explicitly says this
bounded run fails to learn a useful held-out detector. Custom integral mask
AP is distinguished from official COCO;no real-time or paper-accuracy claim.

Original full fixtures and bounded quick run pass. The old unconditional
post-restore mask-convergence assertion conflicted with bounded quick mode,
which can restore the first tied zero-AP checkpoint. Quick mode now checks
finite diagnostics;the explicitly requested full workflow retains the original
mask-loss-decrease gate. Actual quick probe BCE.836033->.869330 is reported,
not relabelled as learning. Core analytic/value/gradient checks are unchanged.

`nn/yolact_models.hpp` has40/40 documented definitions,14 compiled/run public
examples,five clangd hovers and six native tests. Standalone include,original
C++23 executable,Markdown/current website exports pass;all seven media resolve.
Evidence: `build/cnn-yolact-validation/`, `build/cnn-yolact-coverage.json`,
`build/cnn-yolact-hover.json`, `build/cnn-yolact-review/`. Combined coverage is
1695/1695 across67 headers,321 native tests.48 reports remain.


### Tensor Mask R-CNN and shared instance supports completed (80/129 overall)

Five executed C++ blocks,one editable/executed TikZ and six reviewed figures
show aligned features/derivatives,selected-mask gradients,actual fitting supports,
three-task Adam traces and held-out image/truth/predicted-mask triptychs.
Fractional/border values and every feature derivative agree with independent
bilinear arithmetic(error5.45e-7);the affine-plane report additionally checks
Float32 output differences within1e-5. Wrong-class gradients are exactly zero.
Full256/80 mask head executes2643280 learned scalars/all12 gradients;compact
32/10 head50666/all12 nonzero. This is full mask-head validation,not a claim
that the entire paper ResNet/FPN model was trained.

The complete compact stride4 backbone/2304-anchor RPN/aligned box/mask model
has1077290 learned scalars,no moving state. Every branch fits512 visible
Adam(.001) updates,batch4/32 epochs,explicit RPN+box+mask objectives and kernel
L2.64 fitting/16 validation scenes use training-source MNIST;8 test scenes
use independent test-source digits. Validation box AP50 selects epoch30/.868659.
Held-out26-digit box AP50 is.806614;integral mask AP50/.75/mean50:95 are
.806614/.370370/.391733. These integral precision-envelope mask metrics are
explicitly distinct from official COCO101-point,crowd/area protocols. Wrong
masks/classes remain visible;there is no pretrained/benchmark claim.

One shared tensor sampler now optionally retains Bool object supports without
changing RNG draws,pixels,boxes or classes. Legacy plain/instance scene entry
points forward to that sampler and agree exactly with each other;interpolation
may differ by one byte from the old host sampler at borderline rounding.
Tensor mask paste uses bilinear pixel centres and inclusive probability.5.
Mask AP preserves stable score/first-truth ties and duplicate-best matching.
The existing RoI decoder adds optional suppress=false raw class candidates;
default NMS behavior remains unchanged. Original15 two-stage and9 scene native
regressions pass in addition to12 new instance/Mask R-CNN tests.

New `nn/instance_scene_models.hpp` and `nn/mask_rcnn_models.hpp`,plus two private
shared-sampler definitions,add29 documented definitions,16 compiled/run public
examples and six clangd hovers. Direct includes,original C++23 full fixtures/
quick fit,Markdown and current website exports pass;all six media resolve.
Evidence: `build/cnn-mask-validation/`, `build/cnn-mask-coverage.json`,
`build/cnn-mask-hover.json`, `build/cnn-mask-review/`. Combined coverage is
1655/1655 across66 headers,315 native tests.49 reports remain.


### Tensor HRNet completed (79/129 overall)

HRNet adds four executed C++ blocks,one editable/executed TikZ diagram and
six reviewed figures(three SVG,three actual-data/prediction/feature PNGs).
Independent odd-grid identity projections,aligned bilinear interpolation,
centre-tap stride2 sampling and every source derivative pass. The hand-check
fixture has high/low sums700/240 and source-gradient sums47/47. Final branches
retain independent resolutions before projection/concatenation.

Full W32/21-class HRNet executes29546997 learned/58304 BN state scalars and
all920 learned tensor gradients at33px,eight exchanges/four streams and
480-channel final concatenation. Compact W8/stem16/two stage1 bottlenecks,
two basic blocks and one exchange per stage has382580 learned/3744 state
scalars. Every compact gradient is nonzero;96/97px logits align with saved
24/12/6/3 or25/13/7/4 streams. Invalid branch resolutions,width overflow and
RGB input mismatches reject.

The complete compact model fits512 visible Adam(.001) updates,batch4,32 epochs,
64/16/16 independent96px shape scenes,fit-only pixel weights,explicit kernel
L2 and paired learned/BN restoration. Epoch19 validation mIoU.514050 yields
test accuracy.860664/mIoU.480074,with background.952996 versus circle.241481,
rectangle.377457 and triangle.348361. Actual wrong masks remain. Restored
per-stream channel-mean feature maps share one intensity range and are clearly
identified as activations,not class probabilities.

`nn/hrnet_models.hpp` has17/17 definitions,five compiled/run examples,three
hovers and three native tests. Direct include,original C++23 full fixtures/
quick fit and Markdown/current website exports pass,all six media resolved.
Evidence: `build/cnn-hrnet-validation/`, `build/cnn-hrnet-coverage.json`,
`build/cnn-hrnet-hover.json` and reviewed figures. Combined coverage is
1626/1626 across64 headers,303 native tests. There are50 reports remaining,
including four segmentation reports(instance/panoptic/volumetric).


### Tensor DeepLab v3/v3+ and PSPNet completed (78/129 overall)

These two reports add eight executed C++ blocks,two editable/executed TikZ
diagrams and11 reviewed figures(six SVG,five actual-data/prediction PNGs).
Global image pooling/projection/broadcast values and every input derivative,
exact overlapping floor-start/ceil-end bins including bins larger than the
input,and every mean/source derivative match independent oracles.

Full ResNet50 v3/v3+ executes39048533/26696133 learned,56192/70720 state
scalars and all179/203 learned tensor gradients at33px. Compact241740/198004
learned,1584/2256 state models retain five ASPP branches and the optional
separable low-level decoder,with all compact derivatives nonzero. Stride8
rate doubling preserves parameter shapes. Full PSPNet101 executes67949930
learned/110976 state,all334 gradients,4096-channel context concatenation and
auxiliary scores. Compact222704/1424 keeps all1/2/3/6 pools and exact96/97px
main/auxiliary alignment with nonzero gradients. Auxiliary-disable and invalid
rate/bin behavior also pass.

All three compact models fit512 visible Adam(.001) updates each,batch4,
32 epochs,64/16/16 independent96px scenes,fit-only pixel weights and explicit
kernel L2. PSPNet explicitly uses(main CE+.4 auxiliary CE)/1.4 from one forward,
so BN updates once. Strict validation selects paired learned/BN checkpoints.
V3/v3+ epochs16/14 yield validation mIoU.381906/.504644,test.403804/.446050,
with weak circle recognition explicit. PSPNet epoch25 yields validation.522807,
test accuracy.875156/mIoU.547042 and per-class IoUs.945169/.361144/.415453/
.466404. Actual wrong masks remain;no Xception/VOC/Cityscapes claim is made.

`nn/deeplab_v3_models.hpp`, `nn/pyramid_pool_models.hpp` have39/39 definitions,
nine compiled/run examples,five hovers and seven native tests. Direct includes,
two original C++23 full fixture/quick-fit workflows and Markdown/current
website exports pass,all11 media resolved. Evidence:
`build/cnn-pyramid-context-validation/`, `build/cnn-pyramid-context-coverage.json`,
`build/cnn-pyramid-context-hover.json` and reviewed figures. Combined coverage
is1609/1609 across63 headers,300 native tests. There are51 reports remaining;
HRNet is in progress and not yet counted.


### Tensor DeepLab v1/v2 and exact CRF completed (76/129 overall)

DeepLab v1/v2 adds five executed C++ blocks,one editable/executed TikZ diagram
and nine reviewed figures(four SVG,five actual-data/raw/refined-mask PNGs).
Full20505429/44048532 models execute all32/320 learned tensor gradients at33px,
with0/105344 state scalars. Dense ResNet stride32/16/8 preserves176712 learned/
1200 state scalars while changing stage geometry. Compact321452/185944 models
retain all layers and have every learned gradient nonzero at96px. Independent
ASPP tap/scatter values and every input derivative pass.

Exact spatial/bilateral Potts kernels now use detached tensor distance/exp/
degree normalization instead of host image loops. Symmetry,zero self edges,
Float32 RGB-difference/Float64 normalization conventions,independent every-pair
kernel oracle,host mean-field updates and all unary central differences pass.
Inference remains explicit bounded O(P²),not fast permutohedral filtering.
Pixel-centre sampling selects1,4,7,...,94 at96->32 and preserves source
input gradients. Legacy preview now uses these tensor numerics/metrics.

Both complete compact models fit512 visible Adam(.001) updates,batch4,
32 epochs,64/16/16 independent96px shape scenes,fit-only class weights and
explicit kernel L2. V2 restores learned/BN stores together;V1 state is empty.
Selected epochs32/27 yield validation mIoU.537894/.402606,test pixel accuracy
.885959/.853488 and native-resolution test mIoU.610431/.463214. On the separate
first-four-image32px subset,fixed five-step CRF changes.425411->.439245 and
.384999->.409759. Subset gains are explicitly separate from full test metrics;
all failed masks remain visible and no VOC/fast-filtering claim is made.

`nn/deeplab_models.hpp`, `nn/dense_crf_models.hpp` have36/36 definitions,
15 compiled/run examples,seven hovers and seven native tests. Both direct
includes,original C++23 full fixtures/quick fitting and Markdown/current
website exports pass,all nine media resolved. Evidence:
`build/cnn-deeplab-validation/`, `build/cnn-deeplab-coverage.json`,
`build/cnn-deeplab-hover.json` and reviewed figures. Combined coverage is
1570/1570 across61 headers,293 native tests. There are53 reports remaining;
DeepLab v3/v3+ and PSPNet ports are in progress and not yet counted.


### Tensor nested/attention U-Net completed (75/129 overall)

UNet++/attention U-Net adds four executed C++ blocks,one editable/executed
TikZ diagram and seven visually reviewed figures(three SVG,four actual
image/mask/prediction/gate PNGs). Full9156264/31381414 models execute all68/70
learned tensor gradients at32px,with15 dense nested nodes/four supervised
heads or four additive skip gates. Compact573240/491468 models retain every
head/gate,and all derivatives are nonzero. Independent positive/negative/
saturated gate values and both input derivatives pass; the report checks61
analytic curve points. Odd nested95x97 alignment and attention80x96 rounded
geometry,including odd base width3,remain supported.

Both compact models fit512 visible Adam(.001) updates,batch4,32 epochs,
64 fit/16 validation/16 test96px scenes and fit-only weights/kernel L2.
UNet++ explicitly averages CE over all four learned heads; inference uses
the deepest. No BN state or dropout is hidden. Selected epochs27/31 reach
validation mIoU.777953/.732494 and test.817592/.780081; pixel accuracies are
.966973/.956299. Circle/background disparities and all wrong masks are
explicit,without a general ranking/medical benchmark claim. The first test
image's actual12/24/48/96px gates retain a fixed[0,1] display range.

`nn/unet_variant_models.hpp` has24/24 definitions,eight compiled/run examples,
three hovers and four native tests. Standalone/original C++23 target plus
bounded fit and Markdown/current website exports pass,all seven media resolved.
Evidence: `build/cnn-unet-variants-validation/`,
`build/cnn-unet-variants-coverage.json`, `build/cnn-unet-variants-hover.json`
and reviewed figures. Combined coverage is1534/1534 across59 headers,286
native tests. There are54 reports remaining; DeepLab is in progress.


### Tensor SegNet completed (74/129 overall)

SegNet adds four executed C++ blocks,one editable/executed TikZ diagram and
five visually reviewed figures(three SVG,two actual image/mask/prediction PNGs).
Indexed2x2 pooling now computes detached tensor argmax and integer NHWC offsets;
sparse unpool uses differentiable tensor segment scattering. No host pixel
scan/inverse-map numerics remain. Independent negative/odd/multi-batch/channel
values and every original/pooled derivative pass,as do first-maximum ties,
zero odd edges and malformed duplicate/out-of-range record rejection.

Full29449355 learned/15872 moving-state scalars execute all102 gradients at64px.
Compact463132/1984 preserves96/97 grids with all nonzero derivatives. For an
even96px probe,Int64 index cache562176 bytes is half the1124352-byte full
Float32 feature cache; the report excludes pooled tensors/graphs from that
comparison and makes no peak-memory claim. Original source comments now
accurately describe tensor pooling/RNG,also correcting earlier FCN/U-Net seed
and segmentation-loss EDT comments without changing their executed behavior.

The complete compact model fits512 visible Adam(.001) updates,batch4,32 epochs,
64 fit/16 validation/16 test96px scenes,fit-only class weights,explicit kernel
L2 and paired BN checkpoint restoration. Seed17 selects epoch26,validation
mIoU.367691; test pixel accuracy.871738 but mIoU.377209,with circle/triangle
IoUs only.038031/.004332. Failure is explicit and retained in actual masks.

`nn/segnet_models.hpp` has15/15 definitions,six compiled/run examples,three
hovers and four native tests. Standalone/original C++23 target and bounded
fitting plus Markdown/current website exports pass,all five media resolved.
Evidence: `build/cnn-segnet-validation/`, `build/cnn-segnet-coverage.json`,
`build/cnn-segnet-hover.json` and reviewed figures. Combined coverage is
1510/1510 across58 headers,282 native tests. There are55 reports remaining.


### Tensor FCN and U-Net completed (73/129 overall)

FCN32s/16s/8s and U-Net add eight executed C++ blocks,two editable/executed
TikZ diagrams and12 reviewed figures(six SVG,six actual image/mask/prediction
PNGs). Full released FCN counts134346581/134357354/134362751 and every
32/34/36 learned tensor gradient execute at8px with released padding/crops.
Full U-Net31030658/46 executes188->4;572->388 geometry is checked separately.
Compact FCN counts271916/272176/272308 preserve95/96/97px odd grids; compact
U-Net485844 preserves96/112/128px. All compact tensor gradients are nonzero.
Independent bilinear/fusion/crop values and every input derivative pass.

Shared tensor data uses class-independent shape photometry,occlusion and
aligned labels. Tensor RNG replaces legacy mt19937,so source seed pixels
change explicitly. Fit-only inverse-square-root weights include pseudocount1;
weighted CE and confusion/IoU use tensors,with exact Int64 segment-summed
counts and Float64 metrics. Native tests cover deterministic/reseeded pixels,
row-copy independence,weight/CE derivatives,prediction-only/absent-class IoU,
palette order and invalid inputs. No BN state exists in these two models.

Each of the four compact models fits512 visible Adam(.001) updates,batch4,
32 epochs,64 fit/16 validation/16 test96px scenes,seeds11/22/33 and explicit
kernel L2. Validation-selected FCN32/16/8 epochs29/27/22 reach test mean IoU
.463854/.723306/.734045; U-Net epoch32 reaches.768823(.955254 pixel accuracy),
with weaker circle IoU.573475 explicit. All actual masks retain mistakes.
These bounded procedural demonstrations make no benchmark-ranking claim.

`nn/segmentation_data.hpp`, `nn/fcn_models.hpp`, `nn/unet_models.hpp` have
54/54 definitions,23 compiled/run examples,eight hovers and nine native tests.
Three standalone includes,two original C++23 targets with every fixture and
bounded fitting,and Markdown/current website exports resolve all12 figures.
Evidence: `build/cnn-fcn-unet-validation/`, `build/cnn-fcn-unet-coverage.json`,
`build/cnn-fcn-unet-hover.json` and reviewed figures. Combined coverage is
1495/1495 across57 headers,278 native tests. There are56 reports remaining.


### Tensor segmentation losses completed (71/129 overall)

Dice/Tversky, Lovasz-Softmax and detached signed-distance boundary loss add
three executed C++ blocks,one editable/executed TikZ diagram and five reviewed
figures(four SVG,one actual field/target PNG). Region masks,sorting,Jaccard
increments and exact distances now use tensors. The original linear-time host
EDT becomes exact separable tensor minima with O(H*W*(H+W)) work and
O(H*W*max(H,W)) temporary memory, explicitly documented as a tradeoff.

Independent quotient-rule derivatives,every fixed-sort Lovasz subgradient,
negative boundary loss/detached-map derivatives,void/perfect/absent masks and
100 rectangular exhaustive nearest-seed oracles pass,including no seeds and
signed interior/boundary/exterior/full/empty maps. Seven public examples,
five native tests,four clangd hovers,both standalone includes/legacy original
program and Markdown/current website exports pass. Canonical
`nn/segmentation_losses.hpp` has11/11 documented definitions.

Four independent24x24 two-class fields expose all4608 logit leaves before
Adam(.1),then200 explicit updates. All known-mask IoUs reach1; final Dice,
Tversky,Lovasz and Dice+boundary losses are.001614630/.001543164/.001452238/
-.000130733. Actual initial/target/final pixel galleries retain the alignment.
The report explicitly calls this in-sample mask reconstruction,not held-out
CNN segmentation. Evidence: `build/cnn-seg-losses-validation/`,
`build/cnn-seg-losses-coverage.json`, `build/cnn-seg-losses-hover.json` and
reviewed figures. Combined coverage is1441/1441 across54 headers,269 native
tests. There are58 reports remaining,including12 further segmentation examples.


### Detection group finished with tensor YOLO v5-style (70/129 overall)

All17 detection reports are complete. YOLO v5-style adds six executed C++
blocks, one editable/executed TikZ diagram and seven visually reviewed figures
(four SVG,three actual-data PNG galleries). Full80-class CSP/SPPF/PAN executes
all177 tensor gradients at128px, with7235389 learned/19008 state scalars.
Compact10-class grayscale/divisor4 retains every stage,451735 learned/4752
state scalars and64/96/128px dynamic heads. Independent original CSP, pooling,
CIoU frozen-alpha,21-slot ratio/neighbour and exact rectangular mosaic fixtures
pass. Seven canonical native tests also check complete analytic empty/perfect
loss derivatives, fallback/tie collisions, every mosaic pixel, label/box
alignment, flip involution, independent storage and combined-head decoding.

All compact leaves fit512 visible Adam(.001) updates, batch4,32 epochs,
64/96/128px cycles every10 updates, explicit kernel L2 and paired BN checkpoints.
Each fixed-centre mosaic takes four fitting-only IDs, followed by tensor-random
horizontal flips. Nine pixel priors use only64 fitting scenes. Selected epoch20
has validation AP50.127463; test AP50/AP75/AP50:95 are.063752/.002704/.017808
across23 digits/eight scenes with718 detections. Failed generalization and
extra/misplaced boxes are explicit, with no pretrained/benchmark claim.

`nn/yolo_v5_models.hpp` and `nn/detection_mosaic.hpp` have49/49 definitions,
20 compiled/run examples,six hovers and standalone includes. Markdown/current
website exports resolve all seven figures. Evidence: `build/cnn-yolo-v5-validation/`,
`build/cnn-yolo-v5-coverage.json`, `build/cnn-yolo-v5-hover.json` and reviewed
montage. Combined coverage is1430/1430 across53 headers,264 native tests.
There are59 remaining reports, beginning with segmentation.


### Tensor YOLO v3 completed (69/129 overall)

YOLO v3 adds five executed C++ blocks, one editable/executed TikZ diagram and
five reviewed figures (three SVG,two actual-data PNG galleries). Full80/20
class models execute all222 learned tensor gradients at128px, with
61949149/61626049 learned and52608 state scalars. Compact75-convolution
Darknet53/top-down heads have976795 learned/6576 state scalars; all gradients
and64/96/128px dynamic grids pass.

Tensor targets select one global prior/scale, preserve larger/first collision
rules, ignore all-truth prediction overlaps>.5 and retain relative-area
coordinate weights. Independent multi-label BCE, coordinate, background and
ignored fixtures check every derivative across all heads. Zero/extreme logits,
empty-scene objectness gradients and zero class/geometry gradients, global
prior routing, ties, per-image/class cross-level decoding and raw-candidate
counts pass. Legacy interfaces forward and the original program runs every
full fixture plus its bounded fitting/checkpoint workflow.

All compact parameters fit512 visible Adam(.001) updates, batch4,32 epochs,
64/96/128px cycles every10 updates and explicit kernel L2. Nine pixel priors
see only64 fitting scenes. Paired learned/BN checkpoint epoch27 has validation
AP50 .043582, but all test AP metrics are zero across23 digits/eight scenes,
with800 detections. The report explicitly documents failed generalization and
retains thin/mislocalized false boxes. Initialized full checks are distinct
from the compact fit, with no pretrained/benchmark claim.

`nn/yolo_v3_models.hpp` has27/27 definitions, ten compiled/run public examples,
five hovers and seven native tests. Standalone include and Markdown/current
website HTML exports pass with all five media references. A concurrent core
rebuild briefly interrupted linking; `build/mepml-pc/datamunge.pc` now points to
an isolated completed core snapshot in `build/cnn-core/` (recorded source/mtime),
so later runs avoid a changing shared library. All24 recent YOLO native tests
pass against that snapshot. Evidence: `build/cnn-yolo-v3-validation/`,
`build/cnn-yolo-v3-coverage.json`, `build/cnn-yolo-v3-hover.json` and reviewed
figures. Combined coverage is1381/1381 across51 headers, with257 native tests.
There are60 reports remaining; YOLO v5-style is the final detection report.


### Tensor YOLO v2/WordTree completed (68/129 overall)

YOLO v2 adds six executed C++ blocks, one editable/executed TikZ diagram
and nine reviewed figures (five SVG, four actual-data PNG galleries). Diagram
text wraps without hyphenation; joint classification-only loss has its own
readable vertical scale. Full20-class/five-prior and9418-node/three-prior graphs
execute all68 learned tensor gradients at128px, with50655389/79502989 learned
and20672 BN-state scalars. No pretrained/9418-class accuracy is implied.

Fit-only tensor IoU clustering retains source seeding, Double mean updates,
empty centres and area order. Global anchor/cell targets, current-IoU
confidence, overlap ignore, larger collisions and parity labels use tensors.
WordTree sibling softmax/path products and coarse/fine losses use tensors;
only topology metadata uses host indices. Independent fixtures verify every
path/detection/classification-only derivative, selection reversal and weak
objectness, rectangular targets/decoding, empty images and9418-node capacity.
All eight native tests and original full fixtures/bounded flat/joint fits pass.

Both complete compact models have2584 state scalars and fit512 visible
Adam(.001) updates with explicit kernel L2, batch4,32 epochs,64/80/96px cycling
and paired weight/BN checkpoints. Flat801447 scalars use512 detection updates;
joint798351 use102 coarse-box/410 fine-classification-only updates. Priors see
only64 fitting scenes;16 validation scenes select epochs31/29. Eight held-out
scenes/23 digits give AP50/AP75/AP50:95=.149621/.056106/.069738 (flat,715 boxes)
and .020432/0/.005559 (joint,800 boxes). Limited accuracy, different objectives
and update allocation are explicit; no architecture ranking is claimed.

Two canonical headers have48/48 documented definitions,18 compiled/run public
examples (including the indented constructor example), seven hovers and eight
native tests. Standalone headers pass. Markdown and the current website report
exporter pass with all nine local-media references; the old Doxygen-page
export script was removed by the repository's concurrent website migration.
Historical Doxygen-page checks remain recorded; new page checks use website
HTML. Evidence: `build/cnn-yolo-v2-validation/`,
`build/cnn-yolo-v2-coverage.json`, `build/cnn-yolo-v2-hover.json` and reviewed
figures. Combined CNN coverage is1354/1354 across50 headers, with250 native
tests. There are61 reports remaining; YOLO v3/v5-style are next.


### Tensor YOLO v1 completed (67/129 overall)

YOLO v1 adds five executed C++ blocks, one editable/executed TikZ diagram
and six reviewed figures (three SVG, three actual-data PNG galleries).
The actual full448RGB backbone has60155968 scalars and48 finite defined
gradients. Its211547582-scalar dense head is counted from shapes rather
than allocated; total271703550. Both complete compact normalization modes
and every learned tensor gradient pass, with optional BN-state count3296.

All grid targets, collision handling, detached IoU/responsibility, squared
losses and decoding use tensors. Every analytic derivative in four exact/
imperfect/swapped/empty-cell fixtures passes. The canonical affine transform
uses Double tensor inverse pixel-centre sampling/exposure/byte quantization;
identity, translations, border interpolation, scaling, clipping/removal and
row isolation pass. Original augmentation/pretraining fixtures also pass.

Optional first20-convolution crop pretraining is explicit:168 fit/45 validation
crops,353714 learned/2272 BN-state scalars,88 Adam(.001) updates over8 epochs.
Validation NLL selects epoch8 at1.453287/accuracy.488889. Actual crops show
neighbouring/overlapping digits. Name/type/shape/finite validation precedes
exact prefix transfer; incomplete normalization prefixes are rejected before
mutation, leaf identity is retained and last four convolutions/dense heads
remain initialized. A fresh visible Adam(.0005) fits all2801020 detector
scalars for512 updates (batch4,32 epochs), with explicit kernel L2 and paired
BN checkpoints. It selects epoch25 at validation AP50 .010068; held-out
AP50/AP75/AP50:95=.079243/0/.012379,617 boxes. Poor localization/generalization
is explicit, without benchmark claims. Scenes are generated directly at56px.

`nn/detection_augmentation.hpp` and `nn/yolo_v1_models.hpp` have34/34 documented
definitions,14 compiled/run public examples, six hovers and nine native tests.
Legacy numerical interfaces forward; the original program builds/runs all
fixtures and bounded detector fitting. Standalone includes and both exports
pass. Evidence: `build/cnn-yolo-v1-validation/`, `build/cnn-yolo-v1-export/`,
`build/cnn-yolo-v1-coverage.json`, `build/cnn-yolo-v1-hover.json` and reviewed
figures. Combined CNN coverage is1306/1306 across48 headers, with242 native
tests. There are62 reports remaining; YOLO v2/v3/v5-style are next.


### Tensor EfficientDet and DETR completed (66/129 overall)

Two reports add nine executed C++ blocks, two editable/executed TikZ diagrams
and10 reviewed figures (six SVG, four actual-data PNG galleries). Fusion bar
axes retain the full checked derivative range; DETR concept text wraps cleanly.
Full D0/80 validation verifies3874217 learned/47136 state scalars and493 tensor
gradients at128 pixels (49104 anchors at default512). Full DETR verifies
41575061/53120 and424 defined finite gradients,417 nonzero; zero-content
self-attention/key-bias invariance explains initial zeros.

EfficientDet preserves eight-node/nineteen-edge BiFPN, odd grids, positive
normalized edge weights and shared separable heads with per-level BN state.
Independent fusion values and every input/scalar derivative, all-zero edges,
all D0–D7 metadata and full/compact graph gradients pass. The final compact
P7 BN-beta derivative is analytically zero and explicitly documented.

DETR positions, class/box costs, rectangular Hungarian potential/distance
updates, one-to-one targets and query decoding all use tensors. Host arrays
hold only discrete augmenting-path indices. One hundred negative-cost
rectangular problems match exhaustive minima. Every position coordinate,
weighted-class derivative and disjoint-box central difference passes; empty
truths, no-object normalization, auxiliary rematching and no-NMS decoding
semantics are explicit. Potential overflow is rejected and regression-tested.

Both complete compact models fit512 visible Adam(.001) updates, batch4 and
32 epochs with explicit kernel L2 and paired learned/BN-state restoration.
The64 fit/16 validation/eight held-out MNIST-scene split selects epoch21 in
both runs. EfficientDet test AP50/AP75/AP50:95=.004607/0/.000526 (800 boxes);
DETR test AP is zero at every measured threshold (80 query detections).
Reports preserve the failure to generalize, actual false positives and measured
loss curves, without pretrained/benchmark claims or architecture rankings.

Four new canonical headers have59/59 definitions,21 compiled/run examples,
11 hovers and11 native tests. Both original programs compile/run every full
fixture and bounded fitting, standalone includes pass, and both report formats
export. Evidence: `build/cnn-set-fusion-validation/`,
`build/cnn-set-fusion-export/`, `build/cnn-set-fusion-coverage.json`,
`build/cnn-set-fusion-hover.json` and reviewed figures. Combined CNN coverage
is1272/1272 across46 headers, with233 native tests. There are63 reports
remaining; YOLO v1/v2/v3/v5-style are next.


### Tensor RetinaNet, FCOS and CenterNet completed (64/129 overall)

Three reports add15 executed C++ blocks, three editable/executed TikZ diagrams
and16 visually reviewed figures (10 SVG, six actual-data PNG galleries).
Full-weight source probes execute all learned tensor derivatives: RetinaNet
37968692 learned/53120 state scalars,195 tensors; FCOS36426138/53120,218;
CenterNet14046164/10496,81. Full probes use128-pixel inputs without pretraining;
640-pixel RetinaNet/FCOS anchor/point counts76725/8525 are checked separately.

All targets, focal/IoU/heatmap losses and decoding use canonical tensors.
Independent oracles cover every focal derivative, anchor/point coordinate,
FCOS complete objective derivative and CenterNet Gaussian pixel. Threshold,
ignored bands, smallest-area/first ties, larger-object centre collisions,
source score ties, per-image/class budgets, peak plateaus and pre-filter top
caps are explicit and tested. Legacy interfaces delegate and all three original
programs build/run their complete fixtures and bounded checkpoint fitting.
SSD's new optional raw decoder is regression-tested, with defaults unchanged.

Each complete compact model fits512 visible Adam(.001) updates, batch4 and
32 epochs, with explicit kernel L2 and paired learned/BN-state checkpoints.
The common64 fit/16 validation/eight test split uses actual cached MNIST pixels.
Selected epochs18/15/26 yield held-out AP50 .021629/.018984/.025192; AP75
0/.003507/0 and AP50:95 .003207/.005745/.007186. All reach the100/image cap.
Reports explicitly document poor generalization and retain the dense false
positives, rather than claiming useful detectors or paper benchmark accuracy.

The three canonical headers have92/92 documented definitions,36 compiled/run
public examples,13 hovers and22 new native tests; one additional SSD raw-decode
test also passes. All standalone headers and Markdown/Doxygen exports pass.
Evidence: `build/cnn-dense-detection-validation/`,
`build/cnn-dense-detection-export/`, `build/cnn-dense-detection-coverage.json`,
`build/cnn-dense-detection-hover.json`, `build/cnn-center-hover.json` and the
reviewed figures. Combined CNN coverage is1213/1213 across42 headers, with222
native tests. There are65 reports remaining; DETR and EfficientDet are next.


### Tensor SSD completed (61/129 overall)

SSD adds six executed C++ blocks, one editable/executed TikZ diagram and seven
reviewed figures (five SVG, two actual-data PNG galleries). Default geometry
uses equal x/y pixel scales; learned3-4-5 channel normalization and the full
multibox objective/derivatives have independent checks. Per-image mining and
batch-positive normalization are separate, all-background loss/gradients are
exact zero, lower-index mining ties and higher-index source inference ties
are explicit. Every default-coordinate, collision-free target assignment,
pre-area-filter top cap, clipping and class/image suppression is tested.

The actual full300-pixel SSD300 forward/backward verifies26285486 scalars,
8732 predictions and all71 learned tensor gradients, finite/nonzero with
norms[.000053089,28.780021667]. It is explicitly initialized validation, not
trained benchmark output. The complete compact1536-default model fits1167362
scalars, including its learned L2 scale, using512 visible Adam(.001) updates,
batch4 and32 epochs. It selects epoch18 at validation AP50.310301. Eight test
scenes/23 digits yield AP50/AP75/AP50:95=.203434/.069719/.089794,105 detections
and selected mean scale10.035564. Limited generalization and visible mistakes
remain explicit; no paper accuracy or architecture ranking is claimed.

`nn/ssd_models.hpp` has25/25 documented definitions, nine public examples,
five hovers and nine native tests. All compile/run, including full SSD300
backward and compact every-tensor gradients. The canonical model, geometry,
mining, targets and decode use tensors; legacy entry points delegate. The
original C++23 program builds and runs every full fixture plus its bounded
training/checkpoint evaluation. Standalone header and both report exports
pass. Evidence: `build/cnn-ssd-validation/`, `build/cnn-ssd-export/`,
`build/cnn-ssd-coverage.json`, `build/cnn-ssd-hover.json` and reviewed figures.
Combined CNN coverage is1121/1121 across39 headers, with199 native tests.
There are68 reports remaining; RetinaNet, FCOS and CenterNet are next.


### Feature pyramid detector completed (60/129 overall)

The FPN report adds four executed C++ blocks, one editable/executed TikZ diagram
and seven reviewed figures (four SVG, three actual-data PNGs). The controlled
7x9/4x5/2x3 fixture checks every output/input derivative against an independent
nearest-index reference, with zero printed error and35 negative responses
preserved. Different smoothing coefficients expose accidental propagation of
smoothed maps. The extra coarser map samples exactly; constant-level maps
verify canonical224 routing and clamped extremes. The64-pixel detector uses
canonical32,1023 P2–P6 anchors and full2174882 learned parameters.

The visible joint Adam loop makes512 updates, selecting epoch13 at validation
AP50.333201. On eight test scenes/23 digits,100 proposals/image cover
.913043/.608696 at IoU.5/.7. AP50/AP75/AP50:95 are.324489/.176811/.163452,
with207 retained class-specific detections. Limited accuracy and many extra
boxes remain explicit; no paper benchmark or architecture ranking is claimed.

`nn/feature_pyramid_models.hpp` plus the new shared pooled-head finisher add
17 documented definitions, nine compiled/executed public examples, five hovers
and five native tests. Twenty native tests including the earlier two-stage
regression suite pass. All27 public examples across the two affected headers
compile/run; coverage is61/61. Full pyramid/model counts and every learned
Tensor gradient, routed order and feature-only gradients, odd grids, canonical
side boundaries and invalid inputs pass. Legacy FPN and pooled-head names
forward to canonical tensor helpers. Standalone headers, original FPN C++23
syntax and both exports pass. Evidence: `build/cnn-fpn-validation/`,
`build/cnn-fpn-export/`, `build/cnn-fpn-coverage.json`, `build/cnn-fpn-hover.json`
and reviewed images under `build/cnn-detection-review/`. Combined CNN coverage
is1096/1096 across38 headers, with190 accumulated native tests. There are69
reports remaining; SSD and the remaining one-stage detectors are next.


### R-CNN family completed (59/129 overall)

R-CNN, Fast R-CNN and Faster R-CNN add twelve executed C++ blocks, three
retained/executed TikZ diagrams and nineteen reviewed figures (nine SVG and ten
actual/control PNG galleries). All three fit their complete source compact
architectures using visible Adam(.001) loops and explicit kernel L2. The same
64 fit/16 validation/eight test scene protocol uses separate official MNIST
pools. No normalization/dropout/moving model state is hidden. The reports state
source adaptations: softmax/joint regression instead of paper SVM/ridge,
bilinear14-to7 pooling instead of rounded bins/ROIAlign, tensor RNG sequences,
and detached approximate joint RPN/head learning.

R-CNN fits426931 scalars on938 proposal/ground-truth patches (664 foreground,
596 regression-eligible),480 updates, selecting epoch6 by joint validation
loss2.293463. Box refinement gives a mixed result: AP75.492616→.505146,
AP50 stays.682429 and AP50:95 falls.485728→.428570. Fast R-CNN fits946963
scalars in512 updates, selecting epoch13 at validation AP50.557778; test AP50
is.712130, AP75.503667, AP50:95.427539. Its sampler shows14 foreground RoIs
across two images, explicitly up to—not exactly—one quarter. Faster R-CNN fits
1026624 scalars in512 joint updates, selecting epoch13 at validation AP50
.852206;100 proposals/image cover all23 test targets at IoU.5 and.913043 at
.7, while final AP50/AP75/AP50:95 are.659215/.338100/.378088. Small-scene
selection gaps, merged/extra detections and mixed refinement remain visible;
these results do not establish an architecture ranking or paper benchmark.

`nn/two_stage_detection_models.hpp` has44/44 documented definitions, eighteen
compiled/executed public examples, six hovers and fifteen native tests. Counts,
all learned tensor gradients, exact selected-class loss derivatives, source
RoI geometry, target threshold differences, sampled quotas/empty truth/index
mapping, class-wise decoding and RPN matching/targets/clipping/filter/NMS
budgets pass. A native graph check proves head loss reaches the backbone but
not the RPN-specific leaves through detached proposals. All43 now-prettified
public examples in the five detection headers compile/run. Three original
C++23 programs pass syntax; legacy model/loss entry points forward to the core.
Both export formats pass; long Fast/Faster legends were shortened and their
actual training blocks rerun before final visual review.

Combined CNN coverage is1079/1079 across37 headers, with185 native tests.
Evidence: `build/cnn-two-stage-validation/`, `build/cnn-two-stage-export/`,
`build/cnn-two-stage-coverage.json`, `build/cnn-two-stage-hover.json`, native
logs and `build/cnn-detection-review/`. There are70 reports remaining; the
remaining eleven detection reports begin with feature pyramids.


### Tensor scene data and detection baseline completed (56/129 overall)

`detection_baselines` adds five executed C++ blocks, one editable/executed TikZ
block and seven reviewed figures (four SVG, three original-data PNG galleries).
Pixel-centre crops have hand-checkable ramp values. Real MNIST digits are
composited through tensor Float64 sampling/quantization/maximum operations;
tight boxes, jitter crops, backgrounds, window coordinates and component-box
geometry use tensor ops. Host RNG selects integer scene metadata and a host
queue traverses discrete component indices. Native tests compare independent
legacy pixels/annotations/proposals, crop values/eight box derivatives and
exact rendered outlines. Tensor patch RNG differences and possible one-byte
interpolation rounding differences are explicitly documented.

The visible full421771-scalar patch CNN fits972 patches from64 scenes using
372 Adam updates, twelve epochs, batch32, .001 learning rate and explicit
kernel L2. It selects epoch4 by NLL.857149 on198 patches from16 separate
validation scenes; accuracy is.772727. Later epochs overfit visibly. There
is no moving model state. On eight test scenes/23 digits, component proposals
use11.75 candidates/image versus454 windows; proposal recall at IoU.5/.7
is.869565/.695652 versus.782609/.391304. Inclusive source VOC AP50 is
.674392 versus.421627, AP75.345508 versus.037404, and COCO AP50:95.378197
versus.175300. Actual galleries retain merged/extra boxes. These are bounded
scene results using2048 training/512 test digit pools, not paper benchmarks.

New `nn/detection_scene_models.hpp` and `nn/detection_inference.hpp` provide
29 documented definitions and twelve compiled/executed public examples.
Thirteen new native tests, one new inclusive-synthetic-VOC test, five hovers,
independent header compilation and original baseline C++23 syntax pass.
The existing dataset AP API explicitly supports inclusive synthetic VOC
cutoffs while retaining strict official VOC defaults. Three-header affected
coverage is58/58; cumulative CNN coverage is1035/1035 across36 headers,
with170 accumulated native tests. Legacy patch forward calls the canonical
CNN. Markdown/Doxygen exports pass. Evidence:
`build/cnn-detection-scene-validation/`, `build/cnn-detection-scene-export/`,
`build/cnn-detection-scene-coverage.json`, `build/cnn-detection-scene-hover.json`
and `build/cnn-detection-review/`. There are73 reports remaining; R-CNN and
shared two-stage proposal/ROI helpers are next.


### Detection utilities and evaluation completed (55/129 overall)

`nms_matching` and `detection_evaluation` add eight executed C++ blocks, two
editable/executed TikZ diagrams and nine visually reviewed SVGs. Recorded
TensorFlow NMS selections, RPN/SSD target collisions, anchor geometry,1000
Float64 box-code round trips and zero-logit BCE gradients pass. The visible
initial/fresh-Adam loops verify paired parameters/moving-state restoration on
strict validation ties; the original complete warm-start fixture also passes.

Canonical `nn/detection_components.hpp` supplies pure tensor SSD matching,
shared RPN heads/losses and the baseline patch CNN. Canonical
`nn/detection_evaluation.hpp` supplies tensor IoU, cumulative precision/recall,
envelopes and VOC/COCO-style AP, with host metadata controlling only discrete
assignments. Difficult/crowd rules, regular-before-crowd preference, inclusive
versus strict cutoff, stable ties, cap100, ignored-category exclusion and
randomized independent-oracle comparisons pass. Source recall-grid floating
arithmetic is explicitly retained: the three-of-ten VOC fixture gives3/11
because3*.1 lies above3/10. This numerical boundary is explained, not hidden.

Both original C++23 programs build/run, all thirteen public header examples
compile/run, fourteen native tests and four clangd hovers pass, and45/45 new
header definitions are documented. Both reports export standalone Markdown
and Doxygen pages. Optional VOC/COCO prepared external predictions are absent;
the report says so and makes no real-dataset benchmark claim. Evidence:
`build/cnn-detection-validation/`, `build/cnn-detection-export/`,
`build/cnn-detection-coverage.json`, `build/cnn-detection-hover.json` and
`build/cnn-detection-review/`. There are74 reports remaining; shared tensor
scene generation and the detection baseline are next.


### Attention/modern group completed (53/129 overall)

All twelve reports in `1.5_attention_and_modern_convnets` are complete.
Spatial transformers and capsules add eleven executed C++ blocks, two retained
TikZ blocks, six reviewed SVGs and five original/distorted/learned PNG galleries.
Affine sampling checks identity/translation/rotation and all six derivatives;
localization initialization and full paired CNN gradients/counts pass. Distorted
MNIST generation is now entirely tensor based, including affine coefficients,
bilinear sampling and quantization. Its RNG sequence differs explicitly from
legacy host draws. Both full paired fits show320 Adam updates each, with honest
early-checkpoint overfitting and actual selected test warps, without pose claims.

Capsules check regularized squash, stable host routing, through-iteration finite
differences and every full model/decoder gradient. The visible optimizer fits
MARGIN plus .0005 summed-pixel reconstruction and explicit L2; normalized-length
NLL only monitors/selects. Decoder masks use true labels while fitting and
predicted labels for held-out reconstructions. The selected full8215568-scalar
model scores50/64 test digits; blurry similar reconstructions remain visible.
No calibration, high-quality reconstruction or paper benchmark is claimed.

`nn/spatial_transformer_models.hpp` and `nn/capsule_models.hpp` have32/32
documented definitions, fourteen compiled/executed examples, four passing hovers
and eleven native tests. Both original C++23 sources and exports pass. Combined
CNN coverage is961/961 across thirty-two completed headers, with142 accumulated
native tests. Evidence: `build/cnn-stn-capsule-validation/`,
`build/cnn-stn-capsule-export/`, `build/cnn-stn-capsule-coverage.json`,
`build/cnn-stn-hover.json`, `build/cnn-capsule-hover.json` and isolated XML.
There are76 reports remaining. The fundamentals, classic, residual/dense,
efficient/mobile and attention/modern groups are complete; object detection is next.


### Non-local and octave reports completed (51/129 overall)

The two reports execute eleven C++ and two retained TikZ blocks, with eight
reviewed SVGs and two original-CIFAR PNGs. Non-local attention checks all four
affinity rules with/without pooling against dense tensor values/derivatives,
identity initialization and active global support. Its input-before-projection
pooling convention, non-probabilistic dot/concatenation rows and legitimate
bias-shift invariances are explicit. The complete five-module compact fit shows
320 Adam updates and paired learned/BN validation checkpoints.

Octave convolution checks independent four-path host oracles for both source
strides, per-kernel finite differences and exact arithmetic/storage ratios.
A real geometry bug is fixed: a stride-two high-only merge now reads its low
input directly, avoiding the final shortcut's pool/upsample mismatch at odd7x7
resolution. The changed operation and corrected LH MACs are explicit. A real
224-pixel model forward, all five full-width split counts and complete compact
gradients pass. Its visible Adam loop retains every bottleneck/path. Stride-two
pre-pooling is distinguished from ordinary strided convolution; nominal target
budgets are distinguished from count-forward input sizes. Both reports state
modest held-out results without accuracy or runtime rankings.

`nn/nonlocal_models.hpp` and `nn/octave_models.hpp` have 36/36 documented
definitions, twelve compiled/executed examples, four passing hovers and eleven
native tests. Both original C++23 sources and both exports pass. Combined CNN
coverage is 929/929 across thirty completed headers, with 131 accumulated
native tests. Evidence: `build/cnn-nonlocal-validation/`,
`build/cnn-nonlocal-export/`, `build/cnn-nonlocal-coverage.json`,
`build/cnn-nonlocal-hover.json`, corresponding `cnn-octave-*` artifacts and
isolated XML in `build/cnn-docs-audit/xml`. There are 78 reports remaining;
spatial transformers and capsules are the final two in the modern group.


### RepVGG completed (49/129 overall)

Six C++ blocks and retained TikZ execute with three reviewed SVGs and an
original-CIFAR PNG. Production fusion now uses tensor BN folding, centre
padding and grouped one-hot identity kernels, preserving Float64 and every
source input/filter/BN-affine derivative in four fixtures. Complete A0 fusion
and all ten actual full-weight training/deployment counts pass. Counts use
32-pixel forwards; exact 224-pixel MACs scale convolution area by 49 with a
real A0 224-pixel cross-check. Large full-resolution timings are not claimed.
The full 22-block CIFAR A0 fit shows 320 Adam updates, explicit L2 and paired
validation checkpoints. Its selected TRAINED checkpoint is then fused: relative
maximum test-logit error 1.66e-7, zero of 64 prediction disagreements. Poor
14.0625% held-out accuracy is explicit. A four-panel figure retains the initial
BN transient, shows later NLL detail, and compares original/fused test scores.

`nn/repvgg_models.hpp` has 35/35 documented definitions, thirteen
compiled/executed examples, two passing hovers and four native tests. Tests
cover Float64 all-source derivatives, independent classifier storage, complete
A0 geometry/counts, frozen BN gradients and invalid modes/groups/missing state.
Original C++23 syntax and both exports pass. Combined completed CNN coverage
is 893/893 across twenty-eight headers, with 120 accumulated native tests.
Evidence: `build/cnn-repvgg-validation/`, `build/cnn-repvgg-export/`,
`build/cnn-repvgg-coverage.json`, `build/cnn-repvgg-hover.json`.
The documentation audit now uses independent `build/cnn-docs-audit/xml` output
to avoid concurrent shared-output rebuilds; regenerate with the saved Doxyfile.
There are 80 reports remaining.


### ConvNeXt V1/V2 completed (48/129 overall)

The two reports execute ten C++ blocks and two retained TikZ blocks, with six
reviewed SVGs and two original-CIFAR PNGs. V1 shows tiny layer scale, magnified
corrections, exact zero-scale identity derivatives and 7x7 branch support. V2
checks active GRN, approximate scaling invariance and finite identity gradients
for zero/dead channels. Every released V1/V2 table count is analytical, using
checked integer parameter arithmetic; actual full-weight Atto/Tiny counts and
Tiny 224-pixel MACs cross-check the formulas. Other large models, particularly
Huge, are explicitly not allocated. Full-width Tiny backward retains every
block. Both complete Atto fits expose 320 Adam updates, branch dropout, L2 and
validation checkpoints with no unnecessary BN state. Modest test results do not
establish a V1/V2 ranking; FCMAE pretraining is not claimed.

`nn/convnext_models.hpp` has 27/27 documented definitions, twelve compiled
and executed examples, two hovers and six native tests. Tests cover all table
counts, actual Atto leaves, active/zero GRN derivatives, layer-scale identity,
spatial support, full CIFAR gradients, rectangular MACs and overflow rejection.
Both original C++23 sources and exports pass. Combined CNN coverage is 858/858
across twenty-seven completed headers, with 116 accumulated native tests.
Evidence: `build/cnn-convnext-validation/`, `build/cnn-convnext-export/`,
`build/cnn-convnext-coverage.json`, `build/cnn-convnext-hover.json`,
`build/cnn-combined-coverage.json`. There are 81 reports remaining.


### Attention and deformable batch completed (46/129 overall)

SE, CBAM, ECA, selective kernels and deformable v1/v2 now have reports.
Twenty-three C++ blocks and five retained TikZ blocks executed, with seventeen
reviewed SVGs and five original-CIFAR PNGs. Small fixtures expose shared CBAM
MLPs/nontrivial spatial BN, local ECA filtering, normalized SK branch mixtures,
3x3 versus dilated 5x5 derivative supports and bilinear dy/dx sampling with zero
extension. Float64 SE/CBAM native comparisons include every input/parameter
derivative; deformable sampling checks an independent oracle and all eighteen
offset-bias finite differences away from integer boundaries. SK softmax weights
can underflow to zero; its illustrated support fixture avoids saturation.

All full attention counts and sixteen-block backward probes run at bounded
64-pixel geometry. Quarter-width 32-pixel fits show 320 Adam updates, explicit
L2 and paired learned/BN checkpoints; v1/v2 deformable models fit separately.
Modest held-out results are stated without accuracy rankings. Fixed an actual
shared-backbone bug: grouped spatial filters and replacement hooks now receive
the actual conv2 stride, so choosing first-1x1 stride does not downsample twice.
Both stride placements have native gradient/geometry tests. SK fitting retains
the source's spatial-filter stride placement.

`nn/attention_cnn_models.hpp` and `nn/deformable_models.hpp` have 55/55
documented definitions, seventeen compiled/executed examples, three passing
hovers and eleven native tests. All five original C++23 sources and both
exports pass. Combined CNN coverage is 831/831 across twenty-six completed
headers; 110 accumulated native tests pass. Evidence:
`build/cnn-attention-validation/`, `build/cnn-attention-export/`,
`build/cnn-attention-coverage.json`, `build/cnn-attention-hover.json`,
`build/cnn-combined-coverage.json`. There are 83 reports remaining.


### Efficient/mobile group completed (41/129 overall)

All eleven reports in `1.4_efficient_and_mobile_architectures` are complete.
MnasNet, EfficientNet and EfficientNetV2 add sixteen executed C++ blocks and
three editable TikZ blocks, with ten reviewed SVGs and three original-CIFAR
PNGs. Block strings round-trip; SE/linear projection and fused/depthwise
arithmetic are visible. Per-row dropout checks every input derivative and
zero/one endpoints. Five MnasNet budgets use actual 224-pixel forwards. All
eight EfficientNet and seven V2 full-weight models run at 64 pixels for counts;
exact reference-resolution SAME arithmetic uses ceil geometry, with actual
224-pixel B0 forwards as cross-checks. Larger reference forwards are not claimed.
Complete B0/A1 backward probes report finite gradients and legitimate BN-shift
cancellation. Every fit shows 320 Adam updates, L2, augmentations and paired
parameter/BN checkpoints. Modest held-out results are stated without benchmarks.

`nn/mbconv_models.hpp` has 64/64 documented definitions, twenty-one compiled
and executed examples, two passing hovers and five native tests (parsing,
endpoint/row derivative masks, SE finite differences, fused budgets, complete
CIFAR gradients/state). All three original sources and both exports pass.
Combined completed CNN coverage is 776/776 across twenty-four headers, with
99 accumulated native tests. Evidence: `build/cnn-mbconv-validation/`,
`build/cnn-mbconv-export/`, `build/cnn-mbconv-coverage.json`,
`build/cnn-mbconv-hover.json`, `build/cnn-combined-coverage.json`.
There are 88 reports remaining; attention and modern convolutions are next.


### CondenseNet completed (38/129 overall)

The report executes five C++ blocks and retained TikZ, with four reviewed SVGs
and an original-CIFAR PNG. Tensor importance/ranking and mask updates replace
host numerical pruning; tensor gathers pack kernels without Float32 coercion.
An independent oracle checks every pruning stage; stable ties and skipped-stage
catch-up are explicit. Float64 packing preserves values and all input/filter/BN
derivatives after gathering BN state and restoring interleaved output order.
The full C86 fit exposes Adam, group lasso, L2 and next-forward mask timing.
All parameters, BN moments, masks, stages and step are checkpointed together.
The 320-update run selects epoch 18 (34.375% on 64 test images); overfitting is
reported. Analytical converted budgets explicitly assume classifier selection;
no fitted full-network conversion or mobile speedup is claimed.

`nn/condensenet_models.hpp` has 47/47 documented definitions, thirteen
compiled/executed examples, two passing clangd hovers and five native tests.
The original C++23 source and both exports pass. Combined coverage is 712/712
across twenty-three completed CNN headers, with 94 accumulated native tests.
Evidence: `build/cnn-condense-validation/`, `build/cnn-condense-export/`,
`build/cnn-condense-coverage.json`, `build/cnn-condense-hover.json`.
There are 91 reports remaining; MnasNet/EfficientNet/V2 reports are executing.


### RegNet completed (37/129 overall)

The report executes six C++ blocks and an editable TikZ block, with four
reviewed SVGs and an original-CIFAR PNG contact sheet. It shows width generation,
packed/split grouped values and all derivatives, fourteen full parameter/stage
budgets and exact target-resolution MACs. Count-only large models use 64 pixels;
X/Y-400MF also execute 224-pixel forwards to cross-check the budget conversion.
Complete Y-400MF backward checks retain all sixteen blocks. Its visible
320-update Adam fit restores paired learned/moving checkpoints and explicitly
reports poor test generalization.

`nn/regnet_models.hpp` has 25/25 documented definitions, eight compiled and
executed examples, two passing clangd hovers and five native tests. Tests cover
all fourteen designs under changed rounding modes, restoration on width-overflow
exceptions, input-based SE, post-add ReLU, released count/MACs, full CIFAR
gradients/state and invalid settings. Combined coverage is 665/665 across
twenty-two CNN headers; 89 accumulated native tests pass. The original source
compiles and Markdown/Doxygen exports pass. Evidence:
`build/cnn-regnet-validation/`, `build/cnn-regnet-export/`,
`build/cnn-regnet-coverage.json`, `build/cnn-regnet-hover.json` and
`build/cnn-combined-coverage.json`. There are 92 reports remaining; CondenseNet,
MnasNet and EfficientNet variants remain in the efficient/mobile group.

### GhostNet completed (36/129 overall)

The report executes five C++ blocks and an editable TikZ block, with four
reviewed SVGs and an original-CIFAR PNG contact sheet. A controlled odd-width
module checks intrinsic/cheap maps, trimming, exact input derivatives and zero
data gradients for discarded generated channels. All three full width budgets
and measured-shape MACs pass. Complete full-width backward checks separate
thirteen redundant BN shifts from nonredundant parameter derivatives. The
visible 320-update Adam fit saves/restores paired learned/moving-state values
and reports its poor test generalization explicitly.

`nn/ghostnet_models.hpp` has 24/24 documented definitions, nine compiled and
executed examples, two passing clangd hovers and five native tests. The tests
also cover ratio-three channel order and signed zero-branch identities.
Combined coverage is 640/640 across twenty-one CNN headers; 84 accumulated
native tests pass. The original source compiles and Markdown/Doxygen exports
include all figures. Evidence: `build/cnn-ghost-validation/`,
`build/cnn-ghost-export/`, `build/cnn-ghost-coverage.json`,
`build/cnn-ghost-hover.json` and `build/cnn-combined-coverage.json`.
There are 93 reports remaining; RegNet and the remaining efficient models are next.

### ShuffleNet completed (35/129 overall)

The report executes five C++ blocks and an editable TikZ block, with four
reviewed SVGs and an original-CIFAR PNG contact sheet. It checks release/paper
permutations, weighted permutation derivatives, controlled grouped-filter
connectivity, exact v2 even-channel carry derivatives and v1 padded-average
shortcuts. All twelve released count/MAC presets pass. Complete v1/v2 gradient
probes separately check redundant depthwise BN shifts. Two visible 320-update
Adam fits use validation-selected paired learned/moving-state checkpoints.

`nn/shufflenet_models.hpp` has 32/32 documented definitions, thirteen compiled
and executed examples, two passing clangd hovers and six native tests. Combined
coverage is 616/616 across twenty CNN headers; 79 accumulated native tests pass.
The original source compiles and Markdown/Doxygen exports include all figures.
Evidence: `build/cnn-shuffle-validation/`, `build/cnn-shuffle-export/`,
`build/cnn-shuffle-coverage.json`, `build/cnn-shuffle-hover.json` and
`build/cnn-combined-coverage.json`. There are 94 reports remaining; GhostNet
and the remaining efficient/mobile architectures are next.

### SqueezeNet completed (34/129 overall)

The report executes five C++ blocks and an editable TikZ block, with four
reviewed SVGs and an original-CIFAR PNG contact sheet. It verifies Fire expand
centre embedding and every input/parameter derivative, negative-input ceil-pool
values/switches, both full released geometries/counts and optional bypasses.
Complete backward probes keep all widths/modules. Three visible Adam(.0001)
fits perform 320 updates each, explain the ReLU-head learning-rate choice and
report poor small-data generalization without an architecture-ranking claim.

`nn/squeezenet_models.hpp` has 22/22 documented definitions, six compiled and
executed public examples, two passing clangd hovers and five native tests.
Combined coverage is 584/584 across nineteen CNN headers; 73 accumulated native
tests pass. The original source compiles and Markdown/Doxygen exports include
all figures. Evidence: `build/cnn-squeezenet-validation/`,
`build/cnn-squeezenet-export/`, `build/cnn-squeezenet-coverage.json`,
`build/cnn-squeezenet-hover.json` and `build/cnn-combined-coverage.json`.
There are 95 reports remaining; ShuffleNet is next.

### MobileNet v1/v2/v3 completed (33/129 overall)

The three reports execute seventeen C++ blocks and three editable TikZ blocks,
with twelve reviewed SVGs and three original-CIFAR PNG contact sheets. V1 checks
linear factorization and all input/filter derivatives, seven width/resolution
budgets, the dense baseline and pointwise cost shares. V2 shows active-coordinate
spiral recovery, exact signed bottleneck values/derivatives and nine full budgets.
V3 plots activation formulas, checks gate values/derivatives and both searched
tables. Complete full-width backward probes retain every model block; redundant
projection BN shifts are checked separately. Five visible 320-update Adam fits
use paired learned/moving checkpoints and report their small-data limitations.

`nn/mobilenet_models.hpp` has 70/70 documented definitions, twenty-three public
examples compiled/executed, four passing clangd hovers and seven native tests.
The tests include the input-dependent SE gate derivative, exact full counts/MACs,
complete CIFAR gradients/state semantics and invalid dimensions/metadata.
Combined coverage is 562/562 across eighteen CNN headers; 68 accumulated native
tests pass. All three original sources compile and Markdown/Doxygen exports
include PNGs. Evidence: `build/cnn-mobilenet-validation/`,
`build/cnn-mobilenet-export/`, `build/cnn-mobilenet-coverage.json`,
`build/cnn-mobilenet-hover.json` and `build/cnn-combined-coverage.json`.
There are 96 reports remaining; SqueezeNet and ShuffleNet are next.

### RevNet completed; residual/dense family complete (30/129 overall)

The report executes five C++ blocks and an editable TikZ block, with four
reviewed SVGs and an original-CIFAR PNG contact sheet. Nontrivial basic and
bottleneck units invert in training/inference BN modes. All five released
preset counts pass. Complete RevNet-38/164 gradients compare ordinary,
stored-input and reconstructed-input backward, reporting float32 drift.
Measured logical tensor peaks compare all three modes at 3/9/18 units per
stage; the deepest fixture uses 31.308 MiB reversible versus 253.095 MiB
ordinary, with its exact baseline and exclusions stated. The visible 320-step
Adam loop explicitly uses reversible backward and paired validation checkpoints.

`nn/reversible_residual_models.hpp` has 50/50 documented definitions, nineteen
compiled/executed public examples, two passing clangd hovers and five native
tests, including BN updates once rather than again during recomputation.
Combined CNN coverage is 492/492 across seventeen headers; 61 accumulated native
tests pass. The original source compiles and Markdown/Doxygen exports pass.
Evidence: `build/cnn-revnet-validation/`, `build/cnn-revnet-export/`,
`build/cnn-revnet-coverage.json`, `build/cnn-revnet-hover.json` and
`build/cnn-combined-coverage.json`. All twelve residual/dense examples are
complete. There are 99 reports remaining; efficient/mobile architectures are next.

### Stochastic depth completed (29/129 overall)

The report executes four C++ blocks and an editable TikZ block, with three
reviewed SVGs and an original-CIFAR PNG contact sheet. It checks original and
inverted scaling, whole-batch/per-row gates, both probability endpoints,
empirical survival and exact input/weight derivatives. Schedule/shortcut checks
and a complete 110-layer backward probe precede two visible 320-update CIFAR-20
Adam fits. It explicitly distinguishes mask-active depth from executed work:
every branch is computed and its BN updated even when masked.

`nn/stochastic_residual_models.hpp` has 25/25 documented definitions, nine
compiled/executed examples, two passing clangd hovers and five native tests.
Combined CNN coverage is 442/442 across sixteen headers; 56 accumulated native
tests pass. The original source compiles, and Markdown/Doxygen exports pass.
Evidence: `build/cnn-stochastic-validation/`, `build/cnn-stochastic-export/`,
`build/cnn-stochastic-coverage.json`, `build/cnn-stochastic-hover.json` and
`build/cnn-combined-coverage.json`. There are 100 reports remaining; RevNet is next.

### PyramidNet and Res2Net completed (28/129 overall)

Both reports execute eight C++ blocks and two editable TikZ blocks, with seven
reviewed SVGs and two original-CIFAR PNG contact sheets. PyramidNet verifies
signed pooled/zero-padded shortcuts and weighted input derivatives, all ten
original parameter budgets, and complete full-width 272-layer gradients with
181 redundant BN shifts checked separately. Res2Net checks a scalar spatial
oracle, exact support/mass for ordinary/stage/strided/scale-one paths, full
ImageNet geometry/counts and complete compact gradients. Binary support plots
make the 3/5/7/1 receptive fields readable. Visible full-architecture Adam loops
perform 320 updates each and restore paired learned/moving-state checkpoints.

`nn/pyramidal_multiscale_models.hpp` has 37/37 documented definitions, thirteen
compiled/executed examples, four passing clangd hovers and six native tests.
Combined coverage is 417/417 across fifteen headers; 51 accumulated CNN native
tests pass. Both original sources compile. Markdown/Doxygen exports preserve
all local figures. Evidence: `build/cnn-pyramid-scale-validation/`,
`build/cnn-pyramid-scale-export/`, `build/cnn-pyramid-scale-coverage.json`,
`build/cnn-pyramid-scale-hover.json`, `build/cnn-combined-coverage.json`.
There are 101 reports remaining; stochastic depth is executing and RevNet is next.

### ResNeXt and Wide ResNet completed (26/129 overall)

Both reports execute eight C++ blocks and two editable TikZ blocks, with eight
reviewed SVGs and two original-CIFAR PNG contact sheets. ResNeXt checks grouped
versus split-transform-sum values and all input/parameter derivatives at strides
one/two, full ImageNet count/geometry and complete compact CIFAR-29 gradients.
Wide ResNet checks dropout values/derivatives, fresh masks, deterministic
inference, signed identity, three full model counts and width/MAC budgets.
Visible Adam loops perform 320 updates per fit, with three controlled WRN fits;
paired parameter/BN checkpoints precede test evaluation. Poor generalization is
reported explicitly and does not establish an architecture ranking.

`nn/grouped_wide_residual_models.hpp` has 40/40 documented definitions, eleven
compiled/executed examples, four clangd hovers and six native tests. Combined
coverage is 380/380 across fourteen headers; all 45 accumulated CNN native tests
pass. Both original sources compile through aliases. Markdown/Doxygen exports
include PNGs. Evidence: `build/cnn-grouped-wide-validation/`,
`build/cnn-grouped-wide-export/`, `build/cnn-grouped-wide-coverage.json`,
`build/cnn-grouped-wide-hover.json` and `build/cnn-combined-coverage.json`.
There are 103 reports remaining; PyramidNet and Res2Net are next.

### Dual Path Networks completed (24/129 overall)

The report executes five C++ blocks and an editable TikZ block, with four
reviewed SVGs and an original-CIFAR PNG contact sheet. It checks residual
identity, ordered dense preservation, stage projections and input derivatives,
all five full parameter budgets, complete DPN-92 stage geometry and every
compact-model parameter gradient. The visible Adam loop performs 320 updates;
the selected checkpoint scores 20.3% on 64 official-test images.

`nn/dual_path_models.hpp` has 40/40 documented definitions, thirteen compiled
and executed examples, two passing clangd hovers and four passing native tests.
Combined coverage is 340/340 across thirteen headers. The original source
compiles and Markdown/Doxygen exports preserve all figures, including PNG.
Evidence: `build/cnn-dpn-validation/`, `build/cnn-dpn-export/`,
`build/cnn-dpn-coverage.json`, `build/cnn-dpn-hover.json` and
`build/cnn-combined-coverage.json`. There are 105 reports remaining; ResNeXt
and Wide ResNet are next.

### Highway Networks completed (23/129 overall)

The report executes five C++ blocks and an editable TikZ block, with four
reviewed SVGs and an original-MNIST PNG contact sheet. It checks the full
input-dependent gate Jacobian, dense/pointwise-convolution equality, exact
zero/one gate limits, hundred-layer scalar attenuation and the complete
136660-parameter classifier. The visible Adam loop makes 160 updates on 512
digits and selects a checkpoint before a 128-digit official-test evaluation.

`nn/highway_models.hpp` has 13/13 documented definitions, six compiled/executed
examples, two clangd hovers and four focused native tests. All 35 CNN tests pass;
the original Highway source compiles. Combined coverage is 300/300 across
twelve CNN headers. Markdown/Doxygen exports include the PNG. Evidence:
`build/cnn-highway-validation/`, `build/cnn-highway-export/`,
`build/cnn-highway-coverage.json`, `build/cnn-highway-hover.json` and
`build/cnn-combined-coverage.json`. There are 106 reports remaining; DPN is next.

### Pre-activation and DenseNet completed (22/129 overall)

Both reports execute ten C++ blocks and two editable TikZ blocks, with seven
reviewed SVGs and two tensor-generated PNG contact sheets. Worked diagnostics
cover fifty signed identity units, preactivated projections, nonzero-branch
finite differences, dense feature order/direct derivatives, and controlled
compression/pooling values and derivatives. Full reference forward counts/shapes
and complete-depth compact gradients pass. Both explicit 20-epoch Adam loops
save/restore parameter and moving-state checkpoints and show test count matrices.
The unsuccessful short preactivation fit is reported directly; the DenseNet
fit's generalisation gap is also retained.

`nn/preactivation_densenet_models.hpp` has 48/48 documented definitions, 17
compiled/executed examples and four clangd hovers. Five added native tests pass
(31 CNN tests overall); all twelve original residual/dense sources still compile.
Combined coverage is 287/287 across eleven CNN headers. Standalone Markdown and
Doxygen exports include the PNG assets. Evidence:
`build/cnn-preact-dense-validation/`, `build/cnn-preact-dense-export/`,
`build/cnn-preact-dense-coverage.json`, `build/cnn-preact-dense-hover.json` and
`build/cnn-combined-coverage.json`.
There are 107 reports remaining. Highway layers are next being prepared; the
remaining residual/dense reports include DPN, PyramidNet, Res2Net, ResNeXt,
RevNet, stochastic depth and Wide ResNet.

### Basic and bottleneck ResNets completed (20/129 overall)

Both reports execute eight C++ blocks and two editable TikZ blocks, with eight
reviewed local figures. Controlled fixtures check signed post-add gates,
projection shortcut values/derivatives and bottleneck stride placement. Full
18/34/50/101/152 forward geometries and counts pass; complete compact graphs
check every parameter gradient. Five separate 20-epoch quarter-width Adam fits
show losses, augmentation, L2 and paired parameter/BN-state checkpoints explicitly.
Small-subset scores and differences between training/inference moments are
explained without claiming a depth ranking.

`nn/resnet_models.hpp` has 32/32 documented definitions, ten compiled/executed
examples and two clangd hovers. Four added tests pass (26 CNN native tests in
all). All twelve original residual/dense-family sources still compile through
the compatibility header. Combined coverage is 239/239 across ten CNN headers.
Markdown and Doxygen exports pass. Evidence: `build/cnn-resnet-validation/`,
`build/cnn-resnet-export/`, `build/cnn-resnet-coverage.json`,
`build/cnn-resnet-hover.json` and `build/cnn-combined-coverage.json`.
There are 109 reports remaining; continue with DenseNet and pre-activation
components, then the remaining residual/dense family members.

### All classic architectures completed (18/129 overall)

All eight reports in `1.2_classic_architectures` are complete. The two Inception
reports execute nine C++ blocks plus two editable TikZ blocks and retain eight
reviewed local SVGs. They instantiate full-width v2/v3/v4/Inception-ResNet-v2
forward stages/counts and test every parameter gradient on complete compact
topologies. Worked checks cover linear factorisation and factor gradients,
beta-only BN/EMA moments, explicit smoothing updates, and A/B/C residual value,
parameter-gradient and identity-gradient scaling. Dataset blocks expose four
20-epoch Adam fits, smoothing/auxiliary/L2 terms and paired parameter/state
checkpoints. Small-subset failures and generalisation gaps are reported directly.

`nn/inception_models.hpp` has 47/47 documented definitions, 26 compiled/executed
examples and four clangd hovers. All nine new/earlier CNN component headers now
pass the combined 207/207 definition check. Twenty-two native CNN tests pass;
all affected original classic sources compile. Both new reports export to
standalone Markdown and Doxygen. Evidence: `build/cnn-inception-validation/`,
`build/cnn-inception-export/`, `build/cnn-inception-coverage.json`,
`build/cnn-inception-hover.json` and `build/cnn-combined-coverage.json`.
There are 111 reports remaining. Continue with `1.3_residual_and_dense_families`;
ResNet basic/bottleneck are the next shared model components being prepared.

### ZFNet and Xception completed (16/129 overall)

Nine C++ blocks and two editable TikZ blocks execute, with nine reviewed SVGs,
including original RGB previews, 20-epoch explicit Adam traces and a restored
ZFNet conv5 diagnostic comparing deconvnet with an ordinary input gradient.
Full reference forward stage/count checks pass; compact models retain every
block. Xception explicitly saves/restores both learned parameters and BN state.
Nine new architecture tests pass across the two latest batches. The new
`vision_architecture_components.hpp` and `zfnet_xception.hpp` have 43/43 documented
definitions, 16 compiled/executed examples and four clangd hovers. Both original
sources still compile. Markdown and Doxygen exports pass. Evidence:
`build/cnn-zf-xception-validation/`, `build/cnn-zf-xception-export/`,
`build/cnn-zf-xception-coverage.json` and `build/cnn-zf-xception-hover.json`.
There are 113 reports remaining. Inception v2/v3 and v4/ResNet are in progress.

### AlexNet and GoogLeNet completed (14/129 overall)

Both reports execute four C++ blocks and one editable TikZ block, with original
CIFAR-10 image sheets and eight reviewed local SVGs. They instantiate full-width
forward geometry/count checks and keep compact 20-epoch Adam loops visible.
AlexNet checks split/fused grouped values and every derivative, disconnected
channels, LRN finite differences and dropout modes. GoogLeNet checks boundary
branch arithmetic, reduction MACs, auxiliary gradients and main-head independence.
Five new native tests pass. The two canonical headers have 54/54 documented
definitions, 23 compiled/executed examples and four clangd hovers; both original
sources pass C++23 syntax checks. Markdown and Doxygen exports pass.
Evidence: `build/cnn-alex-google-validation/`, `build/cnn-alex-google-export/`,
`build/cnn-alex-google-coverage.json` and `build/cnn-alex-google-hover.json`.
There are 115 reports remaining; next unchecked entries are Inception v2/v3 and
Inception v4/ResNet. ZFNet and Xception are being prepared in the same group.

### NiN and VGG completed (12/129 overall)

Network in Network and VGG are complete. Their reports include original RGB
previews, explicit 30-epoch compact-model Adam loops, independent patch-MLP
parameter gradients, all VGG depth schedules and dense/sliding-head equality.
NiN verifies the full 966986-parameter topology. VGG full counts are explicitly
shape arithmetic; every compact model is instantiated. Small-subset scores
are reported without claiming a benchmark or architecture ranking.

`nn/classic_cnn_models.hpp` owns these architectures, with two legacy forwarding
headers. Ten focused native tests now pass across both CNN batches. The classic
header has 31/31 documented definitions, ten compiled examples and two clangd
hovers; both affected original sources pass C++23 syntax checks. Two more reports
and eight local figures export successfully. Validation: `build/cnn-classic-validation/`,
`build/cnn-classic-export/`, `build/cnn-classic-coverage.json` and
`build/cnn-classic-hover.json`.

### CNN fundamentals completed (10/129 overall)

All ten `1.1_fundamentals` reports execute their tensor computations and explicit
Adam loops where fitting applies. They retain ten editable TikZ diagrams and
26 reviewed local SVGs, including original cached MNIST/CIFAR image previews,
patch/overlap maps, dilation support, receptive fields and checkerboards.
The short image training runs identify their fixed subsets and avoid full-data
benchmark claims. BN comparisons pair kernels and row orders and restore both
parameters and moving state; no requirement that BN must win is imposed.

Canonical `nn/convolution_components.hpp` adds differentiable patch extraction,
zero-expanded dilation and separable-kernel composition. `nn/cnn_models.hpp`
owns the shared MNIST CNN and sparse 60000-parameter LeNet-5; their four relevant
legacy common headers forward to canonical operations. `nn/vision_plots.hpp`
provides matrix/image previews. Seven native tests, 15 compiled header examples,
34/34 documented definitions and two representative clangd hovers pass. Four
original C++ programs also pass C++23 syntax checks. All ten reports export to
nested Doxygen pages; validation artifacts live in `build/cnn-foundations-validation/`.



The next scope is all **129 C++ examples** under
`examples/cpp/source/nn/1_convolutional_neural_networks`, with the same relative
layout beneath `docs/mepml/examples/nn/1_convolutional_neural_networks`.

- Format every executable code block with ordinary indentation and line breaks;
  put declarations, control flow and separate statements on readable lines.
- Show representative input images for image tasks, labelled where useful, and
  plot generated signals, geometry or volumes when those are the actual inputs.
  Describe sample selection and distinguish fixtures, short demonstrations and
  full benchmark runs. Use the real locally cached MNIST/CIFAR data when present.
- Execute each block, inspect figures, verify exports and tick entries only after
  the corresponding report passes. Original runnable sources remain available.

- [x] 1_convolutional_neural_networks/1.1_fundamentals/batch_norm_ablation
- [x] 1_convolutional_neural_networks/1.1_fundamentals/cnn_cifar10
- [x] 1_convolutional_neural_networks/1.1_fundamentals/cnn_mnist
- [x] 1_convolutional_neural_networks/1.1_fundamentals/convolution_im2col
- [x] 1_convolutional_neural_networks/1.1_fundamentals/depthwise_separable
- [x] 1_convolutional_neural_networks/1.1_fundamentals/dilated_convolutions
- [x] 1_convolutional_neural_networks/1.1_fundamentals/global_pooling_vs_flatten
- [x] 1_convolutional_neural_networks/1.1_fundamentals/lenet5
- [x] 1_convolutional_neural_networks/1.1_fundamentals/receptive_fields
- [x] 1_convolutional_neural_networks/1.1_fundamentals/transposed_convolutions
- [x] 1_convolutional_neural_networks/1.2_classic_architectures/alexnet
- [x] 1_convolutional_neural_networks/1.2_classic_architectures/googlenet
- [x] 1_convolutional_neural_networks/1.2_classic_architectures/inception_v2_v3
- [x] 1_convolutional_neural_networks/1.2_classic_architectures/inception_v4_resnet
- [x] 1_convolutional_neural_networks/1.2_classic_architectures/network_in_network
- [x] 1_convolutional_neural_networks/1.2_classic_architectures/vgg
- [x] 1_convolutional_neural_networks/1.2_classic_architectures/xception
- [x] 1_convolutional_neural_networks/1.2_classic_architectures/zfnet
- [x] 1_convolutional_neural_networks/1.3_residual_and_dense_families/densenet
- [x] 1_convolutional_neural_networks/1.3_residual_and_dense_families/dpn
- [x] 1_convolutional_neural_networks/1.3_residual_and_dense_families/highway_networks
- [x] 1_convolutional_neural_networks/1.3_residual_and_dense_families/preactivation_resnet
- [x] 1_convolutional_neural_networks/1.3_residual_and_dense_families/pyramidnet
- [x] 1_convolutional_neural_networks/1.3_residual_and_dense_families/res2net
- [x] 1_convolutional_neural_networks/1.3_residual_and_dense_families/resnet_basic
- [x] 1_convolutional_neural_networks/1.3_residual_and_dense_families/resnet_bottleneck
- [x] 1_convolutional_neural_networks/1.3_residual_and_dense_families/resnext
- [x] 1_convolutional_neural_networks/1.3_residual_and_dense_families/revnet
- [x] 1_convolutional_neural_networks/1.3_residual_and_dense_families/stochastic_depth
- [x] 1_convolutional_neural_networks/1.3_residual_and_dense_families/wide_resnet
- [x] 1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/condensenet
- [x] 1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/efficientnet
- [x] 1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/efficientnet_v2
- [x] 1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/ghostnet
- [x] 1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mnasnet
- [x] 1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mobilenet_v1
- [x] 1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mobilenet_v2
- [x] 1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mobilenet_v3
- [x] 1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/regnet
- [x] 1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/shufflenet
- [x] 1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/squeezenet
- [x] 1_convolutional_neural_networks/1.5_attention_and_modern_convnets/capsule_networks
- [x] 1_convolutional_neural_networks/1.5_attention_and_modern_convnets/cbam
- [x] 1_convolutional_neural_networks/1.5_attention_and_modern_convnets/convnext
- [x] 1_convolutional_neural_networks/1.5_attention_and_modern_convnets/convnext_v2
- [x] 1_convolutional_neural_networks/1.5_attention_and_modern_convnets/deformable_conv
- [x] 1_convolutional_neural_networks/1.5_attention_and_modern_convnets/eca_net
- [x] 1_convolutional_neural_networks/1.5_attention_and_modern_convnets/non_local
- [x] 1_convolutional_neural_networks/1.5_attention_and_modern_convnets/octconv
- [x] 1_convolutional_neural_networks/1.5_attention_and_modern_convnets/repvgg
- [x] 1_convolutional_neural_networks/1.5_attention_and_modern_convnets/senet
- [x] 1_convolutional_neural_networks/1.5_attention_and_modern_convnets/sknet
- [x] 1_convolutional_neural_networks/1.5_attention_and_modern_convnets/spatial_transformer
- [x] 1_convolutional_neural_networks/1.6_object_detection/centernet
- [x] 1_convolutional_neural_networks/1.6_object_detection/detection_baselines
- [x] 1_convolutional_neural_networks/1.6_object_detection/detection_evaluation
- [x] 1_convolutional_neural_networks/1.6_object_detection/detr
- [x] 1_convolutional_neural_networks/1.6_object_detection/efficientdet
- [x] 1_convolutional_neural_networks/1.6_object_detection/fast_rcnn
- [x] 1_convolutional_neural_networks/1.6_object_detection/faster_rcnn
- [x] 1_convolutional_neural_networks/1.6_object_detection/fcos
- [x] 1_convolutional_neural_networks/1.6_object_detection/fpn
- [x] 1_convolutional_neural_networks/1.6_object_detection/nms_matching
- [x] 1_convolutional_neural_networks/1.6_object_detection/rcnn
- [x] 1_convolutional_neural_networks/1.6_object_detection/retinanet
- [x] 1_convolutional_neural_networks/1.6_object_detection/ssd
- [x] 1_convolutional_neural_networks/1.6_object_detection/yolo_v1
- [x] 1_convolutional_neural_networks/1.6_object_detection/yolo_v2
- [x] 1_convolutional_neural_networks/1.6_object_detection/yolo_v3
- [x] 1_convolutional_neural_networks/1.6_object_detection/yolo_v5_style
- [x] 1_convolutional_neural_networks/1.7_segmentation/deeplab_v1_v2
- [x] 1_convolutional_neural_networks/1.7_segmentation/deeplab_v3_plus
- [x] 1_convolutional_neural_networks/1.7_segmentation/fcn
- [x] 1_convolutional_neural_networks/1.7_segmentation/hrnet
- [x] 1_convolutional_neural_networks/1.7_segmentation/mask_rcnn
- [x] 1_convolutional_neural_networks/1.7_segmentation/panoptic_fpn
- [x] 1_convolutional_neural_networks/1.7_segmentation/pspnet
- [x] 1_convolutional_neural_networks/1.7_segmentation/segmentation_losses
- [x] 1_convolutional_neural_networks/1.7_segmentation/segnet
- [x] 1_convolutional_neural_networks/1.7_segmentation/unet
- [x] 1_convolutional_neural_networks/1.7_segmentation/unet_variants
- [x] 1_convolutional_neural_networks/1.7_segmentation/volumetric_segmentation
- [x] 1_convolutional_neural_networks/1.7_segmentation/yolact
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/angular_margin
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/bilinear_cnn
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/context_encoder
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/crnn_ctc
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/dncnn
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/facenet_triplet
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/flownets
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/gatys_style_transfer
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/image_colorization
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/image_retrieval
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/johnson_fast_style
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/monocular_depth
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/multilabel_classification
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/openpose_pafs
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/partial_inpainting
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/pwcnet
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/siamese
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/stacked_hourglass
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/super_resolution
- [x] 1_convolutional_neural_networks/1.8_other_vision_tasks/vqa_baseline
- [x] 1_convolutional_neural_networks/1.9_1d_convolutions/character_cnn
- [x] 1_convolutional_neural_networks/1.9_1d_convolutions/ecg_sensor_cnn
- [x] 1_convolutional_neural_networks/1.9_1d_convolutions/kim_sentence_cnn
- [x] 1_convolutional_neural_networks/1.9_1d_convolutions/tcn
- [x] 1_convolutional_neural_networks/1.9_1d_convolutions/time_series_cnn
- [x] 1_convolutional_neural_networks/1.9_1d_convolutions/wavenet_stack
- [x] 1_convolutional_neural_networks/1.10_3d_and_video_convolutions/c3d
- [x] 1_convolutional_neural_networks/1.10_3d_and_video_convolutions/factorized_video
- [x] 1_convolutional_neural_networks/1.10_3d_and_video_convolutions/i3d
- [x] 1_convolutional_neural_networks/1.10_3d_and_video_convolutions/medical_volume_classification
- [x] 1_convolutional_neural_networks/1.10_3d_and_video_convolutions/point_geometry
- [x] 1_convolutional_neural_networks/1.10_3d_and_video_convolutions/point_models_checks
- [x] 1_convolutional_neural_networks/1.10_3d_and_video_convolutions/pointnet
- [x] 1_convolutional_neural_networks/1.10_3d_and_video_convolutions/pointnet_plus_plus
- [x] 1_convolutional_neural_networks/1.10_3d_and_video_convolutions/slowfast
- [x] 1_convolutional_neural_networks/1.10_3d_and_video_convolutions/two_stream
- [x] 1_convolutional_neural_networks/1.10_3d_and_video_convolutions/voxnet
- [x] 1_convolutional_neural_networks/1.11_interpretability/deepdream
- [x] 1_convolutional_neural_networks/1.11_interpretability/feature_visualization
- [x] 1_convolutional_neural_networks/1.11_interpretability/filter_activations
- [x] 1_convolutional_neural_networks/1.11_interpretability/grad_cam
- [x] 1_convolutional_neural_networks/1.11_interpretability/guided_backprop
- [x] 1_convolutional_neural_networks/1.11_interpretability/integrated_gradients
- [x] 1_convolutional_neural_networks/1.11_interpretability/occlusion_sensitivity
- [x] 1_convolutional_neural_networks/1.11_interpretability/smoothgrad
- [x] 1_convolutional_neural_networks/1.11_interpretability/vanilla_saliency


Native data preparation (2026-10-30): the historical `prepare_cnn_*.py` and
`prepare_nn_*.py` entrypoints mentioned above have been retired. Canonical
`dm::datasets::prepare_example_data()` and the installed `datamunge-prepare-data`
CLI own downloads, checksum checks, archive extraction, metadata and tensor NPZ
preparation. Reports call the library directly, using the shared data cache.
Independent framework oracle scripts remain optional development tools; report
data preparation does not invoke them or install Python environments.


## CNN architecture and confusion figures, 1.2–1.11 (2026-10-30)

Completed the figure pass across all 119 reports in sections 1.2 through 1.11.
The 118 reports defining a tensor network/operator now execute architecture
programs and include 146 SVGs (including refreshed classic architecture variants).
`detection_evaluation` defines evaluation procedures rather than a learned network;
it retains its concept/dataflow figure and explicitly explains that scope.

Architecture canvases fit their content, with per-node rotated label placement,
compact branch spacing and parameter-only transform chains scheduled near their
consumers. Named mobile, dense, recurrent and PWC cost-volume/warp blocks reduce
primitive expansion. Multiple-input `model_stage` boundaries preserve branch
inputs and reachable shared parameters; Layers/Operations detail remains available.
Each report states the traced configuration, shape probe and any fixed-token,
frozen-weight or display-adapter limits. Original worked-example sections remain intact.

Added 110 annotated confusion SVGs from 70 executed programs: 59 images from 43
classifier programs, 30 from nine segmentation programs, three from two further
classifiers, one restored-saliency evaluation, and 17 frozen-count replots in 15
reports. Frozen tables and aggregate multi-label decisions are explicitly identified.
Segmentation includes raw counts and true-row percentages; volume counts use
`[TN,FP;FN,TP]`. Existing task-specific AP, overlap, regression, retrieval and
sequence analyses remain appropriate to their tasks. Existing training curves and
comparison panels are retained alongside the native annotated matrices.

The canonical `ConfusionMatrix` API supplies `from_counts`, `to_svg`, `save_svg`
and controls for labels, normalization, colors, cell size, font size and values.
Rendering is deterministic and does not require a display or plotting backend.
The final typography pass rerendered the exact executed integer counts through
this API; it did not refit models or change the measured predictions.

Validation: all 118 architecture programs, 70 confusion programs and the saliency
training prerequisite execute successfully. All 71 focused native tests and five
website/report-runner tests pass; 32 public API/component examples compile and run,
and the two plotting headers compile standalone. Fresh Doxygen coverage passes
348/348 affected definitions; eight clangd hovers pass. Inkscape checks all 146
architecture SVGs for rendered label bounds, own-glyph separation and forward
edges, and all 110 confusion images plus three extreme-size fixtures for text
bounds. Representative figures were visually reviewed. All 119 reports export
to Markdown and current website HTML, with 996 valid local media references and
241 consistently shaped tables. A graphical browser smoke check is not claimed.
Evidence and repeatable validation helpers: `build/cnn-figures-expansion/`, with
`final-validation.json`, source digests, execution logs, rendered previews and
fresh XML under `docs/xml/` inside that evidence directory.
