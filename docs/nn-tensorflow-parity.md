# TensorFlow example parity for the NN reports

The goal is a runnable Datamunge MepML demonstration of every example in
`~/projects/tflow_examples`. The [coverage inventory](nn-tensorflow-coverage.tsv)
tracks the source reports and the method checklist independently. Its source
baseline is commit `2508214d2df55230c2e405d6ccc0673efd4a04a8` plus the local
example checkout as inspected on 2026-11-18.

## Scope and completion criteria

- Track all 655 example reports, including 642 declared CMake targets and 13
  fixture walkthroughs. Use the 701-item method checklist to find shared
  examples and variants that might otherwise be missed. Treat the 508
  `docs/common/` pages as shared implementation references.
- Examine each source program and its method writeup. A same-name entry in the
  inventory is a candidate. Check the inputs, derivatives or expected
  invariants, training and evaluation split if relevant, and any named
  variants. Resolve checklist items with no inferred source individually.
- Use an existing public native API where it covers the behavior. Add reusable
  tensor operations, layers, losses, model components, or I/O to the C++ core
  where needed and expose them through `dm/dm.hpp`. Keep the experiment and
  training loop in the report so the full program remains editable.
- Give each covered example an exact report destination. A report may cover
  several checklist items if it demonstrates every mapped variant. Explain
  the method and assumptions, plot the fit or diagnostics, and report the
  observed limitations of short or synthetic runs. Use source inputs where
  they matter to the method. Pin or generate fixtures for reproducibility.
- Record verification before marking an example complete. Compile and run
  the report code with the checkout library, exercise numerical or behavioral
  controls, and export HTML and Org with the generated figures. Record the
  backend and fixture used when an example depends on external weights,
  media, or hardware.

## Implementation order

| Batch | Inspect or implement | Verify in the reports |
| --- | --- | --- |
| Foundations | Masks, initializers, layer and optimizer options, data loaders, summaries, checkpoint and inference paths | Shapes, reference values and gradients, empty rows, train/eval behavior, state round trips |
| Vision and autoencoders | Reusable convolutional blocks, detection/matching, rendering and attribution, latent objectives, quantization | Fixture scores or held-out fits, gradient routing, invariance, output galleries |
| Generative and sequence models | Diffusion schedules and samplers, flows and likelihoods, recurrent state and masks, attention/cache, generation | Step and density references, inverse and cache equivalence, causality, sample diagnostics |
| Reinforcement learning | Environments, replay, rollout/return estimators, planning and policy/value components | Seeded episodes, terminal and truncation behavior, reference updates and return curves |
| Graph, time series, recommenders | Graph operators, temporal splitting and evaluation, ranking components | Permutation and leakage controls, future holdout, ranking and forecasting baselines |
| Audio, scientific and later sections | Audio transforms, differential solvers, uncertainty, multimodal alignment, structured data, meta-learning, robustness, compression and deployment | Signal reconstruction, solver convergence, calibration, matched splits and intervention or deployment checks |

Start with foundations and audit the existing candidate reports in each batch
before adding overlapping APIs. Preserve optional backend paths when the
source example specifically tests deployment, acceleration, or an external
device. Track the plain native demonstration and the optional run separately.

For the classic vision architectures, audit the canonical model checks and
the compact CIFAR-10 benchmarks together. Match the source's fixed
12,000/5,000 fitting/validation split and compare on all 10,000 official test
images. Reuse the independent crop/flip API from the CIFAR-10 CNN report.
Check grouped connectivity and LRN for AlexNet, switch-based reconstruction
for ZFNet, and the identity between pooled logits and class-map means for
Network in Network. For VGG, check dense-to-convolution weight reuse across
all three variants. For GoogLeNet and Inception, check auxiliary-head loss
and gradient routing separately from inference. For Xception, check the
depthwise-to-pointwise ordering and moving batch-normalization state.

## Inventory workflow

The inventory has one `example` row per upstream report and one `checklist`
row per method or variant. Follow the upstream locator from the source
checkout. Checklist rows name the concrete program or programs to inspect.
For each example row, replace provisional report candidates with the verified
destination and update its status in the same change as the implementation.
Audit variant rows against their own advertised behavior even if the parent
program already has a mapped report.

Run `python scripts/check_nn_tensorflow_coverage.py` to compare the inventory
with the current upstream tree, reconcile build-target rows against the
`tf_example` declarations in CMake, and check that every verified
Datamunge destination exists. Pass the checkout path as an argument if it lives elsewhere.
The check reports inventory counts. The execution and numerical evidence for
individual reports is recorded below.

The inventory has 52 examples without a filename match. Search for equivalent capabilities under other local
report names before implementing a new one. The first follow-up within the
foundations batch is to check device discovery, tensor views and dtypes, graph
execution and custom operations against their source programs. The
[device report](mepml/nn/0_foundations/0.1_tensors_graphs_sessions/hello_tf.mepml)
now enumerates every physical ordinal visible to CPU, CUDA and Vulkan and
probes each for context availability. The saved run found one CUDA ordinal
and three Vulkan ordinals, including llvmpipe. The NVIDIA name matched
the host driver listing. A CPU-only backend smoke test checked disabled
backend diagnostics and the invalid CPU ordinal. All four report blocks
ran and all three figures exported. The report also queried NVML without
requiring CUDA compute support. GPU name, driver version, compute
capability, and total memory matched `nvidia-smi`. The capture includes
clocks, temperature, power, utilization, and framebuffer usage. These
measurements vary over time. A test library with missing functions
confirmed that unsupported metrics remain empty and that missing required
entry points produce a diagnostic.
The [graph construction report](mepml/nn/0_foundations/0.1_tensors_graphs_sessions/graph_construction.mepml)
now exercises strict placement and an explicit fallback using eager compute
contexts. On the CPU-only build, strict CUDA placement reports the backend
diagnostic, while the fallback computes `[[14, 20], [30, 44]]`. The report
also exercises shared parameter gradients, ordered counter updates, and
continued execution after an invalid matrix product. All three code blocks
ran, and both figures were verified in the exports.

## Verified work

- [Causal attention with padded sequences](mepml/nn/0_foundations/0.2_reusable_layer_library/attention_masks.mepml)
  adds `dm::nn::causal_padding_keep_mask` for CPU integer lengths. The fixture
  checks row normalization, exact zeros for excluded positions, heads
  broadcasting in native tests, and all 120 Q/K/V derivatives.
  Native weights, context and gradients agree with the saved TensorFlow
  Float32 fixture to a maximum absolute difference below `1e-7`.
- [Deep Sets](mepml/nn/18_miscellaneous/deep_sets.mepml) composes the existing
  `masked_mean` and optimizer APIs. Held-out MSE fell from 0.075295 to
  0.002789. All 128 predictions, their permuted controls and both trained
  parameter tensors agree with the TensorFlow fixture to `2e-16`.
- Reran the existing [initializers report](mepml/nn/0_foundations/0.1_tensors_graphs_sessions/variables_initializers.mepml)
  and [autodiff report](mepml/nn/0_foundations/0.1_tensors_graphs_sessions/autodiff.mepml).
  They cover the source's mutation and initialization checks, scalar and
  matrix derivatives, vector-Jacobian products, stop-gradient behavior and
  gradient descent through the corresponding native APIs.

All four reports ran and exported to HTML and Org. The attention and
positional encoding suite passed all 28 tests with the mask addition.

- [Learned index](mepml/nn/18_miscellaneous/learned_index.mepml) uses the
  public `dm::algorithms::LearnedIndex` API to fit piecewise linear predictors
  and certify the correction intervals. All 60,002 source-fixture queries
  agree with binary search. A separate deterministic check verified 510,500
  queries across randomized key sets under UndefinedBehaviorSanitizer.
  The report ran and exported to HTML and Org with its correction-window plot.

- [Frozen random features and echo-state reservoirs](mepml/nn/18_miscellaneous/reservoir_computing.mepml)
  adds `dm::nn::reservoir_features` and `dm::nn::fit_reservoir_readout`.
  Both variants retain the upstream input sequences, chronological split,
  and readout penalty. All input features, fitted coefficients, and held-out
  predictions match the saved TensorFlow arrays within `1.8e-11` absolute
  error. The report ran and exported to HTML and Org with prediction plots.

- [Spiking surrogate gradients](mepml/nn/18_miscellaneous/spiking_surrogate.mepml)
  adds `dm::nn::leaky_integrate_and_fire` with hard spikes, a sigmoid
  surrogate derivative, and a detached reset. The report reproduces the
  two learned weights, 96 held-out counts and 768 binary spikes exactly.
  Held-out accuracy rose from 48/96 to 96/96.
- [Double descent](mepml/nn/18_miscellaneous/double_descent.mepml) uses the
  existing SVD minimum-norm API. All eight upstream input and label
  arrays match exactly, and the 80 training and test errors per split
  agree with TensorFlow to `7.1e-10` absolute error. The paired grokking report closes the shared checklist item.

Both reports ran and exported to HTML and Org, and the 27 native layer
and optimizer tests passed, including the analytic surrogate-gradient check.

- [Grokking](mepml/nn/18_miscellaneous/grokking.mepml) uses the existing NN
  tensor, loss, and AdamW APIs for both 5,500-update runs. All 46 training
  and test checkpoint counts match the supplied TensorFlow results exactly.
  The run with decay reaches 56/58 held-out pairs, compared with 10/58
  without decay. Across both runs the maximum checkpoint logit difference
  is below `1.3e-9`.
- [Neural tangent kernels](mepml/nn/18_miscellaneous/neural_tangent_kernel.mepml)
  adds finite-width and analytic ReLU kernel APIs. The finite Gram matrices
  and ridge predictions match the TensorFlow fixture to `5.4e-15` absolute
  error. The limiting Gram matrices agree to `1.4e-8`. The native test
  also checks the finite kernel against an explicit autograd Jacobian Gram.

Both reports ran and exported to HTML and Org with figures.

- [Poincaré embeddings](mepml/nn/18_miscellaneous/poincare_embeddings.mepml)
  adds paired distances and an interior projected gradient step. The original
  105 pairs and targets match exactly. The learned coordinates and 25 held-out
  distances agree with the TensorFlow fixture within `1.5e-15`. Held-out MSE
  drops from 3.72235709 to 0.63825505. The report plots the embeddings and
  held-out distances, and the native suite verifies the radial derivative,
  coincident-point subgradient and projection behavior.

- [Spline-edge regression](mepml/nn/18_miscellaneous/kolmogorov_arnold_network.mepml)
  adds `dm::nn::linear_spline_basis` and `dm::nn::spline_edge_layer`.
  Both layers' basis arrays, learned coefficients and 128 held-out
  predictions match TensorFlow within `4.5e-16`. Held-out MSE fell from
  0.12857213 to 0.00961294. The native test checks constant extrapolation,
  interpolation and gradients.
- [Implicit coordinate networks](mepml/nn/18_miscellaneous/implicit_coordinates.mepml)
  uses existing sine, Fourier features and SGD APIs. Both interleaved-grid
  inputs and targets match exactly. Across the sine and Fourier variants,
  both fitted weight matrices and all 512 held-out predictions match
  TensorFlow within `1.2e-16`.

Both reports ran and exported to HTML and Org with figures. All 30 native
layer and optimizer tests passed.

- [Differentiable circle rendering](mepml/nn/18_miscellaneous/differentiable_rendering.mepml)
  adds `dm::nn::soft_circle_raster` for ordered sigmoid-coverage circles.
  The recovered centers match the upstream fixture exactly. Target and
  fitted images, alpha layers and starting position gradients agree to
  `2.3e-16` absolute error. All four central differences also pass.
- [Neural cellular automata](mepml/nn/18_miscellaneous/neural_cellular_automata.mepml)
  adds `dm::nn::cellular_automaton_rollout` for a masked local residual
  rule. Fitted weights and translated states agree with TensorFlow to
  `4.5e-16`. Translated weighted MSE fell from 0.40456471 to 0.17162434,
  and the shifted prediction exactly matches the corresponding unshifted
  values on the common region.

Both reports ran and exported to HTML and Org with image galleries. All 32
native layer and optimizer tests passed.

- [Equivariant networks](mepml/nn/18_miscellaneous/equivariant_networks.mepml)
  adds `dm::nn::c4_lift_kernel`, `dm::nn::c4_group_kernel` and
  `dm::nn::euclidean_graph_step`. All 14 serialized arrays match TensorFlow
  within `4.5e-16`. The runnable report checks quarter turns, rigid motions
  and node permutations, and plots rotated convolution channels. All 34
  native layer and optimizer tests passed.

- [Dense layers](mepml/nn/0_foundations/0.2_reusable_layer_library/dense_layer.mepml)
  covers optional bias and activation, named parameter sharing across sequence
  positions, and the five upstream initialization and activation pairs.
  The gradient matches the closed form exactly and central differences
  within `6.8e-10`.
- [Pooling](mepml/nn/0_foundations/0.2_reusable_layer_library/pooling.mepml)
  reproduces the upstream VALID and SAME maps, gradient routing, and global
  and 3-D reductions. A direct window scan independently confirmed the
  one-pixel shift agreement of 0%, 28%, 31%, and 100% across the four depths.
- [Dropout](mepml/nn/0_foundations/0.2_reusable_layer_library/dropout.mepml)
  checks inverted dropout moments and inference identity. The fitted regression
  demonstrates Monte Carlo sampling and checks its finite draws, positive spread
  and exact reproducibility of deterministic predictions after sampling.
  The TensorFlow fixture confirms the same behavior on its own fit.

All nine C++ blocks ran. All three reports exported to HTML and Org, and the
website exporter copied all seven SVG figures.

- [Batch normalization](mepml/nn/0_foundations/0.2_reusable_layer_library/batch_norm.mepml)
  checks biased variance in the forward computation and unbiased reported
  variance, fresh and learned inference statistics, and gradients against
  closed forms and central differences. The 24-layer experiment measures
  activation and gradient scales with and without normalization.
- [Normalization layers](mepml/nn/0_foundations/0.2_reusable_layer_library/normalization_layers.mepml)
  checks layer and RMS normalization against hand calculations, group
  normalization for one through eight groups, and instance normalization
  against the one-channel-per-group case. Applying layer normalization to
  `[2, 4, 6]` agrees exactly with treating the eight positions as rows.
- [Weight and spectral normalization](mepml/nn/0_foundations/0.2_reusable_layer_library/weight_spectral_norm.mepml)
  checks weight magnitudes, gradient orthogonality and closed-form
  gradients. It compares fresh and persisted power iteration against an SVD
  and measures the residual Lipschitz error after a finite number of steps.

All nine blocks ran and the three reports exported to HTML and Org with
eight SVG figures.

- [Recurrent cells](mepml/nn/0_foundations/0.2_reusable_layer_library/recurrent_cells.mepml)
  checks RNN, LSTM and GRU state updates and parameter counts against their
  equations, traces gradients across sequence lengths and gate biases, and
  fits the noisy one-bit latch task with a held-out test set. All four
  blocks ran and the report exported to HTML and Org with four SVG figures.

- [Activations](mepml/nn/0_foundations/0.2_reusable_layer_library/activations.mepml)
  covers the upstream nonlinearities, scalar derivatives, Gaussian summaries
  and deep signal propagation, with additional gated activations.
- [Convolution layers](mepml/nn/0_foundations/0.2_reusable_layer_library/conv_layers.mepml)
  checks output sizes, 1-D through 3-D forward values, depthwise and
  separable parameterizations, transposed convolution as the input adjoint,
  overlap counts, and dilation as a kernel with inserted zeros.
- [Positional encodings](mepml/nn/0_foundations/0.2_reusable_layer_library/positional_encodings.mepml)
  preserves the sinusoidal, learned, rotary and ALiBi demonstrations and
  adds `dm::nn::clipped_relative_position_bias`. The upstream rectangular
  grid verifies 48 table lookups and all 10 accumulated derivatives. The
  API canary also verified radius zero, parameter reuse and overflow rejection.

All nine blocks ran and the reports exported to HTML and Org. The scoped
Doxygen check found documentation for all six positional-variant definitions.

- [Embedding](mepml/nn/0_foundations/0.2_reusable_layer_library/embedding.mepml)
  reproduces lookup and repeated-index gradient accumulation, masked
  pooling, and trained softmax skip-gram. The skip-gram loss fell from
  2.5194 to 1.2875, with stronger within-topic cosine similarity.
  The diamonds fit also ran with the checksum-matched pinned CSV.
- [Attention](mepml/nn/0_foundations/0.2_reusable_layer_library/attention.mepml)
  checks scaled dot-product, additive and multi-head attention against
  independent loops. It checks the softmax adjoint, measures scaling
  and entropy, and fits the upstream key-value lookup task. The softmax
  fit reached 0.0002 held-out MSE and assigned 0.989 mean weight to the
  matching key.

All seven blocks ran and the reports exported to HTML and Org, with six
SVG figures copied by the website exporter. All 13 upstream reusable-layer
examples now have verified report mappings.

- [SGD and momentum](mepml/nn/0_foundations/0.3_optimizers_and_training_loop/sgd_momentum.mepml)
  compares SGD, momentum and Nesterov with 20 scalar-reference updates each.
  It checks counter increments, restoration of initial state, and 10 coupled
  updates evaluated at the same old parameter vector. The quadratic and
  64-row minibatch regression demonstrations also ran. All three rules
  recovered the upstream slope and intercept within 0.005 after 320 updates.
- [Adaptive optimizers](mepml/nn/0_foundations/0.3_optimizers_and_training_loop/adaptive_optimizers.mepml)
  checks kernel formulas, initial updates, AMSGrad and Adamax envelopes,
  and AdamW decay. All seven upstream rules passed the twin-parameter,
  shared-counter and zero-gradient-after-reset checks.
- [Advanced optimizers](mepml/nn/0_foundations/0.3_optimizers_and_training_loop/advanced_optimizers.mepml)
  checks LAMB and LARS trust ratios, Lion's signed direction and momentum,
  and Adafactor's rank-one update and factorized slots. The four rules
  stayed finite on zero gradients and fit the synthetic matrix regression.
- [Learning-rate schedules](mepml/nn/0_foundations/0.3_optimizers_and_training_loop/learning_rate_schedules.mepml)
  verifies the 14 upstream boundary values, rejects six invalid argument
  combinations, and matches scheduled SGD to the product formula.
- [Training loop](mepml/nn/0_foundations/0.3_optimizers_and_training_loop/training_loop.mepml)
  shows short batches, weighted aggregation and an explicit minibatch fit.
  Its saved sampler state reproduces the remaining rows and future shuffles
  exactly, and a malformed snapshot leaves the sampler unchanged.

These five reports ran all 19 blocks, exported to HTML and Org, and copied
all 19 SVG figures through the website exporter.

- [Early stopping and checkpoints](mepml/nn/0_foundations/0.3_optimizers_and_training_loop/early_stopping_checkpoints.mepml)
  verifies both monitor directions, zero patience, repeated-epoch rejection,
  best-state restoration and exact continuation of all Adam slots. A missing
  checkpoint leaves the restored weight and update counter intact.
- [Gradient clipping and accumulation](mepml/nn/0_foundations/0.3_optimizers_and_training_loop/gradient_clipping_accumulation.mepml)
  illustrates value and global-norm limits. Four sample-weighted Adam updates
  match the full-batch weights and slots, with explicit rejection checks for
  empty accumulation and zero batch weight.
- [Metrics](mepml/nn/0_foundations/0.3_optimizers_and_training_loop/metrics.mepml)
  covers confusion counts, top-k ties, ROC AUC and average precision. Its
  detection example uses the upstream boxes and checks duplicate matching,
  absent classes and integral precision envelopes. It obtains mAP 11/12 at
  IoU 0.50 and 5/12 at IoU 0.75.

These three reports ran all 13 blocks and exported to HTML and Org. The
remaining optimizer-section example is mixed precision. The report emulates
half and bfloat16 rounding on CPU. All four CPU blocks reran successfully and
were previously exported to HTML and Org. Upstream native Float16 execution
and the explicit BF16 GPU option still need a native device implementation and
strict-placement validation, with Float32 master weights and loss scaling.

- [IDX loader](mepml/nn/0_foundations/0.4_data_loading_and_preprocessing/idx_loader.mepml)
  uses a generated big-endian image and label fixture. It checks pixel
  normalization, paired row gathering, and rejection of truncated payloads,
  trailing bytes and invalid headers.
- [TFRecord I/O](mepml/nn/0_foundations/0.4_data_loading_and_preprocessing/tfrecord_io.mepml)
  checks empty and binary record round trips, CRC-32C, typed Example feature
  parsing, and missing-required-field rejection. The report also tests
  uncompressed, GZIP and ZLIB streams.

All seven blocks ran. Both reports exported to HTML and Org, and the
website exporter copied all four SVG figures.

- [Audio features](mepml/nn/0_foundations/0.4_data_loading_and_preprocessing/audio_features.mepml)
  verifies stereo PCM16, the 1 kHz STFT peak before and after zero-padding,
  log-mel shape, and the first 13 cepstral coefficients against an independent
  DCT sum. A short signal produces the expected empty feature tensors.
- [CIFAR loading](mepml/nn/0_foundations/0.4_data_loading_and_preprocessing/cifar_loader.mepml)
  loads generated CIFAR-10 and CIFAR-100 files in caller order and checks all
  6,144 pixel coordinates per format, fine and coarse labels, and normalization.
- [CSV features](mepml/nn/0_foundations/0.4_data_loading_and_preprocessing/csv_features.mepml)
  checks the BOM, alternate delimiter, quoting and numeric errors, and verifies
  that training-only features are invariant to column reorder.
- [Image folders](mepml/nn/0_foundations/0.4_data_loading_and_preprocessing/image_folder_loader.mepml)
  loads the checked-in JPEG and generated PNGs in the same reordered batch.
  It checks label alignment, grayscale expansion, half-pixel resize and
  train/validation class consistency.
- [Augmentation](mepml/nn/0_foundations/0.4_data_loading_and_preprocessing/data_augmentation.mepml)
  checks flip, crop and cutout coordinates, brightness and hue, mixup,
  area-correct CutMix, and seeded mixing.
- [Text tokenization](mepml/nn/0_foundations/0.4_data_loading_and_preprocessing/text_tokenization.mepml)
  checks whitespace and Unicode scalar boundaries, BPE serialization,
  WordPiece longest-match and unknown rules, and padding and truncation.

All 20 blocks ran and the reports exported to HTML and Org. All eight
upstream preprocessing examples now have verified report mappings.

- [Checkpoint I/O](mepml/nn/0_foundations/0.5_persistence_and_interoperability/checkpoint_io.mepml)
  reproduces the mixed-state restore using native NPZ checkpoints. The weight
  matrix, counter at `2^40 + 7`, UTF-8 and embedded-NUL strings restore
  exactly into existing and fresh handles. The restored matrix predicts
  `[11, 16]`, and missing-name, shape and dtype errors preserve the original
  values. All three blocks ran and both figures exported to HTML and Org.

- [NumPy weights](mepml/nn/0_foundations/0.5_persistence_and_interoperability/numpy_weights.mepml)
  verifies native stored and compressed NPZ output with NumPy, then checks
  independent NumPy-written stored and compressed archives in Datamunge.
  Both produce the upstream prediction 6.25. Fortran-order Float64,
  big-endian Int32 and NPY versions 2 and 3 decode exactly. All six blocks
  ran, and the report exported to HTML and Org.

- [Polynomial regression](mepml/nn/0_foundations/0.6_classical_machine_learning/polynomial_regression.mepml)
  verifies the finite-ensemble bias–variance identity and compares each
  fitted degree against QR. Training MSE decreases from degree 1 to 3 to 12,
  while degree 3 minimizes held-out latent MSE. All three blocks ran and
  the report exported to HTML and Org with three figures.

- [Linear regression](mepml/nn/0_foundations/0.6_classical_machine_learning/linear_regression.mepml)
  reproduces the 64-row sinusoidal fixture with QR, Cholesky and SGD, checks
  normal-equation stationarity and the 0.005 held-out MSE, and compares the
  direct solvers on the nearly collinear design.
- [Regularized regression](mepml/nn/0_foundations/0.6_classical_machine_learning/regularized_regression.mepml)
  checks ridge, lasso and elastic net on the upstream orthogonal and
  correlated designs. All six trajectories satisfy monotone objective and
  final KKT checks, and the L1 fits have exact zeros.

All nine blocks ran and both reports exported to HTML and Org.

- [K-means](mepml/nn/0_foundations/0.6_classical_machine_learning/kmeans_clustering.mepml)
  reproduces the four-cloud Lloyd experiment. Across eight seeds per rule,
  k-means++ has lower mean initial inertia and reaches the same best
  objective as random initialization. All 600 assignments match after
  aligning the arbitrary cluster IDs, and the empty-center check passes.
- [Nearest neighbors](mepml/nn/0_foundations/0.6_classical_machine_learning/nearest_neighbors.mepml)
  covers distance and vote ties, exact-match weighting and the upstream
  held-out rings. All 800 neighbors agree with an independent sort. Both
  voting rules classify every query correctly and weighted radius MSE
  is below 0.005.

All ten blocks ran and both reports exported to HTML and Org.

- [Principal components](mepml/nn/0_foundations/0.6_classical_machine_learning/principal_components.mepml)
  checks the known linear axes and variances, identity covariance after
  whitening, and matching train and held-out reconstruction error. The
  RBF ring embedding verifies Gram centering, the eigensystem, training
  reprojection, and query-batch independence. All six blocks ran and
  both formats exported.

- [Gaussian mixtures](mepml/nn/0_foundations/0.6_classical_machine_learning/gaussian_mixtures.mepml)
  reproduces the upstream three-group full-covariance EM fit on 1,000
  training rows. It verifies monotone likelihood, component parameters and
  the covariance floor. Held-out log densities and responsibilities on
  500 rows agree with direct two-dimensional Gaussian calculations. The
  maximum log-density error is `3.55e-15`. All four blocks ran and all
  four figures exported to HTML and Org.

- [Cross-validation](mepml/nn/0_foundations/0.6_classical_machine_learning/cross_validation.mepml)
  checks deterministic and stratified splits. It selects polynomial
  degree and ridge strength using fold-local centering and scaling,
  refits on all 67 development rows and scores 401 independent test
  responses.
- [Naive Bayes](mepml/nn/0_foundations/0.6_classical_machine_learning/naive_bayes.mepml)
  covers multinomial and Bernoulli document models and reproduces the
  300-row Gaussian evaluation. It checks the constant-feature variance
  floor, posterior normalization and independent Gaussian log scores.

All eight blocks ran and both reports exported to HTML and Org.

- [Logistic regression](mepml/nn/0_foundations/0.6_classical_machine_learning/logistic_regression.mepml)
  reproduces the regularized binary and three-class fits on independent
  training and test samples. It checks extreme-logit gradients, binary
  stationarity and probability error, and multinomial normalization and
  held-out classification. All five blocks ran and both formats exported.

- [Hidden Markov models](mepml/nn/0_foundations/0.6_classical_machine_learning/hidden_markov_models.mepml)
  checks inference against enumeration and runs 45 Baum–Welch updates
  with monotone training likelihood. Held-out Viterbi accuracy allows
  for permuted state labels, and every posterior row normalizes while
  every best-path score is bounded by its sequence likelihood.
- [Generalized linear models](mepml/nn/0_foundations/0.6_classical_machine_learning/generalized_linear_models.mepml)
  checks log-link losses and derivatives, then fits Poisson, Gamma and
  negative binomial regression to the upstream simulation. Each fit
  passes coefficient and gradient checks and beats the training-mean
  baseline on 1,000 held-out rows.

All ten blocks ran and both reports exported to HTML and Org.

- [Support vector machines](mepml/nn/0_foundations/0.6_classical_machine_learning/support_vector_machines.mepml)
  checks the held-out cluster and rings fits with the same hinge objective.
  The fixed 256-feature Fourier map approximates the exact RBF kernel
  and resolves the rings with more than 25 percentage points of accuracy
  gain over the linear head. All five blocks ran and both formats exported.


- [Original MNIST benchmarks](mepml/nn/0_foundations/0.6_classical_machine_learning/softmax_mnist.mepml)
  verifies one shared executable report for the linear softmax and 128/64-unit
  MLP programs. It uses the upstream checkout's official IDX files, 55,000
  fit rows and 5,000 validation rows. Both validation-selected checkpoints
  restore their recorded NLL and are evaluated on all 10,000 official test
  rows, with accuracies of 92.70% and 97.75%. Each confusion matrix accounts
  for every test row and reproduces its printed accuracy. The MLP companion
  links to the shared benchmark. Both pages ran and exported to HTML and Org.

- [Tensor basics](mepml/nn/0_foundations/0.1_tensors_graphs_sessions/tensor_basics.mepml)
  checks dtype promotion, aliasing and reshape against brute force. The
  borrowed-buffer example uses a standard C++ span over Datamunge's tensor
  storage for the upstream host calculation. Scalar and row reductions,
  broadcasts, casts, matrix products and the Gram matrix match the
  expected values. All four blocks ran and both formats exported.

- [Custom gradients](mepml/nn/0_foundations/0.1_tensors_graphs_sessions/custom_gradient.mepml)
  uses local `nn::Function` callbacks for custom backward rules. It
  verifies erfc, first and second derivatives, the missing history error,
  and straight-through ceiling via `stop_gradient`.
- [Custom kernels](mepml/nn/0_foundations/0.1_tensors_graphs_sessions/custom_op_kernel.mepml)
  implements the upstream ScaleShift and RowSums operations with native
  function objects, validates float32 and float64 execution and rejects
  invalid ranks and integer inputs. Composition passes first- and
  second-derivative checks.
- [Session-style execution](mepml/nn/0_foundations/0.1_tensors_graphs_sessions/sessions.mepml)
  maps feeds, fetches, intermediate substitution and side effects onto
  eager calls. It checks the upstream affine model values and counter,
  and traces per-operation timing with Datamunge's native profiler.

All 14 executable blocks ran across these three reports, and all three
exported to HTML and Org. The illustrative custom-gradient snippet is
rendered as text.

- [Depthwise separable convolution](mepml/nn/1_convolutional_neural_networks/1.1_fundamentals/depthwise_separable.mepml)
  checks direct factor contraction, native forward values and gradients for
  inputs and both factors. Batched SVD verifies the discarded-energy
  identity for multipliers 1, 2 and 5. Equivalent filters on 32×32 inputs run the
  upstream warmup and timing protocol with the corresponding MAC counts.
- [Transposed convolution](mepml/nn/1_convolutional_neural_networks/1.1_fundamentals/transposed_convolutions.mepml)
  checks explicit scatter-add, the inner-product adjoint, input-gradient
  parity and finite differences. Constant inputs expose the expected
  checkerboard phases and the effect of resize-convolution.

All 13 blocks ran across these two reports. Both exported to HTML and Org.

- [Dilated convolution](mepml/nn/1_convolutional_neural_networks/1.1_fundamentals/dilated_convolutions.mepml)
  checks zero-inserted filter equivalence for six padding and dilation
  cases, verifies input and filter derivatives against central
  differences, and traces gridded dependency support. The rate-two
  student recovers the teacher coefficients to within `1e-4`.
  All seven blocks ran and both formats exported.

- [Im2col convolution](mepml/nn/1_convolutional_neural_networks/1.1_fundamentals/convolution_im2col.mepml)
  checks eight padding, stride and dilation geometries. It compares
  products and input/filter gradients to the native convolution path,
  verifies the patch-extraction adjoint, and checks input gradients by
  central differences. All six blocks ran and both formats exported.


- [MNIST CNN](mepml/nn/1_convolutional_neural_networks/1.1_fundamentals/cnn_mnist.mepml)
  trains the 105,866-parameter architecture for six epochs on 55,000
  rows with 5,000 separate validation rows. The recorded validation NLL
  reproduces after checkpoint restoration. Test accuracy is 98.72%
  across all 10,000 original test examples, with NLL 0.039052.
  Independent NumPy counts reproduce the complete confusion matrix.
  All eight blocks ran and both formats exported.

- [Receptive fields](mepml/nn/1_convolutional_neural_networks/1.1_fundamentals/receptive_fields.mepml)
  restores the full MNIST CNN and independently repeats the 98.72% test
  evaluation. It covers all ten digits, checks the chosen trained feature
  against its theoretical 10×10 support, and checks active and
  outside-support perturbations. The report declares the CNN checkpoint's
  producing report as a prerequisite. All seven blocks ran and both
  formats exported.

- [Global pooling versus flatten](mepml/nn/1_convolutional_neural_networks/1.1_fundamentals/global_pooling_vs_flatten.mepml)
  fits both heads on two controlled spatial classification tasks. Global
  pooling scores 100% on appearance and 50% on location, while flatten
  scores 50% on appearance and 100% on location. The report checks the 6
  and 258 parameter counts, the balanced-class loss for location with
  pooling, and the unchanged flatten weights at unseen coordinates. All
  ten blocks ran, the strengthened fit block ran separately, and both
  formats exported.

- [LeNet-5](mepml/nn/1_convolutional_neural_networks/1.1_fundamentals/lenet5.mepml)
  trains on 55,000 MNIST images for eight epochs and restores the checkpoint
  selected on 5,000 validation images. Test accuracy is 97.94% across all
  10,000 official test rows. The report checks disconnected C3 derivatives
  and fixed prototype predictions. An independent check reproduces the
  confusion matrix and the six stage counts totaling 60,000 learned scalars.
  All eight blocks ran and both formats exported.

- [CIFAR-10 CNN](mepml/nn/1_convolutional_neural_networks/1.1_fundamentals/cnn_cifar10.mepml)
  fits 45,000 original RGB images for ten epochs and selects on 5,000
  separate development images. The crop/flip fixture independently
  recovers four distinct transforms on four source rows, including black
  padding. The 545,098-parameter checkpoint reaches 73.17% test accuracy
  and 0.767207 test NLL. NumPy reproduces the confusion matrix against
  the raw official test labels. All nine blocks ran and both formats
  exported.

- [Batch normalization ablation](mepml/nn/1_convolutional_neural_networks/1.1_fundamentals/batch_norm_ablation.mepml)
  fits six models on the same 12,000 MNIST images for six epochs. For
  each seed, both variants share initial weights and minibatch order.
  A moment-update fixture checks training state, and inference checks
  companion invariance and restoration into fresh stores. The three
  differences on the 10,000 official test images average 0.453 percentage
  points in favor of BN, with sample SD 0.203 points. NumPy reproduces
  all six confusion matrices from the original test labels and the
  saved logits. All fourteen blocks ran and both formats exported.

- [AlexNet](mepml/nn/1_convolutional_neural_networks/1.2_classic_architectures/alexnet.mepml)
  checks all canonical stage shapes and nonzero gradients through 22
  parameter tensors. Grouped connectivity, clipped LRN, dropout and the
  regularized training update have separate controls. The compact model
  fits 12,000 CIFAR-10 images for ten epochs and reaches 53.70% accuracy
  on all 10,000 official test images after validation selection. NumPy
  reproduces the full confusion matrix from raw test labels. Nine report
  blocks and the separate training-objective control ran, and both
  formats exported.

- [ZFNet](mepml/nn/1_convolutional_neural_networks/1.2_classic_architectures/zfnet.mepml)
  checks the canonical shapes, 62,357,608 scalars, and nonzero gradients
  through all 16 learned tensors. The compact fit uses 12,000 original
  CIFAR-10 training images for ten epochs and reaches 55.53% test accuracy
  after selection on 5,000 validation images. Independent counts reproduce
  the full confusion matrix. The restored model projects the strongest
  conv5 feature of the first official test image through both Deconvnet and
  ordinary differentiation. All ten blocks ran and both formats exported.

- [Network in Network](mepml/nn/1_convolutional_neural_networks/1.2_classic_architectures/network_in_network.mepml)
  checks pointwise XOR, explicit patch-MLP equivalence and gradients
  through the full 966,986-parameter graph. The quarter-width model fits
  12,000 CIFAR-10 images for twelve epochs and reaches 48.06% on the
  official test split after checkpoint selection. Independent checks
  reproduce the confusion matrix and all 100 class-map means, and
  confirm that the gallery contains the first test image of each class.
  All eleven blocks ran and both formats exported.

- [NMS and matching](mepml/nn/1_convolutional_neural_networks/1.6_object_detection/nms_matching.mepml)
  checks threshold equality, tie ordering, grouped detection budgets, RPN
  and SSD assignments, 1,000 box-coding round trips, zero-logit objectness
  derivatives, and a checkpoint warm start with fresh Adam state.
  Seventeen detection, box and NMS tests passed, including TensorFlow
  reference values. All six blocks ran and both formats exported.

- [Segmentation losses](mepml/nn/1_convolutional_neural_networks/1.7_segmentation/segmentation_losses.mepml)
  checks Dice, Tversky, Lovász and boundary losses, then fits four fields
  with 200 Adam steps each. Every fitted field reconstructs the same
  known circle. Five tests verify gradients, empty and ignored labels,
  and 100 exhaustive rectangular distance oracles. The native separable
  distance calculation returns the same distances as the source and
  uses more work than its linear-time envelope. All six blocks ran and
  both formats exported, including the fitted-mask PNG.

- [Point geometry](mepml/nn/1_convolutional_neural_networks/1.10_3d_and_video_convolutions/point_geometry.mepml)
  checks FPS ties, strict ball-query boundaries and padding, all 64
  gathered features and 96 relative coordinates, and 60 frozen-neighbor
  finite differences. Four native tests cover the original fixture,
  independent batched oracles, duplicate scatter and invalid inputs.
  All six blocks ran and both formats exported.

- [Point model checks](mepml/nn/1_convolutional_neural_networks/1.10_3d_and_video_convolutions/point_models_checks.mepml)
  executes both canonical 1,024-point, 40-class models and both compact
  models on genuine CAD surfaces. It checks every learned gradient,
  inventory counts, exact identity transform initialization, all nine
  orthogonality derivatives and the two models' order dependence. The
  local fixtures use chair, sofa and table meshes, with the canonical
  runtime check on a chair. The upstream compact dataset uses toilet
  meshes for its third class. All nine blocks ran and both formats exported.

- [Vanilla saliency](mepml/nn/1_convolutional_neural_networks/1.11_interpretability/vanilla_saliency.mepml)
  fits the shared small CNN on 1,000 source-selected MNIST images with
  96 explicit Adam updates. The native selected checkpoint reaches 88%
  accuracy on the 200 official-test images, with an independently verified
  confusion matrix. The report checks fixed raw-logit derivatives by
  central differences and preserves initial weights for attribution controls.
  All seven blocks ran and both formats exported.

- [Guided backpropagation](mepml/nn/1_convolutional_neural_networks/1.11_interpretability/guided_backprop.mepml)
  checks the two ReLU gates and compares ten held-out digit maps with
  ordinary, runner-up, initial-weight and reset-head gradients. An
  independent check reproduces all four aggregate L1 changes. Six blocks
  ran and both formats exported.

- [Integrated gradients](mepml/nn/1_convolutional_neural_networks/1.11_interpretability/integrated_gradients.mepml)
  preserves the true and runner-up labels and holds the selected target
  fixed along each path. NumPy re-integrates the saved 64 midpoint
  derivatives for each primary, runner-up, gray-baseline and initial-weight
  control and reproduces all completeness residuals. Five blocks ran and
  both formats exported.

- [SmoothGrad](mepml/nn/1_convolutional_neural_networks/1.11_interpretability/smoothgrad.mepml)
  saves the actual Gaussian inputs and derivatives. Independent means,
  magnitudes, squares, sample variances and standard errors agree for all
  eight groups, including clipping and initialization controls. Six blocks
  ran and both formats exported.

- [Occlusion sensitivity](mepml/nn/1_convolutional_neural_networks/1.11_interpretability/occlusion_sensitivity.mepml)
  compares fixed-logit scores under each actual patch mask. Independent
  overlap accumulation reproduces every coverage and map value for the
  eight patch, baseline, target and model controls. Six blocks ran and both
  formats exported.

- [Grad-CAM and Grad-CAM++](mepml/nn/1_convolutional_neural_networks/1.11_interpretability/grad_cam.mepml)
  derives the two weight formulas and checks them for fixed raw logits.
  NumPy reproduces both heatmaps from saved activations and gradients for
  all ten digits and four controls. Five blocks ran and both formats
  exported.

- [GoogLeNet](mepml/nn/1_convolutional_neural_networks/1.2_classic_architectures/googlenet.mepml)
  checks nine Inception modules, all 15 canonical stage shapes and
  gradients through 128 learned tensors. Independent host cross entropy
  confirms both auxiliary weights. The perturbation changes the training
  objective and the auxiliary bias updates with kernel L2 enabled.
  The compact model reaches 52.07% accuracy after twelve epochs on
  12,000 fitting images and validation selection. Independent counts
  reproduce all 10,000 test predictions in the confusion matrix.
  All nine blocks ran and both formats exported.

- [VGG-11, VGG-16 and VGG-19](mepml/nn/1_convolutional_neural_networks/1.2_classic_architectures/vgg.mepml)
  checks the three canonical forwards, gradients through every learned
  parameter tensor, receptive fields and shared dense/convolutional heads.
  The compact models respectively reach 52.73%, 45.13% and 35.43% on the
  official test split. VGG-19 uses seed 29, with lower validation NLL
  than the first run at seed 17, whose test accuracy was 31.36%.
  The sixteen other report blocks passed in the first run. The training
  variants ran separately, each selected checkpoint was reevaluated, and
  the final report exported both formats with all plots and independently
  verified confusion matrices.

- [DeepDream](mepml/nn/1_convolutional_neural_networks/1.11_interpretability/deepdream.mepml)
  uses frozen learned digit features, optimizes the actual images across
  three resolutions, and records every iteration for the main and control
  objectives. Independent checks confirm nondecreasing recorded objectives
  and clipping for all 75 updates. Five blocks ran and both formats exported.

- [Feature visualization](mepml/nn/1_convolutional_neural_networks/1.11_interpretability/feature_visualization.mepml)
  optimizes seeded noise for all 16 convolutional channels and four
  controls. Independent weighted objective recomputation confirms all
  80 histories and 960 monotonic updates, including exact zero-rate
  pixel retention. Five blocks ran and both formats exported.

The native `dm::nn::split_image_holdout` API now supplies the CIFAR-10
CNN's seeded split and preserves the split under a fitting cap. All nine
dataset tests and both Doxygen examples ran. Doxygen XML exposes the
new function, result type and fields. The report reran all nine blocks
and its labels, logits and confusion counts are bitwise equal to the
previous split implementation.

- [Xception](mepml/nn/1_convolutional_neural_networks/1.2_classic_architectures/xception.mepml)
  checks the signed depthwise/pointwise ordering, all sixteen canonical
  stage shapes and gradients through 156 learned tensors. The twelve-epoch
  compact run confirms checkpoint restoration and immutable inference
  moments, reaching 62.51% accuracy on all official CIFAR-10 test images.
  NumPy reproduces its confusion matrix from the raw labels. All nine
  blocks ran and both formats exported.

- [Filter activations](mepml/nn/1_convolutional_neural_networks/1.11_interpretability/filter_activations.mepml)
  runs frozen official VGG-19 weights on the original Labrador and Kandinsky
  images using max pooling. Independent NumPy checks reproduce 2,944
  channel summaries and all 524,288 first-convolution outputs within
  0.000136. The native report checks constant weights and repeatability.
  All sixteen blocks ran and both formats exported.

- [Bayesian linear regression](mepml/nn/12_probabilistic_and_bayesian_deep_learning/bayesian_linear_regression.mepml)
  adds `dm::nn::ml::FitBayesianLinear` and `PredictBayesianLinear`.
  Fits to all 32 simulated observations compare degrees 1, 2 and 5 and
  average predictive means and variances at 61 points. The quadratic
  receives 93.17% posterior model probability. All 22 saved arrays agree
  with the upstream TensorFlow run to a maximum absolute error of
  1.18e-13. An independent data-space Gaussian solve reproduces the
  evidence, covariance and predictions for each degree. Both blocks ran
  and both formats exported.

- [Inception v2 and v3](mepml/nn/1_convolutional_neural_networks/1.2_classic_architectures/inception_v2_v3.mepml)
  checks 15 and 18 canonical stages and their 141 and 196 learned
  tensors. Rectangular factorization, beta-only normalization, label
  smoothing and primary/auxiliary loss controls have their own fixtures.
  The compact fits score 53.58% and 56.59% on all 10,000 official test rows,
  and independent counts reproduce both confusion matrices. All fourteen
  blocks ran and both formats exported.

- [Maximum likelihood and MAP](mepml/nn/12_probabilistic_and_bayesian_deep_learning/mle_map.mepml)
  refits the source 16-row Gaussian regression with a noise SD of .6 and a
  prior SD of .35. Held-out RMSE on 256 rows is 0.764283 for MLE and 0.677850
  for MAP. All eight saved arrays agree with the TensorFlow run within
  4.8e-14. The report ran and both formats exported. The shared Gaussian
  regression API passed the classical workflow tests, its four public
  examples and the Doxygen audit.

- [Bayesian optimization](mepml/nn/12_probabilistic_and_bayesian_deep_learning/bayesian_optimization.mepml)
  uses the native fixed RBF surrogate and expected improvement on the
  source 201-point grid. It exactly reproduces all 15 evaluation indices,
  their objective values and zero grid regret. All 7,236 saved means,
  variances and acquisition values match TensorFlow within 1.13e-11.
  The report ran and both formats exported.

- [Temperature scaling](mepml/nn/12_probabilistic_and_bayesian_deep_learning/temperature_scaling.mepml)
  adds bounded inverse-temperature fitting and top-label reliability.
  The separate 2,048-row validation and 4,096-row test sets reproduce
  all ten TensorFlow arrays to within 2.3e-16. Independent bin counts give
  held-out ECE 0.182347 before and 0.009467 after, with unchanged predicted
  labels. The report ran and both formats exported.

- [ResNet-18 and ResNet-34](mepml/nn/1_convolutional_neural_networks/1.3_residual_and_dense_families/resnet_basic.mepml)
  checks the canonical inventories and all learned derivatives. Both
  compact fits use the new seeded holdout API for 12,000 fitting and
  5,000 validation rows. After twelve epochs, the restored models score
  67.12% and 65.02% on the complete official CIFAR-10 test set. This run
  uses CPU inference; the report retains device selection. All eight
  blocks ran and both formats exported.

- [Inception v4 and Inception-ResNet-v2](mepml/nn/1_convolutional_neural_networks/1.2_classic_architectures/inception_v4_resnet.mepml)
  checks all 22 and 51 canonical stages and 306 and 496 learned tensors.
  Zero, single and doubled residual scales check the identity path and
  gradient scaling. Both compact models fit for twelve epochs and
  restore their validation-selected learned parameters and moving moments.
  Test accuracy is 34.93% and 59.67%. The report records the first model's
  0.07-percentage-point gap below the upstream sanity floor. Both
  confusion matrices match independent counts on the full test split.
  All thirteen blocks ran and both formats exported.

- [Exact Gaussian process regression](mepml/nn/12_probabilistic_and_bayesian_deep_learning/gaussian_process.mepml)
  adds `dm::nn::fit_gaussian_covariance` and `condition_gaussian` for
  known covariance blocks. All eight candidate log likelihoods and the
  joint posterior at 81 queries agree with the TensorFlow fixture.
  Maximum absolute error across the nine arrays is 7.4e-13. The fit
  selects length .65, and the pointwise observation bands cover 80 of
  81 independently generated test responses. The report ran and both
  formats exported.

### Deterministic autoencoders

All eleven reports in this group ran their 81 C++ blocks and exported HTML
and Org. Independent NumPy checks use all 200 shared held-out MNIST images
and the selected native parameters.

| Report | Reproduced checks |
| --- | --- |
| [Undercomplete](mepml/nn/2_autoencoders/2.1_deterministic_autoencoders/undercomplete_autoencoder.mepml) | The 32-coordinate bottleneck achieves held-out MSE .050326, below the fitting-pixel mean. All 1,400 source IDs are disjoint across partitions. |
| [Tied](mepml/nn/2_autoencoders/2.1_deterministic_autoencoders/tied_autoencoder.mepml) | The tied decoder is exactly the shared transpose. Both complete encoded and reconstructed arrays match independent matrix products. |
| [Sparse](mepml/nn/2_autoencoders/2.1_deterministic_autoencoders/sparse_autoencoder.mepml) | For matched plain, L1, KL and combined fits, every test code, unit mean and reconstruction matches the independent sigmoid calculation. |
| [Denoising](mepml/nn/2_autoencoders/2.1_deterministic_autoencoders/denoising_autoencoder.mepml) | Both fits' reconstructions match at five held-out noise conditions using the actual saved corrupted inputs and clean targets. |
| [k-sparse](mepml/nn/2_autoencoders/2.1_deterministic_autoencoders/k_sparse_autoencoder.mepml) | A stable signed sort reproduces all eight winners and stored codes for every row in all partitions. |
| [Stacked](mepml/nn/2_autoencoders/2.1_deterministic_autoencoders/stacked_autoencoder.mepml) | All eight transferred arrays are exact copies of the selected stage encoders and decoders. Independent sigmoid composition reproduces the pretrained reconstruction. |
| [Masked](mepml/nn/2_autoencoders/2.1_deterministic_autoencoders/masked_autoencoder.mepml) | Both held-out masks retain four distinct patches per image. Independent patchification reproduces masked MSE .079355 and .078687. |
| [Contractive](mepml/nn/2_autoencoders/2.1_deterministic_autoencoders/contractive_autoencoder.mepml) | The independent sigmoid Jacobian matches every probe value, and exact squared penalties agree for all held-out rows under all three fits. |
| [Winner-take-all](mepml/nn/2_autoencoders/2.1_deterministic_autoencoders/winner_take_all_autoencoder.mepml) | Stable feature-wise sorting reproduces the saved gates for the first, partial and selected-checkpoint training probes. |
| [Recurrent](mepml/nn/2_autoencoders/2.1_deterministic_autoencoders/recurrent_autoencoder.mepml) | Independent encoder and decoder recurrences reproduce all 200 codes and all 28 reconstructed rows per image. |
| [Convolutional](mepml/nn/2_autoencoders/2.1_deterministic_autoencoders/convolutional_autoencoder.mepml) | The full 7×7×4 bottleneck and 14×14×8 decoder probe lead to held-out MSE .017783 on the original inputs. |

### Variational and distribution-matching autoencoders

All eighteen reports ran 155 C++ blocks and exported HTML and Org. The
existing independent Python oracles passed against these fresh artifacts.
Documented shell commands are shown as prose examples in the exports.

| Report | Independent checks |
| --- | --- |
| [VAE](mepml/nn/2_autoencoders/2.2_variational_autoencoders/variational_autoencoder.mepml) | 101,652 first derivatives, bit-exact first Adam update, three validation checkpoints and unfiltered prior and latent-grid predictions. |
| [Beta-VAE](mepml/nn/2_autoencoders/2.2_variational_autoencoders/beta_variational_autoencoder.mepml) | 102,832 first derivatives and six eligible checkpoints for the paired beta objectives. |
| [Conditional VAE](mepml/nn/2_autoencoders/2.2_variational_autoencoders/conditional_variational_autoencoder.mepml) | 102,932 first derivatives, exact conditioning inputs, exact first Adam update and all three validation checkpoints. |
| [Convolutional VAE](mepml/nn/2_autoencoders/2.2_variational_autoencoders/convolutional_variational_autoencoder.mepml) | 4,234 initial and selected derivatives, first update and three validation checkpoints. |
| [FactorVAE and beta-TCVAE](mepml/nn/2_autoencoders/2.2_variational_autoencoders/factor_tc_variational_autoencoder.mepml) | 102,929 first derivatives, both total-correlation calculations and six checkpoints. |
| [Categorical VAE](mepml/nn/2_autoencoders/2.2_variational_autoencoders/gumbel_categorical_autoencoder.mepml) | 53,104 first derivatives, exact first Adam update and fixed-temperature soft and hard decode controls. |
| [Importance-weighted](mepml/nn/2_autoencoders/2.2_variational_autoencoders/importance_weighted_autoencoder.mepml) | 102,832 first derivatives and six checkpoints scored with the common validation objective. |
| [Wasserstein](mepml/nn/2_autoencoders/2.2_variational_autoencoders/wasserstein_autoencoder.mepml) | 26,036 derivatives, exact first Adam update, distribution-matching scores and all validation checkpoints. |
| [Adversarial](mepml/nn/2_autoencoders/2.2_variational_autoencoders/adversarial_autoencoder.mepml) | 77,522 derivatives, four independent first Adam states, three-phase update isolation and four validation checkpoints. |
| [Flow posterior](mepml/nn/2_autoencoders/2.2_variational_autoencoders/flow_variational_autoencoder.mepml) | 52,672 first derivatives, exact first Adam update, inverse and log-determinant checks and three checkpoints. |
| [Hierarchical and NVAE](mepml/nn/2_autoencoders/2.2_variational_autoencoders/hierarchical_variational_autoencoder.mepml) | 4,004 first derivatives, six first moving buffers and six validation checkpoints across both models. |
| [InfoVAE and MMD-VAE](mepml/nn/2_autoencoders/2.2_variational_autoencoders/info_variational_autoencoder.mepml) | 52,208 first derivatives, decomposition of both objectives and six validation checkpoints. |
| [Ladder VAE](mepml/nn/2_autoencoders/2.2_variational_autoencoders/ladder_variational_autoencoder.mepml) | 52,196 first derivatives, precision fusion, bit-exact first update and three validation checkpoints. |
| [Posterior collapse](mepml/nn/2_autoencoders/2.2_variational_autoencoders/posterior_collapse.mepml) | Paired KL schedules, selected checkpoints and fixed-prefix latent controls pass the sequence oracle. |
| [Sequence VAE](mepml/nn/2_autoencoders/2.2_variational_autoencoders/sequence_variational_autoencoder.mepml) | Source grammar, teacher forcing, padding, separated free decoding and held-out controls pass the sequence oracle. |
| [VAE-GAN](mepml/nn/2_autoencoders/2.2_variational_autoencoders/vae_gan.mepml) | 64,786 derivatives and three bit-exact first Adam steps with unchanged unrelated parameter groups. |
| [VQ-VAE](mepml/nn/2_autoencoders/2.2_variational_autoencoders/vector_quantized_autoencoder.mepml) | 805 parameter derivatives, exact first update, empirical code use and independent prior sampling. |
| [VQ-VAE-2](mepml/nn/2_autoencoders/2.2_variational_autoencoders/vector_quantized_autoencoder_2.mepml) | 1,157 parameter and 6,272 hierarchy derivatives, exact first update and both levels of prior samples. |

### Adversarial generative models

Ten reports ran 100 C++ blocks and exported HTML and Org with 100 figures.

| Reports | Independent checks |
| --- | --- |
| Vanilla, least-squares and hinge GANs | The first fitting batch and initial and final test forwards match independent NumPy calculations on the original MNIST inputs. Each fitting epoch visits all 1,000 source rows. |
| Conditional GAN and auxiliary classifier GAN | The held-out generated images and class-conditioned diagnostics match independent recomputation. |
| Wasserstein GAN | The clipped and gradient-penalty variants match the independent oracle for the forward values, parameter derivatives and first four updates. |
| DCGAN | Independent forward and backward calculations check transposed convolution, batch-normalization state and alternating updates. |
| Spectral GAN and InfoGAN | Initial, first-batch and final forwards match the independent calculations, including spectral normalization and the categorical and continuous recognition outputs. |
| Bidirectional GAN | The independent oracle checks joint scores and learned encoder and generator outputs on the synthetic rings. |

The other ten GAN reports ran 144 C++ blocks and exported both formats.
The BigGAN and self-attention GAN source forwards match independent NumPy
calculations. The CycleGAN, Pix2Pix, progressive GAN, StarGAN and StyleGAN
oracles check their saved probes. For SinGAN, independent convolutions
reproduce generated and reconstructed outputs from all three scales.
Independent SRGAN and ESRGAN predictions and held-out quality metrics
agree with the saved reports. For GAN diagnostics, the original frozen
Inception graph supplies the features. Independent recomputation matches
all six FID and Inception Score pairs. The real-image controls reproduce
the reference scores. The SinGAN pool uses the reference seed 8101, with
scores 351.225 and 1.214 for the native fitted checkpoint.

### Normalizing flows

All nine reports ran 141 C++ blocks and exported HTML and Org with all
132 figures. Independent NumPy forward and inverse calculations match the
first fitting batch, the initial and trained held-out density values, and
the generated samples and Jacobians. This covers continuous flows, Glow,
IAF, MAF, NICE, Real NVP, rational-quadratic splines, the planar and radial
variants, and the three-family Real NVP study with banana, eight-Gaussian
and checkerboard targets. Every fit used each of its 512 fitting rows
once per epoch for three epochs.

### Autoregressive models

All nine reports ran 89 C++ blocks and exported HTML and Org with 79
figures. Independent forward calculations reproduce held-out logits for
the character RNN, character Transformer, ImageGPT and WaveNet. The
independent MADE, PixelCNN, gated PixelCNN, PixelCNN++ and PixelRNN
oracles reproduce the fitting-batch, initial and trained held-out
forwards. The gated model covers both label-conditioned and unconditional
variants. MADE covers both raster and reverse order.

### Diffusion and flow matching

Eleven reports ran 125 C++ blocks and exported HTML and Org. Independent
calculations check trained held-out predictions, class-conditioning
controls, and selected full DDIM, reverse SDE and probability-flow ODE
trajectories. The new text-conditioned report adds
`dm::nn::models::TokenConditionedDenoiser` and `DdimFromClean`. It
fits all 288 batches from the two-word bar protocol, plots the training
loss, and renders all prompts from matched noise. The independent oracle
reproduces every transition and all 16 template assignments. The
last-batch loss falls from .116350 to .059900 with native initialization
and Adam.

### Energy-based models

All five reports ran 43 C++ blocks and exported HTML and Org. Independent
enumeration reproduces the held-out probabilities for the deep belief,
deep Boltzmann and restricted Boltzmann models. Independent calculus
reproduces the Langevin model’s held-out energy, force and Hessian, and
the Hopfield model’s retrieval steps and Jacobians at all three inverse
temperatures.

### Autoencoder applications

Seven reports ran 75 C++ blocks and exported HTML and Org. The independent
oracles check the restoration estimates, encoder transfers, latent edits
and projections, coded image output and reconstruction anomaly scores.
AutoRec reads the original MovieLens ratings and fits only the training
entries. Its independent oracle reproduces the first masked loss and its
derivatives, verifies split isolation, and matches all 50,944 predictions
from the selected checkpoint.

### Recurrent cells and training controls

Seventeen reports ran 155 C++ blocks and exported HTML and Org. The
independent sequence calculations reproduce initial and trained held-out
predictions and initial fitting gradients for every cell and control.
The clipping checks recompute every applied gradient under the global
norm, coordinatewise and unclipped policies. The dropout checks verify
time-shared and per-step masks, and the truncated-backpropagation checks
use both the full window and the last-three-step window.

### Sequence prediction and decoding

Independent reference calculations reproduce held-out outputs for BiLSTM
tagging, BiLSTM-CRF, IMDB sentiment classification, and the PTB word RNN.
The beam search report adds the original three-token transition fixture
using the public `dm::nn::BeamSearch` API. All 15 terminal or capped
paths agree with exhaustive enumeration, including EOS, separate states
and ties. Width one gives probability .063580 and width two gives .315000.
The decoded translation example also uses the shared beam implementation.

The new recurrent forecasting report fits both direct six-lead models
with `dm::nn::ForecastWindows` and `RecurrentDirectForecast`. The
NumPy recurrence reproduces every prediction and confirms that complete
test targets stay within the test split. Native LSTM and GRU one-step
RMSEs are .145328 and .104865. The report plots both held-out fits and
horizon-specific errors. All six reports ran 40 C++ blocks and exported
HTML and Org.

The current local MIDI, pointer generator and pointer network reports
run. Their source protocols still require work before verification.
The MIDI example uses a different note vocabulary and LSTM width, and
the pointer reports use older task fixtures.

The new toy encoder-decoder report covers the default reversal and
addition tasks. It runs both fits and exports a validation-likelihood
plot. NumPy recreates all 192 unique input and target records per task
and reproduces all held-out teacher-forced logits and free-running
sequences. Both trials achieve 0/32 exact matches after three epochs.
The same upstream executable also offers Bahdanau and dot-product Luong
attention and English-to-Spanish translation. Those variants and the
parent example remain open.

### Language model inference

The native KV cache report now checks all 2,048 held-out steps and the
48-token greedy continuation used by the reference protocol. The
independent NumPy audit matches every key, value, attention weight,
133,120 logits and generated token on the native checkpoint.

The token sampling report reproduces the reference prompt, sampler
settings, MT19937 draws and newline-aware beam on a native fitted
checkpoint. The public `dm::nn::models::DrawRankedToken` preserves
ranked inverse-CDF order alongside the existing vocabulary-order API.
The source oracle matches all 128 decisions and three beam scores.
The report distinguishes the legacy single-distribution fixtures from
the source sequence recordings and plots the effective vocabulary.

The published Small BERT report serves the three original 16-token
requests. Independent WordPiece packing and NumPy inference match all
6,144 sequence outputs and 384 pooled values, with maximum absolute
differences below 3.7e-6. The reference SavedModel replay requires the
external TensorFlow recording. All three reports ran their C++ blocks
and exported HTML and Org.

### One-dimensional convolutional applications

Six reports ran 71 C++ blocks and exported HTML and Org with 62 figures.
Independent counts reproduce every held-out confusion cell from the
original MR review labels and the official GunPoint, ECG200 and
BasicMotions splits. BasicMotions uses the class order declared by the
TS archive. On the speaker-disjoint WaveNet split, independent NLL is
5.535257 after fitting and 5.550587 before fitting. All 768 generated
tokens match independent sampling from the recorded logits and uniforms.

### Predictive uncertainty

Three reports fit native networks to the original simulation protocols
and export HTML and Org with fitted prediction bands. Independent
NumPy predictions, densities and intervals agree with the saved runs.

- [Heteroscedastic regression](mepml/nn/12_probabilistic_and_bayesian_deep_learning/heteroscedastic_regression.mepml)
  fits the conditional mean and variance. Its held-out NLL is .702422
  versus .843726 for a training-residual pooled variance around the
  same fitted mean. The nominal 95% band contains 93.65% of test responses.
- [Split conformal prediction](mepml/nn/12_probabilistic_and_bayesian_deep_learning/conformal_prediction.mepml)
  calibrates the fixed regressor at rank 360 of 399 independent residuals.
  Marginal test coverage is 89.55%, with 93.74% on the left of the input
  range and 85.37% on the right.
- [Deep ensemble](mepml/nn/12_probabilistic_and_bayesian_deep_learning/deep_ensemble.mepml)
  fits five independent Gaussian networks and checks every held-out
  member prediction, moment and mixture log density. Mixture NLL is
  3.502677 versus 3.894182 averaged over the members.

- [Mixture density network](mepml/nn/12_probabilistic_and_bayesian_deep_learning/mixture_density.mepml)
  fits two latent response branches with a stable weighted likelihood.
  The independent check matches component parameters, probabilities,
  likelihood and all 512 conditional draws.
- [Hamiltonian Monte Carlo](mepml/nn/12_probabilistic_and_bayesian_deep_learning/hamiltonian_monte_carlo.mepml)
  uses the public fixed-step proposal API to preserve every transition.
  The independent analytic force reproduces all 14,400 decisions.
  Momentum, uniforms and acceptance flags match TensorFlow exactly.
  The maximum position difference is 2.14e-13.
- [Laplace logistic regression](mepml/nn/12_probabilistic_and_bayesian_deep_learning/laplace_approximation.mepml)
  finds the Bayesian mode and solves its Hessian for the Gaussian
  covariance. All data and Gaussian draws match the source arrays.
  The mode, derivatives, covariance and both predictive curves agree
  with the TensorFlow recording. NumPy independently verifies the
  gradient, Hessian and covariance.
- [Evidential regression](mepml/nn/12_probabilistic_and_bayesian_deep_learning/evidential_regression.mepml)
  fits a normal-inverse-gamma head by marginal Student-t likelihood
  with a residual-weighted evidence penalty. Independent calculations
  reproduce all 512 fitted parameter tuples, test losses and both
  model-implied variance components.
- [Stochastic-gradient Langevin dynamics](mepml/nn/12_probabilistic_and_bayesian_deep_learning/sgld.mepml)
  samples a fitted regression posterior with 32 chains. Independent
  algebra reproduces all 57,600 chain updates. The source data, sampled
  minibatch indices, innovations and schedule match TensorFlow exactly.
  The plots compare the sampled coefficients with the exact Gaussian
  posterior and show two chains over time.
- [Stochastic weight averaging Gaussian](mepml/nn/12_probabilistic_and_bayesian_deep_learning/swag.mepml)
  independently verifies the online means, diagonal and ring buffer,
  all coefficient draws, and 16,384 latent predictions. The trajectory,
  covariance components and predictions agree with the TensorFlow
  recording within 3.6e-7.
- [Bayes by Backprop](mepml/nn/12_probabilistic_and_bayesian_deep_learning/bayes_by_backprop.mepml)
  fits weight and activation reparameterizations with shared objective
  logic. The raw data and random draws match the reference, and NumPy
  reproduces each KL and both banks of latent predictions. The plots
  show the fitted means and the joint effect of sharing weight draws.

- [Deep kernel learning](mepml/nn/12_probabilistic_and_bayesian_deep_learning/deep_kernel.mepml)
  fits the learned embedding and RBF length scale with 24 full-data Adam
  updates. Simulated fitting and test arrays match the TensorFlow source
  bitwise. Independent NumPy Cholesky solves reproduce the fitted likelihood
  and held-out moments, and central differences check all 63 derivatives.
  The report plots 64 held-out responses with predictive intervals.
- [Sparse variational GP](mepml/nn/12_probabilistic_and_bayesian_deep_learning/sparse_variational_gp.mepml)
  fits 192 observations with 12 inducing points. The simulation matches
  TensorFlow bitwise, and the inducing posterior, joint predictions and
  collapsed bound agree within 2e-13 absolute. A dense NumPy observation-space
  calculation independently checks the predictions and the lower bound.
  The plot shows the fitted mean and pointwise 95% latent intervals.

- [Two-layer deep GP](mepml/nn/12_probabilistic_and_bayesian_deep_learning/deep_gp.mepml)
  conditions 1,024 joint hidden trajectories on eight training responses
  and reports an effective sample size of 374.72. The inputs, responses
  and base draws match TensorFlow bitwise. All mixture diagnostics agree
  within 8e-10 absolute and aggregate moments within 7e-12. Independent
  NumPy linear solves reproduce the per-draw and marginal results.

- [Conditional and attentive neural processes](mepml/nn/12_probabilistic_and_bayesian_deep_learning/neural_processes.mepml)
  fit both variants for 576 task updates with the reference task seeds and
  evaluate 24 new functions. Held-out labels match bitwise. NumPy verifies
  Gaussian mixture scores and attention normalization, and reversing context
  rows changes predictions by at most 1.2e-7. Native initializers yield held-out
  NLLs of 0.71928 and 0.68880 versus TensorFlow's 0.72894 and 0.74443.

### Compact residual CNN fits

The residual CNN candidates also ran their compact report fits. Their
upstream full fitting and test protocols remain open in the inventory.

### Grid-world environment replays

The [grid-world report](mepml/nn/6_deep_reinforcement_learning/6.1_environments/grid_world.mepml)
checks transition distributions and runs all four upstream replay policies.
The deterministic manual, risky, safe, and fall-and-recovery rollouts take
9, 13, 17, and 20 moves, with returns of -9, -13, -17, and -119. The fall
returns the agent to the start before the safe route continues. The native
CAD renderer records the actual states in four independently decoded
H.264 clips. The report's three figures and four videos reach the website.

### Tabular policy replays

The dynamic programming, Q-learning, SARSA, TD(lambda), and Monte Carlo
control reports under `6.2_tabular_and_foundational_methods` now record
the eleven original seeded greedy rollouts with the native grid renderer.
They retain the source training schedules and evaluation checks. Each
episode records the visited states from the final policy and writes them
alongside the fitted tables. An independent NumPy model checked every
recorded transition, reconstructed the cliff penalties and returns, and
verified both dynamic programming policies against their Bellman backups.
All 20 report blocks ran and exported to HTML and Org. All eleven videos
decoded with the expected frame counts and were copied into the website
export. Native replay overlays and camera timing use the Datamunge
renderer. The local upstream build attempt could not find its TensorFlow
and vsim headers, so this audit used its source protocol and the independent
transition model rather than newly generated upstream outputs.

### Bandit pull replays

The environment, algorithm comparison, and contextual policy reports now
save and render 30, 40, and 30 individual pulls. The upstream environment
and Thompson source programs ran with a trace-export sink in place of their
video recorder. Every arm and reward matches exactly. The current Thompson
source reports the one best-arm selection in the 40-pull run without a
minimum-count assertion. The contextual policy uses its existing native
initializer and fits the 187 source updates. Its 30 greedy rounds yield
26 unit rewards and retain the checks against random and oracle policies.
All twelve blocks ran, and all three exports contain decodable videos.

### Mountain Car policy replays

The discrete and continuous Mountain Car report and the linear function
approximation report now export three replays of their seeded policies.
Freshly executed upstream environment logic with a trace sink reproduces
every position, velocity and reward for the 114-step discrete pump and the
125-step continuous half-throttle controller. For the 155-step tile-coded
SARSA replay, independent NumPy code checked the greedy tile sums and every
dynamics transition against the fitted weights. The 60 training episodes
retain the upstream update schedule. All eight report blocks ran, and
all three replays decoded and appeared in the website.

### CartPole, Pendulum and Acrobot replays

The three classic-control reports now include four seeded videos. A trace
sink substituted for the upstream video writer and executed the original
policies. Every observation and reward matches the report traces bitwise.
The balancing and always-right CartPole policies last 500 and 9 steps,
the Pendulum swing-up lasts 200 steps, and the Acrobot pump reaches the
target on step 76. All twelve report blocks ran. The four videos decode
with one frame per observation and appear in the website export.

### Planning and exploration maze replays

The Dyna-Q and count-exploration reports record five final greedy rollouts
using the native grid renderer. The Dyna-Q+ route crosses the moved wall
through the only open cell on its left. The count-exploration replays
preserve the source first-seed rule, and an independent check reconstructed
every move from the saved task-only Q-tables. Each route reaches its goal
in 16 steps. All nine report blocks executed and exported to HTML and Org.
The website export includes all five videos, each checked by decoding it
and checking its frame count. This audit verified the native runs against
the source protocol and maze geometry.

### Monte Carlo tree search board replays

The MCTS report now renders all three seeded matches. All 90 moves, the six
benchmark win/loss/draw rows, and the 2,000-search root visit counts match
an execution of the upstream example with its video writer replaced by a
move sink. The native player uses a flat board with turn and outcome labels.
All five report blocks ran, and all three clips decoded and reached the website.

### Board games and environment interfaces

The board-games and environment-interface reports now record the three
seeded random games and the seeded random CartPole episode. All recorded
moves, CartPole observations and rewards, and the three 2,000-game summary
rows match fresh upstream executions with trace sinks. The vectorized
environment report matches the upstream auto-reset summaries and finishes
34 episodes over its 100-step batch. Its additional check compares 300
steps across 64 environments using one and four threads. Throughput depends
on host load. All twelve blocks ran, and the four videos decoded and
appeared in the website export.

### Snake replay

The Snake report exports the 108-step greedy episode as a video and a
PNG filmstrip from the same captured frames. Every one of the 109 board
states matches a fresh execution of the upstream policy with a board
trace sink. All five blocks ran. The video decodes and is included in
the website export.

### Rollout video recording and LunarLander

The recording report now encodes all four upstream trajectories, and the
LunarLander report also encodes its own seeded heuristic episode. Every
observation and reward in the five replays matches fresh upstream runs
using a trajectory sink. Each clip has one frame per observation and
decodes from the website export. The ten report blocks ran and exported
to HTML and Org. Datamunge uses its native renderer with ffmpeg. The
checklist item specifically naming the vsim renderer remains open.

### Self-normalizing tabular regression

The SELU report runs all 96 Adam updates and measures initial activation
moments against a ReLU control using the same weights. The probe inputs
and held-out inputs and labels match TensorFlow bitwise. Independent
NumPy forward passes reproduce activation moments and held-out predictions
from both sets of saved weights. Datamunge's native initialization gives a
held-out RMSE of 0.54820, compared with 0.37378 for the upstream fixture.
Both figures render in HTML and Org and reach the website export.

### Entity embeddings

The entity embeddings report fits two four-dimensional category tables and
a tanh regression network for 192 Adam updates. All held-out rows and
responses match the upstream fixture bitwise. NumPy independently reproduces
both models' lookups, predictions, and losses. Every embedding row updates.
Datamunge's native initialization gives 0.24657 held-out RMSE, compared with
0.23652 in the upstream fixture. The fitted scatter and category update
diagnostic render in HTML and Org and reach the website.

### Deep and cross tabular regression

The report fits two full-matrix cross layers in parallel with a two-layer
tanh tower, using all 144 source Adam updates. Held-out features and
responses match TensorFlow bitwise. NumPy independently reproduces both
models' cross states, deep states, predictions and losses from their saved
weights. Datamunge's native initialization gives 0.15565 held-out RMSE,
compared with the upstream fixture's 0.19786. Both prediction and residual
plots render in HTML and Org and reach the website.

### Neural additive regression

The report trains four univariate neural shape functions for 192 Adam
updates. The test data and curve grid match the TensorFlow fixture bitwise.
Independent NumPy evaluation reproduces both models' effects, centered
intercept, predictions and loss. The Datamunge fit reconstructs every
prediction from its centered effects within 1.71e-7. The held-out RMSE is
0.33435 with native initialization versus 0.33914 for the upstream
fixture. The centered curves also export to CSV.

### Multi-gate expert regression

The report fits the shared-bottom control and the three-expert mixture
for 192 Adam updates each, using the same batch schedule. Held-out arrays
match TensorFlow bitwise. NumPy independently reconstructs the expert
bank, routing, task representations, and predictions of both native and
upstream runs. Native initialization gives task RMSEs of 0.32204 and
0.38839 for the shared bottom, and 0.33085 and 0.26153 for the mixture.
The upstream values are 0.32858 and 0.36454, and 0.26343 and 0.27859.

### Imbalanced binary classification

The report fits four classifiers for 300 updates each with the source
threshold, prevalence, and batches. Both label vectors and all four
training index sequences match the upstream fixture exactly. The
resampled batches contain 16 positives each. NumPy independently
reconstructs the native predictions, both sets of initial losses and
confusion counts, and grouped average precision. Native initialization
changes some threshold counts. The precision-recall tradeoff retains
all four operating points in HTML, Org, and website export.

### Neural oblivious trees

The report trains 12 depth-three trees for the 192 source updates.
The held-out arrays match TensorFlow bitwise. An independent NumPy
active-set solve checks both runs' entmax selections, entmoid splits,
all leaf probabilities and fitted values. Finite differences verify the
implicit entmax derivative. Native initialization produces held-out RMSE
0.22109, compared with 0.20534 upstream. The maximum leaf mass error in
the native report is 1.19e-7.

### DeepSurv and DeepHit

The survival report trains both models for 48 full-cohort updates. The
held-out covariates, censoring times, and causes match the upstream fixture
bitwise. An independent risk-set and competing-risk calculation reproduces
both runs' Cox and DeepHit objectives. With native initialization,
concordance is 0.68887 and the DeepHit held-out NLL is 2.31736, compared
with 0.68700 and 2.35717 upstream. Both diagnostic plots export.

### Mixed-feature FT-Transformer

The report uses the original 192-update schedule and matches held-out
features and labels bitwise. Independent NumPy evaluation reproduces
initial tokenization, both attention tensors, and fitted predictions
from each backend's saved weights. Datamunge's native initialization
gives held-out RMSE 0.08974, compared with 0.08639 upstream. Both plots
render in HTML and Org and reach the website.

### Query-grouped learning to rank

The report fits RankNet, LambdaRank, and ListNet for the full 96-update
schedule. Held-out features and grades match TensorFlow bitwise. NumPy
checks the native scores, both runs' pair weights and objectives, and all
NDCG scores. Datamunge's native initialization gives NDCG@6 of 0.99449,
0.98797, and 0.99818. The upstream fixture gives 0.98953, 0.98791, and
0.99594. Both plots render in HTML and Org and reach the website.

### TabNet regression

The compact TabNet report trains for all 192 upstream updates and matches
held-out inputs and targets bitwise. Independent NumPy calculations verify
both runs' attention logits, sparse masks, decision channels, and predictions.
A central-difference check verifies the active-set sparsemax gradient.
Native held-out RMSE is 0.09231 with 81.6% zero mask entries. The upstream
fixture gives 0.18000 RMSE and 75.5% zero entries. Both figures export.

### Tabular boosting comparison

The report uses the existing exact-split regressor with squared loss and
zero regularization. Training and test arrays match upstream bitwise. All
64 boosting stages and the held-out ensemble predictions agree with the
reference to within 9e-16 absolute. Replaying the upstream tree structure
independently reproduces the stage history and predictions. The native MLP
uses all 144 source updates and achieves 0.47286 held-out RMSE versus
0.49595 upstream. The tree ensemble achieves 0.10161 in both runs.

### Tabular feature preprocessing

The report fits and freezes column-wise population moments, assigns values
at bucket boundaries to the upper bucket, and maps categories with
TensorFlow-compatible FarmHash fingerprints. All seven output arrays matched
the upstream fixture bit for bit, and the category file matched byte for
byte. NumPy independently confirmed the bucket assignments. A separate
comparison with TensorFlow's Fingerprint64 matched all 4,097 binary inputs,
one per length from 0 through 4,096 bytes. The plot rendered in HTML and Org
and was staged for the documentation website.

### Neural program induction

The interpreter selected ADD and MULTIPLY with the same reported soft
probabilities and training MSE as TensorFlow. Its decoded program matched
all 64 held-out triples exactly. The gated rollout used the same training
initialization and batches, then generalized from width 24 to 49. All
seven saved arrays agreed with the TensorFlow fixture within floating-point
rounding. A NumPy recurrence independently reproduced the logits and
4,462 of 4,704 correct predictions. The held-out BCE plot rendered in HTML
and Org and was staged for the documentation website.

The remaining miscellaneous example is GPU NeRF. Reuse the Fourier feature
and device-transfer APIs, add reusable differentiable ray compositing, and
verify the two matrix products on the GPU as well as the novel-view RGB
fixture.

### CUDA neural radiance field

The report ran with both CPU and CUDA matrix products. For the CUDA run,
the executable linked the checkout's CUDA backend against the installed
core. The full Git-flake package build excluded pre-existing untracked
headers referenced by the worktree. The runnable block reported two CUDA
products on the RTX 5080 Laptop GPU. GPU and CPU runs both reproduced
the TensorFlow capture's held-out MSE of 0.0736832 to 0.0679625 and
PSNR of 11.6773 dB. All eight saved arrays matched the TensorFlow fixture,
with a maximum absolute difference of 2.39e-7. NumPy independently
recomputed the novel-view colors and terminal transmittance. The fit panel
and loss curve exported to HTML and Org and staged for the website.

### Foundations mixed precision

All five report blocks ran and all five figures exported. Native CUDA
Float32 and binary16 runs each applied 400 Adam steps. Their MSEs were
7.02e-14 and 1.0525e-7, respectively. The CPU bfloat16 path met the
upstream convergence threshold. Scaling recovered a derivative of
9.99716576e-9, matching the upstream CPU check, and a forced overflow
left already-initialized Adam moments, master parameters, and the
iteration count unchanged. NumPy independently verified both native
forward predictions. The upstream TensorFlow CUDA executable still
failed in a Cast kernel with CUDA_ERROR_INVALID_HANDLE. The Datamunge
native CUDA path ran independently. All foundations examples are verified.

### Frozen affine GraphDef

The frozen-graph report fits the five upstream training observations with
500 Adam steps and exports a standard five-node GraphDef with two embedded
constants. Its input and output signature matches the upstream signature
byte for byte. Reloading after live weight mutation preserves predictions
exactly, and a single-row batch returns 9.5. Datamunge loads the upstream
artifact, and an independent TensorFlow C API check imports both artifacts
and reproduces predictions on batches of sizes three and one without
initialization. The report's diagram and fit exported to HTML and Org.
The combined graph-freezing and constant-folding checklist row is covered
by this report and the graph-optimization report.

### Python-trained SavedModel inference

The report loads the upstream SavedModel and variable checkpoint through
the optional TensorFlow C++ SDK backend. The local fixture's graph,
checkpoint, and reference table match the upstream files byte for byte.
The C++ session resolves the named tensors through the serving SignatureDef.
The four-row predictions match the exported Python values within 1e-6,
and a single row predicts -7.25 within 1e-5. Wrong width, wrong dtype,
and an unknown signature are rejected. All five runnable blocks and
both figure exports passed. The next persistence example uses the
converted TFLite artifact from the same training script. Its report
needs a LiteRT session that reuses the compiled model and buffers.

### TensorFlow Lite inference

The LiteRT report compiles the real converted model on CPU and reuses
the runtime's 8-byte input and 4-byte output buffers. All five predictions,
including the repeated first row, exactly match the exported Python
references. A malformed FlatBuffer is rejected. Two concurrent callers
sharing the compiled model passed 100 total invocations with independent
prediction values. Regenerating the model with the upstream Python
conversion script reproduced both the bundled 1,044-byte artifact and
reference table byte for byte. The four report blocks ran and both
figures exported to HTML and Org. All five persistence and
interoperability examples are now verified.

### CPU graph optimization

The same elementwise fixture ran under baseline, Grappler, and XLA
policies using Datamunge's optional TensorFlow C++ backend. The executed
partitions contained 18, 16, and 6 nodes, respectively, matching the
upstream capture. Grappler and XLA folded the fetched scalar multiplication
to Const. Only the XLA partition contained an XLA executable. All 65,536
values per mode matched a double-precision scalar reference within 3e-7.
The chart compares warmed call medians including tensor copies. In this
capture, Grappler had the lowest median and XLA had the largest first-call
cost. The report ran and the chart exported to HTML and Org.

### Low-rank dense factorization

The report regenerated the upstream rank-three-plus-noise weight matrix
and held-out features. At ranks 1, 2, 3, and 5, Datamunge's reconstructed
weights and predictions matched TensorFlow and an independent NumPy SVD
within 1e-12. The largest prediction difference was 7.82e-14. Comparisons
used factor products because equivalent singular vectors can have
opposite signs. The discarded singular energy matched the squared
Frobenius residual. The storage/error plot exported to HTML and Org.

### CPU inference batching

The report uses the upstream 24-to-64-to-6 network and deterministic
weights. For all five batch sizes, all 128 predictions matched repeated
single-row inference exactly and agreed bit for bit with an independent
TensorFlow run. The timing example verifies the warm-up and measurement
call counts. The median, p95, and aggregate-throughput figure exported to
HTML and Org. Timing values are specific to the host and native runtime.

### Signed hashed weights

The report trains eleven shared coefficients for 36 connections. The
assignment, signs, and data match TensorFlow exactly. Learned values and
predictions agree within 4.45e-16. NumPy reproduced training independently
with an explicit signed scatter-add, and the held-out MSE fell from
0.5271054 to 0.001542223. The loss and occupancy figure exported to HTML
and Org.

### Structured convolution pruning

Selecting channels 1, 5, and 0 by L1 norm reduced the two convolutions
from 126 parameters to 63. The compact and masked versions agreed
within 4.45e-16. All eleven saved arrays matched the TensorFlow
fixture within 9e-16, and NumPy independently reproduced the L1
ranking and compact convolution outputs. The fit comparison exported
to HTML and Org.

### Magnitude pruning and lottery tickets

The report trained all six upstream conditions. All 21 saved arrays
matched the TensorFlow fixture, with exact masks and differences within
floating-point rounding for trained weights. NumPy independently
reproduced the eligibility masks, SGD, and held-out MSE. At 28 active
weights, iterative pruning gave MSE 0.01950 and one-shot pruning
gave 0.02326. Rewinding the iterative mask to its original weights
gave 0.04654, compared with 0.16247 after fresh initialization and
0.45591 for the random mask. The comparison plot exported to HTML
and Org. The next compression example needs symmetric int8 calibration,
integer accumulation, and a clipped straight-through training path.

### Symmetric int8 quantization

The report reproduced float and fake-quantized training, calibrated both
scales on training inputs, and tested on a wider input range. Scales,
integer codes, dot products, and dequantized predictions matched
TensorFlow exactly. Trained float weights agreed within 1.12e-16.
NumPy independently reproduced both training paths and integer inference.
The held-out int8 MSE was 0.00063375 for post-training quantization and
0.00062598 for quantization-aware training. The report's error plot
exported to HTML and Org.

### Knowledge distillation

The report reproduces the larger teacher, plain student, logit student,
feature-hint student, and same-size self-distillation run. All 19 arrays
matched TensorFlow within 9e-16. NumPy independently evaluated the
trained weights and reproduced all five accuracy counts. The plain
student scored 99/128, logit distillation 103/128, feature matching
102/128, and self-distillation 98/128. The accuracy plot exported
to HTML and Org.

### TensorFlow CPU runtime profiling

The report runs the two independent matrix products with 1/1, 2/2,
and 4/2 intra/inter-op thread settings, timing untraced calls after
warm-up. Each separate full trace contains both named products and
seven events, matching the upstream captures. All 65,536 result
entries per setting matched independent Float64 NumPy evaluation
exactly. Python parsed the three Chrome JSON traces and checked
timestamps, nonnegative durations, and required operation names.
The untraced latency plot exported to HTML and Org.

### SavedModel over loopback HTTP

The report loads the actual Python-trained SavedModel and uses Libevent
to serve its signature on an ephemeral IPv4 loopback port. Its curl
round trip matched both direct predictions exactly and verified 400
and 200 responses for malformed input and health checks. An independent
Python HTTP client matched all four exported Python predictions,
rejected nonfinite features, and verified an unknown route returns 404.
The report's prediction agreement plot exported to HTML and Org.

### CUDA binary16 training

The native CUDA report transfers Float32 host tensors, packs binary16
operands on the GPU, accumulates products in Float32, and rounds back
to binary16. Both forward and transposed backward products ran on
the RTX 5080 Laptop GPU. The upstream TensorFlow GPU example reran
successfully and matched the report's held-out MSE of 0.253055 to
0.00148284 at its printed precision. An independent NumPy binary16
replay reproduced every saved epoch weight and prediction bit for bit.
The held-out loss curve exported to HTML and Org. All 11 model
compression examples are verified. The foundations mixed-precision report also exercises this API with
dynamic loss scaling.

### Rolling-origin forecast evaluation

The report evaluates persistence and seasonal naive at 187 chronological
origins and six leads. Origins, targets, point forecasts, scales, and
residual draws matched TensorFlow exactly. MASE, sMAPE, and CRPS agreed
within 6e-14. NumPy independently replayed all scores with the full
pairwise CRPS sum. A future-target perturbation left point forecasts
and draws unchanged. The lead-wise plots exported to HTML and Org.

### Point forecast baselines

The baseline report compares persistence, seasonal naive, a six-step
trailing mean, and a four-lag ridge autoregression on first differences.
The fit uses only observations 0–767. The upstream NumPy audit reproduced
all 4,488 forecasts at 187 origins. Forecasts matched the TensorFlow
fixture within 2.52e-13, with identical observations and origins.
A future-target perturbation preserved the coefficients and forecast.
The recurrence had the lowest MAE at leads one through four, and
seasonal naive led at five and six. Both panels exported to HTML and Org.

### Probabilistic forecast losses

The report fits ordered quantiles and a conditional Gaussian to the same
four-lag windows. The 192 test targets match TensorFlow exactly, and the
maximum absolute prediction differences are 1.61e-6 for pinball and
1.79e-7 for Gaussian. The upstream NumPy audit replayed 576 pinball
terms and 192 Gaussian terms. The interval coverage counts are identical
to TensorFlow at 141 and 155 of 192, respectively. Both prediction panels
exported to HTML and Org.

### DLinear long-horizon forecasts

The report trains separate trend and seasonal projections over 96-sample
windows. The upstream NumPy audit reproduced all 9,312 predictions from
the saved decomposition and weights, and independently checked both
naive baselines. Datamunge and TensorFlow produced identical test trend
and seasonal arrays and differed by at most 1.32e-6 in their normalized
predictions. Changing the first test origin's future targets preserved
its forecast. Both plots exported together to HTML and Org.

### Direct MLP, CNN, and TCN forecasts

A shared report trains all three upstream architectures against the same
chronological split using the existing convolution, dense, and causal
TCN APIs. For each model, all 1,122 fitted test predictions agree with
the corresponding TensorFlow table within 1.44e-6. Observations and both
naive baselines agree exactly. Changing post-origin targets preserves
all three fixed-origin forecasts. The TCN also passes all 24 tests of
causality inside its observed window. Both plot panels export to HTML and Org.

### Offline neural change-point localization

The report fits a shared boundary-scoring MLP and a learned no-change
score. The upstream NumPy audit reproduced the fitted logits and
probabilities. Test input and labels matched TensorFlow exactly, and
probabilities differed by at most 4.18e-7. All 160 changes were located
within two samples, with zero false alarms among 80 unchanged series.
The held-out traces and boundary probabilities export to HTML and Org.

### LSTM time-series anomaly scoring

The report trains on clean histories and calibrates both scores on clean
validation windows. The upstream NumPy audit replayed the two scores and
calibration thresholds, and verified that next-observation spikes left
the network outputs unchanged. TensorFlow and Datamunge produced identical
flags at each test origin. The maximum output difference was 1.69e-5.
The fitted trace and both score panels exported to HTML and Org.

### Directed graph time-series forecasts

The report fits node-local and three-hop shared heads on six synthetic
sensors. The upstream NumPy audit independently replayed all 6,840
predictions, verified node permutation equivariance, and evaluated
reversed-edge inference. Histories, targets, adjacency, and diffusion
features matched TensorFlow exactly. Predictions differed by at most
1.20e-7. The fitted trace and error-by-lead panels exported to HTML and Org.

### Differentiable scalar Kalman filter

The report learns process and measurement variance from observations
alone. The upstream NumPy audit independently replayed all 4,096 test
updates and checked both raw-variance likelihood gradients by central
differences. Datamunge's filtered states match TensorFlow within
2.39e-7 and the gradients within 7.69e-9. The fitted trace and training
likelihood exported to HTML and Org.

### N-BEATS learned basis forecasting

The report fits three generic residual blocks and checks fixed-origin
invariance to changes in future samples. The upstream NumPy audit verified
both backcast subtractions and all three forecast contributions at all 187
origins. All 1,122 predictions in original units match TensorFlow within
1.37e-6, and the block traces within 2.39e-6. The three panels exported to
HTML and Org.

### N-HiTS multi-resolution forecasts

The report trains three residual stacks and verifies invariance to
unobserved future values at a fixed origin. The upstream NumPy audit
replayed pooling, interpolation, residual updates, and all 1,122 test
predictions. Forecasts in original units match TensorFlow within
6.39e-7 and stack traces within 1.44e-6. All three panels exported to HTML
and Org.

### PatchTST overlapping history patches

The report fits a univariate two-head encoder and checks fixed-origin
invariance to future changes. The upstream NumPy audit reconstructed
all 1,309 observed patches and checked the attention distributions.
All 1,122 fitted forecasts match TensorFlow within 1.05e-6, and the
encoder traces within 6e-7. Both panels exported to HTML and Org.
The combined Informer, Autoformer and PatchTST checklist row remains
open for the other two architectures.

### Autoregressive GRU forecasting with additive attention

The report fits the complete free-running decoder and checks that
changing unseen future values preserves a fixed-origin forecast. All
1,122 test predictions match TensorFlow within 4.56e-7. The upstream
audit verified all 26,928 observed-lag attention weights, including
normalization and variation across leads and origins. The weights
match TensorFlow within 2.62e-8. Both panels exported to HTML and Org.

### DeepAR Gaussian path forecasts

The report fits a shared GRU across three related series. The upstream
NumPy audit checked 71,808 Gaussian draws, recurrent feedback, conditional
likelihood, and coverage. The draws match TensorFlow within 3.29e-6 and
both runs give 38.95% coverage for the nominal central 80% interval.
The first 48 six-step intervals and lead-wise coverage exported to HTML
and Org.

### Irregular-time ODE-RNN forecasting

The report fits a shared autonomous field and GRU observation updates.
The upstream NumPy audit replayed all 1,536 integration intervals and
the final-query delay experiment. Inputs and targets match TensorFlow
exactly. All 512 forecasts match within 1.56e-6, and the hidden traces
within 3.04e-6. The held-out irregular-time plot exported to HTML and Org.

### Autoformer circular delay aggregation

The runnable report fits the upstream compact architecture. The independent
NumPy audit verifies both decoder seeds, the top-three lag scores and
mixtures for all three attention layers, and the trend-plus-seasonal
forecast at all 187 test origins. The trained Datamunge forecasts have
RMSE 0.621240 versus 0.607907 in TensorFlow, with different selected
lags. This mapping verifies the architecture and forward calculations.
Both panels exported to HTML and Org. The combined Informer, Autoformer,
and PatchTST checklist row still needs Informer.

### Informer sparse queries and encoder distillation

The report fits both sparse encoder stages and the causal decoder.
The upstream NumPy audit verifies query selection, replacement of
mean context, pairwise max pooling, the decoder seed, and the causal
mask at all 187 origins. Datamunge reached test RMSE 0.407892 versus
0.426594 in TensorFlow. The fitted runs select different active
queries. Both report panels exported to HTML and Org. Informer
completes the combined checklist item with Autoformer and PatchTST.

### InceptionTime and ROCKET time-series classification

The report trains the compact residual classifier and fits a ridge head
on the frozen random-kernel features. The upstream NumPy audit replayed
the trained convolutional model, all ROCKET features, and the fitting-only
ridge solve. ROCKET features match TensorFlow exactly, with scores within
1.51e-13. InceptionTime logits match within 8.29e-6. Both models predicted
all 90 held-out classes correctly. This completes the example mappings in
the time-series and forecasting chapter.

### DenseNet, dual-path, and highway networks

Compared each report with its upstream source and built the three
upstream targets. DenseNet reproduced the feature ordering and direct
gradient checks, the projected average-pooling transition, and canonical
BC-121 and BC-100 shapes and counts. DPN reproduced residual and dense
path arithmetic and gradients and the released parameter counts.
Highway networks reproduced the coupled gate equations and derivatives,
the 100-layer attenuation fixture, and the full MNIST architecture.

All 24 Datamunge blocks ran. The reports use bounded fits with separate
validation and official-test subsets. NumPy reconstructed their logged
confusion matrices from held-out logits and labels. DenseNet classified
20/64, DPN 2/32, and the highway model 83/128 correctly on their respective
subsets. All 21 figures passed HTML, Org, and website staging checks.

### Pre-activation and Wide ResNet

The pre-activation report checks signed identity mappings and their
derivatives, preactivated projections, and a nonzero residual against
finite differences. It verifies the canonical 164- and 1001-layer
budgets and executes backward passes at reduced width. The Wide ResNet
report checks residual-branch dropout, signed shortcuts, parameter and
MAC counts, and paired width and dropout fits. Both upstream targets
compiled. All 17 report blocks ran and exported their figures. The
saved held-out labels match the official CIFAR-10 binary records.
Independent confusion replay found 2/32 correct for the pre-activation
fit and 18/64, 13/64, and 12/64 for the three Wide ResNet fits.

### Bottleneck ResNet, ResNeXt, and Res2Net

The three upstream targets compiled. The bottleneck report checks the
ResNet-50, -101, and -152 stage counts and projected shortcuts, and
fits all three depths. ResNeXt compares grouped convolution with
explicit split and aggregation, including parameter and input
derivatives. Res2Net checks ordinary and stage blocks against a host
reference and verifies the four receptive fields. The reports run full
shape and parameter checks separately from their bounded gradient
and fitting demonstrations. All 23 blocks ran and the figures exported.
Independent replay confirmed official CIFAR-10 labels and reconstructed
the confusion matrices. The three bottleneck fits classified 4, 2,
and 5 of 32 correctly. ResNeXt scored 7/32, and Res2Net scored 9/64.

### PyramidNet, RevNet, and stochastic depth

The three upstream targets compiled. PyramidNet verifies the rounded
width schedule, signed pooled shortcuts and gradients, and the full
PyramidNet-272 gradient structure. The report now fits the upstream
PyramidNet-110 alpha-48 architecture on its bounded CIFAR-10 split.
RevNet checks inversion, reconstruction drift, and parameter gradients
against ordinary backpropagation. Its memory plot measures tracked live
tensor buffers, whereas the upstream example measures process RSS.
Stochastic depth checks both scaling conventions, survival schedules,
mask behavior, and full-depth gradients, then fits paired configurations.

All 23 blocks and 21 figure exports passed. Independent replay confirmed
official CIFAR-10 labels and all confusion counts. PyramidNet classified
10/64 correctly, RevNet 16/64, and the stochastic-depth comparison 20/64
at constant depth and 16/64 with stochastic depth. The residual and dense
families chapter now has 12/12 verified example mappings.

### MobileNet v1, v2, and v3

All three upstream targets compiled. The Datamunge reports check v1's
depthwise factorization and width and resolution budgets, v2's linear
projection and signed inverted residual, and v3's hard activations and
squeeze-and-excitation, including both Large and Small. Each includes
complete-width parameter and gradient checks and bounded CIFAR-10 fits.
All 28 blocks ran. The saved labels match official CIFAR-10 records, and
independent confusion replay gave 11/64 and 6/64 for v1, 10/64 for v2,
and 13/64 and 6/64 for v3. All figures exported to HTML, Org, and the site.

### ShuffleNet, GhostNet, EfficientNet, and MnasNet

The five upstream targets compiled. ShuffleNet verifies channel
permutations, grouped connectivity, and both released unit designs.
GhostNet checks the depthwise-generated features and measured compute
ratio. EfficientNet covers B0–B7 compound scaling and inverted
drop-connect, and EfficientNetV2 contrasts fused and depthwise blocks
and verifies the released budgets. MnasNet checks its decoded block
tables and five width/model budgets. The Datamunge reports evaluate
complete parameter budgets and run gradient probes at reduced spatial
resolutions. All 42 blocks and 34 figures passed. Independent
confusion replay verified the official labels and found 19/64 and
11/64 for ShuffleNet, 2/32 for GhostNet, 2/32 for EfficientNet, 2/64
for EfficientNetV2, and 12/64 for MnasNet on their bounded test sets.

### RegNet and CondenseNet

Both upstream targets compiled. RegNet checks quantized stage widths,
group compatibility, released parameter budgets, and the complete
Y-400MF training graph. CondenseNet checks magnitude-based mask updates,
the packed inference layer, full parameter budgets, and condensation
during training. Their 17 report blocks and 14 figures passed. NumPy
reconstructed the confusion matrices and verified official test labels.

### SqueezeNet

The upstream target compiled. The Datamunge report checks the Fire
module's equivalent centered-filter convolution and input gradients,
ceil-rounded pooling, and both released width budgets. It fits v1.0,
v1.0 with simple bypasses, and v1.1 on the same split. NumPy verified
all 30,000 official-test labels and reconstructed all three confusion
matrices, with 4,425, 1,000, and 4,651 correct, respectively. The ten
code blocks and report figure exports passed. The efficient and mobile
architectures chapter now has 11/11 verified example mappings.

### CBAM, non-local attention, and ConvNeXt

The upstream targets compiled. The CBAM report verifies sequential
channel and spatial gates against host arithmetic. The non-local
report checks all four pairwise functions with and without subsampling,
identity initialization, and receptive fields. ConvNeXt verifies the
layer-scale residual and 7-by-7 support, and ConvNeXt V2 checks GRN
and its rescaling behavior. Both ConvNeXt reports check released
parameter budgets and use reduced-width fits. All 31 blocks ran and
their figures exported. Independent replay confirmed the saved official
CIFAR-10 labels and confusion counts for the four bounded fits.

### Capsule networks and ECA-Net

The two upstream targets compiled. The capsule report checks squash,
routing against a host computation, margin loss, decoder parameter
counts, and all three routing iterations. ECA checks channel-local
gating and adaptive kernel widths against analytic references. All
16 Datamunge blocks ran and exported. Independent confusion replay
validated the 32 official MNIST labels and 64 official CIFAR-10 labels.
The bounded fits classified 24 and 16 of those records, respectively.

### Deformable and octave convolutions

Both upstream targets compiled. Deformable convolution verifies its
zero-offset and initial modulation limits, integer and fractional
sampling, and offset derivatives, then fits both variants. Octave
convolution checks all four streams, alpha-zero equivalence to
ordinary convolution, and operation budgets across alpha values.
All 19 blocks ran and figure exports passed. Independent replay
checked all three official CIFAR-10 label arrays and confusion
matrices. The deformable variants classified 13/64 and 4/64;
Oct-ResNet classified 16/64.

### RepVGG, SENet, SKNet, and spatial transformers

All four upstream targets compiled. RepVGG checks block fusion and
full-model deployment, including inference from the selected trained
checkpoint. SENet checks the channel gates against host arithmetic,
and SKNet checks normalized branch weights and receptive-field
derivatives. The spatial transformer report checks identity,
translation, rotation, and finite differences, then compares baseline
and transformer fits on distorted MNIST. All 33 blocks ran and their
figures exported. Independent replay verified the original test labels
and confusion matrices for all five fits. This completes the 12 example
mappings in the attention and modern ConvNet chapter.

### Detection baselines, two-stage heads, and set and keypoint detectors

Seven upstream targets compiled and 53 Datamunge code blocks ran.
The baseline report compares sliding windows and classical proposals.
Fast and Faster R-CNN check shared feature pooling, with the latter
also training its RPN. DETR checks rectangular matching and per-query
losses. CenterNet and FCOS check target assignment, localization and
decoding. EfficientDet checks normalized BiFPN fusion and its gradients.
The reports use bounded synthetic digit-scene fits. An independent
NumPy pass replayed the eight test prediction sets and matched integral
AP50, 101-point AP75, and AP50:95 within 2e-7. All 46 figures exported.

### Real VOC and COCO box evaluation

The upstream preparation script fetched and checked the dataset archives,
published detector, and COCOeval. The TensorFlow target ran SSD MobileNet-v2
on eight seeded images from each dataset. Its scores matched the independent
VOC and official COCO references within 5e-13. The Datamunge report now
documents preparation and evaluates the same 1,600 exported detections.
Each metric matched the upstream result within the six-decimal output
precision. VOC integral AP50 was 0.766265 and VOC07 AP50 was 0.759431.
COCO AP50, AP75, and AP50:95 were 0.555824, 0.331552, and 0.345538.
All five executable blocks and four figure exports passed.

### FPN, R-CNN, RetinaNet, SSD, and YOLO

All eight upstream targets compiled, and all 71 report blocks ran.
FPN checks top-down fusion and RoI routing. R-CNN compares classical
proposals with and without box regression. RetinaNet and SSD verify
anchor matching and losses. The YOLO reports check successive
responsibility and anchor rules, WordTree partial-label gradients,
multiscale outputs, and CSP, SPPF, PAN, and mosaic geometry.
The plots reflect bounded synthetic fits. The independent AP replay
matched all ten exported tests within 2e-7, and all 62 figures passed
export checks. The object detection chapter now has 17/17 verified
example mappings.

### Segmentation metric replay

After running the segmentation reports, run
`python scripts/check_nn_segmentation_metrics.py` to replay the saved
test outputs without Datamunge. Pass report basenames to check a subset.
The check reconstructs semantic pixel confusion, pixel accuracy and mean
IoU, binary voxel confusion and mean-volume Dice and IoU, and integral
instance-mask AP. It checks that each report supplies the expected number
of prediction archives. Panoptic FPN's export covers its semantic and
instance scores. The report itself checks the panoptic merge and PQ on
controlled fixtures.

The upstream segmentation targets compiled. The full independent replay
agreed with all 19 regenerated prediction archives. The report executions
and figure exports below record which checks support the coverage entries.

PSPNet is verified. Its nine blocks ran, including independent
adaptive-pooling values and input gradients on odd grids and the
checkpoint-selected auxiliary-head fit. Replaying 147,456 held-out
pixels gave accuracy 0.896837 and mean IoU 0.580204. The regenerated
figures passed HTML, Org, and site export checks.

SegNet is verified. Its nine blocks ran, including the pooling winner
and sparse unpool derivative checks and a validation-selected fit.
Replaying 147,456 held-out pixels gave accuracy 0.805739 and mean IoU
0.311656. Figure exports passed in HTML, Org, and website staging.

Mask R-CNN and HRNet are verified. Their 18 blocks ran, including
RoIAlign border and derivative checks, matched mask-channel supervision,
and HRNet's cross-resolution fusion. Replaying the regenerated artifacts
gave mask AP50 0.771561 and AP75 0.407407 for Mask R-CNN, and mean IoU
0.497492 for HRNet. Their figures passed all export checks.

Panoptic FPN is verified. Its ten blocks ran, including the panoptic
merge and PQ fixtures. The independent replay reproduced semantic mean
IoU 0.732621 and integral mask AP50 of zero. The fitted report shows
the difference between material segmentation and instance recognition.
The regenerated figures passed export checks.

DeepLab V3 and V3+ are verified. All twelve blocks ran, including
atrous spatial pyramid pooling, the low-level decoder path, and both
fits. Replaying the held-out masks gave mean IoU 0.412291 and 0.509020.
All figures passed export checks.

The DeepLab V1 and V2 report ran all fifteen blocks and exported fourteen
figures. Independent pixel replay gave mean IoU 0.509540 for V1 and
0.448361 for V2 across 147,456 held-out pixels per fit.

The FCN report ran all fifteen blocks and exported fourteen figures.
The replay matched mean IoU of 0.364558, 0.627222, and 0.652889 for
FCN-32s, FCN-16s, and FCN-8s, respectively. The report's comparison
now reflects these regenerated fits.

### Angular margins, Siamese verification, and multi-label classification

The angular-margin report checked the ArcFace and CosFace target-logit
transformations and fitted both models. Independent confusion replay
matched 607/640 and 582/640 held-out classifications. The Siamese
report checked shared gradients and fitted on MNIST with disjoint
classes for evaluation. Its 512 held-out pair decisions included
256 true negatives, five true positives, and 251 false negatives.
The multi-label report fitted on actual COCO128 photos and calibrated
its decision threshold on validation data. Replaying all 1,920
held-out label decisions gave six true positives, 110 false positives,
60 false negatives, and 1,744 true negatives, for positive micro F1
of 0.065934. All 33 executable blocks ran and all 29 figures passed
HTML, Org, and site export checks.

### Bilinear CNN, FaceNet, CRNN with CTC, and DnCNN

All 41 executable blocks ran and all 35 figures passed HTML, Org, and
site export checks. Independent confusion replay matched 10/48 flower
classifications and 323/512 FaceNet pair decisions. The flower report
now uses the regenerated checkpoint results. Independent greedy CTC
decoding returned empty strings for all six held-out human-annotated
word crops, for 27 edits in 27 characters. Recomputing raw per-image
PSNR for DnCNN gave 19.723765 dB for the learned fit, 20.219740 dB for
the noisy input, and 24.034324 dB for the reflected mean filter. The
denoising report now describes that fit and its moving-BN probe values.

### U-Net and YOLACT

Both reports ran all nine blocks and exported eight figures each.
The U-Net replay matched held-out mean IoU of 0.664712. The YOLACT
replay matched zero integral mask AP over the reported thresholds.
The report shows both the prototype mechanism and the weak held-out
instance predictions from the short fit.

### Inpainting, optical flow, and pose

Context Encoder, partial-convolution inpainting, FlowNetS, PWC-Net,
OpenPose PAFs, and stacked hourglass ran all 71 blocks and exported
61 figures. Independent replay matched hole MSE of 0.051181 for
Context Encoder and 1.225842 for the raw partial-convolution composite.
The known regions were preserved to within 1.5e-8. FlowNetS and PWC-Net
reproduced mean endpoint errors of 4.246750 and 3.946422 pixels.
The stacked-hourglass coordinate export gave mean error of 61.741773
pixels over 116 annotated joints. OpenPose assembled zero held-out
person groups in the regenerated run, and its prose now matches the fit.

### Batch-hard triplet loss

The [new report](mepml/nn/7_graph_neural_networks_and_representation_learning/7.3_metric_learning/batch_hard_triplet.mepml)
uses the existing high-level miner and hinge loss to fit an embedding.
Held-out hinge fell from 5.555956 to zero, and all 24 held-out anchors
satisfied the full margin. NumPy independently reproduced the mined
triplets and the initial and final embedding gradients within 2e-6.
The fitted gap and loss figure passed HTML, Org, and site export checks.

### U-Net variants

The U-Net++ and attention U-Net report ran all thirteen blocks and
exported twelve figures. Replaying 36,864 held-out pixels per model
matched mean IoU of 0.497570 and 0.472300, respectively.

### Gatys style transfer, colorization, and image retrieval

All 32 blocks ran and all 29 figures passed export checks. The Gatys
output archive reproduced the mean absolute pixel change of 0.085390.
The colorization archive reproduced mean clipped RGB PSNR of 24.258774 dB.
The retrieval report verified exact rank and distance parity after gallery
serialization. Its export contains 96 distinct source IDs with unit-norm
descriptors. The regenerated checkpoint reached held-out mAP of 0.402526.

### Volumetric segmentation

All thirteen blocks and twelve figure exports passed. Replaying 32,768
held-out voxels per fit yielded mean-volume Dice of 0.317169 and
0.991557 and mean-volume IoU of 0.190854 and 0.983311. All thirteen
upstream examples in the segmentation chapter now have verified mappings.

### Fast style transfer, monocular depth, and super-resolution

All 40 executable blocks and 36 figure exports passed. Replaying the fast
style outputs reproduced the mean absolute change of 0.127584 across
distinct content inputs. The fixed sensor masks reproduced refined
depth RMSE of 1.388465 m across 16 rooms. Independent central-luminance
replay matched the five super-resolution PSNR values and the common
bicubic score of 26.711192 dB.

### Visual question answering

All twelve blocks and ten figure exports passed. On 89 held-out questions,
independent replay matched the 64-class mapped confusion table. The replay
found 21 exact answers with known majority labels, matching the report's
21/89 diagnostic. Another 32 diagonal entries came from the unknown-answer
bucket. The report separately computes human-answer consensus, which was
0.308989 for the fitted joint model.

### Next metric-learning work

- Extend the normalized angular classifier for SphereFace's monotone
  four-branch continuation and annealing. Check the target logits and
  derivatives against an independent angle calculation, then fit a
  matched experiment with ArcFace and CosFace.
- Have the Datamunge DeepCluster report export initial and fitted
  gallery/query descriptors in the format consumed by retrieval evaluation.
  The retrieval example currently verifies against the upstream archive.

### C3D, R(2+1)D, P3D, and I3D

The three reports ran all 37 blocks using the person-disjoint KTH clip
fixtures. Independent replay reproduced the four held-out confusion
matrices and NLL values. C3D, R(2+1)D, P3D, and I3D classified 2/6, 2/6,
3/6, and 2/6 clips correctly. The C3D and factorized-video discussions
now use the regenerated checkpoint results. All figures passed HTML,
Org, and site export checks.

### PointNet and PointNet++

The two reports ran 22 blocks on sampled original meshes and exported
20 figures. Independent replay matched 4/12 held-out classifications
for each model and NLL of 1.508108 and 1.112266. Both checkpoints label
all held-out objects as chairs. The report discussions now use these
regenerated results.

### Medical volume classification

All eleven blocks ran on the NoduleMNIST3D teaching subset and all ten
figures passed export checks. The held-out confusion replay gave 37/64
correct predictions and NLL 0.677774. The report now reflects the
regenerated comparison with the block 1-NN reference.

### Exact embedding retrieval evaluation

The [report](mepml/nn/7_graph_neural_networks_and_representation_learning/7.3_metric_learning/retrieval_evaluation.mepml)
uses the exact descriptor index with a new cutoff API for hit rate and
macro item recall. A tie and missing-category fixture and the upstream
initial and trained DeepCluster archives all ran. Independent exhaustive
ranking matched both metrics at ranks 1, 3, and 5 to within 1e-12.
The trained descriptors reached a hit rate of 1.0 at rank 5 across 96
eligible queries, with macro item recall of 0.074870. The figure passed
HTML, Org, and site export checks.

### Sequence-to-sequence reversal and addition

Both blocks ran and the loss figure passed HTML, Org, and site export
checks. Independent replay checked the exact answers for all 192 unique
inputs per task and reproduced masked test token NLL of 2.085723 for
reversal and 1.820951 for addition. Both free-running decoders scored
0/32 exact matches in the short run. This verifies the upstream toy
example. Attention and translation remain separate checklist entries.

### Monophonic MIDI generation

The replacement MIDI report fits the 26-token note-duration/rest LSTM
for 48 Adam updates. It reproduces the upstream corpus and all 64
sampling uniforms exactly. Independent replay checked every generated
token, the test NLL of 2.374941, and all 59 note intervals and six rests in
the emitted format-0 file. Teacher-forced and streaming logits agreed.
The validation and piano-roll figure passed HTML, Org, and site export
checks. The combined MIDI LSTM/music-transformer checklist row remains
open until both mappings have been verified.

### Sequence and transformer gaps found in the execution audit

Fresh runs passed for BERT MLM, BERT fine-tuning, sentence embeddings,
Switch Transformer, Transformer-XL, DINO, and ViT. Keep these mappings
open while aligning the current source workflows.

| Reports | Work to match the current source |
| --- | --- |
| BERT MLM, fine-tuning, sentence embeddings | Match the two-block, 24-channel encoder, 64-character windows, and per-update mask resampling. Carry the resulting encoder checkpoint into all three fine-tuning heads and the symmetric sentence-retrieval objective. |
| Switch Transformer | Use the pinned bidirectional masked-character task. Compare accepted and overflow slots and the pre-capacity load-balancing loss against the exported routing fields. |
| Transformer-XL | Match detached two-layer caches and next-character likelihood for both segments, with a memory-off evaluation on the same test rows. |
| DINO | Replace the four RGB views with the upstream MNIST global/local crops. Replay the first student update, teacher EMA and centering, then compare before/after feature exports on the pinned row IDs. |
| ViT | Use official MNIST and pinned split indices. Verify patch order, single/batch parity, and held-out likelihood and predictions. |

### Sparse adjacency propagation

The rectangular COO report uses the new differentiable multiplication
API. Its entries and forward, transpose, and packed-feature results
matched the upstream archive within 1e-6. Independent dense calculations
reproduced the three paths and the six edge-weight derivatives within
1e-12. The feature adjoint matched automatic differentiation. The heatmaps
passed HTML, Org, and site export checks. The combined Cora GCN checklist
row remains open pending a verified Cora model.

### DQN, Double DQN, and C51

These three CartPole reports ran all 18 blocks. Independent checks matched
the worked bootstrap and projection targets and inspected the saved online
and target parameters and terminated episodes. All bounded runs reached
512 environment steps. The final logged greedy return was 16 for DQN,
10 for each branch of the Double DQN comparison, and 51 for C51.
All 15 figures passed HTML, Org, and site export checks. The reports use
a shorter default fitting budget than the source examples.

### Semi-supervised graph convolution

The report fits the two-layer GCN on the exact upstream 48-node graph,
features, and label split. The new dense normalization API adds self-loops
and applies symmetric degree scaling. Independent replay matched the
normalized matrix, all fitted logits, and all 74 gradients from the
16 fitting labels. Held-out NLL dropped from 0.721847 to 0.561152, with
30 of 32 held-out labels predicted correctly. The numerical results are
for the native initialization. Both panels passed HTML, Org, and site export.

### Dueling, implicit-quantile, n-step, and noisy value networks

All 24 blocks and 20 figure exports passed. The hand calculations for
centered advantages, cosine features, discounted prefixes, and factorized
noise matched the saved values. Fresh 512-step runs exported episode
returns and finite losses. The final logged returns were 15 for dueling,
10 for IQN, 10/30/37 for the one-, three-, and five-step comparison, and
10 for the noise-only exploration run. These are bounded-run diagnostics.

The Snake pixel-DQN report ran all six blocks and exported five figures.
The native 6×6 model logged 63 ended episodes in 512 steps. Its final
20-episode greedy mean was −0.60, compared with −0.90 for random actions.
The combined pixel-observation checklist entry remains open for Atari.
The ALE, Atari DQN, and external environment reports executed fallback
branches because the emulator, ROM or vsim scene path was unavailable.
Their mappings remain under audit.

### Chebyshev graph convolution

The fitted report reproduced all 64 upstream weighted-ring test inputs
and labels exactly. Independent eigendecomposition reproduced the four
polynomial terms within 1e-6 for Datamunge and 2e-6 for the upstream
float32 outputs. Replay matched the native fitted logits, 64/64 accuracy,
and NLL of 0.008411. The report checked permutation equivariance of the
basis and invariance of the pooled predictions. The two-panel figure
passed HTML, Org, and site export.

### Prioritized replay, quantile regression DQN, and recurrent replay

All 18 blocks ran and the 15 figures passed export checks. Independent
calculations matched the importance weights, two quantile Huber terms,
and padded sequence mask. The bounded 512-step demonstrations logged 38,
38, and 32 completed episodes, respectively, with final logged returns
of 10, 8, and 20. Recurrent replay implements the upstream zero-start
variant.

### Graph isomorphism network

The report uses the exact upstream 64 path and star test graphs with
shuffled node order and padded rows. Independent calculations reproduced
both fitted masked layers and the readout logits. The native model scored
64/64 with NLL 0.162198, and a fixed relabeling left its predictions
unchanged. The fit figure passed HTML, Org, and site export.

### Rainbow

All six executable blocks and five figures passed export checks. The
terminal C51 target puts all probability on the reward atom. The
512-step CartPole run completed 26 episodes and logged a final
evaluation return of 61 with finite loss. The six components from the
upstream combined agent are exercised in the bounded report.

### A2C and ACER

Both reports ran six blocks and exported five figures each. Independent
calculations checked A2C's one-step advantage and ACER's two-step Retrace
target. Fixed rollout batches brought A2C and ACER to 520 and 560 steps.
Their final logged evaluations were 10 and 62. A3C's report also ran.
Its interleaved worker loop still needs an asynchronous threaded
equivalent to match the upstream example.

### ACKTR, online actor-critic, and exploration bonuses

The three reports ran all 18 blocks and exported all 15 diagnostic
figures. Independent calculations checked the damped K-FAC factors,
one-step TD target, entropy, and count bonus. The bounded CartPole runs
used 560 steps for ACKTR and 512 for online actor-critic, with logged
evaluations of 62 and 8. The exploration report exercised all three
CartPole entropy coefficients and both 200,000-step MountainCar
conditions. Its final logged MountainCar scores were −200 without
the count bonus and −153 with it.

### Sparse GCN on Cora

Native fetching verified the checksums of both pinned source tables. The
loader produced the exact upstream label IDs, normalized edge weights,
and 140/2,568 split. Independent parsing and sparse calculations
reproduced the 13,264 graph entries, word features, both propagation
stages, logits, and all four fitted parameter gradients. The reported
transpose agreed with automatic differentiation. The native initialization
gave held-out NLL 1.865684 and 1,131 of 2,568 correct predictions after
three updates. The sparse adjacency checklist entry is now covered by
both its rectangular fixture and the Cora model. The figure passed HTML,
Org, and site export checks.

### GAE and PPO

The reports ran 12 blocks and exported ten figures. The worked
discounted-advantage and clipped-surrogate calculations matched
independent values. GAE evaluated four lambda values on 560-step
runs. PPO ran both the clipped and adaptive KL variants for 512 steps
each, with final logged evaluations of 69 and 67. IMPALA remains open
because its deterministic actor schedule does not verify the upstream
threaded trajectory queue.

### Graph autoencoders

The deterministic and variational fits reproduced the upstream graph,
features, pair masks, and sampled noise. Independent calculations checked
that all 48 held-out pairs are absent from fitting and that held-out
positive edges are absent from the encoder graph. The replay matched
posterior parameters, sampled latent features, KL, logits, and held-out
scores. Native GAE and VGAE had final held-out AUC 0.736111 and 0.725694.
The fit figure passed HTML, Org, and site export checks.

### REINFORCE, value baselines, and TRPO

All 18 blocks and 15 figure exports passed. Worked calculations matched
the Monte Carlo returns, baseline-adjusted advantages, and trust-region
rescaling. The report compared both REINFORCE variants. TRPO checked
Fisher-vector symmetry, the conjugate-gradient residual, and the
backtracking criteria. Its bounded run logged a final evaluation of 47.

### Inductive node classification with GraphSAGE

The fitted report reproduced the 60-node upstream test graph, labels,
and both sets of neighbor samples exactly. Independent calculations
reproduced the two normalized embeddings and all logits. After 12
updates, native held-out NLL was 0.440885 with 49/60 correct.
Relabeling nodes and remapping their sampled neighbors gave zero
logit error. The figure passed HTML, Org, and site export checks.

### Continuous-control and model-based RL

Action squashing, D4PG, DDPG, MPO, continuous PPO, SAC, and TD3 ran all
42 blocks and exported 35 figures. The tanh density replay agreed
with the two-million-draw histogram to within 2% away from the
boundary. Independent calculations checked projection weights, action
scaling, the Gaussian and corrected tanh log densities, MPO weights,
and TD3's conservative target. The continuous-control results use
bounded budgets. SAC on the exact upstream vsim environment remains
open. The existing report ran on Datamunge's native InvertedPendulum.

AlphaZero ran the tic-tac-toe self-play and search fit, ending with
38 wins and two draws against random play and four wins and 16 draws
against MCTS. Dreamer, MBPO, MPC, MuZero, PETS, PlaNet, and World Models
also ran and exported their model, planning, or policy diagnostics.
The captures showed 16 model updates for Dreamer, MBPO, PETS, and
PlaNet, and 512 CartPole steps for MuZero. World Models exported
2,000 VAE and 1,500 recurrent losses plus actual reconstructed and
imagined frames. All 49 blocks and 41 figures across the eight
model-based reports passed.

### GAT and GATv2

The report fits both single-head scoring rules to the exact upstream
held-out query/key features and labels. Independent replay reproduced
unmasked scores, masked attention and fitted classifier outputs.
GAT scored 149/192 predictions and ranked the matching key first in
96/192 queries. GATv2 scored 192/192 on both measures. The shared API
also handles a receiver with no allowed senders, which returns zero
attention and mixed features. The figure passed all export checks.

### DiffPool, TopK, and SAGPool

The report fits all three pooling methods to matched weighted-ring
fixtures. Independent replay reproduced assignment weights, selected
nodes, the 192 coarsened graphs, auxiliary loss, and fitted logits.
All three variants correctly classified all 64 held-out graphs.
Relabeling the input nodes changed logits by at most 2e-6.
The fit figure passed HTML, Org, and site export checks.

### Relational convolution and recurrent message passing

R-GCN reproduced the upstream directed-edge fixture exactly. Independent
replay checked the normalized relation weights, both basis-decomposed
layers, and original and relation-swapped logits. The native fit
classified 256/256 original queries and 92/256 after swapping relations.
Relabeling nodes and their incidence coordinates preserved the outputs.

The signed-edge MPNN reproduced the upstream ring fixture exactly.
Independent replay matched both message sums, both shared GRU updates
and the gated readout. The native fit scored 64/64 with the original
signs and 32/64 when all edges were positive. The all-positive logits
were identical across the graphs. Both reports passed figure export.

### DeepWalk and node2vec

The report reproduces the upstream bridge graph and every fitting and
test walk, sampling draw, and ordered context pair exactly. Independent
replay checks the transition weights and both fitted skip-gram gradients.
Native final held-out NLL was 1.896746 for DeepWalk and 1.818787 for
node2vec. Immediate returns on the held-out walks were 24.09% and 45.57%.
Both values describe their respective sampler distributions. The fit
figure passed HTML, Org, and site export.

### Knowledge graph embeddings

All four native scorers fit the same upstream cyclic triples and exclude
the exact six held-out positives. Independent real and complex formulas
reproduced all scores, the class-balanced logistic loss, and the filtered
endpoint ranks. Final filtered MRR was 0.116288 for TransE, 0.475000 for
DistMult, 0.743056 for ComplEx, and 0.237897 for RotatE. These results use
the native initialization. The figure passed HTML, Org, and site export.

### Temporal graph networks

The temporal-memory fit reproduced the 96 upstream streams and their
timestamps exactly. The independent replay scored each event before
updating memory, matched both logits, and checked every post-event state.
The report verified that a truncated stream reproduces its full prefix
exactly and that the first event gives equal positive and negative scores.
After four warm-up events, the fitted native model ranked all 384 positive
links above their sampled negatives with suffix NLL 0.001431.

### Molecular graphs

The SchNet-style and DimeNet-style reports used the exact 128 upstream
held-out geometries and energies. Independent calculations matched
Gaussian, sinusoidal, and angular bases and both branches' fitted
predictions. Held-out MSE was 0.009039 and 0.016644, respectively.
Neither branch changed its predictions under the combined rigid motion
and equivalent-atom exchange on this fixture. Both figures passed export
checks. All fifteen examples in the graph neural networks chapter now
have verified mappings.

### Next representation-learning coverage

Start with the upstream paired-view and frozen-feature protocols.
Reuse the current exact retrieval, graph, and tensor APIs where applicable.
Verify disjoint input splits, augmentation reproducibility, and objective
invariants before updating the contrastive-method checklist. Keep original
source-specific differences visible, including negative queues, EMA target
networks, and prototype assignments.

### Paired stochastic views and SimCLR

The paired-view report visualizes a source and both independently sampled
crops on a common intensity scale. The SimCLR report trains the compact
32/24/12 encoder and projector with symmetric NT-Xent and plots loss and
nearest-positive retrieval on a fixed held-out batch.

The native generator reproduced all 32 source surfaces, all 64 crops,
and all augmentation parameters in the upstream SimCLR archive exactly.
Loading all six TensorFlow parameter tensors into the native projection
path reproduced held-out loss exactly. Maximum absolute discrepancies
were 5.96e-8 for normalized rows and 1.91e-6 for scores. A uniform-score
check produced log(3) with four views and zero loss with one pair.
An independent NumPy calculation checked the complete trained native
projection, similarity, loss, and retrieval ranking. The native run
reduced held-out loss from 3.809875 to 1.793901 and increased final
nearest-positive retrieval from 11/64 to 35/64. Both reports passed fresh
execution and figure checks in the HTML, Org and website exports.

### VICReg and Barlow Twins

Both reports fit the shared compact projection MLP on fresh pairs of
stronger crops and plot each objective and its terms on a fixed held-out
batch. VICReg loss fell from 21.727041 to 10.142406. Barlow Twins loss fell
from 3.313872 to 0.592952.

Replaying the complete saved TensorFlow projections matched the VICReg
covariance matrices and Barlow Twins cross-correlation within 1.20e-7
maximum absolute error. The two gradients with respect to the projections
matched within 2.80e-8, and every objective component matched within
5.97e-8. Constant-projection checks gave 24.75 for VICReg and 3 for Barlow
Twins with three features. Both figures passed HTML, Org and website export
checks.

### Frozen representation evaluation

The linear probe and five-neighbor report ran both on a synthetic fixture
and on the original 192 training and 96 held-out DeepCluster descriptors.
On the original archive, every neighbor index and voted label matched the
TensorFlow outputs and an independent NumPy ranking. Maximum squared-distance
error was 7.41e-8. Independent calculations verified held-out logits,
predictions and NLL from each implementation's fitted weights. Using
Datamunge initialization, the probe scored 74/96 on initial features and
84/96 on learned features, compared with 72/96 and 84/96 in TensorFlow.
Both runs scored 93/96 for five-neighbor voting on both representations.
The report figure passed HTML, Org and website export checks.

### SimSiam

The compact SimSiam model fits the shared encoder and predictor on pairs
of stronger crops. On held-out views, alignment loss fell from -0.015601
to -0.987830, target spread fell from 0.209858 to 0.041948, and nearest-positive
retrieval fell from 13/64 to 7/64. The three plots expose the decline in
feature spread and retrieval alongside the improved alignment.

The native generator reproduced the TensorFlow crop matrix exactly.
Loading the ten saved parameter tensors reproduced the raw projection
exactly and both normalized outputs and per-view losses within 2.39e-7.
The detached target had no gradient tracking, and the gradient with respect
to the shared projection matched within 9.72e-10. The figure passed HTML,
Org and website export checks.

### BYOL

The BYOL report uses the core parameter-store copy operation for initial
synchronization and the subsequent 0.95-momentum target updates. On
held-out crops, alignment loss fell from 2.114650 to 0.136598, target
spread went from 0.158957 to 0.123933, and matching-crop retrieval
rose from 13/64 to 18/64.

Independent checks recomputed every alignment loss and all 48 target
updates for both the TensorFlow and Datamunge trajectories. In both runs
the optimizer phase left the target unchanged exactly, and each EMA
stage matched the independently computed weighted average within 3e-8.
The figure passed HTML, Org and website export checks.

### MoCo v1, v2 and v3

The MoCo report fits all three versions with momentum updates before
each training forward. Versions 1 and 2 warm up and maintain a
64-key FIFO dictionary. Version 3 fits an online predictor and uses
symmetric in-batch classification. Held-out loss fell from 3.966862
to 1.944388 for v1, from 3.769516 to 2.101516 for v2, and from
3.175692 to 1.580703 for v3.

Independent NumPy calculations verified all 48 training steps per
version in both implementations, including the pre-forward EMA, the
unchanged target across Adam, dictionary order and loss within 3e-7
absolute error. The native objective also reproduced early, middle
and final saved TensorFlow batch losses for each version. The figure
passed HTML, Org and website export checks.

### SwAV

The SwAV report trains a compact encoder, projector and eight normalized
prototypes with detached Sinkhorn targets. Held-out swapped cross-entropy
fell from 2.921565 to 1.303926, and the largest final column mass error
was 0.000560 from the per-view target of four. The figure shows the held-out
loss and final soft assignment matrix.

Native replay of the saved TensorFlow embeddings and prototypes reproduced
their scores, balanced assignments and swapped loss exactly at Float32
output precision. The stopped-target score gradient matched the analytical
cross-entropy gradient within 4.66e-9. Independent NumPy scaling verified
the fitted native assignments and loss. The figure passed HTML, Org and
website export checks.

### Contrastive predictive coding

The CPC report fits a causal GRU on six-step sequences with four context
steps and two future horizons. Held-out InfoNCE fell from 3.537139 to
1.166413, and future retrieval rose from 2/256 to 140/256. Perturbing
the held-out suffix changed the future scores with zero change in
prefix contexts.

Loading the upstream input and all ten parameter tensors reproduced
the latent features, contexts and both original and perturbed scores
within 2.87e-6 maximum absolute error. The archived held-out InfoNCE and
retrieval checks passed. An independent NumPy calculation verified
the native model's encoder, GRU and both horizon projections. The
figure passed HTML, Org and website export checks.

### Image pretext tasks

The report fits separate compact classifiers for quarter-turn rotation,
four jigsaw orders and four neighbor positions. Final held-out accuracy
was 128/128, 128/128 and 115/128, respectively. The plot shows loss and
accuracy on the fixed held-out split by epoch.

The native run reproduced all three saved TensorFlow image batches,
labels and flattened transformed inputs exactly. Replaying the
TensorFlow parameter tensors reproduced every saved logit exactly
for all three tasks. An independent NumPy computation checked native
fitted logits. The plot passed HTML, Org and website export checks.

### Topic corpus, word windows and word2vec

The word2vec report covers both shared fixture walkthroughs and trains
CBOW and skip-gram with the same four-negative objective and batch
schedule as the upstream program. It plots held-out loss, which fell
from 3.465192 to 1.875455 for CBOW and from 3.468256 to 1.424783 for
skip-gram.

Both seeded corpora, all window sources and all candidate IDs matched
the TensorFlow exports exactly. The unigram weights agreed within
1e-15 relative tolerance. Replaying saved embedding tables matched
candidate logits within 9.54e-7 and both gradients within 8.57e-8.
An independent NumPy computation checked fitted native logits. The plot
passed HTML, Org and website export checks.

### GloVe

The report fits weighted log-count regression with Adagrad to harmonic
co-occurrence counts from the synthetic topic corpus. Held-out loss fell
from 13.776212 to 3.658863. Its figure shows train and held-out loss and
compares fitted with observed held-out log counts.

Both native count tables matched every upstream pair and count exactly.
Replaying saved parameters matched held-out predictions within 2.39e-7,
combined embeddings exactly, and all gradients within 5.97e-8. NumPy
recomputed the native held-out regression loss. The figure passed HTML,
Org and website export checks.

### fastText subwords

The report fits skip-gram with word and hashed subword features and
evaluates three unseen word forms using their subwords alone. Unseen-word
loss fell from 3.522031 to 1.479544. The fitted cosine similarities put
all three unseen forms nearest to a known form of the corresponding verb.

Both corpus splits, all sampled IDs and every hashed feature count matched
the TensorFlow exports exactly. The native replay of the saved embedding
tables reproduced all composed vectors exactly, candidate logits within
9.54e-7 and gradients within 1.38e-6. NumPy independently checked the native
fitted vectors and logits. HTML, Org and website figure exports passed.

### Deep Embedded Clustering

The report pretrains an autoencoder, initializes centers with native
k-means++, then trains the encoder and centers against targets refreshed
from the full training set before each refinement round. Held-out
reconstruction MSE fell from 0.704624 to 0.191754. The fixed-target KL
fell in all three refinement rounds. The report plots both diagnostics
and the held-out latent clustering.

Native replay of all three upstream rounds matched the Student-t
assignments, sharpened targets and before/after KL within 2.39e-7.
Independent NumPy calculations checked the same quantities in every
round of both runs and verified that centers changed. The figure passed
HTML, Org and website export checks.

### DeepCluster

The report alternates k-means over normalized stripe embeddings with
full-batch balanced pseudo-label classification. Each round resets the
linear head and fixes the assignments during eight SGD updates. Weighted
loss fell within each round, from 1.082267 to 0.784440 in the final round.
The figure shows those within-round changes and held-out cluster counts.

All saved TensorFlow pseudo-labels, weights and losses were reproduced
exactly by the native replay. Independent NumPy calculations also
verified all three rounds and the final held-out assignments in both
runs. The frozen-representation evaluation consumed the native archive
successfully, and the figure passed HTML, Org and website export checks.
All examples in section 7.2 now have verified mappings.

### Next metric-learning coverage

Audit the existing metric and angular-margin APIs against the outstanding
section 7.3 examples. Verify the shared-tower contrastive loss, the three
angular heads, both proxy objectives and both structured losses against
their saved fixtures and add runnable training reports for the missing
mappings.

### Shared-tower contrastive margin loss

The report fits the shared tower on fresh balanced pairs and plots
held-out loss and pair distance distributions. Loss fell from 0.853927
to 0.091024; threshold accuracy rose from 135/256 to 235/256. Native
replay reproduced upstream distances, per-pair losses and embedding
gradients within 2e-6. NumPy verified the fitted native distances and
losses, and the pair labels matched the upstream fixture exactly.
The figure passed HTML, Org and website export checks.

### Angular metric heads

The report fits ArcFace, CosFace and SphereFace on matched synthetic
batches. All three reached 24/24 held-out accuracy with margin-free
inference. It plots loss, classification accuracy and the monotonic
SphereFace transform. The native replay checked training and inference
logits for all three heads and the SphereFace values and derivatives
on the complete 257-point TensorFlow grid. The plot passed HTML, Org
and website export checks.

### N-pair and lifted structured loss

The shared-tower report trains both objectives on matching streams of
one-positive-per-class batches. Held-out N-pair loss went from 0.499378
to 0.027495, and lifted structured loss went from 2.353276 to 0.001083.
Both held-out batches matched their TensorFlow inputs and labels exactly.
The native replay matched upstream loss terms and embedding gradients
within 1.36e-7. NumPy independently recomputed fitted loss terms. The
two-panel plot passed HTML, Org and website export checks.

### Proxy-NCA and Proxy Anchor

The report fits normalized class proxies jointly with the embedding
tower on identical batch streams. Both classifiers reached 24/24
held-out cosine accuracy. The inputs and labels reproduced the upstream
archives exactly. Native replay matched cosine scores and both sets of
gradients within 5.97e-7. NumPy independently verified the fitted
similarities and losses. Figure export passed in HTML, Org and the website.
All examples in chapter 7 now have verified report mappings.

### Remaining sequence coverage

Start the next batch by auditing the seven remaining example mappings
in chapter 4 against native sequence capabilities. Prioritize shared
components where more than one report can exercise them.

### Seven-step Neural Turing Machine

The report fits the upstream 16-unit controller and six-slot memory
geometry on distinct bit strings. All 512 split IDs and every held-out
observation and target matched the upstream archive exactly. A NumPy
replay reconstructed both fitted runs' address weights, memory and
read vectors from their recorded interface controls within 2.83e-7,
and independently verified held-out BCE. The Datamunge fit reduced
test BCE from 0.696075 to 0.535830 with 564/768 correct bits and 0/64
exact copies. Validation loss and read/write heatmaps passed HTML,
Org and website export checks.

### Seven-step Differentiable Neural Computer

The DNC copy experiment trains a 16-unit controller with six memory slots
and backward, content and forward read modes. The 512 split IDs and held-out
inputs and targets match the TensorFlow archive exactly. Independent
NumPy replay reconstructed every saved memory access state, allocation
and temporal link in both runs within 1.70e-7 from their respective
controller outputs. It also verified held-out BCE. The native fit
reduced test BCE from 0.691248 to 0.535498 with 564/768 correct bits and
0/64 exact copies. Validation, allocation and temporal-link panels
passed HTML, Org and website exports.

### Two-hop relational memory networks

The report trains the end-to-end full-fact and structured key-value
models on the upstream corpus, matching all 1,280 fact sets, queries
and labels exactly. Both fits passed fact-permutation checks. With
the TensorFlow parameter archives loaded, the native path reproduced
held-out logits, states, attention and NLL within 1.20e-6. Independent
NumPy forward passes also verified both native fits.

Held-out answer counts were 63/128 for full facts and 62/128 for
key-value pairs. Neither native run ranked the expected bridge fact
first on hop one. Validation and attention plots passed HTML, Org
and website export checks. All three examples in section 4.3 now
have verified report mappings.

### Per-example extended pointer vocabulary

The three-step pointer-generator fits the 12-token base and 14-token
extended vocabularies with shared unknown embeddings and accumulated
copy mass. Every generated source, length, target and identity across all
384 cases matched TensorFlow exactly. Independent NumPy checks recombined
the generator, gate and alignment into the reported mixture for both
runs within 5.42e-8 and checked the masked mass and token NLL. The native
run's greedy output was invariant to altered padding. It copied the
correct local ID in all 64 held-out cases and decoded 39 complete
sequences exactly. The validation and attention figure passed HTML,
Org and website exports.

### Sorting and hull pointer network

The seven-slot pointer report fits separate sorting and hull models.
All 768 generated observations and their lengths, targets and target
lengths match the TensorFlow archives exactly. Independent checks
recomputed masked NLL in both implementations and verified the mask on
each padded or previously used slot, absorbing STOP and nonrepeated
greedy indices. Native greedy predictions were invariant to padding
perturbation. Exact test outputs were 56/64 for sorting and 2/64 for
hull traversal. Figure exports passed in HTML, Org and the website.

### Handwriting CTC on composite MNIST strips

The report trains the 24-frame bidirectional model on official MNIST
glyphs. All split glyph IDs, test offsets, pixels, labels and lengths
match the TensorFlow archive exactly. Native CTC replay of TensorFlow
logits matched losses and gradients within 1.93e-6. An independent
dynamic program verified both runs' NLL and all padded gradients were
zero. Native greedy outputs were invariant to padding changes.

Validation NLL fell from 29.899820 to 5.950790. At three epochs the
native model obtained 0/96 exact test strings and 176/176 character
edits. Validation and image panels passed HTML, Org and website exports.

### Text-conditioned handwriting synthesis

The compact two-LSTM model fits a correlated two-component pen density
with a monotone Gaussian text window. Held-out NLL fell from 1.981002
to -0.187603. Sampling with matched draws was deterministic and changed
when the prompt changed. The synthetic test text, strokes, prompts and
sampling noise exactly matched TensorFlow. Native replay checked the
upstream density loss and finite gradients. Independent NumPy calculations
verified both runs' losses and draws within 2.09e-7. Validation and stroke
plots passed HTML, Org and website exports.

All chapter 4 examples now have verified mappings. Next, audit the chapter
5 language model candidates against the upstream corpora and saved model
outputs, beginning with BERT masked-language pretraining.

### Bidirectional masked-character pretraining

The 64-character report fits the two-block, three-head encoder with
a tied 65-character output table. The held-out corruption and operation
codes matched TensorFlow exactly. Held-out NLL was 3.459945 with 53/320
masked selections correct. Changing unselected labels kept loss fixed;
perturbations on either side changed the center output. NumPy checked
masked NLL and attention normalization. Figure exports passed.

Replaying saved weights matched logits, attention and NLL within 1.20e-6
for base BERT, ALBERT, DistilBERT, RoBERTa and static masking. Native training demonstrations cover the variant checklist below.

### ALBERT, DistilBERT and masking schedules

The variants report trains its own BERT teacher, ALBERT with factorized
embeddings and a reused block, and a one-block distilled student. It
compares RoBERTa-style dynamic masks and warm-up/decay against a static
masking control at a constant rate. Every mask operation, batch size,
and learning rate for both schedule runs matched TensorFlow exactly.
The teacher parameters remained identical throughout student training,
and independent NumPy checks reproduced both DistilBERT objectives within
8.3e-8. Native replay also verified the TensorFlow distillation objective.
The validation and schedule figures passed HTML, Org and website exports.

### Shared character BERT paraphrase retrieval

The two-tower report restores the pretrained character encoder and
fine-tunes it using symmetric in-batch contrast. All split IDs and held-out
tokens and lengths match the TensorFlow archive exactly. With the upstream
checkpoint, initial cosine scores matched within 2.99e-7 and held-out
recall at one rose from 3/64 to 61/64. NumPy recomputed cosine matrices
and both losses independently. Embeddings were invariant to altered
padding. The report also executed with the Datamunge pretraining archive.
Validation and cosine panels passed HTML, Org and website exports.

### Response-only instruction fine-tuning

The report initializes from the causal character decoder checkpoint
and trains Copy, Reverse and First on disjoint input strings. All
held-out prompts, labels, lengths, loss masks and split IDs matched
the TensorFlow archive exactly. Both initial and final greedy IDs
agreed for every held-out sequence in the matched-checkpoint run.
The final checkpoint replay matched TensorFlow logits within 1.67e-6
and verified exact zeros for prompt and padding logit gradients.
Independent NumPy checks verified both response losses. Validation
loss fell from 3.682518 to 1.521187, and neither fit produced a
completely correct held-out response at this budget. Figure exports
passed in HTML, Org and the website.

### Character BERT classification, BIO tagging and extractive QA

The native report restores the upstream pretrained character encoder
and updates it jointly with each separate task head. All 640 source
records and the held-out tokens, tags, lengths and answer spans match
TensorFlow exactly. Independent NumPy checks verified all three losses
and all decoded outputs in both runs, including the QA context mask.
The native fit obtained 22/64 classification, 1.0 entity F1 and 35/64
exact QA spans. Classification and QA reached 22/64 and 34/64 in the
upstream run. The validation panels passed HTML, Org and website
exports.

### Character BERT LoRA and residual adapters

The BIO tagging report starts both adaptation methods from the same
pretrained weights. Zero-initialized output projections preserve the
original hidden states, then the adaptation and tagging tensors change
while the pretrained tensors remain identical. The LoRA run uses 893
trainable values and reaches 0.6798 entity F1. The adapter run uses 565
and reaches 0.2430. All source records and held-out token tensors match
TensorFlow exactly. Independent checks recomputed both losses and
decoded tags. Replaying the upstream adapters and frozen parameters
matched held-out logits within 1.42e-5. Figure exports passed in HTML,
Org and the website.

### Grouped-query causal attention

The report exercises one, two and four key/value heads for four query
heads. All generated Q, K, V and output adjoints match the upstream
inputs exactly. Outputs, attention and all three gradients match
TensorFlow within 2.39e-7. NumPy independently checks the shared head
gradient sums. Every causal prefix output equals its full-pass row
exactly, and the figure passes HTML, Org and website exports.

### Exact chunked attention on CUDA

The report uses Float64 CUDA kernels for online softmax and recomputation
of the query, key, and value gradients. With 521 queries and 1,031 keys,
64-by-96 and 113-by-127 tiles gave identical results on an RTX 5080 Laptop
GPU. The generated inputs matched TensorFlow exactly. The largest
absolute error across the output and three gradients was 1.16e-14
against the TensorFlow artifact. A dense NumPy calculation independently
checked all four arrays. The diagnostic figure passed HTML, Org, and
website exports.

### Masked character Switch Transformer

The bidirectional masked-character report uses the original Tiny
Shakespeare fit and test windows and reproduces the fixed held-out
input and targets exactly. It runs 48 Adam updates with batch-wide
capacity. Its independently initialized masked NLL falls from 4.6254
to 3.3137, with 21 of 128 tokens over capacity. A separate replay using
TensorFlow’s fitted weights reproduced every route and capacity slot,
all 14 overflow rows, and held-out NLL. The largest logit difference
was 9.54e-7. Independent calculations checked both runs’ slots,
overflow contributions, masked NLL, and balance loss. The report and
plot passed HTML, Org, and website exports.

### Transformer-XL segment memory

The native model trains for 48 updates on the pinned character corpus.
Its held-out test NLL falls from 4.34189 to 3.02047. Cached and reset
second-segment NLL are 2.96627 and 2.95765. Changing cached inputs
changes the recurrent logits, and the first segment's gradient is
exactly zero. Every held-out input and target matches the TensorFlow
artifact. Replaying the fitted TensorFlow weights matches both logit
paths within 1.2e-6, both attention tensors within 1.8e-7, and the live
boundary gradient within 1e-4. Independent checks recover both
second-segment losses and confirm the causal mask and attention row
sums. The plot passes HTML, Org, and website exports.

### Vision Transformer on MNIST

The two-block model trains for 96 updates on the upstream 1,024-image
fit subset. Fit, validation, and test row IDs, the held-out labels and
pixels, and all extracted patches match exactly. The independent native
fit lowers test NLL from 3.12204 to 1.09612 and classifies 157 of 256
test digits correctly. With the fitted TensorFlow weights the native
model reproduces the patch and embedded arrays exactly, matches logits
within 2.33e-6 and both attention tensors within 5.67e-7. Independent
calculations confirm loss, accuracy, and attention row sums. The
validation and class-attention figure passes HTML, Org, and website
exports. The separately named architecture variants retain their own
checklist entries.

### DINO self-distillation on MNIST

The native fit runs 96 updates with the upstream split and crop stream.
The split IDs and all three first-step crops match the TensorFlow
artifacts exactly. Replaying TensorFlow’s initialization matches
first-step scores and loss within 2.39e-7, all student parameters after
Adam within 1.5e-8, all teacher parameters after EMA within 3.64e-12,
and the center within 5.59e-9. Before and after features from TensorFlow
weights match within 4.77e-7. In the independently initialized fit,
held-out five-nearest-neighbor classification rises from 116 to 121
correct out of 256. A NumPy probe reproduces both counts. The loss and
probe panels pass HTML, Org, and website exports.

### Neural surrogate for steady heat conduction

The report fits a two-layer tanh network for 96 updates to predict all
31 interior temperatures. All 1,024 fit and 128 test parameter triples
and all held-out target temperatures match the upstream artifact exactly.
An independent tridiagonal solve confirms the held-out targets and a
direct calculation recovers both runs’ reported error. The independently
initialized native fit reduces test RMSE from 0.306242 to 0.019555.
The reference and predicted profile and the held-out error curve pass
HTML, Org, and website exports.

### Atari frame preprocessing through ALE

The SDK-enabled report ran against ALE 0.11.2 and its bundled Tetris ROM.
The action set, zero return, 512.6-step mean length across five seeded
episodes, and 0.0012 final stack motion statistic match upstream. The
side-by-side video decodes all 400 frames at 15 fps. HTML, Org and
website exports include the recorded clip. The standalone diagnostic
also runs without the ALE SDK and checks pooling on alternating frames.

### External simulators through ALE and vsim

With both SDKs enabled, the report reproduces the upstream Tetris
episode (546 agent steps, 2,184 emulator frames, zero reward and game
over) and the complete printed final observation. The original vsim
scene contains eight entities, seven bodies and no joints. After
480 steps at 120 Hz the highest dynamic center falls from 2.25 m to
0.30 m. The height plot uses all 481 states. The 273-frame Atari clip
and 120-frame vsim clip decode in full and export with the plot in
HTML, Org and the website. The vsim clip contains one frame for
every four physics steps and runs at 30 fps.

### InvertedPendulum and Hopper with vsim

The optional Datamunge environment adapter loads the original articulated
scenes and steps the same vsim/Jolt SDK. Both seeded random policies,
pendulum feedback, and passive Hopper match the upstream reward means,
ranges, and mean lengths at the displayed precision. Feedback reaches the
1,000-step horizon on all 20 seeds. The clips reproduce the 288-step
feedback sequence, 239-step passive Hopper and seven-step random Hopper.
Their simulation-only frame counts reflect the local capture cadence.
All three clips decode completely and the plots and clips pass HTML,
Org, and website export checks.

### SAC on articulated vsim tasks

The SDK-backed report uses original scenes and an editable replay,
critic, actor, temperature, and target-update loop. It uses 256 hidden
units and a 256-transition batch. Both tasks completed a separate
1,100-transition check with 100 optimizer updates after warmup.
Their deterministic evaluation and policy recordings ran to completion,
and HTML, Org, and website exports include the short-run media. The
30,000-transition pendulum run finished with 29,000 updates and a
ten-episode deterministic return of 1,000, matching the upstream
final evaluation. The 300,000-transition Hopper run finishes
299,000 updates and improves ten-episode mean return from 11.46 to
261.81. Both schedules have finite logged losses and temperatures.
The 750-frame pendulum and 64-frame Hopper recordings decode fully
and export with the full-run plots in HTML, Org and the website.

### Asynchronous advantage actor-critic on CartPole

Four native worker threads collect rollouts and compute gradients
against private snapshots. The shared Adam commits are serialized to
protect its parameter and moment writes. The run commits all 100,000
upstream-budget transitions, with contributions from every worker.
Of 20,815 updates, 20,552 use older snapshots. Final greedy return
averages 151.7 over 20 episodes. This independently initialized
threaded run varies with scheduling. The upstream report has no
completed recorded result for direct numerical comparison. The
evaluation plot and greedy rollout passed HTML, Org, and website
export checks.

### Threaded IMPALA with V-trace on CartPole

Four independent actors generate 5,000 full trajectories at the
100,000-transition budget. The learner consumes 2,293, with 2,255
from older policy versions. The queue drops 2,707 and reaches its
configured peak of 64. The run accounts for every trajectory and
checks V-trace against independent on-policy returns and the
vanishing-ratio limit. It drains remaining work at the budget.
Final greedy return is 401.9 over 20 episodes in this scheduling-
dependent run. The TensorFlow recording reports 212.9. Both use
independent initialization. The evaluation and importance-ratio plot
and the 500-frame rollout pass HTML, Org, and website export checks.

### Atari Double DQN

The report now runs real ALE replay and records a greedy policy. A
byte-exact ring stores shared Atari frames and reconstructs the usual
replay minibatch. Seeded tests against ordinary Float32 replay cover
wraparound, episode resets and all 256 pixel values. The short ALE
check runs 2,008 transitions, including the 2,000-step warmup, and
performs three updates with 32 samples. Its greedy trial lasts 950
steps and scores zero. The full ALE schedule
then finishes 30,000 transitions, 7,001 updates and 55 training
episodes. The archived losses are finite. Both scheduled evaluations
score zero, and the final greedy trial scores zero in 410 steps.
The 205-frame policy video decodes completely and exports with the
full-run loss and reward plot. The upstream capture did not complete,
so it cannot supply a final-score comparison.

### Deep Ritz method for Poisson’s equation

The trial function enforces zero boundary values on all four edges.
The native fit takes 192 optimizer steps and lowers midpoint-grid
energy from 0.001601 to -2.416186 with an RMSE of 0.037947. Evaluation
coordinates, forcing and edge points match upstream exactly. The
spatial derivatives agree with automatic differentiation within
4.8e-7. Replaying the upstream parameters reproduces the field and
derivatives within 9.54e-7 and energy within 2.39e-7. Independent
NumPy calculations recover both runs’ energy and RMSE. The paired
heatmaps pass HTML, Org and website export checks.

### Heat solution operators

DeepONet trains for 288 updates on 192 initial functions and tests on
32 unseen functions with 48 queries each. Its native RMSE drops from
0.374433 to 0.204922. Function coefficients, fit and test inputs,
and held-out targets match upstream exactly. Replaying upstream
weights matches both embeddings and predictions within 1.20e-7.
An independent evaluation of the sine-mode solution checks the
targets and reproduces both runs’ RMSE.

The Fourier operator trains for 48 updates on a 32-point periodic
grid. Its native RMSE is 0.057158 on both the 32- and 64-point grids
using the same weights. Both input and target arrays match upstream
exactly. Independent FFT attenuation checks the targets, and
upstream-weight replay matches predictions on both grids within
3.58e-7. Both reports’ figures pass HTML, Org and website exports.

### Inverse PINN for thermal diffusivity

The 192-update native fit shares the upstream training, collocation
and test arrays exactly, including the sensor noise and mask. The
diffusivity estimate moves from 0.06 to 0.068015 against the true 0.18.
Its held-out temperature RMSE is 0.081824 and residual RMS is
0.295259. The upstream initialization yields diffusivity 0.115437.
Replaying the upstream trained parameters, with inverse softplus
applied to their saved diffusivity, reproduces both the fitted and
held-out field and residual arrays within 5.37e-7. Independent
calculations confirm the masked fitting objectives and test errors.
The temperature and diffusivity plots pass HTML, Org and website
export checks.

### Neural ODE with direct RK4 and continuous adjoint gradients

Both gradient paths fit the same 32 initial states for 12 optimizer
updates each. The trained parameters, 24 held-out predictions,
losses and gradients all match the upstream artifact within 1.12e-16.
The native direct and continuous gradients differ by at most 3.02e-12
at 32 RK4 steps. A separately implemented NumPy solver and central
differences verify the direct gradient within 8.39e-12. The time-one
flow plot passes HTML, Org and website export checks.

### Hamiltonian neural network for pendulum dynamics

The native fit trains for 192 updates and gives a held-out derivative
RMSE of 0.070945. The held-out states and derivative labels match
upstream exactly. Across twelve RK4 trajectories, the maximum learned
energy drift is 2.85e-8 and physical energy drift is 0.05744.
Independent calculation checks both drifts, and automatic
differentiation verifies the analytic vector field. Replaying upstream
weights matches every trajectory coordinate within 1.12e-15. The
phase trajectory and energy plots pass HTML, Org and website exports.

### Lagrangian neural network with position-dependent inertia

The native fit takes 192 updates and reduces held-out acceleration
RMSE from 0.824309 to 0.018955. Held-out states and targets match the
upstream artifact exactly. Automatic differentiation independently
checks the analytic first and mixed derivatives within 3.58e-7.
The inertia and potential satisfy their origin anchors. Replaying
the upstream weights reproduces accelerations and differential
terms within 4.77e-7. Independent calculations recover the target
accelerations and both runs’ RMSE. The inertia and potential plots
pass HTML, Org and website export checks.

### Neural Itô SDE with learned drift and diffusion

The native model fits 192 updates and improves held-out Gaussian
transition NLL from -0.489405 to -1.044428. The inputs, transition
targets, standard normals and reference trajectories match upstream
exactly. Upstream-weight replay reproduces the learned paths within
3.58e-7. Independent calculations check the negative log likelihoods
and endpoint moments. The native endpoint from +1 averages 0.10737
against the reference 0.46685, showing bias over multiple steps
despite the one-step likelihood improvement. The common-noise mean
path plot passes HTML, Org and website export checks.

### Variational latent ODE for irregularly sampled series

The native 60-update fit achieves posterior-mean forecast RMSE of
0.646052 versus 0.997401 for persistence. The upstream run reaches
0.542231. The held-out gaps, observed prefix, target values and
posterior noise match exactly. The upstream artifact includes only
decoder weights. Replaying those weights from the saved posterior
parameters and noise reproduces latent states, sampled forecasts and
posterior-mean forecasts within 4.77e-7. Independent calculations
check the initial latent sample, Gaussian-plus-KL objective and both
forecast errors. The held-out trajectory plot passes HTML, Org and
website export checks.

### Particle graph network with antisymmetric forces

The native fit trains for 144 updates and yields held-out acceleration
RMSE of 0.031758. Its 40-step endpoint RMSE is 0.025859 and the
maximum net-force component is 1.79e-7. The initial states,
acceleration labels and reference trajectories match upstream
exactly. Replaying the upstream weights matches its acceleration
and trajectory arrays within 2.39e-7. Independent checks confirm the
rollout centers of mass and both RMSE calculations. The trajectory
plot passes HTML, Org and website exports.

### Physics-informed networks for four manufactured PDEs

The native PINNs take 72 updates each for heat, wave, forced Burgers
and Taylor–Green Navier–Stokes. Evaluation coordinates, forcing and
exact fields match upstream exactly. Replaying upstream learned
weights matches the fields and equation residuals within 1.20e-7.
Finite differences independently check all four residual formulas
within 1.09e-7. Every native training objective decreases, while
the held-out field RMSE spans 0.22289 to 0.51823. The four fitted-field
panels pass HTML, Org and website export checks.

### SINDy autoencoder for latent dynamics

The native 96-update run reduces held-out objective from 2.950114
to 0.875370. Held-out measurements and derivatives are bitwise
identical to upstream. After thresholding, the native fit retains
linear, quadratic and cubic terms in its learned coordinate.
The independently initialized upstream fit retains constant and linear
terms. Upstream-weight replay matches reconstruction and derivative
diagnostics within 5.97e-7 and matches its objective. Independent
finite differences check the encoder and decoder tangents, and direct
calculations reproduce the effective polynomial and objectives for
both runs. The figure passes HTML, Org and website exports.


### Spectral-flux onset detection and beat tracking

The generated waveform, detected peaks, beat grid, period and phase
match upstream exactly. Flux differs by at most 7.63e-6 from upstream
and 3.86e-6 from an independent NumPy FFT. The report recovers 125 BPM
and matches 25 onsets and 16 beats within 20 ms, including the two
missing downbeats. Silence yields no pulse estimate. The flux and
beat-grid figure passes HTML, Org and website export checks.


### Conv-TasNet separation of spoken-digit mixtures

The 72-update native fit improves held-out permutation-invariant
SI-SNR from -30.351 to -5.278 dB. The mixture baseline reaches
-0.310 dB, and the independently trained upstream model reaches
-6.254 dB. The native held-out mixtures and reference sources match
upstream bitwise. NumPy independently reproduces the loss for both
sets of predictions within 6e-7. Swapping the target sources leaves
the native loss unchanged. The held-out waveform overlay passes
HTML, Org and website export checks.


### DeepSpeech-style convolutional GRU with CTC

The native model reduces held-out CTC loss from 21.29695 to 4.80842
at the upstream three-epoch checkpoint and to 4.20201 after twelve
epochs, with 35 of 192 exact transcripts. The upstream three-epoch
result is 4.37077 with 18 exact transcripts. Held-out labels match
upstream exactly. An independent log-domain alignment recurrence
reproduces both saved losses within 1.4e-7 and reproduces both greedy
transcript counts. The frame probabilities for the first correct
held-out utterance pass HTML, Org and website export checks.


### ESC-50 clean classification and multilabel tagging

The native fit reduces held-out binary cross-entropy from 0.8559
to 0.4642 in three epochs. It scores 22/32 on isolated clips and
0.6500 micro-F1 on two-source mixtures at the fixed 0.5 threshold.
The upstream fit scores 26/32 and 0.7344. Held-out labels match
exactly and the log-mel inputs differ by at most 9.92e-5. Independent
calculations reproduce both losses, clean accuracy and the mixture
confusion counts. The held-out tag scores export to HTML, Org and
the website.


### HiFi-GAN harmonic vocoder

In 36 alternating updates the held-out log-mel MAE falls from 5.98744
to 3.13192. The upstream run reaches 3.04723. The held-out conditions
agree within 3.31e-4, and all critic and generator trace shapes match.
An independent FFT and mel calculation reproduce the native conditions
and generated features within 4.96e-5. Recomputing all scalar losses
from saved critic maps and mel features agrees within 8.8e-6 across
both implementations. The waveform overlay exports to HTML, Org and
the website.


### WaveGlow conditional harmonic vocoder

The 36-update fit reduces training NLL from 1.87546 to -0.16466
nats per sample. Held-out NLL is -0.14042 versus -0.18597 upstream.
The held-out reference waveforms, Gaussian draws and sensitivity
probes match exactly. Log-mel conditioning agrees within 3.31e-4.
Reconstruction error is 1.79e-7 and the forward and reverse log
determinants cancel within 1.53e-5. A centered finite-difference
Jacobian independently confirms the native log determinant within
2.22e-4 and the upstream determinant within 5.25e-5. Independent
Gaussian likelihood calculations reproduce both reported losses.
The synthesis and reconstruction overlay passes HTML, Org and
website export checks.


### Speaker-disjoint keyword spotting

The three-epoch CNN improves validation accuracy from 34/128 at
epoch one to 47/128 at epoch three. It reaches 69/192 on test versus
87/192 upstream. Test labels match exactly, log-mel features agree
within 5.15e-4, and training means agree within 4.77e-7 with identical
scales. Independent calculations recover loss and accuracy from both
sets of logits. The per-keyword recall plot passes HTML, Org and
website export checks. The ESC-50 artifacts remain bitwise identical
after sharing its convolutional backbone with the keyword classifier.


### Residual CNN speech enhancement

The native denoiser starts as exact passthrough and fits for 60
updates. Clean and noisy held-out clips match upstream bitwise.
The enhanced mean SNRs at input levels -5, 0 and 5 dB are 3.318,
5.752 and 6.267 dB, versus upstream values of 3.173, 5.441 and
5.853 dB. Independent calculations reproduce the residual
subtraction, held-out MSE and per-level SNRs for both runs. The
held-out waveform overlay exports to HTML, Org and the website.


### Listen, Attend and Spell

The native three-epoch fit decreases teacher-forced held-out loss
from 2.3946 to 1.0773, versus 1.0476 upstream. Both models decode
39 of 192 held-out pairs and their EOS exactly. The targets match
upstream exactly. Independent calculations confirm both losses,
attention normalization and the use of previous greedy predictions
as feedback. The held-out attention plot exports to HTML, Org and
the website.


### Track-level music genre from pooled MFCCs

The native three-epoch MLP reduces held-out excerpt cross-entropy
from 1.2942 to 0.5367. Averaging three logits per track gives 28/32
correct versus 26/32 upstream. Genre labels match exactly, held-out
features agree within 3.82e-6, and training features agree within
7.63e-6. Independent calculations reproduce both losses and track
scores. The genre recall plot passes HTML, Org and website exports.


### Locally conditioned WaveNet vocoder

The 30-update fit lowers held-out code cross-entropy from 5.5470 to
4.9534, versus 4.8830 upstream. Held-out reference waves and quantized
targets match bitwise. Changing a future history code leaves earlier
logits unchanged, and swapping the mel condition changes the first
logit vector. The receptive-field rollout reproduces full-history
greedy decisions. Both fits emit just two distinct codes in this
short experiment. Independent calculations reproduce both
cross-entropies and confirm the shifted rollout histories. The
native WAV contains two 256-sample channels at 8 kHz, and the waveform
plot exports to HTML, Org and the website.


### Spoken-digit recurrent neural network transducer

Held-out lattice loss falls from 32.70189 to 4.70529 at the upstream
three-epoch checkpoint. The native decoder recovers zero exact pairs
there, versus one upstream with loss 4.67750. At epoch twelve the
native decoder recovers 18/192 pairs and loss rises to 5.34467.
Held-out paired labels match upstream exactly. Independent
enumeration of all 105 two-label paths per example reproduces the
reported losses, and the blank-probability plot exports to HTML,
Org and the website.


### X-vector speaker identification and verification

The native three-epoch fit identifies 62/80 held-out clips from
known speakers versus 60/80 upstream. For two unseen speakers,
enrollment and probe clips yield 120 trials and native AUROC of
0.95778 versus 0.93556 upstream. Identification, enrollment and
probe labels match exactly. Independent double-precision
calculations reproduce both cosine score matrices and both
AUROCs. The verification score plot exports to HTML, Org and the
website.


### Masked wav2vec 2.0 pretraining

The native 48-update run reduces held-out objective from 1.01189
to 0.98972, with a minimum of 0.78536 at epoch two. Upstream reaches
1.18088 from 1.40892. The held-out waveform and mask match exactly.
The upstream NumPy audit verifies hard assignments, product vectors,
masked inputs, similarity logits, duplicate suppression and both
loss terms for each fit. Both runs use 22 distinct product codes.
The native run has eligible negatives for 32/48 masked frames
versus 40/48 upstream. The held-out objective plot exports to
HTML, Org and the website.


### Tacotron 2 text-to-mel synthesis

The native fit reduces teacher-forced loss from 2.1286 to 0.5235.
Free-running normalized mel RMSE reaches 0.9681 versus 0.8630
upstream. Both runs recover all 192 stop lengths in the fixed-duration
fixture. Held-out prompt tokens match exactly and targets agree
within 2.51e-6. Independent checks reproduce the losses, RMSEs
and stopping decisions and confirm zero attention on padded tokens
and raw-frame autoregressive feedback. The mel overlay exports to
HTML, Org and the website.


### Tabular and image fusion

The native 144-update fit reduces held-out binary cross-entropy
from 0.69115 to 0.00033. The upstream fit reaches 0.00042. Held-out
images, metadata and labels match exactly. Both fits classify
256/256 paired rows correctly, 128/256 after swapping images and
116/256 after swapping metadata. Independent calculations confirm
both fitted losses and all six counts. The modality-swap plot
exports to HTML, Org and the website.


### Two-hop stacked attention for visual question answering

Both three-epoch fits answer all 192 held-out synthetic questions
correctly. Native cross-entropy falls from 1.10659 to 0.00041.
Images, tokens and answers match exactly. Replaying the upstream
weights reproduces the logits within 1.91e-6, attention within
2.39e-7, states within 7.16e-7 and regions bitwise. Independent
matrix products confirm both attention-weighted state updates and
changing the padded token has no effect on native logits.
The paired-question attention plot exports to HTML, Org and the website.


### Global and spatial-attention image captioning

The native three-epoch global and attentive models reach held-out
teacher-forced NLLs of 0.31277 and 0.31130 and decode 97/192 and
95/192 captions exactly. Upstream reaches 100/192 and 89/192.
Held-out images and target captions match bitwise. Independent
calculations reproduce all four losses and counts, verify greedy
argmax outputs and confirm normalized spatial attention. The
caption attention plot exports to HTML, Org and the website.


### Audio-visual syllable recognition

The native 96-update fit reduces held-out cross-entropy from 1.41784
to 0.01087. Waveforms, formant magnitudes, rendered frames and labels
match upstream bitwise. Both runs classify 256/256 paired clips,
126/256 with shifted audio and 138/256 with shifted video. Replaying
the upstream weights reproduces all three sets of logits within
1.91e-6. The pairing comparison exports to HTML, Org and the website.


### Serving and fine-tuning a pretrained digit backbone

The native frozen and fully fine-tuned classifiers score 37/200
and 114/200 on held-out rotated MNIST, versus 45/200 and 102/200
upstream. All split IDs, test labels and rotated test pixels match
exactly. The native checkpoint restores both target starts to the
served features, and head-only training preserves all backbone
weights bitwise. Replaying upstream source and target weights
matches held-out features and logits within 9.54e-7. The target
accuracy plot exports to HTML, Org and the website.


### Elastic weight consolidation, synaptic intelligence and replay

All fitting and held-out inputs, reservoir indices and minibatch
draws match upstream exactly. Independent calculations confirm the
empirical Fisher and synaptic importance. Task-A trajectory gradients
agree within 2.09e-7 and final weights for all six modes agree within
1.20e-7. Task-A accuracy is 125/256 after sequential training,
187/256 with EWC and 183/256 with SI. Replay reaches 179/256 versus
150/256 for the matched fresh-only run. The retention and adaptation
plot exports to HTML, Org and the website.


### Random and Bayesian hyperparameter search

Both searches evaluate eight width/rate pairs at 36 updates per
trial. The validation and test targets and random candidate order
match upstream exactly. Native random and Bayesian validation
winners score 0.32184 and 0.24476 test MSE. Both upstream searches
select the same candidate and score 0.26877. Independent GP solves
reproduce the acquisition scores and chosen proposals, and direct
MSE calculations confirm validation ranking and test scores.
The search comparison exports to HTML, Org and the website.


### Hyperband and asynchronous successive halving

The native schedulers select candidate 4 and score 0.24476 on the
same held-out targets as upstream. Hyperband spends 28 units and
ASHA 25, versus 28 and 22 upstream. The fresh candidate orders
match exactly. Independent trace checks reproduce promotion order,
ASHA virtual completion times, incremental resource accounting and
full-budget selection. Overlapping native full-budget candidates
match the fixed-budget search losses, checking that promotions resume
Adam and minibatch state. The full-budget comparison exports to HTML,
Org and the website.


### Few-shot sinusoid hypernetwork

The native 480-update run reduces held-out query MSE from 1.17800
to 0.21019, versus 0.16904 upstream. Support and query observations
match upstream bitwise. Reversing support rows leaves the native
predictions unchanged at the measured precision. Independent
calculations reproduce the upstream generated weights, both
regressors' predictions and both MSEs. The held-out fit plot
exports to HTML, Org and the website.


### Reptile few-shot sinusoid adaptation

The native 384-update run reaches held-out query MSE 0.67438
before adaptation and 0.59980 after, versus 0.82584 with the
adapted random initialization. Initial weights and held-out
observations match upstream exactly. First-step states match
within 1.50e-8, final initialization within 1.20e-7 and per-task
losses within 2.39e-7. Independent derivatives confirm the
support steps and reproduce all three held-out losses. The
adaptation plot exports to HTML, Org and the website.


### Full MAML, first-order MAML and Meta-SGD

After 384 episodes per method, held-out query MSE goes from 0.66549
to 0.60344 for full MAML, 0.67200 to 0.61684 for first-order MAML
and 0.65708 to 0.55775 for Meta-SGD. The sampled episodes match
upstream bitwise. First-step outer gradients agree within 8.95e-8
and final weights, learned rates and per-task losses within 2.39e-7.
Finite differences independently confirm the outer gradients,
including the fixed support derivative for the first-order variant.
The adaptation plot exports to HTML, Org and the website.


### Recurrent learned optimizer

The native 288-update fit reduces held-out mean rollout objective
from 1.062143 to 0.147597. At step eight the learned rule reaches
0.000979, versus 0.000217 upstream and 0.029490 for fixed-rate SGD.
Held-out minimizers, curvatures and starting points match bitwise.
Independent rollout calculations using each fit's weights reproduce
all states within 1.77e-7 and per-step objectives within 9.07e-7.
The convergence plot exports to HTML, Org and the website.


### Learning without Forgetting

The native source fit scores 248/256 on task A. Both target fits
retain 247/256, while task-B accuracy is 220/256 for plain training
and 221/256 with distillation. Upstream task-A scores are 244, 248
and 246 at the three stages. Task-B training inputs and labels and
both held-out label sets match bitwise. Independent calculations
confirm cached teacher probabilities, temperature-scaled distillation
and combined objectives for both fits. The comparison plot exports
to HTML, Org and the website.


### Domain-adversarial adaptation

Target accuracy rises from 307/512 with source-only training to
390/512 with gradient reversal, while source accuracy changes
from 487/512 to 481/512. Upstream target counts are 284/512 and
330/512. Held-out inputs and labels match bitwise. The forward
discriminator is unchanged by reversal. Independent checks
confirm its weight gradient and the -0.6 feature gradient multiplier
for both fits. The source and target comparison exports to HTML,
Org and the website.


### Matching, prototypical and relation networks on Omniglot

Three 384-update fits share identical episode sampling and the
same encoder architecture. The native matching, prototype and
relation models classify 346, 353 and 187 of 480 held-out queries
correctly versus upstream counts of 310, 337 and 172. All held-out
row IDs and support labels match bitwise. Independent calculations
reproduce support attention, class means or sums, learned relation
logits, class probabilities and losses for both implementations.
The accuracy plot exports to HTML, Org and the website.


### Siamese one-shot Omniglot and shared fixture walkthroughs

The native verifier's training BCE falls from 0.67901 to 0.66067
across three epochs. It classifies 71/128 held-out five-way
episodes correctly, versus 77/128 upstream and 43/128 by pixel
distance. Episode rows and correct indices match upstream exactly.
Independent checks reproduce the pixel distances and argmax
decisions. The episodic report also covers the two shared-header
walkthroughs through its validated archive loader, same-alphabet
sampler, shared convolutional encoder and three training modes.
Archive checks confirm 964 background and 659 evaluation characters
with 20 contiguous drawings per character. Both figures export
to HTML, Org and the website.


### PackNet continual regression

The native and upstream fits reserve the same 80 of 161 coordinates.
The held-out fixtures match exactly, all weight snapshots agree
within 8.95e-8, and predictions agree within 2.39e-7. Task A's
reserved coordinates and outputs remain bitwise unchanged after
task B, preserving 0.28965 RMSE. Task-B RMSE falls from 0.99879
to 0.63342. Independent masked-forward calculations reproduce
both runs, and the error plot exports to HTML, Org and the website.


### Progressive network with frozen lateral sources

Held-out inputs and targets match upstream bitwise. Training
the second column preserves the first column's parameters and
predictions exactly in both runs. Native task-A RMSE is 0.42763.
Task-B RMSE is 0.27196 with lateral inputs and 0.32440 when they
are disabled at inference. The upstream comparison is 0.20988
versus 0.53885. Independent forward calculations reproduce all
scores. The ablation plot exports to HTML, Org and the website.


### First-order DARTS and discrete child retraining

Both fits select tanh and then ReLU after 36 alternating updates.
Native mixed validation MSE and fresh-child test MSE are 0.18034
and 0.18453, versus upstream 0.17617 and 0.15288. Validation
inputs and validation and test targets match bitwise. Independent
forward calculations and finite differences confirm both sets of
mixture probabilities, validation predictions and architecture
gradients. The held-out child fit exports to HTML, Org and the website.


### ENAS with shared weights and an autoregressive controller

Both 144-update searches select ReLU followed by tanh. Native
shared-weight validation MSE is 0.09665 and the fresh child reaches
0.16536 test MSE, versus 0.13467 and 0.16708 upstream. Validation and
test targets match bitwise. Independent calculations reproduce
each REINFORCE update, reward baseline, validation ranking and
retrained test error. The controller trajectory exports to HTML,
Org and the website.


### Residual attention rollout

The native example traces the classification token through three
two-head residual blocks. Attention and cumulative rows sum to one.
Independent matrix products reproduce all transitions and the final
rollout for both initializations. Feeding upstream head weights
directly to the Datamunge rollout API reproduces its transitions
and final matrix within 8.95e-8. The per-layer token mass plot
exports to HTML, Org and the website.


### Minimum-cost bounded counterfactuals

The native solution agrees with the upstream recorded coordinates
and costs at printed precision for both modified rows. The first
row has an interior solution costing 0.300833 and the second
saturates its first coordinate and costs 0.111250. Closed-form
active-set calculations reproduce both solutions. The immutable
third feature is preserved, the infeasible row is detected, and
the already feasible row stays unchanged. The coordinate-path
plot exports to HTML, Org and the website.


### Group fairness metrics

Thresholded counts and all available rates match upstream on the
15-row fixture. The first pair has demographic-parity gap 1/14
and equalized-odds gap 5/12. Comparisons involving the third group
preserve the undefined true-positive and equalized-odds gaps.
The rate plot exports to HTML, Org and the website.


### Mean-score parity and adversarial debiasing

All fixtures match upstream bitwise and the three fitted weight
vectors agree within 5.56e-17. The projected model achieves a
training mean-logit gap below 1e-12. Held-out selection-rate gaps
are 0.70625 for ordinary training, 0.08750 for the projection
and 0.18750 for adversarial training. Independent calculations
reproduce all held-out accuracies and conditional rate gaps.
The comparison plot exports to HTML, Org and the website.


### Kernel SHAP with sampled and exhaustive coalitions

The background, query point and sampled masks match upstream
exactly. Coalition means and sampled and exhaustive coefficients
agree within 1.12e-16. Direct Shapley marginal contributions
independently reproduce the exhaustive regression. Both regressions
sum to the full-minus-empty score and the sampled attribution
maximum error is 0.006067. The comparison exports to HTML, Org
and the website.


### LIME-style local linear surrogates

Both anchors and all fitting and held-out perturbations match
upstream bitwise. Kernel weights, predictions and fitted coefficients
agree within 2.23e-15. Independent weighted normal equations
reproduce both surrogates and their held-out weighted RMSEs of
0.032094 and 0.028378. The coefficient plot exports to HTML, Org
and the website.


### Concept activation vectors and TCAV

Fitting and held-out inputs and concept labels match upstream
bitwise. Probe coefficients and directional derivatives agree
within 1.23e-14. Independent ridge fitting and centered finite
differences confirm both steps. The probe predicts 127/128
held-out concept labels. Derivatives are positive for 40/40
class-0 rows and 17/88 class-1 rows. The sensitivity plot exports
to HTML, Org and the website.


### Out-of-distribution confidence scores

Native maximum-softmax, ODIN and negative-energy AUROC are
0.99590, 0.99829 and 0.95958, versus 0.99593, 0.99815 and 0.99841
upstream. Held-out inputs match bitwise. Independent calculations
from the fitted parameters reproduce the logits, ODIN gradient,
clipped inputs, log probabilities and confidence rankings. The
native ODIN score also runs inside an inference no-grad scope.
The AUROC plot exports to HTML, Org and the website.


### Federated averaging on unequal non-IID clients

Training and held-out arrays match upstream bitwise. The
count-weighted one-step update agrees with centralized training
within 1e-12. All eight rounds agree with upstream within 2.23e-16,
and an independent NumPy implementation reproduces the full
trajectory. Held-out half MSE falls from 0.71263 to 0.01401.
The convergence plot exports to HTML, Org and the website.


### Differentially private SGD

The simulated binary classifier takes 24 steps using per-example gradient
clipping and Gaussian noise on the gradient sum. An independent NumPy
calculation reproduces the raw gradients, clipped gradients and weight
updates to within 4e-16 when supplied with the same noise. The classifier
gets 251 of 256 held-out labels correct. The plot tracks the maximum
per-example gradient norm before and after clipping and renders in HTML
and Org. The example omits privacy accounting.


### Randomized smoothing

The report reproduces all 48 upstream inputs, Gaussian draws, class vote
counts and decisions bitwise. The Clopper–Pearson bounds and Gaussian
radii agree within 7.5e-15. Independent calculations reproduce the
classification histograms and confirm each nonzero lower bound by direct
binomial tail summation. The fixed affine classifier certifies 41 of 48
points, and 27 are both correct and certified beyond radius 0.1. The radius
plot exports to HTML, Org and the website.

The same seeded fixture covers the upstream shared toy-regions header.
Its simulated coordinates and three-region label rule are shown in the
report, verified bitwise against the upstream archive and checked
independently against the region boundaries.


### Ridge-regression influence functions

The eight analytic case-weight derivatives differ from central differences
of weighted refits by at most 6.12e-10. An independent NumPy weighted
ridge calculation reproduces the coefficients, influence scores and full
removal changes, and all eight rows match the upstream printed precision.
Row 7 has the largest positive influence. The plot compares the local
derivatives and the actual loss changes and exports to HTML, Org and the
website.


### Filtered recommendation metrics

The five-user fixture matches upstream item IDs, per-user NaNs for the
two excluded users, and all hit rate, NDCG, and MRR values within 1e-14.
The paired mode also evaluates the 96-user two-tower score archive with
diagonal relevance. Independent NumPy rankings reproduce all metrics
for both modes. The fixture plot exports to HTML, Org, and the website.


### Deep & Cross click prediction

DCN and DCN-v2 use bitwise matching inputs and labels and take 141 Adam
steps from native initializations. Native inference replays both upstream
checkpoints within 7.16e-7 across embeddings, both cross states, the deep
branch and logits. An independent NumPy forward pass reproduces native
and upstream outputs. Native seen/unseen log losses are 0.56807/0.79680
for DCN and 0.54227/0.64296 for DCN-v2. The plot shows both held-out groups
over training and exports to HTML, Org and the website.


### Wide & Deep click prediction

Fit and held-out inputs and labels match the upstream data bitwise.
The native model takes 141 Adam steps and improves training log loss
from 0.69984 to 0.50862. Seen and unseen held-out losses finish at
0.51309 and 0.66585. Native inference replays both upstream branch logits
and their sum within 3.58e-7. Independent NumPy calculations reproduce
both implementations and confirm zero wide cross coefficients for the
unseen pairs. The learning curves export to HTML, Org and the website.


### DLRM click prediction

Fit and held-out arrays match upstream bitwise. With native initialization
and 141 Adam updates, training loss falls from 0.70176 to 0.46113. Seen
and unseen pair log losses finish at 0.51601 and 0.59269. Upstream
checkpoint replay matches the bottom MLP, three dot products, combined
features and logits within 4.77e-7. Independent NumPy forward passes
confirm both implementations. The learning curves export to HTML, Org
and the website.


### Factorization machines and DeepFM

Both reports reproduce the upstream fit and held-out inputs and targets
bitwise and train with native initializations. On unseen user–item pairs,
the rating model reaches RMSE 0.30447 versus 0.62870 for the linear
baseline. DeepFM finishes with seen/unseen pair log losses of
0.52427/0.56864. Native checkpoint replay agrees within 1.20e-7 and
4.77e-7, respectively. Independent explicit sums over distinct pairs
verify the low-rank interaction and reproduce both models' predictions.
Both figures export to HTML, Org, and the website.


### Explicit-rating matrix factorization

All 628 fitting and 210 held-out ratings, both initial matrices, and all
three shuffle orders match upstream bitwise. Final SGD and ALS factors
and predictions agree within 3.34e-15. Independent NumPy calculations
reproduce both training trajectories from the archived inputs and orders.
The mean baseline, SGD and ALS test RMSEs are 0.486300, 0.486492 and
0.579280. The held-out prediction scatter exports to HTML, Org and the
website.


### Implicit matrix factorization and Bayesian personalized ranking

Interaction counts, held-out items, and sampled audit triples match
upstream bitwise. A NumPy calculation reproduces scores, regularized
objectives, factor gradients and filtered ranks for each implementation.
With native initializations and 36 Adam updates, the weighted and BPR
objectives reach 0.122779 and 0.231320. Both retrieve the held-out item in
the top five for all 24 users, matching upstream hit counts. The rank
curves export to HTML, Org and the website.


### Neural collaborative filtering

The seen mask and held-out items match upstream bitwise. The native
model trains from Datamunge initialization with four sampled unknowns
per positive and achieves hit@5 of 12/24, versus 11/24 upstream.
Replaying upstream weights matches all held-out ranks exactly and all
catalog logits within 1.20e-7. An independent NumPy forward pass and
masked sort reproduce the scores and ranks of both models. The
ranking curve exports to HTML, Org and the website.


### Two-tower retrieval

Both held-out input views match upstream bitwise. With Datamunge
initialization and 48 Adam updates, catalog cross-entropy drops from
7.351359 to 1.468453, and hit@5 rises from 2/96 to 96/96. Upstream
checkpoint replay reproduces all 96 ranks and embeddings and scores
within 2.87e-6. Independent NumPy calculations confirm both forward
passes, losses and ranks. The before/after plot exports to HTML, Org
and the website. The native retrieval archive includes both independently
encoded views for downstream ranking or nearest-neighbor evaluation.


### Approximate embedding retrieval

The synthetic catalog, queries, hyperplanes, signatures, exact rankings,
approximate rankings and candidate counts match upstream bitwise. An
independent NumPy calculation verifies every bucket probe. Raising the
minimum budget from 16 to 32 increases mean candidates from 18.406 to
34.583 and exact top-five recall from 0.31250 to 0.43125. Saved-index
serving reproduces all 96 upstream results and scan counts without
exhaustive scoring. The same report evaluates native two-tower embeddings,
and the fixture plot exports to HTML, Org and the website.


### GRU4Rec session ranking

Held-out sessions match upstream bitwise. The native initialization and
72 Adam updates reduce TOP1 loss from 0.999530 to 0.577276. Replaying
upstream weights matches all ranks and logits within 9.54e-7. An independent
NumPy calculation reproduces both GRU unrolls, ranks, and losses. The
native report verifies session reset invariance and confirms that changing
the last click leaves earlier scores untouched. The prefix hit-rate plot
exports to HTML, Org and the website.


### SASRec and BERT4Rec

Held-out sessions, model inputs, and the sampled objective inputs and
targets match upstream bitwise for both methods. Separate NumPy
calculations reproduce each implementation's losses and ranks and
verify the attention masks and future-click intervention. Native SASRec
achieves hit@5 of 192/192 and BERT4Rec reaches 185/192, versus 192/192 and
168/192 upstream. Native initialization differs and upstream archives
contain predictions but no weights. The ranking plot exports to HTML,
Org and the website.


### RetNet execution modes

Held-out targets match upstream bitwise. The native model uses Datamunge
initialization and 48 Adam updates, reducing test NLL from 3.725425 to
1.966256. Parallel, recurrent, and four-token chunk outputs agree within
3.82e-6. The native operation reproduces all three upstream arrays
exactly when fed upstream Q, K and V. A separate NumPy implementation
reproduces the retention outputs and next-event NLL in both archives.
The validation plot exports to HTML, Org and the website.


### Hyena sequence mixer

Held-out tokens and targets match upstream bitwise. Training from
Datamunge initialization lowers test NLL from 3.713467 to 1.817176.
Changing the suffix preserves the prefix logits exactly. Replaying the
upstream weights reproduces logits within 1.44e-6 and taps within
5.96e-8. An independent direct-convolution implementation reproduces
both models' filters, logits and losses. The validation and tap plots
export to HTML, Org and the website.


### S4 structured state-space filter

Held-out sequences and targets match upstream bitwise. Native training
reduces test MSE from 0.183042 to 0.009270. Convolution and recurrence
agree within 2.06e-7. Independent NumPy calculations verify the rank-one
structure, reproduce both discrete matrices by direct bilinear solves,
and verify kernels, predictions, and losses from each implementation's
fitted parameters. The validation and held-out fit plots export to
HTML, Org, and the website.


### Mamba selective memory

The held-out sequences and targets match upstream bitwise. Native
initialization and 72 Adam updates reduce test MSE from 0.346459 to
0.274757. Altering the future values preserves earlier predictions
exactly. Loading the upstream weights reproduces outputs, final state
and step sizes within 4.77e-7. Independent NumPy convolution and
recurrence reproduce both runs. The plots export to HTML, Org and the
website.


### RWKV-4 time and channel mixing

Held-out inputs and labels match upstream bitwise. With Datamunge
initialization, test NLL drops from 3.573025 to 1.948274 in 48 Adam
updates. The prefix logits remain identical after changing the suffix.
Replaying upstream weights reproduces logits within 1.67e-6 and
weighted key-value averages within 4.77e-7. An independent NumPy
calculation uses explicit lag weights to reproduce both runs. The
validation and weighted-value plots export to HTML, Org and the website.


### Shortest-path-biased graph transformer

The held-out adjacency matrices, distance IDs, and labels match upstream
bitwise. BFS independently verifies all shortest paths and the connectivity
labels. The model trains from Datamunge initialization and classifies all
128 graphs correctly. Node relabeling changes logits by at most 9.54e-7.
Loading upstream weights matches logits and both attention maps within
3.58e-6. An independent NumPy forward pass confirms both models. The
validation curve exports to HTML, Org, and the website.


### CLIP-style contrastive retrieval and prompt classification

All eight held-out image batches, captions, concept labels and canonical
prompt banks match their upstream fixtures bitwise where archived. Native
initialization with 144 Adam updates lowers validation loss from 3.789672
to 0.256267. Image-to-text and text-to-image hit counts are 112/128 and
119/128, versus upstream 113/128 and 122/128. Reloaded native encoders
classify 114/128 by prompt alone, versus upstream 110/128. Independent
NumPy calculations recover both implementations' scores, losses and
predictions from archived embeddings. The training curve exports to
HTML, Org and the website.


### Music Transformer

All simulated phrases and sampling uniforms match upstream bitwise.
With Datamunge initialization, test NLL drops from 3.823078 to 1.936422.
The prefix intervention leaves earlier logits identical. Checkpoint
replay matches held-out logits, both attention maps and sampled logits
within 1.44e-6 and reproduces all sampled tokens. Independent decoding
verifies the categorical draws and MIDI note times, pitches and
durations in both runs. The validation and sample plots export to HTML,
Org and the website.


### Audio Spectrogram Transformer

Held-out waveforms and labels match upstream bitwise. Independent NumPy
checks reconstruct the log-mel transform, fitting-split normalization,
overlapping patches, and classification loss. With Datamunge initialization,
test NLL falls from 1.603500 to 0.013190 and all 128 test signals are
classified correctly. Replaying upstream weights with native preprocessing
matches mel values, patches and logits within 1.11e-5, 4.54e-6 and 4.50e-6,
respectively. The validation plot exports to HTML, Org and the website.


### Conformer acoustic recognition

All held-out transcripts match upstream bitwise. With Datamunge
initialization, test CTC loss drops from 21.760458 to 0.060676 and
all 96 transcripts are decoded correctly, matching upstream
accuracy. Replaying archived upstream logits and convolution inputs
matches CTC losses within 6.15e-7 and depthwise outputs within 1.20e-7.
Independent NumPy dynamic programming, convolution and greedy decoding
verify both archives. The upstream archive contains intermediate
outputs but no model weights. The validation and frame-probability
plots export to HTML, Org and the website.


### SegFormer

Held-out masks match upstream bitwise after reproducing the simulator's
draw order. With Datamunge initialization, training loss falls from
1.234041 to 0.750136 over 72 updates. The validation-selected third
epoch scores mean IoU 0.317886 on test masks, versus 0.274794 upstream.
Datamunge reproduces the upstream decoder logits exactly from its
archived projected grids and decoder weights. Independent NumPy checks
verify both decoders, attention geometries, and segmentation metrics.
The validation curve and mask gallery export to HTML, Org and the website.


### BEiT masked visual tokens

The report consumes the upstream digit and frozen codebook archives.
Independent nearest-centroid assignments reproduce every target across
all splits. Held-out pixels, targets, and contiguous masks match upstream
bitwise. With Datamunge initialization, masked test NLL falls from
3.411247 to 1.632119, with 574/1200 tokens correct versus upstream
562/1200. The native hidden-pixel and visible-label interventions leave
logits and loss unchanged. Independent masked cross-entropy and accuracy
checks confirm both archives. The validation curve exports to HTML, Org
and the website.


### XLNet permutation language modeling

Character windows and held-out permutation ranks match upstream
bitwise. From Datamunge initialization, test NLL drops from 4.361160
to 3.519795. Mutating forbidden tokens leaves the tested query logits
identical, and changing the rank-zero token affects later queries.
Archived rank masks and cross-entropy are independently verified in
both implementations. Replaying the upstream checkpoint reproduces
both attention maps and logits within 1.20e-6. The validation plot
exports to HTML, Org and the website.


### Temporal Fusion Transformer

Held-out observations, covariates, and targets match upstream bitwise.
With Datamunge initialization and the same schedule, test pinball loss
drops from 0.763691 to 0.070209, versus a final 0.053107 upstream. Later
covariate edits leave the first horizon exactly unchanged and affect
later forecasts. Independent NumPy checks verify both archives' losses,
attention masks and normalized selection weights. The upstream archive
contains predictions and intermediate weights, with no model checkpoint.
The validation and fitted-forecast plots export to HTML, Org and the website.


### Transformer translation and compact T5-style tasks

Held-out sources, targets, teacher prefixes, and archived lengths match
upstream exactly. Native initialization and the upstream Adam schedule
give final test NLL 2.126995 for translation and 1.079726 for T5-style
tasks. Greedy matches are 0/64 and 15/96. Both modes pass causal and
source-sensitivity checks. Upstream checkpoint replay reproduces all
greedy sequences and matches logits and three attention maps within
2.39e-6. Independent masked-loss and attention checks confirm both
archives. The validation plot exports to HTML, Org and the website.


### Gaussian cross-entropy policy search and feedback rollouts

Mirrored starts, Gaussian draws and all elite indices match upstream
exactly. The refitted means agree to 2.99e-8 and variances to 1.87e-9.
Across all candidate rollouts, returns agree to 6.11e-5 absolute.
Held-out mean-policy return improves from -18.609520 to -3.031933.
Independent NumPy checks reproduce the refits, the return calculation,
and the full pre-transition state trace of the final mean policy.
The report covers the shared rollout example as well as CEM. Both
plots export to HTML, Org and the website.


### Augmented random search

Mirrored starts, directions, and selected direction IDs match upstream
exactly. Held-out return improves from -18.609520 to -4.178938 and
agrees with upstream within 8e-6 on all four evaluations. Independent
NumPy checks verify the raw coefficient whitening, rollouts, running
means and variances, selected reward standard deviations, and updates
for both implementations. The report exports to HTML, Org and the website.


### OpenAI evolution strategies

All starts, Gaussian perturbations and centered ranks match upstream
exactly. Held-out return improves from -18.609520 to -3.730335 and
agrees within 1e-5 across the four evaluations. Independent NumPy checks
reproduce candidate rollouts, average ranks, ascent gradients and Adam
steps for both implementations. The plot exports to HTML, Org and
the website.

### Separable natural evolution strategies

The report fits the mean and three independent scales of a diagonal
Gaussian policy search. Initial states and Gaussian draws match upstream
exactly. The independent Fisher-system audit reproduced all three updates
and the returns from 1,152 episodes. Held-out return improved from
-18.609520 to -2.720727 and agreed with upstream within 1e-5 across four
evaluations. The plot exports to HTML and Org.

### Full-covariance CMA-ES

The native optimizer reproduces upstream initial states, Gaussian draws,
and parent selection exactly. The independent audit replays 864 episodes
and verifies the mean, both evolution paths, covariance, and global step
size for all three updates. Held-out return improves from -18.609520 to
-2.625544. The off-diagonal covariance grows from zero to 0.968605 in
absolute value. HTML and Org exports include the return plot.

### MAP-Elites feedback policy archive

Native evaluation matches upstream Gaussian draws, parent IDs, behavior
bins, and archive occupants exactly across all three rounds. The independent
audit reconstructs 192 candidates, their descriptors and replacements, and
29 held-out archive evaluations. Occupancy grows from 19 to 29 of 64 cells.
The training champion earns -2.370273 on held-out starts. The final archive
plot exports to HTML and Org.

### Synchronous population-based training

An independent replay verifies 144 momentum updates and all four checkpoint
transfers, including inherited momentum and perturbed hyperparameters. The
native report matches upstream samples, donors and factors exactly. Best
validation MSE falls from 0.041686 to 0.002953 and the validation-selected
worker has test MSE 0.003700. HTML and Org exports include the plot of all
six worker histories.

### Feed-forward NEAT on XOR

The native report matches upstream genes, depths, parents, splits and
species exactly. Outputs and fitness agree within 3e-7. Independent
reconstruction checked 256 genomes, 60 connection markings and 18
split markings. Best fitness increases from 0.470487 to 0.498984
while species count grows from 1 to 19. The short run ends at two of
four correct XOR signs. HTML and Org exports include the fitness plot.

### Sparse-grid exploration and random network distillation

All three control trajectories match upstream exactly. RND uses the native
parameter initializer and preserves the upstream grid, network geometry,
reward scaling and learning schedule. Independent NumPy replay verifies
12,288 actions, both Q-tables, bonus computations and unchanged target
features. Predictor MSE falls on visited cells in all three RND runs.
Coverage improves on two seeds and ties on one, with no task successes.
The standalone fixture checks callback scheduling and plots coverage over
time. Both reports export their figures to HTML and Org.

### Intrinsic curiosity and shared dynamics model

The shared model report fits inverse-only and joint dynamics objectives
from the same initialization. Joint training reduces forward error from
0.329432 to 0.000207. The independent sparse-grid audit replays 12,288
actions with the native initializer, verifies both Q-tables and recomputes
the forward errors. ICM increases coverage on one seed and decreases it
on two. Inverse accuracy on replay ranges from 81.1% to 88.5%, with no
goal reached. Both reports export the generated plots in HTML and Org.

### Never Give Up-style episodic novelty

The native implementation freezes inverse-dynamics features per episode
and retains the neighbor distance scale across resets. The independent
audit reconstructs the kernel bonuses on all 6,144 learned-policy actions
and replays both Q-tables for all six runs. Coverage increases from
18/17/15 to 25/26/33 cells on the three seeds. Neither condition reaches
the goal. The report uses native network initialization and exports its
coverage plot to HTML and Org.

### Bootstrapped DQN on a sparse grid

The report uses native initialization and reproduces the source's
five-head architecture, persistent episode head, Bernoulli replay masks,
and target update schedule. Independent replay checks 4,096 transitions
and all masks, and reconstructs terminal targets and masked gradients.
The run visits all 49 cells and reaches the goal 27 times during
training. The coverage plot exports to HTML and Org.

### Go-Explore on a sparse grid

The discovery trace, observed graph, recovery labels, archive goal path,
and all 128 noisy imitation trials match TensorFlow exactly. The independent
audit reconstructs archive updates, shortest paths over observed edges,
and noisy actions. The archive covers all 49 cells, finds the goal in round
10, and retains a 16-action path. Native imitation loss falls from 1.323688
to 0.001348. HTML and Org include the archive coverage plot.

### Hindsight experience replay

The native grid corpus, replay order, future relabel choices and final
Q-tables match TensorFlow exactly. Independent replay verifies all 64,854
updates and evaluates all 2,352 ordered start-goal pairs per learner.
The control solves 729 pairs and future-goal replay solves 1,848, with
32,427 updates per learner. The evaluation plot exports to HTML and Org.

### Universal value function approximation

The fit split, held-out split and action-value targets match upstream
exactly. Independent reverse graph search verifies all 1,920 held-out
action targets and 480 policy rollouts across ten unseen goals. With
native initialization, held-out MSE falls from 0.036151 to 0.001102.
The greedy policy solves 473 of 480 pairs, each on a shortest path.
The prediction plot exports to HTML and Org.

### Option-Critic

Native training preserves call-and-return execution, the arrival-value
mixture and the stop-gradient placements. The seeded native initializer
gives different early rollouts. The final two groups reach the goal in
64/64 episodes each, as in TensorFlow. Independent derivatives match
the continuing and terminal actor and termination gradients. The
training plot exports to HTML and Org.

### Discrete HIRO and HAC replay

Segment traces and worker logits match TensorFlow exactly for both
methods. An independent audit of 1,536 segments reconstructs the
likelihoods, replay goals, failed HAC subgoal tests and goal arrivals.
Native log-likelihoods agree within 1e-6. Greedy evaluation solves
four of eight starts for HIRO and two for HAC. The training plot
exports to HTML and Org.

### Distral policy sharing

An independent fixed-point and linear-system audit verifies the task
policies, discounted occupancies, and distillation on all three rounds.
Saved native arrays agree with TensorFlow within 1.1e-6. Greedy policy
success is 24/24, 21/24, and 21/24. The final shared right-action
probability at the center is 0.659965. HTML and Org include the
shared prior plot.

### FeUdal directional manager and worker

The gradient-driven run uses native initialization and the source
training schedule. Independent reconstruction verifies all 546 manager
directions and 1,638 worker rewards against the recorded motion. The
native policy solves all 48 starts in greedy evaluation, matching
TensorFlow's final success count. Per-phase training trajectories vary
with initialization. The training figure exports to HTML and Org.

### Behavior cloning and coordinate policy

The coordinate policy report fits the bottom-row directional change.
The complete cloning experiment reproduces all 3,200 expert labels and
all 48 closed-loop traces exactly. The independent audit verifies action
agreement and goal success for 36 seen and 12 unseen cells. Native
training NLL decreases from 0.487829 to 0.062679 over three epochs.
Both plots export to HTML and Org.

### DAgger on a grid with blocked edges

Initial demonstrations and both aggregation traces match TensorFlow
exactly. An independent replay verifies 2,298 expert queries, their
mixed and learner-only actions, and all greedy evaluations. From
48 starts the policy reaches the goal on 5, 18, and 0 trials across
the three checkpoints, matching upstream. The report plots that
nonmonotonic result and exports to HTML and Org.

### Conservative Q-learning

An independent audit reconstructs both Bellman and conservative
terms and every gradient. Final native arrays differ from TensorFlow
by at most 1.1e-6. With the matched update budget, the control solves
zero of four starts and CQL solves all four. The report plots the
unobserved action values and exports to HTML and Org.

### Discrete batch-constrained Q-learning

An independent audit reconstructs both 30-sweep fits, their support masks,
targets, greedy choices and rollouts. All recorded arrays match TensorFlow
exactly. The control solves zero of four starts and selects three unsupported
actions. BCQ solves all four and selects only supported actions. The
comparison plot exports to HTML and Org.

### Fitted Implicit Q-Learning

The independent audit reconstructs 30 sweeps, expectiles and the fitted
actor. Q-values, state values, TD targets and policy match TensorFlow
exactly. Actor weights agree within 1e-6. The actor solves all four starts.
The value plot exports to HTML and Org.

### Imitation chain and exact-occupancy GAIL

Independent state propagation reproduces the fixture occupancies and
goal probability. The GAIL audit also reconstructs both objectives and
all four policy checkpoints. Native arrays agree with TensorFlow within
4.8e-7. Goal probability rises from 0.25 to 0.999750 while occupancy
total variation falls from 0.562465 to 0.017619. Both reports export
with plots in HTML and Org.

### Adversarial inverse reinforcement learning

The four AIRL checkpoints independently reproduce occupancy, shaped scores,
discriminator logits, both objectives and goal probability. Saved arrays
agree with TensorFlow within 1.7e-6. The stochastic goal probability
increases from 0.25 to 0.999371. The plot exports to HTML and Org.

### Maximum-entropy inverse reinforcement learning

Independent enumeration of all 256 trajectories verifies the dynamic
program's log partition and reward gradient. Trajectory features and
sampled IDs match TensorFlow exactly, and fitted quantities agree
within 2.4e-7. Held-out NLL falls from 5.545177 to 4.110981. The plot
exports to HTML and Org.

### TD3 with behavior cloning

All 1,024 offline transitions and probe noise match TensorFlow.
Independent checks verify smoothed actions, clipped twin targets
and the scaled actor loss. Training makes 96 critic updates and 48
actor updates. With native initialization, return is -0.399751 against
-0.199083 for the behavior policy. The upstream trained return is
-0.614211. The policy plot exports to HTML and Org.

### Decision Transformer

The offline corpus and evaluation states match TensorFlow exactly.
The native fit independently verifies 4,608 return-to-go labels and
1,536 greedy rollout actions and return updates. All validation
positions pass exact causal invariance. The high and low requests
produce 768 and zero matched bits, respectively. Validation NLL falls
from 0.877288 to 0.271035. The plot exports to HTML and Org.

### Trajectory Transformer

The held-out corpus and complete beam-plan and rollout trace match
TensorFlow exactly. The native model uses its own seeded initializer
and lowers held-out NLL from 2.280730 to 0.212238. Independent replay
verifies all executed action and successor pairs. Both fits solve all
four starts. The native prefix check compares every held-out position
in the first sequence. The likelihood plot exports to HTML and Org.

### Independent multi-agent Q-learning

All 1,978 joint transitions and both final Q-tables match TensorFlow
exactly. Independent replay verifies 372 successful training episodes
out of 384 and all 16 greedy start pairs. The training plot exports
to HTML and Org.

### VDN and QMIX

All 96 joint transitions match upstream. Independent checks reconstruct
joint values, targets and per-agent gradients for both mixers and
verify QMIX's decentralized maximum for every state and joint action.
Both native fits reduce TD error but solve zero of 16 starts. The
upstream VDN fit solves all 16 and its QMIX fit also solves zero.
The native example uses the Datamunge initializer. The comparison
plot exports to HTML and Org.

### CommNet majority coordination

Independent checks verify both rounds of peer means, permutation
equivariance and the exact expected team reward. With native seeded
initialization, the trained policy scores 0.971735 with messages and
0.249950 with messages muted. TensorFlow scores 0.997499 and 0.249986
on the same eight contexts. The comparison plot exports to HTML and Org.

### Board action values and Monte Carlo self-play

The board-policy fixture verifies a winning-action fit and an independent
snapshot. The full report independently replays 576 training games and
128 evaluation games, validating all 4,187 outcome labels. With native
initialization, the win/draw/loss counts are 90/3/35, compared with
TensorFlow's 90/2/36. Both plots export to HTML and Org.

### League self-play

The report reuses the board value policy and trains against frozen
snapshots. An independent audit scores every greedy move from the
archived parameters and verifies all 288 training matches, 980 labeled
positions and 504 evaluation matches. The eight native snapshots give
different matchup scores from TensorFlow under the same schedule.
The round-robin plot exports to HTML and Org.

### MADDPG on coupled scalar dynamics

All 128 initial random transitions match TensorFlow. Independent
checks reconstruct all 1,152 native transitions, reward pairs and
both critics' TD targets. Native joint return is -0.227750, compared
with TensorFlow's -0.232039 and a shared zero-action return of
-0.967093. The rollout plot exports to HTML and Org.

### Multi-agent PPO

Independent checks reconstruct GAE, centralized targets and per-agent
ratios for all 3,410 transitions. The native seeded fit records 30, 76
and 101 training arrivals across the three phases, compared with 33,
78 and 107 upstream. Both greedy policies solve all 16 start pairs.
The training plot exports to HTML and Org.

### Direct preference optimization

Prompts, preference draws, completion tokens and the frozen reference
match TensorFlow exactly. An independent audit verifies 512 held-out
comparisons and all 40 gradient coordinates. The final fitted policy
agrees within 4.8e-7. Held-out NLL falls from 0.693147 to 0.566060 and
requested completion probability rises from 0.212500 to 0.482635.
The plot exports to HTML and Org.

### Preference reward and smoothed bigram reference

All 768 held-out segment pairs, oracle scores and labels match TensorFlow
exactly. An independent audit verifies model scores and 80 gradient
coordinates. Native initialization gives held-out NLL 0.787558 initially
and 0.418513 after fitting. The standalone bigram fixture reproduces all
32 completion probabilities across two prompts. DPO reruns with identical
reported results using the shared smoothing API. Plots export to HTML and Org.

### Token-level RLHF PPO

The preference labels, token draws and sampled histories match upstream
exactly. Independent replay verifies 512 labels, 768 sampled completions
and 445 active clipping cases. Maximum difference across saved arrays
is 1.5e-6. Exact requested completion probability rises from 0.212500
to 0.649033. The training figure exports to HTML and Org.

### RL² recurrent bandit policy

All held-out inputs, actions, rewards and environment path masses match
TensorFlow exactly. Independent recurrence checks verify each of the
1,024 paths in both memory conditions. The native seeded policy yields
2.395386 expected reward with carry and 2.247523 with reset. TensorFlow
yields 2.355648 and 2.221565. The plot exports to HTML and Org.

### MAML for contextual bandits

The held-out task tables match TensorFlow exactly. The upstream audit
reconstructs all 24 support outcomes and checks both meta-gradient
coordinates against analytic terms and finite differences at the native
fit. Expected held-out reward after adaptation rises from 0.573388 to
0.599403. The largest difference between the two training histories is
0.000320 in expected reward, and the fitted weight vectors differ by
up to 0.019719. The report and its plot export to HTML and Org.

### PEARL with discrete soft actor-critic

All 768 held-out replay rows, task values and latent noise draws match
TensorFlow exactly. The upstream audit reconstructs the Gaussian
posterior, sampled latent, twin critic predictions, policy and losses
at zero, one and four observations. With native initialization, expected
reward rises from 0.510998 to 0.855463, compared with 0.512552 to 0.856337
in TensorFlow. The four-observation mean posterior variance is 0.083952
in Datamunge and 0.077257 in TensorFlow. The report exports to HTML and Org.

### Loss-based membership leakage

Both seeded input matrices and label vectors match TensorFlow exactly.
The largest absolute loss difference is 2.3e-15. An independent NumPy
solve and threshold scan reproduce the calibration cutoff of 0.365615,
61 true positives and 10 false positives out of 64 rows per group, and
AUROC of 0.848633. The loss-distribution plot exports to HTML and Org.

### Schema-based training-data integrity

TensorFlow and Datamunge use identical clean and corrupted rows, labels,
probe inputs and eligible row indices. A NumPy audit independently
identified all 132 altered rows and confirmed that the repaired probes
produce identical logits to the valid test inputs. With the native
initializer, the corrupted fit falsely flags 194 of 194 negative probes
and the defended fit flags 9. TensorFlow flags 194 and 7, respectively.
The accuracy and false-positive plot exports to HTML and Org.

### Shared patch robustness

The seeded images, classifier kernel, all 145 patch states, held-out
patched images and both logit arrays match the TensorFlow archive
exactly. An independent NumPy calculation verifies all 144 updates and
the held-out insertion. At the unseen placement, all 43 originally
correct class-zero images switch to class one. The margin plot exports
to HTML and Org.

### Linear model inversion on synthetic bars

The images, labels and templates match TensorFlow exactly. After 55
projected updates, the recovered pixels agree with TensorFlow within
2.6e-15 and with the independently solved box-constrained optimum
within 1.9e-15. Template MSE is 0.141284 for horizontal bars and
0.149823 for vertical bars. The image grid exports to HTML and Org.

### Adversarial training on the unit square

The first training batch and held-out features and labels match
TensorFlow exactly for all four fits. A NumPy audit verifies the pixel
and displacement bounds, both cross-entropies, the categorical KL
divergence and the first-batch training objective. Native initialization
gives clean/PGD correct counts of 231/128, 218/128, 217/129 and 220/133
out of 256 for the baseline, FGSM, PGD and TRADES fits, respectively.
TensorFlow gives 226/128, 213/132, 213/132 and 218/134. The accuracy
plot exports to HTML and Org.

### Gradient robustness on the unit square

The seeded test and calibration data match TensorFlow exactly. A NumPy
replay reconstructs the fitted classifier's input derivatives and
checks the FGSM update, all 12 PGD steps, all 14 DeepFool steps,
all 24 shared perturbation steps, and the final Carlini–Wagner
selection across 384 candidate updates. Native correct counts out of
256 are 231, 127, 128, 0, 25 and 194 for clean, FGSM, PGD,
Carlini–Wagner, DeepFool and shared perturbation respectively.
The TensorFlow counts are 226, 129, 130, 0, 27 and 190. The figure
exports to HTML and Org.

### Additive and global dot attention for toy sequence tasks

The 192 generated rows per task and all held-out token tensors match
upstream exactly. The native models use the upstream 1,718-parameter
Bahdanau and 1,462-parameter Luong graphs. The forward portion of the
upstream NumPy audit verifies all held-out logits, alignments, token NLL
and greedy sequences for both tasks. Native and TensorFlow initializers
differ. Held-out reversal/addition NLL is 2.127567/1.863096 for Bahdanau
and 2.090509/1.855340 for Luong. The combined validation curve exports
to HTML and Org. The pinned English–Spanish checklist entry remains
separate.

### Luong English–Spanish translation

The pinned corpus, vocabulary, attribution and held-out token tensors
match the TensorFlow fixture. The upstream NumPy audit reconstructs
teacher-forced logits, all attention alignments, token loss and greedy
sequences for 64 test pairs using the native weights. After 96 updates,
held-out NLL is 2.934701 in Datamunge and 2.934005 in TensorFlow.
Both fits score 64 of 326 teacher-forced tokens and zero of 64 greedy
exact matches. The validation plot exports to HTML and Org.

### MLP-Mixer and ConvMixer

The MNIST training, validation and test IDs, held-out pixels, labels
and extracted patches match TensorFlow exactly. The upstream NumPy
audit reconstructs the two patch embeddings, all 256 logit rows per
model and their NLL, including ConvMixer running normalization.
Datamunge scores 206/256 for MLP-Mixer and 194/256 for ConvMixer.
TensorFlow scores 199/256 and 189/256, respectively. The validation
loss plot exports to HTML and Org.

### Three-stage Pyramid Vision Transformer

The MNIST split IDs, held-out images, labels and patches match the
TensorFlow fixture exactly. The upstream NumPy audit replays all
transition neighborhoods, the 16-by-4, 4-by-1 and 2-by-2 attention maps,
and the 256 held-out predictions with the native weights. Datamunge
scores 205/256 at NLL 0.617389. TensorFlow scores 202/256 at NLL 0.629910.
The validation plot exports to HTML and Org.

### Shifted-window Swin Transformer

The MNIST split IDs, held-out pixels, labels and patch tensors match
TensorFlow exactly. The upstream NumPy audit verifies regular, shifted
and coarse attention, all 21,504 masked wraparound probabilities,
patch merging and all 256 held-out logit rows. Datamunge scores 201/256
at NLL 0.624856. TensorFlow scores 211/256 at NLL 0.546147. The validation
loss plot exports to HTML and Org.

### Offscreen vsim rollout recording

The companion CartPole report runs the seed-5 controller for all 500
transitions and records the reset and every subsequent observation.
The scene uses vsim's offscreen Vulkan renderer and Datamunge's ffmpeg
writer. FFprobe decodes all 501 frames at 320 by 200 and 50 fps, with
500 distinct decoded frame hashes. The clip embeds in HTML and exports
to Org. The four-controller report supplies the independent upstream
trajectory comparisons. Validation used the installed 3D SDK on an RTX
5080 Laptop GPU with a local pkg-config overlay that selected the
active compiler's libm instead of the SDK's older glibc runtime.

### DeiT hard distillation token

The MNIST split IDs, held-out pixels, labels and patches match
TensorFlow exactly. The upstream NumPy audit verifies both attention
maps, both student heads, frozen teacher hard labels, averaged inference
and class-head NLL for all 256 test images. The native teacher scores
236/256 and remains unchanged during student fitting. The student
scores 166/256 at class-head NLL 1.070077. TensorFlow's student scores
156/256 at 1.176424. HTML and Org exports include the validation plot.

### Linformer sequence projections

The MNIST split IDs, held-out images, labels and patches match
TensorFlow exactly. In both blocks, the independent NumPy audit
verifies that the key and value matrices retain rank five and change
during training. It reconstructs both compressed attention maps
and all 256 held-out predictions. Datamunge scores 202/256 at NLL
0.774667. TensorFlow scores 194/256 at NLL 0.744287. The validation
figure exports to HTML and Org.

### Causal Sparse Transformer

The MNIST split IDs, held-out images, labels and patches match
TensorFlow exactly. The upstream NumPy audit reconstructs both
packed factors with dense causal masks, checks all empty slots,
and verifies the final logits and token layout. Changing the final
eight patches leaves the first eight attention rows in each block
bit-identical. Datamunge scores 173/256 at NLL 0.977960. TensorFlow
scores 155/256 at NLL 1.128664. The validation figure exports to
HTML and Org.

### Performer FAVOR+ orthogonal features

The MNIST split IDs, held-out pixels, labels and patches match the
TensorFlow fixture exactly. The upstream NumPy audit verifies
orthogonality within each feature block, positive normalizers,
both attention computations and all 256 held-out logits. The native
seeded projections yield mean absolute attention probability errors
of 0.02960 and 0.02660 against exact softmax in the two blocks.
Datamunge scores 168/256 at NLL 1.071189. TensorFlow scores 154/256
at NLL 1.143035. The validation plot exports to HTML and Org.

### Longformer and BigBird window attention

The MNIST splits, pixels, labels and patches match TensorFlow exactly.
The upstream dense NumPy reconstruction verifies both packed maps
per model, including masked edge slots, seeded random connections,
separate Longformer global projections and all 256 logits rows.
Datamunge scores 161/256 and 167/256, respectively. The validation
plot exports to HTML and Org.

### Reformer LSH chunks

The MNIST splits, held-out images, labels, patches and both Gaussian
rotations match TensorFlow exactly. The independent NumPy audit
reproduces the stable bucket order, masks and probabilities for both
sorted attention traces, as well as all 256 held-out logits.
Datamunge scores 70/256 at NLL 1.999889. TensorFlow scores 90/256
at NLL 1.806947. The validation plot exports to HTML and Org.

### CoAtNet relative attention

The MNIST splits, held-out images, labels and patches match TensorFlow
exactly. The upstream NumPy audit reproduces the final MBConv features,
both relative attention maps and all 256 held-out logit rows. Datamunge
scores 153/256 at NLL 1.125837. TensorFlow scores 167/256 at NLL 1.012843.
The validation plot exports to HTML and Org.

### Perceiver and Perceiver IO

The MNIST splits, held-out images, labels and patches match TensorFlow
exactly. The upstream NumPy audit reproduces both passes of shared
latent attention, both classification heads, the Perceiver IO decoder
attention and all 4,096 coordinate intensities. Datamunge scores 159/256
and 132/256, compared with TensorFlow's 165/256 and 143/256. Native
patch intensity MSE is 0.015815 versus 0.013231 for the fitting-set mean
baseline. The validation plot exports to HTML and Org.

### Pixel-observation DQN checklist audit

All six Snake blocks reran with identical results and exported to HTML
and Org. A fresh run of the Atari report against ALE 0.11.2 and its
bundled ROM completed the 2,008-transition check and produced three
finite replay losses plus a decodable 475-frame video. The archived
full run contains 7,001 finite losses, 55 finite returns and a
decodable 205-frame video. Both report paths are now recorded on
checklist row 571.

The CPU runs verify the Snake and Atari model and environment variants.
The checklist explicitly names GPU execution. Both current native
pixel Q entry points require CPU inputs. To verify the remaining
backend variant, exercise a replay update with CUDA-accelerated
convolutional and dense products and compare its Q values, loss and
gradients against the CPU graph on the same replay batch.

The inventory checker reconciles all 642 declared CMake target names
with their report rows. A fixture with an extra unmapped CMake target
fails validation as expected.

### CUDA pixel replay and completion checks

The CUDA pixel-DQN report consumes eight actual Snake transitions and
eight frame-pooled ALE transitions with the bundled ROM. Both original
architectures use CUDA for convolutional and dense products in the
forward and backward passes. Replay, im2col packing, activations and
Adam remain on the host. On the local NVIDIA device, the CPU and CUDA
Huber losses agree at the reported precision, and the maximum gradient
difference is 1.49e-8 for each network. The post-update Q values also
pass the comparison. Validation used the existing working-tree library
with its CUDA backend compiled and loaded separately. A regular CUDA
build enables the same backend with `DATAMUNGE_ENABLE_CUDA=ON`.

The inventory checker reconciles 642 CMake target declarations, 655
upstream report rows and 701 checklist items. For each distinct
Datamunge destination it checks for executable blocks, captured
results. With `--check-media`, it also verifies referenced local figures
and videos in a checkout with generated media. These are
structural checks. The per-example execution and parity evidence is
recorded in the reports and in this document.
