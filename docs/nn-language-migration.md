# Native NN example migration

The active goal is genuine Python and R implementations of every canonical NN
C++ example. Models, data generation, forward passes, losses, training loops,
evaluation and experiment checks belong in the associated scripts. Shared
high-level experiment `Run` functions and compiled example adapters do not count
as language ports. Reusable tensor operations, layers, autograd, optimizers and
numerical primitives remain in the canonical C++ library, exposed through SWIG.

## Inventory and validation

`just nn-language-status` refreshes `examples/nn/language_status.json`. It scans
both canonical source roots: sections 1 and 2 under `examples/cpp/nn`, and the
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

## Current checkpoint

Thirty-six examples have native Python and R implementations: `cnn_mnist`,
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
