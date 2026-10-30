# Moving `examples/mepml` into `docs/mepml/examples`

Data preparation policy: executable reports obtain their inputs through the
canonical C++ library (`dm::datasets::fetch()` or
`dm::datasets::prepare_example_data()`). Implement missing download, archive,
metadata and numeric-format support in datamunge; do not introduce Python data
scripts, manual archive placement, pip/venv setup, or calls into another checkout.
Use the shared configurable cache, retain verified source provenance, and keep
independent framework oracle generation in optional development tests.


This is the working plan for rewriting the 528 literate reports in
`examples/mepml/*.mepml` as polished reports in `docs/mepml/examples/`, one at a
time. It is written so the work can stop and resume at any point: the checklist
at the bottom is the state, and the first unchecked entry is the next report.

## What the user asked for

- Move the reports one at a time, **rewriting** each one, and **delete the old
  `examples/mepml/<name>.mepml` once its replacement is finished**.
- **Move code into datamunge** whenever a report contains logic that belongs in
  the library (helpers, data generators, summaries, plotting recipes). The
  report's code should only **showcase the high-level API**: short programs
  that call library functions, not re-implementations.
- **Simple worked examples come first** (user, 2026-10-28): open the method with
  small, hand-checkable examples, each with a figure and the maths it
  illustrates (a few points, a tiny graph, a short sequence), so a reader sees
  the mechanism before any real data.
- **Use TikZ concept diagrams when appropriate** (user, 2026-10-28): near the
  introduction of a topic, illustrate its mechanism or geometry with a clear diagram
  when that helps (aligned symbols, graphs, state transitions, partitions, etc.). Keep
  the editable TikZ source and a repeatable rendering command. A diagram should teach
  the concept; use ordinary plots for measured data and numerical results.
- **Save images as local files, not inline SVG output** (user, 2026-10-28):
  render into the report's ignored `build/` directory (or another local build directory),
  then reference/copy the image with `file="figures/..."` and `\image(...)`. Do not
  print SVG XML into result blocks, use SVG/data-URI text as a figure, or embed a
  base64 image in report output. SVG is welcome as a standalone image file.
- **A dataset example comes second, when it fits.** Where the method has a
  natural use on real data, analyse a `dm::datasets::<name>()` object directly
  (add one to `dm::datasets` if needed, see "Adding a dataset"). Many algorithm
  reports (number theory, data structures, geometry kernels, sorting, hashing,
  ...) have no meaningful dataset: skip it rather than force one, and use
  generated inputs (random point sets, synthetic workloads) for scaling and
  property checks instead.
- Produce, where appropriate:
  - **tables** summarizing the data and results (`to_md()`, pipe verbs);
  - **figures** analysing the dataset (plots saved as SVG);
  - **figures and mathematics describing the method**, with derivations
    ("derive things mathematically if possible"): state the problem, derive the
    algorithm or estimator, prove or sketch its key properties (correctness,
    complexity, convergence, bias), and connect the maths to the code's output.
- Use the existing report as a starting point and expand it.
- `docs/mepml/examples/iris_lda.mepml` is the model, but each new report should
  be better than it, especially in describing and deriving the method.
- Keep going until every report is converted.

## Standard for a finished report


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

### Neural-network architecture and confusion figures (2026-10-30)

CNN reports show executed tensor architecture figures when a network is defined.
Use named layer/block boundaries, preserve every branch input, and show the traced
configuration explicitly. Fit the canvas to glyphs and rotated labels, use shared
shape scales, and retain native horizontal scrolling for wide graphs. Parameter-only
transform chains belong near their consumers; avoid tall lists beside the input.
Multi-input stages use the `model_stage(name, vector<Tensor>, output)` overload.

Classification and dense segmentation reports use `ConfusionMatrix::save_svg()`
(or save `to_svg()` to a local file) for annotated matrix images. Prefer the actual
held-out predictions; `from_counts()` also accepts exact integer tensor counts.
True classes run top to bottom, predicted classes left to right. State normalization
and ignored-label rules. Explicitly identify frozen-table replots and aggregate
multi-label decisions. Keep AP, overlap, regression, retrieval and sequence metrics
where those describe the task; do not force a single-label confusion matrix onto them.
Never print SVG XML or data URIs into report results.

### Report contents

1. Front matter: `//? Title:`, `//? Subtitle:`, `//? Exports: both`, `//? Lang: en-GB`;
   an `\abstract(...)` and `\toc`. **The abstract opens by stating the problem the
   method solves** (user, 2026-10-28): what goes in, what comes out, and why it is
   hard or worth solving, in one or two plain sentences, before any mention of the
   method itself, its history or its complexity. Then summarise what the report
   shows and its main findings.
2. Sections, adapted to the topic, in this order:
   - **The method**: the problem stated precisely, then a derivation of the
     algorithm/estimator, its key properties (with proofs or proof sketches)
     and complexity.
   - **Simple examples**: small worked examples, each with a figure and the
     maths it illustrates, small enough to check by hand (e.g. ten points, a
     six-node graph). Show the algorithm's intermediate state, not just its
     answer. These come before any dataset.
   - **Checking the theory**: generated inputs (random, worst-case, scaling
     sweeps) with tables/figures that test the claims: a bound that holds, a
     closed form that matches, a growth rate that fits.
   - **A dataset example** (optional, when it fits): which `dm::datasets`
     object, its shape and a summary table, then short programs using the
     high-level API on it, with tables and figures of the results.
   - **Conclusion**: what the analysis showed, and limits of the method.
3. Every runnable block is a complete program that starts with
   `#include <dm/dm.hpp>` and uses the standard flags
   `flags="-std=c++23 -O2 $(pkg-config --cflags --libs datamunge) -Wl,-rpath,$(pkg-config --variable=libdir datamunge)"`.
   Blocks are short; anything long or reusable moves into the library.
4. Figures: programs write `build/<name>.svg`; the block has
   `file="figures/<name>.svg"`, a `\caption(...)` and an `\alttext(...)`.
   Figures are committed (`.gitignore` re-includes report SVG and PNG assets).
   Pixel-image contact sheets may use `build/<name>.png` and a matching `file=`
   instead: use tensor `save_image_grid` with explicit pixel scaling/spacing,
   and identify the class order in the caption. PNG keeps original image pixels
   compact; keep mathematical plots and editable concept diagrams as SVG.
   **All figure bytes stay in local image files:** use `.save_svg("build/...")` or a
   renderer's file output, never `std::print("{}", plot.to_svg())` or equivalent XML
   dumps. Keep only paths/references and ordinary results in the report.
   For a useful introductory concept diagram, retain a TikZ `.tex`/`.tikz` source
   under `diagrams/` and compile it into `build/`; copy/reference the rendered image
   through `file=` as above. A short shell block may run LaTeX/dvisvgm (it is a
   figure-build step, exempt from the complete-C++-program rule). Document its tool
   requirements, preserve compile logs locally, and visually inspect the exported
   diagram alongside plots. The docs export consumes committed rendered figures and
   does not require rerunning TeX.
   **Make every figure render well in HTML** (user, 2026-10-28). The Doxygen page
   scales each figure to the width of the text column, about 700 to 1100 px:
   - Multi-panel `RLayout` figures must call `.fit_panels()` after `.size(...)`
     (and after `.title(...)`). Without it every panel keeps its 960x640 canvas
     and is shrunk into its cell: tiny text, wide blank bands, and anything near
     a panel's edge (legends, last tick label, long titles) is cut off, because
     browsers clip each panel's nested `<svg>` at its border.
   - At most two panels side by side, at `size(1100, 450..520)`; use a 2x1 or
     2x2 grid for more. A single plot is 600 to 900 px wide. Keep the figure at
     most about 1150 px wide, so text is not scaled below a readable size.
   - Keep everything inside the plot: short panel titles (about 45 characters
     at 1100 px for two panels), legends with headroom (`y_limits`), text labels
     away from the right edge. RPlot does not clip, so lines and segments must
     stay within the axis limits.
   - Check figures as a browser would draw them, with
     `inkscape X.svg --export-type=png --export-filename=out.png -w 1100`.
     ImageMagick (`magick`) ignores the nested-`<svg>` clipping, so it hides
     clipped content; use it only for a quick look.
5. Tables: `results="markdown"` blocks ending in a table, followed by
   `\caption(...)`.
6. Maths uses `\( \)` inline and `$$ $$` display, each display followed by an
   `\alttext(...)` reading of the formula.
7. All results are produced by actually running the report (below); nothing is
   typed in by hand.

## Procedure for one report

1. Read `examples/mepml/<name>.mepml` (and the library code it uses).
2. Design the simple examples and any useful introductory TikZ diagram first;
   ensure all rendered images are saved as local build files, not inline output.
   Then decide whether a dataset fits
   (add one to `dm::datasets` if needed) or the report is better without one.
3. Move reusable logic into the library: header in `include/datamunge/...`,
   source in `src/datamunge/...`, Doxygen comments, a gtest in `tests/cpp/`,
   `lib_sources.txt` for new `.cpp` files. Keep API names consistent with the
   rest of the library. Python/R bindings are not required for report-only
   helpers.
4. Write `docs/mepml/examples/<name>.mepml` (keep the file name; drop a
   redundant `_analysis` suffix only if it reads better, and note it below).
5. Build the working-tree library and run the report:
   ```sh
   nix develop .#cpp-tools --command cmake --build build/debug/tests --target datamunge
   PKG_CONFIG_PATH=$PWD/build/mepml-pc python3 scripts/mepml_run.py docs/mepml/examples/<name>.mepml
   ```
   `build/mepml-pc/datamunge.pc` points `pkg-config` at `include/` and
   `build/debug/tests/_deps/datamunge-build` (the installed copy is stale). The path
   must be absolute: blocks run in the document's directory.
   To view a figure, `inkscape figures/X.svg --export-type=png --export-filename=x.png -w 1100`
   renders it the way a browser does (clipping included).
6. Read every result and figure (render figures with inkscape, see 4); check numbers against theory or an
   independent oracle (R, or `sklearn_oracle_env` Python).
7. Check the export: `mep-mepml convert docs/mepml/examples/<name>.mepml /tmp/x.md`.
8. Delete `examples/mepml/<name>.mepml` and any `examples/mepml/figures/<name>*`.
   Also delete `examples/org/<name>.org` (and its `.tex`/`.html`/figure leftovers),
   with `git rm -f`, and repoint `list-of-algorithms.txt`.
9. Tick the box below (with notes: dataset used or "no dataset", library additions).

## Notes and decisions

- 2026-10-29: Converted the nine remaining continuous-optimization reports, one
  at a time: adaptive gradient, overview, constrained, coordinate/CG,
  derivative-free, nonlinear least squares, proximal, second-order and variance
  reduction. All open with the numerical problem and hand-checkable controls;
  nine retained TikZ diagrams introduce the mechanisms. Iris appears where a
  regression objective is meaningful; generated fixtures isolate curvature,
  constraints, residual fitting, local basins and stochastic scaling elsewhere.
  Shared native `optim/analysis.hpp` fixtures and typed callback recording replace
  duplicated report implementations. Histories distinguish callback requests,
  rejected trials, look-ahead points and explicit post-run replay; native return
  is never presented as a uniform convergence certificate. KKT, proximal-mapping
  and residual-gradient checks are used for their respective interfaces.
  SVRG's snapshot gradient now divides the full sum by n, matching its component
  estimator's mean-loss scale; SGD rejects a zero mini-batch size instead of
  looping indefinitely. Existing option layouts and native algorithm variants
  otherwise remain. Reports explain the restricted projection-like SQP,
  Cauchy-only trust-region update, fixed-beta Nadam, bias-corrected AMSGrad,
  loss-change stopping failures, rejected LM proposals and the default barrier
  run's late exception. No global-optimality or unequal-budget speed ranking is
  inferred from a successful return or seed scan.
  All 37 complete C++ programs and nine TikZ builds execute; 46 local SVG figures
  were rendered with Inkscape at 1100 px and visually reviewed. All 76 optimizer
  tests pass (12 analysis/regression tests), all 175 affected header examples
  compile/run with matching outputs, and twelve clangd hovers pass. NumPy/SciPy
  provide 121 independent comparisons, including exact stochastic trace replay,
  OLS/lasso controls and both original noisy nonlinear curve recipes (maximum
  discrepancy 4.12e-9). Markdown and HTML exports pass for all nine reports using
  the current website export path. The affected 60 header/source files pass
  547/547 documentation definitions; the completed-path gate passes 10914/10914
  at this run. Graphical mep K integration was not available for a UI check.
  The 27 old mepml/org/TeX report copies were removed; standalone C++ and neural
  network optimization examples remain. Evidence: `build/optimization-verification/`.

- 2026-10-29: `mersenne_twister64` converted around exact twist words and all
  four tempering stages before seeded conformance, aligned-block state recovery and
  finite-sample quality diagnostics. A retained TikZ diagram distinguishes seed,
  ordered in-place refill, invertible output transform and upper-53-bit uniforms.
  The native MT engine delegates to inspectable twist/temper steps without changing
  its scalar-seeded stream. Added inverse tempering, raw state/index inspection,
  an aligned 312-output continuation factory with a fresh normal cache, and a
  standard-engine conformance table whose 64-bit values remain exact hex strings.
  Period/storage/effective degree, scalar-seed entropy, call order, cached normal
  pairs, arbitrary degenerate restored states and the distinct 32-bit engine are
  explicitly separated. No external dataset or throughput ranking is forced.
  The report compares 80000 raw words with std::mt19937_64 and predicts 4096
  previously unseen words without error. Single-seed low-byte p=.035514 and
  seven serial-pair p-values above .95 across forty seeds are retained and
  interpreted without post hoc seed changes or an all-tests-must-pass rule.
  Six complete C++ programs plus a TikZ build generate seven local SVGs; all were
  rendered in Inkscape at 1100 px and visually inspected. All 56 generator/quality
  tests (11 new), eight compiled/executed header examples, eight clangd hovers and
  Markdown/Doxygen exports pass. An independent integer/bit-triangular Python
  implementation checks 160000 stream words, exact recovered continuations and
  inverse operations (169246 exact comparisons); NumPy/SciPy replay all 41 batteries
  from 5740000 generated words and compare 674 floating values. Maximum p-value
  difference is 1.68e-13. New analysis files pass 9/9 documentation definitions;
  the MT class's nine contracts also pass. The wider shared random core's existing
  backlog remains outside the completed-file manifest. The completed-path gate
  passes 9920/9920 at this run. Legacy mepml/org sources are removed; no standalone
  C++ example was present. Evidence: `build/mt19937-64-verification/`.

- 2026-10-29: `kernel_ridge_analysis` converted with exact two-point RBF and
  linear-kernel calculations, signed extrapolation weights, a four-row fixed-metric
  versus preprocessing-refitted LOO counterexample, the original 120-row sine-wave
  simulation and a transparent 120/30 Iris regression split. Canonical KRR now
  implements a genuine free intercept through joint kernel/response centring,
  centred query kernels and df including the intercept. Positive-lambda LOO uses
  the centred inverse diagonal without leverage floors; lambda zero uses a
  relative-rank pseudoinverse and actual fixed-metric deletions. Standardisation
  uses training population SDs without a unit-dependent small-SD threshold.
  Added retained-row/scaling/design/Gram accessors, training/deletion diagnostics,
  spectrum, signed prediction weights, full smoother, objective/solve checks,
  all-candidate selection tables, strict labelled scoring, a seeded wave recipe
  and fixed-candidate fold-local validation with learned metrics and original-row
  predictions. Finite/PSD controls, null/query alignment and plotting guards are
  explicit. Existing LOO accessor semantics remain; the corrected estimator changes
  numerical results. The sine's default finite search selects gamma 15.602620 and
  lambda .170125, with mean-curve RMSE .059972; an independent augmented-system
  sweep verifies this change, and its former .05 recovery test bound is now .08.
  Iris selects RBF gamma .1/lambda .01 inside training rows: validation RMSE .402698,
  held-out RMSE .366721 versus training-mean baseline 1.816220 cm.
  Seven complete C++ programs plus retained TikZ source generate eight local SVGs,
  rendered in Inkscape at 1100 px and visually reviewed. All 23 native tests,
  22 compiled/executed header examples, seven clangd hovers and Markdown/Doxygen
  export checks pass. NumPy augmented-system/deletion and sklearn centred-KRR/primal
  ridge oracles compare 894522 values, all 480 automatic candidates, 1217 actual
  fixed-metric deletions and 48 Iris fold fits. Prediction differences are at most
  2.07e-11; ill-conditioned tiny-penalty dual differences reach 1.02e-6 while
  equation/prediction checks remain small. Both canonical files have 130/130
  documented definitions; the completed-path gate passes 9886/9886 at this run.
  Legacy mepml/org sources are removed; no standalone C++ example was present.
  Evidence: `build/krr-verification/`.

- 2026-10-29: `inla_analysis` converted with an exact three-row normal-normal posterior,
  a four-outcome skew Bernoulli posterior and a fixed-success-fraction sequence,
  uncertain log precision, and separate grid-resolution/span checks. Gaussian-strategy
  conditional marginals, hyperparameter mixing, transformed SD moments and inverse-link
  plug-in predictions are explicitly distinguished from exact posterior quantities.
  Newton exhaustion/failed line searches are rejected, likelihood tails and observation
  support are checked, hyper-Hessian probes stay inside the declared box, scalar grid
  boundary cells are clipped/recentered, and grid weights use their actual measures.
  EB no longer reports a mode density as integrated evidence. The core exposes full
  conditional means/covariances, log weights, scores/iterations, covariance and within/
  between variance, fallback/edge/skipped diagnostics and mixture quantiles/densities.
  The mixed-model facade uses normalized uniform priors on bounded packed-factor/log-SD
  coordinates, retains complete-case indices/design/responses, and reports physical SD
  moments after transforming every grid point. Fixed mean-theta SD getters remain
  explicitly labelled plug-in quantities. Numeric group keys retain full precision;
  prediction handles empty/all-missing rows and random-only missing slopes correctly.
  Penguins: 333 complete observations, centered kg mass, fixed-effect prior SD 2.5 and
  island-SD prior [0.01,3], with independent numerical and prior-support sensitivity.
  At 81 grid points, expanding the SD prior bound from 1.5 to 3 to 5 moves its mean
  from 1.1482 to 1.7909 to 2.2920. All three island mixture intervals include zero;
  species composition and non-integrated response predictions remain explicit limits.
  The separate original ten-p-value example now correctly reports Bonferroni/Holm/BH
  rejection counts 2/2/3; the fourth hypothesis passes none at .05.
  Seven complete C++ programs plus retained TikZ source generate eight reviewed local
  SVGs. All 27 native tests, 33 compiled/executed header examples, eight clangd hovers
  and Markdown/Doxygen exports pass. Independent NumPy/SciPy checks cover 61 fits,
  505 conditional evaluations, 43 direct scalar references and 196961 values; maximum
  scaled replay difference is 7.95e-11. A separate exact-Gaussian/adaptive-theta oracle
  finds span-five SD truncation error 1.19e-5 and span-seven error 7.27e-9, both retained
  as evidence rather than conflated with solver convergence. Original standalone C++
  example is retained, its misleading comments corrected, and it compiles/runs.
  Legacy mepml/org/TeX sources are removed; the algorithm list links here.
  Evidence: `build/inla-verification/`.


- 2026-10-29: `id3_analysis` converted with exact four-row entropy/branch arithmetic,
  the original fourteen-row Play Tennis fixture, XOR zero-gain stopping, identifier
  memorisation, depth prefixes and local-majority prediction traces. Reusable native
  split/branch/node/evaluation tables and small-tree plots replace report algorithms.
  A DataFrame wrapper freezes training-only numeric/string category dictionaries,
  reserves -1 for unseen levels and checks schema, finite cells and integer labels.
  Native training now rejects ragged/misaligned inputs; prediction rejects empty
  trees, short visited queries, invalid visited indices and cycles. Existing valid
  tree behavior, including strict positive computed gain and integer-label ties,
  is retained. The neighbouring continuous routine is documented as a restricted
  C4.5-inspired variant, without claiming its missing features.
  The Titanic example selects four columns, drops only two missing Embarked rows,
  declares 711 fitting and 178 holdout rows, and fixes depth two before evaluation.
  It fits 550/711 and classifies 149/178 holdout rows correctly, versus the fitted
  majority's 108/178. The second question reduces entropy without changing hard
  predictions relative to depth one; this equality is checked explicitly.
  Five C++ programs plus retained TikZ source produce six reviewed local SVGs;
  Markdown/Doxygen exports pass. All 17 focused native tests, 19 compiled/executed
  header examples and six representative clangd hovers pass. An independent Python
  implementation verifies 1152 fits, 7532 nodes and 368751 values, including every
  binary label assignment on the eight cube vertices at four depth caps, 120
  multicategory cases, tennis and four Titanic depth caps. Full tree topology,
  gains, branch entropies, traces, predictions, evaluation, entropy telescoping and
  node bounds agree; maximum numeric difference is 4.45e-16. CSV-derived Titanic
  encoding is checked independently. Old mepml/org reports are removed and the
  algorithm list points here. Evidence: `build/id3-verification/`.


- 2026-10-29: `ica_analysis` converted with an exact uncorrelated-but-dependent four-point
  example, hand-checkable cubic update, retained waveform demonstration, independent
  Laplace/uniform/Rademacher sources, Gaussian controls, uniform-mixture geometry and
  an explicitly descriptive Iris example. Corrected the n-1/n whitening/update mismatch
  to population covariance throughout, made rank checks relative, added finite/schema/
  option guards and explicit zero-update failure, and kept orthonormal directions even
  on iteration exhaustion. Native APIs expose whitening/eigenvalues/rotation/initial
  directions, per-component moments/stopping, every completed update, inverse transforms
  and reconstruction checks. Added known-source generators and one-to-one Hungarian
  matching with sign/scale/offset alignment. No finite-sample convergence, kurtosis,
  whitening or exact reconstruction is described as an independence certificate.
  Six C++ programs plus retained TikZ source produce seven reviewed local SVGs;
  Markdown/Doxygen exports pass. All 17 native tests, 15 compiled header examples and
  six clangd hovers pass. Independent NumPy/SciPy checks cover 35 fits, 4726 replayed
  updates and 433722 values; maximum direction difference is 6.65e-13. Four independent
  scikit-learn deflationary FastICA comparisons also pass. All twelve non-Gaussian
  control fits converge (mean matched correlation .98949..99986); seven Gaussian fits
  exhaust their budgets, including an accidentally high .97614 matching score. Iris's
  two-component reconstruction RMSE is .1591888 cm, without a latent-source accuracy or
  prediction claim. Legacy mepml/org reports are removed and the algorithm list repoints.


- 2026-10-29: `gauss_newton_analysis` converted with exact three-row linear and scalar
  tangent examples, quadratic/linear error recurrences, full-step overshoot and rank
  failure, the original generated sinusoid and NIST's observed 14-row Misra1a benchmark.
  The canonical solver now uses column-scaled Householder QR instead of normal equations
  with silently skipped pivots, supports analytic and central-difference derivatives,
  and exposes local residual/Jacobian/step diagnostics and the full evaluated trace.
  Corrected stale final SSE and exhausted-budget iteration counts. Absolute proposed-step
  stopping is explicitly not a minimum certificate, and full-step increases remain visible.
  Added the provenance-linked embedded Misra1a CSV, generated accessor, catalog/load entry
  and dataset test coverage. Six C++ programs plus retained TikZ source produce seven
  reviewed local SVGs. All 28 focused native tests, nine compiled header examples, six
  clangd hovers and Markdown/Doxygen exports pass. Independent NumPy/SciPy checks cover
  61 fits, 549 linearizations and 12939 values. All scaled-QR steps agree with independent
  SVD solves to relative residual-scaled error 4.52e-14. The scaled NIST fit matches
  certified A=238.94212918, b=.00055015643181 and SSE=.12455138894; distant-start
  overshoot, negative intermediate amplitude and raw-rate finite-difference bias are
  explicit. The tiny raw-rate derivative approximation shifts amplitude by about
  2.86e-5; this is retained and checked using an independent analytic stationarity
  correction, not hidden by claiming exact parameter agreement for that variant.
  Old mepml/org sources are removed; no TeX/HTML or figure leftovers remain, and the
  algorithm list links to the conversion.


- 2026-10-29: `gbm_iris_analysis` converted with exact binary and three-class updates,
  full-sample projection/descent evidence, stochastic samples and an overshooting rate,
  followed by a declared 90/30/30 Iris fitting/validation/test split. Native additions
  expose log priors, raw/prefix scores and probabilities, original retained/sample rows,
  per-class residual/tree updates, aggregate step diagnostics, staged evaluation, named
  confusion/importance tables and validation-only rate/round selection. Stable
  log-sum-exp replaces the clipped loss calculation; finite/control/grid checks guard
  the documented contracts. The direct first-order leaf-mean variant is distinguished
  from Newton-leaf classification boosters, and averaged per-tree importance can sum
  below one. Six C++ programs and retained TikZ source produce seven reviewed local
  SVGs; Markdown/Doxygen exports pass. All 29 native GBM tests, 23 compiled header
  examples, six clangd hovers and the original C++ example syntax compilation pass.
  Independent NumPy/SciPy exhaustive-CART checks cover 28 cases, 998391 values and
  2044 class-tree fits, including every Iris validation prefix and the final refit;
  maximum absolute difference is 1.34e-15. Validation selects eta=.3 and 106 rounds;
  the final 120-flower refit classifies 29/30 reserved flowers correctly, with mean
  log loss .21296239. Test metrics do not select settings or imply calibration.
  Old mepml/org report sources are removed; the algorithm list links to the conversion.


- 2026-10-29: `elastic_net_iris_analysis` converted with exact orthogonal thresholding,
  unit-scaling, duplicate-predictor grouping/KKT and explicit fold-refit examples before
  a predeclared 120/30 Iris split, complete paths and alpha sensitivity. Corrected the
  actual coordinate squared-norm denominator and training-only fold centring/scaling;
  removed the workaround that hid large CV errors. Added native objective/KKT/sweep
  diagnostics, path coefficients/status, fold assignments/results, coefficient tables,
  frozen-query scoring and a configurable path endpoint ratio. Existing constructors
  and ridge/lasso wrappers remain. Six C++ programs plus retained TikZ source produce
  seven reviewed local SVGs; Markdown/Doxygen exports pass. All 23 native tests, 20
  compiled header examples, six clangd hovers and the original C++ example compile pass.
  Independent NumPy/scikit-learn checks cover 58 cases, 51498 values and 2290 solver
  fits, including every candidate/fold for the five declared Iris models. Coefficient
  comparisons account for Hessian curvature and measured stationarity tolerances.
  Primary held-out RMSE .3504 cm versus lambda-zero .3403 cm is reported without
  claiming a regularization gain; the ridge search boundary is explicit. Old mepml,
  org and TeX sources are removed and the algorithm list links to the new report.

- 2026-10-29: `factor_analysis_analysis` converted with an exact eight-row,
  three-indicator covariance and scalar PAF recurrence before unit scaling, varimax,
  nine known-model sample fits and Iris boundary diagnostics. Added native named
  loading/variance/covariance tables, observed/common/reproduced covariance, extraction
  history and diagnostics, a seeded factor-model generator and off-diagonal residual RMS.
  Corrected covariance-mode SMC variance units and the observed-trace denominator;
  finite/schema/options checks and explicit raw uniqueness prevent clipped output from
  hiding an improper fit. Existing nonnegative uniqueness output is retained.
  Five C++ programs and a retained TikZ build produce six reviewed local SVG figures;
  Markdown/Doxygen exports pass. Fourteen native tests, sixteen compiled header
  examples and six clangd hovers pass. Independent NumPy/SciPy checks cover 44 cases
  and 102342 values, including fourteen independent two-factor angle optima.
  Iris's converged one-factor fit still has negative raw uniqueness; parameter counting
  and iteration stopping are not presented as admissibility or identification proofs.
  Old mepml/org sources are removed; the algorithm list links to the converted report.

- 2026-10-29: `fft_analysis` converted with exact four-point coefficients, DC/Nyquist
  amplitude calibration, periodic-Hann leakage, zero-padding and linear-versus-circular
  convolution before complex-signal theory checks and the original 256-sample mixture.
  No empirical dataset: known constructed frequencies expose sign, scaling and finite
  record interpretation. Added native direct-DFT, diagnostic, one-sided spectrum and
  full linear convolution APIs; retained the radix-2 FFT conventions and full real-input
  output. Six C++ programs and a retained TikZ butterfly build produce seven reviewed
  local SVGs. Ten native tests, seven compiled header examples and five clangd hovers
  pass. Independent NumPy/SciPy checks cover 44 cases and 8177 values; maximum absolute
  discrepancy is 2.620e-13. Markdown/Doxygen exports preserve all seven figures and
  runnable blocks. Old mepml/org sources are removed and both algorithm-list links updated.

- 2026-10-29: `pca_iris_analysis` converted with exact four-point axes/projection loss,
  unit-rescaling and query-metric counterexamples, and a wide rank-one example before
  Iris spectra/loadings, measured reconstruction and frozen 120/30-row transforms.
  Added native PCA means/scales/covariance/rank accessors, spectrum/loadings tables,
  reconstruction and error curves. Full p-axis output is preserved and documented,
  including null directions; uncentred options describe second moments. Finite/schema
  validation and covariance normalization protect the demonstrated numerical contracts.
  Six C++ programs and one retained TikZ build produce seven reviewed local SVGs.
  Fourteen native tests, seventeen compiled header examples and six clangd hovers pass.
  Independent NumPy SVD/eigh and scikit-learn checks pass on 23 cases (4736 values,
  including 55 separated-prefix projectors and eleven scikit-learn comparisons).
  Old mepml/org/TeX sources are removed; the algorithm list links to the new report.

- 2026-10-29: `clustering_iris_analysis` converted with four-point centroid/Ward costs,
  core/border/noise chains and exact purity/adjusted-Rand examples before Iris. All four
  raw measurements enter the fits; species is used only for post-fit descriptions.
  K-means count/restart sweeps, all four linkages, tree re-cutting and DBSCAN radius
  sensitivity retain explicit coverage and noise conventions. Added native
  `stats/clustering_diagnostics.hpp` composition/agreement and projection/dendrogram APIs.
  Seven complete C++ programs and a retained TikZ build produce eight local SVG figures;
  Inkscape review and Markdown/Doxygen exports pass. Nineteen native tests pass, four
  header examples compile/run and five clangd hovers pass. Independent NumPy/SciPy/
  scikit-learn checks cover objectives, scores, all hierarchical cuts/heights and all
  seven DBSCAN partitions on a shared distance matrix. At epsilon .3, independently
  calculated boundary distances change the partition; the report explains this.
  Old mepml/org/TeX sources are removed and eight algorithm-list links repointed.

- 2026-10-28 backlog: these finished reports use `RLayout` without `fit_panels()`
  and need their multi-panel figures fixed and re-checked with inkscape (rule 4):
  acorn_generator, adaboost, adler32, aho_corasick, almeida_pineda,
  alpha_max_beta_min, arima, arithmetic_coding, a_star, backward_euler, bakery,
  banker, baum_welch, bayesian_modeling, bayesian_sampling, bbp_pi, bead_sort,
  beam_search, beam_stack_search, bellman_ford, bentley_ottmann, berkeley,
  best_first_search, bicubic_interpolation, bidirectional_search, iris_lda,
  warnsdorff, xgb_iris, xiaolin_wu_line, zeller (all `_analysis` except iris_lda).

- 2026-10-28: the user moved simple worked examples (figures + maths) ahead of
  the dataset example and made the dataset optional. Reports finished before
  this date are dataset-first; leave them unless they are revisited. The user
  also asked for `chans_algorithm_analysis` next, out of checklist order, to
  try the new structure.

- A second worker started from the last unchecked report on 2026-10-27, converting
  one at a time in reverse checklist order while the original worker proceeds
  from the front. Reverse-side checkpoints are the individual checked entries
  at the end; the next reverse-side report is the last unchecked entry.

- `examples/mepml/*.mepml` were generated from `examples/org/*.org` by
  `just mepml-from-org`. The user decided (2026-10-27) to delete each report's
  org source when it is converted, so that recipe cannot recreate it. Repoint
  any `report examples/org/<name>.org` line in `list-of-algorithms.txt` to
  `docs/mepml/examples/<name>.mepml`.
- `index`, `plot`, `pspline`, `random_generation`, `dense_linear_algebra`,
  `chacha20`, `pcg32`, `sfc64`, `mersenne_twister64`, `xoroshiro128plus`,
  `xoshiro256starstar` do not follow the `_analysis` naming; decide per file
  (merge into a related report, or convert as-is).

### Adding a dataset

Put the CSV in `data-raw/datasets/`, add its name to `DATASETS` in
`scripts/generate_datasets.py`, run `just generate-datasets`, then add the
accessor to `include/datamunge/datasets/datasets.hpp` (documented),
`src/datamunge/datasets/catalog.cpp` (function, `load()`, `kEntries`),
`lib_sources.txt`, the catalog-size tests (`tests/cpp/datasets_loaders_tests.cpp`,
`tests/python/test_datasets.py`, `tests/testthat/test-datasets.R`) and the
README in `data-raw/datasets/`.

## Progress

Datasets added for reports:
- `sudoku` (download, Norvig's easy50/top95/hardest; used by ac3_analysis).

Library modules added for reports:
- `algorithms::sudoku` (domains/peers/arcs, propagate, exact solve, pencil-mark board);
  `Ac3Result` gained revisions/removals/checks/candidates counters (ac3_analysis).
- `random/quality.hpp`: `quality_tests()` battery (DataFrame), `bit_frequencies()`,
  `bit_periods()`, `bit_plot()` raster, `uniform_sample()`; `stats::ks_test_one_sample_uniform()`;
  `Acorn64::discard()` jump-ahead + `state()` (acorn_generator_analysis). Reuse these in the
  other generator reports (blum_blum_shub, lagged_fibonacci, mersenne_twister64, pcg32, ...);
  a second-level test (battery over ~40 seeds, KS on the p-values) is the convincing check.
- Bug fixed: `RandomGenerator::uniform_u64()` used `x % range` (low bits; biased an ACORN
  bootstrap SE by 7%); now Lemire multiply-shift on the high bits. Shuffles/samples changed.
- `stats::AdaBoostClassifier` (formula + DataFrame; history(), staged_error(), margins(),
  decision_function(), feature_importance(), exact plot_decision_regions()) and
  `stats::train_test_split(frame, test_fraction, seed, stratify)` (adaboost_analysis).
  `algorithms::adaboost_train`'s stump search is now an O(n log n) sweep per feature.
  Use train_test_split + breast_cancer/wine/titanic for the other classifier reports.
- `algorithms/addition_chain.hpp`: binary_chain, window_chain(e, k), is_addition_chain,
  chain_steps, mod_pow_chain, chain_lower_bound (addition_chain_analysis; no dataset fits).
- `algorithms/checksums.hpp` + `checksums.cpp`: adler32_update (NMAX deferred modulo),
  adler32_combine, adler32_roll, adler32_windows, adler32_collision_probability; crc32 is
  now table-driven; the image/zlib and plot copies of Adler-32 forward to it
  (adler32_analysis; datasets digits + titanic names). Generators of other checksum
  reports (crc32, fletcher) should extend this header.
- `algorithms/aho_corasick.hpp`: reusable AhoCorasick class (search, count, longest_at,
  scan_stats, automaton() DataFrame); aho_corasick_search wraps it (aho_corasick_analysis;
  titanic names: titles and surnames).
- Plot gotcha: abline_h/v outside the data range is drawn outside the panel (no clipping);
  set y_limits/x_limits to include it.
- `algorithms/aks.hpp` + `aks.cpp`: aks_trace(n, check_congruences) with the deciding step, aks_r,
  aks_power, aks_congruence_holds, multiplicative_order, perfect_power, euler_totient;
  aks_is_prime now wraps aks_trace; r condition made exact (aks_primality_analysis; no dataset).
- `algorithms/almeida_pineda.hpp` now has a .cpp: almeida_pineda_network, _settle (residuals,
  convergence), _gradient (forward + adjoint states); relaxation is synchronous (Jacobi) so the
  forward/adjoint rate theorem holds. New `stats/almeida_pineda_classifier.hpp`
  (AlmeidaPinedaClassifier: formula + DataFrame, momentum GD, history/outputs/confusion)
  (almeida_pineda_analysis; iris).
- `algorithms/alopex`: split into hpp+cpp; `AlopexResult` gained `initial_cost`, `temperature_history`, `parameter_history` (via `AlopexParameters::record_path`); new `alopex_correlations()` rebuilds the C_i trace (alopex_analysis).
- `algorithms/game_search`: `GameResult` gained `nodes_visited`/`cutoffs`/`best_child`; windowed `alpha_beta(root, max, alpha, beta)`; `random_game_tree`, `order_moves`+`MoveOrder`, `minimal_tree_leaves` (Knuth-Moore), `alpha_beta_branching_factor` (Pearl); `tictactoe::{winner, x_to_move, game_tree}` (alpha_beta_analysis).
- `algorithms/elementary`: `AlphaBetaCoefficients`/`AlphaBetaErrorRange`, closed-form `alpha_max_beta_min_minimax` (tan^2(pi/16)), `_no_underestimate`, `_least_squares`, exact `_error`, k-line `_polygon` + overloads; new `cv/gradient` (`sobel_gradient`, true-derivative scaled) (alpha_max_beta_min_analysis).
- `algorithms/approximate_counting`: split into hpp+cpp; `MorrisCounter` gained `add(events, rng)` (geometric skip), `base()`, `register_bits()`, base>1 check; new `morris_variance` ((a-1)n(n-1)/2) and exact `morris_exponent_pmf` (DP) (approximate_counting_analysis).
- `algorithms/association_rules`: split into hpp+cpp (detail:: helpers now private); new `AprioriLevel` + `apriori_levels` (joined/counted/frequent per k), `TransactionData` + `encode_transactions(DataFrame, columns)` (column=value items), `itemset_label`, `rule_improvements` (Bayardo improvement) (apriori_analysis).
- New built-in dataset `air_passengers` (R AirPassengers, 144x3: year, month, passengers); catalog now 20 rows (C++ and Python tests updated); facade `DataFrame::air_passengers()`. New `stats/time_series`: `difference`, `acf`, `pacf` (Durbin-Levinson; match R to 4 dp), `arma_acf` (like ARMAacf), `ljung_box_test`. `stats::ARIMA` gained `expanded_ar/ma_coefficients()` and `psi_weights()`; airline-model CSS fit matches R exactly (arima_analysis).
- `algorithms/arithmetic_coding` (arithmetic_coding): split into hpp+cpp; ArithmeticEncoder/ArithmeticDecoder (incremental), AdaptiveFrequencyModel, adaptive + context-model encode/decode, arithmetic_bits_trace, arithmetic_intervals, ideal_code_length, symbol_counts, shannon_entropy_bits, adaptive_code_length (exact Laplace), adaptive_redundancy_approx, context_counts. `huffman_coding`: huffman_code_lengths(counts).
- `algorithms/arnoldi_iteration` (arnoldi_iteration): split into hpp+cpp; matrix-free overload (MatVec, optional reorthogonalization), RitzPair + ritz_pairs (residual h|y_m|), arnoldi_orthogonality_loss, arnoldi_relation_residual. New `stats/knn_graph` (KnnGraph: symmetric kNN graph, random_walk_apply, random_walk_spectrum). FIXED nn/linalg.cpp: LAPACK ints were 32-bit while nix OpenBLAS is ILP64 (OPENBLAS_USE64BITINT) -> dgeev param-11 errors/crash.
- `algorithms/terrain_search` (a_star): TerrainGrid, terrain_from_frame, TerrainHeuristic {Zero, Euclidean, Octile, OctileClimb}, terrain_a_star (weight, reopen; records expansion order; float-tie tolerance), terrain_cost_from/to_go, terrain_edges. New dataset `volcano` (R; 5307x3 long format, catalog now 21). RPlot::image() + graphics::terrain_colors() (matches R).
- Baum-Welch: EM derivation + Jensen monotonicity; casino via new sample_hmm (fit, Viterbi accuracy, random restarts); faithful eruptions with new Gaussian HMM (gaussian_hmm.hpp: baum_welch_gaussian, posteriors, viterbi, independent-mixture option), 2- vs 3-state BIC.
- Banker: greedy-safety exactness proof, safe-order counting DP, closed-form unsafe rectangle for 2 processes, simulation banker vs grant-if-available (deadlock rate, throughput); new banker.cpp (count_safe_sequences, deadlocked_processes, simulate_resource_allocation).
- Bakery: lemma-based proof of mutual exclusion/FCFS, BFS model checker over atomic interleavings (holds with choosing flag, 10-step violation without), random-schedule simulator, real BakeryLock vs std::mutex; new bakery.cpp.
- Baillie-PSW: derived strong Fermat (sqrt of 1) and strong Lucas (Frobenius) conditions; census of liars < 1e7 (162 spsp2, 178 slpsp, 0 common); new jacobi_symbol, selfridge_parameters, lucas_sequence_mod, lucas/strong_lucas_probable_prime; fixed is_square overflow near 2^64 and add/sub overflow for n > 2^63.
- `bayes/diagnostics` (new; bayesian_modeling): draws_of, chains_of, effective_sample_size (Geyer, multi-chain), split_rhat, energy_bfmi, PosteriorSummary + summarize_draws, posterior_histogram. New `bayes/normal_models`: normal_mean_posterior, hierarchical_normal_log_marginal, hierarchical_normal_posterior (BDA 5.4 tau quadrature, verified vs scipy). `NUTSResult` gained per-draw divergent/tree_depth/n_leapfrog/energy; `NUTS::sample_chains`; `leapfrog_gaussian_acceptance` (exact, verified vs MC); normal_lpdf with Var sigma; bayes module fully documented. Report rebuilt on eight_schools (pooled conjugate check, step-size stability, centred vs non-centred funnel); old iris logistic part dropped.
- `bayes/grid_posterior` (new; bayesian_sampling): midpoint tensor-grid quadrature (evidence, moments, marginals, edge_mass). `RWMOptions::adapt_covariance` (adaptive Metropolis, Haario 2001; RWMResult::proposal_covariance). `bayes::LogDensity` lambda adapter; `laplace_lpdf`; `draws_mean`/`draws_covariance`; `gaussian_importance_efficiency` ((sqrt(2k^2-1)/k^2)^d). gibbs/rwm/importance_sampling fully documented. SWIG: NUTS::sample_chains and laplace_lpdf %ignored in all .i files. Report: mtcars mpg~wt, Normal vs Laplace errors, grid ground truth, centred vs uncentred mixing, IS scale sweep + Bayes factor.
- `algorithms/pi_and_cordic` (bbp_pi): new bbp_pi_fraction ({16^n pi}), bbp_pi_hex_digits(n, count<=12), pi_hex_digits(count) (exact, Machin on 32-bit limbs; verified vs MPFR to 100k digits). Report: integral proof of BBP, precision study (error ~ sqrt(n) 2^-53; 8 digits/eval safe to 2e4, 10 at 1e6, 8 at 1e7 vs published digits), O(n log n) timing, chi-square normality of first 50k hex digits.
- bead_sort: Ferrers/conjugate-partition proof on faithful waits; algorithms::bead_sort now O(n+max) via conjugate_partition, plus bead_columns and bead_sort_gravity (train-model fall simulation) in new src/datamunge/algorithms/bead_sort.cpp; sorting_hybrid.hpp fully documented. Gotcha: dm::datasets::X() returns by value, so never bind 'const auto& v = dm::datasets::X().get_column(..).values()' (dangling).
- beam_search: volcano terrain (same problem as a_star); new algorithms::terrain_beam_search (layered beam, best-g duplicate detection, incumbent bound U => anytime; unlimited width = queue Bellman-Ford, proved optimal) + BeamRanking (g+h / h only) + TerrainBeamResult; trace_path helper shared with terrain_a_star. Finding: B=1 g+h dead-ends because pruned cells keep their g; h-only never fails but first path stays 15% off; unlimited beams re-expand (5439 > 5307 cells).
- beam_stack_search: volcano terrain; new algorithms::terrain_beam_stack_search (Zhou-Hansen f-range stack, (f,cell) total-order keys, stored-layer duplicate check, upper_bound seed, remember_finished table, max_expansions budget, exhausted flag). Finding: without the finished-cell table B<=16 re-expands exponentially (1 cell 83,392x, >1e7 expansions); with it every width 1..128 proves the optimum in <=205k expansions.
- bellman_ford: e-bike energy map on volcano (climb lambda=100, descent credit kappa); new terrain_energy_edges, BellmanFordResult passes/relaxations_per_pass/edge_checks/negative_cycle, bellman_ford_queue (SPFA, depth>=n detection, fallback extraction), shortest_path_to. Checked against Johnson reweighting with potential kappa*z (Dijkstra) to 4e-12; kappa>lambda gives 2-cell perpetual-motion cycles.
- bentley_ottmann: breast_cancer parallel coordinates (crossings = Kendall discordant pairs); geometry module's first compiled source src/datamunge/geometry/bentley_ottmann.cpp (std::set status, SweepResult with pairs/events/tests, segment_intersections_brute, parallel_coordinate_segments) + stats::kendall_tau (Knight O(n log n), tau_a/tau_b, tie counts). Fixed two robustness bugs found by data: steep segments need pinned crossing height; exactly concurrent triples (real data!) handled as blocks. Finding: at n=569 brute force always beats the sweep; sweep wins 107x at n=32000 (random 1/sqrt(n) segments, k~n/pi).
- berkeley: airquality NY 1973 temps drive tuning-fork crystal drift (crystal_drift_ppm parabolic law) of a 5-node sensor network; new berkeley_sync_window (Gusella-Zatti largest-cluster fault-tolerant average), BerkeleyResult.included, simulate_berkeley_network (+BerkeleyNetworkOptions/Trace) in new src/datamunge/algorithms/clock_synchronization.cpp. Findings: window must satisfy w >= 4eps + drho R (estimates, not clocks); 15-min rounds let a 150 ppm broken clock pass a 0.25 s window and drag the group +375 s.
- Fixed a real bug: berlekamp_factor row-reduced Q-I instead of (Q-I)^T and returned composite 'irreducible' factors (28 test failures). Added factor_gf_p (square-free + Berlekamp), berlekamp_factor_count, count_monic_irreducibles, expected_distinct_irreducible_factors. Digits-as-GF(17)-polynomials match random-polynomial factor statistics.
- berlekamp_massey: new algorithms/lfsr (lfsr_sequence, lfsr_period, linear_complexity_profile, expected_linear_complexity, geffe_sequence, lfsr_seed, lfsr_correlation_scan, bytes_to_bits/bits_to_bytes/xor_bits) and datasets::to_csv (inverse of parse_csv). Faithful CSV LFSR known-plaintext attack, Geffe correlation attack, faithful long/short profile.
- bernstein_vazirani: new algorithms/fourier_sampling (fourier_sampling_state, walsh_spectrum, parity_truth_table, measure_shots, bernstein_vazirani_classical). Noisy oracle P(s)=(1-2eta)^2; breast_cancer NN classifier Fourier spectrum. Reuse for deutsch_jozsa/simon/grover reports.
- best_bin_first: new geometry::BestBinFirstKd (bucketed k-d tree, largest-spread median splits, BinPriority Cell (Arya-Mount incremental) vs Hyperplane (Beis-Lowe), knn with budget, exact flag) + knn_brute_force, in new src/datamunge/geometry/best_bin_first.cpp. Digits: PCA rotation halves exact-search work at d=64 (44% vs 86%).
- best_first_search: algorithms::terrain_best_first (f = alpha g + h, any heuristic table), new algorithms/high_water_mark (Wilt-Ruml bounds on GBFS, minimax Dijkstra + floods), new geometry/proximity_graph_search (NSW/HNSW best-first loop with ef). GBFS with h* expands only its path but is 22% worse than optimal on the volcano.
- matrix-free algorithms::biconjugate_gradient (optional shadow, breakdown detection), page_rank_system, geometry::knn_lists; PageRank on digits 10-NN graph; shadow b is a left eigenvector -> BiCG breaks down at step 1, BiCGSTAB x0=0 NaN at alpha>=0.99
- cubic_convolution_kernel(t,a), bicubic_interpolate(...,a), resample_grid(GridInterpolation); only a=-1/2 reproduces linears (old header wrongly claimed all a do); a=-3/4 is first order; volcano coarsening study
- new algorithms/bidirectional_search.hpp (bidirectional_bfs layer-complete, bidirectional_dijkstra mu rule, bfs/dijkstra_pair_search, ball_sizes); old search_batch bidirectional_search returned invalid paths 9% -> now delegates; digits kNN + volcano Naismith
- dense_linear_algebra (kept name; old P-spline half dropped, it belongs to the pspline report): mtcars least squares. New linalg/svd.hpp (one-sided Jacobi svd, SVDDecomposition rank/solve/pseudo_inverse/reconstruct, condition_number) and linalg/least_squares.hpp (least_squares + LeastSquaresMethod, gram, vandermonde, matrix_cast, equilibrate/standardize_columns, correlation_matrix, relative_error, backward_error, least_squares_accuracy vs long double), QRDecomposition::r_inverse/rank_threshold, graphics::diverging_colors; dense headers fully documented. Bugs fixed: linear_regression covariance via R^-1 (was Cholesky of X^TX, 88% SE error at kappa 1e9); jacobi_eigen absolute stop test (wrong eigenvalues for small-scale matrices); QR rank tolerance absolute -> relative; formula '. - x' kept x; SVG stylesheet text{fill} overrode every RPlot::text() colour. Finding: NE error = u kappa_eq^2, QR = u kappa_eq (equilibrated kappa, van der Sluis); KernelRidgeRegression.RecoversNonlinearFunction already failing (0.060 > 0.05), not caused by this.
- chans_algorithm (first report in the new simple-examples-first layout; dataset faithful, as an optional final section): `geometry/hull_algorithms` moved to a .cpp and added to geometry.hpp (it was not in dm.hpp). FIXED: chans_algorithm scanned every group-hull vertex and globally sorted the input, so it was not O(n log h); now hash dedupe + binary-search tangents (convex_polygon_tangent, with probes). New chans_algorithm_trace + ChanOptions (initial_m, square_guess, binary_tangent, record_geometry) + ChanRound/ChanResult, counting graham_scan/jarvis_march overloads, convex_layers (onion peeling), random_point_cloud(PointCloud::Square/Disk/Gaussian/Parabola, exact 2^-20 grid). Finding: tests < 9 n log2 h always, but the constant is 4-7x Graham's; Chan's wins on counts only for h ~ 12 and loses in wall-clock everywhere.
- frank_wolfe (simple-examples-first layout; dataset faithful waiting times, NPMLE): `algorithms/frank_wolfe` extended (header template + new .cpp). New FrankWolfeOptions (max_iter, tolerance on the FW gap, FrankWolfeVariant Vanilla/AwayStep/Pairwise, FrankWolfeStep Standard/ShortStep/LineSearch, lipschitz, record_path), FrankWolfeResult gained gap/step_size/moves/active_atoms/path/atoms/weights/iterations/converged (old frank_wolfe(f,grad,lmo,x0,iters) unchanged), an initial-active-set overload, l1_ball_lmo, vertex_lmo, MixingWeightsFit + gaussian_location_npmle (Kiefer-Wolfowitz NPMLE, gap = max D_j - 1) + gaussian_location_em. Line search is regula falsi on the slope: golden section on f values only resolves the step to sqrt(eps), so drop steps never fired. Findings: exact line search made vanilla FW slower than 2/(k+2) in both experiments; NPMLE needs every grid point as its own starting atom (a single-vertex start makes the likelihood near-singular).
- `stats::LM` (lm_penguins_analysis, 2026-10-28): fully documented; added `rss()`, `log_likelihood()`/`aic()`/`bic()` (R's logLik.lm), `coefficient_frame()`, `anova_frame()`, `fitted_rows()`, `plot_diagnostics()` (2x2, fit_panels) and free `stats::anova(reduced, full)` -> `NestedAnova` (with std::formatter). Diagnostic plots now match R's plot.lm: zero line, lowess smoothers, clipped qqline, Cook's 0.5/1 contours kept inside the limits, top-3 points labelled by data row. Fixed R² / adjusted R² / F for no-intercept models (TSS about 0, as R). New `stats::lowess` (port of R's clowess, matches R to 1e-15). All checked against R 4.5.2. Dataset: penguins (Simpson's paradox in bill depth; sex x species found from residuals).
- `linalg::symmetric_eigen()` (Householder + implicit QL, O(n^3), ~1 s at n=1000; use it instead of jacobi_eigen for large symmetric matrices). `stats::LLE` rewritten (lle_iris_analysis): kNN bug fixed, `regularization` now means sklearn's `reg` (ridge r*tr(C), was r*tr(C)/k), component indicators shifted above the spectrum so near-zero eigenvalues cannot mix with them, new neighbors()/weights()/reconstruction_errors()/component_labels()/cost(), free `lle_reconstruction_weights()`; matches sklearn to 2e-9. Also `datasets::swiss_roll(n, noise, seed)` (with `arc_length`), `stats::trustworthiness()/continuity()` (matrix or DataFrame columns), `DenseMatrix::from_columns()`, `graphics::viridis_colors()/value_colors()`, `DataFrame::MarkdownOptions::scientific`. Datasets: swiss_roll (generated), iris, digits. Reuse swiss_roll + trustworthiness in the isomap/laplacian_eigenmaps/tsne/umap/diffusion_maps reports.
- `stats::hypothesis_test_table()`, `rank_sum_table()`, `fisher_exact_support()`, `binomial_rejection_table()`, `binomial_operating_characteristics()` and `t_test_reference_plot()` expose classical-test intermediate states and exact count calibration. Tests retain small tails directly (erfc, Student symmetry, complementary beta/gamma); vector inputs are finite, without automatic omission. The one-proportion Wilson CI is uncorrected, binomial two-sided probability ordering is not generally dual to its central Clopper–Pearson CI, and ordinary KS requires an independently specified continuous CDF. Reuse these helpers rather than embedding test/enumeration logic in reports.
- `algorithms/hamming_distance` provides `hamming_bit_distance()`, per-byte `hamming_comparison()`/`plot_hamming_comparison()`, `hamming_distance_matrix()`, finite-code `hamming_code_summary()`/`hamming_nearest()` and q-ary `hamming_shells()`. Strings compare bytes (including NUL), packed words compare bits, and nearest-codeword ties remain explicit. Reuse the helpers rather than embedding mismatch/decoder/shell logic in reports. The introductory TikZ source lives under `diagrams/`; the report renders it with LaTeX/dvisvgm into local `build/` files.
- `stats::KernelPCA` exposes raw/centred kernels, normalized dual coefficients, the full signed spectrum, numerical rank and feature-space energy diagnostics. `transform()` and `projection_details()` use frozen training centring; query batches cannot change a row's projection. RBF centring uses `expm1` for tiny gamma, and the symmetric eigensolver is scaled. Polynomial kernels require positive integer degree/nonnegative offset. Compare score Grams or whole repeated eigenspaces, and interpret energy only within a fixed kernel/metric; no inverse transform or gamma selection is fitted.
- `stats::KernelRegression` exposes ordinary and deletion fits, original retained rows, frozen scaling, query weights/concentration/proximity, the smoother matrix, bandwidth coverage/objectives and held-out scores with a training-mean baseline. Gaussian relative log weights avoid all-weight underflow; compact-kernel automatic tuning requires complete deletion coverage. Explicit `bandwidth_candidates` preserve caller order/first ties. Deletions freeze full-training preprocessing; selected CV is a tuning criterion. Reuse these diagnostics rather than report-side fitting/scoring logic.
- `stats::isomap_graph()` inspects directed neighbours, union edges and components before fitting; `stats::Isomap` exposes graph/geodesic/Gram matrices, signed spectrum, prefix distance/strain diagnostics, pair distances, independent-reference errors and maximum edge stretch, plus graph/Shepard plots. Disconnected graphs are rejected, never bridged. Reuse the shared Swiss-roll `arc_length,height` reference for shortcut checks; compare graph definitions before cross-library coordinates (iris ties can differ).
- `stats::LMM` (lmm_analysis): fixed a transposed-Lambda bug in the Z*Lambda product (also in `stats::GLMM`) that made random-slope SDs, correlations and BLUPs wrong whenever q>1 (likelihood was unaffected); DE is now followed by a Nelder-Mead polish and Lambda column signs are canonicalised; new `theta()`, `deviance_at(theta)` (profiled ML/REML criterion at any theta), `coefficient_frame()`, `random_effects_frame()`. Matches nlme to ~1e-7 on Orthodont (REML and ML, random intercept + slope). New built-in dataset `orthodont` (nlme::Orthodont, 108 x 4). Reuse deviance_at for profile plots and LRTs of variance components (random-slope products and BLUPs rechecked against dense working solves during the glmm_analysis conversion).
- `stats::KNNClassifier` / `KNNRegressor` (knn_iris_analysis): full docs; leave-one-out now recomputes the standardisation without the left-out row (equals refitting n times; matches sklearn Pipeline LOO exactly on iris); distances are raw differences times 1/s_j so raw ties stay exact (rounding of standardised values used to break ties at random); new `loo_accuracy_path(K)` / `loo_rmse_path(K)` (all k from one sort per row), `kneighbors()` (KNNNeighbors, with formatter), `options()`, `training_labels()`/`training_response()`, free `knn_asymptotic_risk(eta, k)` (Cover-Hart limit); `plot_decision_regions` redrawn as run-length rectangles + training points, optional limits box. Shared code in stats/detail/knn_common.hpp (DistanceMetric moved there, still reachable via knn_classifier.hpp).
- stats::NaiveBayesClassifier (naive_bayes_classifier.hpp): user priors, gaussian_frame/categorical_frame, posterior_terms, exact leave-one-out (loo_fitted_classes/loo_accuracy), plot_decision_regions; graphics::add_class_regions/tint (plot/class_regions.hpp), shared with KNNClassifier.
- optim::BayesianOptimization / RBFGaussianProcessSurrogate (optimization_bayesian_analysis): run() -> BayesianOptimizationResult history; candidates + compass refinement of the acquisition; options ctor (C++ only, #ifndef SWIG) with value normalisation, profile marginal-likelihood length scale (+ optional nugget), EI/PI/LCB; predict(), log_marginal_likelihood(), free expected_improvement/probability_of_improvement; verified vs sklearn GaussianProcessRegressor. KernelRidgeRegression 28x faster (symmetric_eigen, eigenbasis reused).
- stats::SplitConformalClassifier / SplitConformalRegressor (conformal_prediction_analysis): free conformal_quantile + conformal_coverage (Beta coverage law), normalised and quantile-band (CQR) regression scores, class-conditional and non-empty sets, p_values(), string-label ctors, per-class coverage and set-size counts; random::beta_pdf/beta_cdf/beta_quantile.
- stats::RandomForestClassifier / DecisionTreeClassifier (rf_iris_analysis): FIXED rows-with-missing-values misalignment (bootstrap/OOB indexed the complete-case rows but sampled the original frame; also in RandomForestRegressor); tree split search is now a sort-and-sweep (identical trees, 3x faster); new tree(t), in_bag_counts(t), tree_votes(x)/vote_share(x) fast paths, oob_tree_counts/oob_vote_share/oob_predictions/oob_margins/oob_accuracy_path/oob_tree_accuracies, strength_correlation() (Breiman's s, rho-bar, bound), permutation_importance() (OOB, with SE), vote-agreement-shaded plot_decision_regions(x, y, res, limits); DecisionTreeClassifier::predict_row/node_frame. datasets::twonorm/threenorm/ringnorm/circle (mlbench-style, known Bayes errors). Dataset iris; sklearn RF OOB and CART importances agree.
- Quantile regression (quantile_regression_analysis): exact vertex-descent simplex is now QuantileRegression's default (ADMM kept as an option; it hit its 10 000-iteration cap from n = 1000 up); fixed-perturbation tie-breaking plus an optimality certificate on the original data (the plain simplex stalled in 8 of 120 tied fits); dual_solution(), basis(), coefficient_table() with IID/NID/bootstrap standard errors, quantile_process(), check_loss/hall_sheather_bandwidth/bofinger_bandwidth; thin Gram-Schmidt QR replaces linalg::qr's n x n Q (3.2 GB at n = 20 000). Checked against scikit-learn HiGHS (120 fits, loss to 1e-15). Changing the default moved the alopex report's 'exact optimum' from 74.2894 to 74.2893.
Backward Euler: stability functions (new OdeMethod/stability_function), local error sign, heat-equation stiffness, Robertson kinetics with new backward_euler_adaptive (step doubling); Jacobian now central differences (exact for quadratic RHS, 70x fewer steps on Robertson).
Backtracking: tree of simple paths, K_n count floor(e(n-2)!) derived, grid SAW counts, knight Hamiltonian paths with Warnsdorff order + connectivity prune; new algorithms/backtracking.hpp.
BSGS: worst/average cost derivation (sqrt n vs sqrt(n/2)), table-size sweep, scaling to 1e12 + DH, subgroup search; new baby_step_giant_step_trace, element_order, primitive_root.
Sieve of Atkin: parity theorem derived from Jacobi r2; new sieve_of_atkin.hpp (form representations, work counters, li, Riemann R); atkin_primes moved and bounded; Chebyshev race to 1e8.

- [x] ac3_analysis
- [x] acorn_generator_analysis
- [x] adaboost_analysis
- [x] addition_chain_analysis
- [x] adler32_analysis
- [x] aho_corasick_analysis
- [x] aks_primality_analysis
- [x] almeida_pineda_analysis
- [x] alopex_analysis
- [x] alpha_beta_analysis
- [x] alpha_max_beta_min_analysis
- [x] approximate_counting_analysis
- [x] apriori_analysis
- [x] arima_analysis
- [x] arithmetic_coding_analysis
- [x] arnoldi_iteration_analysis
- [x] a_star_analysis
- [x] atkin_sieve_analysis
- [x] baby_step_giant_step_analysis
- [x] backtracking_search_analysis
- [x] backward_euler_analysis
- [x] baillie_psw_analysis
- [x] bakery_analysis
- [x] banker_analysis
- [x] baum_welch_analysis
- [x] bayesian_modeling_analysis
- [x] bayesian_sampling_analysis
- [x] bbp_pi_analysis
- [x] bead_sort_analysis
- [x] beam_search_analysis
- [x] beam_stack_search_analysis
- [x] bellman_ford_analysis
- [x] bentley_ottmann_analysis
- [x] berkeley_analysis
- [x] berlekamp_analysis
- [x] berlekamp_massey_analysis
- [x] bernstein_vazirani_analysis
- [x] best_bin_first_analysis
- [x] best_first_search_analysis
- [x] biconjugate_gradient_analysis
- [x] bicubic_interpolation_analysis
- [x] bidirectional_search_analysis
- [ ] bilinear_analysis
- [ ] binary_gcd_analysis
- [ ] binary_search_analysis
- [ ] binary_splitting_analysis
- [ ] birkhoff_interpolation_analysis
- [ ] bisection_analysis
- [ ] bitap_analysis
- [ ] bitonic_sort_analysis
- [ ] bkm_analysis
- [ ] blast_analysis
- [ ] bloom_filter_analysis
- [ ] blossom_matching_analysis
- [ ] blum_blum_shub_analysis
- [ ] booth_analysis
- [ ] bootstrap_analysis
- [ ] boruvka_analysis
- [ ] borwein_pi_analysis
- [ ] boyer_moore_analysis
- [ ] boyer_moore_horspool_analysis
- [ ] branch_and_bound_analysis
- [ ] breadth_first_search_analysis
- [ ] brent_cycle_detection_analysis
- [ ] bresenham_analysis
- [ ] bron_kerbosch_analysis
- [ ] brute_force_search_analysis
- [ ] brzozowski_analysis
- [ ] b_star_analysis
- [ ] bucket_sort_analysis
- [ ] buddy_allocator_analysis
- [ ] bully_election_analysis
- [ ] burrows_wheeler_analysis
- [ ] burstsort_analysis
- [ ] c45_analysis
- [ ] cannon_analysis
- [ ] canopy_analysis
- [ ] cantor_zassenhaus_analysis
- [ ] chacha20_analysis
- [ ] chacha20
- [ ] chakravala_analysis
- [x] chans_algorithm_analysis
- [ ] cheney_analysis
- [ ] chien_search_analysis
- [ ] chinese_remainder_analysis
- [ ] christofides_analysis
- [ ] chudnovsky_pi_analysis
- [ ] chu_liu_edmonds_analysis
- [ ] cipolla_analysis
- [ ] circle_packing_layout_analysis
- [ ] clarke_wright_analysis
- [ ] closest_pair_analysis
- [x] clustering_iris_analysis
- [ ] cnn_analysis
- [ ] cocktail_shaker_sort_analysis
- [ ] cohen_sutherland_analysis
- [ ] comb_sort_analysis
- [ ] computus_analysis
- [ ] cone_algorithm_analysis
- [x] conformal_prediction_analysis
- [ ] conjugate_gradient_analysis
- [ ] connected_components_analysis
- [ ] convex_hull_analysis
- [ ] cordic_analysis
- [ ] counting_sort_analysis
- [ ] crank_nicolson_analysis
- [ ] crc32_analysis
- [ ] cristian_analysis
- [ ] cubic_spline_analysis
- [ ] cuthill_mckee_analysis
- [ ] cycle_sort_analysis
- [ ] cyk_analysis
- [ ] dag_longest_path_analysis
- [ ] damerau_levenshtein_analysis
- [ ] davis_putnam_analysis
- [ ] dda_line_analysis
- [x] de_boor_analysis — no dataset (evaluation of prescribed control nets); hand-computed Bezier/nonuniform triangles, repeated-knot corner/jump, closed-form degree sweep and exact tiny-domain endpoints, local perturbation/support, original cubic; DeBoorSpline validates once and exposes evaluation, traces/tables, uniform samples and piecewise plots; fixed epsilon-shifted endpoint and malformed-input handling; 6 blocks, 6 SVGs, 18 geometry gtests, R/SciPy/rational checks.
- [ ] de_casteljau_analysis
- [ ] delaunay_analysis
- [x] dense_linear_algebra
- [ ] depth_first_search_analysis
- [ ] deutsch_jozsa_analysis
- [ ] dice_coefficient_analysis
- [ ] difference_map_analysis
- [ ] diffie_hellman_analysis
- [ ] diffusion_maps_iris_analysis
- [ ] dijkstra_analysis
- [ ] dinic_analysis
- [ ] distance_transform_analysis
- [ ] dixon_analysis
- [ ] doomsday_analysis
- [ ] double_dabble_analysis
- [ ] dpll_analysis
- [ ] d_star_analysis
- [ ] dt_iris_analysis
- [ ] dynamic_time_warping_analysis
- [ ] earley_analysis
- [ ] eclat_analysis
- [ ] edf_analysis
- [ ] edmonds_karp_analysis
- [x] elastic_net_iris_analysis
- [ ] elias_gamma_analysis
- [ ] ellipsoid_analysis
- [ ] euclidean_mst_analysis
- [ ] euler_method_analysis
- [ ] exact_cover_analysis
- [ ] exp_golomb_analysis
- [ ] explainable_ai_analysis
- [ ] exponential_backoff_analysis
- [ ] extended_euclidean_analysis
- [ ] eytzinger_search_analysis
- [x] factor_analysis_analysis
- [ ] false_position_analysis
- [ ] fem_analysis
- [ ] fermat_factor_analysis
- [ ] fermat_primality_analysis
- [x] fft_analysis
- [ ] fibonacci_coding_analysis
- [ ] fibonacci_search_analysis
- [ ] filtered_back_projection_analysis
- [ ] finite_difference_analysis
- [ ] fisher_yates_analysis
- [ ] flashsort_analysis
- [ ] fletcher_analysis
- [ ] flood_fill_analysis
- [ ] floyd_cycle_detection_analysis
- [ ] floyd_steinberg_analysis
- [ ] floyd_warshall_analysis
- [ ] fnv_analysis
- [ ] force_directed_layout_analysis
- [ ] forward_backward_analysis
- [ ] fp_growth_analysis
- [x] frank_wolfe_analysis
- [ ] freivalds_analysis
- [ ] fuzzy_c_means_analysis
- [ ] gabow_scc_analysis
- [ ] gale_shapley_analysis
- [ ] gaussian_elimination_analysis
- [x] gaussian_process_regression_iris_analysis — iris; exact two-point conditioning and repeated-query covariance, physical noise versus jitter, finite prior limits, likelihood profiling, conditional deletion versus refitted pipelines, full-data fit and stratified 105/45 holdout with interval coverage; GP joint covariance, new-observation predictions, tabular diagnostics/search and scoring; consistent relative jitter and variance-floor likelihood; 7 blocks, 6 SVGs, 25 GP gtests plus format checks, R/scikit-learn/NumPy checks.
- [ ] gauss_jordan_analysis
- [ ] gauss_legendre_pi_analysis
- [x] gauss_newton_analysis
- [ ] gauss_seidel_analysis
- [x] gbm_iris_analysis
- [ ] general_problem_solver_analysis
- [ ] generational_analysis
- [ ] geohash_analysis
- [ ] geometric_hashing_analysis
- [ ] girvan_newman_analysis
- [ ] gjk_analysis
- [x] glm_iris_analysis — binary-group and Poisson hand calculations, replication and separation, iris train/test probability scores and an explicit rounded-count mismatch; core GLM IRLS traces, working/training/coefficient/status tables and scoring; guarded finite-domain fitting, final-weight covariance, weighted residual/AIC and interval corrections.
- [x] glmm_analysis — four-count and binary-group PQL hand calculations, a frozen working-variance profile, generated Poisson recovery and penguin within-/whole-island holdouts; core working traces/profiles, known-SD option, diagnostics, scoring and simulation; Woodbury group solves, complete-state convergence and explicit working-criterion/zero-effect prediction semantics.
- [ ] gnome_sort_analysis
- [ ] goertzel_analysis
- [ ] golden_section_search_analysis
- [ ] goldschmidt_analysis
- [ ] golomb_coding_analysis
- [ ] graham_scan_analysis
- [ ] gram_schmidt_analysis
- [ ] graph_coloring_analysis
- [ ] grasp_analysis
- [ ] grover_analysis
- [ ] halley_analysis
- [ ] hamming_code_analysis
- [x] hamming_distance_analysis — local TikZ positional/cube concept diagram, byte/bit and edit-distance examples, repetition/parity-code correction limits and exact q-ary shells; reusable comparison plots/tables, packed-bit distance, distance matrices, nearest ties and code/shell diagnostics; no dataset.
- [ ] heap_permutations_analysis
- [ ] heapsort_analysis
- [ ] hermite_analysis
- [ ] hirschberg_analysis
- [ ] histogram_equalization_analysis
- [ ] hits_analysis
- [ ] hmac_analysis
- [ ] hopcroft_analysis
- [ ] hopcroft_karp_analysis
- [ ] hopfield_network_analysis
- [ ] huffman_analysis
- [ ] hungarian_analysis
- [x] hypothesis_tests_analysis — worked t/rank/ANOVA/binomial/Fisher calculations, exact size/power enumeration and iris effects, Holm family, grouped correlations and explicit KS limitations; reusable result/rank/support/calibration tables and Student reference plot, stable tails, input guards and robust F-interval bracketing.
- [x] ica_analysis
- [ ] icp_analysis
- [x] id3_analysis — exact four-row split and tennis tree; XOR, identifier bias, depth/fallback controls and a fixed 711/178 Titanic split. Five C++ blocks plus retained TikZ source, six reviewed local SVGs; 17 native tests, 19 compiled header examples, six hovers and exports pass. Independent Python checks: 1152 fits, 7532 nodes, 368751 values.
- [ ] iddfs_analysis
- [ ] index_calculus_analysis
- [ ] index
- [x] inla_analysis — exact normal-normal and skew Bernoulli controls, uncertain-precision mixture, resolution/span checks and centered penguin mixed model with SD-prior sensitivity; corrected independent p-adjust appendix. Seven C++ blocks plus TikZ, eight reviewed local SVGs; 27 tests, 33 compiled examples, eight hovers, exports and 61-fit independent oracle pass.
- [ ] insertion_sort_analysis
- [ ] interpolation_search_analysis
- [ ] inverse_iteration_analysis
- [x] isomap_iris_analysis — four-point line, square-cycle negative spectrum and disconnected triangles; known-intrinsic Swiss-roll neighbourhood/density checks, iris connectivity and MDS comparison; core graph/geodesic/spectral/distance diagnostics, deterministic ties, contiguous shortest paths and scaled symmetric eigensolve.
- [ ] itp_analysis
- [ ] jackknife_analysis
- [ ] jaro_winkler_analysis
- [ ] jarvis_march_analysis
- [ ] johnson_apsp_analysis
- [ ] jump_and_walk_analysis
- [ ] jump_point_search_analysis
- [ ] jump_search_analysis
- [ ] kabsch_analysis
- [ ] kadane_analysis
- [ ] kahan_summation_analysis
- [ ] kalman_filter_analysis
- [ ] karatsuba_analysis
- [ ] karger_min_cut_analysis
- [ ] karn_analysis
- [ ] kdtree_nn_analysis
- [x] kernel_pca_iris_analysis — exact linear/Gaussian three-point and quadratic four-point maps, RBF limits and eight symmetric ring points before iris metric/gamma checks and held-out projections; inspectable raw/centred kernels, signed spectrum, numerical rank, feature-energy diagnostics and frozen-training transform; stable small-gamma centring and PSD parameter guards.
- [x] kernel_regression_iris_analysis — exact three-point weights, boundary bias, duplicate/support counterexample and known-function bandwidth check before full/held-out iris; reusable smoothing/weight/coverage/score diagnostics, stable Gaussian log weights and complete-coverage CV.
- [x] kernel_ridge_analysis — exact free-intercept two-point RBF and linear fits, signed weights and a four-row preprocessing counterexample; known-sine penalty path and fold-refitted Iris regression with an untouched 30-row holdout. Seven C++ blocks plus retained TikZ source, eight reviewed local SVGs; corrected kernel centring/intercept LOO, inspectable diagnostics and fold predictions. All 23 native tests, 22 header examples, seven hovers, exports and independent 480-candidate/48-fold checks pass.
- [ ] k_medoids_analysis
- [ ] kmp_analysis
- [x] knn_iris_analysis
- [ ] knuth_bendix_analysis
- [ ] kosaraju_analysis
- [ ] kruskal_analysis
- [ ] k_way_merge_analysis
- [ ] lagged_fibonacci_analysis
- [ ] lagrange_interpolation_analysis
- [ ] lamport_clock_analysis
- [ ] lanczos_iteration_analysis
- [ ] lanczos_resampling_analysis
- [ ] laplacian_smoothing_analysis
- [ ] lax_wendroff_analysis
- [ ] lcs_analysis
- [ ] lda_iris_analysis
- [ ] leaky_bucket_analysis
- [ ] lenstra_ecm_analysis
- [ ] levenshtein_analysis
- [ ] levenshtein_coding_analysis
- [ ] levinson_analysis
- [ ] lex_bfs_analysis
- [ ] library_sort_analysis
- [ ] linde_buzo_gray_analysis
- [ ] linear_congruential_analysis
- [ ] linear_interpolation_analysis
- [ ] lis_analysis
- [ ] ll1_analysis
- [x] lle_iris_analysis
- [ ] lll_analysis
- [x] lmm_analysis
- [x] lm_penguins_analysis
- [ ] logitboost_analysis
- [ ] long_division_analysis
- [ ] longest_common_substring_analysis
- [ ] lsh_analysis
- [ ] lucas_primality_analysis
- [ ] luhn_analysis
- [ ] lz77_analysis
- [ ] lzw_analysis
- [ ] manacher_analysis
- [ ] marching_squares_analysis
- [ ] mark_compact_analysis
- [ ] mark_sweep_analysis
- [ ] match_rating_analysis
- [ ] matrix_chain_analysis
- [ ] max_clique_dyn_analysis
- [ ] maximum_parsimony_analysis
- [ ] md5_analysis
- [x] mds_iris_analysis — iris; rectangle centring/truncation, indefinite four-cycle, analytic spectrum sweep, Euclidean versus Manhattan residuals, exact unscaled PCA equivalence; MDS::from_distances, Gram/distance accessors, spectrum, prefix diagnostics, pair distances and Shepard plots; 9 blocks, 6 SVGs, 16 gtests, R/NumPy checks.
- [ ] median_filter_analysis
- [ ] memetic_analysis
- [ ] merge_sort_analysis
- [x] mersenne_twister64 — exact odd/even twist and tiny tempering examples, ordered-refill reference streams, inverse tempering and aligned-block continuation recovery; single-stream and forty-seed quality checks with all tails retained. Six C++ blocks plus retained TikZ source, seven reviewed local SVGs; 56 native tests, eight compiled header examples, eight hovers, exports and independent integer/SciPy checks pass. No external dataset needed.
- [ ] metaphone_analysis
- [ ] midpoint_circle_analysis
- [ ] miller_rabin_analysis
- [ ] min_bounding_box_analysis
- [ ] min_conflicts_analysis
- [ ] min_enclosing_circle_analysis
- [ ] minimum_degree_analysis
- [ ] miser_analysis
- [ ] modular_exponentiation_analysis
- [ ] moead_analysis
- [ ] monotone_cubic_analysis
- [ ] montgomery_analysis
- [ ] moore_analysis
- [ ] move_to_front_analysis
- [ ] muller_analysis
- [ ] nagle_analysis
- [x] naive_bayes_penguins_analysis
- [ ] nearest_neighbor_interpolation_analysis
- [ ] nearest_neighbor_tsp_analysis
- [ ] needleman_wunsch_analysis
- [ ] nested_sampling_analysis
- [ ] nesting_analysis
- [ ] neural_network_analysis
- [ ] neville_analysis
- [ ] newell_normal_analysis
- [ ] newton_division_analysis
- [ ] newton_raphson_analysis
- [ ] newton_raphson_division_analysis
- [ ] nonblocking_switch_analysis
- [ ] non_restoring_division_analysis
- [ ] nsga2_analysis
- [ ] nth_root_analysis
- [ ] nysiis_analysis
- [ ] odd_even_sort_analysis
- [ ] odds_algorithm_analysis
- [ ] ode_solver_analysis
- [ ] one_r_analysis
- [ ] online_variance_analysis
- [ ] optics_analysis
- [x] optimization_adaptive_gradient_analysis
- [x] optimization_analysis
- [x] optimization_bayesian_analysis
- [x] optimization_constrained_analysis
- [x] optimization_coordinate_cg_analysis
- [x] optimization_derivative_free_analysis
- [x] optimization_nonlinear_least_squares_analysis
- [x] optimization_proximal_analysis
- [x] optimization_second_order_analysis
- [x] optimization_variance_reduction_analysis
- [ ] ordered_dither_analysis
- [ ] package_merge_analysis
- [ ] packrat_analysis
- [ ] page_rank_analysis
- [ ] painters_algorithm_analysis
- [ ] pancake_sort_analysis
- [ ] patience_sort_analysis
- [ ] pbkdf2_analysis
- [x] pca_iris_analysis
- [ ] pcg32
- [ ] pcnn_analysis
- [ ] pearson_analysis
- [ ] perceptron_analysis
- [ ] permutation_analysis
- [ ] petrick_analysis
- [ ] phase_estimation_analysis
- [ ] pigeonhole_sort_analysis
- [ ] plot
- [ ] pohlig_hellman_analysis
- [ ] point_in_polygon_analysis
- [ ] pollard_kangaroo_analysis
- [ ] pollard_p_minus_1_analysis
- [ ] pollard_rho_analysis
- [ ] pollard_rho_log_analysis
- [ ] polygon_triangulation_analysis
- [ ] polynomial_long_division_analysis
- [ ] postman_sort_analysis
- [ ] power_iteration_analysis
- [ ] powerset_construction_analysis
- [ ] pratt_analysis
- [ ] prim_analysis
- [ ] prufer_analysis
- [x] pspline — faithful; exact three-point shrinkage and flat CV/GCV, cubic coefficient nullspaces, analytic lambda sweep, original oscillating signal with basis-resolution limits, real-data path/residuals/leverage and retuned basis-size comparison; PSR basis matrix, coefficient differences, summary/observation tables and smoothing_path; FDA difference operator, leverage and fixed-lambda deletion scores; 7 blocks, 6 SVGs, 32 P-spline/B-spline gtests, R/SciPy checks.
- [ ] push_relabel_analysis
- [ ] qft_analysis
- [ ] q_learning_analysis
- [ ] qr_algorithm_analysis
- [ ] quadratic_sieve_analysis
- [x] quantile_regression_analysis — faithful; hand five-point envelope (exact tau ranges, non-unique face), fanning cloud vs true quantiles, wild response vs leverage, counting/duality checks, simplex vs ADMM, SE coverage Monte Carlo, ozone lines + quantile process; 8 blocks, 6 SVGs, 16 gtests, scikit-learn check.
- [ ] quickhull_analysis
- [ ] quickselect_analysis
- [ ] quicksort_analysis
- [ ] quine_mccluskey_analysis
- [ ] rabin_karp_analysis
- [ ] radix_sort_analysis
- [ ] ramer_douglas_peucker_analysis
- [ ] random_generation
- [ ] random_restart_hill_climbing_analysis
- [ ] range_coding_analysis
- [ ] ransac_analysis
- [ ] rate_monotonic_analysis
- [ ] rayleigh_quotient_iteration_analysis
- [ ] rbf_network_analysis
- [ ] rc4_analysis
- [ ] recursive_descent_analysis
- [ ] reference_counting_analysis
- [ ] restoring_division_analysis
- [ ] reverse_delete_analysis
- [x] rf_iris_analysis
- [ ] rice_coding_analysis
- [ ] ridders_analysis
- [ ] rotating_calipers_analysis
- [ ] roulette_wheel_analysis
- [ ] rounding_functions_analysis
- [ ] round_robin_analysis
- [ ] runge_kutta4_analysis
- [ ] run_length_encoding_analysis
- [ ] ruzzo_tompa_analysis
- [ ] rvm_analysis
- [ ] salsa20_analysis
- [ ] samplesort_analysis
- [ ] sarsa_analysis
- [ ] scanline_fill_analysis
- [ ] schensted_analysis
- [ ] schonhage_strassen_analysis
- [ ] schreier_sims_analysis
- [ ] scs_analysis
- [ ] secant_analysis
- [ ] segment_intersection_analysis
- [ ] selection_sort_analysis
- [ ] self_organizing_map_analysis
- [ ] sfc64
- [ ] sha1_analysis
- [ ] sha256_analysis
- [ ] shamir_secret_sharing_analysis
- [ ] shamos_hoey_analysis
- [ ] shannon_fano_analysis
- [ ] shannon_fano_elias_analysis
- [ ] shell_sort_analysis
- [ ] shoelace_analysis
- [ ] shortest_job_next_analysis
- [ ] shortest_remaining_time_analysis
- [ ] shunting_yard_analysis
- [ ] sieve_of_eratosthenes_analysis
- [ ] signed_reversals_analysis
- [ ] simon_analysis
- [ ] simplex_analysis
- [ ] siphash_analysis
- [ ] slerp_analysis
- [ ] slr_analysis
- [ ] smith_waterman_analysis
- [ ] sobel_analysis
- [ ] sor_analysis
- [ ] soundex_analysis
- [ ] spea2_analysis
- [ ] spectral_layout_analysis
- [ ] spigot_analysis
- [ ] sqrt_digits_analysis
- [ ] srt_division_analysis
- [ ] sss_star_analysis
- [ ] steinhaus_johnson_trotter_analysis
- [ ] stochastic_universal_sampling_analysis
- [ ] stone_sip_analysis
- [ ] stooge_sort_analysis
- [ ] strand_sort_analysis
- [ ] strassen_analysis
- [ ] structured_svm_analysis
- [ ] subgraph_isomorphism_analysis
- [ ] subset_sum_analysis
- [ ] summed_area_table_analysis
- [ ] sundaram_sieve_analysis
- [ ] sutherland_hodgman_analysis
- [x] svm_iris_analysis — iris; four-point margins, exact two-point cost paths, XOR polynomial/RBF maps, 105/45 held-out comparison; SVM decision_function/training_details/diagnostics/error/plot_decision_regions; 5 SVGs, 8 executed blocks, sklearn predictions and NumPy objectives checked.
- [ ] symbolic_cholesky_analysis
- [ ] tarjan_lca_analysis
- [ ] tarjan_scc_analysis
- [ ] td_learning_analysis
- [ ] tea_analysis
- [ ] ternary_search_analysis
- [ ] thomas_algorithm_analysis
- [ ] timsort_analysis
- [ ] todd_coxeter_analysis
- [ ] token_bucket_analysis
- [ ] tonelli_shanks_analysis
- [ ] toom3_analysis
- [ ] topological_sort_analysis
- [ ] tournament_selection_analysis
- [ ] transitive_closure_analysis
- [ ] trapezoidal_ode_analysis
- [ ] tree_sort_analysis
- [ ] trial_division_analysis
- [ ] tricubic_interpolation_analysis
- [ ] trigram_search_analysis
- [ ] truncated_backoff_analysis
- [ ] truncated_binary_analysis
- [ ] trust_rank_analysis
- [ ] tsne_iris_analysis
- [ ] ukkonen_analysis
- [ ] umap_iris_analysis
- [ ] unary_coding_analysis
- [ ] uniform_binary_search_analysis
- [ ] uniform_cost_search_analysis
- [ ] union_merge_analysis
- [ ] upgma_analysis
- [x] u_statistic_analysis — iris; three-point variance terms/projections/jackknife, exact degenerate four-point law, two-point population rate checks, pooled/within-species scale summaries and conditional Monte Carlo calibration; UStatistic::analyze/term_details/two_point_law/incomplete_path; 7 blocks, 6 SVGs, 25 resampling gtests, R/NumPy checks.
- [ ] vector_clock_analysis
- [ ] velvet_analysis
- [ ] verhoeff_analysis
- [ ] verlet_analysis
- [ ] vincenty_analysis
- [ ] viterbi_analysis
- [ ] voronoi_analysis
- [ ] v_statistic_analysis
- [ ] wang_landau_analysis — in progress, reverse-order agent.
- [x] warnsdorff_analysis — digits 8×8 intensity readout; shared canonical heuristic with partial paths, decision traces, all-start profiles and grid readout helpers. Four blocks and four inspected SVGs export; 12 tests, seven compiled examples, 25/25 new-module definitions and clangd hover. Independent graph/set oracle agrees on all 158 paths and 6182 decisions; NumPy verifies every summary for 1797 images. Distinguishes greedy failures, existence and closure; preserves all legacy paths.
- [x] wildcard_match_analysis — titanic Name; corrected pattern-star handling, shared greedy counters/trace, independent prefix DP, string-column summaries and filters. Four blocks, four inspected SVGs, Markdown export; 13 tests (132496 exhaustive metacharacter cases plus binary/random coverage), eight compiled examples, 29/29 new-module definitions and two clangd hovers. Python fnmatchcase verifies all 7128 record/pattern decisions; an epsilon-NFA verifies all 184 plotted prefix states. Exact adversarial counts demonstrate quadratic worst-case work.
- [x] winnow_analysis — digits with an explicitly engineered two-pixel OR; shared checked Winnow loop, DataFrame encoding/trace/evaluation and Boolean counterfactual probes. Four blocks, four inspected SVGs, Markdown export; 11 tests, nine compiled examples, 18/18 new-module definitions and clangd hover. Exact integer-exponent Python verifies all weights, pass records, split rows, five dimension trials and probe scores; all 1797 records match upstream scikit-learn data.
- [x] xgb_iris_analysis — iris; fixed-model staged metrics/confusion, validation-only lambda/round search and separate final test. Four blocks, four inspected SVGs, Markdown export; 24 tests, 12 compiled examples, 6/6 new-helper definitions and two clangd hovers. Independent Python verifies every partition, 505 validation stages, 58 refit stages, 90 test probabilities and four gain importances; R verifies data and final metrics.
- [x] xiaolin_wu_line_analysis — air_passengers; checked full/clipped endpoint policies, DataFrame polyline mapping, additive pixel summaries and grey-cell plotting. Four blocks, three inspected SVGs, Markdown export; 13 tests, seven compiled examples, 18/18 definitions and clangd hover. Exact rational arithmetic verifies 64 segment-policy cases and every accumulated pixel at both resolutions; R verifies all input values and annual summaries.
- [x] xoroshiro128plus — mtcars mean-mpg bootstrap; raw snapshots, published jumps and held-out binary recurrence diagnostics. Six blocks, four inspected SVGs, 29 tests, seven compiled header examples, 13/13 quality-module definitions and clangd hover; authors’ C reference, independent GF(2) elimination, Python stream/resampling and R summary oracles.
- [x] xoshiro256starstar — faithful waiting-time bootstrap; raw-state snapshots, published jump/long-jump operations, multi-seed quality summaries and caller-supplied bootstrap engines. Five blocks, four inspected SVGs, 11 tests, 14 compiled examples, 37/37 helper definitions documented; authors’ C reference plus independent Python/R oracles.
- [x] z_algorithm_analysis — titanic Name; shared instrumented Z loop, trace, virtual non-byte separator and record-wise search summary. Four blocks, three inspected SVGs, Markdown export; 13 focused tests (including full-byte/random cases), eight compiled header examples, 20/20 Doxygen definitions and clangd hover; Python byte-find verifies all 5346 record/query position lists.
- [x] zeller_analysis — air_passengers; checked Gregorian weekdays/month lengths, zeller_terms, monthly calendar features and cycle counts; fixed negative/extreme-year arithmetic. Six blocks, three inspected SVGs, Markdown export; eight focused tests, six compiled header examples, 13/13 Doxygen definitions; Python datetime verifies all 146097 cycle dates and calendar checks all 144 monthly rows.
- [x] zero_r_analysis — titanic Survived; DataFrame ZeroR training/counts/fixed-model evaluation and per-class confusion. Five blocks, three inspected SVGs, Markdown export; five focused tests, eight compiled header examples, 9/9 new-module Doxygen definitions; independent Python split/count oracle matches all 20 seeds.
- [x] zhu_takaoka_analysis — titanic Name; compiled ZhuTakaoka matcher, actual scan counters/trace and record-wise naive/Horspool/ZT comparison. Five blocks, two inspected SVGs, Markdown export; seven focused tests (16,000 random cases), ten compiled header examples, 27/27 Doxygen definitions; independent Python brute-force suffix/str.find oracle reproduces all 15 query/method totals.
- [x] ziggurat_analysis — eight_schools; ZigguratNormal draw/table/plot_envelope/diagnostics/simulate_measurements, supported-layer validation and finite uniform endpoint fix. Seven blocks, four inspected SVGs, Markdown export; six focused tests, eight compiled header examples, 24/24 Doxygen definitions; independent R checks of both million-draw samples and all school probabilities.
- [x] zobrist_analysis — digits (64 categorical pixels); zobrist_boards, zobrist_transition, zobrist_collision_profile; checked indices/shapes. Seven blocks, three SVGs, Markdown export, five focused tests, seven compiled header examples, 16/16 Doxygen definitions; independent Python XOR/pair-count oracle matches all nine widths.
