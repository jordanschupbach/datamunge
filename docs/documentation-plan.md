# Doxygen documentation completion plan

The goal is useful documentation at every maintained definition, including
private helpers, enums and their values, classes, namespaces and modules,
constructors, functions, aliases, fields, constants, and static variables.
Doxygen comments in the C/C++ core are canonical. Comments must explain actual
behavior so that navigation and symbol help (`K` in mep) answer how to use a
symbol without reading its implementation.

## Completion checklist

- [x] Read repository instructions and inventory the source trees and existing docs.
- [x] Start an active goal for completing this checklist.
- [x] Adopt Doxygen as the documentation format at the user's request.
- [x] Add reusable repository skills and contributor guidance for keeping docs current.
- [x] Establish the baseline and repeatable checks.
  - [x] Repair the standalone Doxygen build and include implementation internals.
  - [x] Audit Doxygen XML for undocumented symbols, including enum values.
  - [x] Record counts by source area and symbol kind; retain actionable symbol lists.
  - [x] Verify the audit detects omissions and rejects an empty or stale inventory.
- [ ] Document the C/C++ core in reviewable batches (headers and implementations).
  - [ ] Top-level API, vectors, data frames, and plotting.
  - [ ] Neural network types, storage, and tensor metadata.
  - [ ] Neural network operations, autodiff, layers, optimizers, training, and I/O.
  - [ ] Algebra and linear algebra.
  - [ ] Algorithms (track individual sub-batches below).
  - [ ] Statistics and Bayesian inference, including detail headers.
  - [ ] Optimization and general autodiff.
  - [ ] Geometry, GIS, FEM, ODEs, and functional data analysis.
  - [ ] Images, computer vision, and filters, including detail headers.
  - [ ] Data structures, random generators, and datasets.
- [ ] Document maintained binding interfaces and native language wrappers.
  - [ ] Inventory generators versus generated proxies and runtime scaffolding.
  - [ ] Carry canonical Doxygen descriptions through supported SWIG doc generation.
  - [ ] Document handwritten Python and R modules, functions, classes, and constants.
  - [ ] Document other maintained language facades using editor-readable native syntax.
  - [ ] Regenerate affected bindings and verify representative runtime help.
- [ ] Document developer code throughout the repository.
  - [ ] Scripts, packaging hooks, and build configuration entry points.
  - [ ] Reusable example helpers and example entry points.
  - [ ] Tests, fixtures, and benchmarks.
- [ ] Validate definition help and prevent regressions.
  - [x] Verify representative Doxygen XML descriptions and generated HTML text.
  - [x] Verify C++ language-server hover for a class, enum value, method, and constant.
  - [ ] Verify mep's `K` integration if its executable/configuration is available.
  - [x] Add CI coverage checks as batches reach zero undocumented symbols.
  - [ ] Run appropriate language checks and verify source changes preserve behavior.
  - [ ] Reach zero remaining undocumented maintained symbols in every audited area.
  - [ ] Review Doxygen parsing warnings and resolve or explicitly track limitations.
  - [ ] Mark the goal complete only after all coverage and editor checks are satisfied.

Repository workflows are captured in [maintain-docs](../skills/maintain-docs/SKILL.md)
and [change-library](../skills/change-library/SKILL.md), indexed in `AGENTS.md`.
Update them when generation commands, conventions, or coverage checks change.

## Documentation standard

Use `/// @brief` before C/C++ declarations and definitions, `///<` for enum values
and short fields, and `@file` with a purpose for each maintained source file.
Document namespaces once at a canonical declaration; do not duplicate their
description at every reopening. Put docs on declarations so language servers can
associate them with symbols; use explicit descriptions or `@copydoc` at separate
definitions when needed, and verify hover rather than assuming support.

Every public symbol needs `@brief`, details, `@param`/`@tparam` for each parameter,
`@return`, `@throws`, a compiled and run `@par Example` block, and `@see` links. When
the behavior comes from an external method, format, or library, it also needs a
`@par References` list that prefers original sources: the introducing paper or
book with its DOI, or the official specification or upstream docs. See the
[maintain-docs skill](../skills/maintain-docs/SKILL.md#required-content-for-every-documented-symbol)
for the template.

Describe parameters, return values, units and shapes, defaults, ownership and
lifetime, mutation, error conditions, and thread-safety when relevant. Document
overloads and template parameters individually where their contracts differ.
Keep an existing accurate explanation; improve missing contracts without
replacing it with a restatement of the identifier. Tiny accessors can have one
precise sentence. Internal state and static locals need purpose and invariants.
Ordinary local loop variables do not need API pages, but non-obvious local state
and lambdas need explanations at their definitions.

Generated files must receive documentation through their canonical headers,
interface files, or generators. Do not hand-edit generated SWIG wrappers or copy
third-party documentation into vendored code. Generated low-level SWIG runtime
internals, external dependencies, downloaded datasets, and build artifacts are
outside maintained-symbol coverage. Record concrete exclusions and regeneration
commands during the binding inventory; generated public APIs still need help.

Doxygen cannot describe every language equally, and an XML coverage count does
not prove comment quality or editor integration. Use native docstrings/help
where a language server requires them, derived from the Doxygen contract, and
track unsupported parsers and local static-variable checks explicitly.

## Commands and evidence

The C++ tools are available in `nix develop .#cpp-tools`. Documentation generation
must not require building the library or downloading a presentation theme.

- `just docs` builds the website in `build/debug/docs/site` (`scripts/website/`) from
  Doxygen's XML (`build/debug/docs/audit/xml`), the mepml reports and the guides.
- `just docs-audit` generates a complete core XML inventory and reports coverage.
- `just docs-audit --check --path include/path.hpp` checks a finished batch.
- `just docs-audit --check --completed` enforces `docs/documented-paths.txt`.
- `just docs-hover` verifies representative definition help through clangd.
- `just test-docs` runs the coverage auditor's Doxygen integration tests.
- `python3 scripts/check_documentation.py --help` lists report options.

The initial audit covers `include/` and `src/datamunge/`. Binding, script, example,
test, and benchmark inventories must be added before claiming repository-wide
completion. Doxygen does not enumerate function-local static variables; audit
them separately in each source batch. Full symbol reports belong under `build/`;
compact baseline counts belong in this directory.

## Batch progress

### Vectors and tensor types

- [x] `include/datamunge/vector.hpp` and `src/datamunge/vector.cpp`: 62 definitions.
- [x] `include/datamunge/nn/dtype.hpp`: 22 definitions, including all enum values
  and type-specialization constants.
- [x] `include/datamunge/nn/tensor.hpp` and `src/datamunge/nn/tensor.cpp`: 100 definitions.

This batch passes the presence audit for all 184 definitions. The vector engine's
function-local static state was reviewed and documented separately. Comments
cover ownership, tensor strides and offsets, gradient state, dtype validation,
in-place mutation checks, vector length checks, random sampling constraints,
factories, and internal rendering helpers. The C++ API and behavior are unchanged.

Validation: standalone XML and HTML generation with Doxygen 1.15.0, ten auditor
integration tests, syntax checks of both changed translation units with Clang
21.1.8, and definition hover for a class, enum value, static constant, and two
methods with clangd 21.1.8. Representative descriptions are present in the generated
HTML. The checked paths are listed in `docs/documented-paths.txt`; Linux CI now
enforces that list and saves full coverage and warning reports. The CI workflow
has been validated locally through its commands; its hosted run is pending.

### Typed vectors and plot composition

The vector API now owns `std::vector<T>` storage and documents one-based
selection, numeric and string coercion in `c()`, logical-mask recycling,
negative-index exclusion, arithmetic recycling, and factor codes. There is no
missing-value sentinel: selected positions beyond storage throw instead of
returning R's NA. Partial recycling cycles are accepted without a warning.

The vector header and implementation pass the presence audit for 86 definitions;
`include/datamunge/plot/compose.hpp` adds 21 documented definitions. The new
composition header is included in `docs/documented-paths.txt`. New subtitle and
palette declarations in `plot.hpp` have adjacent contracts; plotting as a whole
remains outside the completed-file list.

Validation: Doxygen XML presence checks pass for all 107 selected definitions.
Clangd hover checks now include the vector template, templated selection method,
and reusable point-layer options. All 80 focused C++ vector, plotting, and spline
tests pass, including rendered PNG color/opacity checks. The playground builds
through CMake and exports the Iris plot to SVG and PNG. The full test target is
blocked by an existing missing `<algorithm>` include in
`tests/cpp/algorithms_tarjan_lca_tests.cpp` (`std::shuffle` is undeclared).
Regenerated R bindings install and pass the installed-package and spline tests,
including palette recycling and subtitle ownership.
The regenerated Python core extension passes six focused pytest tests, including
native-list palettes and ownership after garbage collection. Color-vector
conversion is registered before parsing palette-taking methods in SWIG.

### Single-header dm API and axis-label pipes

`include/dm/dm.hpp` includes the high-level API, datasets, and the algorithm,
linear-algebra, and neural-network module headers. Its documented `dm` namespace
alias uses the existing library types and linker target. The header is installed
by the existing CMake header-directory rule and is included in the completed-file
manifest. The current dm header and plot composition header pass the presence
audit for 29 definitions, including the label destination enum and axis-label
pipe functions. Clangd returns the descriptions for the namespace alias and the
typed `DataFrame::get_column()` accessor. The rest of the dataframe header remains
outside the completed-file list.

Including the algorithm headers together exposed conflicting declarations:
inverse iteration now reuses the canonical `EigenPair`; graph layout and polygon
filling share an indexed `Point2D`, while scanline fill still accepts standard
array points; weighted path searches return `InformedSearchResult`, distinct from
the sorted-array `SearchResult`. These changes remove the conflicting public
type definitions. Callers explicitly naming the weighted-search return type
should use `InformedSearchResult`.

The playground builds through CMake using only `<dm/dm.hpp>` and qualified names.
An installed-header C++17 syntax check passes. Integration tests exercise typed
column borrowing and mutation, null masks, missing/wrong-type columns, SVG
rendering, and the formerly conflicting algorithm headers together. A broader
test run exposes an existing failure in
`RayleighQuotientIteration.ConvergesToAnEigenpair`; a separate build using the
original inverse-iteration header reproduces it. All 76 remaining focused tests
pass separately from that pre-existing numerical failure. The completed-file
documentation gate passes for all 237 currently listed definitions.

### Grouped plot options, legends, and standard printing

`PlotArgs` owns named title, subtitle, xlab, and ylab settings applied through
`Plot::with()`. Point palettes now use `PointsArgs`. `LegendArgs` owns labels
and colors, recycles colors to the label count, and validates option names and
types. Empty legend options retain automatic entries. Manual legend entries now
render in the viewer and PNG output as well as SVG, including color opacity.

The new `include/datamunge/format.hpp` provides a dataframe formatter for
`std::format` and `std::print`. The dm umbrella includes it when standard format
support is available. Only the playground and formatting test target require
C++23; the core and other examples retain C++17. The playground uses the required
print format string and an explicit `get_column<std::string>()` for Species.

Validation: 88 focused C++ vector, plot, spline, and dm API tests pass, along with
both C++23 formatting tests through CTest. The exact command
`just TARGET=playground run-cpp` prints the dataframe, opens the plot on a temporary
X11 display, and exits successfully after the window closes. Doxygen checks pass
for all 51 definitions in the composition and formatting headers; clangd hover
checks include point options and the dataframe formatter. The formatting header
is included in the completed-file manifest.

### Fluent plot setters

The plotting API also accepts `Plot().set_title().set_subtitle().set_xlab().set_ylab()`,
`points(x, y).set_col()`, and `legend().set_labels().set_col()`. Setters mutate
their receiver, return it for chaining, and own their inputs. Legend fields can
be set in either order; incomplete legends fail before modifying a plot. Colors
retain alpha and recycle as with the existing grouped options. The `.with()`
APIs and label pipes remain supported, and tests compare the fluent Iris plot's
SVG with the grouped version byte for byte.

Validation: all 92 focused C++ tests and both C++23 formatting tests pass.
`just TARGET=playground run-cpp` opens and closes its native figure successfully.
The same playground passes mep's live C++ UI checks for object inspection, an
embedded PNG figure, full-file Run, restart, and teardown with local crepl and
freshly installed headers. Doxygen checks cover all 55 definitions in the
composition and formatting headers and all 264 definitions in the completed-file
manifest. All 18 representative clangd hover checks pass, including the new
plot, point, and legend setters.
The Python and R core wrappers are regenerated. All five focused Python tests
and all ten installed-package R assertions pass, including temporary-owner
retention for the new plot-label chains.

### Playground constructors, callable vectors, and plot colors

The playground now supports constructor-based `Histogram` and `Plot(x, y)`,
typed `apply()` with an owned `function()` callable, the `runif()` alias, and
`set_col()` with RGB(A), basic color names, or hex colors. Histogram binning
reuses `RPlot::hist()` and rejects non-finite observations or unrepresentable
bin widths. Public declarations describe ownership, input validation, callable
exceptions, color replacement, and random-generator sharing. The new histogram
header joins the completed-file manifest.

Validation: all 90 focused C++ vector, plot, composition, umbrella, GGPlot, and
C++23-format tests pass. The updated playground opens and closes all five native
plot windows on a private Xvfb display and exits successfully. All 274 definitions
in the completed-file manifest pass the fresh Doxygen audit, and all 22 clangd
hover checks pass, including the callable adapter, vector application, histogram
constructor, and named-color setter. This does not claim mep UI verification.

Python and R wrappers are regenerated from their SWIG interfaces. All six focused
Python tests and all 17 installed-package R assertions pass, including temporary
plot ownership and invalid constructor, histogram, and color inputs. Shared SWIG
exception translation turns these failures into native errors instead of letting
C++ exceptions escape an interpreter. Python runtime help includes the new
constructor and color contracts. R keeps its native `stats::runif()` unmasked.

### Playground moment statistics

The enabled ToothGrowth example computes skewness and excess kurtosis over the
whole `len` column with the draft's sample-standard-deviation normalization.
`length()`, `sd()`, and scalar/vector decimal `round()` have adjacent contracts
covering empty inputs, non-finite values, rounding ties, and ownership. Powers
use `pow()` and the generic callable uses `dm::function()`; the whole-column
reduction calls that function directly instead of applying it to individual cells.

All 26 vector/API tests and both C++23 formatting tests pass against the rebuilt
core. The numerical regression gives `[-0.143, -1.043]` for ToothGrowth. The full
test build is blocked by unrelated ambiguous `nn::tensor` initializer-list calls
in `tests/cpp/nn_ops_tests.cpp`; the affected suites were compiled separately.
The CMake playground target builds and runs on a private Xvfb display, closing
all five plot windows and printing the expected moment statistics before exiting.
Fresh Doxygen checks pass all 93 definitions
in the vector header/implementation and all 333 completed definitions, and all
25 clangd hover checks pass, including the three new helper contracts. This does
not claim mep UI verification.

### Playground API depth: parameters, returns, and verified examples

Every in-house symbol used by `examples/cpp/source/playground.cpp` now carries
`@param`/`@return`/`@throws`, `@see` cross-links, and at least one
`@par Example` + `@code` block written in `dm::` spelling. Files:
`vector.hpp`, `pipe.hpp`, `info.hpp`, `datasets/datasets.hpp` (newly gated),
`plot/compose.hpp`, `plot/histogram.hpp`, the playground members of
`plot/plot.hpp` (`RGB`, `RPlot`), `dstruct/dataframe.hpp` (`NullableColumn`,
`Row`, `GroupedDataFrame`, and the verbs playground calls), and the `Splom`
facade in `datamunge.hpp`. Each example was compiled and run against the built
library and its output comments match. `just docs-audit --check --completed`
reports 335/335. `plot.hpp`, `dataframe.hpp`, and `datamunge.hpp` are still
ungated because of members outside playground.

mep's C++ hover (`~/projects/mep`, `src/cpp_lsp*.cpp`) renders this style:
Doxygen tags become Parameters/Returns/Throws/Example/See also sections. Code
examples keep their indentation, and trailing `///<` docs are shown.
Signatures keep return `&`/`*` and default arguments. Members of class templates
resolve. The overload matching the call's argument count is shown. Members on
`auto` receivers fall back to a labelled guess by name. Use the same tag
vocabulary (`@par Example` followed by `@code`) in new docs so hover stays
structured.

### Full plot.hpp and dataframe.hpp coverage with references

`include/datamunge/plot/plot.hpp` (209/209) and `include/datamunge/dstruct/dataframe.hpp`
(580/580) are now fully documented and gated in `docs/documented-paths.txt`, including
private helpers, enum values, and fields. Each user-facing member has a compiled and run
example. Methods that mirror R graphics, dplyr/tidyr, or pandas cite the upstream reference
page, and statistical methods cite the original papers: Tukey 1977; McGill, Tukey & Larsen
1978; Wilk & Gnanadesikan 1968; Hyndman & Fan 1996; Wickham 2011 and 2014. Places where the
behaviour deliberately differs from R are noted in the docs. `just docs-audit --check
--completed` reports 1164/1164. The maintain-docs skill now requires examples and references.

### Remaining coverage and editor work

The initial core inventory is 13,448 Doxygen definitions, with 3,288 documented
and 10,160 missing. After the first batch, 3,408 are documented and 10,040 remain
without descriptions. Initial counts by area and kind are preserved in
`docs/documentation-baseline.json`; the current full missing-symbol report is
`build/debug/docs/coverage.json`.

There are 151 existing Doxygen warnings outside the finished batch, including
unfinished formulas, unresolved references, and incomplete parameter docs.
Review these during their source-area batches; the parser inventory is not a
substitute for reviewing code and language-server behavior.

At-definition hover checks pass using a temporary compilation database and no
local clangd configuration. During a separate calling-code probe, methods could
return signatures without their descriptions, so hover from references still
needs investigation. The repository's existing `.clangd` contains developer-specific
include paths and should be reviewed as part of editor integration. The mep
executable cannot open an X display in this environment; its graphical `K`
interaction has not been verified.

Continue with remaining tensor/autograd internals and the top-level API, then
work through the other core areas. Expand the inventory to the maintained
bindings and developer code before claiming repository-wide completion.

## References

Use the official [Doxygen comment syntax](https://www.doxygen.nl/manual/docblocks.html)
and [configuration reference](https://www.doxygen.nl/manual/config.html).
`EXTRACT_ALL` intentionally retains undocumented symbols in XML and disables
Doxygen's own missing-documentation warnings; the coverage checker supplies that
check instead of counting only the already documented symbols.

### Zobrist hashing report conversion

`include/datamunge/algorithms/zobrist_hashing.hpp` and its new implementation
pass the presence audit for all 16 definitions. Contracts cover owned table
storage, zero-category keys, checked indices and board shapes, categorical
DataFrame encoding, exact hexadecimal update traces, distinct-input pair counts,
and the ideal-key assumption behind collision expectations. All seven header
examples compile and run. Five focused C++17 tests pass; an independent Python
XOR and bucket-count calculation reproduces the report at all nine widths.
The converted digits report runs seven blocks, exports to Markdown, and has
three visually inspected SVGs. New helpers are report-only C++ APIs; binding
regeneration and editor UI verification were not part of this conversion.

### Ziggurat sampler report conversion

The Ziggurat header and new implementation pass the presence audit for all
24 definitions, including private counters and sampler state. Eight public
header examples compile and run. Six focused C++17 tests cover table closure,
accepted/rejected proposal conservation, both supported table sizes, scalar
and batch equivalence, tails, invalid inputs and measurement simulation.
Seven report blocks execute and four SVGs have been visually inspected.
Independent R calculations reproduce moments, tail counts and KS statistics
for both million-draw samples, plus all eight analytic measurement probabilities
and Monte Carlo standard errors. R and the library use slightly different
asymptotic KS p-value corrections, documented in the report. No bindings were
regenerated for these report-only C++ diagnostics; editor UI checks remain separate.

Representative clangd definition hovers for `zobrist_boards` and
`ZigguratNormal::simulate_measurements` return their complete adjacent contracts
and examples (headless check; graphical editor UI remains unverified).

### Compiled Zhu-Takaoka search report conversion

The string-search-extra header and implementation pass the presence audit for
all 27 definitions. Ten header examples compile and run; seven focused C++17
tests pass, including 16 000 random cases against std::string-based naive
matching, textbook comparison counts, binary bytes, overlap, trace conservation
and record boundaries. The actual search loop now provides the scan counters
and trace used by the report, with reusable preprocessing. Independent Python
uses brute-force suffix consistency (instead of the core's linear border
preprocessing) and str.find to reproduce all 15 corpus match/work totals.
The report executes five blocks, exports to Markdown and includes two inspected
SVG figures. No generated bindings changed for these report-only C++ helpers.

### ZeroR DataFrame report conversion

The new ZeroR adapter header and implementation pass the presence audit for
all nine definitions. The changed ZeroR declarations in rule_learners.hpp
now document nominal-label semantics, deterministic ties, actual map-based
complexity, empty models, ignored features and training-only accuracy. The
OneR portion remains outside the completed-file manifest. All eight affected
header examples compile and run. Five focused C++17 tests pass, including
invalid labels, empty evaluation, unseen classes and an evaluation set with
a different majority to ensure there is no refitting. Independent Python
reproduces the SplitMix64 shuffle, training/test counts and every one of the
20 split-sensitivity rows. Five report blocks execute, Markdown exports and
three SVGs are visually inspected. Report-only adapters do not change bindings.

Representative clangd hover also returns the compiled ZhuTakaoka::scan
contract and its example. The graphical editor remains a separate check.

The DataFrame zero_r_train overload also passes a representative clangd
definition-hover check, including the no-rounding label contract and example.

### Zeller calendar report conversion

The Zeller header and new implementation pass the presence audit for all 13
definitions. Six header examples compile and run; eight focused C++17 tests
cover leap-century exceptions, invalid dates, negative/extreme years, complete
cycle progression, Friday-the-thirteenth counts and checked DataFrame features.
Independent Python datetime agrees with every one of 146097 dates in the
2000–2399 cycle; calendar/date operations verify all 144 airline monthly rows.
Six report blocks run, Markdown exports, and three SVGs were visually inspected.
Representative clangd hover returns the calendar contract and example. These
are report-only C++ helpers; no generated bindings changed.

### Z-algorithm report conversion

The Z-algorithm header and source pass the presence audit for all 20 definitions.
The shared loop produces arrays, traces and actual successful/failed comparison
counters. A virtual integer separator outside the byte range supports all
256 byte values without copying a joined string. Thirteen C++17 tests pass,
including naive-array comparisons, full-byte inputs, overlaps, empty patterns,
trace conservation and corpus boundaries. Eight header examples compile and run,
and representative clangd hover returns the complete preprocessing contract.
Independent Python byte-find verifies all 5346 record/query position lists.
Four report blocks run, export to Markdown and generate three inspected SVGs.
No generated bindings were changed for these report-only C++ diagnostics.

### xoshiro256** report conversion

The quality and bootstrap headers and implementations pass the presence audit
for all 37 definitions. The changed Xoshiro state/seed/transition/jump contracts
are documented separately; the wider random core retains an existing backlog
and is not added to the completed-file manifest. Fourteen affected header
examples compile and run, and clangd returns the long-jump contract. Eleven
focused tests pass, including the preserved default MT19937-64 bootstrap stream,
caller-supplied engines, factory validation and normal-cache clearing on jumps.
The authors’ public-domain C reference agrees on initial, jump and long-jump
states and outputs. Independent Python reproduces all 5000 bootstrap means; R
agrees on standard errors, bias and intervals. Five report blocks execute and
export to Markdown with four inspected SVGs. Both 40-seed quality batches are
reported, including unusually high and low second-level p-values. No generated
bindings changed for these report-only C++ additions.

### xoroshiro128+ report conversion

The quality header and implementation pass the presence audit for all 13
definitions, including the new held-out binary recurrence diagnostic. The
changed xoroshiro128+ contracts document version 1.0 parameters, low-bit
linearity, state ownership, jump spacing and normal-cache behaviour; the wider
random core remains outside the completed-file manifest. Seven header examples
compile and run, and clangd returns the recurrence diagnostic contract. All
29 focused/existing generator tests pass. The authors’ compiled C reference
agrees on states and eight outputs after initialisation and each jump. An
independent binary Gaussian-elimination oracle verifies minimal recurrence
lengths; separate Python bit arithmetic checks all 2048 raw words and 10000
bootstrap means. R agrees on car data, standard errors, bias and percentile
endpoints. Six report blocks execute, Markdown exports and four SVGs are
visually inspected. These C++ diagnostics require no generated binding changes.

### Xiaolin Wu line report conversion

The line header and new implementation pass the presence audit for all 18
definitions. The existing four-argument full-column API is preserved; an
explicit clipped-column policy accounts for endpoints, including segments
inside one column. Contracts distinguish distance-filter ink from exact area
coverage, projected length from Euclidean length, and additive saturation
from alpha-over compositing. Seven header examples compile and run; clangd
returns the DataFrame polyline contract. Thirteen focused/existing graphics
tests pass, including 1000 random all-octant conservation/reversal checks,
coordinate bounds, missing cells, extreme finite values and coincident points.
Independent exact-fraction Python arithmetic verifies 64 segment-policy cases
and every accumulated airline pixel at both report resolutions. R agrees on
all 144 passenger values and annual summaries. Four report blocks run, export
to Markdown and generate three visually inspected SVGs. The graphical editor
remains unverified; these report-only C++ helpers require no binding changes.

### Iris XGBoost-style report conversion

The new diagnostics header and implementation pass the presence audit for all
six definitions. Changed classifier constructors/options, predictions and new
evaluation/staged/confusion APIs have contracts and examples; the classifier's
older accessor/helper documentation backlog remains outside the completed-file
manifest. Twelve affected header examples compile and run; clangd returns both
the staged-evaluation and validation-search contracts. Twenty-four C++ tests
pass, including prefix-refit equivalence under subsampling, independently
computed scores, complete-case alignment, unknown classes, invalid options and
deterministic selection ties. An independent NumPy implementation uses sorted
prefix-sum split search rather than the core's repeated threshold rescans. It
verifies every partition row, all 505 validation stages, selection of lambda 10
and 57 rounds, all 58 refit test stages, 90 probabilities and four importances.
R agrees on the iris data and final accuracy/log loss/Brier score. Four report
blocks execute, export and produce four inspected SVGs. The report distinguishes
the local diagonal Hessian from the pinned upstream 3.1.0 curvature bound and
keeps validation selection separate from the final test. No bindings regenerated
for these report-only C++ diagnostics; graphical-editor verification is separate.

### Winnow sparse-pixel report conversion

The new Winnow adapter header and implementation pass the presence audit for
all 18 definitions; changed Winnow contracts in the shared online-classifier
header are documented separately, with the Perceptron area outside this batch.
Nine affected header examples compile and run. Eleven C++ tests pass, including
shared-loop trace equivalence, online versus post-pass errors, Boolean/shape
validation, strict ties, numerical range failures and random disjunction bounds.
An exact integer-exponent Python oracle verifies all final weights, complete
pass records, seeded split rows, five dimension trials and counterfactual scores.
All 1797 dataset records agree with upstream scikit-learn's compressed data file.
Four report blocks execute, export and produce four inspected SVGs. The report
derives a potential bound under explicit realizability/update assumptions and
shows why zero observed test errors do not establish full-domain OR equivalence.
Clangd returns the probe contract; graphical-editor verification remains separate.
No generated bindings changed for these report-only C++ adapters.

### Wildcard matching report conversion

The new wildcard header and implementation pass the presence audit for all 29
definitions. The shared last-star loop now checks a pattern star before literal
equality, fixing text containing stars. The changed original wildcard_match
contract documents anchoring, byte semantics and quadratic worst-case work;
the wider substrings module remains outside this completed-file batch. Eight
header examples compile and run; clangd returns both the basic byte contract
and the trace contract. Thirteen C++ tests pass, including 132496 exhaustive
metacharacter-text pairs, 1000 random binary cases, trace/counter conservation,
DP agreement, resource budgets, null fields and exact adversarial counts.
Python fnmatchcase(bytes) agrees on all 7128 record/pattern decisions. An
independent epsilon-NFA agrees on all 184 displayed prefix states. Four report
blocks execute, export and produce four inspected SVGs; pattern table cells
retain their stars in Markdown. The index distinguishes this restricted
non-recursive implementation from recursive wildmat. No generated bindings
changed for these report-only C++ diagnostics; graphical-editor checks are separate.

### Warnsdorff report conversion

The new Warnsdorff header and implementation pass the presence audit for all
25 definitions. The legacy function contract is updated separately; the wider
tour-search module remains outside this completed-file batch. Seven header
examples compile and run; clangd returns the canonical tie-breaking contract.
Twelve C++ tests pass, covering trace decisions, valid partial paths, resource
bounds, grid permutations and invalid inputs. An independent graph/set oracle
verifies all 158 paths and 6182 decisions; legacy complete-or-empty paths are
unchanged. NumPy verifies both sums and both variations for all 1797 digit images.
Four report blocks execute, export and produce four inspected SVGs, including
yellow unvisited squares on the failed attempt. The report distinguishes greedy
failure from impossibility and shows that the tour is not a smoothness optimizer.
No generated bindings changed; graphical-editor verification remains separate.


### SVM report conversion

The SVM header and implementation pass the presence audit for all 83 definitions.
Contracts document training-only sample-SD scaling, sorted pair signs and vote ties,
coefficient pruning, original row indices, missing-data behaviour, dense-Gram storage
and the simplified SMO stopping limits. New C++ APIs expose pair decisions, all
training coefficients/margins, primal/dual/KKT diagnostics, labelled error and
sampled two-predictor decision regions. The KKT check ignores rounding-sized
box-face distances with epsilon=min(1e-8,1e-6*C); an iteration budget does not
certify convergence, and indefinite sigmoid kernels have no convex gap guarantee.

Validation: 14 SVM gtests pass, including exact two-point solutions over scales
and costs, the degenerate XOR polynomial solution, votes reconstructed from pair
decisions, original-row/missing-data handling, empty-string response classes and
plot argument validation. All header examples compile and run; four canonical
symbol descriptions are checked through clangd. Scikit-learn 1.8.0 agrees on all
600 train/holdout predictions across the four predeclared iris kernel models;
NumPy independently reconstructs all 12 pairwise objectives from the exported
coefficients. Eight report blocks execute and export, with five SVGs inspected
at browser width through Inkscape. Legacy mepml/org/tex files are removed and
the algorithm index repointed. SVM is already excluded from generated SWIG APIs;
no generated binding files change. Graphical-editor help is not verified.


### MDS report conversion

The MDS header and implementation pass the presence audit for all 43 definitions.
Contracts correct classical scaling's objective to squared centred-Gram error
(strain), distinguish positive/absolute inertia from direct distance residuals,
and document raw feature units, null-dropping/alignment, precomputed-distance
validation, zero-distance behaviour and arbitrary eigenvector signs/bases.
New report-facing C++ APIs accept explicit dissimilarities, expose the accepted
matrix and untruncated Gram matrix, and return spectrum/prefix/pair diagnostics
and Shepard plots. No change to the existing feature-constructor signature;
negative eigenvalues remain visible and are clamped only for coordinates.

Validation: 16 gtests pass, covering hand-derived rectangle and four-cycle
spectra, strain versus distance loss, scale laws, all-prefix residuals, exact
unscaled PCA geometry, null-row indices, zero distances, invalid inputs and
within-tolerance symmetrisation. Eight header examples compile and run; clangd
checks the class and three new API descriptions. R 4.5.2 cmdscale agrees on the
spectra, fitted Grams and both GOF fractions, including all one- through four-axis
iris fits for both metrics; prcomp confirms the unscaled PCA equality. NumPy
independently reconstructs all spectra, prefix diagnostics and 22362 pair distances
across the rectangle, cycle and two iris fits. All 101 cycle spectra and strain
floors match closed forms. Nine report blocks execute and export with six SVGs
inspected through Inkscape at browser width. Legacy mepml/org/tex files are
removed and the algorithm index updated. The new APIs remain C++ only; existing
SWIG exclusions/facades are unchanged. Graphical-editor help is not verified.


### U-statistic report conversion

The U-statistic header and implementation pass the presence audit for all 43
definitions. Contracts distinguish complete data-sampling diagnostics from
conditional Monte Carlo variability, specify distinct indices within each draw
and replacement across draws, and document fixed-degree/non-degenerate limits,
coverage and NaN conventions, within-STL seeded reproducibility and term-count
overflow. Existing UStatisticResult fields and their leading-projection semantics
are preserved. New owned analysis exposes inclusion means, delete-one values,
pseudo-values, conditional kernel variance/Monte Carlo error and a complete-fit
jackknife. Tuple tracing, repeated-budget calibration and exact two-point laws
keep reusable diagnostic logic in the C++ core. Finite input and callback values
are validated; callback exceptions propagate.

Validation: 25 resampling gtests pass (13 new diagnostics tests plus 12 existing
resampling tests), including hand-computed three-point values, independent
jackknife refits at several degrees, complete/random tuple invariants, sparse
coverage, single-draw/whole-sample limits, exact degenerate and non-degenerate
sampling laws, general-degree agreement with all 32 ordered binary samples,
probability endpoints, Monte Carlo RMSE and overflow before evaluation. Seven
header examples compile and run; clangd checks four canonical descriptions.
R 4.5.2 var, pairwise differences and explicit leave-one-out refits agree on all
eight iris summaries; R dbinom and independent binomial calculations agree on
all 12 exact count laws. NumPy reconstructs every inclusion mean/pseudo-value,
checks 4440 recorded tuples and independently aggregates 1536 incomplete runs.
Seven report blocks execute/export, with six SVGs inspected at browser width
through Inkscape. The report corrects the canonical/overlap variance distinction,
non-degeneracy and growing-degree claims and the interpretation of iris's
balanced species design. Legacy mepml/org sources are removed and the algorithm
index updated. Report additions are C++ only; no generated wrappers change.
Graphical-editor help is not verified.

## P-spline report conversion (2026-10-28)

Completed `include/datamunge/pspline_regression.hpp`,
`src/datamunge/pspline_regression.cpp`, `include/datamunge/fda/pspline.hpp` and
`src/datamunge/fda/pspline.cpp`: 70/70 definitions documented, with zero missing
in fresh Doxygen XML. The completed-path manifest gate also passes (6495/6495
definitions). Contracts distinguish extended uniform knots from repeated
boundary knots, coefficient-index versus physical polynomial nullspaces, summed
loss scaling, bounded GCV search, numerical failure and fixed-lambda deletion.
New C++ diagnostics expose the difference operator, leverage, deletion residuals
and CV, basis evaluation, coefficient differences, scalar/observation tables and
a lambda path. Squared differences replace the cancellation-prone quadratic form
for roughness; finite data/options and in-domain low-level training are checked.
Existing constructor forms and their separate knot conventions are retained.

Validation: 21 P-spline tests pass (11 new checks and ten existing tests), plus 11
existing B-spline and formula tests exercising the shared FDA fitting layer.
Checks cover the exact three-point solution, independent fixed-basis deletion
refits, unit-response smoother diagonals, difference Gram matrices and nullspaces
through order four, monotone RSS/roughness/edf paths, clamped prediction consistency,
undefined interpolation diagnostics and non-finite inputs. All 19 header examples
compile and run; clangd verifies four representative canonical descriptions.
R 4.5.2 splineDesign and augmented QR independently reproduce eight fits, predictions,
GCV minima and all 272 Old Faithful delete-one refits; maximum scalar discrepancy
is 2.6e-9. SciPy basis evaluation, augmented least squares and a generalized penalty
eigensystem verify all 41 lambda-path rows and the two unpenalized modes (largest
scalar discrepancy 1.1e-8). The rewritten `docs/mepml/examples/pspline.mepml` has seven
executed blocks, six SVGs inspected through Inkscape at browser width, and verified
Markdown/Doxygen exports with nine display formulas and working source links.
It covers hand calculations, a flat tuning criterion, boundary/nullspace distinctions,
basis approximation limits on the original oscillating signal, and a 272-row faithful
analysis with residual/leverage and basis-size sensitivity checks. The legacy mepml
source is removed (no org/TeX companion existed), and the conversion plan and algorithm
index are updated. Report diagnostics are C++ only; generated wrappers are unchanged.
Graphical-editor help is not verified.

## De Boor report conversion (2026-10-28)

Completed `include/datamunge/geometry/de_boor.hpp` and
`src/datamunge/geometry/de_boor.cpp`: 39/39 definitions documented in fresh Doxygen
XML, zero missing. The completed-path manifest gate also passes (6650/6650
definitions). Public contracts cover degree and knot counts, finite inputs,
maximum multiplicity and nonempty end spans, right-hand interior-knot values and
left-limit final endpoints, ownership, complexity, parameter versus arc-length
sampling, interpolation traces and discontinuity-aware plotting. The geometry
umbrella now exposes the evaluator. Existing free-function signatures are retained;
the new owned DeBoorSpline validates once for repeated O(log N+p^2) evaluation.
The sampler evaluates the actual right endpoint, fixing its fixed-epsilon shift,
and malformed inputs now fail explicitly. Bounded interpolation and scaled knot
ratios avoid overflowing opposite-sign finite coordinate/knot differences.

Validation: 18 focused geometry tests pass (12 new DeBoorSpline checks, three
existing de Boor checks and three existing geometry regressions). Checks include
hand-computed triangles, nonuniform weights, exact endpoints on tiny domains,
degree zero, unclamped endpoints, repeated-knot continuity and jumps, plotted span
separation, local perturbation versus independent basis values, eight degrees of
basis comparisons, twelve degrees of Bezier/closed-form agreement and work counts,
affine equivariance, convex bounds, extreme finite values and invalid inputs.
All nine header examples compile and run, including two plotting examples; four
canonical descriptions pass clangd hover checks. R 4.5.2 splineDesign and SciPy
BSpline agree on 29 curves and 3668 evaluations, including near-knot side values,
nonuniform/repeated/unclamped knots and tiny/translated domains, with maximum
coordinate discrepancy 1.8e-15. Exact rational arithmetic on the represented
inputs checks 509 recorded blends, with maximum coordinate discrepancy 4.5e-16.

The rewritten `docs/mepml/examples/de_boor_analysis.mepml` has six executed blocks
and six SVGs inspected through Inkscape at browser width. It derives the evaluator
by collecting coefficients in the basis recursion, shows all intermediate steps,
distinguishes absolute control conditioning from relative/knot sensitivity, verifies
closed forms and tiny-domain endpoints, and gives the original cubic a full trace.
No empirical dataset is appropriate for prescribed-curve evaluation. Markdown and
Doxygen exports preserve six programs, six figures, ten display formulas and
source links. Legacy mepml/org files are removed; no companion TeX/HTML or figure
files existed. Conversion and algorithm indexes are updated. These additions are
C++ report helpers; generated wrappers are unchanged. Graphical-editor help is
not verified.

## Gaussian process regression report conversion (2026-10-28)

Completed `include/datamunge/stats/gaussian_process_regression.hpp` and
`src/datamunge/stats/gaussian_process_regression.cpp`: 92/92 definitions documented
in fresh Doxygen XML, zero missing. The completed-path manifest gate also passes
(6776/6776 definitions). Contracts specify empirical preprocessing,
expanded-feature distance, population scale conventions, grid bounds and flags,
variance profiling/floor, standardized versus original-unit log density, row
alignment, finite-input validation, uncertainty targets and numerical failure.
The historical fitted_values/rmse/r_squared accessors retain their fixed-fit
conditional deletion meaning, now labelled explicitly rather than presented as
validation of a refitted preprocessing/tuning pipeline.

The likelihood and final covariance now use the same signal-relative numerical
ridge, configurable by the appended jitter_ratio option. The Gaussian density
retains q/s when the profiled variance floor is active, fixing the constant-response
case. Stable kernel distance scaling avoids inverse-length-square overflow at tiny
or huge positive length scales. New methods expose physical new-observation intervals,
joint posterior covariances, row-aligned prediction/deletion tables, training posterior
means, full hyperparameter-search records, original-unit model summaries and external
scoring (errors, both interval coverages, width and Gaussian log predictive density).
Existing normalized-unit variance/likelihood accessors remain source-compatible.
The options formatter includes the new ridge setting and its focused tests are updated.

Validation: 25 GP gtests pass (11 existing and 14 new), plus two focused formatting
checks. Independent checks cover exact two-point means/covariances/deletions,
repeated-query dependence versus independent observation noise, non-negligible
jitter consistency, variance-floor Gaussian density, response-unit/Jacobian
equivariance, explicit fixed-parameter deletion submatrix refits, PSD/symmetry,
profile-grid maxima, singleton automatic flags, prior limits, external coverage/NLPD,
null alignment, invalid inputs, extreme length scales and the distinction between
conditional deletion and refitting the empirical mean. All eight header examples
compile and run, including a plotting example; four representative canonical
class/method/field descriptions pass clangd hover checks.

R 4.5.2 solves and scikit-learn fixed-kernel GaussianProcessRegressor reproduce nine
fits and 197 joint queries, including constant response, sizeable jitter, repeated
queries, multivariate features and training-only iris fits. Maximum discrepancies
are below 9e-14 for means, 1e-14 for covariance entries and 6e-13 for log densities.
Independent NumPy/SciPy calculations verify all 607 profiled candidates (largest
amplitude/likelihood discrepancy 1.3e-9), every conditional deletion through explicit
submatrix refits (largest discrepancy 4.8e-14), and all eight held-out score fields.
The model is trained on 105 flowers and scored on 45 held out before any preprocessing
or tuning: RMSE 0.402965 cm, 42/45 new-observation coverage versus 14/45 latent-interval
coverage, with 95% nominal pointwise intervals.

The rewritten `docs/mepml/examples/gaussian_process_regression_iris_analysis.mepml`
has seven executed blocks, six SVGs inspected with Inkscape at browser width, and
verified Markdown/Doxygen exports with nine display formulas and source links.
It clarifies interval targets, variance units, fixed-fit conditional deletion and
finite extrapolation limits. Legacy mepml/org/TeX sources are removed; conversion
and algorithm indexes are updated. New diagnostics are C++ only; generated wrappers
are unchanged. Graphical-editor help is not verified.

## Generalized linear model iris report conversion (2026-10-28)

Completed `include/datamunge/stats/glm.hpp` and `src/datamunge/stats/glm.cpp`:
122/122 definitions documented in fresh Doxygen XML, zero missing. The completed-path
manifest gate passes at 6942/6942 definitions. Contracts cover family/link pairs,
response encoding and domains, prior weights, row omission/alignment, finite-value
validation, IRLS initialization and stopping, QR complexity, retained last iterates,
boundary warnings, weighted residuals, inference targets and ownership/thread safety.
The Gamma family AIC approximation and fractional-weight/count conventions are explicit.

New reusable methods expose accepted coefficient/deviance/step traces, reconstructible
working responses and weights, coefficient/Wald tables, original training-row diagnostics,
fit-status summaries and external binary accuracy/Brier/log-loss scores. Numerical
convergence is exposed separately from a near-boundary-probability heuristic; neither
claims to test MLE existence. Logistic inverse links are evaluated stably and bounded
at machine epsilon. Later invalid or deviance-increasing steps are halved; an invalid
first update throws because no preceding feasible coefficient vector is available.
Final covariance and leverage now use weights at the returned means, including at the
iteration limit, and are computed from QR without forming normal equations. Weighted
Gaussian AIC includes its log-weight term; weighted deviance residuals square to total
deviance. No-intercept null deviances use the zero link predictor. Mean-response intervals
validate their levels and respect the positive Gamma inverse-link domain; an interval
crossing zero on that link scale has an infinite upper response endpoint.

Validation:

- 33 C++23 native tests pass: 9 existing GLM tests, 15 new diagnostic/closed-form/edge
  tests and 9 GLMM regression tests. New checks cover exact two-group Bernoulli and
  four-count Poisson information, reconstruction of accepted IRLS updates, final-weight
  covariance after a single update, weighted likelihood/residual identities, row omission,
  scoring losses, separated-but-finite paths, invalid inputs and Gamma interval domains.
- Eight R `glm()` oracle fits cover all four families with and without positive integer
  prior weights, using the actual embedded iris measurements and tight fitting tolerances.
  Across 5776 values, maximum coefficient error is 6.3e-14, fitted-mean error 2.3e-14,
  covariance error 2.3e-9, interval-endpoint error 2.1e-9 and AIC error 2.0e-11.
- The report's exact 70/30 split was exported and refit in R. Accuracy, Brier loss and log
  loss agree within 4.4e-16, and training AIC within 4.3e-14. Full-data petal coefficients
  agree within 5.0e-14. Its standard errors agree within 4.8e-13 with R information
  recomputed at the returned means; R's default summary uses preceding-iteration weights
  and differs by up to 0.00209 on this sharper fit.
- All 13 header code examples compile and run, including prediction/diagnostic tables and
  all four SVG diagnostic recipes. Four clangd hover checks pass for the class, working-data
  method, boundary flag and iteration-limit field. The mep graphical K integration was
  not verified separately.
- The rewritten `docs/mepml/examples/glm_iris_analysis.mepml` executes all seven programs.
  All seven SVGs were rendered with Inkscape and visually inspected at 1100 px, including
  fitted two-panel layouts and corrected distinct series colours. Both Markdown and
  Doxygen exports pass; the latter retains seven C++ programs, seven copied SVGs and
  nine display formulas with rendered reference links.

The report starts with binary-group and four-count hand calculations, then replication
and an explicit separation proof, followed by iris and a prespecified three-model holdout
comparison. The petal model classifies 29/30 test flowers correctly (Brier 0.0254041,
log loss 0.0851926), versus 22/30 for sepals. A deliberately artificial rounded-sepal
Poisson fit is labelled a variance-assumption counterexample, with Pearson X-squared/df
about 0.033. The previous Gaussian-versus-binomial AIC comparison is removed because its
continuous density and discrete mass have incompatible reference measures. Legacy Mepml,
Org, TeX and generated HTML copies were removed; the conversion checklist and algorithm
index point to the replacement.

## Generalized linear mixed model report conversion (2026-10-28)

Completed `include/datamunge/stats/glmm.hpp` and `src/datamunge/stats/glmm.cpp`:
135/135 definitions documented in fresh Doxygen XML, zero missing. The completed-path
manifest gate passes at 7170/7170 definitions. Contracts specify the single-group formula
and random-slope grammar, binary/integer-count response domains, row omission/alignment,
finite-value and optimizer-control validation, covariance scaling, full-state convergence,
boundary heuristics, ownership, const concurrency and low-rank computational complexity.
They distinguish working-ML PQL from exact marginal likelihood and working REML, conditional
Wald SEs from variance-component uncertainty, and b=0 predictions from marginal new-group
means. Historical deviance/aic/bic accessors retain their working-Gaussian meanings with
explicit documentation and summary labels; the report does not rank response models by them.

New reusable methods expose fixed-coefficient/group tables, accepted PQL traces, frozen
working responses/weights, working-criterion profiles, retained training rows/diagnostics,
model status, explicit zero-random-effect predictions, row-aligned prediction status and
binary probability scoring. A known random-intercept SD option supports conditional fits
and exact hand calculations, including the zero-variance GLM limit. The simulator returns
balanced grouped binary/count observations with known effects and conditional means; it
preserves the same Gaussian group-effect draws across cluster-size experiments with the
same seed. Its standard-library distribution draws are not promised portable across C++
standard-library implementations. The handwritten options formatter includes the new field.

Working covariance solves now use q-by-q Woodbury systems, avoiding an n_g-by-n_g Cholesky
per group and candidate. Covariance optimization adds a bounded local polish after DE and
checks the exact zero-variance candidate for random-intercept fits. The outer stopping rule
checks relative changes in fixed coefficients, group effects and covariance factors, fixing
premature stopping when fixed coefficients remain stationary. Stable logistic inverse links,
finite-domain guards and response validation prevent silent nonfinite working fits. Numeric
group keys use enough precision to preserve distinct doubles. Known-group prediction no
longer silently treats missing random slopes as zero; null required slopes produce NaN,
and missing required columns throw. Binary scoring removes response-null rows before query
validation, consistent with joint omission.

Validation:

- 48 C++23 statistical tests pass: 9 existing GLMM tests, 15 new GLMM diagnostics tests,
  and all 24 GLM tests. Two affected formatter tests also pass. New checks include the
  exact first Poisson and binary updates, the analytic first-working-fit variance optimum,
  known-variance penalized score equations, the zero-variance GLM limit, numeric group-key
  collisions, prediction status/null behavior, row alignment, scoring, generator invariants
  and invalid controls/domains.
- The correlated random-intercept/slope implementation is rechecked with independent full
  n-by-n covariance solves: an explicitly non-diagonal covariance factor validates the
  working criterion, and the fitted factor validates first-step fixed effects and every
  group BLUP. This checks the Z*Lambda multiplication and D*Z-transpose BLUP product.
- An independent NumPy/SciPy oracle covers binomial and Poisson fits, both with known SD
  and estimated SD: 4 models, 25 outer updates and 2000 working rows. Maximum working-mean,
  weight and response errors are below 1.8e-15; fixed-step coefficient error is 1.3e-15,
  working-criterion error 2.3e-13 and final covariance error 4.2e-17. A separate bounded
  scalar optimizer agrees in objective within 1.2e-13 and SD within 8.2e-8. Known-variance
  fits also satisfy independently evaluated original penalized-family scores within
  6.3e-14, with no improvement in a direct SciPy joint-mode optimization. R MASS/nlme/lme4
  are unavailable in this environment; no exact-marginal or REML equivalence is claimed.
- All 13 header code examples compile and execute. Four clangd hover checks pass for
  the model class, working-data method, working-criterion accessor and known-SD field.
  The mep graphical K integration was not separately verified.
- `docs/mepml/examples/glmm_analysis.mepml` executes all 8 programs. All 7 SVGs were rendered
  with Inkscape and visually inspected at 1100 px; multi-panel layouts fit their panels.
  Markdown and Doxygen exports pass. The Doxygen export retains 8 C++ programs, 7 copied
  SVGs and 10 display formulas with reference links.

The report starts with four-count and binary-group hand calculations, then profiles a
frozen working variance and checks recovery on simulated Poisson groups. On 333 complete
penguins, adding species changes island SD from 1.179775 to an exact zero boundary, with
only three islands available for variance estimation. The prespecified 233/100 within-island
train/test split improves from 81 to 89 correct predictions after species adjustment, with
Brier loss 0.144562 versus 0.074401 and log loss 0.453253 versus 0.244815. Separate whole-island
holdouts use the explicitly labelled unseen-group b=0 fallback, without test-label BLUPs or
claims of logistic-normal marginal prediction. Legacy Mepml/Org/TeX copies were removed,
and the conversion checklist and algorithm index now point to the replacement.


## Isomap report conversion (2026-10-28)

Completed `include/datamunge/stats/isomap.hpp` and `src/datamunge/stats/isomap.cpp`:
60/60 definitions documented in fresh Doxygen XML, zero missing. The completed-path
manifest gate passes at 7241/7241 definitions. Contracts specify unscaled Euclidean or
Manhattan metrics, joint null omission, retained/original row alignment, finite domains,
2<=k<n, deterministic distance/index neighbour ordering, union rather than mutual edges,
zero-weight duplicate edges, disconnected-graph rejection, ownership and const concurrency.
The historical goodness-of-fit fraction is explicitly positive inertia, not a certificate
of distance preservation. Signed spectra, negative modes, prefix strain, pair-distance
errors and constant-vector correlation conventions are documented separately.

Reusable `isomap_graph()` and `IsomapGraph` expose directed neighbours, union adjacency,
input distances, component labels, degree/edge summaries and projected graph plots without
requiring connectivity. `Isomap` exposes its graph, geodesic and Gram matrices, spectrum,
prefix diagnostics, pair-distance tables, independent-reference diagnostics and Shepard
plots. Reference diagnostics include maximum intrinsic-to-input selected-edge stretch.
Dense contiguous Floyd–Warshall replaces the hash-graph shortest-path path, and a scaled
Householder/QL symmetric eigensolve replaces Jacobi. Extended-precision distance and
centring accumulation, scaled eigenanalysis and explicit finite/squared-range checks
support uniform rescaling while rejecting unrepresentable geometry.

Validation:

- 35 C++23 statistical tests pass: 7 existing Isomap tests, 12 new diagnostics tests and
  all 16 MDS tests. Two affected formatter tests also pass. Closed forms check line
  distances/Gram entries and the square cycle's (2,2,0,-1) spectrum; additional tests cover
  deterministic union ties, disconnected components, zero-weight duplicates, complete-graph
  equivalence to MDS for both metrics, nested neighbourhoods, scaling by 1e-100 and 1e100,
  row alignment, reference-edge stretch including zero denominators, constant correlation
  and invalid domains/options.
- An independent NumPy/SciPy oracle checks iris (150 rows, k=25) and two 300-point Swiss-roll
  fits (Euclidean and Manhattan, k=10). Independently accumulated input distances and
  distance/index ordering reproduce the graph; SciPy Dijkstra and NumPy eigenanalysis
  agree with geodesics within 1.5e-14, Gram matrices within 2.9e-12, embedding Gram matrices
  within 6.4e-12, full spectra within 8.1e-11 and prefix diagnostics within 1.8e-15.
  Stock scikit-learn agrees on both continuous-roll cases (embedding Gram error <=2.2e-12).
  Its iris graph differs because floating-point distance rounding and neighbour ties
  differ: maximum geodesic difference 0.1542, embedding Gram difference 0.3170. No iris
  scikit-learn equivalence or cross-implementation tie portability is claimed.
- All 13 header code examples compile and execute. Four clangd hover checks pass for the
  class, edge-table method, reference-diagnostic method and neighbour-count field. The
  mep graphical K integration was not separately verified.
- `docs/mepml/examples/isomap_iris_analysis.mepml` executes all 8 programs. All 7 SVGs were
  rendered with Inkscape and visually inspected at 1100 px; multi-panel layouts fit their
  panels, titles are readable and square-cycle axis scales preserve the square's shape.
  Markdown and Doxygen exports pass. The Doxygen export retains 8 C++ programs, 7 copied
  SVGs and 6 display formulas with rendered reference links.

The report begins with a four-point line, a square-cycle counterexample (100% positive
inertia and zero residual variance despite 23.91% relative graph-distance error) and two
disconnected triangles. A noiseless 1000-point Swiss roll uses withheld arc length and
height as independent intrinsic coordinates: embedding error is 7.45% at k=6 and 67.53%
at k=40. Moving from k=6 to k=8 introduces a maximum edge stretch of 10.94; a lower-density
500-point sample already exhibits that shortcut at k=6. Iris first connects at k=25 under
the stated rule, with two-dimensional distance error 6.31% against its graph and 17.12%
against ambient distances (MDS ambient error 4.18%). Species only colours the plots;
connectivity and visual class separation are not presented as manifold validation or
predictive performance. Legacy Mepml/Org/TeX copies were removed, and the conversion
checklist and algorithm index point to the replacement.

## Classical hypothesis-tests report conversion (2026-10-28)

Completed 24 affected header/implementation paths for Student/Welch/paired t tests,
Wilcoxon tests, classical ANOVA/Kruskal–Wallis, Pearson/Spearman/Kendall coefficients,
chi-square tests, variance comparison, Shapiro–Francia, KS, proportion/binomial tests,
Fisher's exact test, shared result storage and reusable diagnostics. Fresh Doxygen XML
covers 116/116 definitions with zero missing. The completed-path gate passes at
7408/7408 definitions. Contracts specify finite inputs without automatic omission,
paired alignment, each null/reference assumption, exact versus approximate inference,
tie/continuity conventions, interval estimands, ownership and absence of shared mutable
state. They distinguish equal distributions from equal medians/means, Pearson from
Spearman and pooled from within-group association, and require independent fixed CDF
parameters for ordinary one-sample KS calibration.

New `hypothesis_diagnostics` APIs provide owned result/rank tables, a Student null-tail
plot, Fisher's conditional support, complete binomial rejection tables and deterministic
size/power enumeration. The source is in the library source manifest and stats umbrella.
Normal p-values use erfc, Student tails use symmetry, F tails complementary beta and
chi-square tails upper incomplete gamma; Fisher tails sum directly and binomial tails
use beta identities/probability-ordered log-PMF sums. These avoid cancellation zeros in
the iris results. Finite/control/group/count checks replace several silent NaN results;
constant-rank/zero-variance domains have explicit errors, Pearson n=3 omits an unavailable
Fisher-z interval, and perfect correlations use their limiting reference. Extended-
precision moment/correlation accumulation improves numerical stability. Variance confidence
bounds now invert a lower F tail from a zero bracket and use reciprocal symmetry, fixing
a nonterminating positive-bracket expansion at high confidence and low numerator df.

Validation:

- 73 C++23 native tests pass: 51 existing hypothesis/multiplicity tests, 18 new diagnostics
  tests and 4 existing Kendall/sweep tests. Two affected formatter tests also pass. Checks
  cover the four-point Student closed form, tied U variance, three-group F/H formulas,
  preserved tiny and symmetric tails, rational binomial/Fisher supports, inclusive endpoint
  nulls, non-duality of a probability-ordered p-value and central interval, sparse score
  size inflation, row labels/absent intervals, invalid controls/domains and the analytic
  F(1,1) quantile at 99.9% confidence.
- An R oracle checks 100 reference cases and 516 scalar quantities, including all three
  alternatives, Welch/pooled/paired tests, tie-corrected rank procedures, correlation,
  variance, exact/approximate count tests, endpoint nulls and the report's iris fits.
  Largest statistic/estimate error is 2.1e-10, df error 7.2e-14, p-value error 2.2e-9
  (relative <=3.1e-9), and ordinary interval error 4.0e-10. At the deliberately extreme
  F(1,1) upper interval near 405285, R differs by 5.4e-5 absolute (1.4e-10 relative);
  the implementation also passes the independent tangent-squared quantile formula.
  Shapiro–Francia is checked with an independently written R Blom-score/Royston formula;
  the nortest package is unavailable. KS comparisons check D, not equality with R's
  differently calibrated p-values; three separate SciPy Kolmogorov-survival checks verify
  the implementation's stated asymptotic formula.
- A separate R enumeration covers n=1..100 with both score-correction settings: 200 rows,
  800 size/power values, maximum error 7.1e-14. It independently recomputes exact and score
  rejection rules and sums their Binomial(n,.1) and Binomial(n,.3) probabilities.
- All 29 header code examples compile and run. Five clangd hover checks pass for the
  alternative enum, Welch test, exact binomial test, operating-characteristic helper and
  fixed-null KS contract. The mep graphical K integration was not separately verified.
- `docs/mepml/examples/hypothesis_tests_analysis.mepml` executes all 10 programs. All 10
  SVGs were rendered with Inkscape and visually inspected at 1100 px, including fitted
  multi-panel layouts and the shortened interval legend. Markdown and Doxygen exports
  pass; the latter retains 10 C++ programs, 10 copied SVGs, 7 display formulas and rendered
  reference links.

The rewritten report begins with hand-checkable Student, tied-rank, ANOVA, binomial and
conditional Fisher examples. At n=5,p_null=.1, the uncorrected score test has actual size
8.146%, versus 0.856% for the exact test at nominal 5%. The 0/10,p_null=.3 example has
p=.03884 while its central 95% interval includes .3, illustrating different orderings.
Iris reports a declared four-Welch-test Holm family with raw-unit effects and marginal
intervals (versicolor-minus-virginica petal length -1.292 cm), pooled/within-species
correlations, rounded-data normal QQ diagnostics and an explicit fixed 4.5 cm count
threshold. It makes no causal/confirmatory claim and does not interpret a fitted-normal
KS p-value or call an observed zero cell structurally impossible. Legacy Mepml/Org/TeX
copies were removed, and the conversion checklist and algorithm index point to the new
report.

## Kernel-regression report conversion (2026-10-28)

Completed `include/datamunge/stats/kernel_regression.hpp` and
`src/datamunge/stats/kernel_regression.cpp`: 95/95 definitions documented, zero missing.
Both paths join the completed manifest; the completed-path gate passes 7526/7526
current definitions with zero missing. Baseline coverage is unchanged.

`KernelRegression` now exposes ordinary versus conditional deletion fits, original retained
rows, frozen population scaling, query weights, the smoother matrix, training influence,
bandwidth coverage/objectives and external scores with a training-mean baseline. Explicit
bandwidth candidates preserve caller order and first ties. Gaussian relative log weights
retain predictions when all unnormalized exponentials would underflow; direct deletions
exclude observations before normalization. Automatic compact-kernel tuning requires all
validation predictions, rather than silently selecting on an easier supported subset.
Undefined deletion R-squared remains NaN; missing query rows retain their alignment and
status. Formula plotting accepts only a direct numeric predictor, and compact-support gaps
remain separate curve runs. The options formatter includes the new candidate grid.

Validation:

- 25 native C++23 tests pass: 12 existing kernel-regression tests and 13 new diagnostics
  tests. Two affected formatter tests also pass. They check exact three-point weights and
  deletion identities, boundary bias/convex hull, duplicates, Gaussian far-field queries
  through 1e100, compact endpoints, incomplete CV rejection, caller-order ties, population
  SD and tiny-unit rescaling, null row alignment/schema/levels, held-out coverage and a
  training-only mean baseline, constant-response R-squared and invalid fitting/plot controls.
- An independent NumPy oracle checks 17 fitted cases and 557057 finite values: four radial
  kernels, fixed/automatic bandwidths, one/two predictors, full-data and held-out iris,
  smoothing/query weights, ordinary/deletion predictions, local concentration, candidate
  coverage/objectives and external scores. Maximum normalized-weight error is 1.4e-17,
  prediction error 8.9e-16, effective-sample-size error 7.2e-15, CV-MSE error 2.3e-16 and
  held-out RMSE/MAE error 5.6e-17. Selected bandwidths and coverage match. statsmodels is
  unavailable; no cross-package agreement is claimed.
- All 17 header code examples compile and run. Five clangd definition-hover checks pass
  for the model class, candidate option, bandwidth diagnostics, score and query weights.
  The mep graphical K integration was not separately verified.
- `docs/mepml/examples/kernel_regression_iris_analysis.mepml` executes all seven programs.
  All seven SVGs were rendered with Inkscape and visually inspected at 1100 px, including
  fitted multi-panel layouts, legends and axis labels. Markdown and Doxygen exports pass;
  the latter retains seven C++ programs, seven copied SVGs, five display formulas and
  rendered reference links.

The rewritten report derives the local weighted-constant solution and conditional
smoother/variance/deletion identities before exact three-point and affine-boundary examples.
A duplicate-window counterexample has covered deletion MSE 4 but incomplete coverage;
a broader candidate covers all deletions and is selected despite its higher loss. A
known sinusoid separates apparent, conditional deletion and known-function errors. Iris
then distinguishes full-data smoothing from a prespecified species-stratified 120/30
split: the primary Gaussian width-only test RMSE is 0.456914 cm, versus the training-mean
baseline 1.753375 cm, with all 30 test rows covered. Kernel and two-predictor sensitivity
comparisons are not used for test-set selection. Extrapolation tables distinguish defined
Gaussian predictions from proximity to data and compact-kernel refusal. Legacy
Mepml/Org/TeX copies were removed; the conversion checklist and algorithm index point to
the replacement report.

## Kernel-PCA report conversion (2026-10-28)

Completed `include/datamunge/stats/kernel_pca.hpp` and
`src/datamunge/stats/kernel_pca.cpp`: 61/61 definitions documented, zero missing.
Both join the completed-path manifest. A fresh inventory and completed-path gate pass
7660/7660 current definitions, with zero missing; the initial baseline is unchanged.

`KernelPCA` now exposes raw and centred kernels, normalized dual coefficients, full
signed eigenvalues, numerical rank, spectrum/feature-energy diagnostics and frozen-training
`transform()`/`projection_details()`. Retained `eigenvalues()` keep their Gram-scale
meaning; sample variances divide by n-1. Query batches do not learn new centres, and
invalid predictor rows keep their positions with NaN coordinates and explicit statuses.
The scaled symmetric eigensolver replaces the unscaled Jacobi route; a relative rank
threshold prevents roundoff from masquerading as a component. RBF centring uses expm1,
retaining small-gamma structure even when raw kernel entries round to one. Linear
centering uses centred predictors, preserving small-unit/translated geometry. Active
parameters and retained inputs must be finite; polynomial kernels require a positive
integer degree and nonnegative offset instead of silently accepting indefinite/nonreal
feature maps. Materially negative spectra and insufficient resolved rank are rejected.

Validation:

- 17 native C++23 tests pass: six existing kernel-PCA tests and 11 new diagnostics tests.
  The affected unsupervised-model formatter test also passes. Checks cover exact linear
  centering/normalization and query projection, explicit quadratic features/eigenvalues,
  three-point RBF closed forms, gamma 1e-100 with a rounded all-ones raw kernel, tiny
  predictor units, large translations, coordinated unit/gamma rescaling, duplicate rank,
  training reprojection and spectral residual energy, query-batch invariance, row alignment,
  null/nonfinite status, empty queries and PSD/schema/overflow guards.
- An independent NumPy oracle checks 25 fitted cases and 1237269 finite values, including
  linear/quadratic/cubic kernels, RBF scales, symmetric ring eigenspaces, full iris and
  three frozen-training held-out kernels. It checks raw/centred kernels, all eigenvalues,
  normalization, energy shares/tails, score Gram matrices, query-training inner products
  and query projection norms/losses. Maximum centred-kernel error is 5.7e-14, eigenvalue
  error 7.3e-12, score-Gram error 7.9e-12 and projection-loss error 4.6e-11 (the largest
  absolute errors occur in polynomial feature units). Whole repeated eigenspaces are
  compared without imposing a coordinate basis. Twenty-three cases also agree with
  scikit-learn 1.8.0: eigenvalue error <=7.3e-12, score-Gram error <=6.7e-12 and
  query-training inner-product error <=5.9e-12. The two extremely small-gamma cases use
  the independent stable NumPy formula rather than scikit-learn's rounded raw kernel.
- All ten header code examples compile and run. Five clangd definition-hover checks pass
  for the model class, polynomial degree, spectrum, frozen transform and projection loss.
  The mep graphical K integration was not separately verified.
- `docs/mepml/examples/kernel_pca_iris_analysis.mepml` executes all eight programs. All
  eight SVGs were rendered with Inkscape and visually inspected at 1100 px; iris panels
  were rerun with legend headroom, and long headings shortened. Markdown and Doxygen
  exports pass, retaining eight C++ programs, eight copied SVGs, five display formulas
  and rendered reference links.

The report begins with exact three-point linear/Gaussian and four-point quadratic maps,
then checks scaled linear and equal-eigenvalue RBF limits and eight symmetric ring points.
It distinguishes nonlinear information from principal-axis ordering and explains rotations
within repeated eigenspaces. Iris confirms the unscaled-PCA identity (two-score Gram
Frobenius discrepancy 1.494e-12), then describes gamma sensitivity in a declared raw-unit
metric without species-based selection. At gamma .13 two axes retain 83.45% of resolved
feature energy. A prespecified 120/30 split freezes the training kernel centre and axes;
mean training/test feature-space projection losses are .082928/.080447. These are
representation diagnostics, not original-unit prediction errors or classifier accuracy.
The old Mepml/Org copies were removed; there were no matching generated legacy figures
or TeX files. The conversion checklist and algorithm index point to the new report.


## Hamming-distance report conversion (2026-10-28)

Completed the new `include/datamunge/algorithms/hamming_distance.hpp` and
`src/datamunge/algorithms/hamming_distance.cpp`, now in the verified-path manifest.
The existing `hamming_distance()` byte contract in `string_metrics.hpp` and file-purpose
comments in that header/implementation were refreshed. The four-path audit passed 20/20
definitions; the completed-path gate passed 7725/7725 definitions. The shared string-metric
files were not added to the completion manifest: their other public APIs still need the
full example/reference review. The initial coverage baseline is unchanged.

Reusable helpers expose packed-bit XOR/popcount, per-byte mismatch/cumulative tables and
plots, distance matrices, finite-code minimum separation, all nearest-codeword ties and
q-ary shell/ball counts. Arbitrary bytes and embedded NULs retain their meaning; packed
bits and UTF-8 characters are not silently substituted for bytes. Code-membership detection
and hard nearest-word correction are explicitly distinguished.

Validation: all 22 C++23 tests pass (14 existing string-metric tests and eight new Hamming
geometry tests). Checks include all 65536 pairs of one-byte values, packed/ASCII-bit
agreement, all triples in the three-bit cube, complete binary/four-symbol shell enumeration,
repetition-code correction, parity-code ties and invalid widths/codebooks/ranges. All eight
header examples compile and run, and five clangd hovers pass. The mep graphical K integration
was not separately checked. An independent Python combinatorial calculation also verifies
72 displayed shell/ball/probability values; probability differences are only six-decimal
printing errors (at most 5e-7).

The report executes five complete C++ programs and one local TikZ rendering block, producing
six standalone SVG files. All six were rendered with Inkscape at 1100 px and visually
reviewed; the TikZ diagram was arranged side by side with bounding-box padding, and long
shell-plot legend labels shortened. Markdown/Doxygen exports retain five C++ programs,
one shell build, six copied SVGs and four display formulas. TikZ source is retained in
`docs/mepml/examples/diagrams/hamming_distance_concept.tikz`; LaTeX/dvisvgm save the diagram,
intermediates and logs under local `build/`, without SVG XML in output. Legacy Mepml/Org
copies were removed, and the checklist/index updated. The user's TikZ and local-image-file
preferences are recorded in AGENTS.md, the conversion plan and the reports README.

This validation record was completed after the earlier disk-space interruption was resolved.


## SGD/momentum NN report (2026-10-28)

Completed `nn/optimizer_analysis.hpp/.cpp` (10/10 definitions documented), adding
actual tensor gradient-stream/quadratic traces and analytic fixed-mode spectral radii.
Five new native tests and ten existing fused-optimizer tests pass; they cover independent
scalar recurrences, a closed-form quadratic trajectory, stability boundaries, state/rate
conventions and invalid inputs. All three header examples compile/run and three clangd
hover checks pass. The report executes four C++ programs and one TikZ build, with five
local SVG figures and two display formulas. Figures were rendered using Inkscape and
reviewed; legible colours and scientific loss notation replace hard-to-read pale curves
and rounded-to-zero residuals. Markdown/Doxygen exports pass. Original runnable C++
examples remain as required by the NN conversion plan. Graphical mep K was not tested.

The report separates prescribed gradients, full-gradient quadratic modes and an explicit
sampled minibatch loop. It states the TensorFlow reparameterized Nesterov convention,
shows differing stability intervals and avoids claiming that one optimizer universally
wins. TikZ sources and intermediate/log files remain local and no SVG XML is printed.


## Early-stopping/checkpoint NN report (2026-10-28)

The new report executes three C++ programs and one TikZ build, producing four local SVGs.
Sixteen existing checkpoint/training/schedule/metric tests pass. A strict patience/min_delta
stream, a closed-form scalar SGD trajectory and best-state restoration are checked; full
Adam resume after 17+23 updates reproduces every serialized optimizer value and weight
exactly, while weights-only restoration has a different counter and final weight.
All four figures were rendered using Inkscape and reviewed, with long labels shortened;
Markdown and Doxygen exports preserve the programs, images and displayed stopping rule.
No core API change was needed. Original examples remain. The report states the compatible
configuration, external RNG/model-state and monitor-state requirements for exact resume,
and separates validation selection from untouched-test evaluation.


## Gradient clipping accumulation NN report (2026-10-28)

Three C++ programs and one local TikZ build execute, with four Inkscape-rendered and reviewed SVGs and two display formulas. Markdown/Doxygen exports pass. The (3,4) norm/value geometry, sample-weighted 2/3/2 partition and nonlinear clipping-order counterexample match exact values; the existing weighted-accumulator native regression test passes. The report distinguishes aggregate clipping, per-example privacy clipping and adaptive/momentum updates. No core change was needed; original examples remain.


## Learning rate schedules NN report (2026-10-28)

Three C++ programs and a local TikZ build execute, with four rendered/reviewed SVGs and two display formulas. Markdown/Doxygen exports pass. Seven schedule families have explicit endpoints; scheduled scalar SGD matches its independent product reference after every update; the zero-rate momentum example distinguishes continuing state updates from omitting calls. The four existing native schedule tests already passed in the training-component run. No core change was needed. The report states its completed-update clock, clamped tails and rate-only one-cycle convention; original examples remain.


## Metrics NN report (2026-10-28)

Three C++ programs and a TikZ build execute, with four local SVGs rendered using Inkscape and reviewed. Markdown/Doxygen exports preserve all programs/figures and one display formula. Exact confusion counts, absent-class macro averaging, tied-score AUC=.875/AP=5/6, weighted batch means and strictly-higher-score top-k ties are exposed; the three existing native metric tests passed in the training-component run. No core change was required. Fixed constructed inputs are not described as a classifier-validation study; original examples remain.


## Mixed precision NN report (2026-10-28)

Ported reusable storage-emulation and host loss-scaling logic into the native library;
the old common header forwards without retaining algorithm implementations. All 23
changed-path definitions are documented. Eight header examples compile/run and four
clangd hovers pass. Five native tests pass, including every finite binary16 encoding,
ties/subnormals, signed zero, gradient underflow recovery, independent unscaled buffers,
validation, and unchanged nontrivial Adam state on an overflow skip. The original
mixed-precision example still compiles/runs through the compatibility forwards.

Three report C++ programs and one retained local TikZ build execute. Four SVGs were
rendered with Inkscape and reviewed, with rescaled gradient units and log10 MSE so
small values remain legible. Markdown/Doxygen exports preserve programs, figures and
one display formula. The report explicitly limits claims to CPU Float32 arithmetic
with rounded storage/VJP emulation, and explains the surrogate gradient, finite-check
control flow, and separate host-policy history needed for resume. Original examples
remain; graphical mep K was not tested.


## Training loop NN report (2026-10-28)

Three complete C++ programs and one retained TikZ build execute, producing four local
SVGs rendered through Inkscape and visually reviewed. Markdown/Doxygen exports preserve
all programs, figures and the weighted-mean formula. The fixed eleven-row example
checks short-batch aggregation and dropped-row coverage; the 103-row regression visits
every row once per epoch, takes 350 Adam updates and recovers the affine coefficients
with final MSE about 1.15e-13. Online and fixed-weight evaluation losses are recorded
separately. A local sampler snapshot reproduces forty row choices across the remaining
seven rows and three future epochs, with invalid-state rejection leaving state unchanged.
Six native training tests and the original runnable training-loop example pass. No core
API change was needed. The report explains that sampler-only resume is not a complete
training checkpoint and that shuffle sequences need not be portable across toolchains.


## Optimizer/training-loop section completion (2026-10-28)

All nine NN section entries are checked; seven new reports were completed sequentially
(SGD/momentum, early stopping/checkpoints, clipping/accumulation, schedules, metrics,
mixed precision and training loop). Their 22 complete C++ programs and seven retained
TikZ builds execute and produce 29 local SVG figures. Final Markdown/Doxygen export
checks preserve every program, figure and displayed formula. Report source contains
no inline SVG XML or image data URIs; referenced figures exist and are not ignored.
C++ snippets were formatted for readability after verification. Final fresh Doxygen
XML passes the completed-path gate at 7758/7758 definitions, zero missing. Original
runnable C++ examples remain.

### Foundation report helpers (2026-10-29)

The twelve new `include/datamunge/nn/` headers `artifacts`, `binary`, `cifar`,
`csv`, `tokenization`, `tfrecord`, `linear`, `polynomial`, `validation`, `glm`,
`gaussian_mixture` and `naive_bayes` pass the definition-presence audit for all
175 definitions. Public comments state shapes/dtypes, fitting boundaries,
mutation/lifetimes, errors, numerical conventions and caller preconditions.
Legacy example headers forward to these canonical definitions.

Validation: 22 focused gtests; all 67 header examples compiled and executed;
clangd hovers for FeatureColumns, restore_tensors and GlmNll; eleven affected
original programs compiled and executed with C++23. Header-only workflows need
no generated binding changes. Coverage/hover/example artifacts are under
`build/debug/docs/foundations-coverage.json` and `build/foundations-validation/`.
The 29 new literate reports and their retained figures are tracked in
`docs/mepml/nn-conversion-plan.md`; this does not claim a full MNIST benchmark
or graphical mep K verification. The checked header paths are in the completed
manifest.

### Classical tensor inference for executable reports (2026-10-29)

- [x] `include/datamunge/nn/classical_inference.hpp`: 10/10 definitions documented.
  Contracts cover log-potential shapes/dtypes, differentiable HMM forward evidence,
  stable neighbour selection, exact-match weights, class ties and gathered target
  gradients. Four focused native tests and three compiled header examples pass;
  clangd hovers for HMM evidence and the neighbour class carry their descriptions.
  Evidence lives in `build/classical-tensor-coverage.json` and
  `build/classical-tensor-hover.json`. The accompanying 15 classical ML reports
  now expose tensor optimizer loops instead of hidden model fitting.

### CNN fundamentals components (2026-10-29)

- [x] `nn/convolution_components.hpp`, `nn/cnn_models.hpp`, `nn/vision_plots.hpp`:
  34/34 documented definitions, 15 compiled and executed examples, two clangd
  hovers, seven focused native tests. Contracts cover patch order/padding/gradients,
  composition rank constraints, sparse C3 and fixed-prototype adaptations, plus
  finite heatmap values and RGB/grayscale preview conversion. Evidence is saved
  in `build/cnn-foundations-coverage.json` and `build/cnn-foundations-hover.json`.

### NiN and VGG tensor model components (2026-10-29)

- [x] `include/datamunge/nn/classic_cnn_models.hpp`: 31/31 documented definitions,
  ten compiled/executed examples, two clangd hovers. Ten native CNN tests now
  include explicit NiN patch-MLP gradients, VGG sliding-head equivalence and
  invalid configurations. Legacy NiN/VGG sources compile through forwarding
  headers. The reports expose all optimizer loops and distinguish full shape
  arithmetic from instantiated compact models and their measured subset scores.
  Evidence: `build/cnn-classic-coverage.json`, `build/cnn-classic-hover.json`.

### AlexNet, GoogLeNet and shared branch components (2026-10-29)

- [x] `nn/classic_cnn_components.hpp`, `nn/alexnet_googlenet.hpp`: 54/54
  documented definitions, 23 compiled/executed code examples and four clangd
  hovers. Five native tests cover LRN derivatives, grouped value/gradient
  equivalence and isolation, Inception ordering, reference configuration
  geometry, dropout and auxiliary independence. Both source examples still
  compile via compatibility headers. Reports expose the tensor optimization
  loops. Evidence: `build/cnn-alex-google-coverage.json` and
  `build/cnn-alex-google-hover.json`.

### ZFNet, Xception and normalised/adjoint primitives (2026-10-29)

- [x] `nn/vision_architecture_components.hpp`, `nn/zfnet_xception.hpp`: 43/43
  documented definitions, 16 compiled/executed examples, four clangd hovers.
  Four added native tests cover switch routing, backward-sign gating, exact
  input-grid reconstruction, signed separable ordering and BN-state separation.
  Nine architecture tests now pass alongside the ten earlier CNN tests. Both
  original source examples compile through compatibility headers. The reports
  retain full topology while showing bounded dataset fits and explicit state
  checkpoints. Evidence: `build/cnn-zf-xception-coverage.json` and
  `build/cnn-zf-xception-hover.json`.


## Clustering Iris report and diagnostics (2026-10-29)

Converted `clustering_iris_analysis` into seven complete, formatted C++ programs and
one retained local TikZ build, producing eight SVG figures. Each figure was rendered
using Inkscape and reviewed; the transparent concept diagram was additionally reviewed
on a white background. Markdown/Doxygen exports preserve seven programs, the TikZ
build, all eight images and four displayed formulas. No SVG XML/data URI is printed.
Old mepml/org/TeX sources are removed, all eight algorithm-list references point to
the new report, and the conversion checklist is checked.

Added `stats/clustering_diagnostics.hpp` and its implementation, registered in the
umbrella and build source list. `ClusterAgreement`, `cluster_agreement`,
`cluster_composition`, `plot_clusters` and `plot_dendrogram` replace reusable report
logic. Contracts specify noise (-1), alignment, arbitrary ids, deterministic majority
ties, all-noise NaN purity, all-row ARI with noise as one category, leaf traversal order
and Ward's height-squared/2 SSE convention. Plot validation handles nonfinite inputs,
constant coordinates and unrepresentable ranges. All 17 changed-path definitions are
documented; four header examples compile/run and five clangd hovers pass. Nineteen
native tests pass, including existing clustering tests and seven diagnostic/geometry
regressions. Graphical mep K was not tested.

Independent NumPy/SciPy/scikit-learn checks use the library's exported Iris values:
eight WCSS calculations and twenty agreement/coverage summaries match; the k=3 K-means
partition, all five hierarchical partitions/cuts, and 596 merge heights agree with
independent implementations. All seven DBSCAN partitions agree when given the same
direct-coordinate distance matrix. Independently computing distances changes 28
undirected neighbourhood edges at radius .3 by roundoff near the threshold, changing
that partition despite an equal noise count; the report explains this numerical
sensitivity rather than claiming unconditional cross-implementation equality.

The final fresh completed-path documentation gate passes at 8166/8166 definitions,
zero missing. Species is used only for post-fit descriptions, with raw-feature metric,
predeclared settings, restart-prefix interpretation, noise coverage and purity's
fragmentation bias stated explicitly.

### Full Inception model components (2026-10-29)

- [x] `nn/inception_models.hpp`: 47/47 documented definitions, 26
  compiled/executed examples and four clangd hovers. Three added native tests
  cover beta-only rectangular BN/EMA, residual value/parameter/input scaling,
  and full compact depths/auxiliary heads across all four variants. All 22 CNN
  component tests pass. The combined nine-header inventory is 207/207 with no
  missing descriptions. The original Inception sources still compile through
  compatibility aliases. Reports keep optimization and state checkpointing
  explicit. Evidence: `build/cnn-inception-coverage.json`,
  `build/cnn-inception-hover.json`, `build/cnn-combined-coverage.json`.

### Basic and bottleneck ResNet components (2026-10-29)

- [x] `nn/resnet_models.hpp`: 32/32 documented definitions, ten
  compiled/executed examples, two clangd hovers. Four focused native tests
  cover signed gates/projections, stride placement, complete supported depth
  schedules, feature-only extraction and invalid inputs. All 26 CNN tests pass.
  Twelve original residual/dense sources compile through the forwarding header.
  Combined CNN documentation coverage is 239/239. Reports preserve complete
  depths and expose all optimizer/state-checkpoint code. Evidence:
  `build/cnn-resnet-coverage.json`, `build/cnn-resnet-hover.json` and
  `build/cnn-combined-coverage.json`.


## PCA Iris report and inspectable reconstruction (2026-10-29)

Converted `pca_iris_analysis` into six complete, formatted C++ programs plus a retained
TikZ projection diagram, producing seven local SVG figures. The four-point example
checks axes/eigenvalues and the training-tail SSE identity; a feature-unit experiment
and a single-query counterexample distinguish standardized from original-unit loss.
A two-row/three-feature example separates a full returned basis from numerical rank.
Iris supplies fitted preprocessing/loadings/spectra, raw-versus-scaled reconstruction,
and a fixed 120/30-row split whose singleton and batched query projections agree exactly.
All seven figures were rendered through Inkscape and visually reviewed; diagram labels
and geometry-plot scaling were adjusted for legibility and perpendicular projection.
Markdown/Doxygen exports retain all six programs, the diagram build, seven images and
three displayed formulas. No inline SVG XML/data URI is emitted. Old mepml/org/TeX
sources are removed, the conversion plan is checked and an algorithm-list entry links
to the new report.

PCA now exposes fitted means/scales, the actual covariance/second-moment matrix and
numerical Gram rank, with spectrum/loadings tables, original-unit reconstruction and
measured error curves. Reconstruction references actual supplied entries and does not
refit query preprocessing. Full p-axis output is preserved, including null directions;
constant-column handling and uncentred RMS/second-moment semantics are explicit.
Selected schema must be numeric/distinct, retained/query values finite, and arithmetic
representable. Extended accumulation and a normalized covariance eigensolve address
underflow/overflow of the original eigensolver's norm, while documenting the Gram
method's conditioning limits. Sample normalization and the original public methods
remain. The incorrect old claim of min(n-1,p) nontrivial returned axes is corrected.

All 51 changed-path definitions are documented. Seventeen header examples compile/run
and six clangd hover checks pass. Fourteen native tests pass, covering the seven existing
PCA regressions and seven new diagnostics/numerical contracts. Independent NumPy
SVD/eigh and scikit-learn checks cover 23 cases and 4736 values, including 55 identifiable
prefix projectors and eleven scikit-learn comparisons. Cases include uncentred options,
constant and wide inputs, repeated eigenvalues, near-dependence, Iris fits, query
reconstruction and unscaled amplitudes 1e-100/1e100 (covariances whose squared norms
would underflow/overflow without normalization). Repeated eigenspaces are compared
without asserting a preferred individual eigenvector.

The fresh completed-path gate passes at 8297/8297 definitions, zero missing. Graphical
mep K was not tested. The report states that variance is metric-dependent representation
energy, and does not infer predictive accuracy, biological importance or an optimal k
from its scores or one deterministic holdout split.

### Pre-activation and DenseNet components (2026-10-29)

- [x] `nn/preactivation_densenet_models.hpp`: 48/48 documented definitions,
  17 compiled/executed examples and four clangd hovers. Five focused tests cover
  signed identity/projection derivatives, finite differences, dense feature
  ordering, transition arithmetic, native model counts and invalid grouping or
  compression. All 31 CNN tests pass. Twelve original residual/dense sources
  compile through compatibility headers. Shared residual-output documentation
  now distinguishes post-activation and preactivation sums. Combined coverage:
  287/287 across eleven CNN headers. Reports expose all tensor fitting and state
  checkpoints; their PNG input sheets export alongside SVG plots. Evidence:
  `build/cnn-preact-dense-coverage.json`, `build/cnn-preact-dense-hover.json`,
  `build/cnn-combined-coverage.json`.


## FFT report, spectrum normalization and convolution (2026-10-29)

Converted `fft_analysis` into six complete C++ programs and a retained TikZ
butterfly build, producing seven local SVGs. Four-point complex coefficients,
eight-point DC/interior/Nyquist amplitudes, off-bin leakage, periodic-Hann weights,
zero padding and linear/circular convolution precede a complex-input length sweep
and the original 256-sample signal. The latter's extra term is correctly identified
as deterministic sinusoids, not random noise. Operation counts are distinguished
from timing measurements; finite numerical agreement is distinguished from exact
arithmetic. All seven figures were rendered with Inkscape at 1100 px and visually
reviewed. Markdown and Doxygen exports preserve the seven executable blocks and
images plus five displayed formulas. Results emit no SVG XML or data URI. The old
mepml/org sources are removed and both algorithm-list links point to the new report.

Native APIs now provide an extended-accumulation direct DFT, full-complex forward/
inverse and Parseval diagnostics, calibrated one-sided spectra and full linear FFT
convolution. Contracts specify radix-2 lengths, owned outputs, unchanged inputs,
finite-value validation, periodic versus symmetric Hann, undoubled DC/Nyquist,
observed-window normalization excluding padding, and per-bin mean square versus
per-Hz density. The low-level FFT retains its original forward/inverse conventions,
empty/singleton behavior and full real-input output. Portable pi and a final-stage
length-overflow guard replace its platform constant and unchecked final shift.

All 18 changed-path definitions are documented, all seven header examples compile
and run, and five clangd hover checks pass. Ten native tests pass, including the
existing FFT/Goertzel regressions. Independent NumPy/SciPy comparisons cover 44
cases and 8177 values: complex forward/inverse transforms, arbitrary-length direct
DFTs, fifteen window/padding spectra with integrated mean square, and sixteen
full convolutions checked against both direct and FFT SciPy/NumPy methods. Largest
absolute discrepancy is 2.620e-13. Evidence is retained in `build/fft-verification/`.
The fresh completed-path documentation gate passes at 8328/8328 definitions, zero
missing. Graphical mep K was not tested.

### Coupled-gate Highway model components (2026-10-29)

- [x] `nn/highway_models.hpp`: 13/13 documented definitions, six
  compiled/executed examples and two clangd hovers. Four focused native tests
  cover the complete gate Jacobian, pointwise equivalence, scalar attenuation,
  classifier count/gradients, learned-bias reuse and invalid dimensions.
  All 35 CNN tests pass; the original source compiles through aliases.
  Combined coverage is 300/300 across twelve headers. The report exposes the
  MNIST optimizer loop and shows original image pixels in a PNG contact sheet.
  Evidence: `build/cnn-highway-coverage.json`, `build/cnn-highway-hover.json`,
  `build/cnn-combined-coverage.json`.


## Factor analysis report and explicit boundary evidence (2026-10-29)

Converted `factor_analysis_analysis` to five complete C++ programs and a retained
TikZ common-factor diagram, producing six local SVGs. An exact eight-row/three-
indicator design gives unit variances, correlations .5, loadings sqrt(.5), uniqueness
.5 and a scalar communality recurrence with error reduced by three each step.
Uniform unit scaling checks covariance-mode variance units; correlated generating
factors demonstrate orthogonal rotation invariance. Nine known-model samples compare
rotation-invariant population recovery and same-sample PAF/PCA covariance residuals.
Iris supplies a real-data counterexample: one factor stops after 22 iterations but
has raw petal-length uniqueness -.0324913; two factors exhaust 1000 iterations with
negative parameter-count df and an improper loading solution. The text distinguishes
stopping, identification counts, covariance fit and admissibility. It corrects the old
claims that positive df establishes identification or that clamping produces a clean
solution. No causal-factor, general PAF optimality or likelihood-test claim is made.

Native FactorAnalysis now exposes observed/common/reproduced covariance, fitted
means/scales, named loading and variance tables, covariance pairs, extraction history
and numerical/parameter-count diagnostics. Raw uniqueness and scale-relative Heywood
flags accompany the existing nonnegative uniqueness output. Covariance-mode SMCs now
have the correct variance units, and variance shares use the observed covariance trace
rather than a sum inflated by clamped uniquenesses. Validation rejects duplicate,
missing/nonnumeric schemas, invalid stopping controls, retained nonfinite values,
zero-variance predictors and unrepresentable arithmetic. Null rows remain dropped as
units. Eigenvalue flooring is relative to maximum observed variance and explicitly
identified as regularization, not a pseudoinverse. Symmetric eigenproblems are scaled
before Jacobi. A seeded local Gaussian factor-model generator and off-diagonal
covariance residual helper replace report-local reusable algorithms. Loading plots
separate nearby labels and draw a unit circle only in standardized units.

All 66 changed-path definitions are documented. Sixteen header examples compile and
run, six clangd hovers pass, and all fourteen native tests pass (seven original and
seven new). Independent NumPy eigendecomposition/PAF calculations and SciPy scalar
rotation optimization pass on 44 cases and 102342 values, including fourteen two-
factor angle optima. Cases include exact one-step/converged fits, standardized and
covariance units, heterogeneous feature scales, singular inputs, known independent/
correlated models and Iris. Maximum common-covariance and iteration discrepancies
are 1.553e-11 and 1.567e-11; score discrepancies reach 1.552e-8 with the documented
regularized inverse. Independent angle objectives agree within 4.794e-13.
Evidence is retained in `build/factor-analysis-verification/`.

All six figures were rendered at 1100 px with Inkscape and visually reviewed; panel
headings and the recovery scale were refined. Markdown/Doxygen exports preserve
all six blocks, six local SVGs and six displayed formulas, with no SVG XML/data URI
in results. Old mepml/org sources are removed, the conversion checklist is complete
for this report and a statistics algorithm entry links to the replacement. The fresh
completed-path documentation gate passes at 8434/8434 definitions, zero missing.
Graphical mep K was not tested.


### Dual-path tensor model components (2026-10-29)

- [x] `nn/dual_path_models.hpp`: 40/40 documented definitions, thirteen
  compiled/executed examples, two clangd hovers and four native tests.
  Tests cover ordinary residual/dense preservation, projected path values and
  derivatives, all layouts, complete compact geometry/gradients and invalid
  configurations. The original example compiles through forwarding aliases.
  Combined CNN coverage is 340/340 across thirteen headers. The report exposes
  its Adam loop and original CIFAR pixels; Markdown/Doxygen exports pass.
  Evidence: `build/cnn-dpn-coverage.json`, `build/cnn-dpn-hover.json`,
  `build/cnn-combined-coverage.json`.


### Grouped and wide residual tensor models (2026-10-29)

- [x] `nn/grouped_wide_residual_models.hpp`: 40/40 definitions documented,
  eleven public examples compiled/executed and four clangd hovers passed.
  Six native tests cover aggregation values/all derivatives, dropout mask
  values/derivatives and signed identity, full compact topology/counts,
  moving-state updates and invalid layouts. Original ResNeXt/Wide ResNet
  examples compile through aliases. Both reports expose optimizer loops and
  original CIFAR images; Markdown/Doxygen exports include all local figures.
  Combined CNN coverage is 380/380 across fourteen headers. Evidence:
  `build/cnn-grouped-wide-coverage.json`, `build/cnn-grouped-wide-hover.json`,
  `build/cnn-combined-coverage.json`.


### Pyramidal and internal-scale tensor models (2026-10-29)

- [x] `nn/pyramidal_multiscale_models.hpp`: 37/37 documented definitions,
  thirteen compiled/executed examples, four passing clangd hovers and six
  native tests. Checks cover rounded schedules, signed pooled/zero-padded
  values/derivatives, invalid geometry, hierarchical supports and masses,
  complete compact topology/gradients and moving-state semantics. Original
  PyramidNet/Res2Net examples compile through compatibility aliases.
  Both reports execute their Adam loops and export all figures, including PNGs.
  Combined CNN coverage is 417/417 across fifteen headers. Evidence:
  `build/cnn-pyramid-scale-coverage.json`, `build/cnn-pyramid-scale-hover.json`,
  `build/cnn-combined-coverage.json`.


### Stochastic residual tensor models (2026-10-29)

- [x] `nn/stochastic_residual_models.hpp`: 25/25 documented definitions,
  nine compiled/executed public examples and two passing clangd hovers.
  Five native tests check original/inverted scaling, zero/one survival,
  row/batch mask values and derivatives, schedules, option-A shortcuts,
  complete CIFAR-20 gradients/state, generator consumption and invalid inputs.
  The original source compiles through aliases. The report executes both Adam
  fits and exports its figures, including original CIFAR PNGs. Combined CNN
  coverage is 442/442 across sixteen headers. Evidence:
  `build/cnn-stochastic-coverage.json`, `build/cnn-stochastic-hover.json`,
  `build/cnn-combined-coverage.json`.


### Reversible residual tensor models (2026-10-29)

- [x] `nn/reversible_residual_models.hpp`: 50/50 documented definitions,
  nineteen compiled/executed examples, two passing clangd hovers and five
  native tests. Tests cover nontrivial inverses, Float64 full-model gradient
  equality across three backward modes, BN updates exactly once, complete
  released CIFAR count/geometry and invalid modes/layouts. The original source
  compiles through aliases. The report executes reversible Adam fitting and
  measures actual tensor-buffer peaks, separately reporting Float32 drift.
  Markdown/Doxygen exports include all figures. Combined CNN coverage is
  492/492 across seventeen headers. Evidence: `build/cnn-revnet-coverage.json`,
  `build/cnn-revnet-hover.json`, `build/cnn-combined-coverage.json`.


### MobileNet tensor forward models and arithmetic metadata (2026-10-29)

- [x] `nn/mobilenet_models.hpp`: 70/70 documented definitions, twenty-three
  compiled/executed public examples, four passing clangd hovers and seven
  native tests. Tests check dense/depthwise normalization and logged MACs,
  inverted signed identities, hard activations, the input-dependent SE gate
  derivative, all four released model budgets, full CIFAR gradients/state and
  invalid widths/metadata. Original v1/v2/v3 sources compile through aliases.
  All three reports execute visible Adam loops and export local figures,
  including PNG previews. Combined CNN coverage is 562/562 across eighteen
  headers. Evidence: `build/cnn-mobilenet-coverage.json`,
  `build/cnn-mobilenet-hover.json`, `build/cnn-combined-coverage.json`.


### Fire and SqueezeNet tensor models (2026-10-29)

- [x] `nn/squeezenet_models.hpp`: 22/22 documented definitions, six
  compiled/executed examples, two passing clangd hovers and five native tests.
  Tests cover Fire expand values/all derivatives, negative ceil-pool edges and
  switches, both complete released counts/geometry, CIFAR gradients, dropout
  generator semantics and invalid shapes. The original source compiles through
  aliases. The report explicitly fits three models and exports all local figures.
  Combined CNN coverage is 584/584 across nineteen headers. Evidence:
  `build/cnn-squeezenet-coverage.json`, `build/cnn-squeezenet-hover.json`,
  `build/cnn-combined-coverage.json`.


### ShuffleNet permutations and tensor classifiers (2026-10-29)

- [x] `nn/shufflenet_models.hpp`: 32/32 documented definitions, thirteen
  compiled/executed examples, two passing clangd hovers and six native tests.
  Checks cover exact release/paper permutations, inverse weighted derivatives,
  grouped connectivity, signed even-channel carry, padding-inclusive pooling,
  released count/geometry/MACs, complete CIFAR gradients/state and invalid
  dimensions/groups. The original source compiles through aliases. The report
  executes both Adam fits and exports local figures, including PNG previews.
  Combined CNN coverage is 616/616 across twenty headers. Evidence:
  `build/cnn-shuffle-coverage.json`, `build/cnn-shuffle-hover.json`,
  `build/cnn-combined-coverage.json`.


### GhostNet intrinsic/cheap tensor models (2026-10-29)

- [x] `nn/ghostnet_models.hpp`: 24/24 documented definitions, nine
  compiled/executed examples, two passing clangd hovers and five native tests.
  Tests cover ratio-two/three ceiling and trimming, discarded map gradients,
  signed zero-branch shortcuts, all released width counts/MACs, full CIFAR
  gradients/state and invalid ratios/shapes. The original source compiles
  through aliases. The report executes visible Adam fitting and exports all
  local figures. Combined CNN coverage is 640/640 across twenty-one headers.
  Evidence: `build/cnn-ghost-coverage.json`, `build/cnn-ghost-hover.json`,
  `build/cnn-combined-coverage.json`.


## Elastic-net report, actual coordinate norms and training-only CV preprocessing (2026-10-29)

Converted `elastic_net_iris_analysis` into six complete C++ programs plus retained
TikZ penalty/threshold source, producing seven local SVG figures. Exact four-row
orthogonal examples verify the threshold/denominator, intercept and objective;
measurement-unit changes distinguish raw and penalized coordinates. Duplicate
predictors show unique equal sharing with positive L2 curvature versus nonunique
lasso allocation, while finite sweep budgets expose objective/KKT convergence.
An outlier-containing twelve-row example explicitly refits the exposed folds.
Iris uses a declared 120/30-row split, a four-column formula design including an
interaction, full paths and five declared alpha values. The primary fit has three
nonzero slopes, with 60/60 full-data candidates and 300/300 fold fits converged.
Its reserved-row RMSE .3504 cm exceeds the unpenalized baseline .3403 cm; no
regularization benefit is claimed for this split. Ridge's minimum-grid selection
is identified as a search boundary. Training fit, CV selection, held-out error,
stationarity, sparsity and coefficient identification remain distinct concepts.

Cyclic coordinate descent now divides by actual column squared norm plus L2
curvature. The old unit-norm assumption was invalid both without standardization
and on fold submatrices. Each CV training subset now fits its own predictor and
response offsets and predictor RMS scales. All folds use the declared common
full-training lambda grid and Formula schema; pooled CV MSE weights rows equally.
Every finite CV candidate remains plotted, replacing the old workaround that hid
large errors. Positive/finite option validation, retained-value validation and
extended intermediate arithmetic guard the documented numerical contracts.
Exact zero scales become one. Ridge is documented as having no sparsity threshold,
while permitting exact zeros; positive penalties ensure invertibility rather than
an unconditional small condition number.

Native APIs expose penalty-coordinate and original-unit coefficients, frozen
means/scales, retained-row indices, objective/loss/KKT/change/sweep status, full
path coefficients/diagnostics, fold assignments/results and frozen-query scoring.
The optional path endpoint ratio is forwarded by Ridge/Lasso wrappers; original
constructor signatures and defaults are retained. Numerical exhaustion remains
visible, including when selecting CV candidates, rather than being reported as
convergence. The direct glmnet-style objective is distinguished from the original
2005 rescaled estimator. No selected-coefficient inference is promised.

All 128 changed-path definitions are documented. Twenty header examples compile
and execute, six clangd hovers pass, and all 23 native tests pass (14 existing,
nine new). The original C++ example passes a C++23 syntax compilation. Independent
NumPy ridge/least-squares and scikit-learn fits cover 58 cases, 51498 values and
2290 solver fits, including all candidates and validation folds for the five Iris
models. Cases include raw/normalized predictors, absent intercepts, lambda zero,
duplicate/constant/tiny columns and null-row alignment. Coefficients are compared
using the strongly convex Hessian's curvature and measured stationarity residuals;
near-collinearity amplifies small residuals, so a uniform coefficient tolerance
is not substituted for the documented stopping rule. Maximum path coefficient
and prediction differences are 1.004e-6 and 3.409e-7; maximum fold/pooled CV MSE
differences are 1.029e-7 and 2.716e-8. Every selected lambda agrees independently.
Evidence is retained in `build/elastic-net-verification/`.

All seven SVGs were rendered through Inkscape at 1100 px and visually reviewed;
legend labels/colours and coordinate-score labels were refined. Markdown/Doxygen
exports preserve seven blocks, seven local images and five displayed equations.
No SVG XML/data URI is emitted in results. The old mepml/org/TeX reports are removed,
the conversion checklist is updated and an algorithm-list entry links to the new
report. The original standalone C++ and other language examples remain available.
The fresh completed-path gate passes at 8887/8887 definitions, zero missing.
Graphical mep K was not tested.


### RegNet design metadata and tensor classifiers (2026-10-29)

- [x] `nn/regnet_models.hpp`: 25/25 documented definitions, eight
  compiled/executed examples, two passing clangd hovers and five native tests.
  Tests check all fourteen preset designs under changed rounding modes,
  exception-safe rounding restoration, input-based SE reduction, gated shortcut
  derivatives, full released count/MACs, complete CIFAR gradients/state and
  invalid dimensions/settings. The original source compiles through aliases.
  The report executes visible Adam fitting and exports all local figures,
  distinguishing count-only geometry from target-resolution arithmetic.
  Combined CNN coverage is 665/665 across twenty-two headers. Evidence:
  `build/cnn-regnet-coverage.json`, `build/cnn-regnet-hover.json`,
  `build/cnn-combined-coverage.json`.


## GBM classification report, inspectable residual steps and stable log loss (2026-10-29)

Converted `gbm_iris_analysis` into six complete C++ programs plus retained TikZ
update source, producing seven local SVG figures. Four-row binary and six-row
three-class examples expose priors, simultaneous one-hot-minus-softmax residuals,
leaf means and score/probability updates before the Iris analysis. A noisy example
checks the full-sample projection identity and Hessian-based sufficient descent
bound for 0<eta<4, contrasts half-sample residual fits, and demonstrates a large
learning-rate step increasing training loss. Training accuracy, probability loss,
validation selection and held-out performance are kept distinct.

The canonical classifier now exposes retained/sample original-row indices, log
priors, raw and probability predictions for fixed prefixes, per-row/class residual
and tree updates, aggregate projection diagnostics, staged evaluation, named
confusion tables and importance tables. `gbm_validation_search` selects a rate
and prefix on caller-supplied validation data, including a possible prior-only
choice; it never receives test data. The caller remains responsible for disjoint
partitions. Existing constructors, options order and full-model calls remain.
The algorithm uses direct first-order residual leaf means, without Newton leaf
corrections or a binary single-logit reduction; sklearn classification boosting
is not an equality oracle. Equal-weight averages of per-tree normalized gains
can sum below one because unsplit class trees contribute zeros. A separate
relative-share column normalizes only that display summary.

Stable log-sum-exp replaces the old clipped true-class-probability loss. A noisy
eta=2000 example checks the actual mean loss 1000/6 even though a true-class
probability underflows. Controls, finite encoded predictors/arithmetic, prefix
ranges and grid sizes are checked. Native tests cover exact binary/multiclass
updates, subsampled prefix/refit and sample-sequence agreement, null-row alignment,
unknown response classes, constant-tree importance, prior-only selection/ties,
invalid arguments and the sufficient full-sample loss bound. GBMRegressor is
unchanged and its eight existing tests remain in the 29-test passing native run.

Iris uses petal length and width with a declared 90/30/30 fitting/validation/test
split and a 363-stage grid over three rates and rounds zero through 120. Validation
selects eta=.3 and 106 rounds (mean loss .11568032); the final 120-flower refit
classifies 29/30 reserved flowers correctly, mean loss .21296239 and Brier score
.06703447. One versicolor is classified as virginica. No calibration, causality,
uncertainty or test-driven parameter selection is inferred from this split.

All 83 changed-path definitions are documented; 23 header examples compile and
execute and six representative clangd hovers pass. The original standalone C++
example passes C++23 syntax compilation. Independent NumPy/SciPy checks implement
exhaustive CART and simultaneous residual boosting, independently reproduce the
SplitMix64 sample sequence, and compare 28 cases, 998391 values and 2044 class-tree
fits. Every Iris validation prefix and the final refit agree, including selection;
maximum absolute difference is 1.333e-15. The evidence and check programs are under
`build/gbm-verification/`.

All seven SVGs were rendered with Inkscape at 1100 px and visually reviewed;
headroom and short titles preserve readable legends. Markdown and Doxygen exports
retain six C++ programs, one TikZ build, seven local images and three displayed
equations. No inline SVG XML or data URI appears in results. Old mepml/org/TeX
report sources are removed, the conversion checklist and algorithm list are updated,
and standalone C++/other-language examples are retained. The completed-path gate
passes at 9081/9081 definitions, zero missing. Graphical mep K was not tested.


### CondenseNet tensor pruning and packed inference (2026-10-29)

- [x] `nn/condensenet_models.hpp`: 47/47 documented definitions, thirteen
  compiled/executed examples, two passing hovers and five native tests.
  Production pruning/ranking and kernel packing use tensors; Float64 values
  and every input/filter/BN derivative survive packed grouped conversion.
  Tests cover stable ties, skipped stages, pre-update forward snapshots,
  complete C86 gradients/counts, analytical conversion assumptions and invalid
  masks/configurations. The report exposes Adam/lasso/L2, mask progress and
  all-state validation checkpoints. Original C++23 syntax and both exports pass.
  Evidence: `build/cnn-condense-coverage.json`, `build/cnn-condense-hover.json`.


### MBConv and complete MnasNet/EfficientNet/V2 models (2026-10-29)

- [x] `nn/mbconv_models.hpp`: 64/64 documented definitions, twenty-one
  compiled/executed examples, two passing hovers and five native tests.
  Contracts explain block strings, width/repeat scaling, input-based logistic
  SE, linear versus fused expansion-one projections, row dropout endpoints,
  moving-state mutation and model-specific schedules. Float64 block finite
  differences and dropout derivatives pass; full CIFAR counts/gradients and
  frozen state are checked for all three models. Original C++23 sources and
  sixteen report C++ blocks run; all local figures export to Markdown/Doxygen.
  Evidence: `build/cnn-mbconv-coverage.json`, `build/cnn-mbconv-hover.json`.


### Attention residual backbones and deformable predictors (2026-10-29)

- [x] `nn/attention_cnn_models.hpp`, `nn/deformable_models.hpp`: 55/55
  documented definitions, seventeen compiled/executed examples, three hovers
  and eleven native tests. Float64 SE/CBAM comparisons preserve every gradient;
  ECA has a channel-convolution oracle, SK has normalized weights and exact
  derivative supports, and deformable offsets match independent interpolation
  and all-coordinate finite differences. Complete compact/full counts and
  gradients pass. Grouped/hooked conv2 stride now respects first-1x1 placement,
  with both settings tested. Original sources compile through compatibility
  names; five reports execute twenty-three C++ blocks with visible optimizers
  and export their reviewed figures. Evidence:
  `build/cnn-attention-coverage.json`, `build/cnn-attention-hover.json`.


### ConvNeXt V1/V2 and safe global response normalization (2026-10-29)

- [x] `nn/convnext_models.hpp`: 27/27 documented definitions, twelve
  compiled/executed examples, two hovers and six native tests. Contracts cover
  biased patchify/depthwise filters, LayerNorm/GELU, layer scale, safe zero-norm
  derivatives, active GRN, row dropout and actual rectangular MAC geometry.
  Checked integer analytical counts cover all released sizes without Huge
  allocation and agree with actual Atto/Tiny leaves. Full CIFAR gradients,
  finite differences and identity/support fixtures pass. Both original sources
  compile through aliases; ten report C++ blocks execute visible optimizer
  loops and export reviewed local figures. Evidence:
  `build/cnn-convnext-coverage.json`, `build/cnn-convnext-hover.json`.


## Gauss–Newton report, scaled QR and evaluated-state diagnostics (2026-10-29)

Converted `gauss_newton_analysis` into six complete C++ programs plus retained
TikZ residual/tangent source, producing seven local SVGs. A three-point linear
fit verifies the exact local problem and one-step solution; a scalar square-root
fit distinguishes tangent-predicted residuals from actual nonlinear residuals.
Explicit scalar recurrences check quadratic zero-residual convergence versus a
one-third linear error ratio with nonzero residuals. Another scalar example
raises SSE from .9801 to 600.37250625 after one undamped step, while a zero-start
Jacobian produces an explicit rank error. The original three-parameter generated
sinusoid is retained, with both analytic and central-difference derivative routes.

The canonical header now solves the rectangular least-squares step through the
existing Householder QR implementation, with unit-norm column scaling and an
explicit numerical-rank check. It no longer forms normal equations or silently
skips singular pivots. The original call signature and first three result fields
remain; analytic derivative overloads, local linearization diagnostics, residuals,
completed-update counts, an absolute-step stopping flag and evaluated history are
added. Final SSE and residuals now match the returned parameters, including budget
exhaustion and zero-budget cases. Proposed-step stopping is explicitly distinct
from stationarity or a minimum certificate. The central-difference Jacobian's
computed J^T*r is an approximate gradient. Domain/nonfinite values and malformed
sizes/controls are rejected, and callback exceptions propagate. Full steps remain
unsafeguarded. The report states the explicit-Q backend's O(m*n^2+m^2*n) work and
O(m^2+m*n) temporary storage instead of claiming compact-QR complexity.

Added NIST's observed fourteen-row Misra1a pressure/volume data as a provenance-
linked raw CSV, a deterministic generated embedded source/accessor and catalog/
load entry. The source does not specify physical units and the report preserves
that limitation. Scaled-rate analytic fits from both NIST starts and the scaled
central-difference fit agree with the certified estimates. The start-2 analytic
fit returns A=238.94212917890, b=.00055015643181 and SSE=.12455138894441.
All fourteen observations are used for numerical certification, with no held-out
prediction claim. The distant-start trace retains a negative-amplitude state and
SSE above 27 million. A raw-rate central-difference comparison shows an amplitude
bias about 2.86e-5 from its relatively large perturbation, despite nearly identical
curves and SSE. This accuracy limit is visible rather than described as convergence
to certified parameters for every derivative convention.

All 69 definitions in the five checked paths are documented. Nine new header
examples compile and execute; six clangd hovers pass. Twenty-eight native tests
pass: twelve Gauss–Newton tests (two existing, ten new), thirteen dataset/CSV/cache/
image-loader tests and three existing Davis–Putnam/GRASP tests from the same batch.
Tests cover final-state SSE/counts, exact updates, derivative agreement, overshoot,
error recurrences, disparate parameter units, explicit rank/nonfinite/domain
failures, Misra1a certification and absolute-step stopping without stationarity.
The catalog test includes the new dataset. Generated bindings are not hand-edited.

Independent NumPy/SciPy evidence covers 61 fits, 549 evaluated linearizations and
12939 compared values. Analytic and finite-difference Jacobians are checked
against model derivatives; each recorded rectangular subproblem is independently
solved by SVD after column normalization. Maximum relative residual-scaled step
error is 4.515e-14. Native final curves and objectives are compared with independent
SciPy refinement. Raw-rate central differences have a measured maximum prediction
discrepancy 5.839e-7; a separately recorded analytic stationarity correction verifies
that this is derivative bias, without modifying the native answer or asserting exact
raw-rate parameter agreement. Maximum NIST SSE difference from its rounded printed
reference is 5.564e-12. Check programs, JSON state export, logs, hovers and summaries
are retained under `build/gauss-newton-verification/`.

All seven SVGs were rendered with Inkscape at 1100 px and reviewed; tangent labels,
plot titles and axis spacing were refined. Markdown/Doxygen exports preserve six
C++ programs, the TikZ build, seven local images and six displayed equations, with
no inline SVG XML/data URI. Old mepml/org sources are removed, legacy TeX/HTML and
figure leftovers were checked, and the checklist/algorithm list now point to the
conversion. The Gauss–Newton/updated-dataset scoped audit passes 69/69. The shared
completed-path gate currently reports 9235/9236, with one unrelated missing method
description in `nn/repvgg_models.hpp` (`RepVGGBlockSpec::Validate`). This path belongs
to concurrent work and was left intact. Graphical mep K was not tested.


### RepVGG tensor branch folding and independent deployment (2026-10-29)

- [x] `nn/repvgg_models.hpp`: 35/35 documented definitions, thirteen
  compiled/executed examples, two hovers and four native tests. Tensor folding
  preserves Float64 outputs and every input/filter/BN-affine derivative for
  identity/stride/grouped cases. Deployment copies independent classifier
  storage and omits BN state. Complete A0 counts/gradients and invalid settings
  pass. The report executes all ten full-weight budgets and explicit A0 fitting,
  then fuses the selected trained checkpoint and verifies test predictions.
  Original source syntax and both figure exports pass. Audit XML is isolated
  at `build/cnn-docs-audit/xml` to avoid concurrent rebuild collisions.
  Evidence: `build/cnn-repvgg-coverage.json`, `build/cnn-repvgg-hover.json`.


## ICA report, population whitening and honest source-recovery diagnostics (2026-10-29)

Converted `ica_analysis` into six complete C++ programs plus retained TikZ
whitening/rotation source, producing seven local SVGs. Four equally weighted
Rademacher corners give an exact independent distribution whose rotated coordinates
have identity covariance but remain dependent: their joint-zero probability is zero
while the product of marginal zero probabilities is one quarter. A native cubic
step reproduces raw update (-1.024,-.432) and direction ratio .421875 from (.8,.6).
The original sine/square/sawtooth demonstration remains, with its deterministic
construction distinguished from independent random sources. Separate generated
Laplace/uniform/Rademacher coordinates, Gaussian controls, uniform-mixture geometry
and an explicitly descriptive Iris example complete the report.

The canonical FastICA implementation now uses population covariance 1/n, matching
its 1/n fixed-point averages; the former n-1/n convention mismatch is removed.
Relative numerical-rank checks replace the absolute eigenvalue floor, covariance
formation uses a common scale, and retained values/options/schema are checked.
Duplicate feature names, nonpositive iteration budgets, invalid angle tolerances,
query null/nonfinite/wrong-type columns and degenerate zero updates are explicit
errors. Two-pass deflation and projected initialization retain orthonormal directions
and a valid pseudo-inverse even when a component exhausts its update budget.
Existing constructors/options order and full-model accessors remain; population
rather than sample-unit source scaling is now documented explicitly.

Native APIs expose whitening/eigenvalues, whitened rows, initial and sign-fixed
rotation directions, every completed fixed-point update, per-component moments/
angle stopping, inverse transforms and measured reconstruction checks. Frozen
new-row transforms use only training means/maps and agree for singleton/batch
queries. Known-source generators and Hungarian one-to-one correlation matching
replace optimistic reusable-component scoring in the report. Alignment restores
sign, offset and scale only for supervised benchmarking, never during blind fitting.
Mixing columns are distinguished from unmixing covectors, and full-rank reconstruction
is not described as evidence of independence. Gaussian rotational nonidentifiability,
finite-sample/local contrast limitations, source permutations/scales and standard-
library initialization portability are stated. No new generated binding code is edited.

All twelve non-Gaussian control fits meet the declared angle rule, with mean matched
correlations .98948957 through .99985756. Seven of twelve Gaussian control fits
exhaust 500 updates; an exhausted one nevertheless scores .97614069 against the
arbitrary chosen true basis. Neither that score nor a stopped Gaussian fit resolves
population nonidentifiability. All 24 contrast/seed/family runs remain visible.
The waveform match uses every sample, while only the first 200 traces are plotted.
Iris labels enter only after fitting. Retaining two PCA dimensions preserves
.977685206 of the original covariance trace, with reconstruction RMSE .1591888 cm
per feature cell; all four dimensions reconstruct to floating-point precision.
There is no latent-source accuracy, causal, classification or held-out prediction
claim for these flower measurements.

Both changed paths pass at 84/84 documented definitions. Fifteen header examples
compile and execute, six representative clangd hovers pass, and all seventeen native
ICA tests pass (nine existing, eight new). The whitening test now checks population
variance strictly. Tests cover the exact cubic step, small-mixture whitening/inverse,
frozen singleton inference, strict query validation, exhausted-budget orthogonality,
relative rank across units, PCA projector algebra, distinct source matching and
sign/scale alignment, iid-source recovery and invalid/degenerate inputs.
Independent NumPy/SciPy checks cover 35 fits, 4726 independently replayed updates
and 433722 values, including every control fit, the waveforms, geometry/truncation,
exhausted budgets, disparate units and both Iris fits. Whitening/spectra, fixed-point
steps, deflation, maps/pseudo-inverses/projectors, moments, matching and diagnostics
agree; maximum direction difference is 6.648e-13. Four independent scikit-learn
FastICA comparisons use deflation, the same already-verified white data and actual
initial directions, accounting for signs/permutations. Evidence and check programs
are retained in `build/ica-verification/`.

All seven local SVGs were rendered through Inkscape at 1100 px and visually reviewed;
long titles and legends were shortened. Markdown/Doxygen exports retain six C++
programs, one TikZ build, seven local images and five displayed equations, with no
inline SVG XML/data URI. The old mepml/org reports are removed and no matching legacy
TeX/HTML/figure leftovers remain. The conversion checklist and algorithm list point
to the new report. Fresh Doxygen generation and the shared completed-path gate pass
at 9356/9356 definitions, zero missing. Graphical mep K was not tested.


### Non-local affinities and octave frequency exchange (2026-10-29)

- [x] `nn/nonlocal_models.hpp`, `nn/octave_models.hpp`: 36/36 documented
  definitions, twelve compiled/executed examples, four hovers and eleven
  native tests. Non-local tests compare all affinity/pooling values and every
  derivative, identity/global support, exact five-hook placement and complete
  gradients/state. Octave tests use independent path oracles and kernel finite
  differences, fixed counts, actual224 geometry, corrected high-only stride-two
  merging, full compact gradients and invalid dimensions. Source conventions
  and the corrected final-shortcut operation/MACs are explicit. Both original
  sources compile; eleven report C++ blocks execute visible optimizers and
  export their reviewed figures. Evidence: `build/cnn-nonlocal-coverage.json`,
  `build/cnn-octave-coverage.json`, corresponding hover files, and isolated XML.

### Categorical ID3 conversion and inspectable prediction (2026-10-29)

- [x] `algorithms/decision_tree_induction.hpp`, `algorithms/id3_diagnostics.hpp`
  and `src/datamunge/algorithms/id3_diagnostics.cpp`: 47/47 documented definitions.
  Contracts state categorical codes, strict positive computed gain, feature/label
  tie rules, root-zero depth, empty/zero-width behavior and local-majority fallback.
  Native guards reject ragged/misaligned training data and unsafe visited prediction
  paths. The shared continuous routine remains a restricted C4.5-inspired threshold
  algorithm; its unchecked input preconditions and missing full-C4.5 features are
  documented without changing its behavior.

Split/branch entropy tables, node depths, prediction traces, evaluation with a fitted
majority baseline and small-tree plots are canonical library helpers. A DataFrame
wrapper fits and freezes string/numeric category dictionaries, rejects null/nonfinite
cells and invalid labels/schema, and reserves -1 for unseen levels. Every new or
changed user-facing callable has a runnable example (trivial tree accessor excepted).
Nineteen header examples compile and execute, six representative clangd hovers pass,
and all seventeen focused native tests pass, including the five existing tree/rule
regressions. Tests cover exact gains, greedy XOR failure, identifier memorisation,
depth prefixes, local/global novelty, malformed topology and frozen frame encoding.

An independent Python implementation checks 1152 fits, 7532 nodes and 368751 values:
256 exhaustive binary label assignments at four depth caps, 120 multicategory cases,
tennis and four Titanic depth caps. It independently reconstructs complete trees,
branch frequencies/entropies, traces, predictions, evaluation and entropy telescoping,
and checks majority-error and node-count bounds. Maximum numeric difference is
4.4409e-16. Titanic training/query codes are checked against dictionaries reconstructed
from the source CSV using the declared complete-row split. The report fixes depth
two before holdout evaluation: 550/711 fitting and 149/178 holdout labels are correct,
versus 108/178 for the fitted majority baseline. Its extra Pclass question lowers
entropy without changing hard labels relative to depth one; this is checked.

Five complete C++ programs and one retained TikZ build execute and export six local
SVG figures. All six are rendered with Inkscape at 1100 px and visually reviewed.
Markdown/Doxygen exports preserve six images and three display equations, with no
inline SVG or data URI. The old mepml/org sources are removed, conversion and
algorithm lists are updated, and verification programs/logs live under
`build/id3-verification/`. Fresh Doxygen XML and the shared completed-path gate pass
at 9435/9435 definitions, zero missing. Graphical mep K was not tested.


### Spatial transformer and capsule tensor models (2026-10-29)

- [x] `nn/spatial_transformer_models.hpp`, `nn/capsule_models.hpp`:32/32
  documented definitions, fourteen compiled/executed examples, four hovers and
  eleven native tests. Tensor distorted-digit generation preserves source
  geometric ranges/quantization while explicitly changing RNG draw sequence.
  Affine values/all six derivatives, zero/active localization gradients and full
  paired classifier counts/gradients pass. Capsule routing matches a stable
  host oracle and finite differences through agreement; complete decoder/margin
  gradients and masks/counts pass. Decoder lifetime/ownership and regularization
  at zero are explicit. Both original sources compile; eleven report C++ blocks
  expose actual optimizer loops and export reviewed image galleries/figures.
  Evidence: `build/cnn-stn-capsule-coverage.json` and corresponding hover files.


### Tensor detection components and dataset box AP (2026-10-29)

- [x] `nn/detection_components.hpp` and `nn/detection_evaluation.hpp`:45/45
  definitions documented, thirteen public examples compiled/executed, fourteen
  native tests and four passing clangd hovers. SSD greedy collisions/ties/empty
  targets, shared RPN shapes/counts/all gradients, zero-logit stable BCE,
  Smooth-L1 derivatives and complete patch CNN counts/gradients pass. AP tests
  cover difficult/crowd/regular preference, duplicate/tie/threshold rules,
  max100, absent categories, grid boundaries and randomized host-reference
  agreement. Legacy interfaces forward numerical work to the canonical tensor
  kernels; original C++ programs build/run. Evidence:
  `build/cnn-detection-coverage.json`, `build/cnn-detection-hover.json` and
  the executed/exported detection utility reports.


### Tensor detection scenes, patch sampling and inference (2026-10-29)

- [x] `nn/detection_scene_models.hpp` and `nn/detection_inference.hpp` add29
  documented definitions, twelve public examples, thirteen native tests and
  five clangd hovers. Legacy host pixel/proposal oracles, fractional crops and
  box derivatives, source window order, patch RNG/row isolation, exact box
  outlines, class/image NMS separation, per-image budgets, ties/thresholds,
  empty inputs and record conversion pass. The changed dataset evaluator adds
  a tested explicit inclusive-VOC option for source synthetic scenes while
  retaining strict official defaults. Three-header coverage is58/58. All four
  detection headers compile independently. Evidence:
  `build/cnn-detection-scene-coverage.json`, `build/cnn-detection-scene-hover.json`
  and the executed/exported baseline report with actual selected predictions.

### Gaussian-strategy INLA and bounded hyperparameter mixtures (2026-10-29)

- [x] `bayes/inla.hpp`, `bayes/inla_integration_strategy.hpp`, `bayes/inla.cpp`,
  `bayes/inla_diagnostics.hpp/.cpp` and `stats/inla_mixed_model.hpp/.cpp`:
  186/186 documented definitions across seven canonical paths. Contracts distinguish
  conditional modes from actual posterior means, Gaussian-mixture moments/quantiles
  from exact marginals, transformed SD moments from mean-theta plug-ins, and
  inverse-link mean-field predictions from posterior predictive expectations.

The conditional solver rejects exhausted Newton budgets and failed line searches.
Precision symmetry/finite values, inputs, controls and observation-family support are
checked, and the INLA Bernoulli log likelihood/curvature use stable tails. Outer
curvature probes and grid evaluations respect the declared box. Scalar boundary cells
are clipped/recentered and use their actual measures. Higher-dimensional rotated
boundary cells remain an explicitly documented coarser approximation. EB returns NaN
for integrated evidence. Retained conditional covariance matrices, log integrands,
weights, cell measures, convergence evidence, within/between covariance, grid fallback,
edge and skipped counts make the approximation replayable. Dense scalar reference
quadrature integrates the actual likelihood/prior, without Laplace, for checks.

The formula facade now uses properly normalized bounded uniform priors on packed
factor/log-sigma coordinates. Gaussian sigma support remains data-centered, so its
empirical-prior evidence interpretation is explicit. Training row/design/response
metadata and named fixed/random/physical-SD tables are available. Full-precision
numeric group keys prevent accidental merging, and missing random-only slopes or
all-missing/empty prediction frames no longer silently produce incorrect finite rows
or an empty-design exception. Existing valid plug-in prediction semantics remain.

All 27 focused native tests pass (13 existing and 14 new). Thirty-three header examples
compile and execute, eight representative clangd hovers pass, and the retained original
standalone C++ example compiles/runs with its overclaims corrected. Independent
NumPy/SciPy checks cover 61 fits, 505 conditional evaluations, 43 direct scalar
references and 196961 values. They reconstruct conditional modes/full covariances,
Laplace scores, weights, mixture covariance, densities/quantiles, transformed moments,
plugin predictions and source-array-derived penguin designs. Maximum scaled replay
difference is 7.942e-11. Independent analytic Gaussian integration plus adaptive theta
quadrature measures a span-five SD error of 1.190e-5, reduced to 7.268e-9 at span seven;
the report shows both resolution and span checks. Forty p-adjust table entries and
correct 2/2/3 rejection counts pass an independent SciPy/formula check.

Seven complete C++ blocks plus one retained TikZ build execute and export eight local
SVGs. Each is rendered in Inkscape at 1100 px and visually reviewed. Markdown/Doxygen
exports retain eight figures and four display equations, with no SVG XML/data URI in
results. The old mepml/org/TeX reports are removed, the conversion checklist is checked
and the algorithm list links to the new report. Evidence and repeatable verification
programs are in `build/inla-verification/`. Fresh Doxygen and the shared completed-path
gate pass at 9695/9695 definitions, zero missing. Graphical mep K was not tested.


### Shared tensor R-CNN family models and data (2026-10-29)

- [x] `nn/two_stage_detection_models.hpp`:44/44 documented definitions,
  eighteen compiled/executed public examples, six hovers and fifteen native
  tests. Positive/ignored/negative RPN targets, forced matching, stable top
  budgets, clipping/minimum-side/NMS, sampled RoI quotas/indices/empty truth,
  feature-centre/bilinear pool geometry and exact selected-regressor gradients
  pass. Full R-CNN/Fast/Faster counts and every learned tensor gradient pass.
  Detached approximate joint inference blocks head-loss gradients into RPN
  leaves while retaining shared-backbone gradients. Legacy model/loss wrappers
  preserve parameter names. All43 formatted public examples across the five
  detection headers compile/run; original sources and report exports pass.
  Evidence: `build/cnn-two-stage-coverage.json`, `build/cnn-two-stage-hover.json`,
  `build/cnn-two-stage-validation/` and actual reviewed model/data galleries.


### Tensor feature-pyramid routing and compact detector (2026-10-29)

- [x] `nn/feature_pyramid_models.hpp` and shared pooled box-head finisher add
  seventeen documented definitions, nine compiled/executed examples, five
  hovers and five native tests. Affected coverage61/61; all27 affected public
  examples and twenty native tests (including two-stage regression) pass.
  Independent odd-grid fusion/derivatives prove unsmoothed recursion and
  linear negative responses. Routed constant maps/order/feature-only gradients,
  canonical scales, complete source detector counts/all learned gradients and
  input guards pass. Legacy wrappers preserve names. Evidence:
  `build/cnn-fpn-coverage.json`, `build/cnn-fpn-hover.json`, report exports and
  actual reviewed selected detection galleries.


### Kernel ridge regression, a free intercept and honest validation (2026-10-29)

Completed `include/datamunge/stats/kernel_ridge_regression.hpp` and
`src/datamunge/stats/kernel_ridge_regression.cpp`: 130/130 documented definitions.
The native model now centres both kernel and response for an unpenalised intercept,
centres query kernels using fitted quantities, and includes the constant direction
in smoother trace. The previous response-only centring did not generally solve the
stated free-intercept objective. Positive-penalty deletions refit intercept/centring
at a fixed predictor metric; zero penalty uses explicit deleted pseudoinverse fits.
No arbitrary leverage or denominator floor masks an interpolation limit. Scaled
population SD calculation preserves very small nonconstant predictor units.

New contracts cover optional zero intercept, explicit finite candidate grids,
training and LOO predictions, original kept rows, fitted design/scaling/Gram arrays,
signed response weights, smoother/spectrum/rank, objective/equation residuals,
all-candidate selection tables, finite complete evaluation and training-mean baseline.
Fold validation requires fixed candidates and learns its preprocessing from each
fold's fitting rows; it returns per-fold mean/SD and original-row predictions.
Null omissions, invalid controls, PSD polynomial restrictions, query alignment,
transformed design inference, dense complexity and zero-penalty limitations are
explicit. Public references point to Kimeldorf–Wahba (1971) and Saunders et al.
(1998); algorithm papers were verified through the publisher/author repository.

The converted `docs/mepml/examples/kernel_ridge_analysis.mepml` opens with exact
RBF/linear two-point calculations and a four-row preprocessing counterexample.
Its sine-wave sweep separates training, fixed-metric tuning and generating-mean
error. Correct free-intercept LOO changes default selection and curve recovery:
independent augmented-system checks confirm the .059972 mean-grid RMSE, so the
old .05 recovery threshold becomes .08, still well below noise SD .15. Iris uses
120 fitting flowers, 12 declared candidates, four preprocessing-refitted folds,
and 30 untouched held-out rows. Winner gamma .1/lambda .01 gives holdout RMSE
.366721 cm versus training-mean baseline 1.816220 cm; no uncertainty band or
population generalisation certificate is inferred from the split.

Verification: native library build and all 23 KRR tests (nine existing, fourteen
new); all 22 header code examples compiled and executed; seven representative
clangd hovers; fresh scoped Doxygen audit 130/130 and completed-path gate
9886/9886 with zero missing at this run. Independent NumPy augmented-system
solves and sklearn centred-KRR/primal-ridge comparisons check 894522 values,
480 automatic candidates, 1217 explicit fixed-metric deletions and all 48 Iris
fold fits. Maximum prediction difference is 2.07e-11 and deletion difference
1.69e-10; tiny-penalty dual coefficients are more ill-conditioned (absolute
comparison differences up to 1.02e-6), retained in the evidence. Seven complete
C++ report blocks plus one TikZ build generate eight local SVGs; every figure was
rendered with Inkscape at 1100 px and visually inspected. Markdown and Doxygen
exports resolve all figures and four display equations without inline SVG output.
Legacy mepml/org sources were removed; no standalone C++ source was present.
Evidence lives in `build/krr-verification/`. Graphical mep hover was not exercised;
clangd/Doxygen checks cover the available editor-help path.


### Tensor SSD models, mining and decoding (2026-10-29)

- [x] `nn/ssd_models.hpp`:25/25 definitions, nine compiled/executed public
  examples, five hovers and nine native tests. All default coordinates match
  an independent reference. Known L2 normalization/zero-vector derivatives,
  every analytic multibox derivative, per-image mining/batch normalization,
  all-background/empty losses, source tie conventions, forced targets and
  decode/filter ordering pass. Full SSD300 and compact source counts, grids
  and every learned tensor gradient pass. Legacy numerical entry points
  forward to tensors; the original C++ program builds/runs all full fixtures
  and bounded fitting. Standalone include and report exports pass. Evidence:
  `build/cnn-ssd-coverage.json`, `build/cnn-ssd-hover.json`,
  `build/cnn-ssd-validation/` and actual reviewed detection galleries.


### MT19937-64 word operations, continuation recovery and finite-sample diagnostics (2026-10-29)

Completed `include/datamunge/random/mt19937_64_analysis.hpp` and
`src/datamunge/random/mt19937_64_analysis.cpp`: 9/9 documented definitions, added to
the completed-path manifest. The nine MT class definitions in shared
`random/random.hpp` and `random.cpp` are separately checked through the fresh XML
inventory. The wider random core retains its existing backlog and is not marked
complete. Public contracts cover scalar seeding, period versus storage/entropy,
ordered in-place refills, full raw words, inherited sampling/cache behaviour,
borrowed state lifetime and index, and aligned-output continuation recovery.

The canonical engine shares its twist and tempering operations with native
inspection functions, retaining every existing seeded raw output. New finite
inverse-series tempering recovery is checked independently by triangular bit
substitution. The continuation factory untempers an aligned 312-output block,
sets index to exhausted and starts a fresh normal cache; it does not infer block
alignment, recover a scalar seed, or reconstruct a previous normal cache.
Arbitrary arrays define deterministic continuations, and all-zero input deliberately
yields a degenerate stream outside the nonzero-period claim. The conformance table
compares every raw word with std::mt19937_64, preserving 64-bit seeds and endpoints
as hexadecimal strings. References were verified against the original 1998 paper,
authors' distinct 64-bit version page and the C++ working draft parameters and
10000th reference output. The 32-bit engine's 623-dimensional claim is not transferred.

The converted `docs/mepml/examples/mersenne_twister64.mepml` derives scalar seeding,
bit joining/twisting and the inverse of each nilpotent masked-shift map. Exact
odd/even and input-one calculations precede the reference transcript, state
recovery, single-seed battery and all forty-seed second-level checks. It preserves
the original raw/normal/die call order and explains upper-53-bit floating sampling.
The low-byte p=.035514 and seven upper-tail serial-pair p-values are retained.
Pointwise bit-frequency guides, approximate p-value calibration, multiplicity,
consecutive-seed dependence and the lack of a randomness certification are explicit.
No external dataset is needed for these exact and generated-input algorithm checks.

Verification: native core build and all 56 generator/quality tests (45 existing,
11 new); all eight affected header examples compiled and executed; eight clangd
hover checks; new-file audit 9/9, nine MT class contracts, and completed-path gate
9920/9920, zero missing at this run. Independent Python integer recurrence and
bit-triangular inversion check 169246 exact comparisons, including 160000 raw
stream words and both observed/recovered continuations. NumPy/SciPy replay all
41 batteries from 5740000 independently generated words and compare 674 floating
values; largest statistic discrepancy is 3.42e-13 and largest p-value difference
1.68e-13. Six complete C++ report programs plus retained TikZ source generate
seven local SVGs, all rendered at 1100 px in Inkscape and visually inspected.
Markdown/Doxygen exports resolve seven figures, three display equations and ten
alt texts without inline SVG output. Legacy mepml/org files are removed; no standalone
C++ example was present. No generated language binding changes were needed for
these C++ diagnostic APIs. Graphical mep hover was not exercised; clangd/Doxygen
provide the available help checks. Evidence: `build/mt19937-64-verification/`.

## Hierarchical B-splines (2026-10-29)

Completed `include/datamunge/fda/hierarchical_bspline.hpp` and
`src/datamunge/fda/hierarchical_bspline.cpp`: 187/187 definitions documented, with zero
missing in fresh Doxygen XML. The port adds Oslo knot refinement, `TensorBSpline`,
tensor P-spline fits, the Forsey-Bartels overlay `HierarchicalBSpline` and the
`TruncatedHierarchicalBSpline` (THB). Contracts cover clamping, row-major coefficient
order, cell-local patch coordinates, truncation semantics, raw-value versus residual
fitting, penalty restriction to active functions and solver failure modes.

Validation: 44 new C++ tests plus the 13 existing fda B/P-spline tests pass against the
CMake-built library (C++17 test standard). Checks include exact refinement of random
splines for degrees 0-4 with repeated knots, agreement with `fit_pspline()` for
penalty orders 1-3, THB partition of unity, continuity after a coarse edit across a
refined boundary, exact recovery of a three-level THB function, and least-squares
residual orthogonality. All 43 header examples compile and print their documented
output. Python bindings (`pydatamunge.i`) pass `tests/python/test_hierarchical_bspline.py`
against a fresh core extension; the full wheel build currently stops in the separate
`nn_python_wrap.cpp` extension (`CondenseContext` undeclared), unrelated to this batch.
R and the other bindings are not wired.



### Tensor dense detection models (2026-10-29)

- [x] `nn/retina_detection_models.hpp`, `nn/fcos_detection_models.hpp` and
  `nn/center_detection_models.hpp`:92/92 definitions,36 compiled/executed
  examples,13 hovers and22 native tests. Independent full loss/derivative,
  anchor/point coordinate and Gaussian-pixel references pass, together with
  matching/collision rules, inference ordering and full/compact every-tensor
  gradients/counts. One new SSD optional-raw regression passes as well.
  Legacy programs compile/run all fixtures and bounded fitting. Standalone
  headers and report exports pass. Every fit shows tensor Adam explicitly and
  restores paired learned/BN checkpoints. Actual poor held-out detection is
  documented honestly. Evidence: `build/cnn-dense-detection-coverage.json`,
  `build/cnn-dense-detection-hover.json`, `build/cnn-center-hover.json` and
  `build/cnn-dense-detection-validation/`.


### Tensor weighted pyramids and DETR set matching (2026-10-29)

- [x] `nn/bifpn_models.hpp`, `nn/efficientdet_models.hpp`,
  `nn/tensor_assignment.hpp`, `nn/detr_models.hpp`:59/59 definitions,
  21 compiled/executed public examples,11 hovers and11 native tests.
  Fusion derivatives, odd grids, full D0 metadata/gradients,100 exhaustive
  rectangular assignment references, every sine position, set-loss analytic
  and central-difference derivatives, empty truths and full/compact DETR
  connections pass. Potential overflow is guarded. Legacy interfaces forward
  and original programs run full fixtures and bounded fitting. Standalone
  headers and Markdown/Doxygen exports pass. Actual poor held-out fits are
  documented with explicit whole-model tensor Adam and paired BN checkpoints.
  Evidence: `build/cnn-set-fusion-coverage.json`,
  `build/cnn-set-fusion-hover.json`, `build/cnn-set-fusion-validation/`.


### Tensor YOLO v1 and scene affine augmentation (2026-10-29)

- [x] `nn/detection_augmentation.hpp`, `nn/yolo_v1_models.hpp`:34/34 definitions,
  14 compiled/executed public examples, six hovers and nine native tests.
  Exact loss derivatives, detached responsibility, collision/tie handling,
  decode clipping/class scores, full448 backbone and compact every-tensor
  gradients pass. Tensor affine pixels/boxes match independent fixtures.
  Pretraining crop split isolation, explicit optimizer use and atomic prefix
  transfer/leaf identity/head preservation are verified, including incomplete
  BN-prefix rejection. Original program runs all fixtures and bounded fit.
  Standalone includes and report exports pass. Both fitting stages are visible
  tensor Adam loops with paired learned/BN checkpoints and actual data galleries.
  Evidence: `build/cnn-yolo-v1-coverage.json`, `build/cnn-yolo-v1-hover.json`,
  `build/cnn-yolo-v1-validation/` and reviewed local figures.


### Tensor YOLO v2 and WordTree hierarchy (2026-10-29)

- [x] `nn/word_tree_models.hpp`, `nn/yolo_v2_models.hpp`:48/48 definitions,
  18 compiled/executed public examples, seven hovers and eight native tests.
  Source IoU clustering/targets/ignore/collision/current-confidence contracts,
  every detection and classification-only derivative, hierarchy path/coarse/fine
  gradients and9418-node capacity pass. Full20/9418-class graph counts and all
  learned tensors, compact multiscale shapes, standalone headers and original
  fixtures/bounded fitting pass. Both model fits expose tensor Adam and restore
  paired BN checkpoints, with fit-only priors and actual held-out error galleries.
  Markdown/current website HTML exporter pass (nine local media references).
  Evidence: `build/cnn-yolo-v2-coverage.json`, `build/cnn-yolo-v2-hover.json`,
  `build/cnn-yolo-v2-validation/`.


### Tensor YOLO v3 three-scale detector (2026-10-29)

- [x] `nn/yolo_v3_models.hpp`:27/27 definitions, ten compiled/executed examples,
  five hovers and seven native tests. Full80/20-class every-tensor gradients,
  compact multiscale shapes/nonzero gradients, global prior/collision/area/ignore
  rules, exact multi-label/coordinate/background derivatives, empty-scene masks
  and cross-head class/image suppression pass. Legacy original fixtures and
  bounded fitting pass. Explicit whole-model tensor Adam, fit-only pixel priors,
  paired BN restoration and actual poor held-out predictions are documented.
  Standalone header and Markdown/current website exports pass. Evidence:
  `build/cnn-yolo-v3-coverage.json`, `build/cnn-yolo-v3-hover.json`,
  `build/cnn-yolo-v3-validation/`. Stable validation core is `build/cnn-core/`.


### Continuous optimization examples and callback contracts (2026-10-29)

Completed the nine remaining optimization-report conversions and documented 60
native header/source files: `optim/analysis` plus the 29 optimizer implementations
used by those reports. The scoped audit passes 547/547 definitions, including
private validators, covariance eigenpairs, enum strategies and particle fields.
All 60 paths are in `docs/documented-paths.txt`; the fresh completed-path gate
passes 10914/10914 at this run. Other optimization families and general autodiff
remain outside this completed batch.

Public contracts describe actual update variants, reset state, input/output
shapes, defaults, mutation, borrowed callback lifetimes, callback exceptions,
method-specific stopping and nominal costs. Original sources are linked where
applicable. The new reusable fixtures expose exact quadratic/KKT/proximal controls,
Formula-frozen least-squares designs, original noisy nonlinear curve recipes,
analytic derivatives and explicit post-run diagnostic plots. Typed traces retain
callback order and sampled row indices without inserting fitting calls; full
values, derivative requests, rejected trials, accepted iterates and diagnostic
sampling are not conflated. Normal return is explicitly not a convergence status.

SVRG's full summed snapshot gradient is now divided by the number of terms before
combining it with unscaled component differences. This corrects the biased mixed
sum/mean estimator and aligns its update scale with SGD/SAGA. SGD rejects a zero
mini-batch size. Other solver update rules and option layouts are preserved;
restricted SQP, Cauchy trust-region, fixed-beta Nadam and bias-corrected AMSGrad
variants are identified without inheriting unsupported published guarantees.

Verification: native library build; all 76 optimizer tests (12 fixture/regression
tests); 175 compiled/executed header examples with matching output comments;
twelve representative clangd hovers; and 121 independent NumPy/SciPy numerical
comparisons (maximum discrepancy 4.12e-9). Independent checks reconstruct adaptive,
momentum, coordinate/CG, Newton/Cauchy, SQP, central-path and proximal steps; replay
SGD/SVRG/SAGA from actual sampled indices; compare OLS/lasso controls and both
nonlinear curve fits; and verify bounded best-visited return semantics. All 37
report C++ blocks and nine TikZ builds execute. The 46 saved SVGs were rendered
with Inkscape at 1100 px and visually inspected; all nine reports export Markdown
and HTML through the current website workflow. Evidence lives under
`build/optimization-verification/`. No graphical mep UI was available for K-help
integration verification. Legacy mepml/org/TeX copies are removed, and the
conversion checklist and algorithm links point to the maintained reports.


### Tensor YOLO v5-style and mosaic (2026-10-29)

- [x] `nn/yolo_v5_models.hpp`, `nn/detection_mosaic.hpp`:49/49 definitions,
  20 compiled/executed examples,six clangd hovers,seven native tests plus the
  original independent CSP/SPPF/mosaic/CIoU/assignment/full-gradient fixtures.
  Full/compact177-leaf gradients,64/96/128px heads, every analytic loss
  derivative,21-slot ratio/neighbour assignment,fallback/ties,rectangular
  byte-exact mosaics, independent copies,flip involution and decoding pass.
  Explicit whole-model Adam on fitting-only mosaics/flips,fit-only priors,
  paired BN restoration and actual held-out errors are documented. Both direct
  header includes and Markdown/current website exports pass. Evidence:
  `build/cnn-yolo-v5-coverage.json`, `build/cnn-yolo-v5-hover.json`,
  `build/cnn-yolo-v5-validation/`.


### Tensor segmentation losses and distance maps (2026-10-29)

- [x] `nn/segmentation_losses.hpp`:11/11 definitions,seven compiled/executed
  examples,four clangd hovers,five native tests and the original fixture/fit
  workflow. Independent overlap derivatives,Lovasz subgradients,void/perfect/
  absent behavior,negative boundary loss and detached gradients,100 exact
  rectangular nearest-seed comparisons,signed maps and invalid inputs pass.
  The tensor EDT's higher asymptotic cost is explicit. Four visible Adam field
  fits and aligned pixel galleries demonstrate known-mask reconstruction.
  Direct include and Markdown/current website exports pass. Evidence:
  `build/cnn-seg-losses-coverage.json`, `build/cnn-seg-losses-hover.json`,
  `build/cnn-seg-losses-validation/`.


### Tensor FCN/U-Net and segmentation data/evaluation (2026-10-29)

- [x] `nn/segmentation_data.hpp`, `nn/fcn_models.hpp`, `nn/unet_models.hpp`:
  54/54 definitions,23 compiled/executed examples,eight clangd hovers,nine
  native tests. Full original parameter/every-gradient graphs,compact nonzero
  gradients,dynamic odd/even grids,independent fusion/crop derivatives,
  tensor data determinism/copy isolation,fit-only class weights,weighted CE
  derivatives,exact confusion/IoU and palette contracts pass. Original
  targets run every fixture and bounded fitting. Four compact variants fit
  explicit tensor Adam with validation-selected restoration and actual
  held-out mask galleries. Standalone headers and Markdown/current website
  exports pass. Evidence: `build/cnn-fcn-unet-coverage.json`,
  `build/cnn-fcn-unet-hover.json`, `build/cnn-fcn-unet-validation/`.


### Tensor SegNet indexed pooling and sparse unpool (2026-10-29)

- [x] `nn/segnet_models.hpp`:15/15 definitions,six compiled/executed examples,
  three clangd hovers,four native tests plus original full fixture/quick fit.
  Tensor first-maximum NHWC winners,negative/odd/multiple-batch/channel sparse
  values and every derivative,malformed record checks,full102 gradients,
  compact nonzero gradients/dynamic grids and precise cache bytes pass.
  Explicit whole-model Adam,fit-only weights,paired BN restoration and actual
  failed foreground recognition are documented. Direct include and Markdown/
  current website exports pass. Evidence: `build/cnn-segnet-coverage.json`,
  `build/cnn-segnet-hover.json`, `build/cnn-segnet-validation/`.


### Tensor dense-skip and attention U-Net variants (2026-10-29)

- [x] `nn/unet_variant_models.hpp`:24/24 definitions,eight compiled/executed
  examples,three clangd hovers,four native tests plus original fixtures/fit.
  Full68/70 learned gradients,compact nonzero gradients,dense concatenations,
  all four supervised heads,analytic gate input derivatives,negative ReLU/
  saturated sigmoid,odd-grid and odd-base-width geometry pass. Explicit
  whole-model Adam and four-head loss averaging,fit-only weights,validation
  restoration,actual test masks and fixed-range learned gate maps are shown.
  Direct include and Markdown/current website exports pass. Evidence:
  `build/cnn-unet-variants-coverage.json`, `build/cnn-unet-variants-hover.json`,
  `build/cnn-unet-variants-validation/`.


### Tensor dense ResNet,DeepLab v1/v2 and exact CRF (2026-10-29)

- [x] `nn/deeplab_models.hpp`, `nn/dense_crf_models.hpp`:36/36 definitions,
  15 compiled/executed examples,seven clangd hovers,seven native tests plus
  original full fixtures/quick fit. All full/compact gradients,dense stride
  shapes/counts,independent every-tap ASPP derivatives,every-pair Gaussian/
  bilateral kernels,Potts iterations/unary finite differences,zero self
  edges/symmetry,source pixel-centre sampling and its gradients pass. Complete
  explicit Adam fits,fit-only weights,paired V2 BN restoration and separate
  native/subsampled-CRF metrics retain real errors. Direct headers and
  Markdown/current website exports pass. Evidence:
  `build/cnn-deeplab-coverage.json`, `build/cnn-deeplab-hover.json`,
  `build/cnn-deeplab-validation/`.


### Tensor global/atrous and adaptive pyramid context (2026-10-29)

- [x] `nn/deeplab_v3_models.hpp`, `nn/pyramid_pool_models.hpp`:39/39 definitions,
  nine compiled/executed examples,five clangd hovers,seven native tests plus
  original full fixtures/quick fits. Independent global mean/broadcast and
  overlap-bin values/every derivative,full179/203/334 learned gradients,
  compact nonzero gradients,five context branches,dense stride doubling,
  odd input alignment,auxiliary-disable and invalid rates/bins pass. All
  compact leaves fit explicit Adam,including PSPNet auxiliary CE at.4 weight,
  with fit-only weights,paired BN restoration and actual held-out errors.
  Direct headers and Markdown/current website exports pass. Evidence:
  `build/cnn-pyramid-context-coverage.json`, `build/cnn-pyramid-context-hover.json`,
  `build/cnn-pyramid-context-validation/`.


### Tensor HRNet persistent streams and exchange (2026-10-29)

- [x] `nn/hrnet_models.hpp`:17/17 definitions,five compiled/executed examples,
  three clangd hovers,three native tests plus original full fixture/quick fit.
  Independent odd-grid route values,every interpolation/subsampling source
  derivative,full920 gradients,compact nonzero gradients,persistent branch
  shapes,input alignment and invalid geometry/overflow checks pass. Complete
  explicit Adam,fit-only weights,paired BN restoration,actual failed masks
  and shared-range restored feature maps are documented. Direct include and
  Markdown/current website exports pass. Evidence:
  `build/cnn-hrnet-coverage.json`, `build/cnn-hrnet-hover.json`,
  `build/cnn-hrnet-validation/`.


### Tensor instance supports and Mask R-CNN (2026-10-29)

- [x] `nn/instance_scene_models.hpp`, `nn/mask_rcnn_models.hpp` and private
  shared sampler add29 documented definitions,16 compiled/executed examples,
  six clangd hovers and12 native tests. Independent support/paste/IoU/AP,
  fractional/border alignment/every feature gradient,selected-class BCE,
  full/compact head gradients,nearest truth targets,raw/default box decoding,
  empty gradients and palette/score overlay checks pass. Original two-stage/
  scene regressions,legacy Mask R-CNN executable,standalone includes and report
  Markdown/current website exports pass. All figures reviewed. Evidence:
  `build/cnn-mask-coverage.json`, `build/cnn-mask-hover.json`,
  `build/cnn-mask-validation/`;complete tensor Adam and honest custom mask-AP
  semantics documented without a paper-model accuracy claim.


### Tensor YOLACT prototypes,targets and Fast NMS (2026-10-29)

- [x] `nn/yolact_models.hpp`:40/40 definitions,14 compiled/executed public
  examples,five clangd hovers and six native tests. Independent signed basis/
  crop derivatives,weighted objectives/mining,ignored bands,reordered nearest
  targets,semantic unions,all-predecessor suppression and known decoded masks
  pass. Full50157071-scalar/348-gradient graph and compact every-nonzero
  gradients execute. Explicit all-branch Adam,paired BN restoration and actual
  zero held-out AP/failed masks are documented honestly. Original full fixtures
  and finite bounded quick diagnostics,standalone include,Markdown/current
  website exports and seven reviewed figures pass. Evidence:
  `build/cnn-yolact-coverage.json`, `build/cnn-yolact-hover.json`,
  `build/cnn-yolact-validation/`.


### Tensor panoptic shared models,merge,metrics and scenes (2026-10-29)

- [x] `nn/panoptic_fpn_models.hpp`, `nn/panoptic_models.hpp`,
  `nn/panoptic_scene_models.hpp`:44/44 definitions,19 compiled/executed public
  examples,six clangd hovers,six native tests. Full46027110-scalar/224-gradient
  graph,compact125 nonzero gradients,exact routed Jacobians,overlap trimming,
  strict-IoU/void/crowd PQ,empty/invalid maps and all-pixel textured target
  alignment pass. Explicit four-task Adam,paired BN restoration and actual
  strong stuff/zero thing outcomes are documented without benchmark claims.
  Original full fixtures/quick fit,standalone includes,Markdown/current website
  exports and seven reviewed figures pass. Evidence:
  `build/cnn-panoptic-coverage.json`, `build/cnn-panoptic-hover.json`,
  `build/cnn-panoptic-validation/`.


### Tensor volumetric models,data and sparse weights (2026-10-29)

- [x] `nn/volumetric_models.hpp`, `nn/volumetric_data.hpp`:41/41 definitions,
  19 compiled/executed public examples,six hovers,nine native tests. Every
  independent3D convolution/transpose/crop/PReLU/BN/sparse-CE/squared-Dice
  derivative,full-width census/gradients,compact nonzero stage gradients,
  sparse row pairing/fit-only weights and orthogonal geometry pass. Complete
  visible two-model Adam,paired BN restoration,current-volume test BN and
  actual U-Net oversegmentation/V-Net dense-label results are documented.
  Original full fixtures/three-epoch workflow,standalone headers,Markdown/
  current website exports and seven reviewed figures pass. Evidence:
  `build/cnn-volume-coverage.json`, `build/cnn-volume-hover.json`,
  `build/cnn-volume-validation/`.


### Tensor Siamese models and metric-learning data (2026-10-29)

- [x] `nn/siamese_models.hpp`, `nn/metric_learning_data.hpp`:43/43 definitions,
  23 compiled/executed public examples,six hovers,six native tests. Independent
  head/BCE/extreme derivatives,shared branch gradients,full/compact graphs,
  class-disjoint sampling,gather Jacobians,byte SSD,verification/tie rules and
  every episode gallery pixel pass. Complete visible Adam and poor held-out
  transfer are documented with restored validation scores and actual images.
  Original complete workflow,standalone includes,Markdown/current website
  exports and five reviewed figures pass. Evidence:
  `build/cnn-siamese-coverage.json`, `build/cnn-siamese-hover.json`,
  `build/cnn-siamese-validation/`.


### Tensor NN1 embeddings and triplet mining (2026-10-29)

- [x] `nn/facenet_models.hpp`, `nn/metric_learning.hpp`:35/35 definitions,
  21 compiled/executed public examples,six hovers,six native tests. Full220RGB
  forward/backward and compact all-stage nonzero gradients pass. Independent
  normalization/maxout/triplet Jacobians,empty and hinge-boundary gradients,
  exhaustive strict mining and tie rules,P5K8 sampling,calibration and AUC pass.
  Visible detached mining/recomputed Adam,restored validation checkpoint,
  validation-only threshold and actual held-out retrieval images are documented.
  Original complete workflow,standalone includes,Markdown/current website
  exports and six reviewed figures pass. Evidence:
  `build/cnn-facenet-coverage.json`, `build/cnn-facenet-hover.json`,
  `build/cnn-facenet-validation/`.


### Tensor angular margins and label-free cosine heads (2026-10-29)

- [x] `nn/angular_margin_models.hpp`:24/24 definitions,nine compiled/executed
  public examples,six hovers,five native tests. Full220RGB ArcFace/CosFace
  graphs and compact all29 nonzero gradients pass. Independent target/CE/
  feature/weight Jacobians,central differences,endpoint/tail/zero-margin
  policies,analytic boundaries,scale invariance and tensor confusion pass.
  Visible two-model Adam,plain validation checkpoint restoration,separate
  unseen calibration/retrieval and actual error images are documented.
  Original complete workflow,standalone include,Markdown/current website
  exports and eight reviewed figures pass. Evidence:
  `build/cnn-angular-coverage.json`, `build/cnn-angular-hover.json`,
  `build/cnn-angular-validation/`.


### Tensor bilinear CNN and official flower ingress (2026-10-29)

- [x] `nn/bilinear_models.hpp`, `nn/flower_data.hpp`:38/38 definitions,
  17 compiled/executed public examples,six hovers,seven native tests. Full
  and compact two-stream/RGB gradients,all37 stages,independent local outer/
  signed-root/unit/CE derivatives,metrics,nontrivial color prototypes,every
  prediction tile/bar pixel and row-gather gradients pass. Official archive
  checksums,disjoint split/class metadata and tensor resize preparation pass.
  Visible SGD color fitting and complete Adam,restored validation,immutable
  independent deployment and all48 actual test images are documented.
  Original complete workflow,standalone headers,Markdown/current website
  exports and eight reviewed figures pass. Evidence:
  `build/cnn-bilinear-coverage.json`, `build/cnn-bilinear-hover.json`,
  `build/cnn-bilinear-validation/`.


### Tensor GeM retrieval and exact Int64 host extraction (2026-10-29)

- [x] `nn/retrieval_models.hpp`, `nn/retrieval_data.hpp`,
  `nn/retrieval_index.hpp`:54/54 definitions,20 compiled/run public examples
  (including two changed shared-tensor examples),eight hovers,seven retrieval
  tests plus two new tensor regressions. All35 tensor value tests pass. Existing
  `nn/tensor.hpp` coverage69/69 is separately verified;item/to_vector preserve
  full-range Int64 values without a double intermediate. GeM/power/unit
  derivatives,full/compact all-stage graphs,every source CIFAR pixel/metadata
  row,hard mining,exact ties/self/AP/empty cases,index ownership/serialization,
  pixel references and every gallery pixel pass. Visible complete-model Adam,
  restored validation and immutable deployment are documented with actual
  held-out errors and stronger color references. Original full workflow,
  standalone includes,Markdown/current website exports and seven figures pass.
  Evidence: `build/cnn-retrieval-coverage.json`,
  `build/cnn-retrieval-tensor-coverage.json`, `build/cnn-retrieval-hover.json`,
  `build/cnn-retrieval-validation/`.


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


## Native compute/game/simulation foundation — 2026-10-30

Completed the new headers and implementation files listed under the native
simulation batch in `docs/documented-paths.txt`. Contracts cover explicit device
ownership/transfers and synchronous native kernel dispatch; CPU primitive physics,
constraints, snapshots and rollback; scene/behavior lifecycle; batched RNG/reset;
and the native articulated control task definitions. GPU backend internals are
included in the Doxygen inventory through explicit CUDA/Vulkan feature defines.

Scoped fresh XML coverage is 647/647 definitions with no missing descriptions.
All 116 header examples compile/run; six representative definition hovers pass
through clangd. This does not establish renderer UI help or GPU operation parity.
The generated Python core and R task adapters expose owned output snapshots and
binary masks; their native tests cover garbage collection and episode-end state.

Evidence: `build/simulation-verification/{coverage,examples,hover}.json`,
`ctest.log`, `sanitizers.log`, `gpu-context-tests.log`, `python-tests.log` and
`r-tests.log`. There are 37 passing GTests plus the allocation check; two GPU
execution checks skip in a CPU-only build. Thirty-six CPU tests pass ASan/UBSan
with leak detection disabled because LeakSanitizer cannot run under this host's
ptrace restrictions. Software Vulkan AXPY/buffer execution passes, while CUDA
execution and physical GPU performance remain unverified. The full Python wheel
build has separate existing NN wrapper compile errors; only the generated core
extension and its task tests are qualified here. Full migration status remains
in `docs/simulation-plan.md`.


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


### Native scene import, Courier and renderer contracts (2026-10-30)

The compute/game/sim scope now passes fresh Doxygen coverage for 856/856
definitions, including strict native JSON, explicit legacy behavior mappings,
Courier factories and Vulkan renderer internals. All 137 public-header examples
compile and run against the GPU-enabled core on software Vulkan; nine scoped
clangd definition hovers pass. Verified new paths are in documented-paths.txt.
The native CPU suite passes 52 executed checks, with four optional GPU skips;
all three renderer tests pass separately on software Vulkan. Address/undefined
sanitizers pass 49 CPU cases (LeakSanitizer unavailable under host ptrace).
Physical GPU performance and graphical mep help remain unverified.

### Device-aware tensor storage and transfer contracts (2026-10-30)

The combined compute/game/sim and tensor/storage/autograd scope passes fresh
coverage for 1,235/1,235 definitions. All 154 selected public header examples
compile/run, including native device contracts and explicit CPU NPY serialization;
12 scoped clangd hovers pass. New private storage kernel files and verified
autograd/detail paths are added to documented-paths.txt. Full legacy NN I/O source
documentation remains outside this completed batch.

Behavior evidence: 216 executed CPU checks pass (six optional GPU skips); the
GPU-enabled suite passes 14 checks with two CUDA skips, including actual Vulkan
native compute, rendering and tensor storage execution. Tensor/transfer-autograd
and compute backend host code also pass Address/Undefined sanitizer checks;
unrelated numerical objects were linked without instrumentation. LeakSanitizer
and physical GPU/CUDA execution remain unverified. Native task language tests
still pass (Python 2 tests; R 16 assertions).


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


### Native training adapters and post-solve speed bounds (2026-10-30)

World::step now documents the post-constraint 500 m/s / 100 rad/s dynamic speed
limits. The same cap previously applied only after force integration; full-range
Hopper actions exposed the missing post-solve bound. Native PPO/SAC examples now
collect real transitions and evaluate trained policies. The internal native task
adapters document state/observation ownership, explicit CPU execution, automatic
masked reset and correct time-limit GAE boundaries. Behavioral evidence includes
800 full-range Hopper transitions, three adapter/GAE tests and four native CPU
training smoke runs. Updated coverage/sanitizer counts are retained in the
simulation verification directory and implementation plan.

The final expanded batch passes 1,236/1,236 scoped definitions, 154 executable
header examples, 12 clangd hovers, 220 executed CPU checks and 53 CPU engine/adapter
Address/Undefined sanitizer cases. Four native PPO/SAC smoke runs each verify
nonzero parameter changes. Shader/backend execution uses software Vulkan;
CUDA driver execution and physical GPU performance remain unqualified.

### CUDA toolkit and native matrix adapters (2026-10-30)

The compute contracts now document MatrixType, packed MatrixProduct layouts,
Context::supports_matmul and Context::matmul, including precision, synchronous
ownership, aligned bounded views, alias rejection, empty products and failures.
The nn::matmul contract describes native CUDA packing, batch broadcasting,
Float32/Float64 context restrictions and the remaining broadcast-gradient
reduction limitation. Private cuBLAS ownership and validation helpers have
adjacent Doxygen descriptions; conditional cuBLAS definitions participate in the
XML audit. The matrix implementation is added to documented-paths.txt.

Verification: 263/263 compute/matrix implementation definitions documented,
158 compiled/executed header examples and four representative clangd hovers.
The full core builds inside the new Nix CUDA shell; four physical RTX 5080 CUDA
tests pass against that build, including both-precision numerical and gradient
checks. NVCC-generated sm_120 PTX also executes correctly through Context. The
combined native CUDA/software-Vulkan suite passes 18/18 checks. CPU regression
coverage executes 220 passing checks with eight optional GPU skips. Logs are
retained in build/cuda-verification/. Full GPU NN/physics parity, GPU training,
GPU sanitizers and performance qualification remain open.

### Executable docs header and toolchain repair (2026-10-30)

`just docs` and `just docs-mepml-run` now use the installed-library `cpp` shell;
the lightweight `cpp-tools` shell still supports audits and local development.
Both report entry points check pkg-config, public-header compilation, linking
and execution before modifying report results/statuses. This check also applies
without PCH. Configuration refreshes the cached mep executable while preserving
explicit command-line overrides. The umbrella header's standard-header names
render as code rather than unsupported Doxygen HTML tags.

Twenty-six dataset-preparation blocks now locate the source root, inherit the
selected pkg-config path and reuse cached source archives. HDF5 C++ and a
versioned Python 3.12 support their native preparations without replacing the
website's Python. The style reference subprocess alone receives the native
wheel-library path; mep and native compilation retain their own loader setup.
The VQA preparation command includes its required original-image input path.

Verification in `build/docs-header-fix/`: normal `just docs` succeeds with
report execution ON and failure suppression OFF, producing 4,466 pages for
289 reports with zero failed blocks. Their statuses contain 1,535 successful
blocks and 37 skipped blocks. The initial repair run re-executed 282 stale
reports; focused reruns resolved the remaining preparation failures before the
final full-site build. `just test-docs` passes ten real Doxygen auditor tests
and 39 report toolchain/path/export regressions. Installed C++20/C++23 header
programs compile, link and execute using only pkg-config flags; real mep canaries
pass with PCH on/off, and explicit/stale mep-cache cases pass. Fresh umbrella
coverage is 1/1 definitions with a passing namespace-alias clangd hover and no
unsupported-tag warnings from that header.

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


Native data preparation (2026-10-30): the historical `prepare_cnn_*.py` and
`prepare_nn_*.py` entrypoints mentioned above have been retired. Canonical
`dm::datasets::prepare_example_data()` and the installed `datamunge-prepare-data`
CLI own downloads, checksum checks, archive extraction, metadata and tensor NPZ
preparation. Reports call the library directly, using the shared data cache.
Independent framework oracle scripts remain optional development tools; report
data preparation does not invoke them or install Python environments.

Native report data verification (2026-10-30): 17 adapter names now cover original
source download, streaming checksums, ZIP/tar/gzip ingress, MAT/HDF5 metadata,
source-faithful subset selection, image/audio/video preprocessing, and NPZ data.
Retired all 18 Python preparation entrypoints; compatibility C++ CLIs delegate
to the installed library. Added native IDX writing, explicit NPY Fortran/endian
options and compressed NPZ writing. Required dependencies and the native CLI are
in the C++ Nix shell, with an explicit minimal-build option.

Evidence: `build/native-data-docs/coverage.json` (145/145 definitions),
`hover.json` and `array-hover.json` (six API hovers), nine compiled/run public
examples, 26 native data/format tests, 15 docs workflow checks, and
`data-ingress/summary.json` (69 original preparation/loading blocks from35 reports,
zero failures after the CIFAR acquisition fix). Source preparation was exercised
from empty caches with genuine pinned sources, including the3GB NYU archive;
pose and VQA preparation also passed offline with no programs on PATH, including
VQA paths containing spaces. The minimal build retains compressed NPZ support.
The Nix package and normal C++ shell build and run the installed CLI.

Every canonical MNIST/CIFAR read now fetches absent sources in C++ (163 blocks
checked). Saliency data uses native MNIST preparation. Shared interpretability
checkpoints declare their original C++ producer through `Requires` metadata;
the website validates dependencies, runs producers first and propagates failures.
The original independent NumPy-written interoperability fixtures remain in tests;
optional framework oracle generators are excluded from report data preparation.
Full unrelated model-training reports were not rerun as an acceptance gate;
verification focuses on data ingress and the affected checkpoint dependency.

Final acceptance also ran the genuine vanilla-saliency checkpoint producer and
SmoothGrad consumer in declared dependency order (both passed), plus the standard
`just test-docs` gate (10 auditor fixtures and16 workflow/export tests).
`just docs -DDOCS_MEPML_RUN=OFF` built4,472 pages from289 stored reports;
69 input blocks/35 reports and the checkpoint pair were executed separately.
This is an export/site build check, not a claim that all model training was rerun.


## CNN figure API expansion and report pass (2026-10-30)

Completed the canonical confusion-matrix SVG API in `nn/metrics.hpp` and
`nn/confusion_plot.cpp`: owned integer-count import, exact true-row orientation,
raw and three normalized scales, customizable class labels/colors/cell/font sizes,
XML escaping, accessible descriptions and checked local-file output. Native tests
cover exact counts, normalization/empty support, noncontiguous ownership, invalid
inputs/settings and file failures. `just test-nn-metrics-plot` runs that suite and
adjacent training/metric regressions.

Architecture rendering now fits its canvas and each node's rotated labels, and
packs parameter-only preprocessing chains near their earliest consumers. Added a
multiple-input composite-stage overload and trace-only mobile/dense/recurrent/PWC
annotations, preserving real branches and distinct reachable parameter counts.
Regression checks cover compact geometry, transformed parameter leaves, shared
multi-input gradients and expanded-versus-grouped PWC paths.

Validation: 71 focused native tests and five report/website tests pass. All 27
plotting-header examples and five changed model examples compile and execute;
`metrics.hpp` and `model_plot.hpp` compile standalone. Fresh Doxygen passes all
348 definitions across the two plotting headers, three implementation files and
five annotated model headers; eight clangd hovers pass. Added the newly completed
metrics header and confusion renderer to `docs/documented-paths.txt`.

CNN sections 1.2–1.11 now contain 146 architecture SVGs and 110 native annotated
confusion images. All 119 reports export to Markdown and current website HTML;
996 local media references resolve and 241 tables have consistent column counts.
Rendered SVG bounds, forward edges and label/glyph separation pass; representative
previews were visually reviewed. Browser UI verification is not claimed. Detailed
execution hashes, SVG audits, 32 compiled examples, exports and fresh Doxygen XML
are under `build/cnn-figures-expansion/`; scope and frozen-count distinctions are
recorded in `docs/mepml/nn-conversion-plan.md`.
