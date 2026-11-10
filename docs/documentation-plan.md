# Doxygen documentation completion plan

The goal is useful documentation at every maintained definition, including
private helpers, enums and their values, classes, namespaces and modules,
constructors, functions, aliases, fields, constants, and static variables.
Doxygen comments in the C/C++ core are canonical. Comments must explain actual
behavior so that navigation and symbol help (`K` in mep) answer how to use a
symbol without reading its implementation.

## Completion checklist

- 2026-11-10: DDPM native continuation (2026-11-10): independent NumPy verifies all 28
source archives / 2,781,150 values and every captured optimizer write. It checks
1,097,600 original pixels against all four raw MNIST IDX banks, all 1,400 labels/
source IDs and 89,600 continuous half-pixel resized pixels; no threshold or
substitute pool is introduced. All 288 Double schedule values, source Float32
time features/coefficients, 3,000 batch rows, 397,894 forward values, 539,392
parameter partials, 528,384 exact Adam-updated weights and all 5,504 parameter
coordinate finite differences pass. Manual dense/SiLU pullbacks and all 12,800
first-pixel input-gradient values/directional finite differences pass. All four
ancestral banks (three epoch16 and final32) preserve 128 batched transitions,
655,360 recorded field values and bit-exact reverse formulas on every source
state. Every last transition ignores innovation exactly. Maximum forward/
parameter-partial errors are 1.452e-6/2.218e-8; parameter FD1.394e-11 and input
FD1.178e-10. No native model math, external autodiff or refitting enters this audit.

Canonical `nn/diffusion.hpp` adds CosineDiffusionSchedule, DiffusionExtract,
DiffusionTimeEmbedding, DiffusionNoised, DdpmEpsilon and DdpmReverse with flat
owning records. The configurable cosine schedule owns detached Double tables
without a mutable cache. Integer times are structural. Source Float32 phase/
feature and extracted coefficient rounding survives Double promotion. Live
first/higher applicable state, noise, table and parameter paths survive parent/
store collection. The source72->32SiLU->32SiLU->64 MLP creates kernels/biases
in source order: 5504 values/six leaves, seed11003. No fit, optimizer, noise/timestep sampler or
complete sampling loop is hidden in these operations. Fixed posterior variance,
raw versus clipped x0, nonsmooth clipping/sqrt conventions, last zero-innovation
step, invalid geometry/types/options, finite overflow, strides, ownership and
warmed store/RNG immutability are explicit and tested.

Fourteen new native tests and 638 selected C++ tests, 331 Python checks,
2,783 R expectations in 66 files and 65 metadata/toolchain checks pass. All 33
new definitions, nine compiled examples, nine clangd hovers, checkout/installed
standalone headers and Python/R help pass; affected coverage523/523. Core
delegation and the complete editable C++ fitting/sampling program reproduce all
28 source archives plus 288 update archives / 1,056,864 values exactly. Histories,
batch orders, preparation manifest, every non-timing metric and all six original
PNG files match byte for byte. The program keeps every96write and32step loop
editable and uses canonical preparation/readers/resize/image-grid operations.

The fixed-final source outcome is retained: test epsilon MSE .838227928 versus
Gaussian pixel-prior .233514349 and zero predictor1.014657692. Clipped x0 MSE
is .640106230; actual time-conditioning effect .510688901. Terminal alpha_bar
2.369970271e-6 and average conditional terminal-prior KL6.619617340e-5 per image
are source diagnostics, not a trained likelihood or sample-quality benchmark.
Preparation records the existing C++ balanced distinct-image seed1931 selection;
there is no claim of TensorFlow-identical row selection/initialization.

Report narrative and native figures remain pending. Begin with a concept diagram
and deterministic noising/reverse/clipping worked example, then actual5504value
architecture, all schedule rows and source provenance, full96write program,
epoch/final samples, trajectories, denoising cases, actual time/input-gradient
controls and baseline shortfall. Verify plot values, bounds, palettes, layout and
raw/site export assets before retirement. The original executable and shared
legacy ddpm.hpp remain; that helper still serves other diffusion originals.
Evidence: scripts/check_ddpm.py and build/nn-report-conversion/ddpm{,_core,_report}/.
All416 original sources and earlier CNN retirement revalidation remain in scope.

- 2026-11-10: Three-family flow-density report completed (2026-11-10). The independent
NumPy audit reconstructs every source triple, erfc-Phi component/jitter, final
Float32 observation and source ID: 22,272 data values. It checks 255,957 forward
values, 39,900 parameter partials, all 144 Adam writes / 38,304 exact updated
weights, all 798 parameter-coordinate finite differences, 4,608 batch rows,
1,536 forward and 12,288 inverse Jacobian entries, and all 3,072 prior samples.
The complete editable program and core delegation reproduce all 66 archives /
433,485 values plus 432 phase archives / 76,752 values exactly; every source
history, batch order, protocol, sample-diagnostic JSON and summary row matches.

Canonical `nn/flow_study.hpp` supplies pure FlowStudyPoints,
FlowStudyLogProbability and FlowStudySampleDiagnostics with owning flat records.
Reuse RealNvpDensity/RealNvpInverse rather than duplicating the model. Data,
truth and diagnostic outputs are deliberately detached; the caller owns all
random banks, IDs, three separate stores/Adam states and complete fitting loops.
Three-coordinate Box-Muller grouping, source erfc tails, half-open checker
support, legitimate negative-infinite checker log densities, stable eight-term
mixture log-sum-exp, inclusive .45 mode radius, serial Double curvature sums
and final observation rounding are documented and tested. No boundary clipping,
extra fitting, sample filtering or changed family budgets is introduced.

The verified report contains 14 executed C++ blocks and 13 inspected native
figures: concept diagram, worked affine coupling/three analytic laws, actual
266-value/17-leaf graph, all pools, histories, paired density maps, samples,
latents, quality counts, both path logdets, signed Jacobian residuals and every
normalization identity. All 98,304 density cells, 16 worked checker cells and
all source plot positions/colors/directions are independently checked. Common
scatter limits [-5,7] retain every row and clear the legends; ring history
upper limit 4.9 keeps its identity baseline outside the legend. Raw Markdown,
HTML and actual website exports contain all 13 verified local assets.

The unchanged outcomes are explicit: banana NLL 2.760811806 versus full
Gaussian 2.837374308, curvature .719403856 versus expectation 1; eight-Gaussian
NLL 3.523986816 versus truth .961499012, all eight modes visited but only
161/1024 samples near a mode; checker NLL 3.139697790 exceeds full Gaussian
3.139320070, with only 436/1024 generated samples on support. Fixed final epoch
three remains selected despite banana validation doing better at epoch two.

Nine new native tests, 624 selected C++ tests, 324 Python checks, 2,727 R
expectations in 65 files and 65 metadata/toolchain checks pass. Documentation
covers all 17 new definitions, five compiled public examples, five clangd hovers,
checkout/installed standalone headers and native Python/R help; combined
coverage is 490/490. Original executable and last-consumer legacy
real_nvp_density.hpp/flow_density.hpp are removed. All normalizing-flow density
reports now replace their originals; full Python/R experiment-loop ports and
all remaining numbered NN sections remain in scope.

DDPM is next. Its original run is frozen in 28 archives / 2,781,150 values,
with nine source pins and two prepared-input hashes. All 96 update probes are
pinned in 288 archives / 1,056,864 values; instrumented capture matches every
source record except training_seconds. Existing saliency MNIST
assets provide 1000/200/200 original 28x28 rows, labels/IDs and the preparation
manifest; that manifest records distinct-image C++ selection seed 1931, not a
claim of TensorFlow-identical row selection. Preserve source half-pixel 28->8
continuous scaling (no threshold), native init 11003/order 11019, 5504 values /
six leaves, all 96 Adam writes including eight-row last batches, 32 cosine
levels, state-only ancestral trajectories, clipped x0 and the zero-noise final
transition. Independently audit this frozen run before canonical APIs/bindings,
full report and retirement. Evidence: scripts/check_flow_density_2d.py and
build/nn-report-conversion/flow_density_2d{,_core,_report}/. All 416 sources
remain in scope; earlier CNN report retirement revalidation is unfinished.

- 2026-11-10: CNF native continuation (2026-11-10): independent NumPy checks all 21 source
archives and 48 captured Adam writes, with 107,367 forward values, 2,600 partials,
2,496 exact updated weights and all 52 model-coordinate finite differences.
Manual dense/trace pullbacks and reverse RK4 accumulation agree with every
source gradient; analytic RK4 matrix composition checks actual discrete-map
Jacobians. All 5,888 procedural data values, 1,536 batch rows, 6,784 actual field
Jacobian entries and 1,024 samples are retained. All 200 field parameter-mode
finite differences, eight field input directions and 52 reverse-solver parameter
finite differences pass. Maximum forward/partial errors are 4.977e-6 / 2.783e-7;
model finite differences 7.499e-11, field finite differences 2.183e-12 and inverse
finite differences 1.901e-10. Double normalization residual is 1.051e-10, a
numerical ODE residual. Refinement checks 4/8/16/32 steps without claiming an
exact discrete-map determinant or inverse.

Canonical `nn/cnf_density.hpp` adds CnfField, CnfIntegrate, CnfDensity/CnfInverse
and flat owning records. Hidden/output kernels initialize first; the actual
shared field is 3->8 tanh->2 scaled .3, 50 values / 4 leaves, plus global scale
for 52 / 5. Source seed 10003, BatchOrder 10021, Float32 dt/half/sixth and stage
times before promotion, source RK4 grouping and four chronological quarter
states are preserved. Stage logdets are CUMULATIVE selected-trace integrals;
endpoint divergence/velocity fields retain live first/higher paths through the
entire unrolled ordinary reverse-mode solver. No continuous adjoint, adaptive
solve, constant-memory claim or hidden fitting loop is introduced.

Only exact mode 0 and deterministic paired mode 3 approximate the flow
likelihood. Modes 1/2 integrate fixed-probe diagnostic contractions and generally
do not approximate full logdet or define a density. Tests verify identical
states but different single-probe integrals, plus the paired mean. Finite RK4
reverse paths and integrated divergence retain numerical inverse/Jacobian/
normalization residuals, including a Float32 coefficient/time refinement floor.

Fourteen new native tests and 615 selected C++ tests, 318 Python checks,
2,670 R expectations across 64 files and 65 inventory/toolchain checks pass.
All 34 new definitions, six compiled examples, six clangd hovers, checkout/
installed standalone headers and Python/R help pass; affected coverage 473/473.
Tests cover all field, fitting and inverse coordinates; both actual maps;
quarter endpoints; analytic scalar-tanh ODE refinement; source rounding;
lifetimes/higher graphs, strides, warmed immutability/RNG, invalid geometry/
options and unsafe math. The core APIs and complete editable C++ program match
all 21 source archives / 163,073 values and 144 update captures / 5,040 values
exactly, with every batch/history TSV, protocol and non-timing metric.

Fixed-final test NLL 2.994471788 only slightly improves full Gaussian
2.994631823, remains above diagonal 2.993811360 and truth 2.796480843. Preserve
the source generic hidden=16 metadata for parity while identifying actual hidden
width 8 and one shared field, rather than four independent field networks.
Evidence: `scripts/check_cnf_density.py`, `build/nn-report-conversion/cnf_density_core/`
and `cnf_density_report/fit.cpp`, `source-parity.json`.

All 18 report C++ blocks execute and all 17 native figures are rasterized and
visually reviewed. A native method diagram, analytically solvable scalar tanh
flow, five-resolution RK4 refinement and cross-coordinate probe example precede
the source experiment. Independent checks verify 2,852 worked curve/check
positions, all four trace-mode integrals, the retained coefficient error floor,
3,072 source observation/sample/latent markers, 1,024 quarter input/state markers,
2,048 cumulative/endpoint diagnostics and their time colours, 128 fitted velocity
segments and every field trace/probe row. All 32,768 density cells retain exact
colours, square geometry and source orientation. Every integrated/actual volume
pair, signed residual and normalization row is retained within verified bounds;
legends remain clear. The executed architecture has 52 unique values, sharing
the same 50 field values across both solver halves. Raw Markdown/HTML and actual
website exports include all 17 verified assets.

The original executable and last-consumer legacy CNF helper are removed;
canonical core APIs remain. Shared real_nvp_density/flow_density helpers still
serve the three-dataset flow_density_2d study, which is next. Preserve its
3-D base draws, erfc-based components/jitter, source IDs, equal per-family fit
budgets and all support/mode/curvature diagnostics. All 416 sources remain in
scope; the migration goal remains active.

- 2026-11-10: CNF source preparation (2026-11-10): original frozen in 21 archives / 163,073
values with nine source pins and no external inputs. Instrumented updates
preserve every source array and all six other files; 144 phase captures retain
5,040 values across 48 Adam writes. Independent audit and canonical APIs remain
pending. Preserve source field 3->8 tanh->2 scaled .3, 52 values / 5 leaves,
seed 10003 with hidden/output kernels created first, BatchOrder 10021 and eight
fixed RK4 steps with Float32 coefficients/times before promotion. Differentiate
through unrolled RK4, without a continuous adjoint. Deterministic paired probes
(1,1) and (1,-1) average to exact 2-D field divergence; numerical integrated
trace and reverse RK4 are approximate discrete-map likelihood/inverse paths.
Keep all four quarter states, cumulative logdets, endpoint divergence/velocity,
actual field/whole-map Jacobians and every sample/normalization check.

The generic source metrics retain hidden=16 despite the actual hidden width 8;
coupling_layers=4 counts quarter diagnostics, not independent field networks.
Preserve source JSON parity while explaining the executed architecture. Final
test NLL 2.994471788 barely beats fitted full Gaussian 2.994631823 and remains
above diagonal Gaussian 2.993811360 and truth 2.796480843. Independent manual
field/RK4 differentiation and matrix composition are next. Evidence:
`build/nn-report-conversion/cnf_density/` and `cnf_density_core/capture-parity.json`.

- 2026-11-10: Planar/radial native continuation (2026-11-10): canonical
`nn/planar_radial_density.hpp` adds pure PlanarFlow/RadialFlow, named
ElementaryFlow, PlanarRadialDensity/PlanarRadialInverse and flat owning records.
The planar map retains source angle/frequency/projection/perpendicular/bias
parameterization and 32-bisection inverse with a single stopped-root and
stopped-denominator Newton correction. First derivatives use that source
implicit rule; higher graphs follow the stopped correction, not an exact
implicit Hessian. Radial retains the stable two-branch quadratic radius and
finite center derivative, including the source tiny-positive-radius rule.
Representability checks reject unsafe frequency, radius, discriminant, factors
or global scale; no clipping, refit or hidden fitting loop is added.

Both source stores remain independent: planar 22 values / 21 leaves, radial
18 / 13, seed 9903, block order 0..3 and global scale last. Meaningful public
factor/coordinate fields replace the archive's scale/shift diagnostic names.
Fourteen new native tests and 601 selected C++ tests, 312 Python checks,
2,584 R expectations across 63 files and 65 inventory/toolchain checks pass.
Tests include both whole-map Jacobians, all 34 leaf NLL finite differences,
all 40 inverse-objective parameter-coordinate finite differences, pure
input/parameter partials, the planar stopped higher rule, radial center and
both inverse branches, lifetimes/higher graphs, strides, warmed immutability,
RNG stability, invalid shapes/dtypes and unrepresentable math. All 35 new
definitions, seven compiled examples, seven clangd hovers, checkout/installed
standalone headers and Python/R help pass; affected coverage is 439/439.

The native overlay and complete editable report program match all 48 source
archives / 281,879 values and all 288 update captures / 3,936 values exactly,
including both explicit 48-write fits, every batch/history TSV, protocol,
non-timing metric, strong-contraction fixture and inverse parameter gradient.
Preserve fixed-final planar NLL 2.991511106 and radial 2.999574184 versus full
fitted Gaussian 2.994631823 and truth 2.796480843; radial underperformance remains.
Evidence: `build/nn-report-conversion/planar_radial_density_core/`,
`planar_radial_density_report/fit.cpp`, `source-parity.json` and
`scripts/check_planar_radial_density.py`.

All 24 report C++ blocks execute and all 23 native figures are rasterized and
visually reviewed. Native method and hand-checkable planar/radial maps precede
the original banana data. Both nine-point contraction fixtures, all 40 inverse
parameter partials and executed architecture counts retain their source values.
Independent checks verify 1,308 worked curve/check positions, 5,376 observation/
sample/latent markers, 54 fixture coordinate markers, 36 segment endpoints,
72 fixture analytic/actual determinants, all 4,608 whole-map volume markers,
2,304 signed Jacobian residuals and 2,048 normalization residuals. All 65,536
density cells retain independently verified colours, square physical geometry
and source orientation. Every source sample and high held-out tail is retained;
legends are clear and bounds contain all data. Raw Markdown/HTML and actual
website exports include all 23 verified assets. The original executable and
last-consumer legacy planar/radial helper are removed; canonical core APIs
remain. Shared affine/flow helpers still serve CNF and the remaining flow study.
Next is CNF density, preserving its numerical RK4 inverse/likelihood and paired
trace semantics. All 416 sources remain in scope; the migration remains active.

- 2026-11-10: Planar/radial independent continuation (2026-11-10): both source variants are
frozen in 48 archives / 281,879 values with nine source pins and no prepared
external inputs. Instrumentation preserves every source archive and all 12 other
files; 288 phase captures retain 3,936 values across 96 Adam writes (48 each).
`scripts/check_planar_radial_density.py` independently checks 170,638 forward
values, 2,040 partials, 1,920 exact updated weights, 34 model-leaf finite
differences, all 11,776 procedural data values and 3,072 batch rows. Both whole
maps' actual Jacobians, all 2,048 inverse samples and normalization identities
pass. Maximum forward/partial errors are 6.306e-6 / 4.517e-7; model finite
differences 5.679e-11 and normalization residual 3.553e-15.

Both nine-point strong-contraction fixtures retain every field and both actual
Jacobians (333 values), including the radial center's finite derivative and the
source planar hinge. The inverse objective uses the original first eight prior
rows per variant; all 40 parameter partials and all 40 coordinate finite
differences pass, maximum error 1.877e-10. Planar inversion is numerical:
32 bisections followed by one Newton correction with stopped root and denominator
for source implicit first derivatives. Do not claim a closed-form inverse or
mathematically exact implicit higher derivatives. Radial inversion uses the
stable two-branch positive quadratic radius, with explicit zero-radius handling.

Preserve four elementary maps plus global scale: planar 22 values / 21 leaves,
radial 18 / 13, each seed 9903 and BatchOrder 9921. Fixed-final test NLL is
2.991511106 for planar and 2.999574184 for radial, versus fitted full Gaussian
2.994631823 and truth 2.796480843: retain the radial shortfall. Canonical APIs,
bindings/docs and the full editable report are next; both original variants and
their helper remain. Evidence: `build/nn-report-conversion/planar_radial_density/`
and `planar_radial_density_core/capture-parity.json`. Shared affine/flow helpers
remain in use; all 416 sources remain in scope.

- 2026-11-10: Spline native continuation (2026-11-10): independent NumPy checks all22 source
archives and48 captured Adam writes, with159,943 forward values,43,900 partials,
42,144 exact updated weights and17 model-leaf finite differences. All5,888
procedural data values,1,536 batch rows, both actual whole-map Jacobians and
all1,024 inverse samples are checked. The272-point source fixture retains
12,784 values, including3,264 input/head objective partials;3,192 finite
differences check rows away from nondifferentiable log-derivative knot branches.
Exact knots/neighbours/tails, positive first derivatives and the closed quadratic
inverse are checked separately. Maximum forward/partial errors are7.986e-6/
5.713e-7, model finite differences1.303e-10, fixture finite differences1.304e-9,
and Double normalization residual6.218e-15. The source Float32-rounded softplus
offset0.5397424101829529 is preserved. Evidence: `scripts/check_spline_density.py`
and `build/nn-report-conversion/spline_density/independent-oracle.json`.

Canonical `nn/spline_density.hpp` adds RationalQuadraticSpline, SplineCoupling,
SplineDensity and SplineInverse with flat owning records. The pure primitive
supports K>=2 bins, configurable symmetric bounds and positive bin/slope floors;
Double offset precision is default and source rounded mode is explicit. Actual
bin sizes come from adjacent source-order cumulative knots. Inclusive endpoints,
right-bin knot ownership, strict identity tails and2c/(-b-sqrt(b*b-4ac)) inverse
root preserve the source mechanism. Nonpositive represented bins, slopes,
discriminants or derivative terms reject. No root search or fitting workflow
enters the core. First/higher input/head/parameter graphs survive collection
within structural branches; public widths/heights have meaningful names rather
than the source archive's scale/shift labels. The compact model preserves
878 values/17 leaves, seed9803 and BatchOrder9821. The shared private dense
conditioner comment now covers its calling models' output widths.

Fourteen new native tests and587 selected C++ tests,305 Python checks,2,538 R
expectations in62 files and65 inventory/toolchain checks pass. Generic2/4/7-bin
identity and nonuniform-head derivatives, custom bounds/floors, explicit rounded
Double precision, both actual Jacobians, all17 leaf finite differences, first/
higher graphs, strides, tails, bin metadata lifetimes, immutability and invalid/
unrepresentable cases are verified. All35 new definitions, six compiled examples,
six hovers, checkout/installed standalone headers and Python/R help pass;
affected documentation coverage404/404.

The native overlay matches all22 original archives/235,867 values and six
other files exactly, plus144 update archives/84,336 values. The complete editable
C++ report program also matches every source array, update, batch/history TSV,
protocol and non-timing metric, including the entire272-point fixture and all48
explicit Adam writes. Final test NLL2.871595860 improves on fitting-only
Gaussian2.994631823, above truth2.796480843. Evidence:
`build/nn-report-conversion/spline_density_core/` and
`spline_density_report/fit.cpp`, `source-parity.json`.

All 15 report C++ blocks execute and all 14 native figures are rasterized and
visually reviewed. A hand-checkable two-bin spline on [-2,2], widths/heights 2
and slopes 1,2,1, maps x=-1 to -1.2 with actual derivative .8. The native
method diagram and executed 878-value architecture precede the full fitting
program. All 272 original fixture rows, 816 residuals and 3,264 input/head
gradient cells retain their source values. Independent checks cover 528 worked
curve/marker positions, 1,088 fixture curve/marker positions, all gradient cell
colours and positions, 3,072 observation/sample/latent markers, 32,768 density
cells and 2,304 determinant markers. Every sample, tail and normalization check
is retained within verified display bounds; raw and website exports include all
14 assets. The original executable and unused legacy spline helper are removed.
Shared affine/flow helpers still serve other originals. Next is the combined
planar/radial source, with both independently fitted variants in scope. All 416
sources remain in scope; the migration goal remains active.


- 2026-11-10: IAF report continuation (2026-11-10): canonical `nn/iaf_density.hpp` adds
IafDensity, IafInverse and a23-field flat owning IafDensityOutput. It reuses
MafConditioner/MafMasks and global flow scaling. Density preserves reversed
block order3..0, two recovered-prefix calls plus a final recovered-input head
re-evaluation per block. Sampling uses one parallel pass per block0..3.
Source initialization9703, BatchOrder9721,466 stored values/17 leaves and all
160 masked coordinates are preserved. Live fields retain applicable first/
higher input/parameter graphs through collection. Nine new tests and573
selected C++ tests,300 Python checks and2,497 R expectations in61 files pass.
All27 new definitions, three compiled examples, three hovers, checkout/installed
standalone headers and Python/R help pass; affected coverage369/369.

All15 report C++ blocks execute and14 native figures are rasterized and visually
reviewed. A hand-set affine IAF checks parallel observation generation, explicit
sequential density recovery and both actual triangular Jacobians before the
original banana pools. Strict masks retain192 independently verified cells;
all4,096 actual conditioner partials include3,072 exact forbidden zeros. The
executed IAF density diagram correctly groups blocks3/2 and1/0,232 values each,
plus global scale2. All3,072 observation/sample/latent markers have independently
verified source positions. Both128-square density maps retain all32,768 cells,
shared colours and physical orientation. Every forward/inverse determinant
pair, normalization identity and original sample is retained with clear bounds.

The native overlay and full editable program match22 original archives/226,631
values and144 update captures/44,784 values exactly, including all48 Adam writes,
batch/history TSVs, protocol and every non-timing metric. Fixed-final test NLL
2.879989624 improves on fitting-only Gaussian2.994631823, above truth2.796480843.
Raw Markdown/HTML bundles and actual website figure copies pass. The source
remains a standalone affine exact-density adaptation, with no VAE/ELBO claim.
Evidence: `build/nn-report-conversion/iaf_density_core/`, `iaf_density_report/`
and `scripts/check_iaf_density.py`.

The original executable and last-consumer `common/iaf_density.hpp` and
`common/maf_density.hpp` are removed. Canonical native MAF/IAF headers remain;
shared original affine/flow helpers are still used by other flows. The earlier
independent checkpoint below preceded this completed API/report. Next is spline
density: original frozen in22 archives/235,867 values, nine source pins, no
external prepared inputs. Preserve source4-bin rational-quadratic couplings,
C1 identity tails, analytic quadratic inverse,272-point fixture with knot/tail
and input/head derivative checks,878 values/17 leaves, seed9803 and order9821.
Independent audit, canonical spline API/bindings and full report are pending.


- 2026-11-10: IAF independent continuation (2026-11-10): all22 pinned source archives and
all48 captured Adam writes pass independent NumPy. The audit composes analytic
Jacobians and manual parameter derivatives through sequential sample-to-base
recovery in reversed block order, including the source's explicit final head
re-evaluation on each fully recovered input. Base-to-sample inversion uses
one parallel conditioner call per block in order0..3. All166,727 forward
values,23,300 partials,22,368 exact updated weights,17 leaf finite differences,
160 additional masked-coordinate finite differences,40,704 forbidden conditioner
partials,1,536 batch rows and all1,024 samples are checked. All7,680 masked
partials stay zero and their stored weights remain bit-exact across48 writes.
Maximum forward/partial errors are2.618e-5/3.176e-7; finite differences4.983e-11,
Double normalization residual4.441e-15. Both whole-map determinants agree with
actual Jacobians and remain spatially varying. Source test NLL2.879989624
improves on fitting-only Gaussian2.994631823, above truth2.796480843.

Evidence: `scripts/check_iaf_density.py`,
`build/nn-report-conversion/iaf_density/independent-oracle.json` and
`iaf_density_core/capture-parity.json`. All22 original archives/226,631 values
and six other files are unchanged by instrumentation;144 pinned phase archives
retain44,784 values. Ten source files are pinned, no external inputs. Canonical
IAF APIs, bindings/docs and the editable report remain pending. Preserve466
stored values/17 leaves,160 masked kernel coordinates, reversed forward warmup
with seed9703, BatchOrder9721, three epochs/48 Adam(.003) writes and the full
original protocol. Reuse native MafConditioner/MafMasks and flow scaling;
retain all live recovered stage inputs/means/scales/determinants and applicable
first/higher graphs. Density uses two prefix calls plus one final re-evaluation
per block; sampling uses one parallel call. Keep the original unchanged until
its API/program/report and exported figures pass. The shared original MAF
helper remains needed by IAF; other flows still use affine/flow helpers.


- 2026-11-10: MAF report continuation (2026-11-10): all15 C++ blocks pass and14 native
figures are rasterized and visually reviewed. A worked strict Gaussian
conditioner with explicit dependency-ordered inverse and actual triangular
Jacobian precedes the original banana pools. Both degree orders retain all192
mask cells; independent colour/position checks verify every entry. All4,096
actual final-test conditioner partials are displayed, including3,072 exact
forbidden zeros. All3,072 observation/sample/latent plot markers have verified
source coordinates. The executed network counts466 stored values including160
masked coordinates. Every observation, inverse sample, density cell, forward/
inverse determinant pair, normalization identity and original fitting write
remains visible. Both128-square density maps share independently checked
colours and physical orientation; equal coordinate axes retain all tails.

The complete program matches22 original archives/226,631 values and144 update
archives/44,784 values exactly, with original batch/history TSVs, protocol and
every non-timing metric. Final test NLL2.878228426 improves on fitting-only
Gaussian2.994631823, still above truth2.796480843. Raw Markdown/HTML figure
bundles and actual website copies pass. Metadata/toolchain checks pass65/65.
Evidence: `build/nn-report-conversion/maf_density_report/` and its
`source-parity.json`, `figure-geometry.json`, `plot-values.json`,
`density-colours.json`, `mask-and-points.json` and `website-export.json`.
The original executable is removed. Retain `common/maf_density.hpp` for IAF,
and shared affine/flow headers for remaining originals. The earlier native
checkpoint below preceded this completed report.

Next is IAF density: its unchanged original run is frozen in22 archives
(226,631 values), with ten source pins and no external prepared inputs. It uses
the same strict conditioner in the opposite computational direction: sequential
density recovers coordinates in reversed block order, then re-evaluates heads
on each recovered input for its determinant; inverse sampling is parallel in
block order0..3. Preserve source466 values/17 leaves, reversed forward warmup,
seed9703 and BatchOrder9721. Independent audit and canonical IAF API/report
are pending. This standalone exact-density affine adaptation makes no VAE/ELBO
claim; keep its distinction from the original paper's sigmoid gating explicit.


- 2026-11-10: MAF native continuation (2026-11-10): independent NumPy checks all 22 original
archives/226,631 values, 48 Adam writes and all 144 update archives/44,784 values.
It verifies 166,727 forward values, 23,300 partials, 22,368 exact updated weights,
17 leaf finite differences, 160 masked-coordinate finite differences, 40,704
strict forbidden conditioner Jacobian entries and all 1,024 inverse samples.
Every forbidden kernel coordinate has zero partials and remains unchanged at
all 48 writes (7,680 checks each). The degree inventory has472 values. Analytic
matrix composition verifies parallel density and the sequential inverse in
alternating dependency order. Maximum forward/partial errors are1.907e-5/
2.849e-7, leaf finite-difference error8.547e-11 and Double normalization
residual2.665e-15. Original procedural rows/seeds are unchanged; no external
prepared inputs. Evidence: `scripts/check_maf_density.py` and
`build/nn-report-conversion/maf_density/independent-oracle.json`.

Canonical `nn/maf_density.hpp` adds MafMasks, MafConditioner, MafDensity and
MafInverse. It reuses native degree-mask comparison, masked dense projections
and global flow scaling. Flat owning records retain all23 source stage/input/
mean/log-scale/determinant fields and applicable first/higher graphs after
collection. Source466 values/17 leaves,160 masked coordinates, seed9603 and
order9621 are preserved. The parallel density pass evaluates each conditioner
once; the inverse recomputes it on each recovered prefix and accumulates log
scales in dependency order. Warm inference neither writes leaves nor draws RNG.
Ten new tests and564 selected C++ tests,297 Python checks,2,445 R expectations
in60 files and65 inventory/toolchain checks pass. All38 new definitions,
seven compiled examples, seven hovers, checkout/installed standalone headers
and Python/R help pass; affected documentation coverage342/342.

The native overlay matches all22 original archives and six other files exactly,
plus144 update captures. The complete editable report program also matches every
array, update, batch/history TSV, protocol and non-timing source metric exactly.
It retains actual conditioner Jacobians, both whole-map Jacobians, source masks,
all48 explicit fitting writes and all1,024 inverse samples. Final test NLL
2.878228426 improves on fitting-only Gaussian2.994631823 while truth remains
better at2.796480843. Evidence: `build/nn-report-conversion/maf_density_core/`
and `maf_density_report/fit.cpp`, `source-parity.json`.

The MAF report narrative and figures remain pending. Next include a worked
hand-set strict conditioner, explicit dependency-ordered inverse and Jacobian,
native method/executed architecture diagrams, source degree/mask inventory and
actual forbidden conditioner partials. Use the verified full program, retain
all density cells, samples, per-point determinants and normalization identities,
and verify raw/site exports before retirement. The original MAF executable is
unchanged. Retain `common/maf_density.hpp` for IAF and shared affine/flow helpers
for other originals. The completed Glow report below is the latest retirement.


- 2026-11-10: Glow native continuation (2026-11-10): independent NumPy checks all 23 pinned
archives and all 48 Adam writes, with 126,023 forward fields, 14,800 partials,
14,208 exact updated weights, 28 model-leaf finite differences and four extra
signed-matrix finite differences. It verifies all 5,888 procedural data values,
1,536 batch rows, both actual Jacobians and all 1,024 inverse samples. Additional
captures preserve every uncalibrated leaf and all four actual first-32-row
calibration inputs (five archives/552 values). The independent serial Double
mean/variance calculation reproduces all 16 ActNorm parameters exactly. The
negative-determinant fixture checks log|det|, inverse and weight partials.
Evidence: `scripts/check_glow_density.py` and `glow_density/independent-oracle.json`.

Canonical `nn/glow_density.hpp` adds pure FlowActNormCalibration proposals,
explicit named FlowActNorm, signed FlowMix2D, the compact GlowDensity and
GlowInverse. No implicit calibration, statistics updates or fitting/sampling
workflows enter the model API. Both matrix signs are supported; singular or
unrepresentable squared determinants reject. Live flat fields retain first/
higher graphs after parent/store collection. Source rotation initializers round
to Float32 before promotion and consume no RNG. The forward has no extra final
global scale. Ten new tests and 554 selected C++ tests, 293 Python checks,
2,292 R expectations in 59 files and 65 inventory/toolchain checks pass.
All 43 new definitions, seven compiled examples, seven hovers, native Python/R
help and standalone checkout/installed headers pass; affected coverage304/304.

The native overlay matches all 23 original archives/184,017 values and six
other files exactly, all 144 update archives/28,464 values and all five
calibration archives/552 values. The complete editable C++ report program also
passes those exact array/calibration comparisons, batch/history TSVs, protocol
and every non-timing source metric. It keeps the signed fixture, once-only
first32 sequential calibration, all 48 writes, snapshots, both actual Jacobians
and all inverse samples visible. The original final test NLL2.866107941 improves
on fitted Gaussian2.994631823 while truth remains better at2.796480843.
Evidence: `build/nn-report-conversion/glow_density_core/` and
`glow_density_report/fit.cpp`, `source-parity.json`, `calibration-parity.json`.

The completed Glow report executes all 15 C++ blocks and renders 14 native
figures. Worked population-moment/ridge and signed-matrix examples precede the
original data, once-only sequential calibration and complete editable fitting/
sampling workflow. The executed architecture retains 296 values in 28 leaves;
all 768 observations, 1,024 inverse samples, actual forward/inverse Jacobians,
per-point determinant pairs, normalization and both density grids are retained.
The four actual calibration inputs match source captures exactly; all sixteen
plotted coordinate moments are independently checked. Equal coordinate scales
retain every sample, including the lower tail beyond the original plotting
window. All 32,768 density cells have independently verified common colours and
physical orientation. Every figure is rasterized and visually reviewed; raw
Markdown/HTML bundles and actual website copies pass. Metadata checks pass65/65.
Evidence: `build/nn-report-conversion/glow_density_report/`, including
`source-parity.json`, `calibration-parity.json`, `figure-geometry.json`,
`plot-values.json`, `density-colours.json` and `website-export.json`.
The original executable and last-consumer `common/glow_density.hpp` are removed.
Shared `common/real_nvp_density.hpp` and `common/flow_density.hpp` remain for
other originals.

Next is MAF density: its unchanged original run is frozen in 22 archives
(226,631 values), with nine source pins and no external prepared inputs.
Preserve its strict degree masks, parallel density and sequential inverse,
466 values/17 leaves, initialization9603 and BatchOrder9621. Capture all48
Adam writes and independently audit the actual conditioner Jacobians, masked
partials and inverse dependency order before adding canonical APIs/report.


- 2026-11-10: RealNVP report continuation (2026-11-10): all 13 C++ blocks and 12 native
figures pass with complete visible fitting and inverse sampling. It reproduces
all 21 original archives (142,951 values), all 144 update archives (25,584 values),
batch/history TSVs, data protocol and every non-timing metric exactly. Canonical
`nn/affine_flows.hpp` adds FlowDiagonalScale, bounded RealNvpCoupling, the compact
RealNvpDensity and RealNvpInverse, with flat owning live stage/scale/shift/logdet
fields. Inverse determinants are computed on the inverse path's own points.
The source .8*tanh bound, 266 values in 17 leaves, initializer 9403 and order
seed 9421 are unchanged. Nine new tests and 544 selected C++ tests, 289 Python
checks and 2,240 R expectations in 58 files pass, including first/higher paths,
strides, global scaling, invalid inputs/overflow, immutability and collection
lifetimes. All 33 new definitions, seven compiled examples, seven hovers,
checkout/installed standalone headers and Python/R help pass; affected coverage
is 261/261. Metadata/toolchain checks pass 65/65.

The report shows a worked affine Jacobian, analytic banana coordinates, the
executed 266-value network, all 768 observations and 1,024 inverse samples,
both density grids (32,768 cells), all 1,152 forward/inverse determinant pairs,
residuals and normalization identities. Equal coordinate scales retain every
tail point; common density colours and physical grid orientation are checked.
All 12 figures are rasterized and visually reviewed; raw Markdown/HTML bundles
and actual website copies pass. The original final test NLL 2.879116774 is better
than the fitting-only full Gaussian 2.994631823, still worse than analytic truth
2.796480843. Evidence: `build/nn-report-conversion/real_nvp_density_report/`,
`real_nvp_density_core/` and `scripts/check_real_nvp_density.py`.
The original executable is removed. Retain `common/real_nvp_density.hpp` and
`common/flow_density.hpp`: other flow originals still consume their helpers.
The earlier independent-audit checkpoint below preceded this completed API/report.

Next is Glow density: its unchanged original run is frozen in 23 archives
(184,017 values), with nine source pins and no external data requirement.
Instrumentation preserves all 23 archives and six other files exactly and
captures 144 phase archives (28,464 values). Keep the once-only sequential
ActNorm calibration from the first 32 fitting points, signed-mixing fixture,
296 values/28 leaves, initialization 9503 and BatchOrder 9521. Independent audit,
canonical calibration/mixing/flow operations and its report remain pending.


- 2026-11-10: NICE report continuation (2026-11-10): all 12 C++ blocks and 11 native figures
pass, with complete visible fitting and latent-only inverse sampling. It matches
all 21 source archives (117,586 values), 144 update archives (19,056 values),
batch/history TSVs, the original protocol and every non-timing metric exactly.
Native cumulative sums preserve the original serial Double fitting-only moments
and summaries; no new baseline helper was necessary. Worked nonlinear coupling
and analytic banana examples precede the exact original procedural pools.
The actual 198-value architecture, every observation/sample, both density grids,
both actual Jacobians and all normalization residuals are rendered natively.
All 768 observations, 1,024 inverse samples and transformed test points remain
within equal coordinate scales; the high held-out tail hidden by the source's
plotting window is visible. Every density-map cell is retained (32,768 total)
with common colour limits and correct physical orientation. Readable method
labels, clear legends and the fixed-final negative result are verified.

All 11 figures are rasterized/visually reviewed and both raw export bundles plus
actual website figure copies pass. C++535/Python285/R2165 and metadata65 pass;
28 definitions, ten compiled examples, seven hovers and checkout/installed
standalone headers remain verified. Evidence:
`build/nn-report-conversion/nice_density_report/`, `nice_density_core/` and
`scripts/check_nice_density.py`. Original NICE executable and its last-consumer
`common/nice_density.hpp` are removed. The shared `common/flow_density.hpp`
remains for eight other flow originals. The earlier native-only checkpoint below
records the API work that preceded this report.

The next source is RealNVP density: unchanged original run frozen in 21 archives
(142,951 values), eight source pins and no external prepared inputs. Capturing
all 48 updates preserves those archives and six other files exactly, adding
144 pinned phase archives (25,584 values). Its independent audit now passes; affine coupling
API and report are pending. The source uses four bounded affine couplings,
266 values/17 leaves, store seed 9403 and BatchOrder seed 9421 on the same banana
protocol. Keep its original input-dependent determinant and inverse-path checks.


- 2026-11-10: NICE native flow API and independent original audit complete.
  `nn/normalizing_flows.hpp` and `nn/nice_density.hpp` provide explicit banana
  preparation/truth, Gaussian change-of-variables densities, additive couplings,
  the compact 198-value NICE map and exact-form inverse. Flat fields retain live
  first/higher graphs after parent/store collection; warmed inference is immutable.
  All 28 definitions, ten compiled examples, seven hovers, native Python/R help
  and checkout/installed standalone headers pass; affected coverage is 228/228.
  Source parity is exact across 21 archives (117,586 values), six other files
  and 144 update archives (19,056 values). Independent NumPy verifies all 48
  Adam writes, 17 finite differences, both Jacobians and 1,024 inverse samples.
  C++535/Python285/R2165 in 57 files and metadata65 pass. The report still needs
  authoring; the original NICE executable and shared flow harness remain.
  Evidence: `build/nn-report-conversion/nice_density_core/`,
  `nice_density/independent-oracle.json` and `scripts/check_nice_density.py`.

- 2026-11-10: WaveNet report continuation (2026-11-10): the13-block report now keeps all27
Adam writes, warmup32 and both complete sampled-feedback loops editable. It
matches all22 source NPZ archives/9283192 values,81 update archives/273699 values,
source manifest and batch/history TSVs exactly. All eight WAVs are byte identical
and independently match7170 PCM16 samples. Markdown/HTML players, adjacent WAV
bundles and actual website image/audio copying pass. The website now tracks both
direct/nested audio assets for staleness and copying, with two regression tests.
All11 native figures are inspected: worked RF5/cache and256-code codec checks,
method/executed5068-value architecture, all three384-code held-out references,
all three epochs/baselines,224 losses per method, full64x256 structural partials,
both2049-code cold streams, all three640-code continuations and369 probe errors.
Every displayed waveform position is retained; offsets/physical time are explicit.
The architecture uses readable executed composite stages with actual shapes.

The negative result remains: final7.96727 bits/mu-law sample is worse than
fitting-only unigram6.94609 and transition6.23728. All real FSDD source/speaker
IDs, strict next-code shifts, case-major midpoint MT draws, observed128-code
prefixes and RF33 checks remain. Native per-point Double loss plus serial scalar
summaries preserve original17-digit history values without changing fitting.
Nine new cache tests and527 selected C++ tests,281 Python tests,2119 R expectations
in56 files and65 inventory/toolchain checks pass. All14 new definitions,6 compiled
header examples,6 hovers and standalone checkout/installed headers pass; affected
coverage202/202. Independent PCM/NumPy still verifies every5632 sampled code from
RF33 contexts. Evidence: `build/nn-report-conversion/wavenet_report/`,
`wavenet_core/`, `wavenet/independent-oracle.json` and
`scripts/check_generative_wavenet.py`. Original executable and the now-unused
`common/wavenet_generation.hpp` are removed. The older `common/wavenet_stack.hpp`
shim still has SWIG consumers and remains until those consumers are migrated.

The next independent source is NICE density, whose unchanged48-update procedural
banana run is frozen in21 archives/117586 values with8 source pins and no external
input requirement. Instrumentation preserves all21 archives and6 other files
exactly, capturing144 update archives/19056 values. Its independent audit and
native data/coupling/inverse APIs remain pending. The in-process synthetic density
is explicit, not a substituted real dataset. Existing AutoRec/GAN diagnostics and
character_rnn recipe questions remain pending; ImageGPT's original prepared
palette corpus is also absent from the local cache, so its recipe needs recovery
before that source can run. No substitute assets/splits are used.


- 2026-11-10: WaveNet cache continuation (2026-11-10): native `nn/wavenet_generation.hpp`
adds owning `WaveStackCache` with next/reset/consumed/clone. The constructor copies
required finite Float32 CPU leaves from a warmed store, accepts strides, ignores
unrelated leaves and retains no Tensor/store alias. Each block keeps its OWN
input ring at its dilation; previous-code input taps, scalar Float32 accumulation,
tanh/filter divided by1+exp gate, ordered skips and absent final residual match
the source. Ring/previous-code writes commit only after finite candidate state/head
and owning output allocation; invalid code and arithmetic failure preserve state.
Snapshots, resets, C++ copies and host-language clones are independent; outputs
are owning detached Float32 heads, with no sampling or training hidden in the cache.
R calls its reserved-keyword method `next_code`; C++/Python use `next`.

Canonical WaveStackConfig/Field/Conv/Output/Stack/Nll are also exported before
legacy SWIG imports. Returned gate vectors copy owning Tensor handles, preserving
live graphs after parent/vector/store collection. Thin Python/R documentation
covers the explicit architecture, Float32 cache and warmup choices. Native codec,
audio ingress and PCM16 writer already exist and are reused without duplication.
Nine new C++ tests cover repeated cycles/ring wraps, snapshots, strides, retained
heads, forks/reset, missing/dtype/shape/nonfinite inputs and transactional failed
steps, including a finite ring candidate followed by an overflowing output.
Selected527 C++ tests,281 Python tests,2119 R expectations in56 files and63
metadata checks pass. All14 new definitions,6 compiled examples,6 hovers and
checkout/installed standalone headers pass; combined affected coverage202/202.
The final R help install uses the checkout pkg-config/rpath and its focused130
expectations pass after installation; Python/R help and copied gate lifetimes pass.

Native delegation reproduces22 source archives/9283192 values,16 other files
(including all eight WAVs) and81 captured update archives/273699 values exactly.
The prior independent PCM/NumPy audit verifies all27 writes and5632 sampled codes
from full RF33 contexts. `wavenet_core/nll-probe.cpp` additionally proves source
17-digit history/test summaries: native per-point Double max+logsumexp-minus-target,
then serial row/time summation to preserve legacy scalar formatting. Whole-array
native reductions differ only in the last bits; fitting retains native Float32
WaveNll with start32. Evidence: `build/nn-report-conversion/wavenet_core/`,
`wavenet/native-progress.json`, `wavenet/cache-design.md` and the independent oracle.
The complete report with editable fitting/sampling loops, native figures and all
audio exports remains pending. Original WaveNet source/helpers are retained;
the older stack shim still has SWIG consumers and cannot yet be removed.


- 2026-11-10: PixelCNN++ report continuation (2026-11-10): the complete native12-block
report retains the48-step Adam loop and explicit component/quantized-R/G/B
raster sampler. All24 original archives/1223447 values and144 captured update
archives/135408 values match exactly, including batch/history TSVs. All11 figures
and both exports are reviewed, with516096 original gallery channel values exact.
Two executed worked checks precede real CIFAR inputs: all256 logistic bins need
infinite endpoint tails, and triangular RGB means use preceding observed channels.
The report separately verifies all nine extreme component/channel distributions,
12288 causal entries,64 whole-pixel future flips and448 permanently masked
coordinates. Sampling uses one component for all three channels, already quantized
R/G for subsequent means, and full case/pixel/stream midpoint MT banks. Prefix
completions retain all32 observed whole RGB pixels. The component-frequency plot
retains the dominant.817 observed fraction without clipping; no semantic labels
are assigned to mixture components.

The original32x32 CIFAR bytes use exact integer4x4 area means, half-up codes and
one Float32 signed representation cast. Fixed source rows are128 fit/32 validation/
32 test, with unused labels retained and no TF initial arrays present. Final test
7.48424 bits per8-bit RGB channel improves on fit-only histogram7.91458 and uniform8;
unfiltered samples are noisy and no original-resolution benchmark claim is made.
Native `nn/logistic_pixels.hpp` exposes preparation, exact-bin coupled likelihood
and compact context density as flat owning live records. Tests518 C++,277 Python,
1989 R expectations/55 files and63 inventory/toolchain checks pass. All29 definitions,
6 compiled documentation examples/clangd hovers and standalone checkout/installed
headers pass; combined coverage188/188. Independent NumPy covers379469 forwards,
78420 partials,29 finite differences,48 exact Adam writes and9792 sampled channels.
Evidence: `pixel_cnn_pp_report/`, `pixel_cnn_pp_core/`, `pixel_cnn_pp/independent-oracle.json`
and `scripts/check_pixel_cnn_pp.py` under `build/nn-report-conversion/` as applicable.
The original executable and now-unused `common/pixel_cnn_pp.hpp` and
`common/gated_pixel_cnn.hpp` are removed; shared binary helpers still have consumers.

WaveNet is the next independent source: its unchanged27-update native run is
frozen in22 archives/9283192 values,13 source pins,2 native header pins and31
original/prepared input pins. Its27 original FSDD WAV recordings, release archive,
manifest/source IDs and72/12/24 speaker-disjoint windows remain intact. Existing
canonical audio ingress, stack, tensor codec and PCM16 writer can be reused.
Instrumentation preserves all22 archives and16 other files exactly and captures
81 gradient/update/iteration archives/273699 values. `scripts/check_generative_wavenet.py`
independently verifies27675 original PCM samples,55296 shifted code values,1152
prompt codes,7618560 forwards,163356 partials,34 finite differences and27 exact
Adam writes. It recomputes every one of5632 sampled heads from RF33 contexts,
reproduces all5632 draws and validates7170 PCM16 samples in all eight generated/
reference WAVs. Maximum forward/partial/next-head errors are2.55e-8/7.61e-9/3.06e-8.
All16384 encoded-input causal derivatives retain exact zeros outside RF33 and
in the future. Test7.96727 bits/mu-law sample remains worse than both fitting-only
Laplace baselines. Initial zero post biases can sit on ReLU boundaries; finite
differences use fitted smooth coordinates while initial VJPs retain derivative0
at ReLU ties. Promote the owning per-layer synthesis cache next; exact source
Float32 scalar accumulation and rings of each block's own inputs are essential.
Cache API, bindings and complete report conversion remain pending. Evidence:
`wavenet/native-progress.json`, `wavenet/cache-design.md` and the independent
oracle under `build/nn-report-conversion/`. AutoRec/GAN diagnostics input questions and
the original character_rnn corpus recipe remain pending; no substitute recipes.

- 2026-11-10: MADE density continuation (2026-11-10): native `nn/made_density.hpp` promotes
nonnegative integer degree masks, stopped structural masked dense projections,
the fixed64-pixel/two16-unit sigmoid density in both orders, and a stable
Float64 equal-probability mixture of normalized joint densities. Flat owning
records expose degree/mask inventory, hidden features, logits, mixture NLLs
and posterior component responsibilities. All applicable first/higher paths,
mask detachment, invalid-before-leaves behavior and record lifetimes are tested.
Each model retains2400 values in6 ordered leaves,1256 active and1144 masked;
both begin with the same seed8803 draws but train separate stores and Adam states.
Strict output degrees exclude self and future coordinates in each order.

The12-block report preserves all192 Adam writes and complete sampling loops.
It matches47 original archives/2799518 values and576 update archives/921792 values
exactly. Independent NumPy checks139110 forwards,470400 partials,36 finite
differences,192 Adam writes,6000 row visits,8192 causal derivatives,128 future
flips and10752 sampled pixels. All1144 blocked weights/partials per order stay
exact. Reverse sampling indexes uniforms by generation ordinal, keeps raw
Jacobian coordinates, and observes the bottom half for completion; raster
observes the top half. A four-bit worked density sums to one. A four-state
mixture example and native gradient show why joint logmeanexp differs from
mean component NLL. The final mixture is.552185 bits per binary pixel, between
raster.558043 and reverse.551028; all remain worse than the original fit-only
independent baseline.384447. Eleven figures and both export bundles are reviewed;
four original galleries match1376256 decoded channel values exactly.

All23 new definitions,8 compiled header examples/clangd hovers and standalone
checkout/installed headers pass. Combined affected coverage159/159 is clean.
Selected tests510 C++,273 Python,1966 R expectations in54 files and63 metadata
checks pass. Evidence: `build/nn-report-conversion/made_density_core/`,
`made_density_report/`, `made_density/independent-oracle.json` and
`scripts/check_made_density.py`. The original executable and last-consumer
`common/made_density.hpp` are removed. PixelCNN++ is the next independent source:
its original48-update CIFAR-10 run is frozen in24 archives/1223447 values with
12 source pins and two original binary input pins. Instrumentation retains all24
original archives and7 other files exactly, recording144 update archives/135408
values. Independent NumPy verifies589824 original RGB values,36864 integer-area
codes,379469 forwards,78420 partials,29 finite differences,48 exact Adam writes,
12288 causal entries,64 whole-pixel future flips and9792 sampled channel values.
All nine256-bin tail distributions normalize, all448 blocked weights stay exact,
and generation from independent NumPy heads reproduces every sampled code.
Test7.48424 bits per8-bit RGB channel improves on the fit-only histogram7.91458
and uniform8. The largest absolute forward difference.001451 is an extreme-tail
component log probability (scaled error1.66e-7); max partial difference7.63e-7.
Evidence: `scripts/check_pixel_cnn_pp.py` and `pixel_cnn_pp/independent-oracle.json`.
Its reusable exact RGB likelihood/data primitives and complete report remain
to be promoted and rendered. AutoRec/GAN diagnostics input questions and the
original character_rnn corpus recipe remain pending; no substitute recipes.


- 2026-11-10: PixelRNN continuation (2026-11-10): native `nn/recurrent_pixels.hpp` promotes
an explicit convolutional Row-LSTM step and the compact one-layer binary density.
The step accepts projected16-channel gates, prior hidden/cell rows and an
unbiased1x3 recurrent kernel. Its i/f/o/candidate ordering and fixed forget+1
are explicit; seven flat fields expose all gates and new states. The full model
uses a biased mask-A3x3 input1-to16 projection, width4 row recurrence and biased
1x1 binary head. Source seed8303 creates357 values in5 ordered leaves,277 active
and80 masked. Initial h/c share the first input features times zero, retaining
the connected zero path and signed-zero arithmetic. No learned initial state,
recurrent bias, class conditioning, same-row hidden recurrence or batch coupling.
Seven model fields expose actual zero/gate/state/logit boundaries. All supplied
inputs validate before leaves; applicable first/higher paths and owning records
survive parent/store collection. Core APIs hide no fitting/sampling/optimizer.

The10-block report keeps all96 Adam writes and64 raster-sampling steps editable.
It reproduces22 original archives/2028400 values and288 phase archives/68640 values
byte-for-byte, including batch/history TSVs. Independent NumPy checks719283
forwards,34986 partials,13 finite differences,96 Adam writes,3000 fitting-row
visits,4096 causal derivatives,64 future flips and5376 sampled pixels. All80
forbidden weights and partials remain exact. Nine figures and both export bundles
are inspected; both original PNGs match688128 decoded channel values. A worked
nonzero-cell/zero-kernel example verifies forget memory f^8 and its derivative;
the real model uses linked zero states. The fixed-final test loss.677641 remains
worse than the fit-only Bernoulli baseline.384447 bits per binary8x8 pixel, using
the unrounded Double posterior for the baseline scalar. Actual hidden/cell row
magnitudes are shown without interpreting them as accuracy or attribution.

All19 new definitions,4 compiled header examples/clangd hovers and both
standalone checkout/installed headers pass. Combined affected coverage136/136 is
clean. Selected tests501 C++,269 Python,1932 R expectations in53 files and63
inventory/toolchain checks pass. Evidence: `build/nn-report-conversion/pixel_rnn_core/`,
`pixel_rnn_report/`, `pixel_rnn/independent-oracle.json` and
`scripts/check_pixel_rnn.py`. The original PixelRNN executable and its now-unused
shared header are removed. The next independent source is MADE density: both
raster/reverse orders are frozen in47 archives/2799518 values with9 source pins
and2 prepared-input pins,192 updates total. Instrumentation preserves all47
originals exactly and records576 phase archives/921792 values. Its equal-order density
mixture remains part of that future conversion. AutoRec/GAN diagnostics input
questions and the original character_rnn corpus recipe remain pending; continue
other independent examples without substituting a different recipe.


- 2026-11-09: Gated PixelCNN continuation (2026-11-09): native `nn/gated_pixels.hpp` promotes
vertical3x3/horizontal1x3 masks, paired tanh/sigmoid gates, seven owning stream
boundaries, generic RGB context and mandatory-label binary variants. Three
width4 blocks use unbiased vertical-to-horizontal1x1 links, biased horizontal
projections and residuals above block0. Unconditional1073 values/23 leaves,
conditional1553/29; both retain320 forbidden coordinates. Six learned10x8 tables
supply class offsets before the gates. Core APIs expose no fitting/sampling
loop or optimizer. `MaskedPixelConv` now accepts positive odd rectangular kernels;
old3x3 PixelCNN behavior remains byte-exact. All supplied inputs validate before
leaves, labels are discrete, and applicable image/weight first/higher paths stay
live. The RGB3/30 context has1410/23 and448 masked coordinates; both Float32/64
match the unchanged original RGB source in11800 values/six archives, with5320
independent forward values and3240 analytic partials checked separately.

The13-block report reproduces50 original archives/3231050 values and576 phase
archives/504384 values byte-for-byte, including exact batch/history TSVs and
all192 Adam writes. Independent NumPy checks553830 forwards,257348 partials,
152 finite differences,6000 fitting-row visits,8192 causal Jacobian entries,
128 future flips and15872 sampled pixels. All320 forbidden weights and partials
per variant stay exact. Both class grids reuse four complete uniform streams
across ten classes; only the conditional model responds to changed valid labels.
Both gated models open the plain masked-CNN input28/logit34 blind spot. Twelve
local figures and both export bundles are inspected; all six original PNGs match
2359296 decoded channel values. The fixed-final test losses.582654/.578156 remain
worse than the fit-only Bernoulli baseline.384447 bits per binary8x8 pixel.

All15 new definitions,5 compiled examples/hovers and standalone checkout/installed
headers pass, along with the changed binary-pixel header's25 definitions/9
examples/hovers. Combined40/40 coverage is clean. Selected tests493 C++,266
Python,1913 R expectations in52 files and63 inventory/toolchain checks pass.
Evidence: `build/nn-report-conversion/gated_pixel_cnn_core/`,
`gated_pixel_cnn_report/`, `gated_pixel_cnn/independent-oracle.json` and
`scripts/check_gated_pixel_cnn.py`. The old Gated executable and now-unused
`common/pixel_cnn.hpp` are removed. `common/gated_pixel_cnn.hpp` is reduced to
native compatibility imports for PixelCNN++; PixelRNN includes binary helpers
directly. Both remaining headers compile. A fresh old-PixelCNN run against the
native compatibility adapter reproduces all22 archives/1820572 values exactly.
PixelRNN is the next independent conversion: its unchanged357-value Row-LSTM
run is frozen in22 archives/2028400 values, with288 captured phase archives/
68640 values and exact original-output parity; its original recipe is available.
AutoRec/GAN diagnostics input questions and the original character_rnn corpus
recipe remain unresolved; these do not block the independent conversions.


- 2026-11-09: PixelCNN continuation (2026-11-09): canonical `nn/autoregressive_pixels.hpp`
provides detached Double-resize/Float32-threshold binary preparation, structural
A/B masks, stopped-mask/live-kernel convolutions, six actual PixelCNN boundaries,
a four-field live Bernoulli likelihood and a complete local midpoint-MT uniform
bank. The source network has1841 values/10 leaves,1033 active and808 forbidden
kernel coordinates; four width8 biased3x3 ReLU layers and a1x1 binary head.
It accepts positive gray rectangles and keeps the ordinary masked-CNN blind spot.
All supplied input contracts validate before leaf creation. Core APIs expose no
fitting/sampling loop or optimizer; complete96-step fitting and64-site raster
sampling remain editable in the report. Owning flat records, first/higher paths,
all64 causal sites, fixed mask values, inference immutability, invalid geometry,
unit-range/binary checks and parent/store collection are verified.

The10-block report reproduces22 original archives/1820572 values and288 phase
archives/353568 values byte-for-byte, with exact batch/history TSVs. A separate
NumPy oracle checks498099 forwards,180418 partials,28 finite differences,
all96 Adam writes, all3000 fitting-row visits,4096 causal derivatives,64 future
flips and5376 sampled pixels. All808 forbidden weights remain unchanged and
all their partials are zero. Both source PNGs match688128 decoded channel
values exactly. Nine local figures, native introduction/mask/likelihood examples,
an executed architecture and both Markdown/HTML figure bundles are inspected.
The result deliberately preserves the source's worse density than its fit-only
baseline: fixed-final test.616536 versus.384447 bits per binary8x8 pixel. The
baseline evaluates the unrounded Double posterior, separately from archived
Float32 probabilities. No accuracy or natural-image realism claim is made.

All25 new definitions,9 compiled examples/clangd hovers and checkout/installed
standalone headers pass. Selected C++ tests485, Python263, R1890 expectations
in51 files and63 inventory/toolchain checks pass. Metric umbrella/Python/R exports
are also complete. Evidence: `build/nn-report-conversion/pixel_cnn_core/`,
`pixel_cnn_report/`, `pixel_cnn/independent-oracle.json` and
`scripts/check_pixel_cnn.py`. The PixelCNN executable was retired here; its compatibility header was later
removed in the Gated continuation. The next
Gated PixelCNN source is frozen unchanged with both conditional/unconditional
variants:50 archives/3231050 values,10 source pins,2 prepared-input pins;
192 updates total,96 per variant. Instrumentation preserves all50 originals
exactly and retains576 phase archives/504384 values. This baseline is now converted and retired above. AutoRec and GAN
diagnostics input questions remain pending; four character-language examples
also lack the original character_rnn corpus-preparation recipe. None of these
missing recipes authorizes a substitute split. Continue independent conversions.


- 2026-11-09: GAN diagnostics infrastructure (2026-11-09, report still pending): native
`nn/frozen_graph.hpp`/`src/datamunge/nn/frozen_graph.cpp` provide immutable bounded
GraphDef decoding and read-only numeric subgraph execution, without TensorFlow
or protobuf runtime dependencies. Numeric Float32/64,Int32/64,Bool constants
preserve exact bits/scalar rank. Explicit feeds cut upstream ancestry; output-zero
names, control edges/CheckNumerics, cycle/depth checks, strict wire bounds and
last-consumer release are preserved. Returned fields are detached owning copies;
no internal model storage escapes. Native Python/R bindings include model loading,
node/operation metadata, constant copies and named evaluation. A thin Python
constructor adapter accepts the existing ByteVector proxy; R accepts integer bytes.

Native `prepare_example_data("inception-metric")` obtains the original official
2015 archive, unmodified GraphDef, LICENSE and manifest. Archive88931400 bytes/
SHA256 f6b539e07358fc4622efdd40cdb5577672c46cd416bbfccf9d4d2438f9b93a1c;
graph95673916 bytes/SHA256
009d6814d1bc560d4e7b236e170e9b2d5ca6f4b57bd8037f6db05776204415c6.
Completed graphs are rehashed, including same-size-corruption rejection. Native
file-mirror ingress, repeated preparation and constructor/evaluation are verified.
An isolated optional TensorFlow2.20 oracle verifies all488 numeric constants/
23885410 values byte-for-byte and12751308 values at22 stages for each of two
real CIFAR inputs. Preprocessing matches exactly; max pooled-feature difference
2.3842e-5, max probability difference3.5167e-6. Overall early-stage max difference
.00146484375 reflects Float32 kernel rounding. Both original MatMul and biased
logits are retained and compared; no replacement feature network is used.

Standalone `nn/generative_metrics.hpp` promotes detached Float64 low-rank Gaussian
Frechet, probability-space Inception score and contiguous split/population-spread
records. Source half/mean/covariance conventions and absolute negative-FID tolerance
are explicit. All84 original algebra values match exactly; independent NumPy and
five native mathematical cases cover unequal pools, same moments/different
supports, collapse/uncertainty and unequal ordered splits. The main NN umbrella
and native Python/R now export all three metric functions and their owning flat
records. Mixed Float32/64 unequal pools, stopped fields after parent collection,
ordered splits and invalid probability contracts pass in both bindings.
FrozenGraph and metric bindings are complete.

All77 affected definitions,12 compiled header examples,12 clangd hovers and both
standalone checkout/installed headers pass. Native selected tests478=375 model+
69 prior other+12 frozen graph+5 metric+17 dataset cases; Python260 and R1868
expectations/50 files pass, plus63 inventory/toolchain checks. Current evidence:
`build/nn-report-conversion/gan_diagnostics_core/`,
`gan_diagnostics/frozen-inception-reference.json`, `metric-oracle.json`, and
`scripts/nn_frozen_inception_reference.py`. No additional example is retired:
GAN diagnostics still lacks its original recorded images/Inception/Frechet inputs
and preparation recipe. A user question about their location is pending; do not
repeat it. The native model/extractor infrastructure gap is resolved, while the
full original experiment/report and explicit classifier-bias/generator protocol
still require evidence. Do not substitute a toy extractor, fabricate an original
baseline or retire the source from algebra-only checks. Continue other independent
conversions if those inputs remain unavailable. AutoRec's earlier original-ratings
question also remains pending. All416 originals remain in the goal's scope.

- 2026-11-09: SinGAN continuation (2026-11-09): canonical `nn/single_image_gan.hpp`
provides one local seeded Gaussian stream with optional coarse shared RGB, a
detached Double RMS amplitude with one final cast, five-field residual generation,
an image-only raw patch critic and an eight-field shared-weight two-branch game.
Three biased SAME3x3 convolutions preserve arbitrary positive RGB rectangles;
G371 values/6 leaves, D297/6. Explicit input mixture and previous-image skip remain
live through higher derivatives. Output is unbounded previous+tanh correction.
The compact source uses full G least squares plus10 reconstruction MSE and half
D least squares, rather than the full paper's WGAN-GP. All supplied inputs/aliases
validate before leaves; no API hides sampling, updates, optimizers or no-grad.

The13-block report keeps every three-scale draw and fitting step editable. One
CIFAR-10 training image produces16/24/32 targets with signed antialiased bicubic.
Fixed reconstruction noise is shared RGB seed7411 at the coarse scale and zeros
above. Higher amplitudes are measured once from the frozen lower reconstruction.
Both G/D warm-start from the previous final weights; each scale has cold Adam
(.0005,.5,.999,1e-8) clocks0/0. Each of48 fitting batches per scale refreshes the
frozen lower random draw, then takes D and a fresh G capture,288 total writes.
Completed scales keep no optimizers. Checkpoints restore exactly; all lower G/D
weights and every unselected owner stay bit exact. Random12-image, rectangular
40x48, fixed-reconstruction and all three zero-noise controls remain unchanged.

All21 source archives/488107 values and1038 probe archives/1890142 values match
byte-for-byte. Independent NumPy verifies99309 native partials(max5.96e-7),102
scalar finite differences,800156 forwards(max3.48e-7), every batch seed/input,
14208 exact preparation/noise values and all288 Adam writes exactly. All nine
history rows and measured amplitudes match. Report galleries independently check
1248768 RGB bytes, explicit[-1,1] range, half-up quantization, nearest enlargement
and no gaps. Twelve figures, including a native7-node/7-edge concept diagram and
two executed architectures, pass geometry/export checks and visual review. The
G trace packs211 supplied coordinates solely for plotting and explicitly retains
previous, noise and amplitude branches. Results remain weak: training MSE
.1290045545 and sample pixel variance.006276357, no held-out/realism claim.

All25 header definitions,seven compiled examples,seven clangd hovers and standalone
checkout/installed header checks pass. Twelve new native cases bring selected
coverage to375 model+69 other=444;254 Python checks and1836 R expectations/48
files pass, including owning graph lifetimes, higher paths, unequal pools, cold
warm-start clocks, residual/noise/amplitude derivatives and invalid-before-leaf
validation. All63 metadata checks pass. Source and both last-consumer legacy
single-image/adversarial-SR headers are removed; all reusable SRConv math remains
canonical. Evidence: `build/nn-report-conversion/singan_core/`, `singan_report/`,
and `scripts/check_single_image_gan.py`. GAN diagnostics is next: preserve its
original Inception protocol and resolve preparation/native extraction before
retirement. AutoRec's original ratings preparation and129 older CNN retirement
revalidations remain outstanding; full independent Python/R fits remain separate.

- 2026-11-09: SRGAN/ESRGAN continuation (2026-11-09): canonical `nn/relativistic_gan.hpp`
provides stable unequal-pool relativistic-average losses with LIVE opposite-pool
means, explicit half-scaled D/G terms and owning higher paths. Native
`nn/adversarial_super_resolution.hpp` supplies both residual generators, ordinary/
bias-free critics, frozen first-convolution features and twenty-field two-path
games. Nine generated-path fields expose image/residual/packed/shuffled/head/
body/upsampling and detached optional five-layer BN captures. Fixed source seeds,
Float32 fan-in bounds, normal.001 output kernels, image-level bicubic shortcut,
source RRDB scaling and CRD order are preserved. SRGAN G1591/25,D1673/8 plus
BN40/10; ESRGAN G3511/40,D1672/7,no BN. Frozen1792 official VGG19 values are
outside every optimizer. Native extraction stops all weight/bias paths and target
extracted features; RGB/other target paths remain live. Both feature tensors divide
by255 BEFORE post-ReLU MSE or pre-activation L1; SRGAN MSE thus scales by255².

The complete editable report owns16 pixel-only warm-up G steps, a cold reset of
BOTH Adam states without weight/statistic changes, then32 D-then-fresh-G steps
per variant. Cumulative writes48G/32D end at counters32/32. After every SRGAN G
weight write, explicit BN candidates use moments captured WITH that pre-write G
partial, source-rounded momentum/unbiased correction. No API hides fitting,
capture, detach, updates or later RNG. There is no generator weight EMA. Every
forward evaluates saved-statistic inference and batch-statistic G; source losses
use batch G even for read-only validation. Target canaries leave BOTH G paths
bit exact, while ESRGAN's paths coincide because it has no BN.

All19 original archives/3075843 values, both PNGs, four TSVs and non-timing
metrics match exactly, as do515 phase/probe archives/9487714 values. Independent
NumPy checks362343 native partials/moments(max8.06e-6),10013647 forwards
(max1.88e-6),226 scalar finite differences, all160 Adam transitions bit exactly,
every unselected-store/BN write, all768 row visits and1327104 exact HR/LR/bicubic
pixels. Frozen1792 values match the full official VGG archive. All validation/
final-test records, unequal-pool algebra and target canaries pass. Keeping old
warm-up Adam moments changes the first GAN G write by.000184068/.000194842;
post-write recomputed BN moments differ by.002699554. Another320 per-image
metric values match independent Gaussian-SSIM/PSNR calculations(max5.71e-14).

The18-block/16-figure report includes supplied BGR/feature-scale, CRD, BN and
live-mean derivative checks; a7-node/7-edge native method diagram; three executed
architecture traces; complete native preparation/capture/optimizer code; original
RGB comparisons; all-image quality gains; all48 BN mean commits and counter
resets. All2174448 RGB raster values, plot samples and adjacent HTML/Markdown
figure bundles pass;16 figures were visually reviewed. It preserves small gains:
mean PSNR26.75467688/26.73889463 versus bicubic26.71119211 dB, mean SSIM
.85584107/.85522090 versus.85413543. Mean PSNR differs from pooled PSNR;
one SRGAN image loses slightly. No perceptual/benchmark claim follows from this
32-image fixed-final teaching experiment. Native CIFAR fetch/style preparation
reuse existing official caches; no recorded-feature or other-checkout fallback.

All52 affected definitions,eight compiled documentation examples,two standalone
headers(checkout/installed),eight clangd hovers,363 native model plus69 selected
other cases,249 focused Python checks,1803 R expectations across47 files and63
metadata checks pass. A missing direct image_ops include was caught/fixed by
standalone checks, then bindings/native/docs rebuilt. Tests cover higher/owning
paths,frozen feature stops,rounded coefficients,real reset semantics,explicit BN
commits and two-path target invariance. The completed source is retired and unused
legacy model/BN/feature/relativistic code removed. Only the unchanged general
SRConv compatibility helper remains for SinGAN; an unused legacy include in the
older native-super-resolution forwarding header is removed and both compatibility
headers compile together. Evidence: `build/nn-report-conversion/srgan_esrgan_core/`,
`srgan_esrgan_report/` and `scripts/check_adversarial_sr.py`. Next SinGAN has21
original archives/488107 values,eleven source/helper pins and one native CIFAR
input digest frozen after cleanup; its independent oracle is pending;
AutoRec awaits its original ratings recipe,129 older CNN reports need retirement
revalidation, and independent Python/R fitting loops remain separate work.

- 2026-11-09: StarGAN continuation (2026-11-09): canonical `nn/uniform_categorical.hpp`
provides explicit local-MT rejection/modulo categorical draws with accepted raw
words; a high-cardinality rejection fixture checks skipped words too. Native
`nn/star_domains.hpp` supplies separate original/inversion/rotation domains and
observed-order balanced disjoint selections with exact originals and explicit
unused IDs. Source IDs preserve input dtype; classes/domains validate before
one-hot/gathers. No helper certifies cross-pool provenance; LoadSaliency does.
`nn/star_gan.hpp` provides the conditioned two-skip U-Net1609/10, image-only
shared-trunk dual-head critic688/8, stable three-way CE and fifteen-field game.
All flat records own applicable live graphs/storage, including higher paths.
The domain head initializes before the source head, preserving source RNG order.

One G produces fake(source,target) and reconstructs(fake,original) with shared
weights and both paths live. G=(full fake-to1 mean+fake-domain CE)+10*roundtrip L1;
D=half(real-to1+fake-to0 means)+real-domain CE, without a fake CE term. All inputs
validate before leaves; only missing leaves initialize. No API hides fitting,
noise, captures, detached paths or writes. Source seeds6301/6311 and batch6323
are retained. Uniform target seeds6338/6339/6340 reject4294967295 before modulo3;
fixed held-out requests use next domain. Each epoch visits900 rows with a final
4-row batch:29 batches,87 writes per owner,174 total. Caller steps D then uses
a fresh G forward on the same batch. The saved stale-G control changes the
first generator partial vector by.00003471970558166504.

The17-block/15-figure report supplies transform/partition/CE/shared-derivative/
loss-scale hand checks, both executed architecture traces, an8-node/7-edge
native method diagram and the complete editable C++ capture/Adam schedule.
Original11 archives/4307530 values, PNG, both TSVs and non-timing metrics match
exactly;363 phase/probe archives add2504554 exact values. Independent NumPy
checks203745 native partials(max2.46e-6),2470079 forward values(max5.45e-7),49
scalar finite differences, all174 Adam transitions bit exactly and every phase's
unselected store. All2700 fitting visits/accepted target words, per-class selected/
unused IDs, transforms, validation history and final loss/accuracy records pass.
An additional143472 fitting-only mean/prediction pixels, three known-map errors
and70560 balanced-gallery values pass independent checks. All1375120 raster
pixels, plot samples, architecture counts and adjacent HTML/Markdown figure
bundles pass;15 figures were visually reviewed. The stale-gradient plot uses
explicit1e6 display units so its small values have readable ticks.

Weak original translations remain: known-map MSE.202748341 versus fitting-only
target-domain means.067819011 and unchanged-source.619213694. Real domain accuracy
.683333333, generated requested-domain accuracy one third, roundtrip L1.438569665.
All original54 panels remain, with90 added panels covering all ten digits under
three source domains. No faithful translation/class-preservation/convergence
claim follows from conditioning or roundtrip arithmetic. All52 affected
definitions,twelve compiled documentation examples,three standalone headers in
checkout/installed layouts,twelve clangd hovers,351 native model plus69 other
selected cases,240 focused Python checks,1754 R expectations across46 files and
63 metadata checks pass. Tests cover discrete validation before leaves, real
skip joins, stable extreme CE, higher/owning paths, both shared G contributions,
fresh-G scheduling and exact ownership. The completed source and unused StarGAN
helper are removed. Its last legacy PixConv helper/common/pix2pix.hpp is removed
too; canonical nn/pix2pix.hpp stays. Component checks do not finish independent
Python/R fitting loops. Evidence: `build/nn-report-conversion/star_gan_core/`,
`star_gan_report/` and `scripts/check_star_gan.py`. Next is SRGAN/ESRGAN, using the
already prepared native CIFAR-10 and full official VGG19 cache in read-only mode
for the frozen original run. Its19 archives/3075843 values, ten source/helper
pins and four prepared-input digests are frozen; the independent oracle is pending.
AutoRec awaits its original ratings recipe;129 older CNN reports still need retirement
revalidation.

- 2026-11-09: StyleGAN continuation (2026-11-09): canonical `nn/modulated_conv.hpp`
provides per-example NHWC style modulation and optional live kernel demodulation,
with source row-major tap accumulation, signed/zero gains, rectangular images and
odd square kernels. Non-demodulated RGB retains the source computed factor times
zero plus one path. `nn/spatial_instance_norm.hpp` supplies live per-row/channel
population spatial centering and explicit division/sqrt with the source-rounded
guard; affine style remains separate. `nn/style_gan.hpp` separates latent-only
mapping, W-only synthesis, raw critic and the ten-field least-squares game. Nine
flat synthesis fields expose constant/maps/styles/demodulators; binding fields own
graph/storage. No API hides fitting, supplied noise, truncation, EMA or updates.

Both independently initialized trials preserve Gseed6607/Dseed6619, mapping144/4,
G661/17(AdaIN) or625/19(modulated), D493/6, zero initial noise strengths, full G
fake-to1 mean and half-scaled D. AdaIN normalizes after conv/noise/bias/leaky,
then applies scale/shift; version2 demodulates feature kernels and uses modulated
RGB without demodulation. A single80-value per-row noise bank splits16/64; first
style mixing and doubled actual first gains remain explicit. The complete editable
program owns every fresh D/G capture and Adam write. Actual32 batches per epoch
retain8 final rows,96 writes per owner,384 total across both trials; source comment
claiming96 steps per epoch is corrected in prose, preserving actual arithmetic.

The18-block/15-figure report includes supplied population-moment, live kernel-
derivative, noise-learnability and loss-scale examples, two executed architecture
traces, the complete native schedule, all six original galleries and all validation
records. All19 original archives/1743020 values, six PNGs, four TSVs and non-timing
metrics match exactly;792 phase/probe archives add928688 exact values. NumPy
checks220384 native partials(max1.98e-7),576982 forward values(max1.97e-6),136
scalar finite differences, all384 Adam transitions bit exactly and all unselected
owners. All89600 resized pixels,6000 fitting visits, source Gaussian banks,
initial zero-noise fields, style mixing, truncation and double-gain controls pass.
Another25600 fixed-center/independent-noise pixels match independent forwards
(max7.46e-8), demonstrating psi0 collapse requires shared noise too. Both style
and noise effects remain measurable, while the fixed-final generated images stay
weak. Test D/G=.345204055/.512427390(AdaIN),.342792064/.520632327(modulated).
No image-quality, fidelity, diversity or convergence claim follows from controls.

All1413168 gallery raster pixels, plot samples, architecture counts and adjacent
HTML/Markdown figure bundles pass;15 figures were visually reviewed. All41 new
definitions, nine compiled documentation examples, three standalone headers in
checkout/installed layouts, nine clangd hovers,339 native model plus69 other
selected native cases,232 focused Python checks,1704 R expectations across45
files and63 metadata checks pass. Tests cover higher paths, owning fields,
learnable zero strengths, style mixing, actual gain scaling, common-noise
truncation, explicit fresh-G scheduling, update ownership and before-leaf errors.
The completed source and unused StyleGAN helper are removed; shared PixConv stays
for StarGAN. Component coverage does not finish independent Python/R experiment
loops. Evidence: `build/nn-report-conversion/style_gan_core/`, `style_gan_report/`
and `scripts/check_style_gan.py`. Next StarGAN has11 original archives/4307530 values and eight source/helper
pins frozen after cleanup; its independent oracle is pending. AutoRec awaits its original
ratings preparation and129 older CNN reports still need retirement revalidation.

- 2026-11-09: CycleGAN continuation (2026-11-09): canonical `nn/cycle_domains.hpp`
provides native live rotation/inversion and disjoint first-half-per-class domain
gathers, preserving original IDs and exact original_B pixels. The inverse
orientation is mathematical; floating cancellation need not recover original bits.
Balanced integer metadata and unit images validate before gathers; the existing
LoadSaliency supplies original provenance/cross-pool checks. No known mapping,
label or opposite-domain target enters a generator. `nn/cycle_gan.hpp` reuses
both actual Pix2Pix U-Net skips, adds unconditional661-value patch critics and
exposes19 owning translation/cycle/identity/score/loss fields. Four distinct stores
own4102 values/32 leaves; unequal unpaired row counts are allowed. G keeps full
adversarial means plus cycle10/identity5 with source grouping; each D keeps its
half-scaled patch means. No API hides capture, optimizer, fitting or batch state.

The sixteen-block/sixteen-figure report includes supplied rotation, partition,
coupled-gradient, loss-scale and correspondence-ambiguity examples; two executed
architecture traces; a ten-node/eight-edge native method diagram; and the complete
C++ schedule. Independent orders6233/6247 visit500 fitting rows per domain and
retain the20-row final batch. D_a/D_b step separately, then ONE fresh common G
forward supplies BOTH generator vectors before either G write. Four Adam states
finish48 steps each,192 writes total. The first-step wrong control changes the
second G gradient by.00081092119216918945. All ten original archives/2467653
values, two PNGs, two TSVs and non-timing metrics match exactly, as do347 phase/
probe archives/3988406 values. NumPy checks202388 native partials(max3.02e-6),
4076881 forward values(max1.09e-6),86 scalar finite differences, all192 Adam
transitions bit exactly and every phase's unselected owners. All1500 independent
row pairs, disjoint source IDs, exact original_B targets and every validation/
final-test scalar record pass. Another156800 Double fitting-mean pixels match
exactly; four known-map errors and62720 balanced-gallery values pass. All1417728
raster pixels, all plotted coordinates, two architecture traces and adjacent
HTML/Markdown figure bundles pass; sixteen figures were visually reviewed.

The report preserves weak translations, test cycles.458553344/.469415665 and
mapping MSE.229144769/.210121211. Fitting-only target-domain means achieve
.066571734/.069072913. It preserves both original ten-zero galleries and adds
one ten-class/two-direction gallery; known mappings stay diagnostic only. No
convergence, class-preservation or identification claim follows from valid cycle
arithmetic. All32 affected definitions,six compiled documentation examples,two
standalone public headers(checkout/installed),six clangd hovers,327 native model
plus69 other selected native cases,224 focused Python checks,1649 R expectations
across44 files and63 metadata checks pass. Tests cover unequal unpaired counts,
actual generator coupling, both captures before writes, source-only translation,
owning higher paths, all aliases, scales and before-leaf validation. The completed
source and unused CycleGAN helper are removed. Unused legacy PixGenerate/
PixDiscriminate/PixObjectives are removed too; shared PixConv stays for StyleGAN/
StarGAN consumers. Component coverage does not finish independent Python/R full
experiment loops. Evidence: `build/nn-report-conversion/cycle_gan_core/`,
`cycle_gan_report/` and `scripts/check_cycle_gan.py`. Next StyleGAN has nineteen
original archives/1743020 values and eight source/helper pins frozen after cleanup;
its independent oracle/native promotion/report remain pending. Preserve both AdaIN
and modulated/demodulated variants, explicit spatial noise and source optimizer
schedules. AutoRec awaits its original ratings recipe;129 older CNN
reports still need retirement revalidation.

- 2026-11-09: Pix2Pix continuation (2026-11-09): canonical `nn/quantized_sobel.hpp`
recovers8-bit unit pixels in Double before floor, applies native zero-padded
Sobel convolutions and thresholds raw L1 magnitude inclusively(default256).
The condition tensor is detached. `nn/pix2pix.hpp` exposes a source-only two-level
U-Net with both actual concatenation skips, conditional49-patch raw scores and
seven explicit paired objective fields. All biased kernels use normal(.02),
source names/order and zero biases; G1390/10 leaves(seed6101),D725/6(seed6113).
The two learned stride-two transposes and skip channel order are retained.
D sums two patch logistic means without half; G adds100 times mean paired L1.
No API hides fitting, optimizer writes, captures, no-grad or batch state.

The thirteen-block/thirteen-figure report includes hand-checkable integer Sobel,
L1 and patch-mean derivative examples, an executed seven-stage/nine-node U-Net
trace with both multi-input skips, a native method diagram and the complete
C++ training program. Each batch captures D partials/steps D, then makes a fresh
G forward through updated D and steps G alone. Independent Adam states use
.0002/.5/.999/1e-8. All96 batches keep their final8 rows, giving192 writes and
all3000 fitting row visits. The final checkpoint is fixed at epoch3. All11
original archives/3118891 values, source PNG, two TSVs and non-timing metrics
match exactly, as do393 phase/probe archives/1673895 values. NumPy checks205155
native parameter partials(max1.22e-6),1577832 forward values,43 scalar differences,
all192 Adam transitions bit exactly and every phase's unselected owner.
Another65856 actual skip values match(max6.46e-7), with exact concatenation channel
order; all20 class-error values and200 held-out errors pass. All988128 gallery
pixels independently decode exactly; all plot coordinates, six method nodes/five
edges, thirteen reviewed figures and both adjacent export bundles pass.

The report keeps test L1=.408497557/MSE=.191904112 beside the stronger fitting-only
mean reference(.151756616/.067798074). It preserves the original ten-zero gallery,
adds a ten-class preview and blank-condition G/D controls, and makes no quality
or convergence claim. Both networks depend on source, inference stays immutable,
and single/batch generation agrees exactly. All25 affected definitions,six compiled
header examples,two standalone public headers(checkout/installed) and six clangd
hovers pass. All317 model tests plus69 selected other native cases,217 focused
Python checks,1613 R expectations across43 files and63 metadata tests pass.
Owning fields/higher paths, source-only translation, actual skips, loss scales,
explicit separate Adam writes and before-leaf validation are covered. The completed
source and unused Pix2PixModel adapter are removed; shared pure PixConv/PixGenerate/
PixDiscriminate/PixObjectives remain for CycleGAN consumers. Component coverage
does not complete independent Python/R experiment loops. Evidence:
`build/nn-report-conversion/pix2pix_core/`,`pix2pix_report/`,
`scripts/check_pix2pix.py`. Next CycleGAN has ten original archives/2467653 values
and eight source/helper pins frozen after cleanup. Its independent oracle/native
promotion/report remain pending. Preserve disjoint unpaired partitions,
rotation/inversion, two critics and jointly captured generator gradients.
AutoRec awaits its original ratings recipe;129 older CNN reports need retirement
revalidation.

- 2026-11-09: Progressive GAN continuation (2026-11-09): canonical `nn/pixel_rms_norm.hpp`
provides channel-wise live RMS division with the source-rounded epsilon, first
and higher derivatives, and no centering/affine/batch state. The standalone
`nn/progressive_gan.hpp` exposes the real old/new generator images and discriminator
features, source-ordered active groups and seven forward fields. Both source fade
formats (rank-zero and one-element tensors) work. Eight distinct stores initialize
all2067 values/24 tensors before fitting. G blends post-sigmoid images; D blends
4x4 features before its shared nonlinear score core. G retains its full least-
squares objective without a half factor. Native BigGanResize8 preserves the
source half-pixel Double accumulation and final Float32 cast.

The fourteen-block/fourteen-figure report contains the complete three-stage loop,
all-active-gradient capture before any phase write, selected D writes followed
by a fresh G forward through updated D, and eight independent Adam states.
Stable4/fade8/stable8 use1462/2067/2054 active values. All96 batches retain the
last8 rows, giving576 group writes and final counters96,64,64,64,96,64,64,64.
The final fade batch still steps the old RGB groups at alpha1; stable8 omits them.
A supplied zero-gradient Adam toy explains momentum continuation. Shared cores
continue changing after old RGB parameters freeze. No API hides fitting,
optimizer writes or stage transitions.

Both native delegation and the complete report preserve all eight original
archives/1343565 values, four PNGs, two TSVs and non-timing metrics exactly.
All399 additional phase/probe archives/804784 values match. Independent NumPy
checks180723 native partials (maximum1.14e-7),282436 forward values,66 scalar
finite differences and all576 Adam transitions bit exactly, with192 phase
ownership checks. Another2056 actual discriminator feature values match within
4.64e-8. All814992 gallery pixels independently decode exactly. Ten method nodes,
eleven edges, all plotted coordinates, fourteen visually reviewed figures and
both adjacent export bundles pass. The report preserves the weak nearly uniform
patches and fixed-final test D/G=.344080687/.626963854, without a quality or
convergence claim.

All31 affected definitions,eight compiled documentation examples,two standalone
public headers(checkout/installed) and eight clangd hovers pass. All309 native
model tests plus69 selected other cases,212 focused Python checks,1584 R
expectations across42 files and63 metadata checks pass. Tests cover higher
paths, real fade endpoints, dual alpha formats, owning field lifetimes, explicit
phase ownership, stage counters, inactive stores and the source G loss scale.
The completed source and its unused progressive helper are removed; shared
Pix2Pix/GAN support remains for other consumers. This does not finish independent
Python/R full experiment loops. Evidence: `build/nn-report-conversion/progressive_gan_core/`,
`progressive_gan_report/` and `scripts/check_progressive_gan.py`.
Next Pix2Pix has eleven original archives/3118891 values and seven source/helper
pins frozen; independent oracle/native promotion/report remain pending. Its
integer Sobel threshold, paired U-Net skips/transposed convolutions, conditional
PatchGAN and fresh-G-through-updated-D schedule must be preserved. AutoRec awaits
the original ratings recipe;129 older CNN reports need retirement revalidation.

- 2026-11-09: Wasserstein GAN continuation (2026-11-09): canonical `nn/midpoint_uniform.hpp`
provides local raw-word MT midpoint matrices with one final cast;
`nn/wasserstein_gan.hpp` reuses BiganMap for the source tanh16 generator/critic,
exposes all eleven forward fields and actual/analytic input Jacobians, and clips
all selected critic tensors explicitly, including biases. The detached interpolation
stops external real/alpha/G penalty paths. create_graph retains critic history
through the actual input derivative; evaluation disables that history under no-grad.
Weight0 skips the penalty addition while retaining its diagnostics. No API hides
an optimizer, clipping callback, fitting loop or persistent training-graph flag.

The complete thirteen-block/thirteen-figure report shows both independent trials:
clipped RMSProp5e-5/rho.9/epsilon inside the root, versus GP Adam1e-4/beta1=0/
beta2=.9/coefficient10. Each real batch is reused for three fresh-noise critic
steps, then a fresh G step through updated C. Both initialize G82/C65 values/eight
tensors; clipping occurs before fitting and after every clipped critic write.
Each trial makes117 C and39 G writes over three fixed epochs,312 writes total.
Every unselected store stays bit exact; all1200 row visits per trial are checked.

Both native delegation and the report preserve all five original archives/1900
values, two source PNGs, four TSVs and both complete metric files exactly. All41
phase probes/11112 values match exactly, including pre-clip steps and final
Jacobian/gradient/noise fields. NumPy independently verifies6696 forwards,848
parameter derivatives,42 final scalar differences,554 exact first-update values,
all three clips per trial,1312 ring coordinates and all midpoint/Gaussian banks.
Three additional65-coordinate gradients match closed form(max1.06e-6); the full
vector matches frozen source exactly and its Float32 component sum differs by
at most1.44e-6. All1572864 source raster RGB values match independent decoding;
existing native Image/draw_circle replaces the example's pixel loop. All labelled
SVG point/curve/epoch samples, eight method nodes/eight edges and both adjacent
export bundles pass. The report shows the weak diagonal generated bands and
unconverged gaps, rather than claiming ring coverage or a certified distance.

All19 affected definitions,six compiled documentation examples,two standalone
headers(checkout/installed) and six clangd hovers pass. All299 native model tests
plus69 selected other cases,206 focused Python checks,1556 R expectations across
41 files and63 metadata checks pass. Tests cover higher input/critic paths,
stopped external penalty paths, guarded zero Jacobians, owning field lifetime,
explicit optimizer choice,3-to1 step counts and bias-inclusive clipping. Source
and its unused Wasserstein helper class are removed; shared GAN support remains
for other consumers. Component tests do not finish independent Python/R full
experiment loops. Evidence: `build/nn-report-conversion/wasserstein_gan_core/`,
`wasserstein_gan_report/` and `scripts/check_wasserstein_gan.py`.
Next Progressive GAN has eight archives/1343565 values and eight source/helper
pins frozen after cleanup. Its independent oracle/native promotion/report remain
pending. Preserve eight independent optimizer groups, stage-specific activity,
all-active-gradient capture before any phase write, pixel normalization, resize
rounding and actual old/new fade paths. AutoRec awaits the original ratings
recipe;129 older CNN reports still need retirement revalidation.

- 2026-11-09: DCGAN continuation (2026-11-09): canonical `nn/captured_affine_batch_norm.hpp`
reuses fused learned-affine BN with detached mean/sample-variance captures;
`nn/dcgan.hpp` exposes signed G, raw D, separate joint-D/fresh-G phase functions
and saved-BN inference. Source kernels, names, creation order, transposed4x4
stride2 maps, fused rounding and Double .001 epsilon are preserved. G has6873
values/eight leaves, D989/seven; two saved-moment stores add40 values/six leaves.
The visible loop normalizes concatenated real/fake rows once for D with a stopped
G path, commits D moments with explicit Double .9 retention, then writes D Adam.
A fresh batch-BN G passes through current saved-BN D; both G moment pairs commit
before selected G Adam. No model API owns an optimizer, EMA write or fitting loop.

The complete twelve-block/twelve-figure report preserves all four original
archives/173429 values, both source PNGs, both TSVs and non-timing metrics exactly.
Twelve phase probes match435581 values; direct native phase helpers independently
match121981 frozen values. NumPy checks546880 forwards,22597 parameter derivatives,
41 final scalar differences,22 extreme-logit values, both exact first Adam and
moving updates,3000 row visits, and final validation/test metrics including coarse
MMD. An extra313600 signed-pixel BN-path values match the independent decoder
(max1.29e-7); four first-step effect rows match source checkpoint differences.
All833456 independently decoded gallery pixels, all curve/epoch points, eight
method nodes/nine edges and both adjacent export bundles pass. All four write
owners are checked on every batch,96 writes per Adam over three fixed epochs.

The samples remain blurry checkerboard-like textures. D real/fake accuracy1/.98
and D/G losses.252165/1.994329 are game diagnostics. Test coarse MMD.054826617
is much worse than repeated fitting mean.000135779; the report shows both images
and explains this statistic's limited view. Hand examples establish transposed
scatter/overlap, joint versus separate BN and stable raw-logit objectives before
real data. No labels enter fitting or test/validation score selects a checkpoint.

All45 affected definitions,11 compiled documentation examples,two standalone
headers (checkout/installed) and11 clangd hovers pass. All289 native model tests
plus69 selected other cases,200 focused Python checks,1510 R expectations across
40 files and63 metadata checks pass. Owned flat records retain storage and live
paths through parent/store collection; saved and captured moments stop. Source
and its unused DCGAN helper section are removed; shared `common/gan.hpp` remains
for WGAN and other consumers. Component tests do not finish independent Python/R
full experiment loops. Evidence: `build/nn-report-conversion/dcgan_core/`,
`dcgan_report/` and `scripts/check_dcgan.py`. Next `wasserstein_gan` has five
archives/1900 values and six source/helper pins frozen after the helper cleanup;
its independent oracle, native promotion and report remain pending. Preserve both
clipping/RMSProp and symbolic-gradient-penalty/Adam paths and three critic steps
per fresh G step. AutoRec still awaits its original ratings recipe;129 older CNN
reports need retirement revalidation.

- 2026-11-09: BigGAN continuation (2026-11-09): canonical `nn/captured_batch_norm.hpp`
separates live batch-moment capture from detached moving-value arithmetic;
`nn/gan_priors.hpp` exposes a locally seeded Gaussian rejection bank with every
consumed Double proposal; `nn/big_gan.hpp` exposes hierarchical G, projection D,
raw-kernel orthogonal penalty, both display paths and explicit captured-BN commit.
The visible loop warms each spectral cache 16 times, then advances D cache,
writes D, advances G cache, computes a fresh G gradient through updated D, writes
G and commits moments from that same pre-write forward. There are 96 Adam writes
per network over three fixed epochs. Five stores contain 1030 optimized values
and 342 state values in 60 tensors. Both objectives use batch-statistics fake
images even when the display flag selects saved-BN images; the report explains
this original convention. Hierarchical chunks, shared raw class embedding,
residual shapes, source-rounded normalization and rejection streams are retained.

All 23 original archives/319761 values, three PNGs, two TSVs and non-timing
metrics match exactly, both through native delegation and the complete report.
Ten phase probes match 15268 values exactly. NumPy independently verifies 84708
forward values, 4740 derivative/capture values, 74 final scalar differences,
89600 exact resized pixels, 19669 exact proposals and all 3000 row visits.
Two report-only archives additionally check 813 full-gradient/capture values and
1800 projection components. All 14 C++ blocks, 14 inspected figures, 996432
independently decoded gallery pixels and adjacent Markdown/HTML figure bundles
pass. Hand examples distinguish population normalization from unbiased moving
variance, isolate class projection, derive the off-diagonal penalty and display
accepted/rejected proposals. Short-run samples remain weak; class response,
projection changes and hierarchy/truncation controls do not establish class
accuracy or realistic generation.

All 60 affected definitions, 16 compiled documentation examples, three standalone
headers (checkout and installed) and 16 clangd hovers pass. All 279 native model
cases plus 69 selected other cases, 195 focused Python checks, 1468 R expectations
across 39 files and 63 metadata checks pass. A SWIG owning-copy typemap for nested
moment records fixes the dangling-parent lifetime found by collection tests;
record fields preserve both storage and applicable graphs. Source and unused
`common/big_gan.hpp` are removed; shared `common/gan.hpp` remains for consumers.
Python/R component coverage does not finish independent full experiment loops.
Evidence: `build/nn-report-conversion/big_gan_core/`, `big_gan_report/` and
`scripts/check_big_gan.py`. Next DCGAN has four archives/173429 values and six
source/helper pins frozen; its independent oracle, native promotion and report
remain pending. AutoRec awaits the original ratings recipe; 129 older CNN
reports still need retirement revalidation.

- 2026-11-09: Self-Attention GAN continuation (2026-11-09): canonical
`nn/cached_spectral_dense.hpp` exposes the shared dense cached-spectral map;
`nn/self_attention_gan.hpp` exposes general spatial token attention and compact
patch G/D maps with twelve live forward fields. Existing Plain attention at
scale 1 supplies unscaled softmax; MaePatches/MaeImage supply the raster/patch
permutations. Source projection order, biased maps, normal(.01) positions,
gamma 0 initialization, Float32-rounded initializers/guards and all stopped-cache
versus live-sigma derivatives are retained. The spectral GAN now delegates its
private dense math to the same reusable layer. General attention permits
positive B/T/C/Q/V with validated products and a configurable prefix. All four
SAGAN stores must be distinct and dtype matched; real/fake row counts may differ.

All 39 affected definitions, eleven compiled examples, three standalone headers
and eleven clangd hovers pass. All 264 native model cases plus 69 selected other
cases,189 Python checks,1415 R expectations across 38 files and 63 metadata checks
pass. Record fields retain storage/graphs after parent/store collection; tests
cover zero-gate projection partials versus live gate, first/higher paths, signed
spectral normalization, strides, invalid geometry and all four explicit writes.
The complete native program shows sixteen warm-up iterations per network,
then D-cache advance, D write, G-cache advance, fresh G through updated D with
unchanged D cache, and G write:96 writes per optimizer,192 total, fixed epoch 3.
G 2730,D 650,G-cache 245,D-cache 110 values occupy 51 checkpoint tensors/3735 values;
only 3380 values are optimized. No labels enter fitting. The initial test forward
is diagnostic; no test result selects weights, seeds, hyperparameters or epochs.

Both original-source delegation and the eleven-block/eleven-figure report retain
all ten frozen archives/1226475 values in dtype/shape/value bytes, both source
PNGs, both TSVs and all non-timing metrics. Twelve phase probes match 97010 native
values exactly. The shared dense refactor preserves Spectral GAN's 364381 probe
values and 764429 complete-run values exactly. NumPy independently verifies 892521
forwards,15600 native derivatives,71 final scalar differences,355 cache values,
16 warm-up iterations per network,3735 checkpoint values and 3000 row visits. It
checks patch order, unscaled attention, normalization derivatives, gate identity,
nonlocal/noise controls and state-D-state-fresh-G ownership. All 696672 gallery
pixels, all curve samples, eight method nodes/six edges and both adjacent export
bundles are verified. Small effects use a readable 1e-4 output-change axis.

The run still produces weak grey fields, D real/fake accuracy 0/1 and coarse
MMD.11777806 versus constant fitting mean.0001357794. Trained nonlocal effects
about.000728369(G)/.000387209(D) demonstrate mixing, without certifying realistic
generation or semantic localization. The report distinguishes the controlled toy
from learned weights and the dense patch teaching architecture from the paper's
convolutional benchmark. The standalone source and both now-unused Self-Attention
GAN/shared spectral helper headers are removed. Shared big_gan.hpp/gan.hpp remain
for their consumers. Component coverage does not complete independent Python/R
full experiment loops. Evidence: `build/nn-report-conversion/self_attention_gan_core/`,
`self_attention_gan_report/` and `scripts/check_self_attention_gan.py`.
Next BigGAN has 23 archives/319761 values and seven source/helper pins frozen;
its independent oracle and native promotion remain pending. Preserve hierarchical
noise, class-conditional BN/moving moments, residual geometry, projection,
orthogonal penalty, rejection truncation and per-group spectral state phases.
AutoRec still awaits its original ratings recipe;129 older CNN reports need
retirement revalidation.

- 2026-11-09: Spectral GAN continuation (2026-11-09): canonical `nn/spectral_cache.hpp`
provides detached column power iteration, cached normalization with a live signed
sigma and explicit state-only advances. `nn/spectral_gan.hpp` reuses native
MlpGanGenerate and HingeGanObjectives for five live endpoints. G 26416 and D 25153
values are optimized;849 cached u/v values are separately checkpointed. Guard
constants and D initialization preserve source Float32 rounding for both dtypes.
Sixteen warm-up iterations precede the initial checkpoint; one state advance
precedes each D write, then a fresh G forward uses updated D and the same cache.
No power iteration, schedule, optimizer or MMD is hidden in model evaluation.
Invalid/missing pairs fail before any cache writes. Cache flags follow
ParameterStore, but vector paths stop while first/higher W/sigma paths stay live.

All 20 definitions, seven compiled public examples, two standalone headers and
seven clangd hovers pass. All 255 native model cases plus 69 selected other cases,
184 Python checks,1381 R expectations across 37 files and 63 metadata checks pass.
The nine complete report programs/nine inspected figures preserve all seven
original archives/764429 values byte for byte, both source PNGs, both TSVs and
non-timing JSON. An unchanged-source delegation matches the same artifacts.
Ten phase probes additionally match 364381 initialization/state/gradient/update
values exactly. NumPy independently checks 339561 forwards,313600 control pixels,
129554 native derivative values,22 scalar differences,16 warm-up iterations,849
cache values,52418 checkpoint values, all 3000 row visits and final SVD audits.
All 696672 preview/gallery pixels, sampled plot vertices and epoch points, nine
node boxes/seven edges, and adjacent HTML/Markdown figure bundles are verified.

The final hidden/output audit norms 1.0086809859/1.0000000278 recompute sigma/Wbar
from saved Float32 arrays in Float64 before SVD. Finite power iteration does not
certify exact unit kernel or network norms. Weak grey samples, coarse MMD.11783564
versus fitting-mean.0001357794, and perfect source separation are retained without
claiming realism or convergence. The standalone source and unused model wrapper
are removed; shared SpectralDense remains unchanged for Self-Attention GAN.
No separate Python/R experiment loops are claimed by component coverage.
Evidence: `build/nn-report-conversion/spectral_gan_core/`, `spectral_gan_report/`
and `scripts/check_spectral_gan.py`. Next is Self-Attention GAN: ten archives/
1226475 values and nine source/helper pins are now frozen; its independent oracle
and native promotion remain pending before shared math changes.
AutoRec still awaits original ratings preparation;129 older CNN reports need
retirement revalidation.

- 2026-11-09: InfoGAN continuation (2026-11-09): canonical `nn/info_gan.hpp` exposes
row-interleaved generated priors, explicit noise/category/continuous generation,
independent image-only Q, unit-Gaussian recognition loss and fifteen live forward
endpoints. MlpGanGenerate/Discriminate, LogisticGanObjectives, AuxiliaryClassLoss,
GaussianNoise and ImqMmd supply existing computations. Defaults create G26800,
D25153,Q25516 values in fourteen leaves, with Q hidden/CATEGORY/CONTINUOUS order.
Real and fake pools can differ; finite signed Float32/64 CPU matrices, discrete
codes, bounded continuous priors, strides and representable geometry validate
before leaves. All applicable continuous first/higher paths remain live; record
fields survive parent/store collection. Prior category words use unbiased local
mt19937 rejection, followed by two midpoint words per default row; final Float32
rounding can reach endpoints. Raw words remain exact Float64 audit values.
All 39 definitions, nine compiled examples, standalone header and nine clangd
hovers pass. All 245 native model cases plus 69 other selected cases,180 focused
Python checks,1350 R expectations across 36 files and63 tooling checks pass.

Twelve complete report programs/twelve inspected native figures replace the
standalone source and its unused model-specific helper. No Python/R launcher
existed; component coverage does not complete independent host-language full
experiment loops. The editable C++ program shows all 288 D-Q-G writes,96 per
independent Adam state, all 3000 fitting-row visits and fixed final epoch3.
G minimizes non-saturating source+recognition with weight1; Q minimizes recognition
only. No digit label enters fitting and no entropy constant is subtracted from
G's reported objective. D's write leaves Q's partial bit exact with G fixed.
Every phase recomputes; checkpoint restore, image-only Q, code-only G, single/batch
parity, inference immutability and all 100+55+55 traversal panels are retained.
The initial test forward is a diagnostic control; no test result selects writes,
noise banks, hyperparameters, epochs or weights.

Both unchanged-source delegation and report preserve all eleven original archives/
684712 values in dtype,shape,value bytes, two TSVs, all four source PNGs and
non-timing JSON values; duration is remeasured. NumPy independently checks 516984
forwards/controls,77469 initial derivatives,39 scalar finite differences,30800
exact prior/noise values and all fitting rows. An additional report archive stores
80400 G source/recognition/total partials; its independent matrix-calculus check
agrees within 9.75e-9 and matches all 26800 frozen total-G derivatives exactly.
All 2063856 gallery/preview pixels are independently decoded and verified; all
sampled plot vertices and epoch points avoid legends and all six method edges
avoid node interiors. HTML/Markdown exports include twelve adjacent figure files.

Results remain weak: mostly grey fields, generated-prior category recovery.115,
continuous-coordinate MSE.23072124 and empirical entropy-minus-R score-.67078066.
Code recovery and pixel dependence do not certify digit semantics. The report
explains that the finite Monte Carlo score has no sample-specific information
bound guarantee and that a negative score is not negative true mutual information.
Q's separate unit-variance architecture and the source's D-Q-G schedule differ
from common paper implementations; no convergence or disentanglement is claimed.
Evidence: `build/nn-report-conversion/infogan_core/` and `infogan_report/`.
Next is spectral GAN: seven archives/764429 values and eight source/helper pins
are frozen. Its independent oracle and promotion remain pending. Cached stopped
singular vectors, differentiable sigma,16 warm-up updates and one state-only update
before each D step must be retained; its final exact effective hidden norm 1.00868
is not a strict unit-norm or Lipschitz certificate. AutoRec still needs its original
ratings preparation;129 older CNN reports need retirement revalidation.

- 2026-11-09: Compact dense GAN continuation (2026-11-09): `nn/dense_gan.hpp` and
`nn/conditional_gan.hpp` expose explicit tanh/sigmoid generation, image-only raw
scoring, required Logistic/LeastSquares/Hinge objectives and class-conditioned raw
scoring on both networks. Existing AaeDecode, ConditionalGanGenerate,
LogisticGanObjectives, GaussianNoise and ImqMmd are reused. DenseGan owns five live
endpoints; ConditionalGan owns seven, without optimizer state or hidden quadratic
MMD. Each default unconditional model has 26416+25153=51569 values/eight leaves;
conditional G/D have 26736+25473=52209. Finite Float32/64 CPU matrices, strides,
positive representable geometry and malformed options/labels validate before leaves;
unconditional real/noise pools may differ. Continuous first/higher paths remain
live where differentiable. Hinge uses zero D subgradient at exact margins and
unhinged G=-mean(fake); least-squares preserves .5*(real mean+fake mean) with
0/1 targets, without claiming the different-label Pearson derivation.

All 33 definitions, eleven compiled examples, both standalone headers and eleven
clangd hovers pass. All 235 native model cases plus 69 other selected cases,
174 focused Python checks, 1301 R expectations across 35 files and 63 inventory/
tooling checks pass. Python/R facades delegate all numerical work to C++.
Component coverage does not finish independent host-language experiment loops.

Thirty runnable report programs and thirty inspected native figures replace
vanilla_gan, lsgan, hinge_gan and conditional_gan. Each report includes all fitting,
96 D then 96 fresh-G writes, two independent Adam states, all 3000 fitting-row
visits, fixed final epoch3, exact restore, immutable inference and fixed/changed/
zero-noise controls. An initial test forward is archived only as a diagnostic;
test outcomes never select the schedule or weights. Conditional GAN includes a
hand-checkable one-unit class example, both-network class effects, invalid-label
controls and all100 same-noise class-grid panels, without an auxiliary class head.
The legacy MlpGanModel, its objective enum and unused class/least-squares helpers
are removed after their four consumers retire. Shared common/gan.hpp remains for
other models; InfoGAN has been frozen with 516984 independently checked forwards,
77469 derivatives, 39 scalar finite differences and 30800 exact prior/noise values
for the next conversion. Its native recognition components and report remain pending.

Both unchanged-source core delegation and complete report programs preserve all
29 archives/3134202 values in dtype, shape and value bytes, eight TSVs, nine source
PNGs and non-timing JSON values; duration is remeasured. Independent NumPy checks
2692972 forwards/controls, all206916 initial parameter derivatives and88 scalar
finite differences across the four models, plus all12000 fitting-row visits.
The independently decoded galleries preserve every panel/pixel, native curves
retain every sampled vertex/epoch observation without legend overlap, and all
sixteen method edges avoid node interiors. HTML/Markdown exports include verified
adjacent figure bundles. Coincident logistic D-real/G-fake derivatives are explained.
The result stays weak: mostly grey generated fields, coarse MMD² around.118 versus
.0001358 for the repeated fitting mean. Source accuracies and the conditional
pixel/logit effects are structural/game controls, not evidence of realistic digits,
calibration or convergence. Evidence: `build/nn-report-conversion/mlp_gan_core/` and
`mlp_gan_reports/`. Next is InfoGAN; AutoRec still awaits its original ratings recipe.

- 2026-11-09: Bidirectional GAN continuation (2026-11-09): canonical `nn/gan_data.hpp` and
`nn/bidirectional_gan.hpp` expose ordered seeded Gaussian-ring samples, signed
unbounded tanh/linear maps, explicit joint data/code scores and eight live forward
endpoints. Existing GaussianNoise, AaeDecode and LogisticGanObjectives supply the
computations; no second RNG or MLP implementation is introduced. Three independent
stores create hidden/output W,b in G,E,D order, with 82+82+97=261 default values
in twelve leaves. Finite signed Float32/64 CPU matrices, positive representable
widths, unequal real/prior pools, strides and continuous first/higher paths are
supported. All user input checks precede leaf creation; owned Tensor fields survive
parent/store collection. The deterministic ring cycles component i%K and casts only
after Double centre/noise arithmetic, preserving all 1312 source point values.
All 17 definitions, five compiled examples, both standalone headers and five clangd
hovers pass. All 224 native model cases plus 69 other selected cases, 164 focused
Python checks, 1244 R expectations across 34 files and 63 inventory/tooling
checks pass.

Twelve complete report programs and twelve inspected figures replace the original
C++ source and its unused model-specific helper. No Python/R launchers existed for
this model; component coverage does not finish independent host-language experiment
loops. Shared common/gan.hpp remains for other models. The visible loop recomputes
D then E then G with three independent Adam states: 39 writes/store, 117 updates,
all 1200 fitting-row visits and fixed final epoch 3. E's write leaves G's partial bit
exact with D fixed; reconstruction and latent cycle are live but unfitted diagnostics.
All twelve checkpoint tensors restore exactly and inference is immutable. The
initial test forward is an explicit diagnostic control; no test outcome chooses
updates, schedule, hyperparameters or weights.

All nine archives/6210 values, both TSV files and non-timing JSON values are exact;
duration is remeasured. The core-delegated unchanged source also preserves all three
original PNGs. The report intentionally replaces hand-drawn pixels with annotated
native plots, preserving every point, fixed [-3,3] windows and all outside counts.
Independent SVG projection checks 3360 coordinate values across the three main
scatter panels, all 656 procedural-pool points and both 128-point noise controls,
with equal coordinate aspect. Every plotted curve/point is retained without legend
overlap; all six diagram edges avoid node interiors. Markdown/HTML exports include
all twelve adjacent figure files.

The independent NumPy receipt checks 3398 forwards (maxerror2.36e-7), 768 exact
fixed/changed noise values, 32 scalar finite differences, all 1200 minibatch rows
and the 82 stored G derivatives after first D-Adam. Its first-Adam replay explicitly
uses widened equations then Float32 rounding, rather than claiming bit-exact native
intermediate rounding. The result stays weak: reconstruction MSE 2.1331171 is worse
than fitting-mean 2.0097662; latent-cycle MSE 1.2355026 is a separate unfitted metric.
G concentrates away from ring modes and E produces tight groups rather than a
Gaussian cloud. Joint zero-code/zero-image effects .2945093/.8831886, prior-only G,
data-only E and all restore/immutability controls remain explicit. Idealized
invertibility conditions are cited without claiming convergence or a learned inverse.
Evidence: `build/nn-report-conversion/bigan_core/` and `bigan_report/`.
Next is conditional GAN: its eight archives/845165 values, source pins and independent
oracle are frozen; existing native components reproduce all 52209 initial parameters
and 25219 first-forward values exactly. NumPy checks all 52209 initial derivatives,
22 scalar differences and 732625 forwards. Both networks consume class conditions,
unlike ACGAN's image-only D. AutoRec still needs its original ratings preparation.

- 2026-11-09: Auxiliary-classifier GAN continuation (2026-11-09): canonical `nn/gan_models.hpp`
and `nn/auxiliary_classifier_gan.hpp` expose stable logistic source objectives,
conditional tanh/sigmoid generation, an image-only source/class discriminator,
source-grouped categorical CE, fifteen live forward endpoints and explicit coarse
pixel features. Default G has 26736 values/four leaves, D 25483/six; D creates
hidden, CLASS, SOURCE in the original initializer order. Finite signed Float32/64
CPU matrices, integer targets, strides and positive representable geometry are
validated before leaf creation. All continuous first/higher paths remain live;
caller-owned optimizers select writes. Owned field handles survive parent/store
collection. Extreme raw logits preserve tiny positive-real D derivatives and
strong negative-fake non-saturating G derivatives. All 37 definitions, nine
compiled examples, both standalone includes and seven clangd hovers pass.
All 216 native model cases plus 69 other selected cases, 159 focused Python checks,
1199 R expectations across 33 files and 63 inventory/tooling checks pass.

Twelve complete report programs and twelve inspected figures replace the original
C++ source. No Python/R launchers existed for this source; its unused compatibility helper is
removed; shared common/gan.hpp remains for other models. The full visible loop
performs D then a fresh G forward through the updated D, using two independent
Adam states, all 3000 fitting rows and 96 writes per store/192 updates. Final
epoch 3 is fixed in advance, with no validation selection. An initial test forward
is archived solely as a control; test outcomes never choose writes, epochs or
hyperparameters. Checkpoint restore, inference immutability, isolated G source/class
partials, image-only D invariance, wrong labels, same-noise class grid and zero/
changed-noise controls remain explicit.

All ten archives/919949 values, both TSV files, all three source PNGs and non-timing
JSON values match bit for bit; duration is remeasured. The independent NumPy
checker agrees on 749714 forward values, 53472 isolated G derivatives, 24 scalar
finite differences and every minibatch source row. An independent PNG decoder
checks all 1509808 pixels of five galleries, including all 100 class-grid panels
and fixed two-per-digit fitting previews. Six numerical plots preserve every
curve/point without legend overlap; all four diagram edges avoid node interiors.
Existing native GaussianNoise, ImqKernel and ImqMmd are reused, with 57600 noise
values and 240000 kernel entries previously verified exact; no second RNG/kernel.

The weak result is retained: real/fake source accuracies 1/1 coexist with real/fake
auxiliary accuracies .39/.185, mostly grey samples and mean intensity .4993914.
Coarse generated MMD² .11794877 is worse than repeated fitting mean .0001357794;
the nondiverse mean demonstrates the diagnostic's blind spot. No image-quality,
invertibility or convergence guarantee is claimed. Component bindings do not
finish separate Python/R experiment loops. Evidence: `build/nn-report-conversion/acgan_core/`
and `acgan_report/`. Next is BiGAN: its frozen baseline/oracle and existing
GaussianNoise/AaeDecode/LogisticGanObjectives reuse are verified for all 261 initial
parameters, 1282 initial endpoints, 1312 point coordinates and 512 fixed-noise
values. AutoRec still needs its original ratings preparation recipe.

- 2026-11-09: Reconstruction-anomaly continuation (2026-11-09): canonical
`nn/reconstruction_anomaly.hpp` owns detached per-case residual/mean-ink
scores, explicit nearest-rank nominal calibration, canonical ROC/AP adaptation
and the three fixed paired grayscale stress controls. Finite signed Float32/64
sample rows accept strides and arbitrary matching flattened widths; differences
widen before feature-order Double accumulation. Score computation stops paths
without changing caller flags and rejects computed overflow. Quantiles use the
explicit Double ceil(q*N) rule in(0,1], default .95, without interpolation/fuzz.
Rankings accept unequal/mixed floating score vectors, group ties and use strict
score>threshold. Existing canonical ROC/AP reproduce all nine original rows
bit for bit; no second ranking implementation is added. Float32[N,28,28,1]
patch/darkening/shift results own contiguous stopped storage, preserving unchanged
and shifted signed-zero bits. All 13 definitions, six compiled examples, standalone
include and six clangd hovers pass. All 207 native model cases plus 69 other
selected cases, 154 focused Python checks, 1147 R expectations across
32 files and 63 inventory/tooling checks pass.

Fourteen complete report programs, thirteen inspected figures and both exports
replace the standalone source and generated reference launchers; the unused
compatibility helper is removed. Shared autoencoder helpers remain for remaining
consumers. The visible 96-update prerequisite reproduces both checkpoints exactly
(101,984 parameter values). Application inference adds zero updates and checks
all 50,992 model values for immutability. Both archives/3,198,145 values, every
JSON/TSV value and all ten source PNG strips are exact. Original/core/report
independent receipts agree for 1,568,784 forwards (maxerror3.56e-7), 6,001 exact
scores, 470,400 exact corrupted pixels, all nine rankings and calibration/tie
controls. An independent PNG decoder checks all 624,144 pixels of the two split
galleries and blank panels, all 80 case panels and ten eight-column source strips.

The report explicitly discloses validation reuse: the same 200 images select the
upstream checkpoint and calibrate thresholds, so empirical5% calibration FPR is
not a fresh-holdout guarantee. It retains all 600 paired transformed cases from
200 sources, all ten nominal digit classes and the separately excluded blank.
AE white-patch/shift TPRs are1/.995, but darkening TPR .02, AUROC .4028 and AP
.418652 remain visible. Mean-image darkening AUROC0/TPR0 and ink shift AUROC
.486025/TPR .05 are retained. Nominal test FPRs .09/.07/.06 differ from calibration
.05. The blank score .2130155 exceeds the AE threshold .0748985 but is excluded
from rankings. Equal-pool AP prevalence and single-split/synthetic-pair limits
are explicit; no deployment anomaly or class-rejection guarantee is claimed.
Evidence: `build/nn-report-conversion/anomaly_core/` and `anomaly_report/`.
Next: auxiliary-classifier GAN; AutoRec still awaits its original preparation recipe.

- 2026-11-09: Learned-image-codec continuation (2026-11-09): canonical
`nn/learned_image_codec.hpp` owns explicit fitting-only Float32 coordinate
ranges, half-up/clamped byte symbols, detached dequantization, a live explicit
affine-sigmoid decoder and the original bounded TFLCODE1 packet. Generic ranges
support positive widths; the format fixes 32 coordinates/784 grayscale pixels.
Finite CPU validation, strides, signed/constant ranges, exact little-endian
binary32 bits, deep nested range exports, stopped packet paths, continuous first
and second derivatives, exact length/count/shape guards and malformed fields
are covered. Pack validates all caller fields as strictly as parse; no encoder,
file IO, optimizer, checksum or entropy model is hidden. All 26 definitions,
ten compiled examples, standalone include and nine clangd hovers pass.
All 199 native model cases plus 69 other selected cases pass (five model cases
excluded by the initial name filter were separately run). All 149 focused Python
checks, 1087 R expectations across 31 files and 63 inventory/tooling checks pass.

Twelve complete report programs, ten inspected figures and both exports replace
the standalone source and generated reference launchers; its unused compatibility
helper is removed. Shared autoencoder helpers remain for reconstruction anomaly.
The visible 96-update prerequisite reproduces initial/selected checkpoint values
exactly (101984 parameters). The codec adds zero updates and checks every 50992
model value for immutability. All three archives / 1359472 values, every bitstream
byte, TSV, 210 PNG files and other JSON values remain exact; report Float64 MSE
reductions differ below 2e-15 and global PSNR by 1.73e-13. Prerequisite timing is
measured anew. Original/core/report independent checks agree for 352000 forward
values, 6496 symbols/dequantized coordinates, all 110972 packet bytes and 200 PNG
files/156800 lossless pixels, allowing only the stated metric roundoff. Forward
error is below 5e-6. A separate process decodes 156800 pixel probabilities / 200 IDs
exactly with absent dataset and encoder paths. An independent PNG decoder checks
all 304848 pixels of the fixed ten-case gallery, all 40 panels and ten source strips.

The report keeps all clipping/constants/failures: three constant channels and
nine clipped test-code entries. Actual file accounting is 28 header+256 ranges+
103488 decoder+800 identities+6400 payload=110972 bytes. The complete 5.661837 bpp
file is larger than the measured 49889-byte / 2.545357 bpp PNG collection; the old
uncompressed-PNG comment is not repeated. Payload-only rate .326531 bpp is labeled
separately. Decoded MSE .050327014 versus unquantized .050325811 and global PSNR
12.981988 retain blurry reconstructions and unequal lossy/lossless quality. No
competitive compression or joint rate-distortion training is claimed. Evidence:
`build/nn-report-conversion/codec_core/` and `codec_report/`. Next: reconstruction
anomaly; AutoRec still awaits its original preparation recipe.

- 2026-11-09: Latent-visualization continuation (2026-11-09): canonical
`nn/latent_visualization.hpp` owns finite signed Float32/64 CPU sample-row
flattening, fitting-only centered thin-SVD PCA, exact feature-order double
Euclidean neighbors and posthoc overlap/label summaries. PCA uses N-1 sample
variances, unscaled axes, frozen owning fitting state and live query/inverse
paths; exported state is detached and copied. Prefix inverse projection, strides,
constant/rank-deficient data, row-index distance ties, labels supplied after
retrieval, duplicate overlap semantics and validation/overflow are covered.
All 20 definitions, eight compiled examples, standalone include and seven
clangd hovers pass. Native model tests total 191; all 144 focused Python checks,
1047 R expectations across 30 files and 63 inventory/tooling checks pass.

Eleven complete report programs, ten inspected figures and both exports replace
the source and generated reference launchers. The unused visualization and
legacy tensor-PCA helpers are removed; shared autoencoder helpers remain.
A visible 96-update prerequisite reproduces both checkpoints exactly (101,984
parameter values); the application adds zero updates and checks all 50,992
model values for immutability. Both archives / 3,833,394 values match exactly.
Delegation preserves JSON and both source PNGs exactly. Report MSE reductions
differ below 3.5e-14, elapsed timing is measured anew, and other JSON values
are exact. Independent original/core/report NumPy checks agree for 1,299,200
forward values, 1,190,000 projection/reconstruction values, sign-aligned SVD
bases, all 10,000 neighbor indices and fitting-only display limits; metric
comparisons allow only the stated MSE roundoff. Maximum widened forward error
is below 5e-6. Both native scatter figures contain all 1,000 fitting and 200 test
points, with no points under legends and zero boundary-clipped test points.
Fixed first-per-digit reconstruction/retrieval galleries retain every failure.
AE32 test MSE .050326 is worse than pixel-PCA32 .017631; projecting the code
to two axes then decoding increases MSE to .063673. Pixel/AE32/PCA32 1NN
accuracies are .89/.78/.88, falling to .325/.35 for code/pixel PCA2. The report
states the information loss, unequal objectives/parameterization and single-run
limits. Evidence: `build/nn-report-conversion/visualization_core/` and
`visualization_report/`. Next: learned image codec; AutoRec still awaits its
original asset preparation recipe.

- 2026-11-09: Added canonical `nn/latent_arithmetic.hpp` now owns explicit affine interpolation/edits, detached
fitting-label class centroids and owning path flattening. `UndercompleteDecoder`
in `nn/autoencoder_models.hpp` accepts signed codes alone and creates/reuses
only decoder leaves (25,872 default values). Finite signed Float32/64 CPU,
strides, original grouping/row-order accumulation, input ownership, all affine
mixed second paths, code/decoder derivatives and pre-leaf errors are covered.
No clipping, model fitting or RNG is hidden. All 41 definitions in both affected
headers, 18 compiled examples (five new), standalone include and 13 clangd
hovers (five new) pass. Native tests total 183; all 139 focused Python checks,
1011 R expectations across 29 files and 63 inventory/tooling checks pass.

Eleven complete report programs, ten inspected figures and both exports replace
the original/reference launchers. The unused latent-arithmetic compatibility
header is removed; shared autoencoder/PCA helpers remain for other consumers.
A visible 96-update prerequisite reproduces initial/selected checkpoint values
exactly (101,984 parameters checked). The frozen application adds zero updates
and verifies all 50,992 model values for immutability. Both archives / 1,448,809
values, every JSON value and all five original PNGs are exact. Native norm
matching reproduces every Float32 random direction. Original/core/report NumPy
receipts agree for 13,760 affine path values, 8,160 class means, 651 actual path
derivatives, 320 Gaussian values, 218,720 forward values, both pixel controls
and fixed endpoint identities. Maximum widened forward error is below 5e-6.
The report retains every pair and failure, all 89 negative edited coordinates
and 8,587 clipped pixel-control entries. Decoder nonlinearity reaches 0.179131;
this is not semantic success or a generative-prior result. Evidence:
`build/nn-report-conversion/latent_core/` and `latent_report/`.

- 2026-11-09: Added canonical `nn/encoder_pretraining.hpp` now owns frozen ReLU features, fitting-only
population scaling and a frozen encoder/scaler with a caller-owned linear head.
Finite signed Float32/64, strides, detached deep copies/exports, live input/head
paths, pre-leaf validation and original two-pass double accumulation/casting
are covered. Defaults preserve 25,120 frozen values and two head leaves / 330
trainable values. All 28 definitions, seven compiled examples, standalone
include and three clangd hovers pass. Native model tests total 175; all 134
focused Python checks and 971 R expectations across 28 files pass.
All 63 inventory/tooling checks pass after retirement.

Nine complete report programs, eight inspected native figures and both exports
replace the source and generated reference launchers. The unused C++ transfer
adapter is removed; shared autoencoder helpers remain for other consumers.
The visible prerequisite shows all 96 unsupervised Adam updates and reproduces
initial/selected checkpoint parameters / 101,984 values exactly. The two heads
then receive matching initialization/order and 96 supervised updates each;
total compute is unequal because only one encoder receives pretraining.
All 26 archives / 2,849,964 values remain exact. Delegation preserves all TSV
and non-timing JSON values; report native Float64 probability-NLL reductions
differ by at most 3.11e-15. Other TSV/non-timing JSON values are exact.
Independent original/core/report NumPy receipts agree: 1,320 derivative values
including shifted-label controls, exact first Adam, 240,576 forward values,
6,000 batch rows, six eligible validation candidates, frozen encoder snapshots,
all confusion counts and fixed first-per-class gallery identities. Maximum
forward error is below 5e-6. Both branches select epoch 3; held-out accuracy is
62% pretrained versus 44% random, with NLL 1.286701 versus 1.628766. The report
states the single-seed and unmatched-compute limits. Evidence:
`build/nn-report-conversion/pretraining_core/` and `pretraining_report/`.

- 2026-11-09: Added `nn/patch_restoration.hpp`: generic target-free patch
  visibility and exact observed/predicted composition, without model or optimizer
  state. Contracts cover finite signed Float32/64 values, strides, positive axes,
  detached binary masks, all-visible/all-hidden cases, fill conversion, owning
  lifetime and selected continuous higher paths. Branch selection preserves
  signed zeros and visible bits. All six definitions, two compiled examples,
  standalone include and two clangd hovers pass. Native 168, Python 130,
  R 941 expectations across 27 files and metadata 63 pass. Twelve report programs
  and nine inspected figures preserve four archives / 4,704,817 values. Four
  visible native prerequisite programs reproduce three checkpoint archives and
  115,633 parameters exactly with 288 Adam steps; the application adds zero
  updates. Source delegation preserves TSV/JSON exactly; report MSE/PSNR
  reductions differ below 1.5e-14 / 3e-13. Independent NumPy checks all corruption,
  dense/transformer predictions, mask provenance, hidden-value independence,
  exact visible copies and metrics. Original, reference launchers and three
  unused C++ compatibility headers are retired; host-language helper comments
  point to canonical headers. AutoRec snapshots are pinned, with original asset
  preparation still to be located. Evidence:
  `build/nn-report-conversion/restoration_core/`, `restoration_report/` and
  `autoencoder_restoration/prerequisites/`.

- 2026-11-09: Added `nn/wasserstein_autoencoder.hpp`: deterministic tanh WAE,
  code-only decoder, pixel-summed squared cost and explicitly weighted signed
  unbiased MMD objective. Contracts cover eight source-ordered leaves / 26,036
  scalars, unequal counts, Float32/64, strides, six owning Tensor fields plus
  nested MMD record, explicit independent prior and continuous higher paths.
  Original Tolstikhin/Bousquet/Gelly/Schoelkopf paper and compact fixed-scale
  teaching differences are verified. All 16 definitions, five compiled examples,
  standalone include and five clangd hovers pass. Native 161, Python 126,
  R 910 expectations across 26 files and metadata 63 pass. SWIG declaration order and nested owning
  record getter are covered by Python/R GC tests. Nine report programs/seven
  inspected figures and both exports preserve 18 archives / 1,147,656 values and
  exact histories; only two summary MSE reductions differ below 4e-15. NumPy
  verifies 26,036 first derivatives, exact Adam, 52 endpoint fields and complete
  Gaussian/validation/MMD controls. Original, reference launchers and three
  unused compatibility headers are retired. Restoration's next baseline/oracle
  preserve zero application updates and three frozen checkpoints, checking
  1,593,600 forward values and exact visible copies. Evidence:
  `build/nn-report-conversion/wae_core/`, `wae_report/` and
  `autoencoder_restoration/`.

- 2026-11-09: Added `nn/vector_quantized_autoencoder_2.hpp`: generic two-scale
  VQ inference, coarse-conditioned fine encoding, code-only top/final decoders
  and copied parent/previous-symbol conditional prior. Contracts cover original
  fourteen-leaf order / 1,157 scalars, eleven owning fields and nested records,
  Float32/64, strides, shape/ID/product validation, explicit uniforms, per-image
  BOS, surrogate hierarchy paths and both reconstruction-to-book stops. Original
  Razavi/van den Oord/Vinyals reference and compact gradient-update/Bernoulli
  variant differences are verified. All 36 definitions, eight compiled examples,
  standalone include and six clangd hovers pass. Native153, focused Python121,
  R865/25 files and metadata63 pass. Ten report programs/eight inspected figures
  and both exports preserve fifteen archives / 1,932,136 values and exact TSVs;
  only two summary MSE reductions differ below 3e-15. NumPy verifies 1,157 first
  derivatives, exact first Adam, 6,272 hierarchy derivatives, 68 endpoints and
  complete prior/validation controls. The original, generated launchers and both
  unused legacy VQ headers are retired. WAE's pre-kernel oracle/pins remain
  unchanged; a later zero-update restoration baseline freezes three checkpoints.
  Evidence: `build/nn-report-conversion/vq2_core/` and `vq2_report/`.

- 2026-11-09: Added `nn/vector_quantized_autoencoder.hpp`: nearest dictionary
  assignment/override, selected-vector-preserving identity surrogate, separate
  dictionary/commitment targets, spatial VQ inference and code-only decoder,
  usage diagnostics and explicit-uniform raster Markov prior. Contracts cover
  Float32/64, strides, shapes/IDs/products, tie boundaries, intended surrogate
  higher paths, ownership, creation order, reductions and fitting-only provenance.
  Original van den Oord et al. reference and compact architecture/prior differences
  are verified. All 39 definitions, 12 compiled examples, standalone include and
  six clangd hovers pass. The hover driver closes stdin after LSP exit and allows
  30 seconds for parser-worker shutdown. All 145 native cases, 116 focused Python
  checks and 830 R expectations across 24 files pass; 63 metadata checks pass after
  retirement. Eight report programs/six inspected figures preserve 14 archives/
  759,770 values and exact histories; two MSE tensor reductions differ below
  3e-14. NumPy verifies 805 first surrogate derivatives, exact first Adam, all
  endpoints/priors and every validation candidate. The source, unused legacy
  model/decoder adapters and generated launchers are retired; shared adapters
  remain for VQ-VAE-2. Its 15 archives/1,932,136 values and independent hierarchy
  checks remain exact. The later WAE baseline now has a full independent oracle
  for 26,036 derivatives and signed MMD controls, with original pins preserved.
  Evidence: `build/nn-report-conversion/vq_core/` and `vq_report/`.

- 2026-11-09: Vanilla VAE now uses the existing canonical Gaussian API at
  beta1/784 pixels/latent2/hidden64. Five shared compatibility math functions
  forward to the core, retaining signatures, original eight fields and creation
  order. Five canonical-equivalent examples compile/run; Python/R help includes
  the configuration. New tests cover owned fields through GC, all ten parameter
  gradients, explicit input/noise paths and caller Adam. All 135 native cases,
  112 focused Python checks, 791 R expectations across 23 files and 63 metadata
  checks pass. Nine report programs, seven inspected figures and both exports
  preserve 16 archives/8,817,594 values and exact histories; one tensor Float64
  summary MSE differs by 2.998e-15, with other non-timing JSON exact. NumPy
  verifies all 101,652 first derivatives, exact first Adam, fixed/fresh Gaussian
  streams, 5,747,784 endpoint values and all three validation candidates.
  Both VQ examples retain all 29 archives/2,691,906 values and non-timing JSON
  after shared delegation. The vanilla source/reference launchers are retired;
  the shared helper remains for live consumers. No new public core symbol was
  needed. Next: canonical VQ components and their reports. Evidence:
  `build/nn-report-conversion/vanilla_vae_report/`.

- 2026-11-09: Added `nn/vae_gan.hpp`: explicit code-only generator,
  discriminator feature/logit records, detached-target learned feature distance
  and all nineteen Gaussian VAE-GAN endpoints. Contracts specify source names/
  creation order, separate selected-group objectives, balanced non-saturating
  fake branches, diagnostic-only clipped pixel NLL, unequal prior batches,
  Float32/64, strides, ownership, higher paths and checked architecture products.
  The primary Larsen et al. reference and compact-recipe differences are verified.
  All 30 definitions pass fresh Doxygen; six examples compile/run, the include
  compiles alone and five clangd hovers pass. All 135 native cases, 111 focused
  Python checks and 775 R expectations across 23 files pass; 63 metadata checks
  pass after retirement. Eight report programs/five inspected figures
  retain all 24 archives/3,078,235 values and exact histories. Two tensor Float64
  MSE summary reductions differ below 1.1e-14; every other non-timing metric is
  exact. NumPy verifies 64,786 initial/recomputed parameter derivatives, all
  three exact first Adam states, group isolation, all nineteen forward fields,
  explicit Gaussian banks and every eligible validation candidate. Source,
  unused VAE-GAN helper and generated reference launchers are retired. The next
  untouched vanilla VAE baseline contains 16 archives/8,817,594 finite values.
  Its scratch canonical-API forwarding probe preserves every archive, history
  and non-timing metric; independent checks cover all 101,652 first derivatives,
  exact first Adam, 17,240 Gaussian values and all validation candidates.
  Evidence: `build/nn-report-conversion/vae_gan_core/` and `vae_gan_report/`.

- 2026-11-09: Added `MidpointUniformNoise` to `nn/autoencoder_models.hpp`:
  exact source MT19937 raw-word midpoint uniforms, detached owning Float64,
  explicit full-uint32 local streams, scalar/empty geometry and checked
  element/byte/stride overflow. Contracts distinguish discrete resolution,
  Float32 rounding to one and the separate general uniform generator; the
  original MT reference is verified. All 31 header definitions pass fresh
  Doxygen; all 13 public examples compile/run, the include compiles alone and
  eight clangd hovers pass. All 128 native cases, 106 focused Python checks and
  743 R expectations across 22 files pass; 63 metadata checks pass after
  retirement. The sequence report has eight programs, five inspected figures
  and both computed exports. All 20 archives/1,454,190 values, full TSVs,
  vocabulary and non-timing JSON values remain exact. Independent NumPy verifies
  all 3,243 first BPTT derivatives/exact first Adam, all eighteen endpoint fields,
  all validation candidates, Gaussian/token banks, test scores and generation
  counts. The source, unused sequence helper and reference launchers are retired.
  The untouched VAE-GAN baseline and separate independent phase-gradient oracle
  are ready for the next core/report conversion. Evidence:
  `build/nn-report-conversion/sequence_report/` and `vae_gan/`.

- 2026-11-09: Added sequence_vae_data.hpp with owning grammar pools, independent
  vocabulary copies, complete corpus validation, prefix display/grammar policy
  and original fifteen-array export. Contracts cover every schema/factor/identity,
  split, EOS/PAD, prefix, balance and row-order rule, config shape compatibility,
  I/O errors, shared Tensor storage and ignored post-EOS display suffixes. All 17
  definitions pass fresh Doxygen; six public examples compile/run, the header
  compiles alone and six clangd hovers pass. Python/R pool-vector output typemaps
  copy handles;105 focused Python checks and 730 R expectations across 22 files
  verify owning selected pools after GC and file removal.126 native model cases
  and 63 metadata checks pass. The posterior-collapse report has nine programs,
  six inspected figures and both computed exports. All 35 archives/2,265,740 values,
  full TSVs and non-timing JSON values match the untouched source; independent
  NumPy BPTT/Adam, all endpoints, common validation and mixture/ablation/generation
  checks pass. The original posterior source/helper/reference launchers are
  retired; the sequence adapter was retained at that checkpoint with all 20
  archives/1,454,190 values exact and is now retired above. Evidence:
  `build/nn-report-conversion/posterior_report/`.

- 2026-11-09: Added sequence_variational_autoencoder.hpp and posterior_collapse.hpp:
  PAD-masked GRU encoding, exact source-ordered eighteen-field sentence VAE,
  prefix-only teacher decoder, greedy/explicit-uniform generation, KL schedule
  and signed host-double empirical-mixture diagnostics. Contracts cover fixed
  architecture, IDs/dtypes/strides, ownership, continuous BPTT/noise paths, hard
  choices, raw scores after EOS, finite-sample bias and numerical overflow.
  All 57 definitions pass fresh Doxygen; 13 public examples compile/run, both
  includes compile alone and 10 clangd hovers pass. 121 native model cases include
  all 3,243 Float64 leaf finite differences and noise second derivatives.
  102 focused Python checks, 712 R expectations across 21 files and 63 metadata
  checks pass. Both legacy programs now delegate mathematics to core and retain
  all 55 archives/3,719,930 values, complete TSVs and non-timing JSON values exactly.
  An independent NumPy BPTT/generation/mixture oracle passes on original and
  delegated artifacts, including every eligible validation column and first Adam.
  The report/corpus adapters remain for the next conversion step. Evidence:
  `build/nn-report-conversion/sequence_vae_core/`.

- 2026-11-09: Extended native `prepare_example_data` with sequence-vae-assets:
  all 432 six-factor sentences, sixteen Int32 arrays, explicit previous-target
  inputs, EOS/PAD lengths, 288/72/72 compositional splits and ascending source
  IDs. Public help documents procedural provenance, cache location, supplied
  archive preservation and missing-output behavior. Both public preparation
  examples compile/run; all 69 definitions across assets.hpp, basic.cpp,
  adapters.cpp and support.hpp pass fresh Doxygen, with three clangd hovers.
  All 15 native dataset cases and 63 migration/website checks pass. Both untouched
  sequence-VAE/posterior-collapse programs verify the exhaustive loader contract;
  their 55 baseline archives are retained before shared model migration. The
  constant-beta and standalone sequence trajectories match exactly. Evidence:
  `build/nn-report-conversion/sequence_vae_data/`.

- 2026-11-09: Added `nn/ladder_variational_autoencoder.hpp`: source-ordered
  precision-weighted Gaussian fusion, explicit ancestral conditional-prior sampling
  and a complete sixteen-field two-level Ladder forward. Contracts cover all four
  fusion paths, detached stabilizing shift, extreme-log limits, explicit lower-
  then-top noise order, lower-only decoding, conditional KL dependence, owning
  endpoints and input/product checks before leaves. Existing conditional Gaussian
  KL and ReLU decoding are reused. All 34 definitions pass fresh Doxygen; six
  examples compile/run, the header compiles alone and six clangd hovers pass.
  111 native model tests, 97 focused Python tests and 676 R expectations across
  20 files pass. Seven report programs/five inspected figures preserve all 20
  source archives (2,014,940 values), exact histories and batch orders. NumPy
  checks 52,196 first parameter derivatives, exact first Adam, 16 fusion partials,
  10,956 Gaussian values, 1,489,600 lattice/prediction pixels, all fifteen original
  saved forward endpoints and three eligible validation scores. The full native
  result additionally owns logits. The stronger fitting mean and missing BN/warmup
  benchmark recipe remain explicit. The unused legacy helper is removed.
  Evidence: `build/nn-report-conversion/ladder_vae/`.

- 2026-11-08: Added `nn/kernel_discrepancy.hpp` and
  `nn/info_variational_autoencoder.hpp`: fixed three-scale IMQ, all seven signed
  unbiased MMD endpoints, coefficient-controlled InfoVAE objectives, image-free
  tanh decoder and complete sixteen-field Gaussian forward. Contracts cover
  unequal sample counts, within-only diagonal exclusion, independent explicit
  prior draws, negative estimates, Float32 coefficient rounding, full Float64
  gradients, owned fields, product checks and separate negative-ELBO diagnostics.
  All 38 definitions pass fresh Doxygen; seven examples compile/run, the header
  compiles alone and seven clangd hovers pass. 104 native model tests, 92 focused
  Python tests and 638 R expectations across 19 files pass. Eight report programs
  and six inspected figures preserve all 35 source archives (7,209,428 values),
  exact histories and batch orders. NumPy checks 52,208 parameter derivatives,
  both exact first Adam steps, 29,056 Gaussian values, 2,847,488 lattice/prediction
  pixels, 1,434,984 kernel entries, all sixteen fields and six own-objective
  validation scores. The unused InfoVAE helper is removed. WAE's required kernel
  import delegates to core; its separate 18-archive pre-delegation comparison,
  histories and every non-timing metric remain exact. Evidence:
  `build/nn-report-conversion/info_vae/` and
  `build/nn-report-conversion/wasserstein_vae-prekernel/`.

- 2026-11-08: Added `nn/importance_weighted_autoencoder.hpp`: stable sample-axis
  log-mean-exp, sampled Gaussian IWAE bounds, normalized weights/ESS and complete
  Gaussian/Bernoulli multi-sample forwards. Contracts cover sample-major tiling,
  finite signed NLL, Float32 source rounding versus full Float64 precision,
  full ordinary pathwise gradients, K1 sampled versus analytic-KL estimators,
  finite-draw nonmonotonicity, owning fields and validation before leaf creation.
  All 22 definitions pass fresh Doxygen; five examples compile/run, the header
  compiles alone and five clangd hovers pass. 96 native model tests, 87 focused
  Python tests and 591 R expectations across 18 files pass. Seven report programs
  and five inspected figures preserve all 34 source archives (10,980,972 values),
  exact histories and batch orders. NumPy checks 102,832 parameter derivatives,
  both exact first Adam steps, 36,200 Gaussian values, 9,942,688 lattice/prediction
  pixels, all saved weight/ESS diagnostics and six common validation scores.
  The stronger fitting-mean baseline and concentrated importance weights remain
  explicit. Unused IWAE and compatibility-flow helpers are removed. Evidence:
  `build/nn-report-conversion/importance_vae/`.

- 2026-11-08: Added `nn/hierarchical_variational_autoencoder.hpp`: direct/residual
  Gaussian posteriors, conditional diagonal KL, miniature NVAE cell, image-free
  two-latent decoding, ancestral sampling and paired hierarchy forwards. Contracts
  cover finite shapes/dtypes/overflow, caller-supplied noise, moving versus batch
  moments, unbiased moving-variance updates, explicit BN mutation, separate
  optimizer/state stores, owning endpoints and the two-kernel regularizer.
  All 46 definitions pass fresh Doxygen; nine examples compile/run, the header
  compiles alone and nine clangd hovers pass. 89 native model tests, 82 focused
  Python tests and 560 R expectations across 17 files pass. Nine report programs
  and seven inspected figures preserve all 41 original archives (5,808,422 values)
  and exact histories/batch orders. NumPy checks 4,004 derivatives, both exact
  first Adam updates, all six first moving-buffer updates, 186,252 Gaussian values,
  2,797,312 lattice/prediction pixels, all 19 fields and six validation checkpoints.
  The stronger fitting-mean baseline and miniature-model limitations remain
  explicit. The unused hierarchy helper is removed; the ladder helper remains.
  Evidence: `build/nn-report-conversion/hierarchical_vae/`.

- 2026-11-08: Added `nn/gumbel_categorical_autoencoder.hpp`: explicit midpoint
  uniform/Gumbel banks, categorical relaxation, discrete uniform-prior KL,
  target-free simplex/ID decoding and categorical VAE forwards. Contracts cover
  broadcast rows/strides, source Float32 rounding, full-uint32 local RNG,
  temperature-invariant hard draws, absent straight-through gradients, soft
  surrogate versus hard discrete diagnostics, owning endpoints, overflow and
  validation timing.81 native tests,77 focused Python tests and516 R expectations
  pass. Nine public examples compile/run; the header compiles alone,35/35 definitions
  pass fresh Doxygen and nine clangd hovers pass. Eight report blocks/five inspected
  figures preserve all18 original archives (3471217 values) and exact batch/history
  data. NumPy checks53104 first derivatives, an exact first Adam step,70160 midpoint
  uniforms/Gumbels,2551136 data/prediction pixels and4096 calibrated hard samples.
  The stronger fitting-mean baseline and soft/hard mismatch remain explicit.
  The unused legacy helper is removed. Evidence:
  `build/nn-report-conversion/gumbel_vae/`.

- 2026-11-08: Added `nn/flow_variational_autoencoder.hpp`: configurable conditional
  affine posterior couplings with analytic inverses and signed log determinants,
  plus two-coupling Gaussian VAE forwards. Full contracts cover live image context,
  fixed-context code Jacobians, coordinate ordering, even-width validation,
  creation order,18 leaves/52672 default scalars, caller-supplied noise, sampled
  density signs, owning endpoints/named arrays and direct standard-normal prior
  decoding. No optimizer/fit is hidden.75 native tests,73 focused Python tests
  and477 R expectations pass. Five public examples compile/run; the header compiles
  alone,28/28 definitions pass fresh Doxygen and five clangd hovers pass. Eight
  report blocks/five inspected figures preserve all19 original archives (1984695
  values), exact batch/history data and all18 fields for every saved draw. NumPy
  checks52672 derivatives, an exact first Adam step,7304 Gaussian values,1398656
  data/prediction pixels and all48 full Jacobian entries. The stronger fitting-mean
  baseline and lack of matched no-flow ablation remain explicit. The legacy shared
  header retains only the normalizer required by the pending IWAE conversion.
  Evidence: `build/nn-report-conversion/flow_vae/`.

- 2026-11-08: Added `nn/total_correlation_autoencoder.hpp` with exact finite-mixture
  TC, a signed density-ratio discriminator, balanced discriminator loss, explicit
  detached column permutations and FactorVAE/full-reference betaTCVAE forwards.
  Full contracts distinguish live sample/reference derivatives, fitting-only
  reference provenance, signed sample ratios, Float32 rounded constants, caller-owned
  two-phase Adam, finite-batch product approximations, validation timing and nested
  owning Gaussian results. The unused legacy helper is removed. All69 native
  model tests,69 focused Python tests and447 R expectations pass. Eight public
  examples compile/run; the header compiles alone,19/19 definitions pass fresh
  Doxygen and eight clangd hovers pass. Ten report blocks/six inspected figures
  preserve all45 original archives (3633019 values), exact VAE/D batch order and
  complete histories. NumPy checks102929 derivatives, all first Adam updates,
  12344 Gaussian values,2162272 data/prediction pixels, six eligible criteria and
  fitting-only probes. The stronger fitting-mean reconstruction baseline stays visible.
  Evidence: `build/nn-report-conversion/factor_tc_vae/`.

- 2026-11-08: Added `nn/convolutional_variational_autoencoder.hpp`: spatial
  Gaussian inference with a sample-only dense/repeat/convolution decoder. Full
  contracts cover sixteen owning forward fields, five decoder fields,14 default
  leaves/2117 scalars, all seven stages, mixed He/Glorot initialization, explicit
  noise, reductions, continuous gradients, strides, non-square geometry, overflow
  checks and error timing. Binary NLL differs from soft-target cross-entropy;
  no deterministic spatial bypass, fitting or RNG is hidden. The unused legacy
  helper is removed.63 native tests,65 focused Python tests and421 R expectations
  pass. Four public examples compile/run; the header compiles alone,21/21 definitions
  pass Doxygen and four clangd hovers pass. Ten report blocks/eight inspected figures
  preserve all19 original archives and exact batch/history data. NumPy verifies4234
  initial/selected derivatives, one exact first Adam step,17240 Gaussian values,
  6821584 data/prediction pixels and seven complete spatial stages. Weak measured
  reconstruction and the narrow improvement over the fitting mean remain explicit.
  Evidence: `build/nn-report-conversion/convolutional_vae/`.


- 2026-11-08: Added `nn/conditional_autoencoder.hpp`: configurable Gaussian
  posterior and Bernoulli decoder conditioned on the same required class, plus
  standalone code/class decoding. Full contracts cover all twelve owning fields,
  ten named leaves/102932 default scalars, continuous gradients, computation-dtype
  one-hot rows, discrete Int32/64 bounds, error timing, reductions and fixed shared
  prior. Binary NLL and soft-target cross-entropy are distinguished. The old
  unused helper is removed; Python/R expose the core with native help/tests.
  All59 model tests,61 focused Python tests and393 R expectations pass. Three
  public examples compile/run, the header compiles alone,10/10 definitions pass
  fresh Doxygen and three clangd hovers pass. Nine report blocks/seven inspected
  figures match all18 original archives plus exact batch/history files. NumPy
  independently checks102932 derivatives, one exact first Adam update,17210
  Gaussian values,7158704 data/prediction pixels and fitting-only class means.
  The stronger class-mean baseline and the original seed reuse remain explicit.
  Evidence: `build/nn-report-conversion/conditional_vae/`.


- 2026-11-08: Added `nn/gaussian_autoencoder.hpp` and
  `nn/latent_factor_diagnostics.hpp`: explicit pathwise noise, analytic summed KL,
  configurable ReLU/Bernoulli beta-VAE and standalone decoder, plus generic
  fitting-only standardized ridge/best-axis probes. Contracts cover all nine
  owning forward fields, ten named leaves/51416 default scalars, exact reductions,
  continuous gradients, stride/dtype/range/beta checks, error timing and13 detached
  probe outputs with negative heldout R2. Shared example helpers delegate model,
  decoding and probe arithmetic; Python/R bindings and help use the canonical API.
  All55 native tests,57 focused Python tests and367 R expectations pass. Six public
  examples compile/run,19/19 new-header definitions pass Doxygen and six clangd
  hovers pass. Ten report blocks/eight inspected figures match all30 original NPZ
  archives plus both exact batch/history files. Independent NumPy verifies102832
  first-batch derivatives, two exact first Adam updates,7304 Gaussian values,
  2162272 lattice/prediction pixels and every probe array. Higher beta's weaker
  reconstruction and the limits of linear factor recovery remain explicit.
  Evidence: `build/nn-report-conversion/beta_vae/`.


- 2026-11-08: Added `nn/adversarial_autoencoder.hpp` and `nn/factor_lattice.hpp`,
  plus native beta-vae-assets preparation. Full contracts cover deterministic E/G/D,
  independent prior rows, unit-range soft/binary targets, stable summed Bernoulli
  NLL, all continuous gradients, distinct store identities, ownership/error timing
  and caller-owned snapshot/phase/four-state training. Native lattice generation
  specifies all294 images, exact IDs, compositional splits, dtype/order and local
  procedural provenance.50 model tests and12 dataset tests pass;53 focused Python
  tests and337 R expectations cover owning values and host phase isolation.
  Eight public examples compile/run,88/88 affected definitions pass fresh Doxygen
  (21 in the two new headers), and six new clangd hovers pass. The eight-block/
  six-figure report matches all26 original archives plus exact batch/history data.
  Independent NumPy checks77,522 initial/actual derivatives, four exact first Adam
  states,3,776 Gaussian values and633,472 lattice/prediction pixels. Poor measured
  reconstruction and the limits of finite empirical aggregate matching stay explicit.
  Evidence: `build/nn-report-conversion/adversarial/`.

- 2026-11-08: Revalidated `nn/competitive_autoencoders.hpp` for the lifetime
  winner-take-all conversion. Existing core symbols own stable column-wise
  selection, detached masks and explicit gated-training/full-ReLU-deployment
  endpoints; no new model math is duplicated in host languages. Removed the
  retired example helper and its redundant legacy example bindings. Three public
  examples compile/run,10/10 definitions remain documented and both clangd
  hovers pass.46 native model tests include all lifetime small-model parameter
  derivatives, input first/second checks, context dependence and zero-slot rules.
  49 focused Python tests and295 R expectations across nine files pass. All source
  arrays in29 archives match original C++, report and normal Python/R. Eight report
  blocks/six reviewed figures preserve both96-update loops, ungated validation
  selection and weak measured reconstruction. Independent NumPy checks202,400
  derivatives, two exact first Adam updates, eight validation checkpoints,
  179,200 code values and4,503,296 prediction pixels. Deployed competitive codes
  are96.85% nonzero; training slot counts are not a deployment sparsity guarantee.
  Evidence: `build/nn-report-conversion/winner_take_all/`.

- 2026-11-08: Added `nn/tied_autoencoder.hpp`: live transposed-weight
  ReLU/sigmoid forward values, independent-decoder control and target-free
  decoding. Contracts cover complete geometry, fixed leaf names/order/counts,
  both shared gradient paths, owning tensor fields/live aliases, standalone
  decoder-only leaves, signed codes, error timing and caller-owned fitting.
  All44 native model tests pass, including all original toy activities/eleven
  analytic derivatives, tied/untied first and second finite differences,
  live transpose checks after Adam and independent decoder divergence.
  All three public examples compile/run, the header compiles standalone,
  10/10 definitions pass fresh Doxygen and two clangd hovers pass.47 focused
  Python tests and264 R expectations across eight files pass. All source
  arrays in24 archives agree across original C++, report and normal Python/R.
  The seven-block/five-figure report retains96 updates per model, completed-
  epoch validation selection, raw controls and different-capacity limits.
  Independent NumPy checks76,896 parameter derivatives, two first Adam updates,
  eight validation checkpoints,89,600 code values and4,076,800 prediction pixels.
  Evidence: `build/nn-report-conversion/tied/`.

- 2026-11-08: Added `nn/layered_autoencoders.hpp`: configurable mirrored
  sigmoid stacks with strict encoder narrowing and standalone decoding using
  original layer indices. Contracts cover finite Float32/64, complete geometry,
  output order/code index, independent weights, owning vector/list elements,
  all gradients, error timing and caller-owned fitting/transfer policies.
  All39 model tests pass (five new stack cases), including all24 original toy
  activities/all48 analytic derivatives, first/second finite differences, frozen
  detached caches, independent copies of both decoders and an optimizer created
  before transfer. Both public examples compile/run, the header compiles
  standalone,6/6 definitions pass fresh Doxygen and two clangd hovers pass.
  Native Python/R use thin list/geometry facades;45 focused Python tests and239
  R expectations across seven files pass. All source arrays in42 report archives
  agree across original C++, native Python/R and the seven-block/five-figure
  report. Independent NumPy checks309,984 first derivatives, four first Adam
  updates,16 validation checkpoints and2,873,600 prediction pixels, with transfer,
  cached-code and RMSE-bound checks. Evidence: `build/nn-report-conversion/stacked/`.

- 2026-11-08: Added `nn/activation_sparse_autoencoder.hpp`: finite activation
  L1, dtype-consistent clamped mean-unit Bernoulli KL and complete untied sigmoid
  model objectives. Contracts cover units/normalization, batch coupling, signed
  codes, inclusive clip endpoints, zero abs derivatives, scalar rounding,
  invalid options before leaf creation and owning nested penalty lifetimes.
  All34 model tests pass (five new sparse cases), all four public examples
  compile/run, the header compiles standalone,27/27 definitions pass fresh Doxygen
  and three clangd hovers pass.43 focused Python tests and217 R expectations
  across six files pass. All source arrays in41 archives match original C++,
  native Python/R and the report. Independent NumPy checks404,800 first parameter
  derivatives,10,240 activity derivatives, four exact first Adam updates and
  5,644,800 predictions. All eight blocks, seven inspected figures and both export
  formats pass. Evidence: `build/nn-report-conversion/sparse/`.

- 2026-11-08: Added `nn/sequence_autoencoders.hpp`: checked image-row sequence
  conversion, final-state-only shared tanh encoding, independent repeated-code
  decoding and owning full/partial state histories. Contracts specify finite
  Float32/64, strict scalar bottlenecks, zero initial states, shared parameter
  names, full BPTT, no teacher forcing and undefined partial-result fields.
  All29 model tests pass (five new sequence tests); all five public examples
  compile/run, the header compiles standalone,12/12 definitions pass fresh Doxygen
  and four clangd hovers pass.41 focused Python tests and203 R expectations across
  five files pass. R fixture arithmetic now parenthesizes multiplication before
  modulo and checks [0,1] ranges, so its inputs match the C++/Python fixtures.
  All source arrays in13 archives agree across original C++, native Python and R,
  and with the eight-block/six-figure report. Independent NumPy verifies all4,956
  first derivatives, first Adam rounding,1,724,800 prediction pixels,14,336 state
  values and6,272 fixed-target input derivatives. Evidence:
  `build/nn-report-conversion/recurrent/`.

- 2026-11-08: Added `nn/masked_autoencoder.hpp`: checked patch indexing and
  differentiable inversion, exact source hash selectors, grouped position tables,
  visible-only transformer encoding, independent mask-token decoding and generic
  hidden-pixel MSE. Contracts specify Float32/64, finite model inputs, unique
  detached one-hot selectors, partial-result fields and owning tensor handles,
  parameter layout, discrete masks and continuous target gradients. All24 model
  tests pass; all12 public examples compile/run, the header compiles standalone,
  fresh Doxygen covers30/30 definitions and four clangd hovers pass. Python/R
  native help and owning-handle/error/unsigned-seed tests accompany normal package
  builds and exact source-array parity across all three languages;39 focused
  Python tests and180 R expectations across four files pass. Nine report
  blocks, six reviewed figures and Markdown/HTML exports pass. Independent NumPy
  checks all13,649 first gradients and1,724,800 predicted pixels. Evidence:
  `build/nn-report-conversion/masked/`.

- 2026-11-08: Added `nn/competitive_autoencoders.hpp`: detached stable signed
  top-k/lifetime selectors, explicit training/deployment forward paths and owning
  output handles. Full contracts cover ties, finite values, empty/strided inputs,
  selected slots versus nonzero counts, original parameter layout and winner-only
  local gradients. Extended `model_stage` to retain declared detached producers
  in Blocks diagrams, with unchanged gradients, producer-order independence,
  cycle rejection and documented view-dependent parameter reachability.
  All77 model/layer/tape/plot tests pass; three model and five changed plotting
  documentation examples compile/run,126/126 affected definitions and three
  hovers pass. Python/R help and forwarding accompany their native ownership
  tests;36 focused Python tests and152 R expectations across three files pass.
  Evidence: `build/nn-report-conversion/k_sparse/`.

- 2026-11-08: Added explicit MT19937/Box-Muller Gaussian banks and detached
  clipped Gaussian corruption to `nn/autoencoder_models.hpp`. Contracts cover
  Float32 coefficient/pixel rounding, signed zero, finite/range validation,
  strided/empty inputs, independent RNG state and retained clean targets.
  Thirteen native tests pass,12 header examples compile/run,30/30 definitions
  pass fresh Doxygen and seven hovers pass. Independent checks cover2.8million
  Gaussian values,3.1million corruption pixels and both complete first-step
  parameter gradients/Adam updates. Python/R help and native forwarding replace
  duplicated language-specific RNG/corruption and dense-model implementations.
  Rebuilt bindings pass35 focused Python tests and141 R expectations across
  three files; R numeric seed typemaps cover the full unsigned32 range.
  Evidence: `build/nn-report-conversion/denoising/`.

- 2026-11-08: Added the reusable spatial convolutional autoencoder and standalone
  nearest-repeat decoder to `nn/autoencoder_models.hpp`. Contracts cover NHWC
  geometry, dtype/stride support, original parameter names, initialization,
  partial store mutation on incompatible leaves and retained endpoint gradients.
  Ten model tests pass; all ten header examples compile/run,28/28 definitions
  pass fresh Doxygen and five model hovers pass. Independent Float64 forward and
  adjoint checks cover all741 parameter derivatives and156,800 held-out pixels.
  Python/R ownership and garbage-collection tests pass, with34 focused Python
  runtime tests and127 R expectations across three test files. Evidence:
  `build/nn-report-conversion/convolutional/`.

- 2026-11-08: Extended `nn/autoencoder_models.hpp` with the exact sigmoid
  encoder-Jacobian penalty and untied contractive forward objective. Full
  contracts cover coordinate sums versus batch means, retained activation/weight
  derivatives, strided inputs, nonfinite propagation and parameter-store mutation.
  Seven native tests include independent scalar gradients, full reverse-mode
  Jacobians, second-derivative finite differences, geometry and original model
  inventory. All six header examples compile/run, 16/16 definitions pass fresh
  Doxygen and three model hovers pass. Python/R documentation and value ownership
  accompany the SWIG surface; native garbage-collection tests pass in both.
  Evidence: `build/nn-report-conversion/contractive/`.

- 2026-11-08: Added `nn/autoencoder_models.hpp` with the reusable strict
  undercomplete ReLU/sigmoid model and its differentiable output type; the
  legacy SWIG-visible wrapper forwards to it. Added `Tape::to_markdown()` with
  escaped table cells to avoid nested result fences in executable reports.
  All three new documentation examples compile/run, 28/28 affected definitions
  pass fresh Doxygen coverage, and both new clangd hovers pass. Three model tests
  and the existing tape/model-plot suites pass (59 tests). Evidence:
  `build/nn-report-conversion/coverage.json`, `hover.json`, `native-tests.log`,
  `autoencoder-tests.log` and `header_examples.cpp`.

- 2026-11-08: Extracted Goertzel into a shared checked ordered kernel
  in algorithms/goertzel.hpp/.cpp. The existing Hz/bin signatures preserve the
  legacy offset-then-truncate signed index, unsigned unreduced angle, ordered
  recurrence and raw terminal expression. All7,152 frozen scalar cases/28,608
  lines and15,357 actual selection/sample/final events/43,965 lines match direct,
  installed and rebuilt old execution;2,137 nonempty plans/11,058sample updates/
  25empty returns independently certified. Original16stdout lines/SVG identical;
  four old DSP tests and FFT header unchanged. Finite negative raw power−2^-51
  versus exact2^-104 and DC cancellation at100k/1m retained explicitly, without
  clipping/widening. Finite samples/rate/target, safe truncated int range, actual
  selected angle/coefficient/evaluated state and terminal output checked, including
  invalid arguments before empty return. Existing valid fractional int boundaries
  remain admitted; huge unsigned bins are not silently reduced modulo N.
  Native GoertzelRun/GoertzelTrace expose actual two-state execution and selected
  sample updates, excluding validation/trigonometry/allocation/FLOPs/time. Complete
  fixed initial/sample/stop history is emitted online with default1m records,
  zero optional-off, no replay/truncation/partial result/input mutation. N4H6
  reject5/admit6, empty2reject1/admit2. Input/root/ordinary run+selection/constant
  working state/allocator excluded; records are not entries/bytes/peak RAM.
  Unrounded frequency response is separate and does not alter legacy selection.
  Native sinusoid_signal/dtmf_signal own output, validate every descriptor first,
  admit whole N samples before allocation (default1m/0off), preserve ordered
  descriptor accumulation and the original410sample generator bit-for-bit.
  DtmfPeaks holds eight native responses, first maximum per canonical group and
  aggregate8N updates. It is only a candidate: silence/empty ties choose1; no
  amplitude, timing, tolerance, noise or speech acceptance test is claimed.
  Independent direct complex Fourier sums check7,023moderate-bin cases; large
  unreduced-angle deviations remain disclosed witnesses. Versioned exact Q(sqrt2)
  Fourier fixtures cover4,786cases/35,438input entries without recurrence or FFT.
  Eight strict-fno-wrapv ASAN/UBSAN tests, native21+5dir and combined3,199 with no
  skips pass; whole two paths67definitions audited,17public examples compiled/run
  with exact stdout,11clangd hovers, final installed code/read-only section parity
  and freshness checked. Registry2,415classes/16,260members/1,787KiB; two paths
  added to completed manifest. TI SPRA066 method/implementation/summary and
  official Q.23/Q.24 relevant sections read and PDF pages inspected. Original1958
  bibliography verified; inaccessible full article not claimed read.
  Seven complete C++ programs/seven inspected local SVGs/all108computed lines in
  each Markdown/HTML export and matching figure bundles. Independent report
  Fourier oracle checks29selector rows,8DTMF rows against frozen original powers,
  five candidate cases,36MN work rows,10DC rows and four hand-checkable states.
  Concept diagram near introduction, selector boundaries/negative frequency,
  original410sample pair and unrounded comparison, invalid-input candidates,
  selected work, finite numerical limitations and separate quotas demonstrated.
  Numerical axis uses ppm for distinct ticks; printed percentages unchanged;
  six other SVG hashes unchanged. No PSD/calibrated-power/complete-receiver/
  timing claim or forced external dataset. Website137reports/658blocks/593figures
  verified. Original pair removed only after all gates; eight candidate old images
  absent; catalog API/selector/candidate semantics/destination updated.
  Evidence: build/mepml-conversion/goertzel/.

- 2026-11-08: Extracted gnome sort into a shared checked diagnostic kernel
  in algorithms/gnome_sort.hpp/.cpp. exchange_sorts.hpp forwards the existing
  signature and removes its duplicate declaration; exchange_sorts.cpp loses only
  the old gnome definition. Odd-even/pancake bodies are byte-identical, the prior
  cocktail module is untouched, and all3 original exchange-sort tests remain
  unchanged (including6,000randomarrays and4,000pancake replays).
  All10,396 scalar cases/20,792lines and15,648 actual pointer/pair steps over758
  observed calls/37,198total lines match direct preflight, installed and rebuilt
  old execution. Original5stdoutlines/SVG byte-identical; all27 original example
  visits checked against the frozen old actual observer. Legacy by-value signature,
  boundary short-circuit, left<=right forward, strict adjacent swap/back, duplicate
  stability and int extremes retained. Scalar sorting has constant auxiliary
  state beyond its owned input/output, no identity/history allocation or diagnostic
  counter updates. Lvalue input copying costs O(n).
  GnomeSortRun owns stable original identities carried by actual exchanges and
  checked selected work: visits, key comparisons, swaps, forward/backward steps
  and zero-index boundary visits, excluding copying/identity initialization/
  allocation/loop tests/FLOPs/time. Identity output/initialization cost O(n).
  GnomeSortTrace emits fixed-size initial/actual-step/stop records online, without
  full-array snapshots, replay, truncation, partial return or caller lvalue mutation.
  Every metadata record counts: default1m complete records/0optional-off; empty2
  reject1/admit2, singleton3reject2/admit3, [2,1]6reject5/admit6 including the final
  stop append. Inputs/root/ordinary sorted+identity outputs/constant working state/
  allocator overhead excluded; not numeric entries/bytes/peak RAM. Copying and
  identity allocation can precede history rejection. Output-only/run have no
  history quota. Native adjacent_sort_inputs generates four owned controlled
  arrays: ascending, descending, one middle adjacent inversion and last minimum.
  Whole4n integer entries admitted before allocation, default1m/0off; n-1 int
  representability, 4n overflow and capacity checked, with no overflowing final
  int iota increment. n4whole16reject15/admit16; deep ownership verified.
  Own sorted-prefix/stability proof and potential P=n-i+2I give exact visits
  S=n+2I0, swaps/backwards I0, forwards n+I0 and comparisons S−Z. Z equals one
  plus original strict prefix record minima for nonempty input, zero for empty.
  Independent direct stable key+original-index ranks, pairwise inversions and
  prefix minima verify all10,396 cases; every15,648 recorded step checks actual
  before pair, branch, pointer and resulting values against those counts. Versioned
  fixtures2,097cases/12,428input entries and600actual pair/identity/prefix/potential
  certificates are native regressions. Known family inversion counts tested for
  n0..64 and exact reported36rows/ninepositive sizes. Stability/extremes/empty/
  singleton/online-final-empty limits/ownedcopies/wholegenerator admission pass.
  Seven strict-fno-wrapv ASAN/UBSAN tests, native19+5dir and combined3,178 with no
  skips pass. Whole new2paths44definitions plus existingforwardheader4/total48
  audited; all10public examples compiled/run with exact stdout and7clangd hovers.
  Final counter wording correction rebuilt documentation/core; compiled .text,
  .rodata and .data.rel.ro match the already-verified build exactly, installed
  freshness/code parity holds, and focused24tests/hovers rechecked. Two new paths
  added to manifest; existing exchange header remains once. Registry2,410classes/
  16,231members/1,784KiB. Official NIST entry read in full, original Sarbazi-Azad
  2000 newsletter bibliography verified there; unavailable author pages not
  claimed read, Grune historical-date uncertainty not invented.
  Six complete C++ programs/six inspected local figures/all90computed output lines
  in each Markdown/HTML export and identical image bundles: actual [3,1,2] pairs/
  seven visits/nine records; original27walk/10swaps/24comparisons/3boundary/29records;
  equal keys stable ids[1,3,0,2]; ascending/reverse/adjacent/last-minimum36rows;
  equalI3tenvisits but8versus7comparisons/all_equal4visits3comparisons; complete
  history versus whole four-array entry limits. All six SVG hashes unchanged on
  review. Counted visits/key comparisons/retained records kept distinct; precise
  linear time when I=O(n), no equality with ordinary insertion-sort comparison
  counts and no timing claim. Controlled inputs need no external dataset.
  Website136reports/651blocks/586figures verified. Original paired reports removed
  only after all gates; eight candidate old images absent; catalog native API/
  count semantics/new destination updated. Evidence: build/mepml-conversion/gnome-sort/.

- 2026-11-08: Moved planar GJK into a shared checked native kernel
  in geometry/gjk.hpp/.cpp, keeping the original distance signature/default100,
  ordered double support/projection/triangle recurrence, first support ties,
  absolute squared-norm1e-20/progress1e-10 cutoffs and final norm at the loop cap.
  All4,762 scalar cases/58,680 lines and38,470 actual state records/97,150 lines
  match direct preflight, installed and rebuilt old execution. Original7stdout
  lines and SVG byte-identical; original5 numeric-batch tests, Point2D and unrelated
  sim3D GJK/EPA/collision sources unchanged. Inputs are finite 2D point clouds
  representing convex hulls, including arbitrary order/duplicates/singletons/
  collinear points. No concave-body, curved-support callback, 3D, closest-pair
  witness, penetration depth, time-of-impact or exact contact guarantee is added.
  Complete coordinate validation precedes empty convention, loop budget and
  history work; evaluated nonfinite intermediates are rejected before decisions.
  Frozen NaN/infinity/ignored late NaN/empty-masked NaN/nonfinite norm and finite
  subtraction-overflow behavior is corrected. Finite underflow/cancellation and
  floating triangle predicates remain, with the finite legacy scale limitations
  preserved rather than silently changing the method: positive5e-11 point gap
  returns0; scale1e-6 squares with truegap2e-6 return6e-6; crossingsegments can
  return sqrt2*1e-6. Empty zero is a convention, not geometric contact.
  Native GjkRun owns actual stopping reason, final constant-sized feature and
  selected work. GjkTrace captures complete fixed-size actual state records online,
  including initialization, every queried support/actual returned feature and
  stop/empty metadata, with no numerical replay/truncation/partial return/input
  mutation. Default1m records,0 optional-off; complete records not scalar entries,
  bytes or peak RAM. Input/root/ordinary result/final feature/constant working
  storage/allocator overhead excluded. Separated singleton4reject3/admit4;
  near-origin3reject2/admit3; empty1admit1 and quota0off. Loop-budget0 still initializes
  and returns iteration_limit with2records, a different unit from the history cap.
  Selected actual support calls/dots/strict comparisons/loop visits/progress and
  feature reductions exclude validation/FLOPs/allocation/time. Complete-scan dots
  equal(n+m)*queries, comparisons=dots−2queries. Honest O((K+1)(n+m)) search plus
  validation, O(1) ordinary auxiliary storage; no fixed-convergence/linear-time
  theorem inferred from the bounded default. Native regular_polygon ports the
  report helper's exact ordered angle/coordinate formula, with finite scalar and
  computed output checks, whole vertex count/quota/capacity before allocation,
  default1m/0optional-off, zero-radius repeated points and owned output.
  Independent Fraction oracle enumerates ALL pairwise Minkowski differences,
  constructs an exact monotone-chain hull and exhaustively projects onto its
  boundary, without GJK support/simplex recurrence:4,759nonempty cases/40,454 hull
  vertices;3empty legacy0 cases excluded from geometric comparisons. Moderate-scale
  max absolute error1.78e-15; all-scale errors and3falsezero cases retained/disclosed.
  Versioned exact fixtures1,442cases/8,353storedpoints (225corner-subset pairs,
  961collinear-subset pairs,256randomclouds);400actualsupport-firsttie/feature-
  geometry/work certificates, all stopping reasons and budget/history/ownership/
  malformed/range/generator tests pass. Seven strict-fno-wrapv ASAN/UBSAN cases,
  native14+5dir and combined3,159 with no skips pass. Whole2paths79definitions,
  all11public examples compiled/run with exact stdout and7clangd hovers, final
  installed freshness/code/read-only parity pass. New2paths added to manifest.
  Original GJK1988primaryPDF pages4/6 read/raster inspected for support/descent
  versus scaled numerical criterion/backup; retained closest-feature attribution
  Ericson2005 RTCD Sections5.1.2/5.1.5 verified against author site/table of contents.
  SixcompleteCPP/sixinspectedlocalfigures/allcomputedMDHTMLoutputs/copiedimages:
  analyticalsquaregap3/Minkowskidifference, actualsegment t3/5closest(-.4,.2) and
  triangle-inside branch,27originaldemo/sweepvalues,13scalecontrols/tinygapzero,
  ten actualsize-workrows,record/wholevertex/loop/domaincontrols. Active segment
  highlighted/label moved and duplicate legends removed; other3SVGs unchanged.
  Website135reports/645blocks/580figures verified. No forceddataset for the kernel.
  Original paired reports removed only after gates; eight old image candidates
  absent; catalog API/approximation/convex-hull scope/destination updated.
  Evidence: build/mepml-conversion/gjk/.

- 2026-11-08: Extracted Girvan–Newman into a shared checked native module
  in algorithms/girvan_newman.hpp/.cpp, with network_analysis.hpp forwarding the
  old include. Six non-GN functions (validate/normalize/PageRank/system/TrustRank/
  HITS) remain byte-identical and original7 graph tests unchanged. All9,912 scalar
  cases/137,397 lines and34,011 actual primitive events over390 observed cases/
  171,798 lines match direct preflight, installed and rebuilt old execution;
  original2stdout lines exact. Shared ordered-double Brandes/raw source sums,
  canonical minmax/sort/dedup/loop ignore, adjacency/BFS order, every source,
  lower_bound score-slot search and near-maximum raw1e-12 simultaneous deletion
  preserved. Conventional scores are raw/2. Target is at least: cycle4 overshoots
  target2 to4groups; disconnected early stop and empty n0target1 remain explicit.
  Native edge betweenness, actual-work run/owned primitive traces, original-graph
  modularity and bounded clique-bridge/layered families added. Nonfinite checks
  precede propagation/division/deletion, fixing frozen layer1030 path overflow/
  65,728 nonfinite contributions/NaN maximum/delete-all2060-singleton result.
  A265-vertex/318-edge exact uint64 regression has mathematical2^53+1 paths
  rounded back to2^53; the guarantee now checks the bound BEFORE addition and
  clears correctly, with unchanged actual double recurrence. Layer60 remains
  finite beyond2^53; conservative integer-count guarantee false
  is not an assertion every larger integer rounded. Finite operation order is
  retained. Vertex distance/counter/container capacities checked. Selected work
  includes n² source slots and actual binary-search comparisons, not timing or
  all costs; honest O(r log(r+1)+m n²+n m² log(m+1)) total bound plus terminal
  work, O(n+m) scoring scratch and O(r+n+m) ordinary auxiliary storage with raw
  deduplication capacity and additional history storage explained. Modularity uses original edges including
  removed ones, validates complete disjoint groups and defines edgelessQ0; no
  global modularity optimum or true-community certificate is claimed.
  Online histories count all fixed primitive and canonical/round/component
  metadata records, default1m/0 optional-off. No replay, partial/truncated return
  or caller mutation. Inputs/root/work/final ordinary output/working BFS-preds-
  queues/allocator storage excluded; not numeric entries/bytes/peak RAM. Tiny
  graph33reject32/admit33, emptymetadata4reject3/admit4, standalone edge-score19
  reject18; whole generator vertices6reject5/admit6 are separate units.
  Native bounded GML topology ingress preserves source-id mapping and input
  edge order/direction/multiplicity while skipping unknown balanced attributes;
  malformed/duplicate/missing/unknown ids, int64 and byte/node/edge limits checked,
  iterative deep-list skipping and copied output tested. New karate_club()/fetch
  path reuses native verified file/ZIP extraction/shared cache/offline/mirror;
  original author ZIP970bytes and both members pinned, corrupt cached bytes
  rejected without replacement.34vertices/sourceids1..34 mapid−1,78undirected
  unweighted edges, no faction labels. Native fresh acquisition/all78 independently
  parsed edges/offline reuse pass. Dataset catalog adds graph row, load() directs
  callers to graph accessor, downloadable names9; existing14 offline loader tests
  updated only for catalog24/download9 and graph refusal, all pass.
  Independent exact ALL-simple-path Fraction oracle checks7,356 partitions,
  including all5,406 simplegraph0..5/valid-target cases and799actualraw scores;
  versioned fixtures retain1,100graphs/5,406partitions/scores. Independent karate
  Floyd distances plus explicit shortest-path enumeration/Fraction weights agrees
  with78initial scores,34target requests and all33actual batch stages/modularity.
  Visited Qmax0.4012984878369494 at round21/fivegroups; firstsplit round10/twogroups
  Q0.359960552268,11removals. Full actual174,531-record history/32scored batches/
  78removals retained. Eight algorithm and six reader/cache strict sanitizer cases
  cover exhaustive fixtures,160actual BFS/path/dependency/cut/work certificates,
  direct-pair modularity, domains/order/normalization/quotas/ownership/generators,
  finite and overflowing path counts, malformed GML/10,000-deep lists/limits and
  real cache corruption. Visual review found directed arrows on undirected graph
  figures; added canonical Diagram::link with shared boundary clipping and forced
  no arrowhead, leaving existing edge() semantics intact. Eleven strict diagram
  sanitizer cases (nine existing+two new) pass; three pre-change directed SVGs
  remain byte-identical after the shared implementation refactor. Only bridge,
  diamond and cycle figure SVGs change when using undirected links; other five
  inspected SVG hashes unchanged. Native44graph/data+11diagram+5dir/combined3,145
  pass. Whole4newpath132definitions plus existing2diagram paths63/total195 audited;
  all20 changed/public snippets and stdout,17clangd hover targets, final installed
  freshness/code/read-only parity pass. Four new paths added to manifest; existing
  diagram paths remain once. Eight complete C++ programs/eight inspected local
  figures/all computed outputs in Markdown+HTML and identical figure bundles pass:
  bridge9raw18; actual half-path diamond; original bridge/cycle/disconnected stops;
  native karate firstsplit/layout drawing only; all33 original-edge Q stages;
  selected work family; finite60layer versus1030rejection; online/whole controls.
  Primary GN2002PDFp3 single-edge rule, Brandes2001p7 dependency formula/p9 divide2
  and NG2004PDFp8 original-graph modularity read/raster inspected; Zachary1977
  publisher metadata/author collection README verified. Native batch variant,
  floating limits, no invented labels/timing/global-optimum/social-truth claims
  explicit. Site134/639blocks/574figures verified. Original paired reports removed
  only after all gates; six candidate old image paths absent; catalog API/variant/
  destination updated. Evidence: build/mepml-conversion/girvan-newman/.

- 2026-11-08: Converted geometric hashing to shared checked native kernels
  in geometry/geometric_hashing.hpp/.cpp. Original table/match fields and helper
  names retained; ordered double quotient, absolute squared-basis cutoff1e-18,
  floor/signed32 packing, bucket multiplicity, every scene basis, unordered tally
  iteration/strict global improvements, exact recovery and first-hit verification
  preserved. All2,748 scalar cases/242,814 lines and54,482 primitive events over80
  observed cases/297,376 lines match direct preflight, installed and rebuilt old
  execution. Original6 geometry tests/6stdout lines/SVG and completed Point2D are
  unchanged. Complete finite points/quant/tolerance-square/positive threshold and
  public table-id validation precedes work/capture; four frozen float-cast UBSan
  failures, zero-vote negative-index UBSan and positive-id ASan overflow corrected.
  Required finite intermediates/signed floor ranges checked before casts; no
  arbitrary ordinary point-count cap. Same ordered model/table provenance remains
  caller responsibility. Native checked frame/bin/vector similarity transform,
  ordinary actual-work results and owned primitive histories share the kernels.
  Online complete-record cap defaults1m/0 optional-off, before each append;
  no synthetic initial record, numerical replay, partial/truncated return or
  caller mutation. Input vectors/root/work/final ordinary table or candidate,
  working tally/hash and allocator storage excluded; not numeric entries/bytes/
  peak RAM. Triangle build12reject11/admit12; recognition31reject30/admit31;
  empty0 and complete invalid-before-quota/deep-copy controls pass.
  Nine strict ASan/UBSan/float-cast tests cover1,024 independent exact Fraction
  frame/bin fixtures,2,000 integer-grid table comparisons,120 actual trace/work/
  tally/first-hit certificates, transforms/reflection/tiny cutoff, signed limits,
  malformed inputs and candidate/inlier/duplicate controls. External Fraction
  oracle checks2,688 table cases/201,564 entries and1,552 unambiguous vote maxima/
  49,198 scene bases; exact-real equivalence is not claimed for excluded boundary
  cases. Native17+5dir/combined3,099 pass; whole2path102definitions, all20 public
  snippets/stdout,8clangd hovers and final built-installed code/read-only parity/
  freshness pass. Header/new source added to completed manifest; Point2D once.
  Seven complete programs/seven visually inspected SVGs and all computed lines
  in Markdown/HTML with source-build-export image parity pass: hand(.5,.5) frame;
  actual tiny table/vote and threshold; original6model+4clutter at scale1.7/angle.9/
  translation12,-8 recovered; same scene atq.5 yields8votes/2inliers despite true
  copy; duplicate4inliers against3scene; nextafter floor/reflection/absolutecutoff;
  actual M3..12 cubic frames ending1,320 with all poor candidates retained;
  online records/ownership/invalid/empty. Legend overlap and generic bar labels
  polished; only gh_votes/work/controls SVG hashes change. found means recovered
  candidate, not inlier acceptance; no alternate retry/refinement/one-to-one
  matching. Own complex cancellation/reflection, floor-boundary and frame-count
  derivations; selected counts not total/timing; per-call validation plus expected
  hash O(T+N+M+M^3+V+NM), O(N²) tally and collision-list costs explained.
  Original Lamdan/Wolfson1988 metadata verified and paper238–240 raster read,
  Sections2.1–2.2 distinguished from this native one-model/global-only/first-hit
  variant. No forced dataset, general-affine, portable-tie, timing or universal
  clutter-success claim. Site133/631blocks/566figures verified. Original paired
  reports removed only after all gates; six old image paths absent; catalog
  corrected from general affine to supported planar similarity and destination
  updated. Evidence: build/mepml-conversion/geometric-hashing/.

- 2026-11-08: Converted geohash to a shared checked native kernel in
  geohash.hpp/.cpp. Legacy three-argument encoder preserves longitude-first,
  upper-on-equality, ordered double midpoints, alphabet and all requested steps;
  valid negative precision including INT_MIN still clamps to zero. No arbitrary
  positive precision limit or early stagnation exit. All107980 frozen scalar
  outputs and82060 actual bisections/16412 emissions over544 observed cases/
  206996 lines match installed and rebuilt old execution; original11 calendar/
  geometry tests,20 stdout lines and SVG preserved. Complete finite inclusive
  coordinate checks now precede arithmetic. New run/work and owned bit/emission
  trace share the actual encoder kernel; initial coordinates/bounds plus5p bits
  andp emissions require1+6p complete records. Checked count/capacity/whole quota
  before history initialization/allocation; default1m/0 optional quota off;
  not numeric entries, bytes or peak RAM. Root/work/final ordinary output,
  working and allocator storage excluded; no partial/truncation/replay/mutation.
  Strict fully validated lowercase decoder accepts empty world and returns
  actual unrounded bounds/center/spans and precision flags; no high-precision
  roundtrip guarantee. Ideal angular resolution explicitly distinct from actual
  double endpoint stagnation/collapse and representational underflow. Full-key
  alphabet validation precedes common-prefix comparison, even differing heads.
  Own interleaved integer-grid derivation, ceil/floor5p/2 axis bits, alternating
  8/4 and4/8 contraction, angular area/32 and prefix nesting explained; nearby
  points across a split and equivalent end meridians share zero characters.
  Ten strict ASan/UBSan/float-cast tests include2643 independent exact Fraction
  integer-quantization fixtures,32768 exhaustive three-character integer-grid
  cells,1000 actual random trace certificates, domain/count/alphabet/empty/owned/
  quota/INT_MAX-before-allocation controls and4096-character no-early-exit work.
  Independent Fraction quantization checks74760 frozen queries. Canonical
  point first unchanged bit110, no collapse and positive actual1024 spans while
  ideal spans underflow to zero; pole/end-meridian collapse109 distinguished.
  Native30+5dir/combined3082 pass; whole2path78 definitions, all15 public examples
  and stdout,8 clangd hovers, final installed freshness/code/read-only parity
  pass. Header/source added to completed manifest. Six complete C++ programs/
  six visually inspected local SVGs, computed outputs in Markdown and HTML,
  source/build/export image bytes and site132/624blocks/559figures pass. Report
  covers actual01101→13→e, original ezs42 and London/Greenwich/Sydney keys,
  resolution, boundary counterexamples, endpoint stagnation and record controls.
  Primary Chris Veness author page and source verified; strict-case, empty-key,
  unrounded-center and validation variants documented. No forced dataset,
  physical-distance certificate, historical invention or timing claim. Original
  paired reports removed only after all gates; six candidate image paths absent;
  catalog updated. Evidence: build/mepml-conversion/geohash/.

- 2026-11-08: Converted generational collection to a shared checked native kernel
  in generational.hpp/.cpp. Legacy result layout, binary-age outcomes, root and
  old-field seed order, all-seeds-before-LIFO behavior, ascending classification
  and actual full comparison are preserved. All34036 valid cases/387529 records
  match independent full/young Warshall closures and direct preflight/installed
  output. All60561 primitive events over788 frozen cases/448878 observed lines
  match direct preflight, installed and rebuilt old execution. Original9memory
  tests and completed object_graph.hpp remain byte-for-byte unchanged; original
  7stdout lines/SVG exact. Complete graph validation then exact binary ages now
  precede indexing/copy/quota; four recorded ASan heap overflows (short ages,
  high root/child and short rows) and a nonbinary-age false-free are corrected.
  Slot-only size metadata remains ignored, including invalid unused values.
  Native ordinary work/full marks and owned primitive traces share the actual
  kernel; no numerical replay. Online actual event quota defaults1m/0 disables
  optional quota only; complete primitive records are not numeric entries/bytes/
  peakRAM. Input vectors, root/work/final ordinary output, working and allocator
  storage excluded; no partial/truncated return or caller mutation. Singleton8
  records rejects7/admitted8; empty0records. Separate family generator admits
  whole object counts before allocation and checked int-id capacity: original7
  objects reject6/admit7, with safe excessive-size and zero-old-count guards.
  New owned fixed-age GenerationalHeap validates/indexes once and maintains exact
  old-to-young(source,field) slots on checked existing-field replacement. Sorted
  hinted construction O(N+E+R); ordered-set writes O(log(W+1)); allocate new key
  before mutation, duplicate fields independent, invalid writes atomic and deep
  copies independent. Ages/roots/field counts fixed. Sparse collect shares young
  kernel with O(Y+R+W+E_y) per-call work/storage: no old scan, whole revalidation,
  full comparison or N-dense output. Setup, explicit graph snapshots and writes
  have separate costs. Neither analysis physically promotes/frees/copies objects.
  Legacy total O(N+E+R) includes discovery, dense output and full comparison;
  minor_scanned is only young pops. Unreachable old sources can conservatively
  retain floating young garbage, making minor visits exceed full-root liveness.
  Own last-old-to-young-crossing proof establishes full-live-young subset of
  retained young, not equality; maintained exact-slot update invariant explained.
  Ten strict ASan/UBSan/float-cast tests cover33033 exhaustive all graphs0..3/
  roots/binary ages with independent Warshall and eager/sparse equality,10000
  actual writes independently rechecked after each,300 actual event/work/stack
  certificates, hand seed3,1→visits1,2,3/full3,0,1,2, floating safety, full invalid
  graph/age-before-quota and ignored sizes, online/whole quota/owned controls,
  old chains through100000 with separate setup/constant young work,100000-young
  chain iterative traversal and duplicate-slot/atomic-write/deep-copy behavior.
  Native28+5dir/combined3052 pass; whole2path77definitions/all18public snippets/
  stdout/8clangdhovers and final built-installed code/read-only parity/freshness
  pass. Header and new source added to completed manifest; ObjectGraph once.
  Six complete C++ programs/six inspected figures/all computed result lines in
  both Markdown/HTML exports/source-build-export image parity pass: tiny boundary;
  original7heap (young2/full6, remember3, free6, old/full/validation fields4 and
  dense7); rooted minor2/full3 versus unrooted2/0; maintained writes with slots
  0,1,2,1,0 and survivors0,2,2,1,0, no old scan and explicit separate rebuild
  verification; old1..4096 eager discovery/full grow versus paid setup and fixed
  remembered1/young2/classified3; online8/8/7 versus whole objects7/7/6 plus owned,
  invalid-before-quota, atomic writes and empty controls. Intro arrow/plot legend
  shortened after visual review; final SVG hashes confirm other figures unchanged.
  Original Ungar1984 publisher metadata/DOI10.1145/800020.808261 verified; original
  Sections7–8 read, printed161 §8.1 raster inspected. Store-side remembering/new
  traversal distinguished from copying/survivor-space/tenuring runtime policy.
  Empirical young-death tendency is workload-dependent; no production-version,
  pause-time, timing, external-dataset or universal-performance claim. Site131/
  618blocks/553figures passes. Originals removed only after all gates, six old
  image paths absent, catalog API/destination updated. Evidence:
  build/mepml-conversion/generational/.

- 2026-11-08: Converted General Problem Solver to a dedicated checked forward-planning module
  (general_problem_solver.hpp/.cpp), reexported through search_more.hpp. The old
  recursive state-only visited set could miss bounded plans. One shared heap DFS
  now records best expansion depths and reopens strictly shallower arrivals.
  Legacy record layouts/signature/default32, operator order, add-wins effects,
  initial-goal success, failure placeholders and all64 fact bits are retained.
  Independent BFS checks397686 frozen cases (327680 exhaustive two-fact cases,
  70000 random cases and6 controls):397683 results unchanged,2 false negatives
  recovered and1 formerly successful case selects a different valid first DFS
  path (3 rather than2 actions). No successes lost; all266377 returned plans are
  independently valid. DFS explicitly does not promise shortestness. Direct
  preflight/installed outputs match; original empty-stdout/exit0 make program
  preserved. All2171129 frozen observed lines reproduce from rebuilt old source.
  Of931 primitive event traces,925/6496 events match exactly; all6 changed traces
  contain actual shallower reopenings (6909 new versus6776 old total events).
  Unrelated shared search bodies/declarations and original6tests unchanged.
  Actual visit/test/apply/goal/cutoff/repeat/reopen/backtrack/finish/exhaust events,
  selected operator indices and work come from the executing kernel, without
  replay. Online fixed-schema event-record quota defaults1m/0 disables optional
  quota; each next capture is checked, no partial/truncated return or caller
  mutation. Input operators/name bytes/root/work/final ordinary/working/allocator
  storage excluded; records are not numeric entries, bytes or peak RAM. Original
  make needs6 events including finish: cap5 rejects6 admits; initial goals/zero
  budget still produce2 events. Independent checker validates every index before
  whole p+1 prefix-state quota/capacity and transitions; early failure retains an
  actual valid prefix with no admission discount. Applicability and goal flags
  are separate, names may duplicate/be empty, and indices disambiguate them.
  Native carry-operator generator accepts1..64 bits, with safe high-bit masks.
  Ten strict ASan/UBSan/float-cast tests independently check327680 exhaustive and
  21000 random three/four-fact bounded BFS outcomes,400 actual event/work
  certificates, reduced old counterexample4→6→1→7, goal/depth/repeat precedence,
  precondition failure/full-index-before-quota, online and whole caps, snapshots,
  empty/duplicate/NUL labels, add-wins/highbit63/invalid enum/domain guards.
  Counter family1..16 bits executes deep32768-step plan on32769 heap frames;
  all32769 certificate states independently equal0..32768. Exact work is65536
  visits/65535 checks/32752 repeat prunes/15 depth cutoffs at16 bits, no recursion.
  Native25+5dir/combined3024, whole2path76definitions/all14header snippets/stdout/
  8clangdhovers and final installed code/read-only parity/freshness pass. Two new
  manifest paths; incomplete unrelated search_more files intentionally excluded.
  Five complete C++ programs/five inspected figures/40computed lines per MD/HTML
  export/source-build-export image parity pass: six-event make, independent door
  states0,1,2,6; actual reopen6 depth2→1 with remaining budget1→2 and certified
  path4,6,1,7; budget4/5 nonshortest paths; binary exponential counts; online
  records6/6/5 versus whole certificate states3/3/2 with highbit/identity controls.
  Own bit-mask, soundness/depth-dominance/bounded-completeness/finite-state cost,
  carry progress/optimal family length/exact work and quota derivations. Original
  Fikes/Nilsson IJCAI1971 introduction/II.A–B printed608–610 read; printed609 PDF
  raster inspected. Ground positive-fact forward DFS is distinguished from
  historical GPS means-ends strategy/first-order STRIPS theorem proving; no
  external action, dataset, timing or general shortest-plan claim. All figures
  passed first visual review. Site130/612blocks/547figures/5757pages passes;
  original mepml/org removed only after gates, six old image paths absent and
  catalog API/destination corrected. Evidence: build/mepml-conversion/general-problem-solver/.

- 2026-11-08: Converted Gauss–Seidel with actual mixed-vector coordinate histories
  in gauss_seidel.hpp and the existing conjugate_gradient.cpp stationary kernel.
  Legacy scalar/residual-only signatures/layouts/zero-start/ordered widened sums,
  checked double assignments, double-rounded dense A*x residual and inclusive
  stopping are preserved. Existing residual-only history retains its online cap.
  All20530 systems/16 unrelated CG controls match in146016 frozen records; all
  116331 actual numerator/old/new/before-after coordinate events match in495009
  observed records across direct preflight, installed and rebuilt original.
  Original5stdout lines/SVG exact. Pre-existing cached3.72223e-11 is identified
  as stale: actual unchanged pre-conversion program outputs3.72241e-11, which is
  preserved. Original14CG tests/helpers/kernel/public wrappers/Laplacian fixtures
  and CG/legacy linear headers remain byte-for-byte unchanged. New ordinary
  run/work and full actual coordinate/sweep history share this one execution,
  without numerical replay or additional residual products. Counts include zero
  dense coefficients: kn coordinate assignments/divisions, kn(n-1) offdiagonal
  products, kn² residual products and k+1 norm evaluations, first norm from b.
  Complete shape/finiteness/nonzero diagonal/controls validated before initial
  convergence/quota. New whole history bound n+1+K[n(1+2n)+(1+n)], empty1;
  initial vector/norm, per-coordinate metadata+two vectors and per-sweep metadata
  plus vector included. Fixed metadata contains scalars/work. Units are entries,
  not bytes/peakRAM; original A/b, final ordinary output, root metadata, working
  and allocator storage excluded. Positive quota/default1m,0 disables optional
  quota only; checked counts/capacities precede initialization/history allocation,
  no early-success discount, truncation, partial returned history or caller
  mutation. Ordinary O(kn²) work/O(n) extra storage; actual history O(kn²).
  Exact one-sweep n2 needs16 rejects15/admitted16; zero-RHS n1/K5 needs27 rejects
  cap2 despite initial convergence, while legacy online cap1 admits initial
  success and rejects a needed sweep. Empty/INTMAX budget history needs1.
  Ten strict ASan/UBSan/float-cast tests independently check exact binary q-family
  powers/error/radii/residual including q1.25 divergence, hand mixed reads,
  SDD nilpotent transient residual1→64→0, reversed variables mapped back,
  1000 integer strictly dominant3x3 against Cramer determinants, domain-before-
  quota/nonfinite/negative/zero/overflow/unrepresentable norm and division,
  complete bound/capacity arithmetic/owned snapshots/actual dense work through64.
  Native33target+5dir/combined2999 pass; whole3path90definitions/all8newheader
  snippets/stdout/10clangdhovers pass; final installed code/read-only sections
  and freshness verified. One new manifest header; existing shared two once.
  Five complete C++ programs/five inspected native figures/41computed result
  lines in each Markdown/HTML export/source-build-export image parity pass.
  Actual tiny updates(.5,0)→(.5,.75); original3x3 solution(2,4,3)/13sweeps with
  residual3.722410353787e-11/off78,res117,norm14; q=.25,.5,.75,1.25 observed ratios
  q² versus analytic Jacobi|q| (no Jacobi solver/timing claim); strict dominance
  does not imply monotone residual, triangular1,64,0/order(.5,.75)vs(.75,.5);
  native Laplacian one-sweep dense scaling64→4032off/4096res/8386history entries;
  complete16-16-15 quota and early-zero/online/ownership/validation/empty controls.
  Primary Netlib Templates second-edition author list and full Gauss–Seidel
  subsection read; native dense widened absolute-residual variant stated. Own
  scalar/triangular/error splitting, exact spectral criterion, SDD infinity-norm
  induction/SPD coordinate-energy contraction, q-family closed form and work/
  quota derivations. Corrects universal twice-Jacobi/straightlog/monotone/solely
  matvec claims; residual not forward-error or exact floating residual certificate,
  zero has no fabricated logarithm, scaling affects absolute stopping. No forced
  external dataset or benchmark. Rate-plot legend initially covered divergent
  endpoint; raised headroom, reran5programs and verified four unchanged SVG SHAs.
  Site129reports/607blocks/542figures passes; original mepml/org removed only after
  gates, six old owned image paths absent and catalog API/destination updated.
  Evidence: build/mepml-conversion/gauss-seidel/.

- 2026-11-08: Converted Gauss–Legendre pi into an actual fixed-count double kernel
  in gauss_legendre_pi.hpp/.cpp. Legacy initialization, scalar signature, ordered
  means/left-associated p*delta*delta/t subtraction and doubling are preserved.
  All2227 frozen records match: scalar counts0..1024, full1024 actual before/after
  state/estimate records including last unused p=Inf,103 BBP controls,32 Chudnovsky,
  42 CORDIC and exact40-hex-digit Machin output. New preflight/installed/rebuilt
  original match; original9stdout lines/SVG exact; original8tests unchanged/pass.
  Only AGM body moved; other BBP bodies unchanged. A source-file purpose comment
  refresh is the sole additional source difference and preserves all compiled
  text/rodata/data.rel.ro after fresh XML/registry/core/install verification.
  New owned run/trace/used-correction/change flags/work and reference diagnostics
  share the actual recurrence without replay, early stopping, widening or forced
  equal means. Legacy negatives retain zero-stage compatibility. Checked analysis
  counts are0..binary-double maximum exponent(1024 here); excessive old NaN suffix
  is rejected before work. Every used weight, correction, mean, t and estimate is
  finite; the last unused p overflow is explicitly retained, never used again.
  Scalar legacy evaluates only its final estimate; analysis evaluates initial and
  every stage(N+1) for flags. Mean-state equality excludes the changing weight.
  Plateau is two ULPs above nearest double pi, correcting old last-bit claim.
  Estimate first unchanged at4 while means still change; (a,b,t) first unchanged
  at5 while a,b remain one ULP apart. Actual positive stage4 correction1.358855e-20
  is lost in rounded t, then later actual corrections are zero. Error helper keeps
  rounded-double and wider-rounded differences/local-reference ULP offsets apart,
  exposes extra-bit/log-availability flags and never fabricates an error floor.
  Rounded pi compared with itself has zero rounded error but nonzero wider error;
  wider reference is itself approximate and may provide no extra bits elsewhere.
  Whole admission counts one initial-state record plusN fixed-schema stage
  records, default1m/0 disables optional quota only; complete count/capacity checks
  precede initialization/history allocation. Records are not scalar entries,
  bytes or peakRAM; ordinary final output/root-work metadata/working state/
  allocatorbytes excluded. Exact7records for6stages rejects6/admitted7; zero
  stages has one initial record. No partial output/truncation/caller mutation.
  O(N) arithmetic and history, O(1) ordinary analysis result storage.
  Ten strict ASan/UBSan/float-cast tests independently compare Decimal150 Machin
  pi/true errors, all0..1024 scalar/run results/full actual stage certificates,
  exact positive AGM gap identity within rounding, used correction versus rounded
  t difference, estimate/state stagnation versus unequal means/correct rounding,
  count-boundary/negativeINTMIN compatibility/new negative checks/INTMAX rejection,
  ownership/quota/log-reference units/adjacent ULPs/nonfinite candidate domains,
  long-double precision portability and work through stagnant suffixes. Versioned
  tests/oracles/gauss_legendre_machin.py --write/--check deterministically reproduces
  the reference header with Python standard Decimal, independently of AGM.
  Native27target+5dir/combined2966 pass; whole4path73definitions/all11newheader+
  1changedlegacy examples/stdout/7hovers pass. Final library/install freshness and
  compiled code/read-only parity checked. New2manifest paths/shared2already once.
  Five complete C++ programs/five inspected native figures/44computed lines in
  each Markdown/HTML export/source-build-export image parity pass. Actual first
  averages/correction/t/p; preserved original6stages with double error8.881784e-16
  versus wider7.656636e-16/twoULPs and53vs64bits; oneULP gap/correction-loss/stall
  flags4vs5; actualN+1 square roots andNupdates/noearlystop through1024lastunused
  pInf/finiteestimate; exact recordquota7vs6/owned/count/nonfinite controls and
  roundedconstantzero-no-log but wider1.225148e-16/zero-stage2.9142135623730949.
  Explicit newline diagram labels and absent default panel x labels fix visual
  wrapping; all5programs rerun, unchanged figures verified by SHA.
  Primary Brent1976 full author paper/Sections4-5 printed245-247 read,246-247
  rasters inspected: elliptic/Legendre connection pi=A²/T and better finite
  estimate, quadratic exact theory and own positive-gap derivation. Mathematical
  lower formulas can cease to bracket pi when computed in finite precision.
  Variable-precision guard bits/M(n)log n and non-self-correction distinguished
  from fixed double; no arbitrary-precision/digit/error certificate claim.
  Salamin independent-reference metadata verified (fulltext403/not claimed read),
  C++ nearest-constant specification read; no forced external dataset or timing
  claim. Site128reports/602blocks/537figures passes; original mepml/org removed
  only after gates, six owned old image paths absent and catalog API/destination
  updated. Evidence: build/mepml-conversion/gauss-legendre-pi/.

- 2026-11-08: Converted Gauss–Jordan into checked full augmented reduction
  in gauss_jordan.hpp/.cpp. The shared kernel preserves legacy result layout,
  by-value signature, strict absolute1e-12 cutoff/equality acceptance, earliest
  absolute pivot ties, interleaved full-row double A/identity divisions and
  multiply-subtracts, including zero factors, prior columns and signed zeros.
  All2,255,164 frozen records from35,859 systems match:16,276 actual-event cases/
  118,243 actual pivot/normalize/eliminate/reject snapshots, full solution/inverse/
  final matrix-transformation-RHS-permutation and100 unrelated solver controls.
  New preflight, installed library and rebuilt original match; original12stdout
  lines/SVG exact. Only Gauss–Jordan body moved; shared other bodies, original8
  tests and protected Gaussian126 header/source/tests remain byte-for-byte unchanged.
  Complete square/finite/RHS/control validation precedes indexing, early cutoff,
  quotas, input copies and inverse/state allocation (legacy argument copies can
  precede validation). Old short-row/short-RHS ASan heap overflows, NaN false
  success, extraemptyRHS and finite arithmetic overflow are reproduced and fixed.
  All stored coefficient/transformation/RHS assignments are checked, including
  inverse overflow; rounding/underflow and inaccurate inverses remain possible.
  Actual owned event matrices/identity-transform/RHS/permutation, final reduced
  coefficients/RHS, counted work and raw clearing factors share one execution
  without replay. No duplicated final inverse block; failed legacy solution and
  inverse arrays remain empty. A failed event's transform is a partial row
  transformation, not an inverse. New independent widened unsnapped A*V and V*A
  products/infinity errors/supplied norms replace local report matrix products;
  arbitrary candidates need not be inverses. Supplied norm product is not a
  condition certificate; numerical residuals are not interval accuracy proofs.
  Gaussian126 fixture and solve-residual summary are reused.
  Whole history H=n²+2n+(n²+n)*(1+2n²+4n), default1m/0 disables optional cap only;
  checked count/container/work-counter bounds before copies/state, full validation
  first, no early-stop discount/truncation/partial output/caller mutation. Input
  rows/cells/RHS and event metadata/two matrix row-records-cells/RHS/permutation
  count; ordinary output/root-work metadata/working state/allocatorbytes excluded.
  Entries not bytes/peakRAM. Successful n2history attains110, early zero-pivot
  rejection retains25 but rejects cap25/admitted110; original3actual/bound387;
  empty0. Separate inverse-product quota2n+2n² with default1m/0off. Ordinary
  O(n³) arithmetic/O(n²) storage, full history/copy O(n⁴).
  Ten strict ASan/UBSan/float-cast tests independently use determinant/adjugate/
  Cramer on19,683 ternary3x3 matrices and15,625 small2x2 systems,300 random
  actual-event certificates with accumulated T*Aoriginal=currentA and T*borig=
  currentRHS invariants, pivot ties/swap-all-blocks/normalization/full clearing/
  counters, dominant known solutions through32/both inverse sides, exact work
  identities0..64, raw above-pivot factor100, scaled cutoff/equality/singular and
  inconsistent systems, failed partial transform, all later nonfinite/domain-
  beforequota/early pivot, coefficient+inverse overflow, ownership/successful
  exact versus failed conservative quotas/virtual counts/empty, noncommuting
  unsnapped products/cap/widened range. Native27target+5dir/combined2939 pass;
  whole4path96definitions, all9new+2changedlegacy examples/stdout,8hovers and final
  installed source freshness/code/read-only parity pass;2new manifest paths,
  shared2already listed once.
  Five complete C++ programs/five inspected native figures/36computed lines in
  each Markdown/HTML export/source-build-export image parity pass. Tiny actual
  swap-normalize2-clearbelow1-normalize-.5-clearabove1.5 transforms I into
  [[3,-1],[-2,1]], x[1,2]/6events110. Original3x3 two swaps/12events387, inverse
  errors A*V-I1.110223e-15/V*A-I8.881784e-16 and solve residual2.220446e-16 are
  shown unsnapped. Actual fullblock updates2n²(n-1) versus Gaussian's
  n(n-1)(n+1)/3 give ratio6n/(n+1) for these different tasks, replacing the false
  inverse-free/50% claim; no universal ratio/timing claim. Raw above-pivot100
  has an exact inverse here, while supplied wrongI gives product errors100 and
  normproduct101 versus truecandidate10201. Failed partialtransform errors6/2;
  tiny1e-13I defaultreject/explicit1e-15works, whole110/success110/early25 and
  domain/ownership/productquota/empty controls. Diagram row label shortened and
  plot legend shortened after visual clipping review, all programs rerun.
  Own row-operation composition/inverse-on-full-reduction, cost and quota
  derivations. Primary Tsao1989 NASA TM101466 Section3 printed8-9 read/raster
  inspected; no-pivot diagonal-reduction variant explicitly differs from native
  normalization/partialpivot/full identity. No error-equivalence theorem applied
  to this implementation. Peters/Wilkinson1975 original stability reference
  verified, fulltext403/not claimed read; no unconditional accuracy/rank claim
  or forced external dataset. Site127reports/597blocks/532figures passes; original
  mepml/org removed only after gates, six owned old image paths absent and catalog
  API/destination updated. Evidence: build/mepml-conversion/gauss-jordan/.

- 2026-11-08: Converted Gaussian elimination to a shared checked scalar kernel
  in gaussian_elimination.hpp/.cpp. Legacy LinearSolution layout, by-value
  signature, ordered double recurrence, earliest maximal-absolute pivot ties,
  strict absolute pivot cutoff1e-12 (equality accepted), row permutations and
  early failures are preserved. Only the Gaussian body moves from shared
  linear_solvers.cpp; every other body remains byte-for-byte unchanged. All
  1,264,805 frozen records match:35,859 systems,16,276 full actual-event cases/
  90,477 events, final upper/RHS/row order and100 unrelated solver-control cases.
  All625 small2x2 matrices with25RHS each,19,683 ternary3x3 matrices, scalar/empty,
  500random, original3, scale/threshold/tie/swap andWilkinson controls included.
  New direct-source preflight, installed library and rebuilt original match;
  original9stdout lines/originalSVG exact; original5tests remain unchanged/pass.
  The actual original solution is floating approximate[2,3.0000000000000004,
  -0.99999999999999989]; the cached signedzero differed from this toolchain's
  actual output. New report preserves actual data rather than its stale cache.
  Full square/RHS shape and finite input validation precedes indexing, early
  cutoff and trace quota. Independently reproduced shortrow/shortRHS ASan heap
  overflow, NaN false-success and finite overflow/nonfinite solution are fixed.
  Stored double assignments and widened growth are checked, without widening
  or reordering the solve. Roundoff/underflow remain possible; an absolute cutoff
  failure is not a rank certificate and does not distinguish inconsistent from
  underdetermined or uniformly tiny nonsingular systems. Explicit positive
  finite cutoff on new APIs preserves legacy default semantics.
  Owned actual pivot/reject/eliminate/back snapshots, final upper/transformed
  RHS/current-to-original row order, accepted pivot count, actual work and stored
  coefficient growth share one kernel without replay. No manufactured zeros;
  no packed L factor/blocked BLAS. New original fixture/Wilkinson growth family
  and widened infinity-norm residual/backward summary replace report-local logic.
  Residual summary is independent of solving; distinct from existing Frobenius/
  2-norm linalg::backward_error. A small residual does not certify forward accuracy.
  Whole trace bound n²+2n+(n(n-1)/2+2n)*(1+n²+4n), default1m/0 disables optional
  cap only; checked arithmetic/container/work-counter limits precede input copies
  and state allocation after complete no-allocation input/control validation.
  Input/event matrix rows and all vectors plus one event metadata record counted;
  ordinary final outputs, root/work metadata, working state/allocatorbytes excluded.
  Entries not bytes/peakRAM; no early-stop discount/truncation/partial output or
  caller mutation. Tiny upper73/actual67 rejects67 and admits73; original213/195;
  empty0. Ordinary arithmeticO(n³)/storageO(n²), full history/copy worst-caseO(n⁴).
  Ten strict ASan/UBSan/float-cast tests independently use determinant/Cramer
  formulas for15,625 small2x2 systems and19,683 ternary3x3 matrices,300 random
  actual-event certificates, held row/pivot/permutation/update/back states and
  counters, known dominant systems through64, exact Wilkinson growth through128,
  |multiplier|<=1/no swaps, singular/inconsistent controls, threshold equality/
  scaling, full later nonfinite/domain-beforequota/early failure, finite overflow,
  owned copies/conservative caps/virtual count capacities and empty behavior.
  Native24target+5dir/combined2912 pass. Whole four-path102definitions pass;
  all13newheader+2changedlegacyexamples compile/run with exact stdout,7hovers
  (including existing CG legacy target), final library/install source freshness
  and matching compiled code/read-only sections verified. New2manifest paths;
  existing shared2paths remain listed once.
  Six complete C++ programs/six inspected native figures/40 computed lines in
  both Markdown/HTML exports/source-build-export image parity pass. Actual tiny
  swap/factor0.5/-0.5y=-1/back(1,2); original U diagonal-3,1.667,0.2/roworder1,2,0/
  two swaps/8coefficientupdates/9events/195retained and explicit rounding evidence;
  successful actual triangular row-count/cubic coefficient-count vs virtual
  quartic whole-history entries (no timing claim); Wilkinson family growth2^(n-1)
  with multiplier1/no swaps but zero backward error for selected RHS; deliberately
  wrong(1,2) on diag(1,1e-10) has coordinate error1 despite residual1e-10/
  backwarderror3.333e-11; scaled1e-13I fails1e-12 but solves with1e-15, wholequota/
  owned/fullvalidation/numeric-overflow/empty controls. Log-magnitude figure
  explains its zero-height coordinate bar represents error1, not zero error.
  Own row-equivalence/backsub/work/growth-doubling/history derivations; primary
  LAPACK DGETRF source/LUG read with exactzeroINFO/blocked-factor differences
  explicit, NIST family and original growth-author explicit matrix read.
  Higham/Higham1989 publisher abstract/metadata verifies correctdoi0610012 and
  growth need not yield large error for every RHS. Wilkinson1961 reference verified
  but ACM fulltext403, no claim full paper read. Generated systems are natural;
  no forced external dataset or unconditional numerical-stability claim.
  Site126reports/592blocks/527figures passes; original mepml/org removed only
  after gates, six owned old image paths absent, catalog API/destination updated.
  Evidence: build/mepml-conversion/gaussian-elimination/.

- 2026-11-08: Converted Gale–Shapley into a checked shared FIFO kernel
  in stable_matching.hpp/.cpp. Legacy result layout/signatures, exact FIFO
  proposals, ranking/acceptance/displacement/rejection, maps and proposal metadata
  are preserved across 2,854,683 frozen records from 46,975 profiles: all 46,674
  strict complete profiles through three agents, 300 random cases, original five.
  New Clang preflight, installed GCC and rebuilt original match; 35 original
  stdout lines and original SVG match byte-for-byte. Actual pre-edit program
  yields nine/eight proposals and resident mean ranks 1.8/2.8; cached report's
  1.2/3.2 was stale. Fixture preserves shuffle semantics; exact numeric tests
  separately pin emitted rows because std::shuffle mapping is library-dependent.
  Existing five stable-matching tests remain unchanged and passed originally;
  a new three-agent hypothetical wrong-orientation regression demonstrates the
  correct inverse map, without alleging an existing test bug.
  Full preference shape/range/permutation checks precede candidate indexing and
  quota admission; malformed maps now return false safely. Old short-map UBSan,
  out-of-range partner/short inverse ASan faults and false certification of a
  duplicate nonmatching are independently reproduced and corrected. Valid partial
  candidates remain supported; proposal metadata, including MAX, is ignored by
  stability checks. Empty result is stable/perfect, ranks have no agents and
  means are placeholders. New owned proposal snapshots expose actual used reviewer
  ranks, next choices, before/after FIFO, two maps, incumbent/newly-free/decision;
  work counters share the ordinary kernel without replay. Diagnostic witnesses
  use one-based ranks and unmatched n+1. Orientation reversal preserves execution
  count metadata. Rank summaries require perfect feasible maps but do not certify
  stability. Native full-permutation enumeration reuses the checker, admits n!
  candidates before search and is not an independent oracle. Native original,
  mutual-first and common-order profile generators replace report-local logic.
  Whole trace entries 3n+3n²+Q(1+5n), Q=n(n-1)+1 (empty zero); default one million,
  zero disables optional cap only. Checked arithmetic/container lengths precede
  rank/state/copy allocation, after O(n) validation scratch. Entries are not bytes
  or peak RAM; ordinary outputs, metadata, working state and allocator bytes
  excluded. Tiny bound51/actual45 rejects45 and admits51; original636/actual275.
  Diagnostics admit worst-case n² witnesses; enumeration limits candidates rather
  than stable answers. No truncation, caller mutation, partial history or replay.
  Ten strict ASan/UBSan/float-cast tests independently scan preference positions
  for all complete small profiles/candidates, exact stable sets, proposer-optimal
  and reviewer-pessimal properties; 300 random actual-state certificates check
  held-best-ever, FIFO/free-map equivalence, no repeated pair, next-choice prefix,
  counters and owned copies. All256 small malformed/partial map combinations,
  witness ranks, MAX metadata, domains before quotas, overflow, factorial/empty,
  exact deterministic families and portable fixed-row regression pass.
  Native24 target+5dir/combined2888 pass; whole two-path92 definitions/all26
  header examples/stdout/5hovers pass with final built/installed code freshness.
  Five complete C++ programs/five inspected native figures and both exports retain
  all43 computed lines and identical source/build/export figures: tiny actual
  displacement/queues[1],[0],[]; original synthetic rows/orientations/rank means;
  paper Example1 three stable maps with opposite rank means over six candidates;
  measured proposals n/triangular and n² inverse-rank preparation (not timing);
  malformed/partial/unstable/stable controls with0/3/1/0 witnesses, whole quotas,
  ownership, input validation, virtual overflow and empty controls.
  Read original1962 paper printed11-15, Theorem1 pp12-13 and Theorem2 p14;
  rejection induction, existence/stability/optimality and reviewer tradeoff stated
  within strict complete equal-side model. FIFO proposal bound differs from
  original simultaneous stage count. Ordinal means are not universal utility;
  native enumeration is not the independent test oracle. Official NRMP positions
  and couples sources verify that real matching exceeds this unit-capacity toy;
  no actual residency data or outcome claim. Site125 reports/586 blocks/521 figures
  passes. Source and org removed only after gates; six owned old image paths
  absent, catalog destination/API updated, two paths added to completed manifest.
  Evidence: build/mepml-conversion/gale-shapley/.

- 2026-11-08: Converted Gabow SCC into a validated shared iterative kernel
  in gabow_scc.hpp/.cpp, retaining the legacy signature, raw closure-order labels,
  ascending start and adjacency order/duplicates/loops. Original was already
  iterative. All658,854 frozen graph/label/other-SCC-control/actual preorder-S-B-
  heapframe-nextneighbor/component-assignment records match exactly:66,369 graph
  cases, including all66,067 digraphs through4vertices and833 full-event cases.
  New Clang preflight, installed GCC and rebuilt original match;6 original stdout
  lines/originalSVG are byte-identical. graph_connectivity.hpp and its unrelated
  algorithms remain byte-for-byte unchanged and outside completed manifest.
  Native work and owned enter/edge/collapse/close-root/assign/finish snapshots
  share one kernel without replay. Each primitive is captured at its actual time;
  closure-counter increment follows all assignments. S holds all discovered-
  unassigned vertices, including finished DFS calls; B holds candidate roots;
  active frames are a different stack. Full n/row/endpoint validation precedes
  empty returns, quotas, copies and state allocation. Old short-row/high-endpoint
  ASan heap overflow and negative-edge UBSan offset overflow are documented and
  corrected; negative n/extrarow-empty domains explicit. No caller mutation,
  shared state or partial result. Native partition-bijection comparison permits
  noncontiguous/MAX nonnegative labels and native quotient summary groups/sorts/
  deduplicates arcs with raw internal/cross counters; supplied labels are not
  certified SCCs and arbitrary quotients may be cyclic. Summary sort/map costs
  separated from linear traversal; Gabow IDs strictly decrease on cross arcs.
  Whole history bound(N+E)+(E+4N)*(1+5N), exact event countE+4N and conservative
  vector lengths; default1m/0 disables only optional cap/count capacities remain.
  Admission before copies/state; snapshot frame pair counts one record, root
  metadata/work/final ordinary labels/working storage/allocator bytes excluded,
  entries not bytes or peak RAM. Tiny upper263/actual185 rejects cap185 and
  admits263; original upper1740/actual1173. Empty trace0. History adds worst-case
  O((N+E)N) copies/storage while ordinary kernel retains O(N) state.
  Ten strict ASan/UBSan/float-cast cases independently use transitive reachability
  on all66,067 digraphs, per-primitive state/counter certificates for531 small and
  300 random shuffled duplicate graphs, full membership/preorder/S/B/frame/tree
  index/assignment/closure counter invariants, finished-open cross-edge, single
  edge79collapses/80totalrootpops, dense/DAG exact operation identities,
  200,000-vertex chain/cycle heap traversal, relabel/order/loop/duplicate partition
  invariance, full domains before quota, virtual overflow/conservative caps/owned
  snapshots, label-bijection and arbitrary cyclic quotient controls. Original
  raw labels[3,3,2,2,2,1,1,0]/visits8/edges10/tree7/open3/closed0/closures4/
  collapses4/peaksDFS8-S8-B4/42events retained. Native25 target+5dir/combined2864
  pass. Whole two-path75definitions/all15header snippets/stdout/5hovers pass;
  final allocation-failure contract comments preserve compiled text/read-only
  sections after docs/registry/install freshness verification.
  Five complete C++ programs/five inspected native figures/both exports retain
  every computed line and identical source/build/export images. Actual tiny
  finished1/openS[0,1] versus frames[(0,1)], later2->1 collapse/all3SCC; original
  four-group quotient chain sizes2,3,2,1/sink-firstIDs; reversed neighbor order
  permutes labels but same partition; duplicate loops change work2,1,0→2,6,1/
  rawcross2unique1 and supplied false split produces cyclic quotient. Families
  reuse native rooted_tree_edges/complete_directed_graph_edges, measuredS=E+4N
  versus represented input N+E (no elapsed timing claim); eight-cycleoneedge
  seven boundary pops/eight total, whole263/185 and deep200kcycle1mtransitions/
  199999collapses/default history rejection. Work-plot headroom keeps legend
  clear of highest point. No external dataset is natural for these exact inputs.
  Read full Gabow2000 Section2/Theorem2.1 printed108-110; contracted-path
  boundary invariant, sink removal and amortized proof distinguish open vertices
  from active calls, all-SCC partition from cycle enumeration, and many pops per
  edge from one-comparison claim. Native monotonic preorder/separate component/
  explicit-frame variant described;1999 submission not mistaken for journal date.
  Site124reports/581blocks/516figures passes; originals removed only after all
  gates, catalog API/destination updated/six old figure paths absent; two paths
  added to completed manifest. Evidence: build/mepml-conversion/gabow-scc/.

- 2026-11-08: Converted fuzzy c-means into a checked shared Euclidean kernel
  in fuzzy_c_means.hpp/.cpp, preserving FuzzyCMeansResult layout and legacy
  signature/defaults. All296,880 frozen input/initialized membership/actual
  center-membership-maxchange/final objective records match exactly, including
  zero-dimensional/zero-iteration/last-near-coincident controls and original160
  blobs. Both pre-edit compilers, new Clang preflight, installed GCC and rebuilt
  original match;12 original stdout lines/originalSVG are byte-identical.
  Before any API edits, corrected test-driver multiple RNG draws in unspecified
  argument order; preserved first artifacts, sequenced m/T/seed explicitly and
  verified both compiler paths identical. No baseline rewritten after core edits.
  Native work/seeded and supplied-start owned traces/actual weighted numerators,
  denominators, distances and phase objectives share one kernel without replay.
  Prediction memberships replace report-local lambda; original seed21 blob
  generator ported; native coefficient/entropy summaries and ordered objective.
  Last center within strict distance1e-12 wins one-hot, zero-weight column retains
  old center, returned centers use incoming last memberships (no final refresh),
  max-change<tol strict/tol0 runs cap, zero iterations randomU/zeroV all retained.
  Full shape/finite/parameter/simplex-row validation before quotas/shortcuts/
  copies/RNG; no normalization or scaling. Old ragged ASan heap-overflow/NaN
  results/all-positive-weight underflow are documented; invalid input throws,
  actual nonfinite arithmetic overflows, entirely underflowed positive cluster
  weights throw. Trace phase objectives have additional numeric limits (initial
  zero-center objective can overflow when an ordinary final fit is finite).
  Whole trace upper base2n+c+nd+nc+cd plusT*(1+3c+2n+2cd+2nc) admits before
  copies/draws, default1m/0 disables only optional quota/count capacities remain,
  not bytes/peak RAM or early-stop estimate/no truncation/partial/caller mutation.
  Tiny48vs47, original195004 requested/11485actual, ownership/virtual overflow.
  Nine strict ASan/UBSan/float-cast cases use independent inverse-power
  long-double queries400models/four fuzzifiers, analytic weighted means/row
  optimality grid, every actual phase sum-distance-membership/objective/counter
  certificate, scaling/reflection/translation invariants, simplex/domain/
  numerical overflow-underflow/trace extra limits/last ties/dead clusters/
  zero-empty dimensions/caps/owned history/original fixture controls. Native21
  affected+5dir/combined2839 pass. Whole two-path67definitions/all16header
  examples/stdout/5hovers pass; final comment-only helper exception-contract
  narrowing preserves all compiled text/read-only sections after docs/install.
  Five complete C++ programs/five inspected native figures/both exports retain
  every computed line and identical source/build/export images. Tiny centers
  55/31,255/31/J80.5→14.64516→11.21858660.; original11iters/J623.80448; canonical
  Iris150/four raw centimetre features/species excluded then grouped descriptive
  membership means/33iters/J60.505711/PC.783397/H.395492nats; fixed0,10centers
  acrossm1.5,2,3,5 demonstrate far tails→.5 (not global monotone S or probability/
  noise detector); actual three-iteration24point ratio powers72c² versus0 for
  coincident rows, with all distances computed first. Near-exact midpoint report
  assertion tolerates rounding; Color typo corrected to native RGB.
  Own objective/Lagrange/block-minimizer proof distinguishes exact descent from
  finite threshold/stopping, local versus global solutions, fixed-center m limits
  versus fitted solutions, raw-feature scaling, PC/entropy versus calibrated
  uncertainty/cluster quality/automatic c selection. Dunn1973 publisher verified;
  full1984 Bezdek/Ehrlich/Full pp193-195 read including singularity discussion,
  Euclidean versus paper norm options and nonstatistical membership.
  Site123reports/576blocks/511figures passes; originals removed only after all
  gates, catalog API/destination updated/six old SVG paths absent; two completed
  paths added after checks. Evidence: build/mepml-conversion/fuzzy-c-means/.

- 2026-11-08: Converted Freivalds into a checked shared projection kernel
  in freivalds.hpp/.cpp, reexported through the legacy linear_algebra.hpp.
  Preserved valid finite double accumulation order, SplitMix64 unsigned wrap/
  one word per bit, default seed/20 rounds, absolute tolerance1e-6, all three
  matvecs before row comparisons, and first failing row/round exits. All283,315
  frozen input/verdict/actual challenge-matvec-tested-residual/state records and
  unrelated Gram-Schmidt/power/Strassen controls match bit-for-bit; pre-edit
  installed core and rebuilt frozen source match too. All48,000 original seed
  verdicts plus12 counts match both ordinary calls and the once-per-seed batch;
  original18 stdout lines/originalSVG match. Every other legacy linear-algebra
  implementation body/private helper remains byte-for-byte unchanged; old
  incomplete files excluded from completed manifest, new two paths added.
  Full rectangular validation of all inputs precedes finite validation,
  compatibility/zero-round/quota/copies/RNG. Wellformed incompatible dimensions
  still reject; nonpositive rounds compatible accept vacuously, empty default
  calls perform20 empty rounds/no RNG. Old ragged A/B/C each independently
  produce ASan heap-buffer-overflow; old NaNs silently accept. New invalid input
  throws, actual nonfinite accumulators/compared residuals throw overflow_error;
  no unused AB/norm computations or caller mutation. Supplied zero challenge
  with huge unused product accepts while actual overflowing challenge throws.
  Native work/owned actual round traces/supplied binary probe/exhaustive binary
  audit/fixed original40x40 seed3 fixture share one kernel with no replay. Fixture
  product reuses canonical Strassen and independently matches integer products.
  Whole trace upper bound=input rows+cells+T*(1+n+k+3m), T=max(rounds,0) compatible;
  default1m/0 disables only optional quota, checked count/container limits remain.
  Admission precedes copies/RNG; differences retain only tested row prefix.
  Tiny upper29/actual28 still rejects cap28 and admits29; entries are not bytes
  or peak RAM, no partial result escapes. Exhaustive audit count2^n checked
  before enumeration with separate vector cap. Seed sweep validates nonwrapping
  seed range/aggregate work bounds, executes maximum prefix once/seed, derives
  caller-order/duplicate/negative prefixes from actual zero-based failure index;
  zero-trial/empty/incompatible controls documented and tested.
  Ten strict ASan/UBSan/float-cast cases independently test6,561 exhaustive integer
  matrix pairs/correct-and-corrupted claims/all binary vectors,300 rectangular
  random models, dense exact integer residual oracle and wide-integer proof of
  exactly correct cancellation input. Separate SplitMix indexed-word/state
  oracle, actual frames/work/early exits, epsilon boundary/nextafter/signedzero/
  subnormal, finite domain before shortcuts/caps, actual later-row overflow,
  retained ownership/whole quotas/empty/zero-width/zero-negative rounds/range
  checks pass. Measured original batch4000calls/7838rounds/313520words/37622400
  terms/157520comparisons agrees with all original prefix verdicts. Original
  correct20round product96000terms is preserved.
  Native25 affected+5dir and accumulated2818 tests pass; whole new two-path
  Doxygen audit96definitions, all21 public header examples/stdout,5 hovers,
  final registry and built/installed code-readonly-section/freshness parity pass.
  Five complete C++ programs/five inspected native figures/both exports check
  every computed line and identical source/build/export SVGs. Actual tiny
  3matvec method branch/all4challenges2accept; original seed curve with explicit
  0/4000 at11/12 and log panel omitting zeros without fabricated floor; numerical
  small wrong alwaysaccept and exact-cancellation halfreject; measured3Kn²
  versus analytical schoolbookn³ (n40K20:96000>64000, not runtime/Strassen counts);
  whole29/28 admission/owned/domain/empty/zero controls. Legend text shortened
  after visual inspection to fit canvas. Exact-arithmetic IID binary theorem
  independently derived by conditioning one nonzero coefficient; distinguishes
  deterministic seed fractions and approximate tolerance decisions, no general
  exact-equality or one-sidedness promise. Verified1979 Freivalds publisher
  chapter/Section3 full text pp59-60 (sign probes; ours binary variant).
  Site122reports/571blocks/506figures passes. Originals removed after all gates,
  catalog API/destination updated, six old figure paths verified absent.
  Evidence: build/mepml-conversion/freivalds/.

- 2026-11-08: Converted FP-growth into a checked shared explicit DFS kernel
  in fp_growth.hpp/.cpp, retaining the legacy association_rules.hpp declaration,
  FrequentItemset layout, row-set normalization, canonical ordering and historical
  zero-threshold positive-support-only behavior. Full-width node/parent/header
  IDs replace private int narrowing; node0 alone denotes root, so SIZE_MAX item
  IDs remain valid. All93,020 frozen output/Apriori-Eclat controls and actual
  tree creation/node-parent-header/order/DFS emission/weighted conditional path/
  child-tree records,19 original stdout lines and the original SVG match exactly.
  Pre-edit installed native/preflight/rebuilt frozen source also match. Other
  association helpers/miners/rules/encoding function bodies remain byte-for-byte
  unchanged; source purpose description reflects extraction. That final comment
  refresh preserves compiled text/read-only sections exactly.
  Native measured work, owned actual trees/steps/conditional bases and original
  grocery fixtures share one kernel without replay. Header-node creation order,
  build frequency-descending/ID-ascending ties, mining support-ascending/ID ties,
  nearest-ancestor-first paths, TARGET-node weights, pruned child builds and DFS
  emit/base/build/descent schedule are preserved; no paper single-path shortcut.
  History base N+raw occurrences precedes input copy/normalization. Later tree
  costs1+prefix+nodes+order+sum(1+headerIDs), step costs1+pattern+sum(1+pathitems)
  admitted online before history copies; working records formed first as needed.
  Default1m/zero disables only optional cap; count/container checks remain. Units
  exclude ordinary canonical output/work/normalized input/active trees/buffers/
  allocator bytes, not a whole-memory or peak-RAM quota; no partial result or
  caller mutation. Tiny exact38 versus37 rejection, empty trace2 are exercised.
  Added Diagram::rooted_tree using existing exact native font box measurements,
  iterative DFS leaf intervals/global measured spacing and parent-to-child arrows.
  Full parent/root/cycle/disconnection checks before optional node cap; validation
  may allocate scratch, cap precedes primitives; arbitrary parent-ID order, empty
  renderable trees, owned labels, zero cap and capacity checks documented. Existing
  Diagram methods/instance fields remain byte-for-byte unchanged and six old SVG
  fixtures remain identical. No report-local tree layout or schematic substitution.
  Eleven strict ASan/UBSan/float-cast cases independently count subset membership
  on2,842 exhaustive database-threshold cases and250 seeded duplicate/sparse/MAX
  fixtures plus permutations; certify weighted projected node counts, headers,
  supports, conditional paths/children/DFS emissions, every work counter/history
  entry and peak frame tree occupancy. Exact dense/disjoint identities, online
  caps/ownership/wide admission/zero-empty-MAX/pruned-empty controls, measured
  geometry non-overlap/XML/Unicode/long wrapping/parent domains before cap and a
  2,000-deep heap traversal pass. Grocery has no empty pruned child; a dedicated
  two-path base with counts1+1 at threshold2 tests that case instead.
  Fifty-eight affected plus5 dir native cases and accumulated CTest2,793 pass.
  Whole six-path audit197 definitions/all30 changed-header examples(19 stdout)/
  twelve hovers(six new) pass. Existing four completed paths retained once; new
  FP header/source added after gates, umbrella and source manifest wired.
  Five complete programs/five inspected figures/both exports preserve all55
  computed lines and identical source/build/export images. Tiny actual tree/pair
  support2/header3+3, original11 grocery sets and actual ten-node tree, diapers
  base beer2 plus beer,bread1→conditionalbeer3/history150; canonicalTitanic891rows/
  2673incidences/7items/min45/32sets(7,16,9)/initial23nodes/all58/build17/hops43/
  peak3/history3976/all-three-miner equality; denseU10output1023 versus disjoint10,
  nodes1023/build512/hops1013/peak10/live65/output5120IDs; online38/37/owned/MAX/
  empty2/above-count/zeroFP2 versus Apriori-Eclat3 controls. Publisher2000 paper
  tree/conditional proof sections read; nonempty-extension identity and target
  weights precise. Corrects wrong old hand tree, raw-losslessness/two-total-scans/
  O(Nwidth)/one-tree-memory/universal-compression-runtime/zero-equivalence claims.
  Canonical passenger data and native itemset_label reused; no external data glue.
  Site121reports/566blocks/501figures passes; originals removed only after gates,
  catalog API/destination updated; six old figure paths checked absent. Evidence:
  build/mepml-conversion/fp-growth/.

- 2026-11-08: Converted forward–backward into a shared validated native recursion
  in forward_backward.hpp/.cpp, preserving historical model/result field layouts,
  accessors, two-argument signature and hmm.hpp include. Existing Viterbi and
  Baum–Welch bodies remain byte-for-byte unchanged; incomplete umbrella excluded
  from manifest, Gaussian HMM remains completed once. All87,530 frozen successful
  alpha/beta/gamma/scale/loglikelihood plus Viterbi/BaumWelch control records,
  all24 original stdout lines and one original SVG match exactly. Pre-edit GCC
  O3 and Clang O2 baselines also match. New work is measured in the executing
  kernel; actual table results already expose states, so no redundant trace/replay.
  Complete model shape validation precedes all weights, observations, allocation,
  quotas and empty shortcuts. Finite probabilities in[0,1], stochastic initial/
  transition/emission distributions within absolute1e-12 LD sum; no silent model
  normalization. Observation and supplied path IDs fully checked. Empty model and
  observations have evidence1/log0/empty tables. Whole optional result admission
  4T+3TN for positive dimensions, default1m/zero disables only cap; entry records
  not bytes/RAM, count/container bounds remain. Ordinary API has no optional cap.
  Corrected impossible evidence from old finite prefix score to -infinity with
  explicit zero undefined-posterior placeholders. Boolean prefix support probe
  distinguishes positive mass lost to zero (underflow_error); nonfinite actual
  arithmetic/reciprocal scale throws overflow_error. Individual tiny terms can
  still round away; scaling is not an arbitrary-length/precision guarantee.
  Frozen probes demonstrate old impossible log0, tiny inf/NaN and ragged emission
  UBSan null-reference access. Native bounded widened path probability/enumeration/
  marginals/best path and checked integer N^T replace report-local brute force.
  Separate path cap default1m/zero off plus count/work/capacity checks; positive
  widened prefix underflow throws, no partial results/caller mutation. Original
  model and fourteen-symbol fixtures are native; reference enumeration is O(TN^T),
  not a replacement fast solver or unlimited-precision oracle.
  Ten strict ASan/UBSan/float-cast cases independently recurse through all243
  dyadic two-state models/all15,309 observation-model pairs and200 random models;
  filtering/unscaled suffix relations and sum(alpha*beta)=c_t, analytic future
  influence, impossible/positive-underflow/reciprocal-overflow distinction, full
  shape/weight/row-sum/symbol/path domains before cap/empty, whole quotas/ownership/
  full-width forecasts, actual counters/parity,10,000-step log evidence, marginal
  modes with a forbidden transition and enumeration ties. Existing Viterbi test's
  path identity assertion failed identically on frozen old code; corrected to
  verify recovered joint path score against optimum, preserving production ties.
  Twenty-seven affected legacy HMM/format/Gaussian sampling cases plus5 dir pass;
  accumulated CTest2,766. Fresh59 definitions/all13 examples/five hovers pass.
  Five complete programs/five inspected native figures/all46 computed lines in
  both exports with matching source/build/export images: tiny four-path evidence
  .154 and first-state filtering .1→smoothed .24026; original short16-path evidence
  .0133576 and14-step log-14.9452; marginal modes0→1 mass0 versus Viterbi0→2 mass.4;
  one-state10,000-step finite log evidence despite double exp underflow; actual
  two-pass pair terms versus integer path forecasts and entry20/original140 plus
  domain/enum-cap/virtual-overflow/owned controls. Rabiner1989 primary recurrence
  and scaling sections read, exact native backward convention documented; backward
  rows are not probability distributions, product norm c_t not1. Conditional model
  uncertainty does not establish calibration; no universal threshold/Viterbi or
  arbitrary-length guarantees. Synthetic fixtures need no dataset. Site120reports/
  561blocks/496figures passes; originals removed only after gates/catalog alias
  updated/two old figure paths checked absent. Evidence:
  build/mepml-conversion/forward-backward/.

- 2026-11-08: Converted force-directed layout with a shared checked native kernel
  extracted into force_directed_layout.hpp/.cpp; legacy signature/defaults and
  graph_layout include retained, spectral implementation and Point2D ABI unchanged.
  All118,983 frozen seeded/output/raw/actual-frame/repulsion/net/temperature/mean/
  scale records including66 spectral controls,8,289 frozen supplied-initial reference
  records and4 original stdout lines match. Pre-edit installed native smoke also
  exact; obsolete original stored coordinates replaced by actual source output.
  Actual double operation/update order retained, duplicate/reverse edges attract
  repeatedly, validated loops ignored, no coincident jitter, all forces before all
  moves. Cooling0.1*(1-double(i)/T); n<2 raw early return/no normalization, T0n>=2
  still centered/scaled. Work records actual pair/edge/attraction/move/sweep visits
  and uniform-distribution calls, not raw engine words. Owned traces capture real
  accumulator phases and states without replay; fresh supplied-position schedules.
  Whole bound E+3n+S(1+3n), S=T for n>=2 else0, default1m/zero disables only cap,
  before RNG/copies, checked arithmetic/capacities; entries not bytes or peak RAM.
  All supplied finite coordinates/endpoints validated before quota/early returns;
  actual intermediate arithmetic must remain finite. Extreme nonedge controls
  succeed without evaluating unused overflowing attractions; no partial result or
  caller mutation. Independent long-double geometry and C1 regularized potential
  helpers are queries, not extra kernel work; raw versus final units distinguished.
  Ten strict ASan/UBSan/float-cast cases independently use ordered-vertex widened
  forces on300 fixtures and all64 three-vertex loop/edge graphs, actual move replay,
  potential finite differences including the distance floor, analytic two-point
  cooling/non-descent, counts/seeded parity, translation/rotation/reflection,
  coincidence/singleton/full-width sweeps/zero iteration, domain-before-quota,
  whole admission/ownership/virtual overflow and finite-extreme actual overflow.
  Forty-seven affected plus5 dir native cases pass; accumulated CTest2,739.
  Fresh69 definitions/all14 header examples/five hovers pass; two new canonical
  paths completed; unfinished spectral umbrella excluded, geometry remains once.
  Five complete programs/five inspected native figures/both exports preserve all29
  computed lines and matching source/build/export images. Tiny two-vertex second
  capped step increases potential; original500-sweep path coordinates preserved;
  once-only500-frame raw potential/temperature/movement history; repeated-edge/
  loop/coincidence/singleton/T0 controls; dense actual work and quota21/original6515/
  domain/overflow/owned controls. Native complete_graph_edges reused. No global
  convergence/descent/quality/adjacent-distance guarantees or seed portability
  claims. Author-hosted1991 Fruchterman/Reingold paper read; paper frame clamping
  and coincident random directions explicitly omitted in this variant. No dataset.
  Site119reports/556blocks/491figures passes; originals removed only after gates,
  catalog alias updated; original owns no images. Evidence:
  build/mepml-conversion/force-directed/.

- 2026-11-08: Converted Floyd–Warshall with a shared checked in-place kernel
  preserving the historical AllPairsShortestPaths layout,detector/path signatures,
  strict parallel/tie/diagonal behavior and actual k/i/j read/update order. All
  429,817 frozen distance-bit/next-hop/valid-path/actual-stage/strict-improvement
  records and26 original stdout lines match final core and rebuilt frozen source.
  Left operands remain actual per-j reads,including negative-cycle sweeps;no cached
  left operand,early negative-cycle abort or raw-matrix rewrite. Full endpoint/
  finite-weight validation precedes allocation/admission;every attempted double
  candidate must remain finite,including non-improving candidates. Shared work and
  owned full-stage trace do not replay. Whole upper trace entries E+A+(n+1)(1+A)
  +n cubed,A=2n+2n squared,preflight before copies,default1m/zero disables only cap;
  all stages and strict updates owned,no partial result,entry units not bytes/RAM.
  Pair/mask helpers validate full matrices and classify unbounded exactly through
  reachable negative diagonals;do not trust external global flags. Bounded path
  reconstruction validates shapes/values/all hop IDs before shortcuts,rejects
  affected pairs including diagonals,allows unaffected pairs despite other cycles,
  retains broken-sentinel empty results and throws on cyclic external hop chains.
  Frozen UBSan probe confirms old malformed-next null-offset access. Full query
  validation costs O(n squared),external data not certified optimal by validation.
  Native original9edge fixture;report reuses complete_directed_graph_edges helper.
  Ten strict ASan/UBSan/float-cast cases independently use copied-layer edge-list
  Bellman–Ford and negative-descendant propagation on all4096three-vertex graphs,
  all4096four-vertex DAGs and400 potential graphs;actual stage/improvement replay
  and boolean-reachability count formulas,originalall25path edge costs,ties/
  parallel/self/signedzero/dyadics,unbounded masks/untrustedflags/unaffectedqueries,
  fullmatrix/pathbounds/domain/wholequota/ownership/emptyvirtualbound controls,
  finite overflow including non-improving sums,and dense/disconnected actualwork.
  Fifty-six affected plus five dir native cases pass;accumulated CTest2,722.
  Fresh66 definitions/all16 header examples/five hovers pass;two canonical paths
  added to completed manifest after gates. Existing generic WeightedGraph method
  is independent and outside this scope;affected Dijkstra/Bellman/Floyd/format
  target rebuilt and tested.
  Five complete programs/five inspected native figures/both exports preserve
  all40 computed lines and identical source/build/export images. Tiny9→5via1;
  original5x5 matrix/route0,4,3,2cost-3;actualstages0to2cost8/8/8/8/-1/-3,
  strictupdates2/3/1/9/3,total18,82finite additions;negativecontrolnineunbounded
  pairs but isolated4self safe;dense n cubed versus disconnected n measured on
  powersof2 with explicit log2display. Originalquota560/emptybound1/NaN-beforecap/
  cyclicpath/overflow/ownedcontrols execute. Corrects every-optimal-walk simplicity,
  independent-row claim,globalnegativecycleversuspairmeaning,tiehop/ranking,
  floatingcertificate and validation/path complexity assumptions. Historical1962
  DOI/bibliography verified without full-journal access claims;officialBoost.Graph
  API read for all-pairs/negative-cycle scope,no cross-library performance/floating
  equivalence claims or dataset. Site118reports/551blocks/486figures passes;
  originals removed after gates,catalog alias updated,five old SVG paths absent.
  Evidence: build/mepml-conversion/floyd-warshall/.

- 2026-11-08: Converted Floyd–Steinberg into a shared validated native raster kernel
  preserving its by-value signature/dithering.hpp include,128 cutoff,unclipped
  working values,R7/DL3/D5/DR1 order and exact double transfer/update arithmetic.
  All526,364 frozen pixel/working-quantization/update-bit records and six original
  stdout lines match final core and rebuilt frozen header. Frozen ASan confirms
  old ragged-grid overflow;full row validation now precedes all finite checks.
  All finite source values remain accepted,including outside nominal0..255 and
  single DBLMAX;nonfinite sources reject and nonfinite actual neighbor arithmetic
  throws. Boundary loss uses widened missing-weight accounting,avoiding spurious
  double overflow for a single extreme pixel. Work and owned actual traces share
  once-only prepare/run,no replay/caller views/partial result. Trace preflights
  whole upper bound2R+7N before copying,default1m/zero disables only optional cap;
  returned entries exclude temporary scratch/bytes/peakRAM. Native guarded stencil
  supports virtualSIZE_MAX dimensions;gradient/threshold baseline/column-band
  statistics replace report-local algorithms. Ordinary owns double scratch and
  returned int cells,O(R+N)storage;binary values are not packed one-bit storage.
  Added native equal-scale grayscale panels:zero BLACK255 WHITE,display-only
  rounding,invisible cell borders plus thin outer frame for all-white extents,
  complete shape/finite0..255/range/cap validation and owned primitives.
  Nine strict ASan/UBSan/float-cast cases cover independent exact fixed-point
  gathering for729 ternary2x3 grids/256 uniform levels/4096 dyadic fixtures,
  500 higher-precision conditional gathers and actual update replays,signed edge
  balance,all finite/domain/shape/quota/ownership controls,negativezero/subnormals,
  extreme arithmetic,pure full-width geometry via signed128,original gradient
  bits/band counts,whole-image stats validation and grayscale colors/ownership.
  Forty-seven affected plus five dir cases pass;accumulated CTest2,705. Fresh80
  definitions/all18 new-header examples/five hovers pass. Existing ordered Bayer
  body/documentation remain exactly unchanged;uncompleted umbrella stays outside
  the completed manifest,only four canonical new paths added.
  Five complete programs/five inspected native figures/both exports preserve
  all27 computed lines and identical source/build/export images. Original40x64
  gradient white132/415/738 in column bands0/21/42/64 retains fractions.157143/
  .494048/.838636,total1285white,9930updates,10240slots and signedloss-1275 with
  residual3.921e-13. Tiny2x2uniform128 has input512/output510/loss2;single127.5
  becomes0 and loses127.5. Uniform128 threshold allwhite versus diffusion halfwhite;
  uniform64 size controls demonstrate nonmonotonic finite-size tone errors.
  Derives signed boundary telescoping with floating residual;corrects nearest-
  level127.75,exact local/global tone,constant extra memory,packedbitmap and
  universalquality/speed claims. Speigle/Feng2000primary paper section3 read for
  traditional stencil/cutoff and artifacts;Floyd/Steinberg1976citation verified
  without claiming fullpaper access. No color/serpentine/modulated extension or
  dataset. Site117reports/546blocks/481figures passes;originals removed after all
  gates,catalog alias updated,five old SVG paths absent. Evidence:
  build/mepml-conversion/floyd-steinberg/.

- 2026-11-08: Converted Floyd cycle finding with actual shared-kernel pointer records
  and a reusable full-uint64 quadratic modular transition. Existing ordinary/work
  APIs retain result layout, callback order and the repeated initial f(x0).
  All1,204,440 frozen callback-input/output/result/phase-count records and15
  original stdout lines match final core and rebuilt frozen header/source.
  Owned Floyd pairs come from the actual ordinary kernel without replay or extra
  callbacks; original-orbit indices distinguish repeated state labels. Each row
  precedes its equality test; no-call locate-entry reset is recorded. Separate
  online row quota(default1m/zero unlimited) admits before each movement/reset,
  while global evaluation limits can stop within a round. No partial trace or
  callback-effect rollback is promised. Ordinary/work detector state remains
  O(1);trace storage O(mu+lambda),fixture/callback storage excluded explicitly.
  Eight strict ASan/UBSan/float-cast cases use independent first-seen oracles on
  every map through five states/all seeds,65x64 tail/period fixtures,500 random
  maps,actual pair indices and callback-sequence parity,every evaluation/row cap
  and callback-throw boundary on representative orbits,fulluint64 state labels,
  owned records/fixedpoint4calls,and10,000 full-width quadratic cases plus edge
  cube against independent binary add/double modular arithmetic. Twenty-seven
  affected plus five dir native cases pass; accumulated CTest2,690. Fresh72
  definitions/all20 cycle-header examples/six hovers pass. Existing completed
  cycle header/source manifest entries remain exactly once.
  Five complete programs/five inspected native figures/both exports preserve
  all41 computed lines and identical source/build/export images. Originalrho
  mu3/lambda4/entry3 gives Floyd22/Brent17calls;actualmeetingstate4atindices4/8,
  phasecalls12+6+4 and12pairs. Allseven original quadratic totals/ratios preserved
  (1.12to1.59);controlledperiod8/tail0to32 exposes meeting/cost stairs. Derives
  t=lambda*max(1,ceil(mu/lambda)),completeF=3t+2mu+lambda and entry-reset proof.
  Removes unsupported universal ratio/runtime/sqrt-growth assertions;distinguishes
  exact-state cycle finding from Pollard gcd/factorization and deterministic
  mappings from changing callbacks. Fullwidth polynomial control gives3422;
  eval21/row11each leave21completedcalls,exact22/12succeeds. Author-hosted
  Brent1980paper and Pollard1975paper read;no dataset. Site116reports/541blocks/
  476figures passes;originals removed after gates,catalog alias updated,five
  old SVG paths absent. Evidence: build/mepml-conversion/floyd/.

- 2026-11-08: Converted flood fill into a checked shared native traversal
  preserving the count-returning signature, both direction orders, mark-on-push
  and actual LIFO behavior. All345,762 frozen final-grid/pop/probe/peak records
  and21 original stdout lines match final core and rebuilt frozen header.
  A frozen ASan probe confirms the old ragged-grid buffer overrun. Full row-shape
  validation now precedes all no-ops and mutation; guarded size_t geometry avoids
  narrowing and supports virtual SIZE_MAX dimensions without grid allocation.
  Native work and owned actual trace APIs share once-only prepare/traverse;
  trace preflights the complete upper entry bound2R+(D+3)N or2(R+N) before copying,
  default1m/zero disables only the optional cap. Actual K visits/DK probes retain
  no caller views and do not replay. Native original6x8 fixture is reusable.
  Added full-signed categorical cell panels with a sorted shared nonzero palette;
  zero white/eight-color cycle/numeric identities and owned primitives explicit.
  Refactored shared renderer preserves all five legacy SVGs byte-for-byte and
  retains the old nonnegative ID mapping. Existing completed graphics paths stay
  once; new canonical flood paths added only after all gates.
  Fourteen strict ASan/UBSan/float-cast cases cover independent BFS on all65,536
  binary4x4 masks/both modes plus selected other seeds,600 signed rectangular
  contexts with actual stack/probe replay,full shape-before-noop/quota,whole
  admission bounds/ownership,virtual full-width geometry against signed128,
  full signed values,preexisting replacement barrier,large heap frontier and
  signature/fixture parity. Fifty-one affected plus five dir native cases pass;
  accumulated CTest2,682. Fresh74 definitions/all16 header examples/six hovers pass.
  Five complete programs/five inspected native figures/both exports preserve
  all34 computed lines and identical source/build/export images. Tiny2x2 pop
  order and pending counts are actual;checkerboard fills1vs5;originalfill8 leaves
  lower-right5 unchanged. Equal1024-cell inputs show rectangle/strip/checker4/
  checker8 changed1024/1024/1/512 and peak497/1/1/120. Correct O(R+K) includes
  shape checking;stack worstO(K),traces copyN,entry quota is not bytes. Exact-value
  flood differs from all-nonzero component labeling, same-value return0 counts
  mutations,preexisting replacement is not a bridge. SignedINTMINtoINTMAX,
  ragged rejection/cap31vs32/full-width neighbor controls execute. Official
  scikit-image API/example and IntelIPP connectivity read;no tolerance/scanline/
  timing claims or external dataset. Site115 reports/536 blocks/471 figures
  passes;originals removed after gates,catalog alias updated,five old SVG paths
  absent. Evidence: build/mepml-conversion/flood-fill/.

- 2026-11-08: Converted raw Fletcher-16 into a shared native byte kernel
  with the original const-string signature and checksums.hpp include preserved.
  Nine native APIs provide canonical streaming update, constant-work combination,
  actual owned byte folds, complete bit/replacement scans, analytical swap delta,
  an owned binary byte swap and the original modulo2^32 plain sum. All67,837
  frozen tags (all256 one-byte/all65,536 two-byte messages plus binary/long/
  original controls) and nine original stdout lines remain exact against final
  core and rebuilt frozen header. Raw residues0..254/B-high A-low packing and
  empty0 are explicit; invalid initial lane255 states reject before byte work.
  Trace admits complete byte events, scans admit exact N(8N+1)/257N full-message
  byte work before allocation/evaluation, optional zero caps disable only the
  cap, no partial results/replay/retained views. Shared ordinary folds recompute
  every scan candidate with one owned scratch message. Combine reduces uint64
  right length before multiplying and preserves canonical tags, including a
  virtual uint64-max all-zero suffix without allocation or fake byte views.
  Eleven strict ASan/UBSan/float-cast cases cover independent weighted sums,
  all65,025 canonical/511 invalid states,1,600 binary chunk/compose fixtures,
  actual owned byte events,every one-byte bit/replacement candidate,120 random
  flip/100 replacement contexts,all adjacent two-byte swaps/1,000 random swap
  contexts,composite-modulus aliases,full uint64 analytical distances,whole
  quotas,empty/signed-byte semantics and intentional32-bit sum wrap. Exhaustive
  two-byte multiplicities are64,516 singleton states/508 double states/one
  quadruple state. Fifteen affected native plus five dir cases pass; accumulated
  CTest2,661. Fresh scope63 definitions includes the modified completed checksum
  umbrella;all15 new-header examples and four hovers pass. Existing checksum
  manifest paths remain once; only the new canonical module is appended.
  Five complete programs/five inspected native figures/both exports preserve
  every computed line and identical source/build/export images. OriginalC8F0,
  88/88 bit detections,plain266/266 and Fletcher38411/38667 remain. The report
  proves the tag-held-fixed single-bit guarantee,corrects all-single-byte-error
  and universal strength/speed claims,demonstrates0/255 replacement alias
  (255 changed/254 detected/one missed),leading-zero non-length-binding,and
  nonadjacent product-zero swaps at3×85,5×51,17×15,255×1. It distinguishes
  65,025 possible raw tags from a probability model,canonical raw residues from
  protocol complemented check bytes,and actual per-byte reductions from hardware
  timing. RFC926 appendicesA.3/C.2-C.4 and RFC1146 appendixI read; Fletcher1982
  citation verified without claiming closed-paper access. No external dataset.
  Site114 reports/531 blocks/466 figures passes; originals removed after gates,
  one catalog alias updated,five old owned SVG paths absent. Evidence:
  build/mepml-conversion/fletcher/.

- 2026-11-08: Converted Flashsort into checked shared arithmetic
  classification/carry/insertion kernels, preserving its in-place template and
  sorting_extra.hpp include. Canonical flashsort.hpp plus native class-count,
  actual-class grouping and original mt19937_64 sample helpers expose work and
  owned actual class/carry evidence. Integral differences use the corresponding
  unsigned domain; unsafe floating spans/coefficients use a finite scaled ratio,
  with clamping before integer conversion. All finite values and signed length
  are checked before mutation. Bounded frozen-source probes reproduce signed
  range subtraction overflow and wide/subnormal finite-range invalid casts.
  All 7,458 safe frozen result/sample bit records and seven original stdout lines
  remain exact, including the historical double-map/carry/insertion order.
  Actual preparation is shared once, without replay. Trace admits whole upper
  entry count C=5n+2m (shortcut2n), default1m/zero disables only the optional cap,
  before copying/sorting; ordinary metadata is O(m) and records are owned.
  Eleven strict ASan/UBSan/float-cast cases cover all permutations through eight,
  6,561 duplicate arrays,1,600 independent full int64/uint64 pairs,1,600 wide-
  exponent double arrays,full signed/small-integral/float/long-double/subnormal
  domains,exact power-two class floors,400 actual carry replay/boundary/strict-
  inversion oracles,matched occupancy easy/quadratic families,whole quotas,
  finite-before-mutation,ownership,class aggregation and sample/signature parity.
  Seventeen affected native plus five dir cases pass; accumulated CTest2,646.
  Fresh scope66/66 definitions,eleven compiled examples,four clangd hovers pass.
  Five complete programs/five inspected native figures/both exports preserve
  all57 computed lines and identical source/build/export figures. Original
  fifteen values and 2,000-sample generator are retained. Actual860 classes are
  aggregated43 at a time, correcting the original separate20-value-bin plot;
  its old last bin held only one maximum, versus93 in the real final class group.
  Original sample occupancy-square6742/shifts1058/comparisons3055 are measured;
  ordinary class-vector payload3440 bytes versus8000 input bytes on this platform.
  Reuses native bucket workloads/collision families. Derives class ordering,
  insertion-inversion equality and occupancy upper bound,conditional iid fixed-
  probability expectation,linear metadata and quadratic distinct-cluster tails.
  Identical occupancies give256 versus32896 shifts at n258. Preserves the native
  no-initial-maximum-swap variant and avoids stability/hardware-speed/universal-
  linearity claims. Neubert author introduction,paper and code collection read.
  No external dataset. Site113 reports/526 blocks/461 figures passes; originals
  removed after gates,one catalog alias updated,five old owned SVG paths absent.
  Evidence: build/mepml-conversion/flashsort/.

- 2026-11-08: Converted Fisher–Yates to one shared native swap kernel,
  preserving its vector signature and existing permutations.hpp include. A new
  canonical fisher_yates module reuses SplitMix64/uniform_u64; a counting derived
  engine measures actual raw draws including rejection, without replacing the
  sampler. Ordinary/work/owned actual trace and validated explicit choices share
  the same swaps. Trace preflights complete C=2n+max(n-1,0), default one million
  entries/zero disables the optional cap, before copying or drawing; no replay,
  partial trace or byte/peak-RAM claim. Native complete seeded/ideal choice
  censuses retain all position marginals and observed whole-permutation counts,
  with nonwrapping seed intervals, labelled original positions, checked count/
  container limits and whole C=n²+outcomes*(n+1) admission before allocation.
  Exact comparison models include ascending full-range swaps and rotations.
  All 17,448 frozen seeded result/tally records and 13 original stdout lines
  remain exact against final core and rebuilt original source. The original
  stored text was stale before conversion: frozen current seed42 output is
  3,6,4,2,1,5 and first-position counts are 10118/9980/9856/10141/9886/10019;
  tests and the new report preserve executed values rather than stale text.
  Eleven strict ASan/UBSan cases include 2,400 independent full-product-threshold
  stream oracles; all legal shrinking choices through n=8 versus standard
  permutation enumeration; all 27 naive triples; flat rotation marginals with
  only three joint outputs; the original 60,000 seeds, every marginal and all
  720 observed outputs; forced first-zero-word rejection; exact/under/disabled
  quotas, domains/overflow/empty/zero-trials, signed duplicates and owned copies.
  Eighteen affected native cases plus five dir pass; accumulated CTest 2,629.
  Fresh scoped coverage 47/47 definitions, eleven compiled examples, four hovers.
  Five complete programs/five inspected native figures/both exports retain every
  computed line and identical inspected image bundles. The report derives the
  ideal bijection, contrasts deterministic finite seeds with independent bounded
  draws, proves flat marginals insufficient via exact rotations, and separates
  swap attempts, distinct-position swaps, actual raw words, expected retries and
  budget units from timing. Durstenfeld citation verified; NIST, Haas2020 and
  Lemire2019 primary analyses read. No external dataset. Site112 reports/521
  blocks/456 figures passes; originals removed after gates, one catalog alias
  updated and five old owned SVG paths absent. Evidence:
  build/mepml-conversion/fisher-yates/.

- 2026-11-08: Converted finite differences with shared checked Poisson assembly
  and canonical thomas_solve(), retaining the original four-argument vector API.
  New FiniteDifferencePoissonResult/finite_difference_poisson_solve() expose owned
  grid, actual once-only source samples, boundary-adjusted RHS, solved values and
  residuals evaluated with long-double intermediates. Nonpositive grids, absent
  callbacks, nonfinite data, signed n+2 capacity overflow and nonfinite assembly/
  solution are checked. Both endpoint terms reach the same RHS for n=1. Callback
  order and all 32,896 frozen solution/callback bit records plus nine original
  stdout lines remain exact against the final library and rebuilt frozen source.
  A bounded old-source sanitizer probe reproduces zero-grid invalid access.
  Ten strict ASan/UBSan cases include 400 independent discrete Green systems,
  160 pivoted dense solves, 256 polynomial/maximum-principle grids, sine eigenmode
  predictions, callback ordering/exceptions, domain/capacity/finite-overflow,
  signed-zero endpoints, owned copies and residual definitions. Fourteen affected
  native cases and five dir cases pass; accumulated CTest passes 2,611 cases.
  Fresh scoped coverage is 19/19 definitions, with four compiled examples and
  three clangd hovers. Existing ODE documentation now describes the actual
  bounded numerical-Newton iteration; its implementation is unchanged.
  Five complete programs and five inspected native figures cover a one-unknown
  stencil, three-unknown nonzero boundaries, the original sine profile and all
  five refinements, fixed-grid frequency error versus residual, and owned checked
  output. Both exports retain every computed result line and identical inspected
  source/build/export images. Reuses native interpolation and error scans rather
  than report-local assembly/solve/error-census implementations. Derives SPD,
  linear work, conditional smooth nodal O(h²) convergence and the 1/8 inverse
  bound; distinguishes interpolation, rounding, resolution and discrete residual
  from continuous PDE accuracy. Verified Persson and Sobey primary course notes.
  No external dataset. Site verifies 111 reports/516 executed blocks/451 figures;
  originals removed only after all gates, one catalog alias updated, five old
  owned SVG paths absent. Evidence: build/mepml-conversion/finite-difference/.

- 2026-11-08: Converted filtered back-projection into six checked native
  APIs: originalradon_transform/filtered_back_projection signatures retained,
  sharedfinitegeometry/filter/gatherkernels exposedasram_lak_filter/back_projection,
  reusable disk_phantom/parallel_projection_angles replace reportsetup loops.
  Movedinlineimplementationtocanonicalcpp. All60188frozenforward/reconstruction
  floatingbitrecords(168signedsmallfixtures+original64/96disks)and5originalstdout
  lines remainexact against standalonecheckedkernels,installedcore and rebuilt
  originalheader. Positivegeometry/exactN²/rectangularMxD/finitevalues+nonempty
  angles validatedbeforeallocation/numericalwork; size_tpixel/kernelgeometry,
  guardedsupportbeforefloorcast andfinite-resultoverflowerrors. Oldemptyimage
  readUB/emptyanglezeros observedwithboundedprobes; empty-anglepi/Mundefined,nowreject.
  Empty standalonefilterreturnsempty. Unitspacing/fixedpi/Mweights explicit;
  arbitraryfiniteanglesacceptedwithoutadaptivequadraturepromise. Actualfinite
  pixel-splat A andpaired(pi/M)A-transposeB shareweights; finiteFBP isnotmatrix
  inverse. Directzero-extendedRamLak/summationorderpreserved,notcircularFFT.
  TenstrictASAN/UBSAN/floatcastcases include300independentdensehatA/analyticalH
  matrixfixtures,300scaledadjoint/linearityfixtures,axisexactness/clip/finiteDC/
  phantomradius-boundaries/roundedangles,shape-nonfinite-capacity/finiteoverflow/
  hugeangles/ownership/callablesignatures/originaldisk. Audit22/22definitions,
  6compiledexamples/4clangdhovers/25affected+5dir/combined2597tests pass.
  Fivecompleteprograms/fiveinspectednativefigures/bothexports preserve37computed
  lines/identicalimages:tiny3x3actualsplatweights+adjointcertificate,finiteimpulse/
  constantresponse,original96/R22/M120/D144truth-sinogram-reconstruction-profile
  withcorrectedequalrow-columnimagescale,view/detectorRMSE+retainedmasscontrols,
  affinegain/offsetsamecorrelationbutlargerRMSE,andvalidatedownedpipelinestages.
  OnlineCovariancereusesnativePearsoninsteadlocalformula;reportonlyresidual
  reductions/presentationreshapes. CorrectidealcontinuousvsfiniteA-transposeHA,
  unitfrequencyconvention/filterfiniteDC,detectorcollectedmassvsgatherrange,
  correlationvsgain/offset/pointwiseaccuracy. Originalcorr.9941/center.9860 retained,
  actualRMSE.040605776/maxerror.393672622;gaincontrolcorrelation.994051437
  unchangedwhileRMSErises.596190923. Plain dot(...) computedlabelsfixMarkdown/HTML
  anglebracket-stripping;allresultlinesverified. ReadKak/Slaney1988chapter3Eq45/61
  andverifiedoriginalRamachandran/Lakshminarayanan1971citation/abstract. Native
  buffersareordinaryoutputs,notoptionalrecordhistory;noquotaneeded. Site110reports/
  511blocks/446figurespasses;originalsremovedaftergates/onecatalogaliasandits
  inverseclaimcorrected/fiveoldSVGpathsabsent. Evidence:
  build/mepml-conversion/filtered-back-projection/.

- 2026-11-07: Converted Fibonacci search to one checked exact-match
  ordinary/work/ownedtrace kernel plus pure virtual fibonacci_search_schedule.
  Legacy compiled signature/duplicate index choice/all1157680frozenresult-index-
  probelines and10originalstdoutlines preserved against currentcore andseparately
  rebuilt frozensource. Beforegrowth validate signedindexdomain; checkedgrowth
  addition, initialsigned-1sentinel andclamp-beforeaddition remove unsignedwrap
  idiom without changing indices. Scalar initialsizes/growthadditions,actualreads,
  keycomparisons andmaintransitions exposedseparately. Actualpreposttrios/offsets/
  readvalues/branches includeequality-onlyfallback(onecomparison,noinventedorder)
  andrepeatedclamps. Key8reads4,1,2,3,4(finaltail);key20reads4,7,9,9,9(allmain);
  duplicatescanreturnnonfirstoccurrence. Wholeoptionalrecord C=P/default1m/0off,
  admissionbeforeread/noinputcopy-replay-partialtrace-peakRAMclaim;ordinary/work
  constantstorage/nohistoryquota. TenstrictASANcases independentlycomparestandard
  membership,all165smallnondecreasingarrays,alllengths0..4096edgequeries,600seeded
  wide/duplicatearrays,10000virtualfullsigneddomainscheduleswithwiderinteger
  tableoracle,1000001elementinput,actualstateidentities/primitivebranchcounts/
  clamps/fallback/quota/ownership/callablesignature. Audit71/71definitions,
  all16compiledheaderexamples and8clangdhovers(existing8Eytzsnips/4hoversretained),
  24affected+5dir/combined2572tests pass. Fivecompleteprograms/fiveinspectednative
  figures/bothexports/59computedlines/identicalimages:tiny7valuesactual2reads,
  original10values+5queriesand13path/actualbinarypathcomparison,clamps-tail-
  duplicates-empty-singleton,completepresent+missingquerycensus19lengthsvsnative
  binary,fullsigneddomainvirtualschedule+captureownership. No report-localsearch
  replay. Correctexactfiniteclampingvsasymptoticratios/reads-vsuniqueindices-vs
  primitivecomparisons/initialgrowthvssearchwork andaddition-onlyvshardwarespeed;
  purelookupdistinctfromunimodaloptimization/generalizedasymmetricoptimality.
  ReadNISTofficialdefinition andEfraimidis2010introduction; nativevariantexplicit.
  Othersearches/Eytzinger unchanged;completedpathmanifeststillonce. Site109reports/
  506blocks/441figurespasses;originalsremovedaftergates/onecatalogaliasupdated/
  fiveoldSVGpathsabsent. Evidence: build/mepml-conversion/fibonacci-search/.

- 2026-11-07: Converted Fibonacci coding to a complete immutable92weight
  uint64 table, native exact weight/length APIs and shared ordinary/work/owned
  actual-span encoder/strict whole-frame parser. Old callable signatures and
  all69634validfrozen codeword/decoded/batch/appendrecords plus25originalstdout
  lines remain exact against checkedcore and separately rebuilt originalheader.
  Allpositiveuint64valuesnow supported:MAXhas92payload+1terminatorbits; next
  weight neverformedinuint64,terminatorrecognizedbeforeweightlookup,checked
  index/value sums. Zero rejects beforeappendmutation; totalgeometry/reserve
  beforewriting preservesarbitraryprefix onerrors. Unfinishedtailsnowreject
  insteadofsilentloss. BoundedoldrepresentableF92wordhitslogicaltablebounds
  withhardenedstdlib; longmalformedwordreproducesASANheapoverflow; oldzero
  emittedlone1/decodedemptyobserved. NewtableavoidsunboundedMAXweightgrowth.
  Actualwordspansmarkoverlappingfinal11:highestpayloadone+oneappendedterminator.
  Workcountsactualpayload/terminators/bits/values excludepreflight/greedytests/
  allocation. Ownedtrace C=B+2N/default1m/0off; encoderadmitscompleteCbefore
  copies/emission,decoderadmitsBbeforecopyandtwoentriespercompletewordonline;
  no replay/partialtrace/peakRAMclaim;ordinarynooptionalhistoryquota. Tenstrict
  ASAN/UBSANcases independentlyenumerate17710nonadjacentpayloads,all92weight
  boundaries/MAX/10000seededfullwidthvalues/600variablebatches,stricttruncations/
  valuesumoverflow/prefixappend/actualspans-work/domain/ownership/quotas and
  ordinary1000002bitframe.23578Pythonarbitraryprecisioncodecresults pass.
  Audit44/44definitions,14compiledexamples,4clangdhovers,41affected+5dir and
  combined2548tests pass. Fiveprograms/fiveinspectednativefigures/bothexports
  preserve58computedlines/identicalimages:actual10digits/sum/delimiter,alloriginal
  eightwords/tenvalueroundtrip/tenlengthcasespluspositiveEG(n-1)mapping,fullwidth
  table/guards,85originalframesinglebitflips(82changed/3rejected)andvalid3to4
  mutation,actualspan/capturework. ReusesnativeExpGolomblength/genericbittext;
  no report-localcodec/lengthloop. Correctprefixfromknownstartvsarbitrarysync,
  structuralvalidityvserrordetection/correction/blankettwo-wordrecoveryclaim,
  explicitdomainmappingvsgenericcompressionclaim. ReadoriginalPurdue1985
  Apostolico/Fraenkeltechnicalreportandverified1987IEEEpublicationmetadata;
  nativeorder2variantandlimits explicit. Site108reports/501blocks/436figures
  passes; originalsremovedaftergates,onecatalogaliasupdated,fiveoldSVGpathsabsent.
  Evidence: build/mepml-conversion/fibonacci-coding/.

- 2026-11-07: Converted Fermat primality with checked explicit-base
  FermatBaseResult/fermat_base_test, constant-storage bounded half-open-interval
  FermatBaseCensus/fermat_base_census and one shared bool/work/ownedtrace seeded
  kernel. Canonical exact mod_pow replaces report-local powering; census replaces
  local liar_fraction. Legacy compiled signature/defaults and positive-round
  SplitMix64/modulo stream preserved:19200frozen decisions and13originalstdout
  lines match currentcore and separatelyrebuilt oldsource. Nonpositive rounds
  nowreject before filters; zero/one are notprime rather thancomposite. Actual
  stop/filter/completedrounds/lastbase-gcd-residue plus actual admitted roundrows,
  no replay orinnerbit-history. Onegcd perround now suppliesdiagnosticfactor;
  ordinary/workconstantstorage/nohistoryquota; ownedtrace C=actualroundrecords,
  default1m/0off/pre-drawgcdpoweradmission/no partialtrace/nopeakRAMclaim.
  Exactcensus workcapdefault1m/0off checkedbeforepowers, fulluint64 andempty[n,n)
  supported, invalidbase/intervalcontrols reject beforearithmetic; emptyfraction
  denominators leftundefined. TenstrictASANcases include32640smallbase results
  vsindependentrepeatedmultiplication,censuspartitions throughn512,Carmichael/
  fullgroup-originalinterval-samplersupportcounts,goldenSplitMixwords,3000seeded
  work/trace/boolparity+actualearlywitnesses,fullwidth/filters/INTMAXbudget,
  quota/domain/names/ownership.42710Pythonarbitraryprecisionpow/gcdoraclechecks
  pass. Audit49/49definitions,11compiledexamples,4clangdhovers,26affected+5dir
  andcombined2507tests pass. Fiveprograms/fiveinspectednativefigures/bothexports
  preserve54computedlines/identicalimages:tiny15actualgcd/residuecertificates,
  original97/91/561decisions+fiveconditionalfractions,561nativefactors/first40
  bases/copopulations,samepassdefault182passthen345gcd3failure andseed1a115pass
  thena459gcd51failure,filters/MAXbase,explicitlyhypotheticaliidprobabilitycurves,
  samebaseMillerRabincomparisons/quota/ownership/domain. Correctconditionalvsall
  bases,endpointdenominators,passvsproof,idealindependencevsdeterministicmodulo
  bias,andforwardfailureprobabilityvsposteriorinterpretation. ReadConradFermat
  Test andShoup2008Theorem10.1/pp307-308; noactualseedediidguaranteeclaimed.
  Existingcompletedprimalitypathsremainonceinmanifest. Site107reports/496blocks/
  431figurespasses; originalsremovedaftergates,onecatalogaliasupdated,fiveoldSVG
  pathsabsent. Evidence: build/mepml-conversion/fermat-primality/.

- 2026-11-07: Converted Fermat factorization with one exact bounded native
  kernel shared by preserved fermat_factor signature and new work/trace APIs.
  Actual tests, increments, factors, proper split and even/factor/trivial-pair/
  trial-limit stops replace the report-local counter. Default five-million-test
  cap retained; zero/one now reject as invalid factorization inputs. Trial-limit
  n return remains inconclusive, including composite3000000021; first945split
  is composite27x35. Exact reusable integer_sqrt_u128 uses bit-width brackets,
  division comparisons and wide upper sentinel, supports all128bit patterns and
  UINT128_MAXrootUINT64_MAX without floating arithmetic. Evenzero-work andodd
  squareb0 behavior explicit; guarded budget increments avoid unused wrap.
  Optional same-pass capture C=T whole actual trial records, quota1m/0off,
  admitted before root/square work; no partial trace on failure, no peakRAM claim;
  ordinary/work remain constant storage. Ten strict ASAN/UBSAN tests cover8190
  smallinputs via independentclosest-divisororacle,600seeded clippedproducts,
  10000randomfull128root brackets, fullwidth/odd squares/near squares,allstops,
  hugepositivebudget,quota/domain/names/ownership. Optional Python math.isqrt
  independentoracle passes19405fullwidth/square-boundary comparisons. Frozen4626
  valid legacyfactorlines and14originalfull-programstdoutlines match currentcore
  andseparatelyrebuilt oldsource. Bounded strict originalreportcounterprobe shows
  floatingceilbelowroot andout-of-rangefloatcast for(2^32-1)^2+2; native exact
  ceil fixes report counting without a second factorization loop. Audit44/44
  definitions,12compiledexamples,4clangdhovers,24affected+5dir andcombined2481
  tests pass. Fivecompleteprograms/fiveinspectednativefigures/bothexports preserve
  47computedlines andidenticalsource/build/exportimages:tiny5959actualrows and
  equal-area square/rectangle,originaltwofactorizations/sevengaps,true tests vs
  increments andcontinuousfloor formula,stop/cappedcomposite/multifactor controls,
  exactwidthdiagnostics andownedhistoryquota/workfamilies. Corrected nearestpair
  vs arbitrarypair,splitvsprimefactorization,incrementsvscontinuousexcess and
  fixedwordvsarbitrarybit complexity. Read primaryPomerance1996p1474 viaauthor
  PDF andofficialPython3.14isqrt contract. Site106reports/491blocks/426figures
  passes; originalsremovedaftergates,onecatalogaliasupdated,fiveoldSVGpathsabsent.
  Evidence: build/mepml-conversion/fermat-factor/.

- 2026-11-07: Converted the FEM report with a shared five-source synchronous
  visitor and public FEM1D::solve_transient_builtin, delegating to the same actual
  backward-Euler loop as live callbacks. Existing result/class layouts and solver
  signatures retained. All2345frozen nodal/time/value-bit lines and34original
  full-program output lines remain exact against rebuilt frozen module/currentcore;
  original diffusion2/5/20,reaction4..128,Neumann/Robin,transient20nodes/selected
  times and2Dlinear6x6/Poisson8/16/32/slice cases execute. Source defaults/extra
  parameters/empty polynomial match steady/live behavior without local report
  adapter classes. Rounded t_end=.30000000000000004/dt.1 formerly produced a
  duplicate final time/zero actual_dt/NaN. Checked finite positive time controls,
  node/frame/vector/solution-value counts, positive actual intervals, direct final
  time and stopping as soon as reached correct that case to4finiteframes; huge
  finite nominal products avoid intermediate overflow. Spatial finite geometry,
  coefficient/source arithmetic and nonzero Thomas pivots remain explicit caller
  preconditions; no new solver-wide conditioning certificate claimed.
  Eight strict ASAN/UBSAN tests cover400seeded fixtures×5named/live source families
  bothsteady/transient, defaults/trailing/empty data, independent one-interior
  backward-Euler product on3meshes×3reactions,300rounded-time fixtures,partial
  steps,MAXtime/fenv,nonfinite/capacity controls beforecallbacks,actual quadrature
  calls/exceptions,ownership/checked accessors. Audit68/68definitions,9compiled
  examples,4clangd hovers,29affected+5dir andcombined2457tests pass.
  Sixcompleteprograms/sixinspectednativefigures/bothexports preserve49computed
  lines andidenticalsource/build/exportimages. Tinyactual3nodecoefficient/hat,
  nodalexactnesswithbetween-node.03125error,original1D/2Drefinements+fields,
  original/mirroredconormalboundaries,onesharedtransienthistory/steadygap and
  correctedactualtimegrid/source/decaycontrols execute. Reusednative linear
  interpolation/errorcensus;no report-local assembly/linear/time solver. Corrected
  nodalvswholefield/L2vsmaxnorm claims,conditionalcoerciveBE stabilityvsaccuracy,
  integrand-vs-source quadrature degrees andconormal-vs-physicalflux signs. Read
  officialMFEMweakform andArnold2012norm/dualregularity/semidiscreteBE references.
  Minimalpublicheaderskeepcompleteprogramsidiomaticandfast;Doxygenexamplesuse
  dm/dm.hpp. Site105reports/486blocks/421figurespasses;originalsremovedaftergates,
  noexistingcatalogaliasand20oldownedSVGpathsabsent. Evidence:
  build/mepml-conversion/fem/.

- 2026-11-07: Converted false position to one checked plain/retained-twice
  Illinois kernel with actual true endpoint values kept separate from artificial
  secant weights. Original compiled Illinois signature/RootResult layout/default
  seven-trial exponential and2296 frozen result/everycallback-coordinate bit lines
  remain exact, also against separately rebuilt frozen source. Safe legacy
  interpolation is preserved; prechecked products/differences/division and scaled
  positive weights/std::lerp guard finite-overflow cases. Original huge-coordinate
  probe sends10nonfinitecoordinates and returnsNaN; huge-value denominator raises
  overflow while converging in2trials. Both checked cases use1actualguardedtrial
  without overflow/invalid flags; ordinary floating coordinate rounding explicit.
  Actual outcomes expose callback/guard/halving work, pretrial true/weighted state,
  rounded radius and8distinct stops; independent function/coordinate tolerances,
  input orientation and bothendpoint call order retained. Explicit nonfinite/domain,
  stagnation/weightunderflow/budget behavior and callback exceptions/sideeffects.
  Owned same-pass actual evaluation+step capture C=2+2K whole records, default1m/
  zero disables onlyquota, admission before callbacks/interpolation, no replay or
  partial result/peakRAM claim; ordinary constant storage/nohistory quota.
  Tiny finite secant products underflow in the frozen source (wrong zero estimate,
  cap10); normal-product admission now chooses scaled weights, recovering the
  representable affine root in one actual guarded trial.10000independently scaled
  affine pairs spanning coordinate/function exponents-1020..1020 check roots and
  absence of overflow/invalid flags. Ten strict ASAN/UBSAN tests also cover
  2000seededscaledquadratics bothpolicies plus
  2000affinelongdoubleinterpolationchecks,actualraw/weight/transition/call/capture
  certificates,allcontrols/domain/callbackfailures/endpointorder/reversed/zero/
  units/radius/adjacentstagnation/finiteguards/underflow/ownership/quotas. Audit
  77/77definitions,12compiled examples,4clangd hovers,22affected+5dir andcombined
  2428tests pass. Fivecompleteprograms/fiveinspectednativefigures/bothexports
  preserve40computedlines andidenticalsource/build/exportimages:tinyactual
  secants/damping,originalexpdefaultandmatched16vs8,measuredpowerfamilystall/
  endpointstates,functionunitsvscoordinateerror/radius,andfinite/capturecontrols.
  Correctmathematicalsign-bracketderivation/continuityassumptions,rawvsweights,
  residual-vs-coordinateunits/displayfloors,rounded-not-certifiedradius,finitejump
  caveat andconditionalUUM/efficiencyindexanalysis. ReadprimaryDowell/Jarratt1971
  pp168–171;nativeinitialside0delayvsmost-recent-paperinitializationexplicit.
  Otherrootmethods/RootResult unchanged. Site104reports/480blocks/415figurespasses;
  originalsremovedaftergates,onecatalogaliasupdated,oldSVGpathsabsent. Evidence:
  build/mepml-conversion/false-position/.

- 2026-11-07: Converted Eytzinger search to shared checked complete-tree
  construction and equality-first exact-match kernels with owned actual source-
  rank placements and node-read/decision/next-child probes. Original compiled
  signatures, layouts and found/index/probe outputs retain233716 frozen lines,
  also matched by separately rebuilt frozen source. Existing other sorted-search
  algorithms unchanged. Zero-based child existence checked before multiplication;
  signed-index bounds checked. Unchecked ordering preconditions, exact-match vs
  lower-bound semantics, first-encounter duplicate choice and layout-vs-source
  index explicit. Optional capture is N whole placement or P whole probe records,
  default1m/zero disables only quota, before layout allocations/next node read,
  no partial result/replay/input alias/peakRAM claim; ordinary calls no quota.
  Ten strict ASAN/UBSAN tests use independent BFS rank-interval oracle,165exhaustive
  nondecreasing arrays,all lengths0..1024,600seeded wide/duplicate arrays,
  int endpoints,all actual transitions/counts,root rank,perfect-tree depth sums,
  ownership/exact/short/empty/disabled quotas and1000001element ordinary layout.
  Audit35/35definitions,8compiled examples,4clangd hovers,24affected+5dir and
  combined2406tests pass. Fivecompleteprograms/fiveinspectednativefigures/both
  exports preserve57computed lines and identical source/build/export images.
  Tiny7nodeactualpath,original10values/5queries/4nodewalk,actualsource-rank
  provenance,per-key/complete-tree read counts,explicit hypothetical aligned-four-
  integer blocks and full-width/capture controls execute. Corrected root13 is
  rank6, not median10; mathematical in-order correctness/root-rank/depth-mean
  derivations,preprocessing cost,static/read-only preconditions and probe-vs-
  comparison/cache/runtime distinction. Verified Khuong/Morin2017arXivrevision
  section3.2; no imported speedup/tuned-prefetch claim. Site103reports/475blocks/
  410figures passes;originals removed after gates,onecatalogaliasupdated,oldSVG
  paths absent. Evidence: build/mepml-conversion/eytzinger-search/.

- 2026-11-07: Converted extended Euclid to one checked signed128-bit kernel
  with the original compiled ExtendedGcd three-int64 layout/signatures, unsigned
  gcd certificates, actual work counts and owned same-pass row/inverse traces.
  Raw rows preserve signed truncating quotients and terminal zero coefficients;
  final sign normalization stays separate. C=2+D whole row records, default1m,
  zero disables quota, admission before each allocation/division, no replay or
  partial trace. Ordinary calls retain constant auxiliary storage. Legacy gcd
  overflow for the three magnitude2^63 cases throws; unsigned certificates handle
  every int64 pair. Checked product addition and unsigned magnitudes validate
  arbitrary certificates including a malformed sum of2^127. Modulus validation,
  inverse normalization and legacy modulus-one/zero-pair conventions explicit.
  Five strict frozen-source UBSAN probes reproduce division, negation and unused
  terminal coefficient overflow; old safe outputs match133138 frozen lines.
  Ten strict ASAN/UBSAN tests cover66049 small pairs,10000 full-width pairs,
  powers/neighbours/MIN/MAX, raw identities/quotients/unimodular determinants,
  Fibonacci90/91 divisions, ownership/quotas/domains and legacy layout/ranges.
  Independent Python arbitrary-precision oracle passes11449 cases including every
  actual row and modular inverse. Audit50/50definitions,14compiled examples,
  fourclangd hovers,27affected+5dir+1oracle and combined2382tests pass.
  Six complete programs/six inspected native figures/both exports preserve63
  computed lines and identical source/build/export images. Original100000pair
  randomized check, tiny240/46rows, signed paths, inverse residue cycle, bounded
  Fibonacci work and full-width storage cases execute. Mathematical certificates,
  fixed-word versus bit complexity, actual capture units, signed std::gcd
  precondition and degenerate modulus one explained. Verified HAC1996 algorithms
  2.104/2.107/2.142/example2.109 and C++working draft numeric.ops.gcd references.
  Site102reports/470blocks/405figures passes; originals removed after gates,
  two catalog aliases updated and five old owned SVG paths absent. Evidence:
  build/mepml-conversion/extended-euclidean/.

- 2026-11-07: Converted exponential backoff to checked finite window arithmetic
  and owned actual explicit/seeded idle-wait schedules. Original three inline
  names/signatures and tested u=1 endpoint remain. Valid nonnegative finite base/cap
  including zero/base-above-cap, indices/counts and inclusive explicit u retain all
  5786 frozen legacy output lines, including signed-zero/subnormal/max fixtures and
  separately recompiled frozen header. Negative/nonfinite/domain failures now throw.
  Shared widened exponent-gap/mantissa comparison precedes ldexp, avoiding both
  retry-index loops and finite-cap intermediate overflow. Original zero/INTMAX probe
  completes0.87s rather than timing out; runtime nearMAX probe raises FE_OVERFLOW
  before returning a finite cap. New fixed scalar math branch needs neither scaling
  for that capped case nor for zero/INTMAX. O(N) ordinary windows remain valid when
  hypothetical cumulative waits overflow. Actual step records retain index/window/
  unit variate/product and capped/scaled branch flags; shared schedule emitter
  records prefixes once without replay. Cumulative addition rejects a wait exceeding
  remaining finite capacity, with no partial result; rounding remains ordinary.
  Seeded native xorshift13/7/17 high53bits preserves the original eight values and
  seed88172645463325252; zero seed rejected, local state/noncrypto/same-seed limits
  explicit. Native SplitMix64 provides nonzero population seeds in the report.
  Twelve ASAN/UBSAN tests check12000 seeded wide finite cases and2098 binary-power
  exponents/cap neighbours/large indices against independent overflow-safe doubling,
  signed zeros/endpoints, actual records/prefixes/branch flags, no intermediate
  FE_OVERFLOW, malformed/empty cases, ownership, overflow-vs-pure-output separation,
  seeded stream/prefix/same-seed controls and time-unit scaling. Audit36/36
  definitions,9 compiled exact-output examples,4clangd hovers,31affected
  backoff/networking/format+5dir and combined2354 tests pass. Five complete
  programs/five inspected native figures/both exports preserve34 computed
  lines and identical source/build/export images. Tinyfour-step offsets9.25,
  originaleight waits,1000client repeat/distinct-seed controls,2000schedule prefix
  means and finite-domain diagrams execute. Ideal uniform mean/variance/fixed-bin
  collision probability and geometric/capped-tail idle sums are derived; observed-
  range histogram, nonindependence guarantees, timer quantisation, idle-vs-RPC time,
  unlimited cumulative budget and numeric-vs-TCP/Ethernet scope explicit. Verified
  Brooker2015 and RFC6298 references; no copied service contention outcomes.
  Site101reports/464blocks/399figures passes; original sources removed after gates,
  one catalog alias updated and old owned SVG paths absent. Evidence:
  build/mepml-conversion/exponential-backoff/.

- 2026-11-07: Added batched tensor image explanations under nn/image_explainers
  and converted the original explainable-AI report. Legacy scalar stats explainer
  source/header remain byte-identical, with all five existing tests passing.
  New CPU Float32/Float64 single-image masking uses nonempty rectangular grids,
  Bool coalition batches and tensor gather/where across all channels. Fixed-target
  finite [B] callbacks run under NoGradGuard, with owned detached Float64 score
  copies. Known evaluation quota and complete geometry are checked before calls;
  default1m/zero disables quota only, exact Shapley hard P<=20 remains. Quota counts
  scored images rather than peak RAM, CPU work or callback batches. External
  callback side effects cannot be rolled back. Exact ascending-mask enumeration
  retains actual coalition scores, computes tensor weighted marginal sums and
  reports signed values/endpoints/measured efficiency residual and actual work.
  Anchored full/empty plus seeded MT64 masks feed a Float64 tensor weighted-ridge
  Cholesky fit without inverse or hidden model training. Intercept unpenalized;
  actual masks/scores/weights, weighted in-sample R2 and fitted/full scores retained.
  Anchored weighted means avoid constant-score variance drift. Signed spatial
  expansion divides by actual uneven patch areas by default; optional repetition
  explicitly changes pixel totals. Tensor Pearson comparison retains negatives,
  offers labelled positive-only clipping and returns NaN for constant maps.
  Sixteen ASAN/UBSAN tests include224 random games against independent permutation
  averages,127 seven-player unanimity games,16 independent long-double pivoted
  weighted-ridge fits, actual query/mask/span/work checks, additive/signed/baseline
  controls, uneven channels/noncontiguous data, callback errors/ownership/gradients,
  all quota/domain/geometry limits and undefined/extreme-scale correlation. The reusable native map diagram
  adds square cells, grayscale input, signed blue-white-red scales and explicit
  numeric keys, with finite/label/flag/render geometry and owned presentation checks.
  Audit62/62 definitions,12 compiled exact-output documentation examples and4
  clangd hovers pass.21 affected tensor/legacy tests+5dir and combined2323 pass.
  Six complete programs/ten inspected native figures/both exports preserve38
  computed result lines and identical source/build/export image bytes. Retains
  600 noisy12x12 bars, fixed-position horizontal/vertical examples,4x4 patches,
  800 LIME samples, CNN/dense/z-plus comparisons; replaces constructor-trained
  wrappers with explicit SGD .05/batch16/40CNN+200dense epochs on420fit/180heldout.
  Actual architecture counts6578/4706, optimizer calls1080/5400, held-out confusion,
  fixed raw-logit targets,65,536-coalition queries and524,288 marginal counts execute.
  Baseline/partition/probability controls check full-score invariance and query
  preflight; signed numeric maps/residuals replace clamped-negative displays and
  score differences misleadingly labelled efficiency. Fixed-baseline games are
  distinguished from conditional-expectation SHAP/KernelSHAP and causal claims;
  custom LIME kernel/anchors/ridge and new tensor initialization are explicit,
  without claims to reproduce the old wrapper's fitted numbers. Verified primary
  Ribeiro2016/LundbergLee2017 and Shapley RAND1952 references support derivations.
  Site100reports/459blocks/394figures passes; originals removed after gates;
  no old catalog alias or owned SVG files existed. Evidence:
  build/mepml-conversion/explainable-ai/.

- 2026-11-07: Converted Exp-Golomb to full-range checked uint64 emission and
  bounded-prefix parsing through a compiled shared kernel. Original three inline
  names/signatures/defaults and append/prefix semantics remain. All9094 valid
  legacy scenarios match exactly, including independently recompiled frozen header.
  Decoder count remains an upper limit: complete shorter frames return fewer,
  count0 reads none and meeting the limit leaves any tail unread. Every attempted
  word must be complete; invalid order/count and native overflow throw. Old probes
  reproduce shift64/negative-shift UB, truncated heap overflow and UINT64_MAX
  encoding as a single zero. New sentinel arithmetic emits its actual129bit word
  without q+1 overflow or shifts by64. Reusable size/length/ASCII-debug helpers,
  actual work/consumed boundaries and owned emitted/parsed spans replace local
  report codec helpers. Bool-vector storage defines no byte format/endianness.
  Ordinary output has no optional history/quota. Trace logical C=B+2N reserves
  known encoding C before copies/emission; decoder reserves entire input B then
  two entries online per actual completed word. Unread tails retained/accounted,
  default1m/zero disables only quota, checked arithmetic/container limits remain;
  metadata/runtime/capacity/allocator bytes excluded, no peak-RAM claim.
  Fourteen ASAN/UBSAN cases include all65536 uint16 values at eight orders,
  all64orders with powers/neighbours/maximum and320 seeded full-width sequences
  against independent division/binary-string-carry oracle. All655355 binary-prefix
  requests through length16/orders0..4 check value/boundary or incomplete rejection.
  Proper truncations/allorders, native overflow, append prefix/strong parameter
  errors, count0/up-to/INTMAX framing, exact/base/one-short/zero quotas, ownership,
  all actual span/work/capture certificates and size/threshold controls pass.
  Audit60/60 definitions,17 compiled exact-output snippets,4clangd hovers,
  44affected coding/format+5dir and combined2302 tests pass. Five complete native
  programs/five inspected figures/both exports preserve 72 computed result
  lines and identical source/build/export images. Original codewords/mixed90bit
  sample/ten lengths retained; exact254/255 threshold, order-source costs,
  actual5N/7N work/history,129bit maximum, prefix/append/quota controls execute.
  Derives inverse/prefix-free proof and finite-vs-unbounded scope against verified
  ITU-T H.274(05/2022) section9.1 equation72/Table21; no full video/framing claim.
  Site99reports/453blocks/384figures passes; originals removed after gates,
  catalog updated and old figure paths absent. Evidence:
  build/mepml-conversion/exp-golomb/.

- 2026-11-07: Converted exact cover to a validated iterative dancing-links
  kernel, preserving the original compiled signature/result field layout and all565
  valid legacy selected-row sequences. Stable smallest-column/input-row ordering
  remains; every successful/unsuccessful returned search fully unwinds its mesh.
  Actual work/outcome/trace APIs expose live choices, residual column sizes, selected
  stack and complete initial/final node records, without explanatory replay. An
  explicit frame stack removes recursive call-stack dependence. Root/header/incidence
  geometry and original row IDs fit checked int/container limits; full incidence
  range and within-row uniqueness are validated. Duplicate whole rows and unsorted
  unique entries remain valid; empty rows are ignored and empty universe succeeds.
  Old probes show duplicate incidences incorrectly accepted and bad-column access
  causing ASAN heap overflow. New reusable rectangular domino builder keeps original
  horizontal-before-vertical placement order; native diagram validates board shape,
  adjacency, row IDs, exact cell coverage and scale/render quota. Logical trace
  B=R+I+2K plus sum(1+C+depth) counts input row/incidences, two node-record snapshots
  and actual events/residual vectors/partial rows. Known B checked before graph/copy
  allocation, events before copies; default1m/zero disables only quota, ordinary
  output/runtime/capacity/header/allocator bytes excluded, no partial result or
  peak-RAM claim. Ordinary calls retain no optional history. Full raw meshes agree
  after success/failure and all captures are read from actual runtime state.
  Six ASAN/UBSAN cases check all5058 small binary matrices and500 seeded random
  systems vs independent subset oracles; every event matches independent set-based
  residuals and every initial/final mesh satisfies circular-link/membership/size
  invariants. All63 rectangular encodings match direct placement generation and
  parity/cover proofs; large odd unsatisfiable boards use ordinary work, with bounded
  history on even/small odd cases after an intentional trace-cap test adjustment.
  Exact/base/one-short/zero quotas, ownership, all incidence/geometry/graphics limits,
  full work identities, known singleton/whole-row families and depth4096 complete.
  Two-path audit97/97 definitions, fifteen compiled exact-output examples and four
  clangd hovers pass.24 affected trace/CSP/Sudoku/format+5dir and combined2258 pass.
  Separately compiled frozen source reproduces565 original receipts. Five complete
  programs/five inspected native figures/both exports pass (47 computed lines,
  identical source/build/export images). Tiny99entry actual backtracking/full mesh,
  original4,1,3 seven-column selection/literal-paper comparison, original4x6 tiling,
  equal-node/different-scan/history formulas and domain/quota/deep controls execute.
  Correct branch/exchange-free AlgorithmX proof, exact inverse operations, primary-
  column scope, rectangle parity, exponential search limits and verified Knuth2000
  reference replace loose claims. Site98reports/448blocks/379figures passes; originals
  removed after gates, allthree catalog aliases updated and old SVG paths absent.
  Root storage recovered via two Nix-verified dead source snapshots and byte-verified
  NYU cache relocation preserving its canonical path; receipts remain under the
  conversion root. Evidence: build/mepml-conversion/exact-cover/.

- 2026-11-07: Converted forward Euler to a checked shared canonical
  euler_method.hpp/.cpp with ordinary/work/actual-derivative trace entry points and
  an allocation-free capture-size helper. Shared OdeSystem/OdeSolution moved to
  ode_types.hpp, preserving alias/signature and t/y field order; ode.hpp reexports
  them and Euler. Only the old Euler body moves out of ode.cpp; all other integrator
  bodies/helpers remain. Nonempty finite states, finite endpoints, positive steps,
  nonempty callbacks and full derivative dimensions/finiteness are validated before
  unsafe indexing; finite double increments/updated states are checked. Extended
  nominal h/interior grids avoid needless endpoint-subtraction overflow, pin final
  time exactly and validate strictly advancing nonzero grids before callbacks.
  Reverse/zero intervals remain supported; zero interval keeps N calls/N+1 samples.
  Time underflow/rounded stagnation and arithmetic failures are explicit. Ordinary
  calls retain intrinsic output history, without optional derivative capture/quota.
  Actual derivatives are captured from the same callbacks, never replayed. Complete
  known C=(2N+1)(d+1) is checked before history copies/callbacks; default1m/zero disables
  only quota, excludes other metadata/runtime/capacities/headers/allocator bytes and
  is not peak RAM. Failures expose no partial result; caller exceptions propagate.
  All132 frozen legacy full state trajectories (16268samples), sample counts and
  callback counts match exactly;130 time grids exact, two change by at most1.4211e-14
  under the explicit rounding policy. Separately compiled frozen original sources
  reproduce the original receipt. Old probes confirm accepted nonpositive steps,
  empty dimension, NaN derivatives and repeated rounded times; malformed derivative
  caused UBSan null-reference access. Six ASAN/UBSAN cases check closed discrete
  scalar/vector/oscillator solutions, independent nonautonomous quadrature, actual
  stateful callback sequences, known decay order/stability boundaries, whole quotas,
  ownership, all domains and finite overflow/underflow/forward/reverse/zero limits.
  Five-path audit67/67 definitions (originalODEbaseline34/34), nine compiled exact-
  output snippets and four clangd hovers pass.23 affected trace/oldODE/format+5dir
  and combined2234 cases pass. Six complete programs/six final inspected native
  figures and both exports pass (33 computed lines, identical source/build/export
  images). Actual tiny tangents precede original8step/9refinement decay, exact scalar
  stability disk/multipliers, oscillator energy-pumping proof and measured trace,
  time forcing/vector work and capture/precision controls. Assumptions behind O(h)
  global error, defect normalisation, strict decay vs neutral stability and floating
  limits are explicit; verified Hairer/Lubich and Moler primary references. Site97
  reports/443blocks/374figures passes. Originals removed after all gates; old owned
  euler_method.svg paths absent. Evidence: build/mepml-conversion/euler-method/.

- 2026-11-07: Converted planar Euclidean minimum spanning trees to a
  dedicated canonical euclidean_mst.hpp/.cpp with the original compiled function
  and advanced_graph.hpp forwarding include preserved. Only that old body/declaration
  move; unrelated graph functions remain. Complete lexicographic pairs feed a shared
  stable Kruskal kernel for ordinary result, logical work and actual owned scan
  histories. Generic finite signed weights, parallel edges, self-loops, disconnected
  forests, stable input ties and empty/singleton connectivity remain supported.
  Actual steps retain input index, pre-merge roots, acceptance and post-decision
  component count without replay. Planar trace owns points and complete edges with
  pair distances measured once. Checked divide-before-multiply pair geometry and
  full finite coordinate/weight/endpoint validation precede unsafe allocation/sort;
  differences, hypot distances and accepted-edge totals must stay finite. Old probes
  confirm nonfinite singleton coordinates/weights and nonfinite connected totals
  were previously accepted. Logical capture C=N+M+S (generic M+S) reserves known
  inputs before graph generation/copies and each step online before scanning;
  default1m/zero disables only quota; output/runtime/capacity/allocator bytes excluded,
  no partial result and no peak-RAM claim. Ordinary calls keep no optional history.
  All306 valid legacy scenario lines, including full selected-edge order/ties, match
  new outputs and separately recompiled frozen original sources exactly. Six
  ASan/UBSan cases cover350 planar clouds vs independent dense Prim and tree-path
  bottleneck certificates,30 exhaustive Prüfer-tree minima,400 generic graph/BFS
  scan certificates, original square/cluster-cycle/duplicates, full signed/loop/forest
  cases, exact work/quota/ownership/empty cases, finite extremes and pair/sum overflow,
  known line/grid and scale/translation identities. Four-path audit57/57 definitions,
  fourteen compiled exact-output snippets and four clangd hovers pass.35 affected
  custom/old advanced/Kruskal/Prim/format+5dir and combined2211 cases pass. Five
  complete programs/five inspected native figures/both exports pass (28 computed
  lines, identical source/build/export images). Original square3, actual cluster
  rejection/total11, duplicate zeroedge andscale3, seven known line-family work/capture
  cases and explicit quota/domain controls execute. Exchange proof handles ties;
  complete-pair complexity and early-scan limits, labeled duplicates, geometric
  units and tour/Steiner/squared-objective distinctions are explicit. Original
  Kruskal1956 reference verified. Site96reports/437blocks/368figures passes; original
  mepml/org removed after all gates; no old owned images. Evidence:
  build/mepml-conversion/euclidean-mst/.

- 2026-11-07: Converted the ellipsoid method to a positive triangular
  factor representation with O(n²) central cuts, preserving the original compiled
  function and three-field result layout. Shared actual callback execution supplies
  compact result/work and owned trace APIs, explicit limit/zero-subgradient/support-
  tolerance/rounded-center-stagnation reasons, measured factor log volumes, stable
  exact log-volume ratios and a checked native 2-D outline. Scaled projections and
  hypot avoid squared-norm overflow; positive diagonals are multiplied without
  dense cancellation. Full center/radius/cap/tolerance/callback/oracle/state shape
  and finiteness validation precedes unsafe access. Arithmetic loss is explicit;
  floating geometry does not verify convexity or enclosures. Actual trace quota
  C=Nsamples(1+n+n²)+Nattempts(1+n) is reserved before associated callbacks/copies;
  default1m/zero disables only quota, not a peak-memory bound. Ordinary calls retain
  no optional history. Original dense implementation demonstrably stopped on a
  negative metric with a nonsymmetric invalid matrix; wrong gradient shape triggered
  UBSan null-reference access. New original quadratic best1.98984e-13/support3.17709e-11
  meets tolerance1e-10 with the full500attempt allowance. 196 frozen old/new scenarios
  reviewed:144 short higher-dimensional results agree within measured maxvalue9.95e-12
  and maxpoint1.03e-11;48 corrected 1-D results match independent endpoint bisection.
  Old1D duplicated initial midpoint and omitted final cut center; new every accepted
  center is evaluated consistently. Both long quadratics meet independently known
  optima; plateau/constant exact minima retained, zero-gradient queries drop39→2 and
  20→1. Six ASan/UBSan cases check250 random factor cuts vs independent extended
  matrix formulas and retained-half membership, positive scaling, actual oracle
  sequences, known quadratics, nonsmooth/constant optima,48 interval cases, exact
  work/capture identities, all domains, ownership, quotas and finite arithmetic limits.
  All82 definitions/17 compiled exact-output snippets/four clangd hovers pass;
  20 affected native+5dir and combined2176 cases pass. Five complete report programs,
  five final inspected native figures and both exports (33 computed lines,
  identical source/build/export images) pass. Tiny unit cut and interval bisection
  precede actual original quadratic, exact volume-vs-growing-axis counterexample
  and work/precision controls. Conditional objective support and exact rho replace
  false volume-to-distance and unqualified rate claims; lecture reference verified.
  Site95reports/432blocks/363figures passes; originals removed after all gates,
  old owned SVG paths absent. Evidence: build/mepml-conversion/ellipsoid/.

- 2026-11-07: Extended Elias gamma with full positive uint64 support in a
  shared compiled emitter/strict parser, preserving the three original inline
  entry points and all 1,728 valid legacy encoded/decoded scenario lines. Integer
  bit width and shifts in [0,63] replace the undefined shift64 at n>=2^63. Zero
  encoding, nonbinary text, truncated words and trailing zero prefixes now fail
  explicitly; empty input stays valid and overwide prefixes raise overflow. The
  unchanged old-header probe reproduces shift64 UB and old silent zero omission/
  malformed acceptance. Ordinary/work calls retain no optional history or quota.
  New exact length/size summaries compare gamma to minimal common fixed width,
  excluding framing; ASCII bit characters are explicitly not packed bytes.
  Owned actual encode/decode spans share the real kernels, without replay. Capture
  C=B+2N counts bit characters, values and word records. Encoding validates complete
  known geometry/quota before copies/emission; decoding validates the full alphabet,
  reserves B before copying and two entries per complete word online before append.
  Default one million, zero disables only quota; failures expose no partial result,
  and scalar/runtime/capacity/allocator memory is excluded, not a peak-RAM bound.
  All 54 definitions, fifteen compiled exact-output examples and four clangd hovers
  pass. Six ASan/UBSan cases check all 65,535 positive16bit values against independent
  division codewords, all 131,071 binary streams through16bits against an independent
  codeword dictionary, 600 seeded full-width sequences, every power/neighbour and
  UINT64_MAX, actual spans with from_chars payload checks, exact work/size/capture
  identities, strict malformed/zero/overwide limits, ownership and quotas. 21
  compression/trace/format plus five dir and combined 2,156 cases pass. Five complete
  programs/five inspected native figures and both exports pass (61 computed
  lines, identical source/build/export images). Actual thirteen spans, original
  1..12 codes and74bit mixed sequence/minimalfixed100bits, exact dyadic bucket/Kraft
  identities throughK20,7Nbit/9Nhistory families, controlled magnitude comparisons
  and full-width127bit/129entry controls execute. Correct piecewise implied masses,
  symbolic finite-codebook remainder2^-64, source-dependent payload savings and
  framing/storage distinctions replace earlier loose claims. Site94reports/427blocks/
  358figures passes. Originals removed after all gates; old SVG paths absent.
  Evidence: build/mepml-conversion/elias-gamma/.

- 2026-11-07: Extended Edmonds-Karp through one checked complete-BFS kernel
  with ordinary work and actual owned search/augmentation capture, preserving the
  four-argument function and shared three-field MaxFlowResult layout. Entire finite
  nonnegative graph/geometry validation precedes adjacency allocation. Independent
  residual pairs retain self/parallel/antiparallel identities; successful flows use
  cancellation residuals, avoiding nearly-equal capacity subtraction. Every positive
  update/total must make finite representable progress, with explicit failure and no
  partial output. Full BFS retains original first sink paths and final cut order;
  no redundant uncounted cut traversal. Owned complete levels and first parents,
  path pair before/after values and work are captured without replay. Quota counts
  E+1+P(1+2V)+A+L; minimum first-search geometry precedes copies/adjacency, later
  search/path reservations precede traversal/push. Default one million, zero disables
  only quota; output/runtime/scalar/allocator bytes excluded, no peak-RAM claim.
  Ordinary/work APIs retain no optional history/quota. Reusable native flow/cut
  diagrams preserve edge identities using lanes/loops and caller positions, validate
  local shape without certifying edited results, and display four significant digits.
  Visual inspection caught labels crossing strokes; additive Diagram::label provides
  measured white backing and preserves all existing text/edge methods. Final affected
  figures were rerendered and inspected. All 127 definitions across four flow plus two
  diagram files, thirteen compiled exact-output examples and nine clangd hovers pass.
  Six ASan/UBSan cases include all 53,526 small graph/terminal cases and 600 seeded
  self/parallel/antiparallel networks against enumerated cuts, independent Floyd
  distances and lexicographic shortest-parent DAG checks, every actual residual
  update/work/capture certificate, ownership/quota/invalid/numerical/renderer cases,
  exact parallel counts and a 20,000-edge iterative path. Separate old/new binaries
  preserve all 1,232 baseline full value/flow/cut lines. Old probes confirm accepted
  nonfinite capacities and a wide-scale inconsistent flow; new errors are explicit.
  22 flow/format plus nine diagram and five dir cases and combined 2,135 cases pass. Five complete
  programs/five inspected figures and both exports pass (59 computed lines and
  identical source/build/export images): tiny bottleneck/cut, actual e2 cancellation,
  original CLRS flow23/cut23, exact 2k²+k versus Dinic3k BFS inspections and distinct
  self/parallel/opposite lanes with numeric/capture controls. Exact-arithmetic and
  numerical limits, isolated-vertex initialization, witness/certificate distinction
  and retained-memory exclusions are explicit. Site 93 reports/422 blocks/353 figures
  passes. Originals removed after all gates; old SVG paths absent.
  Evidence: build/mepml-conversion/edmonds-karp/.

- 2026-11-07: Moved periodic EDF/RMS scheduling to a shared compiled unit
  engine with owned ready jobs, stable input-order ties and no discarded backlog.
  Original RTTask/RTSlice/RTScheduleResult field order and both inline simulation
  entry points are preserved; CPU scheduling elsewhere is unchanged. Full domain
  validation rejects nonpositive periods, negative work/horizons, duplicate/reserved
  idle ids and invalid policies. Zero-work jobs complete at release; wide deadlines
  avoid release-plus-period int overflow. A miss is checked through boundary H;
  finishing exactly at a deadline is on time. Exact EDF rational screening uses
  private integer limbs with no tolerance or external bigint dependency. Numeric
  utilization/RMS sufficient-screen rounding is explicit; the stable expm1 RMS
  bound is not a necessary feasibility limit. Actual owned traces record releases,
  executed/completed/missed job outcomes and every selected job id in the same
  kernel. Raw N+H+J geometry and complete quota precede copies/history allocation;
  default one million/zero disables only quota, no partial return, runtime/result
  buffers and allocator bytes excluded. Ordinary/work calls retain no optional
  history or quota. Native task-id-to-row plotting validates slices/ids without
  claiming to certify caller-edited demand/verdicts. Six ASan/UBSan cases compare
  2,592 exhaustive policy/task/horizon scenarios and 600 seeded fixtures (both
  policies plus reversed input order) against an independent offline-arrival,
  flat-ready-job scan oracle, checking every tick/job and exact count/cap/ownership
  certificates. Boundary completion, retained expired jobs, zero work, opaque ids,
  full validation and many-limb exact sums pass. Old probes confirm omitted H=2
  miss and accepted 1+1/INT_MAX overload; corrected outcomes are shown. Separate
  binaries preserve 2,708 feasible baseline timeline/verdict lines. All 79 affected
  definitions, nineteen compiled exact-output snippets and four clangd hovers pass;
  30 scheduling/trace/format plus five dir and combined 2,113 cases pass. Five
  complete programs/five inspected figures and both exports pass with identical
  source/build/export images. Actual preemption, original 13-job/20-busy/4-idle
  hyperperiod, EDF versus RMS missed jobs, harmonic U=1 repetition, exact unit/history
  families, overdue selections and quota/domain controls execute. Model assumptions,
  finite-prefix versus repeated-state proof, sufficient versus necessary RMS claims
  and explicit idle-work complexity are stated. Site 92 reports/417 blocks/348
  figures passes. Originals removed after gates; old SVG paths absent.
  Evidence: build/mepml-conversion/edf/.

- 2026-11-07: Extended Eclat with owned sorted vertical tidsets, explicit DFS
  frames, checked work counters and actual conditional-intersection evidence.
  Original signature, FrequentItemset layout, canonical order, empty/repeated rows,
  duplicate-item normalization, opaque full-width ids and zero-support convention
  are preserved. Other miners, shared normalization, rule/encoding/label bodies are
  byte-unchanged. All sibling intersections precede descent; history records both
  actual conditional parents, failures and separate emission ids. Ordinary/work
  calls retain no optional history or quota. Trace quota counts raw rows/items,
  state headers/item ids/tids and emitted ids; base is checked before copies and
  state/emission costs online. Default one million, zero disables only quota;
  working intersection/result copies and allocator bytes are explicitly excluded,
  so the quota is not a peak-RAM claim. No partial trace escapes failure. Six
  ASan/UBSan cases include 2,842 exhaustive database/threshold cases vs independent
  subset support, 500 seeded sparse/duplicate fixtures plus row permutations,
  every state's direct membership certificate, common-prefix actual operands,
  independent union-based merge-step identities, exact work/count/depth/cap
  certificates, ownership and zero/empty/full-width limits. Separate binaries
  preserve all 2,858 baseline scenario lines, including Titanic. All 94 affected
  definitions, ten compiled exact-output snippets and four clangd hovers pass;
  22 association/trace/format plus five dir cases and combined 2,083 cases pass.
  Five complete programs/five inspected figures and both exports pass (81 computed
  lines, identical source/build/export images). Original eleven groceries, actual
  conditional triple and beer/diapers operands, dense F=2^U-1 and R=F-U, disjoint
  R=U(U-1)/2, row scaling, canonical Titanic (891 rows, 32 sets at support45), and
  exact 28-entry capture demonstrate semantics. Positive-threshold cross-checks,
  FP-growth's current zero-support difference, output/failed-attempt costs and
  conditional-memory bounds are explicit. Site 91 reports/412 blocks/343 figures
  passes. Originals removed after all gates; original SVG paths absent.
  Evidence: build/mepml-conversion/eclat/.

- 2026-11-07: Converted Earley recognition to a shared compiled indexed
  closure while retaining both inline entry points, legacy int column counts,
  original production ids/order and duplicate-rule identity. Copied agenda items,
  indexed waiters and first completed spans replace unsafe vector iteration and
  recognize nullable cycles under either waiter/completion arrival order. Ordinary
  work/boolean/size calls return no input or reason history; sparse runtime columns
  avoid eagerly allocating all indexes. Opaque strings include empty/NUL/UTF-8
  names; an empty RHS differs from an empty-named terminal, and an undefined start
  rejects. Raw index/count geometry precedes allocation. Owned actual traces keep
  the chart and first-insertion parents without replay or a parse-forest claim.
  Capture base 2N+P+sum(rhs lengths)+2 is checked before copies/column metadata;
  each new item/reason reserves two records online, duplicate attempts consume none.
  Default cap is one million; zero disables only quota, and failures expose no
  partial trace. A checked selected-item formatter escapes control symbols without
  promising to certify arbitrary edited traces. All 83 definitions, eleven compiled
  exact-output snippets and four clangd hovers pass. Seven ASan/UBSan cases include
  all 8,191 short a/b words, 7,560 seeded nullable/cyclic grammar queries vs an
  independent least fixed point of interval languages, first-parent deduction
  certificates and final-chart prediction/scan/span/join work identities, rule
  permutations/duplicates, opaque names, caps/ownership/formatting and a 20,000-link
  iterative unit chain. An unchanged old-header probe reproduces heap-use-after-free
  for S->A,A->epsilon; the new probe accepts without sanitizer errors. Separate
  old/current binaries preserve 8,194 exact verdict/chart-size scenarios. Thirty
  affected parser/format plus five dir cases pass; combined targeted set 2,061.
  Five complete programs/five inspected figures and both exports pass, including
  37 computed lines and identical source/build/export images. Original nine
  examples/chart counts, actual four-state two-parent completion, late nullable
  join and empty-RHS/token distinction are shown. Exact linear balanced-family
  items, quadratic ambiguous-family items and cubic waiter/span joins are derived
  and checked, with expected-hash/fixed-grammar and no-all-parses limits explicit.
  The original barplot label was corrected to an explicit title; that final figure
  was re-inspected and four other hashes are unchanged. Website 90 reports/407
  blocks/338 figures passes. Original mepml/org removed after gates; old SVG paths
  absent. Evidence: build/mepml-conversion/earley/.

- 2026-11-07: Converted dynamic time warping with the original inline entry
  point, three-field DTWResult and diagonal/advance-a/advance-b tie order retained.
  Shared compiled canonical kernel uses two double cost rows plus one predecessor
  byte per positive cell. Complete finite input and signed-index/product/capture
  geometry validation precedes allocation and optional capture checks. Nonfinite
  off-path accumulated costs remain infinity; only an unrepresentable final optimum
  rejects an otherwise finite-input solve. Actual owned traces capture both inputs,
  positive-cell costs and selected directions during that same recurrence. Complete
  n+m+2nm logical entry count is checked before capture allocation, default one
  million; zero disables only the optional cap. Ordinary/work calls retain no input
  or cost history. A complete witness validator/cost sum and native alignment plot
  port reusable report logic; plot checks feasible endpoints/steps/exact forward
  cost and finite display offsets, not optimality. All 42 definitions, eight exact
  compiled documentation snippets and three clangd hovers pass. Five ASan/UBSan
  cases cover 14,400 ternary short-sequence pairs with independent exhaustive path
  enumeration and every prefix table/direction, 500 seeded symmetry/scaling/L1
  controls, nonmetric counterexamples, input/cap/index/overflow/witness/plot limits,
  ownership and a 50,001-pair iterative traceback. Separate old/current binaries
  preserve 14,402 distances and exact paths; original NaN/infinity acceptance is
  demonstrated by compiled old-header probes. All 17 affected native and five dir
  cases pass; combined targeted set 2,031. Six complete report programs and six
  inspected native figures pass both computed exports and image-byte checks.
  Original zero-cost tiny path and 30-sample sinusoid values are retained, including
  bit-identical vector-generated inputs and exact path. Fixed canonical GunPoint
  fitting sources 40/5/20 supply recorded 150-sample examples without test use,
  extra normalization, synthetic data or benchmark claims. Exact recurrence work,
  buffer-slot counts versus total storage, rigid L1 versus Euclidean naming,
  zero-distance/triangle failures and unrestricted-variant limits are explicit.
  Website 89 reports/402 blocks/333 figures passes. Original mepml/org removed after
  gates; old owned SVG paths absent. Evidence: build/mepml-conversion/dtw/.

- 2026-11-07: Converted the Iris CART report from repeated disabled chunks
  and one assembled program into six complete native C++ programs. Retained the
  original four-predictor depth-three fit (nine nodes, five leaves, 146/150 correct),
  actual node thresholds, original confusion/importance values, eight-depth curve
  and four new-flower leaf-share predictions. A frozen 201-line receipt additionally
  verifies all 150 fitted labels and exact original summaries, node table and
  predictions before/after the change. Added DecisionTreeClassifier::plot_tree():
  actual model nodes, shortest round-trip thresholds, sample counts and fitted
  Gini/entropy labels, class-coloured leaves and <=/> branches; iterative layout
  preserves left-to-right fitted leaf order. The default 255 complete-node cap is
  checked before layout allocation; zero disables only the optional cap. Returned
  native Diagram owns its labels and outlives the unchanged model. Four new tests
  cover exact/short/disabled limits, tiny and full fits, mixed unsplit root,
  ownership, no refitting/mutation, entropy and XML-escaped labels. All 17 classifier,
  regressor and diagram cases pass under ASan/UBSan, 22 native/dir cases pass and
  combined targeted set 2,014 passes. Both canonical paths audit 72/72 definitions;
  all four existing/new documented programs execute, three with exact expected
  stdout and one checked five-node table; the new method's clangd hover passes.
  Six complete programs/six inspected native figures and both computed exports pass;
  47 computed lines and source/build/export image bytes agree. The final predictor
  axis and region-range corrections were inspected; four other image hashes match.
  A six-row Gini example precedes Iris; actual fitted diagrams, classification/
  importance, 120-train/30-test stratified held-out confusion, original plus held-out
  depth curves and sampled petal regions are executed. Training fit, exploratory
  test reuse, leaf shares, tied splits, importance and data-version limitations
  are explicit; no unexecuted LDA/SVM comparison remains. A nested text fence in
  computed output was interpreted as a new executable block; replacing it with
  the native node table and rerunning all six programs resolved the format issue.
  Website 88 reports/396 blocks/327 figures passes; original mepml/org removed only
  after gates and all old owned figure paths were absent. Evidence:
  build/mepml-conversion/dt-iris/.

- 2026-11-07: Converted D*-style replanning into an explicit persistent D* Lite
  planner while preserving the one-shot function, original tiny route, GridPathResult
  field order and failure convention. Shared canonical C++ keeps exact bounded
  g/rhs labels, an indexed heap with one entry per inconsistent cell, a fixed goal,
  checked Manhattan movement modifier, ordered validated obstacle batches and
  deterministic optimal route extraction. Complete shape/coordinate/product/cap
  validation precedes allocation or mutation. Owned copies retain full queue
  capacity; assignment constructs a replacement before nonallocating swaps.
  Actual lower/raise/key-refresh trace operands include initial/final fields,
  stored queue keys and pending work. Logical capture cost is 5N+Q+P, checked
  before copies/pops; a stopped capture leaves valid resumable progress, with
  work_before separating prior pops from this call. Ordinary repair stores no
  history. Checked grid-mask construction and native route plotting move reusable
  setup/rendering into core; supplied routes are checked for feasibility/cost.
  Eight ASan/UBSan cases pass, including 41,472 static mask/terminal BFS checks,
  29,952 exhaustive flip/restore trace certificates, 2,400 seeded changing-map
  queries and a 50,001-cell corridor. Independent certificates check actual queue
  ordering, Bellman updates, labels, termination, capture and all work counters.
  Separate HEAD/current binaries preserve 1,001 original scenarios, including the
  exact tiny path. All 105 definitions, 15 compiled exact-output examples, four
  clangd hovers, 23 affected native cases and five dir cases pass; combined targeted
  set 1,997. Five complete report programs and five inspected native figures pass
  both computed exports with identical source/build/export image bytes. The report
  distinguishes classic D* from Lite, shows disconnection/reopening and lazy keys,
  and measures reuse without promising each edit beats fresh search. Website
  87 reports/390 blocks/321 figures passes. Original mepml/org removed after gates;
  neither source referenced owned figures. The previously recorded 35 unrelated
  documentation gaps remain outside this batch. Evidence: build/mepml-conversion/d-star/.

- 2026-11-07: Converted DPLL with the original function, two-field SatResult,
  first-occurrence unit/smallest-id pure/first-variable true-first order, raw clause
  order/multiplicity/tautologies and total-model completion retained. Full signed-id
  validation precedes verdicts, dense allocation and optional limits; zero, INT_MIN,
  out-of-bound ids and negative counts are rejected. Widened n+1 arithmetic and
  presence polarity bits avoid signed count/frequency overflow. Heap-backed frames
  replace native recursion, retain parent states and expose actual node arrival,
  unit/pure assignments, conflicting original clause, restored failure returns,
  both-child exhaustion and satisfied partial states without replay. Original
  input/complete assignment event entries are capped online, default one million,
  with input/root cost checked before dense allocation; zero disables only cap.
  Ordinary/work calls retain no history. A validated total-model checker and bounded
  original-order mt19937 three-distinct-variable generator move reusable report code
  into core; invalid generator controls leave engine state untouched. Logical work
  separates outer clause visits from literal visits including both pure passes.
  Full two-path audit covers69 definitions, eleven compiled/executed exact-output
  examples and five clangd hovers. Seven ASan/UBSan cases pass:all512 two-variable
  clause sets and19683 ordered three-clause/three-variable formulas vs exhaustive
  models;800 seeded raw formulas; actual state/reason/pure-polarity/conflict/parent/
  exhaustion/model certificates and node/branch/capture identities; empty/zero-var,
  duplicates/tautologies/unused variables, total-model validation, caps/priority,
  ownership, generator stream/state and1024-decision explicit-stack controls.
  Compiled unchanged HEAD-source probes expose invalid-literal vector-index assertion,
  INT_MAX+1 signed overflow and a negative bound accepted as SAT. Separate old/current
  binaries match2522 original verdicts and full models, including all2520 random
  instances. All21 original rounded fractions match the rerun; effective ratios,
  Wilson intervals and measured median work now qualify their finite scope.
  Visual review found unsupported logical AND/OR rendered as question marks in two
  native diagrams. Added U+2227/U+2228 to the canonical font generator and regenerated
  C++ data through just embed-plot-font twice, byte-identically. A native regression
  requires two distinct actual outlines, neither the fallback glyph. Twenty-five
  affected DPLL, five dir and nine diagram cases pass; combined targeted set1974.
  Five complete report programs/five inspected captioned native figures and both
  computed exports pass;56 result lines and source/build/raw figure bytes verified.
  All final images re-inspected after font correction; three other hashes unchanged.
  Unit proof, complete three-node refutation, original/pure-alternative checked
  models, original finite random sweep and exact tautology-family scan formulas
  through depth1024 are executed. Largest sampled median54.5 nodes at actual4.48
  and nonzero high-ratio counts are stated without an asymptotic threshold claim.
  Website86 reports/385 blocks/316 figures passes. Original mepml/org removed after
  gates; old dpll.svg source/build paths absent. The previously recorded35 unrelated
  completed-manifest documentation gaps remain outside this batch; no broad audit
  was rerun. Evidence: build/mepml-conversion/dpll/.

- 2026-11-07: Converted Double Dabble around a shared compiled shift/add-three
  kernel, preserving both original function signatures, most-significant-first
  unpacked digits, deliberate low-bit truncation and singleton zero at width zero.
  Widths outside0..64 now fail before allocation/subtraction/shifting; explicit
  zero/full-width selection avoids shift-by64. An exact65-entry integer capacity
  lookup replaces floating capacity arithmetic. Ordinary conversion, string output,
  actual trace and work-only calls share the kernel; ordinary/work calls retain no
  history. Owned before/corrected/after/carry vectors preserve requested-width zero
  padding, consumed prefixes, actual input bit indices and logical counters. Maximum
  capture is64 stages of20 nibbles, with no unnecessary history quota. A checked
  inverse validates all unpacked0..9 digits before decimal Horner overflow checks;
  leading zeros are accepted and its multiply-ten/add arithmetic is distinguished
  from the conversion recurrence. Full two-path audit covers35 definitions, nine
  compiled/executed exact-output examples and four clangd hovers. Seven ASan/UBSan
  cases pass:all65536 sixteen-bit values vs standard decimal formatting/round-trips;
  all8191 inputs through twelve bits with full actual stage certificates against
  independent schoolbook decimal doubling; all65 widths with zero/all-one and2210
  total seeded/boundary controls, ignored high bits, decimal/binary thresholds,
  full uint64 max, validation/overflow priority and snapshot ownership/padding.
  Counters verify checks=shifts=B*D and corrections=decimal carries. Separate
  unchanged HEAD-header probe confirms the old shift-by64 at width65 and accepted
  negative-width zero. Separate old/current binaries match138 conversions:all eight
  original cases plus two values at every width0..64. Thirteen affected native plus
  five dir cases pass; combined targeted set1940. Five complete programs/five
  inspected captioned native figures pass first review; all42 computed result lines
  appear in both exports and all source/build/raw figure bytes match. Tiny13 carry
  diagram precedes every actual243 stage, original eight conversions/full64-bit max,
  capacity/value-length comparison, measured work families and truncation/inverse
  validation controls. Local digit/carry and global prefix proofs justify the
  recurrence; report separates software nibble counts from CPU timings or hardware
  cycles, and unpacked BCD/decimal text/checked inverse semantics. Website85 reports/
  380 blocks/311 figures passes. Original mepml/org removed after gates; old generic
  double_dabble_digits.svg source/build paths absent. The previously recorded35
  unrelated completed-manifest gaps remain outside this batch; no broad audit was
  rerun. Evidence: build/mepml-conversion/double-dabble/.

- 2026-11-07: Converted Conway's Doomsday rule to a canonical compiled kernel,
  retaining all four original function names/signatures, Sunday-first codes and
  arbitrary signed-code name normalization. Date calls reuse existing Gregorian
  month validation before lookup/subtraction, rejecting invalid months/days and
  nonexistent leap dates. Widened floor-based century decomposition supports the
  full int-year domain, including astronomical year zero and negatives, without
  signed overflow or negative weekday results. Fixed owned year/date terms expose
  century residue/anchor, within-century year, direct quarters, dozen decomposition,
  leap status, canonical reference, signed offset and final weekday without replay.
  Fixed twelve-month anchors retain March14, leap-adjusted January/February and
  actual ordinal days/weekdays. All public semantics specify proleptic Gregorian
  rules, defaults/ownership, historical-cutover/timezone limits and O(1) work.
  Full two-path audit covers37 definitions, ten compiled/executed exact-output
  examples and four clangd hovers. Six ASan/UBSan cases pass:730485 daily comparisons
  against independent sequential-day/month-length oracles across five complete
  400-year cycles; representative chrono dates and10000 full-int-year samples;
  full monthly anchor/quarter/dozen/ordinal identities, century/leap boundaries,
  extreme invalid month/day inputs, complete validation priority, signed names
  and owned copies. Separate unchanged HEAD-header probes confirm the old month13
  bounds error, INT_MIN day subtraction overflow and negative-year anchor-3.
  Separate old/current binaries match all64 original scenario values (three event
  names,2025 anchor,eight original monthly anchors,twelve month starts,forty yearly
  anchors). Twenty-two affected native plus five dir cases pass; combined targeted
  set1927. Six complete report programs/six inspected captioned native figures and
  both computed exports pass; all71 result lines and source/build/raw figure bytes
  are checked. Tightened cycle-panel labels and widened a signed-century diagram
  gap; both corrected images re-inspected, four other reviewed hashes identical.
  Worked Christmas2025 trace and both full common/leap anchor sets precede original
  events/40-year scatter, exact400-year counts/century controls,144-row AirPassengers
  calendar annotation and negative/extreme/invalid-date controls. Report derives
  the century/dozen/whole-week identities and146097-day period, while separating
  Gregorian conventions and monthly observations from local historical calendars
  or inferred daily traffic. Website84 reports/375 blocks/306 figures passes.
  Original mepml/org removed after gates; old doomsday.svg source/build paths absent.
  The previously recorded35 unrelated completed-manifest gaps remain outside this
  batch; no broad documentation audit was rerun. Evidence: build/mepml-conversion/doomsday/.

- 2026-11-07: Converted Dixon integer factorisation to a dedicated canonical
  dixon.hpp/.cpp, retaining the original two-argument function and its old header
  reexport. Only the old Dixon body and its exclusive prime helper move out of
  number_theory_extra.cpp; other factorisation implementations stay intact.
  Shared exact modular scanning, positive smooth relations, all-row GF(2) reduction
  and original-row selection now use dynamic packed vectors instead of unguarded
  single words. Singleton dependencies are eligible; both exact GCDs are retained,
  with a proper minus divisor selected before plus. Inputs require n>=2 and bases
  in1..4096; full requested scan endpoints are validated before allocation. Zero
  residues are counted/skipped before division. Owned fixed-window collections and
  actual factor traces expose original exponents, selected row identities, full
  even sums, roots, GCDs, rank, successful divisions, whole-row XOR work and logical
  captured cells without replay. Default one-million scalar-cell capture cap is
  checked before each affected capture; zero disables only the optional cap.
  Compact factoring stores no dependency history. Full two-path audit covers49
  definitions, seven compiled/executed exact-output examples and four clangd hovers.
  Seven ASan/UBSan cases pass:9568 small modulus/base/budget combinations and714
  independent full-window comparisons; Boolean elimination and repeated-add modular
  arithmetic verify row selections, congruences, rank, both GCDs, divisions, XORs
  and capture accounting. Full-width moduli/root boundaries, zero residues, primes,
  even/square inputs, exact/one-short caps, validation/endpoint overflow, ownership,
  high parity bits across63/64/65 and127/128/129, and a130-row prime-modulus batch
  that exhausts every dependency through selection index129 are covered. Compiled
  unchanged HEAD-source probes confirm the old shift-by64 and modulus-zero division.
  Eleven affected native plus five dir cases pass; combined targeted set1905.
  Five complete programs/five inspected captioned native figures pass first review;
  all36 computed result lines appear in both exports and all source/build/raw figure
  bytes match. Original four factors/cofactors and25-in4000 density are preserved;
  report adds hand congruences for77/69, original batch counts/ranks/trivial attempts,
  six nested-base density curves, capture/failure controls and dynamic-boundary rows.
  Derivation separates rank-nullity from successful factoring, proper/composite
  divisors from prime factorisation, and bounded consecutive scanning from Dixon's
  random-list expected-time theorem. Website83 reports/369 blocks/300 figures passes.
  Original mepml/org removed after gates; old dixon.svg source/build paths absent.
  The previously recorded35 unrelated completed-manifest gaps remain outside this
  batch; no broad documentation audit was rerun. Evidence: build/mepml-conversion/dixon/.

- 2026-11-07: Converted the Euclidean distance transform around a shared compiled
  one-dimensional lower-envelope kernel, preserving the original grid function
  name/input/return shape and a forwarding detail::dt_1d spelling. The geometry
  umbrella now exposes the APIs through dm/dm.hpp. True infinity candidates are
  skipped instead of using1e20; size_t indices are converted before products and
  factored long-double intersections replace signed int squares. Empty/zero-width
  shapes are safe, ragged grids and invalid costs/sites/counts are rejected, and
  optional complete input/cell caps precede capture. Owned actual 1-D comparisons,
  boundaries, removed candidates, costs/argmins and logical counters explain the
  envelope without replay. The 2-D feature/trace APIs retain squared/Euclidean
  distances, original nearest rows/columns and column-pass intermediates; exact
  ties choose smaller column then row, and no-site sentinels are explicit. A
  bounded site-grid constructor and native distance-field plot replace reusable
  report-local preparation/graphics. Full two-path audit covers54 definitions,
  seven compiled/executed exact-output examples and four clangd hovers. Nine
  ASan/UBSan cases pass:87381 weighted vectors against independent quadratic minima,
  all74954 binary masks through4x4 against all-site distances/argmins/column minima,
  actual envelope pop/boundary/work identities, original9x15, char/tie/empty/no-site
  shapes, caps/validation/ownership/plot controls, a50001-finite-candidate line and
  100001-sample/single-row regressions. Compiled unchanged HEAD-header probes expose
  signed overflow at index46341 and the old zero-width vector-index assertion.
  Fourteen affected native plus five dir cases pass; combined targeted set1894.
  Five complete programs/five inspected captioned native figures, both computed
  exports and byte-identical source/build/raw bundles pass. Every one of the135
  original rounded distances and the row-four profile match the executed report.
  Shortened a clipped envelope legend and compacted two equal-scale diagrams;
  all three corrected figures were re-inspected while the other two reviewed
  hashes stayed identical. Report derives crossings/separability/amortized counts,
  verifies single-site/all-site count formulas, and separates exact metric choice
  from floating rounding, padding, obstacle routes and witness tie conventions.
  Website82 reports/364 blocks/295 figures passes. Original mepml/org removed after
  gates; original distance_transform.svg source/build references checked absent.
  The previously recorded35 unrelated completed-manifest gaps remain outside this
  batch; no broad audit was rerun. Evidence: build/mepml-conversion/distance-transform/.

- 2026-11-07: Converted Dinic into dedicated canonical dinic.hpp/.cpp, retaining
  the four-argument function, CapacityEdge spelling, advanced_graph.hpp reexport
  and MaxFlowResult layout/formatting; only the old Dinic body moves out of the
  shared advanced module. Flat residual pairs preserve original identities for
  loops/parallel/antiparallel arcs. Strict-positive residuals replace the former
  absolute 1e-12 cutoff; a compiled HEAD-source probe returns 7.5 and zero for
  capacities 7.5 and 7.5e-13, while the new solver retains both positive values.
  Iterative current-arc traversal avoids recursive vertex-depth calls. Complete
  validation precedes adjacency allocation, and checked residual/aggregate
  arithmetic rejects nonfinite or lost positive progress before committing a path.
  Shared actual capture owns original input, every BFS level set including final
  failure, pushed paths and oriented pair before/after operands, with no replay.
  History cost P*(V+1)+A+L is checked before BFS/path mutation; default one million
  entries, zero disables only the optional cap. A separate generic numerical
  flow/cut check exposes capacity/conservation/value errors, signed gap, crossing
  identities and distinct feasible/optimal flags with explicit absolute tolerance.
  Checked disjoint/staircase fixture generators test one versus increasing phases.
  Scoped two-path audit covers 79 definitions, six compiled/executed exact-output
  examples and four clangd hovers. Eight ASan/UBSan cases pass, including all 729
  ternary three-vertex networks/all six terminal pairs and 4096 binary four-vertex
  networks/all twelve terminal pairs: 53526 computations agree with independently
  enumerated cut minima. Separate replay verifies levels, bottlenecks, residual
  snapshots, blocked level graphs and DFS-inspections=pointer-advances+path-updates.
  Original/CLRS/cancellation, loops/parallel pairs, capacity scales, exact/one-short
  history and generator caps, arithmetic/validation priority, malformed supplied
  certificates, ownership and a 30000-edge iterative chain are covered. Thirty-one
  affected native plus five dir cases pass; combined targeted set 1880. Five complete
  report programs/five inspected captioned native figures and both computed-result
  exports pass with byte-identical source/build/raw bundles. All figures passed
  first review. Report executes the original7.5, classic23/cut12+7+4, reverse-edge
  rerouting across levels3/5, numerical certificate counterexamples, phase/work
  families and a20000-edge chain. Fresh-XML website81 reports/359 blocks/290 figures
  passes. Broader completed manifest has19700/19735 documented definitions:34
  missing CUDA descriptions and one simulation description are outside this change;
  both files are unchanged from HEAD. Scoped Dinic coverage remains79/79. Original
  mepml/org removed after gates; neither had owned figure assets. Evidence:
  build/mepml-conversion/dinic/.

- 2026-11-07: Converted Dijkstra with the original finite-input distances, strict
  adjacency/tie behavior, public signatures, two-field ShortestPathResult and
  formatting retained. A shared lazy-heap kernel records actual fresh/stale pops,
  outgoing-edge operands/outcomes, original selected arc identities, settlement
  order and logical work without replay or full snapshots. Complete finite
  nonnegative weight/endpoint validation precedes adjacency allocation, including
  unreachable arcs; an examined nonfinite candidate sum now throws explicitly even
  when it would not improve a known route. Online caps count pops plus inspections,
  default one million, checked before each event; zero disables only that budget.
  Path reconstruction retains empty out-of-range/unreachable convenience while
  bounding and checking reachable public-record chains for malformed shapes,
  distances, indices and cycles. A bounded descending parallel-arc generator
  exposes O(E) lazy heaps at V=2 with exact binary64 integer weights. Full two-path
  audit covers 53 definitions, six compiled/executed exact-output examples and
  four clangd hovers. Nine ASan/UBSan cases pass, including all 4096 three-vertex
  zero/one/two/absent-cost digraphs from every source (12288 traversals), independent
  Floyd-Warshall distances and multiset-based complete actual-event certificates;
  original six cities, zero cycles, duplicates, tie rules, bands, parallel arcs,
  event/generator cap boundaries, validation/overflow priority, maximum/subnormal
  finite costs, negative zero, malformed parents and owned copies. Thirty-nine
  affected native cases plus five dir cases pass; combined targeted set 1849.
  Five complete programs and five inspected captioned native figures retain all
  original six-city roads, distances and route A-C-B-D-F at cost eleven. Crowded
  crossing-road labels were separated and a base-two count plot now distinguishes
  heap growth from two settlements; both corrected figures were re-inspected and
  three other reviewed hashes stayed identical. The report proves settlement,
  R+1 push/pop conservation and the unrestricted multigraph versus simple-graph
  bounds, with closed-form band and parallel controls. Both computed-result exports,
  identical packaged source/build figures and website 80 reports/354 blocks/285
  figures pass. Original mepml/org removed after gates; all original figure paths
  checked absent. Evidence: build/mepml-conversion/dijkstra/.

- 2026-11-07: Converted diffusion maps with the existing core constructor, four-field
  option aggregate, defaults, compact model storage and binding facade signatures
  retained. A shared fit kernel exposes owned actual squared distances, Gaussian K,
  alpha-corrected A, degrees, stationary probabilities, transition P, symmetric S,
  full right eigenbasis and retained coordinates without replay. The symmetric
  Householder/QL solver replaces cyclic Jacobi. Nonfinite controls/retained data,
  squared-distance overflow and floating-point disconnected kernels are rejected;
  small roundoff-negative eigenvalues are clipped before nonnegative-time filtering.
  Trace observation/capacity guards precede dense allocation. Generic upper-median
  global/kth-neighbour squared-distance summaries and aligned between/within centroid
  scatter replace report-local helpers. An explicit SwissRollEngine overload retains
  the original 400 mt19937_64/libstdc++ draws at seed 42 while the existing Xoshiro
  API and outputs remain unchanged. Full six-path audit covers 76 definitions,
  eleven compiled/executed exact-output examples and four clangd hovers. Nine new
  ASan/UBSan tests pass, including 108 independent transition-power/spectral fits
  at four times, rational three-point matrices/spectrum/distances, reversibility,
  weighted orthogonality, filtering, null/cap/ownership/overflow controls, all ranks
  at sizes 2..19, centroid scatter/similarity identities and original-generator
  replay controls. Seventeen affected native cases plus five dir cases pass,
  including both legacy facade constructors, owned embedding frames and unchanged
  options/model formatting; combined targeted set 1810 passes. Six complete report
  programs and six inspected captioned native figures pass. Two iris panel lower
  bounds were expanded and re-inspected; the other four final figures retain
  reviewed identical bytes. Both computed-result exports, packaged source/build
  figure bundles and website 79 reports/349 blocks/280 figures pass. The report
  corrects right-eigenvector and stationary-normalization factors, separates
  asymptotic alpha claims from finite labelled separation, retains all original
  iris/roll/PCA/MDS scenarios, and explicitly shows broad-kernel height recovery
  in coordinate three versus local-kernel height failure in the first four.
  Original mepml/org removed after gates; their twelve referenced build images
  were absent. Evidence: build/mepml-conversion/diffusion-maps/.

- 2026-11-07: Converted educational Diffie–Hellman around the existing exact uint64
  right-to-left exponentiation kernel, preserving legacy public/shared-key arithmetic
  names/includes, all valid nonzero-modulus results, modulus-one/zero-exponent
  conventions and original fold/square/shift order including the final unused square.
  Canonical mul_mod now shares the full unsigned 128 product implementation and rejects
  zero before remainder evaluation; the original diffie_hellman include still reexports
  it for Shamir. Actual owned bit traces record remaining exponent, running base,
  accumulator before/after, bit decision and squared base without replay. Recorded
  counts distinguish bit-width squares, popcount accumulator products and their sum;
  complete effective-bit caps (default 64, zero disables optional cap) precede
  allocation/powering, with modulus one returning zero work. A deterministic four-power
  exchange record retains public/raw shared values without storing private exponents. A
  bounded prime-validated generated-cycle API accepts any canonical nonzero element,
  includes identity once, computes its exact order with one enumeration product per
  value including the closing product, and applies online order/capacity guards;
  enumeration counts exclude primality checking and zero disables only the optional
  cap. The report reproduces original 23/5/6/15 public 8/19/shared 2 and original
  31-bit public 744111946/1598667707/shared 490982923. Existing
  baby_step_giant_step_trace recovers exponent 123456 from public inputs with 46,341
  baby products and 3 giant lookups and recomputes the shared value, exposing the toy
  size without adding a new attack algorithm. Actual 5^6 trace shows three squares/two
  folds/result 8 and unused final square 16; 19^6 gives 2. Generated cycles for g
  1/4/5/22 modulo 23 have orders 1/11/22/2; the original 23 plotted exponents include a
  repeated identity and a tiny table recovers exponent representatives 6/15. A
  two-session substitution gives Alice/intermediary 18 and intermediary/Bob 15 despite
  both local agreement checks passing; unvalidated 0/1/22 peers expose collapsed/parity
  outputs. Sparse/dense exponents of 64 bits take 65/128 products; full-width product
  arithmetic gives 2 versus premature uint64-wrap 6. Zero-work and exact 3-bit/22-cycle
  caps are checked. Primary-paper claims distinguish CDH shared-value computation from
  discrete-log recovery sufficiency and reject a scattered plot as evidence of
  hardness; RFC 7919 peer-check discussion stays within its known-group context.
  Variable work/exposed trace bits, missing randomness/authentication/group
  validation/KDF and raw arithmetic versus secure protocol scope remain explicit. Nine
  ASan/UBSan tests pass: 72,900 small arithmetic exchanges and 103,275 complete power
  traces use repeated multiplication/state invariants; 5,000 full-width products and
  500 powers use independent addition-only/left-to-right references. Every element of
  every prime through 127 is enumerated independently; a full-width prime's order-two
  element exercises exact closing multiplication. Original/period/subgroup/substitution
  cases, all 64 exponent widths, zero-modulus/zero-work controls, complete/online caps,
  primality/range guards and owned copies pass. A separate Python arbitrary-precision
  CTest verifies 20,000 product/pow controls. Thirty-six native cases, one Python
  oracle and five dir cases pass; combined targeted CTest passes 1,793. Fresh
  47-definition audit, twelve compiled/executed exact-output documentation examples and
  four clangd hovers pass. Five complete report programs, five inspected captioned
  native figures including fitted cycle panels, both computed-result exports and
  byte-identical source/build/raw-export bundles pass without visual corrections. Site
  verifies 78 reports, 343 executed blocks and 274 source-matching figures. Original
  mepml/org removed after gates. Evidence: `build/mepml-conversion/diffie-hellman/`.
  Migration: 78/427 converted, 349 pending; next is diffusion_maps_iris_analysis.

- 2026-11-07: Converted the difference map into a canonical compiled checked execution
  kernel behind the original callable template, preserving by-value/move-only callback
  support, defaults, four-field result layout/formatter and strict step<tol stop
  semantics. A compiled pre-change probe on A={0,3},B={0,2},beta=.25,x0=2.25 returned
  solution 2/convergedtrue/step0 despite distance 1 to A. The corrected
  positive-iteration result is the actual B(f_A) cross-estimate, not B(x_after); the
  exact control decodes 0 while the fixed iterate remains 2.25 and direct B(x)=2 is
  invalid. Four callback orderings A(x),B(x),A(f_B),B(f_A) are retained per update
  without the old extra final B call; the zero-budget B(x0)/zero-residual
  sentinel/false-stop fallback remains explicit. Complete initial/control checks and
  dimension/finiteness checks after every callback precede indexing;
  reciprocal/relaxed/gap/update/iterate arithmetic is checked. Hypot norms retain
  finite 1.414214e200 and nonzero 1.414214e-200 steps where squaring would
  overflow/underflow. Owned actual eight-vector histories record initial/final
  iterates, cross-estimates, stable step/gap norms, numerical stop reason and
  callback/update counts without replay. A checked 1+8n history cost and online
  actual-record cap (default1m entries; zero disables optional cap) protect
  zero-dimensional metadata too, exclude documented baseline/transient storage, and
  reject before the next callbacks while preserving earlier external effects. Reusable
  affine-hyperplane and finite-set projectors validate complete shapes/finite inputs,
  scale normals/use checked long-double arithmetic, return owned outputs and keep
  earliest numerical nearest-point ties; no exact/full-range projection guarantee is
  claimed. The report preserves the original [2,2]/two-update/zero-residual example,
  distinguishes first good iterate from evaluated agreement, proves cross-projection
  decoding and the beta-one Douglas–Rachford identity, exposes alternating 3/2
  stagnation in the exact finite-set control, derives the independent two-line T_beta
  matrix and norm multiplier, and checks 48 actual iterate norms within 8.882e-16
  relative error. Small-beta disjoint sets give a numerical stop with gap 1; unit-beta
  disjoint sets drift with gap 1; slope-one beta 4 diverges despite convex intersecting
  sets. The legacy step residual is explicitly distinguished from Elser's cross-gap
  error measure. Ten ASan/UBSan tests pass, including 12,000 matrix-oracle updates
  across five slopes/eight betas/25 starts, 1,000 projector trials with independent
  integer nearest-point checks, original/decoded/nonconvex controls, actual callback
  inputs/order, move-only calls, ownership, strict-zero tolerance, zero
  budgets/dimensions, exact online caps/overflow, malformed/nonfinite/throwing
  callbacks, numerical range checks and stop-limit counterexamples. Twenty-four
  affected native cases plus five dir cases pass; combined targeted CTest passes 1,756.
  Fresh 50-definition audit, eleven compiled/executed exact-output examples and four
  clangd hovers pass. Four report programs rerun after a decoder-node label wrap was
  corrected; the changed figure was re-viewed and other inspected hashes remain
  identical. Four captioned inspected native figures, both computed-result exports and
  byte-identical source/build/raw-export bundles pass. Site verifies 77 reports, 338
  executed blocks and 269 matching figures. Original mepml/org removed after all gates.
  Evidence: `build/mepml-conversion/difference-map/`. Migration: 77/427 converted, 350
  pending; next is diffie_hellman_analysis.

- 2026-11-07: Converted byte-bigram Sørensen–Dice into a dedicated canonical packed-key
  consuming kernel, retaining the original string_metrics include, two-string scalar
  call, all five word-pair scores, identity/short-input early paths and literal
  byte/case/space/NUL behavior. Counter storage is now size_t, keys are packed unsigned
  16-bit pairs without temporary substring keys, and reserve requests are bounded by
  the 65,536-key alphabet rather than unbounded input length. A checked count-only API
  validates intersections before total overflow and computes exact integer total/union
  plus floating Dice and multiset Jaccard; its explicit both-empty convention lets
  string traces preserve different-short-string score zero. An owned actual trace
  records sorted union multiplicities and ascending right-offset remaining-count
  transitions from the executing kernel, with no replay; it scans equal/short inputs
  deliberately, and counts actual occurrence-level updates/lookups/matches/misses
  separately from hash probes, collisions, rehashing and trace bookkeeping. Complete
  A+B cap defaults to one million and precedes scanning/copy/allocation; zero disables
  only the optional cap. The report first exposes the shared ht in night/nacht and
  retains scores .25/.4/.8/1/0. Repeated aa/aaaa consumes once then misses twice,
  giving Dice .5 and Jaccard 1/3. Different strings abaca/acaba share the same full bag
  and score one; ab/abc/bc gives d=1−Dice direct 1 versus two-leg 2/3, disproving
  metricity. Literal case, space, NUL, canonically equivalent UTF-8 spellings and
  short/empty controls expose representation limits. A unary length-k/3k family matches
  both score closed forms exactly and records 4k−2 occurrence operations despite one
  distinct key; at k=4096 it has 4,095 shared, 16,382 operations and 12,287 events. A
  count-only four-billion total/three-billion union is checked without constructing
  huge strings, while all constants also fit a 32-bit size_t. Corrected historical
  attribution distinguishes Dice's ecological normalization from later lexical bigrams.
  Nine tests pass ASan/UBSan: 29,282 exhaustive ASCII/binary pairs use independent
  sorted token-list intersections/unions and unconsumed-token event scans; 2,000 seeded
  arbitrary-byte pairs add symmetry and bijective byte-renaming invariance. A
  131,072-byte input covers every 65,536 byte pair and unsigned sorted rows; a large
  unary case checks 100,000 matches and 200,000 misses.
  Original/short/nonmetric/encoding/cap/ownership cases, count boundaries, overflow and
  invalid-before-overflow precedence pass. Thirty-one affected native cases plus five
  dir cases pass; combined targeted CTest passes 1,732. Fresh 39-definition audit,
  eight compiled/executed exact-output examples and four clangd hovers pass. Four
  executed report programs, four inspected captioned native figures, both
  computed-result exports and byte-identical source/build/raw-export figure bundles
  pass. Final prose-only edits clarify bucket reserve requests and floating score
  rounding; all actual program/result hashes remain unchanged. A caption-count audit
  caught and repaired four missing Deutsch–Jozsa captions with its final program/figure
  hashes unchanged and both exports rechecked. Site verifies 76 reports, 334 blocks and
  265 source-matching figures. Original Dice mepml/org removed only after these gates.
  Evidence: `build/mepml-conversion/dice/`. Migration: 76/427 converted, 351 pending;
  next is difference_map_analysis.

- 2026-11-07: Converted Deutsch–Jozsa into a shared canonical compiled circuit while
  retaining the original two-field result layout, formatter, ascending callback order,
  gate sequence, raw probability, >0.5 decision and integer f(x)&1 convention. Checked
  qubit dimensions and empty callbacks reject before shifts/evaluation; capacity checks
  precede allocation. An owned actual trace records all three gate boundaries and
  observed low bits without replay, count-derived mean phase, exact
  constant/balanced/neither table categories, logical oracle applications, actual
  callbacks, Hadamards, amplitude-pair updates and sign flips. A complete default
  1048576-state cap is checked before allocation/calls; zero disables only that
  optional cap, with counter/capacity guards retained. Sequential classical promised
  querying records actual early or worst-case calls with O(1) extra storage and permits
  safe early exit on the largest representable domain. Callback exceptions propagate
  without undoing prior effects. The report retains all five original n=4 oracles and
  adds a native two-bit interference diagram, off-promise examples yielding either
  threshold verdict, a nonlinear balanced spectrum with four quarter-probability
  outcomes, all 17 one-count controls and explicit-log2 measured query/work
  comparisons. At n=16, one logical oracle application costs 65,536 simulator callbacks
  and 1,048,576 amplitude-pair updates; the top-bit classical control uses 32,769
  actual queries. Promise guarantees, floating residuals, deterministic exact versus
  randomised classical comparisons, n=0 conventions and ideal hardware versus
  state-vector simulation costs are explicit. Primary historical checks distinguish the
  original 1992 two-query algorithm from the 1998 single-query phase-kickback form.
  Nine ASan/UBSan tests pass, including all 65,814 Boolean tables through four bits
  against independent direct character sums and all 12,958 promised tables against
  sequential classical decisions; all affine parities/complements through eight bits,
  original outputs, shared Fourier cross-checks, low-bit semantics,
  nonlinear/off-promise tables, ordering, ownership, caps, dimensions and throwing
  callbacks pass. Thirty affected native cases plus five dir cases pass; the combined
  targeted suite passes 1,701. Fresh 35-definition documentation audit, eight
  compiled/executed exact-output examples and four clangd hovers pass. Four report
  programs, four inspected native figures, both computed-result exports and identical
  source/build/raw-export figure bytes pass. Final plot limits keep legends away from
  the largest measured counts; all four programs were rerun and the changed figure
  re-viewed, with unchanged figure hashes verified. Site verifies 75 reports,330 blocks
  and261 source-matching figures. Original mepml/org removed after these gates.
  Evidence: `build/mepml-conversion/deutsch-jozsa/`. Migration: 75/427 converted, 352
  pending; next is dice_coefficient_analysis.

- 2026-11-07: Converted directed depth-first search into a dedicated canonical
  explicit-continuation kernel, retaining the original four-argument call,
  BasicPathResult layout/first-entry count, search_more.hpp include, original input-edge
  order, global marking and immediate goal-entry stop. Heap {vertex,next-successor}
  frames preserve recursive descent/resume order without the process call-stack limit;
  successful paths stay active and no finish/unwind records are invented. Complete
  terminal/endpoint validation precedes adjacency allocation and all caps, including
  start=goal and invalid unreachable arcs. Owned actual discovery/scan/finish/goal
  events retain original edge identities and pre-descent target state;
  parent/edge/depth, preorder/genuine postorder and checked 2n discovery/finish
  sentinels distinguish unseen from entered-but-unfinished vertices. Complete
  source-component traversal uses the same kernel without requesting a goal or
  restarting unreachable components. Complete vertex/edge caps and online event caps
  throw without partial output; zero disables only optional limits. A reusable
  rooted-tree generator produces forward chain/binary/star controls with checked
  complete edge caps. The report retains the original two-node path [0,1], expanded 2,
  derives continuation/reachability/linear work and storage invariants, and
  distinguishes first DFS paths from shortest-hop BFS. A backtracked cycle branch
  finishes [3,1] before path [0,2,4,5,6]; actual entries 7, scans 7 and events 17 remain
  distinct. Moving direct edge 0->6 first gives two entries/one scan. A complete
  duplicate/self-loop/unreachable control has seven entries/finishes, ten scans and 24
  events, with genuine nested intervals and unseen v7. Tree events match 3n-1 regardless
  of branching; at n=512, chain/binary/star active peaks are 512/10/2, displayed with an
  explicitly labelled log2 depth scale. A 200,000-node ordinary path and 20,000-node
  full chain exercise deep entry and finishing without recursive DFS. Nine new tests
  pass ASan/UBSan: all 4,626 directed graphs within the stated loop scopes give 70,210
  goal comparisons and 17,954 full-source comparisons against a recursive
  original-edge-scan oracle. Separate transitive closure checks reachability;
  metadata/count comparisons and independent event/frame/target-state replay check
  actual recording. One thousand seeded ordered multigraphs add loops, duplicate arcs
  and reversed order. Original/trivial/unreachable/early-stop cases, complete
  intervals/DAG reverse postorder, ownership, exact caps, timestamp overflow,
  invalid-before-cap precedence and generator guards pass. Fifteen native
  DFS/search-neighbor cases plus five dir cases pass; combined CTest passes 1,671. All
  61 scoped definitions, ten compiled/executed exact-output documentation examples and
  four clangd hovers pass. Four report programs, four inspected native figures, both
  computed-result exports and all source/build/raw-export figure bytes pass; the fitted
  two-panel work/depth plot keeps chain, binary and star results distinct. Site verifies
  74 reports, 326 executed blocks and 257 source-matching figures. Original mepml/org
  removed after these gates. Evidence: `build/mepml-conversion/depth-first-search/`.
  Migration: 74/427 converted, 353 pending; next is deutsch_jozsa_analysis.

- 2026-11-07: Converted Delaunay triangulation to a dedicated exact-sign
  symbolic-boundary Bowyer-Watson core, preserving the one-vector call, Triangle
  layout/defaults/formatter, original four-face square fan and original seeded 47-face
  cloud. A compiled pre-change probe reproduced zero faces for (0,0),(1,0),(.5,1e-6);
  the new real seed plus hull ghosts returns the required face without finite
  super-triangle coordinates. Complete finite validation precedes caps/trivial results;
  exact duplicate/signed-zero coordinates retain earliest original ids, collinear inputs
  have no seed, strict cocircular ties retain existing faces, and finite output triples
  are CCW/canonical/sorted. New incircle_sign handles all finite binary64 via 48 signed
  products of four decoded dyadics, using two 264-word uint32 accumulators; 8,398
  required bits fit 8,448, without floating products, tolerance or long-double
  dependence. It complements existing exact orientation, with exactness scoped to stored
  coordinates. Actual owned removal/boundary/replacement events and checked
  seed/finite/ghost/cavity counters require no replay; original-record caps precede
  allocation, cumulative conflict caps precede whole scans and event-entry caps precede
  history retention. Local mesh analysis validates indices, CCW/nondegenerate/duplicate
  faces and edge multiplicity, caps full circle-query work, counts
  used/distinct/boundary sites and exact violations, and provides separately scaled
  approximate angles. It explicitly does not certify arbitrary global embedding/coverage
  or use approximate angles for topology. Core cloud generation keeps the original
  seed/distribution draw order and documents cross-library distribution variability. The
  report derives determinant/cavity and Euler edge-incidence relations,
  symbolic-boundary initialization and exact u(u-3) conflict scans, with conservative
  ordered-map complexity and output/history costs. It corrects the no-thin-triangles
  claim: the unchanged cloud has minimum angle 0.364812 degrees despite 1,269
  zero-violation exact circle checks; the only triangle at epsilon 1e-6 has minimum
  0.000114591559 degrees. Native scan counts reach 16,000 at u=128; exact fan limits are
  5 original records, 10 conflicts and 19 entries. Nine new tests pass ASan/UBSan: all
  512 subsets of a 3x3 grid in both orders and 250 seeded integer meshes have
  independent signed128 signs, full site use, exact hull/face area equality, noncrossing
  edges, boundary/Euler counts and complete empty-circle checks. Actual-event replay, 24
  cocircular square permutations, collinear/hull-edge insertion, duplicate
  identities/ownership, mixed largest-finite/subnormal topology, malformed mesh analysis
  and complete caps pass. Independent Fraction translated determinants agree for all
  10,000 controls: 6,000 random full-range and 4,000 cocircular/ULP cases. Forty-three
  native Delaunay/Voronoi/jump-and-walk/FEM/format/trace cases plus the Fraction oracle
  and five dir cases pass all 49 gates; combined CTest passes 1,656. All 69 scoped
  definitions, ten compiled/executed exact-output documentation examples and four clangd
  hovers pass. Four report programs, four inspected native figures, both computed-result
  exports and all source/build/raw-export figure bytes pass. A final prose-only cloud
  finding preserved executed C++ hashes and genuine run metadata. Site verifies 73
  reports, 322 executed blocks and 253 source-matching figures. Original mepml/org
  removed after these gates. Evidence: `build/mepml-conversion/delaunay/`. Migration:
  73/427 converted, 354 pending; next is depth_first_search_analysis.

- 2026-11-07: Converted de Casteljau into a dedicated canonical weighted-double
  evaluator, retaining the original signature, PlanarPoint two-double
  layout/defaults/formatter, interpolation.hpp include, empty (0,0) sentinel and finite
  polynomial extrapolation. Full finite-control/parameter validation precedes caps or
  output; nonfinite weighted results now throw even if a different algebraic arrangement
  could be finite. The original increasing-index (1-t)*left+t*right ordering remains.
  Owned actual rows and measured inner-loop blend counts replace the report-local
  recurrence, with checked cancelled full triangle geometry. Reusable subdivision
  captures the two boundary diagonals during one solve in O(m) storage, retains nominal
  degree and meeting-point copies, and checks a complete combined-output cap. Uniform
  sampling validates once, includes both requested parameter endpoints through
  std::lerp/direct endpoint assignment, handles equal/descending/tiny/extrapolated
  parameter intervals, and checks complete point/prospective-blend caps before
  allocation. Optional zero caps leave arithmetic/capacity checks enabled. The report
  preserves the original four controls, both endpoints, midpoint (2,1.875), ten triangle
  points and 101 curve samples. It derives intermediate Bernstein expansion,
  endpoints/convexity in exact arithmetic, both subdivision reparameterizations, exact
  absolute control-perturbation sensitivity, cancellation/relative-conditioning limits
  and scalar/trace/split/batch costs. A quarter split meets at (0.90625,1.265625); both
  native identity mappings have maximum coordinate residual 1.33226763e-15 over 101
  local parameters. A unit perturbation of the second y control follows 3t(1-t)^2 with
  residual 4.99600361e-16 and sampled maximum 0.444411. Degree 64 records 2080 actual
  point blends, 2145 stored points and 4160 coordinate blends. Nine tests pass
  ASan/UBSan: 300 integer nets at nine eighth-grid parameters check every actual
  triangle entry against an independent signed128 Bernstein numerator; 100 nets at five
  split values including endpoints and seventeen local values check both
  identities/diagonals; 1000 non-dyadic nets check long-double Bernstein, affine
  transformation, hull boxes and extrapolation. Original values, ownership,
  empty/singleton, complete caps, count overflow, finite-before-cap validation, explicit
  arithmetic overflow and sensitivity/cancellation controls pass. A test-only mixed-type
  auto declaration was separated before compilation; all final checks pass. Thirty-four
  native interpolation/de Boor/format/trace cases plus five dir cases pass; combined
  CTest passes 1,612. All 31 scoped definitions, nine compiled/executed exact-output
  examples and four clangd hovers pass. Four final report programs, four inspected
  native figures, both computed-result exports and all source/build/raw-export figure
  bytes pass. Two diagram labels initially crossed paths; they were moved into empty
  regions, all four programs rerun and both corrected diagrams viewed again; other
  viewed figure hashes match. Site verifies 72 reports, 318 executed blocks and 249
  source-matching figures. Original mepml/org removed after these gates. Evidence:
  `build/mepml-conversion/de-casteljau/`. Migration: 72/427 converted, 355 pending; next
  is delaunay_analysis.

- 2026-11-07: Converted DDA line drawing into a dedicated canonical double-accumulation
  kernel while retaining the four-argument legacy call, DDAPoint alias and both original
  rasters. Widened endpoint differences, unsigned sample/count loops and checked
  rounding cover full 32-bit geometry and reject out-of-int accumulated results before
  conversion. Actual owned traces retain both accumulators before rounding,
  exact-rational reference centres, approximate drift/coordinate-error diagnostics and
  precise coordinate add/round counts without replay, endpoint snapping, clamping or tie
  repair. The ordinary path omits all trace/reference arithmetic. Exact random-access
  and full-raster comparisons use portable unsigned64 products and quotient/remainder
  with global signed halfway-away-from-zero rounding, avoiding signed overflow and
  floating selection. Complete point caps check before allocation; zero disables only an
  optional cap. The report derives major-axis sampling, the exact half-pixel bound,
  8-connectivity and reversal, and the floating increment-plus-addition error identity
  and half-pixel-plus-drift budget. Output storage is correctly O(N+1), and measured
  arithmetic is two coordinate additions per transition and two rounds per sample.
  Native equal-scale figures open with five binary-exact samples and preserve original
  shallow 15 and steep 11 sequences. A 12-step midpoint selects (6,0) against exact
  (6,1); a 10-step height 7 reversal selects (5,4) versus(5,3); the translated
  10,000-step control has six changed samples and max coordinate error 0.500500. Exact
  ties are distinguished from Bresenham's directed policy and from integer translation
  across zero. Full-span count 4,294,967,296 and middle exact samples (-1,0)/(0,1)
  allocate no raster. Nine new tests pass ASan/UBSan: all 14,641 signed-grid endpoint
  quadruples compare with an independent signed128 weighted-endpoint oracle and check
  replay, diagnostics, exact endpoints/reversal/8-connectivity; 1,000 seeded short
  lines, 25,000 full-width random-access samples, 2,500 extreme-grid/index combinations,
  coincidence/ownership, complete caps, invalid indices and signed ties pass. An actual
  five-million-step near-INT_MAX overshoot throws before unsafe conversion. The initial
  12-step reversal-witness assertion was corrected to the observed 10-step height 7
  witness; no core defect was hidden, and all final tests pass. Fifteen native
  graphics/trace cases plus five dir cases pass; combined CTest passes 1,578. All 43
  scoped definitions, eight compiled/executed exact-output examples and four clangd
  hovers pass. Four report programs, four inspected native figures, both computed-result
  exports and all source/build/raw-export figure bytes pass; the website verifies 71
  reports, 314 executed blocks and 245 source-matching figures. Original mepml/org
  removed after these gates. Evidence: `build/mepml-conversion/dda-line/`. Migration:
  71/427 converted, 356 pending; next is de_casteljau_analysis.

- 2026-11-07: Converted Davis–Putnam into a canonical checked sparse ordered-clause
  kernel while retaining the legacy Boolean call shape and valid propositional verdicts.
  Complete literal/bound validation now precedes trivial verdicts and resource caps,
  rejects zero/INT_MIN/out-of-range ids, avoids signed loop/negation overflow and
  allocates no dense num_vars array. Canonical literal/clause deduplication and
  tautology removal feed deterministic first-unit, smallest-signed-pure and
  smallest-variable resolution rules. Complete owned initial/after snapshots,
  per-transition parent/tautology/duplicate counts and checked aggregate
  rules/pairs/peak counts come from the actual kernel without replay. Separate
  distinct-clause, cumulative parent-pair and whole-history entry caps throw without a
  partial Boolean verdict; zero disables only optional caps. A reusable grid CNF checks
  ids/output sizes before allocation and exposes first-state 2k+2 -> k²+2 with k² pairs.
  The report derives existential projection, unit/pure correctness, finite termination
  and ordered-set costs; it narrows the old general-first-order decision claim to finite
  propositional SAT and distinguishes a decision/history from assignment or standalone
  proof certificates. Original four formulas remain, with actual resolution work
  replacing input-clause bars. The four-clause contradiction records 4 pairs, 2
  discarded tautologies, opposite units and a final empty clause. At grid order 8,
  initial 18 becomes 66, later peak 263 and total 8200 pairs; first-step theory and
  whole-run measures are explicitly separate. History accounting 13+5+2 exactly meets
  cap 20 and rejects 19. Eight new tests pass ASan/UBSan, independently enumerating all
  512 canonical two-variable formulas, 3,304 three-variable formulas with at most three
  clauses and 1,500 seeded formulas through seven variables; every step is checked under
  both pivot values for every assignment, with independent resolvent/counter
  reconstruction. Ownership, empty/duplicate/tautological inputs, sparse INT_MAX,
  invalid-before-trivial/cap precedence and complete cap/generator guards pass. Thirteen
  native trace/optimization cases plus five dir cases pass; combined CTest passes 1,563.
  All 41 scoped definitions, seven compiled/executed exact-output documentation examples
  and four clangd hovers pass. Four report programs, four inspected native figures, both
  computed-result HTML/Markdown exports and all source/build/raw-export figure bytes
  pass. The site verifies 70 reports, 310 executed blocks and 241 source-matching
  figures. Original mepml/org removed only after these gates. Evidence:
  `build/mepml-conversion/davis-putnam/`. Migration: 70/427 converted, 357 pending; next
  is dda_line_analysis.

- 2026-11-07: Converted unrestricted Damerau–Levenshtein distance to a dedicated
  canonical Lowrance–Wagner kernel while preserving the scalar name/signature,
  string_metrics include and all five original Levenshtein/DL results. Checked
  rectangular/border dimensions and n+m arithmetic replace unchecked
  allocation/candidate sums; safe clipping at the finite marker preserves every
  unit-cost interior minimum. Owned complete traces retain the actual sentinel table,
  all prefix-cell candidates, pre-update last-occurrence coordinates, deterministic
  first-minimum choices and checked cell/byte/match/row/bridge counters without replay
  or string snapshots. A reusable three-row OSA comparison computes the restricted
  distance rather than assuming its value. Checked prefix access and local traceback
  validation expand source-interior deletions, one adjacent swap and target-interior
  insertions into owned current-position byte instructions; checked by-value replay
  verifies enum/payload/position/expected-byte semantics. Complete border-inclusive
  table and output-edit caps reject whole requests before their respective allocations;
  zero disables only optional caps. Byte/NUL/high-byte/case-sensitive semantics and the
  absence of UTF-8 decoding/normalization are explicit. The report derives
  operation-graph metric bounds, the Lowrance–Wagner last-occurrence bridge under its
  unit-cost condition, exact integer clipping, prefix/grid/work costs and
  script-position invariants. The native CA->AC->ABC witness costs two, while computed
  OSA gives three and violates the triangle inequality against its two unit legs.
  Original unsupported frequency ranking and overly narrow strict-improvement claims are
  replaced with scoped historical motivation and the unrestricted edit definition. All
  five original pairs are retained, with nested markers preserving coincident Lev/OSA/DL
  values; actual prefix candidates are (3,3,3,2), and rectangular controls separate nm
  interior updates from complete borders. Nine new cases pass AddressSanitizer and
  UBSan: independent shortest-edit-graph BFS checks all 14,641 pairs of 121 ternary
  strings through length four, with bounded-depth/length completeness and
  outside-alphabet projection arguments; 1,771,561 finite-corpus metric triangles,
  literal script replay, full-table OSA controls, 1,000 seeded repeated/full-byte pairs,
  all 256 byte swaps, NUL/UTF-8 cases, last-occurrence/candidate/work reconstruction,
  overflow/complete caps, ownership and malformed records/instructions pass. All 64
  scoped definitions, thirteen compiled/executed documentation examples and four clangd
  hovers pass. Thirty-one metric/Hamming cases plus five dir cases pass; combined CTest
  passes 1,550 cases. Four report programs, four inspected native figures,
  computed-result HTML/Markdown exports and byte-identical source/build/export figure
  bundles pass. A report-only prefix-program name collision was corrected and all four
  programs rerun successfully; final source/exports contain no compiler diagnostics.
  Site verifies 69 reports, 306 executed blocks and 237 source-matching figures.
  Original mepml/org removed after all gates. Evidence:
  `build/mepml-conversion/damerau-levenshtein/`.
  Migration: 69/427 converted, 358 pending; next is davis_putnam_analysis.

- 2026-11-07: Converted DAG longest paths to a dedicated canonical
  FIFO-Kahn/maximum-relaxation kernel while retaining the original signature,
  LongestPathResult two-vector layout/formatter, routing_batch include and
  source/unreachable predecessor sentinel n. Complete endpoint/finite-weight/source
  checks precede allocation and optional trace n/m caps; every graph component is
  checked for cycles before numerical relaxation. All reachable double candidates must
  be finite, including losing extremely negative routes, replacing silent
  infinity/unreachable confusion with checked overflow; unreachable edges are validated
  but never added. Owned actual traces retain original directed/parallel edges,
  whole-graph topological order, exact predecessor-edge choices, every reachable-edge
  attempt and checked pop/edge/reach/candidate/improvement counts without replay. Strict
  stored ties retain the first witness under the documented FIFO/input-order policy.
  Safe local path recovery checks shapes, markers, visited endpoints and stored
  recurrences, rejects predecessor cycles by a length guard and distinguishes singleton
  source paths from empty unreachable records without claiming to certify arbitrary
  edited graphs globally. A reusable finite-weight forward-band generator uses cancelled
  checked edge-count factors and complete preallocation caps. The report preserves the
  original length-nine diamond and recovers vertices [0,2,3]/edges [1,3], derives
  topological induction and the Hamiltonian-path hardness connection, separates
  negative/unreachable/tied/parallel cases, and derives componentwise-earliest
  precedence-only event times (0,3,2,10,11). Chain/dense workloads measure V+E+E_R
  visits rather than claiming linear work in vertices for dense graphs. An explicit 2^53
  control shows distinct exact route totals collapsing into a stored tie, with the
  exact-real recurrence and ordinary double limitations stated separately. Nine new
  cases pass AddressSanitizer and UBSan: 202,013 ordered-DAG/source cases through six
  vertices match independent enumeration of all source paths; 1,000 seeded relabelled
  multigraphs, actual event replay/topological checks, exact edge witnesses,
  signed/unreachable/tied/source cases, global cycles, full domain/complete-cap checks,
  both overflow signs, rounded ties, all small generator shapes/count guards,
  ownership/malformed local chains and 100,000 isolated vertices pass. All 40 scoped
  definitions, nine compiled/executed documentation examples and four clangd hovers
  pass. Twenty-six routing/format cases plus five dir cases pass; combined CTest passes
  1,519 cases. Four report programs, four inspected native SVGs, computed-result
  HTML/Markdown exports and byte-identical source/build/export figure bundles pass. Site
  verifies 68 reports, 302 executed blocks and 233 source-matching figures. Original
  mepml/org removed after all gates. Evidence:
  `build/mepml-conversion/dag-longest-path/`.
  Migration: 68/427 converted, 359 pending; next is damerau_levenshtein_analysis.

- 2026-11-07: Converted CYK to a canonical compressed-id byte-membership kernel while
  retaining the inline recognizer, original CNFGrammar three-field layout/formatter and
  empty-input rejection. All integer ids, including negative/sparse signed extremes,
  remain valid; duplicate rules are recognition-idempotent, terminals are bytes
  including NUL, and epsilon/unit-rule/general-CFG conversion is explicitly outside the
  representation. Complete owned analyses retain every positive span, sorted distinct
  memberships, actual first split/rule choices and checked
  terminal/split/skip/rule/match/new-state counts without replay. Dense geometry uses
  factor-cancelled checked triangular/cubic counts and complete potential Q*m/K*r caps
  before input/chart/choice allocation; grammar preparation precedes those caps, zero
  disables only optional limits, and actual skips do not reduce potential budgets.
  Checked constant-time borrowed-cell access and iterative preorder first-tree recovery
  validate visited local structures and root consistency without claiming to certify
  arbitrary edited records against an absent original grammar. Complete parse-node caps
  cover the exact 2n-1 tree size. The report derives recurrence soundness/completeness,
  the original a^k b^k language and nonregularity, span/split identities,
  grammar-dependent memory and complete time costs including duplicate terminals and the
  split loop when r=0. It retains all eight original strings, explains all ten aabb
  cells/seven memberships/18 rule tests and reconstructs its seven-node first parse.
  S->SS|a separates five possible aaaa trees from one returned witness, ten chart
  memberships and matching-rule events; duplicate rules change work but not the
  language. Nine new cases pass AddressSanitizer and UBSan: all 8,191 a/b strings
  through length twelve match the proven original language; an independent fixed-point
  language-concatenation oracle checks 100 seeded grammars and 12,700 inputs; recovered
  trees match original productions and bytes. All 256 byte terminals, NUL/UTF-8 byte
  fixtures, signed extremes, duplicate/first-rule choices, exact cubic work, empty-child
  skips, factor-cancelled large geometry, overflow/potential/node budgets,
  ownership/borrows, malformed visited witnesses and a 200-byte right-branching tree
  pass. All 76 scoped definitions, twelve compiled/executed documentation examples and
  four clangd hovers pass. Twenty-four parsing/hash/format cases plus five dir cases
  pass; combined CTest passes 1,493 cases. Four report programs, four inspected native
  figures, computed-result HTML/Markdown exports and byte-identical source/build/export
  figure bundles pass. Giving the original bar chart an explicit title and short legend
  fixes initial clipping; the other final SVG hashes match inspected figures. Site
  verifies 67 reports, 298 executed blocks and 229 source-matching figures. Original
  mepml/org removed after all gates. Evidence: `build/mepml-conversion/cyk/`.
  Migration: 67/427 converted, 360 pending; next is dag_longest_path_analysis.

- 2026-11-07: Converted comparison-based cycle sort into a dedicated canonical
  rank/placement kernel while retaining its by-value signature and distribution_sorts
  include. Output-only, work-only and traced calls share actual placements without a
  second instrumented algorithm. Checked native counts distinguish suffix rank tests,
  duplicate equality tests, successful skips, logical value-cell writes, rank scans,
  starts and nontrivial rotations. Complete owned traces carry real held/displaced keys
  and input identities, target-block starts, destinations and cycle closure, with at
  most n records and no array snapshots. The complete element cap defaults to 4096 and
  rejects before copying, identity/history allocation or sorting; zero disables only
  that optional cap. An independent std::sort-based core reference reports the Hamming
  minimum for key sorting, outside measured work. The report proves that every initially
  mismatching key cell is written once and every initially correct cell is preserved,
  derives exact comparison/scan formulas and the sharp B(n) through B(n)+(n-1)^2
  rank-test bounds, and distinguishes quadratic repeated-rank sorting from Haddon's
  conditional linear variants. Actual duplicate identities (2a,2b,1c) become (1c,2b,2a),
  exposing instability and the difference between the two-write key minimum and a
  three-position stable-record target. Logical working-cell counts explicitly exclude
  by-value copies, held items, identity records, allocation and physical device
  erase/write amplification. The original seven values retain seven writes, 39 rank
  tests and three cycles; all forty seed-three sizes retain the n=200 endpoint of 199
  writes and 57,301 rank tests. Separate native panels make the linear write curve
  visible; a model based on the existing full-width bubble kernel explains its 39,800
  reversal key tests, 19,900 exchanges and 39,800 model writes without inventing a
  crossover. Nine new cases pass AddressSanitizer and UBSan: 88,573 ternary sequences
  through length ten and 46,234 permutations through length eight replay against
  independent sorted targets, unique destinations, held/displaced identities, exact work
  reconstructions and minimum-write checks; 1,000 seeded mixed/full-range inputs, signed
  extremes, exact best/worst/reverse formulas, empty/singleton/ownership/complete-cap
  cases and the original experiment pass. All 41 scoped definitions, seven
  compiled/executed documentation examples and four clangd hovers pass. Thirteen sorting
  cases plus five dir cases pass; combined CTest passes 1,469 cases. Four report
  programs, four inspected native SVGs, computed-result HTML/Markdown exports and
  byte-identical source/build/export figure bundles pass. Site verifies 66 reports, 294
  executed blocks and 225 source-matching figures. Original mepml/org removed after all
  gates. Evidence: `build/mepml-conversion/cycle-sort/`.
  Migration: 66/427 converted, 361 pending; next is cyk_analysis.

- 2026-11-07: Converted Cuthill–McKee to a dedicated checked canonical queue kernel while retaining
  the original signature/default and sparse_linalg include. Raw endpoints and size sums
  are checked before optional trace caps; loops and duplicate entries are ignored, off-
  diagonal symmetry is required after normalization, and degree/id ties plus component
  starts are deterministic. A once-sorted root list avoids quadratic isolated-component
  searches. Owned complete traces record actual discoveries, parents, component
  levels/roots, inverse positions and normalized scan/enqueue work without replay.
  Reusable checked permutation/inversion, band-pattern generation, lower/upper profile
  metrics and sorted structural coordinates replace report-local algorithms; generator
  counts are constant-time checked before allocation. The report derives permutation/SPD
  preservation, BFS level bounds, exact reversal invariance of bandwidth and profile
  exchange, and sorting-inclusive complexity. It preserves the original seed-seven
  forty-row 32-to-2 fixture, explicitly separates portable structural checks from
  std::shuffle-specific ids, and proves a star counterexample where CM worsens optimal
  bandwidth 2 to 3. Separate native panels distinguish bandwidth, skyline allocation and
  exact symbolic fill, including the star's three/one/zero added edges. Nine new cases
  pass AddressSanitizer and UBSan: all 33,868 simple graphs through six vertices match
  an independent dense/linear-choice traversal; all 124,470 graph/order cases through
  five vertices match dense bandwidth/envelope inspection and reversal identities; 1,000
  irregular normalization controls, 500 coordinate/permutation round trips, extreme
  generator counts, empty/singleton/ownership/invalid/budget cases and 100,000 isolated
  vertices pass. All 40 scoped definitions, eleven compiled/executed documentation
  examples and four clangd hovers pass. Twenty-two sparse/format cases plus five dir
  cases pass; combined CTest passes 1,456 cases. Four report programs, four inspected
  native figures, computed-result HTML/Markdown exports and adjacent byte-identical
  source/build/export bundles pass. Additional work-plot headroom prevents the legend
  obscuring its largest point; other final rasters rerender byte-identically to
  inspected copies. Site verifies 65 reports, 290 executed blocks and 221 source-
  matching figures. Original mepml/org removed after all gates. Evidence: `build/mepml-
  conversion/cuthill-mckee/`.
  Migration: 65/427 converted, 362 pending; next is cycle_sort_analysis.

- 2026-11-07: Converted natural cubic spline interpolation into a dedicated canonical
  kernel while retaining the original five-vector CubicSpline layout, names/include and
  zero/constant/line conventions. Full finite/shape/increasing-knot checks and checked
  fit/evaluation arithmetic replace unsafe malformed-record behavior. Legacy mutable
  scalar evaluation explicitly validates in O(n); an owned PreparedCubicSpline
  validates/copies once for O(log n) scalar value/derivative queries and complete
  ordered batches, with right-piece third-derivative and endpoint-cubic extrapolation
  semantics. One fit kernel supplies actual widths, secants, moments, original
  tridiagonal entries/RHS, multipliers, pivots, eliminated RHS and logical work without
  replay. Complete trace caps follow full input validation. Reusable native function-
  error censuses retain actual predictions/references, first sampled maxima and scaled
  RMS, validate finite queries/callback presence before complete
  caps/allocation/invocation, and distinguish sampled maxima from certified suprema. The
  report derives the moment system, SPD uniqueness and bending-energy minimum over the
  knot span; preserves the original Runge nodes and measured 0.02197383 versus
  1.91564305 errors (ratio 87.178404); and restores the formerly clipped polynomial
  peaks. Compatible sine and mismatched quadratic endpoint controls derive/show O(h^4)
  versus boundary-layer O(h^2) behavior. Explicit monotone-data overshoot, negative
  interpolation of nonnegative samples and curved natural extrapolation replace blanket
  smoothness/shape/stability claims. All 51 scoped definitions, eleven compiled/executed
  documentation examples and four clangd hovers pass. Eleven new cases pass
  AddressSanitizer and UBSan, covering exact hand coefficients/moments, 1,000 irregular-
  grid factor-replay/continuity/linearity controls, original interpolation/format
  behavior, all derivative-side conventions,
  malformed/finite/budget/ownership/degenerate cases, both refinement families and
  stable large-error RMS. A permanent optional Fraction oracle solves the full
  value/C1/C2/natural coefficient constraints by exact dense elimination, independently
  of the moment/Thomas formulation: 102 fits and 4,320 value/derivative controls, plus
  coefficients/moments, pass; its native checker is also sanitized. Twenty-five
  interpolation/format/oracle cases plus five dir cases pass; combined CTest passes
  1,434 cases. Four report programs, four inspected native SVGs and both exports pass.
  Shortening the convergence legend fixes initial right-edge clipping; the other final
  SVGs rerender byte-identically to inspected rasters. Final hashes and adjacent build
  copies match. Site verifies 64 reports, 286 executed blocks and 217 matching figures.
  Original mepml/org removed after all gates. Evidence: `build/mepml-conversion/cubic-
  spline/`. Migration: 64/427 converted, 363 pending; next is cuthill_mckee_analysis.

- 2026-11-07: Converted Cristian's clock observation model into a dedicated canonical
  core while preserving the original inline signatures/default minimum/include and
  worked estimate 1020/radius 15/all ten sweep rows. Owned native observations
  explicitly timestamp reply departure, remove known server residence, retain nominal
  relative/absolute receipt intervals and model controlled path asymmetry. Complete
  probe selection validates every observation and derived output before an optional
  cap/allocation, keeps all analyses and selects the first smallest stored radius;
  estimates remain at their own receipt instants with no aging or OS-clock mutation.
  Contradictory/nonfinite/negative parameters reject instead of clamping impossible
  minima to false zero. Half/radius/error collapse, hidden network duration, generated
  interval loss and unrepresentable arithmetic are checked. Subtraction-based
  feasibility avoids doubled-minimum overflow and wider-long-double assumptions; an
  additional 8-byte-long-double guard control passes exact, infeasible and positive-
  radius-collapse subnormal cases. Absolute epoch additions share double precision to
  fix a zero-width double-rounding inconsistency, covered by a nextafter regression. The
  report derives the minimax midpoint, separates symmetry from the conditional bound,
  identifies arrival/departure stamp corrections and known-processing bias, preserves
  the original RTT study, and demonstrates that a faster probe need not have smaller
  actual error. Coarse epochs can collapse stored absolute endpoints while relative
  delays retain width; these are ordinary rounded model values, not outward enclosures,
  drift/error propagation, statistical confidence intervals or a full NTP
  implementation. All 56 scoped definitions, nine compiled/executed new or changed
  documentation examples and four clangd hovers pass. Nine new cases pass
  AddressSanitizer and UBSan, including 27,004 integer exchanges, 10,000 exact binary-
  fraction controls, attained extremes, processing cancellation, selection
  ties/ownership, complete caps and full validation, zero/coarse epochs,
  underflow/overflow and both rounding regressions. Sixteen Cristian/distributed cases
  plus five dir cases pass; combined CTest passes 1,409 cases. Four executed report
  programs, three inspected native SVGs and both exports pass; final SVG hashes match
  inspected copies and adjacent build outputs. Site verifies 63 reports, 282 executed
  blocks and 213 matching figures. Original mepml/org removed after all gates. Evidence:
  `build/mepml-conversion/cristian/`. Migration: 63/427 converted, 364 pending; next is
  cubic_spline_analysis.

- 2026-11-07: Converted CRC-32 into a dedicated canonical immutable-table byte kernel
  shared by the existing inline crc32() and complete actual byte traces; the
  checksums.hpp include path, table name and exact checksum convention remain available.
  Owned native bit toggles fully validate addresses and cancel duplicates in pairs. A
  complete bounded single-bit census reuses one toggled/restored scratch message and
  checks exact n*(8n+1) byte-fold work before allocation/evaluation. Binary/embedded-
  zero/empty semantics, ownership, no-partial-result caps and fixed-length affine
  syndrome meaning are explicit. The report retains standard CBF43926 and original
  B74574DE/152 flips/15.460526 mean, distinguishes raw and finalized registers, and
  diagrams actual first-byte lookup constants. It derives payload-independent syndrome
  composition and the <=32-bit short-burst guarantee, checks all 11,476 original double-
  bit errors via unique syndromes, and replaces false universal low-weight/odd-
  error/avalanche-collision claims with an explicit 33-span, fifteen-flip zero-syndrome
  generator error. All 41 scoped definitions, eight compiled new/changed documentation
  examples and three clangd hovers pass. Nine new cases pass AddressSanitizer and UBSan:
  256 one-byte and 65,536 two-byte messages versus an independent bit loop; 1,000 random
  binary inputs with existing codec-streaming parity; 300 payload/composition controls;
  every original single/double error; 110,591 exhaustive width-1..12 bursts plus 2,000
  sampled width-13..32 bursts; all 120 fitting offsets of the odd-weight collision;
  exact/one-below/disabled budgets, duplicate/invalid bits, empty results and owned
  copies. Thirty-two checksum/Adler/codec cases plus five dir cases pass; combined CTest
  passes 1,393 cases. Three report programs, three inspected native SVGs and both
  exports pass, with final figure hashes matching inspected rasters and adjacent build
  files. A root-disk-full compilation/cache interruption was resolved using project
  TMPDIR, narrowly removing ten verified unreferenced datamunge Nix source snapshots
  through liveness checks, preserving task PCH symlinks and saving a reusable tools-only
  dev-shell profile; the final run has no failures. Site verifies 62 reports, 278
  executed blocks and 210 matching figures. Original mepml/org removed after all gates.
  Evidence: `build/mepml-conversion/crc32/`. Migration: 62/427 converted, 365 pending;
  next is cristian_analysis.

- 2026-11-07: Converted Crank–Nicolson heat evolution into a dedicated canonical core,
  retaining the original callable signature and pde_solvers.hpp include path. One
  validated kernel samples each interior point once, enforces exact zero endpoints,
  reuses one actual Thomas factorization and optionally retains complete actual
  RHS/profile states, factors, solve counts, scaled RMS and componentwise backward
  errors without prefix replay. Scalar domains, finite sampled values, representable
  spacing/interior coordinates, ratios, coefficients, final time and arithmetic are
  checked; valid zero diffusion/time and zero-step conventions are explicit. Complete
  history scalar caps are checked before allocation or callback invocation. Native sine-
  mode controls independently separate continuous, semidiscrete and ideal CN evolution;
  finite modal evaluations are not arbitrary-precision certificates. The report retains
  the sixty-interior-point experiment, fixes the sample/reference index mismatch, opens
  with the hand-solvable alternating 1,-0.6,0.36 case, and distinguishes ideal L2
  stability from positivity, maximum-norm guarantees, stiff damping and finite
  arithmetic. Separate space/time studies give ratios approaching four and expose
  cancellation in combined PDE error; solve residuals are not treated as PDE accuracy.
  All 50 new-module definitions, eight compiled/executed documentation examples and
  three clangd hovers pass. Eleven new cases pass AddressSanitizer and UBSan, including
  500 independent dense long-double Cholesky evolutions, 116 sine-mode controls, actual
  RHS/factor/state checks, both refinement orders, original data, zero-
  step/diffusion/time behavior, callback order/exceptions, exact/one-below/disabled
  history caps, invalid/unrepresentable arithmetic, near-DBL_MAX scaled-RMS identity
  profiles and owned storage. Fifteen heat/PDE plus five dir cases pass; combined CTest
  passes 1,361 cases. Four executed report programs, four inspected native SVGs and both
  exports pass; explicit scientific ticks fix initially rounded-zero temporal axes and
  single legend entries retain visible endpoints. Final SVG hashes match inspected
  copies and adjacent build outputs. Site verifies 61 reports, 275 executed blocks and
  207 matching figures. Original mepml/org removed after all gates. Evidence:
  `build/mepml-conversion/crank-nicolson/`. Migration: 61/427 converted, 366 pending;
  next is crc32_analysis.

- 2026-11-07: Converted counting sort with one canonical histogram/prefix/reverse-
  placement kernel. Legacy overloads retain their hard billion-bucket ceiling and valid-
  empty convention; new owned analysis exposes stable original indices and actual bucket
  boundaries, and a complete trace records each original key, input index, cursor and
  assigned slot without replay. Full bounds/key validation precedes allocation and
  optional bucket/event caps. Unsigned checked range arithmetic supports narrow
  intervals at either signed-64-bit endpoint. Diagnostic defaults bound buckets to one
  million and events to one hundred thousand; zero disables only the optional cap, with
  allocation failures still possible. The report preserves the original eight integers
  and named records, replaces the packed-key pseudo-stability check with genuine equal-
  key identities, derives block correctness/stability, and separates logical n,k,n work
  from initialization, copies, timings and worst-case comparison bounds. All 34 scoped
  definitions, nine compiled/executed documentation examples and three clangd hovers
  pass. Eighteen native cases pass address/undefined sanitizers, including 29,524
  exhaustive ternary sequences through length nine and 1,000 random duplicate-heavy
  identity/event-replay controls. Endpoint ranges, complete exact/one-below caps, full
  invalid input, empty/singleton/owned results and legacy exception preservation pass.
  Eighteen counting-sort plus five dir cases pass; combined CTest passes 1,346 cases.
  Four report programs, four inspected SVGs and both exports pass, with final figure
  hashes matching inspected rasters and adjacent build outputs. Site verifies 60
  reports, 271 executed blocks and 203 matching figures. Original mepml/org removed
  after all gates. Evidence: `build/mepml-conversion/counting-sort/`. Migration: 60/427
  converted, 367 pending; next is crank_nicolson_analysis.

- 2026-11-07: Converted circular CORDIC. One canonical floating recurrence preserves the
  original SinCos two-double layout/default count and actual raw/gain update order while
  adding checked reduction metadata, immutable prepared stage constants, owned actual
  before/after traces and bounded error sweeps. Original atan/sqrt expressions prepare
  the shared table once; per-call work explicitly includes floating multiplications,
  compensation products and extra trace normalizations. The supported domain is finite
  theta and 0..1075 stages, covering every nonzero binary64 scale; zero stages is an
  unrotated approximation and zero residual does not stop the requested steps. Complete
  trace/evaluation caps follow full input validation; sweep rows retain first
  worst-angle ties and explicit absent extrema. The retained
  remainder(theta,2*pi_double) policy is explicitly not high-precision reduction: huge
  original-angle phase errors can survive a tiny residual and near-unit norm. The report
  corrects amplification versus inverse compensation, derives the ideal finite-stage
  bound only for exact reduced-target arithmetic, and distinguishes a floating model
  from multiplier-free hardware. Original four angles and all twelve normalized states
  are preserved; the actual first raw states (1,1), (0.5,1.5), (0.875,1.375) are
  hand-checkable. A 2,001-angle sweep separates stage error from the binary64 floor. All
  87 scoped definitions, fifteen compiled/executed new or changed documentation examples
  and four hovers pass. Twelve new cases pass AddressSanitizer and UBSan. Exact
  output-value parity with the former kernel covers 1,000 seeded angles at counts
  0,1,2,3,12,40,44,64,1075; original trajectory states match independently. Ten thousand
  moderate angles, every nonzero stage scale, fold boundaries, gain/phase identities,
  full invalid-input checks, exact/one-below budgets, sweep empty/tie semantics and
  owned storage pass. A permanent optional CTest oracle uses 430-digit Decimal working
  arithmetic with independent Machin pi/Taylor sine/cosine for 126 controls, including
  subnormals and largest finite doubles; its native checker also passes
  address/undefined sanitizers. Maximum folded-model component error is 1.12767e-13 and
  moderate original-angle error 1.13135e-13, while original-angle error including huge
  inputs reaches 1.83609. Optional standard-library oracles require Python 3.9 and the
  focused runner builds its checker dependency. Thirty CORDIC/pi/format/oracle cases
  plus five dir cases pass; combined CTest passes 1,328 cases. Five report programs,
  four inspected equal-scale figures and both exports pass. Final SVG hashes match
  inspected copies. Site verifies 59 reports, 267 executed blocks and 199 matching
  figures. Original mepml/org removed after all gates. Evidence:
  `build/mepml-conversion/cordic/`. Migration: 59/427 converted, 368 pending; next is
  counting_sort_analysis. Previously recorded unrelated audit, CAD-policy and optional
  C++17 limits remain outside this batch.

- 2026-11-07: Converted convex hull. One canonical Andrew monotone-chain kernel
  preserves the legacy by-value signature and strict CCW/lexicographic-start convention
  while adding earliest original identities, checked actual work and compact actual
  test/pop/push edits without prefix reruns or quadratic stack snapshots. Complete
  finite validation precedes internal indexed allocation/sorting and optional
  point/event caps; signed-zero duplicates retain the earliest matching representation.
  Empty, singleton, two-point, collinear and edge-interior conventions are explicit. A
  reusable exact orientation_sign decodes finite IEEE binary64 values and accumulates
  six dyadic coordinate products in two fixed 4224-bit integer arrays; the proved sum
  bound is at most 4199 bits. No floating products, long-double assumptions or tolerance
  determine its sign. Cancellation determinant one, subnormal triangles and huge
  collinear configurations are classified correctly; the cancellation triangle retains
  all three vertices. The original Point2D layout/equality and double-valued primitives
  remain unchanged in behavior, with their limitations fully documented. Native
  unsnapped mt19937 normal sampling preserves the original full forty-point stream and
  former hull on this toolchain, including eight vertices; permanent tests compare full
  original stream/hull parity rather than assuming a universal normal mapping. The
  seven-point square retains its four original corners. The report derives both envelope
  invariants, lower-dimensional conventions, exact push/pop identity, linear scan bound,
  capacity proof and comparison-model scope. All 80 scoped definitions, nineteen
  compiled/executed documentation examples and four hovers pass. Fourteen new native
  cases pass AddressSanitizer and UBSan. Independent integer supporting-edge enumeration
  checks 512 grid subsets, 6,561 ordered four-record tuples and 1,000 seeded clouds with
  actual edit replay. All 5,040 seven-point permutations retain earliest identities;
  10,000 random triple permutation/similarity controls, full-range/duplicate/degeneracy
  checks, exact/one-below caps, original stream parity and owned states pass. A
  permanent optional CTest Python Fraction oracle validates 20,000 raw-binary64 and
  collinear/one-ULP triples; its native checker also passes address/undefined
  sanitizers. The focused runner builds that oracle dependency when Python is available.
  Thirty-five geometry/oracle cases plus five dir cases pass; combined CTest passes
  1,298 cases. Five report programs, four inspected figures and both exports pass. Final
  SVG hashes match inspected copies. The final proof-only wording correction passes
  fresh documentation/hover and exports without changing executable code. Site verifies
  58 reports, 262 executed blocks and 195 matching figures. Original mepml/org removed
  after all gates. Evidence: `build/mepml-conversion/convex-hull/`. Migration: 58/427
  converted, 369 pending; next is cordic_analysis. Previously recorded unrelated audit,
  CAD-policy and optional C++17 limits remain outside this batch.

- 2026-11-07: Converted connected-component labelling. One canonical checked two-pass
  kernel preserves the original ConnectedComponents fields and row-major
  first-occurrence dense IDs, using rank-balanced union and path halving. Complete
  rectangular validation includes empty-first-row ragged inputs; empty/zero-width shapes
  retain their rows. Nonzero negative/extreme ints remain foreground and all image
  indexing uses size_t, with explicit int-label/counter range checks. Output-only and
  analysis paths omit history allocation; optional traces own actual provisional maps,
  per-foreground decisions, all union attempts, final parents and remapping without
  replay. Complete cell caps and online actual record caps have explicit zero-unlimited
  semantics. Native summaries validate label ranges/dense presence and measure areas,
  inclusive bounds, centroids and exposed orthogonal edges, explicitly trusting supplied
  grouping rather than certifying connectivity. Native original-rectangle/checkerboard
  fixtures and a checked one/two-panel labelled-cell Diagram recipe replace report
  algorithms and scatter-marker geometry. The report derives the partial-scan invariant,
  balanced-union complexity and grid perimeter 4A-2E, shows an actual bridge merge and
  diagonal-contact partition change, and reproduces original areas 9,8,8,16 with 41
  foreground cells. The original rectangles need zero successful unions, corrected
  explicitly; they perform 56 total attempts and 153 finds. All 84 affected definitions,
  sixteen compiled/executed documentation examples and four hovers pass. Thirteen new
  cases pass AddressSanitizer and UBSan. All 65,536 four-by-four binary masks match
  independent full-neighbour flood fill in both modes (131,072 complete canonical
  label-map comparisons). One thousand random masks per mode independently replay actual
  predecessor, union/root/rank and remap transitions. A nonminimum-rank-root witness
  retains first-occurrence output IDs. One thousand random measurement maps match
  independent pixel sets and perimeter identities; 576 rectangular checkerboards, holes,
  diagonal contacts, exact/one-below caps, full invalid-input checks and owned storage
  pass. Two graphics cases check validation, bounds, provisional/full-int labels and
  ownership. Thirty new/legacy image/graphics/format cases plus five dir cases pass;
  combined CTest passes 1,263 cases. Five report programs, four inspected figures and
  both exports pass. Final unchanged figure hashes match their inspected versions; the
  corrected work legend is separately rerendered and inspected. Site verifies 57
  reports, 257 executed blocks and 191 matching figures. Original mepml/org removed
  after all gates. Evidence: `build/mepml-conversion/connected-components/`. Migration:
  57/427 converted, 370 pending; next is convex_hull_analysis. Previously recorded
  unrelated audit, CAD-policy and optional C++17 limits remain outside this batch.

- 2026-11-07: Converted conjugate gradient. One checked executing kernel preserves the legacy function/result layouts and adds dense/matrix-free solves, owned actual x/r/p/Ap histories, explicit stop reasons and actual product counts. Complete finite/shape/size/exact-symmetry/control validation precedes dense work; SPD remains a precondition with encountered curvature failure reported, not a matrix-wide certificate. Inclusive absolute stopping checks the recomputed rounded-operator residual separately from the recursive residual, including initially and at zero budget; residual gaps stop unconverged. Widened inner products/norms and checked double updates handle extreme scales without squared-double failures. Online scalar-entry caps count actual snapshots and solve-only calls omit capture. Forward Gauss-Seidel and its native history share one checked sweep kernel; native dense/matrix-free Dirichlet fixtures and analytical unit-RHS solutions replace report loops. The report corrects the gradient sign, distinguishes energy decrease from nonmonotone residual norms, derives exact Krylov termination and the Chebyshev energy bound with floating limits, and explains active-spectrum termination. The original 20-unknown unit-RHS solve takes ten CG steps and twenty counted products, with computed true residual zero explicitly distinguished from the old recursive 2.3e-16. Forty GS sweeps leave residual 1.712176. Matrix-free dimensions 20/40/80/160 match the analytical solutions exactly on this toolchain. All 91 affected definitions, fifteen compiled/executed new or changed documentation examples and four hovers pass. Fourteen new cases pass AddressSanitizer and UBSan. One thousand seeded SPD problems match an independent long-double Cholesky oracle; actual recurrence replay, energy checks, directly recorded callback arguments, active eigenvalue controls, a residual-rise example, an actual residual-gap witness, zero/empty/budget/inclusive controls, full invalid-input checks, callback failures, curvature, caps, widened scales and owned states pass. Thirty-six new/legacy solver/format cases plus five dir cases pass; combined CTest passes 1,233 cases. Five report programs, four inspected figures and both exports pass. Final figures match inspected SVG bytes. Site verifies 56 reports, 252 executed blocks and 187 matching figures. Original mepml/org removed after all gates. Evidence: `build/mepml-conversion/conjugate-gradient/`. Migration: 56/427 converted, 371 pending; next is connected_components_analysis. Previously recorded unrelated audit, CAD-policy and optional C++17 limits remain outside this batch.

- 2026-11-07: Converted the 2D cone surface diagnostic. A single checked angular-gap kernel preserves the legacy mask signature while adding actual owned direction records, per-query witnesses and exact work counters. Full finite-input/control validation precedes bounded work; every ordered pair is scanned, coincident locations contribute no invented ray, widened norms handle huge/subnormal differences, sparse queries retain their full-turn convention, and thresholds are inclusive. The original 2,500 candidate sampler is native and reproduces 1,948 accepted points, 176 flags and minimum flagged radius 9.336971. Strict radial summaries have explicit empty extrema. The report distinguishes this planar analogue from the published 3D molecular algorithm, derives O(n² + sum k log k) work, demonstrates an interior point falsely flagged locally and shows all 962 inner-core points flagged at radius 0.25. All 62 definitions, twelve compiled/executed documentation examples and three hovers pass. Fifteen new cases pass AddressSanitizer and UBSan. An independent supporting-line oracle checks 6,561 ordered grid tuples, all 512 grid subsets and 1,000 random clouds; 500 local clouds use a separate cross/dot angular oracle. Original sampler/mask parity, exact inclusive controls, one-working-ULP thresholds, duplicates, extreme norms, actual pair/direction caps, full validation and ownership pass. Twenty-two new/legacy cases plus five dir cases pass; combined CTest passes 1,197 cases. Five report programs, four inspected figures and both exports pass. Final figure copies match inspected outputs. Site verifies 55 reports, 247 executed blocks and 183 matching figures. Original mepml/org removed after all gates. Evidence: `build/mepml-conversion/cone-surface/`. Migration: 55/427 converted, 372 pending; next is conjugate_gradient_analysis. Previously recorded unrelated audit, CAD-policy and optional C++17 limits remain outside this batch.

- 2026-11-07: Converted computus. One canonical checked positive-int Gregorian
  Easter kernel preserves the original EasterDate fields, triviality and standard
  layout while adding owned actual integer steps and bounded constant-storage
  range summaries. All positive int years are supported; zero/BCE numbering is
  rejected and early positive dates are explicitly proleptic rule projections.
  Full endpoint/order validation and positive complete-count caps precede scans;
  date extrema preserve first ties and INT_MAX-ending loops cannot wrap. The
  report corrects the golden-number/index, nominal-full-moon and day-of-year
  descriptions, derives both exceptional week cases and a sufficient 5,700,000-year
  rule period without claiming minimality. Original notable dates and the 2000-2040
  earliest/latest witnesses and 33-day spread remain. All 40 definitions have
  coverage; six documentation examples compile/run and three hovers pass. Ten
  new cases pass AddressSanitizer and UBSan. A distinct published USNO Oudin
  algorithm and independent Gregorian weekday calculation check every year in a
  complete 5,700,000-year period; its independently accumulated histogram agrees
  exactly. Published date controls, early proleptic years, 10,000 random large
  years, 10,000 period shifts, exceptional predicates, 1,000 independent summary
  ranges, partition additivity, exact/one-below complete caps, INT_MAX termination
  and owned returns pass. Thirty new/legacy calendar/format cases plus five dir
  cases pass; combined runners pass 1,175 cases. Five report programs, four
  inspected figures and both exports pass against the final rebuilt core. The
  cycle contains all 35 dates and 38,750 week corrections. Final figure hashes
  match the inspected versions. Site verifies 54 reports, 242 executed blocks and
  179 matching figures. Original mepml/org removed after all gates. Evidence:
  `build/mepml-conversion/computus/`. Migration: 54/427 converted, 373 pending;
  next is cone_algorithm_analysis. Previously recorded unrelated audit, CAD-policy
  and optional C++17 limits remain outside this batch.

- 2026-11-07: Converted comb sort. One canonical plain-10/13 gap kernel
  preserves the legacy API while adding checked actual work and optional owned
  pass states/identities without replay. Exact quotient/remainder arithmetic fixes
  gap-times-ten overflow across size_t's range, without a special gap adjustment.
  The original full-width forward bubble comparator is now a named native baseline,
  retaining its terminal zero-swap scan. Positive online trace caps count that
  terminal scan; output-only/work-only calls have no cap. The original seven-value
  run has five passes, 23 comparisons and six swaps. All forty seed-four shuffle
  inputs agree with former report counting loops on this toolchain; n=400 gives
  7,106 comb and 155,610 bubble comparisons. The old frozen bubble count 147,630
  and implementation-specific shuffle mapping are explicitly distinguished.
  The report replaces the unqualified linearithmic claim with a quadratic
  worst-case upper bound, sorted-input Theta(n log n), a derived inversion decrease
  and an actual carried-identity instability example. All 29 affected definitions
  have coverage; eight documentation examples compile/run and three hovers pass.
  Eleven new cases pass AddressSanitizer and UBSan. Independent sorting, inversion
  and stable maximum-left-displacement oracles check 9,841 ternary inputs through
  length eight, all 40,320 eight-element permutations and 1,000 seeded duplicate
  arrays. Actual pass/state counters, signed extremes, full-range gap arithmetic
  against unsigned-128 products, former-loop parity, exact/one-below caps and
  snapshot ownership pass. Fourteen new/legacy comparison-sort cases plus five
  dir cases pass; combined runners pass 1,145 cases. Five report programs, four
  inspected figures and both exports pass. Site verifies 53 reports, 237 executed
  blocks and 175 matching figures. Original mepml/org removed after all gates.
  Evidence: `build/mepml-conversion/comb-sort/`. Migration: 53/427 converted,
  374 pending; next is computus_analysis. Previously recorded unrelated audit,
  CAD-policy and optional C++17 limits remain outside this batch.

- 2026-11-07: Converted Cohen-Sutherland line clipping. A canonical checked
  kernel preserves the legacy call and ClippedSegment field layout while validating
  all finite coordinates and ordered inclusive bounds before trivial decisions.
  Zero-area rectangles, corner tangency and directed point segments are valid.
  Owned actual endpoints/outcodes, selected boundaries, replacement codes and
  terminal decisions share the executing kernel without replay. Widened original-
  line determinants avoid drift from earlier rounded cuts and preserve tested
  integer corner cancellations and huge diagonals. Bounded interpolation handles
  unrepresentable products; current-range projection retains prior half-planes.
  The report explicitly limits floating geometry rather than claiming exact
  predicates, and derives a worst-case four-cut bound from permanently satisfied
  half-planes. A left-only endpoint gains a top bit before rejection, correcting
  the old decreasing-bit termination argument. All 45 definitions have coverage;
  seven documentation programs compile/run and three clangd hovers pass. Eleven
  new cases pass AddressSanitizer and UBSan. An independent exact-rational
  parametric oracle checks all 86,436 grid segment/ordered-rectangle combinations,
  including degenerate rectangles, plus 3,000 seeded integer problems. Original
  results, actual work/state invariants, reversal/exact similarities, every-coordinate
  validation, one-ULP corner controls, huge/subnormal values, extreme thin-slab point
  contact and ownership pass. Twenty-six native new/legacy graphics/format cases
  plus five dir cases pass; combined runners pass 1,131 cases. Five report programs,
  four inspected equal-scale/operation-count figures and both exports pass. The
  complete 2,401-segment report grid attains four cuts on four inputs, with 2,436
  total cuts and 4,837 decisions. Site verifies 52 reports, 232 executed blocks and
  171 matching figures. Original mepml/org removed after all gates. Evidence:
  `build/mepml-conversion/cohen-sutherland/`. Migration: 52/427 converted,
  375 pending; next is comb_sort_analysis. Previously recorded unrelated audit,
  CAD-policy and optional C++17 limits remain outside this batch.

- 2026-11-07: Converted cocktail shaker sort. One canonical adjacent-exchange
  kernel preserves legacy output and directional-bound behavior, adding optional
  owned actual pass states, stable original identities and checked work counters.
  A positive online snapshot cap includes terminal zero-comparison passes; the
  output-only API has no cap. The original seven integers need five directional
  passes, twenty comparisons and ten swaps, stopping with bounds [2,4]. The report
  corrects the old universal half-pass and bounds-must-meet claims, proves the
  outer-region invariant, stability and swaps=inversions, and derives exact
  sorted/reverse/turtle/rabbit comparison families. All 29 affected definitions
  have coverage, four documentation examples compile/run and three hovers pass.
  Nine new cases pass AddressSanitizer and UBSan. Independent stable-sort/inversion
  and pass-state checks cover all 9,841 ternary inputs through length eight,
  40,320 eight-element permutations and 1,000 seeded duplicate arrays. Integer
  extremes, exact/one-below caps, empty stopping passes and owned snapshots pass.
  Twelve new/legacy exchange-sort cases plus five dir cases pass; combined runners
  pass 1,105 cases. Four mep programs, four raster-inspected figures and both
  exports pass. The site verifies 51 reports, 227 executed blocks and 167 matching
  figures. Original mepml/org removed after all gates. Evidence:
  `build/mepml-conversion/cocktail-shaker/`. Migration: 51/427 converted,
  376 pending; next is cohen_sutherland_analysis. Previously recorded unrelated
  audit, CAD-policy and optional C++17 limits remain outside this batch.

- 2026-11-07: Converted CNN analysis with canonical tensor bar generation,
  small convolutional/dense models, signed z-plus relevance and native square-cell
  image diagrams. Model constructors expose forward endpoints and named leaves;
  the report keeps all SGD/Adam training loops explicit. Bar generation matches
  the original complete 600-row MT19937_64 normal/shuffle stream. Relevance uses
  actual max-pool winners, signed raw-score seeds and documented bias/epsilon/
  zero-path limits, without mutating parameters, input gradients or caller grad mode.
  Sixty-two affected definitions have coverage; fourteen documentation programs
  compile/run and four clangd hovers pass. Twenty-one new cases pass AddressSanitizer
  and UBSan, covering independent hand gradients/redistribution, every-parameter
  finite differences, local-pool phase changes, global-pool controls, signed paths,
  invalid domains/overflow, exact parameter counts, real learning and square pixels.
  Twenty-nine new/legacy CNN cases and five dir cases pass; combined runners pass
  1,093 cases. Eight complete report programs reproduce the original bar fit,
  filters, same-target initialized relevance controls and positional-shift results.
  Native verified MNIST archives supply a disjoint 512-row fit, 128-row validation
  and 256-row official-test experiment; checkpoint selection uses validation only.
  All eighteen figures are raster-inspected and match native output bytes; both
  exports pass. The site verifies 50 reports, 223 executed blocks and 163 matching
  figures. Original mepml/org removed after all gates. Evidence:
  `build/mepml-conversion/cnn/`. Migration: 50/427 converted, 377 pending; next is
  cocktail_shaker_sort_analysis. Previously recorded unrelated audit, CAD-policy
  and optional C++17 limits remain outside this batch.

- 2026-11-07: Converted closest-pair analysis. A dedicated canonical checked
  all-pairs kernel retains the legacy call/result layout, validates every finite
  coordinate before complete work caps, compares widened differences/hypot values
  and checks finite-double legacy conversion. Owned indexed results preserve
  original duplicate identities and first computed-distance ties. Actual strict
  improvement records share the executing kernel without replay; their positive
  record cap is online, while the complete pair cap preflights after validation.
  Zero distances do not stop the all-pairs scan. A bounded native rectangle sampler
  preserves the original sixty seed-424242 MT19937 draws and planted pair; standard-
  library uniform-real mapping limits cross-implementation bitwise reproducibility.
  All 45 affected definitions have coverage, ten examples compile/run and three
  hovers pass. Seventeen new tests pass AddressSanitizer and UBSan. Exact integer
  squared-distance oracles check all 6,561 ordered four-record grid tuples, 502
  nontrivial grid subsets and 1,000 seeded integer clouds, plus 720 permutations
  with exact similarity transforms. Original cloud/pair, subnormal/tiny/huge
  distances, wide selected values beyond double, actual record caps, validation
  before limits, owned copies and generator bounds pass. Twenty-two native
  new/legacy/geometry-format cases and five dir cases pass. Production Python help
  and wrapper are regenerated from canonical header/interface sources; scoped
  invalid/overflow failures become ValueError/OverflowError. Nine targeted SWIG
  compatibility tests pass using that exact production rule; a full installed-
  package build is not claimed. The report corrects the old one-square-root and
  exactness claims, derives the invariant/exact work, preserves the original
  62-record example and expands its right triangle. An unequal-real-distance
  counterexample whose computed values tie explicitly limits geometric accuracy.
  Five mep programs, four inspected figures and both exports pass. Combined runners
  pass 1,064 cases; site verifies 49 reports, 215 executed blocks and 145 matching
  figures. Original mepml/org removed after all gates. Evidence:
  `build/mepml-conversion/closest-pair/`. Migration: 49/427 converted, 378 pending;
  next is cnn_analysis. Previously recorded unrelated audit, CAD-policy and
  optional C++17 limits remain outside this batch.

- 2026-11-06: Converted Clarke-Wright savings routing. A dedicated canonical
  parallel-savings kernel retains the legacy call/include and VehicleRoutes layout,
  validates all finite coordinates/nonnegative demands and finite positive capacity,
  and checks widened distances/loads and finite-double final cost. Equal savings
  now use ascending original pair ids; zero savings remain eligible. Actual owned
  endpoint lookups, rejection reasons, reversal calls, accepted customer orders,
  route loads, named work and star/savings/distance accounting share one kernel
  without replay. Complete pair/endpoint/trace caps preflight after validation.
  Independent native route evaluation reports closure, coverage, repeated/missing
  customers, interior depot visits, overloads and widened edge sums; invalid ids
  throw. Depot demand is validated but ignored; capacity comparisons have no hidden
  tolerance. All 73 module definitions, eleven compiled examples and three hovers
  pass. Seventeen new cases pass AddressSanitizer and UBSan. Independent deque
  state checks cover 1,926 line/depot/capacity inputs and 1,000 random Euclidean
  problems. Exhaustive permutation/cut oracles check 100 small problems and the
  fixed six-point counterexample, whose shorter feasible length is 30.520843
  against greedy 30.655229. Exact/one-below caps, full validation before limits,
  literal floating boundaries, zero demands, depot-only/nonzero depot, ownership,
  malformed proposed routes and wide arithmetic pass. Thirty-two routing/tour/
  formatting cases plus five dir cases pass. The report retains the original
  distance-four merge, proves feasibility and exact telescoping savings, states
  cubic endpoint lookup/quadratic storage and distinguishes greedy capacity sweeps
  from exact-optimum monotonicity. Five mep programs, four inspected figures and
  both exports pass. Combined runners pass 1,042 cases; site verifies 48 reports,
  210 executed blocks and 141 matching figures. Original mepml/org copies removed
  after all gates. Evidence: `build/mepml-conversion/clarke-wright/`. Migration:
  48/427 converted, 379 pending; next is closest_pair_analysis. Previously recorded
  unrelated audit, CAD-policy and optional C++17 limits remain outside this batch.

- 2026-11-06: Converted the circle-layout report. Dedicated canonical headers
  retain Point2D/Circle2D layouts and the legacy include/call while the executing
  Jacobi kernel adds checked widened arithmetic, fixed-step controls, complete
  pair-work/snapshot preflight caps, owned actual raw states, measured edge/nonedge
  potential, normalization map and separate returned-disk diagnostics. Full
  endpoint/options validation precedes limits/allocation; loops are ignored,
  reversed duplicates collapse and empty/singleton inputs skip sweeps. Counts
  distinguish relaxation pairs, commits, raw snapshots and output measurements.
  The report corrects the original Koebe-construction and linear-memory claims:
  dense adjacency is quadratic, no universal fixed-step descent is promised,
  and resetting output radii can destroy contacts or create overlaps. Original
  triangle coordinates remain; three raw contacts become zero returned contacts.
  A positive unequal-radius K4 control has all six contacts, while the default
  eight-cycle returns eight overlapping pairs. Actual energy agrees with the
  independent two-circle closed form, and step size two increases it. All 83
  definitions have coverage; thirteen documentation examples compile/run and
  three hovers pass. Fourteen new tests pass AddressSanitizer and UBSan, with
  independent raw-state checks of all 1,100 simple graphs through five vertices,
  1,000 seeded random multigraph/control cases and triangle/two-circle closed
  forms. Endpoint/tolerance/radius validation, owned frames, exact/one-below caps,
  huge preflight work, range checks and zero-step semantics pass. Thirty native
  circle/graph-analysis/formatting cases plus five dir cases pass. Five mep
  programs, four inspected figures and both exports pass. Combined runners pass
  1,010 cases; the site verifies 47 reports, 205 executed blocks and 137 matching
  figures. Catalog corrected; original mepml/org removed after all gates.
  Evidence: `build/mepml-conversion/circle-packing/`. Migration: 47/427 converted,
  380 pending; next is clarke_wright_analysis. Previously recorded unrelated audit,
  CAD-policy and optional C++17 limits remain outside this batch.

- 2026-11-06: Converted Cipolla modular-square-root analysis. The canonical
  module validates every uint64 prime before normalization/shortcuts, uses reduced
  unsigned-128 scalar products and sums, safe canonical differences and a safe
  exponent expression, and verifies the final extension root. Candidate order
  1,2,...,p-1,0 preserves earlier successes and fixes sqrt(1) modulo three, where
  zero is the only suitable auxiliary. The legacy signed API/include remains;
  full unsigned root pairs retain raw identity while a complementary root is used
  when necessary to give a nonnegative representable legacy result. Actual owned
  candidates, exponent-bit transitions, closing squares, path and named counters
  share the kernel without replay. Positive work caps expose no partial result;
  primality's internal arithmetic is excluded explicitly from named product counts.
  All 75 module definitions are documented, twelve examples compile/run and three
  hovers pass. Fourteen new tests, four arithmetic regressions, nine formatting and
  five dir cases pass. All fourteen new cases pass AddressSanitizer and UBSan.
  Independent square sets exhaust 6,338 residue inputs over 55 prime fields through
  257 and normalized aliases, with all 1009 original residue verdicts retained.
  Add-and-double modular arithmetic verifies every actual candidate and pair
  transition. Another 1,000 known squares span four large primes through the top
  of uint64, checking signed fallback, endpoints and widened arithmetic. Composite/
  pseudoprime rejection before zero shortcuts, exact/one-below caps, ownership,
  unused closing-square work and binary/zero/nonresidue shortcuts pass. The report
  preserves the original four root choices and 504-square plot, proves norm and
  Frobenius correctness, distinguishes deterministic search from uniform-random
  expectations, corrects the 1903 attribution, and shows a raw p-3 root returned
  safely as legacy root three. Five mep blocks, three inspected SVGs and both
  exports pass. Combined runners pass 980 cases; site verifies 46 reports, 200
  executed blocks and 133 matching figures. Original mepml/org copies removed
  after all gates. Evidence: `build/mepml-conversion/cipolla/`. Migration: 46/427
  converted, 381 pending; next is circle_packing_layout_analysis. Previously
  recorded unrelated audit, CAD policy and optional C++17 limits remain outside
  this batch.

- 2026-11-06: Converted Chu-Liu/Edmonds directed-tree analysis. A dedicated
  canonical module retains the legacy call/layout while recovering actual original
  edge ids and parents through iterative compact contraction history. Working costs
  and final original-edge sums use checked long double with checked conversion;
  vertex ids no longer narrow to int, and every endpoint/finite signed weight
  validates before caps, including ignored loops and root-entering arcs. One kernel
  captures actual incoming choices, wide reduced arcs, predecessor cycles, original
  partitions, baselines and work without replay. Cycle detection follows each
  non-root vertex once per successful round. Phase/combined-edge-visit caps return
  no partial result; nonexistence retains its actual failed round with empty tree
  arrays. All 63 new-module definitions are documented, twelve examples compile/run
  and three hovers pass. Fourteen new tests, seven advanced-graph cases, nine
  formatting and five dir cases pass; all fourteen new cases pass AddressSanitizer
  and UBSan. Independent enumeration checks 2,187 three-vertex signed ternary
  graph/root combinations, 16,384 four-vertex presence/root combinations and 1,000
  signed random multigraphs. Every phase, reduced cost, cycle, partition, baseline,
  final original edge and named counter is verified. Nested/simultaneous cycles,
  missing vertices versus unreachable closed cycles, original parallel ties,
  ignored malformed arcs, owned copies, exact/one-below caps, size_t domains,
  wide intermediates/fitting totals, final overflow and a 100,000-vertex chain
  pass. The report preserves the original cost six and derives rooted reachability,
  baseline normalisation and the contraction exchange argument. It documents the
  O(E+PV) compact expansion history even for cost-only calls, full trace storage,
  rounded numerical limits and the scope of its two-round 6k scan family. Five
  mep blocks, three inspected SVGs and both exports pass. Combined runners pass
  953 cases; site verifies 45 reports, 195 executed blocks and 130 matching
  figures. Original mepml/org copies removed after all gates. Evidence:
  `build/mepml-conversion/chu-liu/`. Migration: 45/427 converted, 382 pending;
  next is cipolla_analysis. Previously recorded unrelated audit, CAD policy and
  optional C++17 limits remain outside this batch.

- 2026-11-06: Converted Chudnovsky pi analysis. A dedicated canonical module
  replaces separate factorial/power evaluation with a bounded-ratio double
  recurrence, retaining the legacy include/call while documenting its changed
  rounding path and explicit nonpositive-count errors. Actual owned term, sum and
  prefix-estimate records share the kernel without replay. Stagnation alone does
  not stop evaluation; the first underflowed zero is recorded, after which huge
  suffixes are skipped without giant reservation. Native ratio, decimal-gain and
  geometric log-bound APIs document the limiting factor 1728 and separate exact
  truncation theory from rounding and real-accuracy guarantees. All 35 module
  definitions are documented; eight examples compile/run and three hovers pass.
  Ten new tests, eight pi/BBP/CORDIC cases, nine formatting and five dir cases pass.
  All ten new cases pass AddressSanitizer and UBSan. Independent factorial-prefix
  checks, 1,000 full-range random ratio indices, 10,000 uniform-ratio indices,
  exact positive-polynomial identities, one hundred omitted-term bounds and
  counts 1--4096 pass, alongside INT_MAX requests, exact underflow boundaries,
  malformed counts and owned records. A 240-digit Decimal Machin oracle verifies
  all 25 actual native terms/prefixes, twelve finite-series error bounds and seven
  ratios, explicitly marking values below its decimal resolution. Linked-core and
  standalone Clang probe output agrees exactly. The report preserves the original
  four counts, labels both rounded references, removes artificial log floors and
  distinguishes a nonzero lost correction, a later underflowed term and the
  theoretical 14.18-digit rate. Five mep blocks, three inspected figures and both
  exports pass. Combined runners pass 923 cases; site verifies 44 reports, 190
  executed blocks and 127 source-matching figures. Original mepml/org copies are
  removed after all gates. Evidence: `build/mepml-conversion/chudnovsky/`.
  Migration: 44/427 converted, 383 pending; next is chu_liu_edmonds_analysis.
  Previously recorded unrelated audit, CAD policy and optional C++17 limits remain
  outside this batch.

- 2026-11-06: Converted Christofides TSP analysis. The legacy call, result layout
  and general finite nonnegative matrix acceptance remain. A dedicated canonical
  module and shared TourResult header expose actual tree edges, odd vertices,
  matching choices, Euler vertices/stable edge-copy identities, first-visit shortcut
  positions, stage costs and named work without replay. Ties explicitly prefer
  weight/endpoints and ascending matching partners. Widened checked matching and
  route sums prevent invalid partner reconstruction after double overflow; the
  final result conversion is checked. Separate full-matrix metric checks, native
  Euclidean and uniform hub/leaf fixtures, hard odd caps and table preallocation
  caps document the real contract. All 48 new-module definitions are documented,
  ten examples compile/run and three hovers pass. Fifteen new tests, six legacy
  tour/search cases and nine formatting cases pass; five dir cases pass too.
  All fifteen new tests pass AddressSanitizer and UBSan. Independent Prim and
  recursive pairing verify 729 exhaustive four-vertex cost matrices and 1,000
  planar inputs. Exhaustive tour orders check the 3/2 component bounds for 120
  integer shortest-path metrics. Euler copies, shortcut positions, named counts,
  every square start, zero/empty/singleton costs, invalid trailing values, exact
  and one-below caps, ownership, fitting results with oversized matching alternatives,
  final overflow and a 1,200-vertex line pass. The report preserves the original
  square and distinguishes exact metric theory, tolerance/rounding limits and the
  bounded exponential subset matching implementation. A symmetric nonmetric
  counterexample increases cost from five to 103 by shortcutting, versus proven
  optimum four. Five mep blocks, three inspected figures and both exports pass.
  Combined runners pass 896 cases; site verifies 43 reports, 185 executed blocks
  and 124 source-matching figures. Original mepml/org copies removed after all
  gates. Evidence: `build/mepml-conversion/christofides/`. Migration: 43/427
  converted, 384 pending; next is chudnovsky_pi_analysis. Previously recorded
  unrelated audit, CAD policy and optional C++17 limits remain outside this batch.

- 2026-11-06: Converted Chinese remainder analysis. The legacy call and
  CrtSolution layout remain. A shared canonical kernel now widens Bezout updates,
  checks the compatible lcm before narrowing and records actual merge certificates,
  wide correction/candidate values, representable prefixes and operation counts.
  Overflow leaves the trace verdict absent; the ordinary call throws instead of
  returning a wrapped period. No claim is made about an unvisited inconsistent
  suffix. Checked residue normalization and representative verification replace
  local unsafe modulo helpers. All moduli validate before arithmetic, comparisons
  and positive complete-call merge/division caps; cap failures expose no partial
  trace. All 68 header/source definitions are documented; eleven examples compile
  and run, with three clangd hovers passing. Fourteen new cases, ten legacy CRT
  cases and nine formatting cases pass; all 24 CRT cases pass AddressSanitizer and
  UBSan. Independent enumeration checks 12,286 small pair/triple systems; another
  1,000 signed random and 1,000 constructed near-limit systems verify every actual
  Bezout identity, candidate and counter. Exact/oversized periods, wide corrections
  with fitting results, Fibonacci gcd chains, signed endpoints, reordered late
  contradictions, exact/one-below caps, invalid suffixes, ownership and 100,000
  unit constraints pass. The original three controls and coprime grid are retained,
  with actual merge diagrams and an exhaustive 1/g compatibility census. Five mep
  blocks, three inspected figures and both exports pass. Combined runners pass
  866 CTest cases; site verifies 42 reports, 180 executed blocks and 121 matching
  figures. Original mepml/org copies removed after verification. Evidence:
  `build/mepml-conversion/crt/`. Migration: 42/427 converted, 385 pending; next is
  christofides_analysis. Previously recorded unrelated audit, CAD policy and
  optional C++17 limits remain outside this batch.

- 2026-11-06: Converted Chien root-search analysis. The legacy signature,
  normalization, zero/constant/direct-linear paths, distinct primitive-power root
  order and closing updates remain. A shared canonical kernel now records actual
  generator rejections, coefficient states, final orbit closure and checked named
  operation counts without replay. Whole-call point/term caps preflight normalized
  sweeps; generator-step caps reject before the next named operation, with trace
  reservation after successful generator discovery. Linear inversion products are
  separate from sweep budgets. The existing native Horner/formatting APIs replace
  unchecked report helpers. All 143 polynomial-module definitions are documented;
  twenty-eight examples compile/run and three new plus three existing hovers pass.
  Twenty-six native cases include twelve new and fourteen polynomial/formatting
  regressions; all twelve new cases pass AddressSanitizer and UBSan. Independent
  unsigned Horner and generator-order walks exhaust 23,941 coefficient vectors
  through length five over GF(2/3/5/7), checking every actual state, root order,
  generator result and counter. Another 1,000 signed random polynomials and 1,000
  near-limit linear cases cover widened arithmetic and signed endpoints. Exact and
  one-below caps, huge-sweep preflight and generator rejection before giant record
  reservation, malformed fields, ownership, repeated roots and shortcut semantics
  pass. Existing Cantor-Zassenhaus and dir runners are rebuilt and pass. The report
  retains the original thirteen GF(13) values and root, corrects the nine-value
  image versus all-input coverage distinction, and adds actual GF(7) states, generator
  rejections, full operation counts, formal-zero/function-zero examples and explicit
  prime-field limits. The degree-only curve is labelled a theoretical formula;
  no runtime speedup or complete decoder is claimed. Five mep blocks, three inspected
  SVGs and both exports pass. Combined runners pass 833 CTest cases; site verifies
  41 reports, 175 executed blocks and 118 source-matching figures. Removed original
  mepml/org files after verification. Evidence: `build/mepml-conversion/chien/`.
  Migration: 41/427 converted, 386 pending; next is chinese_remainder_analysis.
  Previously recorded unrelated audit, CAD policy and optional C++17 limits remain
  outside this batch.

- 2026-11-06: Converted Cheney copying analysis. The legacy inline call and
  CheneyResult layout retain valid-input BFS order and ignored-size semantics.
  ObjectGraph defaults to empty; full graph/id validation returns actual visit
  counts and optionally checks positive size prefixes. Iterative liveness validates
  before indexing, while size access checks only its requested value. A shared
  append-only kernel now constructs an owned symbolic heap, rewriting every live
  root/reference slot and preserving duplicates, sharing and cycles. Checked size
  offsets and destination capacity are separate from object-slot forwarding. Actual
  chronological root/field/scan events record scan/free boundaries without replay
  or queue snapshots; positive whole-call event caps expose no partial result.
  Growing output rows are reacquired after possible vector reallocations. All
  56 definitions are documented, fourteen examples compile/run and three hovers
  pass. Thirty-three native cases include fifteen new and eighteen existing
  collector/formatting cases. All fifteen new cases pass AddressSanitizer and
  UBSan. Independent deque and closure oracles cover 6,775 exhaustive small graph/
  ordered-root configurations; 1,000 random multigraphs verify every rewritten slot,
  size default, actual event and repeated evacuation. Exact/zero capacities, event
  caps, malformed garbage, ignored metadata, duplicate ordering and owned copies
  pass, alongside a 100,000-object chain and a 50,000-object frontier exercising
  output reallocation. Affected buddy/memory/dir runners are rebuilt and pass.
  The report retains the original tree, adds scrambled labels and a shared cycle,
  and corrects live-only work, constant-space, byte-address, fragmentation/locality
  and runtime-collector claims. It separates full validation from live copying,
  slots from packed units and destination capacity from an equal-semispace model.
  Five mep blocks, three inspected SVGs and both exports pass. Combined runners
  pass 807 CTest cases; site verifies 40 reports, 170 executed blocks and 115
  source-matching figures. Removed original mepml/org files after verification.
  Evidence: `build/mepml-conversion/cheney/`. Migration: 40/427 converted,
  387 pending; next is chien_search_analysis. Previously recorded unrelated
  audit, CAD policy and optional C++17 limits remain outside this batch.

- 2026-11-06: Converted Chakravala/Pell analysis. The legacy call signature and
  PellSolution layout remain through a compatibility include. Dedicated canonical
  code shares ordinary and actual-traced cyclic steps, fixes the negative-one
  norm multiplier choice, compares exact squared errors with smaller-on-ties
  selection, and retains the floor-root/no-shortcut path to positive norm one.
  Quotient/remainder composition supports oversized unreduced numerators when
  reduced coordinates still fit. Checked coordinate overflow and strict complete-
  call update budgets throw without partial results. New exact uint64 roots and
  signed/unsigned 128-bit text replace report helpers; the existing private format
  helper forwards to canonical text. An exact gcd-cancellation checker avoids
  overflowing coordinate squares and accepts nonfundamental solutions honestly.
  All 47 dedicated definitions and sixteen shared-format definitions pass audits;
  eleven documented examples compile/run and three hovers pass. Twenty-eight
  native cases include thirteen new and fifteen existing number-theory/formatting
  cases; thirteen new cases pass UBSan. Independent continued fractions and a
  separate Python big-integer oracle compare 11,001 inputs: 9,215 representable
  fundamentals, 1,685 explicit range failures and 101 squares. All 7,818 actual
  records for 969 coefficients through 1,000 satisfy exact norm/divisibility,
  independently enumerated multiplier minima and path continuity checks. Full-width
  square neighbours, zero/exact budgets, malformed pairs, copied records, 200,000
  small identity comparisons, 50,000 random root brackets, 100,000 squares and
  128-bit decimal endpoints pass. N=661's actual final numerator exceeds signed
  128-bit while its quotient fits, directly checking the widened-numerator fix.
  The report corrects nearest-root, small-intermediate, digit-count and unsupported
  period-complexity claims, retains the original famous pairs/size plot and adds
  actual paths, work counts, numeric limits and nonminimal valid-pair examples.
  Five mep blocks, three inspected SVGs and both exports pass. Combined runners
  pass 774 CTest cases; site verifies 39 reports, 165 executed blocks and 112
  source-matching figures. Removed original mepml/org files after verification.
  Evidence: `build/mepml-conversion/chakravala/`. Migration: 39/427 converted,
  388 pending; next unchecked entry is cheney_analysis. Previously recorded
  unrelated audit, CAD policy and optional C++17 limits remain outside this batch.

- 2026-11-06: Converted the separate buffered ChaCha20 random-engine report.
  Constructors, next_u64 signatures, simulation-seed packing and valid-range output
  are retained. The engine reuses the canonical cipher permutation, retains all
  eight words at the final block counter, then throws before raw state mutation;
  it no longer silently carries into the nonce. Capacity inspection includes both
  ungenerated blocks and buffered tails. Native seed expansion and preflighted
  actual draw traces replace duplicated report logic; traces reserve before drawing
  and preserve the inherited normal cache. Dedicated header/source coverage is 9/9;
  a separate XML inventory verifies all ten changed engine/class/private definitions
  in random.hpp, which remains outside the completed-file manifest. Eight documented
  examples compile/run and three new plus one existing hover pass. Thirty-five native
  cases pass, including eleven new engine cases and twenty-four existing random and
  distribution cases. The eleven new cases also pass UBSan. Independent row-rotation
  blocks verify 1,000 random streams, actual record locations and values, partial
  buffers and near-limit counters. Another 1,000 seed cases verify legacy packing
  and explicit-key replay. Every final-buffer tail, failed complete-call preflight,
  copied/assigned raw state and normal caches, sampler rejection consumption and
  base dispatch pass. The report checks the entire RFC block as eight raw words,
  preserves the original raw/uniform values and corrects its stale stored bounded
  result to 6 without changing the common sampler. It derives buffering/capacity,
  demonstrates exhaustion and copy replay, and explains variable sampler consumption
  and the limits of simulation seeds, copies, entropy and security claims. Four mep
  blocks, two inspected SVGs and both exports pass. Combined runners pass 746 CTest
  cases; site verifies 38 reports, 160 executed blocks and 109 source-matching figures.
  Removed original mepml/org files after verification. Evidence:
  `build/mepml-conversion/chacha20/`. Migration: 38/427 converted, 389 pending;
  next is chakravala_analysis. Previously recorded unrelated audit, CAD policy and
  optional C++17 limits remain outside this batch.

- 2026-11-06: Converted ChaCha20 analysis. Existing block/XOR signatures and
  valid-range output are preserved; complete-call counter overflow now rejects
  before copying or block work. Shared actual quarter, permutation, block and XOR
  kernels expose owned stages, all eighty quarters and twenty rounds, feed-forward
  words and consumed block ranges without replay. Ordinary calls keep no provenance.
  New checked capacity/block-count APIs handle uint64 endpoints, and whole-call trace
  caps reject before allocation. Binary-safe hex and byte histograms replace report
  helpers. All 44 definitions are documented, sixteen examples compile/run and three
  hovers pass. Thirty-three native cases pass including existing crypto/formatting
  checks; twenty-four crypto cases pass UBSan. Independent row-rotation, widened
  quarter and inverse oracles check 500 random blocks, 500 binary messages and
  10,000 quarter rounds, including every actual record. Counter limits, exact caps,
  empty input, all byte values, histogram groups and ownership pass. The report
  checks complete RFC vectors, retains the original 4 KiB sample, demonstrates the
  tautological histogram mean using a constant-byte control, and executes aligned
  and misaligned one-shot chunking and unauthenticated bit flipping. It explains
  raw versus AEAD capacity, discarded tails, key-bearing traces and the limits of
  histogram/security/timing claims. Five mep blocks, three inspected SVGs and both
  exports pass. Combined runners pass 711 CTest cases; site verifies 37 reports,
  156 executed blocks and 107 source-matching figures. Removed original mepml/org
  files after verification. Evidence: `build/mepml-conversion/chacha20_analysis/`.
  Migration: 37/427 converted, 390 pending; next is the separate chacha20 random
  generator report. Previously recorded unrelated audit, CAD policy and optional
  C++17 limits remain outside this batch.

- 2026-11-06: Converted Cantor-Zassenhaus and its canonical prime-field arithmetic.
  Existing GFPoly/PolyDivMod layouts and division/root/factor signatures and default
  seed are retained. Checked prime validation reuses Miller-Rabin; safe normalization,
  widened coefficient sums/products, zero-divisor checks and size_t degrees fix unchecked
  arithmetic. New reusable arithmetic/formatting and squarefree/irreducibility APIs
  replace report-local code. Chien retains its order/zero convention, with safe
  primitive-root bounds and constant/linear shortcuts. Shared factorization validates
  nonzero squarefree odd-field inputs, uses incremental DDF and iterative EDF, and
  avoids p^d exponent overflow through a Frobenius recurrence. Actual owned traces
  include all attempts, gcd/character/complement records, units, residuals and checked
  work counts. Whole-call trial/product caps return no partial factorization; ordinary
  calls allocate no provenance records. All 90 definitions are documented, twenty
  examples compile/run and three hovers pass. Twenty-five native cases pass including
  existing polynomial/formatting cases; sixteen polynomial cases pass UBSan. Independent
  monic trial division verifies 7,800 small inputs: 6,258 squarefree cases factor and
  1,542 repeats reject. A further 1,000 random draws supply squarefree seeded cases;
  300 near-limit arithmetic cases and a separate unsigned-128-bit exponent oracle
  verify actual Frobenius/character/gcd/complement records and counter formulas.
  Exact budgets, invalid fields/divisors, units, helper GF(2) arithmetic, root ordering,
  zero-seed replay and owned copies pass. The report preserves the original quartic
  and bars, distinguishes its single executed DDF step from the residual shortcut,
  checks irreducibility separately from reconstruction, measures 64 seeds and factors
  near INT64_MAX without a uint64 exponent. It clearly limits claims about biased
  deterministic sampling, squarefree decomposition, characteristic two and operation
  models. Five mep blocks, three inspected SVGs and both exports pass. Combined runners
  pass 678 CTest cases; site verifies 36 reports, 151 executed blocks and 104
  source-matching figures. Removed original mepml/org files after verification.
  Evidence: `build/mepml-conversion/cantor_zassenhaus/`. Migration: 36/427 converted,
  391 pending; next checklist entry is chacha20_analysis, followed by the separate
  chacha20 report. Previously recorded unrelated audit, CAD policy and optional C++17
  limits remain outside this batch.

- 2026-11-06: Converted canopy clustering. Existing inline signatures and Canopy
  layout are retained, with an initialized default centre. Finite rectangular/radius
  validation and hypot-based norms replace unchecked indexing and squared-distance
  overflow/underflow. Shared construction preserves strict all-original-point
  membership and index-order seeds, explicitly consuming a selected zero-tight-radius
  seed. Native actual traces expose distances, decisions, seed-pool transitions and
  checked counters; whole-call distance budgets return no partial result. Coverage
  inverts memberships and deduplicates unordered candidate pairs, with checked manual
  records and before-deduplication pair budgets. Centre extraction validates only
  selected indices. A native translated-grid generator preserves the original ninety
  coordinates/order; the dataset directory already covers it in the finished manifest.
  All 45 definitions are documented, ten examples compile/run and three hovers pass.
  Twenty-eight native cases pass, including existing recurrent/PCNN/formatting cases;
  nineteen canopy/recurrent cases pass UBSan. Independent live-list and pair oracles
  cover 6,825 exhaustive configurations and 1,000 random multidimensional cases, with
  500 additional shuffled cases checking the sufficient metric condition. Boundaries,
  budgets, ownership, malformed inputs/records, original grids and huge/tiny norms pass.
  The report corrects the unconditional close-pair guarantee and Gaussian description,
  derives T1>=T2+r, executes missed-pair and nonmonotone-tight-radius examples, and
  adds native iris candidate coverage without claiming measured acceleration or
  downstream accuracy. Five mep blocks, four inspected SVGs and both exports pass;
  iris legends fit and the omitted unordered pair has an undirected link. Combined
  runners pass 653 CTest cases; site verifies 35 reports, 146 executed blocks and 101
  source-matching figures. Removed original mepml/org files after verification.
  Evidence: `build/mepml-conversion/canopy/`. Migration: 35/427 converted, 392 pending;
  next is cantor_zassenhaus_analysis. Previously recorded unrelated audit, CAD policy
  and optional C++17 limits remain outside this batch.

- 2026-11-06: Converted Cannon matrix multiplication. The scalar API signature,
  cyclic order and final closing shift are preserved through a compatibility include.
  Dedicated canonical code adds checked blocked scheduling, allocation-free checked
  work prediction and actual traces shared with the ordinary numeric kernel. Source
  ids are transported alongside numeric buffers; owned snapshots capture actual C
  after each round. Exact shape/divisibility checks, size_t product checks and complete-
  call block-update budgets precede buffer allocation. Ordinary calls keep no trace
  records, inputs can alias and remain unchanged, and nonfinite arithmetic propagates.
  All 29 definitions are documented, seven examples compile/run and three hovers pass.
  Twenty-four native cases pass, including existing sparse/formatting cases; fifteen
  sparse/Cannon cases pass UBSan. Independent integer products and source/partial
  oracles cover 6,561 exhaustive 2-by-2 pairs at both mesh sizes and 1,000 larger
  blocked cases. Complete k visits, identity/zero/aliased inputs, budgets, counts,
  ownership, malformed shapes and count overflow pass; random floating cases match
  long-double dot products within scaled allowances. The report preserves the original
  small product and random comparison design while reusing native random draws and
  DenseMatrix multiplication. It corrects bitwise-equality and memory/communication
  claims, demonstrates cancellation yielding one versus zero, and measures 96-by-96
  scalar/copy/block work across mesh sizes. Four mep blocks, three inspected SVGs and
  both exports pass. The website's separate compiler rerun passes with identical
  inspected SVG hashes. Combined runners pass 625 CTest cases; site verifies 34
  reports, 141 executed blocks and 97 source-matching figures. Removed original
  mepml/org files after verification. Evidence: `build/mepml-conversion/cannon/`.
  Migration: 34/427 converted, 393 pending; next is canopy_analysis. Other sparse
  linear-algebra APIs remain outside this finished batch, alongside the previously
  recorded unrelated audit, CAD policy and optional C++17 limits.

- 2026-11-06: Converted continuous C4.5-inspired induction. Existing inline
  signatures, aggregate layouts and the maximum-ratio selection policy are retained.
  Checked inputs, finite separating midpoints and visited-path prediction remove
  unchecked shape/index access and overflow/rounding failures. Explicit pending
  partitions preserve left-first numbering without recursive construction. Shared
  ordinary/traced training exposes actual rows, candidates, scores and stopping
  reasons; checked whole-call candidate budgets return no partial model. The original
  Gaussian-cloud generator is now a native reusable dataset API with explicit local
  seed/draw order and same-toolchain repeatability. All 69 learner/generator
  definitions and the changed loader file pass fresh XML coverage; fourteen examples
  compile/run and three hovers pass. Twenty-six native cases pass; seventeen tree/rule
  cases pass UBSan. An independent recursive materialized-label trainer matches
  37,324 exhaustive configurations and 1,000 random multifeature/multiclass cases.
  Exact scores/ties, all stopping rules, XOR, duplicates, malformed inputs/paths,
  extreme/adjacent/subnormal thresholds, budgets, ownership and generator boundaries
  pass. The report derives a hand-checkable split, adds native iris five-fold
  evaluation and preserves the original 180-point training plot. It explicitly
  documents the omitted full C4.5 filter, pruning, missing/categorical handling and
  later MDL threshold correction. Four mep blocks, three inspected SVGs and both
  exports pass. Combined runners pass 601 CTest cases; site verifies 33 reports,
  137 executed blocks and 94 source-matching figures. Removed original mepml/org
  files after verification. Both changed areas were already in the finished-paths
  manifest. Evidence: `build/mepml-conversion/c45/`. Migration: 33/427 converted,
  394 pending; next is cannon_analysis. Previously recorded unrelated audit, CAD and
  optional C++17 limits remain outside this batch.

- 2026-11-06: Converted burstsort. The original inline one-argument signature and
  capacity eight are preserved through the compatibility sorting header. A dedicated
  canonical header implements configured sorting, actual nodes/slots/bursts and
  checked prefix reconstruction. Shared index-bucket kernels use flat ownership and
  explicit work stacks; positive capacities and optional complete-call node caps are
  checked, and copy-then-swap preserves caller values on allocation failure. All 62
  definitions are documented, nine examples compile/run and three hovers pass.
  Twenty-three native cases pass, with thirteen sorting cases and the isolated
  allocation checker also passing UBSan. Independent prefix partitions, stable ids
  and burst triggers match 58,593 exhaustive traces; 1,000 random binary arrays,
  all 256 bytes and 103 injected allocation positions pass. A 10,000-byte common
  prefix creates/traverses/destroys 10,001 nodes without recursive ownership.
  The report preserves the original nineteen words and root-slot plot, illustrates
  the actual two-node trie and threshold discontinuity, and distinguishes directory
  scans/comparator calls from unmeasured cache performance. Four mep blocks, three
  inspected SVGs and both exports pass. Combined runners pass 575 CTest cases; site
  verifies 32 reports, 133 executed blocks and 91 source-matching figures. Removed
  original mepml/org files after verification. Evidence:
  `build/mepml-conversion/burstsort/`. Migration: 32/427 converted, 395 pending;
  next is c45_analysis. Other sorting_dist APIs remain outside the finished manifest,
  as do the previously recorded unrelated audit, CAD and optional C++17 limits.

- 2026-11-06: Converted Burrows-Wheeler transformation. The existing inline
  signatures and aggregate layout are preserved, with a safely initialized default
  primary index. Unsigned byte order now matches in both directions, correcting
  high-bit binary round trips; size_t frequencies/starts avoid narrowing and cyclic
  positions avoid overflowing additions. Primary indices are checked before LF or
  output allocation/access. Shared ordinary/traced kernels expose actual rotation
  starts/comparator work, stable occurrence-ranked LF and actual inverse writes.
  Forward budgets throw before excess byte-pair tests; ordinary calls allocate no
  trace records. All 40 definitions are documented, eleven examples compile/run and
  three clangd hovers pass. Twenty-five native cases pass including existing
  compression/formatting cases; sixteen compression cases also pass UBSan. All 9,841
  ternary-byte strings through length eight match independent materialized rotations
  and small prepend/sort inversion. Full byte-alphabet, 1,000 random binary/periodic
  cases, stable occurrence-pair LF, actual counters/write paths, exact budgets and
  malformed primary indices pass. A 200,000-byte unary inverse repeatedly visits row
  zero while writing every byte. Unary forward work reaches 2,400,768 byte-pair tests
  at n=512, verifying B=n*C with actual sort counts. The report preserves banana and
  the original 59-to-20 run result, reuses native RLE/MTF, demonstrates aabb gaining
  runs, explains periodic LF cycles, and distinguishes index validation from image
  membership or corruption detection. Five mep blocks, four inspected SVGs and both
  exports pass; the work title explicitly names byte-pair tests. Combined runners
  pass 552 CTest cases, including the current registry runner; site verifies
  thirty-one reports, 129 executed blocks and 88 source-matching figures. Removed
  original mepml/org files after verification. Evidence:
  `build/mepml-conversion/burrows_wheeler/`. Migration: 31/427 converted, 396 pending;
  next is burstsort_analysis. Unchanged RLE/MTF and other compression documentation,
  and previously recorded unrelated full-audit, CAD policy and optional C++17
  DataFrame limits remain outside this batch.

- 2026-11-06: Converted bully election. The original inline function, result layout
  and valid-input counts remain as the documented highest-responder shortcut; a new
  static fanout API activates every live higher responder once. Shared kernels expose
  actual logical sends, candidate phases, delivery flags and final learned leaders.
  Lean tracing omits only message storage, and complete-call budgets never return a
  partial leader. Checked ids/flags/initiators eliminate unchecked indexing; ignored
  tails and nonzero-byte liveness are retained, with -1 for an inactive system. All
  41 definitions are documented, ten examples compile/run and three clangd hovers
  pass. Twenty-six native cases pass, including existing distributed and formatting
  cases; seventeen distributed cases also pass UBSan. All 9,217 live-initiator choices
  through ten nodes and 1,000 larger random masks match independent formulas, actual
  messages/phases, delivery and learned leaders. A separate delayed FIFO-message
  simulator verifies fanout outcomes, while budgets, recording modes, extra flags,
  malformed inputs and id capacity pass. All-up formulas hold for every initiator
  through 64 nodes. The report preserves the original 6/2/3 counts, contrasts the
  9/3/3 fanout model, corrects shortcut complexity and the immediate-election claim,
  and explicitly omits the original paper's halt/reorganization/recovery machinery.
  Scaling executes 1,048,575 fanout sends at 1,024 nodes versus 3,069 shortcut and
  1,023 top-initiated sends. Four mep blocks, three inspected SVGs and both exports
  pass; visual review adjusted the scaling limits to keep the largest point clear.
  Combined runners pass 527 CTest cases, including the current registry runner;
  site verifies thirty reports, 124 executed blocks and 84 source-matching figures.
  Removed original mepml/org files after verification. Evidence:
  `build/mepml-conversion/bully_election/`. Migration: 30/427 converted, 397 pending;
  next is burrows_wheeler_analysis. Unchanged clock/bakery documentation and previously
  recorded unrelated full-audit, CAD policy and optional C++17 DataFrame limits remain
  outside this batch.

- 2026-11-06: Converted buddy allocation. The canonical inline class retains its
  metadata layout, valid signatures, deterministic order/offset selection, left-child
  splitting and unknown-free no-op policy. Dimension/request/order validation and
  widened class calculation avoid invalid shifts; positive oversized requests safely
  return -1. Ordinary and traced calls share kernels, with actual split paths, XOR
  buddy probes/merges and owned address-sorted free/live snapshots. Host metadata is
  prepared before committing erases, with rollback on allocation failure. All 63
  definitions are documented, sixteen public examples compile/run and three clangd
  hovers pass. Twenty-seven native cases pass, including existing memory/formatting
  tests and isolated failure injection. Seventeen memory cases and the isolated fault
  checker also pass UBSan. All 733 reachable eight-byte states and 13,927 transitions
  match an independent occupancy bitmap; 9,000 larger-arena operations check actual
  traces, exact accounting, maximal coalescence, alignment, snapshot ownership and
  independent copies. All 25 injected host-allocation failure positions across eight
  scenarios preserve state and permit recovery. Invalid dimensions/requests/orders,
  unknown frees, oversized requests and heights through thirty pass. The report keeps
  the original offsets, waste and free-count plot, demonstrates adjacent nonbuddies,
  corrects the minimum-block waste exception and ordered-container complexity, and
  distinguishes logical offsets from backing storage. Four mep blocks, four inspected
  SVGs and both exports pass; the scaling legend was adjusted after visual review.
  Combined runners pass 501 CTest cases, with the current registry runner rebuilt;
  site verifies twenty-nine reports, 120 executed blocks and 81 source-matching
  figures. Removed original mepml/org files after verification. Evidence:
  `build/mepml-conversion/buddy_allocator/`. Migration: 29/427 converted, 398 pending;
  next is bully_election_analysis. Previously recorded unrelated full-audit, CAD
  policy and optional C++17 DataFrame limits remain outside this batch.

- 2026-11-06: Converted integer bucket sort. A dedicated canonical header/source
  retains the by-value signature, stable insertion comparisons and empty/singleton/
  constant shortcuts. The distribution-sort header forwards it. Quotient/remainder
  scaling widens before subtraction and prevents product overflow across the full
  int range and allocation-free bucket counts through SIZE_MAX. Ordinary sorting
  and actual per-bucket snapshots/comparison/shift/store counters share one kernel;
  trace counters throw instead of wrapping. Original paired sample generation and
  a checked matched-load family are native. All 36 affected definitions are
  documented, eight examples compile/run and three clangd hovers pass. Fourteen
  native cases also pass UBSan, including 97,656 exhaustive short arrays, 2,000
  full-range random inputs, independent rational index checks with 20,000 large
  random counts, actual prefix-rank work counts, monotonicity, shortcuts, sampling
  locality, matched-load formulas, validation and generator capacity. All four
  existing distribution-sort cases pass. The report preserves the original small
  array and 100-item occupancy experiment, corrects the claim that occupancy alone
  decides work, and distinguishes fixed-bin independence from observed-range bins.
  At identical occupancies, m=2,048 produces 2,048 versus 2,098,176 comparisons.
  Four mep blocks, three inspected SVGs and both exports pass. Combined runners pass
  474 CTest cases; site verifies twenty-eight reports, 116 executed blocks and 77
  source-matching figures. Removed original mepml/org files after verification.
  Evidence: `build/mepml-conversion/bucket_sort/`. Migration: 28/427 converted,
  399 pending; next is buddy_allocator_analysis. Unchanged pigeonhole/cycle/tree
  documentation and previously recorded unrelated full-audit, CAD policy and
  optional C++17 DataFrame limits remain outside this batch.

- 2026-11-06: Converted Brzozowski automaton minimization. Canonical checked
  automaton operations and powerset construction now share actual stage provenance
  with the ordinary two-reversal algorithm, retaining inline signatures and state
  discovery order. Default DFA safely denotes the empty language (start=-1), with
  its formatter test updated. Whole-language equivalence explores sparse product
  pairs and returns shortest lexicographic witnesses; stage/pair limits throw before
  exceeding the requested budget. Native binary position-language generation shows
  exact exponential intermediate growth. All 63 definitions across four dedicated
  headers are documented, eighteen example programs compile/run and four clangd
  hovers pass. Twenty-eight native cases include existing automata and formatting
  tests; nineteen automata cases pass UBSan. Independent table-filling checks all
  17,626 complete binary machines through three states, 1,000 partial machines and
  500 language pairs. Actual subset/counter invariants, minimality, reachability,
  sparse products, epsilon/zero-alphabet cases, budget boundaries and invalid input
  pass. The report preserves the original six-to-three-to-three counts, corrects
  first-pass minimality for unreachable inputs and derives the exponential family
  reaching 1,024 intermediate states and twelve final states. Four mep blocks, three
  inspected SVGs and both exports pass. Combined runners pass 460 CTest cases; site
  verifies twenty-seven reports, 112 executed blocks and 74 source-matching figures.
  Removed original mepml/org files after verification. Evidence:
  `build/mepml-conversion/brzozowski/`. Migration: 27/427 converted, 400 pending;
  next is bucket_sort_analysis. Unchanged Moore/Hopcroft partition-helper comments
  and previously recorded unrelated full-audit, CAD policy and optional C++17
  DataFrame limits remain outside this batch.

- 2026-11-06: Converted brute-force shortest-path search. A shared explicit-stack
  kernel preserves the original signature, result layout, per-source adjacency order,
  entered-prefix count, strict shortest-path ties and once-per-entry incumbent bound.
  It avoids native recursion depth and validates every used endpoint before graph
  allocation. A common result header documents producer-specific expanded semantics;
  search_more includes the canonical declarations. Typed actual prefixes/incumbents,
  local/global probes, optional path copies and complete-search visit limits expose
  actual work without returning a partial optimum. A checked complete-directed
  generator supports the factorial family. The report keeps the original chain,
  corrects the claim that all goal paths are enumerated, shows the two incumbent
  changes and safe pruning, and derives exact dense-unreachable work with BFS context.
  All 47 affected definitions are documented; seven public examples compile/run and
  three clangd hovers pass. Sixteen native cases also pass UBSan: all 4,165 directed
  graphs through four vertices and 66,129 terminal pairs match independent Floyd
  distances and first-adjacency shortest ties; actual histories/counters, 1,000
  ordered multigraphs, complete-call budgets, invalid inputs and generator size
  constraints pass. A 50,000-vertex chain succeeds with lean records, and the dense
  family reaches 109,601 prefix visits and 876,808 probes with an isolated goal.
  All six existing search-more cases pass. Four mep blocks, three inspected SVGs
  and both exports pass. Combined converted runners pass 432 CTest cases; the site
  verifies twenty-six reports, 108 executed blocks and 71 source-matching SVGs.
  Removed legacy mepml/org files after verification. Evidence:
  `build/mepml-conversion/brute_force/`. Migration: 26/427 converted, 401 pending;
  next is brzozowski_analysis. Unchanged search-more producer documentation and
  previously recorded unrelated full-audit, CAD policy and optional C++17 DataFrame
  limits remain outside this batch.

- 2026-11-06: Converted Bron-Kerbosch maximal-clique enumeration. A dedicated
  canonical inline header retains the original signature and deterministic pivot,
  branch and output order; the original graph-connectivity header includes it for
  compatibility. Union symmetrisation, duplicate/loop treatment, ignored extra rows
  and the empty-graph clique are preserved, while used endpoints/row counts are
  validated before allocation. One shared kernel supplies ordinary output and actual
  preorder R/P/X/pivot frames, emitted-size counts and checked logical probes. A
  positive call budget throws without returning partial output. Native complete-
  multipartite generation supports the exact exponential-output family. The report
  keeps all three original cliques and the histogram, shows why R=[2,1], P=[], X=[0]
  is not maximal, and accounts for explicit member output and snapshot storage.
  All 33 dedicated definitions are documented; five public examples compile/run and
  three clangd hovers pass. Thirteen native cases also pass UBSan: all 33,868 graphs
  through six vertices match exhaustive subset maximality with actual frame/counter
  invariants; 1,000 asymmetric/duplicate/loop inputs, boundary/empty/complete controls,
  limits and generator overflow pass. Multipartite counts reach 6,561 cliques and
  9,841 calls at 24 vertices. Existing 3,000 random clique and 3,000 SCC trials pass.
  Four mep blocks, three inspected SVGs and both exports pass. Combined converted
  runners pass 416 CTest cases; the website verifies twenty-five reports, 104 executed
  blocks and 68 source-matching SVGs. Removed legacy mepml/org files after verification.
  Evidence: `build/mepml-conversion/bron_kerbosch/`. Migration: 25/427 converted,
  402 pending; next is brute_force_search_analysis. Unchanged SCC documentation and
  previously recorded unrelated full-audit, CAD policy and optional C++17 DataFrame
  limits remain outside this batch.

- 2026-11-06: Converted Borwein pi. A shared inline kernel retains the original
  unambiguous one-argument signatures, lower clamp, direct short-run values and
  requested sequence lengths. Stopping after an unchanged stored inverse-pi update
  fixes the old 512-step inf*0/NaN and avoids unchecked late exponent arithmetic.
  Typed actual updates distinguish performed work from copied sequence padding.
  Optional rationalised trace evaluation retains tiny auxiliary y values without
  promising more accurate final pi. A logarithmic helper evaluates the exact-arithmetic
  error model without underflow or a false rounded-result certificate. The report
  keeps the original four-update sequence, corrects last-bit, log-axis and precision-
  cost claims, and separates real convergence theory from double-reference discrepancy.
  All 32 definitions are documented; seven public examples compile/run and three
  clangd hovers pass. Nineteen native cases include eleven focused cases also passing
  UBSan: original values/signature inference, scalar/sequence/actual-prefix contracts,
  positive tiny y after root rounding, negative and INT_MAX requests, padding through
  2048 updates, finite-exception checks, long-double AGM and log-bound range checks.
  Independent 1000-digit Machin, doubled AGM and Decimal calculations verify the
  exact-arithmetic model and both native trace modes. Four mep blocks, three inspected
  SVGs and both exports pass. Combined converted runners pass 403 CTest cases; the
  website verifies twenty-four reports, 100 executed blocks and 65 source-matching
  SVGs. Removed the legacy mepml/org files after verification. Evidence:
  `build/mepml-conversion/borwein_pi/`. Migration: 24/427 converted, 403 pending;
  next is bron_kerbosch_analysis. Previously recorded unrelated full-audit, CAD policy
  and optional C++17 DataFrame limits remain outside this batch.

- 2026-11-06: Converted Bresenham rasterisation. The four-argument inline API
  and original directed inclusive ties are preserved in a shared integer loop;
  differences and errors are widened before arithmetic for all 32-bit int endpoints.
  New actual pre-movement records expose saved errors, axis/joint moves and terminal
  states. An allocation-free uint64 count and optional complete-trace point limit
  reject excessive output before allocation. Ordinary calls omit trace allocation.
  The report retains all fifteen original centres and the steep/diagonal examples,
  derives rational nearest-centre and integer-state invariants, shows reversal ties,
  measures the half-pixel residual and distinguishes output storage from O(1) working
  state. All 22 definitions are documented; six public examples compile/run and three
  clangd hovers pass. Thirteen native cases include eight focused cases also passing
  UBSan: 14,641 exhaustive small endpoint pairs and 1,000 random moderate segments
  match an independent rational oracle; signed extreme short segments, full-range
  counts, preflight limits, axis exchange, reversed ties and all recorded decisions
  pass. Four mep blocks, three inspected SVGs and both exports pass. Combined converted
  runners pass 384 CTest cases; the website verifies twenty-three reports, 96 executed
  blocks and 62 source-matching SVGs. Removed legacy mepml/org files after verification.
  Evidence: `build/mepml-conversion/bresenham/`. Migration: 23/427 converted, 404 pending.
  Next is borwein_pi_analysis, the earliest source still present in the checklist.
  Previously recorded unrelated full-audit, CAD policy and optional C++17 DataFrame
  limits remain outside this batch.

- 2026-11-06: Converted Brent cycle detection. Shared checked Brent/Floyd kernels
  preserve the legacy result layout and transition-call order while exposing exact
  phase/equality counts and the recovered entry without replay. Brent records actual
  movements and no-call checkpoints; ordinary/work entry points retain constant
  storage. Positive global budgets stop before the next excess call in every phase.
  Native rho-map, exact orbit-prefix, decimal-digit and happy-number helpers replace
  report-side logic. The report corrects checkpoint indices to 0,1,3,7,…, derives
  complete call formulas, retains the original graph and six positive digit examples,
  and separates zero and varying cycle entries from the eight-cycle containing four.
  All 56 definitions are documented; sixteen public examples compile/run and three
  clangd hovers pass. Nineteen native cases also pass UBSan: all 3,413 maps through
  five states (16,739 starts), 4,160 controlled tail/period pairs, exact call order,
  per-phase budget boundaries, empty/throwing callbacks, storage/size constraints,
  10,000 string-digit checks, all positive bounded states 1..1620 and 2,000 random
  uint64 classifications. Four mep blocks, three inspected SVGs and both exports
  pass. Combined converted runners pass 371 CTest cases; the website verifies
  twenty-two reports, 92 executed blocks and 59 source-matching SVGs. Removed the
  legacy mepml/org files after verification. Evidence: `build/mepml-conversion/brent/`.
  Migration: 22/427 converted, 405 pending; next is bresenham_analysis. Previously
  recorded unrelated full-audit, CAD policy and optional C++17 DataFrame limits remain.

- 2026-11-06: Converted breadth-first search. A shared FIFO kernel supplies
  the unchanged single-goal result and typed actual dequeue/discovery/queue records,
  plus a full-source traversal with snapshots off by default. Original result layout
  and input-order tie behavior are preserved through a documented common header;
  producer-specific order semantics also describe the existing bidirectional wrapper.
  Validation precedes adjacency allocation. The report keeps the three-vertex chain,
  shows real queue snapshots and first-discovery predecessors, distinguishes pending
  and undiscovered vertices after goal removal, and verifies native grid distances
  and full incidence counts. It states adjacency-copy storage and optional quadratic
  queue-capture costs explicitly. All 36 affected definitions are documented; six
  public examples compile/run and three clangd hovers pass. Seventeen native cases
  compare 66,129 goal searches and 16,585 full traversals with Floyd–Warshall on all
  4,165 directed graphs through four vertices, plus 1,000 shuffled multigraphs in
  both modes, loops/duplicates, invalid inputs, closed-form grids and a 20,000-vertex
  frontier with snapshots omitted. Three mep blocks, two inspected SVGs and both
  exports pass. Combined converted runners pass 352 CTest cases; the website verifies
  twenty-one reports, 88 blocks and 56 source-matching SVGs. Removed legacy mepml/org
  files after verification. Evidence: `build/mepml-conversion/bfs/`. Migration:
  21/427 converted, 406 pending; next is brent_cycle_detection_analysis. Previously
  recorded unrelated full-audit, CAD policy and optional C++17 DataFrame limits remain.

- 2026-11-06: Converted exact knapsack branch and bound. The shared iterative
  take-first kernel records actual decisions, parent links, incumbent changes and
  pruning while retaining KnapsackResult's layout. Exact density cross-products,
  rational fractional bounds and integer floors replace floating-point pruning;
  input validation and checked weight/value/counter arithmetic remove undefined
  or incorrect cases. Signed nonpositive profits are omitted, positive free items
  precede weighted items and original indices resolve density ties. A native fixture
  generator preserves the original MT19937 draw order. All 61 affected definitions
  are documented; nine public examples compile/run and three clangd hovers pass.
  Sixteen native cases include every bound/prune on 55,986 enumerated instances,
  1,000 random signed/free-item instances, 400 existing DP trials, thirteen seeded
  DP checks, near-INT_MAX density/overflow controls, 5,000 free items and a parity-gap
  family with 136,135 visits at eighteen items. All twelve focused cases also pass
  UBSan. The report preserves value 220 and seventeen visits on the original example,
  corrects the full tree to 2^(n+1)-1 and separates node work from timings. Four mep
  blocks, three inspected SVGs, HTML/Markdown and website exports pass. Combined
  converted runners pass 335 CTest cases; the site verifies twenty reports, 85 blocks
  and 54 source-matching figures. Removed legacy mepml/org files after verification.
  Evidence: `build/mepml-conversion/branch_bound/`. Migration: 20/427 converted,
  407 pending; next is breadth_first_search_analysis. Previously recorded unrelated
  full-audit, CAD policy and optional C++17 DataFrame limits remain outside this batch.

- 2026-11-06: Converted Horspool byte search. Public prefix-only shift tables
  and typed actual-alignment records share the existing ordinary/corpus scan;
  comparison-counter overflow is checked. The report contrasts ending and mismatch
  bytes, preserves the original abra example, proves safe shifts and exhibits
  quadratic work even with no matches. Literal Titanic queries agree per record
  with the native naive baseline: five patterns use 30,351 Horspool comparisons
  versus 108,699 naive comparisons. Restored documented direct native reductions
  in summarise_each through a concrete function-pointer overload, preserving
  generic/captured callbacks, null columns and standalone pipe inclusion. All 139
  affected definitions are documented; sixteen examples compile/run and three
  clangd hovers pass. Thirty-six Horspool/Zhu–Takaoka/pipe cases pass, including
  2,047 independent table checks, 129,921 exhaustive text/pattern pairs, 18,000
  random inputs, binary-byte controls and actual corpus counters. Four mep blocks,
  three inspected figures, HTML/Markdown and site exports pass; the existing
  Zhu–Takaoka report also reruns all five blocks with unchanged source/results.
  The combined C++23 runners pass 319 CTest cases; the assembled site verifies
  nineteen reports, 81 blocks and 51 source-matching SVGs. An optional C++17 probe
  fails on pre-existing DataFrame unordered_set::contains calls at lines 3407/3433;
  an isolated unchanged-DataFrame include reproduces the same errors. Required
  C++23 checks pass; the unrelated full-audit/CAD policy limits remain as recorded.
  Removed legacy mepml/org files after verification. Evidence:
  `build/mepml-conversion/horspool/`. Migration: 19/427 converted, 408 pending;
  next is branch_and_bound_analysis.

- 2026-11-06: Converted Boyer–Moore exact byte matching. Public preprocessing
  tables and actual alignment traces share the same kernel as ordinary search;
  removed the report's duplicate implementations. Preserved the original DNA
  example, its six starts and matches at 13/21, and verified its 17 comparisons.
  Added signed border-sentinel and trace-counter guards, exposed negative raw bad
  shifts, and corrected blanket sublinear/linear claims in the header/report/catalog.
  Exact repeated-byte families demonstrate quadratic rechecking of overlapping
  full matches and distinguish unit-shift misses from absent-byte skips. All 27
  affected definitions are documented; six public examples compile/run and two
  clangd hovers pass. Eighteen native cases include 2,047 direct overlap-oracle
  suffix-table checks, 129,921 exhaustive text/pattern pairs against string::find,
  6,000 random inputs and embedded-NUL/high-byte controls. Four mep blocks, three
  inspected SVGs, HTML/Markdown and website exports pass. The combined converted
  runners pass 283 CTest cases; the site verifies eighteen successful reports,
  77 blocks and 48 source-matching figures. Removed only this report's legacy
  mepml/org sources and assets, retaining Horspool for its own next conversion.
  Evidence: `build/mepml-conversion/boyer_moore/`. Migration: 18/427 converted,
  409 pending; next is boyer_moore_horspool_analysis. Previously recorded unrelated
  full-audit and CAD Python-tab policy failures remain outside this batch.

- 2026-11-06: Converted Boruvka minimum spanning forests. The shared native
  kernel now exposes canonical component labels, original-index cheapest choices,
  actual accepted merges and whole-edge-scan counters while preserving the legacy
  API and tie order. Documented shared edge/result types, checked widened weight
  totals and added reusable dyadic_path_edges. The report retains the four-vertex
  example, shows its actual merges and corrects halving for disconnected graphs,
  vertex-array work and trace storage. All 36 affected definitions are documented;
  eight public examples compile/run and two clangd hovers pass. Sixteen native
  cases include independent dense Prim/reachability checks of all 33,868 simple
  graphs through six vertices, 1,000 shuffled multigraphs, negative weights, loops,
  ties, invalid inputs, weight overflow and tight phases through 1,024 vertices.
  Four mep blocks, three inspected SVGs, HTML/Markdown and website exports pass.
  The combined converted-report runners pass 265 CTest cases; the assembled site
  verifies seventeen successful reports, 73 blocks and 45 source-matching figures.
  Removed the legacy mepml/org files after verification and repointed the catalog.
  Evidence: `build/mepml-conversion/boruvka/`. Migration: 17/427 converted, 410
  pending; next is boyer_moore_analysis. Existing unrelated full-audit and CAD
  Python-tab policy failures remain as previously recorded.

- 2026-11-06: Converted Booth multiplication and bootstrap resampling. Booth's
  widened shared signed-digit loop provides the legacy exact product/counts and
  all 32 real accumulator decisions; the report corrects the reversed bit-pair
  table and checks runs, alternating bits and an exact binomial count law.
  Nine native cases and the focused UBSan suite check 65,536 signed-byte pairs,
  300,000 full-width pairs and prefix invariants. All 23 changed Booth definitions,
  five compiled examples and two hovers pass. Bootstrap gains a native median
  and a shared bootstrap_summary API with finite checks, widened centred sums,
  scaled acceleration and ordered percentile fallback across singular BCa cases.
  Corrected its 1987 paper DOI and the conditional mean-SE formula; all eight
  public examples compile/run. The 215-definition scoped audit and fourteen
  hovers pass. Bootstrap/resampling and vector regression targets pass 51 tests,
  including extreme units, finite endpoints, ties, RNG compatibility and caller
  stream advancement. Independent Python calculations of the actual iris
  replicates agree on bias/SE and all four intervals within 2.55e-10. Together
  the converted-report runners pass 249 CTest cases. Both reports pass eight
  mep blocks, six inspected SVGs and HTML/Markdown exports. The assembled site
  verifies sixteen successful reports, 69 blocks and 42 source-matching SVGs.
  Legacy mepml/org sources are removed after each verified conversion. Evidence:
  `build/mepml-conversion/booth/` and `bootstrap/`. Migration: 16/427 converted,
  411 pending; next is boruvka_analysis. Previously recorded unrelated full
  documentation-gate and CAD Python-tab policy failures remain outside this batch.

- 2026-11-06: Converted blossom matching and Blum–Blum–Shub generation. A shared
  Edmonds kernel now exposes actual expanded odd sets, original-vertex lifted
  paths, post-augmentation mate maps and logical counters. Added the pre-allocation
  INT_MAX check, correct duplicate-edge work bound and unmatched-sentinel example.
  Independent subset enumeration verifies all 33,868 simple graphs through six
  vertices plus 1,000 shuffled cases through twelve, including nested blossoms.
  BBS now exposes the same individual-bit transition used by word generation,
  consuming bit records and nonmutating bounded state orbits. Its report preserves
  nibble c, exposes repeated whole words, proves exact small seed-period counts,
  and documents structural validation versus caller primality assumptions and
  fixed-width teaching scope. Overflow-free add/double and Python integer oracles
  verify near-64-bit moduli, 1,920 transitions and thirty packed words. Eight report
  blocks, four inspected figures, HTML/Markdown and website exports pass. Blossom
  covers 28/28 Doxygen definitions; the new BBS files cover 14/14 and the changed
  shared engine class brings its precise symbol scope to 23/23. The remaining
  shared random-file backlog is not marked complete. All fifteen public examples
  compile/run and four clangd hovers pass. Native targets pass seven and thirty
  cases; all fourteen report runners plus dir rebuild and pass 216 CTest cases.
  Regenerated member help is deterministic. The site verifies fourteen successful
  reports, 61 blocks and 36 source-matching SVGs. Legacy sources are removed after
  each report passes. Evidence: `build/mepml-conversion/blossom/` and `bbs/`.
  Migration: 14/427 converted, 413 pending; next is `booth_analysis`.

- 2026-11-06: Converted BLAST and Bloom filters to the new report collection.
  Canonical indexed seed visitation and ungapped extension expose actual byte pairs,
  strict X-drop states, checked scoring/coordinates and int64 score arithmetic.
  The BLAST report retains the original DNA hits and adds missed-seed and cubic-work
  controls. Bloom filters now pack bits, maintain exact occupancy, expose canonical
  probes and validate dimensions; the FNV helper uses the RFC 9923 basis, explicitly
  requiring retained old string hashes to be rebuilt. An original-header control
  verifies unchanged integer-key membership/fill across 10,240 insertion states
  and 1,310,720 queries. Ten mep blocks, six inspected figures, HTML/Markdown and
  website exports pass. Scoped Doxygen covers 54/54 definitions, seventeen public
  examples compile/run and four clangd hovers pass. Native report targets pass
  17 and 16 cases, including exhaustive small seeded intervals, word boundaries,
  boolean-array occupancy, duplicates/copies and official FNV vectors. Refreshed
  generated dm::dir tables include the new inspection records; the generator omits
  empty trailing fields to avoid source whitespace errors, with deterministic output
  and unchanged parsing. All twelve report runners plus five normal GCC dir cases
  pass 179 CTest cases. A separate clang dir run has four passes and one existing
  compiler-specific expectation failure (DenseMatrix<> versus DenseMatrix<double>);
  type-name parsing is unchanged. The site verifies twelve successful reports,
  53 executed blocks and 32 source-matching SVGs. Evidence: `build/mepml-conversion/`
  under `blast/` and `bloom/`. Migration: 12/427 converted, 415 pending; next is
  `blossom_matching_analysis`. Existing full documentation-gate and CAD Python-tab
  policy failures recorded below remain outside this batch.

- 2026-11-06: Converted Bitap, bitonic sorting and BKM to the new report collection.
  Coupled Shift-And states and overlapping Hamming matches share the canonical
  scanner; bounded budgets preserve legacy start/end conventions. Bitonic sort,
  topology and actual integer events share an allocation-free comparator visitor,
  with explicit arbitrary-length merge invariants and dependency stages. BKM now
  checks its real finite domain and exposes actual accepted/rejected factor traces;
  the report corrects the ln(3) plateau and distinguishes truncation from rounding.
  Twelve mep blocks, seven inspected figures, HTML/Markdown and website exports pass.
  Scoped Doxygen covers 80/80 definitions, all 22 public examples compile/run and
  six clangd hovers pass. Native targets pass 15, 12 and 11 tests; controls include
  exhaustive small bit states, all 131,071 binary inputs through sixteen sorting
  wires, dense elementary-function sweeps, exact binary boundaries and every BKM
  prefix. Independent window/suffix and 100-digit decimal oracles check 116 matches,
  1,692 bit states and 2,279 BKM states. All ten converted-report runners rebuild
  together and pass 141 CTest cases. The assembled site verifies ten successful
  reports, 43 executed blocks and 26 source-matching SVG figures. Legacy mepml/org
  sources are removed after each report passes. Evidence: `build/mepml-conversion/`
  under `bitap/`, `bitonic/` and `bkm/`. Migration progress: 10/427 converted,
  417 pending; next is `blast_analysis`. Existing full-gate documentation and CAD
  Python-tab policy failures recorded below remain outside this batch.

- 2026-11-06: Converted binary splitting, Birkhoff interpolation and bisection.
  The new core modules expose exact product-tree nodes, derivative constraint
  matrices/evaluation and canonical bracket histories. Added a checked 21-term
  uint64 boundary, row/column-scaled pivoting, separate coordinate/residual
  stopping controls, finite-value guards and overflow-safe midpoints. Existing
  function signatures and result layouts remain; stricter error contracts are documented.
  The three report runners pass 14 mep blocks, eight inspected figures and
  HTML/Markdown exports; their native CTest targets pass 10, 20 and 17 cases.
  All eighteen public examples compile/run, six clangd hovers pass and scoped
  Doxygen covers 79/79 definitions. Independent Fraction/100-digit Decimal checks
  verify 21 exact series roots/tail bounds, 504 interpolation systems and forty
  exact dyadic bisection brackets/signs/root-distance bounds. Legacy sources are
  removed only after verification. All seven conversion runners rebuild and pass
  103 CTest cases together; the assembled website has seven successful reports,
  31 executed blocks and nineteen source-matching SVG figures.
  Evidence: `build/mepml-conversion/` under
  `binary_splitting/`, `birkhoff/` and `bisection/`. The migration now has seven
  completed conversions and 420 pending; next is `bitap_analysis`.

- 2026-11-06: Converted B* multi-goal search, bilinear interpolation, binary GCD
  and binary search into the newly requested `docs/examples/mepml/` collection.
  Added canonical search/GCD traces, finite/overflow guards for informed search,
  checked/stable bilinear blending, and accurate boundary/closed-interval contracts.
  The new standalone headers and implementations cover 63/63 Doxygen definitions;
  all sixteen public examples compile/run and seven clangd hovers pass.
  Seventeen report blocks, eleven rasterised/reviewed figures and HTML/Markdown
  exports pass. The local installed-library CMake/CTest runners pass 56 tests,
  including 600 Floyd–Warshall graph cases, 2,000 independent interpolation cases,
  all 65,536 byte GCD pairs, 20,000 full-width Euclidean GCD comparisons, exhaustive
  sorted duplicate sequences and exact search-tree probe sums. Both documentation
  collections retain stable site URLs and actual source links (19 website tests).
  One unrelated existing policy test still fails because it forbids the intentional
  Python tabs in `docs/mepml/examples/cad/seeed_devarm.mepml`; it is not counted as
  passing. The full completed-path gate has 35 existing undocumented fields in
  compute/cuda.cpp and sim/world.cpp; all 63 definitions in the new modules pass the scoped
  check. Legacy mepml/org sources are removed only after validation. Evidence:
  `build/mepml-conversion/`; the remaining conversion checklist starts with
  `binary_splitting_analysis`.

- 2026-11-02: Reimplemented the articulated sorting capabilities in the independent
  native solver: full COM inertia, compounds/ellipsoids/STL support hulls, strict
  reusable URDF import, fixed joints, bounded implicit drives, reflected rotors,
  coupled bilateral/Schur projection, mimic gearing and angular friction. Free
  rotation uses midpoint Euler/Cayley integration. Standard CPU/CUDA packages
  have no MuJoCo dependency; the reference backend remains explicit and optional.
  Native tests include rapid-tumble energy/world-momentum conservation, authored
  inertia-frame rotations, effort/inertia/load sharing, actual opposing-jaw
  frictional lifting, conveyor/bin force feedback, snapshots and import rollback.
  Scoped Doxygen covers 1072/1072 definitions; six representative clangd hovers
  and compiled public API examples pass. The no-MuJoCo suite passes 126 tests
  with 14 optional reference/GPU skips; allocation checks retain zero steady-state
  allocations. Evidence is in `build/native-physics/`. The new 20-second native
  transfer video is explicitly not native training: six arrivals, zero correct,
  three missed, three pending, maximum penetration 1.35 mm, bounded effort.
  The playground has separate native training/cache fingerprints; the earlier
  completed MuJoCo comparison remains in `robot_fruit_mujoco.mepml`.


- 2026-11-02: Audited the robot stream's kinematic grasp, release and analytic
  training assumptions. Added optional MuJoCo articulated contact dynamics with
  pinned URDF masses/inertias and official collision assets, free rotating fruit,
  effort-limited servos, sustained bilateral grasp forces and force/dwell-based
  bin scoring. Measured joint/body poses feed rendering; strict commanded joint
  setters retain their limits. Full implicit integration fixes a reproduced
  fast-tumbling asymmetric-body instability; numerical warnings reject a run.
  Scoped Doxygen covers 430/430 definitions, twelve clangd hovers and four new
  compiled/runnable public examples pass. The native simulation suite passes
  115 tests with eight optional GPU skips. Evidence is in `build/physics-audit/`
  (`coverage.json`, `hover.json`, `doc-examples.cpp`, `final-native-tests.log`,
  `gyro-tests.log`, `reproduce6-implicit.log`). The physics audit explicitly
  distinguishes nominal parameters and numerical checks from hardware calibration. The
  completed installed MuJoCo 3.4.0 run retrains four 256-window controllers;
  independent RGB validation selects dueling Double DQN window 256 before test.
  Held-out counts are 111 correct, 8 wrong, 73 missed and 15 pending from 207
  arrivals. The final 20-second/300-frame CUDA video fully decodes and all 353
  decisions independently replay from its NPZ. The compiled Mep block passes
  in 206.32 s with HTML/TeX exports. Raw physics limits, model hashes, per-stream
  results and timing evidence are in
  `benchmarks/results/fruit_render/physics_fidelity.json`.


- 2026-11-02: Added the installed native `datamunge::rl` discrete pipeline API:
  masked Q(λ), Double DQN, dueling Double DQN and clipped categorical PPO.
  Owned observations and explicit terminal/truncation transitions preserve
  elapsed-time discounts; replay targets, GAE and exact forced-chain composition
  are checked analytically. Native autograd/Adam learners improve a masked bandit;
  checkpoint round trips and rejection of incomplete/nonfinite NPZ data pass.
  A borrowing fruit-stream adapter exposes 25 bounded tracked features and an
  independently seeded arrival curriculum, preserving callbacks and physical
  motor configuration. The playground compares matched fresh training budgets
  with separate physical screening, actual-RGB selection and held-out groups,
  retaining the historical controller as a separately labelled reference.
  Scoped Doxygen covers 345/345 definitions; eleven clangd hovers and 33 compiled
  public examples pass. The simulation suite passes 107 tests with eight optional
  GPU skips. Evidence: `build/rl-benchmark/coverage.json`, `hover.json`,
  `doc-examples.log`, `tests.log`, and the report's exported `rl_*.csv` artifacts.
  Final installed CUDA replay passes its one code block in 157.855 s and exports
  HTML/TeX. Validation ties the three neural methods at 217 and selects Double
  DQN window 256 by the declared tie rule; held-out results are 181 correct,
  8 wrong, 35 missed and 10 pending from 234 arrivals. Q(λ)'s held-out weighted
  score is slightly higher (634 versus 631), explicitly reported without
  reselection. The 20-second/300-frame video fully decodes; 20 artifact links
  and all checkpoint copies match exports. All 357 video decisions independently
  reproduce from the selected NPZ. `benchmarks/results/fruit_render/rl_pipelines.json`
  retains full results, hashes, protocol limits and final timings.



- 2026-11-01: Added the native continuing `sim::FruitStreamEnvironment` and
  duration-aware hierarchical Q-learning. Multiple seeded fruit arrive under an
  independent Poisson clock while the belt and real-URDF servos continue; there
  are no between-pick resets, source backpressure or stopped-belt shortcuts.
  Capacity loss is counted, outcomes require settled physical bin support, and
  completed fruit clear after two seconds to bound retained state. The scheduler
  chooses waits and oldest/second-oldest interception gates over a finite queue/
  urgency/spacing/arm-side abstraction. Fixed IK options retain explicit ideal
  grasp, limited collision and privileged-geometry assumptions. An opt-in analytic
  timing/transport model accelerates training; recording on that proxy is rejected.
  Training-channel classification noise is calibrated from independent validation.
  New camera actor rendering/component association supports real multi-fruit RGB;
  a tracked projected centre associates an RGB-derived component without using
  semantic class for routing. Empty perception remains unknown. `cad::draw_text`
  reuses the native bitmap font for live video counters without external assets.
  Archived the previous three experiments in `robot_fruit_episodic.mepml` and
  replaced the main playground with a complete continuing training/evaluation/
  recording program. MobileNet adapts on 512 moving crops (prototypes 0–3),
  validates on 128 (prototype 4), and scores 121/128 on test prototype 5; all
  stream seed groups and balanced quotas are audited. Forty epochs select epoch
  13. Q-learning runs 8,192 continuing windows, totaling 993,947.28 proxy seconds
  (276.10 simulated hours); independent full-physics validation selects window 512
  after later checkpoints plateau. Across sixteen paired 120 s actual-RGB streams,
  learned/early-oldest/late-oldest achieve 155/149/97 correct sorts and 66/73/133
  misses, with 246 identical arrivals and 12 pending fruit for every method.
  The unselected fixed-seed video is exactly 60 s: 900 frames at 15 Hz, no cuts,
  time scaling or resets, actual frozen RGB inference and full native physics.
  Its four arrivals produce three correct sorts and one miss. Scheduled births
  are 7.0163, 36.2092, 44.8636 and 45.4906 s, preserving the Poisson lull and burst.
  Frame/counter conservation, servo limits and exact-window perception counts
  are verified. Maximum recorded joint/finger rates are 1.49983 rad/s and
  .079899 m/s. Native simulation tests pass 79 cases (eight optional GPU cases
  skip), classifier tests pass three, and all 24 affected public header examples
  compile/run. Doxygen covers 359/359 selected definitions and 10 clangd
  hovers pass. Native full learning completed; `DATAMUNGE_STREAM_REPLAY=1 just
  playground` then passes its code block in 95.196 s and exports HTML/
  TeX. Training is the default code path; explicit replay preserves its completed
  checkpoints and evaluation evidence. All 22 linked artifacts resolve and match
  exported copies byte-for-byte; the 960x640 H264 Baseline/YUV420P video decodes
  completely. Evidence: `build/fruit-stream-verification.txt`,
  `build/fruit-stream-audit.json`, `build/playground/run.json`,
  `build/debug/docs/fruit-stream-coverage.json` and
  `build/debug/docs/fruit-stream-hover.json`.

- 2026-11-01: Extended the conveyor's opt-in `FruitVariation::Diverse` protocol
  with six geometry prototypes per class, shared green/yellow/red/brown palettes,
  seeded mottled textures, scale/roughness/leaf variation, full-turn yaw and small
  pitch/roll tilts. Random physical belt entry and across-belt offsets are retained
  through contact-driven arrival. Recorded quaternion/scale determine rendering;
  proxy dimensions and rotated payload collision bounds follow the sampled fruit.
  Default Basic behavior and its earlier checkpoints remain available. Diverse
  manipulation adds five observed lateral bins before grasp, a shared carrying
  table, size-dependent jaws and short-axis wrist alignment. `reset_feed()` and
  `advance_feed()` expose actual feeding separately from efficient training resets.
  The wrist camera adds configurable FOV and `inspect()`: registered empty-scene
  RGB subtraction produces a visible foreground box and a padded classifier crop,
  without using depth/object IDs or class predictions to determine the box.
  Video annotations never enter the model. This is calibrated synthetic
  localization rather than a learned detector. Optional seeded grayscale training
  augmentation discourages a color shortcut. The playground now trains on 1,024
  images from prototypes 0–3, validates on 256 from prototype 4, and tests on 256
  from prototype 5, with balanced classes/prototypes and disjoint seeds. Forty
  epochs select epoch 38 by validation loss: RGB accuracy 241/256 and grayscale
  236/256. Both motor policies train for 8,000 episodes across all six prototypes;
  all 256 test fruits reach the predicted goal, giving 126/128 correct apple sorts
  and 115/128 correct banana sorts. The report includes all twelve prototypes,
  unannotated crop samples, learning curves, confusion counts and every evaluated
  outcome. Its 778-frame, 24 fps video shows four selected successful held-out
  episodes with live green boxes and prediction borders; feeding is shown at 4x
  speed and manipulation at real time. Rotation stabilization, privileged motor
  geometry and ideal grasp/collision assumptions remain explicit. Native checks
  pass 72 simulation cases (eight optional GPU cases skip) and three classifier
  cases. All 30 public header examples compile/run with their documented asset
  prerequisites; Doxygen covers 281/281 definitions and eleven clangd hovers pass.
  `just playground` passes all three blocks in 1348.4 seconds and exports
  HTML/TeX. Evidence: `build/playground/run.json`,
  `build/fruit-variation-verification.txt`,
  `build/debug/docs/fruit-variation-coverage.json`, and
  `build/debug/docs/fruit-variation-hover.json`.

- 2026-11-01: Added a fixed real-URDF wrist RGB camera and visible blue housing,
  a trainable native `nn::MobileNetClassifier`, and caller-selected Red/Yellow
  manipulation goals. Camera extrinsics depend only on the end-link transform;
  background caching still renders independent payload motion and arm occlusion.
  The classifier uses all thirteen MobileNet-v1 depthwise/pointwise blocks at
  width .125, with 57,738 learned parameters and complete weight/BN inference
  checkpoints. The third playground block generates 256 training, 64 validation
  and 128 separate-seed test images, trains twenty epochs, and selects epoch 17
  using validation loss. It loads the earlier red policy and trains yellow with
  native Q-learning. Predicted class alone chooses the bin and policy; ground
  truth is used for labels and scoring. Final evaluation classifies 128/128 and
  sorts 54/54 apples into red and 74/74 bananas into yellow. A 242-frame,
  24 fps video includes the moving camera and live wrist-view inset; a colored
  display-only border shows the latched initial prediction. Saved artifacts
  include the MobileNet/BN checkpoint, yellow Q table, image manifests, every
  evaluation outcome, trajectories, loss curves and confusion matrix. Results
  explicitly concern synthetic CAD imagery and the existing simplified grasp/
  collision model. Four new simulation tests cover rigid camera extrinsics,
  metadata independence, payload updates despite caching, explicit goal selection
  and physical yellow-bin placement. Two NN tests cover fitting, inference
  restoration, missing moments and validation before mutation. The simulation
  runner passes 69 tests (eight optional GPU cases skip); the NN runner passes
  two. All seventeen public examples compile/run, scoped Doxygen covers 172/172
  definitions, and seven relevant clangd hovers pass. The complete `just playground`
  run passes three blocks and exports HTML/TeX in 308.4 seconds. Evidence:
  `build/playground/run.json`, `build/fruit-vision-verification.txt`,
  `build/debug/docs/fruit-vision-coverage.json`, and
  `build/debug/docs/fruit-vision-hover.json`.

- 2026-11-01: Added `sim::DevArmPickEnvironment`, a red-bin-only manipulation
  task using the real B601-DM URDF, bounded/rate-limited Cartesian IK servos,
  an ideal stabilized proximity grasp, and physical post-release settling.
  Both bins are reachable; tool/payload guards prevent crossing workcell solids.
  Full arm self-collision and torque/contact-grasp dynamics are explicitly outside
  this first model. Canonical tabular Q-learning trains from task transitions;
  `PickPolicy` saves/loads validated finite action-value checkpoints. The second
  playground block trains 2,000 episodes, evaluates a separate environment and
  128 separate seeds, and replays the final reloaded policy's first successful
  apple/banana without exploration or scripted corrections. The final run places
  67/67 apples and 61/61 bananas into red; the random-action baseline is 0/64.
  It exports an 8.75-second, 210-frame H.264 video, a native learning-curve SVG,
  a poster, the checkpoint, all evaluation/training outcomes and joint/fruit
  trajectories. All links are copied into the HTML export's asset directory.
  Six offline regression tests use a licensed, geometry-free fixture preserving
  the actual URDF joint tree. The native simulation runner passes 65 tests;
  eight optional GPU tests skip. Seven public API examples compile/run, including
  the documented 4,000-episode training call. Scoped Doxygen covers 155/155
  definitions and three new clangd hovers pass. `just playground` passes both
  blocks (159.8 seconds) and exports HTML/TeX. Evidence:
  `build/playground/run.json`, `build/devarm-pick-verification.txt`,
  `build/debug/docs/devarm-pick-coverage.json`, and
  `build/debug/docs/devarm-pick-hover.json`.

- 2026-11-01: Added the native `sim::ConveyorEnvironment` and the first robot-arm
  playground block. A seeded apple/banana is driven by tangential tread friction,
  stops at a pick sensor and waits beside two physical sorting bins. Observations,
  placement outcomes and guarded next-fruit spawning provide a foundation for an
  external manipulation controller. `Body::surface_velocity` leaves collision
  geometry stationary and wakes nearby payloads when the tread changes. The report
  imports the pinned real DevArm meshes, records a 240 Hz simulation at 24 fps,
  generates an eight-second baseline H.264 MP4, and embeds it with a download link
  and PNG poster. The arm is a parked visual; grasping and arm collision proxies
  are explicitly deferred. Native tests cover both fruit classes, deterministic
  replay, braking/holding, sleeping-payload restart, placement/next-fruit gating,
  invalid inputs and independent scenes. The simulation runner passes 59 tests;
  eight optional GPU tests skip because those backends are not enabled. All new
  public examples compile/run, scoped Doxygen covers 194/194 definitions, and
  three representative clangd hovers pass. The full `just playground` build runs
  its one block and exports HTML/TeX with a playable video element. Evidence:
  `build/playground/run.json`, `build/debug/docs/conveyor-coverage.json`,
  `build/debug/docs/conveyor-hover.json` and `build/conveyor-verification.txt`.

- 2026-11-01: Moved the HP research manuscript to the standalone local project
  `~/papers/hp_splines/hp_splines.mepml`, together with its fifteen figures and
  eighteen outcome/protocol files. Its flake reuses the canonical datamunge C++
  shell, locks the current local implementation, and gives every block the
  project-relative `.#cpp` shell reference. Added full/cached-export just recipes,
  the earlier `playground` alias, dependency refresh instructions and build ignores.
  The standalone full build passes six blocks (514.7 seconds), exports HTML/TeX,
  and reproduces every rendered figure and outcome/protocol byte. All fifteen
  image links and data links resolve. No other report references the moved assets.
  The library retains a clean scratch playground; its zero-block build and three
  native-diagram report checks pass. Evidence:
  `build/hp-paper-project-verification.txt` and the standalone `build/run.json`.

- 2026-11-01: Added an early Steps example with grow data, truth and the
  selected HP curve above a panel of activated and active-support knot ticks.
  Three cubic roots share a 512-span finest grid and each grow 48 splits;
  positive lambda is retuned on every mesh. Independent tuning selects a
  fourteen-split, depth-seven, 63-coefficient fit, with midpoint signal RMSE
  0.02528 at noise SD 0.10. Truth locations are display/assessment information,
  never refinement inputs. Archived data, predictions, knots, root comparisons,
  full path, lambda trials and protocol accompany the complete embedded C++.
  Expanded the depth guard to 0..9 while retaining the 1024-coefficient cap;
  fifteen CTest cases pass, including new depth 7/8/9 affine/grid-budget checks.
  The changed module remains 100/100 documented, and five clangd hovers include
  the updated depth contract. Independently reconstructed the selected tree,
  active functions and every knot tick; checked tuning decisions, sample counts
  and ISE. Earlier experiment CSVs remain byte-identical. `just playground`
  passes six native blocks (496.9 seconds), exporting HTML/TeX with fifteen
  figures, zero tables and resolving all figure/data links. Three native-diagram
  report checks pass. Evidence: `build/hp-steps/`.

- 2026-11-01: Replaced the playground's thirteen rendered result tables with
  native graphical summaries. Five new SVGs show paired mean-ISE ratios,
  search effort/model size/zero-penalty rates, and every observed-data split
  loss with its arithmetic mean and median. The exact example now includes
  a plotted greedy-candidate audit. Multi-panel figures use `fit_panels()`;
  browser-like renders were visually checked. Ratio intervals use the core
  bootstrap API with 10,000 paired index resamples, seed 2026110201; the paper
  distinguishes these percentile intervals from earlier quoted absolute-MCSE
  intervals and retains rare-event caveats. Independently recomputed all 92
  plotted mean ratios and 184 bootstrap endpoints. All seven original outcome
  CSVs are byte-identical; source digests and a new figure-effects CSV record
  the presentation change. `just playground` passes five native C++ blocks
  (469.7 seconds), exports HTML/TeX with zero tables and fourteen images, and
  all image/archive links resolve. Three native-diagram report checks pass.
  Evidence: `build/hp-figures/figure-audit.log`, `export-audit.txt`, and
  `just-playground.log`.

- 2026-11-01: Added `fit_joint_adaptive_pspline()` and analytic spline
  derivatives. The joint search retunes lambda on every accepted mesh, probes
  between seed multipliers, and supports either positive local-MSE-gain priority
  or spanwise integrated curvature priority. Split gains compare the same lambda
  before/after refinement; path and tuning audits disclose subsequent retuning.
  Mesh-dependent matrices are reused across multipliers, with sparse transfer
  supports avoiding zero products. The fixed-path baseline's complete stdout
  and three archived CSVs remain byte-identical. The revised playground retains
  that baseline and adds 600 fresh datasets / 3,600 selected fits plus an
  explicitly diagnostic 30-split Old Faithful revisit. Joint HP has lower mean
  ISE than fixed HP in 18/20 conditions; curvature is not consistently preferable,
  and global P still wins the observed-data mean comparison. Finite basis bounds,
  interpolation/feedback confounding and candidate failures are disclosed.
  Fourteen native tests pass, including an independent per-mesh lambda oracle,
  matched-lambda replay, exact derivative/curvature checks and recursive priority;
  address/undefined-behavior sanitizers also pass with leak checking disabled.
  The changed module has 100/100 documented definitions, eleven compiled public
  examples and four verified clangd hovers. All 360 printed follow-up mean/MCSE
  and paired statistics were independently checked. `just playground` passes all
  five blocks (470.1 seconds), prints results and exports HTML/TeX; nine local
  figures and all archive links resolve. Evidence: `build/hp-joint/` and
  `build/playground/run.json`.

- 2026-10-31: `just playground` now runs the manuscript through native
  `mep-mepml build --run`, prints refreshed block results, and exports HTML/TeX
  with local figures under `build/playground/`. The full Nix library build and
  all four C++ blocks pass (102.2 seconds for block execution); all six HTML
  image links resolve, archived CSV bytes are unchanged, and protocol source
  hashes still match. Evidence: `build/playground/run.json` and
  `build/hp-paper/just-playground.log`.

- 2026-10-31: Wrote the playground as a reproducible research manuscript on
  continuous hierarchical P-splines. Background distinguishes B/P/HB/THB,
  T-splines, LR splines and earlier adaptive statistical smoothing, using ten
  verified references. Added the canonical C++ `adaptive_pspline` module:
  globally fitted continuous HB/THB spaces share a fixed finest-grid penalty;
  Greville-site divided differences preserve physical affine trends at clamped
  boundaries. Both unpenalized and penalized growth choose the largest positive
  counterfactual parent-local training-MSE drop, audit every proposal, and let
  independent validation select root, lambda and path prefix. The paper proves
  fixed-space identifiability, coordinate invariance and objective monotonicity
  without claiming a global tree/risk optimum or field-wide novelty.
  The disclosed grid covers 480 independent simulated datasets and 2,400
  selected fits, plus 30 Old Faithful splits and 150 observed-data fits. Paired
  MCSEs, failures, search counts and tail outcomes are retained; HP improves
  mean ISE over global P in 13/16 conditions but loses the observed-data mean
  comparison. Four complete embedded C++ programs reproduce captured stdout,
  six native SVGs and all archived per-replicate CSV bytes. Nine CTest cases
  pass, also under address/undefined-behavior sanitizers with leak detection
  disabled; tests include an independent augmented-QR oracle, exact solution,
  affine preservation, basis invariance, candidate replay and validation leakage
  checks. The new header/source have 68/68 documented definitions, seven compiled
  API examples and two clangd hovers. Native parsing and HTML/TeX exports pass;
  direct native PDF compilation encounters existing SVG/LaTeX export limitations.
  Evidence: `build/hp-paper/`; compact outcomes and source digests are archived
  under `docs/mepml/examples/results/hp_*`.

- 2026-10-31: Moved the spline comparison into
  `docs/mepml/examples/stats/regression/hierarchical_splines.mepml` and the
  complete Seeed CAD workflow into `docs/mepml/examples/cad/seeed_devarm.mepml`.
  Each report retains its C++/Python/R tabs, cached results and six figures.
  Figure assets move beside their reports without byte changes; spline figure
  names now describe the comparison/refinement rather than the playground.
  Nix shell, interpreter, R library and robot-cache paths are rebased to the
  repository root. All six fence commands run from their new directories and
  reproduce the cached stdout and images; the DevArm programs generate usable
  local glTF exports there. Root/docs indexes and the CAD guide point to the
  new reports. Playground now contains only its title, export and language
  metadata, ready for new examples. Native parsing and HTML/TeX exports pass
  for all three documents. No core or binding changes.
  Evidence: `build/playground-split/`.

- 2026-10-31: Expanded the playground spline comparison to include an ordinary
  least-squares cubic B-spline with 40 coefficients, using the same interior
  breakpoints and cubic spline space as the P-spline. Boundary knot conventions
  are explained; zero-penalty P-spline predictions agree with the baseline. Each C++/Python/R tab also saves a native two-panel HB/HP
  refinement figure from the fitted objects' cell bounds, depths and mapped
  knot sequences. The figures distinguish the 10-coefficient global patch
  (seven spans), the unfitted routing cell and the 32-coefficient local patch
  (29 spans). Prose explains that HB and HP share their mesh and degree, with
  HP changing the coefficient penalty. All three programs run and produce
  identical tables and SVG bytes for both figures; B-spline signal RMSE is
  0.01567. The CAD workflow is preserved byte for byte. No core or binding
  changes are needed. Native parsing and HTML/TeX exports preserve all twelve
  figures and both tab groups, including captions/alt text for the meshes.
  Evidence: `build/playground-refinement/`.

- 2026-10-31: Merged the playground CAD examples into one complete workflow
  per C++/Python/R tab. Each program obtains the pinned Seeed assets with native
  `datasets::fetch("rebot_devarm")`, imports and poses the real arm, imports the
  separate STEP link, exports both as glTF, checks reopened geometry counts and
  renders the exports. The report retains the independent spline example and
  cached figures; generated model links point into its local ignored build
  directory. The catalog adapter verifies every cached/downloaded file by size
  and SHA-256, retains the source license and honors shared cache/offline/mirror
  controls without optional data-preparation dependencies. The private native
  fingerprints match all 38 entries of `examples/cad/rebot-assets.json`.
  All six report programs pass with matching stdout, PNGs and glTF bytes across
  languages; native HTML/TeX exports retain nine figures and two tab groups.
  A fresh native download independently matches all pinned fingerprints and
  the cache reopens offline. Fourteen native dataset/loader tests, eight Python
  tests and 39 R expectations pass. The touched native download/header batch
  has 43/43 documented definitions and a passing clangd fetch hover. Two stale
  binding test counts are refreshed for the existing 23-row table catalog.
  Evidence: `build/cad-workflow/`.

- 2026-10-31: Ported mep's reusable CAD kernel, polygon modeller, importers and
  materials to canonical C++ `dm::cad`, with owned solid operations, headless
  rendering, picking, URDF forward kinematics and thin Python/R facades. The
  feature inventory and inherited format/geometry limits are recorded in
  `docs/cad-port-plan.md` and `docs/cad.md`; source hashes and MIT attribution
  accompany the port. The four new facade/render/URDF headers have 132/132
  documented definitions, five passing clangd hovers and 36 compiled/executed
  documentation programs. Original kernel comments are retained; the remaining
  legacy headers are not claimed as a completed Doxygen batch.
  All 21 native suites pass, including 19 source regressions, 90 API checks and
  independent binary-format oracles. Four focused suites pass address/undefined
  sanitizers with leak detection disabled; 14 installed Python tests and 57 R
  expectations pass. A separate installed CMake consumer exercises mass
  properties and rendering. All nine playground programs execute: spline,
  Seeed DevArm and exact STEP-link groups each agree across C++, Python and R
  on cached Markdown and image bytes. The real robot has 34 mesh parts and
  430,912 triangles; asset downloads are pinned and hash-verified. Native mep
  parsing and HTML/TeX exports preserve all three tab groups and nine figures.
  Evidence: `build/cad-port/`.

- 2026-10-31: Added a P/HB/HP spline comparison to the fresh playground with
  complete C++, Python and R tabs, native SVG plots and rendered RMSE tables.
  HP is explicitly defined as hierarchical penalized patches using the existing
  `fit_cell_penalized` API. The report explains raw-data residual fitting, local
  coordinates, coefficient counts and the additive hierarchy's boundary behavior.
  R now exposes the canonical hierarchical spline surface through SWIG, matching
  Python; generated sources are regenerated using `scripts/prebuild_bindings.py r`.
  Native R help documents construction, fitting, ownership and errors. All three
  programs run and agree on the reported RMSEs; ten new R spline expectations and
  32 existing algebra/matrix expectations pass. Native mep parsing and HTML/TeX
  export preserve all three tabs, result tables and plots. No core API changes.
  Evidence: `build/playground-splines/`.

- 2026-10-31: Moved the playground's examples into standalone language-tab reports:
  `numerics/linear_algebra/matrix_expressions.mepml`,
  `math/computer_algebra.mepml` and `stats/regression/bspline_regression.mepml`
  beneath `docs/mepml/examples/`. All twelve programs and their four C++/Python/R
  tab groups are preserved. Shell/interpreter paths are rebased for each report's
  depth, and all three spline SVGs move into the regression report's figures
  directory without changing bytes. All twelve fence commands execute from their
  new directories and cached output matches stdout; two previously stale CAS
  output blocks are refreshed to include the current programs' printed text.
  Native parsing and HTML/TeX exports pass for all three reports, including all
  tables, equations and image references. The examples README links to the new
  locations; website discovery finds them automatically. Playground retains only
  its original metadata, ready for a new example. No core API changes.
  Evidence: `build/playground-move/`.

- 2026-10-31: Added C++ Expr arithmetic operators (+, -, *, /, ^), unary signs
  and implicit numeric constants, delegating to the canonical immutable-tree
  methods. Literal operands work on either side, assignment retains value
  ownership, and expressions do not convert implicitly back to numbers. Power
  documentation explicitly covers C++ precedence, left associativity and the
  distinction between negative bases and negated powers; the playground uses
  `(x ^ 3) - 2 * x + 1`, `f * 1 + 0` and `(2 * x).sin()`.
  All 90 focused C++17 algebra checks pass, including six new operator cases
  covering mixed operands, calculus, temporary/self-assignment lifetimes and
  compiler parsing of powers. All nine expression-header examples compile/run
  and eight documented output lines match. Fresh Doxygen verifies all eight new
  public contracts/parameters/examples, and six definition hovers pass. The
  existing expression-header documentation backlog remains outside this batch.
  Canonical Python/R SWIG regeneration changes no generated artifacts; these
  C++ conveniences leave the established language method APIs intact. The C++
  playground program runs and its cached results match stdout and independent
  calculus checks; all four tab sets and the other eleven programs/results are
  preserved. HTML/TeX exports pass. Evidence: `build/cas-operators/`.

- 2026-10-31: Reorganized the playground into four native mep tab sets: linear
  algebra, symbolic calculus, polynomial/rational algebra and B-spline regression.
  Every set contains independent C++, Python and R programs with their saved
  results inside the corresponding panel. Shared SWIG extensions expose the
  existing native algebra TeX serializers as owned strings in Python/R; R also
  gains the canonical definite-integral and shifted-Taylor helpers. The native
  normal generator is available as `native_rnorm(mean, sd, n)` in R without
  masking `stats::rnorm`. All spline examples seed the shared generator with 42
  and use the same breakpoints; Python/R use BSplineRegression for the same
  spline space as C++'s explicit sparse-design LM fit. Separate SVG files preserve
  each panel's plot. All 12 programs compile/run, their cached results match
  stdout, and all 1000 plotted positions plus the fitted curves agree exactly
  across the three SVGs. Native parsing and HTML/TeX exports verify four groups
  and twelve code/result panels. Eleven installed-Python tests and 32 R testthat
  expectations pass, including owned TeX strings, error translation, calculus,
  matrix ownership and reproducible native draws. Core SWIG regeneration and
  installed core-binding builds pass against the checkout's validation library;
  Python's existing unrelated NN-wrapper build is isolated. Cached Nix R testthat
  dependencies supply the normal runner without modifying global R libraries.
  Evidence: `build/playground-tabs/`.

- 2026-10-31: Added composable C++ `.to_tex()` methods for algebra `Expr`,
  `Polynomial` and `RationalFunction`, returning owned `dm::Tex` values without
  math delimiters. Expressions preserve operator grouping, negative and nested
  power bases, fractions and named functions; logical multi-letter names are
  escaped literally. Polynomial output handles signs, omitted unit coefficients,
  configurable formal symbols and reduced rational-function denominators. All
  serializers support 1-17 significant digits, classic-locale numbers, scientific
  notation and nonfinite values. Formatting performs no symbolic evaluation.
  All 84 focused C++17 algebra checks pass, including eight TeX regressions.
  Fresh Doxygen covers 10/10 shared-helper definitions and verifies all three
  new public contracts, parameters and examples; all three header examples
  compile and run and all three clangd definition hovers pass. Legacy algebra
  header documentation remains outside this batch and is not marked complete.
  The methods are C++-only companions to the C++ Tex value; canonical Python/R
  SWIG regeneration leaves binding outputs unchanged. Both playground programs
  compile and run, with raw Markdown results matching stdout. Native mep parsing
  recognizes four math displays and four tables in the CAS section; HTML/TeX
  exports and the isolated CAS PDF pass, with the rendered mathematics visually
  inspected. Evidence: `build/cas-tex/`.

- 2026-10-31: Added the owned C++ `dm::Tex` value type and changed matrix
  `.to_tex()` to return it. Nonmutating underbrace/overbrace and underset/overset
  annotations, grouped scripts, fractions, roots, sized delimiters, escaped
  literal text and raw concatenation compose before display/inline delimiters.
  String conversion copies source; string-view conversion borrows it. Both
  standard stream output and string-style `std::format` are supported. The
  playground now displays the expanded normal equations with underbraces and
  the computed beta, retaining its editable program and Markdown result mode.
  All 252 matrix tests, 96 formatting tests, five C++17 Tex tests and the matrix
  allocation checks pass. All 58 affected-header examples compile and run;
  157/157 selected definitions pass fresh Doxygen and seventeen definition
  hovers pass. Canonical Python/R SWIG generation changes no binding outputs.
  The matrix section compiles to PDF with visually checked underbraces, and
  the complete report exports to HTML/TeX. Full-report PDF export remains limited by the later existing SVG
  plot, which the bundled TeX engine rejects. Evidence: `build/tex/`.

- 2026-10-31: Added lazy ordinary matrix `operator^` and free `dm::t(A)`.
  Integral powers use binary powering, zero yields identity, and both -1 and
  -1.0 retain the existing fused solve path. Fractional powers use a checked
  binomial Taylor series with repeated Denman-Beavers square roots and scalar
  scaling; diagonal powers use scalar pow. General fractional inputs require
  the real principal branch (no nonpositive real eigenvalues); singular,
  nonconvergent and overflowing cases fail explicitly. Scalar types and
  lvalue/temporary ownership are retained, with dense power results.
  Free transpose forwards all four value-category/const overloads unchanged.
  All 290 focused matrix/spline checks and 40 sanitizer facade checks pass;
  both inverse exponent spellings construct without allocations and retain the
  sub-1-KiB fused-solve allocation bound. An independent NumPy eigendecomposition
  oracle agrees on 738 nonsymmetric, positive-spectrum/scaled cases, with maximum
  relative infinity-norm difference 1.70e-14. All 37 matrix/format header examples
  compile and run, 182/182 matrix-header/backend definitions are documented,
  and thirteen definition hovers pass. The playground's dense and CSR
  `(t(X) * X) ^ -1` expressions run in its declared Nix shell; results and
  HTML/Markdown exports pass. The backend remains in the existing tracked
  matrix.cpp so Git-filtered Nix sources include it. Evidence: `build/matrix-powers/`.

- 2026-10-31: Added typed `dm::mat(rows, cols)` and `dm::spmat(rows, cols)`
  factories, defaulting to double and forwarding to zero-initialised dense/CSR
  constructors. Writable zero-based coefficient access returns dense references
  or CSR coordinate proxies, preserving checked insertion/removal and read-only
  sparse-zero behavior. Proxy copying assigns values without rebinding; scalar
  formatting supports existing numeric format specifications. Lazy coefficient
  reads retain value semantics after their temporary snapshot is destroyed.
  All 283 focused matrix/spline checks pass, including storage/allocation budgets;
  all 33 facade checks also pass address/undefined-behavior sanitizers (leak
  detection disabled in this supervised environment). All 35 matrix/format-header
  examples compile and run, 109/109 definitions pass the documentation audit and
  eleven definition hovers pass. Updated the playground's last-row indices to
  `(2, 4)` and `(2, 2)`, ran its matrix block in its Nix shell and refreshed the
  results; HTML/Markdown exports pass. Evidence: `build/matrix-factories/`.

- 2026-10-31: Added `.to_tex(type = "bmatrix")` to dense/sparse matrices and
  their lazy expressions. Serialization includes every coefficient, reads owned
  storage directly and evaluates expressions once. Environment names support
  matrix/pmatrix/bmatrix and other letter names with an optional trailing star.
  Seven-significant-digit numbers use locale-independent decimal points, TeX
  scientific notation and explicit infinity/NaN commands; empty dimensions have
  an empty body. All 240 focused matrix checks pass, including typed storage,
  lazy solves, custom delimiters, empty/full output, locale and allocation tests.
  All 29 header examples compile and run, 91/91 header definitions are documented,
  and eight definition hovers pass. Fixed the playground's C++ beta escape,
  executed its matrix block in the declared Nix shell and refreshed its TeX
  results; HTML/Markdown exports pass. Evidence: `build/matrix-tex/`.

- 2026-10-31: Matrix templates now use `dm::Mat<Storage, T>`, defaulting to
  `dm::Dense` and `double`; `dm::Mat<dm::Sparse>` selects double-precision CSR.
  The root storage policies alias the existing native policy types, and
  `dm::la::Mat` follows the same storage-first order. Updated typed solve and
  spline alias regressions, canonical header examples and playground prose.
  All 271 focused matrix/spline checks pass, including allocation regressions.
  All 28 matrix-header examples compile and run, all 90 header definitions pass
  the documentation audit, and seven definition hovers pass. The playground's
  matrix block runs in its declared Nix shell, reproducing both dense and sparse
  coefficients and replacing stale compiler errors; HTML/Markdown exports pass.
  Evidence: `build/matrix-storage-order/`.

- 2026-10-31: Added the C++ `dm::lm(...)` factory, forwarding to existing `LM`
  constructors with their original argument categories, defaults, data ownership
  and dense/CSR solvers. The playground now uses `dm::lm(y, xmat)`. Existing
  dense/sparse/formula comparisons and typed no-intercept regressions exercise
  the factory; all 38 focused spline/model checks pass. The public example
  compiles and prints both documented predictions; fresh Doxygen documents the
  factory and four selected definition hovers pass. Python/R regeneration leaves
  binding outputs unchanged. All four playground blocks pass in their declared
  Nix environments, retaining the source code and regenerating the displayed
  B-spline image. Evidence: `build/lm-factory/`.

- 2026-10-31: Added the empty `dm::plot()` factory and fluent `RPlot::xlim()` /
  `ylim()` aliases needed by the current B-spline playground block. The aliases
  reuse the existing axis-range validation and rendering semantics; invalid
  ranges leave the plot unchanged. Python aliases retain their owning proxy,
  and Python/R translate invalid-range exceptions. Canonical SWIG regeneration
  and local core-binding builds pass (the Python core wheel uses the previously
  documented isolation from the unrelated NN wrapper build). All 89 focused
  plotting checks, ten Python tests and 68 R expectations pass. Both new header
  examples compile and print their documented values; 379 selected Doxygen
  definitions and seventeen definition hovers pass. All four playground blocks
  run in their declared Nix shells with the user's C++ code retained. Replaced
  the stale compiler-error result with the regenerated SVG: 1,000 sampled points,
  one five-pixel fit curve and the requested axis ranges. The displayed SVG
  matches the saved file; HTML/Markdown exports and a visual preview pass.
  Evidence: `build/playground-plot-api/`.

- 2026-10-31: Plot color arguments now spell their shared native type `RGBA`,
  retaining the existing `RGB` type identity, numeric initialization and layout.
  Implicit string/literal conversion centralizes the existing basic color names
  and strict `#RRGGBB`/`#RRGGBBAA` parsing, with alpha last and normalized to [0, 1].
  Invalid text fails before plot mutation. Literal labels and Histogram chaining
  retain their prior behavior; numeric scalars cannot implicitly become colors.
  Added constructor exception translation and regenerated the Python/R wrappers
  from the canonical header, replacing the old numeric-constructor SWIG extension.
  All 115 focused native plot/vector tests pass. Six color/label documentation
  examples compile and execute, four printed-output comments match, 278 selected
  Doxygen definitions pass, and four definition hovers pass. The binding suites
  cover numeric compatibility, parsed alpha, ownership, SVG output and malformed
  input: nine Python tests and 62 R expectations pass after rebuilding the core
  bindings. The Python core wheel is isolated from the previously recorded NN
  wrapper-build issue. The playground demonstrates literal and std::string line colors and uses
  explicit local Git shell references. Nix metadata and the pinned editor's Lua
  helper resolve those references to this checkout; the originally reported
  GitHub-URL failure has not been reproduced. All four report blocks pass with the
  explicit references; the regenerated SVG matches the displayed file and exports
  to HTML/Markdown. Evidence: `build/rgba-colors/`.

- 2026-10-31: Numeric `Vector` arithmetic, scalar broadcasting and element-wise
  math now use lazy expression templates. Named vector leaves are borrowed;
  temporary vectors and child nodes are owned. Owned-vector assignment is
  alias-safe, `.eval()` captures snapshots, and standard-vector consumers
  materialise directly. Recycling remains local to each node, including unequal
  lengths and empty inputs. Arbitrary `apply()` callbacks remain eager; random
  sampling happens once and its owned results are reused across evaluations.
  Legacy eager ABI symbols and the owned-vector layout are retained.
  All 302 focused native tests pass, with nine new lifetime/alias/domain/API
  cases also passing address and undefined-behavior sanitizers. A 100,000-element
  Doppler chain forms with zero allocations and evaluates with one 800,000-byte
  output allocation; constructing the chain with `rnorm()` allocates only its
  sampled noise. Expression reductions allocate no result arrays. Regenerating
  Python/R interfaces produces no changes; eight Python tests and 53 R binding
  expectations pass. All 65 vector-header examples compile and execute, including
  80 checked printed-output comments. All 184 selected Doxygen definitions and
  eleven clangd definition hovers pass. The playground uses a forwarding Doppler
  lambda and whole-vector evaluation. All four blocks run in their declared Nix
  shells; the regenerated SVG matches the displayed file, exports to HTML/Markdown
  and has been visually checked. Existing hex line colors are expressed through
  the plot's RGBA API. Evidence: `build/vector-expressions/`.

- 2026-10-31: Added typed `SpMat<T>`, direct-CSR `Bspline::speval()`, and dense/CSR
  numeric-response LM constructors and point prediction. Sparse fits retain CSR
  designs (including any automatic intercept), use the existing native PCG
  normal-equation solver, and share inference with formula models; coefficient
  covariance remains dense. Sparse diagnostics and sequential ANOVA retain sparse
  row/prefix access. Input validation, ownership, empty predictions, cross-storage
  prediction, rank/solver failures and formula regressions are covered by tests.
  All 275 native checks pass, including a 20,000-row allocation regression: largest
  allocations are 1 MiB for sparse basis evaluation and 512 KiB for sparse LM
  fit/prediction. On the playground design (1000 x 101, 3990 nonzeros), prediction
  differs from dense QR by at most 1.37e-11. All 35 selected header examples compile
  and run; 221 complete-path Doxygen definitions, nine selected basis/facade
  definitions and seven definition hovers pass. Recompiled the local R binding
  and Python core-extension wheel (isolated from the existing NN-wrapper build
  issue); eight Python binding tests and the R binding suite pass, including
  formula-model ownership checks. All four playground blocks run in their Nix
  shells and keep the displayed SVG. Evidence: `build/sparse-lm/`.

- 2026-10-31: Added the C++ `dm::mkdir(path)` free helper for recursive output
  directory creation. Native paths, strings and literals are accepted; existing
  directories are a no-op and conflicting files/filesystem failures propagate
  as exceptions. Four filesystem tests cover missing parents, repeated creation
  with retained contents, conflicting files and empty paths without a cwd change.
  The public example compiles and prints `true` on both initial and repeated
  runs; the new definition has Doxygen documentation and a clangd hover check.
  All four blocks from the starting playground snapshot pass with the Nix
  shells and preserve the SVG. Concurrent user edits introducing a sparse
  regression API were retained after that run; that newer API is outside this
  helper's verification. Evidence: `build/playground-mkdir/`.

- 2026-10-31: Promoted the typed facade to `dm::Mat<T, Storage>`, with
  `dm::la::Mat<T, Storage>` retained as an alias to the same underlying types.
  Concrete dense-double declarations and return types use `Mat<>`; sparse
  storage and expression-template kernels retain their existing types. Tests
  verify root-namespace dense/sparse solves for float, double and long double,
  alias identity and the playground's `dm::Mat<> = basis.eval(x)` assignment.
  All 266 native checks (including allocation budgets) and seven installed
  Python binding tests pass. All 33 selected public-header examples compile and
  run; 711 affected-header Doxygen definitions and eight definition hovers pass.
  All four playground blocks run with their Nix shells; both report exports
  retain the spline figure, whose bytes are unchanged. Evidence:
  `build/root-matrix-api/`.

- 2026-10-31: Added the `bs(lower, upper, knots, degree, intercept)` free factory
  for an owned `Bspline`, and switched the playground and formula-regression
  integration test to that spelling. The factory preserves the existing cubic
  and omitted-first-column defaults, validates immediately and owns temporary
  knot inputs. All 33 spline/model tests pass, the new header example compiles
  and prints `3 4`, all six factory/basis definitions have Doxygen descriptions,
  and three definition hovers pass. All four playground blocks run through mep;
  the displayed spline SVG is byte-for-byte unchanged and remains in the HTML
  and Markdown exports. Evidence: `build/playground-bs-factory/`.

- 2026-10-31: Playground B-spline regression now composes `Bspline::eval()`,
  matrix column labels and `cbind()`, named DataFrame conversion, the existing
  formula LM and native scatter/line plotting. The same clamped basis is reused
  for prediction; dropping its first column by default preserves a full-rank
  model with a formula intercept. The report saves a local SVG and displays the
  copied figure through mep's `file=` result. Six new integration tests and the
  adjacent spline/LM cases pass (32 tests), as do 232 matrix regressions. The
  cubic design matches R's `splines::bs()` exactly on a reference grid. Ten new
  API examples compile and run; the new class/methods and facade constructors
  have Doxygen descriptions, three affected complete headers pass 765/765
  definitions, and two spline definition hovers pass. The SVG was rendered and
  visually reviewed. All four playground blocks run through mep, including the
  Python/R Nix shells; Markdown and HTML exports retain the image reference.
  Evidence: `build/playground-bspline/`.

- 2026-10-31: Playground Python/R examples now use native dense/CSR matrix
  bindings, with owned runtime expression handles, deferred inverse actions and
  fused solves. Added installed-package binding tests, native previews, Python
  docstrings and R help/examples. The adapter passes 34/34 Doxygen definitions,
  nine compiled/executed C++ header examples and two clangd definition hovers.
  Seven Python binding tests, the R binding suite, 232 native matrix tests and
  the allocation regression pass. A 20,000-row binding expression uses a largest
  construction/evaluation allocation of 112 bytes. All three playground blocks
  execute against the local core and installed bindings. The full Python wheel
  build encounters unrelated NN-wrapper namespace/type errors; Python validation
  uses an isolated core-extension wheel. Evidence: `build/playground-bindings/`.

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

### Typed native matrix facade and lazy inverse actions (2026-10-31)

`linalg/matrix.hpp`, `src/datamunge/linalg/matrix.cpp` and `format/matrix.hpp`
implement the playground's `dm::Mat<Storage, T>` API, also available as
`dm::la::Mat<Storage, T>`. Storage comes first and defaults to `dm::Dense`;
the scalar defaults to double. `dm::Sparse` selects existing CSR storage.
The original `dm::la::DMat` and `dm::la::SpMat` policy types remain supported.
Float, double and long double flow through products and solves without implicit
promotion. `dm::Mat<>` and `dm::SpMat<>` are convenient double specialisations.
Nested row initialisers, checked coefficient access, mutation, storage/scalar
conversion, canonical-backend interoperation and formatting have adjacent contracts.

Expression nodes borrow lvalue matrices, own rvalues and copy intermediate nodes.
They recheck dimensions at evaluation, preserve sparse products and avoid dense
transpose buffers. `solve(A)*B*C` is reassociated into one solve of `A Z = B C`;
an inverse is formed only when explicitly evaluating a standalone `solve(A)`.
Sparse arithmetic stays CSR until the dense coefficient/RHS solve boundary.
Two-argument `solve(X,y)` uses existing LU for square systems and Householder QR
for tall systems. The normal-equations condition-number penalty and explicit-Q
memory cost are documented. Move operations leave usable empty source matrices.

Validation: the core CMake build, 233 focused/existing linear algebra checks and
all 95 module-formatting tests pass. All 21 new API cases pass AddressSanitizer and UndefinedBehaviorSanitizer;
LeakSanitizer is unavailable under the environment's ptrace supervision. The
allocation regression measures no allocations during expression formation and a
largest allocation of 32 bytes both for a 20,000-column inverse-action chain and
for the playground expression on a 20,000-row dense design. A billion-column
sparse product retains two stored entries without allocating a dense grid.
The playground runs and exports to HTML/Markdown with coefficients 7/33 and 17/33,
checked independently using exact rational arithmetic. All 25 documentation
examples compile/run with matching output; the three new files pass 143/143
documentation definitions and four representative clangd definition hovers.
Evidence: `build/matrix-api/`. Graphical editor help was not checked.

### Native method diagrams (2026-10-31)

`include/datamunge/plot/diagram.hpp` and `src/datamunge/plot/diagram.cpp` add
`graphics::Diagram`, with owned labelled nodes, measured multiline wrapping,
boundary-anchored arrows, geometric paths/polygons/ellipses and fitted SVG bounds.
Visible text uses the plotting library's own Computer Modern glyph outlines,
including native constructions for Unicode scripts; selectable source text and
accessible labels remain in the SVG. Unsupported glyphs and style/error contracts
are documented. The feature is exposed by the standard public umbrella header.

All 189 former external-renderer diagram blocks in the executable reports now
contain complete native C++ programs. The old diagram sources are removed, and
the report/contributor standards require native graphics. Fifteen misplaced TeX
figure copies were restored from the existing native numerical/architecture SVGs
in their build directories; their training/data computations were not rerun.
The 1,007 pre-existing C++ blocks in the indexed affected reports are unchanged.

Validation: the CMake core and focused runner build; eight native renderer tests
and three report/provenance regression tests pass. MEP executes all 189 changed
diagram programs; their generated SVGs were rasterised at 1100 px and visually
reviewed. All 292 current reports export to HTML and Markdown. All thirteen header
examples compile and run, including the documented `0 0` and `true` outputs.
Both new library files pass the documentation presence audit (59/59 definitions),
and four representative definition hovers pass through clangd. The full completed-path
gate reports 34 unrelated undocumented CUDA parameter fields in `compute/cuda.cpp`;
the new diagram files pass their selected audit. Graphical editor
integration was not checked. Evidence: `build/diagram-migration/`.

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
and the ideal-key assumption behind collision expectations. All thirteen header
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
scalar discrepancy 1.1e-8). The rewritten `docs/mepml/examples/stats/regression/pspline.mepml` has seven
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

The rewritten `docs/mepml/examples/numerics/interpolation/de_boor_analysis.mepml` has six executed blocks
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

The rewritten `docs/mepml/examples/stats/bayes/gaussian_process_regression_iris_analysis.mepml`
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
- The rewritten `docs/mepml/examples/stats/regression/glm_iris_analysis.mepml` executes all seven programs.
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
- `docs/mepml/examples/stats/regression/glmm_analysis.mepml` executes all 8 programs. All 7 SVGs were rendered
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
- `docs/mepml/examples/ml/dimensionality_reduction/isomap_iris_analysis.mepml` executes all 8 programs. All 7 SVGs were
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
- `docs/mepml/examples/stats/inference/hypothesis_tests_analysis.mepml` executes all 10 programs. All 10
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
- `docs/mepml/examples/stats/regression/kernel_regression_iris_analysis.mepml` executes all seven programs.
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
- `docs/mepml/examples/ml/dimensionality_reduction/kernel_pca_iris_analysis.mepml` executes all eight programs. All
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
`docs/mepml/examples/math/coding_theory/diagrams/hamming_distance_concept.tikz`; LaTeX/dvisvgm save the diagram,
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

The converted `docs/mepml/examples/stats/regression/kernel_ridge_analysis.mepml` opens with exact
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

The converted `docs/mepml/examples/random/generators/mersenne_twister64.mepml` derives scalar seeding,
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

- 2026-10-30: CIFAR-10 dataset-study helpers add native stratified preparation,
  persisted split/moment restoration, independent crop/flip augmentation and
  compact ResNet inference; generic proper classification scores, reliability
  bins, validation-only bounded temperature fitting, and exact decoded-image
  overlap auditing with full-byte collision checks. All three new headers pass
  35/35 Doxygen definitions, twelve compiled/executed header examples, five
  clangd definition hovers and nine native offline tests. Source identities,
  convergence, held-out inference and interpretation evidence are retained with
  the dataset report. Evidence: `build/cifar10-report/`; graphical mep K was not
  available for a UI check.

## Realistic robot workcell assets (2026-11-01)

Added the opt-in `WorkcellVisualStyle::Realistic` profile, reusable workcell
lighting/render settings, procedural industrial surfaces, rounded fixtures and
detailed fruit meshes. The default Simple profile and existing aggregate
initializers remain compatible. Visual geometry does not change collision
geometry or consume the environment's random streams. The wrist camera renders
the same complete scene for RGB and registered empty-background inspection.

The public renderer documents optional linear-light GGX/Schlick shading, soft
key shadows, local occlusion, exposure and display mapping, including its
approximate ambient illumination and lack of subsurface scattering. References
were checked against the original publications. New public and private symbols
have adjacent documentation; completed asset files are included in the manifest.

Verification: 84 native simulation/render tests and three classifier tests pass
(eight optional GPU tests skip); the existing CAD API suite passes 90 checks,
with model bookkeeping and format/import/render regressions also passing.
All 22 affected short public-header examples compile and execute against current
headers and the rebuilt library. Fresh scoped Doxygen covers 328/328 definitions
with no missing entries; 13 representative clangd hovers pass. Evidence is in
`build/debug/docs/realistic-workcell-{coverage,hover}.json`,
`build/realistic-workcell-*-tests.log`, and `build/realistic-workcell-doc-examples/`.
This is scoped documentation coverage, not completion of the renderer's existing
private-symbol documentation backlog. Graphical mep UI playback was not checked.

The complete playground learning run adapts vision on 768 disjoint moving-view
crops (120/128 held-out correct), trains 8,192 scheduling windows and evaluates
actual RGB in 48 runs across 16 paired streams. An explicit frozen-checkpoint replay via
`DATAMUNGE_STREAM_REPLAY=1 just playground` completes in 1,642 seconds and
regenerates the uninterrupted minute with the new assets. The 960x640 H.264
video fully decodes, has 900 frames at 15 fps and duration exactly 60 seconds.
All 23 published artifacts match the run directory and exported figure copies;
all corresponding HTML links resolve. The four scheduled birth times match the
previous visual profile, with no changed joint/finger limits. The minute reports
two correct placements and two misses, without cutting unsuccessful events.
Detailed evidence: `build/realistic-workcell-audit.json` and
`build/realistic-workcell-verification.txt`.

## Robot workcell performance baseline (2026-11-01)

Added the independent Release C++ recorded-workcell benchmark, standard CMake
target and `just bench-fruit-render` entrypoint. The benchmark guide explains
replay reconstruction, temporal sampling, warm caches, per-call scopes, worker
controls, the one-pixel preparation proxy, and timing limitations. The measured
implementation plan lives in `PERFORMANCE.md`; compact raw evidence is in
`benchmarks/results/fruit_render/`. No public core symbols or generated bindings
changed, so no new public-symbol documentation batch is claimed.

Validation: the actual standalone CMake configure/build and Python runner pass;
36 full-frame samples and three repetitions per control/component complete.
Worker counts 1/2/4/8/16 produce byte-identical RGBA at the control pose. Three
full 60-second actual-RGB simulations reproduce the frozen report outcomes and
901 endpoint-inclusive snapshots. Full encoding retains 900 frames, 15 fps and
60.000 seconds, fully decodes and matches the published MP4 byte for byte.
`just` recipe parsing/dry-run and Python compilation pass. The full Nix wrapper
was not rerun; measurements use the existing local Release core, identified by
its linked path/hash. Full model-training wall times were not remeasured.

## Lossless robot rendering and streaming pass (2026-11-01)

Added `cad::PreparedScene`, explicit wrist-camera worker budgets and native
`cad::VideoWriter`. Public contracts describe owned immutable geometry, lazy
thread-safe base hierarchy initialization, exact actor-order hierarchy rebuilds,
ID/material mapping, synchronous bounded encoder handoff, explicit publication
and failure/abandonment cleanup. New API examples are included in the public
headers; private video helpers/state also have adjacent documentation.

Fresh scoped Doxygen passes 147/147 definitions across render/video/camera headers
and the video/camera implementations; ten clangd hover checks pass. All 17
affected examples compile and execute against the installed Nix package. Added
the video header/implementation to the completed manifest. This does not claim
to finish the renderer's older private-symbol documentation backlog.

Evidence: 85 simulation/render tests, three classifier tests, prepared-scene and
real-video suites, existing CAD API (90 checks) and import/render regressions; 34
exact reference-buffer/crop/box matches. A complete `just playground` frozen
replay improves from 1,642.1 to 771.3 seconds and reproduces the original MP4, two
stills, both trajectories and perception decisions byte for byte. Its 25 assets
and export links pass the audit. Detailed phase timings and source/settings/model
fingerprints are published with the report. Repeated paired performance data is
under `benchmarks/results/fruit_render/optimized.json`; the original baseline is
retained. Native tests validate private-worker SIGPIPE handling without changing
the caller's handler/mask; the Windows branch has not been runtime-tested.
Nix source-filter checks retain build/binding inputs while excluding benchmark
notes/helper scripts. No graphical mep UI playback check is claimed.

## Compact renderer queries and configurable previews (2026-11-01)

Documented new private compact triangle records, original bounds/centroid lookup,
per-query ray reciprocals with guarded division fallback, cached light/sample
helpers, and their ownership and arithmetic-order constraints. The public API is
unchanged. Fresh Doxygen passes 10/10 definitions in
`src/datamunge/cad/detail/ray_box.hpp`; added that header to the completed manifest.
Three new clangd definition hovers pass. This does not claim coverage of the older
private renderer backlog. Evidence: `build/debug/docs/fruit-pass2-{coverage,hover}.*`.

Native validation passes 85 simulation/render tests (eight optional GPU tests
skip), prepared-scene regressions, 90 CAD API checks, import/render regressions,
and 2,000,040 differential slab comparisons. All 34 reference buffers match
exactly. Three paired rounds in each scheduling mode measure a 5.8–6.1% reduction
in median sampled pipeline time; retained samples and hashes are in
`benchmarks/results/fruit_render/pass2.json`.

The playground now documents and executes a configurable 15-second preview while
recording the unchanged full minute of physics and both 900-frame trajectories.
It explicitly labels cumulative counters and the debugging excerpt, links the
preserved full-minute video, and separates short PNG dumps from full benchmark
inputs. `just playground-preview` reuses frozen checkpoints. The completed mep
compilation/execution scope takes 186.229 seconds; rendering/streaming takes
168.010 seconds. This includes the shorter duration, not only engine gains.
The new MP4 equals an independent encoding of original frames 540–764 byte for
byte; full trajectories, inspection image and poster also match their references.
All 26 assets, HTML links, full video decode and conservation checks pass
`build/fruit-pass2-audit.py`. The benchmark runner validates full-minute simulation
outcomes against preview metrics. No new training run or graphical mep UI
playback check is claimed.

## Explicit CUDA CAD renderer (2026-11-02)

Documented the new C++ `cad::CudaRenderer` ownership, serialized persistent device
cache, synchronous outputs, float32 constraints, precision/bias differences and
explicit availability/error behavior. CPU rendering and wrist sensor input remain
the reference paths. New private shared host/device types have individual field
contracts; added `src/datamunge/cad/detail/cuda_types.hpp` to the completed manifest.
Fresh scoped Doxygen passes 153/153 definitions across that file and the public
render header. Six clangd hovers pass, including the complete CUDA class contract
and constructor. The public class example compiles and executes on an RTX 5080
Laptop GPU. This does not complete the older private renderer documentation backlog.

Real-device tests cover maps, point lights, soft shadows/AO, transparency, camera
projection, sample/style modes, actor/base caching, failed cache replacement,
concurrency and invalid inputs. Existing compute/simulation tests pass 90/93;
three Vulkan-only tests skip. All 34 CPU reference buffers still match exactly.
The bounded four-pose benchmark retains paired timings and RGB/depth/ID errors
in `benchmarks/results/fruit_render/cuda.json`; it generates no video or training.
The measured median complete overview call improves from 476.15 to 133.87 ms,
with float32 image RMSE below 0.72/255 and source-ID mismatch below 0.025%.
Evidence: `build/cad-gpu/`, `build/debug/docs/cad-gpu-{coverage,hover}.*`.

`just playground-gpu` completes a five-second/75-frame clip with the installed
CUDA package: 42.866 seconds for code-block compilation/execution, 26.780 seconds
for rendering/streaming, excluding outer Nix builds. All 26 assets and HTML/TeX
exports are present; video decode and cumulative/physical outcomes pass the audit.
Full trajectories, perception decisions and checkpoints remain unchanged. Both
representative wrist insets are pixel-identical to the CPU reference; complete
stills have RGB RMSE below 0.60/255. `just bench-fruit-render-gpu --repetitions 1`
also passes against the installed Clang package. The wrapper explicitly propagates
the repository's canonical pkg-config override through the nested Nix shell.
No full-minute rendering or fresh model training was run in this GPU pass.

## Exact CPU wrist reference concurrency and caching (2026-11-02)

Documented bounded full/reference render concurrency, sequential one-worker mode,
exact joint/finger invalidation, owned base-only buffers, fresh moving reference
actors and off-image behavior in the camera header and private helpers. API
signatures and the single-writer camera contract are unchanged. Fresh scoped
Doxygen covers 50/50 definitions in the camera header/implementation; five clangd
hovers pass. The class, actor-capture and associated-inspection examples compile
and execute against the real cached DevArm URDF.

Four new tests cover owned RGB/depth/IDs, pose/finger changes, moving non-fruit
reference objects, invalid textures, off-image returns, sequential/parallel parity
and recovery after failures. The native suite passes 89/97; eight optional GPU
tests skip. All 31 saved paired wrist outputs and all 34 older reference buffers
match exactly. Bounded paired measurements show 20.0% lower mean wrist cost;
an earlier variable run measured 13.3% and is retained with the raw acceptance
samples in `benchmarks/results/fruit_render/wrist.json`.

The installed-package five-second playground check reproduces the archived MP4
byte for byte. Its wrist stage improves 15.025→12.067 seconds, and rendering/
streaming improves 26.356→23.525 seconds. All settings, physical/outcome metrics,
perception logs, checkpoints, complete trajectories and 26 report assets/links
pass the audit. `just bench-fruit-wrist --repetitions 1` also passes against the
installed core. No full-minute video or fresh training was run. Evidence:
`build/wrist-performance/`, `build/wrist-perf-oracle/`, and
`build/debug/docs/wrist-performance-{coverage,hover}.*`.

## Exterior workcell presentation and studio illumination (2026-11-02)

Documented `sim::workcell_presentation`, `workcell_presentation_options`, the
opt-in `RenderOptions::studio_environment` field and shared CUDA parameter.
Contracts cover owned copies, deterministic maps, texture/index validation,
world-space cable anchors, cosmetic-only geometry, calibration preservation and
analytic-lighting limitations. Source meshes and transforms remain unchanged;
additional surroundings and cables belong only to exterior display scenes.
Existing ambient shading is retained at the default zero studio strength.
Appending the options field requires recompiling native consumers.

Fresh scoped Doxygen covers **183/183** definitions; **11** clangd hovers and both
affected public examples pass. The simulation suite passes **91/99**, with eight
optional GPU skips. Dedicated studio tests and the real-device CPU/CUDA suite
pass. Two enhanced-scene stills have RGB RMSE below 0.401/255 and source-ID
mismatch below 0.037%; all 34 old reference buffers and 31 wrist outputs match
exactly. Evidence: `build/realism2/` and
`build/debug/docs/realism2-{coverage,hover}.*`. This scoped pass does not complete
the older private renderer documentation backlog.

The installed-package `just playground-gpu` build exports HTML/TeX and a fully
decoded 20-second/300-frame video. The 26 report assets and links pass the audit;
complete trajectories, perception logs, frozen checkpoints and physical outcomes
are unchanged. Both representative wrist insets remain pixel-identical. Raw
quality settings and stage timings are archived in
`benchmarks/results/fruit_render/presentation.json`; no new training or full-minute
video rendering was performed.

## Idle infeed observation behavior (2026-11-02)

Added and documented the optional `FruitStreamOptions::watch_infeed` behavior.
The real rigid camera mount and bounded URDF joints determine an elevated
upstream observation pose. Empty-queue waiting returns through guarded servo
waypoints in 0.125-second budgets; subsequent pick options interrupt the return.
Reset starts in that pose only when enabled. The default false retains existing
controller behavior. Queue sensing remains privileged tracked geometry, and the
watch behavior changes travel times rather than providing a newly trained policy.

Fresh Doxygen covers 155/155 definitions in the stream header/implementation;
three clangd hovers and both affected public examples pass. The native simulation
suite passes 93/101, with eight optional GPU skips. New cases check the complete
randomized entry region, exact wait ticks and independent arrival seeds, both-bin
returns, servo limits and subsequent pickups in full physics and the training
proxy. An actual-mesh wrist render confirms a newly arrived fruit has a nonempty
RGB-derived box (7x9 at 128x128), rather than relying solely on projection.
Evidence: `build/infeed-watch/` and
`build/debug/docs/infeed-watch-{coverage,hover}.*`.

The installed-package playground build exports the default 20-second/300-frame
video, HTML and TeX. All 26 assets/links and full video decoding pass the audit.
Complete joint/fruit trajectories and perception decisions match an independent
simulation-only run; Poisson births/appearance seeds and frozen model checkpoints
remain unchanged. The actual camera aims upstream in 659 of 681 idle frames;
the remaining idle frames include its return motion. The clip ends in the watch
pose and labels the live camera state. The fresh minute yields two correct sorts,
zero wrong and two misses. This is a demonstration using the prior scheduler,
not retraining or a new throughput comparison. Raw evidence is archived in
`benchmarks/results/fruit_render/infeed_watch.json`.

## Synchronized interception and continuing-policy retraining (2026-11-02)

Added optional synchronized approach/descent wrist roll, guarded short return
approaches, coordinated idle return segments, live feasible-action masks and
configurable queued-wait/missed/wrong-bin reward costs. Default options retain
the prior behavior. `StreamPolicy::action(state, mask)` preserves the existing
1680-by-five checkpoint format. Training optionally applies the mask to behavior,
exploration and bootstrap values and uses elapsed-time-discounted replacing
Watkins traces, clearing at nongreedy actions and independent windows. Bounded
geometry caches retain four entries with exact pose/aperture/identity/rotation/
lateral-bin keys; live deadlines are never cached. Observation FK has one exact
joint-pose cache. Reset and unknown-perception invalidation are explicit.

Fresh Doxygen covers 193/193 stream definitions; six clangd hovers and five public
examples pass. Native simulation tests pass 98/106, with eight optional GPU skips.
New tests cover deadline expiry, missing targets, query ownership, unknown-target
invalidation, synchronized capture, servo limits, queued-wait rewards, masked
bootstrapping, trace reproducibility and invalid options. The legacy profile's
sixteen-stream actual-RGB results match the frozen prior library byte for byte.
Eight training rollouts also produce identical Q values, durations and returns
with the asset-free and full URDFs.

Completed 32,768 native training windows (3,980,652.31 simulated seconds), with
watching enabled, trace decay 0.8, masked actions and a broader 0.08–0.30 arrival
curriculum. Physical screening and separate actual-RGB validation select the
checkpoint. A further 24-epoch MobileNet adaptation uses independently seeded
aligned-view crops; its validation-selected epoch is 17. Its held-out prototype
test scores 119/128; this is a different view/seed set from the archived image
experiment. No test images enter gradients or checkpoint selection. Evidence:
`build/sort-training/` and `build/debug/docs/sort-training-{coverage,hover}.*`.

Actual-RGB validation selects window 256. On sixteen further paired 120-second
streams (246 identical arrivals), correct placements improve 143→176, misses
87→56 and wrong-bin placements 5→2; pending outcomes are 11→12. Unknown
perception falls 25→0. Updated early/late-oldest schedules achieve 166/152 correct
placements. The unchanged fixed-seed minute improves two→three correct sorts,
with zero wrong placements and one miss. No test/video seed selects a checkpoint.
These results combine motor changes, new vision fitting and scheduler retraining.

`just playground-gpu` exports the default 20-second/300-frame video, HTML and TeX.
All 30 assets/links, complete video decode, servo bounds, Poisson births, seed
separation, dataset quotas, training histories, validation selection and paired
outcome/perception logs pass the audit. Both prior models remain available. Raw
results and fingerprints are archived in
`benchmarks/results/fruit_render/control_training.json`. Replay takes 149.23 s
including code-block compilation, with 135.26 s program time; Nix builds are
excluded. The full-training recipe has a configurable two-hour timeout to allow
the longer learning/validation workflow; rendering remains a 20-second default.
