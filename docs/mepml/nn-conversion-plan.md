# Neural-network examples as mepml reports

Current continuation (2026-11-10): the destination remains
`docs/mepml/examples/nn/`, as clarified by the user. All remaining numbered
sections, including the CNN/autoencoder programs previously moved to
`examples/cpp/source/nn/`, are in scope. The original inventory is **416 sources**
(415 C++ programs and the NumPy companion); **256 matching reports are present**
and **160 still need conversion**. Presence is not a new verification claim.
Use `just nn-report-status` for the separate retirement count. This remains an
active migration until every replacement and the final `examples/nn/` cleanup
are verified.

Retirement checkpoint: **127 originals removed** — all 58 foundation sources,
all 11 deterministic autoencoders, all 18 variational-autoencoder sources and the
restoration, encoder-pretraining, latent-arithmetic, latent-visualization, learned-codec and reconstruction-anomaly applications, plus auxiliary-classifier, bidirectional, vanilla, least-squares, hinge, conditional, InfoGAN, spectral, Self-Attention GAN, BigGAN, DCGAN, Wasserstein GAN, Progressive GAN, Pix2Pix, CycleGAN, both StyleGAN variants, StarGAN, both SRGAN/ESRGAN variants and SinGAN, plus PixelCNN, both Gated PixelCNN variants and PixelRNN, plus MADE density, PixelCNN++, generative WaveNet, NICE density, RealNVP density, Glow density, MAF density, IAF density, spline density and both planar/radial density variants, CNF density and the three-dataset flow-density study.
All **954 runnable C++ blocks** across these 127 reports pass. Sixteen skipped language-less
listings are explanatory command/code examples.
All **816 local figures** are checked against their generated files, rasterized
and visually reviewed; Markdown/HTML exports include verified adjacent figure
bundles. AutoRec and GAN diagnostics await original preparation/recordings;
the next report is `ddpm`.

DDPM native continuation (2026-11-10): independent NumPy verifies all 28
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

Three-family flow-density report completed (2026-11-10). The independent
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

CNF report completed (2026-11-10): independent NumPy checks all 21 source
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

CNF source preparation (2026-11-10): original frozen in 21 archives / 163,073
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

Planar/radial report completed (2026-11-10): canonical
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

Planar/radial independent continuation (2026-11-10): both source variants are
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

Spline report completed (2026-11-10): independent NumPy checks all22 source
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

IAF report continuation (2026-11-10): canonical `nn/iaf_density.hpp` adds
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
The independent audit, canonical spline APIs/bindings and editable program now
pass as recorded above; the completed report and retirement supersede this earlier checkpoint.

IAF independent continuation (2026-11-10): all22 pinned source archives and
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

MAF report continuation (2026-11-10): all15 C++ blocks pass and14 native
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
seed9703 and BatchOrder9721. The independent audit now passes; canonical IAF
APIs and report remain pending. This standalone exact-density affine adaptation makes no VAE/ELBO
claim; keep its distinction from the original paper's sigmoid gating explicit.

MAF native continuation (2026-11-10): independent NumPy checks all 22 original
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

The MAF report described at this earlier native checkpoint is now complete and
the original is retired, as recorded above. IAF is the next independent report.

Glow native continuation (2026-11-10): independent NumPy checks all 23 pinned
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

MAF is the next report; its independent audit, native API and complete program
are now verified as recorded above. Narrative/figures and retirement are pending.

RealNVP report continuation (2026-11-10): all 13 C++ blocks and 12 native
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

The Glow source baseline described at this earlier checkpoint is now fully
converted and retired, as recorded above. MAF is the current next source.

RealNVP independent continuation (2026-11-10): all 21 pinned archives and all
48 captured Adam writes pass independent NumPy. The audit checks 5,888 original
procedural data values, 85,319 forward fields, 13,300 parameter partials,
12,768 exact updated weights, 17 finite differences, 1,536 batch rows, both
actual Jacobian matrices and all 1,024 inverse samples. Every stage retains
its .8*tanh(raw) bounded log scale; forward and inverse-path determinants vary
spatially, agree with actual positive determinants and cancel on reconstruction.
Every sample satisfies change of variables. Maximum forward/partial errors are
1.614e-5/2.196e-7, finite differences 1.04e-10; Double normalization residuals
stay below 3e-15. The original test score 2.879116774 improves on the fitting-only
full Gaussian 2.994631823, while analytic truth remains better at 2.796480843.
Evidence: `scripts/check_real_nvp_density.py` and
`build/nn-report-conversion/real_nvp_density/independent-oracle.json`.
At that earlier audit checkpoint, native affine APIs/bindings and report were
pending. The completed report above supersedes that state. `common/real_nvp_density.hpp` has shared helper
consumers in Glow, MAF, spline, planar/radial and CNF, so retain it after this
individual executable is eventually retired.

NICE report continuation (2026-11-10): all 12 C++ blocks and 11 native figures
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

NICE native continuation (2026-11-10): the independent NumPy audit passes
all 21 original archives (117,586 values), all 48 Adam writes, 17 finite
differences, both analytic matrix Jacobians and all 1,024 inverse samples.
It independently checks 5,888 data values, 60,566 forward values, 9,900 parameter
partials, 9,504 updated weights, 1,536 batch rows, 512 forward and 4,096 inverse
Jacobian entries. Determinants are positive and agree with the global logdet;
Double normalization errors stay below 4e-15. The original Float32 log(2*pi)
constant is explicit. The short fit remains worse than the fitting-only full
Gaussian: 2.994648218 versus 2.994631823 nats/row; analytic truth is 2.796480843.
The banana observations are generated in-process; no external data is used.

Canonical `nn/normalizing_flows.hpp` adds stopped BananaPoints, live Double
BananaLogProbability, matching floating StandardNormalLogProbability and
NormalFlowDensity with scalar/per-row forward determinants. Canonical
`nn/nice_density.hpp` adds alternating 1->16 tanh->1 additive couplings,
four stages with 198 values in 17 leaves, a positive global scale and the
reverse unscale/subtract inverse. Flat owning records retain first/higher
graphs after parent/store collection. Warmup, fitting, Adam and sampling remain
caller-owned. Only missing leaves initialize; warmed inference is immutable.

The API overlay reproduces all 21 archives, six other files and all 144 update
archives (19,056 values) exactly. Eight new native tests and 535 selected C++
tests, 285 Python checks, 2,165 R expectations in 57 files and 65 inventory/
toolchain checks pass. All 28 new definitions, ten compiled header examples,
seven clangd hovers, Python/R help and standalone checkout/installed headers
pass. Affected documentation coverage is 228/228. The standalone check caught
and fixed an explicit autograd header dependency; numerical behavior is unchanged.
Evidence: `scripts/check_nice_density.py`,
`build/nn-report-conversion/nice_density/independent-oracle.json` and
`build/nn-report-conversion/nice_density_core/`.

At this earlier native-only checkpoint, the NICE report was pending. The
completed report above supersedes this next-action note: write the complete editable report with a worked coupling,
inverse and Jacobian example; native method and executed architecture diagrams;
the exact banana pools; all 48 fitting writes; both Jacobians/roundtrips;
normalization identities; all 1,024 inverse samples; fitting-only baselines;
and the fixed final negative result. Inspect figures and exports before retiring
NICE. Remaining flow sources consume the shared harness, so retain it.

WaveNet report continuation (2026-11-10): the13-block report now keeps all27
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

The preceding NICE source, whose unchanged48-update procedural banana
run is frozen in21 archives/117586 values with8 source pins and no external input
requirement. Independent audit and native data/coupling/inverse APIs now pass;
all144 update archives/19056 values match. The report is now complete, as recorded above. The in-process synthetic density
is explicit, not a substituted real dataset. Existing AutoRec/GAN diagnostics and
character_rnn recipe questions remain pending; ImageGPT's original prepared
palette corpus is also absent from the local cache, so its recipe needs recovery
before that source can run. No substitute assets/splits are used.

WaveNet cache continuation (2026-11-10): native `nn/wavenet_generation.hpp`
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

PixelCNN++ report continuation (2026-11-10): the complete native12-block
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

MADE density continuation (2026-11-10): native `nn/made_density.hpp` promotes
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

PixelRNN continuation (2026-11-10): native `nn/recurrent_pixels.hpp` promotes
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

Gated PixelCNN continuation (2026-11-09): native `nn/gated_pixels.hpp` promotes
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

PixelCNN continuation (2026-11-09): canonical `nn/autoregressive_pixels.hpp`
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

GAN diagnostics infrastructure (2026-11-09, report still pending): native
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

SinGAN continuation (2026-11-09): canonical `nn/single_image_gan.hpp`
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

SRGAN/ESRGAN continuation (2026-11-09): canonical `nn/relativistic_gan.hpp`
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

StarGAN continuation (2026-11-09): canonical `nn/uniform_categorical.hpp`
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

StyleGAN continuation (2026-11-09): canonical `nn/modulated_conv.hpp`
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

CycleGAN continuation (2026-11-09): canonical `nn/cycle_domains.hpp`
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

Pix2Pix continuation (2026-11-09): canonical `nn/quantized_sobel.hpp`
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

Progressive GAN continuation (2026-11-09): canonical `nn/pixel_rms_norm.hpp`
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

Wasserstein GAN continuation (2026-11-09): canonical `nn/midpoint_uniform.hpp`
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

DCGAN continuation (2026-11-09): canonical `nn/captured_affine_batch_norm.hpp`
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

BigGAN continuation (2026-11-09): canonical `nn/captured_batch_norm.hpp`
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

Self-Attention GAN continuation (2026-11-09): canonical
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

Spectral GAN continuation (2026-11-09): canonical `nn/spectral_cache.hpp`
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

InfoGAN continuation (2026-11-09): canonical `nn/info_gan.hpp` exposes
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

Compact dense GAN continuation (2026-11-09): `nn/dense_gan.hpp` and
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

Bidirectional GAN continuation (2026-11-09): canonical `nn/gan_data.hpp` and
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

Auxiliary-classifier GAN continuation (2026-11-09): canonical `nn/gan_models.hpp`
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

Reconstruction-anomaly continuation (2026-11-09): canonical
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

Learned-image-codec continuation (2026-11-09, earlier checkpoint): canonical
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

Latent-visualization continuation (2026-11-09, earlier checkpoint): canonical
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

Latent-arithmetic continuation (2026-11-09, earlier checkpoint): canonical
`nn/latent_arithmetic.hpp` now owns explicit affine interpolation/edits, detached
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

Encoder-pretraining continuation (2026-11-09, earlier checkpoint): canonical
`nn/encoder_pretraining.hpp` now owns frozen ReLU features, fitting-only
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

Restoration continuation (2026-11-09, earlier checkpoint): native `nn/patch_restoration.hpp` now
owns generic target-free visible-patch replacement and observed/predicted
composition. Finite signed Float32/64[B,P,D] values, strides, detached binary
masks[B,P], all-visible/all-hidden cases and checked fill narrowing are covered.
Selected branches preserve bits, including signed zero, with live selected
continuous first/second paths. All six definitions, two compiled examples,
standalone include and two clangd hovers pass. Native model tests total 168;
all 130 focused Python checks and 941 R expectations across 27 files pass.
All 63 inventory/tooling checks pass after retirement.

Twelve complete C++ programs, nine inspected native figures and both exports
replace the original and generated reference launchers. The three unused
restoration, dense-denoising and masked compatibility headers are removed;
shared autoencoder helpers remain for other consumers. Four prerequisite
native programs show all 288 three-model Adam updates and reproduce all three
checkpoint archives / 115,633 parameter values byte-for-byte. The restoration
application adds zero updates and checks all three stores for immutability.
It uses only observations/locations at inference, verifies hidden-value
invariance and copies every visible pixel exactly. All four archives /
4,704,817 values remain exact. Source delegation retains all TSV/JSON values;
report Float64 MSE and PSNR reductions differ below 1.5e-14 and 3e-13.
Original/core/standalone/report NumPy receipts agree for 156,800 Gaussian
values, 627,200 corrupted pixels, 1,593,600 forward values, 200 masks, all four
denoising conditions/PSNRs and all four hidden losses. Maximum forward error
is below 1.05e-6. At sigma 0.35 the denoiser beats raw input but loses to the
ordinary control; hidden MAE MSE 0.080208 loses to fitting mean 0.068530 while
beating zero fill 0.116914 and mismatched DAE 0.128178. All unfiltered failures
and all 200 per-case comparisons are retained. Evidence:
`build/nn-report-conversion/restoration_core/` and `restoration_report/`.

WAE continuation (2026-11-09, earlier checkpoint): native `nn/wasserstein_autoencoder.hpp` now
owns deterministic tanh encoding/decoding, pixel-summed squared reconstruction
cost and the explicitly weighted signed-MMD objective. Defaults preserve eight
leaves / 26,036 scalars, source order and weight 10. Unequal image/prior counts,
Float32/64, strides, six owning Tensor fields plus nested seven-field MMD,
first/second continuous paths and pre-leaf checks are covered. Sampling and
Adam remain caller-owned. SWIG includes the canonical MMD declaration before
the WAE record; its nested getter copies the record for Python/R lifetime.
All 16 definitions, five compiled examples, standalone include and five clangd
hovers pass. Native model tests total 161; all 126 focused Python checks and
910 R expectations across 26 files pass. All 63 metadata checks pass after retirement.

Nine complete report programs, seven inspected native figures and both exports
replace the source and generated reference launchers. The unused WAE, beta-VAE
and kernel-discrepancy compatibility headers are removed; shared autoencoder
and masked/denoising helpers remain for applications. Eighteen archives /
1,147,656 values and TSV histories remain exact; report summary MSE roundoff
is 3.5388e-15 and 2.7894e-15, with other non-timing JSON exact. Core delegation
preserves non-timing JSON exactly. Original/core/standalone/report NumPy receipts
agree: 26,036 first derivatives, exact first Adam, 52 endpoint fields / 658,778
values, 3,776 Gaussian values, signed MMD/translation controls, all three eligible
validation candidates, 630 batch rows and 20 unfiltered prior predictions.
Maximum endpoint error is below 8.17e-6. Epoch 3 selects reconstruction alone:
test squared cost 80.780533, signed MMD 0.689407 and MSE 0.103036, worse than the
fitting-only mean 0.048548. Rising validation discrepancy and weak outputs are
retained; no exact Wasserstein distance, likelihood or generation-quality claim.
Evidence: `build/nn-report-conversion/wae_core/` and `wae_report/`.

Two-scale-VQ continuation (2026-11-09, earlier checkpoint): native
`nn/vector_quantized_autoencoder_2.hpp` now owns coarse-code conditioning,
independent fine quantization, both code-only decoder stages and the copied
parent/previous-symbol conditional prior. Defaults preserve fourteen leaves /
1,157 scalars, original creation order and exact objective grouping. Float32/64,
non-square geometry, strides, checked overrides/products, eleven owning fields
and nested quantizer lifetimes are covered. Reconstruction skips both books;
fine losses reach the coarse encoder through its identity surrogate. Prior
sampling takes explicit Float64 uniforms and resets BOS per image, not per row.
All 36 definitions, eight compiled examples, standalone include and six clangd
hovers pass. Eight new cases bring native model tests to 153; all 121 focused
Python checks and 865 R expectations across 25 files pass. All 63 metadata
checks pass after retirement.

Ten complete report programs, eight inspected native figures and both exports
replace the source and generated reference launchers. Both unused legacy VQ
headers are removed. All fifteen archives / 1,932,136 array values and TSV
histories remain exact; two report Float64 MSE summaries differ by 2.2898e-16
and 2.7894e-15, with other non-timing JSON exact. Core delegation preserves
non-timing JSON exactly. Independent NumPy receipts agree for original,
delegated, standalone and report outputs: 1,157 first derivatives, exact first
Adam, 68 endpoint fields / 1,549,828 values, 6,272 hierarchy derivatives, both
reconstruction-to-book stops, two top-conditioning interventions, all three
validation candidates, 630 batch rows, 4,624 prior count cells, 4,900 uniforms
and twenty unfiltered prior images. Maximum endpoint error is below 3.23e-5.
Epoch 3 test NLL 117.445175 and MSE 0.038228 improve on the fitting-only mean
MSE 0.048548; nine coarse and thirteen fine codes occur. Counterfactual grids,
code-use imbalance and weak unfiltered samples remain visible. These checks
do not establish semantic scale separation, generation quality or a codec.
Evidence: `build/nn-report-conversion/vq2_core/` and `vq2_report/`.

Spatial-VQ continuation (2026-11-09, earlier checkpoint): native
`nn/vector_quantized_autoencoder.hpp` now owns nearest-vector lookup, exact
selected-vector forward/identity surrogate routing, separate stopped dictionary
and commitment penalties, generic spatial Conv/ReLU/avg/repeat inference,
code-only decoding, code-use counts/perplexity and an explicit-uniform raster
Markov prior. Defaults preserve K16/D4/C8, source leaf order and nine leaves/
805 scalars. Float32/64, strides, checked shapes/IDs/products, assignment
overrides, owned nested records and continuous surrogate higher paths are
supported. The prior copies fitting codes, exports independent tables and
scores nats per grid; it is a first-order half-count teaching baseline, not
PixelCNN or image likelihood. No EMA, resets, fitting loop or optimizer is hidden.
All 39 definitions, twelve compiled examples, standalone include and six clangd
hovers pass. The hover driver now closes its input after LSP exit and allows
30 seconds for parser-worker shutdown under concurrent builds. Ten new cases
bring native model tests to 145; 116 focused Python checks and 830 R expectations
across 24 files pass. All 63 metadata checks pass after retirement.

Eight complete report programs, six inspected native figures and both exports
replace the standalone source, unused legacy model adapter and generated
reference launchers. All 21 explicit Adam updates, source batch order,
three eligible validation reconstruction-NLL criteria, fresh selected restore,
ten evenly spaced held-out rows, unused dictionary codes and twenty unfiltered
prior outputs remain visible. Core delegation preserves both VQ source runs:
all 14+15 archives/2,691,906 values, TSVs and non-timing JSON exactly. The report
preserves all 14 archives/759,770 values and histories; two tensor Float64 MSE
reductions differ by 2.1233e-14 and 2.7894e-15, with other non-timing JSON exact.
NumPy checks all 805 first surrogate derivatives, exact first Adam, 40 saved
endpoint fields/463,588 values, complete 630-row batch history, every validation
candidate, all 272 prior counts, 980 uniforms and 20 prior predictions. Maximum
endpoint error is below 3.20e-5; original/delegated/standalone/report receipts agree.
Epoch 3 test NLL 341.475494 and MSE 0.126616 do not beat the fitting-only mean's
MSE 0.048548. Only 5/16 fitting codes occur; pooled perplexity 2.288046 and weak
unfiltered images do not establish generation quality, semantics or a codec.

The next AutoRec source/helper/shared-helper snapshots are pinned in
`build/nn-report-conversion/autorec/preparation.json`. Its original ratings
assets and preparation recipe are not present in the inspected checkout/cache;
the source refers to an unavailable `tflow_examples` task. Clarification was
requested asynchronously while restoration work continued. No numerical
AutoRec baseline or native preparation is claimed yet. Preserve the original
split/recipe rather than inventing a compatible replacement. Latent arithmetic,
latent visualization, learned image codec and reconstruction anomaly are verified
above. AutoRec remains the only unconverted autoencoder original.

The next generative-section ACGAN original source/helper/shared snapshots are
pinned with ten archives/919949 values. Its existing shared GAN header needs an
explicit dataset-assets include for standalone compilation; the baseline build
adds that dependency without editing the original. Independent NumPy passes
749714 forward values (maxerror1.06e-6),53472 isolated source/class generator
partials and24 scalar finite differences, all3000 batch rows, fixed final epoch3,
96 updates/group and192 total. Exact noise/condition controls verify that the
conditional generator depends on requested classes while D is image-only. Coarse
MMD is a game diagnostic: test .117949 is much worse than fitting-mean .000135779,
and real/fake auxiliary accuracies .39/.185 retain weak measured results. Native
promotion/report are pending. Evidence: `build/nn-report-conversion/auxiliary_classifier_gan/`.
The next BiGAN original has nine archives/6,210 values plus all source/shared
snapshots and raster figures frozen. No external dataset is needed: exact procedural
ring points and seeded noise are checked independently. NumPy passes 3,398 forward
values (maxerror2.36e-7), 1,312 exact point values, 768 exact noise values, 82 stored
G partials after a widened first-D Adam replay and 32 scalar finite differences.
All 1,200 batch rows, 39 updates/group and 117 total, joint-pair controls, fixed-final
epoch3 and independence of the G gradient from E with D fixed are checked. The
first-D replay uses widened equations then Float32 parameter rounding, not a
bit-exact native optimizer-state claim. Reconstruction MSE2.133117 is worse than
the fitting-mean2.009766; cycle/reconstruction are untrained diagnostics. Canonical
promotion/report remain pending. Evidence: `build/nn-report-conversion/bidirectional_gan/`.
The 129 older CNN reports still need retirement revalidation.

Vanilla-VAE continuation (2026-11-09): nine complete C++ programs, seven
inspected native figures and both exports replace the standalone source and
generated reference launchers. The existing canonical Gaussian API supplies
beta=1, latent2/hidden64 inference, stable pixel NLL, analytic KL and code-only
decoding. Five legacy math functions now forward to that core, preserving their
signatures, eight-field record, ten leaves and 101,652-scalar initialization.
Five canonical-equivalent documentation examples compile/run. Python/R help
specifies the configuration; new ownership/gradient/caller-Adam checks pass.
All 135 native model cases, 112 focused Python checks, 791 R expectations across
23 files and 63 metadata checks pass. This adds no new core symbols.

All 96 visible updates, fixed/fresh Gaussian banks, three eligible validation
candidates, fresh selected-model restore, ten first-per-digit held-out rows,
twenty unfiltered prior outputs and the Cartesian 9x9 code grid are retained.
All 16 archives/8,817,594 values and complete CSV/TSV histories match the frozen
source exactly. Only the tensor Float64 summary MSE reduction differs, by
2.998e-15; all other non-timing JSON values remain exact. Independent NumPy
checks all 101,652 first derivatives, exact first Adam, 17,240 Gaussian values,
1,097,600 thresholded pixels, 51 endpoint fields/5,747,784 values and all eligible
validation criteria. Maximum endpoint error is below 3.03e-5; original,
delegated, standalone and final report summaries agree. Epoch3 test negative
ELBO is 205.507103; decode(mu) MSE 0.080653 modestly improves on the fitting-only
binary mean's 0.086690. These are tiny statically thresholded MNIST pools, not
an official binarized-MNIST benchmark, and the report makes no semantic-axis claim.

The shared autoencoder helper remains for latent diffusion and other consumers.
Both remaining VQ examples were frozen before shared delegation; source runs
after delegation retain all 14+15 archives/2,691,906 values, histories and
non-timing JSON exactly. Together with vanilla VAE this covers 45 archives/
11,509,500 values. Independent NumPy now verifies both VQ baselines: all 805+1,157 initial
parameter derivatives, both exact first Adam updates, 40+68 saved endpoint
fields/2,013,416 values, every eligible validation candidate and both complete
630-row batch histories. VQ-VAE-2 adds 6,272 fine-to-coarse derivatives,
zero reconstruction gradients to both codebooks and two top-code interventions.
All 272+4,624 prior count cells, explicit MT19937 midpoint uniforms and all
forty unfiltered prior decodes pass. Original/delegated oracle receipts agree
for each model. Maximum endpoint errors are below 3.24e-5; maximum first-gradient
absolute errors are 2.16e-4/2.30e-5 under Float32-relative checks. These are
pre-promotion checks, not completed reports. VQ quantization, spatial decoding
and fitting-only Markov prior are next for canonical promotion. The 129 older CNN reports still need retirement
revalidation; report presence alone is insufficient. Evidence:
`build/nn-report-conversion/vanilla_vae_report/`,
`vector_quantized_autoencoder/baseline.json` and
`vector_quantized_autoencoder_2/baseline.json`.

VAE-GAN continuation (2026-11-09): native `nn/vae_gan.hpp` now owns the
code-only tanh/sigmoid generator, learned discriminator features/logits,
detached-real-target half-summed feature distance and all nineteen forward
endpoints. Three explicit objectives preserve source balanced fake weights,
non-saturating generator terms and diagnostic-only Float32-clipped pixel NLL.
General positive widths, unequal prior batches, Float32/64, strides, owning
fields, continuous higher derivatives and validation before leaf creation are
supported. The caller selects only each intended group and recomputes D then E
then G; no fitting, RNG draws or mode changes are hidden. All 30 definitions,
six compiled examples, standalone include and five clangd hovers pass. Seven
new cases bring native model tests to 135; 111 focused Python checks and 775 R
expectations across 23 files pass, including all nineteen fields through GC and
three isolated Adam states. All 63 metadata checks pass after retirement.

Eight complete report programs, five inspected figures and both exports replace
the standalone source, unused VAE-GAN helper and generated reference launchers.
The report retains all 63 visible updates, source leaf creation/order, explicit
posterior/prior banks, three eligible validation candidates, fresh all-group
restore, first-phase audit, ten unfiltered held-out rows and twenty prior outputs.
Canonical delegation preserves all 24 archives/3,078,235 values, TSVs and
non-timing JSON exactly. The report preserves all archives/TSVs and all other
non-timing JSON; its two tensor Float64 summary MSE reductions differ from host
accumulation by 1.02e-14 and 2.79e-15. Independent NumPy verifies all 64,786
initial/recomputed group derivatives, three exact first Adam steps, unrelated
group isolation, 14,528 Gaussian values, all nineteen endpoints across 304 fields/
2,419,786 values and every validation candidate. Maximum endpoint error is below
3.40e-5; original/delegated/report oracle summaries agree exactly. Epoch3 test
MC pixel MSE 0.247712 is worse than fitting mean 0.048548; no perceptual-quality,
image-ELBO or matched-control improvement is claimed. The next untouched vanilla
VAE baseline retains 16 archives/8,817,594 finite values. Its latent2/hidden64
Gaussian model is the next conversion. A scratch forwarding probe through the
existing canonical Gaussian APIs preserves all 16 archives, TSVs/CSVs and
non-timing JSON exactly. NumPy checks all 101,652 first derivatives, exact first
Adam, 17,240 Gaussian values, 5,747,784 endpoint values and every validation
candidate on original/probe artifacts. The source/shared helper remained unchanged at that checkpoint; the verified
vanilla conversion above now delegates the shared math and retires the source.
The 129 earlier CNN reports still need
retirement revalidation. Evidence: `build/nn-report-conversion/vae_gan_core/`,
`vae_gan_report/` and `variational_autoencoder/baseline.json`.

Sequence-VAE continuation (2026-11-09): eight complete C++ programs, five
inspected native figures and both computed exports replace the original sequence
source, unused compatibility helper and generated reference launchers. Native
`MidpointUniformNoise` owns the exact detached Float64 one-word-per-value MT19937
stream, including full unsigned32 seeds, scalar/empty shapes and pre-allocation
negative/overflow checks. Its explicit uniforms preserve ancestral token draws;
Float32 casts can round to one, so categorical draws retain Float64. All 31
header definitions, 13 compiled examples, standalone include and eight clangd
hovers pass. All 128 native model cases, 106 focused Python checks and 743 R
expectations across 22 files pass; 63 inventory/runner/website checks pass after
retirement. The report preserves all 27 visible Adam updates, source leaf creation,
three eligible validation candidates, fresh selected-model restore, unigram
reference, fixed-code causality and all unfiltered held-out/prior rows.
All 20 archives/1,454,190 values, complete TSVs, vocabulary and every non-timing
JSON value match the untouched source exactly. NumPy checks all 3,243 first BPTT
derivatives, exact first Adam, 10,448 Gaussian values, all 18 endpoints across
319 saved fields/1,396,642 values, every validation column, independent token
uniforms, test scores and EOS/grammar/uniqueness counts. Maximum endpoint error
is below 1.69e-6. Epoch 3 test negative ELBO is 18.680923; teacher token perplexity
16.404448 is worse than fitting-only unigram 15.689205. Neither prior policy
produces a valid grammar sentence; sampled generation misses EOS in 12/20 cases.
The next untouched VAE-GAN baseline retains 24 archives/3,078,235 values;
independent NumPy verifies 64,786 initial/recomputed D/E/G derivatives, three
exact first Adam steps, unrelated-group isolation, 14,528 Gaussian values and
all 19 forward fields. Its core/report conversion is now completed above. The 129 earlier
CNN reports still require retirement revalidation. Evidence:
`build/nn-report-conversion/sequence_report/` and `vae_gan/`.

Posterior-collapse continuation (2026-11-09): the verified report now replaces
its standalone source, unused posterior helper and generated Python/R reference
launchers. Native `nn/sequence_vae_data.hpp` owns the exact vocabulary, exhaustive
432-case grammar/schema loader, owning pool records, original prefix-text/grammar
diagnostics and fifteen-array experiment export. The loader preserves valid row
permutations, accepts original config shapes, verifies every source ID, factor,
split, EOS/PAD, length, balance and previous-target relation, and never prepares
or fits. Python/R preparation delegates to the canonical datasets API; pool
selection copies Tensor handles so collected vectors/files cannot invalidate
fields. All 17 definitions, six compiled examples, standalone include and six
clangd hovers pass. Five new cases bring native model tests to 126;105 focused
Python checks and 730 R expectations across 22 files pass;63 inventory/runner/site
checks pass after retirement. The complete report retains both 27-step fits,
source initialization/order/noise, six common validation candidates, fresh model
restore, original synthetic controls, signed full-mixture diagnostics, four code
ablations and all ten fixed free-decoding rows per schedule. Nine C++ blocks,
six inspected native figures and both verified exports pass. All 35 source
archives/2,265,740 values, every TSV and all non-timing JSON values remain exact.
NumPy checks all 6,486 first parameter derivatives, both exact first Adam steps,
20,736 Gaussian values, all 18 saved endpoints/first forwards,2,147,866 endpoint
values, all six eligible validation criteria, code ablations, free generation and
signed mixture/control densities. Maximum endpoint error is below 1.86e-6.
Constant/warmup test negative ELBOs 18.680922959/18.687372672 and zero active mean
coordinates do not support a warmup-cure or semantic-usefulness claim; unfiltered
free decoding remains poor. The retained sequence example also preserves all 20
archives/1,454,190 values, complete TSVs and non-timing JSON values after loader
delegation. Its shared helper is now retired by the verified sequence report above.
The 129 earlier CNN reports still need their own retirement revalidation; report
presence alone does not satisfy the migration. Evidence:
`build/nn-report-conversion/posterior_report/`.

Sentence-model continuation (2026-11-09): native
`nn/sequence_variational_autoencoder.hpp` owns the exact masked GRU16 encoder,
separate embedding8 books, four-dimensional explicit Gaussian posterior, all
18 forward fields, prefix-only teacher decoder and explicit greedy/ancestral
seven-step generation. Source 22 leaves/3,243 scalars and Float32 operation order
are retained; Float64, strides, owned endpoints and continuous first/second
paths are supported. EOS stops emitted IDs with PAD suffixes while raw decoder
states/scores continue; hard choices have no straight-through gradient.
`nn/posterior_collapse.hpp` owns the explicit source-rounded 18-step KL schedule
and host-double analytic/per-unit, population-variance and full empirical-mixture
diagnostics. General nonempty latent widths, strict activity thresholds, signed
Monte Carlo values, self-inclusion bias and overflow errors are documented.
Legacy helpers now import canonical mathematics; their source-specific training,
corpus validation and JSON adapters remained until their report retirements above. Both source
programs retain all 55 archives/3,719,930 values byte for byte, every TSV and every
non-timing JSON value, including diagnostics. Ten new cases bring native model
tests to 121: all 3,243 Float64 leaf derivatives, noise second derivatives,
causality, exact PAD masks, EOS/CDF controls, invalid input and owned lifetimes.
All 102 focused Python checks and 712 R expectations across 21 files pass.
All 57 definitions in the two headers pass fresh Doxygen, 13 public examples
compile/run, standalone includes compile and 10 clangd hovers pass. NumPy checks
all first BPTT derivatives and exact first Adam for each of three fits, all 18
endpoints including first forward, all 9 eligible validation checkpoints, Gaussian
banks, prior token uniforms, all saved generation controls, unigram baseline and
signed mixture diagnostics. Maximum endpoint error is below 1.86e-6; original and
delegated oracle summaries agree. 63 metadata/runner/website checks pass.
The corpus API and posterior-collapse report are now verified by the continuation
above. The sequence helper was retained at this checkpoint and is now retired above.
Evidence: `build/nn-report-conversion/sequence_vae_core/`.

Sentence-data continuation (2026-11-09): native
`prepare_example_data("sequence-vae-assets")` and `just nn-sequence-vae-assets`
now prepare the exact local 432-sentence grammar, all sixteen Int32 arrays,
fixed vocabulary, compositional 288/72/72 splits and ascending source-ID order.
Provenance distinguishes generated data from caller-supplied archives; supplied
archive/vocabulary bytes are preserved, and missing outputs are repaired.
All 15 native dataset cases and 63 migration/website checks pass. All 69 affected
area definitions pass fresh Doxygen, three representative clangd hovers pass and
the two public preparation examples compile/run. Both untouched programs pass
on this data: posterior collapse preserves 35 archives/2,265,740 values,
and sequence VAE preserves 20 archives/1,454,190 values. Constant-beta
and standalone sequence trajectories match exactly, including all 22 parameter
leaves/3,243 scalars, first gradients/Adam, all three epoch arrays and selected
weights; both schedules share initialization, noise and complete batch orders.
Both select epoch3. Constant/warmup test negative ELBOs are
18.680922959/18.687372672; warmup has zero active units at the stated threshold,
so do not claim it cures collapse. This establishes C++ baselines on the loader's
complete grammar contract, without external-framework trajectory claims.
The sentence-model continuation above now moves the shared GRU and posterior
diagnostics into canonical C++ APIs and exposes their bindings. Corpus validation
and the executable reports still remain; keep the sequence helper until both
reports are verified.
Evidence: `build/nn-report-conversion/sequence_vae_data/`,
`posterior_collapse/baseline.json` and
`sequence_variational_autoencoder/baseline.json` below the same build directory.

Ladder continuation (2026-11-09): `nn/ladder_variational_autoencoder.hpp`
now owns stable source-ordered precision fusion, explicit ancestral lower-prior
sampling and the full sixteen-field two-level forward. ConditionalGaussianKl
and GaussianVaeDecode are reused; only the lower code enters decoding. Lower
noise precedes top noise, whose learned-prior and fused-posterior paths remain
live. The report retains 22 leaves/52,196 scalars, all 21 visible Adam updates,
four-draw validation selection and fresh model-only restore. Epoch3 is selected;
test NLL/lowerKL/topKL are 234.154285/7.508567/4.866975, totaling 246.529827.
Recursive greedy MSE 0.073145999 is worse than fitting mean 0.048548290. All
20 original archives (2,014,940 values), complete histories and batch orders
match byte for byte; MSE reduction roundoff is below 2e-14. Independent NumPy
checks all 52,196 first derivatives, exact first Adam, all 16 fusion partials,
10,956 Gaussian values, 1,489,600 lattice/prediction pixels, all fifteen source
endpoints across each saved draw/intervention and three validation checkpoints.
The native forward additionally owns logits, covered by binding lifetime checks.
Seven new cases bring native tests to 111; 97 focused Python tests and 676 R
expectations across 20 files pass. Six public examples compile/run, the header
compiles alone, all 34 definitions pass fresh Doxygen and six clangd hovers pass.
Seven report programs, five inspected figures and both exports pass. The
standalone source, unused Ladder helper and generated reference launchers are
removed. At that checkpoint, the next posterior-collapse/sequence-VAE examples
needed native sentence preparation before capturing their untouched baselines.
The exact grammar, split, token/previous-target schema and vocabulary are checked
by LoadSequenceVaeData. The sentence-data continuation above resolves this
prerequisite and captures BOTH original runs before moving shared model math.
Evidence: `build/nn-report-conversion/ladder_vae/` and
`build/nn-report-conversion/posterior_collapse/prerequisites.json`.

InfoVAE continuation: `nn/kernel_discrepancy.hpp` now owns the three-scale
IMQ kernel and all seven signed unbiased MMD terms, including unequal sample
counts, live input derivatives and unclipped negative estimates.
`nn/info_variational_autoencoder.hpp` adds explicit objective coefficients,
image-free tanh decoding and the complete sixteen-field Gaussian forward. The
caller supplies posterior and independent prior draws, Adam and epoch selection.
The matched alpha0.5/alpha1 variants retain both 21-step fits and own-objective
four-draw validation, rather than comparing their weighted values as ELBOs.
Both select epoch3; test objectives 404.910553/403.818367 differ from diagnostic
negative ELBOs 406.237358/410.269318. Decode(mu) MSE 0.152413267/0.154711297 is
worse than fitting mean 0.048548290. All 35 original archives (7,209,428 values),
full histories and batch orders match byte for byte, with MSE reduction roundoff
below 2e-14. NumPy checks 52,208 first parameter derivatives, both exact first
Adam updates, 29,056 Gaussian values, 2,847,488 lattice/prediction pixels,
1,434,984 kernel entries, all sixteen saved endpoints and six own-objective
validation checkpoints. Eight new cases bring native model tests to 104;
92 focused Python tests and 638 R expectations across 19 files pass. Seven
public examples compile/run; the header compiles alone, all 38 definitions in
two headers pass fresh Doxygen and seven clangd hovers pass. Eight report
programs, six inspected figures and both exports pass. The standalone source,
unused InfoVAE helper and generated reference launchers are removed. The shared
kernel compatibility include remains for WAE, delegating to canonical C++;
all 18 WAE archives (1,147,656 values), histories, batches and non-timing metrics
remain exact against a separately captured pre-delegation baseline. The next
untouched Ladder VAE baseline has 20 archives. Evidence:
`build/nn-report-conversion/info_vae/` and
`build/nn-report-conversion/wasserstein_vae-prekernel/`.

Importance-weighted continuation: `nn/importance_weighted_autoencoder.hpp`
now owns stable sample-axis log-mean-exp, sampled Gaussian density-ratio bounds,
normalized weights/ESS and the complete multi-sample Gaussian/Bernoulli forward.
Float32 preserves source-rounded log constants and sample-major tiled encoding;
Float64 retains full precision. All five bound and eleven forward fields own
lifetime. Full ordinary pathwise derivatives remain live; analytic KL is only
a diagnostic and no DReG substitution is introduced. Caller-owned Adam and
common fixed K20 validation retain the matched K1/K5 experiment, both 21 updates
and epoch-3 selections. Test K20 negative bounds are 201.516907/195.132950;
decode(mu) MSE 0.069591819/0.066122890 is worse than fitting mean 0.048548290.
All 34 original archives (10,980,972 values), full histories and batch orders
match byte for byte, with MSE reduction roundoff below 2e-14. Independent NumPy
checks all 102,832 first derivatives, both exact first Adam updates, 36,200
Gaussian values, 9,942,688 lattice/prediction pixels, every saved importance
weight/ESS/decoder draw, fitting-only probes and all six common validation scores.
Seven new cases bring native model tests to 96; 87 focused Python tests and
591 R expectations across 18 files pass. Five public examples compile/run;
the header compiles alone, all 22 definitions pass fresh Doxygen and five clangd
hovers pass. Seven report programs, five inspected figures and both exports pass.
Finite-draw nonmonotonicity, poor reconstruction and concentrated importance
weights remain explicit. The standalone source, unused IWAE/flow helpers and
generated reference launchers are removed. The untouched next InfoVAE/MMD-VAE
baseline has 35 archives. Evidence: `build/nn-report-conversion/importance_vae/`.

Spatial hierarchy continuation: `nn/hierarchical_variational_autoencoder.hpp`
now owns direct/residual Gaussian heads, conditional Gaussian KL, the miniature
NVAE depthwise/BN/Swish/SE cell, both-latent image-free decoding, ancestral
sampling and the complete paired forwards. Separate weight/moving-statistic
stores make the optimizer and BN update schedule explicit. All 19 endpoints own
their lifetimes; supplied lower/top noise is ordered explicitly. The ordinary
model has 1,977 scalars and the miniature NVAE has 2,027. They remain different
architectures, not a matched ablation or full NVAE. The report retains both 21-step Adam fits,
four-draw inference-mode validation selection and paired weights/state restore.
Both select epoch 3; test negative ELBO 239.567580/297.563977, greedy
MSE 0.104085294/0.106240368 versus fitting-mean 0.048548290. All 41 original archives
(5,808,422 array values), complete histories and batch orders match byte for byte;
only MSE reduction roundoff is below 2e-14. The independent NumPy oracle checks
4,004 first derivatives, both exact first Adam updates, all six first BN buffer
updates, 186,252 Gaussian values, 2,797,312 lattice/prediction pixels, all 19 forward
fields, isolated prior-kernel regularization and six eligible validation scores.
Eight new native cases bring the model suite to 89; 82 focused Python tests and
560 R expectations across 17 files pass. All nine public header examples compile
and run; the header compiles alone. Fresh 46/46 definitions and nine clangd hovers pass.
Nine report programs, seven visually inspected figures and both exports pass.
The unused hierarchy helper and generated reference launchers are removed;
the ladder helper remains for its pending conversion. The untouched IWAE
baseline is saved before migrating its math. Evidence:
`build/nn-report-conversion/hierarchical_vae/`.

The first new report in this continuation is the undercomplete autoencoder.
Its reusable forward model now lives in `nn/autoencoder_models.hpp`; the report
retains the full 784–32–784 geometry, explicit Adam(0.003) minibatch loop,
validation checkpoint selection, genuine MNIST and reconstruction controls.
Six blocks execute, all 156,800 held-out pixels agree with an independent NumPy
forward check within 3.6e-7, and the selected model obtains MSE0.050325811 versus
0.067798078 for the fitting pixel mean. Three model tests plus the tape/model-plot
suites pass (59 tests total); all three new documentation examples compile/run,
28/28 affected Doxygen definitions pass, and both new clangd hovers pass.

Retirement revalidation exposed duplicated fenced results in four early reports.
`Tape::to_markdown()` now provides escaped table output without nested fences;
the duplicate result fragments are removed. All eight tensor/graph reports run
32 runnable blocks. The shared report runner now serializes siblings in the
same working directory, preventing mep from copying another report's newest SVG;
independent directories retain parallel execution. Local evidence and exports:
`build/nn-report-conversion/`.

Contractive autoencoder continuation: the exact sigmoid-encoder penalty and
untied sigmoid/MSE forward model now live in `nn/autoencoder_models.hpp` and
are exposed through the functional SWIG API. Python/R examples delegate those
operations to C++ while retaining their own optimizer loops. A nested penalty
is returned as an owning value so it survives parent collection. All six report
blocks and figures pass; seven native model tests, 33 focused Python tests and
115 R expectations across three test files pass. Six header examples compile/run,
16/16 header definitions pass Doxygen and all three model hovers pass. The
original C++, Python and R three-epoch runs produce **31 byte-identical NPZ
archives**; the report's nine initial/first-update/selected parameter snapshots
also match the original. The optional development oracle
`scripts/check_contractive_autoencoder.py` checks all470,400 reconstructed pixels,
602,112 full Jacobian entries, complete first-batch chain-rule gradients and
bit-identical Float32 Adam updates. The original C++ program and its now-unused
`common/contractive_autoencoder.hpp` are removed. Evidence:
`build/nn-report-conversion/contractive/`.


Convolutional autoencoder continuation: the reusable 741-scalar spatial forward
model and 369-scalar standalone decoder now live in `nn/autoencoder_models.hpp`.
The legacy component wrappers delegate their arithmetic to those functions.
The six-block report preserves the original 28×28×1 → 7×7×4 geometry, all 96
visible Adam(.003) updates, validation-only selection and mean/pool/initial/
zero/shuffled controls. Epoch3 validation MSE is0.018339728; held-out MSE is
0.017783357 versus0.067798074 for the fit mean and0.041953766 for pool/repeat.
All six parameter snapshots match the executed original bit for bit. The
optional `scripts/check_convolutional_autoencoder.py` independently checks
all741 first-batch derivatives (max error9.9e-9), the exact first Adam update,
all156,800 held-out pixels (max error7.1e-7), six activation maps and official
disjoint source IDs. All five figures and both export bundles pass inspection.
Ten native model tests, ten compiled header examples,28/28 header definitions
and five clangd hovers pass. Python's34 focused runtime tests and R's127 expectations across three test files
pass, together with63 inventory/runner tests. All eleven saved NPZ archives
match across the original C++, Python and R runs byte for byte. The
original standalone C++ source is retired; the compatibility header remains
for independent language scripts and later autoencoder applications. Evidence:
`build/nn-report-conversion/convolutional/`.

Denoising autoencoder continuation: native `models::GaussianNoise` and
`models::CorruptForDenoising` now own the explicit MT19937/Box-Muller banks and
detached Float32 corruption. Shared C++, Python and R helpers forward to them;
the language helpers also use the canonical undercomplete forward model.
The eight-block report preserves clean targets, matched initialization/batching,
three fresh epoch noise banks, both96-update Adam(.003) fits and selection on
the same fixed noisy validation task. Both select epoch3. Main held-out MSE is
0.059952738 for the denoiser,0.062168205 raw and0.053709699 for the ordinary
control; low-noise/clean-input failures remain explicit. All six figures and
both export bundles pass review. The original standalone source is retired;
`common/denoising_autoencoder.hpp` remains because restoration examples use it.
The optional `scripts/check_denoising_autoencoder.py` independently checks
2,822,400 Gaussian values (max error4.5e-16),3,136,000 bit-identical corruption
pixels,101,984 parameter derivatives (max error7e-11),both exact first Adam
updates and1,568,000 held-out predictions (max error5.4e-7). All source arrays
in21 report archives match the original, including12 parameter snapshots.
Thirteen native model tests include signed-zero rounding before clipping;
all12 header examples compile/run,30/30 definitions pass and seven hovers pass.
R seed typemaps preserve the full unsigned32 range through numeric values,
including2147483648 and4294967295, rather than narrowing them to R integers.
Both rebuilt packages pass35 focused Python tests and141 R expectations in
three test files, including full-range seed and signed-zero/ownership checks.
All25 C++, Python and R output archives are byte-identical;63 inventory/runner
tests pass. The next original `k_sparse_autoencoder` has been executed as a saved baseline.
Evidence: `build/nn-report-conversion/denoising/`.

k-sparse continuation: `nn/competitive_autoencoders.hpp` now owns the stable
signed selector and both per-row/lifetime forward paths; legacy wrappers and
native Python/R helpers delegate the arithmetic to the core. The eight-block
report keeps a linear784–64–784 encoder/decoder, eight signed winners at
training and deployment, matched plain control, both96-update Adam(.003) fits,
first/partial/selected probes and validation-only selection. Signed selection
is explicitly distinguished from magnitude projection, including non-idempotence
and a tie-discontinuity counterexample. Both select epoch3; held-out MSE is
0.050034922 sparse,0.041335177 plain and0.067798074 fitting mean. The original
source is retired; its compatibility header remains for winner-take-all.

Figure review exposed a missing detached selector in architecture tracing.
`model_stage` now retains explicitly declared data-only producers in Blocks
view, with every declared consumer branch, declaration-order independence and
cycle rejection, without adding gradient edges. Expanded views keep recorded
differentiable ancestry; only-declared producer parameters can differ by view,
as documented/tested. All77 native model/layer/tape/plot tests pass, including
three new tracing tests and five competitive-model tests. All six figures and
both export bundles pass review. Three new model and five changed plotting
header examples compile/run;126/126 affected definitions and three hovers pass.
The optional `scripts/check_k_sparse_autoencoder.py` independently checks
202,400 derivatives (max1.7e-10), both exact Float32 first Adam updates, all
2,822,400 pool/control predictions (max5.1e-7), signed masks and a minimum
cutoff margin6.5e-5. All source arrays in27 report archives match by shape,
dtype and every value byte, including12 parameter snapshots. Both k-sparse
and the shared lifetime model produce29 byte-identical archives each across
original C++, Python and R runs.36 focused Python tests and152 R expectations
across three files pass, together with63 inventory/runner tests. The ownership
regression gate now inspects retired reports' actual C++ optimizer loops instead
of expecting deleted standalone files. The next masked-autoencoder original
has been executed and saved as a baseline. Evidence: `build/nn-report-conversion/k_sparse/`.

The masked-autoencoder report is verified and its standalone original retired.
`nn/masked_autoencoder.hpp` now owns checked differentiable patch conversion,
grouped fixed positions, source-compatible unsigned-hash selection, visible-only
encoding, target-free mask-token decoding and the generic hidden-pixel loss.
The full model retains all35 tensors/13,649 scalars and its exact source trajectory:
three epochs,96 explicit Adam updates, changing training masks and fixed validation
masks. Every source array in16 report archives agrees by dtype, shape and value
bytes, including all parameter snapshots, masks, first gradients, every pool and
control. All source arrays in16 original archives also agree across normal
C++, Python and R workflows. The optional `scripts/check_masked_autoencoder.py`
independently verifies all13,649 first derivatives (max5.3e-9), bit-exact first
Adam updates,4,600 mask rows and1,724,800 prediction pixels (max3.8e-7), with
attention, hidden-input independence and visible-order invariance checks.

All nine blocks and six inspected figures export successfully.24 native model
tests pass, including six new masked-model tests; all12 public header examples
compile/run, the header compiles standalone,30/30 definitions pass Doxygen and
four clangd hovers pass. 39 focused Python tests and180 R expectations across four files pass, alongside
63 inventory/runner tests. Binding ownership/error/unsigned-seed tests pass in both
languages. The remaining restoration compatibility consumer still runs through a
thin delegate, so its shared header is retained. The report states the measured
limitation: test hidden MSE0.079354655 is worse than the fitting-mean0.067858952,
and misaligned visible content barely changes the score (0.079606451). Raw model
predictions and visible-copy composites are separate gallery rows; no downstream
quality claim follows. Evidence: `build/nn-report-conversion/masked/`.

The recurrent-autoencoder report is verified and its standalone original retired.
`nn/sequence_autoencoders.hpp` owns checked row-sequence conversion, final-state
encoding and an independent repeated-code tanh decoder, preserving all 28+28
steps, eight tensors and 4,956 scalars. All source arrays in 13 report archives
agree with the original by shape, dtype and every value byte, including the
initial/first-update/epoch/selected parameters, both state histories, every pool
and frozen controls. The same 13 source archives agree across normal C++, native
Python and R runs. The fitting loop retains all 96 explicit Adam updates and
validation-only selection. Test MSE 0.079884621 is worse than the fitting mean
0.067798074; shifted codes give 0.079947607, so correct code alignment adds little
under this short budget. Reversed inputs are scored against the original order.

All eight programs and six inspected figures export successfully. All 29 native
model tests pass, including five new sequence tests and Float64 first/second
finite-difference checks; all five public examples compile/run, the header
compiles standalone, 12/12 definitions pass fresh Doxygen and four clangd hovers
pass. There are 41 focused Python tests and 203 R expectations across five files;
all 63 inventory/runner tests pass after retirement.
R test fixtures now parenthesize multiplication before modulo and assert [0,1]
ranges; their generated inputs match C++/Python. Independent NumPy checks all
4,956 first derivatives (max 2.0e-9), bit-exact first Adam updates, 1,724,800
prediction pixels (max 1.1e-6 overall; 1.9e-7 at selected weights), 14,336 state
values and 6,272 fixed-target input derivatives (max 1.7e-10). The RMS first-row
input sensitivity is 3.04e-6 of the last row's; a log10 figure distinguishes a
valid full gradient path from useful retained information. The unused shared
`common/recurrent_autoencoder.hpp` was removed after its sole C++ consumer was
retired; native Python/R now call the core directly. Evidence:
`build/nn-report-conversion/recurrent/`.

The activation-sparse autoencoder is verified and its standalone original retired.
`nn/activation_sparse_autoencoder.hpp` owns activation L1, dtype-consistent
clamped mean-unit Bernoulli KL and the complete untied sigmoid model/objective.
Four variants retain identical seed1601 initialization, seed1607 batch order,
three epochs and 96 explicit Adam updates each. Checkpoints use each variant's
own full Float32 validation objective; those differently weighted totals are
not cross-variant model-selection scores. All source arrays in 41 report archives
agree by shape, dtype and value bytes with the original and across normal C++,
Python and R runs. The unused `common/sparse_autoencoder.hpp` was removed after
its only C++ consumer retired; native languages now delegate to the core.

All eight blocks and seven inspected figures export. All 34 model tests pass,
including five new penalty/model tests, inclusive clipping endpoints, signed/zero
L1 conventions, dtype rounding, batch coupling and small-model first/second finite
differences. All four public examples compile/run, the header compiles standalone,
27/27 definitions pass fresh Doxygen and three clangd hovers pass. There are
43 focused Python tests and 217 R expectations across six files; all 63
inventory/runner tests pass after retirement. Nested penalty
and tensor handles survive parent collection, retaining encoder gradients.
The optional `scripts/check_activation_sparse_autoencoder.py` verifies 404,800
first parameter derivatives (max 1.2e-10), four bit-exact first Adam updates,
10,240 first/final-minibatch activity derivatives (max 2.5e-11), and 5,644,800
pool/control prediction pixels (max 1.4e-7), including component and mean checks.

Test MSE is 0.067947765/0.064767495/0.060774161/0.064606958 for plain/L1/KL/combined,
versus fitting-mean 0.067798074. L1 maximizes the observed fraction below0.1;
combined minimizes mean activity, but even its0.2462 remains above target0.05.
Shifting KL codes raises MSE to0.067903786, near the constant mean. The report
separates these properties, shows a mean-aggregation counterexample and documents
clamp derivatives, batch dependence and the absence of exact-zero/downstream
claims. Evidence: `build/nn-report-conversion/sparse/`.

The stacked-autoencoder report is verified and its standalone original retired.
`nn/layered_autoencoders.hpp` owns configurable mirrored sigmoid stacks and a
standalone decoder with original layer indices. The report keeps both greedy
784–64–784/64–16–64 fits, detached frozen caches, all eight encoder/decoder
transfers into 784–64–16–64–784, joint refinement and the independent scratch
control. The four explicit Adam loops perform 96 updates each. Epoch zero is
eligible for joint/scratch selection; selected epochs remain 3/3/3/2. All source
arrays in 42 report archives match the original and native Python/R byte for byte;
all four batch orders match exactly and scalar histories agree within 1e-12.
Both language examples delegate only model arithmetic to the core. The unused
`common/stacked_autoencoder.hpp` was removed after its sole C++ consumer retired.

All seven programs and five inspected figures export successfully. All 39 native
model tests pass, including all 24 source toy activities/48 analytic derivatives,
Float64 first/second derivatives, standalone decoding, frozen caches and an
optimizer created before transfer. Both public examples compile/run, the header
compiles standalone, 6/6 definitions pass fresh Doxygen and two clangd hovers pass.
45 focused Python tests, 239 R expectations across seven files and all 63
inventory/runner tests pass. The optional `scripts/check_stacked_autoencoder.py`
checks 309,984 first derivatives (max 7.6e-10), four exact first Adam updates,
16 validation checkpoints and 2,873,600 prediction pixels (max 1.5e-7), plus all
eight transfers, cached codes and the reconstruction-error bound. The measured
pretrained RMSE 0.261689 satisfies the loose 0.356668 Lipschitz/Frobenius bound.
Test MSE is 0.068480899 pretrained, 0.066063591 fine-tuned, 0.068178502 scratch
and 0.067798074 fitting mean. The staged path uses three times the epoch passes
across differently sized models; it is not a compute-matched superiority result.
The report preserves weak/blurry outputs and explains that depth, a valid bound
and intact gradient paths do not imply useful representations. Evidence:
`build/nn-report-conversion/stacked/`.

The tied-autoencoder report is verified and its standalone original retired.
`nn/tied_autoencoder.hpp` owns the live differentiable encoder-matrix transpose,
ReLU/sigmoid forward result and standalone target-free decoder. The independent
control starts with exactly the same full initial function, then learns a separate
decoder matrix. Three versus four leaves contain 25,904 versus 50,992 scalars;
both receive 96 explicit Adam updates with identical order, learning rate and
validation criterion. All 25,088 shared decoder coordinates are checked after
every update. Every source array in 24 report archives matches the saved original
and normal Python/R runs by shape, dtype and value bytes. Both batch orders match
exactly and scalar fitting/validation histories agree within 1e-12. Epoch zero is
measured as a control but remains ineligible; both select completed epoch3.
The unused `common/tied_autoencoder.hpp` was removed after its sole consumer retired.

All seven blocks and five inspected figures export. All 44 native model tests
pass, including all original toy activities/eleven analytic derivatives and the
encoder-plus-decoder shared-gradient identity. Tied and untied parameter/input
finite differences, first/second derivatives, noncontiguous rows, live updates,
decoder-only names/counts and ReLU's zero convention pass. All three public
examples compile/run, the header compiles standalone,10/10 definitions pass
fresh Doxygen and two clangd hovers pass.47 focused Python tests and264 R
expectations across eight files pass, as do63 inventory/runner tests. Language
checks retain result fields and live aliasing after collection/optimizer updates.
The optional `scripts/check_tied_autoencoder.py` checks76,896 full first derivatives
(max7.4e-11), two bit-exact first Adam updates, eight validation checkpoints,
89,600 code values and4,076,800 prediction pixels (max4.9e-7). Exact matching,
independent-weight divergence and the fitting-only mean are checked separately.

Test MSE is0.049649292 tied,0.049003992 independent and0.067798074 fitting mean;
tied shifted/zero-code MSE is0.075479784/0.210690986. The report derives both
shared gradient paths, gives a null-vector counterexample to inverse decoding,
and keeps raw blurred/distorted outputs. Different capacity and optimization
under one seed/budget do not establish a general ranking or bitrate claim.
The next winner-take-all original has been executed as a saved baseline. Evidence:
`build/nn-report-conversion/tied/`.

The winner-take-all report is verified and its standalone original retired,
completing all 11 deterministic-autoencoder conversions. Existing canonical
`CompetitiveMask` and `CompetitiveAutoencoder` own the lifetime mode: ReLU
activities, ceil(B/8) selected observations per feature during training and full
ReLU deployment without competition. Both matched 101,200-scalar models retain
three epochs,96 explicit Adam updates, seed1931 initialization and seed1933
batch order. Fitting uses training_reconstruction; checkpoint selection uses
ungated full validation reconstruction, with epoch zero ineligible. Both select
completed epoch3. Every source array in29 report archives matches the original
and normal Python/R by dtype, shape and value bytes; both batch-order TSVs match
exactly and scalar histories agree within1e-12. The now-unused shared
`common/competitive_autoencoder.hpp` is removed. Its legacy example bindings
are removed too; core API symbols are unchanged. Python/R keep their own state,
loops and small invariant checks, using the canonical model and selector directly.

All eight report programs and six inspected figures export.46 native model tests
pass, including two new lifetime cases: every small-model parameter derivative,
input first/second finite differences on separated positive support, context-
dependent training with context-independent deployment, and selected-zero slots.
49 focused Python tests and295 R expectations across nine files pass, including
owning endpoints and lifetime batch/zero/partial-count cases. All63 inventory/
runner checks pass. The three existing public examples compile/run,10/10 header
definitions remain documented and both clangd hovers pass. The optional
`scripts/check_winner_take_all_autoencoder.py` checks202,400 first derivatives
(max7.6e-11), both bit-exact first Adam updates, eight validation checkpoints,
179,200 deployed code values and4,503,296 prediction pixels (max8.2e-7), including
first/partial/selected training probes and original disjoint MNIST source IDs.

Test MSE0.067503750 competitive is only0.43% below the fitting mean0.067798074,
versus0.041842456 plain. Competitive shifted/zero-code MSE is0.083673118/
0.191458436. The selected competitive deployment code is96.85% nonzero, compared
with84.03% plain, despite12.5% selected training slots in32-row batches. The report
derives the winner/ReLU gradient, demonstrates batch dependence and a finite
positive-tie jump under a1e-9 perturbation, and distinguishes selected slots from
nonzero values. Raw weak reconstructions and the stronger plain control remain
visible; no deployed-sparsity, classifier, sparse-kernel or bitrate claim follows.
Evidence: `build/nn-report-conversion/winner_take_all/`.

The adversarial-autoencoder report is verified and its standalone original retired.
New `nn/adversarial_autoencoder.hpp` owns deterministic tanh/linear E/G/D components,
stable per-image summed Bernoulli NLL and all three explicit objectives. It retains
all mathematical paths while callers choose update groups. The default12 leaves/
26,133 scalars, three phases, shared E+G reconstruction gradient snapshot and four
independent Adam states are preserved. Twenty-one minibatches give21 steps per
optimizer and84 updates. Validation reconstruction NLL selects completed epoch3;
epoch zero is only a control. Every source array in26 report archives, both
initial/actual gradients, first phase snapshots, fixed/training noise, model-only
checkpoints and all outputs matches the saved original by dtype, shape and value
bytes. Batch TSV and all source validation-history values match exactly.
The unused `common/adversarial_autoencoder.hpp` is removed.

Native `KnownFactorLattice` and the beta-vae-assets preparation adapter generate
all294 original binary circle/square images, fixed compositional holdouts, ascending
source IDs and provenance; no external fixture generation is needed. The unchanged
original checker independently accepts all230,496 pixels, all factors, unique IDs,
split residues and marginal balances before the baseline run. The adapter retains
raw Int32 NPZ pixels; the tensor API returns detached Float32 images. The new
`just nn-beta-vae-assets` entry uses the normal native preparation command. Dataset
tests cover exhaustive geometry, deterministic storage, reuse and missing-output
repair. The installed-library dataset-test target now links the namespaced core
target so public include paths propagate correctly.

All eight programs and six inspected figures export.50 native autoencoder tests
and12 offline dataset tests pass. Four new AAE cases verify summed NLL and target
partials, every small-model parameter derivative, input/prior first/second checks,
separate prior batch sizes, the reconstruction gradient barrier, distinct stores
and phase isolation.53 focused Python tests and337 R expectations across ten
files pass, with owning data/results and host-owned four-state/three-phase checks.
All63 inventory/runner tests pass. Eight public examples compile/run, all88 affected
definitions pass fresh Doxygen coverage and six new clangd hovers pass; the two new
headers account for21 definitions. The optional `scripts/check_adversarial_autoencoder.py`
checks77,522 initial/actual phase derivatives (max9.1e-8), four bit-exact first Adam
states, unrelated-group immutability,3,776 Gaussian values, four validation points,
and633,472 lattice/prediction pixels, including all20 prior-decoder outputs.

Test NLL380.600250 and MSE0.148934076 remain poor: the fitting-mean MSE is0.048548290.
The report retains raw reconstructions, error maps and unfiltered prior probabilities,
derives real/fake/non-saturating loss signs and distinguishes aggregate matching
from per-example KL. Finite deterministic empirical codes cannot equal a continuous
Gaussian measure; these finite adversarial losses do not certify useful generation,
prior matching, disentanglement or an ELBO. The architecture trace explicitly splits
an image/prior shape probe to retain both independent data branches. Evidence:
`build/nn-report-conversion/adversarial/`.

The beta-VAE report is verified and its standalone original retired. New
`nn/gaussian_autoencoder.hpp` owns explicit pathwise Gaussian sampling, analytic
per-image KL, the ReLU Bernoulli decoder and configurable beta-weighted forward
model. `nn/latent_factor_diagnostics.hpp` owns generic fitting-only standardized
ridge/single-axis probes and all13 output arrays. Ten default leaves/51,416 scalars,
matched seed2237 initialization, all row orders and training/evaluation/prior noise
remain unchanged. Both beta1/4 use21 explicit Adam(.003) updates and the same
four-draw unweighted validation NLL+KL criterion, selecting epoch3. The shared
beta helper remains for later VAE consumers, with model/decoder/probe math now
forwarded into the canonical core. The obsolete generated Python/R reference launchers are removed; no unfinished
Python/R experiment is marked native merely because the reusable API is bound.

All30 original archives, both batch TSVs and both complete training histories
match the report by value bytes, shape and dtype, including all first gradients,
first updates, epochs, selected weights, four-draw posterior outputs, priors,
traversals and factor diagnostics. All ten report programs/eight reviewed figures
and both adjacent exports pass.55 native tests cover all35 small-model parameter
derivatives, input first/second checks, paired-beta/noise controls, strides,
validation-before-creation, generic latent/target counts and fitting-only axis
selection.57 focused Python tests and367 R expectations across eleven files pass.
Six public examples compile/run;19/19 definitions in both new headers pass fresh
Doxygen and all six new clangd hovers pass. Binding help documents reductions,
noise ownership, inference controls, error timing and the diagnostic's limits.

The independent NumPy oracle `scripts/check_beta_variational_autoencoder.py`
checks all102,832 first-batch derivatives (max1.8e-7), two exact Float32 first Adam
updates,7,304 Gaussian values,2,162,272 lattice/prediction pixels, six eligible
validation candidates and all13 probe arrays per model. Selected test beta1/4
NLL is239.924421/260.338864, KL29.774689/11.192016 and decode(mu) pixel MSE
0.075139197/0.097220306, both worse than fitting-mean0.048548290. Figures retain
unfiltered reconstructions/errors, prior probabilities and coordinate traversals;
linear recovery does not certify disentanglement. Evidence:
`build/nn-report-conversion/beta_vae/`.

The conditional-VAE report is verified and its standalone original, unused
`common/conditional_variational_autoencoder.hpp` and generated Python/R reference
launchers are retired. New `nn/conditional_autoencoder.hpp` owns configurable
q(z|image,class), class-conditioned target-free decoding and an owning result
with all Gaussian endpoints plus exact one-hot/encoder-input/decoder-input
fields. The shared N(0,I) prior is fixed, supplied classes are required at inference,
and invalid Int32/64 conditions are rejected before leaf creation. This is not a
classifier or the structured-output paper's learned conditional-prior architecture.
Soft targets give cross-entropy rather than a continuous-observation density.

The original102,932 scalars/ten leaves, seed2111 initialization,96 explicit
Adam(.003) updates, balanced1000/200/200 real MNIST pools and static byte>=128
threshold are preserved. Four fixed posterior draws select completed epoch3 by
validation NLL+KL. Training-epoch2 and validation both use seed2131 in the source;
the report explicitly retains/documents that noise-stream reuse rather than
claiming independent banks. All18 archived outputs, exact batch TSV and complete
training/validation history match the original by shape, dtype and value bytes.
No unconverted host-language experiment is marked native merely because its
reusable forward is bound.

All nine report blocks/seven reviewed figures and adjacent HTML/Markdown bundles
pass.59 native tests include all47 configurable small-model parameter derivatives,
input first/second checks, a fourteen-scalar hand condition example, target-free
decoding, noise controls and validation-before-creation.61 focused Python tests
and393 R expectations across twelve files pass, including owning inherited fields,
Float64 one-hot computation, Int64 conditions and explicit caller-owned Adam.
All63 inventory/runner/website tests pass. Three public examples compile/run,
the header compiles alone,10/10 definitions
pass fresh Doxygen and all three new clangd hovers pass. The optional independent
`scripts/check_conditional_variational_autoencoder.py` checks all102,932 first
parameter derivatives (max1.36e-7), one exact Float32 first Adam update,17,210
Gaussian values,7,158,704 data/prediction pixels, three eligible checkpoints,
condition concatenations and exact fitting-only class means.

Selected test NLL187.190332, KL6.617313 and negative ELBO193.807644 nats/image
remain distinct from exact conditional likelihood. Decode(mu) MSE0.076372894
beats the global fitting mean0.086689513, but loses to the condition-aware fitting
class mean0.071591939; shifting the decoder class at the same mu gives0.081708886.
Changing only class changes posterior means (max0.421315) and fixed-prior decoder
probabilities (max0.239766). Unfiltered digit reconstructions/errors, all fifty
class/prior outputs and the full81-point class3 grid retain weak identity/detail.
Evidence: `build/nn-report-conversion/conditional_vae/`. The original three-epoch
convolutional-VAE baseline is captured separately before its model math changes.

The convolutional-VAE report is verified and its original, unused
`common/convolutional_variational_autoencoder.hpp` and generated Python/R
reference launchers are retired. New `nn/convolutional_variational_autoencoder.hpp`
owns configurable pooled Gaussian inference and sample-only projection/repeat/
convolution decoding. Fourteen default leaves/2117 scalars, all seven spatial
stages and the mixed initialization remain exact: ReLU convolutions use He,
dense Gaussian heads and even the ReLU dense projection use ordinary Glorot,
as does the output convolution; biases zero. Only z reaches the decoder, with
no image/label input or deterministic pool2 bypass. The six-leaf decoder exposes
five owning fields; the full model owns sixteen Gaussian/spatial endpoints.

All19 archives, exact batch TSV and complete training/validation history match
the saved original by dtype, shape and every value byte. The original statically
thresholded1000/200/200 MNIST pools, seed2011 weights,96 explicit Adam(.003) updates,
all training/evaluation/prior noise, initial and selected-state gradients, every
epoch and restored checkpoint remain unchanged. Four fixed validation draws
select completed epoch3 by unweighted NLL+KL; the post-selection two-image gradient
review updates nothing. All ten report blocks/eight inspected figures and both
adjacent export bundles pass, including a hand pooling/repetition adjoint,
executed sample-only architecture, all-channel stage display, weak reconstructions,
zero-code controls, twenty unfiltered priors and the full81-point latent grid.

All63 native model tests pass, including every84 small-model leaf derivative,
image/epsilon first/second checks, strided/non-square inputs, all default shapes,
standalone decoding and validation-before-creation.65 focused Python tests and421
R expectations across thirteen files pass with owning inherited/decoder fields
and caller-owned Adam. All63 inventory/runner/website tests pass. Four public examples compile/run;
the header compiles alone,
all21 definitions pass fresh Doxygen and four new clangd hovers pass. The optional
`scripts/check_convolutional_variational_autoencoder.py` independently checks all
4234 initial/selected derivatives (max1.34e-5/7.15e-6 in Float32 comparisons), one
exact first Adam update,17240 Gaussian values,6821584 data/prediction pixels,
all seven stages (largest raw-logit error3.82e-6) and three eligible checkpoints.
Posterior probability error stays below2.71e-7.

Selected test NLL205.425587, KL3.692729 and negative ELBO209.118317 nats/image
remain separate from exact log likelihood. Decode(mu) MSE0.085688548 improves
only slightly over fitting-only mean0.086689513. Blurred/lost identities stay
visible; two Gaussian coordinates and three epochs do not certify semantic
factors, useful generation or bitrate. Evidence:
`build/nn-report-conversion/convolutional_vae/`. The untouched matched FactorVAE/
betaTCVAE baseline is captured separately, with45 output archives, before its
model mathematics changes.

The FactorVAE/full-reference betaTCVAE report is verified; its original,
unused `common/total_correlation_autoencoder.hpp` and generated Python/R reference
launchers are retired. New `nn/total_correlation_autoencoder.hpp` owns exact finite
Gaussian-mixture log ratios, the tanh density-ratio discriminator and balanced
loss, detached column permutation, and both Gaussian VAE forwards. The betaTC
variant uses all210 fitting posteriors with live shared encoder/head derivatives;
it is not the paper's minibatch estimator. Signed sample means are retained.
FactorVAE preserves the z gradient through the fixed discriminator in the VAE
phase, then uses detached fresh samples from an independent fitting batch after
the VAE update; the explicit second optimizer updates only D. Opposite stores
are checked unchanged after every phase. The API hides no fitting or randomness.

All45 original archives match by dtype, shape and every byte (3,633,019 values).
VAE/discriminator batch TSV, complete histories and every non-timing metric other
than MSE match exactly; library MSE reduction changes only rounding below1.6e-15.
Both shared initial arrays,21 VAE updates/model and21 discriminator updates,
all noise banks, all epochs and restored selected arrays remain exact. Four-draw
common unweighted validation NLL+analytic KL selects epoch3 for both; D is restored
at that same epoch. All ten report blocks and six inspected figures pass, including
the signed two-component hand calculation, native concept diagram, common learning
curves, unfiltered heldout gallery,20 identical prior codes and all36 traversals/model.
Both adjacent Markdown/HTML figure bundles pass. Mean reconstruction MSE0.08413/
0.07537 remains worse than the fitting-only mean0.04855; descriptive heldout probes
are explicitly distinguished from disentanglement benchmarks.

All69 native model tests pass, including first/second derivatives of all three
mixture inputs, all44 tiny FactorVAE and35 betaTC leaf derivatives, live reference
paths, strided/detached exact source shuffle, balanced losses,
explicit phase separation and validation before creating leaves.69 focused Python
tests and447 R expectations across fourteen files pass. Owning nested Gaussian
results survive parent and endpoint collection; caller-owned host Adam steps,
reference-input gradients and zero-weight ablation pass. The SWIG interface
excludes private validation helpers to avoid flattened-name collisions. These
native components do not mark incomplete Python/R experiment loops as native.
All63 inventory/runner/website tests pass. Eight public examples compile/run,
the public header compiles alone,19/19 definitions pass Doxygen and eight clangd
hovers pass. The optional `scripts/check_factor_tc_variational_autoencoder.py`
independently checks102929 first-step VAE/D derivatives (max1.80e-7/1.85e-8), exact
Float32 first Adam updates,12344 Gaussian values,2162272 data/prediction pixels,
all six eligible criteria and both13-array fitting-only probes. Probability error
is below2.02e-7 and signed TC error below4.77e-7. Evidence:
`build/nn-report-conversion/factor_tc_vae/`. The untouched affine-flow VAE
baseline is saved with19 archives before changing its math.

The conditional affine-flow VAE report is verified and its standalone C++ original
and generated Python/R reference launchers are retired. New
`nn/flow_variational_autoencoder.hpp` owns configurable even-width affine
couplings, analytic inverses/signed log determinants, and a Gaussian VAE with two
alternating live image-context couplings. The eighteen default leaves/52672
scalars and creation order remain exact. Only final z reaches the decoder;
standard-normal prior final codes bypass posterior inference. Training uses
NLL+logq0-forward_logdet-logp(final_z), preserving signed sample ratios rather
than substituting base analytic KL. Both inverse calls share the original leaves.
The old shared header is reduced to its rounded normalizer, still used by the
unconverted importance-weighted example; its flow/model math is removed.

All19 original archives match by dtype, shape and every byte (1,984,695 values),
including all18 endpoints for zero noise and four draws in every pool, all52,672
first derivatives, the first update, all epochs, selected arrays,48 full Jacobian
entries and Gaussian banks. Batch/source TSV, complete histories and non-timing
metrics other than MSE match exactly; MSE reduction rounding is below2.8e-15.
All eight report blocks and five inspected figures pass, with verified adjacent
Markdown/HTML image bundles. The report keeps all21 explicit Adam(.003) updates,
fixed four-draw validation and fresh restore. Epoch3 is selected. Heldout
NLL226.569573+sampled log ratio36.835837=negative ELBO263.405410 nats/image;
decode(f(mu)) MSE0.068817722 remains worse than fitting-only mean0.048548290.
Unfiltered reconstructions/all20 prior codes, density signs, inverse bounds and
inference immutability remain visible. No matched-baseline or semantic-quality
claim is made; nonlinear f(mu) is not generally the flowed posterior mean.

All75 native model tests pass, including first/second code/context derivatives
for both swap/inverse settings, all63 small-model leaves and image/noise paths,
full4x4 Jacobian determinants, invalid-input timing, strides, complete default
census, owning named fields and target-free decoding.73 focused Python tests and
477 R expectations across fifteen files pass, with live context gradients,
collection-safe endpoint/named-array handles and explicit host Adam. The SWIG
named-vector declaration precedes the new output method; no complete independent
Python/R experiment loop is marked native. All63 inventory/runner/website tests
pass. Five public examples compile/run; the header compiles alone,28/28 definitions
pass Doxygen and five clangd hovers pass. The optional
`scripts/check_flow_variational_autoencoder.py` independently checks52672 first
derivatives (max2.08e-7), one bit-exact first Adam update,7304 Gaussian values,
1398656 data/prediction pixels, all18 endpoints and three eligible checkpoints.
All48 Jacobian entries agree within6.28e-8; prediction error stays below1.25e-7,
density-field error below6.76e-5 and inverse roundtrip error1.20e-6. Evidence:
`build/nn-report-conversion/flow_vae/`. The untouched Gumbel categorical VAE
baseline is saved before changing its math.

The Gumbel categorical-VAE report is verified and its standalone original,
unused `common/gumbel_categorical_autoencoder.hpp` and generated Python/R
reference launchers are retired. New `nn/gumbel_categorical_autoencoder.hpp`
owns local midpoint-uniform/Gumbel banks, soft categorical sampling, discrete
uniform-prior KL, target-free simplex/ID decoders and configurable categorical
VAE forwards. Eight default leaves/53104 scalars, all initial arrays, source
Float32 scalar rounding, explicit noise and21 Adam(.003) updates remain exact.
Temperature affects the relaxation, not q or hard IDs; no straight-through path
is introduced. Soft reconstruction plus discrete KL is explicitly a biased
continuous surrogate, not an exact discrete ELBO or Concrete KL. Matched hard
samples provide a separate Monte Carlo discrete negative-ELBO diagnostic.

All18 original archives match by dtype, shape and every byte (3,471,217 values),
including all13 source fields at each saved draw, calibration, first gradients/
update, all epochs, selected arrays, noise banks and all80 prior uniforms/IDs.
Batch/source TSV, complete histories and non-timing metrics other than MSE match
exactly; library MSE reduction changes rounding below3.25e-15. All eight report
blocks/five inspected figures and both adjacent export bundles pass. The native
diagram separates shared perturbed scores from soft fitting/hard diagnostics.
All prescribed heldout cases and20 uniform-categorical priors remain unfiltered.
Four fixed soft validation draws plus discrete KL select completed epoch3;
hard diagnostics and test data do not select. Heldout soft NLL321.932229,
hard NLL313.407284 and categorical KL3.115642 give relaxed325.047870 versus
hard MC negative ELBO316.522925 nats/image. Decode(q)/MAP MSE0.114726214/0.103664132
remain worse than fitting mean0.048548290. Neither deterministic decode is a
posterior predictive mean; no useful categorical semantics/codec claim is made.

All81 native model tests pass: all39 small-model parameter derivatives and
image/noise first/second derivatives, all12 hand score/Gumbel partials,
row broadcasting/strides, hard temperature invariance, fixed-engine midpoint
banks/full-uint32 seeds, both target-free decoders, Float64 one-hots, default
census, zero model, host Adam and validation/overflow before allocation/leaves.
77 focused Python tests and516 R expectations across sixteen files pass with
collection-safe bank/all14 forward fields, live continuous paths, Int32/64 IDs,
full-range seeds and explicit host optimization. These native components do not
mark incomplete Python/R experiment loops as native. All63 inventory/runner/
website tests pass. Nine public examples compile/run; the header compiles alone,
35/35 definitions pass Doxygen and nine clangd hovers pass. The optional
`scripts/check_gumbel_categorical_autoencoder.py` independently checks53104 first
derivatives (max1.05e-7), one bit-exact first Adam update,70160 midpoint uniforms,
Gumbels (max8.89e-16),2551136 data/prediction pixels, all13 source endpoints and
three eligible criteria. Soft/hard probability error stays below9.67e-8;
minimum hard score margin is0.000547275. Calibration4096 counts846/1243/2007
satisfy the specified6sigma check. Evidence:
`build/nn-report-conversion/gumbel_vae/`. The untouched paired spatial hierarchy/
miniature-NVAE baseline is saved with41 archives before its math changes.

Binding revalidation also found a pre-existing import failure: 42 compatibility
namespaces contain using-declarations SWIG has not exported (the same 42 groups
are absent in the HEAD-generated proxy). Python component discovery now loads
requested groups lazily and still raises on an unavailable group, allowing
independent available groups such as `ae` to run. The complete historical CNN
namespace/architecture binding audit remains open; no claim that every old CNN
component is bound is made. `legacy-binding-groups.json` records before/after
evidence, with zero newly missing groups. Rebuilding the normal Python package
required the documented core SWIG regeneration too: its old proxy referred to
the removed internal `CanonicalEdgeHash` type. Current generated core and NN
sources compile, and both language packages load from the local build.


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
  stages explicitly. Use native `dm::graphics::Diagram` concept diagrams when
  they teach the method; extend the C++ graphics library as needed. Earlier batch
  notes about TikZ record historical verification, superseded by the native migration.
  Keep architecture graphs left to right with counterclockwise node labels; choose
  Flow, Graph or LeNet style and block/layer/operation detail explicitly. Demonstrate
  relevant plot settings and disclose shape scales, visibility floors and sampled
  sheet/neuron counts. Use shared linear dimensions for proportional views; label
  square-root/logarithmic compression explicitly when readability calls for it.
- For pixel-image contact sheets, prefer tensor `save_image_grid` to a local PNG
  with explicit class order and integer pixel scaling. This avoids large SVG
  rectangle payloads for raster data. Keep scalar plots, heatmaps and concept
  diagrams as local SVGs; both formats are build output
  (not committed), rendered again when the reports run, and exported.
- The user authorized source retirement on 2026-11-08. Remove each original only
  after its replacement executes every runnable block and its local figures and
  exports are checked. Refresh the C++ audit, language ledger and optional example
  registry; keep native Python/R scripts independently discoverable. Preserve shared
  headers until their consumers use the canonical library API. Retain original paths,
  source digests and execution evidence in `nn-conversion-status.json`.
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


## Continuation checklist: autoencoders through reinforcement learning

The existing foundation/CNN checklists above record their original conversions.
This continuation checklist records the remaining report work; retirement is
tracked separately for every source in `nn-conversion-status.json`. Foundation
retirement is complete. Continue with the first unchecked report below.

- [x] 2_autoencoders/2.1_deterministic_autoencoders/contractive_autoencoder
- [x] 2_autoencoders/2.1_deterministic_autoencoders/convolutional_autoencoder
- [x] 2_autoencoders/2.1_deterministic_autoencoders/denoising_autoencoder
- [x] 2_autoencoders/2.1_deterministic_autoencoders/k_sparse_autoencoder
- [x] 2_autoencoders/2.1_deterministic_autoencoders/masked_autoencoder
- [x] 2_autoencoders/2.1_deterministic_autoencoders/recurrent_autoencoder
- [x] 2_autoencoders/2.1_deterministic_autoencoders/sparse_autoencoder
- [x] 2_autoencoders/2.1_deterministic_autoencoders/stacked_autoencoder
- [x] 2_autoencoders/2.1_deterministic_autoencoders/tied_autoencoder
- [x] 2_autoencoders/2.1_deterministic_autoencoders/undercomplete_autoencoder
- [x] 2_autoencoders/2.1_deterministic_autoencoders/winner_take_all_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/adversarial_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/beta_variational_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/conditional_variational_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/convolutional_variational_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/factor_tc_variational_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/flow_variational_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/gumbel_categorical_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/hierarchical_variational_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/importance_weighted_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/info_variational_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/ladder_variational_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/posterior_collapse
- [x] 2_autoencoders/2.2_variational_autoencoders/sequence_variational_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/vae_gan
- [x] 2_autoencoders/2.2_variational_autoencoders/variational_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/vector_quantized_autoencoder
- [x] 2_autoencoders/2.2_variational_autoencoders/vector_quantized_autoencoder_2
- [x] 2_autoencoders/2.2_variational_autoencoders/wasserstein_autoencoder
- [x] 2_autoencoders/2.3_applications/autoencoder_restoration
- [ ] 2_autoencoders/2.3_applications/autorec
- [x] 2_autoencoders/2.3_applications/encoder_pretraining
- [x] 2_autoencoders/2.3_applications/latent_arithmetic
- [x] 2_autoencoders/2.3_applications/latent_visualization
- [x] 2_autoencoders/2.3_applications/learned_image_codec
- [x] 2_autoencoders/2.3_applications/reconstruction_anomaly
- [x] 3_generative_models/3.1_gans/auxiliary_classifier_gan
- [x] 3_generative_models/3.1_gans/bidirectional_gan
- [x] 3_generative_models/3.1_gans/big_gan
- [x] 3_generative_models/3.1_gans/conditional_gan
- [x] 3_generative_models/3.1_gans/cycle_gan
- [x] 3_generative_models/3.1_gans/dcgan
- [ ] 3_generative_models/3.1_gans/gan_diagnostics
- [x] 3_generative_models/3.1_gans/hinge_gan
- [x] 3_generative_models/3.1_gans/infogan
- [x] 3_generative_models/3.1_gans/lsgan
- [x] 3_generative_models/3.1_gans/pix2pix
- [x] 3_generative_models/3.1_gans/progressive_gan
- [x] 3_generative_models/3.1_gans/self_attention_gan
- [x] 3_generative_models/3.1_gans/singan
- [x] 3_generative_models/3.1_gans/spectral_gan
- [x] 3_generative_models/3.1_gans/srgan_esrgan
- [x] 3_generative_models/3.1_gans/star_gan
- [x] 3_generative_models/3.1_gans/style_gan
- [x] 3_generative_models/3.1_gans/vanilla_gan
- [x] 3_generative_models/3.1_gans/wasserstein_gan
- [ ] 3_generative_models/3.2_autoregressive_models/character_rnn_lm
- [x] 3_generative_models/3.2_autoregressive_models/gated_pixel_cnn
- [ ] 3_generative_models/3.2_autoregressive_models/image_gpt
- [x] 3_generative_models/3.2_autoregressive_models/made_density
- [x] 3_generative_models/3.2_autoregressive_models/pixel_cnn
- [x] 3_generative_models/3.2_autoregressive_models/pixel_cnn_pp
- [x] 3_generative_models/3.2_autoregressive_models/pixel_rnn
- [ ] 3_generative_models/3.2_autoregressive_models/transformer_lm
- [x] 3_generative_models/3.2_autoregressive_models/wavenet
- [x] 3_generative_models/3.3_normalizing_flows/cnf_density
- [x] 3_generative_models/3.3_normalizing_flows/flow_density_2d
- [x] 3_generative_models/3.3_normalizing_flows/glow_density
- [x] 3_generative_models/3.3_normalizing_flows/iaf_density
- [x] 3_generative_models/3.3_normalizing_flows/maf_density
- [x] 3_generative_models/3.3_normalizing_flows/nice_density
- [x] 3_generative_models/3.3_normalizing_flows/planar_radial_density
- [x] 3_generative_models/3.3_normalizing_flows/real_nvp_density
- [x] 3_generative_models/3.3_normalizing_flows/spline_density
- [ ] 3_generative_models/3.4_diffusion/conditional_diffusion
- [ ] 3_generative_models/3.4_diffusion/consistency_model
- [ ] 3_generative_models/3.4_diffusion/ddim
- [ ] 3_generative_models/3.4_diffusion/ddpm
- [ ] 3_generative_models/3.4_diffusion/diffusion_guidance
- [ ] 3_generative_models/3.4_diffusion/flow_matching
- [ ] 3_generative_models/3.4_diffusion/latent_diffusion
- [ ] 3_generative_models/3.4_diffusion/noise_conditional_score
- [ ] 3_generative_models/3.4_diffusion/score_sde
- [ ] 3_generative_models/3.4_diffusion/signal_diffusion
- [ ] 3_generative_models/3.5_energy_based_models/deep_belief
- [ ] 3_generative_models/3.5_energy_based_models/deep_boltzmann
- [ ] 3_generative_models/3.5_energy_based_models/energy_langevin
- [ ] 3_generative_models/3.5_energy_based_models/modern_hopfield
- [ ] 3_generative_models/3.5_energy_based_models/restricted_boltzmann
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/bidirectional_rnn
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/clockwork_rnn
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/conv_lstm
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/gradient_clipping
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/gru
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/indrnn
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/layer_norm_lstm
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/lstm
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/mgu
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/peephole_lstm
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/phased_lstm
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/qrnn
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/recurrent_dropout
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/sru
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/stacked_rnn
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/truncated_bptt
- [ ] 4_recurrent_and_sequence_models/4.1_recurrent_cells/vanilla_rnn
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/bahdanau
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/beam_search
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/bilstm_crf
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/bilstm_tagging
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/ctc_handwriting
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/handwriting
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/imdb_lstm
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/lstm_forecast
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/luong
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/midi_lstm
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/pointer_generator
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/pointer_network
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/seq2seq
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/translation
- [ ] 4_recurrent_and_sequence_models/4.2_sequence_tasks/word_rnn_lm
- [ ] 4_recurrent_and_sequence_models/4.3_memory_augmented_networks/dnc
- [ ] 4_recurrent_and_sequence_models/4.3_memory_augmented_networks/key_value
- [ ] 4_recurrent_and_sequence_models/4.3_memory_augmented_networks/memnet
- [ ] 4_recurrent_and_sequence_models/4.3_memory_augmented_networks/ntm
- [ ] 5_attention_and_transformers/5.1_attention_fundamentals/attention_masks
- [ ] 5_attention_and_transformers/5.1_attention_fundamentals/attention_visualization
- [ ] 5_attention_and_transformers/5.1_attention_fundamentals/multi_head_attention
- [ ] 5_attention_and_transformers/5.1_attention_fundamentals/scaled_dot_attention
- [ ] 5_attention_and_transformers/5.1_attention_fundamentals/transformer_positions
- [ ] 5_attention_and_transformers/5.2_language_transformers/albert_distilbert
- [ ] 5_attention_and_transformers/5.2_language_transformers/bert_finetune
- [ ] 5_attention_and_transformers/5.2_language_transformers/bert_mlm
- [ ] 5_attention_and_transformers/5.2_language_transformers/gpt_lm
- [ ] 5_attention_and_transformers/5.2_language_transformers/instruction_tuning
- [ ] 5_attention_and_transformers/5.2_language_transformers/kv_cache
- [ ] 5_attention_and_transformers/5.2_language_transformers/lora_adapters
- [ ] 5_attention_and_transformers/5.2_language_transformers/pretrained_bert_serve
- [ ] 5_attention_and_transformers/5.2_language_transformers/roberta_tweaks
- [ ] 5_attention_and_transformers/5.2_language_transformers/sentence_embeddings
- [ ] 5_attention_and_transformers/5.2_language_transformers/switch_transformer
- [ ] 5_attention_and_transformers/5.2_language_transformers/t5_text
- [ ] 5_attention_and_transformers/5.2_language_transformers/token_sampling
- [ ] 5_attention_and_transformers/5.2_language_transformers/transformer_nmt
- [ ] 5_attention_and_transformers/5.2_language_transformers/transformer_xl
- [ ] 5_attention_and_transformers/5.2_language_transformers/xlnet
- [ ] 5_attention_and_transformers/5.3_vision_transformers/beit_pretraining
- [ ] 5_attention_and_transformers/5.3_vision_transformers/convmixer
- [ ] 5_attention_and_transformers/5.3_vision_transformers/deit
- [ ] 5_attention_and_transformers/5.3_vision_transformers/dino
- [ ] 5_attention_and_transformers/5.3_vision_transformers/hybrid_cnn_transformer
- [ ] 5_attention_and_transformers/5.3_vision_transformers/mae_pretraining
- [ ] 5_attention_and_transformers/5.3_vision_transformers/mlp_mixer
- [ ] 5_attention_and_transformers/5.3_vision_transformers/pyramid_vision_transformer
- [ ] 5_attention_and_transformers/5.3_vision_transformers/swin_transformer
- [ ] 5_attention_and_transformers/5.3_vision_transformers/vision_transformer
- [ ] 5_attention_and_transformers/5.4_efficient_and_long_context_attention/bigbird
- [ ] 5_attention_and_transformers/5.4_efficient_and_long_context_attention/linformer
- [ ] 5_attention_and_transformers/5.4_efficient_and_long_context_attention/longformer
- [ ] 5_attention_and_transformers/5.4_efficient_and_long_context_attention/performer
- [ ] 5_attention_and_transformers/5.4_efficient_and_long_context_attention/reformer
- [ ] 6_deep_reinforcement_learning/6.1_environments/acrobot
- [ ] 6_deep_reinforcement_learning/6.1_environments/ale_envs
- [ ] 6_deep_reinforcement_learning/6.1_environments/bandits
- [ ] 6_deep_reinforcement_learning/6.1_environments/board_games
- [ ] 6_deep_reinforcement_learning/6.1_environments/cartpole
- [ ] 6_deep_reinforcement_learning/6.1_environments/env_interface
- [ ] 6_deep_reinforcement_learning/6.1_environments/external_envs
- [ ] 6_deep_reinforcement_learning/6.1_environments/grid_world
- [ ] 6_deep_reinforcement_learning/6.1_environments/lunar_lander
- [ ] 6_deep_reinforcement_learning/6.1_environments/mountain_car
- [ ] 6_deep_reinforcement_learning/6.1_environments/pendulum
- [ ] 6_deep_reinforcement_learning/6.1_environments/render_video
- [ ] 6_deep_reinforcement_learning/6.1_environments/snake
- [ ] 6_deep_reinforcement_learning/6.1_environments/vec_env
- [ ] 6_deep_reinforcement_learning/6.1_environments/vsim_envs
- [ ] 6_deep_reinforcement_learning/6.2_tabular_and_foundational_methods/bandit_algorithms
- [ ] 6_deep_reinforcement_learning/6.2_tabular_and_foundational_methods/contextual_bandit
- [ ] 6_deep_reinforcement_learning/6.2_tabular_and_foundational_methods/dynamic_programming
- [ ] 6_deep_reinforcement_learning/6.2_tabular_and_foundational_methods/linear_fa
- [ ] 6_deep_reinforcement_learning/6.2_tabular_and_foundational_methods/monte_carlo_control
- [ ] 6_deep_reinforcement_learning/6.2_tabular_and_foundational_methods/q_learning
- [ ] 6_deep_reinforcement_learning/6.2_tabular_and_foundational_methods/sarsa
- [ ] 6_deep_reinforcement_learning/6.2_tabular_and_foundational_methods/td_lambda
- [ ] 6_deep_reinforcement_learning/6.3_value_based_deep_rl/c51
- [ ] 6_deep_reinforcement_learning/6.3_value_based_deep_rl/double_dqn
- [ ] 6_deep_reinforcement_learning/6.3_value_based_deep_rl/dqn
- [ ] 6_deep_reinforcement_learning/6.3_value_based_deep_rl/dqn_atari
- [ ] 6_deep_reinforcement_learning/6.3_value_based_deep_rl/dueling_dqn
- [ ] 6_deep_reinforcement_learning/6.3_value_based_deep_rl/iqn
- [ ] 6_deep_reinforcement_learning/6.3_value_based_deep_rl/n_step_dqn
- [ ] 6_deep_reinforcement_learning/6.3_value_based_deep_rl/noisy_networks
- [ ] 6_deep_reinforcement_learning/6.3_value_based_deep_rl/pixel_dqn
- [ ] 6_deep_reinforcement_learning/6.3_value_based_deep_rl/prioritized_replay
- [ ] 6_deep_reinforcement_learning/6.3_value_based_deep_rl/qr_dqn
- [ ] 6_deep_reinforcement_learning/6.3_value_based_deep_rl/r2d2
- [ ] 6_deep_reinforcement_learning/6.3_value_based_deep_rl/rainbow
- [ ] 6_deep_reinforcement_learning/6.4_policy_gradient_and_actor_critic/a2c
- [ ] 6_deep_reinforcement_learning/6.4_policy_gradient_and_actor_critic/a3c
- [ ] 6_deep_reinforcement_learning/6.4_policy_gradient_and_actor_critic/acer
- [ ] 6_deep_reinforcement_learning/6.4_policy_gradient_and_actor_critic/acktr
- [ ] 6_deep_reinforcement_learning/6.4_policy_gradient_and_actor_critic/actor_critic
- [ ] 6_deep_reinforcement_learning/6.4_policy_gradient_and_actor_critic/entropy_exploration
- [ ] 6_deep_reinforcement_learning/6.4_policy_gradient_and_actor_critic/gae
- [ ] 6_deep_reinforcement_learning/6.4_policy_gradient_and_actor_critic/impala
- [ ] 6_deep_reinforcement_learning/6.4_policy_gradient_and_actor_critic/ppo
- [ ] 6_deep_reinforcement_learning/6.4_policy_gradient_and_actor_critic/reinforce
- [ ] 6_deep_reinforcement_learning/6.4_policy_gradient_and_actor_critic/reinforce_baseline
- [ ] 6_deep_reinforcement_learning/6.4_policy_gradient_and_actor_critic/trpo
- [ ] 6_deep_reinforcement_learning/6.5_continuous_control/action_tricks
- [ ] 6_deep_reinforcement_learning/6.5_continuous_control/d4pg
- [ ] 6_deep_reinforcement_learning/6.5_continuous_control/ddpg
- [ ] 6_deep_reinforcement_learning/6.5_continuous_control/mpo
- [ ] 6_deep_reinforcement_learning/6.5_continuous_control/ppo_continuous
- [ ] 6_deep_reinforcement_learning/6.5_continuous_control/ppo_vsim
- [ ] 6_deep_reinforcement_learning/6.5_continuous_control/sac
- [ ] 6_deep_reinforcement_learning/6.5_continuous_control/sac_vsim
- [ ] 6_deep_reinforcement_learning/6.5_continuous_control/td3
- [ ] 6_deep_reinforcement_learning/6.6_model_based_rl_and_planning/alphazero
- [ ] 6_deep_reinforcement_learning/6.6_model_based_rl_and_planning/dreamer
- [ ] 6_deep_reinforcement_learning/6.6_model_based_rl_and_planning/dyna_q
- [ ] 6_deep_reinforcement_learning/6.6_model_based_rl_and_planning/mbpo
- [ ] 6_deep_reinforcement_learning/6.6_model_based_rl_and_planning/mcts
- [ ] 6_deep_reinforcement_learning/6.6_model_based_rl_and_planning/mpc
- [ ] 6_deep_reinforcement_learning/6.6_model_based_rl_and_planning/muzero
- [ ] 6_deep_reinforcement_learning/6.6_model_based_rl_and_planning/pets
- [ ] 6_deep_reinforcement_learning/6.6_model_based_rl_and_planning/planet
- [ ] 6_deep_reinforcement_learning/6.6_model_based_rl_and_planning/world_models
- [ ] 6_deep_reinforcement_learning/6.7_exploration/count_exploration
