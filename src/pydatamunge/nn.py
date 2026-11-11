"""Functional SWIG bindings to datamunge::nn; experiment loops live in Python."""
from contextlib import contextmanager
from operator import index as _index
from types import SimpleNamespace

from .nn_native import *
from . import nn_native as _native



def CosineDiffusionSchedule(timesteps=32, offset=.008, beta_cap=.999):
    """Construct nine owning detached Float64[T] cosine diffusion tables.

    T>=2, finite offset>=0 and finite0<beta_cap<1. Source defaults32/.008/.999.
    f(t)=cos^2((t+offset)/(1+offset)*pi/2); beta=min(cap,1-f(next)/f(current)),
    alpha=1-beta, serial alpha_bar products. Return beta,alpha,alpha_bar,
    alpha_bar_previous,sqrt_alpha_bar,sqrt_one_minus_alpha_bar,
    posterior_variance,posterior_coefficient1,posterior_coefficient2.
    First posterior variance is exactly0. Scalar Double arithmetic; no learned
    variance, sampling, RNG, fitting or static mutable cache. O(T) work/storage.
    Invalid options or degenerate/nonfinite coefficients raise RuntimeError.

    Example: CosineDiffusionSchedule().alpha_bar has32 entries, last~2.36997e-6.
    Reference: Nichol and Dhariwal(2021), Improved Denoising Diffusion
    Probabilistic Models, https://arxiv.org/abs/2102.09672 ; compact fixed-variance
    adaptation. See DiffusionExtract, DiffusionNoised and DdpmReverse.
    """
    return _native.CosineDiffusionSchedule(timesteps, offset, beta_cap)


def DiffusionExtract(coefficients, time, sample_rank=4, dtype=None):
    """Select a live coefficient table at structural integer timestep indices.

    Finite Float32/64 CPU coefficients[T],T>0 and Int32/64 CPU time[N],N>0,
    values0..T-1, arbitrary strides. Gather, cast to dtype(default Float32),
    reshape to [N,1,...,1] with sample_rank>=1 axes. Float64 output does not
    first round to Float32. Owns applicable table first/higher graph after
    parent collection; no mutation/RNG/fit. O(T+N+sample_rank) work and O(N)
    values plus shape metadata. Invalid inputs/options/cast overflow raise
    RuntimeError. DiffusionNoised/DdpmReverse explicitly round coefficients
    to Float32 even when their sample dtype is Float64.

    Example: DiffusionExtract(CosineDiffusionSchedule().beta,
      from_vector_int64([0,31],[2])) has shape[2,1,1,1].
    Reference: Ho, Jain and Abbeel(2020), Denoising Diffusion Probabilistic
    Models, https://arxiv.org/abs/2006.11239 . See CosineDiffusionSchedule.
    """
    return _native.DiffusionExtract(coefficients, time, sample_rank, DType.Float32 if dtype is None else dtype)


def DiffusionTimeEmbedding(time, maximum_time=31., dtype=None):
    """Encode structural integer times as source-rounded sine/cosine features.

    Nonnegative Int32/64 CPU time[N],N>0, arbitrary strides. Divide in Float32
    by finite positive Float32 maximum_time(default31), multiply by Float32
    pi*[1,2,4,8], concatenate four sines then four cosines, finally cast to
    dtype(default Float32). Float32 normalization/phases/features remain rounded
    even with Float64 output. Owning detached[N,8], no time derivative, fitting,
    mutation or RNG. O(N) work/storage. Invalid inputs/options or unsafe phases
    raise RuntimeError. Example: time0 gives [0,0,0,0,1,1,1,1].
    Reference: original compact DDPM teaching protocol; see DdpmEpsilon.
    """
    return _native.DiffusionTimeEmbedding(time, maximum_time, DType.Float32 if dtype is None else dtype)


def DiffusionNoised(clean, noise, time, schedule):
    """Compute a live forward conditional from explicit clean samples and noise.

    Matching finite Float32/64 CPU clean/noise[N,...],positive axes/rank>=2,
    Int32/64 CPU time[N]. Consume schedule.sqrt_alpha_bar and
    sqrt_one_minus_alpha_bar: select as Float32, then promote to sample dtype,
    even for Double. Return signal*clean+sigma*noise. Matching table lengths,
    selected signal>0 and sigma>=0 required; arbitrary user-table consistency
    is caller-owned. Applicable input/table first/higher paths and owning
    storage survive collection. No no-grad, sampling, fit, mutation or RNG.
    O(sample elements+table length) work/storage; strides accepted. Invalid
    inputs/tables or nonfinite arithmetic raise RuntimeError.

    Example: noise=zeros([1,2]) gives clean multiplied by the selected signal.
    Reference: Ho, Jain and Abbeel(2020), https://arxiv.org/abs/2006.11239 .
    See CosineDiffusionSchedule, DiffusionExtract and DdpmReverse.
    """
    return _native.DiffusionNoised(clean, noise, time, schedule)


def DdpmEpsilon(parameters, state, time):
    """Evaluate the compact5504-value DDPM epsilon MLP with live hidden endpoints.

    Finite matching Float32/64 CPU state[N,8,8,1],N>0, caller-owned floating
    store and Int32/64 time[N] in0..31, arbitrary strides. Flatten64pixels,
    append eight source-rounded time features,72->32 SiLU->32 SiLU->64.
    SiLU=x*sigmoid(x). Create missing hidden1/W,b,hidden2/W,b,output/W,b in that
    order before evaluation; normal(.1) kernels,zero biases,source seed11003.
    Six leaves/5504 values. No added prefix; use separate stores for independent
    models. Warm calls do not mutate/draw RNG. Validate inputs before leaves;
    invalid existing leaves/math may follow partial missing-leaf creation.
    Returns owning flat epsilon[N,8,8,1],time_embedding[N,8],hidden1/2[N,32].
    Epsilon/hidden first/higher state/parameter graphs survive parent/store
    collection; time features are detached structural constants. No fit,
    optimizer, timestep/noise sampler or hidden no-grad. O(N*5504) work and
    transient storage; synchronize shared stores. Invalid geometry/type/time,
    nonfinite leaves/preactivations/results raise RuntimeError.

    Example: DdpmEpsilon(ParameterStore(11003),zeros([1,8,8,1]),
      from_vector_int64([0],[1])).hidden1 has shape[1,32].
    Reference: Ho, Jain and Abbeel(2020), https://arxiv.org/abs/2006.11239 ;
    compact teaching MLP, not the paper's image U-Net. See DdpmReverse.
    """
    return _native.DdpmEpsilon(parameters, state, time)


def DdpmReverse(state, time, epsilon, innovation, schedule):
    """Apply one live fixed-posterior DDPM step; the caller owns the full loop.

    Finite matching Float32/64 CPU state/epsilon/innovation[N,...],rank>=2,
    positive axes and Int32/64 CPU time[N]. Consume signal,sigma,c1,c2,variance
    tables, with equal lengths. Select Float32 coefficients then promote even
    for Double. raw=(state-sigma*epsilon)/signal; clip to[-1,1];
    mean=c1*clipped+c2*state; value=mean+sqrt(variance)*innovation. Selected
    signal>0,sigma/variance>=0. Schedule consistency is caller-owned. Generated
    schedule at time0 has exactly zero innovation effect; innovation still
    must be finite/correctly shaped. Return owning flat value/mean/x0_raw/
    x0_clipped matching sample shape and variance[N,1,...,1]. Applicable
    input/coefficient first/higher graphs survive collection. Min/max use core
    tie subgradients; clipping at bounds and sqrt(trainable variance) at zero
    are nonsmooth. No integer-time derivative, fit, sampler, RNG, hidden no-grad
    or mutation. O(sample elements+table length) work/storage, strides accepted.
    Invalid inputs/tables or nonfinite intermediates (before clipping) raise
    RuntimeError. Example: at time0, output.value equals output.mean.
    References: Ho et al.(2020), https://arxiv.org/abs/2006.11239 ; Nichol and
    Dhariwal(2021), https://arxiv.org/abs/2102.09672 . No learned variance here.
    See CosineDiffusionSchedule, DiffusionNoised and DdpmEpsilon.
    """
    return _native.DdpmReverse(state, time, epsilon, innovation, schedule)


_native._set_doc(DiffusionScheduleOutput, "Nine owning detached Float64[T] cosine/fixed-posterior tables; first posterior variance0. No model/RNG/static cache; callers own consistency of manually supplied tables.")
_native._set_doc(DdpmEpsilonOutput, "Owning flat live epsilon[N,8,8,1], hidden1/hidden2[N,32] and detached source-rounded time_embedding[N,8]. Applicable first/higher graphs survive parent/store collection.")
_native._set_doc(DdpmReverseOutput, "Owning live flat value/mean/x0_raw/x0_clipped[N,...], variance[N,1,...,1]; source Float32 coefficients retained even for Double samples. No complete trajectory hidden inside.")


def FlowStudyPoints(standard, kind, dtype=None):
    """Transform explicit Gaussian triples into detached two-dimensional observations.

    Finite Float32/64 CPU standard[N,3],N>0, arbitrary strides. kind=0 banana
    (u,v+.5*(u*u-1)), component=-1; kind=1 equal eight-Gaussian ring (radius2,
    sd.15); kind=2 eight checkerboard cells on [-2,2)^2. For kinds1/2 use
    c=min(7,int(8*Phi(w))) with Phi=.5*erfc(-w/sqrt(2)); checker jitter uses
    Phi(u),Phi(v). Double scalar math, then one cast to dtype (default Float32).
    Tails/Float32 rounding can place checker points on boundaries; no clipping.
    Returns owning flat points[N,2] and Int32 components[N]. Caller owns RNG,
    flat3D draw grouping, splits and source IDs; components are diagnostics.
    No fit, mutation, sampling, graph or global state. O(N) work/storage.
    Invalid geometry/type/kind or nonfinite output raises RuntimeError.

    Example: FlowStudyPoints(zeros([1,3]),0).points is [[0,-.5]].
    Reference: original flow_density_2d three-family teaching protocol.
    See FlowStudyLogProbability and FlowStudySampleDiagnostics.
    """
    return _native.FlowStudyPoints(standard, kind, DType.Float32 if dtype is None else dtype)


def FlowStudyLogProbability(points, kind):
    """Evaluate detached Float64[N] analytic log densities in nats.

    Finite Float32/64 CPU points[N,2],N>0, arbitrary strides. kind=0 banana
    unit-determinant inverse, kind=1 equal eight-component Gaussian mixture
    with radius2/sd.15 using stable Double log-sum-exp, kind=2 checkerboard.
    Checker log density is -log(8) on [-2,2)^2 with even floor(x+2)+floor(y+2)
    parity, negative infinity elsewhere (legitimate zero PDF). Half-open
    boundaries are exact. No autograd, fit, sampling, mutation or RNG. Owns
    stopped storage after collection; O(N) work/storage, eight terms for kind1.
    Invalid inputs/kind or nonfinite kind0/1 calculations raise RuntimeError.

    Example: FlowStudyLogProbability(from_vector([-1.5,-1.5],[1,2]),2)=-log(8).
    Reference: original flow_density_2d analytic benchmark laws.
    See FlowStudyPoints and BananaLogProbability.
    """
    return _native.FlowStudyLogProbability(points, kind)


def FlowStudySampleDiagnostics(samples):
    """Measure all three family diagnostics for explicit samples without fitting.

    Finite Float32/64 CPU samples[N,2],0<N<=INT32_MAX, arbitrary strides.
    Returns owning Int32 mode_counts[8], checkerboard_support_fraction,
    fraction_near_modes, integer modes_covered and banana_curvature_moment.
    Nearest radius2 ring centers use Double hypot and strict comparisons (first
    exact tie wins); inclusive distance<=.45 is three sigma. Eight covered
    modes does not guarantee concentrated samples. Checker support uses exact
    half-open parity; curvature is mean(y*(x*x-1)), true banana expectation1.
    Source serial Double sums divide each contribution by N before adding.
    Detached values survive parent collection. No graph, fit, sampling,
    mutation or RNG; O(8N) work/O(N) temporary storage. Invalid input/counts
    or unsafe curvature arithmetic raises RuntimeError.

    Example: FlowStudySampleDiagnostics(zeros([1,2])).modes_covered=0.
    Reference: original flow_density_2d all-family sample diagnostic protocol.
    See FlowStudyLogProbability and FlowStudyPoints.
    """
    return _native.FlowStudySampleDiagnostics(samples)


_native._set_doc(FlowStudyPointsOutput, "Owning detached points[N,2] and Int32 components[N], -1 for banana or0..7; components are diagnostics, never model features.")
_native._set_doc(FlowStudyDiagnosticsOutput, "Owning detached Int32 mode_counts[8], checkerboard_support_fraction, fraction_near_modes, modes_covered and banana_curvature_moment. Complete mode coverage does not establish concentration.")


def CnfField(parameters, input, time, trace_mode=3):
    """Native Evaluate the source time-conditioned velocity field and exact or paired-probe divergence.
    
    Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides, finite
    Float32 scalar time. Time is a structural float argument, promoted after source
    rounding even for Float64 tensors; no derivative with respect to that scalar.
    Append time via input_col0*0+Double(time). Hidden3->8 tanh has W normal(.3),
    b=.05; output8->2 linear has W normal(.1),b=0; velocity=.3*output. Prefixes
    field/hidden and field/output. Create missing hidden W then output W BEFORE
    biases, preserving source kernel-first initialization with seed10003.50 values
    in4 leaves; no global scale here. Only missing leaves initialize; existing
    leaves unchanged and warmed calls consume no RNG.
    
    Let a,b be the two spatial rows of hidden W, c,d the output W columns, and
    s=1-hidden^2. exact_trace=.3*sum(s*(a*c+b*d)); plus=.3*sum(s*(a+b)*(c+d));
    minus=.3*sum(s*(a-b)*(c-d)). trace_mode0 selects exact,1 plus,2 minus,
    3(default) .5*(plus+minus), always returning all diagnostics. The deterministic
    pair equals the exact2-D trace in exact arithmetic; a single probe generally
    differs and is not the source fitting likelihood. These are field traces,
    not finite RK4-map log determinants.
    
    All applicable first/higher input/parameter paths remain live, including trace
    partials. Owning flat fields survive collection. No autograd differentiation
    inside the operation, fitting/optimizer/sampler/hidden no-grad or random probes.
    O(8N) work/storage; synchronize shared stores. Validate input/time/mode before
    leaves; stored-leaf/math failure can follow partial creation. Reject nonfinite
    leaves, preactivations, velocity or traces without clipping. Extreme finite
    inputs/parameters can reject; finite tanh saturation remains natural.

    Invalid shapes/dtypes/options, nonfinite fields or unsafe math raise RuntimeError.
    """
    return _native.CnfField(parameters, input, time, trace_mode)


def CnfIntegrate(parameters, input, inverse=False, steps=8, trace_mode=3):
    """Native Integrate the source augmented velocity/selected-trace ODE through fixed differentiable RK4 steps.
    
    Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides. steps>=4
    and divisible by4(default8); trace_mode0..3(default3 paired mean). Integrate
    state derivative f and accumulated selected field trace from t0=0
    to1, or1 to0 if inverse=true. Initial cumulative logdet is input_col0*0.
    Source arithmetic computes dt=(inverse?-1.f:1.f)/steps, half=.5f*dt,
    sixth=dt/6.f and ALL stage times in Float32, then promotes coefficients/time
    for tensor arithmetic, even in Float64. Preserve RK4 grouping(a+d)+2*(b+c)
    for both velocity and divergence. Four field stages per step and one additional
    endpoint field at each quarter. Expose four states, cumulative logdets and
    endpoint fields, in chronological order along the supplied direction. Reverse
    quarter endpoints run approximately.75,.5,.25,0; they are not forward checkpoints.
    
    No global scale here:50 values/4 leaves, kernel-first source seed10003. Only
    missing leaves initialize; warmed calls change no leaves or RNG. All applicable
    first/higher state/trace/parameter paths remain live through the ENTIRE unrolled
    RK4 graph. Ordinary reverse mode, no continuous adjoint, adaptive solver or
    constant-memory claim. Owns flat fields after collection. No fitting, optimizer,
    sampling stream or hidden no-grad. O(steps*8N) work and retained graph storage;
    synchronize shared stores. Inputs/options validate before leaves; stored-leaf/
    math failures may follow partial creation. Nonfinite fields/states reject.
    
    The ODE flow is mathematically invertible under its smooth field, but fixed
    finite RK4 reverse integration is a NUMERICAL inverse. Integrated divergence
    approximates continuous-flow logdet for modes0/3. Modes1/2 integrate fixed
    probe contractions and generally do not approximate the full logdet. No mode
    computes the exact discrete RK4-map logdet.
    Roundtrip, actual-Jacobian and change-of-variables residuals remain. Float32
    coefficients/times can limit refinement even with Float64 tensor arithmetic.

    Invalid shapes/dtypes/options, nonfinite fields or unsafe math raise RuntimeError.
    """
    return _native.CnfIntegrate(parameters, input, inverse, steps, trace_mode)


def CnfDensity(parameters, input, steps=8, trace_mode=3):
    """Native Map observations through the source numerical continuous flow and positive global scale.
    
    Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides. CnfIntegrate
    forward uses steps(default8,>=4,divisible4) and trace_mode(default3,0..3), then
    FlowDiagonalScale. Preserve source Float32 RK4 coefficients/times, ordinary
    reverse-mode differentiation through every step and all quarter diagnostics.
    52 values/5 leaves: field hidden3->8 tanh->2 output,50 values/4 leaves, then
    log_scale[2]. Source seed10003 creates hidden W then output W before biases,
    and global scale last. Only missing leaves initialize; warmed inference changes
    no leaves/RNG. Quarter logdets remain CUMULATIVE field integrals before scale.
    This is one shared field over time, not four independently parameterized maps;
    the original generic metadata hidden16 does not describe its actual width8.
    
    All applicable first/higher input/parameter paths stay live through the unrolled
    solver; flat owning fields survive collection. No fitting, optimizer, sampling
    stream, adaptive solve, continuous adjoint or hidden no-grad. O(steps*8N) work/
    retained graph storage; synchronize stores. Inputs/options validate before leaves;
    stored-leaf/math failures can follow partial creation. Reject nonfinite results
    or unsafe global scale. Modes0/3 approximate the flow likelihood; modes1/2 are
    fixed-probe diagnostic integrals, generally not a density. No mode computes
    exact discrete RK4-map logdet; finite numerical residuals remain.

    Invalid shapes/dtypes/options, nonfinite fields or unsafe math raise RuntimeError.
    """
    return _native.CnfDensity(parameters, input, steps, trace_mode)


def CnfInverse(parameters, latent, steps=8, trace_mode=3):
    """Native Numerically reverse the compact continuous-flow density and accumulate its own inverse-path trace.
    
    Finite matching Float32/64 CPU latent[N,2],N>0, arbitrary strides. Validate
    steps(default8,>=4,divisible4)/trace_mode(default3,0..3) before leaves. Unscale
    supplied latent by exp(-log_scale), integrate1->0 with CnfIntegrate and subtract
    sum(log_scale) from its reverse accumulated trace. Recompute every RK4 field
    on that supplied inverse path; no cached forward points. Source Float32 times/
    coefficients apply even for Float64 tensors. Owns live value/logdet after
    collection; applicable first/higher input/parameter paths differentiate through
    the full unrolled reverse solver. This is a numerical reverse ODE path, not an
    exact inverse of the finite forward RK4 map. Its integrated trace likewise
    approximates continuous inverse logdet for modes0/3 only. Modes1/2 integrate
    fixed-probe diagnostic contractions, generally not full inverse logdet.
    Numerical residuals remain.
    
    A forward-warmed store reuses52 values/5 leaves without writes or RNG; missing
    inverse leaves initialize global scale before field kernels/biases, so warm
    CnfDensity first with source seed10003 for source ordering. No fitting, optimizer,
    sampling stream, adaptive solve, continuous adjoint, hidden no-grad or inverse
    refit. O(steps*8N) work/retained graph storage; synchronize stores. Input/options
    validate before leaves; stored-leaf/math failures may follow partial creation.
    Reject nonfinite fields/results or unsafe global scale without clipping.

    Invalid shapes/dtypes/options, nonfinite fields or unsafe math raise RuntimeError.
    """
    return _native.CnfInverse(parameters, latent, steps, trace_mode)


_native._set_doc(CnfFieldOutput, "Five owning live flat fields: velocity[N,2], divergence/exact_trace/hutchinson_plus/hutchinson_minus[N]. Fixed plus/minus probes average to the exact2D field trace within roundoff; first/higher paths survive collection.")
_native._set_doc(CnfDensityOutput, "Nineteen owning live flat fields: stage0..3[N,2], CUMULATIVE stage_logdet0..3[N], endpoint stage_divergence0..3[N]/stage_velocity0..3[N,2], unscaled/latent[N,2] and logdet[N]. Modes0/3 integrated trace approximates continuous-flow logdet, not exact discrete RK4-map logdet; modes1/2 fixed-probe diagnostic integrals generally do not approximate full logdet. All applicable first/higher paths survive collection.")


def PlanarFlow(input, angle, log_frequency, raw_projection, perpendicular, bias, inverse=False):
    """Native Apply a constrained two-dimensional planar map or its numerical implicit-gradient inverse.
    
    Finite Float32/64 CPU input[N,2],N>0, arbitrary strides; all five parameters
    are finite matching CPU scalar Tensors. Set e=exp(log_frequency), direction=
    (cos(angle),sin(angle)), tangent=(-sin(angle),cos(angle)), projection=
    softplus(raw_projection)-.9, w=e*direction, u=(projection/e)*direction+
    perpendicular*tangent. Forward t=sum(input*w)+bias, value=input+u*tanh(t),
    factor=1+projection*(1-tanh(t)^2), logdet=log(factor) per row.
    
    Inverse solves t+projection*tanh(t)=c=sum(input*w)+bias using32 bisections
    bracketed by c+/- (abs(projection)+1), then ONE Newton correction: stop root
    and denominator, t=root+((c-root)-projection*tanh(root))/denominator. Return
    input-u*tanh(t) and negative log(factor) evaluated on corrected t. This is a
    numerical inverse, not a closed form. First gradients use the source implicit
    correction, avoiding discrete bisection branches. Higher graphs remain live
    through the correction but are NOT exact higher implicit derivatives because
    the denominator/root are stopped. Floating residuals remain.
    
    Owns all fields after collection; forward applicable first/higher paths stay
    live. No store/RNG/fitting/optimizer/sampler/hidden no-grad. O(32N) inverse or
    O(N) forward work/storage. Calls share no mutable state. Validate inputs before
    arithmetic; reject zero/overflowed frequency/reciprocal, nonfinite intermediates
    and nonpositive represented determinants without clipping or fallback.

    Invalid shapes/dtypes, nonfinite fields or unrepresentable math raise RuntimeError.
    """
    return _native.PlanarFlow(input, angle, log_frequency, raw_projection, perpendicular, bias, inverse)


def RadialFlow(input, center, raw_alpha, raw_beta, inverse=False):
    """Native Apply a constrained radial map or its stable quadratic-radius inverse.
    
    Finite Float32/64 CPU input[N,2],N>0, center[2], raw_alpha/raw_beta scalars,
    all matching dtype and arbitrary strides. alpha=.1+softplus(raw_alpha)>0,
    beta=softplus(raw_beta)-alpha > -alpha. Offset=input-center, distance is the
    source stabilized radius: sqrt(max(sum(offset^2),1e-30)) for positive squared
    radius, explicitly zero with zero radius derivative at center. Forward uses
    r=distance, factor=1+beta/(alpha+r), radial_factor=1+beta*alpha/(alpha+r)^2;
    value=center+offset*factor and logdet=log(factor)+log(radial_factor).
    
    Inverse recovers r from r^2+(alpha+beta-distance)*r-alpha*distance=0.
    With a=alpha+beta-distance and disc=sqrt(a*a+4*alpha*distance), use
    2*alpha*distance/max(a+disc,1e-20) for a>=0, otherwise .5*(disc-a).
    Return center+offset/factor and negated logdet at that recovered radius.
    Both quadratic branches are evaluated; inactive denominator is stabilized.
    At center the value derivative is factor*I forward and I/factor inverse;
    source radius derivative is zero, including logdet's chosen center derivative.
    Do not infer a smooth mathematical Hessian of radius/logdet at center.
    
    Owning live flat fields survive collection; applicable first/higher input and
    parameter graphs remain within structural branches and the source radius rule.
    No store/RNG/fitting/optimizer/sampler/hidden no-grad. O(N) work/storage, no
    shared mutable state. Inputs validate before arithmetic; reject nonpositive
    represented alpha, alpha+beta, discriminant, denominator or factors and any
    nonfinite intermediate/result. Extreme finite parameters may reject; floating
    inverse residuals and the source tiny-radius stabilization remain.

    Invalid shapes/dtypes, nonfinite fields or unrepresentable math raise RuntimeError.
    """
    return _native.RadialFlow(input, center, raw_alpha, raw_beta, inverse)


def ElementaryFlow(parameters, input, block, radial=False, inverse=False):
    """Native Apply one named source planar or radial flow with caller-owned parameters.
    
    Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides, block0..3.
    radial=false(default) creates flow<block>/angle,log_frequency,raw_projection,
    perpendicular,bias scalar leaves in that order: angle=Double(.4f+.65f*block),
    log_frequency=0, raw_projection=.4, perpendicular=Double(.02f*(odd?-1:1)),
    bias=.05. Five values/five leaves. radial=true creates center[2] normal(.2),
    raw_alpha=.4, raw_beta=.56: four values/three leaves. Calls PlanarFlow or
    RadialFlow with those leaves; diagnostics retain family-specific semantics.
    Use a separate store per family: their shared prefix does not convert or clear
    other-family leaves. Only missing leaves initialize; existing leaves unchanged.
    Warm source blocks0..3 with seed9903 before inverses to preserve source order.
    
    Owns outputs/graphs after collection. Planar inverse first/higher graphs follow
    its stopped-root/denominator correction; higher paths are not exact implicit
    higher derivatives. Radial center/tiny-radius rules are source conventions.
    No fitting/optimizer/sampler/hidden no-grad or warmed RNG. O(N) forward/radial
    or O(32N) planar inverse work/storage. Synchronize shared stores. Inputs validate
    before leaves; stored-leaf/math errors can follow partial creation.

    Invalid shapes/dtypes, nonfinite fields or unrepresentable math raise RuntimeError.
    """
    return _native.ElementaryFlow(parameters, input, block, radial, inverse)


def PlanarRadialDensity(parameters, input, radial=False):
    """Native Map observations through four source planar or radial stages and global scale.
    
    Finite matching Float32/64 CPU input[N,2],N>0, arbitrary strides. Apply
    ElementaryFlow blocks0..3 in order, accumulating each per-row logdet, then
    FlowDiagonalScale. Use one family per store: radial=false(default) planar has
    22 values/21 leaves; radial=true has18 values/13 leaves. Source seed9903 warms
    each block in order then log_scale[2]. Only missing leaves initialize; warmed
    inference changes no leaves and consumes no RNG. Owns all live flat stages,
    factors, coordinates and density intermediates after collection. Applicable
    first/higher forward paths remain live under the source radius conventions.
    No fitting/optimizer/sampler/hidden no-grad or VAE/ELBO workflow. O(N) work/
    storage for four fixed stages. Synchronize shared stores. Inputs validate before
    leaves; stored-leaf/math failure may follow partial creation. Reject nonfinite
    or unrepresentable flow/scale math; floating inverse residuals remain.

    Invalid shapes/dtypes, nonfinite fields or unrepresentable math raise RuntimeError.
    """
    return _native.PlanarRadialDensity(parameters, input, radial)


def PlanarRadialInverse(parameters, latent, radial=False):
    """Native Invert the compact planar/radial density and accumulate its own path determinant.
    
    Finite matching Float32/64 CPU latent[N,2],N>0, arbitrary strides. Unscale
    globally, then ElementaryFlow blocks3..0 with inverse=true. Recompute each
    stage on its supplied inverse path, independent of cached forward points.
    Planar uses32 bisections plus the stopped-root/denominator Newton correction:
    source implicit first gradients remain live, higher graphs follow that stopped
    correction and are not exact implicit higher derivatives. Radial uses stable
    quadratic radius inversion and explicit finite center derivative. Missing
    leaves initialize global scale then reversed blocks; warm PlanarRadialDensity
    first with seed9903 for source ordering. Use a separate store per family.
    
    Owns live value/logdet after collection; applicable input/parameter graph paths
    remain. No fitting/optimizer/sampler/hidden no-grad or warmed writes/RNG.
    O(128N) planar or O(N) radial work/storage. Synchronize shared stores. Inputs
    validate before leaves; stored-leaf/math failures may follow partial creation.
    Reject nonfinite or unrepresentable math without clipping/fallback. Numerical
    planar and floating radial residuals remain; neither path refits parameters.

    Invalid shapes/dtypes, nonfinite fields or unrepresentable math raise RuntimeError.
    """
    return _native.PlanarRadialInverse(parameters, latent, radial)


_native._set_doc(ElementaryFlowOutput, "Four owning live flat fields: value[N,2], signed logdet[N], factor/coordinate[N,1]. Planar factor is determinant, coordinate solved t; radial factor tangential, coordinate source radius. Planar inverse higher graphs follow stopped correction, not exact implicit higher derivatives.")
_native._set_doc(PlanarRadialDensityOutput, "Nineteen owning live flat fields: flow/logdet/factor/coordinate0..3, unscaled/latent and full logdet. Factor/coordinate diagnostics are not affine scales/shifts; applicable graphs/storage survive collection.")


def RationalQuadraticSpline(input, heads, inverse=False, bound=4.0,
                            min_bin_fraction=.01, min_derivative=.001,
                            rounded_offset=False):
    """Native monotone rational-quadratic scalar spline and analytic quadratic inverse.

    Matching finite Float32/64 CPU input[N,1], heads[N,3*K-1], N>0,K>=2,
    arbitrary strides. FirstK width logits, nextK height logits, lastK-1 slope
    logits. Declared sizes=2*bound*(fraction+(1-K*fraction)*softmax(logits));
    returned widths/heights are actual cumulative-knot differences. Interior
    slopes=floor+softplus(raw+log(expm1(1-floor))), boundary slopes1.
    Positive finite bound(default4), fraction(default.01,K*fraction<1),
    floor(default.001,strictly0..1). rounded_offset=True rounds the softplus
    offset to Float32 before promotion for original-source replay; False keeps
    Double. Zero heads are near identity with floating rounding.
    Right-bin ownership at exact knots, analytic inverse2c/(-b-sqrt(b*b-4ac)),
    inclusive spline endpoints, strict outside identity tails/logdet0. No root
    search. Owns five live flat fields: value[N,1],logdet[N],widths/heights[N,K],
    derivatives[N,K+1]. First/higher input/head graphs survive collection within
    structural branches; log-derivative partials can be one-sided at knots.
    No store, RNG, fitting, optimizer, sampler or hidden no-grad. Invalid shapes/
    options, nonpositive represented bins/slopes/discriminants/derivative terms
    or nonfinite math raise RuntimeError. Extreme finite fields may reject.
    """
    return _native.RationalQuadraticSpline(input, heads, inverse, bound,
                                          min_bin_fraction, min_derivative, rounded_offset)


def SplineCoupling(parameters, input, block, inverse=False):
    """Native source four-bin rational-quadratic coupling on two coordinates.

    Finite matching Float32/64 CPU[N,2], positive rows/strides, block0..3.
    Preserve raw coordinate block%2;1->16 tanh->11 conditioner, hidden W normal
    (.3),b=.05, heads W normal(.01),b=0, hidden leaves first.219 values/4 leaves
    per block. Pure spline K4,bound4,fraction.01,floor.001,rounded_offset=True
    changes the other coordinate. Owns value[N,2],logdet[N],widths/heights[N,4],
    derivatives[N,5]. Bin sizes are not log scales or shifts. First/higher graphs
    survive collection within structural bins. Only missing leaves initialize;
    no existing writes, warmed RNG, fitting, optimizer, sampler or hidden no-grad.
    Invalid/nonfinite inputs/leaves or unrepresentable spline math raise
    RuntimeError, possibly after partial leaf creation. Synchronize stores.
    """
    return _native.SplineCoupling(parameters, input, block, inverse)


def SplineDensity(parameters, input):
    """Native compact spline flow: four rational-quadratic couplings and global scale.

    Matching finite Float32/64 CPU[N,2], positive rows/strides. Source blocks0..3,
    accumulating their own per-row logdets, then FlowDiagonalScale. Source
    seed9803,878 values/17 leaves; only missing leaves initialize. Owns23 live
    flat fields: coupling0..3[N,2],coupling_logdet0..3[N],coupling_widths/heights0..3
    [N,4],coupling_derivatives0..3[N,5],unscaled/latent[N,2],full logdet[N]. Source
    scale/shift archive fields correspond to public actual widths/heights.
    First/higher paths survive collection within structural bins. No fitting,
    optimizer, sampler, root search, hidden no-grad or warmed writes/RNG.
    Invalid/nonfinite inputs/leaves or unrepresentable spline/scale math raise
    RuntimeError, possibly after partial creation. Synchronize mutable stores.
    """
    return _native.SplineDensity(parameters, input)


def SplineInverse(parameters, latent):
    """Native compact spline inverse with per-point determinants on its own path.

    Matching finite Float32/64 CPU[N,2], positive rows/strides. Unscale globally,
    reverse blocks3..0, recomputing heads on unchanged conditioning coordinates.
    Inverse bins use height edges and closed quadratic roots. Owns live
    value[N,2]/logdet[N], applicable first/higher graphs after collection within
    structural bins. Warm SplineDensity for source seed9803 initialization order;
    missing inverse leaves initialize global scale then reversed blocks.
    No cached forward state, root search, fitting, optimizer, sampler or warmed
    writes/RNG. Invalid/nonfinite inputs/leaves or unrepresentable spline/scale
    math raise RuntimeError, possibly after partial creation. Floating inverse
    errors remain. Synchronize shared mutable stores.
    """
    return _native.SplineInverse(parameters, latent)


_native._set_doc(RationalQuadraticSplineOutput, "Five owning live floating fields: value[N,1] for pure spline or[N,2] for coupling, signed logdet[N], actual widths/heights[N,K] and positive knot derivatives[N,K+1]; first/higher graphs survive collection within structural bins.")
_native._set_doc(SplineDensityOutput, "Twenty-three owning live flat coupling, actual bin geometry and Gaussian-density fields; widths/heights are positive sizes, not log scales/shifts. Applicable first/higher graphs survive parent/store collection.")


def IafDensity(parameters, input):
    """Native affine IAF density with sequential recovery and final head re-evaluation.

    Finite matching Float32/64 CPU[N,2], positive rows/strides. Reverse blocks3..0;
    recover coordinates0,1(even) or1,0(odd), recomputing MafConditioner on each
    prefix before u_c=(observed_c-mu_c)*exp(-s_c). Re-evaluate heads on fully
    recovered coordinates for negative sum(s), preserving the original graph
    and rounding. Final global FlowDiagonalScale. Three conditioner calls/block.
    Source seed9703 warms blocks3,2,1,0 then scale;466 values/17 leaves,160 masked
    entries with zero partials. Owns23 live flat fields: conditioner_input and
    coupling0..3[N,2] are recovered inputs; coupling_scale/shift0..3[N,2] are
    final LOG scales/means; coupling_logdet0..3[N], unscaled= coupling0[N,2],
    latent[N,2], full logdet[N]. First/higher paths survive collection. Only
    missing leaves initialize; no existing writes, warmed RNG, fitting, optimizer,
    sampler or hidden no-grad. Standalone affine exact-density adaptation, not
    the paper's sigmoid-gated VAE/ELBO workflow. Invalid/nonfinite fields/scales
    raise RuntimeError, possibly after partial leaf creation. Synchronize stores.
    """
    return _native.IafDensity(parameters, input)


def IafInverse(parameters, latent):
    """Native parallel IAF observation map and its own inverse-path determinant.

    Finite matching Float32/64 CPU[N,2], positive rows/strides. Unscale globally,
    then blocks0..3: one MafConditioner call on each known input; h=h*exp(s)+mu,
    add sum(s) per row. Four calls total, versus twelve for sequential density.
    Owns live value[N,2]/logdet[N], applicable first/higher graphs after collection.
    Forward warmup preserves source seed9703 initialization order; missing
    inverse leaves initialize global scale then blocks0..3. No cached forward
    state, fitting, optimizer, sampler, VAE workflow or warmed writes/RNG.
    Invalid/nonfinite inputs/leaves/scales/results raise RuntimeError, possibly
    after partial creation. Both global scale directions positive finite.
    Floating roundtrip errors remain. Synchronize shared mutable stores.
    """
    return _native.IafInverse(parameters, latent)


_native._set_doc(IafDensityOutput, "Twenty-three owning live flat recovered IAF stage/head/density fields, applied in source block order3,2,1,0; unscaled aliases coupling0. Applicable first/higher graphs survive collection; no cached forward or fitting state.")


def MafMasks(block, dtype=None):
    """Native detached strict Gaussian conditioner degrees and masks.

    Block0..3 alternates input degrees[1,2]/[2,1]; hidden floor(j/8),16 units,
    head degrees repeat input for mu0,mu1,raw_s0,raw_s1. Input mask<=, output
    strict<. Owns five flat fields: Int32 input[2]/hidden[16]/output[4] degrees,
    Float32(default)/64 input_mask[2,16],output_mask[16,4]. Forty forbidden
    kernel entries/block, no own/future head dependency. No store, RNG, fitting,
    mutation or graph. Invalid block/dtype raises RuntimeError.
    """
    return _native.MafMasks(block, DType.Float32 if dtype is None else dtype)


def MafConditioner(parameters, input, block):
    """Native strict Gaussian conditioner, used by compact MAF and paired IAF.

    Finite matching Float32/64 CPU[N,2], positive rows/strides, block0..3.
    autoregressive<block>/hidden:2->16 tanh, missing W normal(.3), b=.05;
    /heads:16->4 linear, W normal(.01), b=0. Hidden W/b then heads W/b;
    strict structural mask multiplication stays live inside native dense ops.
    Owns mean/scale[N,2]; scale=.8*tanh(raw), a LOG scale, not exponential.
    Four leaves/116 values,40 forbidden kernel coordinates with zero partials.
    First/higher graphs survive parent/store collection. Only missing leaves
    initialize; no existing writes, fitting, optimizer, sampling or warmed RNG.
    Invalid/nonfinite inputs/leaves/results raise RuntimeError; math failures
    may follow partial leaf creation. Synchronize shared mutable stores.
    """
    return _native.MafConditioner(parameters, input, block)


def MafDensity(parameters, input):
    """Native compact MAF with parallel density and four strict Gaussian conditioners.

    Matching finite Float32/64 CPU[N,2], positive rows/strides. Blocks0..3:
    parallel (x-mu)*exp(-s), logdet=-sum(s), then global FlowDiagonalScale.
    Source seed9603,466 stored values/17 leaves including160 masked entries.
    Owns23 live flat fields: conditioner_input/coupling/scale/shift0..3[N,2],
    coupling_logdet0..3[N], unscaled/latent[N,2], full logdet[N]. Scale fields
    are bounded LOG scales; own/future conditioner input partials stay zero.
    First/higher graphs survive collection. No fitting, optimizer, sampler,
    hidden no-grad, existing-leaf writes or warmed RNG. Invalid/nonfinite
    inputs/leaves/scale/results raise RuntimeError, possibly after creation.
    """
    return _native.MafDensity(parameters, input)


def MafInverse(parameters, latent):
    """Native sequential compact MAF inverse with its own per-point determinant.

    Matching finite Float32/64 CPU[N,2], positive rows/strides. Unscale globally,
    reverse blocks3..0; for each block recover0,1(even) or1,0(odd), recomputing
    MafConditioner on the recovered prefix before each coordinate. Set
    x_c=z_c*exp(s_c)+mu_c, accumulating each used s in dependency order.
    Two conditioner passes/block; forward density uses one. Owns live value[N,2]
    and logdet[N]; applicable first/higher paths survive collection. Warm
    MafDensity for source seed9603 ordering; missing inverse leaves initialize
    in reverse order. No cached forward state, fitting, optimizer, sampler,
    existing writes or warmed RNG. Invalid/nonfinite results raise RuntimeError,
    possibly after leaf creation. Floating roundtrip errors remain.
    """
    return _native.MafInverse(parameters, latent)


_native._set_doc(MafMaskInventory, "Five owning detached fields: Int32 input/hidden/output degrees and floating input/output masks; no own/future head dependency, survives collection without a store or graph.")
_native._set_doc(MafConditionerOutput, "Two owning live floating[N,2] mean and bounded LOG scale fields, with strict masked first/higher graphs surviving collection.")
_native._set_doc(MafDensityOutput, "Twenty-three owning live flat conditioner/stage/density fields; all means/log scales/inputs and per-row determinants retain first/higher graphs after collection.")


def FlowActNormCalibration(input, epsilon=1e-6):
    """Native detached ActNorm moments and proposed parameters from explicit rows.

    Finite Float32/64 CPU[N,D], positive axes/strides; epsilon finite positive,
    default1e-6. Serial divide-first Double mean/population variance, divisor N;
    bias=-mean, log_scale=-.5*log(var+epsilon), cast once to input dtype.
    Owns flat mean/variance Float64[D], bias/log_scale matching floating[D].
    Constant columns and N=1 supported. No store, writes, selection, calibration
    flag, RNG or retained input graph. Caller installs fitting-only parameters
    once, sequentially at each stage. Invalid/nonfinite fields raise RuntimeError;
    a later transform may reject extreme unrepresentable exponentials.
    """
    return _native.FlowActNormCalibration(input, epsilon)


def FlowActNorm(parameters, name, input, inverse=False):
    """Native named ActNorm without automatic calibration or batch statistics.

    Finite matching floating CPU[N,D], nonempty name, positive axes/strides.
    Missing name/bias[D], name/log_scale[D] zero. Forward (x+bias)*exp(scale),
    scalar logdet=sum(scale); inverse x*exp(-scale)-bias, negative logdet.
    Both exponentials positive finite. Owns live value[N,D], scalar logdet;
    first/higher graphs survive parent/store collection. No calibration, fitting,
    optimizer, sampling, existing-leaf writes or warmed RNG. Invalid/nonfinite
    values raise RuntimeError; leaf/math failure may follow partial creation.
    """
    return _native.FlowActNorm(parameters, name, input, inverse)


def FlowMix2D(parameters, name, input, angle=.35, inverse=False):
    """Native signed invertible row-matrix mixing on matching floating CPU[N,2].

    Nonempty name, finite angle(default .35 radians), positive rows/strides.
    Missing name/W[2,2] is cos/sin rotation, Double arithmetic rounded once to
    Float32 then promoted if needed; consumes no RNG. Forward x@W, inverse
    x@adjugate(W)/det(W); scalar logdet=+/-.5*log(det(W)^2). Negative det valid,
    singular or unrepresentable positive finite squared det rejects. Row input
    Jacobian is W transpose. Owns live value/logdet, first/higher paths survive
    collection. No fitting, calibration, optimizer, sampling or existing-leaf
    mutation; invalid/nonfinite results raise RuntimeError, possibly after
    partial leaf creation. Synchronize shared mutable stores.
    """
    return _native.FlowMix2D(parameters, name, input, angle, inverse)


def GlowDensity(parameters, input):
    """Native compact two-channel Glow: ActNorm, signed mixing and bounded affine coupling.

    Finite matching Float32/64 CPU[N,2], positive rows/strides. Four stages:
    named actnorm<i> (missing zeros), mix<i> rotation(.35+.2*i), RealNvpCoupling
    block i. Source seed9503,296 values/28 leaves; no final global scale.
    Accumulate ActNorm+(mix+coupling) logdet in source order. Owns31 live flat
    source fields, including actnorm_input/output0..3, mixed0..3, coupling0..3,
    coupling_logdet/scale/shift0..3, unscaled/latent[N,2], total logdet[N].
    Scale fields are bounded LOG scales[N,1], per-block logdet[N]. Applicable
    first/higher graphs survive parent/store collection. No automatic calibration,
    batch statistics, fitting, optimizer, sampling, existing-leaf writes or
    warmed RNG. Caller selects first32 fitting points and installs sequential
    ActNorm proposals explicitly for source replay. Invalid/nonfinite fields or
    scales/matrices raise RuntimeError, possibly after partial creation.
    """
    return _native.GlowDensity(parameters, input)


def GlowInverse(parameters, latent):
    """Native exact-form compact Glow inverse and its own per-point log determinant.

    Matching finite floating CPU[N,2], positive rows/strides. Reverse stages3..0:
    inverse affine coupling, inverse row mixing, inverse ActNorm. Compute each
    determinant on that inverse path; negative matrix signs use log absolute
    determinants. Owns live value[N,2], logdet[N]; applicable first/higher graphs
    survive parent/store collection. Warm forward for source seed9503 ordering;
    missing inverse leaves may initialize in reverse order. No calibration,
    fitting, optimizer, sampling or warmed writes/RNG. Invalid/nonfinite fields
    raise RuntimeError, possibly after partial creation. Floating errors remain.
    """
    return _native.GlowInverse(parameters, latent)


_native._set_doc(ActNormCalibration, "Four owning detached flat CPU[D] moments/proposals: mean and population variance Float64, bias/log_scale matching input floating dtype; survives collection, no store writes or calibration state.")
_native._set_doc(GlowDensityOutput, "Thirty-one owning live flat source ActNorm/mixing/affine fields, unscaled/latent and per-row full logdet; first/higher graphs survive parent/store collection, no hidden calibration state.")


def FlowDiagonalScale(parameters, input, incoming_logdet, inverse=False):
    """Native live global diagonal scale with an explicit incoming log determinant.

    Finite matching Float32/64 CPU input[N,D], incoming scalar or[N], positive
    axes/strides. Missing log_scale[D] initializes zero. Forward multiplies by
    exp(log_scale), adds sum(log_scale); inverse divides and subtracts the sum.
    Both scale directions must be positive finite. Returns owning live flat
    value[N,D], scalar or[N] logdet; first/higher graphs survive parent/store
    collection. No fitting/optimizer/sampling or warmed-store writes/RNG.
    Invalid geometry/dtypes or nonfinite values raise RuntimeError; leaf/math
    failure may follow partial creation. Synchronize shared mutable stores.
    """
    return _native.FlowDiagonalScale(parameters, input, incoming_logdet, inverse)


def RealNvpCoupling(parameters, input, block, inverse=False):
    """Native compact bounded affine coupling, block0..3, on floating CPU[N,2].

    Finite matching store/input; condition on raw coordinate block%2. Prefix
    coupling<block>/hidden1->16 tanh uses missing W normal(.3), b=.05;
    /heads16->2 uses normal(.01), b0. s=.8*tanh(head0), t=head1;
    y_B=x_B*exp(s)+t, inverse (y_B-t)*exp(-s). Owns live flat value[N,2],
    logdet[N] (+s or-s), scale/shift[N,1]. scale is bounded LOG scale, with .8
    rounded to computation dtype. Four leaves/66 values. Applicable first/
    higher paths survive parent/store collection. Only missing leaves initialize;
    no fitting/optimizer/sampling/warmed-store writes or hidden no-grad. Invalid
    inputs/nonfinite results raise RuntimeError; partial creation can precede
    stored-leaf/math failures. Arbitrary strides; synchronize shared stores.
    """
    return _native.RealNvpCoupling(parameters, input, block, inverse)


def RealNvpDensity(parameters, input):
    """Native four-stage bounded affine RealNVP with global positive diagonal scale.

    Finite matching Float32/64 CPU[N,2], positive rows/strides. Source seed9403
    warms266 values/17 leaves: four1->16 tanh->2 conditioners, then log_scale[2].
    Owns19 live flat fields: coupling0..3/unscaled/latent[N,2], total logdet[N],
    coupling_logdet0..3[N], coupling_scale0..3/coupling_shift0..3[N,1]. Scales are
    bounded log scales, not exp(s). Accumulate determinants in source stage
    order. Applicable first/higher graphs survive parent/store collection.
    Only missing leaves initialize; warmed inference never mutates or samples.
    No fitting/optimizer/normalization/dropout/image multiscale stack/no-grad.
    Both global scale directions must stay positive finite; invalid inputs or
    nonfinite results raise RuntimeError, possibly after partial leaf creation.
    """
    return _native.RealNvpDensity(parameters, input)


def RealNvpInverse(parameters, latent):
    """Native exact-form RealNVP inverse and its own per-point log determinant.

    Finite matching floating CPU[N,2], positive rows/strides. Unscale globally,
    then reverse affine blocks3,2,1,0, evaluating each log scale/shift on the
    inverse path's current conditioning coordinate. No cached forward determinant.
    Owns live flat value[N,2], logdet[N]; first/higher paths survive collection.
    Warm RealNvpDensity first for source seed9403 order; missing inverse leaves
    initialize scale then reversed conditioners. No fitting/optimizer/sampling
    or warmed-store writes/RNG. Both global scale directions positive finite;
    invalid inputs/nonfinite results raise RuntimeError, possibly after partial
    leaf creation. Floating roundtrip errors remain; synchronize shared stores.
    """
    return _native.RealNvpInverse(parameters, latent)


_native._set_doc(FlowScaleOutput, "Two owning live flat value[N,D] and scalar or per-row[N] logdet fields; matching floating dtype, applicable first/higher graphs/storage survive parent/store collection.")
_native._set_doc(RealNvpCouplingOutput, "Four owning live flat value[N,2], logdet[N], bounded LOG scale/shift[N,1]; applicable first/higher graphs survive parent/store collection.")
_native._set_doc(RealNvpDensityOutput, "Nineteen owning live flat source affine-stage, log-scale, shift, logdet, unscaled and latent fields; total logdet per row, applicable first/higher graphs survive parent/store collection.")


def BananaPoints(standard, curvature=.5, dtype=None):
    """Native detached banana observations from an explicit finite CPU[N,2] bank.

    Float32/64 bank and finite curvature(default .5); Double arithmetic computes
    (u,v+curvature*(u*u-1)), then casts once to dtype(Float32 default/Float64).
    Owns contiguous stopped data; accepts strides. No RNG, fitting, row selection,
    input mutation or retained graph. Nonfinite coordinates/invalid options raise
    RuntimeError. The source uses GaussianNoise([2],N,seed) as its Double bank.
    """
    return _native.BananaPoints(standard, curvature, DType.Float32 if dtype is None else dtype)


def BananaLogProbability(points, curvature=.5):
    """Native live Float64[N] analytic banana log density, in nats.

    Finite Float32/64 CPU[N,2], finite curvature(default .5), arbitrary strides.
    Invert the quadratic bend and use full Double log(2*pi). Input first/higher
    paths survive parent collection. No sampling/fitting/mutation; invalid or
    nonfinite computed values raise RuntimeError.
    """
    return _native.BananaLogProbability(points, curvature)


def StandardNormalLogProbability(latent, rounded_constant=False):
    """Native live matching Float32/64[N] Gaussian log density on finite CPU[N,D].

    Positive axes/strides; -.5*sum(z*z,axis1)-.5*D*log(2*pi). Explicit
    rounded_constant=True first rounds log(2*pi) to Float32 even for Double,
    preserving the compact flow source. Default False uses full Double constant
    before dtype promotion. No sampling/fitting/mutation. Nonfinite/invalid
    inputs or results raise RuntimeError; applicable higher paths remain live.
    """
    return _native.StandardNormalLogProbability(latent, rounded_constant)


def NormalFlowDensity(latent, forward_logdet, rounded_constant=False):
    """Native Gaussian change-of-variables density and mean row NLL, in nats.

    Finite matching floating CPU latent[N,D], determinant scalar or[N]; positive
    axes/strides. Determinant describes x->z. Return owning live flat
    base_log_probability[N], log_probability[N] and scalar loss; fields/graphs
    survive parent collection. rounded_constant=True preserves source Float32
    log(2*pi). Does not prove invertibility/compute Jacobians, fit, sample or
    mutate. Invalid geometry/dtypes or nonfinite results raise RuntimeError.
    """
    return _native.NormalFlowDensity(latent, forward_logdet, rounded_constant)


def NiceCoupling(parameters, input, block, inverse=False):
    """Native live source additive NICE block0..3 on matching floating CPU[N,2].

    Condition on coordinate block%2; named coupling<block>/hidden1->16 tanh
    normal(.3)/bias.05, shift16->1 normal(.01)/bias0. Add shift to other raw
    coordinate, or subtract for inverse=True; analytic determinant one. Only
    missing leaves initialize. Four leaves/49 values per block. No fitting,
    optimizer, sampling, existing-leaf writes or hidden no-grad. First/higher
    paths survive store/parent collection. Invalid/nonfinite values raise
    RuntimeError; stored-leaf failures may follow partial creation.
    """
    return _native.NiceCoupling(parameters, input, block, inverse)


def NiceDensity(parameters, input):
    """Native four-stage compact NICE map and global positive diagonal scale.

    Finite matching Float32/64 CPU[N,2], positive rows/strides. Blocks0..3, then
    exp(log_scale[2]); zero scale initialization. Source seed9301 creates198
    values/17 leaves. Seven owning live fields coupling0..3, unscaled,latent[N,2]
    and scalar logdet=sum(log_scale). Graphs survive store/parent collection.
    Only missing leaves initialize; no optimizer/fitting/sampling/mutation after
    warmup. Nonfinite values, zero/overflow scale or invalid inputs raise
    RuntimeError; leaf/math failures may follow partial creation.
    """
    return _native.NiceDensity(parameters, input)


def NiceInverse(parameters, latent):
    """Native exact-form NICE inverse: unscale, subtract shifts3,2,1,0.

    Matching finite floating CPU[N,2], positive rows/strides. Owns live flat
    value[N,2], scalar logdet=-sum(log_scale); first/higher paths survive parent
    collection. Warm with NiceDensity first to preserve source initialization
    order. Missing leaves can initialize in inverse order; warmed stores never
    mutate. No fitting/optimizer/sampling/RNG after warmup. Both scale and its
    reciprocal must be positive finite; invalid/nonfinite results raise
    RuntimeError. Floating roundtrip errors remain.
    """
    return _native.NiceInverse(parameters, latent)


_native._set_doc(NormalFlowDensityOutput, "Three owning live flat base_log_probability/log_probability[N] and scalar mean NLL loss; matching floating dtype; applicable graphs/storage survive parent collection.")
_native._set_doc(NiceDensityOutput, "Seven owning live flat coupling0..3/unscaled/latent[N,2] and scalar global logdet; matching floating dtype, applicable graphs/storage survive parent/store collection.")
_native._set_doc(FlowInverseOutput, "Two owning live flat value[N,2] and scalar inverse logdet; applicable input/parameter graphs and storage survive parent/store collection.")


_native._set_doc(WaveStackConfig, """Native architecture metadata: positive residual/skip widths,
levels1..12, cycles1..4 and classes>=2. Defaults32/64/6/2/256. Set the compact
source explicitly to4/8/5/1/256. Metadata is copied by WaveStackCache.""")
_native._set_doc(WaveStackField, """Native exact field2+cycles*(2**levels-1), including the initial
width2 ingress and every dilated gate. Positive residual/skip, levels1..12,
cycles1..4,classes>=2; invalid config RuntimeError. No fitting/store mutation.""")
_native._set_doc(WaveStackConv, """Native live explicitly left-padded causal convolution.

Matching floating encoded[N,T,C] and caller store, positive axes/output width,
kernel(default1),dilation(default1), nonempty name. Create/reuse name/W of shape
[1,kernel,C,out], Glorot uniform sqrt(6/(kernel*(C+out))), and name/b zero[out].
Owning live[N,T,out], same-time inclusive causal input. No activation, fitting,
optimizer or hidden no-grad. Invalid geometry/dtype/stored leaves RuntimeError.""")
_native._set_doc(WaveStackOutput, """Native owning live logits[N,T,classes], skip[N,T,S] and copied
vector of gates[N,T,R] for every block. Tensor handles retain graph/storage
after output/vector/store collection; no mutable synthesis state or fitting.""")
_native._set_doc(WaveStack, """Native differentiable two-tap gated causal stack.

Matching floating encoded[N,T,classes] and caller ParameterStore; positive
axes. Glorot kernels/zero biases create or reuse input, block1..L/gate+skip,
all-but-last residual, post and output leaves. Whole same-time code is visible;
logits predict the following code. Returns owning live logits[N,T,classes],
skip[N,T,S], and a copied vector of owning gates[N,T,R]. Each survives parent
collection; no fitting, optimizer, dropout, conditioning or normalization.
Config default32/64/6/2/256; source4/8/5/1/256,RF33,5068 values/34 leaves.
Invalid geometry/config/dtype/leaves raise RuntimeError. Warm before optimizer
or cache construction. Input is one-hot code data or continuous encoded probes.""")
_native._set_doc(WaveNll, """Native mean next-code cross entropy after an explicit warmup.

logits floating[N,T,classes], targets Int32/64[N,T] valid IDs, classes>=2
(default256), 0<=start<T(default127). Source generative run uses start32.
Owning live scalar mean over N*(T-start), earlier logit adjoints exactly zero.
No parameter mutation/fitting; invalid geometry/IDs/config RuntimeError.""")
_native._set_doc(WaveStackCache, """Native owning Float32 WaveStack snapshot and circular per-layer histories.

WaveStackCache(parameters,config) requires warmed finite Float32 CPU leaves of
matching architecture, copies required W/b values and ignores unrelated leaves.
Strides allowed. Store writes/collection cannot affect the cache. next(code)
consumes a code0..classes-1 and returns owning detached Float32[classes] next-code
logits, using rings of each block's OWN inputs and source scalar accumulation
order. Last block has no residual projection. No sampling, codec, RNG, fitting,
optimizer or autograd path. Distinct caches are independent; synchronize one
shared cache. Full convolution can differ by Float32 roundoff. reset() clears
histories and previous-code tap; clone() forks independent current state.
Invalid construction/code/nonfinite candidate raises RuntimeError; failed next
preserves all state. consumed() counts successful inputs. Explicit config,
no architecture inference; compact source4/8/5/1/256 has RF33 and5068 values.""")
_native._set_doc(WaveStackCache.__init__, "Native independent Float32 CPU snapshot of required warmed WaveStack leaves; explicit matching config, unrelated leaves ignored; invalid config/dtype/shape/nonfinite/missing leaf RuntimeError. No store writes or RNG. Initial consumed0.")
_native._set_doc(WaveStackCache.next, "Native consume one valid code and return owning detached Float32[classes] logits for the following code. Snapshot never trains/samples; Float32 scalar cache order. Invalid code/count limit or nonfinite candidate rejects without changing history. No autograd/store writes.")
_native._set_doc(WaveStackCache.reset, "Native clear every ring, previous-code tap and consumed count to true beginning; fixed snapshot and existing heads/clones unchanged. No allocation/RNG.")
_native._set_doc(WaveStackCache.consumed, "Native nonnegative successful input-code count since construction/reset; failed steps do not increment.")
_native._set_doc(WaveStackCache.clone, "Native owning independent fork of snapshot, all rings, previous code and consumed count. Future next/reset calls on either cache leave the other unchanged.")


def RgbAreaPixels(original, block_size=4):
    """Native stopped integer RGB area means and half-up byte quantization.

    Int32/64 CPU[N,H,W,3] values0..255; positive block_size(default4) divides
    H/W. Arbitrary strides. Returns owning codes Int32[N,H/B,W/B,3] and images
    Float32 of the same shape: double(code)/127.5-1 cast once. Source32->8 uses
    exact4x4 integer sums. No fitting, source mutation, RNG or graph. Invalid
    geometry/type/range or unsafe block sum raises RuntimeError.
    """
    return _native.RgbAreaPixels(original, block_size)


def DiscretizedLogisticRGB(image, parameters, mixtures=3):
    """Native stable channel-coupled discretized RGB logistic mixture.

    Matching finite Float32/64 CPU image[...,3] in[-1,1], parameters[...,10*K],
    same positive leading axes/strides, K=mixtures>0(default3). Component-major
    slots: logit,meanRGB,logscaleRGB,coeffRG/BR/BG. Tanh coefficients couple G/B
    means to observed R/G; scales lower-clip at-7 with live raw derivative at
    ties. Stable exact-bin identity with source Float32 half/full bin widths
    promoted even for Double. Endpoints x<=float(-.999),>=float(.999) use infinite
    tails. No CDF subtraction or small-mass fallback. Stop the mixture maximum,
    retain all applicable first/higher image/parameter paths. Eight owning flat
    fields: pixel_log_probability[P], component/mixture_log_probability[P,K],
    means/log_scales/coefficients/channel_log_mass[P,K,3], mean_channel_nll scalar
    nats/channel. No fitting, mutation, sampling or no-grad. Invalid inputs or
    nonfinite outputs raise RuntimeError; extreme finite parameters may reject.
    """
    return _native.DiscretizedLogisticRGB(image, parameters, mixtures)


def PixelCnnPP(parameters, image):
    """Native compact whole-pixel-causal three-block RGB logistic density.

    Matching finite Float32/64 CPU[N,H,W,3], positive rectangles/strides,
    values[-1,1]. Caller store owns normal(.1) kernels/zero biases. Seed8609
    reproduces1410 values/23 leaves,962 active/448 masked; width4 dual streams,
    three components and30 head parameters per pixel. Twelve owning live flat
    fields: parameters[N,H,W,30], vertical/horizontal[N,H,W,4], seven likelihood
    boundaries and per_image_nll[N]/loss scalar nats per RGB channel. Within-pixel
    coupling is in the likelihood; all30 context parameters exclude the whole
    current/future pixel. No labels, multiscale, dropout, optimizer, fitting,
    sampling or hidden no-grad. Applicable first/higher paths outlive store and
    parent. Validate input before leaves; invalid/nonfinite outputs RuntimeError.
    """
    return _native.PixelCnnPP(parameters, image)


def AutoregressiveDegreeMask(input_degrees, output_degrees, strict=False, dtype=None):
    """Native detached degree comparison mask for a dense autoregressive network.

    Nonempty nonnegative Int32/64 CPU[In]/[Out], mixed integer widths/strides
    allowed. Entry(i,j) is input_degree[i]<=output_degree[j], or strict<.
    Owning Float32(default)/64[In,Out] exact0/1; zero degrees allow constants.
    Does not validate a complete network/order. No store, RNG, fitting, mutation
    or retained input graph. Invalid vectors/dtype/product RuntimeError.
    """
    return _native.AutoregressiveDegreeMask(input_degrees, output_degrees, strict,
                                             DType.Float32 if dtype is None else dtype)


def MadeMasks(reverse, dtype=None):
    """Native source64-pixel MADE degrees and masks, six owning stopped fields.

    Input degrees1..64 or64..1; hidden1[i]=1+floor(i*62/15), hidden2=hidden1[5i%16].
    Int32 input_degrees[64]/hidden1_degrees[16]/hidden2_degrees[16]. Floating
    input_mask[64,16], hidden_mask[16,16] use<=; output_mask[16,64] strict<.
    Float32(default)/64 masks,1144 forbidden weights in the2400-value model.
    No store, RNG, fitting, mutation or global order state. Invalid dtype RuntimeError.
    """
    return _native.MadeMasks(reverse, DType.Float32 if dtype is None else dtype)


def MaskedDensityDense(parameters, name, input, mask):
    """Native live bias_add(input@(W*stopped_mask),b), with no activation.

    Finite matching floating CPU input[N,In]/mask[In,Out], positive axes/strides,
    exact0/1 mask, nonempty name. Missing name/W normal(.1), name/b0. Input/W/b
    first/higher paths live; structural mask stopped and masked partials zero.
    Owning[N,Out], no fitting/optimizer/sampling/existing-leaf writes/hidden
    no-grad; primitive alone does not establish a complete causal order.
    Inputs validate before leaves; stored-leaf/math errors may follow creation.
    Synchronize shared stores; invalid arguments/nonfinite output RuntimeError.
    """
    return _native.MaskedDensityDense(parameters, name, input, mask)


def MadeDensity(parameters, image, reverse=False):
    """Native fixed-order64-pixel MADE, source store seed8803,2400 values/6 leaves.

    Finite matching Float32/64 CPU[N,8,8,1],N>0, signed values/strides allowed.
    Flatten64, two16-unit sigmoid layers,64 logits; <=,<=,strict< degree masks.
    Missing normal(.1) W, b0;1144 masked/1256 active values. Three owning live
    hidden1/hidden2[N,16] and logits[N,8,8,1], image/weight first/higher paths.
    Reverse changes the mask order per call, not raw output layout. Source fits
    separate raster/reverse stores; shared-weight order training is a different
    protocol. No fitting/sampling/optimizer, existing-leaf writes, normalization,
    batch coupling, dropout or hidden no-grad. Inputs validate before leaves;
    stored-leaf/math errors may follow creation. Invalid calls RuntimeError.
    """
    return _native.MadeDensity(parameters, image, reverse)


def EqualDensityMixture(per_model_nll):
    """Native live Float64 equal-probability mixture of supplied joint densities.

    Finite floating CPU[N,K], positive axes/strides; signed NLLs allowed. Cast
    to Double; per row -log(sum(exp(-nll))/K), with stopped row maximum for
    stability. Three owning live per_image_nll[N]/mean_nll scalar/
    responsibilities[N,K], first/higher paths including Float32 input casts.
    Responsibilities=softmax(-nll), derivative of per-image mixture NLL.
    Supply normalized joint densities on the same base measure for that
    interpretation; numbers alone cannot establish normalization. Use image
    NLL sums, not mean pixel losses. This is not mean component NLL.
    No model evaluation/fitting/sampling/mutation/clipping/hidden no-grad.
    Invalid geometry/dtype/values or nonfinite outputs RuntimeError.
    """
    return _native.EqualDensityMixture(per_model_nll)


_native._set_doc(MadeMaskInventory, "Six owning stopped input_degrees/hidden1_degrees/hidden2_degrees/input_mask/hidden_mask/output_mask fields; Int32 degrees, floating structural masks, parent-independent storage.")
_native._set_doc(MadeDensityOutput, "Three owning live hidden1/hidden2/logits fields for the chosen MADE degree order; graph/storage survive parent/store collection, raw logits remain raster-shaped.")
_native._set_doc(DensityMixtureOutput, "Three owning live Float64 per_image_nll/mean_nll/responsibilities fields; equally weighted normalized joint densities, not mean component NLL, graph/storage survive parent collection.")


def PixelRowLstmStep(input_gates, previous_hidden, previous_cell, recurrent_kernel):
    """Native explicit width4 convolutional Row-LSTM transition with forget bias+1.

    Finite matching Float32/64 CPU input_gates[N,1,W,16], previous h/c[N,1,W,4],
    live kernel[1,3,4,16], positive N/W and arbitrary strides. Total gates=input
    projection+unbiased SAME1x3 conv(previous_hidden,kernel). Split i/f/o/candidate:
    sigmoid, sigmoid(+1), sigmoid, tanh. cell=f*previous_cell+i*candidate;
    hidden=o*tanh(cell). Seven owning live gates/input_gate/forget_gate/
    output_gate/candidate/cell/hidden fields, all applicable first/higher paths.
    Caller owns state sequencing and causal features. No store/parameter creation,
    mutation, fitting, optimizer, sampling, RNG or hidden no-grad. Invalid calls
    RuntimeError; nonfinite summed gates reject. Independent work shares no state.
    """
    return _native.PixelRowLstmStep(input_gates, previous_hidden, previous_cell, recurrent_kernel)


def PixelRnn(parameters, image):
    """Native compact one-layer Row-LSTM raster density, source store seed8303.

    Finite matching Float32/64 CPU[N,H,W,1], positive rectangles, signed values,
    strides allowed. Biased3x3 mask-A1->16 input projection; unbiased previous-row
    hidden1x3 kernel4->16; i/f/o/candidate gates, forget bias+1; biased1x1 head4->1.
    Initial h/c are the same first input row's first4 features times zero, keeping
    the source connected zero path. Columns advance in parallel, rows recur.
    Missing normal(.1) W, b0;357 values/5 leaves,277 active/80 masked coordinates.
    Seven owning live initial_hidden/initial_cell/input_gates/row_gates/hidden/
    cell/logits fields survive parent/store collection; image/weight first/higher
    paths stay live through all rows. No learned initial state, recurrent bias,
    class conditioning, batch coupling, normalization, dropout or pooling.
    Inputs validate before leaves; only missing leaves initialize. No existing-
    leaf writes, fitting/sampling/optimizer, later RNG or hidden no-grad.
    Work/storage O(NHW), graph depth O(H); synchronize shared mutable stores.
    Invalid arguments or nonfinite math RuntimeError. Source run uses8x8 inputs.
    """
    return _native.PixelRnn(parameters, image)


_native._set_doc(PixelRowLstmOutput, "Seven owning live gates/input_gate/forget_gate/output_gate/candidate/cell/hidden fields; width4 Row-LSTM transition, i/f/o/candidate ordering, forget bias+1, graph/storage survive parent collection.")
_native._set_doc(PixelRnnOutput, "Seven owning live initial_hidden/initial_cell/input_gates/row_gates/hidden/cell/logits fields; initial h/c alias the same connected first-feature*0 tensor, no learned initial states; graph/storage survive parent/store collection.")


def GatedPixelMask(input_channels, vertical, first, output_channels=8, dtype=None):
    """Native detached dual-stream raster mask, exact0/1, Float32(default)/64.

    Positive representable Cin/Cout. Vertical[3,3,Cin,Cout]: first top row only,
    later top+current row. Horizontal[1,3,Cin,Cout]: first left, later left+center.
    First block excludes all current raw channels; no RGB channel ordering.
    Owning CPU data, no model/global RNG/store writes. Invalid calls RuntimeError.
    """
    return _native.GatedPixelMask(input_channels, vertical, first, output_channels,
                                  DType.Float32 if dtype is None else dtype)


def GatedPixelGate(preactivation):
    """Native live tanh(first4)*sigmoid(last4) for finite floating CPU[N,H,W,8].

    Positive axes, strides allowed; owning[N,H,W,4], first/higher paths to both
    halves, no hidden no-grad, fitting, mutation or RNG. The gate alone does not
    enforce causality. Wrong geometry/dtype/nonfinite values raise RuntimeError.
    """
    return _native.GatedPixelGate(preactivation)


def GatedPixelConvolutional(parameters, image, labels=None, conditional=False,
                            input_channels=1, output_channels=1):
    """Native three-block width4 dual-stream gated raster context, source seed8501.

    Finite matching Float32/64 CPU[N,H,W,Cin], positive rectangles, strides/signed
    values allowed. Vertical3x3 and horizontal1x3 masked layers to8, paired gates
    to4, unbiased vertical4->horizontal8 links, biased4->4 projections; horizontal
    residuals in blocks1/2, biased1x1 head4->Cout. Normal(.1) W, zero biases.
    Conditional mode adds six learned10x8 tables; Int32/64[N] labels0..9 required.
    Unconditional mode accepts omitted labels or validates and ignores supplied
    ones. All current raw channels excluded, no within-pixel RGB density here.
    Gray binary1073/23 or1553/29 values/leaves,320 masked; RGB3/30 context1410/23,
    448 masked. Seven owning live vertical0/horizontal0/vertical1/horizontal1/
    vertical/horizontal/logits fields, applicable image/weight first/higher paths.
    Keep architecture mode fixed per store; pre-existing unused leaves remain.
    Inputs validate before leaves. No fitting/sampling/optimizer, normalization,
    dropout, pooling, batch coupling, existing-leaf writes or hidden no-grad.
    Synchronize shared stores; invalid arguments RuntimeError.
    """
    return _native.GatedPixelConvolutional(parameters, image, Tensor() if labels is None else labels,
                                          conditional, input_channels, output_channels)


def GatedPixelCnn(parameters, image, labels, conditional):
    """Native binary gated context with mandatory valid labels in both variants.

    GatedPixelConvolutional with Cin=Cout=1. Finite matching floating CPU[N,H,W,1],
    positive rectangles, mandatory Int32/64[N] digit labels0..9. False validates
    then ignores labels; True adds six10x8 tables. Source seed8501;1073/23 or
    1553/29 values/leaves,320 masked coordinates. Seven owning live stream/logit
    boundaries survive parent/store collection. Strict causal images, opened
    ordinary-CNN blind spot, no batch coupling. First/higher paths stay live.
    No likelihood/fitting/sampling/optimizer, existing-leaf writes or hidden
    no-grad. Inputs validate before leaves; invalid arguments RuntimeError.
    """
    return _native.GatedPixelCnn(parameters, image, labels, conditional)


_native._set_doc(GatedPixelOutput, "Seven owning live vertical0/horizontal0/vertical1/horizontal1/vertical/horizontal/logits fields; width4 stream maps, positive NHWC rectangles; graph/storage survive parent and store collection.")


def BinaryPixels(images):
    """Native deterministic28->8 binary preparation from Float32[N,28,28,1].

    Finite positive CPU batch, normalized[0,1], strides allowed. Detached Double
    half-pixel bilinear arithmetic, one Float32 cast then gray>=.5 threshold.
    Owning stopped gray/images[N,8,8,1], no fitting, random binarization or mutation.
    Invalid geometry/range/dtype RuntimeError; independent calls share no state.
    """
    return _native.BinaryPixels(images)


def CausalPixelMask(input_channels, output_channels, type_b, dtype=None):
    """Native detached3x3 raster mask[3,3,Cin,Cout], positive representable widths.

    Float32(default)/64 exact0/1; A allows top row and middle-left, B also center.
    No RGB channel ordering; use A for raw gray pixels, B for causal features.
    No RNG/store writes. Invalid widths/dtype RuntimeError.
    """
    return _native.CausalPixelMask(input_channels, output_channels, type_b, DType.Float32 if dtype is None else dtype)


def MaskedPixelConv(parameters, name, input, mask):
    """Native biased odd rectangular SAME conv(input,W*stopped_mask)+b, stride1.

    Finite matching floating CPU[N,H,W,Cin], mask[Kh,Kw,Cin,Cout] exact0/1, positive odd Kh/Kw.
    Nonempty name, normal(.1) missing W, zero b; validates inputs before leaves.
    Input/W/b first/higher paths live; masked partials zero, mask structural.
    Owning[N,H,W,Cout], no activation/fitting/optimizer/existing-leaf writes.
    Causality depends on mask/input features. Invalid arguments RuntimeError.
    """
    return _native.MaskedPixelConv(parameters, name, input, mask)


def PixelCnn(parameters, input):
    """Native compact causal four-ReLU PixelCNN, source store seed8401.

    Finite matching Float32/64 CPU[N,H,W,1], positive rectangles, strides allowed.
    Four biased SAME3x3 width8 layers: mask A then three B; unmasked1x1 head.
    Missing W normal(.1), b0;1841 values/10 leaves,808 masked coordinates.
    Six owning live fields input_features/head/hidden0/hidden1/hidden/logits.
    Strict raster causality, ordinary masked-CNN blind spot, no batch coupling.
    Input/weight first/higher paths live. No loss/fitting/sampling/optimizer,
    existing-leaf writes or hidden no-grad. Invalid arguments RuntimeError.
    """
    return _native.PixelCnn(parameters, input)


def BernoulliPixels(logits, binary):
    """Native stable teacher-forced Bernoulli loss, exact binary targets.

    Finite matching floating CPU[N,H,W,1], positive axes, strides allowed.
    Four owning live probability/pixel_nll/per_image_nll/loss fields: sigmoid,
    softplus(z)-x*z, HWC sum in nats/image, overall mean in nats/binary pixel.
    Both applicable input paths live, including higher derivatives. No fitting,
    clipping, bit conversion, mutation or hidden no-grad. Invalid calls RuntimeError.
    """
    return _native.BernoulliPixels(logits, binary)


def BinaryPixelUniforms(rows, seed):
    """Native owning detached Double[N,64] row-then-raster midpoint MT bank.

    Positive representable rows, uint32 seed; (raw+.5)/2**32 strictly in(0,1).
    Draw the full bank even for prefix completion; no hidden model/sampling loop,
    global RNG or store mutation. Source sample8423, completion8443.
    Invalid rows RuntimeError. Independent calls have local streams.
    """
    return _native.BinaryPixelUniforms(rows, seed)


_native._set_doc(BinaryPixelData, "Owning detached Float32 gray/images[N,8,8,1]; Double resize, Float32 threshold.")
_native._set_doc(PixelCnnOutput, "Six owning live input_features/head/hidden0/hidden1/hidden/logits boundaries; graph/storage survive parent collection.")
_native._set_doc(BernoulliPixelOutput, "Four owning live probability/pixel_nll/per_image_nll/loss fields, natural-log units; graph/storage survive parent collection.")


def GaussianFrechet(real, fake):
    """Native detached Float64 low-rank Gaussian Frechet feature distance.

    Finite floating CPU real[N,F],fake[M,F], N,M>=2,F>0; unequal rows/dtypes and
    strides allowed. Center each pool and divide sqrt(its rows-1), then thin SVD
    of[N,M] cross product. Seven owning detached fields: fid,raw_fid,mean_term,
    trace_a,trace_b,cross_trace,cross_singular_values[min(N,M)]. Raw distance is
    squared mean difference+both covariance traces-2*cross trace; raw<-1e-7
    rejects, tiny negative round-off clamps zero. Fixed absolute source tolerance.
    No fitting, sampling, extractor, mutation or gradient path. FID naming assumes
    the actual declared pretrained Inception protocol; arbitrary features give a
    Gaussian moment distance. Invalid calls RuntimeError; independent local work.
    """
    return _native.GaussianFrechet(real,fake)


def InceptionScore(probability):
    """Native detached Float64 exp(mean conditional-to-marginal class KL).

    Finite floating CPU[N,C],N>0,C>1, values in[0,1], row sums within3e-6 of1;
    arbitrary strides allowed. Does not renormalize. Cast stopped inputs to Double,
    marginal=mean rows; both log arguments floor1e-300, zero classes contribute0.
    Owning scalar, no network, fitting, sampling, mutation or gradient path.
    Balanced confident C classes score C; uncertain uniform and collapsed score1.
    No accuracy, uniqueness, recall or protection against copying is implied.
    Invalid arguments or nonfinite result RuntimeError.
    """
    return _native.InceptionScore(probability)


def InceptionScoreSplits(probability, splits=4):
    """Native detached source-order contiguous split scores and population spread.

    InceptionScore probability contract; positive splits<=rows, default4.
    Boundaries floor(i*N/splits), with every row used once, no shuffle/drop/RNG.
    Three owning Float64 fields: scores[splits],mean,population_std (divisor
    splits, not splits-1). Order-dependent spread from one pool is not a confidence
    interval over independently fitted models. No extractor/fitting/mutation or
    gradient path. Invalid arguments RuntimeError.
    """
    return _native.InceptionScoreSplits(probability,splits)


_native._set_doc(GaussianFrechetOutput, "Seven owning detached Float64 fid/raw_fid/mean_term/trace_a/trace_b/cross_trace/cross_singular_values fields; low-rank unbiased feature-moment comparison.")
_native._set_doc(InceptionSplitScores, "Three owning detached Float64 scores/mean/population_std fields; contiguous source-order partitions, population divisor splits.")


_native_frozen_graph_init = FrozenGraph.__init__


def _frozen_graph_init(self, model_bytes, max_bytes=512 * 1024 * 1024):
    """Adapt the existing SWIG ByteVector proxy to its sequence constructor typemap."""
    if isinstance(model_bytes, _native.ByteVector):
        model_bytes = list(model_bytes)
    _native_frozen_graph_init(self, model_bytes, max_bytes)


FrozenGraph.__init__ = _frozen_graph_init
_native._set_doc(FrozenGraph, """Native immutable numeric TensorFlow GraphDef, without TensorFlow runtime.

FrozenGraph(model_bytes, max_bytes=512MiB) accepts bytes, an integer sequence
or ByteVector; FrozenGraph.load(path,
max_bytes=512MiB). The positive limit bounds serialized bytes and total decoded
constants. Numeric Float32/64,Int32/64,Bool preserve exact bits/scalar rank;
malformed wire fields, names, shapes, payloads, duplicate/missing nodes and
nonzero serialized input ports reject. Unsupported constant types/operations
may remain outside fetched ancestry. No variables, random ops, JPEG decoding,
function calls, downloading, training or model-identity assumptions.

node_names() returns serialized-order names; operation(name) returns its kind;
constant(name) returns an independent detached mutable constant copy. evaluate
takes a sequence of (name,Tensor) feeds and fetch-name strings, returning named
owning detached Tensor copies in requested order, including duplicate fetches.
Feed overrides cut that node's entire ancestry. Numeric native CPU feeds, declared
dtypes checked; no implicit pretrained preprocessing. Control edges execute,
CheckNumerics rejects nonfinite input, cycles/depth>1024 reject, intermediates
release after last use. Concurrent calls share immutable parsed constants only;
callers must synchronize mutation of their own feed storage.

Supported: Const,Placeholder,Identity,StopGradient,CheckNumerics,Relu,Add/AddV2,
Sub,Mul,float Div/RealDiv,BiasAdd,MatMul,Reshape,Squeeze,ExpandDims,Concat/
ConcatV2,Softmax,Float32 ResizeBilinear,NHWC Float32/64 Conv2D/AvgPool/MaxPool
and legacy BatchNormWithGlobalNormalization. SAME/VALID,spatial strides/dilation,
transpose flags,old BN epsilon/gamma and resize coordinate flags preserved.
Invalid format/arguments/runtime restrictions raise RuntimeError.
""")
_native._set_doc(FrozenGraph.__init__, "Native owned GraphDef bytes (bytes, integer sequence or ByteVector); positive max_bytes defaults512MiB and bounds serialized bytes and decoded numeric constants. No TensorFlow runtime; malformed wire/names/shapes and unsupported input ports reject.")
_native._set_doc(FrozenGraph.load, "Native immutable GraphDef from a regular local file, positive max_bytes default512MiB. No downloads; read/format errors RuntimeError. Owns model independently of file lifetime.")
_native._set_doc(FrozenGraph.node_names, "Native owned list of all node names in serialized order, including unused unsupported operations; no execution or mutable model metadata exposure.")
_native._set_doc(FrozenGraph.operation, "Native copied operation name for an existing node or node:0. Unknown names, controls and other ports raise RuntimeError; no execution.")
_native._set_doc(FrozenGraph.constant, "Native independent owning detached copy of an exact numeric Const, preserving dtype/shape/bits. Internal model storage cannot be mutated. Non-Const, unsupported dtype or unknown endpoint raises RuntimeError.")
_native._set_doc(FrozenGraph.evaluate, "Native read-only supported numeric subgraph execution. Sequence(name,Tensor) CPU feeds override nodes and cut ancestry; fetch strings retain order/duplicates in named owning detached copies. Control/CheckNumerics execute, input higher paths stop, last-use storage releases. See FrozenGraph for supported ops/dtypes/bounds; errors RuntimeError.")


def SingleImageNoise(rows, height, width, shared_rgb, seed, dtype=_native.DType_Float32):
    """Native detached Gaussian RGB bank[N,H,W,3] from one local MT stream.

    Positive representable axes; Float32(default) or Float64. Midpoint raw-MT
    Box-Muller in NHWC order: shared_rgb draws one channel then tiles RGB;
    otherwise one stream draws all three channels. Cast once, no global RNG,
    fitting, state writes or retained graph. Invalid calls raise RuntimeError.
    """
    return _native.SingleImageNoise(rows,height,width,shared_rgb,seed,dtype)


def SingleImageAmplitude(real, previous_reconstruction, factor=.1):
    """Native detached factor times Double RMS error, cast once to input dtype.

    Finite signed matching Float32/64 CPU[N,H,W,3], positive axes, strides allowed.
    Sum squared Double differences then divide numel, sqrt, multiply factor;
    owned[1] result stops both image paths. Factor finite nonnegative(default .1).
    Caller separately sets coarse amplitude one. No fitting/sampling/state writes.
    Invalid inputs or nonfinite computation raise RuntimeError.
    """
    return _native.SingleImageAmplitude(real,previous_reconstruction,factor)


def SingleImageGenerate(generator, previous, noise, amplitude):
    """Native residual RGB generator from previous plus explicit scaled noise.

    Matching finite signed Float32/64 CPU[N,H,W,3], positive rectangles/strides;
    matching amplitude scalar or[1], signed/zero allowed. Biased SAME3x3
    fan-in-uniform3->4->4->3, leaky(.2) twice then tanh, image=previous+residual
    without clipping. Seed7307 gives371 values/6 head/body/output W/b leaves.
    Five owning live input/head/body/residual/image fields retain applicable
    first/higher previous/noise/amplitude/weight paths after parent/store collection.
    No normalization, pooling, dropout, target image, sampling, optimizer, hidden
    no-grad or existing-leaf writes; only missing leaves initialize. Supplied
    invalid inputs fail before leaves, RuntimeError; synchronize shared stores.
    """
    return _native.SingleImageGenerate(generator,previous,noise,amplitude)


def SingleImageDiscriminate(discriminator, image):
    """Native image-only raw local scores[N,H,W,1], nominal7x7 receptive field.

    Matching finite signed Float32/64 CPU RGB[N,H,W,3], positive rectangles,
    strides allowed. Biased fan-in-uniform SAME3x3 3->4->4->1, two leaky(.2);
    source seed7309 gives297 values/6 head/body/output W/b leaves. No pooling,
    sigmoid, normalization, fitting, optimizer, hidden no-grad or existing-leaf
    writes. Owning output retains applicable first/higher image/weight paths.
    Supplied invalid inputs fail before leaves, RuntimeError; shared stores need sync.
    """
    return _native.SingleImageDiscriminate(discriminator,image)


def SingleImageGan(generator, discriminator, real, previous, noise, rec_previous, rec_noise, amplitude):
    """Native explicit compact single-scale game with shared fixed-reconstruction G.

    Distinct matching Float32/64 G/D stores, seeds7307/7309,371/297 values in6/6
    leaves. Finite signed positive RGB CPU rectangles, common H/W; previous/noise
    match, rec_previous/rec_noise exactly match real. Random and real pool rows
    may differ; each score pool uses its own mean. Matching scalar or[1] amplitude
    shared by both G calls. D=.5*(mean((Dreal-1)^2)+mean(Dfake^2));
    G=mean((Dfake-1)^2)+10*mean((reconstruction-real)^2), without a G half.
    Eight owning live flat fields expose fake,reconstruction,real_score,fake_score,
    d_loss,g_loss,adversarial_loss,reconstruction_loss. Both G branches and
    applicable first/higher input/weight paths remain live. No gradient penalty,
    BN, target-conditioned D, clipping, sampling, optimizer or hidden detach/writes.
    Caller freezes lower scales, measures fixed amplitudes, warm-starts BOTH
    groups with cold Adam, then takes D and a fresh G capture. Validate supplied
    inputs/aliases before leaves; errors RuntimeError, synchronize shared stores.
    Source teaching variant of https://arxiv.org/abs/1905.01164 uses LSGAN.
    """
    return _native.SingleImageGan(generator,discriminator,real,previous,noise,rec_previous,rec_noise,amplitude)


_native._set_doc(SingleImageGenerated, "Five owning live input/head/body/residual/image fields; previous plus tanh residual, applicable higher paths survive parent/store collection.")
_native._set_doc(SingleImageGanOutput, "Eight owning live random/reconstruction/patch-score/loss fields; full G LSGAN+10MSE, half D LSGAN, no hidden writes or detach.")


def RelativisticAverageGan(real, fake):
    """Native live relativistic-average logistic losses, including unequal pools.

    Finite matching Float32/64 CPU real[N,1],fake[M,1], N,M>0. rr=real-mean(fake),
    rf=fake-mean(real), BOTH means live. D=.5*(mean softplus(-rr)+mean softplus(rf));
    G=.5*(mean softplus(rr)+mean softplus(-rf)). Four owning live fields retain
    higher paths through both pools; common shifts cancel up to rounding. No
    clipping, fitting, mutation or hidden detach. Invalid arguments RuntimeError.
    """
    return _native.RelativisticAverageGan(real,fake)


def AdversarialSRGenerate(generator, stats, lr, bicubic, enhanced=False, batch_statistics=False):
    """Native compact x2 SRGAN/ESRGAN with explicit bicubic shortcut and BN mode.

    Distinct matching Float32/64 G/stats stores; signed finite CPU lr[N,16,16,3],
    bicubic[N,32,32,3], positive aligned rows. Gseed7001/stats0. SRGAN1591/25
    values/leaves,40/10 saved BN: two BN/PReLU residual blocks, CRD shuffle2.
    ESRGAN3511/40: three5-conv growth2 dense blocks,.2 residuals,nearest2,
    up/HR leaky convs,no BN. Both output kernels normal.001 and add an unbounded
    residual to bicubic. Other kernels source float fan-in uniform. Nine owning
    live fields expose head/body/upsampled/packed/shuffled/image/residual.
    SRGAN batch_statistics=True returns detached moments_mean and
    moments_unbiased_variance[5,4] in order residual0/bn1,bn2,residual1/bn1,bn2,
    body_bn; otherwise undefined, as for ESRGAN. Saved statistics stop at every
    derivative order, applicable LR/bicubic/weight higher paths live. No hidden
    fitting, updates, EMA, no-grad or later RNG. Invalid arguments RuntimeError.
    """
    return _native.AdversarialSRGenerate(generator,stats,lr,bicubic,enhanced,batch_statistics)


def AdversarialSRDiscriminate(critic, image, relative=False):
    """Native compact raw RGB critic: finite signed matching Float32/64 CPU
    image[N,32,32,3], positive rows. Fan-in-uniform head3->4, stride2 4->8->8,
    leaky.2,global mean,score1. Ordinary native Glorot dense+bias1673/8;
    relative=True bias-free source sqrt(6.f/9) kernel1672/7, seed7013. Owns
    live[N,1], applicable higher paths; no BN, sigmoid, fitting or hidden writes.
    Invalid arguments RuntimeError.
    """
    return _native.AdversarialSRDiscriminate(critic,image,relative)


def AdversarialSRFeatures(rgb, weight, bias):
    """Native frozen VGG19 first-convolution features, without ReLU or scaling.

    Finite matching Float32/64 CPU rgb[N,H,W,3], positive axes; weight[3,3,3,64],
    bias[64],1792 supplied values. RGB*255->BGR minus source float-rounded
    ImageNet means, SAME conv+bias. Always stops all weight/bias paths; RGB
    higher paths remain live. Owning[N,H,W,64] output, no fitting, mutation or
    substitute random extractor. Caller verifies actual pretrained provenance.
    Invalid geometry/dtypes/products/nonfinite computation raises RuntimeError.
    """
    return _native.AdversarialSRFeatures(rgb,weight,bias)


def AdversarialSR(generator, critic, stats, lr, target, bicubic, feature_weight, feature_bias, enhanced=False, adversarial=False):
    """Native explicit two-path SRGAN/ESRGAN objectives and detached BN captures.

    Three distinct matching Float32/64 stores; finite signed CPU lr[N,16,16,3],
    target/bicubic[N,32,32,3], positive aligned rows, frozen1792 feature values.
    SRGAN G1591/25,D1673/8,BN40/10; ESRGAN G3511/40,D1672/7,no BN.
    Always evaluate saved-statistic and batch-statistic G; losses use training G.
    SRGAN: pixelMSE,post-ReLU featureMSE,ordinary logistic GAN. ESRGAN: pixelL1,
    pre-activation featureL1,half-scaled relativistic means. Features divide255
    BEFORE loss, target extracted features stop. gan_loss=content+float-rounded
    .001/.005*adversarial, plus ESRGAN float-rounded .01*pixel. g_loss is
    (1-phase)*pixel+phase*gan_loss, phase=adversarial, all components computed.
    Twenty owning flat fields retain higher paths/storage; optional detached BN
    moments[5,4], row order documented in AdversarialSRGenerate. Caller owns
    pixel-only warm-up, cold reset of BOTH Adam states, D then fresh G steps;
    AFTER each G weight write commit BN candidates from that SAME pre-write
    capture, momentum double(float32(.9)). Exclude BN/frozen values from Adam;
    no generator weight EMA, hidden fitting/captures/writes/no-grad/later RNG.
    Validate all supplied inputs/aliases before leaves. Errors RuntimeError.
    Compact teaching variants: https://arxiv.org/abs/1609.04802 and /abs/1809.00219.
    """
    return _native.AdversarialSR(generator,critic,stats,lr,target,bicubic,feature_weight,feature_bias,enhanced,adversarial)


_native._set_doc(RelativisticGanLoss, "Four owning live discriminator,generator,real_relative,fake_relative fields, including both opposite-pool mean paths.")
_native._set_doc(AdversarialSRGenerated, "Nine owning image,residual,packed,shuffled,moments_mean,moments_unbiased_variance,head,body,upsampled fields; BN captures detached and optional.")
_native._set_doc(AdversarialSROutput, "Twenty owning two-path pixel/score/feature/objective fields and optional detached five-layer BN captures; no hidden optimizer or state write.")


def UniformCategorical(rows, seed, classes=3):
    """Native explicit local-MT categorical draw, with accepted raw-word audit.

    Positive rows and classes1..INT32_MAX. Accept word<floor(2^32/classes)*classes,
    then word%classes; default3 rejects4294967295. Returns owning detached
    Int32 labels[N],Float64 raw[N] exactly representing accepted uint32 words;
    rejected words omitted, frequencies not forced balanced. No global RNG,
    fitting or mutable state. Invalid arguments raise RuntimeError.
    """
    return _native.UniformCategorical(rows,seed,classes)


def StarDomainImages(original, domain):
    """Native per-row square grayscale teaching domains:0 unchanged,1 inversion,
    2 rotation out(y,x)=original(H-1-x,y), without intensity inversion. Finite
    signed Float32/64 CPU[N,H,H,1], positive axes; Int32/64 domain[N] in0..2.
    No clipping, fitting or RNG. Owning live input-shape/dtype output retains
    applicable higher paths and storage. Invalid arguments raise RuntimeError.
    """
    return _native.StarDomainImages(original,domain)


def SplitStarDomains(images, labels, source_ids, per_class=30):
    """Native disjoint observed-order three-domain split with explicit unused IDs.

    Unit finite Float32/64 CPU images[N,28,28,1], N>0 divisible10; Int32/64
    labels[N] exactly balanced0..9, unique nonnegative Int32/64 source_ids[N].
    per_class>0<=N//30: first per_class occurrences per digit domain0, next1,
    next2, remaining IDs unused, global order preserved. Default30 gives900/1000;
    explicit6 gives180/200. Flat owning image,original,domain,digit,source,unused;
    domain/digit Int32, IDs preserve dtype, applicable image paths live. No RNG,
    pairing, fitting or cross-pool/provenance certification; LoadSaliency checks
    original data provenance separately. Invalid arguments raise RuntimeError.
    """
    return _native.SplitStarDomains(images,labels,source_ids,per_class)


def StarGanGenerate(generator, source, target):
    """Native source+requested-domain U-Net with both real concatenation skips.

    Finite signed matching Float32/64 CPU source[N,28,28,1], N>0; Int32/64
    target[N] in0..2 checked before one-hot/leaves. Concat source,tiled one-hot3,
    normal.02 biased stride2 conv4->4->8/ReLU, learned up8->4,concat encoder,
    learned up8->2,concat full conditioned input,3x3 sigmoid6->1. Source seed6301
    creates1609 values/10 leaves. Nine owning live fields expose code/input/skips/
    image. No original-domain input, paired target image, noise, normalization,
    fitting or hidden writes; applicable higher paths live. Errors RuntimeError.
    """
    return _native.StarGanGenerate(generator,source,target)


def StarGanDiscriminate(critic, image):
    """Native image-only source/domain critic with a shared leaky.2 trunk.

    Finite signed matching Float32/64 CPU[N,28,28,1], N>0; biased normal.02
    stride2 conv1->4->8. Pooled8->3 Glorot domain head initializes BEFORE3x3
    normal source head8->1, seed6311:688 values/8 leaves. Five owning live fields
    source[N,7,7,1],domain_logits/probability[N,3],down14/down7; no requested
    domain, noise, fitting or hidden writes. Higher paths live. Errors RuntimeError.
    """
    return _native.StarGanDiscriminate(critic,image)


def StarDomainLoss(logits, target):
    """Native stable mean three-domain cross entropy, source-grouped one-hot times
    log_softmax. Finite Float32/64 CPU logits[N,3], positive rows; aligned Int32/64
    target[N] in0..2 checked before one-hot. No clipping, mutation or fitting;
    owning scalar retains applicable higher logits paths. Errors RuntimeError.
    """
    return _native.StarDomainLoss(logits,target)


def StarGan(generator, critic, source, original, target):
    """Native single-generator three-domain game with explicit shared-G round trip.

    Distinct matching Float32/64 stores; signed finite CPU source[N,28,28,1],
    N>0, aligned Int32/64 original/target[N] in0..2. G1609/10,D688/8 values/leaves,
    seeds6301/6311. fake=G(source,target);reconstruction=G(fake,original), SAME G.
    G=(full fake-to1 source mean+fake-domain CE)+10*roundtrip meanL1;
    D=half(real-to1+fake-to0 source means)+real-domain CE, no fake CE in D loss.
    Fifteen owning flat live fields retain graph/storage and higher paths through
    both G calls/current D. Caller captures/steps D then fresh G with same batch,
    steps G only. No hidden detach, optimizer, fitting, no-grad, batch state,
    noise or later RNG. Invalid calls fail before leaves; errors RuntimeError.
    Compact teaching variant: https://arxiv.org/abs/1711.09020.
    """
    return _native.StarGan(generator,critic,source,original,target)


_native._set_doc(UniformCategoricalDraw, "Owning detached Int32 labels and exact accepted Float64 raw words; no retained engine.")
_native._set_doc(StarDomains, "Owning image/original/domain/digit/source/unused fields, disjoint observed-order selected IDs and explicit unused IDs; live image paths.")
_native._set_doc(StarGanGenerated, "Nine owning code/input/down14/down7/up14/joined14/up28/joined28/image fields; both actual skips and higher paths.")
_native._set_doc(StarGanDiscriminated, "Five owning live source/domain_logits/domain_probability/down14/down7 fields; image-only shared trunk.")
_native._set_doc(StarGanOutput, "Fifteen owning live fake/reconstruction, real/fake source/domain fields and explicit losses; shared-G round trip, no hidden writes.")


def modulated_conv2d(image, weight, style, demodulate=True, epsilon=None):
    """Native per-example same-padded NHWC modulated convolution.

    Finite matching Float32/64 CPU image[N,H,W,C], odd square weight[K,K,C,O],
    style[N,C]; positive axes, signed/zero gains permitted. Effective taps are
    (weight*style)*rsqrt(sum(weight*style squared)+epsilon); sum over K,K,C,
    row-major tap accumulation. demodulate=False returns unit factors using the
    source live zero path. epsilon positive finite representable; native default
    is double(float32(1e-8)). Owns live image[N,H,W,O],demodulator[N,O] with first
    and higher input/weight/style paths. No fitting, mutation or hidden state.
    Invalid inputs, products and nonfinite arithmetic raise RuntimeError.
    """
    # Omit epsilon to retain the canonical source-rounded default.
    if epsilon is None:
        return _native.modulated_conv2d(image, weight, style, demodulate)
    return _native.modulated_conv2d(image, weight, style, demodulate, epsilon)


def spatial_instance_normalize(input, epsilon=None):
    """Native live per-row/per-channel population spatial normalization.

    Finite Float32/64 CPU input[N,H,W,C], positive axes. Center over H,W, divide
    by sqrt(population variance+epsilon), no affine/state; constant channels zero.
    Default epsilon double(float32(1e-8)); explicit epsilon positive finite and
    representable. Owns input-shape/dtype output with first/higher paths; inputs
    unchanged. Invalid geometry/guard/nonfinite arithmetic raises RuntimeError.
    """
    if epsilon is None:
        return _native.spatial_instance_normalize(input)
    return _native.spatial_instance_normalize(input, epsilon)


def StyleGanMap(generator, z):
    """Native latent-only mapping: matching finite CPU[N,8], pixel RMS then two
    dense8/leaky.2 layers,144 values/4 leaves. Float32/64; source seed6607.
    Call before synthesis for source initialization order. Owns live W[N,8];
    no truncation, EMA, noise, fitting or existing-leaf writes. Errors RuntimeError.
    """
    return _native.StyleGanMap(generator, z)


def StyleGanSynthesize(generator, w0, w1, noise4, noise8, version2=False, first_style_override=None):
    """Native compact8x8 synthesis from supplied per-layer W and spatial noise.

    Finite matching Float32/64 CPU w0,w1[N,8], noise4[N,4,4,1],noise8[N,8,8,1].
    Learned4x4x4 constant,styled3x3 layer4,nearest up2,styled3x3 layer8,RGB sigmoid.
    AdaIN(default): conv+noise*zero-initialized strength+bias,leaky.2,population
    instance norm then learned scale/shift. version2 uses per-example modulated/
    demodulated weights, no AdaIN; RGB modulation has no demodulation and uses w1.
    Optional first_style_override[N,4] replaces actual first gains only for v2;
    affine leaves still initialize. Synthesis517/13 or481/15 values/leaves; mapping
    adds144/4 for661/17 or625/19. Nine owning live fields retain graphs/storage.
    No hidden noise RNG, updates, fitting, EMA or schedule. Invalid args RuntimeError.
    """
    if first_style_override is None:
        return _native.StyleGanSynthesize(generator,w0,w1,noise4,noise8,version2)
    return _native.StyleGanSynthesize(generator,w0,w1,noise4,noise8,version2,first_style_override)


def StyleGanDiscriminate(critic, image):
    """Native compact raw8x8 critic: finite matching Float32/64 CPU[N,8,8,1],
    positive rows. Biased normal conv4x4/stride2 1->4/leaky.2,3x3 4->8/leaky.2,
    flatten128,dense raw1;493 values/6 leaves, source seed6619. Owns live[N,1]
    with first/higher paths; no state, sigmoid, fitting or update. Errors RuntimeError.
    """
    return _native.StyleGanDiscriminate(critic,image)


def StyleGan(generator, critic, real, z, noise4, noise8, version2=False):
    """Native two-store AdaIN/modulated least-squares game, with explicit noise.

    Distinct matching Float32/64 stores; finite signed CPU real[N,8,8,1],z[M,8]
    and aligned noise4[M,4,4,1],noise8[M,8,8,1], N,M>0, unequal real rows allowed.
    Generator661/17(default AdaIN) or625/19(v2 modulation); critic493/6.
    D=.5*(mean((real_score-1)^2)+mean(fake_score^2)); G=mean((fake_score-1)^2),
    NO half on G. Owns ten live fields, first/higher paths. Caller captures/steps
    D then computes a fresh G forward using the SAME latent/noise batch and steps
    G. No hidden optimizer writes, RNG, fitting, truncation, EMA or batch state.
    Invalid arguments raise RuntimeError. Compact teaching variants, not full
    paper architectures: https://arxiv.org/abs/1812.04948 and /abs/1912.04958.
    """
    return _native.StyleGan(generator,critic,real,z,noise4,noise8,version2)


_native._set_doc(ModulatedConv2DOutput, "Owning live image and per-example/output-channel demodulator; graph/storage survive parent collection.")
_native._set_doc(StyleGanSynthesis, "Nine owning live image,constant,first,upsampled,second,per-layer styles/demodulators; supplied noise only.")
_native._set_doc(StyleGanOutput, "Ten owning live fake,mapped,first/second styles/demodulators,real/fake scores,d_loss,g_loss; no hidden update.")


def RotateInvertGrayscale(image, inverse=False):
    """Native live grayscale90-degree rotation and intensity inversion.

    Finite signed Float32/64 CPU[N,H,W,1], positive axes, strides allowed.
    Default out(y,x)=1-in(H-1-x,y); inverse orientation is1-in(x,W-1-y).
    Owns [N,W,H,1] output/graph; no clipping, RNG, fitting or mutation. Floating
    cancellation means inverse round trips need not recover original bits.
    Invalid inputs raise RuntimeError; keep original_B pixels for exact diagnostics.
    """
    return _native.RotateInvertGrayscale(image, inverse)


def SplitCycleDomains(images, labels, source_ids):
    """Native disjoint first-half-per-class unpaired grayscale teaching domains.

    Finite unit Float32/64 CPU images[N,28,28,1], N>=20 divisible by20.
    Integer Int32/64 labels[N] have exactly N/10 rows per class0..9; source_ids[N]
    are nonnegative/unique within the pool. Take first N/20 observed rows of each
    class for A and the rest for original_B, preserving global order; B is native
    rotation/inversion of original_B. Owns a,b,ids_a,ids_b,original_b fields with
    image gather paths live and ID dtype preserved. No pairing, random split,
    fitting, cross-pool check or provenance certification; use LoadSaliency first.
    Labels/known maps are not model inputs. Invalid arguments raise RuntimeError.
    """
    return _native.SplitCycleDomains(images, labels, source_ids)


def CycleGanDiscriminate(critic, image):
    """Native unconditional49-patch critic, without condition concatenation.

    Finite signed matching Float32/64 CPU[N,28,28,1], positive rows. Biased
    normal.02 SAME4x4/stride2 conv1->4/leaky.2,4->8/leaky.2,then3x3 raw8->1.
    Source seeds6217/6221 create661 values/6 leaves. Owns live[N,7,7,1] raw
    scores with first/higher input/weight paths. No sigmoid, batch state, fitting,
    optimizer writes or hidden no-grad. Invalid inputs/stores raise RuntimeError.
    """
    return _native.CycleGanDiscriminate(critic, image)


def CycleGan(a_to_b, b_to_a, critic_a, critic_b, a, b, cycle_weight=10, identity_weight=5):
    """Native four-store unpaired game with one common coupled generator objective.

    Four distinct matching Float32/64 stores and finite signed CPU a[N,28,28,1],
    b[M,28,28,1], N,M>0; unequal independent counts allowed. Both generators
    reuse the actual two-skip U-Net1390/10 leaves; critics661/6 each;4102/32 total.
    Full G adversarial means plus cycle_weight(default10) times the two cycle-L1
    means and identity_weight(default5) times the two identity-L1 means, preserving
    source grouping. Each D loss is half the sum of real-to1/fake-to0 patch means.
    Nonnegative finite representable weights; all diagnostics evaluate at zero.
    Owns19 flat live fields; all applicable first/higher paths retain graphs/storage.
    Capture D_a/step D_a,D_b/step D_b, then fresh ONE G forward, capture BOTH G
    vectors BEFORE either G write. No hidden capture, optimizer, fitting, no-grad,
    pairing, labels, known transform, image buffer or mutable batch state. Invalid
    inputs/stores/weights raise RuntimeError. See Zhu et al.(2017),
    https://arxiv.org/abs/1703.10593, compact U-Net teaching variant.
    """
    return _native.CycleGan(a_to_b, b_to_a, critic_a, critic_b, a, b, cycle_weight, identity_weight)


_native._set_doc(CycleDomains, "Owning live a,b,original_b image fields and original ids_a/ids_b, from disjoint per-class halves; no pairing or fitted state.")
_native._set_doc(CycleGanOutput, "Owning19 live translation/cycle/identity,raw patch-score and objective fields; one shared coupled G loss, no capture/update/schedule hidden.")


def quantized_sobel_edges(image, threshold=256):
    """Native detached binary Sobel conditions from finite unit grayscale CPU pixels.

    Float32/64 [N,H,W,1], positive axes, strides allowed. Recover 8-bit integers
    with floor(Double(pixel)*255+.5), raw 3x3 zero-padded Gx/Gy, then inclusive
    abs(Gx)+abs(Gy)>=threshold (nonnegative integer, default256). Processes
    borders against zeros. Owns binary input-shape/dtype output, no gradient,
    mutation, fitting or RNG. Invalid arguments raise RuntimeError.
    """
    return _native.quantized_sobel_edges(image, threshold)


def Pix2PixGenerate(generator, source):
    """Native source-only two-level U-Net with both genuine concatenation skips.

    Finite signed matching Float32/64 CPU source[N,28,28,1], N>0. Biased normal
    .02 SAME convs/ReLU encode4/8 channels; learned transpose/ReLU decode4/2.
    Joins (up14,down14), then (up28,source); final3x3 sigmoid. Seed6101 creates
    1390 values in10 leaves. Returns owning live down14,down7,up14,joined14,
    up28,joined28,image fields. First/higher source/weight paths sum through
    skips; no target, labels, noise, normalization, fitting or optimizer writes.
    Inputs/store errors raise RuntimeError; synchronize shared stores.
    """
    return _native.Pix2PixGenerate(generator, source)


def Pix2PixDiscriminate(discriminator, source, image):
    """Native conditional raw49-patch map of a source/image pair.

    Finite signed matching Float32/64 CPU [N,28,28,1], aligned positive rows.
    concat(source,image), two4x4/stride2 biased normal.02 convs/leaky.2, then
    raw3x3 score. Seed6113 creates725 values in6 leaves. Owns live[N,7,7,1]
    logits with first/higher paths through both inputs and weights. No sigmoid,
    batch state, fitting or optimizer writes; validation errors raise RuntimeError.
    """
    return _native.Pix2PixDiscriminate(discriminator, source, image)


def Pix2Pix(generator, discriminator, target, source, l1_weight=100):
    """Native paired patch-logistic game plus explicit mean L1, without fitting.

    Distinct matching floating G/D stores; finite signed CPU target/source
    [N,28,28,1], aligned positive rows. G reads source alone. D loss sums real
    and fake patch means without half; G=mean(softplus(-fake))+weight*mean L1.
    Finite nonnegative representable weight defaults100; zero retains L1 field.
    Owns seven live fake/raw-logit/objective fields through parent/store collection.
    First/higher paths remain live, including G through D. Caller selects writes:
    capture D partials/step D, then fresh forward for G through updated D. No
    hidden optimizer/capture/schedule, batch state or no-grad. Invalid inputs,
    stores, weight or nonfinite math raise RuntimeError; synchronize shared stores.
    See Isola et al.(2017), https://arxiv.org/abs/1611.07004, compact teaching variant.
    """
    return _native.Pix2Pix(generator, discriminator, target, source, l1_weight)


_native._set_doc(Pix2PixGenerated, "Owning live down14,down7,up14,joined14,up28,joined28,image fields; both skip concatenations and their graphs outlive parents/stores.")
_native._set_doc(Pix2PixOutput, "Owning live fake,real_logit,fake_logit,g_adversarial,l1,discriminator_loss,generator_loss fields; paired L1 weight100, no hidden optimizer or fitting.")


def pixel_rms_normalize(input, epsilon=None):
    """Native last-axis RMS division, without centering, learned affine or batch coupling.

    Finite Float32/64 CPU[...,C], rank>=1, positive axes and positive finite
    representable epsilon. Default is Float32-rounded 1e-8 even in Float64.
    Explicit input/sqrt(mean(square(input),last,keepdims=True)+epsilon) preserves
    source grouping; first/higher input paths live. Owns output shape/dtype;
    no mutation, RNG, state, optimizer or fitting. Errors raise RuntimeError.
    """
    if epsilon is None:
        return _native.pixel_rms_normalize(input)
    return _native.pixel_rms_normalize(input, epsilon)


def ProgressiveActiveGroups(stage, generator=True):
    """Native source-ordered active group indices as a fresh Python list.

    Stage0 G[0,1], stage1 G[0,1,2,3], stage2 G[0,2,3]; D adds4. Activity is
    determined by stage, not alpha: old RGB groups still advance Adam on the
    final alpha1 fade batch. Inactive groups receive no step/counter advance.
    No fitting, store access or writes; invalid stages raise RuntimeError.
    """
    return list(_native.ProgressiveActiveGroups(stage, generator))


def ProgressiveGanGenerate(g_core, g_old_rgb, g_new_block, g_new_rgb, noise, alpha):
    """Native shared-core old4/new8 generator with genuine sigmoid-image fading.

    Four distinct matching Float32/64 CPU stores, finite noise[N,16], positive
    N, matching scalar or[1] alpha in[0,1], optionally live. Source tanh stem,
    normal(.02) biased convs, leaky.2 and pixel RMS with rounded epsilon. Counts
    1236,5,296,5 in12 leaves. Owns old4[N,4,4,1],new8/image[N,8,8,1]. Both
    branches always evaluate; first/higher noise/alpha/weight paths live. No
    stage selection, optimizer or fitting. Frozen old RGB does not freeze the
    shared core. Inputs/errors follow native validation; errors raise RuntimeError.
    """
    return _native.ProgressiveGanGenerate(g_core, g_old_rgb, g_new_block, g_new_rgb, noise, alpha)


def ProgressiveGanDiscriminate(d_core, d_old_rgb, d_new_block, d_new_rgb, image, alpha):
    """Native old/new discriminator feature fade before a shared raw-score core.

    Four distinct matching Float32/64 CPU stores, finite signed image[N,8,8,1],
    N>0, matching scalar or[1] alpha in[0,1]. Old input pools before RGB; new
    RGB/two-conv path pools afterward. Biased normal(.02) convs/leaky.2, no RMS
    in D, dense raw score. Counts213,8,296,8 in12 leaves. Owns old_path/new_path/
    mixed/base[N,4,4,4],score[N,1]; both branches evaluate at endpoints.
    Applicable first/higher paths live; no stage, optimizer, sigmoid or fitting.
    Invalid inputs/stores/nonfinite math raise RuntimeError.
    """
    return _native.ProgressiveGanDiscriminate(d_core, d_old_rgb, d_new_block, d_new_rgb, image, alpha)


def ProgressiveGan(g_core, g_old_rgb, g_new_block, g_new_rgb,
                   d_core, d_old_rgb, d_new_block, d_new_rgb, image, noise, alpha):
    """Native eight-store progressive game, with source G loss having NO half factor.

    Eight distinct matching Float32/64 CPU stores, finite aligned image[N,8,8,1]
    and noise[N,16], N>0; matching scalar or[1] alpha in[0,1], optionally live.
    All2067 values/24 leaves always evaluate. D=.5*(mean((real-1)**2)+mean(fake**2)),
    G=mean((fake-1)**2). Seven owning endpoint/score/loss fields retain storage
    and first/higher paths through parent/store collection. No hidden optimizer,
    capture, stage, freeze, fitting or checkpoint. Capture ALL active D gradients
    before any D write, then fresh G and ALL its active gradients before G writes.
    Inactive Adam counters stay unchanged; old RGB remains active at fade alpha1.
    Synchronize stores; input/store/nonfinite errors raise RuntimeError.
    """
    return _native.ProgressiveGan(g_core, g_old_rgb, g_new_block, g_new_rgb,
                                 d_core, d_old_rgb, d_new_block, d_new_rgb, image, noise, alpha)


_native._set_doc(ProgressiveGanGenerated, "Owning live old4,new8,image fields, including both sigmoid endpoints and actual fade; tensors/graphs outlive parents/stores.")
_native._set_doc(ProgressiveGanDiscriminated, "Owning live old_path,new_path,mixed,base features[N,4,4,4] and raw score[N,1]; tensors/graphs outlive parents/stores.")
_native._set_doc(ProgressiveGanOutput, "Owning seven live fake/old4/new8/raw-score/loss fields; G loss has no half factor; no stage, capture, optimizer or fitting hidden.")


def MidpointUniform(rows, columns=1, seed=6601, dtype=None):
    """Native local-MT19937 midpoint matrix with one final Float32/64 cast.

    Positive representable rows*columns, uint32 seed; Float32 default. Each raw
    word uses (Double(word)+.5)/2**32. Double output is strictly inside(0,1);
    Float32 can round to1. Owns detached CPU[rows,columns]; repeat calls repeat
    bits without consuming another stream. No fitting or retained RNG. Invalid
    geometry/dtype raises RuntimeError; allocation errors may propagate.
    """
    if dtype is None:
        return _native.MidpointUniform(rows, columns, seed)
    return _native.MidpointUniform(rows, columns, seed, dtype)


def ClipWassersteinCritic(critic, bound=.01):
    """Native explicit clipping of every critic tensor, including biases.

    Nonempty finite Float32/64 CPU store; positive finite representable bound.
    Computes/validates all candidates before writes, preserving names/order/flags.
    Call once after initialization and after EVERY clipped-WGAN critic step;
    no G call, optimizer/counter reset, fitting or Lipschitz certificate. Avoid
    reusing pre-write graphs; synchronize the store. Errors raise RuntimeError.
    """
    return _native.ClipWassersteinCritic(critic, bound)


def WassersteinGenerate(generator, noise):
    """Native compact Wasserstein G:2->16 tanh->2 unbounded coordinates.

    Finite matching Float32/64 CPU noise[N,2], N>0, strides allowed; source
    Glorot initialization/names,82 values/four leaves via BiganMap. Owns live
    [N,2] with first/higher continuous paths. No critic, clipping, optimizer,
    fitting or checkpoint; invalid inputs/nonfinite math raise RuntimeError.
    """
    return _native.WassersteinGenerate(generator, noise)


def WassersteinCritic(critic, point):
    """Native compact Wasserstein critic:2->16 tanh->1 raw score, without sigmoid.

    Finite matching Float32/64 CPU point[N,2], N>0; source Glorot names/order,
    65 values/four leaves via BiganMap. Owns live raw[N,1] with first/higher
    continuous paths. No G, clip, optimizer, fitting or probability interpretation.
    Invalid inputs/stores/nonfinite math raise RuntimeError.
    """
    return _native.WassersteinCritic(critic, point)


def WassersteinGan(generator, critic, real, noise, alpha, penalty_weight=10., penalty_graph=True):
    """Native eleven-field Wasserstein game with symbolic critic-input penalty.

    Distinct matching Float32/64 CPU stores, finite real/noise[N,2] and alpha[N,1]
    in[0,1], positive matching rows. Detached interpolation stops external real,
    alpha and G penalty paths. Enables recording internally for the actual input
    gradient even under no-grad; penalty_graph=True retains critic history for
    gradient-penalty fitting. False detaches it; outer no-grad plus False gives
    detached evaluation. Analytic Jacobian is a separate diagnostic.
    Norm=sqrt(sum(J**2)+1e-12), penalty=mean((norm-1)**2), gap=mean(real)-mean(fake),
    G=-mean(fake), C=-gap+weight*penalty. Weight0 skips addition but retains all
    diagnostics. Base C loss still has a G path: choose update groups explicitly.
    No optimizer, clipping, fitting or distance certificate. Owned fields/graphs
    outlive parents/stores; invalid inputs/nonfinite math raise RuntimeError.
    """
    return _native.WassersteinGan(generator, critic, real, noise, alpha, penalty_weight, penalty_graph)


_native._set_doc(WassersteinGanOutput, """Owning eleven Wasserstein fields with independent tensor storage/graph lifetimes.

Fake,raw scores,detached interpolation,analytic/symbolic Jacobians,norm,penalty,gap
and G/C losses outlive parents/stores. Penalty stops external real/alpha/G paths;
symbolic critic history is optional. No optimizer, clipping callback or mode retained.
""")


def capture_affine_batch_norm(parameters, moving, name, input, batch_statistics=False, epsilon=.001):
    """Native fused learned-affine BN, with detached captures and no moving write.

    Finite matching Float32/64 CPU[...,C], rank>=2, positive axes; two distinct
    matching stores, nonempty prefix and positive finite representable epsilon.
    Creates gamma/beta before saved mean/variance; defaults Double epsilon .001.
    Batch mode owns live value and detached mean/unbiased_variance[C]; inference
    owns live value with undefined captures and stopped saved statistics. First/
    higher continuous paths live. No EMA, optimizer or fitting; errors raise RuntimeError.
    """
    return _native.capture_affine_batch_norm(parameters, moving, name, input, batch_statistics, epsilon)


def DcganGenerate(generator, moving, noise, batch_statistics=False):
    """Native projection/fused-BN/two-transposed-map generator with signed tanh pixels.

    Distinct matching Float32/64 CPU stores; finite noise[N,16], N>0. Source
    normal(.02) kernels, 6873/eight leaves; moving24/four leaves. Returns owning
    image[N,28,28,1] and four detached batch moment vectors (undefined in inference).
    Saved statistics by default; first/higher continuous paths live, saved/captured
    moments stop. No moving write after initialization, optimizer or fitting.
    Synchronize stores; invalid geometry/stores/nonfinite math raises RuntimeError.
    """
    return _native.DcganGenerate(generator, moving, noise, batch_statistics)


def DcganDiscriminate(discriminator, moving, image, batch_statistics=False):
    """Native two-stride convolutional discriminator with raw logits and fused-BN captures.

    Distinct matching Float32/64 CPU stores; finite signed image[N,28,28,1].
    Source normal(.02) kernels/LeakyReLU(.2),989/seven leaves; moving16/two leaves.
    Owns logit[N,1] and detached mean/variance[8], undefined with saved BN(default).
    Applicable first/higher continuous paths live; saved/captured moments stop.
    No sigmoid, joint batch construction, moving write, optimizer or fitting.
    Invalid inputs/stores or nonfinite math raises RuntimeError.
    """
    return _native.DcganDiscriminate(discriminator, moving, image, batch_statistics)


def DcganDiscriminatorPhase(generator, discriminator, g_moving, d_moving, image, noise):
    """Native joint-real/fake batch-BN D loss with a stopped G path and detached captures.

    Four distinct matching Float32/64 CPU stores, finite real[N,28,28,1] in unit
    coordinates without clipping, noise[N,16], positive matching rows. G uses batch
    BN under no-grad; D runs once on concat(2*real-1,signed_fake) with batch BN.
    Owns raw logits, stable logistic scalar and detached G/D captures. No writes;
    commit only D captures with explicit Double .9 retention, then selected D Adam.
    Applicable real/D first/higher paths live; G paths stop. Errors raise RuntimeError.
    """
    return _native.DcganDiscriminatorPhase(generator, discriminator, g_moving, d_moving, image, noise)


def DcganGeneratorPhase(generator, discriminator, g_moving, d_moving, image, noise):
    """Native fresh G objective with batch-BN G and current saved-BN D, without writes.

    Same four-store/aligned real[N,28,28,1]/noise[N,16] contract as D phase. Real
    values are used only for validation. Returns signed fake, raw fake logits,
    mean softplus(-logit), detached G moments; real_logit/down7 captures undefined.
    First/higher G/noise/D-weight paths live: explicitly select G gradients. Call
    after D capture commit/Adam; commit both G layers with Double .9, then G Adam.
    Saved/captured moments stop; no optimizer/fitting hidden. Errors raise RuntimeError.
    """
    return _native.DcganGeneratorPhase(generator, discriminator, g_moving, d_moving, image, noise)


def Dcgan(generator, discriminator, g_moving, d_moving, image, noise):
    """Native saved-BN DCGAN inference maps with raw logistic losses and unit pixels.

    Four distinct matching Float32/64 CPU stores; finite aligned real[N,28,28,1]
    in unit coordinates without clipping, noise[N,16]. Owns seven fields: fake[N,784],
    real/fake raw logits and sigmoid diagnostics, d_loss/g_loss scalars. Both maps
    read saved moments, without moving updates. Applicable first/higher continuous
    paths live; saved moments stop. No MMD, optimizer, fitting or checkpoint hidden.
    Compute GanCoarsePixels/ImqMmd separately; errors raise RuntimeError.
    """
    return _native.Dcgan(generator, discriminator, g_moving, d_moving, image, noise)


_native._set_doc(AffineBatchNormCapture, "Owning live affine normalized value and optional detached mean/unbiased_variance; fields outlive parent/store collection.")
_native._set_doc(DcganGenerated, "Owning signed image[N,28,28,1] and four optional detached BN moment vectors; fields/graphs outlive parent/store collection.")
_native._set_doc(DcganDiscriminated, "Owning raw logit[N,1] and optional detached mean/variance[8]; fields/graphs outlive parent/store collection.")
_native._set_doc(DcganPhaseOutput, "Owning signed fake, phase-specific raw logits/loss and detached G/D BN captures; G phase has undefined real/D-capture fields; no hidden write.")
_native._set_doc(DcganOutput, "Owning seven saved-BN fields: unit fake[N,784], raw/probability real/fake scores and logistic d_loss/g_loss; fields/graphs outlive parents/stores.")


def capture_batch_norm(stats, name, input, batch_statistics=False, epsilon=None):
    """Native unparameterized BN with live captures and no moving-statistic write.

    Finite matching Float32/64 CPU input[...,C], rank>=2, positive axes, nonempty
    prefix. Creates missing moving_mean/var only; saved values are detached.
    Batch mode uses mean(square(x-mean)), division/sqrt and source-rounded
    unbiased correction; mean/variance fields are undefined in inference mode.
    Default epsilon is Float32-rounded 1e-5 even in Float64. Strides and applicable
    first/higher input/capture paths live. No affine, EMA, optimizer or fitting.
    Owns value/mean/unbiased_variance fields. Invalid inputs raise RuntimeError;
    stored-state errors may occur after earlier missing state was initialized.
    """
    if epsilon is None:
        return _native.capture_batch_norm(stats, name, input, batch_statistics)
    return _native.capture_batch_norm(stats, name, input, batch_statistics, epsilon)


def batch_norm_moving_values(old_mean, old_variance, captured_mean, captured_unbiased_variance, momentum=None):
    """Native detached candidate moments: m*old+(1-m)*captured, with no store write.

    Finite matching Float32/64 CPU[C] vectors, C>0, nonnegative variances and
    momentum in[0,1]. Default m is Float32-rounded .9. All input paths stop;
    owning mean/variance fields outlive their parent. Compute all selected-layer
    candidates before committing state, after the optimizer using these captures.
    Invalid/nonfinite inputs or results raise RuntimeError.
    """
    if momentum is None:
        return _native.batch_norm_moving_values(old_mean, old_variance, captured_mean, captured_unbiased_variance)
    return _native.batch_norm_moving_values(old_mean, old_variance, captured_mean, captured_unbiased_variance, momentum)


def GaussianRejectionPrior(rows, features, seed, limit=float('inf'), dtype=None):
    """Native local-MT Box-Muller rejection bank and complete Double proposal audit.

    Positive representable rows/features, uint32 seed, positive non-NaN absolute
    limit (infinity default), Float32(default)/Float64 output. Midpoint uniforms,
    cosine then cached sine, threshold BEFORE one final dtype cast. Returns owning
    detached z[rows,features], proposals[K], including rejected and last accepted
    candidates, without an unused trailing sine. No clipping or global RNG.
    Strict K<1000*rows*features work guard; exhaustion raises RuntimeError.
    """
    if dtype is None:
        return _native.GaussianRejectionPrior(rows, features, seed, limit)
    return _native.GaussianRejectionPrior(rows, features, seed, limit, dtype)


def BigGanResize8(images):
    """Native half-pixel28->8 grayscale resize in Double, then one input-dtype cast.

    Finite signed Float32/64 CPU[N,28,28,1], N>0, strides and applicable input
    paths. Returns owning live[N,8,8,1], without clipping, normalization or data
    loading. Preserves the source Float32 pixel rounding. Invalid input raises RuntimeError.
    """
    return _native.BigGanResize8(images)


def BigGanGenerate(generator, spectral, moving, noise, labels, batch_statistics=False):
    """Native compact hierarchical class-conditional8x8 generator with captured BN.

    Three distinct matching Float32/64 CPU stores; finite signed noise[N,12],
    Int32/64 classes[N] in[0,10), N>0. Three4-coordinate chunks drive stem and
    two conditional BNs with a shared raw class embedding. Residual nearest-up
    block, normal(.02) spectral /W kernels, learned output affine, sigmoid pixels.
    G789/17 leaves, spectral241/18, moving24/6. Returns owning image and nested
    six-vector moments; moment fields are undefined for inference-only calls.
    First/higher continuous paths live, saved moments/columns stop. No advance,
    EMA, optimizer or fitting. Synchronize stores; errors raise RuntimeError.
    """
    return _native.BigGanGenerate(generator, spectral, moving, noise, labels, batch_statistics)


def BigGanDiscriminate(discriminator, spectral, image, labels):
    """Native residual-down discriminator with a spectral class projection.

    Distinct matching Float32/64 CPU stores, finite signed image[N,8,8,1],
    Int32/64 classes[N] in[0,10). Owns score/source/projection[N,1], features[N,4].
    D241/9 leaves, spectral77/10. Source score plus class-feature inner product;
    no sigmoid/class posterior, G call, BN, advance, optimizer or fitting.
    Applicable first/higher continuous paths live; cached columns stop while
    sigma's W path lives. Invalid inputs/nonfinite math raise RuntimeError.
    """
    return _native.BigGanDiscriminate(discriminator, spectral, image, labels)


def ModifiedOrthogonalPenalty(parameters, coefficient=None):
    """Native off-diagonal raw /W Gram-squared penalty, with connected zero other partials.

    Flattens all but the last axis; excludes Gram diagonal. Default coefficient
    is Float32-rounded .5e-4. Every other leaf contributes0*sum, including raw
    class embedding, bias and output affine. Source creation-order accumulation.
    Finite nonempty Float32/64 CPU store, rank>=2 kernels, positive representable
    products and finite nonnegative coefficient. No state write/optimizer.
    Owns live scalar with applicable higher kernel paths; errors raise RuntimeError.
    """
    if coefficient is None:
        return _native.ModifiedOrthogonalPenalty(parameters)
    return _native.ModifiedOrthogonalPenalty(parameters, coefficient)


def BigGanCommitMoments(moving, captured):
    """Native atomic validation then six BN writes using already captured pre-Adam moments.

    Existing bn1/bn2/output_bn mean/variance[4] and matching captured vectors;
    nonnegative variances. Computes all candidates before any copy; invalid/missing
    data leaves state unchanged. Retention is Float32-rounded .9. Call AFTER G
    Adam using moments from the SAME forward, without recomputing at new weights.
    D phases skip this function. No initialization, spectral advance or optimizer.
    Flags stay unchanged; synchronize shared stores. Errors raise RuntimeError.
    """
    return _native.BigGanCommitMoments(moving, captured)


def BigGan(generator, discriminator, generator_spectral, discriminator_spectral, moving,
           image, noise, labels, display_batch_statistics=False):
    """Native compact BigGAN with both G paths, explicit losses and captured moments.

    Five distinct matching Float32/64 CPU stores, finite signed image[N,8,8,1]
    and noise[N,12], Int32/64 classes[N] in[0,10), positive matching rows.
    1030 optimized values,342 state values,60 tensors. Returns owning twelve-field
    record including nested moments. Display flag selects fake/fake_score only;
    BOTH objectives ALWAYS use the batch-statistics fake. D has hinge margins;
    G=-mean(training fake score)+modified raw-kernel orthogonal penalty.
    Caller warms SN16 per network, then D-SN/D write/G-SN/fresh G gradient/G write/
    captured BN commit. No advance, EMA, optimizer, fit, freeze or checkpoint hidden.
    First/higher continuous paths live; labels discrete, cache/saved BN stop.
    Input checks precede leaves; existing-leaf errors may occur later. Synchronize stores.
    """
    return _native.BigGan(generator, discriminator, generator_spectral, discriminator_spectral, moving,
                          image, noise, labels, display_batch_statistics)


_native._set_doc(BatchNormCapture, """Owning normalized value and optional live mean/unbiased_variance[C].

Moment fields are undefined in saved-statistics mode. Fields outlive parent/store
collection; no EMA write, optimizer or deferred callback is retained.
""")
_native._set_doc(BatchNormMovingValues, """Owning detached moving mean/variance candidates, without any write.

No store reference, graph, optimizer or update callback retained; fields outlive parent.
""")
_native._set_doc(GaussianRejectionDraw, """Owning detached accepted z[rows,features] and Float64 proposals[K].

Audit includes rejects and last acceptance, without unused trailing sine. Threshold
precedes dtype rounding; no global generator or fitting state is retained.
""")
_native._set_doc(BigGanMoments, """Owning six live captured channel vectors: bn1/bn2/output mean/unbiased variance.

All tensor fields are undefined on inference-only G calls. Values and graphs
outlive parent/store collection; commit these SAME pre-write captures after G Adam.
""")
_native._set_doc(BigGanGenerated, """Owning image[N,8,8,1] and nested BigGanMoments, without hidden state writes.

Fields retain storage and applicable first/higher paths after parent/store collection.
""")
_native._set_doc(BigGanDiscriminated, """Owning score,source,projection[N,1] and sum-pooled features[N,4].

Score=source+projection, without class probabilities. Fields retain storage/graphs
after parent/store collection; no optimizer or state mutation is retained.
""")
_native._set_doc(BigGanOutput, """Owning both G paths, raw scores, explicit game losses and nested captured moments.

Display fake/fake_score select inference by default, but ALL losses use the batch
fake. Tensor fields and moments outlive parent/stores; no updates or optimizer hidden.
""")


def cached_spectral_dense(parameters, state, name, x, units, activation=None):
    """Native dense map with cached columns, live signed sigma and source-rounded initialization.

    Finite matching Float32/64 CPU x[N,in], distinct matching stores, nonempty
    prefix and positive representable units/products. Creates prefix/W,b and
    state/u,v only when missing; no existing cache advance. Uniform limits and
    column constants preserve Float32 rounding even in Float64. First/higher
    x/W/b paths live, cache paths stop. Output owns storage. Default activation
    is None/raw affine. Exclude state from Adam; synchronize shared stores.
    Invalid inputs/nonfinite output raise RuntimeError; existing-leaf errors can occur later.
    """
    if activation is None:
        return _native.cached_spectral_dense(parameters, state, name, x, units)
    return _native.cached_spectral_dense(parameters, state, name, x, units, activation)


def SpectralSpatialAttention(parameters, state, input, query_width=2, value_width=4, name="attention"):
    """Native unscaled spatial attention with spectral projections and a zero residual gate.

    Finite matching Float32/64 CPU input[B,T,C], positive representable geometry,
    distinct stores and nonempty prefix. Creates query/key/value/project W,b,
    then scalar gamma0; default Q2,V4. probability=softmax(Q@K.T), without sqrt
    scaling, masking or dropout. output=input+gamma*project(probability@V).
    Reuses native Plain attention at scale1. Owns input/probability/output fields;
    first/higher paths live, cache paths stop while sigma's W path lives.
    No hidden cache advance, fitting or optimizer. Synchronize shared stores.
    Input checks precede leaves; existing-leaf/nonfinite errors may occur later.
    """
    return _native.SpectralSpatialAttention(parameters, state, input, query_width, value_width, name)


def SelfAttentionGanGenerate(generator, state, noise):
    """Native spectral patch G: explicit noise[B,16] to sigmoid raster pixels[B,784].

    Input dense128/tanh, sixteen tokens with8 features, unscaled spectral spatial
    attention Q2/V4, shared8->49 sigmoid patches, native MaeImage permutation.
    G2730/13 leaves and cache245/12 leaves; source order and Float32-rounded
    initialization. Owns pixels and attention_input/probability/output fields.
    Finite matching Float32/64 CPU input, positive rows, strides/higher paths.
    No noise draw beyond missing leaves, cache advance, optimizer or fitting.
    Distinct stores required; errors raise RuntimeError. Synchronize shared stores.
    """
    return _native.SelfAttentionGanGenerate(generator, state, noise)


def SelfAttentionGanDiscriminate(discriminator, state, image):
    """Native spectral patch D: finite signed image[B,784] to raw score[B,1].

    Native MaePatches, shared49->8/tanh, learned normal(.01) positions[16,8],
    unscaled spatial attention Q2/V4, token mean, spectral8->1. Creates D650/14
    leaves and cache110/12 leaves in source order. Owns score and all attention
    fields; Float32/64 CPU stores/input match, positive rows, strides/higher paths.
    No sigmoid, labels, cache advance, optimizer or fitting. Distinct stores,
    synchronized mutation; invalid input/nonfinite errors raise RuntimeError.
    """
    return _native.SelfAttentionGanDiscriminate(discriminator, state, image)


def SelfAttentionGan(generator, discriminator, generator_state, discriminator_state, image, noise):
    """Native twelve-field patch SAGAN with unscaled attention in both networks.

    Four distinct matching Float32/64 CPU stores; finite signed image[N,784]
    and noise[M,16], positive possibly unequal N/M. G2730,D650,G-cache245,
    D-cache110,51 tensors/3735 checkpoint values;3380 optimized values.
    D minimizes hinge margins with zero boundary partials; G=-mean(fake_score).
    Caller initializes, warms both cache stores16 times, then advances D cache,
    writes D, advances G cache, recomputes fresh G through updated D with SAME
    D cache and writes G. No advance, fit, optimizer or MMD is hidden.
    Owns all fields; applicable first/higher paths live, cached columns stop.
    Input checks precede leaves; existing-leaf failures may occur later.
    Synchronize stores; approximate normalization is not a Lipschitz certificate.
    """
    return _native.SelfAttentionGan(generator, discriminator, generator_state, discriminator_state, image, noise)


_native._set_doc(SpectralSpatialOutput, """Owning live input/output[B,T,C] and row probability[B,T,T].

Fields retain storage and applicable first/higher paths after parent/store collection.
No optimizer, fitting mode or cache reference; the initial zero gate gives identity.
""")
_native._set_doc(SelfAttentionGenerated, """Owning live pixels[B,784] and three G attention fields.

attention_input/output[B,16,8], attention_probability[B,16,16] retain storage and
applicable first/higher paths after parent/store collection. No fitting state.
""")
_native._set_doc(SelfAttentionDiscriminated, """Owning raw score[B,1] and three D attention fields.

attention_input/output[B,16,8], attention_probability[B,16,16] retain storage and
applicable first/higher paths after parent/store collection. No sigmoid or fitting state.
""")
_native._set_doc(SelfAttentionGanOutput, """Twelve owning live patch SAGAN fields with explicit hinge objectives.

Fake pixels, real/fake scores, D/G losses, G input/probability/output, real-D
input/probability/output and fake-D probability survive parent/store collection.
No optimizer, cache advance, MMD or fitting state is retained.
""")


def cached_spectral_power_iteration(weight, u0, iterations=1, epsilon=None):
    """Native detached column iterations v=unit(Wu), then u=unit(W.T v).

    Finite matching Float32/64 CPU kernel rank>=2, u0[out,1]; all leading
    kernel axes flatten as input. No initial u normalization, graph or mutation.
    unit(x)=x/sqrt(sum(x*x)+epsilon); default is Float32-rounded 1e-12.
    Positive representable guard/count required. Owns detached u/v columns;
    zero kernels give zeros. Raises RuntimeError for invalid/nonfinite math.
    """
    if epsilon is None:
        return _native.cached_spectral_power_iteration(weight, u0, iterations)
    return _native.cached_spectral_power_iteration(weight, u0, iterations, epsilon)


def cached_spectral_kernel(weight, u, v, floor=None):
    """Native W/max(sigma,floor), sigma=sum(detach(v)*(flatten(W)*detach(u))).

    Finite matching Float32/64 CPU kernel rank>=2 and columns u[out,1],v[in,1].
    Returns owning weight/sigma fields; first/higher W paths including sigma
    stay live, u/v paths stop. Raw sigma is signed, without abs. Positive floor
    defaults to Float32-rounded 1e-6. No iteration or mutation is hidden; an
    approximate sigma is not an exact norm certificate. Invalid math raises RuntimeError.
    """
    if floor is None:
        return _native.cached_spectral_kernel(weight, u, v)
    return _native.cached_spectral_kernel(weight, u, v, floor)


def advance_cached_spectral_state(parameters, state, iterations=1, epsilon=None):
    """Advance existing prefix/u,v for every prefix/W, writing only cache state.

    Distinct matching Float32/64 stores; columns must already exist. All next
    pairs compute before any write; invalid/missing/nonfinite pairs leave both
    stores unchanged. Biases ignored; no kernels is a no-op. No leaf creation,
    RNG, optimizer or weight update. Exclude state from optimizers; its flags
    stay unchanged and cache math always detaches. Synchronize shared stores.
    Positive iterations, guard defaults to Float32-rounded 1e-12. Errors raise RuntimeError.
    """
    if epsilon is None:
        return _native.advance_cached_spectral_state(parameters, state, iterations)
    return _native.advance_cached_spectral_state(parameters, state, iterations, epsilon)


def SpectralGanDiscriminate(discriminator, state, image, features=784, hidden=32):
    """Native cached spectral F->tanh H->raw source score, default784/32.

    Distinct matching Float32/64 CPU stores and finite signed image[N,F], N>0.
    Creates hidden/output W,b and cache columns u,v only when missing; Float32-
    rounded uniform limits/constants preserve the source even in Float64.
    Default D25153/4 leaves, cache849/4 leaves. Does not advance existing cache.
    First/higher image/W paths including sigma live, cache paths stopped.
    No warm-up/fit/optimizer; call advance_cached_spectral_state explicitly.
    Positive representable dimensions required; invalid math raises RuntimeError.
    """
    return _native.SpectralGanDiscriminate(discriminator, state, image, features, hidden)


def SpectralGan(generator, discriminator, state, image, noise, features=784, hidden=32):
    """Native compact sigmoid G and cached spectral D with explicit hinge losses.

    Three distinct matching Float32/64 CPU stores; finite signed image[N,F],
    noise[M,Z], positive possibly unequal N/M and representable dimensions.
    Returns five owning live DenseGanOutput fields; D has zero hinge boundary
    derivative, G=-mean(fake_score) stays unhinged. Default G26416,D25153,state849.
    Initialize, explicitly warm cache16 times, then each batch advance cache,
    write D, recompute through updated D with SAME cache, and write only G.
    No schedule/optimizer/MMD/noise draw is hidden beyond leaf initialization.
    Cache paths stop but sigma's W path stays live. Synchronize shared stores.
    Inputs validate before leaves; existing-leaf/nonfinite failures may occur
    later. Errors raise RuntimeError. Finite iteration is not a norm certificate.
    """
    return _native.SpectralGan(generator, discriminator, state, image, noise, features, hidden)


_native._set_doc(SpectralPowerPair, """Owning detached columns u[out,1],v[in,1] from ordered power iteration.

Squared-norm epsilon allows lengths below one; zero kernels give zeros.
Fields retain storage after parent/input collection; no store or graph retained.
""")
_native._set_doc(CachedSpectralKernel, """Owning live weight=W/max(sigma,floor) and raw signed scalar sigma.

Fields retain first/higher W gradients after parent collection; cached u/v paths
are stopped. Finite power iteration does not certify an exact unit operator norm.
""")


def BernoulliPixelNll(logits, target):
    """Native per-row sum of softplus(logits)-target*logits, with both gradients.

    Finite matching nonempty Float32/64 [B,F]; target values in [0,1]. Returns
    owning [B], without pixel averaging or clipping. Invalid inputs raise
    RuntimeError. Finite large logits use stable softplus; dtype overflow is not repaired.
    """
    return _native.BernoulliPixelNll(logits, target)


def _codec_vector_field(descriptor, vector_type, export):
    """Copy packet vector fields on export and accept any sequence on assignment."""
    def assign(packet, values):
        descriptor.__set__(packet, values if isinstance(values, vector_type) else vector_type(list(values)))
    return property(lambda packet: export(descriptor.__get__(packet, type(packet))), assign)


# Export the vector fields as tuple/bytes copies and keep Python's ordinary sequence ingress.
CodecPacket.source_ids = _codec_vector_field(CodecPacket.source_ids, _native.IntVector, tuple)
CodecPacket.symbols = _codec_vector_field(CodecPacket.symbols, _native.ByteVector, bytes)


def _codec_bytes(values):
    """Adapt Python bytes/sequences to the canonical native byte container."""
    return values if isinstance(values, _native.ByteVector) else _native.ByteVector(list(values))


_native._set_doc(CodecQuantizer, """Owning finite Float32 minimum/maximum vectors of equal positive width.
Every native consumer revalidates ordered bounds. Copies own vector values;
fitting is explicit and stopped. Constant ranges are allowed. CPU only.
""")
_native._set_doc(CodecPacket, """Explicit self-contained TFLCODE1 packet, with no encoder state.
Fields: quantizer (32 Float32 ranges), weights Float32[32,784], bias Float32[784],
source_ids (1..10000 nonnegative Int32 identities), symbols (N*32 bytes).
Their getters return owning tuple/bytes copies; assign a sequence to replace them.
Tensor fields retain caller handles; UnpackCodec creates independent detached
storage. The nested quantizer getter returns an owning copy: assign it back
after edits. Pack/decode validate all fields. No checksum or entropy coding.
""")


def FitCodecQuantizer(code):
    """Native fitting-only per-coordinate min/max of finite Float32 CPU[N,D].

    Positive axes, signed values and strides supported. Returned vector ranges
    own values and retain no fitting gradient; no labels or hidden optimization.
    Invalid input raises RuntimeError.
    """
    return _native.FitCodecQuantizer(code)


def QuantizeCodec(code, quantizer):
    """Native detached row-major bytes from finite matching Float32 CPU[N,D].

    Uses Double floor(clamp((v-low)/(high-low)*255,0,255)+.5), preserving
    half-up ties and clipping. Constant coordinates emit zero. No range refit
    or straight-through gradient. Invalid rows/ranges raise RuntimeError.
    """
    return _native.QuantizeCodec(code, quantizer)


def DequantizeCodec(symbols, quantizer):
    """Native owning detached Float32[N,D] from complete nonempty byte rows.

    Evaluates Float32(low+symbol*(high-low)/255) in Double before casting.
    Finite ordered explicit ranges; constant ranges allowed. No decoder/refit.
    Invalid ranges/partial rows raise RuntimeError.
    """
    return _native.DequantizeCodec(_codec_bytes(symbols), quantizer)


def SymbolTensor(symbols, width=32):
    """Copy complete nonempty byte rows to owning detached Int32[N,width].

    Positive width defaults to32; no quantization or fitting. Invalid geometry
    raises RuntimeError. Byte values retain their exact nonnegative values.
    """
    return _native.SymbolTensor(_codec_bytes(symbols), width)


def LearnedCodecDecode(codes, weights, bias):
    """Native live sigmoid(codes @ weights + bias), with no encoder or store.

    Finite matching signed Float32/64 CPU codes[N,D], weights[D,F], bias[F],
    positive axes and strides. All continuous first/higher paths remain live.
    Invalid values/geometry or nonfinite computed logits raise RuntimeError.
    """
    return _native.LearnedCodecDecode(codes, weights, bias)


def PackCodec(packet):
    """Native exact TFLCODE1 bytes: 103772+36*N, including decoder overhead.

    Fixed N/32/28/28/1 header, interleaved Float32 ranges, W,b, U32 identities,
    then bytes. Little endian, finite binary32 bits including signed zeros.
    All fields validated; N in1..10000. No file IO, gradients or checksum.
    Malformed fields raise RuntimeError.
    """
    return _native.PackCodec(packet)


def UnpackCodec(data):
    """Native bounded parser yielding owning detached TFLCODE1 packet fields.

    Checks magic/fixed shape/count/exact length before decoder allocation,
    every finite scalar, ordered range and nonnegative Int32 identity.
    Rejects truncation/trailing bytes with RuntimeError. No files, encoder
    or source assets needed; no checksum detects arbitrary bit corruption.
    """
    return _native.UnpackCodec(_codec_bytes(data))


def DecodeCodec(packet):
    """Native fresh packet-only decoder to detached Float32[N,784] probabilities.

    Validates fixed packet, dequantizes its bytes and uses only its W/b.
    Caller gradient flags stay unchanged; no original encoder/images, model
    store, fitting, RNG or per-image normalization. Invalid values raise RuntimeError.
    """
    return _native.DecodeCodec(packet)


_native._set_doc(InfoCodes, """Owning generated-prior category Int32[N], continuous Float32/64[N,U], raw Float64[N,U+1].
Raw words store accepted category then continuous draws, without rejected words.
Fields survive parent collection, have no graph and retain no RNG. IDs are not
MNIST labels; consumers revalidate ranges. See InfoPrior for exact stream.
""")
_native._set_doc(InfoRecognition, """Owning image-only Q logits/probability [N,C] and unbounded means [N,U].
All fields retain native graphs after parent/store collection. Q is separate
from D; variance is fixed at one. No code, noise, target or digit label input.
Probability is a convenience; the recognition loss consumes logits and mean.
""")
_native._set_doc(InfoRecognitionLoss, """Owning live categorical, continuous, total and lower_bound scalars.
R is categorical CE plus per-row summed unit-Gaussian NLL. Bound is fixed
uniform-prior entropy minus R; it can be negative and is not measured true MI.
""")
_native._set_doc(InfoGanOutput, """Fifteen owning live separate-Q InfoGAN fields without fitting state.
Includes generated pixels, image-only D logits/reporting sigmoids, D/G/R and
component losses, variational bound and Q logits/probability/means. Fields retain
native storage/graphs after record/store collection. No hidden MMD or optimizer.
""")


def InfoPrior(rows, seed, classes=10, continuous=2, dtype=None):
    """Native local mt19937 row-interleaved uniform categorical/continuous prior.

    Category word rejection below floor(2**32/C)*C precedes U midpoint-uniform
    words per row; continuous Double arithmetic then one final Float32/64 cast.
    Float32 rounding can attain endpoints -1/1. Raw accepted words stay Float64.
    Positive representable counts, C<=INT32_MAX, U>0; Float32 default. Owns
    detached category/continuous/raw tensors; no global RNG, data or fitting.
    Invalid options raise RuntimeError; standard allocation failures propagate.
    """
    return _native.InfoPrior(rows, seed, classes, continuous,
                             DType.Float32 if dtype is None else dtype)


def InfoGanGenerate(generator, noise, category, continuous, features=784, classes=10, hidden=32):
    """Native concat(noise,onehot generated category,continuous)->tanh->sigmoid.

    Finite signed matching Float32/64 CPU noise[M,Z], continuous[M,U] in[-1,1],
    Int32/64 category[M] in[0,C), all axes positive; strides and continuous
    first/higher paths live. Reuses MlpGanGenerate. Default G has26800 values,
    four hidden/output W,b leaves. No image/D/Q/target label/fit/input RNG.
    Inputs validate before leaves; incompatible leaves/nonfinite results may
    fail later with RuntimeError. Output owns life; synchronize shared stores.
    """
    return _native.InfoGanGenerate(generator, noise, category, continuous, features, classes, hidden)


def InfoGanRecognize(recognizer, image, features=784, classes=10, continuous=2, hidden=32):
    """Native image-only independent Q: tanh hidden, category logits then means.

    Creation order hidden/category/continuous W,b gives25516 default values
    in six leaves. Finite signed matching Float32/64 CPU[N,F], positive widths,
    strides and continuous first/higher paths. Means unbounded; variance fixed1.
    No code, noise, target, G/D or fit. Returns owning logits/probability/mean.
    Checks precede leaves; invalid inputs/leaves/nonfinite output raise RuntimeError.
    """
    return _native.InfoGanRecognize(recognizer, image, features, classes, continuous, hidden)


def InfoGanRecognitionLoss(prediction, category, continuous):
    """Native categorical CE plus unit-Gaussian per-row-summed continuous NLL.

    Logits[N,C] and means/targets[N,U] share finite Float32/64 CPU dtype and
    positive axes; category Int32/64[N] in[0,C), targets in[-1,1]. Probability
    convenience field is unused. R=CE+.5*mean(sum((target-mean)^2))
    +.5*U*log(2*pi). Bound=(log(C)+U*log(2))-R assumes independent uniform
    priors; negative bound allowed and does not measure actual MI. First/higher
    logits/mean/target paths live; discrete categories. Owning four scalars;
    no fitting/RNG. Invalid inputs/overflow raise RuntimeError.
    """
    return _native.InfoGanRecognitionLoss(prediction, category, continuous)


def InfoGan(generator, discriminator, recognizer, image, noise, category, continuous,
            features=784, classes=10, hidden=32):
    """Native separate-Q InfoGAN; caller recomputes D then Q then G writes.

    Three distinct matching Float32/64 stores, positive CPU image[N,F], noise[M,Z],
    generated category Int32/64[M] in[0,C), continuous[M,U] in[-1,1]. Unequal
    positive N/M, strides and applicable continuous first/higher paths supported.
    Continuous input is also recognition target. Defaults77469 values/14 leaves;
    Q hidden/category/continuous order, G/D hidden/output. D image-only logistic,
    Q recognition R, G non-saturating source+R (weight1, no entropy subtraction).
    Returns fifteen owning fields; bound may be negative. No true digit label,
    hidden fit/RNG/freeze/optimizer/checkpoint/MMD. All user checks precede leaves;
    incompatible leaves/nonfinite results can fail later with RuntimeError.
    """
    return _native.InfoGan(generator, discriminator, recognizer, image, noise, category, continuous,
                           features, classes, hidden)


_native._set_doc(DenseGanLoss, """Owning live scalar discriminator/generator objectives, with no fitting state.
LeastSquares uses a0,b=c1 half-weighted means; Hinge uses positive D margins
and unhinged negative-mean G. Fields survive parent GC; raw scores are not probabilities.
""")
_native._set_doc(DenseGanOutput, """Five owning live dense-GAN endpoints: fake[M,F], real_score[N,1],
fake_score[M,1], discriminator_loss and generator_loss. No optimizer, mode,
noise draw, checkpoint or quadratic MMD. Fields survive parent/store GC.
""")
_native._set_doc(ConditionalGanOutput, """Seven owning live endpoints with class conditions on BOTH G and D.
Fake[N,F], real/fake logits and source probabilities[N,1], scalar D/G losses.
There is no auxiliary class head or CE. Fields survive parent/store GC.
""")


def LeastSquaresGanObjectives(real, fake):
    """Native D=.5*(mean((real-1)^2)+mean(fake^2)), G=.5*mean((fake-1)^2).

    Preserves source grouping and a0,b=c1 coding. Finite matching Float32/64
    CPU[N,1]/[M,1], positive unequal pools and strides; first/higher paths live.
    No sigmoid, clipping, fit or probability calibration. Inputs unchanged;
    owning scalar fields survive parent GC. Invalid pools/computed square
    overflow raise RuntimeError. This is not the a-1,b1,c0 Pearson derivation.
    """
    return _native.LeastSquaresGanObjectives(real, fake)


def HingeGanObjectives(real, fake):
    """Native D=mean(relu(1-real))+mean(relu(1+fake)), G=-mean(fake).

    D uses zero ReLU subgradient at exact real1/fake-1 margins; G derivative is
    always -1/M, without a hinge. Raw scores unbounded, G loss may be negative.
    Matching finite Float32/64 CPU[N,1]/[M,1], positive unequal pools/strides.
    First/higher paths live away from kinks; classical second derivatives at
    kinks are undefined. No spectral normalization, sigmoid, clipping or fit.
    Owning scalar fields; invalid pools/nonfinite computation raise RuntimeError.
    """
    return _native.HingeGanObjectives(real, fake)


def MlpGanGenerate(generator, noise, features=784, hidden=32):
    """Native canonical tanh/linear map followed by sigmoid, from explicit noise only.

    Finite signed Float32/64 CPU[M,Z] matching store, positive geometry/strides.
    Hidden/output W,b order; default Z16,H32,F784 gives four leaves/26416 values.
    First/higher paths live; no class, D, fit, optimizer or input RNG draw.
    Inputs validate before leaves; incompatible leaves/nonfinite output can fail
    later with RuntimeError. Output owns life; synchronize shared mutable stores.
    """
    return _native.MlpGanGenerate(generator, noise, features, hidden)


def MlpGanDiscriminate(discriminator, image, features=784, hidden=32):
    """Native image-only tanh/LINEAR raw score, without sigmoid or a class condition.

    Canonical AaeDecode math; default F784,H32 gives four leaves/25153 values.
    Finite signed matching Float32/64 CPU[N,F], positive axes/products/strides,
    continuous first/higher paths. Owning [N,1], no fit/target/G/optimizer.
    Input checks precede leaves; invalid input/leaves/nonfinite output raise
    RuntimeError. Raw scores are not calibrated probabilities.
    """
    return _native.MlpGanDiscriminate(discriminator, image, features, hidden)


def DenseGan(generator, discriminator, image, noise, objective, features=784, hidden=32):
    """Native five-field unconditional dense GAN with required DenseGanObjective.

    Logistic, LeastSquares or Hinge selects losses only; architecture/weights
    remain unchanged. Distinct matching Float32/64 stores, finite signed CPU
    image[N,F]/noise[M,Z], positive widths/products, unequal pools and strides.
    Caller updates only D, recomputes, then updates only G. Applicable higher
    paths live; hinge has zero boundary subgradient and unhinged G. No hidden
    optimizer, fitting, input RNG, freeze, checkpoint or quadratic diagnostic.
    Inputs/options validate before leaves; invalid input/leaves/nonfinite output
    raise RuntimeError. Fields own life; synchronize shared stores.
    """
    return _native.DenseGan(generator, discriminator, image, noise, objective, features, hidden)


def ConditionalGanDiscriminate(discriminator, image, labels, features=784, classes=10, hidden=32):
    """Native concat(image,onehot class)->tanh->raw logit, with BOTH pair parts as inputs.

    Hidden/output W,b; default F784,C10,H32 gives four leaves/25473 values.
    Finite signed matching Float32/64 CPU[N,F], integer CPU[N] classes in[0,C),
    positive widths/products/strides. No auxiliary head, CE, G, sigmoid or fit.
    Continuous first/higher paths live, labels discrete. Input checks precede
    leaves; malformed input/leaves/nonfinite output raise RuntimeError.
    """
    return _native.ConditionalGanDiscriminate(discriminator, image, labels, features, classes, hidden)


def ConditionalGan(generator, discriminator, image, noise, labels, features=784, classes=10, hidden=32):
    """Native seven-field logistic GAN with class conditions on BOTH G and D.

    Reuses ConditionalGanGenerate and stable LogisticGanObjectives, without
    auxiliary CE. Distinct matching Float32/64 stores, finite signed CPU
    image[N,F]/noise[N,Z], integer labels[N] in[0,C), positive geometry/strides.
    Caller writes D then recomputes through updated D for G. Continuous higher
    paths live; no hidden fit, optimizer, input noise, freeze, checkpoint or MMD.
    Inputs validate before leaves; malformed input/leaves/nonfinite output raise
    RuntimeError. Seven fields own life after parent/store GC; synchronize stores.
    """
    return _native.ConditionalGan(generator, discriminator, image, noise, labels, features, classes, hidden)


_native._set_doc(BiganOutput, """Eight owning live BiGAN endpoints, without optimizer or fitting state.
Fields: fake[M,F], encoded[N,Z], real_score[N,1], fake_score[M,1], scalar
D/EG losses, reconstruction[N,F] and latent_cycle[M,Z]. Cycle residuals never
enter the fit losses. Tensor fields survive parent/store GC; no inverse guarantee.
""")


def GaussianRingPoints(rows, seed=5507, components=8, radius=2., deviation=.08, dtype=None):
    """Native ordered ring: row i uses component i%components and explicit Gaussian noise.

    Centres, noise and sums use Double; only final [rows,2] coordinates cast to
    Float32 default or Float64. Positive rows/components, representable rows*2,
    finite nonnegative radius/sd; zero radius/sd allowed. Owning stopped CPU
    output, no global RNG or fit. Equal arguments repeat bits. Invalid options
    or nonfinite computed/cast coordinates raise RuntimeError.
    """
    return _native.GaussianRingPoints(rows, seed, components, radius, deviation,
                                     DType.Float32 if dtype is None else dtype)


def BiganMap(parameters, input, output=2, hidden=16):
    """Native inferred-width hidden tanh then LINEAR output coordinates; no sigmoid.

    Reuses AaeDecode math and hidden/output W,b order. Default I=O2,H16 gives
    four leaves/82 values. Finite signed Float32/64 CPU[N,I] matching store,
    positive axes/products and strides. First/higher continuous paths live.
    No encoder, D, fit or optimizer. Inputs validate before leaves; incompatible
    leaves/nonfinite computation can fail later with RuntimeError. Output owns life.
    """
    return _native.BiganMap(parameters, input, output, hidden)


def BiganJointScore(discriminator, image, code, hidden=16):
    """Native concat(image,code)->hidden tanh->raw source logit, with both pair parts live.

    Finite signed matching Float32/64 CPU image[N,F]/code[N,Z] and store dtype,
    positive geometry and strides. Default F=Z2,H16 gives four leaves/97 values.
    No encoder, generator, pair label or probability is hidden. Owning [N,1]
    output retains first/higher paths. Invalid pairs/products, incompatible
    leaves or nonfinite computation raise RuntimeError; synchronize shared stores.
    """
    return _native.BiganJointScore(discriminator, image, code, hidden)


def BidirectionalGan(generator, encoder, discriminator, image, prior, hidden=16):
    """Native eight-field bidirectional joint GAN; three distinct matching-dtype stores.

    Finite signed Float32/64 CPU data[N,F]/prior[M,Z], positive axes/products,
    strides and unequal pools allowed. G prior->data, E data->code, D scores
    (data,E(data)) versus (G(prior),prior). D loss averages softplus(-real) plus
    softplus(fake); EG averages softplus(real) plus softplus(-fake). No cycle or
    reconstruction term is fitted. All continuous first/higher paths remain live.
    Caller updates D, recomputes E, then recomputes G, selecting only each store.
    With D fixed, E's write leaves G's EG partial unchanged. No hidden fit,
    optimizer, noise draw, freeze, checkpoint or inverse guarantee. Input checks
    precede leaves; incompatible leaves/nonfinite computed states can fail later
    with RuntimeError. Fields own life after parent/store GC; synchronize stores.
    """
    return _native.BidirectionalGan(generator, encoder, discriminator, image, prior, hidden)


_native._set_doc(LogisticGanLoss, """Owning live scalar discriminator and non-saturating generator losses.
No optimizer, store, labels or fitting state. Tensor fields survive parent GC.
""")
_native._set_doc(AuxiliaryDiscriminator, """Owning image-only source logits and auxiliary class logits/probabilities.
No class target enters D. Three live Tensor fields survive parent/store GC.
""")
_native._set_doc(AuxiliaryClassifierGanOutput, """Fifteen owning live conditional GAN endpoints and source/class losses.
No optimizer, fitting counter, checkpoint, noise draw or quadratic MMD diagnostic.
Tensor fields survive parent GC; choose update groups and recompute after writes.
""")


def LogisticGanObjectives(real, fake):
    """Native stable source losses: D=mean softplus(-real)+mean softplus(fake),
    G=mean softplus(-fake), preserving tiny real and non-saturating fake gradients.

    Finite same Float32/64 CPU [N,1]/[M,1], positive unequal sizes/strides allowed.
    Owning scalar fields retain first/higher paths. No fitting or store mutation.
    Invalid inputs or nonfinite computation raise RuntimeError.
    """
    return _native.LogisticGanObjectives(real, fake)


def ConditionalGanGenerate(generator, noise, labels, features=784, classes=10, hidden=32):
    """Native concat(noise,onehot)->hidden tanh->output sigmoid conditional G.

    Finite signed Float32/64 CPU noise[N,Z] matching store, integer labels[N] in
    [0,classes), positive widths/axes. Strides and continuous higher paths live.
    Default Z16 creates four leaves/26736 values, hidden then output W,b.
    No D, optimizer, fitting or hidden noise draw. Inputs validated before leaves;
    invalid inputs/incompatible leaves/nonfinite computation raise RuntimeError.
    """
    return _native.ConditionalGanGenerate(generator, noise, labels, features, classes, hidden)


def AuxiliaryGanDiscriminate(discriminator, image, features=784, classes=10, hidden=32):
    """Native image-only hidden tanh->class logits/softmax and raw source logit.

    Finite signed Float32/64 CPU image[N,features] matching store; positive axes
    and strides. Creates hidden, CLASS, SOURCE W,b in this order (six leaves,
    25483 default values). No labels, targets, noise or update enter D. Three
    owning fields retain continuous first/higher paths. Invalid inputs/leaves
    or nonfinite computation raise RuntimeError; synchronize shared stores.
    """
    return _native.AuxiliaryGanDiscriminate(discriminator, image, features, classes, hidden)


def AuxiliaryClassLoss(logits, labels):
    """Native mean(-sum(onehot(labels)*log_softmax(logits),axis=1)).

    Finite Float32/64 CPU logits[N,C], integer CPU labels[N] in [0,C), positive
    axes; strides accepted. Owning scalar retains logit first/higher paths.
    No clipping, smoothing or fitting. Invalid inputs/nonfinite computation
    raise RuntimeError. Labels are discrete.
    """
    return _native.AuxiliaryClassLoss(logits, labels)


def AuxiliaryClassifierGan(generator, discriminator, image, noise, labels, features=784, classes=10, hidden=32):
    """Native fifteen-field conditional G / image-only auxiliary D forward.

    D loss=source_D+(real_CE+fake_CE); G loss=non-saturating source_G+fake_CE.
    Labels condition G and supply CE targets, never D inputs. Distinct matching
    Float32/64 stores, finite signed CPU image[N,F]/noise[N,Z], integer labels[N]
    in [0,C), positive geometry, strides. All continuous paths remain live.
    Select only D parameters, update D, recompute, then select only G parameters.
    No hidden Adam, fitting, checkpoint, random noise or MMD diagnostic. Inputs
    validated before leaves; invalid inputs/leaves/nonfinite computation raise
    RuntimeError. Returned fields own lifetime; synchronize shared stores.
    """
    return _native.AuxiliaryClassifierGan(generator, discriminator, image, noise, labels, features, classes, hidden)


def GanCoarsePixels(pixels):
    """Native 7x7 block averages /4, mapping 28x28 grayscale rows to [N,16].

    Finite signed Float32/64 CPU [N,...784], N>0, rank>=2, strides accepted.
    Owning output retains continuous first/higher paths. No model, fit, clipping,
    kernel or quadratic diagnostic; call native ImqMmd separately. Invalid inputs
    or nonfinite computation raise RuntimeError. This is a coarse game diagnostic.
    """
    return _native.GanCoarsePixels(pixels)


def ReconstructionScores(original, reconstruction):
    """Detached native Float64[N] per-case squared residuals, with feature-order Double sums.

    Finite signed Float32/64 CPU samples[N,...], positive axes, matching N and
    flattened D; trailing shapes and dtypes may differ. Widens before subtraction,
    sums squared deltas in feature order and divides by D. Strides accepted.
    No fitting, clipping or training gradient; caller flags stay unchanged.
    Invalid samples/alignment or computed overflow raise RuntimeError.
    """
    return _native.ReconstructionScores(original, reconstruction)


def InkScores(images, fit_mean):
    """Detached native Float64[N] absolute mean-coordinate deviations from a supplied mean.

    Finite signed Float32/64 CPU samples[N,...], positive axes; widens then sums
    features in order in Double and divides by flattened D. No reference fitting,
    model, labels or clipping. Strides accepted; caller flags stay unchanged.
    Invalid reference/samples or computed overflow raise RuntimeError.
    """
    return _native.InkScores(images, fit_mean)


def NominalThreshold(scores, quantile=.95):
    """Native nearest-rank quantile of an explicit nominal calibration score vector.

    Finite signed Float32/64 CPU[N], N>0. Sorts a copy and selects one-based
    ceil(quantile*N), evaluated in Double, bounded by N. quantile in(0,1],
    default .95; no interpolation/fuzz, dropping NaNs, labels or test selection.
    The caller supplies the intended calibration pool. Flags/input unchanged.
    Invalid vectors/probability raise RuntimeError. Reference: R quantile type1,
    inverse empirical CDF, with no fuzz and excluding probability zero.
    """
    return _native.NominalThreshold(scores, quantile)


_native._set_doc(ReconstructionRanking, """Owning detached AUROC, AP and strict operating-point fractions.
Fields auroc, average_precision, normal_fpr, anomaly_tpr come from RankAnomalies.
Larger scores rank supplied anomalies higher; AUROC gives half credit to ties,
AP processes equal-score groups. AP depends on supplied prevalence. FPR/TPR
use score>threshold, so exact threshold ties are unflagged. No Tensor lifetime.
""")


def RankAnomalies(normal_scores, anomaly_scores, threshold):
    """Native explicit normal/anomaly vector ranking and strict score>threshold flags.

    Nonempty finite signed Float32/64 CPU vectors may differ in length/dtype.
    Delegates tied AUROC/AP to canonical nn.roc_auc/nn.average_precision; normal
    label0, anomaly1. Constant equal scores give AUROC .5 and AP prevalence.
    No calibration, fitting, weighting or gradient paths. Inputs/flags unchanged;
    record fields own values. Invalid vectors or nonfinite threshold raise RuntimeError.
    """
    return _native.RankAnomalies(normal_scores, anomaly_scores, threshold)


def AnomalyCorruption(original, kind):
    """Native owning detached Float32[N,28,28,1] paired grayscale stress controls.

    Finite signed Float32 CPU input, N>0, strides accepted. kind0 replaces top-left
    8x8 with1; kind1 multiplies by Float32(.35); kind2 shifts6 columns right with
    zero fill. No RNG, clipping, labels or anomaly inference. Unmodified/shifted
    values keep bits; caller input/gradient flags unchanged. Invalid kind, geometry,
    dtype or nonfinite values raise RuntimeError. Synthetic changes are not real
    anomaly labels or a deployment guarantee.
    """
    return _native.AnomalyCorruption(original, kind)


def DoubleSampleRows(value):
    """Native owning Float64[N,D] rows from finite signed CPU samples[N,...].

    Float32/64, rank>=1, all axes positive; trailing axes flatten in logical
    order (a vector has D=1). Strides and input gradients survive widening and
    copying. No centering, fitting, labels, clipping or RNG. Invalid values,
    shape or dtype raise RuntimeError; non-CPU storage is unsupported.
    """
    return _native.DoubleSampleRows(value)


_native._set_doc(SvdPca, """Native frozen tensor PCA using centered thin SVD, without variance scaling.

SvdPca(fitting, retained=2) accepts finite signed Float32/64 CPU samples[N,...],
N>=2 and positive axes; R lies in [1,min(N,flattened D)]. Uses only fitting
rows, widens to Float64, subtracts the fitted mean, retains columns of V and
sample variances s**2/(N-1). No labels, null dropping, whitening, query refit or
optimizer is hidden. Zero singular values are allowed; R is not numerical rank.
Signs follow native svd; degenerate bases and backend rounding can differ.

mean(), components(), variances(), scores() export independent detached copies
with shapes [1,D], [D,R], [R], [N,R]. Private fitted state owns lifetime.
Transform(query) returns owning Float64[M,R] using the frozen fitting mean/axes;
Reconstruct(scores[M,K]) uses the first K axes, 1<=K<=R, and returns owning
Float64[M,D] in original units, without pixel clipping. Float32/64 query/score
paths remain live, including higher derivatives; fitting paths are stopped.
Strides are accepted. Invalid inputs/counts or nonfinite computations raise
RuntimeError; native SVD convergence failures propagate. Const calls have no
shared mutable caches. Reference: R prcomp(center=TRUE, scale.=FALSE),
https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prcomp.html .
""")


def NearestFitRows(fitting, query, neighbors=10):
    """Native deterministic Euclidean fitting neighbors, with row-index distance ties.

    Finite signed Float32/64 CPU samples[N,...] and [M,...], positive axes and
    equal flattened widths. Widens before feature-order double squared-distance
    accumulation; sorts (distance, fitting row), nearest first. Positive
    neighbors<=N defaults to10; N must fit Int32. Returns owning detached
    Int32[M,k], without an M-by-N distance matrix. Accepts strides; no labels,
    standardization, metric fitting, query refit or RNG. Invalid inputs/products
    and distance overflow raise RuntimeError. Inputs unchanged.
    """
    return _native.NearestFitRows(fitting, query, neighbors)


def NeighborOverlap(reference, candidate):
    """Native row-wise reference membership fraction for neighbor tables[M,k].

    Positive matching CPU Int32/64 table shapes and nonnegative IDs; integer
    dtypes may differ. Each reference entry counts once if found anywhere in
    its candidate row. Repeated reference IDs count separately, repeated
    candidates add no hits. Strides accepted; no labels or fitting. Returns
    scalar hits/(M*k) in [0,1]. Invalid tables raise RuntimeError.
    """
    return _native.NeighborOverlap(reference, candidate)


def NeighborLabelAccuracy(neighbors, fit_labels, query_labels):
    """Native posthoc first-neighbor label accuracy, without majority voting.

    Nonnegative CPU Int32/64 neighbors[M,k], fit_labels[N], query_labels[M],
    positive axes, with every neighbor ID in [0,N). Only column0 predicts;
    labels enter after fixed neighbors are supplied. Query classes absent from
    fitting score incorrect. Strides/mixed integer dtypes accepted. Returns a
    descriptive scalar ratio, with no fitting or mutations. Invalid inputs,
    alignment or IDs raise RuntimeError.
    """
    return _native.NeighborLabelAccuracy(neighbors, fit_labels, query_labels)


def LatentLerp(a, b, alpha):
    """Native explicit (1-alpha)*a+alpha*b paths, preserving source grouping.

    Finite signed matching Float32/64 CPU endpoints[B,D], levels[T], all axes
    positive; accepts strides and arbitrary levels, including extrapolation.
    Returns owning [B,T,D], with first/higher derivatives through all inputs.
    Two products then sum determine rounding and signed zeros. No clipping,
    projection, norm matching, decoder, fit or RNG. Invalid input/product raises
    RuntimeError; computed overflow is not repaired. Inputs are unchanged.
    """
    return _native.LatentLerp(a, b, alpha)


def LatentDirection(source, direction, scale):
    """Native source+scale*direction affine edits, retaining negative codes.

    Finite signed matching Float32/64 CPU source/direction[B,D], scale[T],
    positive axes and arbitrary levels. Owning [B,T,D] accepts strides and
    first/higher paths through all three inputs. No projection, clipping,
    normalization, decoder, fit, RNG or semantic-factor guarantee is hidden.
    Invalid inputs/products raise RuntimeError; computed overflow is not repaired.
    """
    return _native.LatentDirection(source, direction, scale)


def LatentClassMeans(codes, labels, classes=10):
    """Native detached fitting-label means with row-order double accumulation.

    Finite signed Float32/64 CPU codes[B,D], aligned Int32/64 labels[B] in
    [0,classes), classes positive and every class present. Returns owning
    detached [classes,D] in code dtype, with one final cast after division.
    Accepts strides and live inputs; no encoder updates/derivatives pass through
    the fitted summary. Caller supplies fitting rows only. Label centroids are
    supervised posthoc summaries, not unsupervised factor discovery. Invalid
    tensors, missing classes or accumulation/cast overflow raise RuntimeError.
    """
    return _native.LatentClassMeans(codes, labels, classes)


def LatentFlatten(codes):
    """Native independent contiguous [B*T,D] rows from finite signed [B,T,D].

    Float32/64 CPU, positive axes, with path-major then level-major order.
    Strides and gradients remain live; storage is independent even for a
    contiguous input. No decoding, fitting, clipping or mutation. Invalid
    inputs/products raise RuntimeError.
    """
    return _native.LatentFlatten(codes)


def UndercompleteDecoder(parameters, code, features=784, latent=32):
    """Native code-only sigmoid decoder, with no image or label input.

    Finite signed strided Float32/64 CPU code[B,latent], B>0, matching store;
    0<latent<features. Creates/reuses decoder/W[latent,features], decoder/b[features]
    only: two leaves/25872 default scalars. Uses Glorot/zero initialization if
    absent; restore learned arrays before deployed inference. A decoder-only
    initialization consumes a different RNG sequence from the full model.
    Owns [B,features], with code/decoder first/higher paths. No fitting, clipping,
    encoder use, mutation of existing leaves or optimizer. Invalid inputs fail
    before creation; incompatible leaves or computed overflow may fail later.
    """
    return _native.UndercompleteDecoder(parameters, code, features, latent)


def FrozenEncoderFeatures(image, weight, bias):
    """Native relu(image @ weight + bias), with frozen weight/bias paths.

    Finite signed matching Float32/64 matrices image[B,F], weight[F,L] and
    bias[L], all axes positive. Strides and input gradients are preserved;
    encoder gradients are stopped. Owns result lifetime, mutates no input,
    creates no parameters and fits no scaling. Invalid input raises RuntimeError.
    """
    return _native.FrozenEncoderFeatures(image, weight, bias)


def FitTransferScaling(code, floor=1e-6):
    """Native detached population center/scale fitted from explicit code rows.

    Finite signed Float32/64 [B,L], B,L positive; accepts strides. Computes
    double row-order means, then a second-pass population variance, floors its
    square root and casts once to code dtype. Constant/singleton scales equal
    floor. Returns an owning TransferScaling with detached center[L]/scale[L].
    Caller chooses fitting rows; no labels or other splits are read. Invalid
    tensors, nonpositive/nonfinite/unrepresentable floor or accumulation
    overflow raise RuntimeError. No input mutation or random state.
    """
    return _native.FitTransferScaling(code, floor)


_native._set_doc(FrozenEncoderClassifier, """Native frozen ReLU encoder and caller-owned linear classifier.

Constructor(weight[F,L], bias[L], center[L], scale[L], classes=10) validates
finite matching Float32/64 arrays, positive scales/dimensions, and owns detached
deep copies. Getter methods export independent detached copies. Forward(head,
image[B,F], labels=undefined) creates/reuses linear/W[L,C] and linear/b[C] only.
Head dtype must match; optional Int32/64 labels[B] lie in [0,C). Returns owning
code, standardized features, logits, probability and loss (undefined without
labels). Loss is mean categorical log-softmax cross-entropy without clipping.
Input/head paths remain live; encoder/scaling paths are stopped. Default
L=32,C=10 gives two head leaves/330 scalars. Optimizer, fitting, split selection,
head seed and updates remain caller-owned. Invalid input raises RuntimeError
before head creation; incompatible existing leaves may fail later.
""")


def VisiblePatchValues(patches, hidden, fill=0):
    """Native target-free patch visibility, preserving observed bits exactly.

    Finite signed Float32/64 patches[B,P,D], all axes positive, and detached
    matching-dtype binary hidden[B,P]. Hidden patches become fill; visible
    values, including signed zero, are copied exactly. All-visible/all-hidden
    masks and strides are accepted. Float32 narrows finite fill once; overflow
    raises RuntimeError. Every input value must be finite, including unselected
    values. Owning output retains visible-input derivatives and zero hidden-input
    derivatives; no clipping, targets, model, fitting or random draws enter.
    """
    return _native.VisiblePatchValues(patches, hidden, fill)


def CompositePatchValues(observed, prediction, hidden):
    """Native patch composition: predictions at hidden sites, exact visible copies.

    Finite matching signed Float32/64 [B,P,D] arrays and a detached same-dtype
    binary hidden[B,P], all axes positive. One selects prediction, zero observed.
    Strides and all-visible/all-hidden masks are accepted; selected value bits,
    including signed zero, are preserved. Owns result lifetime and retains
    selected first/second continuous paths, with zero derivatives to unselected
    branches. No blending, clamping, targets or optimizer is hidden. All values
    must be finite, even in unselected branches. Invalid inputs raise RuntimeError.
    """
    return _native.CompositePatchValues(observed, prediction, hidden)


def WaeSquaredPixelCost(image, probability):
    """Native per-image pixel-summed squared cost, with both continuous paths.

    Finite matching nonempty Float32/64 [B,F] inputs lie in [0,1]. Returns
    owning [B] without pixel averaging, density interpretation or clipping.
    Strides and first/second derivatives remain live; invalid input raises RuntimeError.
    """
    return _native.WaeSquaredPixelCost(image, probability)


def WaeObjective(squared_cost, mmd, weight=10):
    """Native mean(squared_cost)+weight*mmd, retaining signed unbiased MMD.

    Cost is a finite nonnegative Float32/64 [B] vector and mmd is one finite
    matching-dtype value. Finite weight>=0 defaults to10; Float32 narrows it.
    Positive underflow/overflow and invalid tensors raise RuntimeError. Both
    paths stay live, including zero derivatives at weight0. This penalized
    objective is not an exact Wasserstein distance, image likelihood or ELBO.
    """
    return _native.WaeObjective(squared_cost, mmd, weight)


def WaeDecode(parameters, code, features=784, latent=4, hidden=16):
    """Native code-only tanh/linear decoder logits[B,features], without fitting.

    Finite signed matching Float32/64 code[B,latent], B>0; positive widths.
    Four decoder/output leaves contain13408 default scalars. Apply sigmoid
    for decoder values; no image, encoder, prior draw or pixel sampling enters.
    Strides and code/parameter derivatives remain live; invalid input raises RuntimeError.
    """
    return _native.WaeDecode(parameters, code, features, latent, hidden)


def WassersteinAutoencoder(parameters, image, prior, weight=10, features=784,
                            latent=4, hidden=16):
    """Native deterministic WAE-MMD; caller owns prior samples, batching and Adam.

    Finite matching Float32/64 image[B,features] in[0,1], prior[P,latent],
    B,P>=2 even when weight0; counts may differ. Default784/4/16 gives eight
    Glorot/zero-bias leaves and26036 scalars in encoder/code/decoder/output order.
    Returns six owning tensor fields and an owning nested seven-field mmd record.
    Squared pixel sums average over images; weight10 multiplies the signed
    unbiased average of IMQ scales1,4,16. Prior changes only the regularizer,
    not deterministic codes/reconstructions. Strides, Float64 and continuous
    gradients remain live. No posterior noise, KL, discriminator or hidden fit.
    Independence and split provenance are caller responsibilities. Invalid
    arguments raise RuntimeError before leaves; incompatible stored leaves can fail later.
    See Tolstikhin et al.(2018), https://arxiv.org/abs/1711.01558.
    """
    return _native.WassersteinAutoencoder(parameters, image, prior, weight,
                                          features, latent, hidden)


def VectorQuantize(encoded, book, entries=16, dimensions=4, override_indices=None):
    """Native nearest-vector lookup and separate surrogate/dictionary/commitment paths.

    Finite matching Float32/64 encoded[N,D], book[K,D], N,K,D>0. Lowest index
    wins ties. Optional Int32 N-element override replaces nearest assignments.
    Owns indices, one-hot assignment, quantized, straight_through and two scalar
    element-mean penalties. Reconstruction through straight_through has identity
    encoded-only surrogate gradients; codebook_loss is book-only and commitment
    is encoded-only. These are not derivatives of hard argmin. Strides accepted;
    signed zero may normalize. Invalid inputs/products/distance overflow raise
    RuntimeError; no EMA, RNG, resets or fitting occurs.
    """
    if override_indices is None:
        return _native.VectorQuantize(encoded, book, entries, dimensions)
    return _native.VectorQuantize(encoded, book, entries, dimensions, override_indices)


def VqLookup(ids, book, entries=16):
    """Native valid Int32[B,T] grid to floating[B*T,D] dictionary lookup.

    Nonempty finite Float32/64 book[K,D], K=entries; accepts strides, owns its
    result and retains book gradients. Invalid IDs, dtype, shape or products
    raise RuntimeError. No encoder or target enters.
    """
    return _native.VqLookup(ids, book, entries)


def VqDecoder(parameters, codes, height=28, width=28, dimensions=4, channels=8):
    """Native code-only repeat/Conv3-ReLU/repeat/Conv3-linear Bernoulli logits.

    Finite floating[B*T,D] matching store dtype, T=H*W/16; H,W positive multiples
    of four. Four shared decoder1/output leaves (369 default scalars). Owns
    [B,H*W] with continuous code/parameter paths; apply sigmoid for probabilities.
    No image, dictionary, fit or noise draw. Invalid inputs/products raise
    RuntimeError before leaf creation; incompatible existing leaves fail later.
    """
    return _native.VqDecoder(parameters, codes, height, width, dimensions, channels)


def VectorQuantizedAutoencoder(parameters, image, entries=16, dimensions=4,
                               channels=8, commitment_beta=.25):
    """Native spatial VQ-VAE: Conv/ReLU/avg2 twice, learned vectors and code-only decoder.

    Finite unit-range Float32/64 grayscale[B,H,W,1], positive H,W divisible by4.
    Defaults K16,D4,C8 retain nine leaves/805 scalars, source creation order and
    codebook U(-.5,.5). Owns encoded, book, probability, per-image nll, scalar
    reconstruction_loss/loss and nested six-field vq. loss=mean(pixel-summed
    NLL)+element-mean codebook square+beta*element-mean commitment square.
    Reconstruction follows the encoded-only surrogate and skips dictionary.
    Strides accepted, intended continuous paths live; no EMA, resets, prior,
    optimizer or hidden fit. Invalid inputs/options/products raise RuntimeError
    before creation; incompatible leaves or computed overflow can fail later.
    """
    return _native.VectorQuantizedAutoencoder(parameters, image, entries, dimensions,
                                              channels, commitment_beta)


def Vq2TopDecoder(parameters, top, height=28, width=28, dimensions=4):
    """Native coarse code vectors[B*H*W/16,D] to conditioning[B,H/2,W/2,D].

    Finite matching Float32/64, H/W positive multiples of4; strides accepted.
    Shared top_decoder Conv3/ReLU has two default leaves/148 scalars. Owns all
    continuous code/leaf paths; no image, target, dictionary, RNG or update.
    Invalid shapes/dtypes/products raise RuntimeError before leaf creation.
    """
    return _native.Vq2TopDecoder(parameters, top, height, width, dimensions)


def Vq2Decoder(parameters, bottom, top_feature, channels=8):
    """Native fine vectors plus coarse-derived conditioning to pixel logits.

    Finite matching Float32/64 bottom[B*h*w,D], top_feature[B,h,w,D], even h,w.
    Conv3(2D,C)/ReLU,repeat2,Conv3(C,1) gives owning[B,4*h*w] logits; apply sigmoid
    for probabilities. Four shared decoder/output leaves,657 default scalars.
    Both continuous input paths and all leaf paths remain live; no image enters.
    Invalid inputs/products raise RuntimeError; incompatible leaves fail later.
    """
    return _native.Vq2Decoder(parameters, bottom, top_feature, channels)


def VectorQuantizedAutoencoder2(parameters, image, top_indices=None, entries=16,
                                dimensions=4, channels=8, commitment_beta=.25):
    """Native two-scale VQ-VAE with explicit top conditioning and two dictionary routes.

    Unit-range matching Float32/64 grayscale[B,H,W,1], H/W positive multiples
    of4. K16,D4,C8 defaults retain source14 leaves/1157 scalars and creation
    order. Owns nine Tensor endpoints and two nested six-field quantizers;
    fine encoding and final decoding both use the coarse code-derived feature.
    Optional valid Int32 top assignments (B*H*W/16 elements) override lookup
    while retaining the encoder surrogate. No continuous image bypass reaches
    decoding. loss=mean(pixel-summed NLL)+(top_dictionary+bottom_dictionary)+
    beta*(top_commitment+bottom_commitment), each penalty averaged at its own
    resolution. Reconstruction skips both books; fine losses can reach coarse
    encoding. No EMA, resets, fit, noise bank or optimizer is hidden. Strides
    and intended continuous/surrogate higher paths remain; invalid inputs,
    options or products raise RuntimeError before leaves, incompatible leaves
    or computed overflow can fail later. Caller owns Adam and prior fitting.
    """
    if top_indices is None:
        top_indices = _native.Tensor()
    return _native.VectorQuantizedAutoencoder2(parameters, image, top_indices, entries,
                                               dimensions, channels, commitment_beta)


Vq2ConditionalPrior = _native.Vq2ConditionalPrior
_native._set_doc(Vq2ConditionalPrior, """Native copied fine raster prior conditional on aligned top codes.

Vq2ConditionalPrior(top,bottom,top_height=7,top_width=7,entries=16,smoothing=.5)
requires valid Int32[B,Ht*Wt] coarse and Int32[B,4*Ht*Wt] fitting-only fine codes.
For each fine raster position, use its two-by-two aligned coarse parent and
previous fine symbol; BOS=K resets per image. counts()/probabilities() export
independent Int32/Float64[K,K+1,K] tables. Nll(top,bottom) averages complete-grid
conditional nats; it is not image likelihood. Sample(top,uniforms) takes explicit
Float64[B,4*Ht*Wt] in[0,1), strict inverse-CDF boundaries and final-code fallback.
No RNG/model update; caller enforces split provenance. Read-only methods are
thread-safe. Invalid options/IDs/geometry/uniforms/count overflow raise RuntimeError.
This parent/previous-symbol half-count teaching baseline is not PixelCNN.
""")


def VqUsage(ids, entries=16):
    """Native owning Int32[K] counts and pooled exp(entropy) usage perplexity.

    Nonempty valid Int32 tensor of any shape, with count total fitting Int32.
    Dead codes remain visible. No likelihood/semantic interpretation. Invalid
    IDs/dtype/vocabulary/count overflow raise RuntimeError.
    """
    return _native.VqUsage(ids, entries)


VqMarkovPrior = _native.VqMarkovPrior
_native._set_doc(VqMarkovPrior, """Native copied first-order raster prior, fitted only to caller-supplied codes.

VqMarkovPrior(fitting, entries=16, smoothing=.5) copies nonempty valid Int32[B,T]
codes. BOS=K resets at each row; positive half-count smoothing is the default.
counts()/probabilities() export independent Int32/Float64[K+1,K] tensors.
Nll(grid) returns mean nats per grid, not per token or image likelihood.
Sample(uniforms) takes explicit Float64[B,T] in[0,1), without RNG, with strict
inverse-CDF boundaries and last-code fallback. State is immutable to readers;
invalid input/options/count overflow raise RuntimeError. This teaching Markov
baseline is not the paper's PixelCNN. Caller enforces fitting-only provenance.
""")


def AaeEncode(encoder, image, features=784, latent=4, hidden=16):
    """Native deterministic tanh/linear encoder with hidden/W,b and code/W,b.

    Finite matching Float32/64 unit-range image[B,features], positive widths.
    Owns signed code[B,latent] and all input/parameter gradients. Default12628
    scalars; no posterior noise, KL, normalization or fit. Invalid inputs fail
    before creation; incompatible leaves can fail later (RuntimeError).
    """
    return _native.AaeEncode(encoder, image, features, latent, hidden)


def AaeDecode(decoder, code, features=784, latent=4, hidden=16):
    """Native code-only tanh/linear Bernoulli decoder; returns logits, not probabilities.

    Finite matching Float32/64 code[B,latent], positive widths, signed/zero codes.
    Owns [B,features] with code/parameter gradients. Default13408 scalars in
    hidden/W,b and output/W,b. No image, target, discriminator or hidden fitting.
    Same validation/error timing as AaeEncode; apply sigmoid for probabilities.
    """
    return _native.AaeDecode(decoder, code, features, latent, hidden)


def AaeDiscriminate(discriminator, code, latent=4, hidden=16):
    """Native tanh/linear latent discriminator returning prior/real log-odds[B,1].

    Finite matching Float32/64 code[B,latent], positive widths. Creates/reuses
    hidden/W,b and logit/W,b (97 default scalars), retaining code/parameter
    gradients and ownership. No labels, detach, loss or optimizer is hidden.
    Invalid input fails before creation; incompatible leaves can fail later.
    """
    return _native.AaeDiscriminate(discriminator, code, latent, hidden)


def AdversarialAutoencoder(encoder, decoder, discriminator, image, prior,
                           features=784, latent=4, hidden=16):
    """Canonical deterministic AAE with three explicit losses and no hidden update.

    Distinct matching floating stores; finite unit-range image[B,features] and
    finite prior[P,latent], B,P>0 (P may differ). Owns code, logits, probability,
    prior_logit, encoded_logit, per-row nll and scalar reconstruction_loss,
    discriminator_loss, adversarial_loss. Reconstruction sums pixels then averages
    rows; D labels prior real and codes fake; E uses non-saturating real labels.
    All mathematical paths remain, including E gradients of D loss. Caller chooses
    requested groups, one reconstruction gradient snapshot, later phase recomputes
    and four independent Adam states. Default12 leaves/26133 scalars. No posterior
    sampling, per-example KL or ELBO. Prior changes do not alter codes or pixels.
    Invalid inputs/aliased stores fail before creation; incompatible leaves and
    nonfinite computed logits may fail later, raising RuntimeError.
    """
    return _native.AdversarialAutoencoder(encoder, decoder, discriminator, image, prior,
                                         features, latent, hidden)


def VaeGanFeatureNll(real, reconstruction):
    """Native half-summed feature residual [B] with a detached real target.

    Finite matching nonempty Float32/64 [B,D]; coordinates are summed, with no
    Gaussian constant or pixel-likelihood interpretation. Owns its result and
    reconstructed-feature gradients; real-target gradients stop at every order.
    Invalid shapes/dtypes/nonfinite values raise RuntimeError.
    """
    return _native.VaeGanFeatureNll(real, reconstruction)


def VaeGanDecode(generator, code, features=784, latent=4, hidden=16):
    """Native code-only tanh/sigmoid generator; owning probabilities, not draws.

    Finite Float32/64 [P,latent] matching generator dtype, positive widths.
    Retains code/leaf gradients with hidden/W,b and output/W,b. Four default
    leaves/13408 scalars; no posterior, RNG, fit or optimizer. Invalid inputs,
    widths/products or incompatible leaves raise RuntimeError.
    """
    return _native.VaeGanDecode(generator, code, features, latent, hidden)


def VaeGanDiscriminator(discriminator, image, features=784, feature_width=8):
    """Native tanh feature [P,D] and real/fake logit [P,1], with owning fields.

    Finite matching Float32/64 nonempty rows, signed inputs and strides allowed.
    Four default leaves/6289 scalars, named feature/W,b and logit/W,b. No labels,
    clipping, detach, sigmoid or fit. Invalid geometry/input/leaves raises
    RuntimeError; fields retain input/parameter gradients after record GC.
    """
    return _native.VaeGanDiscriminator(discriminator, image, features, feature_width)


def VaeGan(encoder, generator, discriminator, image, epsilon, prior,
           features=784, latent=4, hidden=16, feature_width=8):
    """Native nineteen-field Gaussian VAE-GAN with three explicit objectives.

    Distinct same-dtype Float32/64 stores; unit-range image[B,F], finite explicit
    posterior noise[B,L] and independent prior[P,L]. P may differ from B.
    Encoder loss is mean(feature NLL+KL); generator adds balanced non-saturating
    fake-as-real losses; discriminator separates real from both fake branches.
    The real feature target alone is detached. Pixel NLL uses source Float32
    clipping endpoints and is diagnostic, not a fitted pixel ELBO. Owns all
    fields through GC and retains continuous paths. Select only D/E/G leaves
    for their own losses and recompute after each caller-owned optimizer step.
    No RNG, fitting, mode change or freezing is hidden. Defaults:14 leaves,
    32393 scalars. Invalid inputs/options/stores/leaves raise RuntimeError.
    """
    return _native.VaeGan(encoder, generator, discriminator, image, epsilon, prior,
                          features, latent, hidden, feature_width)


def MidpointUniformNoise(shape, seed):
    """Native owning Float64 uniforms (raw32+.5)/2**32 from a fresh local MT19937.

    Exactly one word per element, logical row order, unsigned32 seed. No shared
    RNG, distribution adapter, clipping or history. Empty axes and scalar shape
    are supported. Values stay strictly inside(0,1) in Float64; a Float32 cast
    can round up to1. Useful for explicit SequenceGenerate categorical draws.
    Invalid shape/count raises RuntimeError; independent calls repeat exactly.
    """
    return _native.MidpointUniformNoise(shape, seed)


def prepare_example_data(name, directory=""):
    """Call canonical datamunge::datasets native preparation, returning an absolute cache.

    sequence-vae-assets produces the complete local432-sentence grammar below
    sequence_vae/. The directory is its shared parent. No Python preparation,
    fitting or hidden substitute data. Existing supplied files are retained with
    unverified provenance for the procedural sentence adapter.
    inception-metric prepares the original TensorFlow 2015 ImageNet archive,
    unmodified GraphDef, license and provenance under inception_metric by default.
    Archive and graph bytes/SHA256 are pinned; completed graph caches are rehashed
    on every call. No TensorFlow runtime or synthetic features. FrozenGraph uses
    Float32 RGB ExpandDims:0 inputs in0..255,2048 features,1008 classes. The
    caller explicitly chooses classifier bias policy. See assets.hpp for all adapters.
    """
    return _native.prepare_example_data(name, directory)


def SequenceVocabulary():
    """Independent mutable copy of the nineteen words, PAD0/BOS1/EOS2 first."""
    return list(_native.SequenceVocabulary())


def LoadSequenceVaeData(root):
    """Native exhaustive grammar/schema validation; owning fit/validation/test pool list.

    Reads sentences.npz and exact vocabulary.txt. Verifies all432 unique rows,
    factors, IDs, held-out splits, EOS/PAD, previous-target inputs and balance.
    Preserves valid supplied row permutations. Each pool owns tokens, inputs,
    lengths, factors, source after the returned list is collected. No preparation,
    shuffling or fitting. Invalid data/I/O raises RuntimeError.
    """
    return list(_native.LoadSequenceVaeData(str(root)))


def SequenceDataArrays(pools):
    """Native fully checked fifteen experiment arrays as owning (name, Tensor) pairs.

    Requires exactly fit/validation/test pools with the complete valid grammar;
    config is omitted. Tensor storage is shared, row order preserved. No I/O,
    hidden copies, fitting or RNG. Invalid pools raise RuntimeError.
    """
    if not isinstance(pools, _native.SequenceVaePools):
        vector = _native.SequenceVaePools()
        for pool in pools:
            vector.push_back(pool)
    else:
        vector = pools
    return [(name, value) for name, value in _native.SequenceDataArrays(vector)]


def SequenceText(tokens, row):
    """Native selected integer-row display through first EOS, including [eos].

    Retains [pad]/[bos], and all words if no EOS. Ignores suffix after EOS.
    Prefix IDs0..18, positive integer[B,T], zero-based row; invalid input raises RuntimeError.
    """
    return _native.SequenceText(tokens, row)


def SequenceGrammarValid(tokens, row):
    """Native fixed-grammar prefix test through first EOS; later suffix ignored.

    Malformed/missing-EOS prefixes return False. Invalid matrix/row raises
    RuntimeError. This preserves the original prefix diagnostic; generated PAD
    suffixes are a separate invariant. No mutation or fitting.
    """
    return _native.SequenceGrammarValid(tokens, row)


def SequenceEncoder(parameters, tokens):
    """Native PAD-masked GRU16 encoder, separate embedding8/vocabulary19.

    Int32/64[B,T] IDs0..18; positive B,T, including extra PAD columns. Float32/64
    store; owns state[B,16] and states[B,T,16], with live parameter paths. Seven
    leaves/1352 scalars. PAD preserves state; no EOS parsing or hidden fitting.
    Invalid inputs raise RuntimeError before leaves; stored incompatibility may fail later.
    """
    return _native.SequenceEncoder(parameters, tokens)


def SequenceDecoderInputs(tokens):
    """Native owning Int32[B,7] BOS1-then-previous-target prefixes.

    Int32/64 targets[B,7], IDs0..18; no target leakage, mutation or gradients.
    The final prefix of a short sentence is EOS2 before its PAD0 target.
    Invalid shape/dtype/IDs raise RuntimeError.
    """
    return _native.SequenceDecoderInputs(tokens)


def SequenceTeacher(parameters, z, inputs):
    """Native causal prefix-only decoder, without targets or an encoder.

    Finite matching Float32/64 z[B,4]; Int32/64 prefixes[B,7], IDs0..18. Owns
    initial, states, logits, probability, predicted_ids. Eleven leaves/1755
    scalars. Future prefixes cannot affect earlier scores at fixed z. All
    continuous z/parameter paths stay live; no fit, hidden draws or EOS stopping.
    RuntimeError on invalid inputs.
    """
    return _native.SequenceTeacher(parameters, z, inputs)


def SequenceVariationalAutoencoder(parameters, tokens, epsilon):
    """Native explicit-noise sentence VAE with eighteen owning endpoints.

    Int32/64 targets[B,7], IDs0..18; finite matching Float32/64 noise[B,4].
    Separate embeddings8, masked encoder GRU16, Gaussian4 and causal decoder
    GRU16 initialized from z. Source22 leaves/3243 scalars. NLL includes EOS,
    masks PAD and sums time; loss=mean(NLL+KL). No grammar parsing, optimizer,
    RNG beyond lazy initialization or hidden warmup. BPTT/noise paths remain live.
    Inputs are validated before leaves; invalid input raises RuntimeError.
    """
    return _native.SequenceVariationalAutoencoder(parameters, tokens, epsilon)


def AnnealedSequenceObjective(net, beta):
    """Native mean(net.nll+beta*net.kl), with finite beta in[0,1].

    Reads finite matching Float32/64 vectors only; both paths stay connected,
    with zero KL derivative at beta0. Source scalar rounding, no clipping or fit.
    Returns an owning scalar; invalid vectors or beta raise RuntimeError.
    """
    return _native.AnnealedSequenceObjective(net, beta)


def SequenceGenerate(parameters, z, sample=False, uniforms=None):
    """Native target-free seven-step greedy or explicit-uniform generation.

    Finite matching Float32/64 z[B,4]. Sampling requires Float64[B,7] draws
    strictly between0 and1; greedy ignores uniforms. CDF ties count cdf<=u.
    Starts BOS1, emits PAD0 after EOS2, while all raw scores still evolve.
    Owns ids, logits, probability, ended. Scores retain continuous paths for
    fixed hard prefixes; IDs have no straight-through gradient. No hidden draws
    or NoGrad guard. Same11 decoder leaves/1755 scalars. RuntimeError on invalid inputs.
    """
    return _native.SequenceGenerate(parameters, z, sample, Tensor() if uniforms is None else uniforms)


def CollapseBeta(warmup, step, ramp_steps=18):
    """Native Float32 min(1,step/(ramp_steps-1)), or constant1.

    Explicit zero-based step>=0 and ramp>=2; no stored counter or optimizer.
    Ramp18 reaches1 at step17. Invalid arguments raise RuntimeError.
    """
    return _native.CollapseBeta(warmup, step, ramp_steps)


def CollapseStatistics(mu, log_variance, samples, active_threshold=.01):
    """Native host-double empirical-mixture posterior diagnostics.

    Finite matching Float32/64[N,D], N>1,D>0; nonempty aligned draws of the
    same shape/dtype. Owns per-unit KL/mean variances, analytic and sampled KL,
    mixture information, aggregate KL, and active-unit count using strict
    variance>threshold. Includes each row's own posterior: finite-sample bias,
    not population MI or a semantic score. Signed MC estimates are retained.
    No gradients, mutation or hidden draws. O(S*N*N*D) work. Invalid inputs or
    nonfinite computed diagnostics raise RuntimeError.
    """
    return _native.CollapseStatistics(mu, log_variance, samples, active_threshold)


def LadderPrecisionFusion(evidence_mu, evidence_lv, prior_mu, prior_lv):
    """Native Gaussian precision fusion with four live evidence/prior paths.

    Finite nonempty matching Float32/64[B,L], strides accepted. Owns mu and
    log_variance. Uses a detached maximum shift and source-rounded weights;
    no raw precision exponentials, residual-head substitution or clipping.
    Equal distributions halve variance, up to rounding. First/second paths stay
    live; extreme-log cancellation is not repaired. Invalid inputs raise RuntimeError.
    """
    return _native.LadderPrecisionFusion(evidence_mu, evidence_lv, prior_mu, prior_lv)


def LadderPrior(parameters, top, lower_noise, features=784, lower_latent=4, top_latent=2,
                hidden=32, top_hidden=16):
    """Native image-free ancestral p(z1|z2), with explicit top codes and lower noise.

    Finite matching Float32/64 top[B,T], lower_noise[B,L], positive widths.
    Owns top_z, lower_noise, prior_mu, prior_lv, lower_z, probability. All input
    and parameter paths stay live. Only lower_z enters the ReLU decoder; top
    affects it through the conditional prior. Default ten leaves/26216 scalars
    if created alone. No encoder, image, RNG draws, fit or hidden no-grad.
    GaussianVaeDecode returns matching lower-code-only logits. RuntimeError on invalid inputs.
    """
    return _native.LadderPrior(parameters, top, lower_noise, features, lower_latent,
                               top_latent, hidden, top_hidden)


def LadderVariationalAutoencoder(parameters, image, epsilon, features=784, lower_latent=4,
                                 top_latent=2, hidden=32, top_hidden=16):
    """Native two-level Ladder VAE, preserving precision fusion and explicit noise order.

    Unit-range finite Float32/64 image[B,F], matching epsilon[B,L+T], B>0.
    LOWER noise first, TOP noise last. Twenty-two default leaves/52196 scalars;
    all sixteen endpoints own lifetime. Top noise changes the learned lower
    prior and fused posterior; lower noise changes only the lower sample and
    decoder. Only lower_z enters GaussianVaeDecode. loss=mean(NLL+(lowerKL+topKL));
    lower conditional KL depends on sampled top codes. No BN, warmup, free bits,
    clipping, optimizer or checkpoint selection. Caller owns draws/Adam/selection.
    All image/noise/parameter paths stay live. Invalid inputs raise RuntimeError.
    """
    return _native.LadderVariationalAutoencoder(parameters, image, epsilon, features,
                                                 lower_latent, top_latent, hidden, top_hidden)


def ImqKernel(x, y):
    """Native equally averaged c/(c+distance_squared), c=1,4,16, for all sample pairs.

    Finite matching Float32/64[N,D] and[M,D], N,M,D positive; unequal counts and
    strides accepted. Owns[N,M] with live first/second derivatives through both
    samples. Work/storage O(N*M*D). No fitted bandwidth, dimension normalization
    or clipping. RuntimeError on invalid inputs/products; computed overflow is not repaired.
    """
    return _native.ImqKernel(x, y)


def ImqMmd(x, y):
    """Native signed unbiased IMQ MMD squared, retaining all seven owning endpoints.

    Finite matching Float32/64[N,D],[M,D], N,M>=2, unequal counts/strides allowed.
    Owns xx,yy,xy,within_x,within_y,cross,value with both sample paths live.
    Within diagonals only are excluded; all cross pairs count. Negative estimates
    and negative self-comparisons remain visible. No sqrt or calibrated test.
    Work/storage O((N*N+M*M+N*M)*D). Invalid inputs raise RuntimeError.
    """
    return _native.ImqMmd(x, y)


def InfoVaeObjective(nll, kl, mmd, alpha=.5, lambda_weight=10):
    """Native mean(NLL+(1-alpha)*KL)+(alpha+lambda_weight-1)*signed_MMD.

    Finite matching Float32/64 vectors[B] and one same-dtype finite MMD value;
    signed values/strides accepted, scalar owns lifetime and all three paths.
    alpha in[0,1], lambda_weight>=1. Float32 coefficients preserve source rounding;
    Float64 keeps precision. Positive alpha underflow/lambda overflow is rejected.
    alpha1 removes local KL, alpha0/lambda1 gives diagnostic negative ELBO.
    General objectives are not ordinary ELBOs or mutual-information estimates.
    """
    return _native.InfoVaeObjective(nll, kl, mmd, alpha, lambda_weight)


def InfoVaeDecode(parameters, code, features=784, latent=4, hidden=16):
    """Native image-free tanh/linear InfoVAE decoder logits, not sigmoid probabilities.

    Finite matching Float32/64[B,latent], positive widths. Four decoder/output
    leaves reuse the full model; default13408 scalars. Signed/zero codes and live
    code/weight gradients accepted. No encoder, MMD, noise, fitting or optimizer.
    Invalid inputs/products fail before leaves; incompatible stored leaves can fail later.
    """
    return _native.InfoVaeDecode(parameters, code, features, latent, hidden)


def InfoVariationalAutoencoder(parameters, image, epsilon, prior, alpha=.5, lambda_weight=10,
                               features=784, latent=4, hidden=16):
    """Native Gaussian/tanh InfoVAE with explicit posterior noise and independent prior.

    Unit-range finite image[B,F], matching noise[B,L] and prior[P,L], B,P>=2.
    Unequal counts/strides and Float32/64 accepted. Owns all sixteen fields;
    full image/noise/prior/parameter paths remain live. Ten default leaves/26104
    scalars. Aggregate IMQ MMD uses sampled z, not means; negative estimates stay
    signed. loss and diagnostic negative_elbo differ. Prior affects only MMD/loss.
    Caller owns Adam, statistical independence of prior draws and epoch selection.
    No fit, new draws or hidden optimizer. RuntimeError on invalid inputs/controls.
    """
    return _native.InfoVariationalAutoencoder(parameters, image, epsilon, prior, alpha,
                                              lambda_weight, features, latent, hidden)


def IwaeLogMeanExp(log_weights):
    """Native sample-axis log-mean-exp of finite Float32/64 log weights[K,B].

    Owns[B] nats with full first/second derivatives. Uses a stopped column maximum
    and source-rounded Float32 log(K); Float64 keeps full precision. Positive K,B,
    strided matrices accepted. Invalid inputs raise RuntimeError. No RNG or fitting.
    """
    return _native.IwaeLogMeanExp(log_weights)


def ImportanceWeightedBound(mu, log_variance, z, nll, samples):
    """Native sampled Gaussian importance bound, normalized weights and ESS.

    Finite matching Float32/64 Gaussian[KB,L], nll[KB], positive samples=K dividing
    rows. Signed nll accepted. Owns log_weights[K,B], normalized_weights[K,B],
    negative_bound[B], effective_samples[B], scalar loss. Ordinary full pathwise
    derivatives through all four inputs; no analytic-KL substitution or DReG.
    Source log constants for Float32, full precision for Float64. RuntimeError
    for invalid inputs; computed overflow is not repaired. Inputs unchanged.
    """
    return _native.ImportanceWeightedBound(mu, log_variance, z, nll, samples)


def ImportanceWeightedAutoencoder(parameters, image, epsilon, features=784, latent=4, hidden=32):
    """Native Gaussian/Bernoulli IWAE with explicit noise and caller-owned updates.

    Unit-range finite image[B,F], matching finite Float32/64 epsilon[K,B,L].
    Tiles sample-major and reuses Gaussian VAE encoder/decoder names, ten leaves
    and51416 default scalars. Owns all eleven endpoints through collection; all
    parameter/image/noise gradients stay live. kl is diagnostic only; loss uses
    sampled log p(x,z)-log q(z|x). K=1 is sampled negative ELBO, not analytic KL.
    GaussianVaeDecode performs image-free final-code decoding. No fit, noise draw,
    optimizer or checkpoint selection is hidden. RuntimeError on invalid inputs.
    """
    return _native.ImportanceWeightedAutoencoder(parameters, image, epsilon, features, latent, hidden)


def HierarchyPosterior(prior_mu, prior_lv, head_mu, head_lv, residual=True):
    """Native direct Gaussian heads or additive residual corrections to a learned prior.

    Four finite nonempty matching Float32/64[B,L] matrices, with strides. Returns
    owning mu/log_variance; residual=True adds each head to its prior, False uses
    heads directly. All applicable paths stay live; direct mode has no prior path.
    Zero residuals reproduce the prior exactly. No clipping, RNG or precision fusion.
    """
    return _native.HierarchyPosterior(prior_mu, prior_lv, head_mu, head_lv, residual)


def ConditionalGaussianKl(mu, lv, prior_mu, prior_lv):
    """Native analytic diagonal-Gaussian KL, summing latent coordinates per row.

    Four finite matching Float32/64[B,L] matrices. Owns[B] nats; all four gradients
    and strides retained. Identical distributions yield exact zero; negative
    roundoff is not clipped. Source parenthesization is retained; exp overflow is
    not repaired. No sampling or averaging over coordinates.
    """
    return _native.ConditionalGaussianKl(mu, lv, prior_mu, prior_lv)


def HierarchyNvaeCell(parameters, state, x, training=False, update_stats=False):
    """Native miniature NVAE residual cell with separate trainable and moving-statistic stores.

    Finite matching Float32/64 NHWC[B,H,W,C]; bias-free expansion C->2C, BN/Swish,
    depthwise3x3, BN/Swish, projection C, BN, spatial-mean SE gate; x+.1*project*gate.
    Source Glorot initialization, BN momentum.9/epsilon.001; C4 gives13 leaves/198
    scalars plus six buffers/40 scalars. Training uses population batch moments;
    update_stats=True also moves mean/unbiased variance and requires training.
    Inference detaches moving buffers; initial calls may create missing entries.
    Caller excludes state from gradients/Adam and checkpoints BOTH stores. Existing
    weights never change; existing state changes only on explicit update passes.
    Owning live outputs, no hidden optimizer, noise or fitting. Invalid inputs and
    aliased/mismatched stores fail before creation; incompatible leaves may fail later.
    """
    return _native.HierarchyNvaeCell(parameters, state, x, training, update_stats)


def HierarchyDecode(parameters, state, lower, top, nvae=False, height=28, width=28, channels=4):
    """Native image-free Bernoulli decoding of BOTH explicit hierarchy latent groups.

    Finite matching Float32/64 lower[B,Hq*Wq], top[B,T], T>0. Safe positive
    multiples-of-four geometry, separate matching stores. Top projection/cell plus
    lower map feeds repeat/convolution decoding, without an image/encoder bypass.
    NVAE uses detached moving statistics; restore BOTH stores for trained inference.
    Initial calls can create missing entries; existing values are unchanged. Owning
    probabilities[B,H*W] retain code/parameter gradients; no no-grad scope/RNG/fitting.
    """
    return _native.HierarchyDecode(parameters, state, lower, top, nvae, height, width, channels)


def HierarchyPrior(parameters, state, top, lower_noise, nvae=False, height=28, width=28, channels=4):
    """Native explicit ancestral draw: supplied top z2, learned p(z1|z2), supplied lower epsilon.

    Owns top_z/lower_noise/prior_mu/prior_lv/lower_z/probability, with all continuous
    paths retained. Finite matching Float32/64[B,T] and[B,Hq*Wq], safe geometry and
    separate same-dtype stores. No image, encoder, posterior substitution or RNG.
    NVAE uses moving BN buffers; restore both stores. Lazy calls can create missing
    entries; existing values do not change. No hidden fit or no-grad scope.
    """
    return _native.HierarchyPrior(parameters, state, top, lower_noise, nvae, height, width, channels)


def HierarchicalVariationalAutoencoder(parameters, state, image, epsilon, nvae=False,
                                      training=False, update_stats=False, height=28,
                                      width=28, channels=4, top_latent=2):
    """Native spatial Gaussian hierarchy or miniature residual-posterior NVAE forward.

    Finite unit-range Float32/64 image[B,H*W], matching supplied noise[B,Hq*Wq+T]:
    lower noise first, top noise last. Both sampled groups reach decoding. Default
    ordinary26 leaves/1977 scalars; NVAE37/2027 plus six BN buffers/40 scalars.
    Owns19 fields. loss=mean(pixel NLL+(conditional lower KL+top KL)); NVAE train_loss
    adds1e-4 times the TWO smoothed rank-one prior-kernel norms (epsilon1e-12).
    This is a tiny teaching adaptation, without full NVAE's all-kernel schedule.
    Explicit training/update_stats flags separate batch moments from moving updates;
    extra gradient passes use training=True,update_stats=False. Caller owns Adam,
    excludes state from gradients and checkpoints BOTH stores. Inference detaches
    moving buffers. No noise, optimizer, fitting or selection is hidden. Geometry/
    dtype/batch/store validation precedes leaf creation; all continuous paths remain.
    """
    return _native.HierarchicalVariationalAutoencoder(parameters, state, image, epsilon,
        nvae, training, update_stats, height, width, channels, top_latent)


def GumbelCategorical(logits, noise, temperature=.7):
    """Native explicit Gumbel-softmax draw and independent hard Gumbel-max indices.

    Finite matching Float32/64 logits[B,C], noise[M,C], C>=2; rows match or one is1.
    Owns log_probability/probability[B,C], relaxed[max(B,M),C], Int64 indices.
    Positive representable temperature is narrowed for Float32. Hard samples and
    posterior q ignore temperature; relaxed values/gradients do not. Both continuous
    inputs retain gradients, without a straight-through estimator or hidden RNG.
    Invalid inputs raise RuntimeError; computed overflow is not repaired.
    """
    return _native.GumbelCategorical(logits, noise, temperature)


def CategoricalUniformKl(probability, log_probability):
    """Native sum(q*(logq+log C),axis1), per-row nats; not a Concrete-density KL.

    Finite matching Float32/64[B,C], C>=2, q in[0,1]. Caller supplies normalized q
    and matching logq; consistency is not checked. Both gradients stay live; stable
    finite log-softmax accepts underflowed q=0. Float32 log C is source-rounded.
    Owning result, strides preserved, no clipping/sampling or hidden fitting.
    """
    return _native.CategoricalUniformKl(probability, log_probability)


def GumbelNoise(n, samples, seed, groups=4, classes=8):
    """Native detached explicit uniform/Gumbel banks, local full-uint32 MT19937 seed.

    Positive n/samples/groups, classes>=2, checked element/byte-count overflow.
    Returns owning uniform/gumbel Float64 and noise Float32[S,B,G,C]. Each engine
    word r gives u=(r+.5)/2**32 then -log(-log(u)), preserving open endpoints.
    No global RNG, distribution-library sampler or clipping. Model calls consume
    supplied noise and never draw implicitly.
    """
    return _native.GumbelNoise(n, samples, seed, groups, classes)


def GumbelDecodeSoft(parameters, simplex, features=784, groups=4, classes=8, hidden=32):
    """Native target-free grouped-simplex decoder, returning Bernoulli probabilities.

    Finite matching Float32/64[B,G,C] in[0,1], each group summing to1 within strict
    2e-6/1e-12 tolerance. Shares decoder/output leaves with the full model, default
    four leaves/26928 scalars. Owning output and all code/parameter gradients;
    no encoder, image target, RNG, no-grad scope or fitting. Checks precede creation.
    """
    return _native.GumbelDecodeSoft(parameters, simplex, features, groups, classes, hidden)


def GumbelDecodeIds(parameters, ids, features=784, groups=4, classes=8, hidden=32):
    """Native target-free Int32/64[B,G] IDs decoder; bounds are0<=ID<classes.

    Positive widths, classes>=2 fitting Int32 IDs, checked product width. One-hot
    codes match the floating store dtype; output owns lifetime and decoder gradients.
    No continuous derivative through IDs, encoder/image/RNG or fitting. Invalid
    codes fail before creating leaves. Shares all soft-decoder leaves.
    """
    return _native.GumbelDecodeIds(parameters, ids, features, groups, classes, hidden)


def GumbelCategoricalAutoencoder(parameters, image, noise, temperature=.7,
                                  features=784, groups=4, classes=8, hidden=32):
    """Native categorical VAE with a soft reconstruction plus analytic discrete-KL surrogate.

    Unit-range finite image[B,F], explicit finite matching noise[B,G,C], Float32/64.
    Eight default leaves/53104 scalars. Owns14 posterior/sample/decoder/objective
    fields, with Int32 IDs and same-dtype one-hots. loss=mean(soft NLL+discrete KL),
    a biased continuous surrogate, not an exact discrete ELBO or Concrete KL.
    hard_objective=mean(hard NLL+discrete KL) uses matched actual categorical draws
    as a Monte Carlo discrete negative ELBO diagnostic. No straight-through gradient.
    Temperature affects soft values/gradients, not q/KL/hard IDs. All continuous paths
    remain; caller owns noise, gradients, Adam, fitting and checkpoint selection.
    """
    return _native.GumbelCategoricalAutoencoder(parameters, image, noise, temperature,
                                                features, groups, classes, hidden)


def ConditionalAffinePosteriorFlow(parameters, name, code, context, swap=False, inverse=False,
                                   latent=4, hidden=16):
    """Native affine coupling with live context and code derivatives, owning four fields.

    Matching finite Float32/64 code[B,latent] and context[B,C], positive even latent,
    positive hidden/C and nonempty prefix. The unchanged half plus context feeds
    a Glorot ReLU/linear conditioner. Scale=.5*tanh(raw), shift unbounded; forward
    b*exp(scale)+shift, inverse (b-shift)*exp(-scale), signed sum-scale logdet.
    swap chooses which half is unchanged; output coordinate ordering is preserved.
    Context is held fixed for the code Jacobian. No detach, optimizer or fitting.
    Reuse the same prefix, context, swap and widths to invert. Invalid inputs
    raise RuntimeError before creating leaves; incompatible leaves may fail later.
    """
    return _native.ConditionalAffinePosteriorFlow(parameters, name, code, context,
                                                 swap, inverse, latent, hidden)


def FlowVariationalAutoencoder(parameters, image, epsilon, features=784, latent=4,
                               hidden=32, flow_hidden=16):
    """Native Gaussian VAE with two alternating conditional affine posterior couplings.

    Finite unit-range Float32/64 image[B,features], matching explicit noise[B,latent],
    positive widths/even latent. Default18 leaves/52672 scalars. Encoder context
    conditions both flows; ONLY final z reaches GaussianVaeDecode. Owns18 fields
    from context through inverse_logdet. logq=logq0-forward_logdet, logp is N(0,I)
    at final z, loss=mean(summed_pixel_nll+logq-logp). Signed sampled ratios remain;
    no analytic base KL replacement, fitting, RNG or checkpoint selection is hidden.
    All continuous parameter/image/noise paths stay live. Zero noise gives f(mu),
    not generally the flowed posterior mean. Prior sampling decodes standard-normal
    final codes directly, without context or posterior flow. Inverse fields are
    diagnostics with shared leaves; invalid inputs fail before leaf creation.
    """
    return _native.FlowVariationalAutoencoder(parameters, image, epsilon, features,
                                              latent, hidden, flow_hidden)


def EmpiricalTotalCorrelation(z, mu, log_variance):
    """Native exact finite equal-weight Gaussian-mixture log ratio, with all three gradients.

    Finite matching Float32/64 z[B,L], mu/log_variance[R,L], B,R,L>0. Returns
    owning log_joint[B],log_product[B],per_case[B],mean scalar. All references
    stay live; fitting-only provenance belongs to the caller. This is a full
    empirical mixture, not a minibatch-weighted estimator. Signed sample ratios
    are retained; no clipping, RNG or detach. Overflow is not repaired.
    """
    return _native.EmpiricalTotalCorrelation(z, mu, log_variance)


def FactorDiscriminator(parameters, z, latent=4, hidden=16):
    """Native tanh/linear density-ratio logits[B], positive=joint, negative=product.

    Matching finite nonempty Float32/64 codes[B,latent], positive widths; four
    default Glorot/zero-bias leaves hidden/W,b;ratio/W,b hold97 scalars. Owning
    output retains all code/parameter gradients; no detach, labels or hidden fit.
    """
    return _native.FactorDiscriminator(parameters, z, latent, hidden)


def FactorDiscriminatorLoss(parameters, joint, product, latent=4, hidden=16):
    """Native0.5*(mean softplus(-D(joint))+mean softplus(D(product))).

    Separate finite matching Float32/64 nonempty joint/product matrices may have
    different row counts. All mathematical paths remain; callers must detach
    posterior samples and update only D during a discriminator phase. No hidden
    optimizer or permutation. Input checks precede leaf creation (RuntimeError).
    """
    return _native.FactorDiscriminatorLoss(parameters, joint, product, latent, hidden)


def PermuteLatentColumns(joint, seed):
    """Native independent-column std::shuffle using one local explicit MT19937 seed.

    Finite nonempty Float32/64[B,L], with strides; owns detached same-shape/dtype
    rows preserving each column's multiset exactly. No global RNG or input change.
    Reproducibility requires the same C++ standard-library shuffle implementation.
    Finite shuffled rows are a product-marginal surrogate, not population draws.
    """
    return _native.PermuteLatentColumns(joint, seed)


def FactorVariationalAutoencoder(parameters, discriminator, image, epsilon,
                                 extra_weight=3, features=784, latent=4, hidden=32, disc_hidden=16):
    """Native base Gaussian VAE plus mean discriminator-logit TC surrogate.

    Returns owning net (all Gaussian endpoints/base losses),tc scalar,loss scalar:
    loss=net.unweighted_loss+extra_weight*tc. Default gamma3,51416 VAE scalars and
    97 discriminator scalars. Distinct same-dtype stores; finite unit-range images,
    explicit matching noise, positive widths, representable nonnegative weight.
    No implicit detach: VAE gradients pass through D to z while caller updates
    VAE leaves only. A separate D phase detaches fresh posterior samples. No RNG,
    fitting/selection/clipping is hidden; negative TC surrogates remain visible.
    """
    return _native.FactorVariationalAutoencoder(parameters, discriminator, image, epsilon,
                                                extra_weight, features, latent, hidden, disc_hidden)


def BetaTcVariationalAutoencoder(parameters, image, epsilon, reference_images,
                                 extra_weight=3, features=784, latent=4, hidden=32):
    """Native base Gaussian VAE plus exact full finite-reference mixture TC sample mean.

    All supplied reference_images[R,features] MUST be fitting rows; their shared
    encoder/head parameter paths stay live. No reference decoder or RNG. Default
    extra3 gives beta_TC4, with other decomposition weights1. Returns owning
    net,tc,loss=base NLL+analytic KL+extra_weight*tc. Matching finite Float32/64
    unit-range images/references, explicit matching noise, positive widths and
    nonnegative representable weight. No training/selection/detach is hidden.
    Signed estimates are retained. This replaces the paper's minibatch estimator
    with all provided equally weighted references and does not certify disentanglement.
    """
    return _native.BetaTcVariationalAutoencoder(parameters, image, epsilon, reference_images,
                                                extra_weight, features, latent, hidden)


def ConvolutionalVaeDecode(parameters, code, height=28, width=28, latent=2, channels=8, base_channels=4):
    """Native Gaussian-code-only spatial decoder, owning five differentiable endpoints.

    Dense/ReLU projection -> NHWC quarter-resolution base -> repeat2/He-ReLU
    convolution -> repeat2/Glorot-linear convolution. No image, target, label,
    encoder features or skips. Fields decoder_base,decoder1,spatial_logits,
    logits,probability; probabilities are flattened[B,H*W], not pixel draws.
    Default six leaves/957 scalars; projection uses Glorot even with ReLU.
    Finite signed/zero Float32/64 code[B,latent] matches store dtype; H/W are
    positive multiples of four and widths positive. Input checks precede leaf
    creation; incompatible leaves can fail later. No fitting/RNG/clipping.
    """
    return _native.ConvolutionalVaeDecode(parameters, code, height, width, latent, channels, base_channels)


def ConvolutionalVariationalAutoencoder(parameters, image, epsilon, latent=2, channels=8, base_channels=4):
    """Native pooled convolutional Gaussian VAE; explicit sampled z is the sole decoder bridge.

    Finite unit-range Float32/64 image[B,H,W,1], H/W positive multiples of four,
    explicit epsilon[B,latent] matching dtype/batch, positive widths. Two He-ReLU
    conv/average-pool stages -> linear Gaussian heads -> reparameterized sample
    -> independent repeat/convolution decoder. Default14 leaves/2117 scalars.
    Owns all nine GaussianVaeOutput fields plus conv1,pool1,conv2,pool2,
    decoder_base,decoder1,spatial_logits. All continuous gradients retained.
    loss=unweighted_loss=mean(summed-pixel NLL+summed-coordinate KL), beta1.
    Binary targets give negative ELBO; soft targets give cross-entropy, not a
    continuous-observation density. No skips, hidden RNG, fitting or selection.
    Decode(mu) is not the posterior predictive mean. Invalid input fails before
    creation; incompatible/nonfinite computed states may fail later (RuntimeError).
    """
    return _native.ConvolutionalVariationalAutoencoder(parameters, image, epsilon, latent, channels, base_channels)


def ConditionalVaeDecode(parameters, code, labels, features=784, classes=10, latent=2, hidden=64):
    """Native code+supplied-class ReLU/linear decoder; returns logits[B,features].

    No image, posterior, RNG or label prediction. Finite signed/zero Float32/64
    code[B,latent] matches store dtype; labels Int32/64[B] in [0,classes).
    Positive widths; creates/reuses decoder/W,b and output/W,b, matching the
    full conditional VAE. Owns output/code/leaf gradients. Invalid input fails
    before creation; incompatible leaves can fail later (RuntimeError).
    """
    return _native.ConditionalVaeDecode(parameters, code, labels, features, classes, latent, hidden)


def ConditionalVariationalAutoencoder(parameters, image, labels, epsilon,
                                     features=784, classes=10, latent=2, hidden=64):
    """Native q(z|image,class) and p(image|z,class) with a fixed shared N(0,I) prior.

    Concatenates image/one-hot class before ReLU+mu/log-variance heads and z/class
    before the ReLU/linear Bernoulli decoder. Explicit epsilon, no RNG or hidden
    fit. Twelve owning fields: all nine GaussianVaeOutput fields plus condition,
    encoder_input and decoder_input. loss=unweighted_loss=mean(sum_pixel NLL+
    sum_latent KL). Binary targets give Bernoulli NLL; soft targets give cross-entropy,
    not a continuous-observation density. Ten leaves/102932 default scalars.
    Finite unit-range Float32/64 image[B,features] and epsilon[B,latent] match
    store dtype; labels Int32/64[B] in [0,classes), widths positive. Input checks
    precede creation; incompatible/nonfinite computed states can fail later.
    All continuous gradients retained; conditions have none. Supplied classes
    remain required at inference; decode(mu) is not a posterior predictive mean.
    """
    return _native.ConditionalVariationalAutoencoder(parameters, image, labels, epsilon,
                                                     features, classes, latent, hidden)


def GaussianReparameterize(mu, log_variance, epsilon):
    """Native mu+exp(log_variance/2)*epsilon, with no hidden RNG or detach.

    Finite matching nonempty Float32/64 [B,L] inputs, including strides. Returns
    an owning [B,L] sample with all three gradients. Computed overflow is not
    repaired. Invalid shapes/dtypes/values raise RuntimeError; inputs unchanged.
    """
    return _native.GaussianReparameterize(mu, log_variance, epsilon)


def DiagonalGaussianKl(mu, log_variance):
    """Native analytic KL(N(mu,diag(exp(lv))) || N(0,I)), summed over coordinates.

    Finite matching Float32/64 nonempty [B,L] inputs. Owns differentiable [B]
    nats, without coordinate averaging or clipping; zero mu/lv gives exact zero.
    Invalid inputs raise RuntimeError. Computed overflow is not repaired.
    """
    return _native.DiagonalGaussianKl(mu, log_variance)


def GaussianVaeDecode(parameters, code, features=784, latent=4, hidden=32):
    """Native code-only ReLU/linear Gaussian-VAE decoder returning logits[B,features].

    Finite nonempty matching Float32/64 [B,latent] codes, positive widths.
    Creates/reuses decoder/W,b and output/W,b with Glorot weights/zero biases.
    Same names as BetaVariationalAutoencoder; apply sigmoid for probabilities.
    The vanilla teaching VAE uses features=784, latent=2, hidden=64.
    No images, targets or RNG enter. Owning output retains code/leaf gradients.
    Invalid inputs fail before leaf creation; incompatible leaves can fail later.
    """
    return _native.GaussianVaeDecode(parameters, code, features, latent, hidden)


def BetaVariationalAutoencoder(parameters, image, epsilon, beta=1, features=784, latent=4, hidden=32):
    """Native Gaussian VAE with explicit noise, nine owning fields and no hidden fit.

    ReLU encoder -> linear mu/log_variance heads -> pathwise z -> ReLU/linear
    Bernoulli decoder. Owns mu, log_variance, z, logits, probability,
    reconstruction_nll[B], kl[B], scalar loss=mean(NLL+beta*KL) and scalar
    unweighted_loss=mean(NLL+KL). Pixels and latent coordinates are summed.
    Ten default leaves/51416 scalars; ordinary Glorot weights/zero biases.
    beta=1, features=784, latent=2, hidden=64 reproduces the vanilla teaching
    architecture with ten leaves/101652 scalars and the original names/order.
    Decode(mu) is not the posterior predictive mean. All continuous gradients
    remain. Finite unit-range image[B,features] and epsilon[B,latent] must match
    store Float32/64; widths/beta positive, beta representable in store dtype.
    Invalid input fails before creation; incompatible/nonfinite stored/computed
    states may fail later (RuntimeError). Caller owns noise/optimizer/selection.
    """
    return _native.BetaVariationalAutoencoder(parameters, image, epsilon, beta, features, latent, hidden)


def LatentFactorDiagnostics(fit_mu, test_mu, fit_factors, test_factors, fit_log_variance, ridge=.001):
    """Return13 owning native arrays for fitting-only standardized ridge/single-axis probes.

    Means/log variances are finite Float32/64 matrices; numeric factors have matching
    rows. Best axes and target/latent centers use fitting rows only. Test R2 can be
    negative and requires variable targets. No training labels enter the VAE. Native
    C++ performs every calculation; these probes do not certify disentanglement.
    """
    return [(name, value) for name, value in _native.LatentFactorDiagnostics(
        fit_mu, test_mu, fit_factors, test_factors, fit_log_variance, ridge)]


def KnownFactorLattice():
    """Native exhaustive 294-image circle/square fixture, returned as named tensors.

    dict(result) has nine owning detached arrays in fit/validation/test order:
    *_images Float32[N,28,28,1], *_factors Int32[N,4], *_source_ids Int32[N].
    Factor order shape(2),size(3),x(7),y(7); rows have ascending lattice IDs.
    Residue (x+2*y+size+3*shape)%7 assigns test0, validation1, fit2..6.
    Radius=size+2, centre=(3*x+5,3*y+5), inclusive binary boundaries.
    Counts210/42/42; no external data or random state. This is a procedural
    teaching fixture, not dSprites or a downloaded benchmark.
    """
    return [(name, value) for name, value in _native.KnownFactorLattice()]


def TiedAutoencoder(parameters, image, features=784, latent=32, tied=True):
    """Native ReLU encoder and sigmoid decoder with a live differentiable W transpose.

    Finite matching Float32/64 image[B,features], B>0, 0<latent<features.
    Owns encoder_weight, decoder_weight, code and reconstruction. Three leaves:
    encoder/W,b and decoder/b (25904 scalars at default 784–32–784); both W
    gradient paths accumulate. tied=False creates an independent decoder/W
    (50992 scalars); matching initialization is the caller's responsibility.
    Inputs/configuration/conflicting decoder/W errors precede creation and raise
    RuntimeError; incompatible existing leaves may fail later. No fit/loss policy.
    """
    return _native.TiedAutoencoder(parameters, image, features, latent, tied)


def TiedAutoencoderDecode(parameters, code, features=784, latent=32, tied=True):
    """Native target-free sigmoid decoder using explicit finite code[B,latent].

    Same geometry/dtype/error rules as TiedAutoencoder. Zero/signed codes allowed.
    True creates/reuses encoder/W and decoder/b only; false uses decoder/W,b.
    Handles own lifetimes and retain code/matrix/bias gradients. No image input,
    encoder bias, ReLU, clipping or hidden fit; transpose is recomputed each call.
    """
    return _native.TiedAutoencoderDecode(parameters, code, features, latent, tied)


def SigmoidAutoencoder(parameters, image, widths):
    """Native symmetric sigmoid layers, returning an owning Python list of tensors.

    Odd integer widths length>=3 are positive, mirrored and strictly decreasing
    into the central code. Finite matching Float32/64 image[B,widths[0]], B>0.
    layer<i>/W,b use independent Glorot weights and zero biases. Every layer is
    sigmoid; code is outputs[len(widths)//2-1], reconstruction is outputs[-1].
    All input/parameter gradients remain. Noninteger widths raise TypeError.
    Native input/geometry errors raise RuntimeError
    before leaf creation; incompatible leaves can fail later. No fitting or loss.
    """
    return list(_native.SigmoidAutoencoder(parameters, image, [_index(w) for w in widths]))


def SigmoidAutoencoderDecode(parameters, code, widths):
    """Native decoder half using only explicit finite Float32/64 code[B,central_width].

    Complete mirrored integer geometry uses original layer indices, matching the
    forward stack. Output owns lifetime and retains code/decoder gradients.
    No image, target or copied pixels; zero/signed codes are accepted. Invalid
    input/widths/dtype raise RuntimeError before creation; incompatible leaves can
    fail later. A fresh store creates decoder leaves only.
    """
    return _native.SigmoidAutoencoderDecode(parameters, code, [_index(w) for w in widths])


def ActivationSparsity(code, l1_weight, kl_weight, rho=.05, epsilon=1e-5):
    """Native activation-L1 and clamped mean-unit Bernoulli-KL penalties.

    Finite nonempty Float32/64 code[B,F]; weights>=0,0<rho<1,0<epsilon<.5.
    Constants/complements/endpoints round to code dtype and must stay interior.
    L1 averages B*F absolute activities; KL averages F clamped batch unit means.
    Owns scalar l1,kl,total and unit_mean/clipped_mean[F], retaining code gradients.
    abs passes zero gradient at zero; clip passes endpoints, zeros outside.
    Signed codes are accepted; no mutation or hard zeros. Invalid inputs raise
    RuntimeError. Batch splitting generally changes the KL objective.
    """
    return _native.ActivationSparsity(code, l1_weight, kl_weight, rho, epsilon)


def SparseAutoencoder(parameters, image, l1_weight, kl_weight, features=784, latent=64,
                      rho=.05, epsilon=1e-5):
    """Canonical sigmoid autoencoder with complete MSE+activation-L1+KL loss.

    Finite matching Float32/64 image[B,features], positive widths; default
    784–64–784 creates/reuses encoder/W,b and decoder/W,b (101200 scalars).
    Owns code, reconstruction, mse, loss and an owning nested penalty result.
    All gradients remain, including direct targets if image requires gradients.
    Invalid input/options fail before leaf creation; incompatible leaves can fail
    later. The caller owns optimizer, minibatches, epochs and selection.
    """
    return _native.SparseAutoencoder(parameters, image, l1_weight, kl_weight,
                                    features, latent, rho, epsilon)


def ImageRowSequence(images, steps=28, features=28):
    """Native ordered image rows [B,T,F], T=F=28 by default; no resize or clipping.

    Accepts finite Float32/64 [B,T,F,1], [B,T,F] or [B,T*F], B,T,F>0.
    Preserves values, dtype, order and gradients, materializing strides if needed.
    Invalid shape/dtype/nonfinite inputs raise RuntimeError; no mutation.
    """
    return _native.ImageRowSequence(images, steps, features)


def RecurrentEncode(parameters, sequence, steps=28, features=28, units=32):
    """Native final-state-only tanh encoder with full gradients over all T steps.

    Finite matching Float32/64 sequence[B,T,F], T>1, F>0, 0<U<T*F.
    Shares encoder/Wx, Wh and b across time; zero initial state on every call.
    Owns code[B,U] and encoder_states[B,T,U]; decoder fields are undefined.
    Invalid input geometry/dtype/nonfinite values fail before leaf creation.
    """
    return _native.RecurrentEncode(parameters, sequence, steps, features, units)


def RecurrentDecode(parameters, code, steps=28, features=28):
    """Independent repeated-code tanh decoder, with no image/target/teacher forcing.

    Finite matching Float32/64 code[B,U], B,U,T,F>0, retains all gradients.
    Each step receives the same code and starts from a fresh zero initial state.
    Owning code, reconstruction[B,T,F], decoder_states[B,T,U]; encoder history
    is undefined. Shares decoder/Wx, Wh, b and output/W,b over every step.
    Invalid inputs raise RuntimeError before decoder leaf creation.
    """
    return _native.RecurrentDecode(parameters, code, steps, features)


def RecurrentAutoencoder(parameters, sequence, steps=28, features=28, units=32):
    """Native strict final-state bottleneck and independent repeated-code decoder.

    Finite Float32/64 sequence[B,T,F] matches the store; T>1,F>0,0<U<T*F.
    Default T=F=28,U=32 creates/reuses eight leaves and4956 learned scalars.
    Owns code, reconstruction and both state histories, preserving continuous
    gradients through all encoder/decoder steps. Zero initial states, no bypass,
    attention, teacher forcing or truncation. Invalid inputs raise RuntimeError.
    The caller owns objective, optimizer, batching and checkpoint selection.
    """
    return _native.RecurrentAutoencoder(parameters, sequence, steps, features, units)


def MaeSelection(source, epoch=0, seed=1907, dtype=None):
    """Detached [B,4,16] visible selectors from Int32/64 uint32-range source IDs.

    Four distinct one-hot rows use the source's wrapping hash rank order.
    Epoch is uint32-range and seed is uint32; no global RNG is changed.
    Output is Float32 by default, or the supplied Float32/64 dtype.
    Invalid IDs/epoch/dtype raise RuntimeError; empty source vectors are accepted.
    """
    if dtype is None:
        return _native.MaeSelection(source, epoch, seed)
    return _native.MaeSelection(source, epoch, seed, dtype)


def MaePosition(width, dtype=None):
    """Detached fixed [16,width] row-sine/row-cosine/column-sine/column-cosine table.

    Width is a positive multiple of four. Float32 is the default; Float64 is
    supported. Invalid widths/dtypes raise RuntimeError. Coordinates are grid
    indices 0..3 and frequency k is 10000**(-4*k/width), without global state.
    """
    return _native.MaePosition(width) if dtype is None else _native.MaePosition(width, dtype)


def MaePixel(patch, within):
    """Flat pixel index [0,783] of patch [0,15] and within-patch pixel [0,48].

    Both orders are row-major; invalid indices raise RuntimeError.
    """
    return _native.MaePixel(patch, within)


def MaeBlock(parameters, name, x, width):
    """Native two-head pre-norm residual transformer block with ratio-2 tanh-GELU.

    Requires matching finite Float32/64 x[B,T,width], positive B,T and even width,
    and a nonempty parameter prefix. No dropout/causal mask; LayerNorm epsilon
    is 1e-5. Returns owning output and attention[B,2,T,T] handles with gradients.
    Invalid arguments raise RuntimeError; the caller owns fitting.
    """
    return _native.MaeBlock(parameters, name, x, width)


def MaePatches(image):
    """Differentiable finite Float32/64 NHWC [B,28,28,1] to [B,16,49].

    Patches are nonoverlapping row-major 7x7 blocks. Requires B>0; supports
    strides, preserves input values/gradients and raises RuntimeError on bad inputs.
    """
    return _native.MaePatches(image)


def MaeImage(patches):
    """Differentiable inverse patch permutation [B,16,49] to flat [B,784].

    Float32/64 values and bits are retained without range/finiteness checks;
    empty batches and strided tensors are accepted. Invalid shapes raise RuntimeError.
    """
    return _native.MaeImage(patches)


def MaeEncode(parameters, image, selector):
    """Native visible-only width-32 encoder; no target, scoring or fitting.

    image is finite Float32/64 [B,28,28,1], B>0. The matching selector[B,4,16]
    is detached, with four distinct one-hot rows. Invalid geometry/dtypes/selectors
    fail before parameter creation. Returns owning patches, visible, mask,
    encoder_input, code and encoder_attention; decoder-only fields are undefined.
    Gradients through continuous inputs and learned leaves are retained.
    """
    return _native.MaeEncode(parameters, image, selector)


def MaeDecode(parameters, code, selector):
    """Decode finite [B,4,32] visible codes and locations, without an image/target.

    Matching detached selector[B,4,16] contains four distinct one-hot rows.
    Returns owning projected, restored, decoder_input, prediction, code, mask
    and decoder_attention; encoder-only fields are undefined. Continuous
    gradients remain; invalid inputs raise RuntimeError before decoder creation.
    """
    return _native.MaeDecode(parameters, code, selector)


def MaskedAutoencoder(parameters, image, selector):
    """Canonical 35-tensor, 13649-scalar visible-only transformer MAE; no fitting.

    Matching finite Float32/64 images[B,28,28,1] and detached unique one-hot
    selectors[B,4,16] feed a width-32 four-token encoder and width-16 sixteen-token
    mask decoder. Both use one two-head pre-norm block, ratio-2 tanh-GELU MLP,
    fixed grouped 2D positions and no dropout. Raw sigmoid patch predictions
    never copy visible pixels. All endpoint handles own lifetimes and retain
    continuous gradients. Invalid inputs raise RuntimeError. The caller owns
    masked loss, Adam, batching and validation selection.
    """
    return _native.MaskedAutoencoder(parameters, image, selector)


def MaeLoss(prediction, patches, mask):
    """Mean squared error over hidden pixels in matching Float32/64 [B,P,D].

    Detached binary mask[B,P] uses one for hidden; divides by D*sum(mask).
    At least one patch must be hidden. Prediction/target gradients and strides
    are retained; inputs are not mutated. Invalid shapes/dtypes/masks raise
    RuntimeError. Nonfinite arithmetic may propagate through zero weights.
    """
    return _native.MaeLoss(prediction, patches, mask)


def CompetitiveMask(scores, lifetime, k=8):
    """Detached binary selector for finite Float32/64 scores[B,F].

    Per-row mode keeps min(k,F) largest signed activities per observation;
    lifetime mode keeps ceil(B/8) observations per feature. Equal scores prefer
    lower indices. k is nonnegative, including when ignored in lifetime mode.
    Empty axes and strides are accepted. Multiplying scores by this selector
    passes gradients through winners only; the ranking itself is discrete.
    Malformed/nonfinite arguments raise RuntimeError; values are newly owned.
    """
    return _native.CompetitiveMask(scores, lifetime, k)


def CompetitiveAutoencoder(parameters, input, lifetime, plain=False, features=784, latent=64, k=8):
    """Canonical hard-competition encoder/decoder; the caller owns fitting.

    Matching nonempty Float32/64 input[B,features] creates/reuses independent
    encoder/W,b and decoder/W,b, with Glorot weights/zero biases. Per-row mode
    uses a linear encoder and retains top-k signed activities at deployment.
    Lifetime mode uses ReLU and ceil(B/8) winners per feature for training,
    bypassing the gate at deployment. plain=True bypasses competition while
    retaining that encoder activation. Positive widths and k>=0 are required.
    Returns owning raw, mask, training_code, code, training_reconstruction and
    reconstruction handles. All continuous paths retain gradients; mask is
    detached. Invalid arguments raise RuntimeError. Defaults give 101200 scalars.
    """
    return _native.CompetitiveAutoencoder(parameters, input, lifetime, plain, features, latent, k)


def GaussianNoise(shape, samples, seed):
    """Detached Float64 [samples,shape...] bank from explicit MT19937/Box-Muller.

    The uint32 seed (0 through 4294967295) is local to each call; there is no shared RNG or
    standard-library normal distribution. Negative dimensions/counts raise
    RuntimeError; empty banks are allowed. Odd final counts discard the spare
    sine partner. This stream differs from normal() and Mt19937.normal64().
    """
    return _native.GaussianNoise(shape, samples, seed)


def CorruptForDenoising(clean, standard_noise, sigma):
    """Detached Float32 augmentation: clamp(float64(clean)+float32(sigma)*noise).

    Finite unit-range clean pixels have rank >= 2; noise is finite Float64 with
    shape [1,*clean.shape()]. The addition rounds to Float32 before clamping to
    [0,1]. sigma is finite/nonnegative in original pixel units. Invalid shapes,
    dtypes or values raise RuntimeError. Strided/empty inputs are accepted; inputs
    are not mutated. The unchanged clean image remains the reconstruction target.
    """
    return _native.CorruptForDenoising(clean, standard_noise, sigma)


def UndercompleteAutoencoder(parameters, input, features=784, latent=32):
    """Canonical untied ReLU/sigmoid forward model, without fitting.

    input is matching Float32/64 [N,features]. Requires 0 < latent < features;
    defaults preserve the 784-to-32-to-784 network. Creates/reuses encoder/W,
    encoder/b, decoder/W and decoder/b. Returns owning code and reconstruction
    handles with their gradients retained; invalid arguments raise RuntimeError.
    """
    return _native.UndercompleteAutoencoder(parameters, input, features, latent)


def SigmoidEncoderContraction(code, encoder_weight):
    """Exact squared sigmoid-encoder Jacobian norms and their batch mean.

    code is Float32/64 [N,K]; encoder_weight is matching [D,K]. Norms sum
    all D*K derivative coordinates and average only over observations. Both
    gradient paths and higher derivatives are retained. Returns owning tensor
    handles in per_example and mean; malformed shapes/dtypes raise RuntimeError.
    See Rifai et al. (2011), https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf.
    """
    return _native.SigmoidEncoderContraction(code, encoder_weight)


def ContractiveAutoencoder(parameters, input, weight, features=784, latent=32):
    """Evaluate the canonical untied sigmoid encoder/decoder; performs no fitting.

    input is nonempty Float32/64 [N,features], matching parameters' dtype.
    Creates/reuses encoder/W, encoder/b, decoder/W and decoder/b. Positive
    widths may be overcomplete. Finite weight >= 0 multiplies the exact
    squared encoder Jacobian norm; reconstruction MSE averages all pixels.
    Returns owning code, reconstruction, mse, penalty and loss handles.
    Invalid arguments raise RuntimeError. The caller owns batching and Adam.
    The source variant uses independent weights and MSE; the Rifai et al. paper
    (https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf) uses tied weights
    and cross-entropy in its experiments.
    """
    return _native.ContractiveAutoencoder(parameters, input, weight, features, latent)


def ConvolutionalAutoencoder(parameters, image):
    """Canonical spatial 741-parameter grayscale encoder/decoder; no fitting.

    image is nonempty matching Float32/64 NHWC [N,H,W,1], H,W multiples of
    four. Two ReLU 3x3 convolutions with 2x2 average pooling produce
    [N,H/4,W/4,4]; nearest repetition and two independent convolutions restore
    the image with a sigmoid output. Returns owning conv1, pool1, conv2, code,
    decoder1 and reconstruction handles with gradients retained. Creates or
    reuses encoder1/W,b, encoder2/W,b, decoder1/W,b and decoder2/W,b.
    Invalid shapes/dtypes raise RuntimeError. The caller owns loss and Adam.
    """
    return _native.ConvolutionalAutoencoder(parameters, image)


def ConvolutionalDecoder(parameters, code):
    """Decode matching Float32/64 [N,H,W,4] into [N,4H,4W,1].

    Returns owning hidden and reconstruction handles; both retain gradients.
    Positive dimensions are required. Creates/reuses decoder1/W,b (He uniform)
    and decoder2/W,b (Glorot uniform), with zero biases and 369 scalars in all.
    Uses nearest repetition, not transposed convolution. No fitting or mutation
    of input. Invalid shapes/dtypes raise RuntimeError.
    """
    return _native.ConvolutionalDecoder(parameters, code)

DenseGanObjective = SimpleNamespace(**{name: getattr(_native, 'DenseGanObjective_' + name)
                                     for name in ['Logistic', 'LeastSquares', 'Hinge']})

DType = SimpleNamespace(**{name: getattr(_native, 'DType_' + name) for name in
                          ('Bool', 'Int32', 'Int64', 'Float32', 'Float64')})
Activation = SimpleNamespace(**{name: getattr(_native, 'Activation_' + name) for name in
    ('Relu', 'Relu6', 'LeakyRelu', 'Elu', 'Selu', 'Gelu', 'GeluTanh', 'Silu', 'Mish',
     'Softplus', 'Softsign', 'Sigmoid', 'Tanh')}, Identity=_native.Activation__None)
Padding = SimpleNamespace(**{name: getattr(_native, 'Padding_' + name) for name in ('Same', 'Valid', 'Explicit')})
OptimizerKind = SimpleNamespace(**{name: getattr(_native, 'OptimizerKind_' + name) for name in
    ('Sgd', 'Momentum', 'Nesterov', 'Adagrad', 'RMSProp', 'Adam', 'AdamW', 'Nadam',
     'AMSGrad', 'Adamax', 'Lamb', 'Lars', 'Lion', 'Adafactor', 'Adadelta')})
init = SimpleNamespace(**{name[5:]: getattr(_native, name) for name in dir(_native) if name.startswith('init_')})
schedule = SimpleNamespace(**{name[9:]: getattr(_native, name) for name in dir(_native) if name.startswith('schedule_')})
AttentionImpl = SimpleNamespace(**{name: getattr(_native, 'AttentionImpl_' + name)
                                  for name in ('Auto', 'Plain', 'Fused')})
RotaryLayout = SimpleNamespace(**{name: getattr(_native, 'RotaryLayout_' + name)
                                for name in ('Interleaved', 'HalfSplit')})


def _enum(prefix):
    return SimpleNamespace(**{name[len(prefix) + 1:]: getattr(_native, name) for name in dir(_native)
                              if name.startswith(prefix + '_') and not callable(getattr(_native, name))})


for _kind in ('PadMode', 'ResizeMethod', 'DepthToSpaceMode', 'GridSampleMode', 'GridSamplePadding', 'WavFormat',
              'Reduction'):
    if _kind not in globals() or not isinstance(globals()[_kind], SimpleNamespace):
        globals()[_kind] = _enum(_kind)
augment = SimpleNamespace(**{name[8:]: getattr(_native, name) for name in dir(_native) if name.startswith('augment_')})


def _image_size(size):
    """Convert a spatial pair to the canonical C++ value type."""
    if isinstance(size, _native.ImageSize):
        return size
    height, width = size
    result = _native.ImageSize()
    result.height, result.width = _index(height), _index(width)
    return result


def resize_bilinear(images, size, align_corners=False, half_pixel_centers=False):
    return _native.resize_bilinear(images, _image_size(size), align_corners, half_pixel_centers)


def resize_nearest(images, size, align_corners=False, half_pixel_centers=False):
    return _native.resize_nearest(images, _image_size(size), align_corners, half_pixel_centers)


@contextmanager
def no_grad():
    previous = is_grad_enabled()
    set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(previous)
