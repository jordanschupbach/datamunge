\name{nn}
\alias{nn}
\title{Functional C++ neural network API}
\description{
DoubleSampleRows(input) copies finite signed Float32/64 CPU samples[N,...],
rank at least1 with positive axes, into independent Float64[N,D] rows with
live input derivatives. Trailing axes flatten in logical order; a vector has D1.
SvdPca(fitting,retained=2) fits centered unscaled thin SVD to explicit fitting
rows only, N>=2 and1<=retained<=min(N,D). Stores Float64 means, axes as columns,
variances s squared/(N-1) and fitting scores. Fitting is detached; zero singular
values are allowed and retained count is not numerical rank. No labels, query
refit, variance scaling, whitening or null dropping enters. Getter methods
mean(),components(),variances(),scores() export independent detached copies.
Transform(query) retains query derivatives through frozen mean/axes, returning
Float64[M,R]. Reconstruct(projection[M,K]) retains projection derivatives through
first K axes,1<=K<=R, returning Float64[M,D] in original units without clipping.
Finite strided signed Float32/64 CPU queries/scores are accepted; outputs own
lifetime. Invalid inputs/counts or nonfinite computations raise errors; native
SVD solver failures propagate. Native signs and degenerate subspaces can vary
across backends. Reference: R prcomp(center=TRUE,scale.=FALSE),
\url{https://stat.ethz.ch/R-manual/R-devel/library/stats/html/prcomp.html}.
NearestFitRows(fitting,query,neighbors=10) widens signed floating CPU sample
rows, accumulates squared Euclidean distances in double feature order and sorts
(distance,fitting row), with smaller-row ties. Positive k<=N returns owning
detached Int32[M,k]; N must fit Int32, widths match and sample axes are positive.
No labels, normalization or fitting enters. NeighborOverlap(reference,candidate)
returns per-entry membership hits/(M*k) for matching nonnegative Int32/64[M,k]
tables: repeated reference IDs count separately; repeated candidates add no hits.
NeighborLabelAccuracy(neighbors,fit_labels,query_labels) compares column0's
fitting label with the aligned query label; later neighbors do not vote.
All indices must lie in the fitting label pool; labels are nonnegative integers,
with query-only classes allowed. Invalid geometry/dtype/values or distance
overflow raise errors. Strides and mixed integer dtypes are accepted.
LatentLerp(a,b,alpha) forms (1-alpha)*a+alpha*b with the original two-product
then sum grouping. LatentDirection(source,direction,scale) forms affine edits
source+scale*direction. Finite signed matching Float32/64 CPU endpoint/direction
matrices[B,D] and level vectors[T] have positive axes; strides and all continuous
first/higher paths remain live. Owning results[B,T,D] retain negative codes
and arbitrary extrapolation levels. No clipping, norm matching, projection,
decoder, fitting, random draws or semantic guarantee is implicit.
LatentFlatten(codes) copies finite matching [B,T,D] to independent contiguous
[B*T,D] rows, path-major then level-major, with gradients preserved.
LatentClassMeans(codes,labels,classes=10) groups finite signed Float32/64 CPU
codes[B,D] by aligned Int32/64[B] IDs in [0,classes). Every positive configured
class must occur. Row-order double sums/division followed by one final dtype
cast produce owning detached means[classes,D]. Caller chooses fitting rows;
class-label summaries are supervised posthoc constructions. Invalid inputs,
missing classes, products or accumulation/cast overflow raise errors.
UndercompleteDecoder(parameters,code,features=784,latent=32) creates/reuses only
decoder/W[latent,features] and decoder/b[features], two default leaves/25872 values.
Finite signed matching floating CPU codes[B,latent], positive B and
0<latent<features, map to owning sigmoid reconstructions[B,features]. Strides and
code/decoder higher paths stay live. No images, labels, encoder evaluation,
clipping, fitting or updates enter. Restore learned arrays before deployment;
initializing a decoder alone uses a different RNG sequence from a full model.
Invalid inputs fail before creation; incompatible existing leaves may fail later.
FrozenEncoderFeatures(image,weight,bias) applies a frozen ReLU encoder to
finite matching Float32/64 matrices image[B,F],weight[F,L] and bias[L].
All axes are positive; strides and input derivatives remain live, while
weight/bias paths are stopped. FitTransferScaling(code,floor=1e-6) returns
owning detached center[L]/scale[L] for finite matching code[B,L]. It uses
logical row-order double means and a second-pass population variance, floors
the deviation and casts once to code dtype; singleton/constant scales equal
floor. The caller supplies fitting rows only. Invalid shapes/values or a
nonpositive, nonfinite or unrepresentable floor raise an R error.
FrozenEncoderClassifier(weight,bias,center,scale,classes=10) owns detached deep
copies and exports independent copies with encoder_weight(),encoder_bias(),
center(),scale(). Forward(head,image,labels) creates/reuses only linear/W[L,C]
and linear/b[C], with live head/input derivatives and frozen encoder/scaling.
Optional undefined labels give target-free inference and undefined loss;
Int32/64[B] labels in [0,C) give mean categorical log-softmax cross-entropy.
Its owning result fields are code,features,logits,probability,loss.
The matching-dtype caller-owned head has 330 scalars/two leaves for L32/C10.
No optimizer, fitting loop, split selection or updates are hidden.
VisiblePatchValues(patches,hidden,fill=0) and
CompositePatchValues(observed,prediction,hidden) provide target-free generic
patch selection. Finite signed matching Float32/64 arrays[B,P,D] have positive
axes; hidden is a detached same-dtype binary[B,P] mask, one meaning hidden.
Visibility replaces hidden patches with one finite fill; composition takes
predicted values at hidden sites. Visible observations are copied bit-for-bit,
including signed zero. All-visible/all-hidden masks and strides are accepted.
Every input value must be finite, including unselected branches. Float32 fill
narrows once; overflow is rejected. Outputs own lifetime and retain selected
continuous first/second paths; unselected derivatives are zero. No clamping,
blending, clean targets, model state, fitting or random draws are added.
Invalid inputs raise errors. Use MaePatches/MaeImage for image mapping.

WassersteinAutoencoder(parameters,image,prior,weight=10,features=784,latent=4,hidden=16)
provides deterministic WAE-MMD with eight encoder/code/decoder/output leaves
and26036 default scalars. Matching finite Float32/64 image[B,F] in[0,1] and
explicit independent prior[P,L] require B,P>=2, including weight0; unequal
counts and strides are accepted. Six owning Tensor fields and a nested
seven-field MMD record retain continuous paths through garbage collection.
WaeSquaredPixelCost sums squared errors over pixels; WaeObjective averages
those costs over images then adds weight times signed unbiased IMQ MMD.
Negative finite-sample estimates are retained; this is not a density, ELBO
or exact Wasserstein distance. Weight is finite/nonnegative and representable
in the computation dtype. WaeDecode is code-only tanh/linear logits, with
four decoder/output leaves and13408 default scalars; sigmoid gives unit-range
values, without pixel draws. Prior changes only the discrepancy, not codes
or reconstruction. Caller owns prior sampling, split provenance, batches,
optimizer and checkpoints. Invalid inputs/options/products raise errors before
leaves; incompatible existing leaves can fail later. No fitting or RNG beyond
lazy initialization is hidden. Reference: Tolstikhin et al.(2018),
\url{https://arxiv.org/abs/1711.01558}.

Vq2TopDecoder transforms coarse vectors[B*H*W/16,D] into code-derived
conditioning[B,H/2,W/2,D] with two top_decoder leaves (148 default scalars).
Vq2Decoder combines fine vectors[B*h*w,D] and conditioning[B,h,w,D], even h,w,
into logits[B,4*h*w], using four decoder/output leaves (657 default scalars).
VectorQuantizedAutoencoder2 accepts unit-range Float32/64 grayscale NHWC and
an optional Int32 top-assignment override. Default K16/D4/C8 gives fourteen
leaves/1157 scalars. Nine owning Tensor endpoints and two nested six-field
quantizers retain intended gradient paths through GC. Fine encoding and
final decoding use coarse-derived features; no continuous image bypass reaches
the decoder. Pixel-summed image-mean NLL plus both element-mean dictionary and
beta0.25 commitment penalties preserves source reductions and creation order.
Reconstruction skips both books; fine losses can reach the coarse encoder.
Vq2ConditionalPrior(top,bottom,top_height=7,top_width=7,entries=16,smoothing=.5)
copies fitting-only valid Int32[B,Ht*Wt]/Int32[B,4*Ht*Wt] grids. Parent alignment
uses two-by-two coarse cells and previous fine symbol with BOS=K per image.
counts()/probabilities() export independent Int32/Float64[K,K+1,K] tables.
Nll(top,bottom) returns mean conditional nats per complete fine grid.
Sample(top,uniforms) consumes explicit Float64[B,4*Ht*Wt] uniforms in[0,1),
with strict inverse-CDF boundaries and no RNG. Invalid inputs/options/products
raise errors. This compact parent/previous-symbol prior is not PixelCNN or an
image likelihood. Caller owns optimizer steps, prior fitting and split provenance.

VectorQuantize supplies nearest-vector assignment with lowest-index ties,
optional valid Int32 assignment overrides, owning six-field results and
independent straight-through, dictionary and commitment gradient paths.
Finite matching Float32/64 encoded[N,D] and book[K,D] accept strides. The
surrogate is identity to encoded and stops book; its forward retains selected
vectors without cancellation against encoded. Both penalties average N*D.
VqLookup maps valid Int32[B,T] to floating[B*T,D]. VqDecoder is code-only
repeat/Conv3-ReLU/repeat/Conv3-linear logits[B,H*W], sharing four decoder1/output
leaves (369 default scalars). VectorQuantizedAutoencoder defaults K16/D4/C8,
beta0.25, nine leaves/805 scalars; owns encoded/book/probability/nll, two loss
scalars and nested quantizer fields. Grayscale unit-range NHWC images require
positive height/width divisible by four. No EMA, resets or fitting is hidden.
VqUsage owns Int32[K] counts and pooled empirical exp(entropy) perplexity.
VqMarkovPrior(fitting,entries=16,smoothing=.5) copies fitting-only valid Int32[B,T]
grids. counts()/probabilities() return independent Int32/Float64[K+1,K] tables;
BOS=K resets per row. Nll(grid) averages complete grid nats; Sample(uniforms)
uses explicit Float64[B,T] in[0,1) with strict inverse-CDF boundaries and no RNG.
It is a teaching first-order baseline, not PixelCNN or an image likelihood.
All invalid inputs/options/products raise errors; caller owns optimizer steps,
random banks and split provenance. Nested fields survive parent collection.
SWIG bindings to datamunge::nn. R code composes models and owns
training and evaluation loops; C++ implements tensor operations and differentiation.}
\format{An environment containing tensor factories, named ParameterStore,
functional layers, gradients, Optimizer, BatchOrder, datasets and tensor I/O.}
\details{
The vanilla two-coordinate VAE uses BetaVariationalAutoencoder with beta=1,
features=784, latent=2 and hidden=64: ten leaves/101652 scalars, with the original
names and creation order. GaussianVaeDecode with the same dimensions accepts
codes alone. The original shared example adapters now forward to these core
operations; the caller still owns noise, Adam, validation and checkpoint selection.

VaeGan owns all nineteen source Gaussian VAE-GAN endpoints and three separate
encoder/generator/discriminator objectives. Distinct matching Float32/64 stores,
unit-range image[B,F], explicit posterior noise[B,L] and independent prior[P,L]
are required; P may differ from B. Defaults give14 leaves/32393 scalars.
VaeGanDecode is code-only tanh/sigmoid generation; VaeGanDiscriminator returns
owning tanh features and real/fake logits. VaeGanFeatureNll sums half squared
feature residuals against a detached real target, without a Gaussian constant.
Only reconstructed features receive its gradients. Encoder loss combines feature
distance and KL; generator loss adds balanced non-saturating adversarial terms;
discriminator loss has a real term and half of each fake term. Pixel NLL uses
the source Float32 clipping endpoints and is diagnostic, not a fitted pixel ELBO.
Select only D, E or G leaves for each caller-owned optimizer and recompute after
each step. No fit, RNG, mode change or implicit freezing. Invalid input/geometry
raises errors; outputs retain ownership and available continuous gradients.

MidpointUniformNoise(shape,seed) returns owning detached Float64 uniforms from
one local MT19937 raw32 word per element: (raw+.5)/2^32. All unsigned32 seeds,
scalar shape and empty axes are supported; invalid dimensions/counts error.
The32-bit-resolution values are strictly inside(0,1) in Float64, without shared
RNG state or clipping. Casting to Float32 can round up to1. SequenceGenerate
accepts the original Float64[B,7] values for reproducible token sampling.

The native prepare_example_data(name,directory) delegates to datamunge::datasets;
sequence-vae-assets prepares the exhaustive local grammar with provenance.
LoadSequenceVaeData(root) returns an owning fit/validation/test list after checking
all432 unique sentences, IDs, factors, splits, EOS/PAD and previous-target inputs.
Supplied valid row order is retained. Each pool's tokens, inputs, lengths, factors
and source survive collection of the list. SequenceDataArrays exports fifteen
fully validated named arrays, without config. SequenceVocabulary returns an
independent nineteen-word copy. SequenceText includes literal special tokens and
stops at the first EOS; SequenceGrammarValid tests that prefix only, ignoring any
later suffix. Matrix/row errors raise errors; malformed grammar returns FALSE.

SequenceVariationalAutoencoder owns eighteen endpoints of the fixed vocabulary19,
embedding8, masked encoder GRU16, Gaussian4 and causal decoder GRU16 model.
Int32/64 targets are [B,7]; finite explicit Gaussian noise [B,4] matches the
Float32/64 store. Twenty-two leaves/3243 scalars; NLL sums non-PAD targets,
including EOS, and loss is mean(NLL+KL). SequenceEncoder accepts arbitrary
positive T to test exact appended-PAD invariance. SequenceDecoderInputs returns
owning Int32 BOS-then-previous-target prefixes. SequenceTeacher consumes only
fixed z and prefixes and owns all five decoder endpoints, with live continuous
paths. It and SequenceGenerate share eleven decoder leaves/1755 scalars.
SequenceGenerate uses greedy argmax or explicit Float64 open-interval [B,7]
uniforms with cdf<=u ties. It emits PAD after EOS while raw scores still evolve;
hard IDs have no straight-through gradient. No hidden fit, RNG draws or no-grad.
AnnealedSequenceObjective computes mean(NLL+beta*KL), beta in [0,1]. CollapseBeta
uses the source-rounded explicit step/(ramp-1), default ramp18, or constant1.
CollapseStatistics reads finite matching Float32/64[N,D], N>1, and nonempty
aligned draws. Owns host-double per-unit KL, population mean variances and
analytic/sampled conditional KL, finite empirical-mixture information/aggregate
KL, and active-unit count (variance strictly above the default .01). Own-posterior
mixture inclusion and finite-sample bias are explicit; negative estimates remain.
No gradients or semantic/population-information claims. Invalid inputs and
nonfinite computed diagnostics raise errors. The caller owns Adam and selection.
Use \code{nn$ParameterStore(seed)}, \code{nn$dense}, \code{nn$conv2d},
\code{nn$grad} and \code{nn$Optimizer} in R model and training code.
Enums are grouped under \code{nn$DType}, \code{nn$Activation},
\code{nn$Padding} and \code{nn$OptimizerKind}. Initializers are under \code{nn$init}.
\code{nn$no_grad(function() ...)} restores gradient mode even on errors.
Shapes and tensor row indices follow the zero-based C++ API.
\code{nn$load_npz} and \code{nn$save_npz} use named lists of tensors.
\code{nn$tensor_list} and \code{nn$tensor_vector} convert native tensor vectors.
R's reserved word next is exposed as \code{batch$`_next`()} on BatchOrder.
Tensor references retain their owning containers through garbage collection.}
\section{Disjoint unpaired domains and jointly captured CycleGAN generators}{
\code{nn$RotateInvertGrayscale(image,inverse=FALSE)} uses native reverse/transpose
and1-minus on finite signed Float32/64 CPU[N,H,W,1], positive axes. Default
out(y,x)=1-in(H-1-x,y); inverse orientation is1-in(x,W-1-y). Owns live[N,W,H,1]
output, no clipping/fitting/state. Floating cancellation can prevent bit-exact
round trips; retain original pixels for exact known-mapping diagnostics.
\code{nn$SplitCycleDomains(images,labels,source_ids)} requires unit grayscale
[N,28,28,1], N>=20 divisible by20, balanced integer labels0..9 and aligned unique
nonnegative integer source IDs(Int32/64). First N/20 observed rows of each class
become A, remaining rows original_B, preserving row order; B is rotated/inverted.
Owns a,b,ids_a,ids_b,original_b fields; image paths stay live and ID dtype remains.
It checks this pool, without certifying original provenance or cross-pool identity;
use LoadSaliency first. No random split,pairing,parameter creation or fitting.

\code{nn$CycleGanDiscriminate(critic,image)} returns unconditional raw[N,7,7,1]
patch scores for finite signed matching positive grayscale28x28 rows. Biased
normal(.02) convs/leaky.2 create661 values/6 leaves, without sigmoid or batch state.
\code{nn$CycleGan(ab,ba,da,db,a,b,cycle_weight=10,identity_weight=5)} requires four
distinct matching Float32/64 stores; independent A/B rows may have unequal counts.
G maps reuse the genuine two-skip U-Net1390/10 each; all groups4102/32. G has full
least-squares adversarial means plus10 summed cycle-L1 and5 summed identity-L1;
each D has half the two patch means. Nonnegative finite representable weights;
zero keeps diagnostics. Owns19 live translation/cycle/identity/score/loss fields.
First/higher paths retain storage/graphs. Caller steps D_a then D_b, makes ONE
fresh common G forward, captures BOTH generator vectors before either G write,
then steps both from that snapshot. Recomputing after the first G write differs.
No hidden capture,optimizer,fitting,no-grad,paired targets,known mapping or labels.
Errors raise R errors. See cycle_domains.hpp/cycle_gan.hpp and Zhu et al.(2017),
https://arxiv.org/abs/1703.10593 .
}
\section{Paired U-Net translation and conditional patch scores}{
\code{nn$quantized_sobel_edges(image,threshold=256L)} accepts finite unit-range
Float32/64 CPU[N,H,W,1] with positive axes. Quantization is
floor(Double(pixel)*255+.5), then raw3x3 zero-padded Sobel and inclusive
abs(gx)+abs(gy)>=threshold. Nonnegative threshold; borders process against zeros.
Owns detached binary input-shape/dtype output; no fitting or input derivative.

\code{nn$Pix2PixGenerate(g,source)} accepts finite signed matching Float32/64
CPU[N,28,28,1], N>0. Two-level biased normal(.02) U-Net has real concatenation
skips(up14,down14) and(up28,source), learned transpose4x4/stride2 maps, ReLU
features and final3x3 sigmoid. Seed6101 creates1390 values/10 leaves. Owns live
encoder/decoder/skip/image fields, without target, labels, noise or batch state.
\code{nn$Pix2PixDiscriminate(d,source,image)} requires matching aligned inputs,
concatenates condition before candidate, then returns49 raw logits per image
[N,7,7,1]. Two strided leaky.2 maps and3x3 scores use725 values/6 leaves(seed6113).
All applicable first/higher input/weight paths remain live, and fields own storage.

\code{nn$Pix2Pix(g,d,target,source,l1_weight=100)} requires distinct matching stores
and aligned finite signed pairs. D sums two patch-mean logistic losses without
half. G is mean softplus(-fake logits)+weight*mean absolute paired pixel error.
Finite nonnegative representable weight; zero retains diagnostic L1. Owns seven
live image/logit/objective fields. Caller captures D partials/steps D, then makes
a fresh G forward through updated D and explicitly steps G. No fitting, optimizer,
capture, hidden no-grad or batch-state mutation. Invalid inputs raise R errors.
See pix2pix.hpp and Isola et al.(2017), https://arxiv.org/abs/1611.07004 .
}
\section{Progressive image and feature fades}{
\code{nn$pixel_rms_normalize(input,epsilon=9.99999993922529e-9)} divides last-axis
features by sqrt(mean(square(input))+epsilon), without centering, affine or batch
coupling. Finite Float32/64 CPU[...,C], rank>=1, positive axes and representable
epsilon; default is Float32-rounded even in Float64. First/higher paths live.
\code{nn$ProgressiveActiveGroups(stage,generator=TRUE)} returns an R integer
vector: stage0 G(0,1), stage1 G(0,1,2,3), stage2 G(0,2,3); D adds4. Activity
depends on stage, not alpha: old RGB still advances Adam at the final alpha1 fade
batch. Inactive groups receive no step/counter advance. No fitting or store writes.

\code{nn$ProgressiveGanGenerate(Gcore,Gold,Gblock,Grgb,z,alpha)} uses four distinct
matching Float32/64 CPU stores, finite noise[N,16] and matching scalar or[1] alpha
in[0,1]. Source tanh stem, normal(.02) biased convs, LeakyReLU(.2), pixel RMS,
old4/new8 sigmoid images and actual pixel fade. Counts1236,5,296,5;12 leaves.
\code{nn$ProgressiveGanDiscriminate(Dcore,Dold,Dblock,Drgb,image,alpha)} accepts
signed[N,8,8,1] with the same four-store/fade contract. Old path pools input before
RGB; new path pools after RGB/two convs. Counts213,8,296,8;12 leaves. Owns four
4x4x4 feature fields and a raw score. Both networks always evaluate both branches.

\code{nn$ProgressiveGan(Gcore,Gold,Gblock,Grgb,Dcore,Dold,Dblock,Drgb,image,z,alpha)}
requires eight distinct matching stores and positive aligned image/noise rows.
Owns seven live endpoints/scores/losses. D=.5*(mean((real-1)^2)+mean(fake^2));
G=mean((fake-1)^2), with NO half factor. Capture ALL active D gradients before any
D write, then a fresh G forward through updated D and ALL active G gradients before
G writes. Retain the graph across per-group calls; replace unused partials by zeros.
All2067 values/24 leaves initialize; stage-selected counts1462/2067/2054. No hidden
optimizer, gradient capture, stage, freeze, fitting or checkpoint. Freezing old RGB
weights does not freeze the shared core. Owning tensor fields and first/higher
input/fade/weight paths survive parent/store collection. Synchronize stores; invalid
inputs/stores/nonfinite computation raise errors.
}

\section{Wasserstein maps and explicit constraints}{
\code{nn$MidpointUniform(rows,columns=1,seed=6601,dtype=Float32)} consumes one
local MT19937 word per coordinate and applies (Double(word)+.5)/2^32 before one
final cast. Positive representable dimensions, Float32/64. Owns detached CPU matrix;
Double is strictly inside(0,1), Float32 may round to1. No other RNG stream consumed.
\code{nn$WassersteinGenerate(G,z)} and \code{nn$WassersteinCritic(C,x)} accept
finite matching Float32/64 CPU[N,2], N>0. Both reuse native tanh16/linear maps;
G82 and C65 values/four leaves each. G returns unbounded[N,2], C raw[N,1].
\code{nn$ClipWassersteinCritic(C,bound=.01)} validates all finite selected tensors
before any write and clips every leaf including biases; names/order/flags retained.
Call after initialization and EVERY selected clipped-WGAN critic step. No optimizer
or fitting hidden; no Lipschitz certificate. Synchronize stores, avoid old graphs.

\code{nn$WassersteinGan(G,C,real,z,alpha,penalty_weight=10,penalty_graph=TRUE)}
requires distinct matching stores, finite aligned real/noise[N,2], alpha[N,1] in[0,1].
A detached interpolation leaf stops external real/alpha/G penalty paths. The actual
input Jacobian is computed under internally enabled recording, even during no-grad.
TRUE retains critic parameter history for penalty fitting; FALSE detaches it.
Outer no-grad plus FALSE gives fully detached evaluation. The analytic Jacobian
is a separate diagnostic. Norm is sqrt(sum(J^2)+1e-12), penalty mean((norm-1)^2),
gap mean(real)-mean(fake), G loss -mean(fake), critic loss -gap+weight*penalty.
Weight0 skips the added term while reporting all diagnostics. Base critic loss
still has a G path, so caller chooses update groups. All eleven owning fields
retain storage/applicable graphs through parent/store collection. No clipping,
optimizer, fitting, checkpoint or certified distance hidden. Errors raise R errors.
}

\section{DCGAN and fused affine BN captures}{
\code{nn$capture_affine_batch_norm(parameters,moving,name,input,batch_statistics=FALSE,epsilon=.001)}
uses native fused learned-affine BN with Double epsilon. Finite matching Float32/64
CPU input[...,C], rank>=2, positive axes; distinct matching stores, nonempty prefix.
Creates gamma/beta before moving mean/variance. Batch mode returns owning live value
and detached mean/unbiased_variance; inference uses stopped saved moments and undefined
captures. No moving write after initialization. First/higher continuous paths live.

\code{nn$DcganGenerate(G,Gstate,z,batch_statistics=FALSE)} accepts finite matching
noise[N,16], N>0 and distinct stores. Projection7x7x8, two4x4 stride2 transposed maps,
fused BN, source normal(.02) kernels, signed tanh pixels. G6873/eight leaves, state24/four.
Owns image[N,28,28,1] and four detached moment vectors, undefined in inference.
\code{nn$DcganDiscriminate(D,Dstate,image,batch_statistics=FALSE)} accepts signed
[N,28,28,1], uses two strided maps, LeakyReLU(.2), fused BN and a raw head.
D989/seven leaves, state16/two. Owns raw logit[N,1] and optional detached moments[8].

\code{nn$DcganDiscriminatorPhase(G,D,Gstate,Dstate,real,z)} requires four distinct
matching stores and finite aligned real[N,28,28,1]/noise[N,16]. Real unit coordinates
map to2*real-1 without clipping. G uses batch BN with stopped paths; D batch-normalizes
concat(real,fake) jointly. Returns logistic loss and detached G/D captures without
writes. Commit only D captures using \code{batch_norm_moving_values(...,.9)}, then
selected D Adam. \code{nn$DcganGeneratorPhase(G,D,Gstate,Dstate,real,z)} runs fresh
batch-BN G through current saved-BN D. Select G gradients, commit both G layers with
explicit Double .9 retention, then G Adam. Real-logit/D-capture fields undefined.
D weight paths remain available, so caller chooses the update group.

\code{nn$Dcgan(G,D,Gstate,Dstate,real,z)} owns unit pixels[N,784], raw/probability
scores and logistic d_loss/g_loss with saved BN in both maps. Compute coarse MMD
separately. All result fields retain storage/applicable graphs through parent/store
collection; captures/saved moments stop. No implicit EMA, optimizer, fitting or
checkpoint. Invalid inputs/stores/nonfinite math raise errors. Synchronize stores.
}

\section{Spatial hierarchy and miniature NVAE}{
\code{nn$LadderPrecisionFusion(evidence_mu,evidence_lv,prior_mu,prior_lv)}
combines Gaussian precisions using a detached maximum shift. Four finite matching
Float32/64[B,L] matrices retain all input paths. Its owning mu/log_variance
results use the source operation order, not residual-head addition. Equal
sources halve variance, up to rounding; extreme-log cancellation is not repaired.
\code{nn$LadderPrior(parameters,top,lower_noise,features=784L,lower_latent=4L,
top_latent=2L,hidden=32L,top_hidden=16L)} performs image-free ancestral sampling
from explicit matching top[B,T] and lower_noise[B,L]. All six fields own lifetime.
Only lower_z enters decoding; top affects it through the learned conditional
prior. Default prior-only creation has ten leaves/26216 scalars. No encoder,
image, new draws, optimizer or hidden no-grad is invoked.
\code{nn$LadderVariationalAutoencoder(parameters,image,epsilon,features=784L,
lower_latent=4L,top_latent=2L,hidden=32L,top_hidden=16L)} accepts unit-range
image[B,F] and explicit noise[B,L+T], lower noise first and top noise last.
All sixteen fields own lifetime. Twenty-two default leaves have52196 scalars.
Loss is mean(NLL+(conditional lower KL+top KL)); lower KL depends on sampled top
codes. Only the lower code reaches GaussianVaeDecode. All image/noise/parameter
paths remain live, and the caller owns Adam, draws and validation selection.
No BN, warmup, free bits or variance clipping is hidden. Invalid native inputs
raise errors before leaves where possible; incompatible existing leaves may fail later.

\code{nn$ImqKernel(x,y)} averages c/(c+distance_squared) for c=1,4,16,
returning owning[N,M] from finite matching Float32/64[N,D] and[M,D]. Unequal
counts, strides and both input gradients are supported, with O(N*M*D) storage.
\code{nn$ImqMmd(x,y)} requires N,M>=2 and returns seven owning fields xx,yy,xy,
within_x,within_y,cross,value. Only within diagonals are excluded; every cross
pair counts. Negative unbiased MMD squared and negative self-comparisons remain
signed, with no square root or calibrated hypothesis test.
\code{nn$InfoVaeObjective(nll,kl,mmd,alpha=.5,lambda_weight=10)} combines
mean(NLL+(1-alpha)*KL)+(alpha+lambda_weight-1)*MMD. Finite matching vectors and
one finite same-dtype MMD value keep all derivatives. Alpha lies in[0,1] and
lambda_weight>=1; Float32 coefficients preserve source rounding. Alpha1 removes
local KL; the general objective is not an ordinary ELBO or information estimate.
\code{nn$InfoVaeDecode(parameters,code,features=784L,latent=4L,hidden=16L)}
returns image-free tanh/linear logits. Its four leaves/default13408 scalars share
the full model's decoder. No posterior, MMD or optimizer is invoked.
\code{nn$InfoVariationalAutoencoder(parameters,image,epsilon,prior,alpha=.5,
lambda_weight=10,features=784L,latent=4L,hidden=16L)} accepts unit-range image[B,F],
explicit noise[B,L] and independent prior[P,L], B,P>=2. All sixteen fields own
lifetime; ten default leaves have26104 scalars. Prior changes affect MMD/loss,
not posterior/decoder endpoints. Negative MMD is retained, negative_elbo is a
separate diagnostic, and the caller owns Adam and epoch selection. Invalid
inputs/controls/products raise native errors before leaf creation where possible.

\code{nn$IwaeLogMeanExp(log_weights)} reduces finite Float32/64 log weights[K,B]
to owning[B] log-mean-exp values with full derivatives. A stopped column maximum
avoids raw exponential overflow; Float32 preserves the source-rounded log(K),
Float64 uses full precision. Both dimensions must be positive.
\code{nn$ImportanceWeightedBound(mu,log_variance,z,nll,samples)} accepts matching
finite Gaussian[KB,L], signed finite nll[KB] and positive K dividing KB. Its five
owning fields are log_weights, normalized_weights, negative_bound,
effective_samples and scalar loss. Full ordinary pathwise gradients pass through
all inputs; no analytic-KL or doubly reparameterized estimator is substituted.
\code{nn$ImportanceWeightedAutoencoder(parameters,image,epsilon,features=784L,
latent=4L,hidden=32L)} accepts unit-range image[B,F] and explicit noise[K,B,L].
Ten leaves/51416 default scalars use the existing Gaussian VAE names. All eleven
fields own lifetime, kl is diagnostic only, and the caller owns optimizer and
checkpoint selection. K=1 is sampled negative ELBO. GaussianVaeDecode handles
image-free final-code decoding. Invalid inputs raise native errors; computed
nonfinite values are not repaired.

\code{nn$HierarchyPosterior(prior_mu,prior_lv,head_mu,head_lv,residual=TRUE)}
returns owning Gaussian mu/log_variance from direct heads or additive residual
corrections. Matching finite Float32/64[B,L] inputs retain all applicable paths;
zero residuals give the prior exactly. This is not precision-weighted fusion.
\code{nn$ConditionalGaussianKl(mu,lv,prior_mu,prior_lv)} sums the analytic
conditional diagonal-Gaussian KL over latent coordinates, returning[B] nats.
All four finite matching inputs retain gradients; negative roundoff is not clipped.
\code{nn$HierarchyNvaeCell(parameters,state,x,training=FALSE,update_stats=FALSE)}
uses a bias-free expansion/depthwise/projection BN/Swish cell and spatial-mean SE
gate, returning x+.1*project*gate for finite NHWC features. With four channels it
has13 trainable leaves/198 scalars and six buffers/40 scalars. BN uses momentum.9,
epsilon.001, population variance for normalization and unbiased variance in moving
updates. Training uses batch moments; update_stats=TRUE also updates moving moments
and requires training. Inference detaches moving buffers. Initial calls may create
missing entries; existing weights never change. State belongs in a separate same-dtype
store, excluded from gradients/Adam, and BOTH stores must be checkpointed/restored.
\code{nn$HierarchyDecode(parameters,state,lower,top,nvae=FALSE,height=28L,
width=28L,channels=4L)} decodes finite matching[B,Hq*Wq] lower and[B,T] top samples
without image/encoder input. Both sampled groups feed the convolutional decoder.
\code{nn$HierarchyPrior} instead accepts top samples and lower Gaussian epsilon,
returns six owning ancestral endpoints, and evaluates learned p(z1|z2) without a
posterior. Both inference functions use moving NVAE moments; no RNG, no-grad scope,
fitting or existing-state change is hidden. Lazy calls can create missing entries.
\code{nn$HierarchicalVariationalAutoencoder(parameters,state,image,epsilon,
nvae=FALSE,training=FALSE,update_stats=FALSE,height=28L,width=28L,channels=4L,
top_latent=2L)} has26 default leaves/1977 scalars, or37/2027 plus six BN buffers
for NVAE. Owns19 endpoints. Image is finite unit-range[B,H*W], supplied noise is
matching[B,Hq*Wq+T], lower first/top last. Loss is mean(summed pixel NLL+
conditional lower KL+top KL). NVAE train_loss adds1e-4 times two smoothed prior
kernel norms, epsilon1e-12, not an all-kernel schedule. No optimizer or fitting is
hidden. Validation precedes creation; incompatible leaves/nonfinite computed
states may fail later. All continuous paths/strides remain. Separate matching
floating stores and safe positive multiples-of-four geometry are required.
See Vahdat and Kautz, \url{https://arxiv.org/abs/2007.03898}, and Sonderby et al.,
\url{https://arxiv.org/abs/1602.02282}. This is a small two-group teaching adaptation.
}
\section{Gumbel categorical components}{
\code{nn$GumbelCategorical(logits,noise,temperature=.7)} accepts finite matching
Float32/64[B,C] logits and[M,C] explicit noise, C>=2; rows match or one is1.
Owns posterior log probabilities/probabilities[B,C], relaxed[max(B,M),C] and
Int64 hard Gumbel-max indices. Positive representable temperature is narrowed
for Float32; q and hard IDs ignore temperature, relaxed values/gradients do not.
Both continuous inputs retain gradients; there is no straight-through path or RNG.
\code{nn$CategoricalUniformKl(q,logq)} sums q*(logq+log C) by row, keeping both
gradients. Matching finite floating[B,C], C>=2, q in[0,1]; normalized q and matching
logq are the caller's obligation. Float32 log C is source-rounded. This is a
categorical KL, not a continuous Concrete-density KL; roundoff is not clipped.
\code{nn$GumbelNoise(n,samples,seed,groups=4L,classes=8L)} owns detached
uniform/gumbel Float64 and noise Float32[S,B,G,C]. Positive dimensions/C>=2 and
element/byte overflow are checked. A local full-uint32 MT19937 gives u=(r+.5)/2^32
and -log(-log(u)), without global RNG or endpoint clipping.
\code{nn$GumbelDecodeSoft(parameters,simplex,features=784L,groups=4L,
classes=8L,hidden=32L)} takes finite matching floating[B,G,C] unit-range simplex
codes, each group summing to1 within strict2e-6(Float32)/1e-12(Float64).
\code{nn$GumbelDecodeIds} takes Int32/64[B,G] IDs in[0,C). Both return owning
Bernoulli probabilities, share decoder/output leaves, and preserve decoder
parameter gradients (also continuous simplex gradients for soft codes).
No image, encoder, RNG, no-grad scope or fitting is hidden; validation precedes creation.
\code{nn$GumbelCategoricalAutoencoder(parameters,image,noise,temperature=.7,
features=784L,groups=4L,classes=8L,hidden=32L)} has eight default leaves/53104
scalars and14 owning endpoints. Inputs are unit-range finite image[B,F] and
explicit finite noise[B,G,C] matching the Float32/64 store. Loss is mean(soft
reconstruction NLL+discrete categorical KL): a biased continuous surrogate,
not an exact discrete ELBO or Concrete KL. The matched hard draw gives a
separate Monte Carlo discrete negative-ELBO diagnostic. All continuous paths
stay live; no straight-through estimator, hidden fit, optimizer or selection.
See Jang et al., \url{https://arxiv.org/abs/1611.01144}, and Maddison et al.,
\url{https://arxiv.org/abs/1611.00712}.
}
\section{Conditional affine-flow posterior components}{
\code{nn$ConditionalAffinePosteriorFlow(parameters,name,code,context,
swap=FALSE,inverse=FALSE,latent=4L,hidden=16L)} applies an affine coupling to
finite Float32/64[B,latent] code with matching[B,C] context. Positive even latent,
positive hidden/C and nonempty prefix are validated before creating leaves.
The unchanged half plus context feeds a Glorot ReLU/linear conditioner with
scale=.5*tanh(raw) and unbounded shift. Forward b*exp(scale)+shift and inverse
(b-shift)*exp(-scale) return signed sum-scale logdet. Swap chooses the unchanged
half; output coordinate ordering is preserved. Context stays fixed for the code
Jacobian; code, context and parameter gradients all remain live. Reuse exactly
the same prefix/context/swap/widths for inversion. All four output fields own lifetime.
\code{nn$FlowVariationalAutoencoder(parameters,image,epsilon,features=784L,
latent=4L,hidden=32L,flow_hidden=16L)} has eighteen default leaves/52672 scalars.
Matching finite unit-range image[B,features] and explicit noise[B,latent] produce
Gaussian z0 then two alternating context-conditioned couplings. Only final z
reaches the Bernoulli decoder. Eighteen owning fields retain all continuous paths.
Loss is mean(summed pixel NLL+logq0-forward_logdet-logp(final_z)), with signed
sample log ratios retained. No analytic base-KL replacement, RNG, optimizer,
checkpoint selection or fitting is hidden. Zero noise gives decode(f(mu)), not
generally a posterior predictive mean. Standard-normal prior final codes use
GaussianVaeDecode directly, without context or posterior flow. Inverse endpoints
are diagnostics using the same leaves. Malformed inputs fail before creation;
incompatible leaves/nonfinite computed arithmetic may fail later.
See Rezende and Mohamed, \url{https://proceedings.mlr.press/v37/rezende15.html},
and Dinh et al., \url{https://arxiv.org/abs/1605.08803}.
}
\section{Total-correlation variational components}{
\code{nn$EmpiricalTotalCorrelation(z,mu,log_variance)} uses finite matching
Float32/64 nonempty z[B,L] and reference heads[R,L]. It returns owning
log_joint[B],log_product[B],per_case[B],mean scalar, with gradients through
samples AND reference heads. All R components form an equal-weight empirical
Gaussian mixture; this is not a minibatch-weighted estimator. Signed ratios and
finite-sample means are retained; overflow is not repaired. References must be
fitting-only, a provenance obligation of the caller.
\code{nn$FactorDiscriminator(parameters,z,latent=4L,hidden=16L)} is a tanh/linear
four-leaf97-scalar density-ratio model with hidden/W,b and ratio/W,b; positive
logit means joint, negative means product. \code{nn$FactorDiscriminatorLoss}
returns0.5*(mean softplus(-D(joint))+mean softplus(D(product))), allowing distinct
positive batch sizes. All continuous paths remain. Callers detach posterior
samples and request/update D leaves only in its separate phase.
\code{nn$PermuteLatentColumns(joint,seed)} independently shuffles columns of
finite Float32/64[B,L] rows using one local MT19937 stream, preserving each
marginal exactly. Owns detached same-shape/dtype rows, without changing global
RNG or input. Reproducibility requires matching C++ shuffle implementations.
\code{nn$FactorVariationalAutoencoder(parameters,discriminator,image,epsilon,
extra_weight=3,features=784L,latent=4L,hidden=32L,disc_hidden=16L)} returns owning
net (Gaussian endpoints/base losses),tc and augmented loss. Distinct matching
floating stores, unit-range nonempty images, explicit matching noise, positive
widths and finite representable nonnegative weight are validated before creation.
VAE loss adds extra_weight times mean D(z), preserving the z path through a
fixed D mapping when the caller updates only VAE parameters. No detachment,
training, RNG or checkpoint selection is hidden.
\code{nn$BetaTcVariationalAutoencoder(parameters,image,epsilon,reference_images,
extra_weight=3,features=784L,latent=4L,hidden=32L)} adds exact finite-mixture TC
at sampled z, using ALL supplied fitting-only reference images and live shared
encoder/head gradients. Default extra3 corresponds to beta_TC4; base NLL and
analytic KL retain weight1. There is no reference decoder, hidden RNG or fitting.
Negative surrogates are retained. Neither finite discriminator logits nor these
linear/recovery experiments certify true TC or disentanglement.
See Kim and Mnih, \url{https://proceedings.mlr.press/v80/kim18b.html}; Chen et al.,
\url{https://arxiv.org/abs/1802.04942}.
}
\section{Convolutional Gaussian variational autoencoders}{
\code{nn$ConvolutionalVariationalAutoencoder(parameters,image,epsilon,
latent=2L,channels=8L,base_channels=4L)} takes finite unit-range Float32/64
NHWC[B,H,W,1], H/W positive multiples of four, and explicit finite[B,latent]
noise matching store dtype/batch. Two He-ReLU convolutions with nonoverlapping
average pooling feed separate linear Gaussian heads. The explicit sampled z is
the ONLY decoder bridge; pool2 is an inspection endpoint, never a skip input.
Fourteen default leaves hold2117 scalars. Gaussian heads, ReLU dense projection
and final convolution use Glorot uniform; ReLU convolutions use He uniform;
biases zero. Returns all nine GaussianVaeOutput fields plus conv1,pool1,conv2,
pool2,decoder_base,decoder1,spatial_logits, with owning continuous gradients.
loss and unweighted_loss both equal mean(summed pixel NLL+summed latent KL),
beta1, fixed N(0,I) prior. Binary targets give a negative ELBO estimate; soft
values give cross-entropy rather than a continuous-observation density.
\code{nn$ConvolutionalVaeDecode(parameters,code,height=28L,width=28L,
latent=2L,channels=8L,base_channels=4L)} requires only finite signed/zero matching
Float32/64 codes[B,latent]. Dense/ReLU projection, reshape, nearest repeat2,
He-ReLU convolution, repeat2 and Glorot-linear convolution give five owning
fields: decoder_base,decoder1,spatial_logits,logits,probability. Last two are
flattened[B,H*W]; sigmoid probabilities are not sampled pixels. Six default
leaves hold957 scalars. Positive widths/representable H,W multiples of four
are checked before leaf creation; invalid inputs raise an R error. Incompatible
leaves/nonfinite computed states may fail later. No image, labels, hidden RNG,
fitting, clipping or spatial bypass. Decode(mu) is not the predictive mean.
See Kingma and Welling, \url{https://arxiv.org/abs/1312.6114}.
}
\section{Class-conditioned variational autoencoders}{
\code{nn$ConditionalVariationalAutoencoder(parameters,image,labels,epsilon,
features=784L,classes=10L,latent=2L,hidden=64L)} conditions both the Gaussian
posterior and Bernoulli decoder on supplied classes. Finite nonempty unit-range
images[B,features] and explicit epsilon[B,latent] match store Float32/64 dtype.
Labels are Int32/64[B] in [0,classes), widths positive. Validation precedes leaf
creation; incompatible leaves/nonfinite computed states can fail later. All
continuous image/epsilon/parameter gradients remain; conditions are discrete.
Ten Glorot/zero-bias leaves hold102932 default scalars. Returns an owning
ConditionalVaeOutput with all nine GaussianVaeOutput fields plus condition,
encoder_input and decoder_input. The same condition is concatenated after
images and after samples. loss and unweighted_loss both equal mean(summed-pixel
Bernoulli NLL + summed-coordinate Gaussian KL), beta1, fixed shared N(0,I) prior.
Binary targets give a negative ELBO estimate; soft targets give cross-entropy,
not a continuous-observation likelihood density.
No RNG, label prediction, annealing, fitting or selection is hidden.
\code{nn$ConditionalVaeDecode(parameters,code,labels,features=784L,
classes=10L,latent=2L,hidden=64L)} takes finite signed/zero codes[B,latent] and
required supplied labels only. Returns owning differentiable logits[B,features]
using the same decoder/W,b and output/W,b leaves; sigmoid gives probabilities.
Invalid conditions raise an R error rather than producing all-off one-hot rows.
Supplied classes are required at inference. Decode(mu) is not the posterior
predictive mean. See Kingma and Welling, \url{https://arxiv.org/abs/1312.6114}.
}
\section{Gaussian variational autoencoders}{
\code{nn$GaussianReparameterize(mu,log_variance,epsilon)} computes
mu+exp(log_variance/2)*epsilon with no RNG, clipping or detachment. Finite
matching nonempty Float32/64 [B,L] inputs retain all three gradients and strides.
\code{nn$DiagonalGaussianKl(mu,log_variance)} returns per-row KL to N(0,I),
summed over latent coordinates, in nats. Zero means/log variances give exact zero.
Computed exponent overflow is not repaired; invalid inputs raise an R error.
\code{nn$BetaVariationalAutoencoder(parameters,image,epsilon,beta=1,
features=784L,latent=4L,hidden=32L)} creates/reuses ten Glorot/zero-bias leaves:
encoder/W,b; mu/W,b; log_variance/W,b; decoder/W,b; output/W,b. ReLU hidden
layers and separate linear Gaussian heads give51416 default scalars. Images
are finite unit-range [B,features]; epsilon is explicit finite [B,latent], both
matching the store Float32/64 dtype. Positive beta must remain representable in
that dtype; all widths are positive. Input checks precede creation; incompatible
stored leaves/nonfinite computed states may fail later. It owns mu,
log_variance,z,logits,probability,reconstruction_nll,kl,loss,unweighted_loss.
Pixels and coordinates are summed before averaging images: loss is mean(NLL+
beta*KL), unweighted_loss is mean(NLL+KL). No fitting/annealing/selection is
hidden. Decode(mu) is not the posterior predictive mean.
\code{nn$GaussianVaeDecode(parameters,code,features=784L,latent=4L,hidden=32L)}
uses only finite signed/zero code[B,latent] with the same decoder/output leaves;
returns owning differentiable logits[B,features]. Apply sigmoid for probabilities.
\code{nn$LatentFactorDiagnostics(fit_mu,test_mu,fit_factors,test_factors,
fit_log_variance,ridge=.001)} returns thirteen named detached owning arrays.
Fitting population means/std (floor1e-8), centered targets, ridge coefficients
and maximum absolute fitting-correlation axes use fitting rows only. Means and
log variances are finite matching Float32/64; numeric factor matrices share the
respective split row counts and target dtype. Test targets must each vary. Ridge
is finite positive. Results: latent_mean/std,mu_variance,mean_kl_per_unit [L];
factor_mean [K]; fit/test_correlation,ridge_coefficients [L,K]; Int32
best_axis_by_fit [K]; ridge/single_prediction [Nt,K]; ridge/single_r2 [K].
Other arrays are Float64. Ties choose lower axes; test labels never change fitting.
Negative heldout R-squared is retained. These linear-recovery probes are not
classification accuracy or a certification/benchmark of disentanglement.
See Kingma and Welling, \url{https://arxiv.org/abs/1312.6114}; Higgins et al.,
\url{https://openreview.net/forum?id=Sy2fzU9gl}.
}
\section{Canonical autoencoder components}{
\code{nn$MaeSelection(source,epoch=0L,seed=1907,dtype=nn$DType$Float32)}
returns detached [B,4,16] selectors from Int32/64 uint32-range IDs, with four
distinct one-hot rows in hash order. Epoch is uint32-range; seed supports the
full uint32 range and no R/global RNG is changed. Empty ID vectors are allowed.
\code{nn$MaePosition(width,dtype)} returns grouped row-sine/row-cosine/column-sine/
column-cosine [16,width] positions; width is a positive multiple of four and
dtype defaults to Float32 (Float64 also supported). \code{nn$MaePixel(patch,within)}
maps zero-based patch [0,15] and pixel [0,48] indices into a flat 28-by-28 image.
\code{nn$MaePatches(image)} maps finite nonempty Float32/64 [B,28,28,1] to
[B,16,49]; \code{nn$MaeImage(patches)} differentiably inverts it to [B,784],
including strided tensors and empty batches for the inverse.
\code{nn$MaeBlock(parameters,name,x,width)} is a two-head pre-norm block with
LayerNorm epsilon1e-5, ratio-2 tanh-GELU MLP and no dropout/causal mask.
\code{nn$MaskedAutoencoder(parameters,image,selector)} composes a width-32
four-token encoder and width-16 sixteen-token decoder (35 leaves,13649 scalars).
Matching detached one-hot selectors are validated before parameter creation.
\code{nn$MaeEncode} and \code{nn$MaeDecode} expose the two halves; the latter
accepts finite [B,4,32] codes and locations only, with no image or target.
Results own endpoint handles and continuous gradients. Encode-only fields are
patches, visible, mask, encoder_input, code and encoder_attention; decode-only
fields are code, mask, projected, restored, decoder_input, prediction and
decoder_attention. Other partial fields are undefined. Full forwards populate
all fields. \code{nn$MaeLoss(prediction,patches,mask)} scores matching floating
[B,P,D] against a detached binary hidden mask[B,P], divided by D*sum(mask).
At least one patch must be hidden. It retains prediction/target gradients and
does not clip values; nonfinite arithmetic may propagate through zero weights.
Invalid shapes/dtypes/selectors raise an R error. Shared stores require external
synchronization; all fitting and selection loops remain caller-owned.
See He et al., \url{https://arxiv.org/abs/2111.06377}; this tiny MNIST adaptation
uses hash masks and raw-pixel sigmoid targets, not the ImageNet benchmark.
\code{nn$CompetitiveMask(scores, lifetime, k=8L)} returns detached binary
Float32/64 winners with the same shape as finite scores[B,F]. Per-row mode
keeps min(k,F) largest signed values; lifetime mode keeps ceil(B/8) rows per
feature. Ties prefer lower indices. Empty axes and strided tensors are supported;
negative k and malformed/nonfinite inputs raise an R error.
\code{nn$CompetitiveAutoencoder(parameters,input,lifetime,plain=FALSE,
features=784L,latent=64L,k=8L)} creates/reuses independent Glorot encoder/W,b
and decoder/W,b leaves (101200 scalars by default). Per-row mode has a linear
encoder and retains competition at deployment; lifetime mode uses ReLU and
bypasses competition at deployment. The matched plain control bypasses the mask.
Both forward paths are explicit: owning raw, mask, training_code, code,
training_reconstruction and reconstruction handles remain valid after parent
collection. Every continuous path retains gradients; the mask is detached.
No optimization is performed. Input must be nonempty matching Float32/64
[B,features], widths positive and k nonnegative. Invalid arguments raise an R
error. See Makhzani and Frey, \url{https://arxiv.org/abs/1312.5663} and
\url{https://arxiv.org/abs/1409.2752}; this small untied sigmoid/MSE variant
does not reproduce the original classification benchmarks.
\code{nn$GaussianNoise(shape, samples, seed)} creates a detached Float64 bank
\code{[samples,shape...]} using a local 32-bit Mersenne Twister and explicit
Box-Muller cosine/sine pairs. Nonnegative empty dimensions/counts are allowed;
negative dimensions raise an R error. This stream differs from \code{nn$normal}
and \code{nn$Mt19937} normal samplers and does not change R's random state.
Seeds are numeric whole values in [0,4294967295], including values beyond R's
signed integer range; fractional, nonfinite and out-of-range seeds raise an error.
\code{nn$CorruptForDenoising(clean, standard_noise, sigma)} requires finite
unit-range Float32 clean pixels of rank at least two, a finite Float64 bank
\code{[1,clean.shape()...]} and finite nonnegative Float32 sigma. It evaluates
the sum in Float64, rounds to Float32, then clips to [0,1]. Returns newly owned,
detached augmentation data with clean's shape. Inputs are not mutated; strides
and empty axes are supported. Invalid values/shapes/dtypes raise an R error.
Clean pixels remain the target of the separately evaluated reconstruction loss.
See Vincent et al. (2008), \url{https://doi.org/10.1145/1390156.1390294}; this
example uses clipped Gaussian rather than the paper's masking corruption.
\code{nn$UndercompleteAutoencoder(parameters, input, features=784L, latent=32L)}
uses an independent ReLU encoder and sigmoid decoder, requiring a strictly
smaller positive code width. \code{nn$ContractiveAutoencoder(parameters, input,
weight, features=784L, latent=32L)} uses sigmoid on both layers and accepts any
positive code width. Both require a nonempty floating matrix
\code{input[N,features]} matching the parameter store's dtype. They create or
reuse \code{encoder/W}, \code{encoder/b}, \code{decoder/W} and \code{decoder/b};
they perform no optimization. R owns batching, optimizer updates and selection.
The contractive result contains \code{code}, \code{reconstruction}, \code{mse},
\code{penalty} and \code{loss}. Its objective is MSE over all pixels plus finite
nonnegative \code{weight} times the mean squared encoder Jacobian norm.
\code{nn$SigmoidEncoderContraction(code, encoder_weight)} accepts matching
Float32/64 matrices \code{[N,K]} and \code{[D,K]}. It returns \code{per_example}
norms summing all D*K derivative coordinates and their batch \code{mean}.
Both activation and weight gradient paths are retained. Returned tensors and
nested penalty values remain valid after their parent result is collected.
Malformed shapes, dtypes or settings raise an R error. Inputs are not clipped
or checked for finite values. See Rifai et al. (2011),
\url{https://icml.cc/Conferences/2011/papers/455_icmlpaper.pdf}; the migrated
variant keeps independent weights and MSE rather than the paper's tied weights
and cross-entropy experiments.
\code{nn$ConvolutionalAutoencoder(parameters, image)} accepts nonempty NHWC
Float32/64 \code{[N,H,W,1]} matching the store, with H,W positive multiples of
four. It returns owning \code{conv1}, \code{pool1}, \code{conv2}, \code{code},
\code{decoder1} and \code{reconstruction} tensors, retaining gradients.
The 741-scalar model has a \code{[N,H/4,W/4,4]} code, two ReLU convolutions
with average pooling, then two independent convolutions with nearest repetition
and a sigmoid output. \code{nn$ConvolutionalDecoder(parameters, code)} accepts
\code{[N,H,W,4]} and returns owning \code{hidden[N,2H,2W,8]} and
\code{reconstruction[N,4H,4W,1]} tensors, creating/reusing 369 decoder scalars.
The eight named leaves are \code{encoder1/W,b}, \code{encoder2/W,b},
\code{decoder1/W,b}, \code{decoder2/W,b}. Neither function fits or clips inputs.
Invalid shapes/dtypes raise an R error. Nearest repetition is distinct from
transposed convolution; this untied small model is not the stacked pretraining
experiment in Masci et al. (2011),
\url{https://people.idsia.ch/~ciresan/data/icann2011.pdf}.
}

\section{Native recurrent sequence autoencoder}{
\code{nn$ImageRowSequence(images, steps=28L, features=28L)} accepts finite
Float32/64 images \code{[B,T,F,1]}, sequences \code{[B,T,F]} or flat rows
\code{[B,T*F]}, with positive dimensions. It preserves row order, dtype,
values and gradients without resizing or clipping.
\code{nn$RecurrentEncode(parameters, sequence, steps=28L, features=28L, units=32L)}
uses a zero-initialized shared tanh cell and exposes only its final code to the
independent decoder. Requires T>1, F>0 and 0<U<T*F, matching dtype and shape.
\code{nn$RecurrentDecode(parameters, code, steps=28L, features=28L)} repeats
finite \code{[B,U]} codes through an independent zero-initialized tanh cell,
then a shared sigmoid row projection; T,F are positive. It has no image,
target, teacher forcing or encoder-state bypass. Every step retains gradients.
\code{nn$RecurrentAutoencoder} composes these halves and returns owning
\code{code[B,U]}, \code{reconstruction[B,T,F]} and both \code{[B,T,U]} histories.
Partial encoder/decoder results leave unrelated fields undefined. At defaults,
eight tensors contain 4956 scalars. Shapes/dtypes/nonfinite inputs raise an R
error before new leaves are created; incompatible existing leaves can also fail.
The caller owns loss, Adam, loops and selection. See Elman (1990),
\url{https://doi.org/10.1207/s15516709cog1402_1}, and Werbos (1990),
\url{https://doi.org/10.1109/5.58337}; the repeated-code architecture retains
this small MNIST source experiment rather than a language translation model.
}

\section{Native activation sparsity}{
\code{nn$ActivationSparsity(code, l1_weight, kl_weight, rho=.05, epsilon=1e-5)}
accepts finite nonempty Float32/64 \code{code[B,F]}, nonnegative finite weights,
0<rho<1 and 0<epsilon<.5. Constants, complements and clipping endpoints round
to the code dtype and must remain finite/interior. L1 averages absolute activities
over B*F; KL averages target-to-clipped-batch-unit-mean Bernoulli divergences over
F. Owning fields \code{l1,kl,total,unit_mean,clipped_mean} retain code gradients;
abs has zero derivative at zero, clip passes endpoint derivatives and zeros those
outside. Signed codes are accepted; KL then concerns clipped means. No hard
zeros or weight penalty is imposed. Batch splitting can change the KL objective.
\code{nn$SparseAutoencoder(parameters,image,l1_weight,kl_weight,features=784L,latent=64L,rho=.05,epsilon=1e-5)}
uses matching finite \code{[B,features]} rows and independent sigmoid dense layers.
Default four leaves \code{encoder/W,b,decoder/W,b} contain 101200 scalars.
Returns owning \code{code,reconstruction,mse,loss,penalty}; nested penalty handles
remain valid when the parent is collected. Invalid options/inputs raise R errors
before new leaves; incompatible existing leaves can fail later. Gradients remain;
the caller owns Adam, loops, scoring and selection. This averages unit KL and adds
activation L1 rather than weight decay; see A. Ng, Sparse autoencoder notes,
\url{https://web.stanford.edu/class/cs294a/sparseAutoencoder.pdf}.
}

\section{Native symmetric sigmoid stacks}{
\code{nn$SigmoidAutoencoder(parameters,image,widths)} returns an owning R list of
all layer outputs, excluding input and loss. Widths are positive whole int64
values, odd in count (at least three), mirrored and strictly decreasing into
the code. Finite Float32/64 image rows \code{[B,widths[1]]}, B>0, match the store.
Each \code{layer<i>/W,b} uses independent Glorot weights and zero biases, and every
layer uses sigmoid. In R the code is at list index \code{length(widths)\%/\%2},
and reconstruction is the last entry. All gradients remain; no fitting is hidden.
\code{nn$SigmoidAutoencoderDecode(parameters,code,widths)} uses the same full
geometry and original decoder layer indices, taking only finite
\code{[B,widths[length(widths)\%/\%2+1]]} codes. Zero/signed codes are allowed.
A fresh store creates decoder leaves only. Output handles and list elements
retain lifetimes after their container is collected. Invalid geometry/input
fails before creation, while incompatible stored leaves can fail later.
The caller owns objectives, cached representations, parameter transfer, Adam,
batching and validation selection. These components retain the source's all-sigmoid
MSE architecture; greedy training is motivated by Bengio et al., NIPS2006,
\url{https://papers.nips.cc/paper_files/paper/2006/hash/5da713a690c067105aeb2fae32403405-Abstract.html}.
}

\section{Tied weights}{
\code{nn$TiedAutoencoder(parameters,image,features=784L,latent=32L,tied=TRUE)}
returns owning encoder_weight, decoder_weight, ReLU code and sigmoid reconstruction.
Finite matching Float32/64 image[B,features], B>0 and 0<latent<features are required.
The decoder is the live differentiable transpose of encoder/W; both gradient paths
accumulate into one leaf. Separate encoder/b and decoder/b give 25904 scalars.
FALSE creates independent decoder/W (50992 scalars); the caller matches initialization.
\code{nn$TiedAutoencoderDecode} consumes explicit code[B,latent], including zero or
signed values, creating only the selected matrix and decoder/b. No image, target,
encoder bias, optimizer or fitting policy is hidden. Invalid input or a tied store
containing decoder/W raises an error before leaf creation; incompatible shapes may
fail later. Output tensors retain lifetimes and gradients after parent collection.
}

\section{Known-factor lattice}{
\code{nn$KnownFactorLattice()} returns nine owning named tensors in fit,
validation, test order: images Float32[N,28,28,1], factors Int32[N,4] and
source_ids Int32[N]. Counts are210/42/42. Shape(circle/square), size(3),
x(7), y(7) give294 binary images; radius=size+2, centre=(3*x+5,3*y+5).
Inclusive pixel boundaries, ascending source IDs and residue
(x+2*y+size+3*shape) modulo7 define test0, validation1, fit2..6.
This is a deterministic procedural fixture with no external source or random
state, not dSprites or a downloaded benchmark. Native data preparation under
beta-vae-assets writes the same raw pixels as Int32 NPZ with provenance.
}

\section{Deterministic adversarial autoencoder}{
\code{nn$AdversarialAutoencoder(encoder,decoder,discriminator,image,prior,
features=784L,latent=4L,hidden=16L)} requires three distinct matching floating
stores, finite unit-range image[B,features] and finite prior[P,latent], B,P>0.
It returns owning code, logits, probability, prior_logit, encoded_logit, per-row
nll and scalar reconstruction_loss, discriminator_loss and adversarial_loss.
The deterministic encoder is tanh/linear; the decoder is tanh/linear with sigmoid
probabilities; the latent discriminator is tanh/linear. Default12 leaves/26133
scalars. Reconstruction sums pixel Bernoulli NLL then averages rows; D labels
prior=real and encoded=fake; E uses the non-saturating real-label objective.
All mathematical gradients remain, including encoder gradients of D loss.
The caller selects update groups, computes both reconstruction gradients before
either write, recomputes later phases and owns four independent Adam states.
There is no posterior sampling, per-example KL or ELBO. Prior changes cannot
change code or reconstruction. Invalid input/store identities fail before leaf
creation; incompatible leaves/nonfinite computed logits may fail later.
\code{nn$AaeEncode}, \code{nn$AaeDecode} and \code{nn$AaeDiscriminate}
expose the separate networks; decoding consumes only signed/zero finite codes
and returns logits. \code{nn$BernoulliPixelNll(logits,target)} sums stable
softplus(logits)-target*logits over each row, retaining both argument gradients.
Finite nonempty matching Float32/64 matrices and targets in[0,1] are required.
See \url{https://arxiv.org/abs/1511.05644}; this small teaching model does not
certify aggregate matching, useful generation or the published benchmark scores.
}

\section{Byte-quantized learned transform}{
\code{nn$FitCodecQuantizer(code)} fits only supplied finite Float32 CPU
\code{[N,D]} rows, N,D positive, to owning stopped minimum/maximum vectors.
\code{nn$QuantizeCodec(code,q)} uses explicit Double
floor(clamp((v-low)/(high-low)*255,0,255)+.5), returning row-major bytes;
constant ranges emit0. No refit or straight-through gradients.
\code{nn$DequantizeCodec(symbols,q)} returns detached Float32 rows using
Float32(low+symbol*(high-low)/255), computed in Double.
\code{nn$SymbolTensor(symbols,width=32L)} copies complete byte rows into Int32.
All ranges must be finite, ordered and match the positive row width.
\code{nn$LearnedCodecDecode(codes,W,b)} keeps all continuous Float32/64 paths
through sigmoid(codes W+b), with explicit matching finite CPU matrices and bias.
\code{nn$CodecPacket()} holds quantizer, Float32 W[32,784], b[784], nonnegative
Int32 source_ids[N] and bytes[N*32], N in1..10000. Nested quantizer getters copy:
assign the modified record back to the packet. Tensor fields retain handles.
\code{nn$PackCodec(packet)} writes exact TFLCODE1 little-endian bytes including
all decoder parameters, range values, identities and symbols:103772+36*N bytes.
\code{nn$UnpackCodec(bytes)} validates fixed architecture, bounded count,
exact length before allocation, finite scalars, ordered ranges and identities.
It rejects truncation/trailing bytes and returns independent detached storage.
\code{nn$DecodeCodec(packet)} uses only packet values to produce detached
Float32[N,784] probabilities; caller gradient flags stay unchanged. There is no
encoder, dataset, fitting policy, RNG, file IO, checksum or entropy coder hidden
in these operations. Invalid input raises an error. This is the original teaching
format, not a competitive codec or a guarantee of detecting arbitrary corruption.
}

\section{Reconstruction anomaly diagnostics}{
\code{nn$ReconstructionScores(original,reconstruction)} returns detached
Float64[N] per-case mean squared residuals. Finite signed Float32/64 CPU sample
axes flatten to matching N,D; trailing shapes/dtypes may differ. Values widen
before subtraction; squared differences sum in feature order, divided by D.
\code{nn$InkScores(images,fit_mean)} uses the same Double feature-order sum
for absolute mean-coordinate deviation from the explicit finite reference.
Neither is a training loss or fits a reference. Inputs and gradient flags stay unchanged.
\code{nn$NominalThreshold(scores,quantile=.95)} sorts a copy of nonempty finite
Float32/64 scores[N] and selects nearest rank ceil(quantile*N), using the Double
product and bounding the last rank. Probability is in(0,1]; there is no interpolation,
fuzz, dropping nonfinite scores, label use or hidden calibration split selection.
See R quantile type1, inverse empirical CDF, with no fuzz and excluding probability0.
\code{nn$RankAnomalies(normal_scores,anomaly_scores,threshold)} returns owning
auroc, average_precision, normal_fpr, anomaly_tpr. Nonempty finite signed Float32/64
CPU score vectors may differ in length/dtype. Canonical ROC/AP handle score ties:
half-credit AUROC and noninterpolated equal-score group AP. AP depends on supplied
prevalence. Both flag fractions use strictly score>threshold; threshold ties are
unflagged. No fitting, weighting, calibration or input gradient paths.
\code{nn$AnomalyCorruption(original,kind)} accepts finite signed Float32 CPU
[N,28,28,1], N positive, including strides. Kind0 sets the upper-left8x8 to1;
kind1 multiplies by Float32(.35); kind2 shifts6 columns right with zero fill.
The owning contiguous result is detached; original/flags stay unchanged.
These fixed paired synthetic controls do not define real anomalies. Invalid
geometry/dtype/values, nonfinite references/thresholds, bad probabilities and
computed score overflow raise errors.
}

\section{Conditional auxiliary-classifier GAN components}{
\code{nn$ConditionalGanGenerate(generator,noise,labels,features=784,classes=10,hidden=32)}
concatenates explicit noise[N,Z] and one-hot labels[N,C], then hidden tanh and
output sigmoid. Default Z16 creates four leaves/26736 values: hidden then output W,b.
\code{nn$AuxiliaryGanDiscriminate(discriminator,image,features=784,classes=10,hidden=32)}
sees only image[N,F], never labels/noise/targets. It creates hidden, CLASS, SOURCE
W,b in this initialization order (six leaves/25483 default values), returning
owning live source logits[N,1], class logits and softmax probabilities[N,C].
\code{nn$AuxiliaryClassLoss(logits,labels)} preserves mean(-sum(onehot times
log-softmax,axis1)), without clipping or smoothing. \code{nn$LogisticGanObjectives(real,fake)}
returns owning live scalar discriminator mean softplus(-real)+mean softplus(fake)
and non-saturating generator mean softplus(-fake); unequal positive N/M pools are
allowed. Raw logits preserve small positive-real derivatives at80 and strong
negative-fake generator gradients. No softplus(x)-x cancellation is substituted.
\code{nn$AuxiliaryClassifierGan(generator,discriminator,image,noise,labels,features=784,classes=10,hidden=32)}
returns fifteen owning live endpoints. D loss=source_D+(real_CE+fake_CE),
G loss=source_G+fake_CE. Labels condition G and supply CE targets, never enter D.
Select only D parameters, update D, recompute, then select only G parameters.
Separate optimizers and experiment loops belong to R; no fitting, checkpoint,
noise draw or MMD is hidden. All continuous first/higher paths remain live.
Finite signed Float32/64 CPU inputs match distinct store dtypes, with positive
axes/products and integer Int32/64 labels[N] in[0,C). Strides are allowed;
input validation precedes leaf creation. Incompatible leaves or nonfinite computed
states may fail later. Outputs retain lifetime after parent/store collection.
\code{nn$GanCoarsePixels(pixels)} flattens 28x28 grayscale samples, applies7x7
block averaging and divides by4, yielding live[N,16]. It accepts finite signed
floating CPU[N,...784] rows and strides. Use existing \code{nn$ImqMmd} separately;
this coarse game diagnostic is not an image-quality score. Invalid inputs,
aliased/mismatched stores, targets, dimensions or nonfinite computation raise errors.
}

\section{Bidirectional joint adversarial components}{
\code{nn$GaussianRingPoints(rows,seed=5507,components=8,radius=2,deviation=.08,dtype=Float32)}
returns owning detached CPU[rows,2]. Row i uses component i modulo components;
centres, canonical seeded Gaussian noise and addition are Double before one final
Float32/64 cast. Positive rows/components, representable rows*2 and finite
nonnegative radius/sd are required; zeros are allowed. No random component
assignment, partition, fit or global RNG state; equal arguments repeat bits.
\code{nn$BiganMap(parameters,input,output=2,hidden=16)} infers input width and
reuses canonical AaeDecode tanh/LINEAR layers, without sigmoid or clipping.
Hidden then output W,b order gives four leaves/82 default values. Inputs and
outputs are signed coordinates, not pixel probabilities or posterior samples.
\code{nn$BiganJointScore(discriminator,image,code,hidden=16)} concatenates explicit
image[N,F] then code[N,Z], predicts raw source logits[N,1], and retains both pair
parts' continuous paths. Default F=Z2,H16 creates four leaves/97 values. No
encoding, generation, target label or sigmoid is hidden.
\code{nn$BidirectionalGan(generator,encoder,discriminator,image,prior,hidden=16)}
returns owning fake[M,F], encoded[N,Z], real_score[N,1], fake_score[M,1], scalar
D/EG losses, reconstruction[N,F] and latent_cycle[M,Z]. Three distinct matching
Float32/64 stores create G,E,D hidden/output leaves in order,261 default values
in12 leaves. Finite signed CPU data[N,F]/prior[M,Z], positive representable axes,
strides and unequal N/M are accepted. D minimizes mean softplus(-real)+mean
softplus(fake); E/G minimize mean softplus(real)+mean softplus(-fake). Each pool
is averaged separately. Reconstruction/cycle paths are live diagnostics with no
term in the fitting losses. No learned-inverse, likelihood or convergence guarantee.
Caller selects D leaves, writes D, recomputes E, writes E, then recomputes G.
With D fixed, E's write leaves G's EG partial unchanged. No hidden optimizer,
fitting loop, noise draw, freeze, checkpoint or evaluation guard. All continuous
first/higher paths stay live. Inputs validate before leaf creation; incompatible
leaves or computed nonfinite states can fail later. Invalid options, pairs,
aliased/mismatched stores or nonfinite computations raise errors. Returned Tensor
fields survive parent/store collection; synchronize shared mutable stores.
}

\section{Compact dense and conditional GAN objectives}{
\code{nn$DenseGanObjective} contains Logistic,LeastSquares,Hinge. The required
objective argument of \code{nn$DenseGan(G,D,image,noise,objective,features=784,hidden=32)}
changes only the losses. All three use the same four-leaf G and four-leaf D,
26416+25153 default values. \code{nn$MlpGanGenerate(G,noise,features=784,hidden=32)}
reuses canonical tanh/linear math followed by sigmoid; \code{nn$MlpGanDiscriminate(D,image,features=784,hidden=32)}
returns raw unbounded scores without sigmoid. Five owning live output fields are
fake[M,F], real_score[N,1], fake_score[M,1], discriminator_loss,generator_loss.
Finite signed Float32/64 CPU matrices, matching distinct store dtypes, positive
representable geometry, unequal N/M and strides are accepted. No class labels.
\code{nn$LeastSquaresGanObjectives(real,fake)} preserves source grouping
D=.5*(mean((real-1)^2)+mean(fake^2)), G=.5*mean((fake-1)^2): a0,b=c1 coding,
not the a-1,b1,c0 Pearson-divergence derivation. \code{nn$HingeGanObjectives(real,fake)}
uses D=mean(relu(1-real))+mean(relu(1+fake)), G=-mean(fake). At exact real1/fake-1
margins D has zero ReLU subgradient; G always retains -1/M per fake score.
Negative G loss is valid. No spectral normalization, score clipping, calibration
or sigmoid is hidden. First/higher paths live away from hinges; a classical
second derivative at the kink is undefined. Both return owning live scalar fields.
\code{nn$ConditionalGanDiscriminate(D,image,labels,features=784,classes=10,hidden=32)}
concatenates images with one-hot integer classes before the tanh/linear score.
Both networks consume conditions in \code{nn$ConditionalGan(G,D,image,noise,labels,features=784,classes=10,hidden=32)};
it reuses existing ConditionalGanGenerate and stable logistic objectives. Default
G26736+D25473 values in eight leaves; hidden then output W,b order in each store.
No auxiliary class head or categorical CE. A single Int32/64[N] vector in[0,C)
conditions both pools, requiring matching data/noise row counts. Seven owning
fields: fake[N,F], real/fake logits and source probabilities[N,1], D/G scalar losses.
All continuous applicable higher paths stay live, labels are discrete. Caller
selects only D leaves, writes D, recomputes through updated D and selects G leaves.
No optimizer, fitting loop, input RNG, freeze, checkpoint or quadratic MMD is hidden;
use explicit GanCoarsePixels/ImqMmd for the separate coarse game diagnostic.
Input/options checks precede leaves; incompatible leaves or nonfinite computed
outputs may fail later. Inputs unchanged, fields survive parent/store collection;
synchronize shared stores. Invalid inputs/labels/dtypes/products/objectives,
aliased stores or nonfinite losses (including square overflow) raise errors.
}

\section{Generated codes and separate InfoGAN recognition}{
\code{nn$InfoPrior(rows,seed,classes=10,continuous=2,dtype=nn$DType$Float32)}
returns owning category Int32, continuous Float32/64 and raw Float64 Tensor fields.
A local mt19937 draws an accepted unbiased categorical word followed by U midpoint
uniform words per row. Double continuous arithmetic precedes one final cast;
Float32 rounding can reach -1 or1. C lies in 1..INT32_MAX, axes/products positive.
No global RNG or dataset is consulted; category IDs are generated-prior codes.

\code{nn$InfoGanGenerate(G,z,category,u)} concatenates noise, one-hot category
and continuous prior, reusing the native tanh/sigmoid map. The image-only D uses
\code{nn$MlpGanDiscriminate}. Independent image-only Q is
\code{nn$InfoGanRecognize(Q,image,features=784,classes=10,continuous=2,hidden=32)}.
It creates hidden, CATEGORY, CONTINUOUS W,b in order, with25516 default values/six
leaves. Returns owning logits, probability and unbounded mean fields; no code or
noise enters Q and variance is fixed at1. G has26800 values/four leaves, D25153/four.

\code{nn$InfoGanRecognitionLoss(prediction,category,u)} uses logits-based native
CE plus .5*mean(sum((u-mean)^2))+.5*U*log(2*pi). The unused probability convenience
field does not enter the loss. Returns owning categorical, continuous, total and
lower_bound scalars. Bound=log(C)+U*log(2)-total assumes independent uniform priors;
it may be negative and is not measured true mutual information or disentanglement.

\code{nn$InfoGan(G,D,Q,image,z,category,u,features=784,classes=10,hidden=32)}
returns fifteen live owning fields. D minimizes logistic source loss; Q minimizes
recognition R; G minimizes non-saturating source+R with weight1, without subtracting
prior entropy. The host recomputes D, then Q, then G with independent optimizer
states and selected-store writes. No MMD, fit, freeze, checkpoint or input RNG is
hidden. Distinct matching Float32/64 CPU stores, finite signed image/noise, positive
possibly unequal real/fake rows, integer categories in[0,C), continuous in[-1,1],
strides and positive representable geometry are required. Continuous code is also
recognition target; all applicable first/higher paths live. Inputs validate before
leaves; incompatible leaves/nonfinite computation can fail later. Fields outlive
parent/store collection; synchronize shared stores. See native info_gan.hpp.

\code{nn$cached_spectral_power_iteration(W,u0,iterations=1,epsilon)} returns
owning detached columns u[out,1],v[flattened input,1]. Finite matching Float32/64
CPU kernel of rank at least two, positive axes and u0[out,1] are required.
Each ordered iteration computes v=unit(W*u), then u=unit(t(W)*v), with
unit(x)=x/sqrt(sum(x*x)+epsilon), without initial u normalization. Default guard
is Float32-rounded 1e-12 even for Float64. Zero kernels produce finite zeros.
No graph, mutation, initialization or RNG is hidden.

\code{nn$cached_spectral_kernel(W,u,v,floor)} returns owning live weight and
sigma fields: sigma=sum(detach(v)*(flatten(W)*detach(u))), weight=W/max(sigma,floor).
The raw sigma is signed, without abs. Default floor is Float32-rounded 1e-6.
First/higher W paths including sigma stay live; u/v paths always stop. No power
iteration or state update occurs. Finite iteration is not an exact unit norm
certificate. Positive guards must remain finite/positive in the tensor dtype.

\code{nn$advance_cached_spectral_state(D,state,iterations=1,epsilon)} updates
existing prefix/u,v for each prefix/W, ignoring biases. All candidates compute
before any writes; invalid/missing state leaves both stores unchanged. Distinct
matching floating stores required; no kernels is a no-op. No weight write,
initialization, optimizer or RNG. Existing cache handles see copied values and
keep their flags; exclude state from optimizers. Synchronize shared stores.

\code{nn$SpectralGanDiscriminate(D,state,image,features=784,hidden=32)} maps
finite signed matching floating CPU[N,F] through two cached spectral kernels,
hidden tanh then a raw source score. Missing hidden/output W,b and state u,v are
initialized in source order with Float32-rounded uniform limits and constants.
Default D25153/four leaves, cache849/four leaves. No existing state advance.

\code{nn$SpectralGan(G,D,state,image,noise,features=784,hidden=32)} returns five
owning DenseGanOutput fields, reusing native tanh/sigmoid G (26416 values) and
hinge losses: D=mean(relu(1-real))+mean(relu(1+fake)), G=-mean(fake). D margin
subgradient is zero; G is unhinged. Three distinct matching Float32/64 CPU stores,
finite signed image[N,F]/noise[M,Z], positive possibly unequal rows and
representable dimensions required. Strides and applicable higher continuous
paths live; cache paths stop but sigma's W path lives. Caller initializes,
warms cache16 times, then per batch advances cache, writes D, recomputes through
updated D with SAME cache, and writes only G. No fitting, optimizer, MMD or noise
draw beyond missing leaf initialization is hidden. Input checks precede leaves;
incompatible existing leaves/nonfinite math may fail later. Fields outlive parent
collection; errors raise R errors. See spectral_cache.hpp and spectral_gan.hpp.

\code{nn$cached_spectral_dense(parameters,state,name,x,units,activation)} is the
shared native dense map with stopped cache columns and live signed sigma. Finite
matching Float32/64 CPU x[N,in], distinct matching stores, nonempty prefix and
positive representable geometry required. Creates name/W,b and state/u,v only
when missing, with source Float32-rounded uniform limits and column constants.
Default activation is raw affine. No existing cache advance or optimizer is hidden.

\code{nn$SpectralSpatialAttention(parameters,state,input,query_width=2,value_width=4,
name="attention")} returns owning input/output[B,T,C] and probability[B,T,T].
All tokens interact through softmax(Q*K-transpose), without sqrt scaling, masks
or dropout. Native Plain attention uses scale1. Four biased spectral maps are
created query/key/value/project, then scalar gamma initialized0. Output is
input+gamma*project(probability*V). At initialization, output values equal input;
output-loss projection partials vanish but gamma can receive a gradient. Finite
matching Float32/64 CPU tokens, positive axes/widths/products, distinct stores and
nonempty prefix required. Strides and applicable first/higher paths live; columns
stop and sigma's W path stays live. No cache advance is hidden.

\code{nn$SelfAttentionGanGenerate(G,Gstate,noise)} maps noise[B,16] through
spectral input dense128/tanh, sixteen8-feature tokens, unscaled attention Q2/V4,
shared8-to49 sigmoid patches and native MaeImage raster permutation. It owns
pixels[B,784], attention_input/output[B,16,8] and attention_probability[B,16,16].
G2730/13 leaves and G cache245/12 leaves; source creation order preserved.

\code{nn$SelfAttentionGanDiscriminate(D,Dstate,image)} maps image[B,784] through
native MaePatches, shared49-to8/tanh, learned normal(.01) position[16,8], unscaled
attention, token mean and spectral8-to1 raw score. Owns score[B,1] and all three
attention fields. D650/14 leaves and D cache110/12 leaves; no sigmoid or class head.

\code{nn$SelfAttentionGan(G,D,Gstate,Dstate,image,noise)} returns twelve owning
live fields with hinge D margins and unhinged G=-mean(fake_score). Four distinct
matching Float32/64 CPU stores, finite signed image[N,784]/noise[M,16], positive
possibly unequal rows and representable geometry required. Total51 checkpoint
tensors/3735 values,3380 optimized values. Caller initializes, warms both cache
stores16 iterations, then per batch advances D cache, writes D, advances G cache,
recomputes fresh G through updated D with SAME D cache and writes only G.
No fitting, optimizer, cache advance, freeze or MMD is hidden. Exclude both cache
stores from Adam. Input checks precede leaves; incompatible existing leaves or
nonfinite computation can fail later. Fields retain lifetime/continuous graphs
after parent/store collection. Synchronize shared stores. Errors raise R errors;
finite iteration is not an exact unit norm certificate. See self_attention_gan.hpp.


\code{nn$capture_batch_norm(stats,name,input,batch_statistics=FALSE,epsilon)}
normalizes finite matching Float32/64 CPU input[...,C], rank at least2, using
saved or batch moments. Creates only missing moving_mean/var; never writes an
existing statistic. Batch mean and source-rounded unbiased variance are live
capture fields; both are undefined in saved-statistics mode. Uses division/sqrt
and Float32-rounded1e-5 default epsilon, without affine/fused reassociation.
Positive axes, nonempty prefix and training count>1 required. Strides/higher paths
live; saved moments stop. \code{nn$batch_norm_moving_values(old_mean,old_variance,
captured_mean,captured_unbiased_variance,momentum)} computes detached owning
mean/variance candidates m*old+(1-m)*capture, without any store write. Default m
is Float32-rounded.9; finite matching vectors, nonnegative variances and m in[0,1].

\code{nn$GaussianRejectionPrior(rows,features,seed,limit,dtype)} returns owning
detached z and Float64 proposals. Local MT midpoint uniforms, Box-Muller cosine
then cached sine, Double absolute-threshold rejection before final dtype cast.
No clipping or global RNG. Positive counts, uint32 seed, positive non-NaN limit
(infinity default), Float32(default)/Float64 required. Strict proposal budget
K<1000*rows*features; exhaustion raises an R error. Audit includes all rejects
and final acceptance, without an unused trailing sine.

\code{nn$BigGanResize8(images)} reuses native half-pixel bilinear28-to8 in Double,
then casts once to input Float32/64. Finite signed CPU[N,28,28,1], N>0; strides
and applicable input paths retained, no clipping/rescaling or data loading.

\code{nn$BigGanGenerate(G,GSN,moving,z,labels,batch_statistics=FALSE)} uses
three distinct matching Float32/64 CPU stores, finite signed z[N,12], Int32/64
classes[N] in[0,10). Three latent chunks drive a spectral stem and two conditional
BNs sharing a raw class embedding. Residual nearest-up block, spectral normal(.02)
kernels, output affine and sigmoid produce image[N,8,8,1]. Owns image and nested
six-vector moments; capture fields are undefined for inference-only calls.
G789/17 leaves, G-SN241/18, moving24/6. No advance, EMA or optimizer hidden.

\code{nn$BigGanDiscriminate(D,DSN,image,labels)} owns raw score/source/projection
[N,1] and features[N,4]. Distinct matching stores, finite signed image[N,8,8,1]
and valid classes required. Residual down block and sum-pooling feed a spectral
linear score plus class-feature projection. D241/9 leaves, D-SN77/10. No sigmoid,
class posterior, G call, BN, advance or fitting.

\code{nn$ModifiedOrthogonalPenalty(G,coefficient)} sums squared OFF-diagonal raw
/W kernel Gram entries, excluding diagonal; default coefficient Float32-rounded
.5e-4. Other leaves, including raw class embedding/affine/bias, contribute0*sum
with defined zero partials. Creation-order grouping, live higher kernel paths;
finite nonempty floating store and nonnegative coefficient, representable products.

\code{nn$BigGanCommitMoments(moving,captured)} validates ALL six candidate vectors
before any writes, with Float32-rounded.9 retention. Uses the SAME pre-Adam G
forward captures after its optimizer write; no recomputation at updated weights.
No initialization, spectral update or optimizer. Missing/invalid state fails
without writes. Exclude all state stores from Adam and synchronize mutations.

\code{nn$BigGan(G,D,GSN,DSN,moving,image,z,labels,display_batch_statistics=FALSE)}
returns both G paths, raw scores, explicit losses and nested moments. Five distinct
matching stores, finite signed matching-row inputs, valid labels required. Total60
tensors:1030 trainable+342 state values. Display flag selects fake/fake_score ONLY;
both objectives ALWAYS use batch-statistics fake, even with inference display.
D hinge; G=-mean(training score)+modified orthogonal. Caller warms each SN16 times,
then D-SN advance/D write/G-SN advance/fresh G gradient/G write/captured BN commit.
No fitting, optimizer, EMA, advance or checkpoint hidden. Continuous first/higher
paths live; classes discrete, columns/saved moments stopped. Fields and nested
records outlive parent/store collection. Input checks precede leaves; stored-leaf
failures may occur later. Errors raise R errors. See big_gan.hpp and captured_batch_norm.hpp.

}

\section{Compact StyleGAN and native normalization}{
\code{nn$spatial_instance_normalize(input,epsilon)} accepts finite Float32/64 CPU
NHWC positive axes, centers each row/channel over H,W and divides by population
standard deviation with a positive representable guard (default double(float32(1e-8))).
No affine parameters or state; constant channels zero; first/higher paths live.
\code{nn$modulated_conv2d(image,weight,style,demodulate=TRUE,epsilon)} takes
NHWC[N,H,W,C], odd square[K,K,C,O] and signed/zero gains[N,C]. Returns owning live
image[N,H,W,O] and demodulator[N,O]. Source row-major tap accumulation; factors
rsqrt(sum((weight*style)^2)+guard) over K,K,C. With demodulate=FALSE factors are
ones with the source zero path; no state or mutation. All inputs matching CPU
Float32/64; products, guards and arithmetic must be finite/representable.
\code{nn$StyleGanMap(g,z)} maps finite matching z[N,8] through pixel RMS and
2 dense8/leaky.2 layers (144 values/4 leaves). Call before synthesis for source
seed6607 initialization order. \code{nn$StyleGanSynthesize(g,w0,w1,n4,n8,version2=FALSE)}
uses aligned W[N,8] and explicit noise[N,4,4,1]/[N,8,8,1]. Learned constant,
styled3x3 layer4,nearest up2,styled3x3 layer8,RGB sigmoid. AdaIN conv/noise/bias/leaky
then population normalization and learned scale/shift; version2 modulated/demodulated
feature weights without AdaIN, RGB modulation without demodulation using w1.
Optional first_style_override[N,4] replaces actual first gains for version2 only,
still initializing affine leaves. G totals661/17 or625/19 values/leaves with mapping.
Nine owning flat live synthesis fields expose image/constant/first/upsampled/second,
per-layer styles and demodulators; no hidden noise RNG, fitting or updates.
\code{nn$StyleGanDiscriminate(d,image)} requires matching finite signed[N,8,8,1],
493 values/6 leaves, source seed6619, returns owning raw[N,1] scores.
\code{nn$StyleGan(g,d,real,z,n4,n8,version2=FALSE)} requires distinct matching stores,
positive real/latent rows (unequal allowed), returns ten owning flat live fields:
fake,mapped,first_style,second_style,first_demodulator,second_demodulator,
real_score,fake_score,d_loss,g_loss. D half the sum of squared real-to1/fake-to0 means;
G full squared fake-to1 mean. Capture/step D then fresh G with same latent/noise
batch; caller owns Adam and every write. All applicable first/higher paths live;
outputs retain graph/storage after parents are collected. No hidden EMA/truncation,
batch state, labels, RNG or fitting. Invalid calls raise R errors. Compact teaching
variants: Karras et al., \url{https://arxiv.org/abs/1812.04948},
\url{https://arxiv.org/abs/1912.04958}; canonical contracts in style_gan.hpp.
}

\section{Three-domain StarGAN and explicit categorical targets}{
\code{nn$UniformCategorical(rows,seed,classes=3)} uses local raw-MT rejection:
accept word below floor(2^32/classes)*classes, return word modulo classes.
Positive rows/cardinality1..INT32_MAX; default3 rejects4294967295. Owns detached
Int32 labels[N] and exact Float64 accepted raw[N] words, no retained engine.
\code{nn$StarDomainImages(original,domain)} accepts finite signed Float32/64 CPU
[N,H,H,1] with positive square axes, aligned Int32/64 domains[N] in0..2.
0 unchanged,1 inversion,2 rotation out(y,x)=original(H-1-x,y), no inversion on2.
Owns same-shape/dtype live output, applicable higher paths, no clipping or fitting.
\code{nn$SplitStarDomains(images,labels,source_ids,per_class=30)} requires finite
unit matching images[N,28,28,1], positive N divisible10, balanced integer classes
0..9, nonnegative unique integer IDs. Positive per_class<=N/30, default30 selects
900/1000 fitting rows; use6 for180/200 held-out. First per_class occurrences of
 each digit enter domain0, next1, next2; rest retained as unused IDs. Preserves
global order and ID dtype, returns owning image,original,domain,digit,source,unused;
domain/digit Int32, image gathers live. No pairing, RNG, fitting or cross-pool
provenance checks; LoadSaliency supplies checked original data provenance.
\code{nn$StarGanGenerate(g,source,target)} requires finite signed matching
Float32/64 CPU source[N,28,28,1], aligned Int32/64 target[N] in0..2 before one-hot.
Tiled one-hot3 concatenates with source, two stride2 conv/ReLU maps4->4->8,
learned transpose8->4,concat encoder,transpose8->2,concat full condition,3x3
sigmoid6->1. Normal.02 kernels/zero biases, seed6301,1609 values/10 leaves.
Nine owning live fields expose code/input/both actual skips/image; no original
domain, paired target image, noise, normalization, fitting or hidden writes.
\code{nn$StarGanDiscriminate(d,image)} reads image only with same floating image
contract. Leaky.2 stride2 conv1->4->8, pooled8->3 Glorot domain head BEFORE3x3
normal source head8->1, seed6311,688 values/8 leaves. Five owning live fields
source[N,7,7,1],domain_logits/probability[N,3],down14/down7. Source raw patches
are not probabilities; softmax is on the domain head only. Higher paths live.
\code{nn$StarDomainLoss(logits,target)} computes stable mean three-way CE with
source one-hot/log_softmax grouping. Finite Float32/64 CPU[N,3], aligned integer
[N] domains0..2, positive rows; owning live scalar and applicable higher paths.
\code{nn$StarGan(g,d,source,original,target)} requires distinct matching stores,
the above aligned image/domain contracts, permits target=original. Fake is
G(source,target), reconstruction G(fake,original) with SAME G. G=(full fake-to1
source MSE+fake-domain CE)+10*roundtrip meanL1. D=half(real-to1+fake-to0 source
means)+real-domain CE; no fake CE term. Fifteen owning flat live fields retain
storage/higher paths through both G calls and current D. Caller captures/steps
D, then makes a fresh G forward with same source/original/target batch and steps
only G. No hidden detach, capture, optimizer, fitting, no-grad, noise or batch
state. Existing leaves unchanged, only missing leaves initialize; all supplied
inputs/aliases validate before leaves. Invalid calls raise R errors. This is a
compact teaching variant, not the full paper architecture: Choi et al.(2018),
\url{https://arxiv.org/abs/1711.09020}; canonical contracts in star_gan.hpp.
}

\section{Native Gaussian feature distance and probability-space Inception scores}{
\code{nn$GaussianFrechet(real,fake)} takes finite floating CPU[N,F]/[M,F],
N,M at least2,F positive; unequal row counts/dtypes and strides allowed.
Stop both inputs and cast Float64, center each pool and divide sqrt(its rows-1).
Thin SVD of the[N,M] cross product supplies the covariance cross trace without
forming[F,F] covariance matrices. Seven owning detached fields: fid,raw_fid,
mean_term,trace_a,trace_b,cross_trace,cross_singular_values[min(N,M)]. Raw distance
is squared mean difference+both covariance traces-2*cross trace. Raw below-1e-7
rejects; tiny negative round-off clamps zero. This is a fixed absolute tolerance.
No extractor, fitting, sampling, mutation or gradient path. Calling it FID
requires the actual specified pretrained Inception protocol; arbitrary features
still define a Gaussian moment distance, without testing full-distribution equality.
\code{nn$InceptionScore(probability)} takes finite floating CPU[N,C],N positive,
C at least2, probabilities in[0,1],row sums within3e-6 of1. No renormalization.
Double stopped inputs; marginal mean over rows, log arguments floor1e-300,
exp(mean row KL) gives an owning detached Float64 scalar. Balanced confident C
classes score C; uncertain uniform and class-collapsed pools both score1. No
accuracy, uniqueness, recall or protection against copying is implied.
\code{nn$InceptionScoreSplits(probability,splits=4)} uses contiguous boundaries
floor(i*N/splits),positive splits no greater than N. Every row occurs once;
no shuffle/drop/RNG. Three owning Float64 fields: scores[splits],mean,
population_std with divisor splits, not splits-1. This order-dependent spread
from one pool is not an independent-model confidence interval. No fitting,
extractor, mutation or gradient path; errors become R errors. Original sources:
Heusel et al.(2017),\url{https://arxiv.org/abs/1706.08500}; Salimans et al.(2016),
\url{https://arxiv.org/abs/1606.03498}. Canonical contract: generative_metrics.hpp.
}
\section{Native immutable frozen GraphDef execution}{
\code{nn$FrozenGraph(as.integer(bytes),max_bytes)} parses complete model bytes;
\code{nn$FrozenGraph_load(path,max_bytes)} loads a local regular file. The default
512MiB positive limit bounds serialized bytes and total decoded numeric constants.
No TensorFlow/protobuf runtime, training, downloading, variables or global RNG.
Float32/64,Int32/64,Bool constant bits and scalar rank are exact. Malformed wire,
names/shapes/payloads, duplicate/missing nodes and nonzero serialized input ports
reject. Unsupported constants/operations may remain outside fetched ancestry.
\code{graph$node_names()} lists serialized-order names;
\code{graph$operation(name)} reads its declared kind;
\code{graph$constant(name)} returns an independent detached mutable Tensor copy.
\code{graph$evaluate(nn$named_tensors(feeds),fetches)} executes only fetched ancestry;
convert returned pairs with \code{nn$named_list}. CPU feeds are named Tensor lists,
declared dtypes checked, output names optionally suffixed:0. Feeds cut all input
and control ancestry of the overridden node. Results are owning detached copies
in requested order, including duplicates; no model or input writes occur.
First/higher input gradients always stop. Independent concurrent calls share
immutable parsed constants only; callers synchronize their feed mutation.
Supported: Const,Placeholder,Identity,StopGradient,CheckNumerics,Relu,Add/AddV2,
Sub,Mul,float Div/RealDiv,BiasAdd,MatMul,Reshape,Squeeze,ExpandDims,Concat/
ConcatV2,Softmax,Float32 ResizeBilinear,NHWC Float32/64 Conv2D/AvgPool/MaxPool,
legacy BatchNormWithGlobalNormalization. SAME/VALID,spatial strides/dilation,
matrix transpose,resize coordinate flags and old BN epsilon/gamma are preserved.
Control dependencies execute too; CheckNumerics rejects nonfinite input,
cycles/depth above1024 reject, intermediates release after last consumer.
No JPEG/string,variable,function,random or multi-output execution. Errors become
R errors. Supplied files do not imply pretrained identity or preprocessing.
Canonical contracts: frozen_graph.hpp and TensorFlow's published GraphDef schemas.
\code{nn$prepare_example_data("inception-metric",directory)} obtains the official
2015 ImageNet archive, unmodified GraphDef, license and manifest; default cache
is inception_metric below the shared data root. Archive and graph SHA256/size
are pinned, completed graph caches are rehashed, no TensorFlow runtime or
substitute features. Feed ExpandDims:0 Float32 RGB in0..255; caller chooses
classifier bias policy for the original2048 features and1008 classes.
}
\section{Explicit single-image residual patch game}{
\code{nn$SingleImageNoise(rows,height,width,shared_rgb,seed,dtype=nn$DType$Float32)}
draws owned detached CPU[N,H,W,3] from one local midpoint-MT Box-Muller stream.
Positive representable axes, Float32 or Float64. Shared mode draws one channel,
casts once then tiles RGB; independent mode draws all three in NHWC order.
No global RNG, retained graph, fitting or state writes.
\code{nn$SingleImageAmplitude(real,previous_reconstruction,factor=.1)} takes
finite signed matching floating CPU RGB rectangles with positive axes and arbitrary
strides. Sum squared Double differences, divide by numel, sqrt and multiply factor,
then cast once to input dtype; owning detached[1] result stops both image paths.
Factor finite nonnegative; caller separately sets coarse amplitude one.
\code{nn$SingleImageGenerate(g,previous,noise,amplitude)} takes matching finite
signed Float32/64 CPU[N,H,W,3], positive rectangular axes, matching finite amplitude
scalar or[1], signed/zero allowed. Biased source fan-in-uniform SAME3x3
3->4->4->3, two leaky(.2),tanh, image=previous+residual without clipping.
Seed7307 gives371 values in6 head/body/output W/b leaves. Five owning live
input/head/body/residual/image fields retain applicable first/higher paths after
parent/store collection. No BN,pooling,dropout,target image,sampling,optimizer,
hidden no-grad or existing-leaf writes. Only missing leaves initialize.
\code{nn$SingleImageDiscriminate(d,image)} takes matching signed finite RGB
rectangles, biased fan-in-uniform SAME3x3 3->4->4->1,two leaky(.2),raw
patch scores[N,H,W,1],nominal7x7 receptive field. Seed7309 gives297 values/6
leaves; no pooling,sigmoid,normalization,noise,amplitude or hidden writes.
\code{nn$SingleImageGan(g,d,real,previous,noise,rec_previous,rec_noise,amplitude)}
requires distinct matching stores, common spatial axes, matching previous/noise,
and rec_previous/rec_noise exactly real-shaped. Random/real rows may differ;
each score pool uses its own mean. Shared371-value G generates random fake and
fixed reconstruction; image-only297-value D. D is half sum of real-to-one/fake-to-zero
least-squares means; G is FULL fake-to-one mean plus10*RGB reconstruction MSE.
Eight owning live flat fields: fake,reconstruction,real_score,fake_score,d_loss,
g_loss,adversarial_loss,reconstruction_loss. No gradient penalty, hidden detach,
fitting, sampling, clipping or optimizer. Caller freezes lower scales, measures
fixed amplitudes, warm-starts BOTH groups with cold Adam, then D and fresh G
captures/writes. Supplied inputs/aliases validate before leaves; errors become R
errors, shared stores require synchronization. Rott Shaham, Dekel and Michaeli
(2019), SinGAN: Learning a Generative Model from a Single Natural Image,
\url{https://arxiv.org/abs/1905.01164}; compact source variant uses LSGAN.
}
\section{Explicit SRGAN/ESRGAN and relativistic-average objectives}{
\code{nn$RelativisticAverageGan(real,fake)} accepts finite matching Float32/64
CPU[N,1]/[M,1], positive unequal rows allowed. rr=real-mean(fake),rf=fake-mean(real),
both pool means LIVE. D=.5*(mean softplus(-rr)+mean softplus(rf)),G with signs
reversed, source grouping. Four owning live fields retain higher cross-pool paths,
no clipping, fitting or hidden detach. Common score shifts cancel up to rounding.
\code{nn$AdversarialSRGenerate(g,stats,lr,bicubic,enhanced=FALSE,batch_statistics=FALSE)}
requires distinct matching Float32/64 stores; signed finite CPU LR[N,16,16,3] and
bicubic[N,32,32,3], positive aligned rows. Source seeds7001/0. SRGAN G1591/25
values/leaves,BN40/10, two BN/PReLU residual blocks and CRD shuffle2. ESRGAN
G3511/40,three growth2 five-conv dense blocks,.2 residual scaling,nearest2,up/HR
leaky convs,no BN. Both output kernels normal.001; add unbounded residual to
bicubic, other kernels source float fan-in uniform. Nine owning live fields expose
image,residual,packed,shuffled,head,body,upsampled and optional detached BN
moments_mean/moments_unbiased_variance[5,4], order residual0/bn1,bn2,
residual1/bn1,bn2,body_bn. Captures defined only for batch-statistic SRGAN; otherwise
undefined. No hidden EMA/state update, fitting, optimizer, no-grad or later RNG.
\code{nn$AdversarialSRDiscriminate(d,image,relative=FALSE)} requires matching
finite signed CPU[N,32,32,3]. Fan-in-uniform head3->4,stride2 4->8->8,leaky.2,
global mean,raw score1. Ordinary Glorot dense+bias1673/8,relative bias-free
sqrt(6.f/9) score kernel1672/7,source seed7013. No sigmoid, BN or hidden writes.
\code{nn$AdversarialSRFeatures(rgb,weight,bias)} requires matching finite floating
CPU RGB[N,H,W,3],positive axes,weight[3,3,3,64],bias[64]. RGB*255 reversed to BGR
minus source float-rounded ImageNet means,SAME conv+bias,no ReLU or /255 here.
Both frozen inputs always detach; applicable RGB higher paths live. Owns
[N,H,W,64]; caller certifies actual pretrained values, no random substitution.
\code{nn$AdversarialSR(g,d,stats,lr,target,bicubic,feature_weight,feature_bias,enhanced=FALSE,adversarial=FALSE)}
requires three distinct matching stores and aligned finite signed LR/HR geometry;
all supplied inputs/aliases validate before leaves. Always evaluate saved-statistic
and batch-statistic G, losses use training G. SRGAN pixelMSE/post-ReLU featureMSE/
ordinary logistic; ESRGAN pixelL1/pre-activation featureL1/half-scaled relativistic.
Both feature tensors divide by255 BEFORE loss, target extracted features stop.
GAN objective is content+source-float-rounded .001/.005*adversarial, plus ESRGAN
float-rounded .01*pixel. G=(1-phase)*pixel+phase*GAN, computing every component.
Twenty owning flat fields retain applicable higher paths/storage; detached optional
BN captures as above. Caller owns warm-up,cold reset of BOTH Adam states,D then
fresh G captures/writes; AFTER each G write commit moving candidates from the
SAME pre-write G capture,momentum double(float32(.9)). BN/frozen values excluded
from optimizers; no generator weight EMA or hidden fitting/state write/no-grad.
Invalid calls raise R errors. Compact teaching variants: Ledig et al.(2017),
\url{https://arxiv.org/abs/1609.04802};Wang et al.(2018),
\url{https://arxiv.org/abs/1809.00219}. Canonical contracts in
adversarial_super_resolution.hpp and relativistic_gan.hpp.
}

\section{Native binary pixels and masked PixelCNN}{
\code{nn$BinaryPixels(images)} accepts normalized finite Float32 CPU
\code{[N,28,28,1]}, positive N, arbitrary strides. Detached Double half-pixel
bilinear resize, one Float32 cast, then gray >= .5. Two owning detached
\code{gray/images[N,8,8,1]} fields; no fitting or random binarization.
\code{nn$CausalPixelMask(Cin,Cout,type_b,dtype=nn$DType$Float32)} returns
stopped exact zero/one \code{[3,3,Cin,Cout]}. A allows top row and middle-left;
B also center. Positive representable widths; no RGB channel ordering.
\code{nn$MaskedPixelConv(parameters,name,input,mask)} applies biased odd rectangular SAME
stride1 convolution with live kernel times stopped structural mask. Matching
finite floating CPU input \code{[N,H,W,Cin]}, mask \code{[Kh,Kw,Cin,Cout]}
exact zero/one, positive odd Kh/Kw. Missing name/W normal(.1), name/b zeros. Input/kernel/bias
first/higher paths live; masked partials zero. No activation or fitting.
\code{nn$PixelCnn(parameters,input)} uses source store seed8401: four biased
SAME3x3 width8 ReLU layers, A then three B, unmasked1x1 Bernoulli head.
Finite matching Float32/64 CPU \code{[N,H,W,1]}, positive rectangles and strides
allowed;1841 values in10 leaves,808 masked coordinates. Six owning live fields
\code{input_features/head/hidden0/hidden1/hidden/logits}. Strict raster causality
and ordinary masked-CNN blind spot, no batch coupling. No hidden optimizer,
fitting/sampling, existing-leaf writes or no-grad; first/higher paths remain live.
\code{nn$BernoulliPixels(logits,binary)} requires matching finite floating CPU
\code{[N,H,W,1]}, exact zero/one targets. Four owning live fields:
\code{probability} sigmoid, \code{pixel_nll} softplus(z)-x*z,
\code{per_image_nll} HWC sum in nats, \code{loss} mean in nats per binary pixel.
No clipping, bit conversion, fitting or mutation; both applicable paths stay live.
\code{nn$BinaryPixelUniforms(rows,seed)} draws a full owning stopped
Float64 \code{[N,64]} raw-MT19937 midpoint bank in row-then-ordinal order,
(raw+.5)/2^32. Positive representable N, uint32 seed; draw all64 sites even
when preserving a prefix. Source8423 for samples,8443 for completions.
All inputs validate before model leaves; invalid arguments raise errors.
Tensor fields survive parent/input/store collection. Independent calls share no
mutable state; synchronize shared parameter stores. No global RNG is used.
}

\section{Native dual-stream gated raster context}{
\code{nn$GatedPixelMask(Cin,vertical,first,Cout=8L,dtype=nn$DType$Float32)}
returns owning stopped exact zero/one Float32/64 masks. Positive representable
widths. Vertical shape[3,3,Cin,Cout]: first block top row, later top+current row.
Horizontal shape[1,3,Cin,Cout]: first left, later left+center. First block excludes
all current raw channels. No RGB channel ordering, RNG or store writes.
\code{nn$GatedPixelGate(preactivation)} accepts finite floating CPU[N,H,W,8],
positive axes, arbitrary strides. Owning live[N,H,W,4] tanh(first4)*sigmoid(last4),
first/higher paths to both halves; the gate alone does not enforce causality.
\code{nn$GatedPixelConvolutional(parameters,image,labels,conditional=FALSE,
input_channels=1L,output_channels=1L)} computes three width4 dual-stream blocks.
Image finite matching Float32/64 CPU[N,H,W,Cin], positive signed rectangles,
strides allowed. Vertical3x3/horizontal1x3 masked layers to8, paired gates to4,
unbiased vertical4-to-horizontal8 links, biased4-to4 horizontal projection;
blocks1/2 add a horizontal residual; biased1x1 head4-to-Cout. Missing W and
tables normal(.1), b0, source seed8501. Conditional mode adds six learned10x8
tables before gates; Int32/64[N] labels0..9 required. Unconditional mode accepts
omitted labels or validates and ignores supplied labels. Seven owning live
fields: vertical0/horizontal0/vertical1/horizontal1/vertical/horizontal/logits.
Gray binary head1073 values/23 leaves unconditional,1553/29 conditional,
320 masked coordinates. RGB3/30 context1410/23,448 masked; its output supplies
context parameters, not a complete within-pixel RGB density.
\code{nn$GatedPixelCnn(parameters,image,labels,conditional)} uses Cin=Cout=1
and requires valid labels in both modes, preserving the source binary protocol.
All applicable image/weight first/higher paths remain live after parent/store
collection. Inputs/configuration validate before leaves; stored-leaf errors may
follow initialization. Keep architecture mode fixed per store; unused old leaves
remain. No normalization, dropout, pooling, batch coupling, optimizer, fitting,
sampling, existing-leaf writes or hidden no-grad. Independent work shares no
mutable state; synchronize shared stores. Invalid arguments raise errors.
}

\section{Native Row-LSTM binary pixels}{
\code{nn$PixelRowLstmStep(input_gates,previous_hidden,previous_cell,recurrent_kernel)}
accepts finite matching Float32/64 CPU input_gates[N,1,W,16], previous h/c[N,1,W,4]
and live recurrent kernel[1,3,4,16], positive N/W, arbitrary strides. Total gates
add an unbiased SAME1x3 convolution of the previous hidden row. Gate ordering is
input, forget, output, candidate: sigmoid, sigmoid(+1), sigmoid, tanh.
Cell=f*previous_cell+i*candidate; hidden=o*tanh(cell). Seven owning live fields:
gates/input_gate/forget_gate/output_gate/candidate/cell/hidden. Applicable
first/higher input/state/kernel paths remain live. This primitive alone imposes
no image mask; caller owns causal features and state sequencing. No mutation,
store, parameter creation, fitting, optimizer, sampling, RNG or hidden no-grad.
\code{nn$PixelRnn(parameters,image)} supplies a compact one-layer width4 Row-LSTM
image density, source store seed8303. Finite matching floating CPU[N,H,W,1],
positive signed rectangles and strides allowed. Biased3x3 mask-A input1-to16,
unbiased previous-row hidden1x3 kernel4-to16, paired row recurrence with forget
bias+1, biased1x1 hidden4-to1 Bernoulli logits. Missing W normal(.1), biases0;
357 values/5 leaves,277 active/80 masked input weights. Initial h/c are the same
first input row's first4 features times zero, preserving the connected zero path.
Seven owning live fields: initial_hidden/initial_cell/input_gates/row_gates/
hidden/cell/logits. All H rows recur, each row's columns advance in parallel;
no same-row hidden recurrence, learned initial states or recurrent bias.
No class conditioning, batch coupling, normalization, dropout or pooling. Inputs
validate before leaves; stored-leaf/math errors may follow creation. Only missing
leaves initialize; no existing-leaf writes, later RNG, fitting/sampling/optimizer
or hidden no-grad. First/higher image/weight paths and flat fields survive
parent/store collection. Work/storage O(NHW), graph depth O(H); synchronize shared
stores. Invalid geometry/dtypes/values or nonfinite gate sums raise errors.
}

\section{Native MADE degrees and equal-probability density mixtures}{
\code{nn$AutoregressiveDegreeMask(input_degrees,output_degrees,strict=FALSE,dtype)}
compares nonempty nonnegative Int32/64 CPU[In]/[Out], arbitrary strides and mixed
integer widths. Entry(i,j) is input degree<=output degree, or strict<. Owning
stopped Float32(default)/64[In,Out] exact0/1. Zero degrees permit constants.
The comparison alone does not validate a complete network/order. Invalid
vectors/dtype or overflowing product raise errors; no RNG/store/mutation.
\code{nn$MadeMasks(reverse,dtype)} returns six owning stopped fields: Int32
input_degrees[64], hidden1_degrees[16], hidden2_degrees[16]; floating input_mask
[64,16], hidden_mask[16,16], output_mask[16,64]. Input degrees1..64 or64..1;
hidden1[i]=1+floor(i*62/15), hidden2[i]=hidden1[(5*i)mod16]. Input/hidden masks
use<=, output strict<. Default Float32 masks, Float64 allowed;1144 forbidden
weights in the source model. No store/fitting/RNG/global order state.
\code{nn$MaskedDensityDense(parameters,name,input,mask)} computes live
bias_add(input@(W*stopped_mask),b). Finite matching floating CPU[N,In] and
[In,Out], positive axes/strides, exact0/1 mask and nonempty name. Missing W
normal(.1), b0. Input/W/b first/higher paths live, structural mask stopped,
forbidden partials zero. No activation/fitting/optimizer/sampling/existing-leaf
writes or hidden no-grad. Inputs validate before leaves; stored-leaf errors
may follow creation; nonfinite output rejects. Synchronize shared stores.
\code{nn$MadeDensity(parameters,image,reverse=FALSE)} accepts finite matching
Float32/64 CPU[N,8,8,1], signed values/strides allowed. Flatten64, two16-unit
sigmoid layers,64 logits with<=,<=,strict< masks. Source seed8803,2400 values/
6 leaves,1256 active/1144 masked. Owning live hidden1/hidden2[N,16] and logits
[N,8,8,1]; image/weight first/higher paths survive parent/store collection.
Reverse changes per-call causal order, not raw output layout. Source fits separate
stores, not one shared-weight order-agnostic model. Inputs validate before leaves;
no hidden optimizer, fitting/sampling, normalization, dropout, batch coupling,
existing-leaf writes or no-grad. Invalid calls raise errors.
\code{nn$EqualDensityMixture(per_model_nll)} accepts finite floating CPU[N,K],
positive axes, arbitrary strides and signed NLLs. Cast to Double and compute
-log(sum(exp(-nll))/K) per observation with a stopped row maximum. Three owning
live Float64 fields: per_image_nll[N], mean_nll scalar, responsibilities[N,K].
First/higher paths survive parent/input collection, including Float32 input casts.
Responsibilities=softmax(-nll), the per-image NLL derivative. A normalized mixture
interpretation requires normalized component joints on the same base measure;
that cannot be verified from arbitrary numbers. Use per-image sums, not mean
pixel NLLs. No model evaluation, fitting/sampling, mutation, clipping or hidden
no-grad. Invalid inputs/nonfinite outputs raise errors. This is a probability
mixture, not the arithmetic mean of NLLs or a shared-weight training claim.

\code{nn$RgbAreaPixels(original,block_size=4)} accepts Int32/64 CPU[N,H,W,3]
values0..255, positive axes/strides and positive B dividing H/W. Compute exact
integer square-block sums and nearest half-up byte codes. Two owning detached
fields: codes Int32[N,H/B,W/B,3] and images Float32, double(code)/127.5-1 cast
once. No source mutation, fitting, RNG or graph. Invalid geometry/range/type or
unsafe sums raise errors. Source32x32 with B4 is exact compact8x8 preparation.
\code{nn$DiscretizedLogisticRGB(image,parameters,mixtures=3)} accepts finite
matching Float32/64 CPU[...,3] observations in[-1,1] and[...,10*K] parameters,
positive matching leading axes/strides. Component-major slots: logit,meanRGB,
logscaleRGB,coeffRG/BR/BG. Tanh coefficients couple G/B means to observed R/G.
Scales lower-clip at-7(ties differentiate raw). Stable exact-bin log identity,
Float32 half/full bin widths promoted even for Double, infinite tails for
x<=float(-.999) or>=float(.999), stable mixture logsumexp with stopped maximum.
No CDF subtraction or small-mass fallback. Eight owning live flat fields:
pixel_log_probability[P], component/mixture_log_probability[P,K], means,
log_scales,coefficients,channel_log_mass[P,K,3], mean_channel_nll scalar nats per
RGB channel. Applicable first/higher paths survive input/parent collection.
No fitting, RNG, sampling, mutation or hidden no-grad. Invalid/nonfinite output
raises errors; extreme finite parameters may overflow/underflow and reject.
\code{nn$PixelCnnPP(parameters,image)} accepts finite matching floating
CPU[N,H,W,3] signed RGB[-1,1], positive rectangles/strides and caller store.
Three width4 dual streams,30 context head channels,3 logistic components.
Source seed8609 gives1410 values/23 leaves,962 active/448 masked. All30 head
parameters exclude the whole current/future pixel; observed R/G coupling is in
likelihood. Twelve owning live fields: parameters[N,H,W,30], vertical/horizontal
[N,H,W,4], pixel_log_probability[P], component/mixture_log_probability[P,3],
means/log_scales/coefficients/channel_log_mass[P,3,3], per_image_nll[N], loss
scalar nats/channel. First/higher paths survive store/parent collection. Inputs
validate before leaves. No fitting, optimizer, sampling, labels, dropout,
multiscale stack or hidden no-grad. Invalid inputs/nonfinite outputs raise errors.

\code{nn$WaveStackConfig()} owns integer architecture fields residual(default32),
skip64, levels6, cycles2, classes256. Positive widths, levels1..12,cycles1..4,
classes>=2. Set compact source fields4/8/5/1/256 explicitly; RF33.
\code{nn$WaveStack(parameters,encoded,config)} evaluates the canonical causal
two-tap gated stack on matching floating CPU[N,T,classes]. Creates/reuses Glorot
kernels and zero biases; warm before optimizer/cache construction. Flat owning
live logits[N,T,classes] and skip[N,T,S], plus a copied vector of owning gate
Tensor handles[N,T,R]. Saved gates survive parent collection. All-but-last
blocks have residual projections. Same-time input predicts the following code;
no fitting/optimizer/dropout/normalization/conditioning. Invalid calls raise errors.
\code{nn$WaveNll(logits,target,classes=256,start=127)} returns a live scalar mean
next-code NLL over N*(T-start). Floating logits[N,T,classes], matching bounded
Int32/64 targets[N,T], 0<=start<T. Earlier logit adjoints are exactly zero.
The generative source uses start32, not127. No fitting or parameter writes.
\code{nn$WaveStackCache(parameters,config)} snapshots required finite Float32
CPU warmed kernels/biases of an explicit matching WaveStack architecture.
Unrelated leaves are ignored, strided leaves copied. Owns weights and each
block's OWN circular input history at its dilation. Last block omits residual
projection. Store writes/collection cannot affect it. No Float64 cache, sampling,
codec, RNG, fitting, optimizer, parameter mutation or autograd path.
\code{cache$next_code(code)} consumes0..classes-1 and returns detached owning
Float32[classes] next-code logits using source scalar multiply/add and gate
arithmetic. Existing returned heads survive future steps/reset/cache collection.
\code{cache$consumed()} counts successful fed codes since construction/reset.
\code{cache$reset()} restores zero feature histories/no previous code and count0,
without changing weights, old heads or other caches. \code{cache$clone()} forks
all current state into an owning independent cache. One cache needs caller
synchronization; distinct caches do not share mutable state. Invalid code/count
limit/nonfinite candidate rejects without changing history. Invalid construction,
missing/misshaped/nonfinite/non-CPU/non-Float32 leaves raise errors. Cache/full
convolution may differ by Float32 roundoff. Keep warmup/sampling loops explicit.

\code{nn$WaveStackField(config)} returns2+cycles*(2^levels-1), including the
initial two-tap ingress. Invalid architecture metadata raises errors.
\code{nn$WaveStackConv(parameters,name,x,out,kernel=1,dilation=1)} applies a
live explicitly left-padded convolution to matching floating CPU[N,T,C].
Positive axes/output/kernel/dilation; named kernels[1,kernel,C,out] use Glorot
uniform sqrt(6/(kernel*(C+out))), biases[out] zero. Owning live[N,T,out],
same-time inclusive causal input; no activation, fitting or optimizer.
Missing leaves initialize, existing leaves reuse; invalid calls raise errors.
\code{WaveStackOutput} owns live logits,skip and a copied vector of Tensor
handles. Saved gate vectors/tensors outlive the original output and store.

\code{nn$BananaPoints(standard,curvature=.5,dtype=nn$DType$Float32)} takes a
finite floating CPU[N,2] explicit bank, computes (u,v+c*(u*u-1)) in Double and
casts once. Owning stopped contiguous observations, no RNG/fitting/selection or
input mutation/graph. Float32/64 output; finite curvature, negative/zero allowed.
\code{nn$BananaLogProbability(points,curvature=.5)} returns live Float64[N]
analytic log probabilities in nats using full Double log(2*pi) and unit triangular
determinant. Applicable first/higher input paths survive parent collection.
\code{nn$StandardNormalLogProbability(latent,rounded_constant=FALSE)} accepts
finite floating CPU[N,D], positive axes/strides; return matching live[N]
-.5*sum(z*z,axis1)-.5*D*log(2*pi). TRUE rounds log(2*pi) to Float32 even for
Double, reproducing the original compact flow source. No RNG/fitting/mutation.
\code{nn$NormalFlowDensity(latent,forward_logdet,rounded_constant=FALSE)} uses a
matching finite scalar or[N] determinant describing x->z. Return three owning
live flat fields: base_log_probability[N], log_probability=base+logdet[N], loss
scalar negative row mean in nats. Applicable first/higher graphs and storage
survive parent collection. Does not derive Jacobians or prove invertibility.
\code{nn$NiceCoupling(parameters,input,block,inverse=FALSE)} takes matching
finite floating CPU[N,2], block0..3; condition on coordinate block modulo2.
Names coupling<block>/hidden1->16 tanh normal(.3)/bias.05, /shift16->1 linear
normal(.01)/bias0. Add shift to the other coordinate in raw axis order, or
subtract for inverse TRUE. Determinant one;49 values/four leaves per block.
\code{nn$NiceDensity(parameters,input)} applies blocks0..3 then exp(log_scale[2])
with missing scale zero; source seed9301 gives198 values/17 leaves. Seven owning
live flat fields coupling0..3,unscaled,latent[N,2], logdet scalar sum(log_scale).
\code{nn$NiceInverse(parameters,latent)} unscales then subtracts shifts3,2,1,0.
Two owning live fields value[N,2] and scalar logdet=-sum(log_scale). Warm forward
first for source initialization order; missing inverse leaves initialize scale
then reversed conditioners. Warmed stores have no writes/RNG; no fitting,
optimizer, sampling or hidden no-grad. Both forward/inverse scales must be
positive finite. Applicable input/parameter first/higher paths survive parent
and store collection; arbitrary strides. Synchronize shared mutable stores.
All operations reject invalid geometry/dtypes/nonfinite inputs or computed
values; stored-leaf/math failures can follow partial leaf creation. No clipping;
finite precision roundtrip error remains. Output record types are
\code{NiceDensityOutput}, \code{FlowInverseOutput}, \code{NormalFlowDensityOutput}.

\code{nn$FlowDiagonalScale(parameters,input,incoming_logdet,inverse=FALSE)}
accepts finite matching floating CPU[N,D], scalar or[N] determinant, positive
axes/strides. Missing log_scale[D] initializes zero. Forward multiplies by
exp(log_scale), adds its sum; inverse divides and subtracts. Both scale directions
must stay positive finite. Owns live flat value[N,D], scalar or[N] logdet.
\code{nn$RealNvpCoupling(parameters,input,block,inverse=FALSE)} accepts matching
finite floating CPU[N,2], block0..3; conditions on coordinate block modulo2.
Named coupling<block>/hidden1->16 tanh normal(.3)/bias.05, heads16->2 linear
normal(.01)/bias0. s=.8*tanh(head0), t=head1. Forward b*exp(s)+t, inverse
(b-t)*exp(-s), unchanged raw conditioning coordinate. Four owning live fields
value[N,2], logdet[N] (+s or-s), scale/shift[N,1]. Scale is bounded LOG scale,
not its exponential. Computation-dtype .8 bound;66 values/four leaves per block.
\code{nn$RealNvpDensity(parameters,input)} applies four affine blocks then
global positive diagonal scale; source seed9403 gives266 values/17 leaves.
Nineteen owning live flat fields: coupling0..3/unscaled/latent[N,2], logdet[N],
coupling_logdet0..3[N], coupling_scale0..3/coupling_shift0..3[N,1]. First/higher
graphs/storage survive parent/store collection. Total determinant varies by row.
\code{nn$RealNvpInverse(parameters,latent)} unscales then reverses blocks3,2,1,0.
Compute each scale/shift on its OWN inverse path; return live value[N,2],
logdet[N]. Warm forward first for source initialization order. Missing inverse
leaves initialize scale then reverse conditioners. No cached forward determinant.
All operations own flat Tensor fields with live first/higher paths, accept
arbitrary strides, reject invalid geometry/dtypes/nonfinite values and perform
no fitting, optimizer, sampling or warmed-store mutation/RNG. Inputs validate
before leaves; stored-leaf/math failures can follow partial creation. Synchronize
shared mutable stores. Floating roundtrip errors remain. Records:
\code{FlowScaleOutput}, \code{RealNvpCouplingOutput}, \code{RealNvpDensityOutput}.


\code{nn$FlowActNormCalibration(input,epsilon=1e-6)} takes finite floating
CPU[N,D], positive axes/strides and positive finite epsilon. Serial divide-first
Double row mean/population variance, divisor N; proposes bias=-mean and
log_scale=-.5*log(var+epsilon), cast once to input dtype. Flat owning detached
mean/variance Float64[D], bias/log_scale matching floating[D]. Constant columns
and N=1 supported. No store, writes, selection, RNG, graph or initialized flag.
Caller explicitly selects fitting rows and installs proposals once per stage.
\code{nn$FlowActNorm(parameters,name,input,inverse=FALSE)} accepts matching
finite floating CPU[N,D], nonempty name. Missing name/bias and /log_scale[D]
zero. Forward (x+bias)*exp(scale), scalar sum(scale); inverse x*exp(-scale)-bias,
negative sum(scale). No automatic calibration, batch statistics or moving state.
\code{nn$FlowMix2D(parameters,name,input,angle=.35,inverse=FALSE)} accepts
matching finite floating CPU[N,2], nonempty name and finite angle in radians.
Missing name/W[2,2] rotation is Double cos/sin rounded to Float32 then promoted
if needed, with no RNG draws. Forward x@W, inverse x@adjugate(W)/det(W), scalar
logdet=+/-.5*log(det(W)^2). Negative determinants valid; singular/unrepresentable
positive finite squared determinants reject. Row input Jacobian is W transpose.
\code{nn$GlowDensity(parameters,input)} applies four ActNorm/mixing/bounded
RealNvpCoupling stages; source seed9503,296 values/28 leaves, no final global
scale. Missing ActNorm parameters zero; mix<i> angle=.35+.2*i. Owns31 flat live
fields: actnorm_input/output0..3, mixed0..3, coupling0..3, coupling_logdet/scale/
shift0..3, unscaled,latent,logdet. Coordinate maps[N,2], stage/total logdets[N],
bounded LOG scale/shift[N,1]. Caller owns sequential first32 fitting calibration
and all fitting/sampling; no automatic calibration or batch statistics.
\code{nn$GlowInverse(parameters,latent)} reverses affine coupling, mixing and
ActNorm in stages3..0; each per-point determinant follows its OWN inverse path.
Negative matrix signs use log absolute determinants. Warm forward first for
source ordering; missing inverse leaves initialize in reverse order. No cached
forward determinant. Live flat value[N,2], logdet[N] retain first/higher paths.
All transforms accept arbitrary strides, own applicable graphs/storage after
parent/store collection, never change existing leaves and consume no warmed RNG.
No fitting, optimizer, sampling or hidden no-grad. Both scale exponentials and
squared mixing determinants must remain positive finite; invalid geometry,
types, names/options or nonfinite fields raise errors. Supplied inputs validate
before leaves; stored-leaf/math failures can follow partial creation. Synchronize
shared stores; floating roundtrip errors remain. Calibration itself is stopped.
Records: \code{ActNormCalibration}, \code{GlowDensityOutput}, \code{FlowScaleOutput}.

}

\section{Compact masked autoregressive flow}{
\code{nn$MafMasks(block,dtype=nn$DType$Float32)} returns five detached owning
fields: Int32 input degrees[2], hidden degrees[16], output degrees[4], and
matching floating input_mask[2,16]/output_mask[16,4]. Block0..3 alternates
input degrees(1,2)/(2,1); hidden degrees are floor(j/8); heads repeat input
degrees for mean0,mean1,raw_scale0,raw_scale1. Input mask uses<=, output strict<,
excluding own and future input dependencies. Forty forbidden kernel entries
per block. Float32/64 masks only; no store, RNG, mutation, fitting or graph.

\code{nn$MafConditioner(parameters,input,block)} accepts finite matching
Float32/64 CPU[N,2], positive rows and arbitrary strides. Missing leaves under
\code{autoregressive<block>/hidden} initialize W[2,16] normal(.3), b[16]=.05,
then \code{/heads} W[16,4] normal(.01), b[4]=0. Strict masks multiply kernels
inside the graph. The owning flat mean/scale[N,2] retain applicable first/higher
graphs after collection. scale=.8*tanh(raw) is a log scale. Four leaves/116
stored values,40 masked coordinates with exact zero partials. Source MAF seed9603,
paired IAF seed9703. Only missing leaves initialize; no existing writes, fitting,
optimizer, sampling or warmed RNG.

\code{nn$MafDensity(parameters,input)} applies four parallel masked Gaussian
transforms (x-mu)*exp(-scale), each with negative sum(scale) logdet, then a
learned global positive diagonal scale. Matching floating CPU[N,2], positive
rows/strides. Source seed9603,466 values/17 leaves including160 masked entries.
Owns23 flat live fields: conditioner_input/coupling/coupling_scale/coupling_shift0..3
[N,2], coupling_logdet0..3[N], unscaled/latent[N,2] and total logdet[N]. No fitting,
optimizer, sampler, hidden no-grad, warmed writes or RNG.

\code{nn$MafInverse(parameters,latent)} unscales globally and reverses blocks3..0.
Within each block recover coordinates0,1(even) or1,0(odd), recomputing the strict
conditioner on the recovered prefix before each x_c=z_c*exp(s_c)+mu_c. Accumulate
used scales in dependency order. Two conditioner passes per block; density uses
one. Owns live value[N,2]/logdet[N], with applicable first/higher graphs after
collection. The determinant uses its own recovered points; no cached forward
state. Forward warmup preserves source initialization order; inverse creation
orders missing leaves differently. No fitting, optimizer, sampling or warmed
writes/RNG. Floating roundtrip error remains.

Invalid inputs/block/dtype, nonfinite leaves/results or unrepresentable global
scales raise errors; stored-leaf/math failures may follow partial creation.
Synchronize shared mutable stores. Records: \code{MafMaskInventory},
\code{MafConditionerOutput}, \code{MafDensityOutput}, \code{FlowScaleOutput}.
}

\section{Compact affine inverse autoregressive flow}{
\code{nn$IafDensity(parameters,input)} accepts finite matching Float32/64
CPU[N,2], positive rows and arbitrary strides. Source blocks3..0 recover
coordinates0,1(even) or1,0(odd) sequentially from masked Gaussian heads.
Recompute \code{MafConditioner} on each recovered prefix before
u_c=(observed_c-mu_c)*exp(-s_c), then once more on the fully recovered input
for the stage's negative sum(s) determinant. This third call preserves the
source graph and rounding. Apply \code{FlowDiagonalScale} last. Source seed9703
warms blocks3,2,1,0 then scale;466 stored values/17 leaves including160 masked
kernel coordinates. Owns23 live flat fields: conditioner_input/coupling0..3[N,2]
are recovered inputs, coupling_scale/shift0..3[N,2] are final log scales and
means, coupling_logdet0..3[N], unscaled= coupling0[N,2], latent[N,2], logdet[N].
Applicable first/higher graphs survive parent/store collection. Only missing
leaves initialize; no existing writes, warmed RNG, fitting, optimizer, sampler
or hidden no-grad. This standalone affine exact-density adaptation is distinct
from the paper's sigmoid-gated variational posterior; no VAE/ELBO workflow.

\code{nn$IafInverse(parameters,latent)} unscales globally, then applies
blocks0..3 with one parallel conditioner pass each on the known current input:
h=h*exp(s)+mu and add sum(s) per row. Four calls total, versus twelve density
calls. The inverse determinant uses its own points, without cached forward
state. Matching floating CPU[N,2], positive rows/strides. Owns live value[N,2]
and logdet[N] with applicable first/higher graphs after collection. Forward
warmup preserves source initialization order; inverse creation starts global
scale then blocks0..3. No fitting, optimizer, sampler, VAE workflow or warmed
writes/RNG. Floating roundtrip errors remain.

Invalid inputs, unsafe activations, nonfinite fields or unrepresentable global
scales raise errors; stored-leaf/math failures may follow partial creation.
Both global scale directions must stay positive finite. Synchronize shared
mutable stores. Records: \code{IafDensityOutput}, \code{FlowScaleOutput}.
Shared strict degrees and Gaussian heads use \code{MafMasks}/\code{MafConditioner}.
}

\section{Rational-quadratic splines and compact density flows}{
\code{nn$RationalQuadraticSpline(input,heads,inverse=FALSE,bound=4,
min_bin_fraction=.01,min_derivative=.001,rounded_offset=FALSE)} accepts matching
finite floating CPU input[N,1] and heads[N,3*K-1], positive N,K>=2 and arbitrary
strides. FirstK logits give widths, nextK heights, finalK-1 interior derivatives.
Declared sizes=2*bound*(fraction+(1-K*fraction)*softmax(logits)); returned actual
sizes are adjacent cumulative-knot differences. Slopes=floor+softplus(raw+
log(expm1(1-floor))), boundaries1. Bound positive finite, fraction positive with
K*fraction<1, derivative floor strictly0..1. Rounded mode rounds the softplus
offset once to Float32 before promotion, preserving the original source even
in Double; default keeps Double. Zero heads are near identity with floating
rounding. Exact knots use the right bin, capped atK-1. Inverse uses y edges
and2c/(-b-sqrt(b*b-4ac)), with no root search. Inclusive endpoints use the spline;
strict outside tails preserve input and zero logdet, giving C1 tails in exact
arithmetic. Log-derivative partials can have distinct one-sided knot values.
Owns live value[N,1],logdet[N],actual widths/heights[N,K],derivatives[N,K+1].
Applicable first/higher input/head graphs survive collection within structural
bins. No store, RNG, fitting, optimizer, sampler or hidden no-grad.

\code{nn$SplineCoupling(parameters,input,block,inverse=FALSE)} accepts matching
finite floating CPU[N,2],positive rows/strides and block0..3. Preserve coordinate
block modulo2, condition1->16 tanh->11 heads. Missing hidden W normal(.3),b=.05,
then head W normal(.01),b=0;219 values/4 leaves. Apply the pure spline with
K4,bound4,fraction.01,floor.001,rounded mode on the other coordinate. Owns
value[N,2],logdet[N],actual widths/heights[N,4],derivatives[N,5]. Only missing
leaves initialize; warmed inference writes no leaves or RNG. First/higher graphs
survive collection within structural bins. No fitting, optimizer or sampler.

\code{nn$SplineDensity(parameters,input)} applies source blocks0..3 and a final
global positive diagonal scale to matching finite floating CPU[N,2],positive
rows/strides. Source seed9803,878 stored values/17 leaves. Owns23 live flat fields:
coupling0..3[N,2],coupling_logdet0..3[N],coupling_widths/heights0..3[N,4],
coupling_derivatives0..3[N,5],unscaled/latent[N,2],full logdet[N]. Archival source
scale/shift fields are actual widths/heights, not log scales or affine shifts.

\code{nn$SplineInverse(parameters,latent)} unscales globally, then reverses
couplings3..0 using their own conditioning coordinates, height bins and analytic
quadratic roots. Owns live observation value[N,2]/inverse-path logdet[N]. Forward
warmup preserves source initialization order; inverse creation starts global
scale then reversed blocks. No cached forward values, root search, fitting,
optimizer, sampler, hidden no-grad or warmed writes/RNG. Applicable first/higher
graphs survive collection within structural bins. Floating roundtrip errors remain.

Invalid inputs/options or nonpositive represented bins/slopes/discriminants/
derivative terms and nonfinite math raise errors; extreme finite parameters
may reject. Stored-leaf/math failure may follow partial leaf creation. Both
global scale directions must stay positive finite. Synchronize shared mutable
stores. Records: \code{RationalQuadraticSplineOutput},\code{SplineDensityOutput},
\code{FlowScaleOutput}.
}

\section{Planar and radial density maps}{
\code{nn$PlanarFlow(input,angle,log_frequency,raw_projection,perpendicular,bias,inverse=FALSE)}
accepts finite matching floating CPU input[N,2],positive rows/arbitrary strides,
and five scalar Tensor parameters. Frequency=exp(log_frequency), direction=
(cos(angle),sin(angle)), projection=softplus(raw_projection)-.9, w=frequency*direction,
u=(projection/frequency)*direction+perpendicular*(-sin(angle),cos(angle)).
Forward t=sum(input*w)+bias, value=input+u*tanh(t), factor=1+projection*(1-tanh(t)^2),
logdet=log(factor) per row. Inverse solves t+projection*tanh(t)=c with32 bisections
then ONE Newton correction whose root and denominator are stopped. Returns
input-u*tanh(t) and negated logdet. This is a numerical inverse, not closed form.
First derivatives follow the source implicit correction; higher graphs follow
that stopped correction, NOT exact higher implicit derivatives.

\code{nn$RadialFlow(input,center,raw_alpha,raw_beta,inverse=FALSE)} accepts matching
center[2] and scalar Tensors. Alpha=.1+softplus(raw_alpha), beta=softplus(raw_beta)-alpha.
Radius uses sqrt(max(sum(offset^2),1e-30)) at positive squared radius; explicitly
zero with zero radius derivative at center. Forward factor=1+beta/(alpha+r),
radial_factor=1+beta*alpha/(alpha+r)^2; value=center+offset*factor,
logdet=log(factor)+log(radial_factor). Inverse uses the stable two-branch positive
quadratic radius and center+offset/factor, negating logdet on its own path.
At center value Jacobians are factor*I and I/factor. Radius/logdet center
partials follow the source zero convention; no smooth center Hessian is promised.

Both return owning live \code{ElementaryFlowOutput}: value[N,2],logdet[N],
factor/coordinate[N,1]. Planar coordinate is t; radial coordinate is source
radius, recovered for inverse. Factor is not a log scale. Applicable input/
parameter graphs survive collection within the stated source conventions.

\code{nn$ElementaryFlow(parameters,input,block,radial=FALSE,inverse=FALSE)}
creates missing flow<block> leaves for block0..3. Planar has five scalars:
angle=Double(.4f+.65f*block), log_frequency0, raw_projection.4,
perpendicular=Double(.02f*(odd?-1:1)),bias.05. Radial has center[2] normal(.2),
raw_alpha.4 and raw_beta.56. Use a separate matching floating store per family.

\code{nn$PlanarRadialDensity(parameters,input,radial=FALSE)} applies blocks0..3
then global diagonal scale. Planar22values/21leaves, radial18/13, source seed9903.
Returns19 owning flat \code{PlanarRadialDensityOutput} fields: flow/logdet/factor/
coordinate0..3,unscaled/latent[N,2],full logdet[N].
\code{nn$PlanarRadialInverse(parameters,latent,radial=FALSE)} unscales globally
then reverses blocks3..0, returning live \code{FlowScaleOutput} value/logdet.
Warm the forward first for source initialization order. Inverse gradients remain
live with the stated planar stopped correction and radial center rules.

No fitting, optimizer, sampler, hidden no-grad or warmed writes/RNG. Only missing
leaves initialize; synchronize mutable stores. Invalid/nonfinite inputs/leaves,
nonpositive represented factors or unrepresentable math raise errors, possibly
after partial leaf creation. Both frequency/global-scale directions must remain
positive finite. Floating inverse residuals remain; no fallback refit or clipping.
}

\section{Continuous flow field and differentiable RK4}{
\code{nn$CnfField(parameters,input,time,trace_mode=3L)} accepts finite matching
Float32/64 CPU input[N,2],positive rows/arbitrary strides and a finite structural
Float32 time scalar. Time rounds to Float32 before promotion even in Double.
Append time; hidden3->8 tanh(W normal(.3),b=.05), output8->2(W normal(.1),b=0),
velocity=.3*output. Source seed10003 creates missing hidden W then output W before
biases:50values/4leaves. Exact trace=.3*sum((1-hidden^2)*(a*c+b*d)), with a,b
spatial hidden-W rows and c,d output-W columns. Plus probe(1,1) uses(a+b)*(c+d),
minus(1,-1) uses(a-b)*(c-d). Mode0 exact,1 plus,2 minus,3(default) deterministic
pair mean. Both probes average to exact2D field divergence in exact arithmetic;
a single probe generally differs. No stochastic probe stream is used.
Returns five owning live \code{CnfFieldOutput} fields: velocity[N,2],
divergence/exact_trace/hutchinson_plus/hutchinson_minus[N]. All applicable first/
higher input/parameter paths survive collection; time is structural, not live.

\code{nn$CnfIntegrate(parameters,input,inverse=FALSE,steps=8L,trace_mode=3L)}
integrates state and cumulative selected field trace with fixed RK4, from0 to1
or1 to0 if inverse=TRUE. Steps at least4 and divisible by4. Source dt,half,sixth
and ALL stage times use Float32 arithmetic before promotion even for Float64
tensors; preserve grouping(a+d)+2*(b+c). Four field evaluations per step plus
four quarter endpoints. Ordinary reverse mode differentiates through the entire
unrolled solver. No adaptive solve, continuous adjoint or constant-memory claim.
Returns19 flat owning \code{CnfDensityOutput} fields: stage0..3[N,2],
CUMULATIVE stage_logdet0..3[N], endpoint stage_divergence0..3[N] and
stage_velocity0..3[N,2],unscaled/latent[N,2] and full logdet[N]. For the bare
integrator final latent and unscaled alias stage3; no global scale is created.
Inverse checkpoints follow reverse chronological endpoints, not forward states.

\code{nn$CnfDensity(parameters,input,steps=8L,trace_mode=3L)} integrates forward
then globally scales latent and adds sum(log_scale).52values/5leaves, source
seed10003, one shared field of actual hidden width8. The original generic
archived hidden16 metadata does not describe this architecture.
\code{nn$CnfInverse(parameters,latent,steps=8L,trace_mode=3L)} unscales then
integrates1 to0 and subtracts global log scale, returning owning live
\code{FlowScaleOutput} value[N,2]/logdet[N]. Warm forward first for source
initialization order; inverse creation starts global scale before field leaves.
Every stage recomputes on the supplied inverse path. First/higher paths remain
live through all steps after parent/store collection.

The finite RK4 reverse path is a NUMERICAL inverse, not the exact inverse of
the discrete forward map. Integrated divergence approximates continuous-flow
logdet for modes0/3 only, not the exact determinant of the finite RK4 map.
Modes1/2 integrate fixed-probe diagnostic contractions and generally do not
approximate full logdet or define a density. Actual-Jacobian,
roundtrip and normalization residuals remain. Float32 coefficients/times can
limit refinement even with Double tensors. No fit, optimizer, sampler, hidden
no-grad or warmed writes/RNG. Only missing leaves initialize; synchronize
shared stores. O(steps*8N) work and retained graph storage. Invalid inputs/options,
nonfinite leaves/preactivations/fields/states or unsafe global scale raise errors,
possibly after partial leaf creation; no clipping, inverse refit or fallback.
}

\section{Three-family flow-study data and diagnostics}{
\code{nn$FlowStudyPoints(standard,kind,dtype=nn$DType$Float32)} takes finite
floating CPU[N,3] explicit Gaussian triples, N>0, arbitrary strides. Family0
is banana, family1 eight equal Gaussian components (radius2,sd.15), family2
eight checkerboard unit cells on [-2,2) squared. Component selection is
min(7,int(8*Phi(w))), Phi=.5*erfc(-w/sqrt(2)); checker jitter uses Phi(u),Phi(v).
Double scalar math precedes one final observation cast. Extreme tails/Float32
rounding can create boundary points; no clipping. Owning detached flat
\code{FlowStudyPointsOutput} has points[N,2] and Int32 components[N] (-1 for
banana, otherwise0..7). Components are diagnostics, never model features.
Caller owns RNG, the complete flat3D draw stream, splits and source IDs.
\code{nn$FlowStudyLogProbability(points,kind)} returns detached Float64[N]
analytic log densities: banana inverse, stable Double eight-term Gaussian
log-sum-exp, or checkerboard -log(8) with negative infinity off support.
Checker support is half-open [-2,2) squared and even floor(x+2)+floor(y+2)
parity; negative infinity is a legitimate zero PDF. Inputs are finite floating
CPU[N,2]. No autograd is retained. Kind0/1 overflow is rejected.
\code{nn$FlowStudySampleDiagnostics(samples)} accepts finite floating
CPU[N,2],0<N<=INT32_MAX, arbitrary strides. Owning detached
\code{FlowStudyDiagnosticsOutput} has Int32 mode_counts[8], scalar
checkerboard_support_fraction, fraction_near_modes, integer modes_covered and
banana_curvature_moment. Compute every diagnostic for every family. Nearest
radius2 ring centers use Double hypot, strict comparisons (first exact tie)
and inclusive distance<=.45. Eight covered modes does not establish sample
concentration. Checker support uses the same half-open parity. Curvature is
mean(y*(x*x-1)), true banana expectation1; source serial Double sums divide
by N before adding. Overflow is rejected. All functions are pure, O(N) work
and storage, with no fitting, sampling, mutation or RNG state. Invalid
geometry/type/kind and unsafe arithmetic raise errors. Values survive parent
collection. Example: FlowStudyPoints(zeros(c(1L,3L)),0L) gives (0,-.5);
FlowStudySampleDiagnostics(zeros(c(1L,2L))) has modes_covered=0.
Reference: original flow_density_2d three-family teaching protocol.
}

\section{Explicit diffusion schedules and compact DDPM operations}{
\code{nn$CosineDiffusionSchedule(timesteps=32L,offset=.008,beta_cap=.999)}
returns nine detached owning Float64[T] tables in \code{DiffusionScheduleOutput}:
beta,alpha,alpha_bar,alpha_bar_previous,sqrt_alpha_bar,sqrt_one_minus_alpha_bar,
posterior_variance,posterior_coefficient1,posterior_coefficient2. T>=2,
finite offset>=0 and0<cap<1; scalar Double cosine ratios, serial products,
fixed posterior variance (first exactly zero). Invalid/degenerate math rejects.
\code{nn$DiffusionExtract(coefficients,time,sample_rank=4L,dtype=nn$DType$Float32)}
selects a finite floating CPU[T] table at Int32/64 CPU[N] indices0..T-1,
casts then reshapes to[N,1,...,1]. Positive result rank, arbitrary strides;
applicable table first/higher paths remain live. Float64 output does not first
round to Float32. Invalid geometry/type/index/options or cast overflow rejects.
\code{nn$DiffusionTimeEmbedding(time,maximum_time=31,dtype=nn$DType$Float32)}
accepts nonnegative structural Int32/64 CPU[N], finite positive Float32 maximum.
Float32 normalization and pi*(1,2,4,8) phases, then four sines/four cosines,
are retained even before final Float64 cast. Returns owning detached[N,8];
time0 gives(0,0,0,0,1,1,1,1). No integer-time derivatives.
\code{nn$DiffusionNoised(clean,noise,time,schedule)} accepts matching finite
floating CPU[N,...],positive axes/rank>=2 and Int32/64 time[N]. Consume signal
and sigma tables as Float32, then promote to sample dtype even for Double.
Return signal*clean+sigma*noise. Selected signal>0,sigma>=0; equal table lengths.
Returns owning live applicable first/higher input/table paths. User-supplied
schedule consistency and explicit noise banks are caller-owned.
\code{nn$DdpmEpsilon(parameters,state,time)} accepts matching finite floating
CPU[N,8,8,1] and caller-owned store, Int32/64 CPU[N] times0..31. Flatten64pixels,
append8rounded time features,72->32SiLU->32SiLU->64; SiLU=x*sigmoid(x).
Create missing hidden1/W,b,hidden2/W,b,output/W,b in that order before evaluation,
normal(.1) kernels/zero biases. Source11003 gives5504values/six leaves. No
prefix; use separate stores. Inputs validate before leaves; invalid existing
leaves/math can follow partial creation. Warmed calls consume no RNG/mutation.
Flat \code{DdpmEpsilonOutput}: epsilon[N,8,8,1],time_embedding[N,8],hidden1/2[N,32].
Epsilon/hidden first/higher state/parameter graphs survive parent/store collection;
time features are detached. Not the paper's image U-Net. Synchronize shared stores.
\code{nn$DdpmReverse(state,time,epsilon,innovation,schedule)} accepts matching
finite floating CPU[N,...],rank>=2 and structural Int32/64 time[N]. Extract
signal,sigma,posterior c1/c2,variance as Float32 then promote even for Double.
raw=(state-sigma*epsilon)/signal; clipped=min(max(raw,-1),1);
mean=c1*clipped+c2*state; value=mean+sqrt(variance)*innovation. Selected signal
positive, sigma/variance nonnegative, all consumed tables finite/equal lengths;
user consistency is caller-owned. First generated posterior variance is exactly
zero, so time0 innovation has no effect but still must be finite/matching.
Flat \code{DdpmReverseOutput}: value,mean,x0_raw,x0_clipped match sample shape;
variance[N,1,...,1]. Applicable first/higher paths survive collection. Core min/max
tie subgradients apply; clip bounds and sqrt(trainable variance) at zero are
nonsmooth. Nonfinite intermediate raw values reject before clipping.
All operations own storage, accept documented arbitrary strides and use no
hidden fit, optimizer, timestep/noise sampler, full ancestral loop or mutable
cache. Schedule O(T); extraction O(T+N); noising/reverse O(sample elements+T);
predictor O(N*5504). Invalid inputs/unsafe math raise errors. Complete loops
belong to each language example. References: Ho,Jain,Abbeel(2020),Denoising
Diffusion Probabilistic Models, https://arxiv.org/abs/2006.11239 ;
Nichol,Dhariwal(2021),Improved Denoising Diffusion Probabilistic Models,
https://arxiv.org/abs/2102.09672 . This source uses32levels and fixed variance.
}
