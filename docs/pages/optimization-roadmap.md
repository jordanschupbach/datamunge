# Optimization roadmap

This is the planned expansion of `datamunge::optim`.  Items are ordered by the
smallest useful extension to the current function interfaces, not by a promise
of release order.

## Implemented recently

- Cyclic coordinate descent
- Randomized cyclic block coordinate descent
- Nesterov accelerated gradient
- Nonlinear conjugate gradient (Polak-Ribiere+)
- Nelder-Mead simplex search
- CMA-ES
- AdaGrad
- RMSProp
- AMSGrad
- Nadam
- AdaDelta
- SVRG
- SAGA
- Proximal gradient descent (with `ProximalFunction`)
- FISTA
- Levenberg-Marquardt (with `ResidualFunction`)
- Newton and trust-region Newton (with `HessianFunction`)
- Augmented Lagrangian, SQP, and interior-point methods (with
  `EqualityConstrainedFunction` / `InequalityConstrainedFunction`)
- Bayesian optimization (with `BayesianSurrogate`, plus an
  `RBFGaussianProcessSurrogate` implementation)

### Population-based methods

Alongside the existing `GeneticAlgorithm`, `PSO`, `DifferentialEvolution`, and
`CMA-ES`:

- Ant colony optimization for continuous domains (`ACOR`)
- Artificial bee colony (`ArtificialBeeColony`)
- Estimation of distribution / EMNA-style full-covariance Gaussian
  (`EstimationOfDistribution`)
- Harmony search (`HarmonySearch`)
- Classic self-adaptive (mu,lambda)/(mu+lambda) evolution strategy
  (`EvolutionStrategy`) -- the pre-CMA-ES lineage, single global step size,
  no covariance adaptation
- Cross-entropy method (`CrossEntropyMethod`) -- diagonal covariance,
  exponentially-smoothed update
- Parallel tempering / replica exchange (`ParallelTempering`) -- the
  population-based generalization of `SimulatedAnnealing`, same
  no-bounds/no-tolerance interface
- Grey wolf optimizer (`GreyWolfOptimizer`), cuckoo search (`CuckooSearch`),
  whale optimization algorithm (`WhaleOptimization`), and firefly algorithm
  (`FireflyAlgorithm`) -- included for breadth; this family of post-2008
  "nature-inspired" metaheuristics has been criticized in the optimization
  research community (Sorensen 2015; Camacho-Villalon, Dorigo & Stutzle) as
  offering little genuine novelty over established mechanisms like PSO/DE,
  but each is implemented faithfully to its original paper's formulas

## Next: current interfaces are sufficient

## Requires a new function interface

None currently identified -- every function-type interface this module's
methods need (`ArbitraryFunction`, `DifferentiableFunction`,
`DifferentiableSeparableFunction`, `ProximalFunction`, `HessianFunction`,
`EqualityConstrainedFunction`, `InequalityConstrainedFunction`,
`ResidualFunction`, `BayesianSurrogate`) already exists.

Each new method should retain the existing contract: core C++ implementation,
an umbrella-header export, deterministic options where randomness is involved,
C++ tests, and SWIG interface inclusion for binding regeneration. Note: as of
this writing, 10 of the pre-population-batch additions (FISTA,
ProximalGradient, LevenbergMarquardt, Newton, TrustRegionNewton,
AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization,
RBFGaussianProcessSurrogate) are C++-only and not yet wired into the 15
language `.i` files -- a known, not-yet-closed gap, tracked separately from
the population-based batch above (which does have full SWIG wiring).
