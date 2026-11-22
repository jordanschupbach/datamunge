# Stochastic search implementation and MEpML plan

Inventory checked against the C++ headers, source, tests and MEpML tree on
2026-11-21. The principal scope is derivative-free stochastic search over
continuous, discrete and mixed domains. Multi-objective, simulation-based
and budgeted hyperparameter search have their own batches. Stochastic gradient
optimizers and deterministic search methods are outside this inventory.

**Implementation status** describes the current checkout. **Report status**
links each method to a dedicated MEpML report or a runnable comparison.
The inventory includes both newly added optimizers and existing graph, RL,
NN and Bayesian APIs. The main native entry point is `<dm/dm.hpp>`.

## A. Existing continuous solvers and report gaps

Existing boxed solvers already use an objective callback, mutate coordinates and return the
value at the returned point.

| Method | Implementation status | Current MEpML report status | Next work |
| --- | --- | --- | --- |
| Particle swarm optimization (PSO) | `optim::PSO`, global and ring topologies, constant and linear-decay inertia | [Runnable comparison](mepml/optim/stochastic-search/population/particle_swarm.mepml) | Verified HTML and Org |
| Simulated annealing (SA) | `optim::SimulatedAnnealing`, Gaussian proposals, geometric cooling | [Runnable comparison](mepml/optim/stochastic-search/annealing/simulated_annealing.mepml) | Verified HTML and Org |
| Differential evolution (DE) | `optim::DifferentialEvolution`, rand/1, best/1, current-to-best/1, rand/2, binomial and exponential crossover | [Runnable comparison](mepml/optim/stochastic-search/population/differential_evolution.mepml) | Verified HTML and Org |
| Real-valued genetic algorithm (GA) | `optim::GeneticAlgorithm`, tournament, roulette or rank selection, single-point, uniform or blend crossover | [Runnable comparison](mepml/optim/stochastic-search/population/genetic_algorithm.mepml) | Verified HTML and Org |
| Covariance matrix adaptation evolution strategy (CMA-ES) | `optim::CMAES` | [Runnable comparison](mepml/optim/stochastic-search/population/cma_es.mepml) | Verified HTML and Org |
| Classical evolution strategy | `optim::EvolutionStrategy`, comma and plus selection | [Runnable comparison](mepml/optim/stochastic-search/population/evolution_strategy.mepml) | Verified HTML and Org |
| Cross-entropy method (CEM) | `optim::CrossEntropyMethod` | [Runnable comparison](mepml/optim/stochastic-search/population/cross_entropy.mepml) | Verified HTML and Org |
| Gaussian estimation of distribution algorithm (EDA) | `optim::EstimationOfDistribution` | [Runnable comparison](mepml/optim/stochastic-search/population/gaussian_eda.mepml) | Verified HTML and Org |
| Continuous ant colony optimization (ACOR) | `optim::ACOR`, archive-based Gaussian kernels | [Runnable comparison](mepml/optim/stochastic-search/population/acor.mepml) | Verified HTML and Org |
| Parallel tempering | `optim::ParallelTempering` | [Runnable comparison](mepml/optim/stochastic-search/annealing/parallel_tempering.mepml) | Verified HTML and Org |
| Artificial bee colony (ABC) | Native `optim` implementation | [Runnable comparison](mepml/optim/stochastic-search/swarm/artificial_bee_colony.mepml) | Verified HTML and Org |
| Cuckoo search | Native `optim` implementation | [Runnable comparison](mepml/optim/stochastic-search/swarm/cuckoo_search.mepml) | Verified HTML and Org |
| Firefly algorithm | Native `optim` implementation | [Runnable comparison](mepml/optim/stochastic-search/swarm/firefly.mepml) | Verified HTML and Org |
| Grey wolf optimizer | Native `optim` implementation | [Runnable comparison](mepml/optim/stochastic-search/swarm/grey_wolf.mepml) | Verified HTML and Org |
| Harmony search | Native `optim` implementation | [Runnable comparison](mepml/optim/stochastic-search/swarm/harmony_search.mepml) | Verified HTML and Org |
| Whale optimization algorithm | Native `optim` implementation | [Runnable comparison](mepml/optim/stochastic-search/swarm/whale_optimization.mepml) | Verified HTML and Org |
| Gaussian-process Bayesian optimization with expected improvement | `optim::BayesianOptimization` and `RBFGaussianProcessSurrogate` | [Dedicated existing report](mepml/optim/optimization_bayesian.mepml) | Existing EI and lower-confidence-bound comparison |

## B. Continuous baselines, random walks and hybrid local search

| Method | Implementation status | MEpML report status |
| --- | --- | --- |
| Uniform independent random search | `optim::RandomSearch`, uniform design | [Runnable comparison](mepml/optim/stochastic-search/baselines/random_designs.mepml) |
| Randomized Latin hypercube search | `optim::RandomSearch`, jittered strata | [Runnable comparison](mepml/optim/stochastic-search/baselines/random_designs.mepml) |
| Scrambled Sobol search | `optim::RandomSearch`, binary digital shift, up to 16 dimensions | [Runnable comparison](mepml/optim/stochastic-search/baselines/random_designs.mepml) |
| Randomized Halton search | `optim::RandomSearch`, uniform shifts modulo one | [Runnable comparison](mepml/optim/stochastic-search/baselines/random_designs.mepml) |
| Stochastic hill climbing | `optim::StochasticLocalSearch`, fixed scale | [Runnable comparison](mepml/optim/stochastic-search/baselines/stochastic_local.mepml) |
| Random-restart hill climbing | Native `algorithms::random_restart_hill_climbing`, with a loss-to-score adapter example | [Runnable comparison](mepml/optim/stochastic-search/baselines/optimizer_adapters.mepml) |
| Random-restart local minimization | `optim::HybridRandomSearch`, multistart with Solis–Wets | [Runnable comparison](mepml/optim/stochastic-search/baselines/hybrid_local.mepml) |
| Solis–Wets adaptive random search | `optim::SolisWets`, box-projected paired proposals | [Runnable comparison](mepml/optim/stochastic-search/baselines/solis_wets.mepml) |
| Luus–Jaakola shrinking-region search | `optim::StochasticLocalSearch`, shrinking rectangle | [Runnable comparison](mepml/optim/stochastic-search/baselines/stochastic_local.mepml) |
| Adaptive step-size `(1+1)` evolution strategy | `optim::StochasticLocalSearch`, one-fifth success rule | [Runnable comparison](mepml/optim/stochastic-search/baselines/stochastic_local.mepml) |
| Basin hopping | `optim::HybridRandomSearch`, local Solis–Wets with Metropolis acceptance | [Runnable comparison](mepml/optim/stochastic-search/baselines/hybrid_local.mepml) |
| Multilevel single linkage (MLSL) | `optim::HybridRandomSearch`, normalized single-linkage filtering | [Runnable comparison](mepml/optim/stochastic-search/baselines/hybrid_local.mepml) |
| Stochastic tunneling | `optim::AnnealingSearch`, transformed-loss acceptance | [Runnable comparison](mepml/optim/stochastic-search/annealing/annealing_variants.mepml) |
| Tsallis generalized simulated annealing | `optim::AnnealingSearch`, generalized visiting and acceptance | [Runnable comparison](mepml/optim/stochastic-search/annealing/annealing_variants.mepml) |
| Very fast simulated reannealing (VFSA) | `optim::AnnealingSearch`, coordinate cooling and counted sensitivity probes | [Runnable comparison](mepml/optim/stochastic-search/annealing/annealing_variants.mepml) |
| Annealing with reheating and restarts | `optim::AnnealingSearch`, stagnation-triggered reheating and optional restarts | [Runnable comparison](mepml/optim/stochastic-search/annealing/annealing_variants.mepml) |
| Equi-energy search | `optim::EquiEnergySearch`, finite-history ring jumps for optimization | [Runnable comparison](mepml/optim/stochastic-search/annealing/equi_energy.mepml) |
| Alopex correlation-based search | Native `algorithms::alopex_minimize`, with a common loss callback example | [Runnable comparison](mepml/optim/stochastic-search/baselines/optimizer_adapters.mepml) |

## C. Discrete, permutation and neighborhood search

| Method | Implementation status | MEpML report status |
| --- | --- | --- |
| Ant System (AS) | `optim::ant_colony_tour`, all-ant reinforcement | [Runnable comparison](mepml/optim/stochastic-search/discrete/ant_colony.mepml) |
| Ant Colony System (ACS) | `optim::ant_colony_tour`, greedy choice and local/global updates | [Runnable comparison](mepml/optim/stochastic-search/discrete/ant_colony.mepml) |
| MAX–MIN Ant System (MMAS) | `optim::ant_colony_tour`, trail bounds and restarts | [Runnable comparison](mepml/optim/stochastic-search/discrete/ant_colony.mepml) |
| Rank-based Ant System | `optim::ant_colony_tour`, ranked reinforcements | [Runnable comparison](mepml/optim/stochastic-search/discrete/ant_colony.mepml) |
| Elitist Ant System | `optim::ant_colony_tour`, elite reinforcement | [Runnable comparison](mepml/optim/stochastic-search/discrete/ant_colony.mepml) |
| Discrete simulated annealing | `optim::neighborhood_search`, Metropolis acceptance | [Runnable comparison](mepml/optim/stochastic-search/discrete/acceptance_rules.mepml) |
| Discrete replica exchange | `optim::discrete_replica_exchange` | [Runnable comparison](mepml/optim/stochastic-search/discrete/replica_exchange.mepml) |
| Tabu search with aspiration | `optim::tabu_search`, typed inverse-move keys | [Runnable comparison](mepml/optim/stochastic-search/discrete/tabu_and_opt_moves.mepml) |
| Reactive tabu search | `optim::tabu_search`, recurrence-driven tenure | [Runnable comparison](mepml/optim/stochastic-search/discrete/tabu_and_opt_moves.mepml) |
| Iterated local search (ILS) | `optim::iterated_local_search` | [Runnable comparison](mepml/optim/stochastic-search/discrete/variable_neighborhoods.mepml) |
| Variable neighborhood descent (VND) | `optim::variable_neighborhood_search`, descent mode with optional shuffled order | [Runnable comparison](mepml/optim/stochastic-search/discrete/variable_neighborhoods.mepml) |
| Basic variable neighborhood search (VNS) | `optim::variable_neighborhood_search`, basic mode | [Runnable comparison](mepml/optim/stochastic-search/discrete/variable_neighborhoods.mepml) |
| Reduced variable neighborhood search (RVNS) | `optim::variable_neighborhood_search`, reduced mode | [Runnable comparison](mepml/optim/stochastic-search/discrete/variable_neighborhoods.mepml) |
| Guided local search (GLS) | `optim::guided_local_search` | [Runnable comparison](mepml/optim/stochastic-search/discrete/guided_local_search.mepml) |
| Late acceptance hill climbing (LAHC) | `optim::neighborhood_search`, late acceptance | [Runnable comparison](mepml/optim/stochastic-search/discrete/acceptance_rules.mepml) |
| Record-to-record travel | `optim::neighborhood_search`, record-relative acceptance | [Runnable comparison](mepml/optim/stochastic-search/discrete/acceptance_rules.mepml) |
| Threshold accepting | `optim::neighborhood_search`, current-relative acceptance | [Runnable comparison](mepml/optim/stochastic-search/discrete/acceptance_rules.mepml) |
| Great deluge and extended great deluge | `optim::neighborhood_search`, fixed decrement and target-scheduled variants | [Runnable comparison](mepml/optim/stochastic-search/discrete/acceptance_rules.mepml) |
| Iterated greedy / ruin and recreate | `optim::large_neighborhood_search`, destroy and reconstruct | [Runnable comparison](mepml/optim/stochastic-search/discrete/large_neighborhood.mepml) |
| Large neighborhood search (LNS) | `optim::large_neighborhood_search` | [Runnable comparison](mepml/optim/stochastic-search/discrete/large_neighborhood.mepml) |
| Adaptive large neighborhood search (ALNS) | `optim::large_neighborhood_search`, separate adaptive destroy and repair weights | [Runnable comparison](mepml/optim/stochastic-search/discrete/large_neighborhood.mepml) |
| GRASP | `optim::grasp`, typed construction and local descent | [Runnable comparison](mepml/optim/stochastic-search/discrete/grasp_and_path_relink.mepml) |
| Reactive GRASP | `optim::grasp`, selection of greediness by past outcomes | [Runnable comparison](mepml/optim/stochastic-search/discrete/grasp_and_path_relink.mepml) |
| GRASP with path relinking | `optim::grasp`, elite pool and typed relinking | [Runnable comparison](mepml/optim/stochastic-search/discrete/grasp_and_path_relink.mepml) |
| Scatter search and path relinking | `optim::scatter_search` and `optim::path_relink` | [Runnable comparison](mepml/optim/stochastic-search/discrete/grasp_and_path_relink.mepml) |
| Memetic algorithm | Native `algorithms::memetic_max_cut`, which couples genetic operators with graph-specific local improvement | [Dedicated report](mepml/algorithm/optimization/memetic.mepml) |
| Stochastic 2-opt and 3-opt | `optim::sample_two_opt`, `optim::sample_three_opt` and typed applicators with the neighborhood driver | [Runnable comparison](mepml/optim/stochastic-search/discrete/tabu_and_opt_moves.mepml) |
| Iterated Lin–Kernighan | `optim::iterated_lin_kernighan`, bounded variable-depth exchanges and tour kicks | [Runnable comparison](mepml/optim/stochastic-search/discrete/iterated_lin_kernighan.mepml) |

## D. Population, evolutionary and distribution variants

| Method | Implementation status | MEpML report status |
| --- | --- | --- |
| Binary genetic algorithm | `optim::discrete_genetic_search`, binary encoding | [Runnable comparison](mepml/optim/stochastic-search/discrete/distribution_search.mepml) |
| Permutation genetic algorithm | `optim::discrete_genetic_search`, order crossover and swap or inversion | [Runnable comparison](mepml/optim/stochastic-search/discrete/genetic_tour_search.mepml) |
| Steady-state genetic algorithm | `optim::discrete_genetic_search`, steady-state replacement | [Runnable comparison](mepml/optim/stochastic-search/discrete/genetic_tour_search.mepml) |
| Island-model GA or DE | `optim::discrete_genetic_search`, GA islands with ring migration | [Runnable comparison](mepml/optim/stochastic-search/discrete/genetic_tour_search.mepml) |
| Binary PSO | `optim::binary_particle_swarm` | [Runnable comparison](mepml/optim/stochastic-search/discrete/distribution_search.mepml) |
| Bare-bones PSO | `optim::ParticleSwarmSearch`, Gaussian positions | [Runnable comparison](mepml/optim/stochastic-search/population/particle_swarm_variants.mepml) |
| Fully informed PSO (FIPS) | `optim::ParticleSwarmSearch`, neighborhood-wide guidance | [Runnable comparison](mepml/optim/stochastic-search/population/particle_swarm_variants.mepml) |
| Comprehensive learning PSO (CLPSO) | `optim::ParticleSwarmSearch`, coordinate-specific learned guides | [Runnable comparison](mepml/optim/stochastic-search/population/particle_swarm_variants.mepml) |
| Adaptive differential evolution (jDE) | `optim::AdaptiveDifferentialEvolution`, jDE | [Runnable comparison](mepml/optim/stochastic-search/population/adaptive_de.mepml) |
| Strategy adaptation DE (SaDE) | `optim::AdaptiveDifferentialEvolution`, SaDE | [Runnable comparison](mepml/optim/stochastic-search/population/adaptive_de.mepml) |
| JADE | `optim::AdaptiveDifferentialEvolution`, JADE | [Runnable comparison](mepml/optim/stochastic-search/population/adaptive_de.mepml) |
| SHADE | `optim::AdaptiveDifferentialEvolution`, SHADE | [Runnable comparison](mepml/optim/stochastic-search/population/adaptive_de.mepml) |
| L-SHADE | `optim::AdaptiveDifferentialEvolution`, L-SHADE | [Runnable comparison](mepml/optim/stochastic-search/population/adaptive_de.mepml) |
| Separable CMA-ES | `optim::CMASearch`, diagonal covariance | [Runnable comparison](mepml/optim/stochastic-search/population/cma_es_variants.mepml) |
| IPOP-CMA-ES | `optim::CMASearch`, increasing population restarts | [Runnable comparison](mepml/optim/stochastic-search/population/cma_es_variants.mepml) |
| BIPOP-CMA-ES | `optim::CMASearch`, alternating large and small restart regimes | [Runnable comparison](mepml/optim/stochastic-search/population/cma_es_variants.mepml) |
| Natural evolution strategies (xNES and SNES) | `optim::NaturalEvolutionSearch`, xNES and SNES | [Runnable comparison](mepml/optim/stochastic-search/population/natural_evolution.mepml) |
| OpenAI evolution strategy | `optim::NaturalEvolutionSearch`, antithetic OpenAI ES | [Runnable comparison](mepml/optim/stochastic-search/population/natural_evolution.mepml) |
| Augmented random search (ARS) | `optim::NaturalEvolutionSearch`, ARS | [Runnable comparison](mepml/optim/stochastic-search/population/natural_evolution.mepml) |
| Evolutionary programming | `optim::EvolutionaryProgramming`, self-adapting mutation and survivor tournaments | [Runnable comparison](mepml/optim/stochastic-search/population/evolutionary_programming.mepml) |
| Population-based incremental learning (PBIL) | `optim::discrete_distribution_search`, PBIL | [Runnable comparison](mepml/optim/stochastic-search/discrete/distribution_search.mepml) |
| Univariate marginal distribution algorithm (UMDA) | `optim::discrete_distribution_search`, UMDA | [Runnable comparison](mepml/optim/stochastic-search/discrete/distribution_search.mepml) |
| Compact genetic algorithm (cGA) | `optim::discrete_distribution_search`, cGA | [Runnable comparison](mepml/optim/stochastic-search/discrete/distribution_search.mepml) |
| Mutual-information maximizing input clustering (MIMIC) | `optim::discrete_distribution_search`, MIMIC | [Runnable comparison](mepml/optim/stochastic-search/discrete/distribution_search.mepml) |
| Bayesian optimization algorithm (BOA, discrete EDA) | `optim::discrete_distribution_search`, sparse Bayesian network | [Runnable comparison](mepml/optim/stochastic-search/discrete/distribution_search.mepml) |
| Linkage tree genetic algorithm (LTGA) | `optim::linkage_search`, paired-parent linkage mixing | [Runnable comparison](mepml/optim/stochastic-search/discrete/distribution_search.mepml) |
| Gene-pool optimal mixing EA (GOMEA) | `optim::linkage_search`, gene-pool donors | [Runnable comparison](mepml/optim/stochastic-search/discrete/distribution_search.mepml) |
| Discrete cross-entropy search | `optim::discrete_distribution_search`, product categorical model | [Runnable comparison](mepml/optim/stochastic-search/discrete/distribution_search.mepml) |

## E. Further swarm and nature-inspired methods

These follow the classical core and should each demonstrate a clearly
specified proposal rule on the same evaluation-budget controls used by the
earlier batches.

| Method | Implementation status | MEpML report status |
| --- | --- | --- |
| Bacterial foraging optimization | `optim::CollectiveSearch`, `bacterial_foraging` | [Runnable comparison](mepml/optim/stochastic-search/swarm/bacterial_foraging.mepml) |
| Bacterial chemotaxis search | `optim::CollectiveSearch`, `bacterial_chemotaxis` | [Runnable comparison](mepml/optim/stochastic-search/swarm/bacterial_chemotaxis.mepml) |
| Artificial fish swarm algorithm | `optim::CollectiveSearch`, `artificial_fish` | [Runnable comparison](mepml/optim/stochastic-search/swarm/artificial_fish.mepml) |
| Bat algorithm | `optim::CollectiveSearch`, `bat` | [Runnable comparison](mepml/optim/stochastic-search/swarm/bat.mepml) |
| Gravitational search algorithm | `optim::CollectiveSearch`, `gravitational` | [Runnable comparison](mepml/optim/stochastic-search/swarm/gravitational.mepml) |
| Krill herd algorithm | `optim::CollectiveSearch`, `krill_herd` | [Runnable comparison](mepml/optim/stochastic-search/swarm/krill_herd.mepml) |
| Biogeography-based optimization | `optim::CollectiveSearch`, `biogeography` | [Runnable comparison](mepml/optim/stochastic-search/swarm/biogeography.mepml) |
| Shuffled frog leaping | `optim::CollectiveSearch`, `shuffled_frog` | [Runnable comparison](mepml/optim/stochastic-search/swarm/shuffled_frog.mepml) |
| Invasive weed optimization | `optim::CollectiveSearch`, `invasive_weed` | [Runnable comparison](mepml/optim/stochastic-search/swarm/invasive_weed.mepml) |
| Flower pollination algorithm | `optim::CollectiveSearch`, `flower_pollination` | [Runnable comparison](mepml/optim/stochastic-search/swarm/flower_pollination.mepml) |
| Teaching-learning-based optimization | `optim::CollectiveSearch`, `teaching_learning` | [Runnable comparison](mepml/optim/stochastic-search/swarm/teaching_learning.mepml) |
| Imperialist competitive algorithm | `optim::CollectiveSearch`, `imperialist` | [Runnable comparison](mepml/optim/stochastic-search/swarm/imperialist.mepml) |
| Clonal selection (CLONALG) | `optim::CollectiveSearch`, `clonal_selection` | [Runnable comparison](mepml/optim/stochastic-search/swarm/clonal_selection.mepml) |
| Opt-aiNet artificial immune network | `optim::CollectiveSearch`, `artificial_immune_network` | [Runnable comparison](mepml/optim/stochastic-search/swarm/artificial_immune_network.mepml) |
| Stochastic fractal search | `optim::CollectiveSearch`, `stochastic_fractal` | [Runnable comparison](mepml/optim/stochastic-search/swarm/stochastic_fractal.mepml) |
| Charged system search | `optim::CollectiveSearch`, `charged_system` | [Runnable comparison](mepml/optim/stochastic-search/swarm/charged_system.mepml) |
| Salp swarm algorithm | `optim::CollectiveSearch`, `salp_swarm` | [Runnable comparison](mepml/optim/stochastic-search/swarm/salp_swarm.mepml) |

## F. Multi-objective, diversity and structured search

| Method | Implementation status | MEpML report status |
| --- | --- | --- |
| NSGA-II | Native `optim::NSGA2` | [Dedicated](mepml/optimization/multi-objective/nsga2.mepml) |
| MOEA/D | Native `optim::MOEAD` | [Dedicated](mepml/optimization/multi-objective/moead.mepml) |
| SPEA2 | Native `optim::SPEA2` | [Dedicated](mepml/optimisation/multi-objective/spea2.mepml) |
| NSGA-III | `optim::ParetoSearch`, reference-direction niching | [Runnable comparison](mepml/optim/stochastic-search/pareto/reference_and_indicator.mepml) |
| Pareto archived evolution strategy (PAES) | `optim::ParetoSearch`, PAES | [Runnable comparison](mepml/optim/stochastic-search/pareto/grid_archive.mepml) |
| Pareto envelope-based selection algorithm (PESA-II) | `optim::ParetoSearch`, PESA-II | [Runnable comparison](mepml/optim/stochastic-search/pareto/grid_archive.mepml) |
| Multi-objective PSO (MOPSO) | `optim::ParetoSearch`, MOPSO | [Runnable comparison](mepml/optim/stochastic-search/pareto/grid_archive.mepml) |
| Generalized DE (GDE3) | `optim::ParetoSearch`, GDE3 | [Runnable comparison](mepml/optim/stochastic-search/pareto/grid_archive.mepml) |
| Indicator-based EA (IBEA) | `optim::ParetoSearch`, additive epsilon indicator | [Runnable comparison](mepml/optim/stochastic-search/pareto/reference_and_indicator.mepml) |
| Hypervolume-estimation algorithm (HypE) | `optim::ParetoSearch`, sampled hypervolume contributions | [Runnable comparison](mepml/optim/stochastic-search/pareto/reference_and_indicator.mepml) |
| S-metric selection evolutionary multi-objective algorithm (SMS-EMOA) | `optim::ParetoSearch`, exact two-objective hypervolume selection | [Runnable comparison](mepml/optim/stochastic-search/pareto/reference_and_indicator.mepml) |
| Strength Pareto or Pareto-ranked discrete ACO | `optim::pareto_ant_colony_tour`, archive-ranked tour reinforcement | [Runnable comparison](mepml/optim/stochastic-search/pareto/pareto_ant_colony.mepml) |
| MAP-Elites | `optim::map_elites_search`, typed MAP-Elites search | [Runnable comparison](mepml/optim/stochastic-search/pareto/quality_diversity.mepml) |
| Novelty search | `optim::novelty_search`, descriptor-distance archive | [Runnable comparison](mepml/optim/stochastic-search/pareto/quality_diversity.mepml) |
| Novelty search with local competition | `optim::novelty_search`, local competition | [Runnable comparison](mepml/optim/stochastic-search/pareto/quality_diversity.mepml) |
| NEAT | Native specialized `nn` implementation | [Dedicated report](mepml/nn/6_deep_reinforcement_learning/6.12_evolutionary_and_black_box_alternatives/neat.mepml) |
| Tree-based genetic programming | `optim::genetic_programming_search`, expression trees | [Runnable comparison](mepml/optim/stochastic-search/pareto/genetic_programming.mepml) |
| Cartesian genetic programming | `optim::genetic_programming_search`, acyclic graphs with neutral nodes | [Runnable comparison](mepml/optim/stochastic-search/pareto/genetic_programming.mepml) |
| Gene expression programming | `optim::genetic_programming_search`, head-tail chromosomes | [Runnable comparison](mepml/optim/stochastic-search/pareto/genetic_programming.mepml) |

## G. Noisy, surrogate-assisted and resource-aware search

| Method | Implementation status | MEpML report status |
| --- | --- | --- |
| Simultaneous perturbation stochastic approximation (SPSA) | `optim::StochasticApproximation`, Bernoulli perturbations | [Runnable comparison](mepml/optim/stochastic-search/baselines/stochastic_approximation.mepml) |
| Random-direction finite-difference search (RDSA) | `optim::StochasticApproximation`, spherical perturbations | [Runnable comparison](mepml/optim/stochastic-search/baselines/stochastic_approximation.mepml) |
| Kim–Nelson fully sequential ranking and selection | `optim::simulation_search`, paired sequential elimination | [Runnable comparison](mepml/optim/stochastic-search/resource/simulation_racing.mepml) |
| Optimal computing budget allocation (OCBA) | `optim::simulation_search`, variance and gap-based allocation | [Runnable comparison](mepml/optim/stochastic-search/resource/simulation_racing.mepml) |
| Confidence-bound racing / F-race | `optim::simulation_search`, confidence-bound and Friedman elimination | [Runnable comparison](mepml/optim/stochastic-search/resource/simulation_racing.mepml) |
| Iterated racing | `optim::iterated_racing`, repeated configuration proposals and races | [Runnable comparison](mepml/optim/stochastic-search/resource/simulation_racing.mepml) |
| Bayesian optimization with GP lower confidence bound | `optim::BayesianOptimization` with lower-confidence-bound acquisition | [Dedicated report](mepml/optim/optimization_bayesian.mepml) |
| Thompson-sampling Bayesian optimization | `optim::joint_bayesian_search`, correlated posterior draw | [Runnable comparison](mepml/optim/stochastic-search/resource/joint_bayesian_search.mepml) |
| Noisy expected improvement | `optim::joint_bayesian_search`, joint latent sampling and repeat observations | [Runnable comparison](mepml/optim/stochastic-search/resource/joint_bayesian_search.mepml) |
| Tree-structured Parzen estimator (TPE) | `optim::mixed_search`, conditional Parzen densities | [Runnable comparison](mepml/optim/stochastic-search/resource/mixed_models.mepml) |
| Sequential model-based algorithm configuration (SMAC) | `optim::mixed_search`, random-forest acquisition | [Runnable comparison](mepml/optim/stochastic-search/resource/mixed_models.mepml) |
| Surrogate-assisted evolutionary search | `optim::mixed_search`, surrogate-only proposal evolution | [Runnable comparison](mepml/optim/stochastic-search/resource/mixed_models.mepml) |
| Successive halving | `optim::resource_search`, successive halving | [Runnable comparison](mepml/optim/stochastic-search/resource/trial_schedulers.mepml) |
| Hyperband | `optim::resource_search`, Hyperband brackets | [Runnable comparison](mepml/optim/stochastic-search/resource/trial_schedulers.mepml) |
| Asynchronous successive halving (ASHA) | `optim::resource_search`, completion-order simulation | [Runnable comparison](mepml/optim/stochastic-search/resource/trial_schedulers.mepml) |
| BOHB | `optim::bohb_search`, conditional models by fidelity and Hyperband brackets | [Runnable comparison](mepml/optim/stochastic-search/resource/mixed_models.mepml) |
| Population-based training (PBT) | `optim::population_based_training`, checkpoint inheritance | [Runnable comparison](mepml/optim/stochastic-search/resource/trial_schedulers.mepml) |

## Delivery order and API design

1. **Continuous report backlog and baselines.** Write dedicated PSO,
   annealing, ACOR, DE, GA and CMA-ES reports first. Add uniform random
   search, randomized designs and Solis–Wets, then document the remaining
   implemented solvers from section A.
2. **Discrete search.** Implement the shared state, neighborhood and budget
   primitives with a small permutation example. Deliver Ant System, ACS,
   MMAS, tabu search, discrete SA, ILS, VNS and GRASP integration first.
   Add destroy/repair, path relinking and adaptive variants after their
   primitive operators have tests.
3. **Continuous hybrids and variants.** Add multistart, basin hopping and
   the ES, DE, PSO and EDA variants where differences affect actual search.
   Reuse selection, sampling and replacement code. Expose minor choices as
   options when they share the same algorithm loop.
4. **Pareto and archive search.** Extend the existing multi-objective
   interface with NSGA-III, PAES, MOPSO and the other archive and indicator
   methods. Preserve the dedicated graph, RL and NN entry points.
5. **Noisy, mixed and budgeted search.** Add replication and fidelity to
   the evaluation model before implementing simulation racing or mixed
   hyperparameter methods. Finish the remaining swarm and structured
   search backlog against the same benchmark and report checks.

### Integration contract

- Keep `dm::optim` as the continuous entry point. Use the existing
  `ArbitraryFunction`, method-specific options and boxed
  `optimize(objective, coordinates, lower, upper)` convention. Preserve
  existing overloads, return values and stopping behavior.
- Reuse `trace_optimize`, `OptimizationTrace` and `OptimizationRun`
  where they capture real objective calls. Introduce explicit evaluation
  caps and stop reasons in new entry points only as needed. Count
  initialization, restarts and nested local search against the same budget.
- Use typed state and moves for permutations, categorical values and
  other discrete states. Use explicit admissible choices, state scoring
  and move callbacks, with documented feasibility and minimization
  conventions. Use a type-safe adapter where a specialized solver already
  exists. Model multi-objective search with `MultiObjectiveFunction`
  and `ParetoPoint`.
- Seed local random engines through public options. Document replay scope,
  bound handling, constraints, candidate validity, callback exceptions and
  how duplicate evaluations are counted. Keep tracing opt-in and bounded.
- Review SWIG exposure only after the native contract is settled. Keep the
  existing bindings and umbrella entry points working and add normal
  binding tests for any newly exposed wrapper.

### Completion checks per method

- Build the real native entry point and test seeded replay, exact work
  counts, best-so-far consistency, and method-specific transitions. Cover
  invalid inputs and budgets that end during initialization or a generation.
  Use known answers or enumerated small instances where practical.
- Create one cached, runnable MEpML report per distinct method. Closely
  related options may share a report with a named section and runnable
  comparison. Put new reports under
  `docs/mepml/optim/stochastic-search/<family>/` and link to existing
  dedicated reports where verified.
- Use objective evaluations as the horizontal axis for comparisons.
  Predeclare multiple seeds, publish every result and include variability.
  Use multimodal and rotated continuous surfaces, and exact small tours or
  graph cuts for the discrete methods. Attribute algorithm rules and
  describe what the implementation actually supports.
- Build report figures with the native plotting or diagram API. Save them
  to the report build directory and reference the generated figure files.
  Run cached examples fresh, verify HTML and Org exports and regenerate
  the report catalogue before marking a report done.

## Verification

- The release C++ library built successfully. The focused suite passed 74
  new optimizer tests and 122 existing optimizer regression tests.
- All 66 new MEpML reports ran their examples and exported to HTML and Org
  with native figures. The report catalogue was regenerated.
- The Doxygen audit covered 1,230 definitions. All 179 public example blocks
  in the new optimizer headers and the Gaussian-process surrogate compiled
  and ran with the expected output.
- The additions are exposed in the native C++ umbrella. The new optimizer
  sources are listed in the package manifest. Language bindings for the
  new entry points remain a separate API review.

## References for shared method families

- Independent and adaptive random search. [Solis and Wets (1981)](https://doi.org/10.1287/moor.6.1.19)
- Discrete ant colony optimization. [Dorigo, Maniezzo and Colorni (1996)](https://doi.org/10.1109/3477.484436)
- Tabu search and adaptive memory. [Glover (1989)](https://doi.org/10.1287/ijoc.1.3.190)
- Variable neighborhood search. [Hansen and Mladenović (2001)](https://www.sciencedirect.com/science/article/pii/S0377221700001004)
- Scatter search and path relinking. [Glover, Laguna and Martí (2004)](https://www.sciencedirect.com/science/article/pii/S0377221704005430)
