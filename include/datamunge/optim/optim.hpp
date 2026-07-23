#pragma once

// Convenience umbrella header -- includes the full function-type taxonomy plus every
// optimizer built on top of it.

#include <datamunge/optim/acor.hpp>
#include <datamunge/optim/adam.hpp>
#include <datamunge/optim/adagrad.hpp>
#include <datamunge/optim/adadelta.hpp>
#include <datamunge/optim/amsgrad.hpp>
#include <datamunge/optim/artificial_bee_colony.hpp>
#include <datamunge/optim/augmented_lagrangian.hpp>
#include <datamunge/optim/bayesian_optimization.hpp>
#include <datamunge/optim/conjugate_gradient.hpp>
#include <datamunge/optim/cma_es.hpp>
#include <datamunge/optim/coordinate_descent.hpp>
#include <datamunge/optim/cross_entropy_method.hpp>
#include <datamunge/optim/cuckoo_search.hpp>
#include <datamunge/optim/differential_evolution.hpp>
#include <datamunge/optim/estimation_of_distribution.hpp>
#include <datamunge/optim/evolution_strategy.hpp>
#include <datamunge/optim/firefly_algorithm.hpp>
#include <datamunge/optim/function_types.hpp>
#include <datamunge/optim/fista.hpp>
#include <datamunge/optim/genetic_algorithm.hpp>
#include <datamunge/optim/gradient_descent.hpp>
#include <datamunge/optim/grey_wolf_optimizer.hpp>
#include <datamunge/optim/harmony_search.hpp>
#include <datamunge/optim/interior_point.hpp>
#include <datamunge/optim/lbfgs.hpp>
#include <datamunge/optim/levenberg_marquardt.hpp>
#include <datamunge/optim/nelder_mead.hpp>
#include <datamunge/optim/nesterov_accelerated_gradient.hpp>
#include <datamunge/optim/nadam.hpp>
#include <datamunge/optim/newton.hpp>
#include <datamunge/optim/parallel_tempering.hpp>
#include <datamunge/optim/pso.hpp>
#include <datamunge/optim/proximal_gradient.hpp>
#include <datamunge/optim/randomized_block_coordinate_descent.hpp>
#include <datamunge/optim/rmsprop.hpp>
#include <datamunge/optim/rbf_gaussian_process_surrogate.hpp>
#include <datamunge/optim/sgd.hpp>
#include <datamunge/optim/simulated_annealing.hpp>
#include <datamunge/optim/saga.hpp>
#include <datamunge/optim/svrg.hpp>
#include <datamunge/optim/sqp.hpp>
#include <datamunge/optim/trust_region_newton.hpp>
#include <datamunge/optim/whale_optimization.hpp>
