#pragma once

// Convenience umbrella header -- includes the full function-type taxonomy plus every
// optimizer built on top of it.

#include <datamunge/optim/adam.hpp>
#include <datamunge/optim/differential_evolution.hpp>
#include <datamunge/optim/function_types.hpp>
#include <datamunge/optim/genetic_algorithm.hpp>
#include <datamunge/optim/gradient_descent.hpp>
#include <datamunge/optim/lbfgs.hpp>
#include <datamunge/optim/pso.hpp>
#include <datamunge/optim/sgd.hpp>
#include <datamunge/optim/simulated_annealing.hpp>
