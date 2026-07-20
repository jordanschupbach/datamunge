#pragma once

// Convenience umbrella header -- includes the whole Bayesian inference module: a
// distribution log-density library, an autodiff-to-DifferentiableFunction adapter, and
// MAP/HMC/NUTS/RWM/Gibbs/ImportanceSampling/INLA inference algorithms.

#include <datamunge/bayes/autodiff_model.hpp>
#include <datamunge/bayes/distributions.hpp>
#include <datamunge/bayes/gibbs.hpp>
#include <datamunge/bayes/hmc.hpp>
#include <datamunge/bayes/importance_sampling.hpp>
#include <datamunge/bayes/inla.hpp>
#include <datamunge/bayes/map.hpp>
#include <datamunge/bayes/nuts.hpp>
#include <datamunge/bayes/rwm.hpp>
