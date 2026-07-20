#pragma once

// Convenience umbrella header -- includes the whole state-estimation/filtering module: the
// linear Kalman filter (plus its RTS smoother), the Extended and Unscented Kalman filters for
// nonlinear models, the Information filter (KalmanFilter's algebraic dual), the Ensemble
// Kalman filter and a bootstrap particle filter for high-dimensional/non-Gaussian nonlinear
// problems, and the simple constant-gain alpha-beta / alpha-beta-gamma trackers.

#include <datamunge/filter/alpha_beta_filter.hpp>
#include <datamunge/filter/ensemble_kalman_filter.hpp>
#include <datamunge/filter/extended_kalman_filter.hpp>
#include <datamunge/filter/information_filter.hpp>
#include <datamunge/filter/kalman_filter.hpp>
#include <datamunge/filter/particle_filter.hpp>
#include <datamunge/filter/unscented_kalman_filter.hpp>
#include <datamunge/filter/vector_function.hpp>
