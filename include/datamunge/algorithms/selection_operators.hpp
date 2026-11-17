/// @file selection_operators.hpp
/// @brief Compatibility aggregation of checked roulette, SUS and tournament producers.
#pragma once
#include <datamunge/algorithms/selection_rng.hpp>
#include <datamunge/algorithms/stochastic_universal_sampling.hpp>
#include <datamunge/algorithms/tournament_selection.hpp>

// The classic selection operators of evolutionary computation: given the fitness
// of each individual in a population, choose which ones "reproduce". Selection is
// what turns random variation into directed search -- it sets the *selection
// pressure*, the bias toward fitter individuals. Three standard schemes:
//
//   * Fitness-proportionate (roulette-wheel) selection -- probability of choosing
//     an individual is its fitness divided by the total, as if spinning a roulette
//     wheel whose slot sizes are the fitnesses.
//   * Stochastic universal sampling (SUS) -- one spin with N equally-spaced
//     pointers picks all N parents at once, giving the same expected counts as
//     roulette but far less variance (no individual is over- or under-picked by
//     more than one).
//   * Tournament selection -- repeatedly draw k random individuals and keep the
//     fittest; the tournament size k tunes the selection pressure directly.
//
// SUS/roulette require nonnegative weights; tournaments require only finite ordered fitness.

#include <cstddef>
#include <cstdint>
#include <datamunge/algorithms/roulette_wheel.hpp>
#include <vector>
