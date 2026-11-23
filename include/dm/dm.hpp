#pragma once

/// @file dm.hpp
/// @brief Single C++ entry point for the library's public modules under the dm namespace.
/// Includes the high-level API, bundled datasets, algorithms, linear algebra,
/// neural networks, vectors, and plot composition, plus the standard headers most analyses use
/// (`<iostream>`, `<string>`, `<vector>`, `<filesystem>`, `<cmath>`, `<format>`, and
/// `<print>` under C++23).
/// Link the usual datamunge library; dm names refer to the same types and functions as
/// datamunge names. Works with C++20; the std::format formatters for library types are added
/// automatically under C++23.

#include <datamunge/datamunge.hpp>
#include <datamunge/random/sampling_analysis.hpp>
#include <datamunge/random/poisson_binomial.hpp>
#include <datamunge/random/count_distributions.hpp>
#include <datamunge/random/simplex.hpp>
#include <datamunge/random/matrix_distributions.hpp>
#include <datamunge/random/noncentral_distributions.hpp>
#include <datamunge/algorithms/algorithms.hpp>
#include <datamunge/algorithms/learned_index.hpp>
#include <datamunge/datasets/datasets.hpp>
#include <datamunge/info.hpp>
#include <datamunge/dir.hpp>
#include <datamunge/pipe.hpp>
#include <datamunge/linalg/linalg.hpp>
#include <datamunge/nn/nn.hpp>
#include <datamunge/midi.hpp>
#include <datamunge/rl/discrete.hpp>
#include <datamunge/compute/context.hpp>
#include <datamunge/compute/nvidia_telemetry.hpp>
#include <datamunge/sim/world.hpp>
#include <datamunge/sim/batch.hpp>
#include <datamunge/sim/tasks.hpp>
#include <datamunge/sim/conveyor.hpp>
#include <datamunge/sim/devarm_pick.hpp>
#include <datamunge/sim/devarm_camera.hpp>
#include <datamunge/sim/fruit_stream.hpp>
#include <datamunge/sim/articulation.hpp>
#include <datamunge/game/clock.hpp>
#include <datamunge/game/simulation.hpp>
#include <datamunge/game/io.hpp>
#include <datamunge/game/courier.hpp>
#include <datamunge/game/render.hpp>
#include <datamunge/cad/cad.hpp>
#include <datamunge/fda/adaptive_pspline.hpp>

// Public analysis APIs used by the literate reports.
#include <datamunge/algorithms/dfa_minimization_analysis.hpp>
#include <datamunge/algorithms/fnv_analysis.hpp>
#include <datamunge/algorithms/golomb_analysis.hpp>
#include <datamunge/algorithms/gram_schmidt_analysis.hpp>
#include <datamunge/algorithms/graph_coloring_analysis.hpp>
#include <datamunge/algorithms/grasp_analysis.hpp>
#include <datamunge/algorithms/grover_analysis.hpp>
#include <datamunge/algorithms/hamming_code_analysis.hpp>
#include <datamunge/algorithms/heapsort.hpp>
#include <datamunge/algorithms/histogram_equalization_analysis.hpp>
#include <datamunge/algorithms/hmac_analysis.hpp>
#include <datamunge/algorithms/hopcroft_karp_analysis.hpp>
#include <datamunge/algorithms/hopfield_analysis.hpp>
#include <datamunge/algorithms/huffman_analysis.hpp>
#include <datamunge/algorithms/hungarian_analysis.hpp>
#include <datamunge/algorithms/index_calculus_analysis.hpp>
#include <datamunge/algorithms/interpolation_search_analysis.hpp>
#include <datamunge/algorithms/inverse_iteration_analysis.hpp>
#include <datamunge/algorithms/iterative_deepening_analysis.hpp>
#include <datamunge/algorithms/itp_analysis.hpp>
#include <datamunge/algorithms/jaro_winkler_analysis.hpp>
#include <datamunge/algorithms/johnson_analysis.hpp>
#include <datamunge/algorithms/kabsch_analysis.hpp>
#include <datamunge/algorithms/lanczos_analysis.hpp>
#include <datamunge/algorithms/logitboost_study.hpp>
#include <datamunge/algorithms/lsh_study.hpp>
#include <datamunge/algorithms/merge_sort.hpp>
#include <datamunge/algorithms/quickselect.hpp>
#include <datamunge/algorithms/quicksort.hpp>
#include <datamunge/algorithms/radix_sort.hpp>
#include <datamunge/algorithms/rice_coding.hpp>
#include <datamunge/algorithms/structured_svm_analysis.hpp>
#include <datamunge/algorithms/union_merge_analysis.hpp>
#include <datamunge/filter/kalman_analysis.hpp>
#include <datamunge/geometry/gjk.hpp>
#include <datamunge/geometry/graham_analysis.hpp>
#include <datamunge/geometry/icp_analysis.hpp>
#include <datamunge/geometry/jarvis_analysis.hpp>
#include <datamunge/geometry/jump_walk_analysis.hpp>
#include <datamunge/geometry/kdtree_analysis.hpp>
#include <datamunge/geometry/laplacian_analysis.hpp>
#include <datamunge/nn/cifar10_study.hpp>
#include <datamunge/nn/classification_analysis.hpp>
#include <datamunge/nn/image_overlap.hpp>
#include <datamunge/nn/tabular_classification.hpp>
#include <datamunge/plot/grayscale_images.hpp>
#include <datamunge/random/lagged_fibonacci_analysis.hpp>
#include <datamunge/random/linear_congruential_analysis.hpp>
#include <datamunge/random/sfc64_analysis.hpp>
#include <datamunge/stats/embedding_diagnostics.hpp>
#include <datamunge/stats/jackknife_analysis.hpp>

// Standard headers an analysis script almost always needs, so `#include <dm/dm.hpp>` is the only
// include a short program or report block has to write.
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#if __has_include(<format>)
#include <format>
#endif
#if __has_include(<print>) && __cplusplus > 202002L
#include <print>
#endif

// The std::formatter specializations need C++23 (std::formattable and range formatting), so
// they are only pulled in when the standard library provides those; C++20 builds still get
// everything else.
#if __has_include(<version>)
#include <version>
#endif
#if defined(__cpp_lib_format_ranges)
#include <datamunge/algorithms/insertion_sort_analysis.hpp>
#include <datamunge/format.hpp>
#endif

#include <dm/data.hpp>

#include <datamunge/rl/environments.hpp>

#include <datamunge/rl/tabular.hpp>
#include <datamunge/rl/count_exploration.hpp>
#include <datamunge/rl/tile_coding.hpp>
#include <datamunge/rl/bandit_selection.hpp>
#include <datamunge/rl/environments/sparse_rooms.hpp>
#include <datamunge/rl/tensor_components.hpp>
#include <datamunge/rl/policy_search.hpp>
#include <datamunge/rl/map_elites.hpp>
#include <datamunge/rl/intrinsic_exploration.hpp>
#include <datamunge/rl/curiosity_model.hpp>
#include <datamunge/rl/episodic_novelty.hpp>
#include <datamunge/rl/bootstrapped_dqn.hpp>
#include <datamunge/rl/go_explore.hpp>
#include <datamunge/rl/hindsight_replay.hpp>
#include <datamunge/rl/uvfa.hpp>
#include <datamunge/rl/option_critic.hpp>
#include <datamunge/rl/hierarchical_replay.hpp>
#include <datamunge/rl/distral.hpp>
#include <datamunge/rl/feudal.hpp>
#include <datamunge/rl/imitation_policy.hpp>
#include <datamunge/rl/dagger.hpp>
#include <datamunge/rl/conservative_q.hpp>
#include <datamunge/rl/discrete_bcq.hpp>
#include <datamunge/rl/implicit_q.hpp>
#include <datamunge/rl/policy_occupancy.hpp>
#include <datamunge/rl/maxent_irl.hpp>
#include <datamunge/rl/td3_bc.hpp>
#include <datamunge/rl/decision_transformer.hpp>
#include <datamunge/rl/trajectory_beam.hpp>
#include <datamunge/rl/independent_q.hpp>
#include <datamunge/rl/value_decomposition.hpp>
#include <datamunge/rl/commnet.hpp>
#include <datamunge/rl/board_policy.hpp>
#include <datamunge/rl/maddpg.hpp>
#include <datamunge/rl/mappo.hpp>
#include <datamunge/rl/preference_objectives.hpp>
#include <datamunge/rl/rl_squared.hpp>
#include <datamunge/rl/maml_bandit.hpp>
#include <datamunge/rl/pearl.hpp>
#include <datamunge/nn/membership_audit.hpp>
#include <datamunge/nn/feature_schema.hpp>
#include <datamunge/nn/patch_robustness.hpp>
#include <datamunge/nn/linear_prototypes.hpp>
#include <datamunge/nn/projected_robustness.hpp>
#include <datamunge/nn/decision_robustness.hpp>
#include <datamunge/nn/attentive_seq2seq.hpp>
#include <datamunge/nn/image_mixers.hpp>
#include <datamunge/nn/pyramid_digit_transformer.hpp>
#include <datamunge/nn/swin_digit_transformer.hpp>
#include <datamunge/nn/coatnet_digits.hpp>
#include <datamunge/nn/perceiver_digits.hpp>
#include <datamunge/nn/cuda_pixel_q.hpp>

#include <datamunge/rl/replay.hpp>

#include <datamunge/rl/returns.hpp>
#include <datamunge/rl/environments/normalized_mountain_car.hpp>

#include <datamunge/rl/continuous_actions.hpp>
#include <datamunge/rl/sequence_replay.hpp>
#include <datamunge/rl/environments/masked_cartpole.hpp>
#include <datamunge/rl/off_policy.hpp>

#include <datamunge/rl/mcts.hpp>
#include <datamunge/rl/transition_data.hpp>
#include <datamunge/rl/environments/planning_tasks.hpp>

#include <datamunge/rl/environments/lunar_lander.hpp>

#include <datamunge/rl/environments/atari_frames.hpp>
#include <datamunge/rl/atari_replay.hpp>

#include <datamunge/rl/environments/physics_tasks.hpp>

#include <datamunge/rl/bootstrapped_returns.hpp>

#if defined(DATAMUNGE_ENABLE_VSIM_TASKS)
#include <datamunge/rl/environments/vsim_tasks.hpp>
#endif
