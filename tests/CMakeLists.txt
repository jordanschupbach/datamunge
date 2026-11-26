cmake_minimum_required(VERSION 3.14...3.22)

project(DatamungeTests LANGUAGES C CXX)

# ---- Options ----

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)

# --- Import tools ----

# include(../cmake/tools.cmake)

# ---- Dependencies ----

include(../cmake/CPM.cmake)

CPMAddPackage(NAME datamunge SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# Prefer a preinstalled GTest (offline-friendly), then fall back to fetching.
find_package(GTest QUIET CONFIG)
if(NOT GTest_FOUND)
  find_package(GTest QUIET)
endif()
if(NOT GTest_FOUND)
  CPMAddPackage(
    NAME googletest
    GITHUB_REPOSITORY google/googletest
    VERSION 1.17.0
    OPTIONS "INSTALL_GTEST OFF" "gtest_force_shared_crt ON"
  )
endif()

# if(TEST_INSTALLED_VERSION) find_package(datamunge REQUIRED) else() CPMAddPackage(NAME datamunge
# SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..) endif()

enable_testing()
add_executable(run_nn_autoencoder_models_tests cpp/nn_autoencoder_models_tests.cpp cpp/nn_competitive_autoencoders_tests.cpp cpp/nn_masked_autoencoder_tests.cpp cpp/nn_sequence_autoencoders_tests.cpp cpp/nn_activation_sparse_autoencoder_tests.cpp cpp/nn_layered_autoencoders_tests.cpp cpp/nn_tied_autoencoder_tests.cpp cpp/nn_adversarial_autoencoder_tests.cpp cpp/nn_gaussian_autoencoder_tests.cpp cpp/nn_conditional_autoencoder_tests.cpp cpp/nn_convolutional_vae_tests.cpp cpp/nn_total_correlation_tests.cpp cpp/nn_flow_vae_tests.cpp cpp/nn_gumbel_vae_tests.cpp cpp/nn_hierarchical_vae_tests.cpp cpp/nn_importance_weighted_tests.cpp cpp/nn_info_vae_tests.cpp cpp/nn_ladder_vae_tests.cpp cpp/nn_sentence_vae_tests.cpp cpp/nn_sequence_vae_data_tests.cpp cpp/nn_vae_gan_tests.cpp cpp/nn_vector_quantized_autoencoder_tests.cpp cpp/nn_vector_quantized_autoencoder_2_tests.cpp cpp/nn_wasserstein_autoencoder_tests.cpp cpp/nn_patch_restoration_tests.cpp cpp/nn_encoder_pretraining_tests.cpp cpp/nn_latent_arithmetic_tests.cpp cpp/nn_latent_visualization_tests.cpp cpp/nn_learned_image_codec_tests.cpp cpp/nn_reconstruction_anomaly_tests.cpp cpp/nn_auxiliary_classifier_gan_tests.cpp cpp/nn_bidirectional_gan_tests.cpp cpp/nn_dense_gan_tests.cpp cpp/nn_info_gan_tests.cpp cpp/nn_spectral_gan_tests.cpp cpp/nn_self_attention_gan_tests.cpp cpp/nn_big_gan_tests.cpp cpp/nn_dcgan_tests.cpp cpp/nn_wasserstein_gan_tests.cpp cpp/nn_progressive_gan_tests.cpp cpp/nn_pix2pix_tests.cpp cpp/nn_cycle_gan_tests.cpp cpp/nn_style_gan_tests.cpp
    cpp/nn_star_gan_tests.cpp
    cpp/nn_adversarial_sr_tests.cpp cpp/nn_single_image_gan_tests.cpp)
target_link_libraries(run_nn_autoencoder_models_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_autoencoder_models_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
include(GoogleTest)
gtest_discover_tests(run_nn_autoencoder_models_tests TEST_PREFIX "nn-autoencoder.")
add_executable(run_nn_frozen_graph_tests cpp/nn_frozen_graph_tests.cpp)
target_link_libraries(run_nn_frozen_graph_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_frozen_graph_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_frozen_graph_tests TEST_PREFIX "nn-frozen-graph.")
add_executable(run_nn_generative_metrics_tests cpp/nn_generative_metrics_tests.cpp)
target_link_libraries(run_nn_generative_metrics_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_generative_metrics_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_generative_metrics_tests TEST_PREFIX "nn-generative-metric.")

add_executable(run_nn_autoregressive_pixels_tests cpp/nn_autoregressive_pixels_tests.cpp)
target_link_libraries(run_nn_autoregressive_pixels_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_autoregressive_pixels_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_autoregressive_pixels_tests TEST_PREFIX "nn-autoregressive-pixel.")
add_executable(run_nn_gated_pixels_tests cpp/nn_gated_pixels_tests.cpp)
target_link_libraries(run_nn_gated_pixels_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_gated_pixels_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_gated_pixels_tests TEST_PREFIX "nn-gated-pixel.")
add_executable(run_nn_recurrent_pixels_tests cpp/nn_recurrent_pixels_tests.cpp)
target_link_libraries(run_nn_recurrent_pixels_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_recurrent_pixels_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_recurrent_pixels_tests TEST_PREFIX "nn-recurrent-pixel.")
add_executable(run_nn_made_density_tests cpp/nn_made_density_tests.cpp)
target_link_libraries(run_nn_made_density_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_made_density_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_made_density_tests TEST_PREFIX "nn-made-density.")
add_executable(run_informed_search_tests cpp/algorithms_informed_search_tests.cpp)
target_link_libraries(run_informed_search_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_informed_search_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
include(GoogleTest)
gtest_discover_tests(run_informed_search_tests TEST_PREFIX "informed-search.")
add_executable(run_bilinear_tests cpp/algorithms_bilinear_tests.cpp cpp/algorithms_interpolation_extra_tests.cpp
  cpp/interpolation_batch39_tests.cpp)
target_link_libraries(run_bilinear_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bilinear_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bilinear_tests TEST_PREFIX "bilinear.")
add_executable(run_binary_gcd_trace_tests cpp/algorithms_binary_gcd_trace_tests.cpp
  cpp/algorithms_number_theory_tests.cpp)
target_link_libraries(run_binary_gcd_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_binary_gcd_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_binary_gcd_trace_tests TEST_PREFIX "binary-gcd.")
add_executable(run_binary_search_trace_tests cpp/algorithms_binary_search_trace_tests.cpp
  cpp/algorithms_sorted_search_tests.cpp)
target_link_libraries(run_binary_search_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_binary_search_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_binary_search_trace_tests TEST_PREFIX "binary-search.")
add_executable(run_binary_splitting_tests cpp/algorithms_binary_splitting_tests.cpp
  cpp/algorithms_elementary_more_tests.cpp)
target_link_libraries(run_binary_splitting_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_binary_splitting_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_binary_splitting_tests TEST_PREFIX "binary-splitting.")
add_executable(run_birkhoff_tests cpp/algorithms_birkhoff_tests.cpp cpp/interpolation_batch39_tests.cpp)
target_link_libraries(run_birkhoff_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_birkhoff_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_birkhoff_tests TEST_PREFIX "birkhoff.")
add_executable(run_bisection_tests cpp/algorithms_bisection_trace_tests.cpp cpp/algorithms_root_finding_tests.cpp)
target_link_libraries(run_bisection_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bisection_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bisection_tests TEST_PREFIX "bisection.")
add_executable(run_bitap_tests cpp/algorithms_bitap_trace_tests.cpp cpp/algorithms_merge_ternary_bitap_tests.cpp)
target_link_libraries(run_bitap_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bitap_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bitap_tests TEST_PREFIX "bitap.")
add_executable(run_bitonic_tests cpp/algorithms_bitonic_network_tests.cpp cpp/algorithms_sorting_extra_tests.cpp)
target_link_libraries(run_bitonic_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bitonic_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bitonic_tests TEST_PREFIX "bitonic.")
add_executable(run_bkm_tests cpp/algorithms_bkm_trace_tests.cpp cpp/geometry_numeric_batch_tests.cpp)
target_link_libraries(run_bkm_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bkm_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bkm_tests TEST_PREFIX "bkm.")
add_executable(run_blast_tests cpp/algorithms_blast_trace_tests.cpp cpp/algorithms_bioinformatics_batch_tests.cpp)
target_link_libraries(run_blast_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_blast_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_blast_tests TEST_PREFIX "blast.")
add_executable(run_bloom_tests cpp/algorithms_bloom_packed_tests.cpp cpp/algorithms_bioinformatics_batch_tests.cpp)
target_link_libraries(run_bloom_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bloom_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bloom_tests TEST_PREFIX "bloom.")
add_executable(run_dir_tests cpp/dir_tests.cpp)
target_link_libraries(run_dir_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_dir_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_dir_tests TEST_PREFIX "dir.")
add_executable(run_blossom_tests cpp/algorithms_blossom_trace_tests.cpp cpp/algorithms_blossom_tests.cpp)
target_link_libraries(run_blossom_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_blossom_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_blossom_tests TEST_PREFIX "blossom.")
add_executable(run_bbs_tests cpp/random_bbs_analysis_tests.cpp cpp/datamunge_random_tests.cpp)
target_link_libraries(run_bbs_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bbs_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bbs_tests TEST_PREFIX "bbs.")
add_executable(run_booth_tests cpp/algorithms_booth_trace_tests.cpp cpp/algorithms_multiplication_tests.cpp)
target_link_libraries(run_booth_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_booth_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_booth_tests TEST_PREFIX "booth.")
add_executable(run_boruvka_tests cpp/algorithms_boruvka_trace_tests.cpp cpp/algorithms_routing_batch_tests.cpp)
target_link_libraries(run_boruvka_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_boruvka_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_boruvka_tests TEST_PREFIX "boruvka.")
add_executable(run_boyer_moore_tests cpp/algorithms_boyer_moore_trace_tests.cpp cpp/algorithms_boyer_moore_tests.cpp)
target_link_libraries(run_boyer_moore_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_boyer_moore_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_boyer_moore_tests TEST_PREFIX "boyer-moore.")
add_executable(run_horspool_tests cpp/algorithms_horspool_trace_tests.cpp cpp/algorithms_string_search_extra_tests.cpp cpp/algorithms_zhu_takaoka_tests.cpp cpp/pipe_native_reduction_tests.cpp cpp/pipe_tests.cpp)
target_link_libraries(run_horspool_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_horspool_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_horspool_tests TEST_PREFIX "horspool.")
add_executable(run_branch_bound_tests cpp/algorithms_knapsack_trace_tests.cpp cpp/algorithms_combinatorial_search_tests.cpp)
target_link_libraries(run_branch_bound_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_branch_bound_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_branch_bound_tests TEST_PREFIX "branch-bound.")
add_executable(run_bfs_tests cpp/algorithms_bfs_trace_tests.cpp cpp/algorithms_search_batch_tests.cpp)
target_link_libraries(run_bfs_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bfs_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bfs_tests TEST_PREFIX "bfs.")
# Ported windowless CAD regression suites retain their independent numerical,
# topology, round-trip and constraint-solving oracles.
file(GLOB cad_regression_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/cpp/cad/cad_*_test.cpp")
foreach(cad_test_source IN LISTS cad_regression_sources)
  get_filename_component(cad_test_name "${cad_test_source}" NAME_WE)
  add_executable("run_${cad_test_name}" "${cad_test_source}")
  target_include_directories("run_${cad_test_name}" PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../src/datamunge/cad")
  target_compile_definitions("run_${cad_test_name}" PRIVATE DATAMUNGE_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
  target_link_libraries("run_${cad_test_name}" datamunge::datamunge)
  set_target_properties("run_${cad_test_name}" PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
  add_test(NAME "cad.${cad_test_name}" COMMAND "run_${cad_test_name}")
endforeach()
# Algebra is header-only: this focused runner needs no native shared-library rebuild.
add_executable(run_algebra_tex_tests cpp/algebra_tex_tests.cpp cpp/algebra_expression_tests.cpp
  cpp/algebra_calculus_tests.cpp cpp/algebra_polynomial_core_tests.cpp
  cpp/algebra_roots_and_factorization_tests.cpp cpp/algebra_multivariate_groebner_tests.cpp)
target_include_directories(run_algebra_tex_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_link_libraries(run_algebra_tex_tests GTest::gtest_main)
set_target_properties(run_algebra_tex_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
include(GoogleTest)
gtest_discover_tests(run_algebra_tex_tests TEST_PREFIX "algebra-tex.")
add_executable(run_tex_tests cpp/tex_tests.cpp)
target_link_libraries(run_tex_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_tex_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
include(GoogleTest)
gtest_discover_tests(run_tex_tests TEST_PREFIX "tex.")
# Native concept diagrams have an offline runner independent of dataset/model fitting.
add_executable(run_diagram_tests cpp/diagram_tests.cpp)
target_link_libraries(run_diagram_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_diagram_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
include(GoogleTest)
gtest_discover_tests(run_diagram_tests TEST_PREFIX "diagram.")
add_executable(run_filesystem_tests cpp/filesystem_tests.cpp)
target_link_libraries(run_filesystem_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_filesystem_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_filesystem_tests TEST_PREFIX "filesystem.")
add_executable(run_plot_color_tests cpp/plot_tests.cpp cpp/plot_compose_tests.cpp cpp/ggplot_tests.cpp cpp/dm_api_tests.cpp)
target_link_libraries(run_plot_color_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_plot_color_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_plot_color_tests TEST_PREFIX "plot-colors.")
add_executable(run_bootstrap_tests cpp/stats_bootstrap_summary_tests.cpp cpp/stats_resampling_tests.cpp
  cpp/stats_bootstrap_designs_tests.cpp cpp/stats_discovery_inference_tests.cpp
  cpp/stats_restricted_randomization_tests.cpp)
target_link_libraries(run_bootstrap_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bootstrap_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bootstrap_tests TEST_PREFIX "bootstrap.")

add_executable(run_conformal_tests cpp/stats_conformal_tests.cpp cpp/stats_conformal_calibration_tests.cpp)
target_link_libraries(run_conformal_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_conformal_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_conformal_tests TEST_PREFIX "conformal.")

add_executable(run_stats_group_comparison_tests cpp/stats_group_comparison_tests.cpp
  cpp/stats_robust_summary_tests.cpp cpp/stats_sample_moments_tests.cpp
  cpp/stats_robust_inference_tests.cpp cpp/stats_kernel_density_tests.cpp cpp/stats_agreement_tests.cpp
  cpp/stats_anova_measures_tests.cpp cpp/stats_table_symmetry_tests.cpp cpp/stats_delong_tests.cpp
  cpp/random_simplex_tests.cpp cpp/random_matrix_distribution_tests.cpp cpp/stats_power_transform_tests.cpp
  cpp/stats_energy_tests.cpp cpp/stats_nonparametric_tests.cpp cpp/stats_edf_tests.cpp
  cpp/random_noncentral_tests.cpp cpp/stats_power_tests.cpp cpp/stats_standardized_effects_tests.cpp
  cpp/stats_line_regression_tests.cpp cpp/stats_lm_covariance_tests.cpp
  cpp/stats_robust_regression_tests.cpp cpp/stats_lm_specification_tests.cpp cpp/stats_gls_tests.cpp
  cpp/stats_lm_marginal_means_tests.cpp cpp/stats_lm_marginal_anova_tests.cpp
  cpp/stats_profile_likelihood_tests.cpp cpp/stats_nonlinear_least_squares_tests.cpp
  cpp/stats_segmented_regression_tests.cpp cpp/stats_glm_binomial_tests.cpp
  cpp/stats_firth_logistic_tests.cpp cpp/stats_conditional_logistic_tests.cpp
  cpp/stats_multinomial_logistic_tests.cpp cpp/stats_ordinal_regression_tests.cpp
  cpp/stats_negative_binomial_regression_tests.cpp cpp/stats_zero_augmented_count_tests.cpp
  cpp/stats_beta_regression_tests.cpp cpp/stats_tweedie_regression_tests.cpp cpp/stats_gam_tests.cpp
  cpp/stats_isotonic_regression_tests.cpp cpp/stats_group_lasso_tests.cpp cpp/stats_fused_lasso_tests.cpp
  cpp/stats_variance_components_tests.cpp cpp/stats_likelihood_glmm_tests.cpp cpp/stats_mixed_model_inference_tests.cpp
  cpp/stats_gee_tests.cpp
  cpp/stats_mmrm_tests.cpp
  cpp/stats_repeated_measures_anova_tests.cpp
  cpp/stats_panel_fixed_effects_tests.cpp
  cpp/stats_hausman_tests.cpp
  cpp/stats_arellano_bond_tests.cpp
  cpp/stats_hotelling_tests.cpp
  cpp/stats_manova_tests.cpp
  cpp/stats_canonical_correlation_tests.cpp
  cpp/stats_simpls_tests.cpp
  cpp/stats_regularized_discriminant_tests.cpp
  cpp/stats_covariance_shrinkage_tests.cpp
  cpp/stats_minimum_covariance_determinant_tests.cpp
  cpp/stats_graphical_lasso_tests.cpp
  cpp/stats_multivariate_normality_tests.cpp
  cpp/stats_cfa_tests.cpp
  cpp/stats_oblique_rotation_tests.cpp
  cpp/stats_latent_class_tests.cpp
  cpp/stats_gaussian_mixture_selection_tests.cpp
  cpp/stats_functional_pca_tests.cpp
  cpp/stats_scalar_on_function_tests.cpp
  cpp/stats_spectral_clustering_tests.cpp
  cpp/stats_hdbscan_tests.cpp
  cpp/stats_mean_shift_tests.cpp
  cpp/stats_birch_tests.cpp
  cpp/stats_silhouette_tests.cpp
  cpp/stats_gap_statistic_tests.cpp
  cpp/stats_consensus_clustering_tests.cpp
  cpp/stats_nested_cross_validation_tests.cpp
  cpp/stats_stability_selection_tests.cpp
  cpp/stats_knockoffs_tests.cpp
  cpp/stats_hodges_lehmann_tests.cpp
  cpp/stats_mvn_probability_tests.cpp
  cpp/stats_maximum_likelihood_tests.cpp
  cpp/stats_likelihood_tests_tests.cpp
  cpp/stats_tolerance_limits_tests.cpp
  cpp/stats_dunnett_tests.cpp
  cpp/stats_dip_test_tests.cpp
  cpp/stats_equivalence_tests.cpp
  cpp/stats_factorial_design_tests.cpp
  cpp/stats_response_surface_tests.cpp
  cpp/stats_d_optimal_tests.cpp
  cpp/stats_group_sequential_tests.cpp
  cpp/stats_arima_exact_tests.cpp
  cpp/stats_arima_regression_tests.cpp
  cpp/stats_unit_root_tests.cpp
  cpp/stats_stl_tests.cpp
  cpp/stats_var_tests.cpp
  cpp/stats_cointegration_tests.cpp
  cpp/stats_garch_tests.cpp
  cpp/stats_structural_tests.cpp
  cpp/stats_spectral_tests.cpp
  cpp/stats_lomb_scargle_tests.cpp
  cpp/stats_changepoint_tests.cpp
  cpp/stats_markov_switching_tests.cpp
  cpp/stats_diebold_mariano_tests.cpp
  cpp/stats_nelson_aalen_tests.cpp
  cpp/stats_survival_tests_tests.cpp
  cpp/stats_rmst_tests.cpp
  cpp/stats_cox_counting_tests.cpp
  cpp/stats_cox_strata_tests.cpp
  cpp/stats_cox_zph_tests.cpp
  cpp/stats_aft_tests.cpp
  cpp/stats_turnbull_tests.cpp
  cpp/stats_multistate_tests.cpp
  cpp/stats_gray_tests.cpp
  cpp/stats_fine_gray_tests.cpp
  cpp/stats_cox_cluster_tests.cpp
  cpp/stats_survival_accuracy_tests.cpp
  cpp/stats_survey_tests.cpp
  cpp/stats_survey_calibration_tests.cpp
  cpp/stats_pps_sampling_tests.cpp
  cpp/stats_survey_tables_tests.cpp
  cpp/stats_survey_glm_tests.cpp
  cpp/stats_missing_data_tests.cpp
  cpp/stats_multiple_imputation_tests.cpp)
target_link_libraries(run_stats_group_comparison_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_group_comparison_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_group_comparison_tests TEST_PREFIX "group-comparison.")

add_executable(run_bayes_predictive_tests cpp/bayes_predictive_tests.cpp cpp/bayes_slice_sampling_tests.cpp
  cpp/bayes_evidence_tests.cpp)
target_link_libraries(run_bayes_predictive_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bayes_predictive_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bayes_predictive_tests TEST_PREFIX "bayes-predictive.")

add_executable(run_vector_expression_tests cpp/vector_expression_tests.cpp cpp/vector_tests.cpp)
target_link_libraries(run_vector_expression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_vector_expression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_vector_expression_tests TEST_PREFIX "vector-expression.")
add_executable(check_vector_expression_allocations vector_expression_allocations.cpp)
target_link_libraries(check_vector_expression_allocations datamunge::datamunge)
set_target_properties(check_vector_expression_allocations PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
add_test(NAME vector-expression.allocations COMMAND check_vector_expression_allocations)
# The facade's expression and solver contract plus existing canonical dense/sparse regressions.
add_executable(run_matrix_api_tests cpp/matrix_api_tests.cpp cpp/linalg_dense_tests.cpp
  cpp/linalg_sparse_coo_tests.cpp cpp/linalg_formats_tests.cpp cpp/linalg_solvers_tests.cpp
  cpp/linalg_svd_least_squares_tests.cpp cpp/format/linalg_format_tests.cpp)
target_link_libraries(run_matrix_api_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_matrix_api_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_matrix_api_tests TEST_PREFIX "matrix-api.")
add_executable(run_adaptive_pspline_tests cpp/fda_adaptive_pspline_tests.cpp)
target_link_libraries(run_adaptive_pspline_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_adaptive_pspline_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_adaptive_pspline_tests TEST_PREFIX "adaptive-pspline.")
add_executable(run_bspline_design_tests cpp/bspline_design_tests.cpp cpp/fda_bspline_tests.cpp cpp/stats_lm_tests.cpp)
target_link_libraries(run_bspline_design_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bspline_design_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bspline_design_tests TEST_PREFIX "bspline-design.")
add_executable(check_matrix_allocations matrix_allocations.cpp)
target_link_libraries(check_matrix_allocations datamunge::datamunge)
set_target_properties(check_matrix_allocations PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
add_test(NAME matrix-api.allocations COMMAND check_matrix_allocations)
add_executable(check_sparse_lm_allocations sparse_lm_allocations.cpp)
target_link_libraries(check_sparse_lm_allocations datamunge::datamunge)
set_target_properties(check_sparse_lm_allocations PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
add_test(NAME bspline-design.sparse-allocations COMMAND check_sparse_lm_allocations)
# ---- Create binary ----

file(GLOB cpp_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.cpp)
list(REMOVE_ITEM cpp_sources
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/algebra_tex_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/tex_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/cpp23_format_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/filesystem_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/stats_group_comparison_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/stats_robust_summary_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/stats_sample_moments_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/vector_expression_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/bspline_design_tests.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/matrix_api_tests.cpp)
file(GLOB c_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/*.c)
message(STATUS "c_sources: ${c_sources}")
message(STATUS "cpp_sources: ${cpp_sources}")
add_executable(run_tests ${c_sources} ${cpp_sources})
target_link_libraries(run_tests datamunge::datamunge GTest::gtest_main)
target_include_directories(run_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../prebindings/nn/include)
set_target_properties(run_tests PROPERTIES CXX_STANDARD 17)

include(GoogleTest)
gtest_discover_tests(run_tests)

# Focused engine runner keeps simulation iteration independent of the large NN test suite.
add_executable(run_simulation_tests
  cpp/rl_discrete_tests.cpp cpp/sim_articulated_tests.cpp cpp/sim_native_dynamics_tests.cpp
  cpp/compute_context_tests.cpp cpp/compute_gpu_tests.cpp cpp/sim_world_tests.cpp cpp/sim_conveyor_tests.cpp cpp/sim_devarm_pick_tests.cpp cpp/sim_devarm_camera_tests.cpp cpp/sim_fruit_stream_tests.cpp
  cpp/sim_batch_tests.cpp cpp/sim_tasks_tests.cpp cpp/game_simulation_tests.cpp cpp/game_io_tests.cpp
  cpp/game_courier_tests.cpp cpp/game_render_tests.cpp cpp/sim_rl_adapter_tests.cpp)
target_link_libraries(run_simulation_tests datamunge::datamunge GTest::gtest_main)
target_include_directories(run_simulation_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../prebindings/nn/include)
set_target_properties(run_simulation_tests PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_simulation_tests TEST_PREFIX "simulation.")
add_executable(check_simulation_allocations simulation_allocations.cpp)
target_link_libraries(check_simulation_allocations datamunge::datamunge)
set_target_properties(check_simulation_allocations PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
add_test(NAME simulation.allocations COMMAND check_simulation_allocations)

# Device-storage changes also exercise the existing tensor/gradient/memory/I/O contract.
add_executable(run_nn_device_tests
  cpp/nn_device_tests.cpp cpp/nn_tensor_tests.cpp cpp/nn_views_tests.cpp
  cpp/nn_autograd_tests.cpp cpp/nn_ops_tests.cpp cpp/nn_memory_tests.cpp cpp/nn_io_tests.cpp)
target_link_libraries(run_nn_device_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_device_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_device_tests TEST_PREFIX "nn-device.")

# Model diagrams also exercise the existing layer, parameter, derivative and tape contracts.
add_executable(run_nn_model_plot_tests
  cpp/nn_model_plot_tests.cpp cpp/nn_layers_tests.cpp cpp/nn_derivatives_tape_tests.cpp)
target_link_libraries(run_nn_model_plot_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_model_plot_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_model_plot_tests TEST_PREFIX "nn-model-plot.")

# Confusion SVG rendering shares the native training/metric contract.
add_executable(run_nn_metrics_plot_tests cpp/nn_metrics_plot_tests.cpp cpp/nn_training_tests.cpp)
target_link_libraries(run_nn_metrics_plot_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_metrics_plot_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_metrics_plot_tests TEST_PREFIX "nn-metrics-plot.")

# Proper scores and native CIFAR study helpers have small offline analytic fixtures.
add_executable(run_cifar10_study_tests cpp/nn_classification_analysis_tests.cpp)
target_link_libraries(run_cifar10_study_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cifar10_study_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cifar10_study_tests TEST_PREFIX "cifar10-study.")

# Formatting/printing is a C++20/23 feature; keep the other binding-contract tests on C++17.
# One file per module under cpp/format/ (each module's std::formatter specializations).
file(GLOB cpp23_format_sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/cpp/format/*.cpp)
add_executable(run_cpp23_tests cpp/cpp23_format_tests.cpp ${cpp23_format_sources})
target_link_libraries(run_cpp23_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cpp23_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cpp23_tests)

# ---- code coverage ----
if(ENABLE_TEST_COVERAGE)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    foreach(test_target IN ITEMS run_tests run_cpp23_tests)
      target_compile_options(${test_target} PRIVATE -O0 -g --coverage)
      target_link_options(${test_target} PRIVATE --coverage)
    endforeach()

    # Ensure the datamunge library (built via CPM in this build) is also instrumented.
    if(TARGET datamunge)
      target_compile_options(datamunge PRIVATE -O0 -g --coverage)
      target_link_options(datamunge PRIVATE --coverage)
    endif()
  else()
    message(WARNING "ENABLE_TEST_COVERAGE is ON but coverage flags are only wired for Clang/GNU.")
  endif()
endif()

# # enable compiler warnings if(NOT TEST_INSTALLED_VERSION) if(CMAKE_CXX_COMPILER_ID MATCHES "Clang"
# OR CMAKE_CXX_COMPILER_ID MATCHES "GNU") target_compile_options(datamunge PUBLIC -Wall -Wpedantic
# -Wextra -Werror) elseif(MSVC) target_compile_options(datamunge PUBLIC /W4 /WX)
# target_compile_definitions(${PROJECT_NAME} PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS) endif() endif()

# ---- Add DatamungeTests ----

# enable_testing()
#
# # Note: doctest and similar testing frameworks can automatically configure CMake tests. For other
# # testing frameworks add the tests target instead: add_test(NAME ${PROJECT_NAME} COMMAND #
# ${PROJECT_NAME})
#
# include(${doctest_SOURCE_DIR}/scripts/cmake/doctest.cmake) doctest_discover_tests(${PROJECT_NAME})
#
# # ---- code coverage ----
#
# if(ENABLE_TEST_COVERAGE) target_compile_options(datamunge PUBLIC -O0 -g -fprofile-arcs
# -ftest-coverage) target_link_options(datamunge PUBLIC -fprofile-arcs -ftest-coverage) endif()

if(DATAMUNGE_ENABLE_DATA_ASSETS)
  find_package(nlohmann_json 3.10 REQUIRED)
  add_executable(run_dataset_assets_tests dataset_assets.cpp array_io.cpp cpp/nn_io_tests.cpp)
  target_link_libraries(run_dataset_assets_tests PRIVATE datamunge::datamunge GTest::gtest_main nlohmann_json::nlohmann_json)
  set_target_properties(run_dataset_assets_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
  gtest_discover_tests(run_dataset_assets_tests)
endif()

# Small native MobileNet fitting/checkpoint contract, independent of the large NN suite.
add_executable(run_mobile_classifier_tests cpp/nn_mobile_classifier_tests.cpp)
target_link_libraries(run_mobile_classifier_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mobile_classifier_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mobile_classifier_tests TEST_PREFIX "mobile-classifier.")

add_executable(run_cycle_work_tests cpp/algorithms_cycle_work_tests.cpp cpp/algorithms_cycle_detection_tests.cpp)
target_link_libraries(run_cycle_work_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cycle_work_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cycle_work_tests TEST_PREFIX "cycle-work.")

add_executable(run_bresenham_tests cpp/algorithms_bresenham_trace_tests.cpp cpp/algorithms_compression_graphics_batch_tests.cpp)
target_link_libraries(run_bresenham_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bresenham_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bresenham_tests TEST_PREFIX "bresenham.")

add_executable(run_borwein_pi_tests cpp/algorithms_borwein_trace_tests.cpp cpp/arithmetic_batch40_tests.cpp)
target_link_libraries(run_borwein_pi_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_borwein_pi_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_borwein_pi_tests TEST_PREFIX "borwein-pi.")

add_executable(run_bron_kerbosch_tests cpp/algorithms_bron_kerbosch_trace_tests.cpp cpp/algorithms_graph_connectivity_tests.cpp)
target_link_libraries(run_bron_kerbosch_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bron_kerbosch_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bron_kerbosch_tests TEST_PREFIX "bron-kerbosch.")

add_executable(run_brute_force_tests cpp/algorithms_brute_force_trace_tests.cpp cpp/algorithms_search_more_tests.cpp)
target_link_libraries(run_brute_force_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_brute_force_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_brute_force_tests TEST_PREFIX "brute-force.")

add_executable(run_brzozowski_tests cpp/algorithms_brzozowski_trace_tests.cpp cpp/algorithms_logic_automata_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_brzozowski_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_brzozowski_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_brzozowski_tests TEST_PREFIX "brzozowski.")

add_executable(run_bucket_sort_tests cpp/algorithms_bucket_sort_trace_tests.cpp cpp/algorithms_distribution_sorts_tests.cpp)
target_link_libraries(run_bucket_sort_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bucket_sort_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bucket_sort_tests TEST_PREFIX "bucket-sort.")

add_executable(run_buddy_tests cpp/algorithms_buddy_trace_tests.cpp cpp/algorithms_memory_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_buddy_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_buddy_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_buddy_tests TEST_PREFIX "buddy.")
add_executable(check_buddy_allocations buddy_allocator_allocations.cpp)
target_link_libraries(check_buddy_allocations datamunge::datamunge)
set_target_properties(check_buddy_allocations PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
add_test(NAME buddy.MetadataAllocationFailurePreservesState COMMAND check_buddy_allocations)

add_executable(run_bully_tests cpp/algorithms_bully_trace_tests.cpp cpp/algorithms_distributed_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_bully_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bully_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bully_tests TEST_PREFIX "bully.")

add_executable(run_bwt_tests cpp/algorithms_bwt_trace_tests.cpp cpp/algorithms_compression2_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_bwt_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bwt_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bwt_tests TEST_PREFIX "bwt.")

add_executable(run_burstsort_tests cpp/algorithms_burstsort_trace_tests.cpp cpp/algorithms_sorting_dist_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_burstsort_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_burstsort_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_burstsort_tests TEST_PREFIX "burstsort.")
add_executable(check_burstsort_allocations burstsort_allocations.cpp)
target_link_libraries(check_burstsort_allocations datamunge::datamunge)
set_target_properties(check_burstsort_allocations PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
add_test(NAME burstsort.HostAllocationFailurePreservesInput COMMAND check_burstsort_allocations)

add_executable(run_c45_tests cpp/algorithms_c45_trace_tests.cpp cpp/algorithms_tree_rule_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_c45_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_c45_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_c45_tests TEST_PREFIX "c45.")

add_executable(run_cannon_tests cpp/algorithms_cannon_trace_tests.cpp cpp/algorithms_sparse_linalg_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_cannon_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cannon_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cannon_tests TEST_PREFIX "cannon.")

add_executable(run_canopy_tests cpp/algorithms_canopy_trace_tests.cpp cpp/algorithms_recurrent_pcnn_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_canopy_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_canopy_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_canopy_tests TEST_PREFIX "canopy.")

add_executable(run_cantor_tests cpp/algorithms_cantor_trace_tests.cpp cpp/algorithms_polynomial_gf_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_cantor_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cantor_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cantor_tests TEST_PREFIX "cantor.")

add_executable(run_chacha20_tests cpp/algorithms_chacha_trace_tests.cpp cpp/crypto_batch24_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_chacha20_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_chacha20_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_chacha20_tests TEST_PREFIX "chacha20.")

add_executable(run_chacha_engine_tests cpp/random_chacha20_analysis_tests.cpp cpp/datamunge_random_tests.cpp)
target_link_libraries(run_chacha_engine_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_chacha_engine_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_chacha_engine_tests TEST_PREFIX "chacha-engine.")

add_executable(run_chakravala_tests cpp/algorithms_chakravala_trace_tests.cpp cpp/algorithms_number_theory_advanced_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_chakravala_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_chakravala_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_chakravala_tests TEST_PREFIX "chakravala.")

add_executable(run_cheney_tests cpp/algorithms_cheney_trace_tests.cpp cpp/algorithms_memory_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_cheney_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cheney_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cheney_tests TEST_PREFIX "cheney.")

add_executable(run_chien_tests cpp/algorithms_chien_trace_tests.cpp cpp/algorithms_polynomial_gf_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_chien_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_chien_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_chien_tests TEST_PREFIX "chien.")

add_executable(run_crt_tests cpp/algorithms_crt_trace_tests.cpp cpp/algorithms_chinese_remainder_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_crt_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_crt_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_crt_tests TEST_PREFIX "crt.")

add_executable(run_christofides_tests cpp/algorithms_christofides_trace_tests.cpp cpp/algorithms_tour_search_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_christofides_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_christofides_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_christofides_tests TEST_PREFIX "christofides.")

add_executable(run_chudnovsky_tests cpp/algorithms_chudnovsky_trace_tests.cpp cpp/algorithms_pi_and_cordic_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_chudnovsky_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_chudnovsky_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_chudnovsky_tests TEST_PREFIX "chudnovsky.")

add_executable(run_chu_liu_tests cpp/algorithms_chu_liu_trace_tests.cpp cpp/algorithms_advanced_graph_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_chu_liu_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_chu_liu_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_chu_liu_tests TEST_PREFIX "chu-liu.")

add_executable(run_cipolla_tests cpp/algorithms_cipolla_trace_tests.cpp cpp/algorithms_arithmetic_extra_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_cipolla_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cipolla_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cipolla_tests TEST_PREFIX "cipolla.")

add_executable(run_circle_packing_tests cpp/algorithms_circle_packing_trace_tests.cpp cpp/algorithms_graph_analysis_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_circle_packing_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_circle_packing_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_circle_packing_tests TEST_PREFIX "circle-packing.")

add_executable(run_clarke_wright_tests cpp/algorithms_clarke_wright_trace_tests.cpp cpp/algorithms_tour_search_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_clarke_wright_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_clarke_wright_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_clarke_wright_tests TEST_PREFIX "clarke-wright.")

add_executable(run_closest_pair_tests cpp/geometry_closest_pair_trace_tests.cpp cpp/geometry_closest_pair_tests.cpp cpp/format/geometry_format_tests.cpp)
target_link_libraries(run_closest_pair_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_closest_pair_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_closest_pair_tests TEST_PREFIX "closest-pair.")

add_executable(run_small_image_cnn_tests cpp/nn_small_image_cnn_tests.cpp cpp/stats_cnn_classifier_tests.cpp)
target_link_libraries(run_small_image_cnn_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_small_image_cnn_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_small_image_cnn_tests TEST_PREFIX "small-image-cnn.")

add_executable(run_cocktail_sort_tests cpp/algorithms_cocktail_trace_tests.cpp cpp/algorithms_exchange_sorts_tests.cpp)
target_link_libraries(run_cocktail_sort_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cocktail_sort_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cocktail_sort_tests TEST_PREFIX "cocktail-sort.")

add_executable(run_cohen_sutherland_tests cpp/algorithms_cohen_trace_tests.cpp cpp/algorithms_graphics_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_cohen_sutherland_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cohen_sutherland_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cohen_sutherland_tests TEST_PREFIX "cohen-sutherland.")

add_executable(run_comb_sort_tests cpp/algorithms_comb_trace_tests.cpp cpp/algorithms_comparison_sorts_tests.cpp)
target_link_libraries(run_comb_sort_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_comb_sort_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_comb_sort_tests TEST_PREFIX "comb-sort.")

add_executable(run_computus_tests cpp/algorithms_computus_trace_tests.cpp cpp/algorithms_calendar_geo_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_computus_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_computus_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_computus_tests TEST_PREFIX "computus.")

add_executable(run_cone_surface_tests cpp/geometry_cone_trace_tests.cpp cpp/misc_batch44_tests.cpp cpp/format/geometry_format_tests.cpp)
target_link_libraries(run_cone_surface_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cone_surface_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cone_surface_tests TEST_PREFIX "cone-surface.")

add_executable(run_conjugate_gradient_tests cpp/algorithms_cg_trace_tests.cpp cpp/algorithms_linear_solvers_tests.cpp cpp/algorithms_extra_solvers_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_conjugate_gradient_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_conjugate_gradient_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_conjugate_gradient_tests TEST_PREFIX "conjugate-gradient.")

add_executable(run_component_label_tests cpp/algorithms_component_trace_tests.cpp cpp/algorithms_image_batch_tests.cpp cpp/plot_label_images_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_component_label_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_component_label_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_component_label_tests TEST_PREFIX "component-label.")

add_executable(run_convex_hull_tests cpp/geometry_hull_trace_tests.cpp cpp/geometry_convex_hull_tests.cpp cpp/geometry_hull_algorithms_tests.cpp cpp/geometry_batch2_tests.cpp cpp/format/geometry_format_tests.cpp)
target_link_libraries(run_convex_hull_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_convex_hull_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_convex_hull_tests TEST_PREFIX "convex-hull.")

# Independent exact-rational predicate controls use Python's standard library when available.
find_package(Python3 3.9 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
  add_executable(check_orientation_fraction_oracle cpp/geometry_orientation_oracle.cpp)
  target_link_libraries(check_orientation_fraction_oracle datamunge::datamunge)
  add_dependencies(run_convex_hull_tests check_orientation_fraction_oracle)
  set_target_properties(check_orientation_fraction_oracle PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_test(NAME convex-hull.FractionOracle COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/oracles/orientation_fraction.py $<TARGET_FILE:check_orientation_fraction_oracle>)
endif()

add_executable(run_cordic_trace_tests cpp/algorithms_cordic_trace_tests.cpp cpp/algorithms_pi_and_cordic_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_cordic_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cordic_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cordic_trace_tests TEST_PREFIX "cordic.")

if(Python3_Interpreter_FOUND)
  add_executable(check_cordic_decimal_oracle cpp/algorithms_cordic_oracle.cpp)
  target_link_libraries(check_cordic_decimal_oracle datamunge::datamunge)
  set_target_properties(check_cordic_decimal_oracle PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_dependencies(run_cordic_trace_tests check_cordic_decimal_oracle)
  add_test(NAME cordic.DecimalOracle COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/oracles/cordic_decimal.py $<TARGET_FILE:check_cordic_decimal_oracle>)
endif()

add_executable(run_counting_sort_tests cpp/algorithms_counting_trace_tests.cpp cpp/algorithms_counting_sort_tests.cpp)
target_link_libraries(run_counting_sort_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_counting_sort_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_counting_sort_tests TEST_PREFIX "counting-sort.")

add_executable(run_crank_nicolson_tests cpp/algorithms_crank_nicolson_trace_tests.cpp cpp/algorithms_pde_solvers_tests.cpp)
target_link_libraries(run_crank_nicolson_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_crank_nicolson_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_crank_nicolson_tests TEST_PREFIX "crank-nicolson.")

add_executable(run_crc32_tests cpp/algorithms_crc32_trace_tests.cpp cpp/algorithms_error_detection_batch_tests.cpp cpp/algorithms_adler32_tests.cpp cpp/image_zlib_codec_tests.cpp)
target_link_libraries(run_crc32_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_crc32_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_crc32_tests TEST_PREFIX "crc32.")

add_executable(run_cristian_tests cpp/algorithms_cristian_tests.cpp cpp/algorithms_distributed_batch_tests.cpp)
target_link_libraries(run_cristian_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cristian_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cristian_tests TEST_PREFIX "cristian.")

add_executable(run_cubic_spline_tests cpp/algorithms_cubic_spline_trace_tests.cpp cpp/algorithms_interpolation_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_cubic_spline_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cubic_spline_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cubic_spline_tests TEST_PREFIX "cubic-spline.")

if(Python3_Interpreter_FOUND)
  add_executable(check_natural_spline_fraction_oracle cpp/algorithms_cubic_spline_oracle.cpp)
  target_link_libraries(check_natural_spline_fraction_oracle datamunge::datamunge)
  set_target_properties(check_natural_spline_fraction_oracle PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
  add_dependencies(run_cubic_spline_tests check_natural_spline_fraction_oracle)
  add_test(NAME cubic-spline.FractionOracle COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/oracles/natural_spline_fraction.py $<TARGET_FILE:check_natural_spline_fraction_oracle>)
endif()

add_executable(run_cuthill_mckee_tests cpp/algorithms_cuthill_mckee_trace_tests.cpp cpp/algorithms_minimum_degree_trace_tests.cpp cpp/algorithms_sparse_linalg_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_cuthill_mckee_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cuthill_mckee_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cuthill_mckee_tests TEST_PREFIX "cuthill-mckee.")

add_executable(run_cycle_sort_tests cpp/algorithms_cycle_sort_trace_tests.cpp cpp/algorithms_distribution_sorts_tests.cpp)
target_link_libraries(run_cycle_sort_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cycle_sort_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cycle_sort_tests TEST_PREFIX "cycle-sort.")

add_executable(run_cyk_tests cpp/algorithms_cyk_chart_tests.cpp cpp/algorithms_parsing_hashing_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_cyk_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_cyk_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_cyk_tests TEST_PREFIX "cyk.")

add_executable(run_dag_longest_path_tests cpp/algorithms_dag_longest_trace_tests.cpp cpp/algorithms_routing_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_dag_longest_path_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_dag_longest_path_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_dag_longest_path_tests TEST_PREFIX "dag-longest.")

add_executable(run_damerau_tests cpp/algorithms_damerau_trace_tests.cpp cpp/algorithms_string_metrics_tests.cpp cpp/algorithms_hamming_distance_tests.cpp)
target_link_libraries(run_damerau_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_damerau_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_damerau_tests TEST_PREFIX "damerau.")

add_executable(run_davis_putnam_tests cpp/algorithms_davis_putnam_trace_tests.cpp cpp/optimization_batch45_tests.cpp)
target_link_libraries(run_davis_putnam_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_davis_putnam_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_davis_putnam_tests TEST_PREFIX "davis-putnam.")

add_executable(run_dda_tests cpp/algorithms_dda_trace_tests.cpp cpp/algorithms_graphics_batch_tests.cpp)
target_link_libraries(run_dda_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_dda_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_dda_tests TEST_PREFIX "dda.")

add_executable(run_de_casteljau_tests cpp/algorithms_de_casteljau_trace_tests.cpp cpp/algorithms_interpolation_tests.cpp cpp/geometry_de_boor_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_de_casteljau_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_de_casteljau_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_de_casteljau_tests TEST_PREFIX "de-casteljau.")

add_executable(run_delaunay_tests cpp/geometry_delaunay_trace_tests.cpp cpp/geometry_delaunay_tests.cpp cpp/geometry_batch38_tests.cpp cpp/geometry_voronoi_tests.cpp cpp/datamunge_fem_tests.cpp cpp/format/geometry_format_tests.cpp)
target_link_libraries(run_delaunay_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_delaunay_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_delaunay_tests TEST_PREFIX "delaunay.")
if(Python3_Interpreter_FOUND)
  add_executable(check_incircle_fraction_oracle cpp/geometry_incircle_oracle.cpp)
  target_link_libraries(check_incircle_fraction_oracle datamunge::datamunge)
  set_target_properties(check_incircle_fraction_oracle PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
  add_dependencies(run_delaunay_tests check_incircle_fraction_oracle)
  add_test(NAME delaunay.FractionOracle COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/oracles/incircle_fraction.py $<TARGET_FILE:check_incircle_fraction_oracle>)
endif()

add_executable(run_depth_first_tests cpp/algorithms_depth_first_trace_tests.cpp cpp/algorithms_search_more_tests.cpp)
target_link_libraries(run_depth_first_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_depth_first_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_depth_first_tests TEST_PREFIX "depth-first.")

add_executable(run_deutsch_jozsa_tests cpp/algorithms_deutsch_jozsa_trace_tests.cpp cpp/algorithms_quantum_batch_tests.cpp cpp/algorithms_fourier_sampling_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_deutsch_jozsa_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_deutsch_jozsa_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_deutsch_jozsa_tests TEST_PREFIX "deutsch-jozsa.")

add_executable(run_dice_tests cpp/algorithms_dice_trace_tests.cpp cpp/algorithms_string_metrics_tests.cpp cpp/algorithms_hamming_distance_tests.cpp)
target_link_libraries(run_dice_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_dice_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_dice_tests TEST_PREFIX "dice.")

add_executable(run_difference_map_tests cpp/algorithms_difference_map_trace_tests.cpp cpp/heuristic_batch50_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_difference_map_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_difference_map_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_difference_map_tests TEST_PREFIX "difference-map.")

add_executable(run_diffie_hellman_tests cpp/algorithms_diffie_hellman_trace_tests.cpp cpp/crypto_batch23_tests.cpp cpp/algorithms_modular_exponentiation_tests.cpp cpp/algorithms_miller_rabin_tests.cpp cpp/algorithms_montgomery_trace_tests.cpp cpp/algorithms_addition_chain_tests.cpp)
target_link_libraries(run_diffie_hellman_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_diffie_hellman_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_diffie_hellman_tests TEST_PREFIX "diffie-hellman.")
if(Python3_Interpreter_FOUND)
  add_executable(check_modular_integer_oracle cpp/algorithms_modular_integer_oracle.cpp)
  target_link_libraries(check_modular_integer_oracle datamunge::datamunge)
  set_target_properties(check_modular_integer_oracle PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
  add_dependencies(run_diffie_hellman_tests check_modular_integer_oracle)
  add_test(NAME diffie-hellman.IntegerOracle COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/oracles/modular_integer.py $<TARGET_FILE:check_modular_integer_oracle>)
endif()

add_executable(run_diffusion_maps_tests cpp/stats_diffusion_maps_trace_tests.cpp cpp/stats_diffusion_maps_tests.cpp cpp/stats_diffusion_maps_facade_tests.cpp)
target_link_libraries(run_diffusion_maps_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_diffusion_maps_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_diffusion_maps_tests TEST_PREFIX "diffusion-maps.")

add_executable(run_dijkstra_trace_tests cpp/algorithms_dijkstra_trace_tests.cpp cpp/algorithms_dijkstra_tests.cpp cpp/algorithms_bellman_ford_tests.cpp cpp/algorithms_floyd_warshall_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_dijkstra_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_dijkstra_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_dijkstra_trace_tests TEST_PREFIX "dijkstra.")

add_executable(run_dinic_tests cpp/algorithms_dinic_trace_tests.cpp cpp/algorithms_advanced_graph_tests.cpp cpp/algorithms_edmonds_karp_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_dinic_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_dinic_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_dinic_tests TEST_PREFIX "dinic.")

add_executable(run_distance_transform_tests cpp/geometry_distance_transform_trace_tests.cpp cpp/geometry_numeric_batch_tests.cpp)
target_link_libraries(run_distance_transform_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_distance_transform_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_distance_transform_tests TEST_PREFIX "distance-transform.")

add_executable(run_dixon_tests cpp/algorithms_dixon_trace_tests.cpp cpp/algorithms_factorization_tests.cpp)
target_link_libraries(run_dixon_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_dixon_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_dixon_tests TEST_PREFIX "dixon.")

add_executable(run_doomsday_tests cpp/algorithms_doomsday_trace_tests.cpp cpp/algorithms_calendar_geo_batch_tests.cpp cpp/algorithms_zeller_tests.cpp)
target_link_libraries(run_doomsday_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_doomsday_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_doomsday_tests TEST_PREFIX "doomsday.")

add_executable(run_double_dabble_tests cpp/algorithms_double_dabble_trace_tests.cpp cpp/algorithms_parsing_hashing_batch_tests.cpp)
target_link_libraries(run_double_dabble_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_double_dabble_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_double_dabble_tests TEST_PREFIX "double-dabble.")

add_executable(run_dpll_tests cpp/algorithms_dpll_trace_tests.cpp cpp/algorithms_combinatorial_search_tests.cpp cpp/optimization_batch45_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_dpll_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_dpll_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_dpll_tests TEST_PREFIX "dpll.")

add_executable(run_d_star_tests cpp/algorithms_d_star_lite_tests.cpp cpp/algorithms_search_more_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_d_star_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_d_star_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_d_star_tests TEST_PREFIX "d-star.")

add_executable(run_dt_iris_tests cpp/stats_decision_tree_diagram_tests.cpp cpp/stats_decision_tree_tests.cpp)
target_link_libraries(run_dt_iris_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_dt_iris_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_dt_iris_tests TEST_PREFIX "dt-iris.")

add_executable(run_dtw_tests cpp/geometry_dtw_trace_tests.cpp cpp/geometry_dtw_and_frechet_tests.cpp cpp/format/geometry_format_tests.cpp)
target_link_libraries(run_dtw_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_dtw_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_dtw_tests TEST_PREFIX "dtw.")

add_executable(run_earley_tests cpp/algorithms_earley_trace_tests.cpp cpp/parsing_batch25_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_earley_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_earley_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_earley_tests TEST_PREFIX "earley.")

add_executable(run_eclat_tests cpp/algorithms_eclat_trace_tests.cpp cpp/algorithms_association_rules_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_eclat_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_eclat_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_eclat_tests TEST_PREFIX "eclat.")

add_executable(run_edf_tests cpp/algorithms_edf_trace_tests.cpp cpp/algorithms_scheduling_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_edf_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_edf_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_edf_tests TEST_PREFIX "edf.")

add_executable(run_edmonds_karp_trace_tests cpp/algorithms_edmonds_karp_trace_tests.cpp cpp/algorithms_edmonds_karp_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_edmonds_karp_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_edmonds_karp_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_edmonds_karp_trace_tests TEST_PREFIX "edmonds-karp.")

add_executable(run_gamma_tests cpp/algorithms_elias_gamma_trace_tests.cpp cpp/algorithms_compression2_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_gamma_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_gamma_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_gamma_tests TEST_PREFIX "gamma.")

add_executable(run_ellipsoid_tests cpp/algorithms_ellipsoid_trace_tests.cpp cpp/algorithms_min_conflicts_trace_tests.cpp cpp/algorithms_csp_convex_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_ellipsoid_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_ellipsoid_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_ellipsoid_tests TEST_PREFIX "ellipsoid.")

add_executable(run_euclidean_mst_tests cpp/algorithms_euclidean_mst_trace_tests.cpp cpp/algorithms_advanced_graph_tests.cpp cpp/format/algorithms_format_tests.cpp cpp/algorithms_kruskal_tests.cpp cpp/algorithms_prim_tests.cpp)
target_link_libraries(run_euclidean_mst_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_euclidean_mst_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_euclidean_mst_tests TEST_PREFIX "euclidean-mst.")

add_executable(run_euler_method_tests cpp/algorithms_euler_method_trace_tests.cpp cpp/algorithms_ode_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_euler_method_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_euler_method_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_euler_method_tests TEST_PREFIX "euler-method.")

add_executable(run_exact_cover_tests cpp/algorithms_exact_cover_trace_tests.cpp cpp/algorithms_csp_convex_tests.cpp cpp/algorithms_sudoku_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_exact_cover_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_exact_cover_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_exact_cover_tests TEST_PREFIX "exact-cover.")

add_executable(run_exp_golomb_tests cpp/algorithms_exp_golomb_trace_tests.cpp cpp/algorithms_coding_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_exp_golomb_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_exp_golomb_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_exp_golomb_tests TEST_PREFIX "exp-golomb.")

add_executable(run_tensor_image_explainers_tests cpp/nn_image_explainers_tests.cpp cpp/stats_image_explainers_tests.cpp)
target_link_libraries(run_tensor_image_explainers_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_tensor_image_explainers_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_tensor_image_explainers_tests TEST_PREFIX "tensor-image-explainers.")

add_executable(run_backoff_tests cpp/algorithms_backoff_trace_tests.cpp cpp/algorithms_networking_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_backoff_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_backoff_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_backoff_tests TEST_PREFIX "backoff.")

add_executable(run_euclid_tests cpp/algorithms_euclid_trace_tests.cpp cpp/algorithms_extended_euclidean_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_euclid_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_euclid_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_euclid_tests TEST_PREFIX "euclid.")
if(Python3_Interpreter_FOUND)
  add_executable(check_euclid_integer_oracle cpp/algorithms_euclid_integer_oracle.cpp)
  target_link_libraries(check_euclid_integer_oracle datamunge::datamunge)
  set_target_properties(check_euclid_integer_oracle PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
  add_dependencies(run_euclid_tests check_euclid_integer_oracle)
  add_test(NAME euclid.IntegerOracle COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/oracles/euclid_integer.py $<TARGET_FILE:check_euclid_integer_oracle>)
endif()

add_executable(run_eytzinger_tests cpp/algorithms_eytzinger_trace_tests.cpp cpp/algorithms_sorted_search_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_eytzinger_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_eytzinger_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_eytzinger_tests TEST_PREFIX "eytzinger.")

add_executable(run_false_position_tests cpp/algorithms_false_position_trace_tests.cpp cpp/algorithms_extra_root_finding_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_false_position_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_false_position_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_false_position_tests TEST_PREFIX "false-position.")

add_executable(run_fem_report_tests cpp/fem_named_transient_tests.cpp cpp/datamunge_fem_tests.cpp cpp/format/fem_format_tests.cpp)
target_link_libraries(run_fem_report_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_fem_report_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_fem_report_tests TEST_PREFIX "fem-report.")

add_executable(run_fermat_factor_tests cpp/algorithms_fermat_factor_trace_tests.cpp cpp/algorithms_factorization_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_fermat_factor_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_fermat_factor_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_fermat_factor_tests TEST_PREFIX "fermat-factor.")
if(Python3_Interpreter_FOUND)
  add_executable(check_integer_sqrt128_oracle cpp/algorithms_integer_sqrt128_oracle.cpp)
  target_link_libraries(check_integer_sqrt128_oracle datamunge::datamunge)
  set_target_properties(check_integer_sqrt128_oracle PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
  add_dependencies(run_fermat_factor_tests check_integer_sqrt128_oracle)
  add_test(NAME fermat-factor.IntegerOracle COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/oracles/integer_sqrt128.py $<TARGET_FILE:check_integer_sqrt128_oracle>)
endif()

add_executable(run_fermat_primality_tests cpp/algorithms_fermat_primality_trace_tests.cpp cpp/algorithms_primality_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_fermat_primality_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_fermat_primality_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_fermat_primality_tests TEST_PREFIX "fermat-primality.")
if(Python3_Interpreter_FOUND)
  add_executable(check_fermat_primality_oracle cpp/algorithms_fermat_primality_oracle.cpp)
  target_link_libraries(check_fermat_primality_oracle datamunge::datamunge)
  set_target_properties(check_fermat_primality_oracle PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
  add_dependencies(run_fermat_primality_tests check_fermat_primality_oracle)
  add_test(NAME fermat-primality.IntegerOracle COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/oracles/fermat_primality.py $<TARGET_FILE:check_fermat_primality_oracle>)
endif()

add_executable(run_fibonacci_coding_tests cpp/algorithms_fibonacci_coding_trace_tests.cpp cpp/algorithms_coding_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_fibonacci_coding_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_fibonacci_coding_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_fibonacci_coding_tests TEST_PREFIX "fibonacci-coding.")
if(Python3_Interpreter_FOUND)
  add_executable(check_fibonacci_coding_oracle cpp/algorithms_fibonacci_coding_oracle.cpp)
  target_link_libraries(check_fibonacci_coding_oracle datamunge::datamunge)
  set_target_properties(check_fibonacci_coding_oracle PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
  add_dependencies(run_fibonacci_coding_tests check_fibonacci_coding_oracle)
  add_test(NAME fibonacci-coding.IntegerOracle COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/oracles/fibonacci_coding.py $<TARGET_FILE:check_fibonacci_coding_oracle>)
endif()

add_executable(run_fibonacci_search_tests cpp/algorithms_fibonacci_search_trace_tests.cpp cpp/algorithms_sorted_search_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_fibonacci_search_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_fibonacci_search_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_fibonacci_search_tests TEST_PREFIX "fibonacci-search.")

add_executable(run_fbp_tests cpp/algorithms_filtered_back_projection_tests.cpp cpp/numerical_batch42_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_fbp_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_fbp_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_fbp_tests TEST_PREFIX "fbp.")

add_executable(run_finite_difference_tests cpp/algorithms_finite_difference_tests.cpp cpp/algorithms_pde_solvers_tests.cpp)
target_link_libraries(run_finite_difference_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_finite_difference_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_finite_difference_tests TEST_PREFIX "finite-difference.")

add_executable(run_fisher_yates_tests cpp/algorithms_fisher_yates_tests.cpp cpp/algorithms_permutations_tests.cpp)
target_link_libraries(run_fisher_yates_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_fisher_yates_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_fisher_yates_tests TEST_PREFIX "fisher-yates.")

add_executable(run_flashsort_tests cpp/algorithms_flashsort_tests.cpp cpp/algorithms_sorting_extra_tests.cpp)
target_link_libraries(run_flashsort_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_flashsort_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_flashsort_tests TEST_PREFIX "flashsort.")

add_executable(run_fletcher_tests cpp/algorithms_fletcher16_tests.cpp cpp/algorithms_error_detection_batch_tests.cpp)
target_link_libraries(run_fletcher_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_fletcher_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_fletcher_tests TEST_PREFIX "fletcher.")

add_executable(run_flood_fill_tests cpp/algorithms_flood_fill_tests.cpp cpp/algorithms_compression_graphics_batch_tests.cpp cpp/plot_label_images_tests.cpp)
target_link_libraries(run_flood_fill_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_flood_fill_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_flood_fill_tests TEST_PREFIX "flood-fill.")

add_executable(run_floyd_trace_tests cpp/algorithms_floyd_trace_tests.cpp)
target_link_libraries(run_floyd_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_floyd_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_floyd_trace_tests TEST_PREFIX "floyd.")

add_executable(run_floyd_steinberg_tests cpp/algorithms_floyd_steinberg_tests.cpp cpp/algorithms_image_batch_tests.cpp)
target_link_libraries(run_floyd_steinberg_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_floyd_steinberg_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_floyd_steinberg_tests TEST_PREFIX "floyd-steinberg.")

add_executable(run_floyd_warshall_trace_tests cpp/algorithms_floyd_warshall_trace_tests.cpp cpp/algorithms_floyd_warshall_tests.cpp)
target_link_libraries(run_floyd_warshall_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_floyd_warshall_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_floyd_warshall_trace_tests TEST_PREFIX "floyd-warshall.")

add_executable(run_force_directed_tests cpp/algorithms_force_directed_layout_tests.cpp cpp/algorithms_graph_analysis_tests.cpp)
target_link_libraries(run_force_directed_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_force_directed_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_force_directed_tests TEST_PREFIX "force-directed.")

add_executable(run_forward_backward_tests cpp/algorithms_forward_backward_analysis_tests.cpp cpp/algorithms_hmm_tests.cpp cpp/format/algorithms_format_tests.cpp cpp/algorithms_gaussian_hmm_tests.cpp)
target_link_libraries(run_forward_backward_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_forward_backward_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_forward_backward_tests TEST_PREFIX "forward-backward.")

add_executable(run_fp_growth_tests cpp/algorithms_fp_growth_trace_tests.cpp cpp/algorithms_association_rules_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_fp_growth_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_fp_growth_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_fp_growth_tests TEST_PREFIX "fp-growth.")

add_executable(run_freivalds_tests cpp/algorithms_freivalds_tests.cpp cpp/algorithms_linear_algebra_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_freivalds_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_freivalds_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_freivalds_tests TEST_PREFIX "freivalds.")

add_executable(run_fuzzy_c_means_tests cpp/algorithms_fuzzy_c_means_tests.cpp cpp/algorithms_clustering_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_fuzzy_c_means_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_fuzzy_c_means_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_fuzzy_c_means_tests TEST_PREFIX "fuzzy-c-means.")

add_executable(run_gabow_scc_tests cpp/algorithms_gabow_scc_tests.cpp cpp/misc_batch46_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_gabow_scc_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_gabow_scc_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_gabow_scc_tests TEST_PREFIX "gabow-scc.")

add_executable(run_gale_shapley_tests cpp/algorithms_stable_matching_trace_tests.cpp cpp/algorithms_stable_matching_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_gale_shapley_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_gale_shapley_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_gale_shapley_tests TEST_PREFIX "gale-shapley.")

add_executable(run_gaussian_elimination_tests cpp/algorithms_gaussian_elimination_trace_tests.cpp cpp/algorithms_linear_solvers_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_gaussian_elimination_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_gaussian_elimination_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_gaussian_elimination_tests TEST_PREFIX "gaussian-elimination.")

add_executable(run_gauss_jordan_tests cpp/algorithms_gauss_jordan_trace_tests.cpp cpp/algorithms_extra_solvers_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_gauss_jordan_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_gauss_jordan_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_gauss_jordan_tests TEST_PREFIX "gauss-jordan.")

add_executable(run_gauss_legendre_pi_tests cpp/algorithms_gauss_legendre_pi_trace_tests.cpp cpp/algorithms_pi_and_cordic_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_gauss_legendre_pi_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_gauss_legendre_pi_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_gauss_legendre_pi_tests TEST_PREFIX "gauss-legendre-pi.")

add_executable(run_gauss_seidel_tests cpp/algorithms_gauss_seidel_history_tests.cpp cpp/algorithms_cg_trace_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_gauss_seidel_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_gauss_seidel_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_gauss_seidel_tests TEST_PREFIX "gauss-seidel.")

add_executable(run_general_problem_solver_tests cpp/algorithms_general_problem_solver_tests.cpp cpp/algorithms_search_more_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_general_problem_solver_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_general_problem_solver_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_general_problem_solver_tests TEST_PREFIX "general-problem-solver.")

add_executable(run_generational_tests cpp/algorithms_generational_trace_tests.cpp cpp/algorithms_memory_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_generational_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_generational_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_generational_tests TEST_PREFIX "generational.")

add_executable(run_geohash_tests cpp/algorithms_geohash_trace_tests.cpp cpp/algorithms_calendar_geo_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_geohash_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_geohash_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_geohash_tests TEST_PREFIX "geohash.")

add_executable(run_geometric_hashing_tests cpp/geometry_geometric_hashing_trace_tests.cpp cpp/geometry_batch38_tests.cpp cpp/format/geometry_format_tests.cpp)
target_link_libraries(run_geometric_hashing_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_geometric_hashing_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_geometric_hashing_tests TEST_PREFIX "geometric-hashing.")

add_executable(run_girvan_newman_tests cpp/algorithms_girvan_newman_trace_tests.cpp cpp/datasets_network_data_tests.cpp cpp/datasets_loaders_tests.cpp cpp/algorithms_graph_analysis_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_girvan_newman_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_girvan_newman_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_girvan_newman_tests TEST_PREFIX "girvan-newman.")

add_executable(run_gjk_tests cpp/geometry_gjk_trace_tests.cpp cpp/geometry_numeric_batch_tests.cpp cpp/format/geometry_format_tests.cpp)
target_link_libraries(run_gjk_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_gjk_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_gjk_tests TEST_PREFIX "gjk.")

add_executable(run_gnome_sort_tests cpp/algorithms_gnome_sort_trace_tests.cpp cpp/algorithms_exchange_sorts_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_gnome_sort_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_gnome_sort_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_gnome_sort_tests TEST_PREFIX "gnome-sort.")

add_executable(run_goertzel_tests cpp/algorithms_goertzel_trace_tests.cpp cpp/algorithms_dsp_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_goertzel_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_goertzel_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_goertzel_tests TEST_PREFIX "goertzel.")

# Actual golden-section execution, preserved original searches and record formatting.
add_executable(run_golden_section_tests cpp/algorithms_golden_section_trace_tests.cpp cpp/algorithms_search_order_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_golden_section_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_golden_section_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_golden_section_tests TEST_PREFIX "golden-section.")

add_executable(run_nn_logistic_pixels_tests cpp/nn_logistic_pixels_tests.cpp)
target_link_libraries(run_nn_logistic_pixels_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_logistic_pixels_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_logistic_pixels_tests TEST_PREFIX "nn-logistic-pixel.")

add_executable(run_nn_wavenet_generation_tests cpp/nn_wavenet_generation_tests.cpp)
target_link_libraries(run_nn_wavenet_generation_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_wavenet_generation_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_wavenet_generation_tests TEST_PREFIX "nn-wave-generation.")

add_executable(run_nn_nice_density_tests cpp/nn_nice_density_tests.cpp)
target_link_libraries(run_nn_nice_density_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_nice_density_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_nice_density_tests TEST_PREFIX "nn-nice-density.")

add_executable(run_nn_affine_flow_tests cpp/nn_affine_flow_tests.cpp)
target_link_libraries(run_nn_affine_flow_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_affine_flow_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_affine_flow_tests TEST_PREFIX "nn-affine-flow.")

add_executable(run_nn_glow_density_tests cpp/nn_glow_density_tests.cpp)
target_link_libraries(run_nn_glow_density_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_glow_density_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_glow_density_tests TEST_PREFIX "nn-glow-density.")

add_executable(run_nn_maf_density_tests cpp/nn_maf_density_tests.cpp)
target_link_libraries(run_nn_maf_density_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_maf_density_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_maf_density_tests TEST_PREFIX "nn-maf-density.")

add_executable(run_nn_iaf_density_tests cpp/nn_iaf_density_tests.cpp)
target_link_libraries(run_nn_iaf_density_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_iaf_density_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_iaf_density_tests TEST_PREFIX "nn-iaf-density.")

add_executable(run_nn_spline_density_tests cpp/nn_spline_density_tests.cpp)
target_link_libraries(run_nn_spline_density_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_spline_density_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_spline_density_tests TEST_PREFIX "nn-spline-density.")

add_executable(run_nn_planar_radial_density_tests cpp/nn_planar_radial_density_tests.cpp)
target_link_libraries(run_nn_planar_radial_density_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_planar_radial_density_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_planar_radial_density_tests TEST_PREFIX "nn-planar-radial-density.")

add_executable(run_nn_cnf_density_tests cpp/nn_cnf_density_tests.cpp)
target_link_libraries(run_nn_cnf_density_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_cnf_density_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_cnf_density_tests TEST_PREFIX "nn-cnf-density.")

add_executable(run_nn_flow_study_tests cpp/nn_flow_study_tests.cpp)
target_link_libraries(run_nn_flow_study_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_flow_study_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_flow_study_tests TEST_PREFIX "nn-flow-study.")

add_executable(run_nn_diffusion_tests cpp/nn_diffusion_tests.cpp)
target_link_libraries(run_nn_diffusion_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_diffusion_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_diffusion_tests TEST_PREFIX "nn-diffusion.")

add_executable(run_mlp_classifier_search_tests cpp/mlp_classifier_search_tests.cpp cpp/stats_mlp_classifier_tests.cpp)
target_link_libraries(run_mlp_classifier_search_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mlp_classifier_search_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mlp_classifier_search_tests TEST_PREFIX "mlp-search.")

add_executable(run_fnv_tests cpp/algorithms_fnv_tests.cpp)
target_link_libraries(run_fnv_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_fnv_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_fnv_tests TEST_PREFIX "fnv.")

add_executable(run_goldschmidt_tests cpp/algorithms_goldschmidt_tests.cpp)
target_link_libraries(run_goldschmidt_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_goldschmidt_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_goldschmidt_tests TEST_PREFIX "goldschmidt.")

add_executable(run_golomb_tests cpp/algorithms_golomb_tests.cpp)
target_link_libraries(run_golomb_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_golomb_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_golomb_tests TEST_PREFIX "golomb.")

add_executable(run_graham_trace_tests cpp/geometry_graham_trace_tests.cpp)
target_link_libraries(run_graham_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_graham_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_graham_trace_tests TEST_PREFIX "graham.")

add_executable(run_gram_schmidt_trace_tests cpp/algorithms_gram_schmidt_trace_tests.cpp)
target_link_libraries(run_gram_schmidt_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_gram_schmidt_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_gram_schmidt_trace_tests TEST_PREFIX "gram_schmidt.")

add_executable(run_graph_coloring_trace_tests cpp/algorithms_graph_coloring_trace_tests.cpp)
target_link_libraries(run_graph_coloring_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_graph_coloring_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_graph_coloring_trace_tests TEST_PREFIX "graph_coloring.")

add_executable(run_grasp_trace_tests cpp/algorithms_grasp_trace_tests.cpp)
target_link_libraries(run_grasp_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_grasp_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_grasp_trace_tests TEST_PREFIX "grasp.")

add_executable(run_grover_trace_tests cpp/algorithms_grover_trace_tests.cpp)
target_link_libraries(run_grover_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_grover_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_grover_trace_tests TEST_PREFIX "grover.")

add_executable(run_halley_trace_tests cpp/algorithms_halley_trace_tests.cpp)
target_link_libraries(run_halley_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_halley_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_halley_trace_tests TEST_PREFIX "halley.")

add_executable(run_hamming_code_trace_tests cpp/algorithms_hamming_code_trace_tests.cpp)
target_link_libraries(run_hamming_code_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_hamming_code_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_hamming_code_trace_tests TEST_PREFIX "hamming_code.")

add_executable(run_heap_trace_tests cpp/algorithms_heap_trace_tests.cpp)
target_link_libraries(run_heap_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_heap_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_heap_trace_tests TEST_PREFIX "heap_permutations.")

add_executable(run_heapsort_trace_tests cpp/algorithms_heapsort_trace_tests.cpp)
target_link_libraries(run_heapsort_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_heapsort_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_heapsort_trace_tests TEST_PREFIX "heapsort_trace.")

add_executable(run_hermite_tests cpp/algorithms_hermite_tests.cpp)
target_link_libraries(run_hermite_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_hermite_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_hermite_tests TEST_PREFIX "hermite.")

add_executable(run_hirschberg_tests cpp/algorithms_hirschberg_tests.cpp)
target_link_libraries(run_hirschberg_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_hirschberg_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_hirschberg_tests TEST_PREFIX "hirschberg.")

add_executable(run_equalization_tests cpp/algorithms_equalization_tests.cpp)
target_link_libraries(run_equalization_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_equalization_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_equalization_tests TEST_PREFIX "equalization.")

add_executable(run_hits_trace_tests cpp/algorithms_hits_trace_tests.cpp)
target_link_libraries(run_hits_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_hits_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_hits_trace_tests TEST_PREFIX "hits_trace.")

add_executable(run_hmac_trace_tests cpp/algorithms_hmac_trace_tests.cpp)
target_link_libraries(run_hmac_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_hmac_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_hmac_trace_tests TEST_PREFIX "hmac_trace.")

add_executable(run_hopcroft_trace_tests cpp/algorithms_hopcroft_trace_tests.cpp)
target_link_libraries(run_hopcroft_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_hopcroft_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_hopcroft_trace_tests TEST_PREFIX "hopcroft_trace.")

add_executable(run_hopcroft_karp_trace_tests cpp/algorithms_hopcroft_karp_trace_tests.cpp cpp/algorithms_hopcroft_karp_tests.cpp)
target_link_libraries(run_hopcroft_karp_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_hopcroft_karp_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_hopcroft_karp_trace_tests TEST_PREFIX "hopcroft_karp_trace.")

add_executable(run_hopfield_trace_tests cpp/algorithms_hopfield_trace_tests.cpp cpp/algorithms_neural_batch_tests.cpp)
target_link_libraries(run_hopfield_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_hopfield_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_hopfield_trace_tests TEST_PREFIX "hopfield_trace.")

add_executable(run_huffman_trace_tests cpp/algorithms_huffman_trace_tests.cpp)
target_link_libraries(run_huffman_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_huffman_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_huffman_trace_tests TEST_PREFIX "huffman_trace.")

add_executable(run_hungarian_trace_tests cpp/algorithms_hungarian_trace_tests.cpp cpp/algorithms_hungarian_tests.cpp)
target_link_libraries(run_hungarian_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_hungarian_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_hungarian_trace_tests TEST_PREFIX "hungarian_trace.")

add_executable(run_icp_trace_tests cpp/geometry_icp_trace_tests.cpp cpp/numerical_batch41_tests.cpp)
target_link_libraries(run_icp_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_icp_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_icp_trace_tests TEST_PREFIX "icp_trace.")

add_executable(run_iterative_deepening_tests cpp/algorithms_iterative_deepening_tests.cpp cpp/algorithms_search_more_tests.cpp)
target_link_libraries(run_iterative_deepening_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_iterative_deepening_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_iterative_deepening_tests TEST_PREFIX "iterative_deepening.")

add_executable(run_index_calculus_trace_tests cpp/algorithms_index_calculus_trace_tests.cpp cpp/algorithms_index_calculus_tests.cpp)
target_link_libraries(run_index_calculus_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_index_calculus_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_index_calculus_trace_tests TEST_PREFIX "index_calculus_trace.")

add_executable(run_insertion_sort_trace_tests cpp/algorithms_insertion_sort_trace_tests.cpp cpp/algorithms_comparison_sorts_tests.cpp)
target_link_libraries(run_insertion_sort_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_insertion_sort_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_insertion_sort_trace_tests TEST_PREFIX "insertion_sort_trace.")

add_executable(run_interpolation_search_trace_tests cpp/algorithms_interpolation_search_trace_tests.cpp cpp/algorithms_sorted_search_tests.cpp)
target_link_libraries(run_interpolation_search_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_interpolation_search_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_interpolation_search_trace_tests TEST_PREFIX "interpolation_search_trace.")

add_executable(run_inverse_iteration_trace_tests cpp/algorithms_inverse_iteration_trace_tests.cpp cpp/eigen_batch47_tests.cpp)
target_link_libraries(run_inverse_iteration_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_inverse_iteration_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_inverse_iteration_trace_tests TEST_PREFIX "inverse_iteration_trace." TEST_FILTER "InverseIteration*")

add_executable(run_itp_trace_tests cpp/algorithms_itp_trace_tests.cpp cpp/algorithms_extra_root_finding_tests.cpp)
target_link_libraries(run_itp_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_itp_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_itp_trace_tests TEST_PREFIX "itp_trace.")

add_executable(run_jackknife_trace_tests cpp/stats_jackknife_trace_tests.cpp cpp/stats_resampling_tests.cpp)
target_link_libraries(run_jackknife_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_jackknife_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_jackknife_trace_tests TEST_PREFIX "jackknife_trace.")

add_executable(run_jaro_winkler_trace_tests cpp/algorithms_jaro_winkler_trace_tests.cpp cpp/algorithms_string_metrics_tests.cpp)
target_link_libraries(run_jaro_winkler_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_jaro_winkler_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_jaro_winkler_trace_tests TEST_PREFIX "jaro_winkler_trace.")

add_executable(run_jarvis_trace_tests cpp/geometry_jarvis_trace_tests.cpp cpp/geometry_hull_algorithms_tests.cpp)
target_link_libraries(run_jarvis_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_jarvis_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_jarvis_trace_tests TEST_PREFIX "jarvis_trace.")

add_executable(run_johnson_trace_tests cpp/algorithms_johnson_trace_tests.cpp cpp/algorithms_routing_batch_tests.cpp)
target_link_libraries(run_johnson_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_johnson_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_johnson_trace_tests TEST_PREFIX "johnson_trace.")

add_executable(run_jump_walk_trace_tests cpp/geometry_jump_walk_trace_tests.cpp cpp/geometry_batch38_tests.cpp cpp/format/geometry_format_tests.cpp)
target_link_libraries(run_jump_walk_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_jump_walk_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_jump_walk_trace_tests TEST_PREFIX "jump_walk_trace.")

add_executable(run_jump_point_search_tests cpp/algorithms_jump_point_search_tests.cpp cpp/algorithms_search_more_tests.cpp)
target_link_libraries(run_jump_point_search_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_jump_point_search_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_jump_point_search_tests TEST_PREFIX "jump_point_search.")

add_executable(run_jump_search_trace_tests cpp/algorithms_jump_search_trace_tests.cpp)
target_link_libraries(run_jump_search_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_jump_search_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_jump_search_trace_tests TEST_PREFIX "jump_search_trace.")

add_executable(run_kabsch_analysis_tests cpp/algorithms_kabsch_analysis_tests.cpp cpp/algorithms_calendar_geo_batch_tests.cpp)
target_link_libraries(run_kabsch_analysis_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_kabsch_analysis_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_kabsch_analysis_tests TEST_PREFIX "kabsch_analysis.")

add_executable(run_kadane_trace_tests cpp/algorithms_kadane_trace_tests.cpp cpp/algorithms_substrings_tests.cpp)
target_link_libraries(run_kadane_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_kadane_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_kadane_trace_tests TEST_PREFIX "kadane_trace.")

add_executable(run_kahan_trace_tests cpp/algorithms_kahan_trace_tests.cpp cpp/algorithms_elementary_tests.cpp)
target_link_libraries(run_kahan_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_kahan_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_kahan_trace_tests TEST_PREFIX "kahan_trace.")

add_executable(run_karn_event_tests cpp/algorithms_karn_event_tests.cpp cpp/algorithms_networking_batch_tests.cpp)
target_link_libraries(run_karn_event_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_karn_event_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_karn_event_tests TEST_PREFIX "karn_event.")

add_executable(run_karger_trace_tests cpp/algorithms_karger_trace_tests.cpp cpp/algorithms_advanced_graph_tests.cpp)
target_link_libraries(run_karger_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_karger_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_karger_trace_tests TEST_PREFIX "karger_trace.")

add_executable(run_karatsuba_trace_tests cpp/algorithms_karatsuba_trace_tests.cpp cpp/algorithms_number_theory_tests.cpp)
target_link_libraries(run_karatsuba_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_karatsuba_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_karatsuba_trace_tests TEST_PREFIX "karatsuba_trace.")

add_executable(run_kdtree_trace_tests cpp/geometry_kdtree_trace_tests.cpp cpp/geometry_kdtree_tests.cpp cpp/format/geometry_format_tests.cpp)
target_link_libraries(run_kdtree_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_kdtree_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_kdtree_trace_tests TEST_PREFIX "kdtree_trace.")

add_executable(run_kalman_analysis_tests cpp/filter_kalman_analysis_tests.cpp cpp/filter_kalman_tests.cpp cpp/filter_ekf_ukf_tests.cpp cpp/filter_information_enkf_particle_tests.cpp cpp/filter_alpha_beta_tests.cpp cpp/format/filter_format_tests.cpp)
target_link_libraries(run_kalman_analysis_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_kalman_analysis_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_kalman_analysis_tests TEST_PREFIX "kalman_analysis.")

add_executable(run_k_medoids_trace_tests cpp/algorithms_k_medoids_trace_tests.cpp cpp/algorithms_clustering_batch_tests.cpp)
target_link_libraries(run_k_medoids_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_k_medoids_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_k_medoids_trace_tests TEST_PREFIX "k_medoids_trace.")

add_executable(run_kmp_trace_tests cpp/algorithms_kmp_trace_tests.cpp cpp/algorithms_kmp_tests.cpp)
target_link_libraries(run_kmp_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_kmp_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_kmp_trace_tests TEST_PREFIX "kmp_trace.")

add_executable(run_kb_trace_tests cpp/algorithms_kb_trace_tests.cpp cpp/algorithms_knuth_bendix_tests.cpp)
target_link_libraries(run_kb_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_kb_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_kb_trace_tests TEST_PREFIX "kb_trace.")

add_executable(run_kosaraju_trace_tests cpp/algorithms_kosaraju_trace_tests.cpp cpp/algorithms_graph_connectivity_tests.cpp)
target_link_libraries(run_kosaraju_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_kosaraju_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_kosaraju_trace_tests TEST_PREFIX "kosaraju_trace.")

add_executable(run_kruskal_report_tests cpp/algorithms_kruskal_report_tests.cpp cpp/algorithms_kruskal_tests.cpp cpp/algorithms_euclidean_mst_trace_tests.cpp)
target_link_libraries(run_kruskal_report_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_kruskal_report_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_kruskal_report_tests TEST_PREFIX "kruskal_report.")

add_executable(run_merge_trace_tests cpp/algorithms_merge_trace_tests.cpp)
target_link_libraries(run_merge_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_merge_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_merge_trace_tests TEST_PREFIX "merge_trace.")

add_executable(run_lfg_analysis_tests cpp/random_lagged_fibonacci_analysis_tests.cpp cpp/datamunge_random_tests.cpp)
target_link_libraries(run_lfg_analysis_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lfg_analysis_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lfg_analysis_tests TEST_PREFIX "lfg_analysis.")

add_executable(run_lagrange_trace_tests cpp/algorithms_lagrange_trace_tests.cpp cpp/algorithms_interpolation_tests.cpp cpp/algorithms_interpolation_extra_tests.cpp)
target_link_libraries(run_lagrange_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lagrange_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lagrange_trace_tests TEST_PREFIX "lagrange_trace.")

add_executable(run_lamport_event_tests cpp/algorithms_lamport_event_tests.cpp cpp/algorithms_distributed_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_lamport_event_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lamport_event_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lamport_event_tests TEST_PREFIX "lamport_event.")

add_executable(run_lanczos_analysis_tests cpp/algorithms_lanczos_analysis_tests.cpp cpp/krylov_batch48_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_lanczos_analysis_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lanczos_analysis_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lanczos_analysis_tests TEST_PREFIX "lanczos_analysis.")

add_executable(run_lanczos_resampling_trace_tests cpp/algorithms_lanczos_resampling_trace_tests.cpp cpp/interpolation_batch39_tests.cpp)
target_link_libraries(run_lanczos_resampling_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lanczos_resampling_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lanczos_resampling_trace_tests TEST_PREFIX "lanczos_resampling_trace.")

add_executable(run_laplacian_trace_tests cpp/geometry_laplacian_trace_tests.cpp cpp/geometry_batch38_tests.cpp)
target_link_libraries(run_laplacian_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_laplacian_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_laplacian_trace_tests TEST_PREFIX "laplacian_trace.")

add_executable(run_lax_wendroff_trace_tests cpp/algorithms_lax_wendroff_trace_tests.cpp)
target_link_libraries(run_lax_wendroff_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lax_wendroff_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lax_wendroff_trace_tests TEST_PREFIX "lax_wendroff_trace.")

add_executable(run_lcs_trace_tests cpp/algorithms_lcs_trace_tests.cpp)
target_link_libraries(run_lcs_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lcs_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lcs_trace_tests TEST_PREFIX "lcs_trace.")

add_executable(run_lda_report_tests cpp/stats_lda_report_tests.cpp cpp/stats_lda_tests.cpp)
target_link_libraries(run_lda_report_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lda_report_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lda_report_tests TEST_PREFIX "lda_report.")

add_executable(run_leaky_bucket_trace_tests cpp/algorithms_leaky_bucket_trace_tests.cpp)
target_link_libraries(run_leaky_bucket_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_leaky_bucket_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_leaky_bucket_trace_tests TEST_PREFIX "leaky_bucket_trace.")

add_executable(run_lenstra_ecm_trace_tests cpp/algorithms_lenstra_ecm_trace_tests.cpp)
target_link_libraries(run_lenstra_ecm_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lenstra_ecm_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lenstra_ecm_trace_tests TEST_PREFIX "lenstra_ecm_trace.")

add_executable(run_levenshtein_trace_tests cpp/algorithms_levenshtein_trace_tests.cpp)
target_link_libraries(run_levenshtein_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_levenshtein_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_levenshtein_trace_tests TEST_PREFIX "levenshtein_trace.")

add_executable(run_levenshtein_coding_trace_tests cpp/algorithms_levenshtein_coding_trace_tests.cpp cpp/algorithms_coding2_batch_tests.cpp)
target_link_libraries(run_levenshtein_coding_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_levenshtein_coding_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_levenshtein_coding_trace_tests TEST_PREFIX "levenshtein_coding_trace.")

add_executable(run_levinson_trace_tests cpp/algorithms_levinson_trace_tests.cpp)
target_link_libraries(run_levinson_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_levinson_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_levinson_trace_tests TEST_PREFIX "levinson_trace.")

add_executable(run_lex_bfs_trace_tests cpp/algorithms_lex_bfs_trace_tests.cpp cpp/algorithms_search_order_tests.cpp)
target_link_libraries(run_lex_bfs_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lex_bfs_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lex_bfs_trace_tests TEST_PREFIX "lex_bfs_trace.")

add_executable(run_library_sort_trace_tests cpp/algorithms_library_sort_trace_tests.cpp cpp/misc_batch46_tests.cpp)
target_link_libraries(run_library_sort_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_library_sort_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_library_sort_trace_tests TEST_PREFIX "library_sort_trace.")

add_executable(run_lbg_trace_tests cpp/algorithms_lbg_trace_tests.cpp)
target_link_libraries(run_lbg_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lbg_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lbg_trace_tests TEST_PREFIX "lbg_trace.")

add_executable(run_lcg_analysis_tests cpp/random_linear_congruential_analysis_tests.cpp cpp/datamunge_random_tests.cpp)
target_link_libraries(run_lcg_analysis_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lcg_analysis_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lcg_analysis_tests TEST_PREFIX "lcg_analysis.")

add_executable(run_linear_interpolation_trace_tests cpp/algorithms_linear_interpolation_trace_tests.cpp cpp/algorithms_monotone_cubic_trace_tests.cpp cpp/algorithms_interpolation_extra_tests.cpp)
target_link_libraries(run_linear_interpolation_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_linear_interpolation_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_linear_interpolation_trace_tests TEST_PREFIX "linear_interpolation_trace.")

add_executable(run_lis_trace_tests cpp/algorithms_lis_trace_tests.cpp cpp/algorithms_subsequences_tests.cpp)
target_link_libraries(run_lis_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lis_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lis_trace_tests TEST_PREFIX "lis_trace.")

add_executable(run_ll1_trace_tests cpp/algorithms_ll1_trace_tests.cpp cpp/parsing_batch25_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_ll1_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_ll1_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_ll1_trace_tests TEST_PREFIX "ll1_trace.")

add_executable(run_lll_trace_tests cpp/algorithms_lll_trace_tests.cpp)
target_link_libraries(run_lll_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lll_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lll_trace_tests TEST_PREFIX "lll_trace.")

add_executable(run_logitboost_trace_tests cpp/algorithms_logitboost_trace_tests.cpp)
target_link_libraries(run_logitboost_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_logitboost_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_logitboost_trace_tests TEST_PREFIX "logitboost_trace.")

add_executable(run_long_division_trace_tests cpp/algorithms_long_division_trace_tests.cpp)
target_link_libraries(run_long_division_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_long_division_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_long_division_trace_tests TEST_PREFIX "long_division_trace.")

add_executable(run_lcsubstring_trace_tests cpp/algorithms_lcsubstring_trace_tests.cpp cpp/algorithms_substrings_tests.cpp)
target_link_libraries(run_lcsubstring_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lcsubstring_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lcsubstring_trace_tests TEST_PREFIX "lcsubstring_trace.")

add_executable(run_lsh_trace_tests cpp/algorithms_lsh_trace_tests.cpp cpp/algorithms_sampling_batch_tests.cpp)
target_link_libraries(run_lsh_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lsh_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lsh_trace_tests TEST_PREFIX "lsh_trace.")

add_executable(run_lucas_primality_trace_tests cpp/algorithms_lucas_primality_trace_tests.cpp cpp/algorithms_primality_tests.cpp)
target_link_libraries(run_lucas_primality_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lucas_primality_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lucas_primality_trace_tests TEST_PREFIX "lucas_primality_trace.")

add_executable(run_luhn_trace_tests cpp/algorithms_luhn_trace_tests.cpp cpp/algorithms_error_detection_batch_tests.cpp)
target_link_libraries(run_luhn_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_luhn_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_luhn_trace_tests TEST_PREFIX "luhn_trace.")

add_executable(run_lz77_trace_tests cpp/algorithms_lz77_trace_tests.cpp)
target_link_libraries(run_lz77_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lz77_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lz77_trace_tests TEST_PREFIX "lz77_trace.")

add_executable(run_lzw_trace_tests cpp/algorithms_lzw_trace_tests.cpp cpp/algorithms_compression2_batch_tests.cpp)
target_link_libraries(run_lzw_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_lzw_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_lzw_trace_tests TEST_PREFIX "lzw_trace.")

add_executable(run_manacher_trace_tests cpp/algorithms_manacher_trace_tests.cpp cpp/algorithms_manacher_tests.cpp)
target_link_libraries(run_manacher_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_manacher_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_manacher_trace_tests TEST_PREFIX "manacher_trace.")

add_executable(run_marching_squares_trace_tests cpp/algorithms_marching_squares_trace_tests.cpp cpp/graphics_batch26_tests.cpp)
target_link_libraries(run_marching_squares_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_marching_squares_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_marching_squares_trace_tests TEST_PREFIX "marching_squares_trace.")

add_executable(run_mark_compact_trace_tests cpp/algorithms_object_liveness_trace_tests.cpp cpp/algorithms_mark_compact_trace_tests.cpp cpp/algorithms_memory_batch_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_mark_compact_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mark_compact_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mark_compact_trace_tests TEST_PREFIX "mark_compact_trace.")

add_executable(run_mark_sweep_trace_tests cpp/algorithms_mark_sweep_trace_tests.cpp)
target_link_libraries(run_mark_sweep_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mark_sweep_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mark_sweep_trace_tests TEST_PREFIX "mark_sweep_trace.")

# Batch224–233: prepare all ten canonical APIs/reports before shared compilation.
add_executable(run_matrix_chain_trace_tests cpp/algorithms_matrix_chain_trace_tests.cpp cpp/algorithms_combinatorial_search_tests.cpp)
target_link_libraries(run_matrix_chain_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_matrix_chain_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_matrix_chain_trace_tests TEST_PREFIX "matrix_chain_trace.")

add_executable(run_midpoint_circle_trace_tests cpp/algorithms_midpoint_circle_trace_tests.cpp cpp/algorithms_graphics_batch_tests.cpp)
target_link_libraries(run_midpoint_circle_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_midpoint_circle_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_midpoint_circle_trace_tests TEST_PREFIX "midpoint_circle_trace.")

add_executable(run_median_filter_trace_tests cpp/algorithms_median_filter_trace_tests.cpp cpp/algorithms_image_batch_tests.cpp)
target_link_libraries(run_median_filter_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_median_filter_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_median_filter_trace_tests TEST_PREFIX "median_filter_trace.")

add_executable(run_merge_sort_trace_tests cpp/algorithms_merge_sort_trace_tests.cpp)
target_link_libraries(run_merge_sort_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_merge_sort_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_merge_sort_trace_tests TEST_PREFIX "merge_sort_trace.")

add_executable(run_fitch_trace_tests cpp/algorithms_fitch_trace_tests.cpp)
target_link_libraries(run_fitch_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_fitch_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_fitch_trace_tests TEST_PREFIX "fitch_trace.")

add_executable(run_md5_trace_tests cpp/algorithms_md5_trace_tests.cpp)
target_link_libraries(run_md5_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_md5_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_md5_trace_tests TEST_PREFIX "md5_trace.")

add_executable(run_match_rating_trace_tests cpp/algorithms_match_rating_trace_tests.cpp)
target_link_libraries(run_match_rating_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_match_rating_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_match_rating_trace_tests TEST_PREFIX "match_rating_trace.")

add_executable(run_metaphone_trace_tests cpp/algorithms_metaphone_trace_tests.cpp cpp/algorithms_phonetic_tests.cpp)
target_link_libraries(run_metaphone_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_metaphone_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_metaphone_trace_tests TEST_PREFIX "metaphone_trace.")

add_executable(run_max_clique_trace_tests cpp/algorithms_max_clique_trace_tests.cpp cpp/algorithms_graph_hard_tests.cpp)
target_link_libraries(run_max_clique_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_max_clique_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_max_clique_trace_tests TEST_PREFIX "max_clique_trace.")

add_executable(run_memetic_trace_tests cpp/algorithms_memetic_trace_tests.cpp cpp/heuristic_batch50_tests.cpp cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_memetic_trace_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_memetic_trace_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_memetic_trace_tests TEST_PREFIX "memetic_trace.")

add_executable(run_mepml_batch234_tests
  cpp/algorithms_miller_rabin_tests.cpp
  cpp/algorithms_modular_exponentiation_tests.cpp
  cpp/algorithms_montgomery_trace_tests.cpp
  cpp/algorithms_minimum_degree_trace_tests.cpp
  cpp/algorithms_min_conflicts_trace_tests.cpp
  cpp/algorithms_monotone_cubic_trace_tests.cpp
  cpp/geometry_bounding_rectangle_trace_tests.cpp
  cpp/geometry_enclosing_circle_trace_tests.cpp
  cpp/geometry_min_enclosing_circle_tests.cpp
  cpp/geometry_rotating_calipers_tests.cpp
  cpp/algorithms_miser_trace_tests.cpp
  cpp/numerical_batch42_tests.cpp
  cpp/optim_moead_trace_tests.cpp
  cpp/optim_multi_objective_tests.cpp)
target_link_libraries(run_mepml_batch234_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch234_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch234_tests TEST_PREFIX "mepml_batch234.")
target_sources(run_mepml_batch234_tests PRIVATE
  cpp/algorithms_sparse_linalg_tests.cpp
  cpp/algorithms_csp_convex_tests.cpp
  cpp/algorithms_arithmetic_extra_tests.cpp
  cpp/algorithms_interpolation_extra_tests.cpp
  cpp/format/algorithms_format_tests.cpp
  cpp/format/geometry_format_tests.cpp
  cpp/format/optim_format_tests.cpp)

add_executable(run_mepml_batch244_tests
 cpp/algorithms_moore_trace_tests.cpp
 cpp/algorithms_mtf_trace_tests.cpp
 cpp/algorithms_muller_trace_tests.cpp
 cpp/algorithms_nagle_trace_tests.cpp
 cpp/algorithms_nearest_neighbor_trace_tests.cpp
 cpp/algorithms_nearest_tsp_trace_tests.cpp
 cpp/algorithms_needleman_wunsch_trace_tests.cpp
 cpp/algorithms_nested_sampling_trace_tests.cpp
 cpp/algorithms_nesting_trace_tests.cpp
 cpp/nn_tabular_classification_tests.cpp
 cpp/algorithms_logic_automata_batch_tests.cpp
 cpp/algorithms_compression2_batch_tests.cpp
 cpp/algorithms_extra_root_finding_tests.cpp
 cpp/algorithms_networking_batch_tests.cpp
 cpp/algorithms_tour_search_tests.cpp
 cpp/algorithms_sequence_alignment_tests.cpp
 cpp/algorithms_boosting_bayes_batch_tests.cpp
 cpp/krylov_batch48_tests.cpp
 cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_mepml_batch244_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch244_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch244_tests TEST_PREFIX "mepml_batch244.")

add_executable(run_mepml_batch254_tests
 cpp/algorithms_neville_trace_tests.cpp
 cpp/algorithms_newell_trace_tests.cpp
 cpp/algorithms_newton_division_trace_tests.cpp
 cpp/algorithms_newton_raphson_trace_tests.cpp
 cpp/algorithms_division_trace_tests.cpp
 cpp/algorithms_nonblocking_switch_trace_tests.cpp
 cpp/optim_nsga2_trace_tests.cpp
 cpp/algorithms_nth_root_trace_tests.cpp
 cpp/algorithms_nysiis_trace_tests.cpp
 cpp/algorithms_division_tests.cpp
 cpp/algorithms_phonetic_tests.cpp
 cpp/optim_multi_objective_tests.cpp
 cpp/algorithms_extra_root_finding_tests.cpp
 cpp/algorithms_interpolation_extra_tests.cpp
 cpp/algorithms_networking_batch_tests.cpp
 cpp/format/algorithms_format_tests.cpp
 cpp/format/optim_format_tests.cpp)
target_link_libraries(run_mepml_batch254_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch254_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch254_tests TEST_PREFIX "mepml_batch254.")

add_executable(run_mepml_batch264_tests
 cpp/algorithms_odd_even_sort_trace_tests.cpp
 cpp/algorithms_odds_trace_tests.cpp
 cpp/ode_trace_tests.cpp
 cpp/algorithms_one_r_trace_tests.cpp
 cpp/algorithms_online_variance_trace_tests.cpp
 cpp/algorithms_optics_trace_tests.cpp
 cpp/algorithms_ordered_dither_trace_tests.cpp
 cpp/algorithms_package_merge_trace_tests.cpp
 cpp/algorithms_packrat_trace_tests.cpp
 cpp/algorithms_page_rank_trace_tests.cpp
 cpp/ode_solver_tests.cpp
 cpp/algorithms_clustering_batch_tests.cpp
 cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_mepml_batch264_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch264_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch264_tests TEST_PREFIX "mepml_batch264.")

add_executable(run_mepml_batch274_tests
 cpp/mepml_batch274_tests.cpp
 cpp/algorithms_exchange_sorts_tests.cpp
 cpp/algorithms_sorting_extra_tests.cpp
 cpp/algorithms_online_linear_classifiers_tests.cpp
 cpp/algorithms_recurrent_pcnn_batch_tests.cpp
 cpp/algorithms_parsing_hashing_batch_tests.cpp
 cpp/crypto_batch24_tests.cpp
 cpp/graphics_batch26_tests.cpp
 cpp/datamunge_random_tests.cpp)
target_link_libraries(run_mepml_batch274_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch274_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch274_tests TEST_PREFIX "mepml_batch274.")

add_executable(run_mepml_batch284_tests
 cpp/mepml_batch284_tests.cpp
 cpp/algorithms_quantum_batch_tests.cpp
 cpp/algorithms_distribution_sorts_tests.cpp
 cpp/algorithms_factorization_tests.cpp
 cpp/algorithms_pollard_rho_tests.cpp
 cpp/algorithms_number_theory_more_tests.cpp
 cpp/algorithms_algebra_batch_tests.cpp
 cpp/geometry_polygon_triangulation_tests.cpp
 cpp/geometry_polygon_and_segment_tests.cpp
 cpp/bspline_regression_tests.cpp
 cpp/format/algorithms_format_tests.cpp)
target_link_libraries(run_mepml_batch284_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch284_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch284_tests TEST_PREFIX "mepml_batch284.")

add_executable(run_mepml_batch294_tests
 cpp/mepml_batch294_tests.cpp
 cpp/algorithms_polynomial_gf_tests.cpp
 cpp/algorithms_sorting_dist_tests.cpp
 cpp/algorithms_linear_algebra_tests.cpp
 cpp/parsing_batch25_tests.cpp
 cpp/algorithms_prim_tests.cpp
 cpp/algorithms_prufer_tests.cpp
 cpp/algorithms_advanced_graph_tests.cpp
 cpp/algorithms_dinic_trace_tests.cpp
 cpp/algorithms_quantum_batch_tests.cpp
 cpp/algorithms_tabular_rl_tests.cpp
 cpp/mepml_batch284_tests.cpp)
target_link_libraries(run_mepml_batch294_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch294_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch294_tests TEST_PREFIX "mepml_batch294.")

add_executable(run_mepml_batch304_tests cpp/mepml_batch304_tests.cpp)
target_link_libraries(run_mepml_batch304_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch304_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch304_tests TEST_PREFIX "mepml_batch304.")

add_executable(run_mepml_batch304_regression_tests
 cpp/algorithms_graphics_batch_tests.cpp
 cpp/algorithms_rabin_karp_tests.cpp
 cpp/algorithms_quicksort_tests.cpp
 cpp/algorithms_quickselect_tests.cpp
 cpp/algorithms_quadratic_sieve_tests.cpp
 cpp/eigen_batch47_tests.cpp
 cpp/geometry_batch2_tests.cpp
 cpp/algorithms_radix_sort_tests.cpp
 cpp/algorithms_logic_automata_batch_tests.cpp
 cpp/krylov_batch48_tests.cpp
 cpp/datamunge_random_tests.cpp)
target_link_libraries(run_mepml_batch304_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch304_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch304_regression_tests TEST_PREFIX "mepml_batch304_regression.")

add_executable(run_mepml_batch314_tests cpp/mepml_batch314_tests.cpp cpp/eigen_batch47_tests.cpp)
target_link_libraries(run_mepml_batch314_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch314_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch314_tests TEST_PREFIX "mepml_batch314.")

add_executable(run_mepml_batch314_regression_tests
 cpp/algorithms_division_trace_tests.cpp
 cpp/algorithms_division_tests.cpp
 cpp/algorithms_scheduling_batch_tests.cpp
 cpp/algorithms_coding2_batch_tests.cpp
 cpp/algorithms_edf_trace_tests.cpp
 cpp/algorithms_estimation_batch_tests.cpp
 cpp/algorithms_neural_batch_tests.cpp
 cpp/algorithms_memory_batch_tests.cpp
 cpp/crypto_batch24_tests.cpp
 cpp/heuristic_batch50_tests.cpp
 cpp/parsing_batch25_tests.cpp)
target_link_libraries(run_mepml_batch314_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch314_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch314_regression_tests TEST_PREFIX "mepml_batch314_regression.")

add_executable(run_mepml_batch324_tests cpp/mepml_batch324_tests.cpp)
target_link_libraries(run_mepml_batch324_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch324_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch324_tests TEST_PREFIX "mepml_batch324.")

add_executable(run_mepml_batch324_regression_tests
 cpp/algorithms_golomb_tests.cpp
 cpp/algorithms_compression_graphics_batch_tests.cpp
 cpp/algorithms_ode_tests.cpp
 cpp/algorithms_bwt_trace_tests.cpp
 cpp/algorithms_scheduling_batch_tests.cpp
 cpp/algorithms_coding_batch_tests.cpp
 cpp/algorithms_routing_batch_tests.cpp
 cpp/algorithms_root_finding_tests.cpp
 cpp/arithmetic_batch40_tests.cpp
 cpp/algorithms_subsequences_tests.cpp
 cpp/geometry_rotating_calipers_tests.cpp
 cpp/format/algorithms_format_tests.cpp
 cpp/selection_batch49_tests.cpp)
target_link_libraries(run_mepml_batch324_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch324_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch324_regression_tests TEST_PREFIX "mepml_batch324_regression.")

add_executable(run_mepml_batch334_tests cpp/mepml_batch334_tests.cpp)
target_link_libraries(run_mepml_batch334_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch334_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch334_tests TEST_PREFIX "mepml_batch334.")

add_executable(run_mepml_batch334_regression_tests
 cpp/algorithms_advanced_ml_batch_tests.cpp
 cpp/algorithms_tabular_rl_tests.cpp
 cpp/crypto_batch24_tests.cpp
 cpp/graphics_batch26_tests.cpp
 cpp/algorithms_distribution_sorts_tests.cpp
 cpp/algorithms_extra_root_finding_tests.cpp
 cpp/algorithms_root_finding_tests.cpp
 cpp/algorithms_permutations_tests.cpp
 cpp/algorithms_schreier_sims_tests.cpp
 cpp/algorithms_multiplication_tests.cpp
 cpp/algorithms_subsequences_tests.cpp)
target_link_libraries(run_mepml_batch334_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch334_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch334_regression_tests TEST_PREFIX "mepml_batch334_regression.")

add_executable(run_mepml_batch344_tests cpp/mepml_batch344_tests.cpp)
target_link_libraries(run_mepml_batch344_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch344_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch344_tests TEST_PREFIX "mepml_batch344.")

add_executable(run_mepml_batch344_regression_tests
 cpp/algorithms_compression2_batch_tests.cpp
 cpp/algorithms_coding2_batch_tests.cpp
 cpp/datamunge_random_tests.cpp
 cpp/crypto_batch23_tests.cpp
 cpp/crypto_batch24_tests.cpp
 cpp/algorithms_neural_batch_tests.cpp
 cpp/geometry_batch1_utility_tests.cpp
 cpp/geometry_polygon_and_segment_tests.cpp
 cpp/algorithms_hmac_trace_tests.cpp
 cpp/format/algorithms_format_tests.cpp
 cpp/format/random_format_tests.cpp)
target_link_libraries(run_mepml_batch344_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch344_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch344_regression_tests TEST_PREFIX "mepml_batch344_regression.")

add_executable(run_mepml_batch354_tests cpp/mepml_batch354_tests.cpp)
target_link_libraries(run_mepml_batch354_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch354_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch354_tests TEST_PREFIX "mepml_batch354.")
add_executable(run_mepml_batch354_regression_tests
 cpp/algorithms_bioinformatics_batch_tests.cpp
 cpp/geometry_polygon_and_segment_tests.cpp
 cpp/algorithms_number_theory_tests.cpp)
target_link_libraries(run_mepml_batch354_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch354_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch354_regression_tests TEST_PREFIX "mepml_batch354_regression.")

add_executable(run_mepml_batch364_tests cpp/mepml_batch364_tests.cpp)
target_link_libraries(run_mepml_batch364_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch364_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch364_tests TEST_PREFIX "mepml_batch364.")
add_executable(run_mepml_batch364_regression_tests
 cpp/algorithms_elementary_more_tests.cpp cpp/algorithms_phonetic_tests.cpp
 cpp/algorithms_sequence_alignment_tests.cpp cpp/algorithms_linear_solvers_tests.cpp
 cpp/parsing_batch25_tests.cpp cpp/optim_multi_objective_tests.cpp)
target_link_libraries(run_mepml_batch364_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch364_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch364_regression_tests TEST_PREFIX "mepml_batch364_regression.")

add_executable(run_mepml_batch374_tests cpp/mepml_batch374_tests.cpp)
target_link_libraries(run_mepml_batch374_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch374_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch374_tests TEST_PREFIX "mepml_batch374.")
add_executable(run_mepml_batch374_regression_tests
 cpp/algorithms_division_tests.cpp cpp/algorithms_game_search_tests.cpp
 cpp/algorithms_permutations_tests.cpp cpp/algorithms_linear_algebra_tests.cpp
 cpp/algorithms_graph_hard_tests.cpp cpp/algorithms_sorting_hybrid_tests.cpp
 cpp/algorithms_sorting_extra_tests.cpp)
target_link_libraries(run_mepml_batch374_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch374_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch374_regression_tests TEST_PREFIX "mepml_batch374_regression.")

add_executable(run_mepml_batch384_tests cpp/mepml_batch384_tests.cpp)
target_link_libraries(run_mepml_batch384_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch384_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch384_tests TEST_PREFIX "mepml_batch384.")
add_executable(run_mepml_batch384_regression_tests
 cpp/algorithms_combinatorial_extra_tests.cpp cpp/algorithms_graphics_batch_tests.cpp
 cpp/algorithms_graph_connectivity_tests.cpp cpp/crypto_batch24_tests.cpp
 cpp/algorithms_merge_ternary_bitap_tests.cpp cpp/algorithms_primality_tests.cpp
 cpp/algorithms_tarjan_lca_tests.cpp cpp/algorithms_tabular_rl_tests.cpp
 cpp/algorithms_sparse_linalg_tests.cpp)
target_link_libraries(run_mepml_batch384_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch384_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch384_regression_tests TEST_PREFIX "mepml_batch384_regression.")

add_executable(run_mepml_batch394_tests cpp/mepml_batch394_tests.cpp)
target_link_libraries(run_mepml_batch394_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch394_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch394_tests TEST_PREFIX "mepml_batch394.")
add_executable(run_mepml_batch394_regression_tests
 cpp/algorithms_linear_solvers_tests.cpp cpp/algorithms_sorting_hybrid_tests.cpp
 cpp/algorithms_todd_coxeter_tests.cpp cpp/algorithms_number_theory_tests.cpp
 cpp/algorithms_multiplication_tests.cpp cpp/algorithms_topological_sort_tests.cpp
 cpp/selection_batch49_tests.cpp cpp/algorithms_routing_batch_tests.cpp
 cpp/algorithms_pde_solvers_tests.cpp)
target_link_libraries(run_mepml_batch394_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch394_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch394_regression_tests TEST_PREFIX "mepml_batch394_regression.")

add_executable(run_mepml_batch404_tests cpp/mepml_batch404_tests.cpp)
target_link_libraries(run_mepml_batch404_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch404_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch404_tests TEST_PREFIX "mepml_batch404." PROPERTIES TIMEOUT 10)
add_executable(run_mepml_batch404_regression_tests
 cpp/algorithms_distribution_sorts_tests.cpp cpp/algorithms_factorization_tests.cpp
 cpp/algorithms_substrings_tests.cpp cpp/stats_tsne_tests.cpp cpp/stats_umap_tests.cpp cpp/algorithms_advanced_graph_tests.cpp cpp/algorithms_coding2_batch_tests.cpp cpp/numerical_batch41_tests.cpp cpp/algorithms_search_techniques_tests.cpp cpp/algorithms_networking_batch_tests.cpp)
target_link_libraries(run_mepml_batch404_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch404_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch404_regression_tests TEST_PREFIX "mepml_batch404_regression.")

add_executable(run_mepml_batch414_tests cpp/mepml_batch414_tests.cpp)
target_link_libraries(run_mepml_batch414_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch414_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch414_tests TEST_PREFIX "mepml_batch414.")
add_executable(run_mepml_batch414_regression_tests
 cpp/algorithms_merge_ternary_bitap_tests.cpp cpp/algorithms_coding2_batch_tests.cpp
  
  cpp/algorithms_bioinformatics_batch_tests.cpp
  cpp/algorithms_ode_tests.cpp cpp/algorithms_calendar_geo_batch_tests.cpp cpp/algorithms_distributed_batch_tests.cpp cpp/algorithms_error_detection_batch_tests.cpp cpp/algorithms_search_order_tests.cpp cpp/algorithms_search_techniques_tests.cpp)
target_link_libraries(run_mepml_batch414_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch414_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch414_regression_tests TEST_PREFIX "mepml_batch414_regression.")

add_executable(run_mepml_batch424_tests cpp/mepml_batch424_tests.cpp)
target_link_libraries(run_mepml_batch424_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch424_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch424_tests TEST_PREFIX "mepml_batch424.")
add_executable(run_mepml_batch424_regression_tests cpp/algorithms_hmm_tests.cpp
 cpp/algorithms_forward_backward_analysis_tests.cpp cpp/numerical_batch43_tests.cpp
 cpp/stats_resampling_tests.cpp cpp/geometry_delaunay_tests.cpp)
target_link_libraries(run_mepml_batch424_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_mepml_batch424_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_mepml_batch424_regression_tests TEST_PREFIX "mepml_batch424_regression.")

add_executable(run_nn_diffusion_components_tests cpp/nn_diffusion_components_tests.cpp)
target_link_libraries(run_nn_diffusion_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_diffusion_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_diffusion_components_tests TEST_PREFIX "nn-diffusion-components.")

add_executable(run_nn_recurrent_components_tests cpp/nn_recurrent_components_tests.cpp)
target_link_libraries(run_nn_recurrent_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_recurrent_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_recurrent_components_tests TEST_PREFIX "nn-recurrent-components.")

add_executable(run_nn_recurrent_extensions_tests cpp/nn_recurrent_extensions_tests.cpp)
target_link_libraries(run_nn_recurrent_extensions_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_recurrent_extensions_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_recurrent_extensions_tests TEST_PREFIX "nn-recurrent-extensions.")

add_executable(run_nn_autorec_tests cpp/nn_autorec_tests.cpp)
target_link_libraries(run_nn_autorec_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_autorec_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_autorec_tests TEST_PREFIX "nn-autorec.")

add_executable(run_nn_energy_components_tests cpp/nn_energy_components_tests.cpp)
target_link_libraries(run_nn_energy_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_energy_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_energy_components_tests TEST_PREFIX "nn-energy-components.")

add_executable(run_nn_dnc_components_tests cpp/nn_dnc_components_tests.cpp)
target_link_libraries(run_nn_dnc_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_dnc_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_dnc_components_tests TEST_PREFIX "nn-dnc-components.")

add_executable(run_nn_attention_components_tests cpp/nn_attention_components_tests.cpp)
target_link_libraries(run_nn_attention_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_attention_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_attention_components_tests TEST_PREFIX "nn-attention-components.")

add_executable(run_rl_environment_components_tests cpp/rl_environment_components_tests.cpp)
target_link_libraries(run_rl_environment_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_rl_environment_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_rl_environment_components_tests TEST_PREFIX "rl-environment-components.")

add_executable(run_rl_foundation_components_tests cpp/rl_foundation_components_tests.cpp)
target_link_libraries(run_rl_foundation_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_rl_foundation_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_rl_foundation_components_tests TEST_PREFIX "rl-foundation-components.")

add_executable(run_nn_deep_q_components_tests cpp/nn_deep_q_components_tests.cpp)
target_link_libraries(run_nn_deep_q_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_deep_q_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_deep_q_components_tests TEST_PREFIX "nn-deep-q-components.")

add_executable(run_nn_policy_gradient_components_tests cpp/nn_policy_gradient_components_tests.cpp)
target_link_libraries(run_nn_policy_gradient_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_policy_gradient_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_policy_gradient_components_tests TEST_PREFIX "nn-policy-gradient-components.")

add_executable(run_nn_continuous_off_policy_components_tests cpp/nn_continuous_off_policy_components_tests.cpp)
target_link_libraries(run_nn_continuous_off_policy_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_continuous_off_policy_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_continuous_off_policy_components_tests TEST_PREFIX "nn-continuous-off-policy-components.")

add_executable(run_nn_planning_components_tests cpp/nn_planning_components_tests.cpp)
target_link_libraries(run_nn_planning_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_planning_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_planning_components_tests TEST_PREFIX "nn-planning-components.")

add_executable(run_nn_pixel_environment_components_tests cpp/nn_pixel_environment_components_tests.cpp)
target_link_libraries(run_nn_pixel_environment_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_pixel_environment_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_pixel_environment_components_tests TEST_PREFIX "nn-pixel-environment-components.")

add_executable(run_nn_recovered_language_tests cpp/nn_recovered_language_tests.cpp)
target_link_libraries(run_nn_recovered_language_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_recovered_language_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_recovered_language_tests TEST_PREFIX "nn-recovered-language.")

add_executable(run_nn_completion_components_tests cpp/nn_completion_components_tests.cpp)
target_link_libraries(run_nn_completion_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_completion_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_completion_components_tests TEST_PREFIX "nn-completion.")

add_executable(run_nn_classical_components_tests cpp/nn_classical_components_tests.cpp)
target_link_libraries(run_nn_classical_components_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_nn_classical_components_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_classical_components_tests TEST_PREFIX "nn-classical-components.")

add_executable(run_nn_binding_compatibility_tests cpp/nn_binding_compatibility_tests.cpp)
target_link_libraries(run_nn_binding_compatibility_tests datamunge::datamunge GTest::gtest_main)
target_include_directories(run_nn_binding_compatibility_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../prebindings/nn/include)
set_target_properties(run_nn_binding_compatibility_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_nn_binding_compatibility_tests TEST_PREFIX "nn-binding-compatibility.")

add_executable(run_causal_inference_tests cpp/causal_inference_tests.cpp)
target_link_libraries(run_causal_inference_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_causal_inference_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_causal_inference_tests TEST_PREFIX "causal.")

add_executable(run_causal_matching_tests cpp/causal_matching_tests.cpp)
target_link_libraries(run_causal_matching_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_causal_matching_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_causal_matching_tests TEST_PREFIX "causal.")

add_executable(run_causal_weighting_tests cpp/causal_weighting_tests.cpp)
target_link_libraries(run_causal_weighting_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_causal_weighting_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_causal_weighting_tests TEST_PREFIX "causal.")

add_executable(run_causal_aipw_tests cpp/causal_aipw_tests.cpp)
target_link_libraries(run_causal_aipw_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_causal_aipw_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_causal_aipw_tests TEST_PREFIX "causal.")

add_executable(run_causal_msm_tests cpp/causal_msm_tests.cpp)
target_link_libraries(run_causal_msm_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_causal_msm_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_causal_msm_tests TEST_PREFIX "causal.")

add_executable(run_causal_did_tests cpp/causal_did_tests.cpp)
target_link_libraries(run_causal_did_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_causal_did_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_causal_did_tests TEST_PREFIX "causal.")

add_executable(run_causal_rdd_tests cpp/causal_rdd_tests.cpp)
target_link_libraries(run_causal_rdd_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_causal_rdd_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_causal_rdd_tests TEST_PREFIX "causal.")

add_executable(run_causal_synth_tests cpp/causal_synth_tests.cpp)
target_link_libraries(run_causal_synth_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_causal_synth_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_causal_synth_tests TEST_PREFIX "causal.")

add_executable(run_causal_weak_iv_tests cpp/causal_weak_iv_tests.cpp)
target_link_libraries(run_causal_weak_iv_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_causal_weak_iv_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_causal_weak_iv_tests TEST_PREFIX "causal.")

add_executable(run_stats_gmm_estimation_tests cpp/stats_gmm_estimation_tests.cpp)
target_link_libraries(run_stats_gmm_estimation_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_gmm_estimation_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_gmm_estimation_tests TEST_PREFIX "stats.")

add_executable(run_causal_mediation_tests cpp/causal_mediation_tests.cpp)
target_link_libraries(run_causal_mediation_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_causal_mediation_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_causal_mediation_tests TEST_PREFIX "causal.")

add_executable(run_causal_rosenbaum_tests cpp/causal_rosenbaum_tests.cpp)
target_link_libraries(run_causal_rosenbaum_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_causal_rosenbaum_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_causal_rosenbaum_tests TEST_PREFIX "causal.")

add_executable(run_bayes_psis_loo_tests cpp/bayes_psis_loo_tests.cpp)
target_link_libraries(run_bayes_psis_loo_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bayes_psis_loo_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bayes_psis_loo_tests TEST_PREFIX "bayes.")

add_executable(run_bayes_smc_tests cpp/bayes_smc_tests.cpp)
target_link_libraries(run_bayes_smc_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bayes_smc_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bayes_smc_tests TEST_PREFIX "bayes.")

add_executable(run_bayes_advi_tests cpp/bayes_advi_tests.cpp)
target_link_libraries(run_bayes_advi_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bayes_advi_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bayes_advi_tests TEST_PREFIX "bayes.")

add_executable(run_bayes_sbc_tests cpp/bayes_sbc_tests.cpp)
target_link_libraries(run_bayes_sbc_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_bayes_sbc_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_bayes_sbc_tests TEST_PREFIX "bayes.")

add_executable(run_stats_spatial_autocorrelation_tests cpp/stats_spatial_autocorrelation_tests.cpp)
target_link_libraries(run_stats_spatial_autocorrelation_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_spatial_autocorrelation_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_spatial_autocorrelation_tests TEST_PREFIX "stats.")

add_executable(run_stats_geostatistics_tests cpp/stats_geostatistics_tests.cpp)
target_link_libraries(run_stats_geostatistics_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_geostatistics_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_geostatistics_tests TEST_PREFIX "stats.")

add_executable(run_stats_spatial_regression_tests cpp/stats_spatial_regression_tests.cpp)
target_link_libraries(run_stats_spatial_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_spatial_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_spatial_regression_tests TEST_PREFIX "stats.")

add_executable(run_stats_point_pattern_tests cpp/stats_point_pattern_tests.cpp)
target_link_libraries(run_stats_point_pattern_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_point_pattern_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_point_pattern_tests TEST_PREFIX "stats.")

add_executable(run_stats_hawkes_tests cpp/stats_hawkes_tests.cpp)
target_link_libraries(run_stats_hawkes_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_hawkes_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_hawkes_tests TEST_PREFIX "stats.")

add_executable(run_stats_circular_tests cpp/stats_circular_tests.cpp)
target_link_libraries(run_stats_circular_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_circular_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_circular_tests TEST_PREFIX "stats.")

add_executable(run_stats_meta_analysis_tests cpp/stats_meta_analysis_tests.cpp)
target_link_libraries(run_stats_meta_analysis_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_meta_analysis_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_meta_analysis_tests TEST_PREFIX "stats.")

add_executable(run_stats_network_meta_tests cpp/stats_network_meta_tests.cpp)
target_link_libraries(run_stats_network_meta_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_network_meta_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_network_meta_tests TEST_PREFIX "stats.")

add_executable(run_stats_generalizability_tests cpp/stats_generalizability_tests.cpp)
target_link_libraries(run_stats_generalizability_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_generalizability_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_generalizability_tests TEST_PREFIX "stats.")

add_executable(run_stats_reliability_tests cpp/stats_reliability_tests.cpp)
target_link_libraries(run_stats_reliability_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_reliability_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_reliability_tests TEST_PREFIX "stats.")

add_executable(run_stats_item_response_tests cpp/stats_item_response_tests.cpp)
target_link_libraries(run_stats_item_response_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_item_response_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_item_response_tests TEST_PREFIX "stats.")

add_executable(run_stats_gage_rr_tests cpp/stats_gage_rr_tests.cpp)
target_link_libraries(run_stats_gage_rr_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_gage_rr_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_gage_rr_tests TEST_PREFIX "stats.")

add_executable(run_stats_process_capability_tests cpp/stats_process_capability_tests.cpp)
target_link_libraries(run_stats_process_capability_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_process_capability_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_process_capability_tests TEST_PREFIX "stats.")

add_executable(run_stats_acceptance_sampling_tests cpp/stats_acceptance_sampling_tests.cpp)
target_link_libraries(run_stats_acceptance_sampling_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_acceptance_sampling_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_acceptance_sampling_tests TEST_PREFIX "stats.")

add_executable(run_stats_partial_correlation_tests cpp/stats_partial_correlation_tests.cpp)
target_link_libraries(run_stats_partial_correlation_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_partial_correlation_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_partial_correlation_tests TEST_PREFIX "stats.")

add_executable(run_stats_distance_correlation_tests cpp/stats_distance_correlation_tests.cpp)
target_link_libraries(run_stats_distance_correlation_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_distance_correlation_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_distance_correlation_tests TEST_PREFIX "stats.")

add_executable(run_stats_hsic_tests cpp/stats_hsic_tests.cpp)
target_link_libraries(run_stats_hsic_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_hsic_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_hsic_tests TEST_PREFIX "stats.")

add_executable(run_stats_polychoric_tests cpp/stats_polychoric_tests.cpp)
target_link_libraries(run_stats_polychoric_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_polychoric_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_polychoric_tests TEST_PREFIX "stats.")

add_executable(run_stats_copula_tests cpp/stats_copula_tests.cpp)
target_link_libraries(run_stats_copula_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_copula_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_copula_tests TEST_PREFIX "stats.")

add_executable(run_stats_gev_tests cpp/stats_gev_tests.cpp)
target_link_libraries(run_stats_gev_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_gev_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_gev_tests TEST_PREFIX "stats.")

add_executable(run_stats_gpd_tests cpp/stats_gpd_tests.cpp)
target_link_libraries(run_stats_gpd_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_gpd_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_gpd_tests TEST_PREFIX "stats.")

add_executable(run_stats_tail_index_tests cpp/stats_tail_index_tests.cpp)
target_link_libraries(run_stats_tail_index_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_tail_index_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_tail_index_tests TEST_PREFIX "stats.")

add_executable(run_stats_extremal_index_tests cpp/stats_extremal_index_tests.cpp)
target_link_libraries(run_stats_extremal_index_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_extremal_index_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_extremal_index_tests TEST_PREFIX "stats.")

add_executable(run_stats_compositional_tests cpp/stats_compositional_tests.cpp)
target_link_libraries(run_stats_compositional_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_compositional_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_compositional_tests TEST_PREFIX "stats.")

add_executable(run_stats_dirichlet_regression_tests cpp/stats_dirichlet_regression_tests.cpp)
target_link_libraries(run_stats_dirichlet_regression_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_dirichlet_regression_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_dirichlet_regression_tests TEST_PREFIX "stats.")

add_executable(run_stats_permanova_tests cpp/stats_permanova_tests.cpp)
target_link_libraries(run_stats_permanova_tests datamunge::datamunge GTest::gtest_main)
set_target_properties(run_stats_permanova_tests PROPERTIES CXX_STANDARD 23 CXX_STANDARD_REQUIRED ON)
gtest_discover_tests(run_stats_permanova_tests TEST_PREFIX "stats.")
