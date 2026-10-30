# General-purpose examples, grouped by topic. NN examples are configured separately.
set(EXAMPLES_
    source/playground.cpp
    # Asset-driven offscreen simulation demo; invoked with explicit scene/output paths.
    ../simulation/courier_offscreen.cpp
    # Data
    source/data/datamunge_ex.cpp
    source/data/dataframe_ex.cpp
    source/data/datasets_ex.cpp
    # Linear algebra
    source/linear_algebra/dense_ex.cpp
    source/linear_algebra/linalg_ex.cpp
    source/linear_algebra/tensor_ex.cpp
    # Numerics
    source/numerics/autodiff_ex.cpp
    source/numerics/bspline_ex.cpp
    source/numerics/hierarchical_bspline_ex.cpp
    source/numerics/ode_ex.cpp
    source/numerics/optim_ex.cpp
    # Statistics
    source/statistics/glm_iris_ex.cpp
    source/statistics/glmm_ex.cpp
    source/statistics/hypothesis_tests_ex.cpp
    source/statistics/lm_ex.cpp
    source/statistics/lmm_ex.cpp
    source/statistics/regression_ex.cpp
    # Bayesian
    source/bayesian/bayes_ex.cpp
    source/bayesian/bayes_sampling_ex.cpp
    source/bayesian/inla_ex.cpp
    # Machine learning
    source/machine_learning/clustering_ex.cpp
    source/machine_learning/elastic_net_iris_ex.cpp
    source/machine_learning/forest_iris_ex.cpp
    source/machine_learning/gaussian_process_regression_iris_ex.cpp
    source/machine_learning/gbm_iris_ex.cpp
    source/machine_learning/kernel_regression_iris_ex.cpp
    source/machine_learning/knn_iris_ex.cpp
    source/machine_learning/lda_iris_ex.cpp
    source/machine_learning/mds_ex.cpp
    source/machine_learning/naive_bayes_penguins_ex.cpp
    source/machine_learning/pca_ex.cpp
    source/machine_learning/svm_iris_ex.cpp
    source/machine_learning/tree_iris_ex.cpp
    source/machine_learning/xgboost_iris_ex.cpp
    # Time series
    source/time_series/kalman_filter_ex.cpp
    source/time_series/timeseries_ex.cpp
    # Visualization
    source/visualization/ggplot_ex.cpp
    source/visualization/splom_ex.cpp
    source/visualization/plot_ex.cpp
    source/visualization/rplot_ex.cpp
    # Graphs and geometry
    source/graphs_and_geometry/geometry_ex.cpp
    source/graphs_and_geometry/gis_ex.cpp
    source/graphs_and_geometry/graph_ex.cpp
    # Computer vision
    source/computer_vision/cv_ex.cpp
    source/computer_vision/image_ex.cpp
)
