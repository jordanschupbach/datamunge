
{
  'targets': [
    {
      'target_name': 'datamungejs',
      'sources': [ 'src/datamunge/datamunge.cpp', 'src/datamunge/datamunge_c.cpp', 'src/datamunge/plot.cpp', 'src/datamunge/plot/ggplot.cpp', 'src/datamunge/random.cpp', 'src/datamunge/stats/agglomerative_clustering.cpp', 'src/datamunge/stats/dbscan.cpp', 'src/datamunge/stats/decision_tree_classifier.cpp', 'src/datamunge/stats/decision_tree_regressor.cpp', 'src/datamunge/stats/kmeans.cpp', 'src/datamunge/stats/detail/xgboost_tree.cpp', 'src/datamunge/stats/elastic_net.cpp', 'src/datamunge/stats/formula.cpp', 'src/datamunge/stats/gaussian_process_regression.cpp', 'src/datamunge/stats/gbm_classifier.cpp', 'src/datamunge/stats/gbm_regressor.cpp', 'src/datamunge/stats/glm.cpp', 'src/datamunge/stats/glmm.cpp', 'src/datamunge/stats/inla_mixed_model.cpp', 'src/datamunge/stats/kernel_regression.cpp', 'src/datamunge/stats/knn_classifier.cpp', 'src/datamunge/stats/knn_regressor.cpp', 'src/datamunge/stats/detail/mixed_model_formula.cpp', 'src/datamunge/stats/lda.cpp', 'src/datamunge/stats/lm.cpp', 'src/datamunge/stats/lmm.cpp', 'src/datamunge/linalg/tensor.cpp', 'src/datamunge/autodiff/tape.cpp', 'src/datamunge/optim/function_types.cpp', 'src/datamunge/optim/gradient_descent.cpp', 'src/datamunge/optim/adam.cpp', 'src/datamunge/optim/lbfgs.cpp', 'src/datamunge/optim/sgd.cpp', 'src/datamunge/optim/simulated_annealing.cpp', 'src/datamunge/optim/pso.cpp', 'src/datamunge/optim/differential_evolution.cpp', 'src/datamunge/optim/genetic_algorithm.cpp', 'src/datamunge/bayes/autodiff_model.cpp', 'src/datamunge/bayes/map.cpp', 'src/datamunge/bayes/hmc.cpp', 'src/datamunge/bayes/nuts.cpp', 'src/datamunge/bayes/inla.cpp', 'src/datamunge/bayes/rwm.cpp', 'src/datamunge/bayes/gibbs.cpp', 'src/datamunge/bayes/importance_sampling.cpp', 'src/datamunge/stats/naive_bayes_classifier.cpp', 'src/datamunge/stats/random_forest_classifier.cpp', 'src/datamunge/stats/random_forest_regressor.cpp', 'src/datamunge/stats/svm.cpp', 'src/datamunge/stats/xgboost_classifier.cpp', 'src/datamunge/stats/xgboost_regressor.cpp', 'src/datamunge/datasets/iris.cpp', 'src/datamunge/datasets/penguins.cpp', 'src/datamunge/stats/arima.cpp', 'src/datamunge/stats/exponential_smoothing.cpp', 'src/datamunge/stats/detail/ranking.cpp', 'src/datamunge/stats/t_test.cpp', 'src/datamunge/stats/wilcoxon_test.cpp', 'src/datamunge/stats/ks_test.cpp', 'src/datamunge/stats/chi_squared_test.cpp', 'src/datamunge/stats/anova_test.cpp', 'src/datamunge/stats/correlation_test.cpp', 'src/datamunge/stats/variance_test.cpp', 'src/datamunge/stats/proportion_test.cpp', 'src/datamunge/stats/fisher_exact_test.cpp', 'src/datamunge/stats/normality_test.cpp', 'src/datamunge/stats/p_adjust.cpp', 'src/datamunge/fda/bspline.cpp', 'src/datamunge/image/netpbm.cpp', 'src/datamunge/image/bmp.cpp', 'src/datamunge/image/png.cpp', 'src/datamunge/image/zlib_codec.cpp', 'src/datamunge_js_wrap.cpp' ],
      'include_dirs': [
        "include",
        "<!@(node -p \"require('node-addon-api').include\")",
        '/nix/store/yl9p47yg3qzw1xf9b3pnfav0mgy1qik9-libxml2-2.15.1-dev/include/libxml2'
      ],
      'dependencies': ["<!(node -p \"require('node-addon-api').gyp\")"],
      'cflags': ['-std=c++23'],
      'cflags!': [ '-fno-exceptions', '-fno-rtti'], # ,
      'cflags_cc': ['-std=c++23'],
      'cflags_cc!': [ '-fno-exceptions', '-fno-rtti'], # '-fno-rtti'
      # 'GCC_ENABLE_CPP_RTTI': 'NO',              # -fno-rtti   ???
      'xcode_settings': {
        'GCC_ENABLE_CPP_EXCEPTIONS': 'YES',
        'CLANG_CXX_LIBRARY': 'libc++',
        'MACOSX_DEPLOYMENT_TARGET': '10.14',
        'GCC_ENABLE_CPP_RTTI': 'YES'
      },
      'libraries' : [ '-lxml2' ], #   '-lblas', '-llapack', '-llapacke', '-lcblas'
      'library_dirs' : [ '/usr/lib' ],
      'msvs_settings': {
        'VCCLCompilerTool': { 'ExceptionHandling': 1 },
      }
    }
  ]
}