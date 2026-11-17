classdef MLPClassifierSearch < datamunge.NativeObject
  % MLPClassifierSearch  A two-predictor MLP classifier with Bayesian hidden-width selection and a held-out test.
  %
  %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature)
  %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature, min_hidden)
  %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature, min_hidden, max_hidden)
  %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature, min_hidden, max_hidden, initial_samples)
  %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature, min_hidden, max_hidden, initial_samples, iterations)
  %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature, min_hidden, max_hidden, initial_samples, iterations, epochs)
  %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature, min_hidden, max_hidden, initial_samples, iterations, epochs, seed)
  %
  %   Splits complete labelled data by class into 60% fitting, 20% validation and 20%
  %   test rows (per-class rounding applies). Four random widths followed by eight
  %   Gaussian-process expected-improvement proposals are the default search budget.
  %   Proposals in [min_hidden,max_hidden] are rounded to integer widths; repeated
  %   widths reuse their cached validation log loss. Smaller loss wins; ties retain
  %   the first proposal. Each candidate has one tanh hidden layer, softmax output,
  %   training-only standardization, SGD learning rate 0.05, batch size 32 and L2
  %   penalty 0.0001. Initialization/shuffling use seed for every candidate.
  %
  %   After selection the network is refitted on fitting plus validation rows,
  %   with new standardization statistics, then scored once on the untouched test.
  %   Validation loss belongs to the candidate fit, not the final refit, and is an
  %   optimistic selection score. Neither split nor test performance is a guarantee
  %   for other seeds or datasets. Owns all state, borrows inputs only during calls,
  %   does not change data or global RNG state. Const methods may run concurrently.
  %   Search cost is at most (initial_samples+iterations) fits plus one refit;
  %   cached repeats are cheaper. Prediction/plot costs follow MLPClassifier.
  %
  %   Example:
  %     const std::unique_ptr<dm::DataFrame> iris(dm::data::iris());
  %     dm::MLPClassifierSearch fit(*iris, "Species", "Petal.Length", "Petal.Width");
  %     const bool valid = fit.best_hidden_nodes() >= 2 && fit.best_hidden_nodes() <= 24; // true
  %
  %   References:
  %   - D. R. Jones, M. Schonlau and W. J. Welch (1998), Efficient Global Optimization
  %     of Expensive Black-Box Functions, https://doi.org/10.1023/A:1008306431147
  %
  %   Methods:
  %     best_hidden_nodes  Selected integer width of the single hidden layer.
  %     validation_log_loss  Minimum validation mean negative log probability of the true class.
  %     test_accuracy  Held-out test accuracy of the final development-data refit.
  %     training_rows  Number of rows used to fit each search candidate.
  %     validation_rows  Number of rows used to choose hidden width.
  %     test_rows  Number of untouched rows used for the final evaluation.
  %     summary  Summarize split sizes, selected width, validation loss and final test accuracy.
  %     trials  Copy the complete Bayesian proposal history into an owned dataframe.
  %     predict  Predict labels from the final refit using only its two predictor columns.
  %     plot_decision_regions  Plot final predicted class regions and development observations in the predictor plane.
  %
  %   C++: class datamunge::MLPClassifierSearch
  %
  %   See also: datamunge.MLPClassifier, datamunge.BayesianOptimization, datamunge.train_test_split()
  methods
    function obj = MLPClassifierSearch (varargin)
      % MLPClassifierSearch  Construct a C++ datamunge::MLPClassifierSearch.
      %
      %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature)
      %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature, min_hidden)
      %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature, min_hidden, max_hidden)
      %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature, min_hidden, max_hidden, initial_samples)
      %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature, min_hidden, max_hidden, initial_samples, iterations)
      %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature, min_hidden, max_hidden, initial_samples, iterations, epochs)
      %   obj = datamunge.MLPClassifierSearch (data, response, x_feature, y_feature, min_hidden, max_hidden, initial_samples, iterations, epochs, seed)
      %
      %   Example:
      %     const std::unique_ptr<dm::DataFrame> iris(dm::data::iris());
      %     dm::MLPClassifierSearch fit(*iris, "Species", "Petal.Length", "Petal.Width", 2, 24, 4, 8, 400, 42);
      %     const auto n = fit.test_rows(); // 30
      %
      %   Tune a single hidden layer, refit on development rows and evaluate the test rows.
      %
      %   Inputs:
      %     data  datamunge.DataFrame: Borrowed labelled frame; unchanged. Every selected value must be present and finite.
      %     response  char row: String response column, with nonempty labels and at least five rows per class.
      %     x_feature  char row: First numeric predictor, distinct from response and y_feature.
      %     y_feature  char row: Second numeric predictor; these two are the only model inputs.
      %     min_hidden  integer, default 2: Smallest hidden width, default 2; at least 1 and below max_hidden.
      %     max_hidden  integer, default 24: Largest hidden width, default 24; at most 1000000.
      %     initial_samples  integer, default 4: Random initial proposals, default 4; at least 1.
      %     iterations  integer, default 8: Surrogate-guided proposals, default 8; zero permits initial design only.
      %     epochs  integer, default 400: SGD passes for each candidate and the final refit, default 400; at least 1.
      %     seed  integer, default 42: Local split/search/initialization seed, default 42; validation split uses seed+1 modulo 2^64.
      %
      %   Errors:
      %     datamunge:invalid_argument  For invalid columns, labels, nonfinite/missing data or search controls. (C++ std::invalid_argument)
      %     datamunge:native  If fitting or scoring fails to produce finite probabilities. (C++ std::runtime_error)
      %
      %   C++: datamunge::MLPClassifierSearch::MLPClassifierSearch(const DataFrame & data, const std::string & response, const std::string & x_feature, const std::string & y_feature, std::size_t min_hidden = 2, std::size_t max_hidden = 24, std::size_t initial_samples = 4, std::size_t iterations = 8, std::size_t epochs = 400, std::uint64_t seed = 42)
      obj@datamunge.NativeObject(struct('rosetta_skip__', true));
      if nargin == 1 && isstruct (varargin{1}) && isfield (varargin{1}, 'rosetta_skip__')
        return;
      end
      __datamunge__(-2, obj, 2938, false, varargin{:});
    end
    function r = best_hidden_nodes (obj)
      % best_hidden_nodes  Selected integer width of the single hidden layer.
      %
      %   r = obj.best_hidden_nodes ()
      %
      %   Output:
      %     r  integer: A value in the requested inclusive search range.
      %
      %   C++: std::size_t datamunge::MLPClassifierSearch::best_hidden_nodes() const
      r = __datamunge__(2928, obj);
    end
    function r = validation_log_loss (obj)
      % validation_log_loss  Minimum validation mean negative log probability of the true class.
      %
      %   r = obj.validation_log_loss ()
      %
      %   Output:
      %     r  double: Finite nonnegative loss of the selected candidate before refitting.
      %
      %   C++: double datamunge::MLPClassifierSearch::validation_log_loss() const
      r = __datamunge__(2929, obj);
    end
    function r = test_accuracy (obj)
      % test_accuracy  Held-out test accuracy of the final development-data refit.
      %
      %   r = obj.test_accuracy ()
      %
      %   Output:
      %     r  double: Fraction correctly classified in [0,1]; test rows were never used for selection.
      %
      %   C++: double datamunge::MLPClassifierSearch::test_accuracy() const
      r = __datamunge__(2930, obj);
    end
    function r = training_rows (obj)
      % training_rows  Number of rows used to fit each search candidate.
      %
      %   r = obj.training_rows ()
      %
      %   Output:
      %     r  integer: Fitting partition size, excluding validation and test.
      %
      %   C++: std::size_t datamunge::MLPClassifierSearch::training_rows() const
      r = __datamunge__(2931, obj);
    end
    function r = validation_rows (obj)
      % validation_rows  Number of rows used to choose hidden width.
      %
      %   r = obj.validation_rows ()
      %
      %   Output:
      %     r  integer: Validation partition size.
      %
      %   C++: std::size_t datamunge::MLPClassifierSearch::validation_rows() const
      r = __datamunge__(2932, obj);
    end
    function r = test_rows (obj)
      % test_rows  Number of untouched rows used for the final evaluation.
      %
      %   r = obj.test_rows ()
      %
      %   Output:
      %     r  integer: Test partition size.
      %
      %   C++: std::size_t datamunge::MLPClassifierSearch::test_rows() const
      r = __datamunge__(2933, obj);
    end
    function r = summary (obj)
      % summary  Summarize split sizes, selected width, validation loss and final test accuracy.
      %
      %   r = obj.summary ()
      %
      %   Example:
      %     const std::unique_ptr<dm::DataFrame> iris(dm::data::iris());
      %     dm::MLPClassifierSearch fit(*iris, "Species", "Petal.Length", "Petal.Width");
      %     const bool named = fit.summary().find("Test accuracy") != std::string::npos; // true
      %
      %   Output:
      %     r  char row: Owned multiline text, without a trailing newline.
      %
      %   C++: std::string datamunge::MLPClassifierSearch::summary() const
      %
      %   See also: datamunge.trials()
      r = __datamunge__(2934, obj);
    end
    function r = trials (obj)
      % trials  Copy the complete Bayesian proposal history into an owned dataframe.
      %
      %   r = obj.trials ()
      %
      %   Example:
      %     const std::unique_ptr<dm::DataFrame> iris(dm::data::iris());
      %     dm::MLPClassifierSearch fit(*iris, "Species", "Petal.Length", "Petal.Width");
      %     const std::unique_ptr<dm::DataFrame> history(fit.trials());
      %     const auto n = history->nrows(); // 12
      %
      %   Output:
      %     r  datamunge.DataFrame or []: New caller-owned frame with evaluation (1-based), hidden_nodes, validation_log_loss, best_log_loss, cached (0/1), and phase (initial/bayesian). One row per proposal, including repeated cached widths. Bindings own the result.
      %
      %   C++: DataFrame * datamunge::MLPClassifierSearch::trials() const
      %
      %   See also: datamunge.summary()
      r = __datamunge__(2935, obj);
    end
    function r = predict (obj, newdata)
      % predict  Predict labels from the final refit using only its two predictor columns.
      %
      %   r = obj.predict (newdata)
      %
      %   Example:
      %     const std::unique_ptr<dm::DataFrame> iris(dm::data::iris());
      %     dm::MLPClassifierSearch fit(*iris, "Species", "Petal.Length", "Petal.Width");
      %     const auto n = fit.predict(*iris).size(); // 150
      %
      %   Inputs:
      %     newdata  datamunge.DataFrame: Borrowed frame, unchanged; missing predictors yield empty labels.
      %
      %   Output:
      %     r  cell array of char: One class label per input row, in input order.
      %
      %   Errors:
      %     datamunge:out_of_range  If a required column is absent. (C++ std::out_of_range)
      %     datamunge:invalid_argument  If predictor types are incompatible. (C++ std::invalid_argument)
      %
      %   C++: std::vector<std::string> datamunge::MLPClassifierSearch::predict(const DataFrame & newdata) const
      %
      %   See also: datamunge.predict()
      r = __datamunge__(2936, obj, newdata);
    end
    function r = plot_decision_regions (obj, varargin)
      % plot_decision_regions  Plot final predicted class regions and development observations in the predictor plane.
      %
      %   r = obj.plot_decision_regions ()
      %   r = obj.plot_decision_regions (grid_resolution)
      %
      %   Pale merged rectangles show the predicted class; saturated points show
      %   observed development labels in the same palette. Test rows are excluded.
      %   Axis ranges cover development points plus 10% padding, in original units.
      %   Rendering samples cell centers, so boundaries are a grid approximation.
      %
      %   Example:
      %     const std::unique_ptr<dm::DataFrame> iris(dm::data::iris());
      %     dm::MLPClassifierSearch fit(*iris, "Species", "Petal.Length", "Petal.Width");
      %     auto plot = fit.plot_decision_regions();
      %     plot.save_svg("build/iris-mlp.svg");
      %
      %   Inputs:
      %     grid_resolution  integer, default 120: Cells per axis, default 120; at least 2.
      %
      %   Output:
      %     r  datamunge.RPlot: Owned editable native plot; save it to an SVG/PNG file rather than printing XML.
      %
      %   Errors:
      %     datamunge:invalid_argument  If grid_resolution is less than 2. (C++ std::invalid_argument)
      %
      %   C++: datamunge::graphics::RPlot datamunge::MLPClassifierSearch::plot_decision_regions(std::size_t grid_resolution = 120) const
      %
      %   See also: datamunge.add_class_regions(), datamunge.predict()
      r = __datamunge__(2937, obj, varargin{:});
    end
  end
end
% Generated by Rosetta from datamungeoctave/rosetta/bindings.hpp: class datamunge::MLPClassifierSearch. Do not edit.
