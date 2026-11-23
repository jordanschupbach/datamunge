% CMD: just TARGET=gaussian_process_regression_iris_ex run-prolog
% Gaussian process regression of Petal.Length on Petal.Width (iris), with confidence intervals.
:- use_module(library(datamunge)).
:- initialization(main, main).

formula("Petal.Length ~ Petal.Width").

main :-
    formula(Formula),
    data_iris(Iris),
    data_frame_nrows(Iris, Rows), data_frame_ncols(Iris, Cols),
    format("iris: ~w rows x ~w cols~n", [Rows, Cols]),
    format("formula: ~w~n~n", [Formula]),

    % Both the length scale and the noise ratio are auto-selected by maximizing the exact log
    % marginal likelihood.
    gaussian_process_regression_new(Iris, Formula, Model),
    gaussian_process_regression_print_summary(Model),

    gaussian_process_regression_plot_fit(Model, Iris, FitPlot),
    plot_save(FitPlot, "gpr_iris_fit.svg"),
    gaussian_process_regression_plot_length_scale_profile(Model, ProfilePlot),
    plot_save(ProfilePlot, "gpr_iris_length_scale_profile.svg"),
    format("~nSaved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg~n"),

    % Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
    % interval at every point -- including far outside the training data, where it should widen
    % substantially as the model's uncertainty grows.
    data_frame_new(Query),
    data_frame_add_numeric_column(Query, "Petal.Width", [0.2, 1.3, 2.5, 10.0]),
    gaussian_process_regression_predict_frame(Model, Query, "confidence", Detail),
    data_frame_to_string(Detail, DetailText),
    format("~nPredictions with 95% confidence intervals:~n"),
    format("~w~n", [DetailText]),
    format("(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its~n interval is than the in-range predictions.)~n").
