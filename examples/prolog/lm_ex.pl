% CMD: just TARGET=lm_ex run-prolog
% Linear model: mpg ~ hp + wt + transmission, its ANOVA table and predictions.
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    data_frame_new(Cars),
    data_frame_add_numeric_column(Cars, "hp", [110.0, 110.0, 93.0, 110.0, 175.0, 105.0, 245.0, 62.0, 95.0, 123.0]),
    data_frame_add_numeric_column(Cars, "wt", [2.62, 2.875, 2.32, 3.215, 3.44, 3.46, 3.57, 3.19, 3.15, 3.44]),
    data_frame_add_string_column(Cars, "transmission", ["manual", "manual", "manual", "automatic", "automatic",
                                                        "automatic", "automatic", "automatic", "automatic", "automatic"]),
    data_frame_add_numeric_column(Cars, "mpg", [21.0, 21.0, 22.8, 21.4, 18.7, 18.1, 14.3, 24.4, 22.8, 19.2]),

    format("Fitting: mpg ~~ hp + wt + transmission~n~n"),
    lm_new(Cars, "mpg ~ hp + wt + transmission", Model),
    lm_print_summary(Model),

    lm_anova(Model, Anova), data_frame_to_string(Anova, AnovaText),
    format("~nSequential ANOVA:~n~w~n", [AnovaText]),

    data_frame_new(NewCars),
    data_frame_add_numeric_column(NewCars, "hp", [150.0, 90.0]),
    data_frame_add_numeric_column(NewCars, "wt", [3.0, 2.5]),
    data_frame_add_string_column(NewCars, "transmission", ["manual", "automatic"]),
    lm_predict_frame(Model, NewCars, "confidence", Pred), data_frame_to_string(Pred, PredText),
    format("~nPredictions with 95% confidence intervals:~n~w~n", [PredText]),

    lm_save_diagnostic_plots(Model, "lm_ex_diagnostics"),
    format("~nSaved diagnostic plots as lm_ex_diagnostics_*.svg~n").
