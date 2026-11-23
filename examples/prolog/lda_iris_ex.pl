% CMD: just TARGET=lda_iris_ex run-prolog
% Linear discriminant analysis on iris: summary, confusion matrix, predictions and discriminant plot.
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    data_iris(Iris),
    data_frame_nrows(Iris, Rows), data_frame_ncols(Iris, Cols),
    format("iris: ~w rows x ~w cols~n~n", [Rows, Cols]),

    lda_new(Iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width", Model),
    lda_print_summary(Model),

    lda_confusion_matrix(Model, Confusion), data_frame_to_string(Confusion, ConfusionText),
    format("~nConfusion matrix (rows = actual, cols = predicted):~n~w~n", [ConfusionText]),

    data_frame_new(NewData),
    data_frame_add_numeric_column(NewData, "Sepal.Length", [5.1, 6.0, 6.5, 6.2]),
    data_frame_add_numeric_column(NewData, "Sepal.Width", [3.5, 2.7, 3.0, 2.8]),
    data_frame_add_numeric_column(NewData, "Petal.Length", [1.4, 4.5, 5.5, 4.8]),
    data_frame_add_numeric_column(NewData, "Petal.Width", [0.2, 1.5, 2.0, 1.8]),
    lda_predict_frame(Model, NewData, Pred), data_frame_to_string(Pred, PredText),
    format("~nPredictions for new flowers:~n~w~n", [PredText]),

    lda_save_discriminant_plot(Model, "lda_iris_discriminants.svg"),
    format("~nSaved discriminant plot as lda_iris_discriminants.svg~n").
