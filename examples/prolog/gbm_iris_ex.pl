% CMD: just TARGET=gbm_iris_ex run-prolog
% Gradient-boosted trees (GBM) on iris: summary, confusion matrix, misclassified rows, 5 vs 100 rounds.
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    data_iris(Iris),
    data_frame_nrows(Iris, N), data_frame_ncols(Iris, Cols),
    format("iris: ~w rows x ~w cols~n~n", [N, Cols]),

    gbm_classifier_new(Iris, "Species ~ Petal.Length + Petal.Width", Model),
    gbm_classifier_print_summary(Model),

    gbm_classifier_confusion_matrix(Model, Confusion), data_frame_to_string(Confusion, ConfusionText),
    format("~nConfusion matrix (rows = actual, cols = predicted):~n~w~n", [ConfusionText]),

    format("~nMisclassified rows:~n"),
    gbm_classifier_predict(Model, Iris, Predictions),
    print_misclassified(Iris, Predictions, Misclassified),
    Percent is 100.0 * Misclassified / N, cs(Percent, PercentText),
    format("~w of ~w misclassified (~w%)~n", [Misclassified, N, PercentText]),

    gbm_classifier_plot_training_deviance(Model, DeviancePlot),
    plot_save(DeviancePlot, "gbm_iris_training_deviance.svg"),
    gbm_classifier_plot_decision_regions(Model, "Petal.Length", "Petal.Width", RegionPlot),
    plot_save(RegionPlot, "gbm_iris_decision_regions.svg"),
    format("~nSaved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg~n"),

    % A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
    % at the training loss, gradually sharpening the decision boundary.
    % gbm_classifier_new(Data, Formula, NTrees, LearningRate, MaxDepth, MinSamplesSplit,
    %                    MinSamplesLeaf, Subsample, Seed, Model)
    gbm_classifier_new(Iris, "Species ~ Petal.Length + Petal.Width", 5, Few),
    nl,
    report("5-round ensemble:  ", Few),
    report("100-round ensemble:", Model),
    gbm_classifier_plot_decision_regions(Few, "Petal.Length", "Petal.Width", FewPlot),
    plot_save(FewPlot, "gbm_iris_decision_regions_5rounds.svg"),
    format("Saved gbm_iris_decision_regions_5rounds.svg~n").

report(Label, Model) :-
    gbm_classifier_training_accuracy(Model, Accuracy), A is 100 * Accuracy, cs(A, AS),
    gbm_classifier_training_deviance(Model, Deviance), last(Deviance, D), cs(D, DS),
    format("~w training accuracy=~w%  deviance=~w~n", [Label, AS, DS]).

% Prints the rows whose prediction differs from Species; Count is how many.
print_misclassified(Iris, Predictions, Count) :-
    foldl(print_row(Iris), Predictions, 0-0, _-Count).

print_row(Iris, Predicted, I-C0, I1-C) :-
    I1 is I + 1,
    data_frame_string_at(Iris, "Species", I, Actual),
    (   Predicted == Actual
    ->  C = C0
    ;   C is C0 + 1,
        data_frame_numeric_at(Iris, "Petal.Length", I, L), cs(L, LS),
        data_frame_numeric_at(Iris, "Petal.Width", I, W), cs(W, WS),
        format("  row ~w: Petal.Length=~w Petal.Width=~w  actual=~w  predicted=~w~n", [I, LS, WS, Actual, Predicted])
    ).

% cs(+Number, -String): Number as C#'s default ToString() prints it: integers as they are,
% doubles in their shortest round-trip digits, in E notation when the decimal exponent is below
% -4 or at least 15 (6, 0.1, 1E-05, 1E+15).
cs(X, S) :- integer(X), !, number_string(X, S).
cs(X, S) :- float_class(X, nan), !, S = "NaN".
cs(X, S) :- float_class(X, infinite), !, ( X < 0 -> S = "-∞" ; S = "∞" ).
cs(X, S) :-
    format(codes(C0), "~w", [X]),
    (   C0 = [0'-|C1] -> Sign = "-" ; C1 = C0, Sign = "" ),
    (   append(M, [0'e|E], C1) -> number_codes(Exp, E) ; M = C1, Exp = 0 ),
    (   append(IP, [0'.|FP], M) -> true ; IP = M, FP = [] ),
    append(IP, FP, D0),
    length(IP, L0),
    P0 is L0 + Exp,
    strip_zeros(D0, P0, D1, Point),
    reverse(D1, R1), strip_trailing(R1, R2), reverse(R2, Digits),
    cs_digits(Digits, Point, Sign, S).

strip_zeros([0'0|T], P0, D, P) :- !, P1 is P0 - 1, strip_zeros(T, P1, D, P).
strip_zeros(D, P, D, P).
strip_trailing([0'0|T], R) :- !, strip_trailing(T, R).
strip_trailing(R, R).

cs_digits([], _, Sign, S) :- !, string_concat(Sign, "0", S).
cs_digits([D|Rest], Point, Sign, S) :-
    Exp10 is Point - 1,
    (   ( Exp10 < -4 ; Exp10 >= 15 )
    ->  ( Rest == [] -> Frac = "" ; string_codes(R, Rest), string_concat(".", R, Frac) ),
        ( Exp10 < 0 -> ES = "-" ; ES = "+" ),
        A is abs(Exp10),
        format(string(S), "~w~c~wE~w~|~`0t~d~2+", [Sign, D, Frac, ES, A])
    ;   Point =< 0
    ->  Z is -Point, length(Zs, Z), maplist(=(0'0), Zs),
        format(string(S), "~w0.~s~s", [Sign, Zs, [D|Rest]])
    ;   length([D|Rest], N), Point >= N
    ->  Z is Point - N, length(Zs, Z), maplist(=(0'0), Zs),
        format(string(S), "~w~s~s", [Sign, [D|Rest], Zs])
    ;   length(Int, Point), append(Int, Frac, [D|Rest]),
        format(string(S), "~w~s.~s", [Sign, Int, Frac])
    ).
