% CMD: just TARGET=forest_iris_ex run-prolog
% Random forest classifier on iris: summary, confusion matrix, misclassifications, OOB accuracy.
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    data_iris(Iris),
    data_frame_nrows(Iris, N), data_frame_ncols(Iris, Cols),
    format("iris: ~w rows x ~w cols~n~n", [N, Cols]),

    random_forest_classifier_new(Iris, "Species ~ Petal.Length + Petal.Width", Model),
    random_forest_classifier_print_summary(Model),

    format("~nConfusion matrix (rows = actual, cols = predicted):~n"),
    random_forest_classifier_confusion_matrix(Model, Confusion), data_frame_to_string(Confusion, ConfusionText),
    format("~w~n", [ConfusionText]),

    format("~nMisclassified rows:~n"),
    random_forest_classifier_predict(Model, Iris, Predictions),
    print_misclassified(Iris, Predictions, Misclassified),
    Percent is 100.0 * Misclassified / N, cs(Percent, PercentText),
    format("~w of ~w misclassified (~w%)~n", [Misclassified, N, PercentText]),

    random_forest_classifier_plot_classification(Model, Iris, "Petal.Length", "Petal.Width", ClassPlot),
    plot_save(ClassPlot, "forest_iris_classification.svg"),
    random_forest_classifier_plot_decision_regions(Model, "Petal.Length", "Petal.Width", RegionPlot),
    plot_save(RegionPlot, "forest_iris_decision_regions.svg"),
    format("~nSaved forest_iris_classification.svg and forest_iris_decision_regions.svg~n"),

    % A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
    % are added -- the defining random forest effect a single decision tree cannot demonstrate.
    % random_forest_classifier_new(Data, Formula, NTrees, MaxDepth, MinSamplesSplit, MinSamplesLeaf,
    %                              MaxFeatures, Criterion, Bootstrap, SampleFraction, Seed, Model)
    random_forest_classifier_new(Iris, "Species ~ Petal.Length + Petal.Width", 5, SmallForest),
    accuracies(SmallForest, T1, O1),
    format("~n5-tree forest:   training accuracy=~w%  OOB accuracy=~w%~n", [T1, O1]),
    accuracies(Model, T2, O2),
    format("100-tree forest: training accuracy=~w%  OOB accuracy=~w%~n", [T2, O2]),
    random_forest_classifier_plot_decision_regions(SmallForest, "Petal.Length", "Petal.Width", SmallPlot),
    plot_save(SmallPlot, "forest_iris_decision_regions_5trees.svg"),
    format("Saved forest_iris_decision_regions_5trees.svg~n").

accuracies(Model, Training, Oob) :-
    random_forest_classifier_training_accuracy(Model, T), T100 is T * 100, cs(T100, Training),
    random_forest_classifier_oob_accuracy(Model, O), O100 is O * 100, cs(O100, Oob).

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
