% CMD: just TARGET=naive_bayes_penguins_ex run-prolog
% Naive Bayes on penguins: mixed numeric/categorical predictors, confusion matrix, misclassified rows, plots.
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    % Note: NaiveBayesClassifier drops incomplete rows for its own formula columns internally
    % when fitting, matching every other formula-based model in this library.
    data_penguins(Penguins),
    data_frame_nrows(Penguins, N), data_frame_ncols(Penguins, Cols),
    format("penguins: ~w rows x ~w cols~n~n", [N, Cols]),

    % A mix of numeric (Gaussian likelihood) and categorical (frequency-table likelihood)
    % predictors in one formula -- each modeled independently given the class, per the naive
    % Bayes assumption.
    naive_bayes_classifier_new(Penguins, "species ~ bill_length_mm + bill_depth_mm + island + sex", Model),
    naive_bayes_classifier_print_summary(Model),

    naive_bayes_classifier_confusion_matrix(Model, Confusion), data_frame_to_string(Confusion, ConfusionText),
    format("~nConfusion matrix (rows = actual, cols = predicted):~n~w~n", [ConfusionText]),

    format("~nMisclassified rows:~n"),
    naive_bayes_classifier_predict(Model, Penguins, Predictions),
    Columns = ["species", "bill_length_mm", "bill_depth_mm", "island", "sex"],
    foldl(print_row(Penguins, Columns), Predictions, 0-0, _-Misclassified),
    format("~w misclassified (of ~w rows, some incomplete)~n", [Misclassified, N]),

    % PlotDecisionRegions requires exactly two NUMERIC predictors, so build a separate
    % two-predictor model (bill measurements alone) just for visualization.
    naive_bayes_classifier_new(Penguins, "species ~ bill_length_mm + bill_depth_mm", BillOnly),
    naive_bayes_classifier_training_accuracy(BillOnly, Accuracy), A100 is 100 * Accuracy, cs(A100, AText),
    format("~nbill-measurements-only model training accuracy: ~w%~n", [AText]),
    naive_bayes_classifier_plot_classification(BillOnly, Penguins, "bill_length_mm", "bill_depth_mm", ClassPlot),
    plot_save(ClassPlot, "naive_bayes_penguins_classification.svg"),
    naive_bayes_classifier_plot_decision_regions(BillOnly, "bill_length_mm", "bill_depth_mm", RegionPlot),
    plot_save(RegionPlot, "naive_bayes_penguins_decision_regions.svg"),
    format("Saved naive_bayes_penguins_classification.svg and naive_bayes_penguins_decision_regions.svg~n").

% One prediction (row I): skipped when a model column is null, printed when misclassified.
print_row(Penguins, Columns, Predicted, I-C0, I1-C) :-
    I1 is I + 1,
    (   member(Column, Columns), data_frame_is_null(Penguins, Column, I, true)
    ->  C = C0
    ;   data_frame_string_at(Penguins, "species", I, Actual),
        (   Predicted == Actual
        ->  C = C0
        ;   C is C0 + 1,
            data_frame_numeric_at(Penguins, "bill_length_mm", I, L), cs(L, LS),
            data_frame_numeric_at(Penguins, "bill_depth_mm", I, D), cs(D, DS),
            data_frame_string_at(Penguins, "island", I, Island),
            data_frame_string_at(Penguins, "sex", I, Sex),
            format("  row ~w: bill_length=~w bill_depth=~w island=~w sex=~w  actual=~w  predicted=~w~n",
                   [I, LS, DS, Island, Sex, Actual, Predicted])
        )
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
