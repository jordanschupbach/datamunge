% CMD: just TARGET=knn_iris_ex run-prolog
% k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for several k.
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    data_iris(Iris),
    data_frame_nrows(Iris, N), data_frame_ncols(Iris, Cols),
    format("iris: ~w rows x ~w cols~n~n", [N, Cols]),

    knn_classifier_new(Iris, "Species ~ Petal.Length + Petal.Width", Model),
    knn_classifier_print_summary(Model),

    format("~nConfusion matrix (leave-one-out; rows = actual, cols = predicted):~n"),
    knn_classifier_confusion_matrix(Model, Confusion), data_frame_to_string(Confusion, ConfusionText),
    format("~w~n", [ConfusionText]),

    format("~nLeave-one-out misclassified rows:~n"),
    knn_classifier_predict(Model, Iris, Fitted),
    print_misclassified(Iris, Fitted, Misclassified),
    Percent is 100.0 * Misclassified / N, cs(Percent, PercentText),
    format("~w of ~w misclassified (~w%)~n", [Misclassified, N, PercentText]),

    knn_classifier_plot_decision_regions(Model, "Petal.Length", "Petal.Width", RegionPlot),
    plot_save(RegionPlot, "knn_iris_decision_regions_k5.svg"),
    format("~nSaved knn_iris_decision_regions_k5.svg~n"),

    % k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
    % every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
    % underfit boundary). knn_classifier_new(Data, Formula, K, Metric, Weighted, Standardize, Model)
    forall(member(K, [1, 25]),
           ( knn_classifier_new(Iris, "Species ~ Petal.Length + Petal.Width", K, Knn),
             format(string(File), "knn_iris_decision_regions_k~w.svg", [K]),
             knn_classifier_training_accuracy(Knn, Accuracy), A100 is Accuracy * 100, cs(A100, AText),
             format(string(KText), "~w~t~2|", [K]),  % C#'s {k,-2}
             format("~nk=~w leave-one-out accuracy: ~w%~n", [KText, AText]),
             knn_classifier_plot_decision_regions(Knn, "Petal.Length", "Petal.Width", KnnPlot),
             plot_save(KnnPlot, File),
             format("Saved ~w~n", [File]) )).

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
