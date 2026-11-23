% CMD: just TARGET=svm_iris_ex run-prolog
% Support vector machine on iris: RBF vs. linear kernels, confusion matrix and predictions.
:- use_module(library(datamunge)).
:- initialization(main, main).

formula("Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width").

main :-
    formula(Formula),
    data_iris(Iris),
    data_frame_nrows(Iris, Rows), data_frame_ncols(Iris, Cols),
    format("iris: ~w rows x ~w cols~n~n", [Rows, Cols]),

    format("=== RBF kernel (default) ===~n"),
    svm_new(Iris, Formula, RbfModel),
    svm_print_summary(RbfModel),
    svm_confusion_matrix(RbfModel, Confusion), data_frame_to_string(Confusion, ConfusionText),
    format("~nConfusion matrix (rows = actual, cols = predicted):~n~w~n", [ConfusionText]),

    format("~n=== Linear kernel, for comparison ===~n"),
    svm_new(Iris, Formula, "linear", LinearModel),
    svm_training_accuracy(LinearModel, Accuracy), Percent is Accuracy * 100, cs(Percent, PercentText),
    format("Training accuracy: ~w%~n", [PercentText]),
    svm_num_support_vectors(LinearModel, NumSv),
    format("Support vectors: ~w~n", [NumSv]),

    data_frame_new(NewData),
    data_frame_add_numeric_column(NewData, "Sepal.Length", [5.1, 6.0, 6.5, 6.2]),
    data_frame_add_numeric_column(NewData, "Sepal.Width", [3.5, 2.7, 3.0, 2.8]),
    data_frame_add_numeric_column(NewData, "Petal.Length", [1.4, 4.5, 5.5, 4.8]),
    data_frame_add_numeric_column(NewData, "Petal.Width", [0.2, 1.5, 2.0, 1.8]),
    svm_predict_frame(RbfModel, NewData, Pred), data_frame_to_string(Pred, PredText),
    format("~nRBF predictions for new flowers (votes out of 3 one-vs-one pairs):~n~w~n", [PredText]).

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
