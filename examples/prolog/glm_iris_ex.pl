% CMD: just TARGET=glm_iris_ex run-prolog
% Generalized linear models on iris: logistic regression (binomial) and Poisson regression.
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    data_iris(Iris),
    data_frame_nrows(Iris, NRows), LastRow is NRows - 1,
    numlist(0, LastRow, Rows),

    % Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
    % the other two on these predictors, which sends logistic regression's coefficients toward
    % +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
    % binary split for a demo.
    include([I]>>(data_frame_string_at(Iris, "Species", I, Sp), memberchk(Sp, ["versicolor", "virginica"])), Rows, Kept),
    maplist([I, V]>>(data_frame_string_at(Iris, "Species", I, Sp), (Sp == "virginica" -> V = 1.0 ; V = 0.0)), Kept, IsVirginica),
    maplist([I, V]>>data_frame_numeric_at(Iris, "Petal.Length", I, V), Kept, PetalLength),
    maplist([I, V]>>data_frame_numeric_at(Iris, "Petal.Width", I, V), Kept, PetalWidth),

    data_frame_new(Sub),
    data_frame_add_numeric_column(Sub, "Petal.Length", PetalLength),
    data_frame_add_numeric_column(Sub, "Petal.Width", PetalWidth),
    data_frame_add_numeric_column(Sub, "is_virginica", IsVirginica),

    format("=================== Logistic regression (binomial, logit link) ===================~n"),
    glm_new(Sub, "is_virginica ~ Petal.Length + Petal.Width", "binomial", Logit),
    glm_print_summary(Logit),

    glm_fitted_values(Logit, Fitted),
    length(IsVirginica, N),
    aggregate_all(count, (nth0(I, Fitted, F), nth0(I, IsVirginica, Y), same_side(F, Y)), Correct),
    Accuracy is 100.0 * Correct / N, cs(Accuracy, AccuracyText),
    format("~nResubstitution accuracy at 0.5 threshold: ~w%~n", [AccuracyText]),

    glm_save_diagnostic_plots(Logit, "glm_logistic_iris"),
    format("~nSaved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg~n"),

    % Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
    % classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
    GridN = 100,
    sum_list(PetalWidth, WidthSum), length(PetalWidth, WidthCount), WidthMean is WidthSum / WidthCount,
    min_list(PetalLength, PlMin0), max_list(PetalLength, PlMax0),
    PlMin is PlMin0 - 0.3, PlMax is PlMax0 + 0.3,
    LastGrid is GridN - 1,
    findall(X, (between(0, LastGrid, I), X is PlMin + (PlMax - PlMin) * I / (GridN - 1)), GridLength),
    length(GridWidth, GridN), maplist(=(WidthMean), GridWidth),
    data_frame_new(Grid),
    data_frame_add_numeric_column(Grid, "Petal.Length", GridLength),
    data_frame_add_numeric_column(Grid, "Petal.Width", GridWidth),
    glm_predict_frame(Logit, Grid, "confidence", CurveFrame),
    format("~nPredicted-probability curve (first 5 rows):~n"),
    data_frame_to_string(CurveFrame, 5, CurveText),
    format("~w~n", [CurveText]),

    % Poisson regression, for contrast: same IRLS engine, different family/link.
    format("~n=================== Poisson regression (log link) ===================~n"),
    maplist([I, V]>>data_frame_numeric_at(Iris, "Sepal.Width", I, V), Rows, SepalWidth),
    maplist([I, V]>>data_frame_numeric_at(Iris, "Petal.Length", I, V), Rows, AllPetalLength),
    maplist([I, V]>>(data_frame_numeric_at(Iris, "Sepal.Length", I, L), V is float(floor(L + 0.5))), Rows, Count),
    data_frame_new(CountData),
    data_frame_add_numeric_column(CountData, "Sepal.Width", SepalWidth),
    data_frame_add_numeric_column(CountData, "Petal.Length", AllPetalLength),
    data_frame_add_numeric_column(CountData, "count", Count),
    glm_new(CountData, "count ~ Sepal.Width + Petal.Length", "poisson", Poisson),
    glm_print_summary(Poisson).

% (F >= 0.5) == (Y >= 0.5)
same_side(F, Y) :- ( F >= 0.5 -> Y >= 0.5 ; Y < 0.5 ).

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
