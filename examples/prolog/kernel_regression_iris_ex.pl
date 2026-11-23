% CMD: just TARGET=kernel_regression_iris_ex run-prolog
% Nadaraya-Watson kernel regression on iris: CV-selected bandwidth vs. too-small and too-large ones.
:- use_module(library(datamunge)).
:- initialization(main, main).

formula("Petal.Length ~ Petal.Width").

main :-
    formula(Formula),
    data_iris(Iris),
    data_frame_nrows(Iris, Rows), data_frame_ncols(Iris, Cols),
    format("iris: ~w rows x ~w cols~n", [Rows, Cols]),
    format("formula: ~w~n~n", [Formula]),

    kernel_regression_new(Iris, Formula, Model),  % bandwidth auto-selected via leave-one-out CV
    kernel_regression_print_summary(Model),

    kernel_regression_plot_fit(Model, Iris, FitPlot),
    plot_save(FitPlot, "kernel_regression_iris_fit.svg"),
    kernel_regression_plot_cv_curve(Model, CvPlot),
    plot_save(CvPlot, "kernel_regression_iris_cv.svg"),
    format("~nSaved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg~n"),

    % A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
    % bias-variance story. kernel_regression_new(Data, Formula, Kernel, Bandwidth, NBandwidth, Standardize, Model)
    kernel_regression_new(Iris, Formula, "gaussian", 0.05, Small),
    kernel_regression_r_squared(Small, SmallR2), kernel_regression_rmse(Small, SmallRmse),
    cs(SmallR2, A1), cs(SmallRmse, A2),
    format("~nbandwidth=0.05 (too small): LOO R-squared=~w  LOO RMSE=~w~n", [A1, A2]),
    kernel_regression_plot_fit(Small, Iris, SmallPlot),
    plot_save(SmallPlot, "kernel_regression_iris_fit_small_bandwidth.svg"),

    kernel_regression_new(Iris, Formula, "gaussian", 5.0, Large),
    kernel_regression_r_squared(Large, LargeR2), kernel_regression_rmse(Large, LargeRmse),
    cs(LargeR2, B1), cs(LargeRmse, B2),
    format("bandwidth=5.0 (too large):  LOO R-squared=~w  LOO RMSE=~w~n", [B1, B2]),
    kernel_regression_plot_fit(Large, Iris, LargePlot),
    plot_save(LargePlot, "kernel_regression_iris_fit_large_bandwidth.svg"),

    kernel_regression_bandwidth(Model, Bandwidth), kernel_regression_r_squared(Model, R2),
    kernel_regression_rmse(Model, Rmse),
    cs(Bandwidth, C0), cs(R2, C1), cs(Rmse, C2),
    format("bandwidth=~w (CV-selected): LOO R-squared=~w  LOO RMSE=~w~n", [C0, C1, C2]),
    format("~nSaved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg~n").

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
