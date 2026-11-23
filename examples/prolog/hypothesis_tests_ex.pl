% CMD: just TARGET=hypothesis_tests_ex run-prolog
% Classical hypothesis tests (t, Wilcoxon, ANOVA, correlation, F, normality, chi-squared, binomial) on iris.
:- use_module(library(datamunge)).
:- initialization(main, main).

print_result(Label, R) :-
    hypothesis_test_result_alternative(R, Alt),
    (   Alt == less -> Alternative = "less"
    ;   Alt == greater -> Alternative = "greater"
    ;   Alternative = "two.sided"
    ),
    hypothesis_test_result_statistic(R, Statistic), cs(Statistic, StatisticS),
    format("~w: statistic=~w", [Label, StatisticS]),
    hypothesis_test_result_parameter1(R, P1),
    ( P1 =\= 0 -> cs(P1, P1S), format(", df1=~w", [P1S]) ; true ),
    hypothesis_test_result_parameter2(R, P2),
    ( P2 =\= 0 -> cs(P2, P2S), format(", df2=~w", [P2S]) ; true ),
    hypothesis_test_result_p_value(R, PValue), cs(PValue, PValueS),
    format(", p=~w (~w)", [PValueS, Alternative]),
    hypothesis_test_result_has_conf_int(R, HasConfInt),
    (   HasConfInt == true
    ->  hypothesis_test_result_conf_int_lower(R, Lo), hypothesis_test_result_conf_int_upper(R, Hi),
        cs(Lo, LoS), cs(Hi, HiS),
        format(", CI=[~w, ~w]", [LoS, HiS])
    ;   true
    ),
    hypothesis_test_result_method(R, Method),
    format(" -- ~w~n", [Method]).

column_for_species(Df, Column, Species, Values) :-
    data_frame_nrows(Df, N), Last is N - 1,
    findall(V, ( between(0, Last, I), data_frame_string_at(Df, "Species", I, Species),
                 data_frame_numeric_at(Df, Column, I, V) ), Values).

column(Df, Column, Values) :-
    data_frame_nrows(Df, N), Last is N - 1,
    findall(V, ( between(0, Last, I), data_frame_numeric_at(Df, Column, I, V) ), Values).

count_long(Median, Values, Count) :- aggregate_all(count, (member(X, Values), X > Median), Count).

main :-
    data_iris(Iris),

    column_for_species(Iris, "Petal.Length", "setosa", SetosaPetal),
    column_for_species(Iris, "Petal.Length", "versicolor", VersicolorPetal),
    column_for_species(Iris, "Petal.Length", "virginica", VirginicaPetal),

    format("=================== t-tests: petal length, setosa vs. versicolor ===================~n"),
    t_test_two_sample(SetosaPetal, VersicolorPetal, R1), print_result("Welch two-sample t-test", R1),
    wilcoxon_rank_sum_test(SetosaPetal, VersicolorPetal, R2), print_result("Wilcoxon rank-sum test", R2),

    format("~n=================== One-way ANOVA / Kruskal-Wallis across all 3 species ===================~n"),
    append([SetosaPetal, VersicolorPetal, VirginicaPetal], AllPetal),
    maplist(length, [SetosaPetal, VersicolorPetal, VirginicaPetal], Sizes),
    one_way_anova(AllPetal, Sizes, R3), print_result("One-way ANOVA", R3),
    kruskal_wallis_test(AllPetal, Sizes, R4), print_result("Kruskal-Wallis", R4),

    format("~n=================== Correlation: sepal length vs. petal length ===================~n"),
    column(Iris, "Sepal.Length", SepalLength),
    column(Iris, "Petal.Length", PetalLength),
    pearson_correlation_test(SepalLength, PetalLength, R5), print_result("Pearson correlation", R5),
    spearman_correlation_test(SepalLength, PetalLength, R6), print_result("Spearman correlation", R6),

    format("~n=================== F-test: petal length variance, setosa vs. virginica ===================~n"),
    f_test_variance(SetosaPetal, VirginicaPetal, R7), print_result("F test", R7),

    format("~n=================== Normality: is sepal length normally distributed within setosa? ===================~n"),
    column_for_species(Iris, "Sepal.Length", "setosa", SetosaSepal),
    shapiro_francia_test(SetosaSepal, R8), print_result("Shapiro-Francia", R8),
    ks_test_one_sample_normal(SetosaSepal, 5.006, 0.3525, R9), print_result("KS vs. fitted normal", R9),

    format("~n=================== Chi-squared / Fisher: is petal length \"long\" independent of species? ===================~n"),
    msort(AllPetal, Sorted), length(AllPetal, NAll), Mid is NAll // 2, nth0(Mid, Sorted, MedianAll),
    count_long(MedianAll, SetosaPetal, SetosaLong),
    length(SetosaPetal, NSetosa), SetosaShort is NSetosa - SetosaLong,
    count_long(MedianAll, VersicolorPetal, VersicolorLong),
    length(VersicolorPetal, NVersicolor), VersicolorShort is NVersicolor - VersicolorLong,
    format("table: setosa=[~w,~w] versicolor=[~w,~w]~n", [SetosaLong, SetosaShort, VersicolorLong, VersicolorShort]),
    maplist([C, F]>>(F is float(C)), [SetosaLong, SetosaShort, VersicolorLong, VersicolorShort], Table),
    chi_squared_test_independence(Table, 2, 2, R10), print_result("Chi-squared independence", R10),
    fisher_exact_test_2x2(SetosaLong, SetosaShort, VersicolorLong, VersicolorShort, R11),
    print_result("Fisher's exact test", R11),

    format("~n=================== Proportion / binomial: fraction of \"long\" petals overall ===================~n"),
    count_long(MedianAll, AllPetal, LongCount),
    proportion_test_one_sample(LongCount, NAll, 0.5, R12), print_result("One-sample proportion test (vs 0.5)", R12),
    binomial_test(LongCount, NAll, 0.5, R13), print_result("Exact binomial test (vs 0.5)", R13).

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
