% CMD: just TARGET=timeseries_ex run-prolog
% Time series: ARIMA, seasonal ARIMA and exponential smoothing (SES, Holt, Holt-Winters).
:- use_module(library(datamunge)).
:- initialization(main, main).

seed(N) :- dotnet_random_new(N, Rng), nb_setval(rng, Rng).
uniform(U) :- nb_getval(rng, Rng), rng_sample(Rng, X), U is 1 - X.  % (0, 1]: avoids log(0) in the Box-Muller transform.
gauss(G) :- uniform(U1), uniform(U2), G is sqrt(-2 * log(U1)) * cos(2 * pi * U2).

joined(Values, Text) :- maplist(cs, Values, Texts), atomic_list_concat(Texts, ', ', Text).

% A random walk with drift: integrated ARMA(0,1) increments.
random_walk(0, _, _, []) :- !.
random_walk(N, Level0, PrevShock, [Level|Ys]) :-
    gauss(Shock),
    Level is Level0 + (0.3 + 0.4 * PrevShock + Shock),
    N1 is N - 1,
    random_walk(N1, Level, Shock, Ys).

% A seasonal series: trend + 12-period sine + AR(1) noise.
seasonal_series(N, N, _, []) :- !.
seasonal_series(I, N, Prev0, [Y|Ys]) :-
    gauss(G), Prev is 0.5 * Prev0 + G,
    Y is 20 + 0.2 * I + 5 * sin(2 * pi * I / 12) + Prev,
    I1 is I + 1,
    seasonal_series(I1, N, Prev, Ys).

main :-
    format("=================== ARIMA(1,1,1) on a simulated random walk with drift ===================~n"),
    seed(42),
    random_walk(150, 100.0, 0.0, Y),

    arima_options_new(Options),
    arima_options_set_p(Options, 1), arima_options_set_d(Options, 1), arima_options_set_q(Options, 1),
    arima_options_set_de_population_size(Options, 80), arima_options_set_de_max_generations(Options, 400),
    arima_new(Y, Options, Model),

    arima_ar_coefficients(Model, [Ar|_]), cs(Ar, ArS),
    format("AR coefficient: ~w~n", [ArS]),
    arima_ma_coefficients(Model, [Ma|_]), cs(Ma, MaS),
    format("MA coefficient: ~w~n", [MaS]),
    arima_sigma2(Model, Sigma2), arima_aic(Model, Aic), arima_bic(Model, Bic),
    cs(Sigma2, Sigma2S), cs(Aic, AicS), cs(Bic, BicS),
    format("sigma^2: ~w, AIC: ~w, BIC: ~w~n", [Sigma2S, AicS, BicS]),

    arima_forecast_with_intervals(Model, 6, ForecastValues-StdErrors),
    joined(ForecastValues, F1), format("6-step forecast: ~w~n", [F1]),
    joined(StdErrors, F2), format("forecast std. errors: ~w~n", [F2]),

    format("~n=================== SARIMA(1,0,0)(1,1,0)_12 on a seasonal series ===================~n"),
    seed(7),
    seasonal_series(0, 120, 0.0, S),
    arima_options_new(Options2),
    arima_options_set_p(Options2, 1),
    arima_options_set_seasonal_p(Options2, 1),
    arima_options_set_seasonal_d(Options2, 1),
    arima_options_set_seasonal_period(Options2, 12),
    arima_options_set_de_population_size(Options2, 100),
    arima_options_set_de_max_generations(Options2, 500),
    arima_new(S, Options2, Model2),
    arima_ar_coefficients(Model2, [Ar2|_]), cs(Ar2, Ar2S),
    arima_seasonal_ar_coefficients(Model2, [Sar|_]), cs(Sar, SarS),
    format("AR coefficient: ~w, seasonal AR coefficient: ~w~n", [Ar2S, SarS]),
    arima_forecast(Model2, 12, Forecast2), joined(Forecast2, F3),
    format("12-step forecast: ~w~n", [F3]),

    format("~n=================== Exponential smoothing: Holt-Winters on retail-style seasonal data ===================~n"),
    SeasonalShape = [0.8, 0.75, 0.9, 0.95, 1.0, 1.05, 1.1, 1.05, 1.0, 1.1, 1.3, 1.6],
    seed(11),
    findall(V, ( between(0, 47, I), M is I mod 12, nth0(M, SeasonalShape, Shape), gauss(G),
                 V is (100 + 2 * I) * Shape + G * 3 ), Y2),
    exponential_smoothing_options_new(EsOptions),
    exponential_smoothing_options_set_trend(EsOptions, additive),
    exponential_smoothing_options_set_seasonal(EsOptions, multiplicative),
    exponential_smoothing_options_set_seasonal_period(EsOptions, 12),
    exponential_smoothing_options_set_de_population_size(EsOptions, 60),
    exponential_smoothing_options_set_de_max_generations(EsOptions, 300),
    exponential_smoothing_new(Y2, EsOptions, EsModel),
    exponential_smoothing_alpha(EsModel, Alpha), exponential_smoothing_beta(EsModel, Beta),
    exponential_smoothing_gamma(EsModel, Gamma),
    cs(Alpha, AS), cs(Beta, BS), cs(Gamma, GS),
    format("alpha=~w beta=~w gamma=~w~n", [AS, BS, GS]),
    exponential_smoothing_sigma2(EsModel, EsSigma2), exponential_smoothing_aic(EsModel, EsAic),
    cs(EsSigma2, ES1), cs(EsAic, ES2),
    format("sigma^2: ~w, AIC: ~w~n", [ES1, ES2]),
    exponential_smoothing_forecast(EsModel, 12, EsForecast), joined(EsForecast, F4),
    format("12-month forecast: ~w~n", [F4]),

    format("~n=================== Simple exponential smoothing vs. Holt's linear trend ===================~n"),
    Flat = [50.2, 49.8, 50.5, 49.6, 50.1, 50.3, 49.9, 50.0, 50.4, 49.7],

    exponential_smoothing_options_new(SesOptions),
    exponential_smoothing_new(Flat, SesOptions, Ses),
    exponential_smoothing_alpha(Ses, SesAlpha), cs(SesAlpha, SesAlphaS),
    format("SES alpha= ~w~n", [SesAlphaS]),
    exponential_smoothing_forecast(Ses, 5, SesForecast), joined(SesForecast, F5),
    format("SES 5-step forecast: ~w~n", [F5]),

    exponential_smoothing_options_new(HoltOptions),
    exponential_smoothing_options_set_trend(HoltOptions, additive),
    exponential_smoothing_new(Flat, HoltOptions, Holt),
    exponential_smoothing_alpha(Holt, HoltAlpha), exponential_smoothing_beta(Holt, HoltBeta),
    cs(HoltAlpha, HA), cs(HoltBeta, HB),
    format("Holt alpha=~w beta=~w~n", [HA, HB]),
    exponential_smoothing_forecast(Holt, 5, HoltForecast), joined(HoltForecast, F6),
    format("Holt 5-step forecast: ~w~n", [F6]).

% System.Random(Seed) of .NET (the seeded, Knuth subtractive generator), so the simulated data
% match the C# example's exactly. The state is a term rng(Seeds, Inext, Inextp) updated in place
% (nb_setarg/3); Seeds holds the 56 seeds, seed I in argument I + 1.
dotnet_random_new(Seed, rng(Seeds, 0, 21)) :-
    MBig = 2147483647,
    functor(Seeds, seeds, 56),
    forall(between(0, 55, I), seed_set(Seeds, I, 0)),
    Mj is 161803398 - abs(Seed),
    seed_set(Seeds, 55, Mj),
    init_seeds(1, Seeds, Mj, 1),
    forall(( between(1, 4, _), between(1, 55, I) ),
           ( seed_get(Seeds, I, V0),
             J is 1 + (I + 30) mod 55,
             seed_get(Seeds, J, W),
             V1 is V0 - W,
             ( V1 < 0 -> V is V1 + MBig ; V = V1 ),
             seed_set(Seeds, I, V) )).

init_seeds(55, _, _, _) :- !.
init_seeds(I, Seeds, Mj, Mk) :-
    II is (21 * I) mod 55,
    seed_set(Seeds, II, Mk),
    Mk1 is Mj - Mk,
    ( Mk1 < 0 -> Mk2 is Mk1 + 2147483647 ; Mk2 = Mk1 ),
    I1 is I + 1,
    init_seeds(I1, Seeds, Mk, Mk2).

seed_get(Seeds, I, V) :- A is I + 1, arg(A, Seeds, V).
seed_set(Seeds, I, V) :- A is I + 1, nb_setarg(A, Seeds, V).

rng_sample(Rng, X) :-
    MBig = 2147483647,
    Rng = rng(Seeds, Inext0, Inextp0),
    ( Inext0 + 1 >= 56 -> Inext = 1 ; Inext is Inext0 + 1 ),
    ( Inextp0 + 1 >= 56 -> Inextp = 1 ; Inextp is Inextp0 + 1 ),
    seed_get(Seeds, Inext, A),
    seed_get(Seeds, Inextp, B),
    V0 is A - B,
    ( V0 =:= MBig -> V1 is V0 - 1 ; V1 = V0 ),
    ( V1 < 0 -> V is V1 + MBig ; V = V1 ),
    seed_set(Seeds, Inext, V),
    nb_setarg(2, Rng, Inext),
    nb_setarg(3, Rng, Inextp),
    X is V * (1.0 / MBig).
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
