% CMD: just TARGET=glmm_ex run-prolog
% Generalized linear mixed models: a binomial GLMM on penguins and a Poisson GLMM on simulated store data.
:- use_module(library(datamunge)).
:- initialization(main, main).

uniform(U) :- nb_getval(rng, Rng), rng_sample(Rng, U).

% Random.Next(MinValue, MaxValue): [MinValue, MaxValue).
next_int(Min, Max, N) :- uniform(X), N is truncate(X * (Max - Min)) + Min.

gauss(G) :-
    uniform(U), U1 is 1 - U, uniform(U2),  % 1 - u keeps log() away from 0
    G is sqrt(-2 * log(U1)) * cos(2 * pi * U2).

% Knuth's Poisson sampler: the number of uniform factors before the product drops to Threshold, less one.
poisson(Lambda, K) :-
    Threshold is exp(-Lambda),
    knuth(Threshold, 1, 1.0, K).

knuth(Threshold, K0, P0, K) :-
    uniform(U), P is P0 * U,
    (   P > Threshold
    ->  K1 is K0 + 1, knuth(Threshold, K1, P, K)
    ;   K is K0 - 1
    ).

main :-
    dotnet_random_new(4242, Rng), nb_setval(rng, Rng),

    format("=================== Binomial (logistic) mixed model on a real dataset (penguins) ===================~n"),
    % Predicting sex from body mass, grouped by island: unlike species (which is almost perfectly
    % confounded with island in this dataset -- Gentoo penguins occur on Biscoe island only), sex
    % is not tied to location, so this is a well-behaved fit.
    data_penguins(Penguins),
    data_frame_nrows(Penguins, N), Last is N - 1,
    findall(IsMale-Mass-Island,
            ( between(0, Last, I),
              \+ ( member(C, ["sex", "body_mass_g", "island"]), data_frame_is_null(Penguins, C, I, true) ),
              data_frame_string_at(Penguins, "sex", I, Sex), ( Sex == "male" -> IsMale = 1.0 ; IsMale = 0.0 ),
              data_frame_numeric_at(Penguins, "body_mass_g", I, Mass),
              data_frame_string_at(Penguins, "island", I, Island) ),
            Kept),
    findall(M, member(M-_-_, Kept), IsMaleColumn),
    findall(B, member(_-B-_, Kept), BodyMass),
    findall(Is, member(_-_-Is, Kept), IslandColumn),
    data_frame_new(SexDf),
    data_frame_add_numeric_column(SexDf, "is_male", IsMaleColumn),
    data_frame_add_numeric_column(SexDf, "body_mass_g", BodyMass),
    data_frame_add_string_column(SexDf, "island", IslandColumn),
    glmm_new(SexDf, "is_male ~ body_mass_g + (1 | island)", "binomial", SexModel),
    glmm_print_summary(SexModel),

    format("~n=================== Poisson mixed model on simulated multi-site count data ===================~n"),
    % 25 stores, ~20 days each: daily visit counts depend on a promo intensity score, but both
    % the baseline traffic and the promo's effectiveness vary by store.
    NStores = 25,
    length(StoreEffect, NStores), maplist([X]>>(gauss(G), X is G * 0.4), StoreEffect),
    TrueIntercept = 2.0,
    TrueSlope = 0.3,

    LastStore is NStores - 1,
    findall(SF-Promo-Visits,
            ( between(0, LastStore, S),
              next_int(0, 11, Extra), NDays is 15 + Extra,
              nth0(S, StoreEffect, Effect),
              between(1, NDays, _),
              uniform(U), Promo is U * 3,
              Lambda is exp(TrueIntercept + Effect + TrueSlope * Promo),
              poisson(Lambda, K),
              SF is float(S), Visits is float(K) ),
            Rows),
    findall(St, member(St-_-_, Rows), Store),
    findall(Pr, member(_-Pr-_, Rows), PromoColumn),
    findall(Vi, member(_-_-Vi, Rows), VisitsColumn),
    data_frame_new(Df),
    data_frame_add_numeric_column(Df, "store", Store),
    data_frame_add_numeric_column(Df, "promo", PromoColumn),
    data_frame_add_numeric_column(Df, "visits", VisitsColumn),
    glmm_new(Df, "visits ~ promo + (1 | store)", "poisson", StoreModel),

    glmm_print_summary(StoreModel),

    cs(TrueIntercept, TI), cs(TrueSlope, TS),
    format("~nTrue generating values: intercept=~w, slope=~w, random-intercept SD (log scale)=0.4~n", [TI, TS]),

    format("~n--- BLUPs for a few stores ---~n"),
    glmm_group_labels(StoreModel, Labels),
    forall(between(0, 2, Idx),
           ( glmm_random_effects_for_group(StoreModel, Idx, [Re0|_]), cs(Re0, R0),
             nth0(Idx, Labels, Label),
             format("store ~w: intercept shift=~w~n", [Label, R0]) )),

    format("~n--- Prediction: population-level vs. store-adjusted ---~n"),
    data_frame_new(Population),
    data_frame_new(Store0),
    data_frame_add_numeric_column(Population, "promo", [1.5]),
    data_frame_add_numeric_column(Store0, "promo", [1.5]),
    data_frame_add_numeric_column(Store0, "store", [0.0]),
    glmm_predict(StoreModel, Population, [P0|_]), cs(P0, P0S),
    format("promo=1.5, unseen store:   ~w expected visits (fixed effects only)~n", [P0S]),
    glmm_predict(StoreModel, Store0, [P1|_]), cs(P1, P1S),
    format("promo=1.5, store 0 (known): ~w expected visits (fixed effects + store 0's BLUP)~n", [P1S]).

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
