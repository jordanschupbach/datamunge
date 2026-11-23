% CMD: just TARGET=lmm_ex run-prolog
% Linear mixed models: random intercept (penguins) and random intercept + slope (simulated schools).
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    dotnet_random_new(2024, Rng), nb_setval(rng, Rng),

    format("=================== Random intercept on a real dataset (penguins) ===================~n"),
    data_penguins(Penguins),
    lmm_new(Penguins, "body_mass_g ~ flipper_length_mm + bill_length_mm + (1 | species)", PenguinModel),
    lmm_print_summary(PenguinModel),

    format("~n=================== Random intercept + slope on a simulated multi-school dataset ===================~n"),
    % 30 schools, ~25 students each: score depends on study_hours, but both the baseline score
    % and the return on an extra hour of study vary by school -- the textbook case for a
    % random-intercept-and-slope model.
    NSchools = 30,
    length(SchoolIntercept, NSchools), maplist([X]>>(gauss(G), X is G * 6), SchoolIntercept),
    length(SchoolSlope, NSchools), maplist([X]>>(gauss(G), X is G * 1.2), SchoolSlope),
    TrueIntercept = 60.0,
    TrueSlope = 3.0,

    LastSchool is NSchools - 1,
    findall(S-Hours-Score,
            ( between(0, LastSchool, S),
              next_int(0, 21, Extra), NStudents is 15 + Extra,
              nth0(S, SchoolIntercept, SI), nth0(S, SchoolSlope, SS),
              between(1, NStudents, _),
              next_double(D), Hours is D * 10,
              gauss(Noise),
              Score is TrueIntercept + SI + (TrueSlope + SS) * Hours + Noise * 4 ),
            Rows),
    findall(SF, (member(S-_-_, Rows), SF is float(S)), School),
    findall(H, member(_-H-_, Rows), StudyHours),
    findall(Sc, member(_-_-Sc, Rows), Scores),
    data_frame_new(Df),
    data_frame_add_numeric_column(Df, "school", School),
    data_frame_add_numeric_column(Df, "study_hours", StudyHours),
    data_frame_add_numeric_column(Df, "score", Scores),

    lmm_new(Df, "score ~ study_hours + (1 + study_hours | school)", Model),
    lmm_print_summary(Model),

    cs(TrueIntercept, TI), cs(TrueSlope, TS),
    format("~nTrue generating values: intercept=~w, slope=~w, random-intercept SD=6.0, random-slope SD=1.2, residual SD=4.0~n", [TI, TS]),

    format("~n--- Best Linear Unbiased Predictors (BLUPs) for a few schools ---~n"),
    lmm_group_labels(Model, Labels),
    forall(between(0, 2, Idx),
           ( lmm_random_effects_for_group(Model, Idx, [Re0, Re1|_]),
             nth0(Idx, Labels, Label), cs(Re0, R0), cs(Re1, R1),
             format("school ~w: intercept shift=~w, slope shift=~w~n", [Label, R0, R1]) )),

    format("~n--- Prediction: population-level vs. school-adjusted ---~n"),
    data_frame_new(Population),
    data_frame_new(School0),
    data_frame_add_numeric_column(Population, "study_hours", [5.0]),
    data_frame_add_numeric_column(School0, "study_hours", [5.0]),
    data_frame_add_numeric_column(School0, "school", [0.0]),
    lmm_predict(Model, Population, [P0|_]), cs(P0, P0S),
    format("5 study hours, unseen school:      ~w (fixed effects only)~n", [P0S]),
    lmm_predict(Model, School0, [P1|_]), cs(P1, P1S),
    format("5 study hours, school 0 (known):    ~w (fixed effects + school 0's BLUP)~n", [P1S]).

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

% Random.NextDouble(): [0, 1).
next_double(X) :- nb_getval(rng, Rng), rng_sample(Rng, X).

% Random.Next(MinValue, MaxValue): [MinValue, MaxValue).
next_int(Min, Max, N) :- next_double(X), N is truncate(X * (Max - Min)) + Min.

uniform(U) :- next_double(X), U is 1 - X.  % (0, 1]: avoids log(0) in the Box-Muller transform.
gauss(G) :- uniform(U1), uniform(U2), G is sqrt(-2 * log(U1)) * cos(2 * pi * U2).

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
