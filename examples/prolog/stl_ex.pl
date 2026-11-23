% CMD: just TARGET=stl_ex run-prolog
% C++ std::vector and std::pair values in Prolog: vectors are lists, pairs are Key-Value terms.
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    N = 100,
    Last is N - 1,
    findall(X, (between(0, Last, I), X is I * 1.5), V),
    forall(member(X, V), (cs(X, S), writeln(S))),
    findall(K, (between(0, Last, I), K is floor(I * 1.5)), V2),
    forall(member(K, V2), writeln(K)),

    P = 3-4,
    P = A-B,
    format("p: (~w, ~w)~n", [A, B]),
    P2 = 10-20,
    P2 = C-D,
    format("p2: (~w, ~w)~n", [C, D]),

    % A C++ std::pair<double,double> comes back as a Key-Value pair.
    make_dpair(1.25, 2.75, P3),
    P3 = E-F, cs(E, ES), cs(F, FS),
    sum_dpair(P3, Sum), cs(Sum, SumS),
    format("make_dpair(1.25, 2.75): (~w, ~w), sum_dpair = ~w~n", [ES, FS, SumS]).

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
