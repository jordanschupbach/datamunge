% CMD: just TARGET=dataframe_ex run-prolog
% DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
:- use_module(library(datamunge)).
:- initialization(main, main).

frame_text(Frame, Text) :- data_frame_to_string(Frame, Text).

main :-
    data_frame_new(Sales),
    data_frame_add_string_column(Sales, "region", ["west", "west", "east", "south", "south", "south"]),
    data_frame_add_string_column(Sales, "product", ["widget", "widget", "widget", "gizmo", "gizmo", "gizmo"]),
    data_frame_add_numeric_column(Sales, "sales", [10.0, 10.0, 14.0, 8.0, 0.0, 11.0], [1, 1, 1, 1, 0, 1]),
    data_frame_add_string_column(Sales, "quarter", ["Q1", "Q1", "Q1", "Q2", "Q2", ""], [1, 1, 1, 1, 1, 0]),

    frame_text(Sales, T1),
    format("raw data~n~w~n~n", [T1]),

    data_frame_drop_duplicates(Sales, ["region", "product", "sales", "quarter"], Cleaned),
    data_frame_fill_null_string(Cleaned, "quarter", "unknown"),
    data_frame_fill_null_numeric(Cleaned, "sales", 0.0),
    frame_text(Cleaned, T2),
    format("after drop_duplicates + fill_null~n~w~n~n", [T2]),

    data_frame_select(Cleaned, ["region", "sales", "quarter"], Selected0),
    data_frame_sort_by(Selected0, "sales", false, Selected),
    frame_text(Selected, T3),
    format("selected + sorted~n~w~n~n", [T3]),

    data_frame_group_by_sum(Cleaned, ["region"], ["sales"], Grouped0),
    data_frame_sort_by(Grouped0, "sales", false, Grouped),
    frame_text(Grouped, T4),
    format("group_by_sum(region)~n~w~n~n", [T4]),

    data_frame_new(Targets),
    data_frame_add_string_column(Targets, "region", ["west", "east", "south"]),
    data_frame_add_numeric_column(Targets, "target", [18.0, 12.0, 25.0]),
    data_frame_join(Grouped, Targets, "region", "region", "left", Joined),
    frame_text(Joined, T5),
    format("joined with targets~n~w~n~n", [T5]),

    data_frame_shape(Cleaned, [Rows, Cols|_]),
    format("shape = (~w, ~w)~n", [Rows, Cols]),
    data_frame_numeric_count(Cleaned, "sales", Count),
    format("sales count = ~w~n", [Count]),
    data_frame_numeric_null_count(Cleaned, "sales", Nulls),
    format("sales nulls = ~w~n", [Nulls]),
    data_frame_numeric_sum(Cleaned, "sales", Sum), cs(Sum, SumS),
    format("sales sum = ~w~n", [SumS]),
    data_frame_numeric_mean(Cleaned, "sales", Mean), cs(Mean, MeanS),
    format("sales mean = ~w~n", [MeanS]).

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
