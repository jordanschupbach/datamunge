% CMD: just TARGET=tensor_ex run-prolog
% Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
:- use_module(library(datamunge)).
:- initialization(main, main).

shape_string(Tensor, Text) :-
    tensor_shape(Tensor, Shape),
    atomic_list_concat(Shape, ', ', Joined),
    format(string(Text), "[~w]", [Joined]).

text(Tensor, Text) :- tensor_to_string(Tensor, Text).

main :-
    format("=================== Construction ===================~n"),
    tensor_zeros([2, 3], Zeros), text(Zeros, T1),
    format("zeros([2,3]): ~w~n", [T1]),
    tensor_eye(3, Eye), text(Eye, T2),
    format("eye(3): ~w~n", [T2]),

    tensor_arange(0.0, 12.0, 1.0, R0), tensor_reshape(R0, [3, 4], R), text(R, T3),
    format("arange(0,12).reshape([3,4]): ~w~n", [T3]),

    format("~n=================== Shape ops ===================~n"),
    tensor_transpose(R, RT), shape_string(RT, T4),
    format("transpose -> shape ~w~n", [T4]),
    tensor_slice(R, 1, 1, 3, Slice), text(Slice, T5),
    format("slice(axis=1, start=1, stop=3): ~w~n", [T5]),

    format("~n=================== Broadcasting arithmetic ===================~n"),
    tensor_from_values([3, 1], [1.0, 2.0, 3.0], Col),
    tensor_from_values([1, 4], [10.0, 20.0, 30.0, 40.0], Row),
    tensor_add(Col, Row, Sum), text(Sum, T6),
    format("(3,1) + (1,4) -> ~w~n", [T6]),

    format("~n=================== Reductions ===================~n"),
    tensor_sum(R, RSum), tensor_mean(R, RMean), cs(RSum, T7), cs(RMean, T8),
    format("r.sum() = ~w, r.mean() = ~w~n", [T7, T8]),
    tensor_mean_axis(R, 0, ColMeans), text(ColMeans, T9),
    format("column means (axis=0): ~w~n", [T9]),

    format("~n=================== Linear algebra ===================~n"),
    tensor_from_values([2, 3], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], A),
    tensor_from_values([3, 2], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0], B),
    tensor_matmul(A, B, AB), text(AB, T10),
    format("matmul(2x3, 3x2) -> ~w~n", [T10]),
    tensor_from_values([3], [1.0, 2.0, 3.0], V1),
    tensor_from_values([3], [4.0, 5.0, 6.0], V2),
    tensor_dot(V1, V2, Dot), cs(Dot, T11),
    format("dot([1,2,3], [4,5,6]) = ~w~n", [T11]),
    tensor_outer(V1, V2, Outer), text(Outer, T12),
    format("outer(v1, v2) -> ~w~n", [T12]),

    format("~n=================== Comparisons & masks ===================~n"),
    tensor_full([3, 4], 6.0, Sixes),
    tensor_greater_equal(R, Sixes, Mask), text(Mask, T13),
    format("r >= 6 -> ~w~n", [T13]),
    tensor_sum(Mask, Count), cs(Count, T14),
    format("count(r >= 6) = ~w~n", [T14]),

    format("~n=================== A real dataset as a Tensor ===================~n"),
    data_iris(Iris),
    data_frame_nrows(Iris, N), Last is N - 1,
    Columns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],
    findall(V, ( between(0, Last, I), member(Column, Columns), data_frame_numeric_at(Iris, Column, I, V) ), Flat),
    tensor_from_values([N, 4], Flat, X),
    shape_string(X, T15),
    format("iris feature tensor shape: ~w~n", [T15]),

    tensor_mean_axis(X, 0, FeatureMeans),
    tensor_reshape(FeatureMeans, [1, 4], MeansRow),
    tensor_subtract(X, MeansRow, Centered),
    tensor_transpose(Centered, CenteredT),
    tensor_matmul(CenteredT, Centered, Scatter),
    text(FeatureMeans, T16),
    format("feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): ~w~n", [T16]),
    tensor_to_string(Scatter, 16, T17),
    format("(X-mean)^T (X-mean) [4x4 scatter matrix]: ~w~n", [T17]),

    findall(S, ( between(0, Last, I), data_frame_string_at(Iris, "Species", I, S) ), Species),
    tensor_from_string_values([N], Species, SpeciesTensor),
    tensor_from_string_values([1], ["setosa"], Setosa),
    tensor_equal(SpeciesTensor, Setosa, SetosaMask),
    tensor_sum(SetosaMask, SetosaCount), cs(SetosaCount, T18),
    format("setosa count = ~w (of ~w rows)~n", [T18, N]).

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
