% CMD: just TARGET=autodiff_ex run-prolog
% Automatic differentiation: forward mode (Dual), reverse mode (Tape/Var), Hessians (HyperDual).
:- use_module(library(datamunge)).
:- initialization(main, main).

% Note: the C++ autodiff_ex.cpp additionally demonstrates the generic template drivers
% (derivative/gradient_forward/gradient_reverse/jacobian_forward/hessian), which can't be
% bound. This port demonstrates the same ideas with the bound named-method API.
% var_index/2 is the Var's position in the adjoint vector tape_backward/3 returns.

rosenbrock_dual(X0, X1, F) :-
    dual_new(1.0, 0.0, One), dual_subtract(One, X0, A),
    dual_multiply(X0, X0, X0Sq), dual_subtract(X1, X0Sq, B),
    dual_multiply(A, A, AA), dual_multiply(B, B, BB), dual_multiply_scalar(BB, 100.0, BB100),
    dual_add(AA, BB100, F).

rosenbrock_value(X0, X1, F) :- F is (1.0 - X0) ** 2 + 100.0 * (X1 - X0 * X0) ** 2.

% Build the Rosenbrock graph on Tape; F is the output Var.
rosenbrock_var(Tape, X0, X1, F) :-
    var_new(Tape, 1.0, One), var_subtract(One, X0, A),
    var_multiply(X0, X0, X0Sq), var_subtract(X1, X0Sq, B),
    var_multiply(A, A, AA), var_multiply(B, B, BB), var_multiply_scalar(BB, 100.0, BB100),
    var_add(AA, BB100, F).

rosenbrock_hyper_dual(X0, X1, F) :-
    hyper_dual_new(1.0, 0.0, 0.0, 0.0, One), hyper_dual_subtract(One, X0, A),
    hyper_dual_multiply(X0, X0, X0Sq), hyper_dual_subtract(X1, X0Sq, B),
    hyper_dual_multiply(A, A, AA), hyper_dual_multiply(B, B, BB), hyper_dual_multiply_scalar(BB, 100.0, BB100),
    hyper_dual_add(AA, BB100, F).

adjoint(Adjoints, V, A) :- var_index(V, I), nth0(I, Adjoints, A).

% f(x,y) = [x^2, xy, y^3]
vector_fn(X, Y, [F1, F2, F3]) :-
    dual_multiply(X, X, F1), dual_multiply(X, Y, F2), dual_multiply(Y, Y, YY), dual_multiply(YY, Y, F3).

gradient_descent(Step, NSteps, _, X, Y, X, Y) :- Step >= NSteps, !.
gradient_descent(Step, NSteps, LearningRate, X0, Y0, X, Y) :-
    tape_new(Tape),
    var_new(Tape, X0, VX), var_new(Tape, Y0, VY),
    rosenbrock_var(Tape, VX, VY, VF),
    tape_backward(Tape, VF, Grad),
    var_value(VF, Loss),
    adjoint(Grad, VX, GX), adjoint(Grad, VY, GY),
    X1 is X0 - LearningRate * GX,
    Y1 is Y0 - LearningRate * GY,
    (   ( Step =:= 0 ; Step =:= NSteps - 1 )
    ->  cs(Loss, LS), cs(X1, XS), cs(Y1, YS),
        format("step ~w: loss = ~w, x = [~w, ~w]~n", [Step, LS, XS, YS])
    ;   true
    ),
    Step1 is Step + 1,
    gradient_descent(Step1, NSteps, LearningRate, X1, Y1, X, Y).

main :-
    format("=================== Forward mode: scalar derivative ===================~n"),
    dual_new(2.0, 1.0, X),  % seed derivative = 1 to read df/dx directly
    dual_multiply(X, X, XX), dual_multiply(XX, X, XXX), dual_multiply_scalar(X, 2.0, X2),
    dual_subtract(XXX, X2, F),  % x^3 - 2x
    dual_derivative(F, D), cs(D, DS),
    format("f(x) = x^3 - 2x, f'(2) = ~w (exact: 10)~n", [DS]),

    format("~n=================== Reverse mode: build a graph by hand ===================~n"),
    tape_new(Tape),
    var_new(Tape, 2.0, A),
    var_new(Tape, 3.0, B),
    var_multiply(A, B, AB), var_sin(A, SinA), var_add(AB, SinA, Y),
    var_value(Y, YV), cs(YV, YS),
    format("y = a*b + sin(a) at a=2, b=3 -> y = ~w~n", [YS]),
    tape_backward(Tape, Y, Adjoint),
    adjoint(Adjoint, A, DA), cs(DA, DAS),
    format("dy/da = ~w (exact: b + cos(a))~n", [DAS]),
    adjoint(Adjoint, B, DB), cs(DB, DBS),
    format("dy/db = ~w (exact: a)~n", [DBS]),

    format("~n=================== Forward vs reverse mode agree on the Rosenbrock function ===================~n"),
    PX = 0.0, PY = 0.0,
    % Forward mode: one Dual pass per partial derivative, seeding the direction of interest.
    dual_new(PX, 1.0, DX1), dual_new(PY, 0.0, DY1), rosenbrock_dual(DX1, DY1, R1), dual_derivative(R1, GX),
    dual_new(PX, 0.0, DX2), dual_new(PY, 1.0, DY2), rosenbrock_dual(DX2, DY2, R2), dual_derivative(R2, GY),
    % Reverse mode: one Tape/Var pass computes every partial at once.
    tape_new(Tape2),
    var_new(Tape2, PX, VX), var_new(Tape2, PY, VY),
    rosenbrock_var(Tape2, VX, VY, RV),
    tape_backward(Tape2, RV, Grad),
    rosenbrock_value(PX, PY, F00), cs(F00, F00S),
    format("f(0,0) = ~w~n", [F00S]),
    cs(GX, GXS), cs(GY, GYS),
    format("gradient (forward mode): [~w, ~w]~n", [GXS, GYS]),
    adjoint(Grad, VX, AX), adjoint(Grad, VY, AY), cs(AX, AXS), cs(AY, AYS),
    format("gradient (reverse mode): [~w, ~w]~n", [AXS, AYS]),

    format("~n=================== Jacobian of a vector-valued function ===================~n"),
    % f(x,y) = [x^2, xy, y^3] at (2,3) -- one Dual pass per (output, input) pair.
    findall(Column,
            ( member(SX-SY, [1.0-0.0, 0.0-1.0]),
              dual_new(2.0, SX, JX), dual_new(3.0, SY, JY),
              vector_fn(JX, JY, Outputs),
              maplist(dual_derivative, Outputs, Column) ),
            [Col0, Col1]),
    format("f(x,y) = [x^2, xy, y^3] at (2,3), Jacobian:~n"),
    forall(nth0(Row, Col0, J0),
           ( nth0(Row, Col1, J1), cs(J0, J0S), cs(J1, J1S),
             format("  [~w, ~w]~n", [J0S, J1S]) )),

    format("~n=================== Hessian via second-order forward mode (HyperDual) ===================~n"),
    % One HyperDual pass per (i, j) pair reads off d^2f/dxi dxj directly from eps1eps2.
    MX = 1.0, MY = 1.0,  % the Rosenbrock function's minimum
    Seeds = [1.0-0.0, 0.0-1.0],
    format("Hessian of the Rosenbrock function at its minimum (1,1):~n"),
    forall(member(E1X-E1Y, Seeds),
           ( findall(H,
                     ( member(E2X-E2Y, Seeds),
                       hyper_dual_new(MX, E1X, E2X, 0.0, HX), hyper_dual_new(MY, E1Y, E2Y, 0.0, HY),
                       rosenbrock_hyper_dual(HX, HY, HF), hyper_dual_eps1eps2(HF, H) ),
                     [H0, H1]),
             cs(H0, H0S), cs(H1, H1S),
             format("  [~w, ~w]~n", [H0S, H1S]) )),

    format("~n=================== Gradient descent driven by reverse-mode gradients ===================~n"),
    gradient_descent(0, 2000, 0.001, -1.2, 1.0, _, _),  % the classic Rosenbrock start
    format("(true minimum is at [1, 1] with loss 0)~n").

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
