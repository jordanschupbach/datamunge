% CMD: just TARGET=optim_advanced_ex run-prolog
% Advanced datamunge optimizers (ProximalGradient, FISTA, LevenbergMarquardt, Newton, TrustRegionNewton,
% AugmentedLagrangian, SQP, InteriorPoint, BayesianOptimization, RBFGaussianProcessSurrogate) from Prolog.
%
% Every one of these optimizers takes an interface (ProximalFunction / HessianFunction /
% EqualityConstrainedFunction / InequalityConstrainedFunction / ResidualFunction, or
% ArbitraryFunction for BayesianOptimization) that the C++ library leaves abstract: callers
% implement the objective. From Prolog, extend the interface class (CLASS_extend(Module, Data, Object))
% and give Module override/1 clauses named after its virtual methods; the optimizer then calls the
% Prolog overrides. Coordinates updated in place (the C++ parameter is a non-const
% std::vector<double>&) are an input and an output argument: newton_optimize(N, F, Start0, Start, Value).
:- use_module(library(datamunge)).
:- initialization(main, main).

% Newton's method objective: f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
quadratic:override(evaluate(_Self, [X, Y], F)) :-
    F is (X - 1) ** 2 + 10 * (Y + 2) ** 2.
quadratic:override(gradient(_Self, [X, Y], [GX, GY])) :-
    GX is 2 * (X - 1), GY is 20 * (Y + 2).
quadratic:override(hessian(_Self, _Coordinates, [[2.0, 0.0], [0.0, 20.0]])).

% Levenberg-Marquardt residuals of y = a * exp(b t) against exact data from a = 2, b = 0.5; the
% object's data is the list of sample times.
exponential:override(residuals(Self, [A, B], Residuals)) :-
    object_data(Self, Ts),
    findall(R, (member(T, Ts), R is A * exp(B * T) - 2 * exp(0.5 * T)), Residuals).
exponential:override(jacobian(Self, [A, B], Jacobian)) :-
    object_data(Self, Ts),
    findall([D1, D2], (member(T, Ts), D1 is exp(B * T), D2 is A * T * exp(B * T)), Jacobian).

% An interface method a subclass leaves out throws when the optimizer calls it.
incomplete:override(evaluate(_Self, _Coordinates, 0.0)).

% C# prints these objects with the binding's default ToString, the class name and "()"; Prolog has
% no such text, so the class (object_class/2) stands in.
print_object(Label, Object) :-
    object_class(Object, Class),
    format("~w: ~w()~n", [Label, Class]).

main :-
    format("=================== RBFGaussianProcessSurrogate (concrete BayesianSurrogate) ===================~n"),
    % Fit a 1D surrogate to a simple bowl y = (x-1)^2 sampled at a few points,
    % then check the acquisition function favors an unexplored region.
    Xs = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
    maplist([X, Y]>>(Y is (X - 1) * (X - 1)), Xs, Ys),
    rbf_gaussian_process_surrogate_new(1.0, 1.0e-6, Surrogate),
    maplist([X, [X]]>>true, Xs, Points),
    rbf_gaussian_process_surrogate_fit(Surrogate, Points, Ys),
    min_list(Ys, MinY), Incumbent is min(0.0, MinY),  % best observed value so far
    cs(Incumbent, IncumbentText),
    format("incumbent (best observed y): ~w~n", [IncumbentText]),
    forall(member(X, [0.25, 1.0, 2.5]),
           ( rbf_gaussian_process_surrogate_acquisition(Surrogate, [X], Incumbent, Acq),
             format("acquisition(x=~2f) = ~6f~n", [X, Acq]) )),

    format("~n=================== Options structs + optimizer construction (all 10 new classes) ===================~n"),
    % Every new Options struct and optimizer class constructs cleanly, proving
    % the binding for the whole batch is wired correctly.
    proximal_gradient_options_new(PgOptions), proximal_gradient_options_set_step_size(PgOptions, 0.05),
    proximal_gradient_new(PgOptions, Pg), print_object("ProximalGradient", Pg),
    fista_options_new(FistaOptions), fista_new(FistaOptions, Fista), print_object("FISTA", Fista),
    levenberg_marquardt_options_new(LmOptions), levenberg_marquardt_options_set_max_iterations(LmOptions, 200),
    levenberg_marquardt_new(LmOptions, Lm), print_object("LevenbergMarquardt", Lm),
    newton_options_new(NewtonOptions), newton_new(NewtonOptions, Newton), print_object("Newton", Newton),
    trust_region_newton_options_new(TrnOptions), trust_region_newton_new(TrnOptions, Trn),
    print_object("TrustRegionNewton", Trn),
    augmented_lagrangian_options_new(AlOptions), augmented_lagrangian_new(AlOptions, Al),
    print_object("AugmentedLagrangian", Al),
    sqp_options_new(SqpOptions), sqp_new(SqpOptions, Sqp), print_object("SQP", Sqp),
    interior_point_options_new(IpOptions), interior_point_new(IpOptions, Ip), print_object("InteriorPoint", Ip),
    bayesian_optimization_options_new(BoOptions), bayesian_optimization_options_set_initial_samples(BoOptions, 8),
    bayesian_optimization_new(BoOptions, Bo), print_object("BayesianOptimization", Bo),

    format("~n=================== Prolog objectives (subclassing the C++ interfaces) ===================~n"),
    % Newton's method on f(x, y) = (x - 1)^2 + 10 (y + 2)^2 with exact gradient and Hessian.
    hessian_function_extend(quadratic, none, Quadratic),
    newton_optimize(Newton, Quadratic, [5.0, 5.0], [S0, S1|_], Value),
    fmt_g(Value, 6, ValueText),
    format("Newton: f = ~w at (~6f, ~6f) (exact minimum 0 at (1, -2))~n", [ValueText, S0, S1]),

    % Levenberg-Marquardt fitting y = a * exp(b t) to exact data from a = 2, b = 0.5.
    residual_function_extend(exponential, [0.0, 0.5, 1.0, 1.5, 2.0], Exponential),
    levenberg_marquardt_optimize(Lm, Exponential, [1.0, 0.1], [PA, PB|_], Sse),
    fmt_g(Sse, 3, SseText),
    format("LevenbergMarquardt: residual = ~w, a = ~6f, b = ~6f (exact: 2, 0.5)~n", [SseText, PA, PB]),

    % An interface method a subclass leaves out throws when the optimizer calls it.
    hessian_function_extend(incomplete, none, Incomplete),
    catch(( newton_optimize(Newton, Incomplete, [0.0, 0.0], _, _), Err = none ), E, Err = E),
    (   Err == none
    ->  Ok = "True", Mentions = "False"
    ;   Ok = "False",
        message_text(Err, Text),
        ( sub_string(Text, _, _, _, "gradient") -> Mentions = "True" ; Mentions = "False" )
    ),
    format("missing gradient(): ok=~w, error mentions it: ~w~n", [Ok, Mentions]).

message_text(error(native_error(_, Message), _), Message) :- !.
message_text(E, Text) :- term_string(E, Text).

% fmt_g(+X, +Precision, -String): X as .NET's ToString("G<Precision>") prints it (C's %g, with
% .NET's E+XX exponent).
fmt_g(X, _, "0") :- X =:= 0, !.
fmt_g(X, Precision, S) :-
    P1 is Precision - 1,
    format(string(Sci), "~*e", [P1, X]),      % d.ddddde[+-]xx, correctly rounded
    sub_string(Sci, E, 1, _, "e"), !,
    sub_string(Sci, 0, E, _, Mantissa0),
    E1 is E + 1, sub_string(Sci, E1, _, 0, ExpText), number_string(Exponent, ExpText),
    (   ( Exponent < -5 ; Exponent >= Precision )
    ->  trim_zeros(Mantissa0, Mantissa),
        A is abs(Exponent), ( Exponent < 0 -> Sign = "-" ; Sign = "+" ),
        format(string(S), "~wE~w~|~`0t~d~2+", [Mantissa, Sign, A])
    ;   Digits is P1 - Exponent,
        format(string(Fixed), "~*f", [Digits, X]),
        trim_zeros(Fixed, S)
    ).

trim_zeros(S0, S) :-
    (   sub_string(S0, _, _, _, ".")
    ->  string_codes(S0, C0), reverse(C0, R0),
        drop_zeros(R0, R1), ( R1 = [0'.|R2] -> true ; R2 = R1 ),
        reverse(R2, C), string_codes(S, C)
    ;   S = S0
    ).
drop_zeros([0'0|T], R) :- !, drop_zeros(T, R).
drop_zeros(R, R).

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
