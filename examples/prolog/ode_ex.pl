% CMD: just TARGET=ode_ex run-prolog
% ODEs: built-in named systems, harmonic-oscillator energy check and a Lorenz phase plot.
:- use_module(library(datamunge)).
:- initialization(main, main).

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

% Approximate C's %g: six significant digits, no padding.
fmt_g6(X, S) :- fmt_g(X, 6, S).

last_state(Sol, State) :-
    ode_solution_size(Sol, N), Last is N - 1,
    ode_solution_state_at(Sol, Last, State).

% The time vector followed by the requested state components (0-based), one value per step.
solution_series(Sol, Components, Times, Series) :-
    ode_solution_size(Sol, N), Last is N - 1,
    numlist(0, Last, Steps),
    maplist(ode_solution_state_at(Sol), Steps, States),
    maplist(ode_solution_time_at(Sol), Steps, Times),
    maplist([C, Column]>>maplist(nth0(C), States, Column), Components, Series).

rk4_solver(Step, Solver) :-
    ode_options_new(Options),
    ode_options_set_method(Options, rk4),
    ode_options_set_step_size(Options, Step),
    ode_solver_new(Options, Solver).

% NOTE: like the Lua version this example uses the built-in named systems via solve_builtin.

main :-
    format("=================== Every named built-in system (no director needed) ===================~n"),
    ode_solver_new(Solver),
    Beta is 8.0 / 3,
    Systems = [ system("exponential_decay", [1.0], [1.0]),
                system("logistic_growth", [1.0, 1.0], [0.5]),
                system("harmonic_oscillator", [1.0], [1.0, 0.0]),
                system("van_der_pol", [1.0], [2.0, 0.0]),
                system("lorenz", [10.0, 28.0, Beta], [1.0, 1.0, 1.0]) ],
    forall(member(system(Name, Parameters, Y0), Systems),
           ( ode_solver_solve_builtin(Solver, Name, Parameters, Y0, 0.0, 1.0, Sol),
             ode_solution_steps_taken(Sol, Steps),
             last_state(Sol, State),
             maplist(fmt_g6, State, StateText),
             atomic_list_concat(StateText, ', ', Joined),
             format("~w~t~22| steps=~w~t~13+ final state=[~w]~n", [Name, Steps, Joined]) )),

    format("~n=================== Harmonic oscillator (energy conservation) ===================~n"),
    rk4_solver(0.01, Rk4a),
    ode_solver_solve_builtin(Rk4a, "harmonic_oscillator", [1.0], [1.0, 0.0], 0.0, 20.0, HSol),
    last_state(HSol, [X, V|_]),
    Cos20 is cos(20.0),
    format("x(20) = ~6f (cos(20) = ~6f)~n", [X, Cos20]),
    Energy is X * X + V * V,
    format("energy x^2+v^2 = ~6f (should stay near 1.0)~n", [Energy]),
    solution_series(HSol, [0, 1], Ts, [Xs, Vs]),
    r_plot_plot(Ts, Xs, "l", "x(t)", HPlot),
    r_plot_lines(HPlot, Ts, Vs, "v(t)", _),
    r_plot_title(HPlot, "Harmonic Oscillator", H1), r_plot_x_label(H1, "t", H2), r_plot_y_label(H2, "state", _),
    plot_save_svg(HPlot, "ode_harmonic_oscillator_prolog.svg"),
    format("wrote ode_harmonic_oscillator_prolog.svg~n"),

    format("~n=================== Lorenz attractor (phase plane) ===================~n"),
    rk4_solver(0.005, Rk4b),
    ode_solver_solve_builtin(Rk4b, "lorenz", [10.0, 28.0, Beta], [1.0, 1.0, 1.0], 0.0, 25.0, LSol),
    ode_solution_steps_taken(LSol, LSteps),
    format("steps_taken = ~w~n", [LSteps]),
    solution_series(LSol, [0, 2], _, [LXs, LZs]),
    r_plot_plot(LXs, LZs, "l", "trajectory", LPlot),
    r_plot_title(LPlot, "Lorenz Attractor (x-z phase plane)", L1), r_plot_x_label(L1, "x", L2), r_plot_y_label(L2, "z", _),
    plot_save_svg(LPlot, "ode_lorenz_phase_plane_prolog.svg"),
    format("wrote ode_lorenz_phase_plane_prolog.svg~n").
