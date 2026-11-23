% Smoke tests of the Rosetta-generated SWI-Prolog binding (just test-prolog):
% swipl tests/prolog/test_datamunge.pl, with the module on the library path (build/prolog/prefix/prolog).
:- use_module(library(datamunge)).
:- initialization(main, main).

% Prolog extensions of the C++ Callback class: override/1 clauses named after the C++ method (call).
doubler:override(call(_Self, X, Y)) :- Y is 2 * X.
failing:override(call(_Self, _X, _Y)) :- throw(from_prolog).

:- dynamic checks/1.
checks(0).

check(Goal, Message) :-
    retract(checks(N)), N1 is N + 1, assertz(checks(N1)),
    (   catch(Goal, E, (print_message(error, E), fail))
    ->  true
    ;   format("check failed: ~w~n", [Message]), halt(1)
    ).

main :-
    % Smoke test: call into the native library.
    hello,

    % std::vector<double> and std::pair<double,double> round trips.
    make_dvector(1.0, 2.0, 3.0, V),
    check(V == [1.0, 2.0, 3.0], 'std::vector<double> is a list'),
    check((sum_dvector(V, S), S =:= 6), sum_dvector),
    check((sum_dvector([0.5, 1.5], S2), S2 =:= 2), 'lists convert to std::vector'),
    make_dpair(1.25, 2.75, P),
    check(P == 1.25-2.75, 'std::pair is a Key-Value pair'),
    check((sum_dpair(P, S3), S3 =:= 4), sum_dpair),

    % The C++ Callback class, and Prolog extensions overriding its virtual call (a director).
    callback_new(Native),
    check((call_with_callback(3.0, Native, R1), R1 =:= 3), 'native Callback'),
    check((map_dvector_with_callback([1.0, 2.0, 3.0], Native, M1), sum_dvector(M1, T1), T1 =:= 6), 'native map'),
    callback_extend(doubler, none, Doubler),
    check((call_with_callback(3.0, Doubler, R2), R2 =:= 6), 'the Prolog override runs'),
    check((map_dvector_with_callback([1.0, 2.0, 3.0], Doubler, M2), sum_dvector(M2, T2), T2 =:= 12), 'override from C++ loops'),
    close_object(Native),

    % A Prolog exception inside an override is raised again after C++ returns.
    callback_extend(failing, none, Failing),
    check(catch((call_with_callback(1.0, Failing, _), fail), from_prolog, true), 'the Prolog exception propagates'),

    % C++ exceptions raise error(native_error(Kind, Message), _).
    check(catch((runif(1.0, 0.0, 3, _), fail), error(native_error(invalid_argument, _), _), true),
          'C++ exceptions raise native_error'),

    % Classes, methods and the policy namespace data.
    data_frame_new(Frame),
    data_frame_add_numeric_column(Frame, "x", [1.0, 2.0, 3.0]),
    check((data_frame_nrows(Frame, Rows), Rows =:= 3), data_frame),
    close_object(Frame),
    seql(0.0, 1.0, 5, Seq),
    check(Seq == [0.0, 0.25, 0.5, 0.75, 1.0], seql),
    check((data_iris(Iris), data_frame_nrows(Iris, IrisRows), IrisRows =:= 150), 'the data namespace'),

    checks(N),
    format("test_datamunge.pl: all ~d checks passed~n", [N]).
