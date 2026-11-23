% CMD: just TARGET=bspline_ex run-prolog
% Tensor-product B-spline regression: fit z ~ bs(x, y) to a smooth surface and predict.
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    % Sample a smooth surface z = sin(x) + y^2 on an 8x8 grid over [0, 1]^2.
    findall(XV-YV, (between(0, 7, IY), between(0, 7, IX), XV is IX / 7.0, YV is IY / 7.0), Grid),
    pairs_keys_values(Grid, X, Y),
    findall(ZV, (member(XV-YV, Grid), ZV is sin(XV) + YV * YV), Z),
    data_frame_new(Data),
    data_frame_add_numeric_column(Data, "x", X),
    data_frame_add_numeric_column(Data, "y", Y),
    data_frame_add_numeric_column(Data, "z", Z),

    lm_new(Data, "z ~ bs(x, y)", Surface),
    format("Fitted z ~~ bs(x, y)~n"),
    lm_print_summary(Surface),

    data_frame_new(NewPoints),
    data_frame_add_numeric_column(NewPoints, "x", [0.25, 0.75]),
    data_frame_add_numeric_column(NewPoints, "y", [0.50, 0.25]),
    lm_predict(Surface, NewPoints, [P0, P1|_]),
    format("Predictions: [~6f, ~6f]~n", [P0, P1]).
