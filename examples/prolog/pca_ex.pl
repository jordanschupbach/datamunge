% CMD: just TARGET=pca_ex run-prolog
% PCA on iris: summary, PC1 loadings, scores frame, score/scree plots, scaled vs unscaled.
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    data_iris(Iris),
    FeatureColumns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],

    % species labels, for the grouped (colored-by-species) scores plot
    data_frame_nrows(Iris, N), Last is N - 1,
    findall(S, (between(0, Last, I), data_frame_string_at(Iris, "Species", I, S)), Species),

    format("=================== PCA on iris (scaled) ===================~n"),
    pca_new(Iris, FeatureColumns, true, true, Pca),
    pca_print_summary(Pca),

    format("~nPC1 loadings (which original features drive it):~n"),
    pca_feature_names(Pca, Names),
    pca_component_loadings(Pca, 0, Loadings),
    forall(nth0(J, Names, Name),
           ( nth0(J, Loadings, Loading),
             format("  ~w: ~4f~n", [Name, Loading]) )),

    pca_scores_frame(Pca, Scores), data_frame_to_string(Scores, ScoresText),
    format("~nTraining scores as a DataFrame:~n~w~n", [ScoresText]),

    pca_plot_scores_grouped(Pca, Species, ScoresPlot),
    plot_save_svg(ScoresPlot, "pca_iris_scores.svg"),
    format("~nScores scatter (colored by species) saved as pca_iris_scores.svg~n"),

    pca_plot_scree(Pca, ScreePlot),
    plot_save_svg(ScreePlot, "pca_iris_scree.svg"),
    format("Scree plot saved as pca_iris_scree.svg~n"),

    format("~n=================== PCA on iris (unscaled) ===================~n"),
    pca_new(Iris, FeatureColumns, true, false, Unscaled),
    pca_explained_variance_ratio(Unscaled, [U0|_]),
    pca_explained_variance_ratio(Pca, [S0|_]),
    UP is 100 * U0, SP is 100 * S0,
    format("PC1 explains ~2f% of variance (vs. ~2f% scaled) -- Sepal.Length's larger raw variance dominates the unscaled covariance matrix.~n", [UP, SP]).
