% CMD: just TARGET=mds_ex run-prolog
% Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
:- use_module(library(datamunge)).
:- initialization(main, main).

main :-
    data_iris(Iris),
    FeatureColumns = ["Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"],
    data_frame_nrows(Iris, N), Last is N - 1,
    findall(S, (between(0, Last, I), data_frame_string_at(Iris, "Species", I, S)), Species),

    format("=================== Classical MDS on iris (euclidean) ===================~n"),
    mds_new(Iris, FeatureColumns, 2, "euclidean", Mds),
    mds_print_summary(Mds),

    mds_embedding_frame(Mds, Embedding), data_frame_to_string(Embedding, EmbeddingText),
    format("~nEmbedding as a DataFrame:~n~w~n", [EmbeddingText]),

    mds_plot_embedding_grouped(Mds, Species, Plot),
    plot_save_svg(Plot, "mds_iris_embedding.svg"),
    format("~nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg~n"),

    format("~n=================== Classical MDS on iris (manhattan) ===================~n"),
    mds_new(Iris, FeatureColumns, 2, "manhattan", ManhattanMds),
    mds_goodness_of_fit(Mds, G1),
    mds_goodness_of_fit(ManhattanMds, G2),
    format("Goodness of fit: euclidean=~4f, manhattan=~4f~n", [G1, G2]).
