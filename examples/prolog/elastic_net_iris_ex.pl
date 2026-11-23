% CMD: just TARGET=elastic_net_iris_ex run-prolog
% Ridge, Lasso and Elastic Net on iris, with coefficient-path and CV-curve figures.
:- use_module(library(datamunge)).
:- initialization(main, main).

formula("Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width").

main :-
    formula(Formula),
    data_iris(Iris),
    data_frame_nrows(Iris, Rows), data_frame_ncols(Iris, Cols),
    format("iris: ~w rows x ~w cols~n", [Rows, Cols]),
    format("formula: ~w~n~n", [Formula]),

    format("=================== Ridge ===================~n"),
    ridge_new(Iris, Formula, Ridge),
    ridge_print_summary(Ridge),

    format("~n=================== Lasso ===================~n"),
    lasso_new(Iris, Formula, Lasso),
    lasso_print_summary(Lasso),

    format("~n================= Elastic Net =================~n"),
    % elastic_net_new(Data, Formula, Alpha=0.5, Lambda, NLambda, CvFolds, Standardize, Seed, Model)
    elastic_net_new(Iris, Formula, 0.5, Elastic),
    elastic_net_print_summary(Elastic),

    format("~nSaved figures showing how each model's coefficients respond to the regularization strength, and the cross-validation curve used to pick it:~n"),

    ridge_plot_coefficient_path(Ridge, RidgePath), ridge_plot_cv_curve(Ridge, RidgeCv),
    lasso_plot_coefficient_path(Lasso, LassoPath), lasso_plot_cv_curve(Lasso, LassoCv),
    elastic_net_plot_coefficient_path(Elastic, ElasticPath), elastic_net_plot_cv_curve(Elastic, ElasticCv),
    Figures = [ figure(RidgePath, RidgeCv, "ridge", "ridge:      "),
                figure(LassoPath, LassoCv, "lasso", "lasso:      "),
                figure(ElasticPath, ElasticCv, "elasticnet", "elastic net:") ],
    forall(member(figure(Path, Cv, Name, Label), Figures),
           ( format(string(PathFile), "elastic_net_~w_path.svg", [Name]),
             format(string(CvFile), "elastic_net_~w_cv.svg", [Name]),
             plot_save(Path, PathFile),
             plot_save(Cv, CvFile),
             format("  ~w ~w, ~w~n", [Label, PathFile, CvFile]) )).
