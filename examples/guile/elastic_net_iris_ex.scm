(use-modules (datamunge))

(define FORMULA "Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width")

(define iris (DataFrame-iris))
(format #t "iris: ~a rows x ~a cols\n" (DataFrame-nrows iris) (DataFrame-ncols iris))
(format #t "formula: ~a\n\n" FORMULA)

(format #t "=================== Ridge ===================\n")
(define ridge (new-Ridge iris FORMULA))
(Ridge-print-summary ridge)

(format #t "\n=================== Lasso ===================\n")
(define lasso (new-Lasso iris FORMULA))
(Lasso-print-summary lasso)

(format #t "\n================= Elastic Net =================\n")
(define elastic (new-ElasticNet iris FORMULA 0.5))
(ElasticNet-print-summary elastic)

(format #t "\nSaved figures showing how each model's coefficients respond to the regularization strength, and the cross-validation curve used to pick it:\n")

(Plot-save (Ridge-plot-coefficient-path ridge) "elastic_net_ridge_path.svg")
(Plot-save (Ridge-plot-cv-curve ridge) "elastic_net_ridge_cv.svg")
(format #t "  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg\n")

(Plot-save (Lasso-plot-coefficient-path lasso) "elastic_net_lasso_path.svg")
(Plot-save (Lasso-plot-cv-curve lasso) "elastic_net_lasso_cv.svg")
(format #t "  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg\n")

(Plot-save (ElasticNet-plot-coefficient-path elastic) "elastic_net_elasticnet_path.svg")
(Plot-save (ElasticNet-plot-cv-curve elastic) "elastic_net_elasticnet_cv.svg")
(format #t "  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg\n")
