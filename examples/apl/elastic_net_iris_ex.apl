⍝ CMD: just TARGET=elastic_net_iris_ex run-apl
⍝ Ridge, lasso and elastic-net regression on iris, with coefficient-path and cross-validation plots.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
FORMULA←'Petal.Length ~ Sepal.Length + Sepal.Width + Petal.Width + Sepal.Length:Petal.Width'
IRIS←dm∆data∆iris
PUT 'iris: ',(⍕dm∆DataFrame∆nrows IRIS),' rows x ',(⍕dm∆DataFrame∆ncols IRIS),' cols'
PUT 'formula: ',FORMULA,NL

PUT '=================== Ridge ==================='
RIDGE←dm∆Ridge∆new IRIS FORMULA
dm∆Ridge∆print_summary RIDGE

PUT NL,'=================== Lasso ==================='
LASSO←dm∆Lasso∆new IRIS FORMULA
dm∆Lasso∆print_summary LASSO

PUT NL,'================= Elastic Net ================='
⍝ dm∆ElasticNet∆new data formula alpha lambda n_lambda cv_folds standardize seed
ELASTIC←dm∆ElasticNet∆new IRIS FORMULA 0.5
dm∆ElasticNet∆print_summary ELASTIC

PUT NL,'Saved figures showing how each model''s coefficients respond to the regularization strength, and the cross-validation curve used to pick it:'

P1←dm∆Ridge∆plot_coefficient_path RIDGE
P1 dm∆Plot∆save 'elastic_net_ridge_path.svg'
P2←dm∆Ridge∆plot_cv_curve RIDGE
P2 dm∆Plot∆save 'elastic_net_ridge_cv.svg'
PUT '  ridge:       elastic_net_ridge_path.svg, elastic_net_ridge_cv.svg'

P3←dm∆Lasso∆plot_coefficient_path LASSO
P3 dm∆Plot∆save 'elastic_net_lasso_path.svg'
P4←dm∆Lasso∆plot_cv_curve LASSO
P4 dm∆Plot∆save 'elastic_net_lasso_cv.svg'
PUT '  lasso:       elastic_net_lasso_path.svg, elastic_net_lasso_cv.svg'

P5←dm∆ElasticNet∆plot_coefficient_path ELASTIC
P5 dm∆Plot∆save 'elastic_net_elasticnet_path.svg'
P6←dm∆ElasticNet∆plot_cv_curve ELASTIC
P6 dm∆Plot∆save 'elastic_net_elasticnet_cv.svg'
PUT '  elastic net: elastic_net_elasticnet_path.svg, elastic_net_elasticnet_cv.svg'
dm∆Delete IRIS RIDGE LASSO ELASTIC P1 P2 P3 P4 P5 P6
