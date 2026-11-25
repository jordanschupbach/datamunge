⍝ CMD: just TARGET=gaussian_process_regression_iris_ex run-apl
⍝ Gaussian process regression of Petal.Length on Petal.Width with posterior confidence intervals.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
FORMULA←'Petal.Length ~ Petal.Width'
IRIS←dm∆data∆iris
PUT 'iris: ',(⍕dm∆DataFrame∆nrows IRIS),' rows x ',(⍕dm∆DataFrame∆ncols IRIS),' cols'
PUT 'formula: ',FORMULA,NL

⍝ Both the length scale and the noise ratio are auto-selected by maximizing the exact log
⍝ marginal likelihood.
MODEL←dm∆GaussianProcessRegression∆new IRIS FORMULA
dm∆GaussianProcessRegression∆print_summary MODEL

FIT←MODEL dm∆GaussianProcessRegression∆plot_fit IRIS
FIT dm∆Plot∆save 'gpr_iris_fit.svg'
PROFILE←dm∆GaussianProcessRegression∆plot_length_scale_profile MODEL
PROFILE dm∆Plot∆save 'gpr_iris_length_scale_profile.svg'
PUT NL,'Saved gpr_iris_fit.svg and gpr_iris_length_scale_profile.svg'

⍝ Unlike every other regressor in this suite, a GP gives a genuine posterior confidence
⍝ interval at every point -- including far outside the training data, where it should widen
⍝ substantially as the model's uncertainty grows.
QUERY←dm∆DataFrame∆new ⍬
QUERY dm∆DataFrame∆add_numeric_column 'Petal.Width' (0.2 1.3 2.5 10.0)
DETAIL←MODEL dm∆GaussianProcessRegression∆predict_frame QUERY 'confidence'
PUT NL,'Predictions with 95% confidence intervals:'
PUT DETAIL dm∆DataFrame∆to_string ⍬
PUT '(Petal.Width=10.0 is far outside the training range [0.1, 2.5] -- note how much wider its'
PUT ' interval is than the in-range predictions.)'
dm∆Delete IRIS MODEL FIT PROFILE QUERY DETAIL
