⍝ CMD: just TARGET=kernel_regression_iris_ex run-apl
⍝ Nadaraya-Watson kernel regression of Petal.Length on Petal.Width: CV bandwidth vs too small/large.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ G X: up to 6 significant digits, roughly C's %g ('-' for '¯', 'e' exponent).
∇Z←G X;⎕PP
 ⎕PP←6
 Z←⍕X
 Z[(Z='¯')/⍳⍴Z]←'-'
 Z[(Z='E')/⍳⍴Z]←'e'
∇
⍝ SAVE writes the plot handle P to PATH and releases it.
∇PATH SAVE P
 P dm∆Plot∆save PATH
 dm∆Delete P
∇
⍝ FITLINE describes a fit's leave-one-out R-squared and RMSE.
∇Z←FITLINE M
 Z←'LOO R-squared=',(G dm∆KernelRegression∆r_squared M),'  LOO RMSE=',G dm∆KernelRegression∆rmse M
∇
FORMULA←'Petal.Length ~ Petal.Width'
IRIS←dm∆data∆iris
PUT 'iris: ',(⍕dm∆DataFrame∆nrows IRIS),' rows x ',(⍕dm∆DataFrame∆ncols IRIS),' cols'
PUT 'formula: ',FORMULA,NL

⍝ The bandwidth is auto-selected via leave-one-out CV.
MODEL←dm∆KernelRegression∆new IRIS FORMULA
dm∆KernelRegression∆print_summary MODEL

'kernel_regression_iris_fit.svg' SAVE MODEL dm∆KernelRegression∆plot_fit IRIS
'kernel_regression_iris_cv.svg' SAVE dm∆KernelRegression∆plot_cv_curve MODEL
PUT NL,'Saved kernel_regression_iris_fit.svg and kernel_regression_iris_cv.svg'

⍝ A too-small and a too-large bandwidth, for comparison -- the classic kernel regression
⍝ bias-variance story: dm∆KernelRegression∆new data formula kernel bandwidth.
SMALL←dm∆KernelRegression∆new IRIS FORMULA 'gaussian' 0.05
PUT NL,'bandwidth=0.05 (too small): ',FITLINE SMALL
'kernel_regression_iris_fit_small_bandwidth.svg' SAVE SMALL dm∆KernelRegression∆plot_fit IRIS

LARGE←dm∆KernelRegression∆new IRIS FORMULA 'gaussian' 5.0
PUT 'bandwidth=5.0 (too large):  ',FITLINE LARGE
'kernel_regression_iris_fit_large_bandwidth.svg' SAVE LARGE dm∆KernelRegression∆plot_fit IRIS

PUT 'bandwidth=',(G dm∆KernelRegression∆bandwidth MODEL),' (CV-selected): ',FITLINE MODEL
PUT NL,'Saved kernel_regression_iris_fit_small_bandwidth.svg and kernel_regression_iris_fit_large_bandwidth.svg'
dm∆Delete IRIS MODEL SMALL LARGE
