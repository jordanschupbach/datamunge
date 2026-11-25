⍝ CMD: just TARGET=glm_iris_ex run-apl
⍝ Generalized linear models on iris: logistic regression (versicolor vs virginica) and Poisson regression.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ D FMT X: the number X with D decimals, C style ('-' for '¯', '0.5' for '.5').
∇Z←D FMT X
 Z←D⍕X
 Z←(∨\Z≠' ')/Z
 Z←((Z[1]='.')/'0'),Z
 Z←(1↑Z),(('¯.'≡2↑Z)/'0'),1↓Z
 Z[(Z='¯')/⍳⍴Z]←'-'
∇
⍝ ROUND X: round half to even (Python's round()), elementwise.
∇Z←ROUND X;F
 F←⌊X
 Z←F+(X-F)>0.5
 Z←Z+((X-F)=0.5)∧2|F
∇
IRIS←dm∆data∆iris
SPECIES←IRIS dm∆DataFrame∆pull_string 'Species'

⍝ Logistic regression: versicolor vs virginica only -- setosa is perfectly separable from
⍝ the other two on these predictors, which sends logistic regression's coefficients toward
⍝ +/-infinity (a genuine degeneracy of the method, not a bug) -- so this is the well-behaved
⍝ binary split for a demo.
KEEP←SPECIES∊'versicolor' 'virginica'
IS_VIRGINICA←1.0×(KEEP/SPECIES)∊⊂'virginica'
PETAL_LENGTH←KEEP/IRIS dm∆DataFrame∆pull_numeric 'Petal.Length'
PETAL_WIDTH←KEEP/IRIS dm∆DataFrame∆pull_numeric 'Petal.Width'

SUB←dm∆DataFrame∆new ⍬
SUB dm∆DataFrame∆add_numeric_column 'Petal.Length' PETAL_LENGTH
SUB dm∆DataFrame∆add_numeric_column 'Petal.Width' PETAL_WIDTH
SUB dm∆DataFrame∆add_numeric_column 'is_virginica' IS_VIRGINICA

PUT '=================== Logistic regression (binomial, logit link) ==================='
LOGIT←dm∆GLM∆new SUB 'is_virginica ~ Petal.Length + Petal.Width' 'binomial'
dm∆GLM∆print_summary LOGIT

FITTED←dm∆GLM∆fitted_values LOGIT
CORRECT←+/(FITTED≥0.5)=IS_VIRGINICA≥0.5
PUT NL,'Resubstitution accuracy at 0.5 threshold: ',(1 FMT 100×CORRECT÷⍴IS_VIRGINICA),'%'

LOGIT dm∆GLM∆save_diagnostic_plots 'glm_logistic_iris'
PUT NL,'Saved glm_logistic_iris_{residuals_vs_fitted,normal_qq,scale_location,residuals_vs_leverage}.svg'

⍝ Predicted-probability curve across Petal.Length, with Petal.Width held at its mean -- the
⍝ classic sigmoid shape of a fitted logistic regression, with a 95% confidence band.
WIDTH_MEAN←(+/PETAL_WIDTH)÷⍴PETAL_WIDTH
GRID_N←100
PL_MIN←(⌊/PETAL_LENGTH)-0.3
PL_MAX←(⌈/PETAL_LENGTH)+0.3
GRID_X←PL_MIN+(PL_MAX-PL_MIN)×(¯1+⍳GRID_N)÷GRID_N-1
GRID←dm∆DataFrame∆new ⍬
GRID dm∆DataFrame∆add_numeric_column 'Petal.Length' GRID_X
GRID dm∆DataFrame∆add_numeric_column 'Petal.Width' (GRID_N⍴WIDTH_MEAN)
CURVE←LOGIT dm∆GLM∆predict_frame GRID 'confidence'
PUT NL,'Predicted-probability curve (first 5 rows):'
PUT CURVE dm∆DataFrame∆to_string 5

⍝ Poisson regression, for contrast: same IRLS engine, different family/link.
PUT NL,'=================== Poisson regression (log link) ==================='
COUNT←ROUND IRIS dm∆DataFrame∆pull_numeric 'Sepal.Length'
COUNT_DATA←dm∆DataFrame∆new ⍬
COUNT_DATA dm∆DataFrame∆add_numeric_column 'Sepal.Width' (IRIS dm∆DataFrame∆pull_numeric 'Sepal.Width')
COUNT_DATA dm∆DataFrame∆add_numeric_column 'Petal.Length' (IRIS dm∆DataFrame∆pull_numeric 'Petal.Length')
COUNT_DATA dm∆DataFrame∆add_numeric_column 'count' COUNT

POISSON←dm∆GLM∆new COUNT_DATA 'count ~ Sepal.Width + Petal.Length' 'poisson'
dm∆GLM∆print_summary POISSON
dm∆Delete IRIS SUB LOGIT GRID CURVE COUNT_DATA POISSON
