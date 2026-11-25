⍝ CMD: just TARGET=lda_iris_ex run-apl
⍝ Linear discriminant analysis of iris: summary, confusion matrix, predictions and a discriminant plot.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
IRIS←dm∆data∆iris
PUT 'iris: ',(⍕dm∆DataFrame∆nrows IRIS),' rows x ',(⍕dm∆DataFrame∆ncols IRIS),' cols',NL

MODEL←dm∆LDA∆new IRIS 'Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width'
dm∆LDA∆print_summary MODEL

PUT NL,'Confusion matrix (rows = actual, cols = predicted):'
CM←dm∆LDA∆confusion_matrix MODEL
PUT CM dm∆DataFrame∆to_string ⍬

NEWDATA←dm∆DataFrame∆new ⍬
NEWDATA dm∆DataFrame∆add_numeric_column 'Sepal.Length' (5.1 6.0 6.5 6.2)
NEWDATA dm∆DataFrame∆add_numeric_column 'Sepal.Width' (3.5 2.7 3.0 2.8)
NEWDATA dm∆DataFrame∆add_numeric_column 'Petal.Length' (1.4 4.5 5.5 4.8)
NEWDATA dm∆DataFrame∆add_numeric_column 'Petal.Width' (0.2 1.5 2.0 1.8)

PUT NL,'Predictions for new flowers:'
PRED←MODEL dm∆LDA∆predict_frame NEWDATA
PUT PRED dm∆DataFrame∆to_string ⍬

MODEL dm∆LDA∆save_discriminant_plot 'lda_iris_discriminants.svg'
PUT NL,'Saved discriminant plot as lda_iris_discriminants.svg'
dm∆Delete IRIS MODEL CM NEWDATA PRED
