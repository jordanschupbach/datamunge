⍝ CMD: just TARGET=svm_iris_ex run-apl
⍝ Support vector machine on iris: RBF vs linear kernel, confusion matrix and predictions for new flowers.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
FORMULA←'Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width'
IRIS←dm∆data∆iris
PUT 'iris: ',(⍕dm∆DataFrame∆nrows IRIS),' rows x ',(⍕dm∆DataFrame∆ncols IRIS),' cols',NL

PUT '=== RBF kernel (default) ==='
RBF←dm∆SVM∆new IRIS FORMULA
dm∆SVM∆print_summary RBF
PUT NL,'Confusion matrix (rows = actual, cols = predicted):'
CM←dm∆SVM∆confusion_matrix RBF
PUT CM dm∆DataFrame∆to_string ⍬

PUT NL,'=== Linear kernel, for comparison ==='
LINEAR←dm∆SVM∆new IRIS FORMULA 'linear'
PUT 'Training accuracy: ',(⍕100×dm∆SVM∆training_accuracy LINEAR),'%'
PUT 'Support vectors: ',⍕dm∆SVM∆num_support_vectors LINEAR

NEWDATA←dm∆DataFrame∆new ⍬
NEWDATA dm∆DataFrame∆add_numeric_column 'Sepal.Length' (5.1 6.0 6.5 6.2)
NEWDATA dm∆DataFrame∆add_numeric_column 'Sepal.Width' (3.5 2.7 3.0 2.8)
NEWDATA dm∆DataFrame∆add_numeric_column 'Petal.Length' (1.4 4.5 5.5 4.8)
NEWDATA dm∆DataFrame∆add_numeric_column 'Petal.Width' (0.2 1.5 2.0 1.8)

PUT NL,'RBF predictions for new flowers (votes out of 3 one-vs-one pairs):'
FRAME←RBF dm∆SVM∆predict_frame NEWDATA
PUT FRAME dm∆DataFrame∆to_string ⍬
dm∆Delete IRIS RBF CM LINEAR NEWDATA FRAME
