⍝ CMD: just TARGET=xgboost_iris_ex run-apl
⍝ XGBoost classifier on iris: summary, confusion matrix, training deviance and the effect of L2 regularization.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ MISCLASSIFIED prints every row whose prediction differs from Species; Z is their count.
∇Z←IRIS MISCLASSIFIED PRED;I;ACTUAL
 Z←0
 I←0
LOOP:→(I≥dm∆DataFrame∆nrows IRIS)/0
 ACTUAL←IRIS dm∆DataFrame∆string_at 'Species' I
 →(ACTUAL≡I⊃PRED)/NEXT
 Z←Z+1
 PUT '  row ',(⍕I),': Petal.Length=',(⍕IRIS dm∆DataFrame∆numeric_at 'Petal.Length' I),' Petal.Width=',(⍕IRIS dm∆DataFrame∆numeric_at 'Petal.Width' I),'  actual=',ACTUAL,'  predicted=',I⊃PRED
NEXT:I←I+1
 →LOOP
∇
⎕IO←0
IRIS←dm∆data∆iris
N←dm∆DataFrame∆nrows IRIS
PUT 'iris: ',(⍕N),' rows x ',(⍕dm∆DataFrame∆ncols IRIS),' cols',NL

MODEL←dm∆XGBoostClassifier∆new IRIS 'Species ~ Petal.Length + Petal.Width'
dm∆XGBoostClassifier∆print_summary MODEL

PUT NL,'Confusion matrix (rows = actual, cols = predicted):'
CM←dm∆XGBoostClassifier∆confusion_matrix MODEL
PUT CM dm∆DataFrame∆to_string ⍬

PUT NL,'Misclassified rows:'
MISS←IRIS MISCLASSIFIED MODEL dm∆XGBoostClassifier∆predict IRIS
PUT (⍕MISS),' of ',(⍕N),' misclassified (',(⍕100×MISS÷N),'%)'

P1←dm∆XGBoostClassifier∆plot_training_deviance MODEL
P1 dm∆Plot∆save 'xgboost_iris_training_deviance.svg'
P2←MODEL dm∆XGBoostClassifier∆plot_decision_regions 'Petal.Length' 'Petal.Width'
P2 dm∆Plot∆save 'xgboost_iris_decision_regions.svg'
PUT NL,'Saved xgboost_iris_training_deviance.svg and xgboost_iris_decision_regions.svg'

⍝ XGBoost's defining lever is regularization: a heavily L2-regularized model (large lambda)
⍝ grows the same deep trees but keeps every leaf weight small, producing a much smoother
⍝ decision boundary than the lightly-regularized default.
⍝ dm∆XGBoostClassifier∆new data formula n_trees learning_rate max_depth lambda alpha gamma
⍝                         min_child_weight min_samples_leaf subsample colsample_bytree seed
OTHER←dm∆XGBoostClassifier∆new IRIS 'Species ~ Petal.Length + Petal.Width' 100 0.3 6 50
PUT NL,'lambda=1 (default):   training accuracy=',(⍕100×dm∆XGBoostClassifier∆training_accuracy MODEL),'%  deviance=',⍕↑⌽dm∆XGBoostClassifier∆training_deviance MODEL
PUT 'lambda=50 (heavy L2): training accuracy=',(⍕100×dm∆XGBoostClassifier∆training_accuracy OTHER),'%  deviance=',⍕↑⌽dm∆XGBoostClassifier∆training_deviance OTHER
P3←OTHER dm∆XGBoostClassifier∆plot_decision_regions 'Petal.Length' 'Petal.Width'
P3 dm∆Plot∆save 'xgboost_iris_decision_regions_heavy_lambda.svg'
PUT 'Saved xgboost_iris_decision_regions_heavy_lambda.svg'
dm∆Delete IRIS MODEL CM P1 P2 OTHER P3
