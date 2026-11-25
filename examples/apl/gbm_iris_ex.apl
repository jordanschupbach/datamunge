⍝ CMD: just TARGET=gbm_iris_ex run-apl
⍝ Gradient boosting classifier on iris: summary, confusion matrix, training deviance and few vs many rounds.
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

MODEL←dm∆GBMClassifier∆new IRIS 'Species ~ Petal.Length + Petal.Width'
dm∆GBMClassifier∆print_summary MODEL

PUT NL,'Confusion matrix (rows = actual, cols = predicted):'
CM←dm∆GBMClassifier∆confusion_matrix MODEL
PUT CM dm∆DataFrame∆to_string ⍬

PUT NL,'Misclassified rows:'
MISS←IRIS MISCLASSIFIED MODEL dm∆GBMClassifier∆predict IRIS
PUT (⍕MISS),' of ',(⍕N),' misclassified (',(⍕100×MISS÷N),'%)'

P1←dm∆GBMClassifier∆plot_training_deviance MODEL
P1 dm∆Plot∆save 'gbm_iris_training_deviance.svg'
P2←MODEL dm∆GBMClassifier∆plot_decision_regions 'Petal.Length' 'Petal.Width'
P2 dm∆Plot∆save 'gbm_iris_decision_regions.svg'
PUT NL,'Saved gbm_iris_training_deviance.svg and gbm_iris_decision_regions.svg'

⍝ A handful of boosting rounds vs a well-boosted ensemble: each additional round chips away
⍝ at the training loss, gradually sharpening the decision boundary.
⍝ dm∆GBMClassifier∆new data formula n_trees learning_rate max_depth min_samples_split min_samples_leaf subsample seed
OTHER←dm∆GBMClassifier∆new IRIS 'Species ~ Petal.Length + Petal.Width' 5
PUT NL,'5-round ensemble:   training accuracy=',(⍕100×dm∆GBMClassifier∆training_accuracy OTHER),'%  deviance=',⍕↑⌽dm∆GBMClassifier∆training_deviance OTHER
PUT '100-round ensemble: training accuracy=',(⍕100×dm∆GBMClassifier∆training_accuracy MODEL),'%  deviance=',⍕↑⌽dm∆GBMClassifier∆training_deviance MODEL
P3←OTHER dm∆GBMClassifier∆plot_decision_regions 'Petal.Length' 'Petal.Width'
P3 dm∆Plot∆save 'gbm_iris_decision_regions_5rounds.svg'
PUT 'Saved gbm_iris_decision_regions_5rounds.svg'
dm∆Delete IRIS MODEL CM P1 P2 OTHER P3
