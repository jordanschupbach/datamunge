⍝ CMD: just TARGET=knn_iris_ex run-apl
⍝ k-nearest-neighbours classifier on iris: leave-one-out accuracy and decision regions for k = 1, 5, 25.
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

MODEL←dm∆KNNClassifier∆new IRIS 'Species ~ Petal.Length + Petal.Width'
dm∆KNNClassifier∆print_summary MODEL

PUT NL,'Confusion matrix (leave-one-out; rows = actual, cols = predicted):'
CM←dm∆KNNClassifier∆confusion_matrix MODEL
PUT CM dm∆DataFrame∆to_string ⍬

PUT NL,'Leave-one-out misclassified rows:'
MISS←IRIS MISCLASSIFIED MODEL dm∆KNNClassifier∆predict IRIS
PUT (⍕MISS),' of ',(⍕N),' misclassified (',(⍕100×MISS÷N),'%)'

P5←MODEL dm∆KNNClassifier∆plot_decision_regions 'Petal.Length' 'Petal.Width'
P5 dm∆Plot∆save 'knn_iris_decision_regions_k5.svg'
PUT NL,'Saved knn_iris_decision_regions_k5.svg'

⍝ k=1 memorizes every training point exactly (jagged, overfit boundary with an island around
⍝ every point, including noise); k=25 averages over a much larger neighborhood (very smooth,
⍝ underfit boundary). dm∆KNNClassifier∆new data formula k metric weighted standardize
K1←dm∆KNNClassifier∆new IRIS 'Species ~ Petal.Length + Petal.Width' 1
PUT NL,'k=1  leave-one-out accuracy: ',(⍕100×dm∆KNNClassifier∆training_accuracy K1),'%'
P1←K1 dm∆KNNClassifier∆plot_decision_regions 'Petal.Length' 'Petal.Width'
P1 dm∆Plot∆save 'knn_iris_decision_regions_k1.svg'
PUT 'Saved knn_iris_decision_regions_k1.svg'

K25←dm∆KNNClassifier∆new IRIS 'Species ~ Petal.Length + Petal.Width' 25
PUT NL,'k=25 leave-one-out accuracy: ',(⍕100×dm∆KNNClassifier∆training_accuracy K25),'%'
P25←K25 dm∆KNNClassifier∆plot_decision_regions 'Petal.Length' 'Petal.Width'
P25 dm∆Plot∆save 'knn_iris_decision_regions_k25.svg'
PUT 'Saved knn_iris_decision_regions_k25.svg'
dm∆Delete IRIS MODEL CM P5 K1 P1 K25 P25
