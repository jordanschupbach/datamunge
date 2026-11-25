⍝ CMD: just TARGET=tree_iris_ex run-apl
⍝ Decision tree classifier on iris: summary, confusion matrix, misclassified rows and decision regions.
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

MODEL←dm∆DecisionTreeClassifier∆new IRIS 'Species ~ Petal.Length + Petal.Width'
dm∆DecisionTreeClassifier∆print_summary MODEL

PUT NL,'Confusion matrix (rows = actual, cols = predicted):'
CM←dm∆DecisionTreeClassifier∆confusion_matrix MODEL
PUT CM dm∆DataFrame∆to_string ⍬

PUT NL,'Misclassified rows:'
MISS←IRIS MISCLASSIFIED MODEL dm∆DecisionTreeClassifier∆predict IRIS
PUT (⍕MISS),' of ',(⍕N),' misclassified (',(⍕100×MISS÷N),'%)'

P1←MODEL dm∆DecisionTreeClassifier∆plot_classification IRIS 'Petal.Length' 'Petal.Width'
P1 dm∆Plot∆save 'tree_iris_classification.svg'
P2←MODEL dm∆DecisionTreeClassifier∆plot_decision_regions 'Petal.Length' 'Petal.Width'
P2 dm∆Plot∆save 'tree_iris_decision_regions.svg'
PUT NL,'Saved tree_iris_classification.svg and tree_iris_decision_regions.svg'

⍝ A shallower tree, for comparison, showing a coarser (but still fairly accurate) decision boundary.
SHALLOW←dm∆DecisionTreeClassifier∆new IRIS 'Species ~ Petal.Length + Petal.Width' 2
PUT NL,'Depth-2 tree training accuracy: ',(⍕100×dm∆DecisionTreeClassifier∆training_accuracy SHALLOW),'% (',(⍕dm∆DecisionTreeClassifier∆leaf_count SHALLOW),' leaves)'
P3←SHALLOW dm∆DecisionTreeClassifier∆plot_decision_regions 'Petal.Length' 'Petal.Width'
P3 dm∆Plot∆save 'tree_iris_decision_regions_depth2.svg'
PUT 'Saved tree_iris_decision_regions_depth2.svg'
dm∆Delete IRIS MODEL CM P1 P2 SHALLOW P3
