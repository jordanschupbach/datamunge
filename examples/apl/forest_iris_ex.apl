⍝ CMD: just TARGET=forest_iris_ex run-apl
⍝ Random forest classifier on iris: summary, confusion matrix, misclassified rows and out-of-bag accuracy.
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

MODEL←dm∆RandomForestClassifier∆new IRIS 'Species ~ Petal.Length + Petal.Width'
dm∆RandomForestClassifier∆print_summary MODEL

PUT NL,'Confusion matrix (rows = actual, cols = predicted):'
CM←dm∆RandomForestClassifier∆confusion_matrix MODEL
PUT CM dm∆DataFrame∆to_string ⍬

PUT NL,'Misclassified rows:'
MISS←IRIS MISCLASSIFIED MODEL dm∆RandomForestClassifier∆predict IRIS
PUT (⍕MISS),' of ',(⍕N),' misclassified (',(⍕100×MISS÷N),'%)'

P1←MODEL dm∆RandomForestClassifier∆plot_classification IRIS 'Petal.Length' 'Petal.Width'
P1 dm∆Plot∆save 'forest_iris_classification.svg'
P2←MODEL dm∆RandomForestClassifier∆plot_decision_regions 'Petal.Length' 'Petal.Width'
P2 dm∆Plot∆save 'forest_iris_decision_regions.svg'
PUT NL,'Saved forest_iris_classification.svg and forest_iris_decision_regions.svg'

⍝ A small forest, for comparison, showing how out-of-bag accuracy stabilizes as more trees
⍝ are added -- the defining random forest effect a single decision tree cannot demonstrate.
⍝ dm∆RandomForestClassifier∆new data formula n_trees max_depth min_samples_split min_samples_leaf
⍝                              max_features criterion bootstrap sample_fraction seed
SMALL←dm∆RandomForestClassifier∆new IRIS 'Species ~ Petal.Length + Petal.Width' 5
PUT NL,'5-tree forest:   training accuracy=',(⍕100×dm∆RandomForestClassifier∆training_accuracy SMALL),'%  OOB accuracy=',(⍕100×dm∆RandomForestClassifier∆oob_accuracy SMALL),'%'
PUT '100-tree forest: training accuracy=',(⍕100×dm∆RandomForestClassifier∆training_accuracy MODEL),'%  OOB accuracy=',(⍕100×dm∆RandomForestClassifier∆oob_accuracy MODEL),'%'
P3←SMALL dm∆RandomForestClassifier∆plot_decision_regions 'Petal.Length' 'Petal.Width'
P3 dm∆Plot∆save 'forest_iris_decision_regions_5trees.svg'
PUT 'Saved forest_iris_decision_regions_5trees.svg'
dm∆Delete IRIS MODEL CM P1 P2 SMALL P3
