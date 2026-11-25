⍝ CMD: just TARGET=bspline_ex run-apl
⍝ Fits a two-dimensional B-spline surface with the bs(x, y) formula term.
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ G X: up to 10 significant digits ('-' for '¯').
∇Z←G X
 Z←⍕X
 Z[(Z='¯')/⍳⍴Z]←'-'
∇
⍝ An 8×8 grid on [0,1]², x varying fastest: z = sin(x) + y².
X←,8 8⍴(¯1+⍳8)÷7
Y←,⍉8 8⍴(¯1+⍳8)÷7
Z←(1○X)+Y*2
DATA←dm∆DataFrame∆new ⍬
DATA dm∆DataFrame∆add_numeric_column 'x' X
DATA dm∆DataFrame∆add_numeric_column 'y' Y
DATA dm∆DataFrame∆add_numeric_column 'z' Z

SURFACE←dm∆LM∆new DATA 'z ~ bs(x, y)'
PUT 'Fitted z ~ bs(x, y)'
dm∆LM∆print_summary SURFACE

NEWPOINTS←dm∆DataFrame∆new ⍬
NEWPOINTS dm∆DataFrame∆add_numeric_column 'x' (0.25 0.75)
NEWPOINTS dm∆DataFrame∆add_numeric_column 'y' (0.5 0.25)
PRED←SURFACE dm∆LM∆predict NEWPOINTS
PUT 'Predictions: [',(G PRED[1]),', ',(G PRED[2]),']'
dm∆Delete DATA SURFACE NEWPOINTS
