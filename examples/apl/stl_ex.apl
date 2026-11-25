⍝ CMD: just TARGET=stl_ex run-apl
⍝ C++ std::vector and std::pair values in APL: both are plain numeric vectors.
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ G X: the number X, '-' for '¯'.
∇Z←G X
 Z←⍕X
 Z[(Z='¯')/⍳⍴Z]←'-'
∇
N←100
⍝ A std::vector<double> is a numeric APL vector.
V←1.5×¯1+⍳N
PUT¨G¨V
⍝ Integers: a std::vector<int> is an integer APL vector too.
V2←⌊1.5×¯1+⍳N
PUT¨⍕¨V2
P←3 4
PUT 'p: (',(⍕P[1]),', ',(⍕P[2]),')'
P2←10 20
PUT 'p2: (',(⍕P2[1]),', ',(⍕P2[2]),')'
⍝ A C++ std::pair<double,double> comes back as a 2-element vector.
P3←dm∆make_dpair 1.25 2.75
PUT 'make_dvector(1, 2, 3) = [',(⊃{⍺,', ',⍵}/G¨dm∆make_dvector 1 2 3),']'
PUT 'sum_dvector of the 100 doubles above = ',G dm∆sum_dvector V
PUT 'make_dpair(1.25, 2.75): (',(G P3[1]),', ',(G P3[2]),'), sum_dpair = ',G dm∆sum_dpair P3
