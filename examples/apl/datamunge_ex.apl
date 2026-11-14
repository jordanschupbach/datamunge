⍝ CMD: just TARGET=datamunge_ex run-apl
⍝ Smoke test of the basic bindings: vectors, pairs and a native Callback object.
TAB←⎕UCS 9
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
dm∆hello
V←dm∆make_dvector 1 2 3
PUT 'sum_dvector:',TAB,⍕dm∆sum_dvector V
P←dm∆make_dpair 1.25 2.75
PUT 'sum_dpair:',TAB,⍕dm∆sum_dpair P
CB←dm∆Callback∆new ⍬
PUT 'call_with_callback(3.0):',TAB,⍕dm∆call_with_callback 3 CB
V2←dm∆map_dvector_with_callback (dm∆make_dvector 1 2 3) CB
PUT 'sum_dvector(map_dvector_with_callback(1,2,3)):',TAB,⍕dm∆sum_dvector V2
dm∆Delete CB
