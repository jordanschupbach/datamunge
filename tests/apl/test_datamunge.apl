#!/usr/bin/env apl --script
⍝ Smoke tests of the datamunge GNU APL binding (just test-apl, nix checks.apl):
⍝   apl --script --OFF -f datamunge.apl -f tests/apl/test_datamunge.apl
⍝ with DATAMUNGE_APL_LIBRARY naming libdatamunge_gnu_apl.so. The documentation checks read the
⍝ committed script src/datamungeapl/apl/datamunge.apl (DATAMUNGE_APL_DOCS). Prints
⍝ "DATAMUNGE APL PASS n checks" when every check passes, "FAIL ..." lines otherwise.

CHECKS←0
FAILURES←0
DONE←0   ⍝ set when TEST ran to its end (an APL error stops it and carries on below)

∇CHECK B;LABEL;GOT;WANT
 (LABEL GOT WANT)←B
 CHECKS←CHECKS+1
 →(GOT≡WANT)/0
 FAILURES←FAILURES+1
 'FAIL ',LABEL,': got ',(⍕GOT),', want ',⍕WANT
∇

⍝ Executes EXPR, which must signal an APL error whose dm∆last_error kind is KIND and whose message
⍝ contains FRAGMENT.
∇FAILS B;LABEL;EXPR;KIND;FRAGMENT;R;E
 (LABEL EXPR KIND FRAGMENT)←B
 CHECKS←CHECKS+1
 R←⎕EC EXPR
 →(0=↑R)/CAUGHT
 FAILURES←FAILURES+1
 'FAIL ',LABEL,': no error'
 →0
CAUGHT:E←dm∆last_error
 →((KIND≡↑E)∧(∨/FRAGMENT⍷2⊃E)∧∨/'dm: '⍷2⊃E)/0
 FAILURES←FAILURES+1
 'FAIL ',LABEL,': got ',(↑E),' "',(2⊃E),'"'
∇

⍝ The value of environment variable B ('' when unset).
∇Z←GETENV B;V
 V←⎕ENV B
 V←(V[;1]∊⊂B)⌿V
 Z←''
 →(0=↑⍴V)/0
 Z←⊃V[1;2]
∇

⍝ An APL implementation of the C++ Callback's virtual call() (a director), and one that fails.
∇Z←H Twice∆call X
 Z←2×X
∇
∇Z←H Broken∆call X
 Z←X÷0
∇

∇TEST;BEFORE;V;P;CB;D;F;FRAME;IRIS;DOC;SRC;PATH
 BEFORE←dm∆live
 ⍝ Smoke test: call into the native library (it prints a greeting).
 CHECK 'hello' (0≠↑⎕EC 'dm∆hello') 1
 ⍝ std::vector<double> and std::pair<double, double> round trips.
 V←dm∆make_dvector 1 2 3
 CHECK 'std::vector<double> is a numeric vector' V (1 2 3)
 CHECK 'sum_dvector' (dm∆sum_dvector V) 6
 CHECK 'vectors convert to std::vector' (dm∆sum_dvector 0.5 1.5) 2
 P←dm∆make_dpair 1.25 2.75
 CHECK 'std::pair is a 2-vector' P (1.25 2.75)
 CHECK 'sum_dpair' (dm∆sum_dpair P) 4
 ⍝ The C++ Callback class, and an APL implementation of its virtual call().
 CB←dm∆Callback∆new ⍬
 CHECK 'native Callback' (dm∆call_with_callback 3 CB) 3
 D←'Twice' dm∆Callback∆subclass ⍬
 CHECK 'the APL override runs' (dm∆call_with_callback 3 D) 6
 CHECK 'override from C++ loops' (dm∆sum_dvector dm∆map_dvector_with_callback (1 2 3) D) 12
 ⍝ An APL error inside an override reaches the caller as a callback error.
 F←'Broken' dm∆Callback∆subclass ⍬
 FAILS 'the APL error propagates' 'dm∆call_with_callback 1 F' 'callback' 'DOMAIN ERROR'
 ⍝ C++ exceptions signal APL errors with the message dm: kind: what.
 FAILS 'std::invalid_argument' 'dm∆runif 1 0 3' 'invalid_argument' 'upper must be greater'
 ⍝ Classes and methods; dm∆Delete releases the C++ object.
 FRAME←dm∆DataFrame∆new ⍬
 FRAME dm∆DataFrame∆add_numeric_column 'x' (1 2 3)
 CHECK 'DataFrame' (dm∆DataFrame∆nrows FRAME) 3
 CHECK 'std::vector<std::string> is a nested vector' (dm∆DataFrame∆columns FRAME) (,⊂,'x')
 dm∆Delete FRAME
 FAILS 'a deleted object refuses calls' 'dm∆DataFrame∆nrows FRAME' 'binding' 'closed'
 ⍝ The policy module datamunge::data is dm∆data∆.
 IRIS←dm∆data∆iris
 CHECK 'dm∆data∆iris' (dm∆DataFrame∆nrows IRIS) 150
 dm∆Delete IRIS CB D F
 CHECK 'every handle released' dm∆live BEFORE
 ⍝ The functions carry the C++ documentation as ⍝ comments (⎕CR, mep's APL server).
 DOC←,⎕CR 'dm∆seql'
 CHECK 'function documentation' (∨/'evenly spaced'⍷DOC) 1
 ⍝ The committed script holds the same documentation (usage forms with the C++ defaults).
 PATH←GETENV 'DATAMUNGE_APL_DOCS'
 →(0=⍴PATH)/SKIP
 SRC←19 ⎕CR (⎕FIO[26] PATH)
 CHECK 'committed method documentation' (∨/'Z←obj dm∆DataFrame∆sort_by column_name ascending'⍷SRC) 1
SKIP:DONE←1
∇

TEST
⍞←(DONE∧FAILURES=0)/'DATAMUNGE APL PASS ',(⍕CHECKS),' checks',⎕UCS 10
⍞←(FAILURES≠0)/'DATAMUNGE APL FAIL ',(⍕FAILURES),' of ',(⍕CHECKS),' checks failed',⎕UCS 10
⍞←(~DONE)/'DATAMUNGE APL FAIL: the tests stopped at an error',⎕UCS 10
