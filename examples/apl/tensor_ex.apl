⍝ CMD: just TARGET=tensor_ex run-apl
⍝ Tensors: construction, shape ops, broadcasting, reductions, linear algebra, masks, iris.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ G X: the number X, '-' for '¯'.
∇Z←G X
 Z←⍕X
 Z[(Z='¯')/⍳⍴Z]←'-'
∇
⍝ The shape vector X as [a, b].
∇Z←SHAPE X
 Z←'[',(⊃{⍺,', ',⍵}/⍕¨X),']'
∇
⍝ The tensor T as a string; deletes it (every Tensor operation returns a new handle).
∇Z←TAKE T
 Z←T dm∆Tensor∆to_string ⍬
 dm∆Delete T
∇

PUT '=================== Construction ==================='
PUT 'zeros([2,3]): ',TAKE dm∆Tensor∆zeros 2 3
PUT 'eye(3): ',TAKE dm∆Tensor∆eye 3
ARANGE←dm∆Tensor∆arange 0 12 1
R←ARANGE dm∆Tensor∆reshape 3 4
PUT 'arange(0,12).reshape([3,4]): ',R dm∆Tensor∆to_string ⍬

PUT NL,'=================== Shape ops ==================='
RT←R dm∆Tensor∆transpose ⍬
PUT 'transpose -> shape ',SHAPE dm∆Tensor∆shape RT
PUT 'slice(axis=1, start=1, stop=3): ',TAKE R dm∆Tensor∆slice 1 1 3

PUT NL,'=================== Broadcasting arithmetic ==================='
COL←dm∆Tensor∆from_values (3 1) (1 2 3)
ROW←dm∆Tensor∆from_values (1 4) (10 20 30 40)
PUT '(3,1) + (1,4) -> ',TAKE COL dm∆Tensor∆add ROW

PUT NL,'=================== Reductions ==================='
PUT 'r.sum() = ',(G dm∆Tensor∆sum R),', r.mean() = ',G dm∆Tensor∆mean R
PUT 'column means (axis=0): ',TAKE R dm∆Tensor∆mean_axis 0

PUT NL,'=================== Linear algebra ==================='
A←dm∆Tensor∆from_values (2 3) (1 2 3 4 5 6)
B←dm∆Tensor∆from_values (3 2) (7 8 9 10 11 12)
PUT 'matmul(2x3, 3x2) -> ',TAKE A dm∆Tensor∆matmul B
V1←dm∆Tensor∆from_values (,3) (1 2 3)
V2←dm∆Tensor∆from_values (,3) (4 5 6)
PUT 'dot([1,2,3], [4,5,6]) = ',G V1 dm∆Tensor∆dot V2
PUT 'outer(v1, v2) -> ',TAKE V1 dm∆Tensor∆outer V2

PUT NL,'=================== Comparisons & masks ==================='
FULL←dm∆Tensor∆full (3 4) 6
MASK←R dm∆Tensor∆greater_equal FULL
PUT 'r >= 6 -> ',MASK dm∆Tensor∆to_string ⍬
PUT 'count(r >= 6) = ',G dm∆Tensor∆sum MASK
dm∆Delete ARANGE R RT COL ROW A B V1 V2 FULL MASK

PUT NL,'=================== A real dataset as a Tensor ==================='
IRIS←dm∆data∆iris
N←dm∆DataFrame∆nrows IRIS
⍝ Row-major N×4 values: the four feature columns side by side.
COLUMNS←'Sepal.Length' 'Sepal.Width' 'Petal.Length' 'Petal.Width'
FLAT←,⍉⊃(⊂IRIS) dm∆DataFrame∆pull_numeric¨COLUMNS
X←dm∆Tensor∆from_values (N 4) FLAT
PUT 'iris feature tensor shape: ',SHAPE dm∆Tensor∆shape X

FEATUREMEANS←X dm∆Tensor∆mean_axis 0
MEANS1X4←FEATUREMEANS dm∆Tensor∆reshape 1 4
CENTERED←X dm∆Tensor∆subtract MEANS1X4
CENTEREDT←CENTERED dm∆Tensor∆transpose ⍬
SCATTER←CENTEREDT dm∆Tensor∆matmul CENTERED
PUT 'feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): ',FEATUREMEANS dm∆Tensor∆to_string ⍬
PUT '(X-mean)^T (X-mean) [4x4 scatter matrix]: ',SCATTER dm∆Tensor∆to_string 16

SPECIES←IRIS dm∆DataFrame∆pull_string 'Species'
SPECIESTENSOR←dm∆Tensor∆from_string_values (,N) SPECIES
SETOSA←dm∆Tensor∆from_string_values (,1) (,⊂'setosa')
SETOSAMASK←SPECIESTENSOR dm∆Tensor∆equal SETOSA
PUT 'setosa count = ',(G dm∆Tensor∆sum SETOSAMASK),' (of ',(⍕N),' rows)'
dm∆Delete IRIS X FEATUREMEANS MEANS1X4 CENTERED CENTEREDT SCATTER SPECIESTENSOR SETOSA SETOSAMASK
