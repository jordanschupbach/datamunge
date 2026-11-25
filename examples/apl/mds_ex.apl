⍝ CMD: just TARGET=mds_ex run-apl
⍝ Classical multidimensional scaling of iris (euclidean and manhattan distances).
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ D FMT X: the number X with D decimals, C style ('-' for '¯', '0.5' for '.5').
∇Z←D FMT X
 Z←D⍕X
 Z←(∨\Z≠' ')/Z
 Z←((Z[1]='.')/'0'),Z
 Z←(1↑Z),(('¯.'≡2↑Z)/'0'),1↓Z
 Z[(Z='¯')/⍳⍴Z]←'-'
∇
IRIS←dm∆data∆iris
FEATURES←'Sepal.Length' 'Sepal.Width' 'Petal.Length' 'Petal.Width'
SPECIES←IRIS dm∆DataFrame∆pull_string 'Species'

PUT '=================== Classical MDS on iris (euclidean) ==================='
MDS←dm∆MDS∆new IRIS FEATURES 2 'euclidean'
dm∆MDS∆print_summary MDS

PUT NL,'Embedding as a DataFrame:'
EMBEDDING←dm∆MDS∆embedding_frame MDS
PUT EMBEDDING dm∆DataFrame∆to_string ⍬

SCATTER←MDS dm∆MDS∆plot_embedding_grouped ⊂SPECIES
SCATTER dm∆Plot∆save_svg 'mds_iris_embedding.svg'
PUT NL,'Embedding scatter (colored by species) saved as mds_iris_embedding.svg'

PUT NL,'=================== Classical MDS on iris (manhattan) ==================='
MANHATTAN←dm∆MDS∆new IRIS FEATURES 2 'manhattan'
PUT 'Goodness of fit: euclidean=',(4 FMT dm∆MDS∆goodness_of_fit MDS),', manhattan=',4 FMT dm∆MDS∆goodness_of_fit MANHATTAN
dm∆Delete IRIS MDS EMBEDDING SCATTER MANHATTAN
