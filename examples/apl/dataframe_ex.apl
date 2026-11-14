⍝ CMD: just TARGET=dataframe_ex run-apl
⍝ DataFrame basics: nulls, drop_duplicates, fill_null, select/sort, group-by, join and summaries.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
SALES←dm∆DataFrame∆new ⍬
SALES dm∆DataFrame∆add_string_column 'region' ('west' 'west' 'east' 'south' 'south' 'south')
SALES dm∆DataFrame∆add_string_column 'product' ('widget' 'widget' 'widget' 'gizmo' 'gizmo' 'gizmo')
⍝ valid_mask: 0 marks a null.
SALES dm∆DataFrame∆add_numeric_column 'sales' (10 10 14 8 0 11) (1 1 1 1 0 1)
SALES dm∆DataFrame∆add_string_column 'quarter' ('Q1' 'Q1' 'Q1' 'Q2' 'Q2' '') (1 1 1 1 1 0)
PUT 'raw data',NL,(SALES dm∆DataFrame∆to_string ⍬),NL

⍝ One argument that is itself a vector (a list of names) is enclosed.
CLEANED←SALES dm∆DataFrame∆drop_duplicates ⊂'region' 'product' 'sales' 'quarter'
CLEANED dm∆DataFrame∆fill_null_string 'quarter' 'unknown'
CLEANED dm∆DataFrame∆fill_null_numeric 'sales' 0
PUT 'after drop_duplicates + fill_null',NL,(CLEANED dm∆DataFrame∆to_string ⍬),NL

PICKED←CLEANED dm∆DataFrame∆select ⊂'region' 'sales' 'quarter'
SELECTED←PICKED dm∆DataFrame∆sort_by 'sales' 0
PUT 'selected + sorted',NL,(SELECTED dm∆DataFrame∆to_string ⍬),NL

SUMS←CLEANED dm∆DataFrame∆group_by_sum (,⊂'region') (,⊂'sales')
GROUPED←SUMS dm∆DataFrame∆sort_by 'sales' 0
PUT 'group_by_sum(region)',NL,(GROUPED dm∆DataFrame∆to_string ⍬),NL

TARGETS←dm∆DataFrame∆new ⍬
TARGETS dm∆DataFrame∆add_string_column 'region' ('west' 'east' 'south')
TARGETS dm∆DataFrame∆add_numeric_column 'target' (18 12 25)
JOINED←GROUPED dm∆DataFrame∆join TARGETS 'region' 'region' 'left'
PUT 'joined with targets',NL,(JOINED dm∆DataFrame∆to_string ⍬),NL

SHAPE←dm∆DataFrame∆shape CLEANED
PUT 'shape = (',(⍕SHAPE[1]),', ',(⍕SHAPE[2]),')'
PUT 'sales count = ',⍕CLEANED dm∆DataFrame∆numeric_count 'sales'
PUT 'sales nulls = ',⍕CLEANED dm∆DataFrame∆numeric_null_count 'sales'
PUT 'sales sum = ',⍕CLEANED dm∆DataFrame∆numeric_sum 'sales'
PUT 'sales mean = ',⍕CLEANED dm∆DataFrame∆numeric_mean 'sales'
dm∆Delete SALES CLEANED PICKED SELECTED SUMS GROUPED TARGETS JOINED
