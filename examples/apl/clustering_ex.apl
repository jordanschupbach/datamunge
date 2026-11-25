⍝ CMD: just TARGET=clustering_ex run-apl
⍝ Clustering iris: K-means, agglomerative (ward/complete/average, re-cut) and DBSCAN, with per-cluster purity.
NL←⎕UCS 10
⍝ PUT writes a line (⍞ output: GNU APL wraps ⎕← output at ⎕PW columns).
∇PUT S
 ⍞←S,⎕UCS 10
∇
⍝ REPORT labels species n_clusters: size and majority species of each cluster (negative labels are noise).
∇REPORT B;LABELS;SPECIES;K;C;M;U;COUNTS;BEST;TOTAL
 (LABELS SPECIES K)←B
 C←¯1
LOOP:→(K≤C←C+1)/0
 M←LABELS=C
 →(0=TOTAL←+/M)/LOOP
 U←∪M/SPECIES
 COUNTS←+/U∘.≡M/SPECIES
 BEST←COUNTS⍳⌈/COUNTS
 PUT '  cluster ',(⍕C),': ',(⍕TOTAL),' points, majority ',(⊃U[BEST]),' (',(⍕COUNTS[BEST]),'/',(⍕TOTAL),')'
 →LOOP
∇
IRIS←dm∆data∆iris
SPECIES←{IRIS dm∆DataFrame∆string_at 'Species' ⍵}¨¯1+⍳dm∆DataFrame∆nrows IRIS
FEATURES←'Sepal.Length' 'Sepal.Width' 'Petal.Length' 'Petal.Width'

PUT '=================== K-means (k=3) on iris ==================='
KMEANS←dm∆KMeans∆new IRIS FEATURES 3
PUT 'Inertia: ',(⍕dm∆KMeans∆inertia KMEANS),', iterations (best run): ',⍕dm∆KMeans∆iterations_used KMEANS
REPORT (dm∆KMeans∆labels KMEANS) SPECIES 3

PUT NL,'=================== Agglomerative clustering on iris ==================='
∇AGGLOMERATIVE LINKAGE;MODEL
 MODEL←dm∆AgglomerativeClustering∆new IRIS FEATURES 3 LINKAGE
 PUT '--- linkage=',LINKAGE,' ---'
 REPORT (dm∆AgglomerativeClustering∆labels MODEL) SPECIES 3
 dm∆Delete MODEL
∇
AGGLOMERATIVE 'ward'
AGGLOMERATIVE 'complete'
AGGLOMERATIVE 'average'

PUT NL,'--- Re-cutting the ward dendrogram at k=2 without refitting ---'
WARD←dm∆AgglomerativeClustering∆new IRIS FEATURES 3 'ward'
REPORT (WARD dm∆AgglomerativeClustering∆cut 2) SPECIES 2

PUT NL,'=================== DBSCAN on iris ==================='
DBSCAN←dm∆DBSCAN∆new IRIS FEATURES 0.6 5
PUT 'Clusters found: ',(⍕dm∆DBSCAN∆n_clusters DBSCAN),', noise points: ',(⍕dm∆DBSCAN∆n_noise DBSCAN),' (of ',(⍕dm∆DBSCAN∆observations DBSCAN),')'
REPORT (dm∆DBSCAN∆labels DBSCAN) SPECIES (dm∆DBSCAN∆n_clusters DBSCAN)
dm∆Delete IRIS KMEANS WARD DBSCAN
