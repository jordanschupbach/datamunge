package main

import (
	"datamunge"
	"fmt"
)

func svector(values []string) datamunge.SVector {
	out := datamunge.NewSVector(int64(len(values)))
	for i, v := range values {
		out.Set(i, v)
	}
	return out
}

type labelKey struct {
	label int64
	sp    string
}

func reportPurity(labels []int64, species []string, nClusters int64) {
	votes := make(map[labelKey]int)
	for i, label := range labels {
		if label >= 0 {
			votes[labelKey{label, species[i]}]++
		}
	}
	for c := int64(0); c < nClusters; c++ {
		total := 0
		best := 0
		bestSp := ""
		for key, cnt := range votes {
			if key.label == c {
				total += cnt
				if cnt > best {
					best = cnt
					bestSp = key.sp
				}
			}
		}
		if total > 0 {
			fmt.Printf("  cluster %d: %d points, majority %s (%d/%d)\n", c, total, bestSp, best, total)
		}
	}
}

func szVecToInt64(v datamunge.SizeVector) []int64 {
	out := make([]int64, v.Size())
	for i := 0; i < int(v.Size()); i++ {
		out[i] = int64(v.Get(i))
	}
	return out
}

func ivVecToInt64(v datamunge.IVector) []int64 {
	out := make([]int64, v.Size())
	for i := 0; i < int(v.Size()); i++ {
		out[i] = int64(v.Get(i))
	}
	return out
}

func main() {
	iris := datamunge.DataFrameIris()
	n := iris.Nrows()
	species := make([]string, 0, n)
	for i := int64(0); i < n; i++ {
		species = append(species, iris.String_at("Species", i))
	}
	features := svector([]string{"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"})

	fmt.Println("=================== K-means (k=3) on iris ===================")
	kmeans := datamunge.NewKMeans(iris, features, int64(3))
	fmt.Printf("Inertia: %g, iterations (best run): %d\n", kmeans.Inertia(), kmeans.Iterations_used())
	reportPurity(szVecToInt64(kmeans.Labels()), species, 3)

	fmt.Println("\n=================== Agglomerative clustering on iris ===================")
	for _, linkage := range []string{"ward", "complete", "average"} {
		model := datamunge.NewAgglomerativeClustering(iris, features, int64(3), linkage)
		fmt.Printf("--- linkage=%s ---\n", linkage)
		reportPurity(szVecToInt64(model.Labels()), species, 3)
	}

	fmt.Println("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---")
	wardModel := datamunge.NewAgglomerativeClustering(iris, features, int64(3), "ward")
	reportPurity(szVecToInt64(wardModel.Cut(int64(2))), species, 2)

	fmt.Println("\n=================== DBSCAN on iris ===================")
	dbscan := datamunge.NewDBSCAN(iris, features, 0.6, int64(5))
	fmt.Printf("Clusters found: %d, noise points: %d (of %d)\n", dbscan.N_clusters(), dbscan.N_noise(), dbscan.Observations())
	reportPurity(ivVecToInt64(dbscan.Labels()), species, dbscan.N_clusters())
}
