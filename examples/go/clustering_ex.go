// CMD: just TARGET=clustering_ex run-go

//go:build ignore

// Clustering iris: k-means, agglomerative (several linkages, dendrogram re-cut) and DBSCAN.
package main

import (
	"datamunge"
	"datamunge/data"
	"fmt"
)

var species []string

// For each cluster, print its size and its majority species (negative labels are noise).
func reportPurity(clusterLabels []int, nClusters int) {
	votes := map[int]map[string]int{}
	order := map[int][]string{} // first-seen order of species per cluster, for deterministic ties
	for i, label := range clusterLabels {
		if label < 0 {
			continue
		}
		if votes[label] == nil {
			votes[label] = map[string]int{}
		}
		if votes[label][species[i]] == 0 {
			order[label] = append(order[label], species[i])
		}
		votes[label][species[i]]++
	}
	for c := 0; c < nClusters; c++ {
		counts := votes[c]
		if counts == nil {
			continue
		}
		total, best, bestSpecies := 0, -1, ""
		for _, s := range order[c] {
			total += counts[s]
			if counts[s] > best {
				best, bestSpecies = counts[s], s
			}
		}
		fmt.Printf("  cluster %d: %d points, majority %s (%d/%d)\n", c, total, bestSpecies, best, total)
	}
}

func main() {
	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	for i := 0; i < datamunge.Must(iris.Nrows()); i++ {
		species = append(species, datamunge.Must(iris.StringAt("Species", i)))
	}
	featureColumns := []string{"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"}

	fmt.Println("=================== K-means (k=3) on iris ===================")
	kmeans := datamunge.Must(datamunge.NewKMeansWithNClusters(iris, featureColumns, 3))
	defer kmeans.Close()
	fmt.Printf("Inertia: %v, iterations (best run): %d\n",
		datamunge.Must(kmeans.Inertia()), datamunge.Must(kmeans.IterationsUsed()))
	reportPurity(datamunge.Must(kmeans.Labels()), 3)

	fmt.Println("\n=================== Agglomerative clustering on iris ===================")
	for _, linkage := range []string{"ward", "complete", "average"} {
		model := datamunge.Must(datamunge.NewAgglomerativeClusteringWithLinkage(iris, featureColumns, 3, linkage))
		fmt.Printf("--- linkage=%s ---\n", linkage)
		reportPurity(datamunge.Must(model.Labels()), 3)
		model.Close()
	}

	fmt.Println("\n--- Re-cutting the ward dendrogram at k=2 without refitting ---")
	wardModel := datamunge.Must(datamunge.NewAgglomerativeClusteringWithLinkage(iris, featureColumns, 3, "ward"))
	defer wardModel.Close()
	reportPurity(datamunge.Must(wardModel.Cut(2)), 2)

	fmt.Println("\n=================== DBSCAN on iris ===================")
	dbscan := datamunge.Must(datamunge.NewDBSCANWithMinSamples(iris, featureColumns, 0.6, 5))
	defer dbscan.Close()
	fmt.Printf("Clusters found: %d, noise points: %d (of %d)\n",
		datamunge.Must(dbscan.NClusters()), datamunge.Must(dbscan.NNoise()), datamunge.Must(dbscan.Observations()))
	reportPurity(datamunge.Must(dbscan.Labels()), datamunge.Must(dbscan.NClusters()))
}
