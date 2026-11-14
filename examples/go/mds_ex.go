// CMD: just TARGET=mds_ex run-go

//go:build ignore

// Classical multidimensional scaling of iris: euclidean embedding, plot, and a manhattan comparison.
package main

import (
	"datamunge"
	"datamunge/data"
	"fmt"
)

func check(err error) {
	if err != nil {
		panic(err)
	}
}

func main() {
	iris := datamunge.Must(data.Iris())
	defer iris.Close()
	featureColumns := []string{"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"}
	species := make([]string, datamunge.Must(iris.Nrows()))
	for i := range species {
		species[i] = datamunge.Must(iris.StringAt("Species", i))
	}

	fmt.Println("=================== Classical MDS on iris (euclidean) ===================")
	mds := datamunge.Must(datamunge.NewMDSWithMetric(iris, featureColumns, 2, "euclidean"))
	defer mds.Close()
	check(mds.PrintSummary())

	fmt.Println("\nEmbedding as a DataFrame:\n" + datamunge.Must(mds.EmbeddingFrame()).String())

	check(datamunge.Must(mds.PlotEmbeddingGrouped(species)).SaveSvg("mds_iris_embedding.svg"))
	fmt.Println("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg")

	fmt.Println("\n=================== Classical MDS on iris (manhattan) ===================")
	manhattanMds := datamunge.Must(datamunge.NewMDSWithMetric(iris, featureColumns, 2, "manhattan"))
	defer manhattanMds.Close()
	fmt.Printf("Goodness of fit: euclidean=%.4f, manhattan=%.4f\n",
		datamunge.Must(mds.GoodnessOfFit()), datamunge.Must(manhattanMds.GoodnessOfFit()))
}
