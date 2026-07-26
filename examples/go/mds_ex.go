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

func main() {
	iris := datamunge.DataFrameIris()
	features := svector([]string{"Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width"})

	n := iris.Nrows()
	speciesT := make([]string, 0, n)
	for i := int64(0); i < n; i++ {
		speciesT = append(speciesT, iris.String_at("Species", i))
	}
	species := svector(speciesT)

	fmt.Println("=================== Classical MDS on iris (euclidean) ===================")
	mds := datamunge.NewMDS(iris, features, int64(2), "euclidean")
	mds.Print_summary()

	fmt.Println("\nEmbedding as a DataFrame:")
	fmt.Println(mds.Embedding_frame().To_string())

	scatter := mds.Plot_embedding_grouped(species)
	scatter.Save_svg("mds_iris_embedding.svg")
	fmt.Println("\nEmbedding scatter (colored by species) saved as mds_iris_embedding.svg")

	fmt.Println("\n=================== Classical MDS on iris (manhattan) ===================")
	manhattanMds := datamunge.NewMDS(iris, features, int64(2), "manhattan")
	fmt.Printf("Goodness of fit: euclidean=%.4f, manhattan=%.4f\n",
		mds.Goodness_of_fit(), manhattanMds.Goodness_of_fit())
}
