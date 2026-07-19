from pydatamunge import datamunge as dm

iris = dm.DataFrame.iris()
print(f"iris: {iris.nrows()} rows x {iris.ncols()} cols\n")

model = dm.LDA(iris, "Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width")
model.print_summary()

print("\nConfusion matrix (rows = actual, cols = predicted):")
print(model.confusion_matrix().to_string())

newdata = dm.DataFrame()
newdata.add_numeric_column("Sepal.Length", [5.1, 6.0, 6.5, 6.2])
newdata.add_numeric_column("Sepal.Width", [3.5, 2.7, 3.0, 2.8])
newdata.add_numeric_column("Petal.Length", [1.4, 4.5, 5.5, 4.8])
newdata.add_numeric_column("Petal.Width", [0.2, 1.5, 2.0, 1.8])

print("\nPredictions for new flowers:")
print(model.predict_frame(newdata).to_string())

model.save_discriminant_plot("lda_iris_discriminants.svg")
print("\nSaved discriminant plot as lda_iris_discriminants.svg")
