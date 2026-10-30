#include <datamunge/datasets/datasets.hpp>
#include <datamunge/dstruct/dstruct.hpp>
#include <datamunge/linalg/tensor.hpp>

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

using datamunge::linalg::Tensor;

int main() {
    std::cout << std::fixed << std::setprecision(4);

    std::cout << "=================== Construction ===================\n";
    auto z = Tensor::zeros({2, 3});
    std::cout << "zeros({2,3}): " << z.to_string() << "\n\n";

    auto id = Tensor::eye(3);
    std::cout << "eye(3): " << id.to_string() << "\n\n";

    auto r = Tensor::arange(0.0, 12.0, 1.0).reshape({3, 4});
    std::cout << "arange(0,12).reshape({3,4}): " << r.to_string() << "\n\n";

    std::cout << "=================== Shape ops ===================\n";
    const auto rt = r.transpose();
    std::cout << "transpose -> shape [" << rt.shape()[0] << ", " << rt.shape()[1] << "]\n";
    const auto sliced = r.slice(1, 1, 3);
    std::cout << "slice(axis=1, start=1, stop=3): " << sliced.to_string() << "\n\n";

    std::cout << "=================== Broadcasting arithmetic ===================\n";
    const auto col = Tensor::from_values({3, 1}, {1, 2, 3});
    const auto row = Tensor::from_values({1, 4}, {10, 20, 30, 40});
    const auto broadcast_sum = col.add(row);
    std::cout << "(3,1) + (1,4) -> " << broadcast_sum.to_string() << "\n\n";

    std::cout << "=================== Reductions ===================\n";
    std::cout << "r.sum() = " << r.sum() << ", r.mean() = " << r.mean() << "\n";
    const auto col_means = r.mean_axis(0);
    std::cout << "column means (axis=0): " << col_means.to_string() << "\n\n";

    std::cout << "=================== Linear algebra ===================\n";
    const auto a = Tensor::from_values({2, 3}, {1, 2, 3, 4, 5, 6});
    const auto b = Tensor::from_values({3, 2}, {7, 8, 9, 10, 11, 12});
    std::cout << "matmul(2x3, 3x2) -> " << a.matmul(b).to_string() << "\n\n";

    const auto v1 = Tensor::from_values({3}, {1, 2, 3});
    const auto v2 = Tensor::from_values({3}, {4, 5, 6});
    std::cout << "dot([1,2,3], [4,5,6]) = " << v1.dot(v2) << "\n";
    std::cout << "outer(v1, v2) -> " << v1.outer(v2).to_string() << "\n\n";

    std::cout << "=================== Comparisons & masks ===================\n";
    const auto mask = r.greater_equal(Tensor::full({3, 4}, 6.0));
    std::cout << "r >= 6 -> " << mask.to_string() << "\n";
    std::cout << "count(r >= 6) = " << mask.sum() << "\n\n";

    std::cout << "=================== A real dataset as a Tensor ===================\n";
    const auto iris = datamunge::datasets::iris();
    std::vector<double> flat;
    flat.reserve(iris.nrows() * 4);
    for (std::size_t i = 0; i < iris.nrows(); ++i) {
        flat.push_back(iris.double_at("Sepal.Length", i));
        flat.push_back(iris.double_at("Sepal.Width", i));
        flat.push_back(iris.double_at("Petal.Length", i));
        flat.push_back(iris.double_at("Petal.Width", i));
    }
    const auto X = Tensor::from_values({iris.nrows(), 4}, flat);
    std::cout << "iris feature tensor shape: [" << X.shape()[0] << ", " << X.shape()[1] << "]\n";

    const auto feature_means = X.mean_axis(0);
    const auto centered = X.subtract(feature_means.reshape({1, 4}));
    const auto scatter = centered.transpose().matmul(centered);
    std::cout << "feature means (Sepal.Length, Sepal.Width, Petal.Length, Petal.Width): " << feature_means.to_string()
              << "\n";
    std::cout << "(X-mean)^T (X-mean) [4x4 scatter matrix]: " << scatter.to_string(16) << "\n\n";

    std::vector<std::string> species;
    species.reserve(iris.nrows());
    for (std::size_t i = 0; i < iris.nrows(); ++i) species.push_back(iris.string_at("Species", i));
    const auto species_tensor = Tensor::from_string_values({iris.nrows()}, species);
    const auto setosa_mask = species_tensor.equal(Tensor::from_string_values({1}, {"setosa"}));
    std::cout << "setosa count = " << setosa_mask.sum() << " (of " << iris.nrows() << " rows)\n";

    return 0;
}
