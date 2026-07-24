#include <datamunge/fem/mesh1d.hpp>

#include <stdexcept>

namespace datamunge::fem {

Mesh1D make_uniform_mesh1d(double a, double b, std::size_t num_elements) {
    if (num_elements == 0) {
        throw std::invalid_argument("make_uniform_mesh1d: num_elements must be positive");
    }
    if (b <= a) {
        throw std::invalid_argument("make_uniform_mesh1d: b must be greater than a");
    }

    Mesh1D mesh;
    mesh.nodes.resize(num_elements + 1);
    const double h = (b - a) / static_cast<double>(num_elements);
    for (std::size_t i = 0; i <= num_elements; ++i) {
        mesh.nodes[i] = a + static_cast<double>(i) * h;
    }
    mesh.nodes[num_elements] = b; // avoid float drift on the last node
    return mesh;
}

} // namespace datamunge::fem
