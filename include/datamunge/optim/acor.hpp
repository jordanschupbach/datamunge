#pragma once

#include <datamunge/optim/function_types.hpp>

#include <cstddef>
#include <cstdint>

namespace datamunge::optim {

struct ACOROptions {
    /// @brief k: number of solutions kept in the archive.
    std::size_t archive_size{50};
    /// @brief m: new ants generated each iteration.
    std::size_t samples_per_iteration{40};
    /// @brief q: locality of the search; smaller values concentrate sampling more tightly
    ///        around the best-ranked archive solutions.
    double locality{0.5};
    /// @brief xi (ξ): convergence speed; larger values widen sampling around each archive
    ///        solution, slowing convergence.
    double convergence_speed{0.85};
    std::size_t max_iterations{500};
    /// @brief Stops after the best-ever value has improved by less than this for 20 consecutive
    ///        iterations.
    double tolerance{1e-10};
    std::uint64_t seed{42};
};

/// @brief Ant Colony Optimization for continuous domains (ACOR), Socha & Dorigo (2008). A
///        solution archive of size k is maintained; each iteration, new ants are sampled from a
///        Gaussian kernel mixture built from the archive (ranked by quality, so better solutions
///        attract denser sampling), then the archive is truncated back to its k best members
///        merged with the new ants. Derivative-free, box-constrained.
class ACOR {
public:
    explicit ACOR(ACOROptions options = {});

    /// @brief Minimizes @p function within [@p lower_bound, @p upper_bound], updating
    ///        @p coordinates in place to the best point found, and returns its value. The
    ///        initial @p coordinates (clamped into bounds) seed one member of the archive.
    double optimize(ArbitraryFunction& function, std::vector<double>& coordinates,
                     const std::vector<double>& lower_bound, const std::vector<double>& upper_bound) const;

private:
    ACOROptions options_;
};

} // namespace datamunge::optim
