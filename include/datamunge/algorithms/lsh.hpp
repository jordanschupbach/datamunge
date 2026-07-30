#pragma once

/// \file lsh.hpp
/// \brief Locality-sensitive hashing (LSH) for approximate nearest-neighbor search
///        under cosine similarity, using random hyperplanes (Charikar 2002).
///
/// Exact nearest-neighbor search in high dimensions is slow -- every query scans the
/// whole dataset. LSH trades exactness for speed with hash functions that are
/// *locality-sensitive*: nearby points are far more likely to collide (hash to the
/// same bucket) than distant ones. For *cosine* similarity the classic family is the
/// *random hyperplane* hash: pick a random vector \f$r\f$ and let
/// \f$h_r(x) = \operatorname{sign}(r\cdot x)\f$. The probability two vectors hash to the
/// same bit is exactly
/// \f[
///   \Pr[h_r(x)=h_r(y)] = 1 - \frac{\theta(x,y)}{\pi},
/// \f]
/// where \f$\theta\f$ is the angle between them -- so bit-agreement (Hamming similarity
/// of signatures) directly estimates angular closeness.
///
/// To turn this into sublinear ANN search we AND and OR the hashes: each of \f$L\f$
/// tables concatenates \f$k\f$ random-hyperplane bits into a bucket key (an AND that
/// makes accidental collisions rare), and a query gathers candidates from its bucket
/// in *every* table (an OR that recovers true neighbors missed by any single table),
/// then verifies candidates by exact cosine similarity.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <unordered_map>
#include <vector>

namespace datamunge::algorithms {

/// A random-hyperplane LSH index for cosine-similarity approximate nearest neighbors.
class CosineLSH {
 public:
    /// Build \p num_tables hash tables, each keyed by \p bits_per_table random-hyperplane bits.
    CosineLSH(const std::vector<std::vector<double>>& data, std::size_t num_tables, std::size_t bits_per_table,
              std::uint64_t seed = 0)
        : data_(&data), dim_(data.empty() ? 0 : data.front().size()), tables_(num_tables),
          bits_(bits_per_table) {
        std::mt19937_64                  rng(seed);
        std::normal_distribution<double> gauss(0.0, 1.0);
        // Random hyperplane normals: hyperplanes_[table][bit] is a dim-vector.
        hyperplanes_.assign(num_tables, {});
        for (std::size_t t = 0; t < num_tables; ++t) {
            hyperplanes_[t].assign(bits_per_table, std::vector<double>(dim_));
            for (auto& h : hyperplanes_[t])
                for (double& v : h) v = gauss(rng);
        }
        buckets_.assign(num_tables, {});
        for (std::size_t i = 0; i < data.size(); ++i)
            for (std::size_t t = 0; t < num_tables; ++t) buckets_[t][signature(data[i], t)].push_back(i);
    }

    /// Approximate nearest neighbor of \p query by cosine similarity (index into the data, or
    /// SIZE_MAX if no candidate collided in any table). \p out_examined receives the candidate count.
    std::size_t query(const std::vector<double>& q, std::size_t* out_examined = nullptr) const {
        std::vector<char> seen(data_->size(), 0);
        std::size_t       best     = static_cast<std::size_t>(-1);
        double            best_sim = -2.0;
        std::size_t       examined = 0;
        for (std::size_t t = 0; t < tables_; ++t) {
            auto it = buckets_[t].find(signature(q, t));
            if (it == buckets_[t].end()) continue;
            for (std::size_t id : it->second) {
                if (seen[id]) continue;
                seen[id] = 1;
                ++examined;
                const double s = cosine(q, (*data_)[id]);
                if (s > best_sim) {
                    best_sim = s;
                    best     = id;
                }
            }
        }
        if (out_examined) *out_examined = examined;
        return best;
    }

    /// Cosine similarity between two vectors.
    static double cosine(const std::vector<double>& a, const std::vector<double>& b) {
        double dot = 0.0, na = 0.0, nb = 0.0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        const double denom = std::sqrt(na) * std::sqrt(nb);
        return denom > 0.0 ? dot / denom : 0.0;
    }

    /// The \p bits_per_table-bit signature of \p x in table \p t (a random-hyperplane sign pattern).
    std::uint64_t signature(const std::vector<double>& x, std::size_t t) const {
        std::uint64_t sig = 0;
        for (std::size_t b = 0; b < bits_; ++b) {
            double dot = 0.0;
            const auto& h = hyperplanes_[t][b];
            for (std::size_t i = 0; i < dim_; ++i) dot += h[i] * x[i];
            if (dot >= 0.0) sig |= (std::uint64_t{1} << b);
        }
        return sig;
    }

 private:
    const std::vector<std::vector<double>>*                       data_;
    std::size_t                                                   dim_;
    std::size_t                                                   tables_;
    std::size_t                                                   bits_;
    std::vector<std::vector<std::vector<double>>>                 hyperplanes_;  // [table][bit][dim]
    std::vector<std::unordered_map<std::uint64_t, std::vector<std::size_t>>> buckets_;
};

/// \brief Theoretical collision probability of a single random-hyperplane bit for angle \p theta.
inline double lsh_collision_probability(double theta) { return 1.0 - theta / M_PI; }

}  // namespace datamunge::algorithms
