/// @file diffusion_maps.cpp
/// @brief Shared dense Gaussian fit and optional actual-matrix capture.
#include <datamunge/stats/diffusion_maps.hpp>

#include <datamunge/linalg/eigen.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <limits>
#include <stdexcept>

namespace datamunge::stats {

/// @brief Stateless fit and plot implementation helpers.
namespace {

/// @brief Fixed six-colour palette for grouped plots.
const graphics::RGB kSeriesColors[] = {
    {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
};

/// @brief Copies, sorts and deduplicates group labels.
/// @param labels Borrowed groups.
/// @return Owned sorted unique groups.
std::vector<std::string> unique_sorted(const std::vector<std::string>& labels) {
    std::vector<std::string> result(labels.begin(), labels.end());
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

/// @brief Computes raw squared row distance, rejecting nonfinite results.
/// @param X Finite numeric matrix.
/// @param i First row.
/// @param j Second row.
/// @return Finite squared distance.
/// @throws std::overflow_error Unrepresentable squared distance.
double row_distance_squared(const linalg::DenseMatrix<double>& X, const std::size_t i, const std::size_t j) {
    double sum = 0.0;
    for (std::size_t k = 0; k < X.cols(); ++k) {
        const double d = X(i, k) - X(j, k);
        sum += d * d;
    }
    if (!std::isfinite(sum)) throw std::overflow_error("DiffusionMaps: squared distance overflow");
    return sum;
}

/// @brief Fits once, optionally retaining the actual intermediate matrices.
/// @param data Borrowed input.
/// @param feature_columns_ Metric columns.
/// @param options_ Fit options.
/// @param capture Whether to retain square intermediate matrices.
/// @param cap Complete-case cap; zero disables it.
/// @return Owned actual fit results.
DiffusionMapsTrace compute_diffusion(const dstruct::DataFrame& data,
    const std::vector<std::string>& feature_columns_, DiffusionMapsOptions options_, bool capture, std::size_t cap) {
    DiffusionMapsTrace result;
    auto& kept_row_indices_=result.kept_row_indices;
    auto& eigenvalues_=result.eigenvalues;
    auto& embedding_=result.embedding;
    if (feature_columns_.empty()) throw std::invalid_argument("DiffusionMaps: feature_columns must not be empty");
    if (options_.n_components == 0) throw std::invalid_argument("DiffusionMaps: n_components must be at least 1");
    if (!std::isfinite(options_.heat_kernel_epsilon) || options_.heat_kernel_epsilon <= 0.0)
        throw std::invalid_argument("DiffusionMaps: heat_kernel_epsilon must be positive");
    if (!std::isfinite(options_.alpha) || options_.alpha < 0.0 || options_.alpha > 1.0)
        throw std::invalid_argument("DiffusionMaps: alpha must be in [0, 1]");
    if (!std::isfinite(options_.diffusion_time) || options_.diffusion_time < 0)
        throw std::invalid_argument("DiffusionMaps: diffusion_time must be finite and nonnegative");
    for (const auto& col : feature_columns_) {
        if (!data.has_column(col)) throw std::invalid_argument("DiffusionMaps: column '" + col + "' not found");
        if (data.column_type(col) != dstruct::DataFrame::ColumnType::Numeric)
            throw std::invalid_argument("DiffusionMaps: column '" + col + "' must be numeric");
    }

    kept_row_indices_.clear();
    for (std::size_t i = 0; i < data.nrows(); ++i) {
        bool ok = true;
        for (const auto& col : feature_columns_) {
            if (data.is_null(col, i)) {
                ok = false;
                break;
            }
        }
        if (ok) kept_row_indices_.push_back(i);
    }

    const std::size_t n = kept_row_indices_.size();
    const std::size_t p = feature_columns_.size();
    // One eigenvalue/eigenvector pair (the trivial ~1 one) is always discarded, so we need
    // n_components() usable pairs beyond it.
    if (n < 3 || options_.n_components >= n - 1)
        throw std::invalid_argument(
            "DiffusionMaps: n_components must be less than (number of complete observations - 1)");

    if ((cap && n>cap) || n>std::vector<double>().max_size()/n || n>std::vector<double>().max_size()/p)
        throw std::length_error("DiffusionMaps: dense observation/capacity limit");
    linalg::DenseMatrix<double> X(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) {
            X(i,j)=data.double_at(feature_columns_[j],kept_row_indices_[i]);
            if (!std::isfinite(X(i,j))) throw std::invalid_argument("DiffusionMaps: features must be finite");
        }

    // Dense all-pairs squared Euclidean distance matrix (hardcoded Euclidean, per the algorithm's
    // standard literature definition).
    linalg::DenseMatrix<double> D2(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = i + 1; j < n; ++j) {
            const double d2 = row_distance_squared(X, i, j);
            D2(i, j) = d2;
            D2(j, i) = d2;
        }
    }

    // Dense heat kernel: K(i,j) = exp(-D2(i,j) / epsilon). K(i,i) = exp(0) = 1 automatically.
    linalg::DenseMatrix<double> K(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) K(i, j) = std::exp(-D2(i, j) / options_.heat_kernel_epsilon);

    // Underflow can split the graph; a single discarded mode then need not be constant.
    std::vector<bool> seen(n,false);std::vector<std::size_t> frontier{0};seen[0]=true;
    for(std::size_t u=0;u<frontier.size();++u)
        for(std::size_t j=0;j<n;++j) if(!seen[j] && K(frontier[u],j)>0){seen[j]=true;frontier.push_back(j);}
    if(frontier.size()!=n) throw std::runtime_error("DiffusionMaps: floating-point kernel is disconnected");
    // Alpha degree correction; density independence is an asymptotic statement.
    std::vector<double> q(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) sum += K(i, j);
        q[i] = sum;
    }
    std::vector<double> q_pow_alpha(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) q_pow_alpha[i] = std::pow(q[i], options_.alpha);

    linalg::DenseMatrix<double> Kp(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) Kp(i, j) = K(i, j) / (q_pow_alpha[i] * q_pow_alpha[j]);

    // Row-normalize into a (generally non-symmetric) row-stochastic Markov matrix P; d(i) is its
    // row-sum normalizer.
    std::vector<double> d(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (std::size_t j = 0; j < n; ++j) sum += Kp(i, j);
        d[i] = sum;
    }

    // Symmetrize for eigendecomposition: P_sym(i,j) = Kp(i,j) / (sqrt(d(i)) * sqrt(d(j))).
    // P_sym has the same eigenvalues as P; if v is an eigenvector of P_sym, D^{-1/2} v is the
    // corresponding right eigenvector of P.
    std::vector<double> sqrt_d(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) sqrt_d[i] = std::sqrt(d[i]);

    linalg::DenseMatrix<double> Psym(n, n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) Psym(i, j) = Kp(i, j) / (sqrt_d[i] * sqrt_d[j]);

    const auto eig = linalg::symmetric_eigen(Psym);
    eigenvalues_ = eig.eigenvalues;
    for(auto& value:eigenvalues_) {
        if(value < -128.0*n*std::numeric_limits<double>::epsilon())
            throw std::runtime_error("DiffusionMaps: materially negative Gaussian spectrum");
        value=std::max(0.0,value);
    }

    // Discard the top (trivial, ~1) eigenpair; use the next n_components pairs.
    embedding_ = linalg::DenseMatrix<double>(n, options_.n_components, 0.0);
    for (std::size_t c = 0; c < options_.n_components; ++c) {
        const std::size_t eig_index = c + 1;
        const double lambda = eigenvalues_[eig_index];
        const double scale = std::pow(lambda, options_.diffusion_time);
        for (std::size_t i = 0; i < n; ++i) {
            const double psi_k_i = eig.eigenvectors(i, eig_index) / sqrt_d[i];
            embedding_(i, c) = scale * psi_k_i;
        }
    }

    if(capture) {
        result.squared_distances=std::move(D2);result.kernel=std::move(K);
        result.kernel_degrees=std::move(q);result.normalized_kernel=std::move(Kp);
        result.degrees=d;result.symmetric=std::move(Psym);
        result.stationary.resize(n);double total=0;for(double degree:d)total+=degree;
        result.transition=linalg::DenseMatrix<double>(n,n);
        result.right_eigenvectors=linalg::DenseMatrix<double>(n,n);
        for(std::size_t i=0;i<n;++i){
            result.stationary[i]=d[i]/total;
            for(std::size_t j=0;j<n;++j){
                result.transition(i,j)=result.normalized_kernel(i,j)/d[i];
                result.right_eigenvectors(i,j)=eig.eigenvectors(i,j)/sqrt_d[i];
            }
        }
    }
    return result;
}
} // namespace

DiffusionMapsTrace diffusion_maps_trace(const dstruct::DataFrame& data,const std::vector<std::string>& feature_columns,
    DiffusionMapsOptions options,std::size_t max_observations){
    return compute_diffusion(data,feature_columns,options,true,max_observations);
}

DiffusionMaps::DiffusionMaps(const dstruct::DataFrame& data,const std::vector<std::string>& feature_columns,
    DiffusionMapsOptions options):feature_columns_(feature_columns),options_(options){fit(data);}

void DiffusionMaps::fit(const dstruct::DataFrame& data){
    auto result=compute_diffusion(data,feature_columns_,options_,false,0);
    kept_row_indices_=std::move(result.kept_row_indices);eigenvalues_=std::move(result.eigenvalues);
    embedding_=std::move(result.embedding);observations_=kept_row_indices_.size();
}

std::vector<double> DiffusionMaps::dimension(const std::size_t index) const {
    if (index >= embedding_.cols()) throw std::out_of_range("DiffusionMaps::dimension index out of range");
    return embedding_.col(index);
}

std::string DiffusionMaps::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void DiffusionMaps::print_summary() const { print_summary(std::cout); }

void DiffusionMaps::print_summary(std::ostream& os) const {
    os << "Diffusion Maps\n";
    os << "Features: ";
    for (std::size_t j = 0; j < feature_columns_.size(); ++j) os << (j ? ", " : "") << feature_columns_[j];
    os << "\n";
    os << "Number of obs: " << observations_ << ", dimensions: " << n_components() << "\n";
    os << std::fixed << std::setprecision(4);
    os << "Heat kernel epsilon: " << options_.heat_kernel_epsilon << ", alpha: " << options_.alpha
       << ", diffusion time: " << options_.diffusion_time << "\n\n";

    os << "Eigenvalues of symmetrized Markov matrix (top is trivial, discarded):\n";
    for (std::size_t c = 0; c < eigenvalues_.size() && c < n_components() + 1; ++c)
        os << "  lambda" << c << ": " << eigenvalues_[c] << (c == 0 ? " (trivial, discarded)\n" : "\n");
}

graphics::RPlot DiffusionMaps::plot_embedding(const std::size_t dimension_x, const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("DiffusionMaps::plot_embedding dimension index out of range");

    auto plot = graphics::RPlot::create();
    plot.points(embedding_.col(dimension_x), embedding_.col(dimension_y), "", kSeriesColors[0]);
    plot.title("Diffusion Maps")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

graphics::RPlot DiffusionMaps::plot_embedding(const std::vector<std::string>& group_labels, const std::size_t dimension_x,
                                          const std::size_t dimension_y) const {
    if (dimension_x >= n_components() || dimension_y >= n_components())
        throw std::out_of_range("DiffusionMaps::plot_embedding dimension index out of range");
    if (group_labels.size() != observations_)
        throw std::invalid_argument(
            "DiffusionMaps::plot_embedding: group_labels must have one entry per fitted observation");

    auto plot = graphics::RPlot::create();
    const auto groups = unique_sorted(group_labels);
    for (std::size_t g = 0; g < groups.size(); ++g) {
        std::vector<double> xs, ys;
        for (std::size_t i = 0; i < observations_; ++i) {
            if (group_labels[i] != groups[g]) continue;
            xs.push_back(embedding_(i, dimension_x));
            ys.push_back(embedding_(i, dimension_y));
        }
        plot.points(xs, ys, groups[g], kSeriesColors[g % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))]);
    }
    plot.title("Diffusion Maps")
        .x_label("Dim" + std::to_string(dimension_x + 1))
        .y_label("Dim" + std::to_string(dimension_y + 1));
    return plot;
}

} // namespace datamunge::stats
