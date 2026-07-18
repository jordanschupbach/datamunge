#include <datamunge/stats/lda.hpp>

#include <datamunge/linalg/dense.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

namespace {

std::string format_stat(double v, int precision = 5) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Inf" : "-Inf";
    std::ostringstream oss;
    const double mag = std::fabs(v);
    if (v != 0.0 && (mag >= 1e6 || mag < 1e-4)) {
        oss << std::scientific << std::setprecision(3) << v;
    } else {
        oss << std::fixed << std::setprecision(precision) << v;
    }
    return oss.str();
}

// Solves L x = b for lower-triangular L (forward substitution).
std::vector<double> forward_solve(const linalg::DenseMatrix<double>& L, const std::vector<double>& b) {
    const std::size_t n = L.rows();
    std::vector<double> x(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        double s = b[i];
        for (std::size_t j = 0; j < i; ++j) s -= L(i, j) * x[j];
        x[i] = s / L(i, i);
    }
    return x;
}

// Solves L^T v = y for lower-triangular L (backward substitution).
std::vector<double> back_solve_transpose(const linalg::DenseMatrix<double>& L, const std::vector<double>& y) {
    const std::size_t n = L.rows();
    std::vector<double> v(n, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        double s = y[i];
        for (std::size_t j = i + 1; j < n; ++j) s -= L(j, i) * v[j];
        v[i] = s / L(i, i);
    }
    return v;
}

const plot::RGB kSeriesColors[] = {
    {37, 99, 235}, {220, 38, 38}, {22, 163, 74}, {217, 119, 6}, {124, 58, 237}, {8, 145, 178},
};

} // namespace

LDA::LDA(const dstruct::DataFrame& data, const std::string& formula, LDAOptions options) : formula_(formula) {
    fit(data, std::move(options));
}

void LDA::fit(const dstruct::DataFrame& data, LDAOptions options) {
    design_ = formula_.resolve(data, ResponseKind::Categorical);

    auto              cd = design_.build_classification_matrix(data);
    const std::size_t n  = cd.X.rows();
    const std::size_t p  = cd.X.cols();

    predictor_names_ = design_.coefficient_names;
    observations_    = n;
    training_labels_ = cd.labels;

    classes_.clear();
    for (const auto& label : cd.labels)
        if (std::find(classes_.begin(), classes_.end(), label) == classes_.end()) classes_.push_back(label);
    std::sort(classes_.begin(), classes_.end());
    const std::size_t k = classes_.size();
    if (k < 2)
        throw std::invalid_argument("LDA: response must have at least 2 distinct classes, found "
                                    + std::to_string(k));

    std::vector<std::size_t> class_index(n);
    std::vector<std::size_t> class_count(k, 0);
    for (std::size_t i = 0; i < n; ++i) {
        const auto it   = std::lower_bound(classes_.begin(), classes_.end(), cd.labels[i]);
        class_index[i]  = static_cast<std::size_t>(it - classes_.begin());
        ++class_count[class_index[i]];
    }

    if (n <= k + p)
        throw std::runtime_error("LDA: not enough observations (" + std::to_string(n) + ") for " + std::to_string(k)
                                 + " classes and " + std::to_string(p) + " predictors");
    for (std::size_t c = 0; c < k; ++c)
        if (class_count[c] < 2)
            throw std::runtime_error("LDA: class '" + classes_[c] + "' has fewer than 2 observations");

    priors_.assign(k, 0.0);
    if (options.priors) {
        if (options.priors->size() != k)
            throw std::invalid_argument("LDA: priors size must equal the number of classes (" + std::to_string(k)
                                        + ")");
        double sum = 0.0;
        for (const double pr : *options.priors) {
            if (!(pr > 0.0)) throw std::invalid_argument("LDA: priors must be strictly positive");
            sum += pr;
        }
        if (std::fabs(sum - 1.0) > 1e-6) throw std::invalid_argument("LDA: priors must sum to 1");
        priors_ = *options.priors;
    } else {
        for (std::size_t c = 0; c < k; ++c)
            priors_[c] = static_cast<double>(class_count[c]) / static_cast<double>(n);
    }

    group_means_.assign(k, std::vector<double>(p, 0.0));
    overall_mean_.assign(p, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            group_means_[class_index[i]][j] += cd.X(i, j);
            overall_mean_[j] += cd.X(i, j);
        }
    }
    for (std::size_t c = 0; c < k; ++c)
        for (std::size_t j = 0; j < p; ++j) group_means_[c][j] /= static_cast<double>(class_count[c]);
    for (std::size_t j = 0; j < p; ++j) overall_mean_[j] /= static_cast<double>(n);

    linalg::DenseMatrix<double> W(p, p, 0.0);
    linalg::DenseMatrix<double> B(p, p, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        const auto& mean = group_means_[class_index[i]];
        for (std::size_t a = 0; a < p; ++a) {
            const double da = cd.X(i, a) - mean[a];
            for (std::size_t b = 0; b < p; ++b) W(a, b) += da * (cd.X(i, b) - mean[b]);
        }
    }
    for (std::size_t c = 0; c < k; ++c) {
        for (std::size_t a = 0; a < p; ++a) {
            const double da = group_means_[c][a] - overall_mean_[a];
            for (std::size_t b = 0; b < p; ++b)
                B(a, b) += static_cast<double>(class_count[c]) * da * (group_means_[c][b] - overall_mean_[b]);
        }
    }

    const auto chol = linalg::cholesky(W);
    if (!chol.ok)
        throw std::runtime_error("LDA: pooled within-class covariance is not positive definite "
                                 "(too few observations relative to predictors, or collinear predictors)");
    const auto& L = chol.L;

    // Cache Bayes discriminant terms:
    //   Sigma_w = W / (n - k)  =>  Sigma_w^{-1} mu_c = (n - k) * W^{-1} mu_c
    const double dof_scale = static_cast<double>(n - k);
    precision_means_.assign(k, std::vector<double>(p, 0.0));
    mahalanobis_offset_.assign(k, 0.0);
    for (std::size_t c = 0; c < k; ++c) {
        const auto w_inv_mu = chol.solve(group_means_[c]);
        for (std::size_t j = 0; j < p; ++j) precision_means_[c][j] = dof_scale * w_inv_mu[j];
        double offset = 0.0;
        for (std::size_t j = 0; j < p; ++j) offset += group_means_[c][j] * precision_means_[c][j];
        mahalanobis_offset_[c] = offset;
    }

    // Discriminant axes: generalized eigenproblem B v = lambda W v, reduced
    // via the Cholesky factor W = L L^T to a standard symmetric eigenproblem
    // M y = lambda y with M = L^{-1} B L^{-T}, then v = L^{-T} y.
    linalg::DenseMatrix<double> Y(p, p, 0.0); // Y = L^{-1} B
    for (std::size_t col = 0; col < p; ++col) {
        const auto solved = forward_solve(L, B.col(col));
        for (std::size_t i = 0; i < p; ++i) Y(i, col) = solved[i];
    }
    linalg::DenseMatrix<double> M(p, p, 0.0); // M = L^{-1} Y^T == L^{-1} B L^{-T}
    for (std::size_t col = 0; col < p; ++col) {
        const auto solved = forward_solve(L, Y.row(col));
        for (std::size_t i = 0; i < p; ++i) M(i, col) = solved[i];
    }
    for (std::size_t i = 0; i < p; ++i)
        for (std::size_t j = i + 1; j < p; ++j) {
            const double avg = 0.5 * (M(i, j) + M(j, i));
            M(i, j) = avg;
            M(j, i) = avg;
        }

    const auto       eig    = linalg::jacobi_eigen(M);
    const std::size_t num_ld = std::min(k - 1, p);

    scaling_ = linalg::DenseMatrix<double>(p, num_ld, 0.0);
    proportion_of_trace_.assign(num_ld, 0.0);
    double eigen_total = 0.0;
    for (const double v : eig.eigenvalues) eigen_total += std::max(v, 0.0);
    const double sqrt_dof = std::sqrt(dof_scale);
    for (std::size_t ld = 0; ld < num_ld; ++ld) {
        const auto v = back_solve_transpose(L, eig.eigenvectors.col(ld));
        for (std::size_t j = 0; j < p; ++j) scaling_(j, ld) = v[j] * sqrt_dof;
        proportion_of_trace_[ld] = (eigen_total > 0.0) ? std::max(eig.eigenvalues[ld], 0.0) / eigen_total : 0.0;
    }

    const auto training_result = classify(cd.X);
    fitted_classes_          = training_result.class_label;
    training_discriminants_  = training_result.discriminants;
}

LDAPrediction LDA::classify(const linalg::DenseMatrix<double>& X) const {
    const std::size_t n      = X.rows();
    const std::size_t k      = classes_.size();
    const std::size_t num_ld = scaling_.cols();

    LDAPrediction result;
    result.class_label.resize(n);
    result.posterior.assign(n, std::vector<double>(k, 0.0));
    result.discriminants.assign(n, std::vector<double>(num_ld, 0.0));

    std::vector<double> delta(k);
    std::vector<double> expd(k);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t c = 0; c < k; ++c) {
            double dot = 0.0;
            for (std::size_t j = 0; j < X.cols(); ++j) dot += X(i, j) * precision_means_[c][j];
            delta[c] = dot - 0.5 * mahalanobis_offset_[c] + std::log(priors_[c]);
        }
        const double max_delta = *std::max_element(delta.begin(), delta.end());
        double       sum_exp   = 0.0;
        for (std::size_t c = 0; c < k; ++c) {
            expd[c] = std::exp(delta[c] - max_delta);
            sum_exp += expd[c];
        }
        std::size_t best = 0;
        for (std::size_t c = 0; c < k; ++c) {
            result.posterior[i][c] = expd[c] / sum_exp;
            if (delta[c] > delta[best]) best = c;
        }
        result.class_label[i] = classes_[best];

        for (std::size_t ld = 0; ld < num_ld; ++ld) {
            double proj = 0.0;
            for (std::size_t j = 0; j < X.cols(); ++j) proj += (X(i, j) - overall_mean_[j]) * scaling_(j, ld);
            result.discriminants[i][ld] = proj;
        }
    }
    return result;
}

double LDA::training_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < fitted_classes_.size(); ++i)
        if (fitted_classes_[i] == training_labels_[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

linalg::DenseMatrix<double> LDA::confusion_matrix() const {
    const std::size_t k = classes_.size();
    linalg::DenseMatrix<double> matrix(k, k, 0.0);
    for (std::size_t i = 0; i < training_labels_.size(); ++i) {
        const auto actual_it    = std::lower_bound(classes_.begin(), classes_.end(), training_labels_[i]);
        const auto predicted_it = std::lower_bound(classes_.begin(), classes_.end(), fitted_classes_[i]);
        const auto actual       = static_cast<std::size_t>(actual_it - classes_.begin());
        const auto predicted    = static_cast<std::size_t>(predicted_it - classes_.begin());
        matrix(actual, predicted) += 1.0;
    }
    return matrix;
}

std::vector<std::string> LDA::predict(const dstruct::DataFrame& newdata) const {
    return predict_detail(newdata).class_label;
}

LDAPrediction LDA::predict_detail(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "LDA::predict");
    const auto dm = design_.build_matrix(newdata, /*require_response=*/false);
    const auto compact = classify(dm.X);

    LDAPrediction result;
    result.class_label.assign(newdata.nrows(), "");
    result.posterior.assign(newdata.nrows(), {});
    result.discriminants.assign(newdata.nrows(), {});
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        const auto row               = dm.used_row_indices[i];
        result.class_label[row]      = compact.class_label[i];
        result.posterior[row]        = compact.posterior[i];
        result.discriminants[row]    = compact.discriminants[i];
    }
    return result;
}

std::string LDA::summary() const {
    std::ostringstream out;
    out << "Call:\nlda(formula = " << formula_.text() << ")\n\n";

    out << "Prior probabilities of groups:\n";
    for (const auto& c : classes_) out << std::setw(12) << c;
    out << "\n";
    for (const double pr : priors_) out << std::setw(12) << format_stat(pr, 4);
    out << "\n\n";

    out << "Group means:\n";
    out << std::setw(14) << "";
    for (const auto& name : predictor_names_) out << std::setw(14) << name;
    out << "\n";
    for (std::size_t c = 0; c < classes_.size(); ++c) {
        out << std::left << std::setw(14) << classes_[c] << std::right;
        for (const double v : group_means_[c]) out << std::setw(14) << format_stat(v, 4);
        out << "\n";
    }
    out << "\n";

    out << "Coefficients of linear discriminants:\n";
    out << std::setw(14) << "";
    for (std::size_t ld = 0; ld < scaling_.cols(); ++ld) out << std::setw(14) << ("LD" + std::to_string(ld + 1));
    out << "\n";
    for (std::size_t j = 0; j < predictor_names_.size(); ++j) {
        out << std::left << std::setw(14) << predictor_names_[j] << std::right;
        for (std::size_t ld = 0; ld < scaling_.cols(); ++ld) out << std::setw(14) << format_stat(scaling_(j, ld));
        out << "\n";
    }
    out << "\n";

    out << "Proportion of trace:\n";
    for (std::size_t ld = 0; ld < proportion_of_trace_.size(); ++ld)
        out << std::setw(10) << ("LD" + std::to_string(ld + 1));
    out << "\n";
    for (const double p : proportion_of_trace_) out << std::setw(10) << format_stat(p, 4);
    out << "\n\n";

    out << "Training accuracy: " << format_stat(training_accuracy() * 100.0, 2) << "%  (" << observations_
        << " observations, " << classes_.size() << " classes)\n";

    return out.str();
}

void LDA::print_summary(std::ostream& os) const { os << summary(); }

void LDA::print_summary() const { print_summary(std::cout); }

plot::ScatterPlot LDA::plot_discriminants() const {
    auto plot = plot::ScatterPlot::create();
    const bool has_ld2 = scaling_.cols() > 1;

    for (std::size_t c = 0; c < classes_.size(); ++c) {
        std::vector<double> xs;
        std::vector<double> ys;
        for (std::size_t i = 0; i < training_labels_.size(); ++i) {
            if (training_labels_[i] != classes_[c]) continue;
            xs.push_back(training_discriminants_[i][0]);
            ys.push_back(has_ld2 ? training_discriminants_[i][1] : 0.0);
        }
        const auto& color = kSeriesColors[c % (sizeof(kSeriesColors) / sizeof(kSeriesColors[0]))];
        plot.points(xs, ys, classes_[c], color);
    }

    plot.title("Linear Discriminant Analysis").x_label("LD1").y_label(has_ld2 ? "LD2" : "(single discriminant)");
    return plot;
}

void LDA::save_discriminant_plot(const std::string& path) const { plot_discriminants().save(path); }

} // namespace datamunge::stats
