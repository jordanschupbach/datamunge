#include <datamunge/stats/svm.hpp>

#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/random/random.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

namespace {

std::string format_stat(double v, int precision = 4) {
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

const char* kernel_name(SVMKernel kernel) {
    switch (kernel) {
        case SVMKernel::Linear: return "linear";
        case SVMKernel::Polynomial: return "polynomial";
        case SVMKernel::Radial: return "radial";
        case SVMKernel::Sigmoid: return "sigmoid";
    }
    return "unknown";
}

struct SMOResult {
    std::vector<double> alpha;
    double               bias{0.0};
};

// Simplified Sequential Minimal Optimization (Platt's algorithm, the
// textbook two-multiplier-update variant): solves the soft-margin SVM dual
//   maximize   sum(alpha_i) - 0.5 * sum_ij alpha_i alpha_j y_i y_j K(i,j)
//   subject to 0 <= alpha_i <= C,  sum(alpha_i y_i) = 0
// Random pair selection uses a fixed seed so fits are deterministic.
SMOResult train_binary_smo(const linalg::DenseMatrix<double>& K, const std::vector<double>& y, double C,
                           double tol, std::size_t max_passes) {
    const std::size_t m = y.size();
    std::vector<double> alpha(m, 0.0);
    double               b = 0.0;

    random::SplitMix64 rng(0x5A17ULL + m);

    auto decision = [&](std::size_t i) {
        double s = b;
        for (std::size_t k = 0; k < m; ++k) s += alpha[k] * y[k] * K(i, k);
        return s;
    };

    std::size_t passes           = 0;
    std::size_t total_sweeps     = 0;
    const std::size_t sweep_cap = std::max<std::size_t>(200, max_passes * 20);

    while (passes < max_passes && total_sweeps < sweep_cap) {
        ++total_sweeps;
        std::size_t num_changed = 0;

        for (std::size_t i = 0; i < m; ++i) {
            const double E_i = decision(i) - y[i];
            const bool   violates = (y[i] * E_i < -tol && alpha[i] < C) || (y[i] * E_i > tol && alpha[i] > 0.0);
            if (!violates) continue;

            std::size_t j = i;
            while (j == i) j = static_cast<std::size_t>(rng.uniform01() * static_cast<double>(m)) % m;

            const double E_j = decision(j) - y[j];
            const double alpha_i_old = alpha[i];
            const double alpha_j_old = alpha[j];

            double L, H;
            if (y[i] != y[j]) {
                L = std::max(0.0, alpha[j] - alpha[i]);
                H = std::min(C, C + alpha[j] - alpha[i]);
            } else {
                L = std::max(0.0, alpha[i] + alpha[j] - C);
                H = std::min(C, alpha[i] + alpha[j]);
            }
            if (L >= H) continue;

            const double eta = 2.0 * K(i, j) - K(i, i) - K(j, j);
            if (eta >= 0.0) continue;

            double alpha_j_new = alpha[j] - y[j] * (E_i - E_j) / eta;
            alpha_j_new         = std::clamp(alpha_j_new, L, H);
            if (std::abs(alpha_j_new - alpha_j_old) < 1e-7) continue;

            const double alpha_i_new = alpha[i] + y[i] * y[j] * (alpha_j_old - alpha_j_new);

            const double b1 = b - E_i - y[i] * (alpha_i_new - alpha_i_old) * K(i, i)
                             - y[j] * (alpha_j_new - alpha_j_old) * K(i, j);
            const double b2 = b - E_j - y[i] * (alpha_i_new - alpha_i_old) * K(i, j)
                             - y[j] * (alpha_j_new - alpha_j_old) * K(j, j);

            alpha[i] = alpha_i_new;
            alpha[j] = alpha_j_new;

            if (alpha[i] > 0.0 && alpha[i] < C) b = b1;
            else if (alpha[j] > 0.0 && alpha[j] < C) b = b2;
            else b = 0.5 * (b1 + b2);

            ++num_changed;
        }

        passes = (num_changed == 0) ? passes + 1 : 0;
    }

    return SMOResult{std::move(alpha), b};
}

} // namespace

SVM::SVM(const dstruct::DataFrame& data, const std::string& formula, SVMOptions options) : formula_(formula) {
    fit(data, std::move(options));
}

double SVM::kernel(const std::vector<double>& a, const std::vector<double>& b) const {
    switch (options_.kernel) {
        case SVMKernel::Linear: {
            double dot = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
            return dot;
        }
        case SVMKernel::Polynomial: {
            double dot = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
            return std::pow(effective_gamma_ * dot + options_.coef0, options_.degree);
        }
        case SVMKernel::Radial: {
            double sq = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) {
                const double d = a[i] - b[i];
                sq += d * d;
            }
            return std::exp(-effective_gamma_ * sq);
        }
        case SVMKernel::Sigmoid: {
            double dot = 0.0;
            for (std::size_t i = 0; i < a.size(); ++i) dot += a[i] * b[i];
            return std::tanh(effective_gamma_ * dot + options_.coef0);
        }
    }
    return 0.0;
}

std::vector<double> SVM::scale_row(const std::vector<double>& raw) const {
    std::vector<double> out(raw.size());
    for (std::size_t j = 0; j < raw.size(); ++j) out[j] = (raw[j] - feature_mean_[j]) / feature_scale_[j];
    return out;
}

void SVM::fit(const dstruct::DataFrame& data, SVMOptions options) {
    options_ = options;
    design_  = formula_.resolve(data, ResponseKind::Categorical);

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
        throw std::invalid_argument("SVM: response must have at least 2 distinct classes, found " + std::to_string(k));

    effective_gamma_ = (options_.gamma > 0.0) ? options_.gamma : 1.0 / static_cast<double>(p);

    feature_mean_.assign(p, 0.0);
    feature_scale_.assign(p, 1.0);
    if (options_.scale) {
        for (std::size_t j = 0; j < p; ++j) {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) sum += cd.X(i, j);
            feature_mean_[j] = sum / static_cast<double>(n);
        }
        for (std::size_t j = 0; j < p; ++j) {
            double sq = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                const double d = cd.X(i, j) - feature_mean_[j];
                sq += d * d;
            }
            const double variance = (n > 1) ? sq / static_cast<double>(n - 1) : 0.0;
            feature_scale_[j]     = (variance > 1e-12) ? std::sqrt(variance) : 1.0;
        }
    }

    scaled_training_X_ = linalg::DenseMatrix<double>(n, p, 0.0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < p; ++j) scaled_training_X_(i, j) = (cd.X(i, j) - feature_mean_[j]) / feature_scale_[j];

    pair_models_.clear();
    for (std::size_t ci = 0; ci < k; ++ci) {
        for (std::size_t cj = ci + 1; cj < k; ++cj) {
            std::vector<std::size_t> members;
            for (std::size_t i = 0; i < n; ++i)
                if (training_labels_[i] == classes_[ci] || training_labels_[i] == classes_[cj]) members.push_back(i);

            const std::size_t m = members.size();
            std::vector<double> y(m);
            for (std::size_t idx = 0; idx < m; ++idx) y[idx] = (training_labels_[members[idx]] == classes_[ci]) ? 1.0 : -1.0;

            linalg::DenseMatrix<double> K(m, m, 0.0);
            for (std::size_t a = 0; a < m; ++a) {
                for (std::size_t bIdx = a; bIdx < m; ++bIdx) {
                    const double value = kernel(scaled_training_X_.row(members[a]), scaled_training_X_.row(members[bIdx]));
                    K(a, bIdx) = value;
                    K(bIdx, a) = value;
                }
            }

            const auto smo = train_binary_smo(K, y, options_.cost, options_.tolerance, options_.max_passes);

            PairModel pair;
            pair.class_i = ci;
            pair.class_j = cj;
            pair.bias    = smo.bias;
            for (std::size_t idx = 0; idx < m; ++idx) {
                if (smo.alpha[idx] > 1e-8) {
                    pair.support_indices.push_back(members[idx]);
                    pair.support_alpha_y.push_back(smo.alpha[idx] * y[idx]);
                }
            }
            pair_models_.push_back(std::move(pair));
        }
    }

    fitted_classes_ = classify(scaled_training_X_).class_label;
}

SVMPrediction SVM::classify(const linalg::DenseMatrix<double>& X) const {
    const std::size_t n = X.rows();
    const std::size_t k = classes_.size();

    SVMPrediction result;
    result.class_label.resize(n);
    result.votes.assign(n, std::vector<double>(k, 0.0));

    for (std::size_t row = 0; row < n; ++row) {
        const auto x = X.row(row);

        for (const auto& pair : pair_models_) {
            double decision = pair.bias;
            for (std::size_t sv = 0; sv < pair.support_indices.size(); ++sv)
                decision += pair.support_alpha_y[sv] * kernel(scaled_training_X_.row(pair.support_indices[sv]), x);

            if (decision >= 0.0) result.votes[row][pair.class_i] += 1.0;
            else result.votes[row][pair.class_j] += 1.0;
        }

        std::size_t best = 0;
        for (std::size_t c = 1; c < k; ++c)
            if (result.votes[row][c] > result.votes[row][best]) best = c;
        result.class_label[row] = classes_[best];
    }
    return result;
}

double SVM::training_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < fitted_classes_.size(); ++i)
        if (fitted_classes_[i] == training_labels_[i]) ++correct;
    return static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

linalg::DenseMatrix<double> SVM::confusion_matrix() const {
    const std::size_t k = classes_.size();
    linalg::DenseMatrix<double> matrix(k, k, 0.0);
    for (std::size_t i = 0; i < training_labels_.size(); ++i) {
        const auto actual_it    = std::lower_bound(classes_.begin(), classes_.end(), training_labels_[i]);
        const auto predicted_it = std::lower_bound(classes_.begin(), classes_.end(), fitted_classes_[i]);
        matrix(static_cast<std::size_t>(actual_it - classes_.begin()),
               static_cast<std::size_t>(predicted_it - classes_.begin())) += 1.0;
    }
    return matrix;
}

std::size_t SVM::num_support_vectors() const {
    std::vector<bool> is_sv(observations_, false);
    for (const auto& pair : pair_models_)
        for (const auto idx : pair.support_indices) is_sv[idx] = true;
    return static_cast<std::size_t>(std::count(is_sv.begin(), is_sv.end(), true));
}

std::vector<std::size_t> SVM::support_vectors_per_class() const {
    std::vector<bool> is_sv(observations_, false);
    for (const auto& pair : pair_models_)
        for (const auto idx : pair.support_indices) is_sv[idx] = true;

    std::vector<std::size_t> counts(classes_.size(), 0);
    for (std::size_t i = 0; i < observations_; ++i) {
        if (!is_sv[i]) continue;
        const auto it = std::lower_bound(classes_.begin(), classes_.end(), training_labels_[i]);
        ++counts[static_cast<std::size_t>(it - classes_.begin())];
    }
    return counts;
}

std::string SVM::summary() const {
    std::ostringstream out;
    out << "Call:\nsvm(formula = " << formula_.text() << ", kernel = \"" << kernel_name(options_.kernel)
        << "\", cost = " << format_stat(options_.cost) << ")\n\n";

    out << "Parameters:\n";
    out << "   SVM-Type: C-classification (one-vs-one)\n";
    out << " SVM-Kernel: " << kernel_name(options_.kernel) << "\n";
    out << "       cost: " << format_stat(options_.cost) << "\n";
    out << "      gamma: " << format_stat(effective_gamma_) << "\n";
    if (options_.kernel == SVMKernel::Polynomial || options_.kernel == SVMKernel::Sigmoid)
        out << "     coef.0: " << format_stat(options_.coef0) << "\n";
    if (options_.kernel == SVMKernel::Polynomial) out << "     degree: " << options_.degree << "\n";
    out << "\n";

    const auto per_class = support_vectors_per_class();
    out << "Number of Support Vectors: " << num_support_vectors() << "\n(";
    for (const auto count : per_class) out << " " << count;
    out << " )\n\n";

    out << "Number of Classes: " << classes_.size() << "\n";
    out << "Levels: ";
    for (const auto& c : classes_) out << c << " ";
    out << "\n\n";

    out << "Training accuracy: " << format_stat(training_accuracy() * 100.0, 2) << "%  (" << observations_
        << " observations)\n";

    return out.str();
}

void SVM::print_summary(std::ostream& os) const { os << summary(); }

void SVM::print_summary() const { print_summary(std::cout); }

std::vector<std::string> SVM::predict(const dstruct::DataFrame& newdata) const {
    return predict_detail(newdata).class_label;
}

SVMPrediction SVM::predict_detail(const dstruct::DataFrame& newdata) const {
    design_.validate_categorical_levels(newdata, "SVM::predict");
    auto dm = design_.build_matrix(newdata, /*require_response=*/false);
    for (std::size_t i = 0; i < dm.X.rows(); ++i)
        for (std::size_t j = 0; j < dm.X.cols(); ++j) dm.X(i, j) = (dm.X(i, j) - feature_mean_[j]) / feature_scale_[j];

    const auto compact = classify(dm.X);

    SVMPrediction result;
    result.class_label.assign(newdata.nrows(), "");
    result.votes.assign(newdata.nrows(), {});
    for (std::size_t i = 0; i < dm.used_row_indices.size(); ++i) {
        const auto row          = dm.used_row_indices[i];
        result.class_label[row] = compact.class_label[i];
        result.votes[row]       = compact.votes[i];
    }
    return result;
}

} // namespace datamunge::stats
