#include <datamunge/stats/cnn_classifier.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>

namespace datamunge::stats {

namespace {

std::string format_stat(double v, int precision = 4) {
    if (std::isnan(v)) return "NaN";
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(precision) << v;
    return oss.str();
}

// Everything a backward pass needs from a forward pass, feature maps stored row-major and flat.
struct Cache {
    std::vector<std::vector<double>> conv_pre; // [F][Hc*Wc]
    std::vector<std::vector<std::size_t>> pool_arg; // [F][Hp*Wp] -> index into the Hc*Wc conv map
    std::vector<double> flat;                       // F*Hp*Wp (the pooled activations)
    std::vector<double> h_pre, h;                   // dense hidden pre-activation and activation
    std::vector<double> probs;                      // K softmax outputs
};

void softmax_inplace(std::vector<double>& v) {
    const double mx = *std::max_element(v.begin(), v.end());
    double sum = 0.0;
    for (double& x : v) { x = std::exp(x - mx); sum += x; }
    for (double& x : v) x /= sum;
}

Cache forward(const linalg::DenseMatrix<double>& img, const std::vector<linalg::DenseMatrix<double>>& filters,
              const std::vector<double>& cbias, const linalg::DenseMatrix<double>& W1, const std::vector<double>& b1,
              const linalg::DenseMatrix<double>& W2, const std::vector<double>& b2, std::size_t k, std::size_t pool,
              std::size_t Hc, std::size_t Wc, std::size_t Hp, std::size_t Wp) {
    const std::size_t F = filters.size();
    Cache c;
    c.conv_pre.assign(F, std::vector<double>(Hc * Wc, 0.0));
    c.pool_arg.assign(F, std::vector<std::size_t>(Hp * Wp, 0));
    c.flat.assign(F * Hp * Wp, 0.0);

    for (std::size_t f = 0; f < F; ++f) {
        for (std::size_t i = 0; i < Hc; ++i)
            for (std::size_t j = 0; j < Wc; ++j) {
                double s = cbias[f];
                for (std::size_t a = 0; a < k; ++a)
                    for (std::size_t b = 0; b < k; ++b) s += img(i + a, j + b) * filters[f](a, b);
                c.conv_pre[f][i * Wc + j] = s;
            }
        for (std::size_t pi = 0; pi < Hp; ++pi)
            for (std::size_t pj = 0; pj < Wp; ++pj) {
                double best = -std::numeric_limits<double>::infinity();
                std::size_t best_idx = 0;
                for (std::size_t a = 0; a < pool; ++a)
                    for (std::size_t b = 0; b < pool; ++b) {
                        const std::size_t idx = (pi * pool + a) * Wc + (pj * pool + b);
                        const double relu = std::max(0.0, c.conv_pre[f][idx]);
                        if (relu > best) { best = relu; best_idx = idx; }
                    }
                c.pool_arg[f][pi * Wp + pj] = best_idx;
                c.flat[f * (Hp * Wp) + pi * Wp + pj] = best;
            }
    }

    const std::size_t Hh = W1.rows(), flat = c.flat.size();
    c.h_pre.assign(Hh, 0.0);
    c.h.assign(Hh, 0.0);
    for (std::size_t o = 0; o < Hh; ++o) {
        double s = b1[o];
        for (std::size_t i = 0; i < flat; ++i) s += W1(o, i) * c.flat[i];
        c.h_pre[o] = s;
        c.h[o] = std::max(0.0, s);
    }
    const std::size_t K = W2.rows();
    c.probs.assign(K, 0.0);
    for (std::size_t o = 0; o < K; ++o) {
        double s = b2[o];
        for (std::size_t i = 0; i < Hh; ++i) s += W2(o, i) * c.h[i];
        c.probs[o] = s;
    }
    softmax_inplace(c.probs);
    return c;
}

} // namespace

CNNClassifier::CNNClassifier(const std::vector<linalg::DenseMatrix<double>>& images,
                             const std::vector<std::string>& labels, CNNClassifierOptions options)
    : options_(options) {
    if (images.empty()) throw std::invalid_argument("CNNClassifier: no images");
    if (images.size() != labels.size())
        throw std::invalid_argument("CNNClassifier: images and labels must have equal length");
    if (options_.kernel_size == 0 || options_.pool_size == 0)
        throw std::invalid_argument("CNNClassifier: kernel_size and pool_size must be positive");
    if (options_.dense_hidden == 0) throw std::invalid_argument("CNNClassifier: dense_hidden must be at least 1");
    if (options_.max_epochs == 0) throw std::invalid_argument("CNNClassifier: max_epochs must be at least 1");
    if (options_.learning_rate <= 0.0) throw std::invalid_argument("CNNClassifier: learning_rate must be positive");
    training_labels_ = labels;
    fit(images);
}

std::size_t CNNClassifier::n_parameters() const {
    std::size_t total = filters_.size() * options_.kernel_size * options_.kernel_size + conv_bias_.size();
    total += W1_.rows() * W1_.cols() + b1_.size();
    total += W2_.rows() * W2_.cols() + b2_.size();
    return total;
}

std::vector<double> CNNClassifier::forward_probs(const linalg::DenseMatrix<double>& image) const {
    return forward(image, filters_, conv_bias_, W1_, b1_, W2_, b2_, options_.kernel_size, options_.pool_size, conv_h_,
                   conv_w_, pool_h_, pool_w_)
        .probs;
}

void CNNClassifier::fit(const std::vector<linalg::DenseMatrix<double>>& images) {
    const std::size_t n = images.size();
    height_ = images[0].rows();
    width_  = images[0].cols();
    for (const auto& im : images)
        if (im.rows() != height_ || im.cols() != width_)
            throw std::invalid_argument("CNNClassifier: all images must have the same dimensions");

    const std::size_t k = options_.kernel_size, pool = options_.pool_size, F = options_.n_filters;
    if (k > height_ || k > width_) throw std::invalid_argument("CNNClassifier: kernel is larger than the image");
    conv_h_ = height_ - k + 1;
    conv_w_ = width_ - k + 1;
    if (pool > conv_h_ || pool > conv_w_)
        throw std::invalid_argument("CNNClassifier: pool_size is larger than the convolved map");
    pool_h_  = conv_h_ / pool;
    pool_w_  = conv_w_ / pool;
    flat_dim_ = F * pool_h_ * pool_w_;
    observations_ = n;

    classes_.clear();
    for (const auto& label : training_labels_)
        if (std::find(classes_.begin(), classes_.end(), label) == classes_.end()) classes_.push_back(label);
    std::sort(classes_.begin(), classes_.end());
    if (classes_.size() < 2) throw std::invalid_argument("CNNClassifier: need at least 2 distinct classes");
    const std::size_t K = classes_.size();
    const std::size_t Hh = options_.dense_hidden;

    std::vector<std::size_t> target(n);
    for (std::size_t i = 0; i < n; ++i)
        target[i] = static_cast<std::size_t>(
            std::lower_bound(classes_.begin(), classes_.end(), training_labels_[i]) - classes_.begin());

    // He initialization.
    std::mt19937_64 rng(options_.seed);
    auto init_matrix = [&](std::size_t rows, std::size_t cols, std::size_t fan_in) {
        std::normal_distribution<double> nd(0.0, std::sqrt(2.0 / static_cast<double>(fan_in)));
        linalg::DenseMatrix<double> M(rows, cols, 0.0);
        for (std::size_t r = 0; r < rows; ++r)
            for (std::size_t c = 0; c < cols; ++c) M(r, c) = nd(rng);
        return M;
    };
    filters_.clear();
    conv_bias_.assign(F, 0.0);
    for (std::size_t f = 0; f < F; ++f) filters_.push_back(init_matrix(k, k, k * k));
    W1_ = init_matrix(Hh, flat_dim_, flat_dim_);
    b1_.assign(Hh, 0.0);
    W2_ = init_matrix(K, Hh, Hh);
    b2_.assign(K, 0.0);

    const std::size_t batch = std::min(options_.batch_size == 0 ? n : options_.batch_size, n);
    std::vector<std::size_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    loss_curve_.assign(options_.max_epochs, 0.0);

    for (std::size_t epoch = 0; epoch < options_.max_epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), rng);
        for (std::size_t start = 0; start < n; start += batch) {
            const std::size_t end = std::min(start + batch, n);
            const double inv = 1.0 / static_cast<double>(end - start);

            std::vector<linalg::DenseMatrix<double>> gF;
            for (std::size_t f = 0; f < F; ++f) gF.emplace_back(k, k, 0.0);
            std::vector<double> gcb(F, 0.0), gb1(Hh, 0.0), gb2(K, 0.0);
            linalg::DenseMatrix<double> gW1(Hh, flat_dim_, 0.0), gW2(K, Hh, 0.0);

            for (std::size_t bi = start; bi < end; ++bi) {
                const std::size_t idx = order[bi];
                const auto& img = images[idx];
                const Cache c = forward(img, filters_, conv_bias_, W1_, b1_, W2_, b2_, k, pool, conv_h_, conv_w_,
                                        pool_h_, pool_w_);

                std::vector<double> delta2 = c.probs;
                delta2[target[idx]] -= 1.0;
                for (std::size_t o = 0; o < K; ++o) {
                    gb2[o] += delta2[o];
                    for (std::size_t i = 0; i < Hh; ++i) gW2(o, i) += delta2[o] * c.h[i];
                }
                std::vector<double> delta_h(Hh, 0.0);
                for (std::size_t i = 0; i < Hh; ++i) {
                    double s = 0.0;
                    for (std::size_t o = 0; o < K; ++o) s += W2_(o, i) * delta2[o];
                    delta_h[i] = (c.h_pre[i] > 0.0) ? s : 0.0;
                }
                std::vector<double> delta_flat(flat_dim_, 0.0);
                for (std::size_t o = 0; o < Hh; ++o) {
                    gb1[o] += delta_h[o];
                    for (std::size_t i = 0; i < flat_dim_; ++i) {
                        gW1(o, i) += delta_h[o] * c.flat[i];
                        delta_flat[i] += W1_(o, i) * delta_h[o];
                    }
                }

                // Route pooled gradients back to the argmax positions, through ReLU, into the filters.
                for (std::size_t f = 0; f < F; ++f) {
                    std::vector<double> delta_conv(conv_h_ * conv_w_, 0.0);
                    for (std::size_t pi = 0; pi < pool_h_; ++pi)
                        for (std::size_t pj = 0; pj < pool_w_; ++pj) {
                            const std::size_t src = c.pool_arg[f][pi * pool_w_ + pj];
                            const double g = delta_flat[f * (pool_h_ * pool_w_) + pi * pool_w_ + pj];
                            delta_conv[src] += (c.conv_pre[f][src] > 0.0) ? g : 0.0;
                        }
                    for (std::size_t i = 0; i < conv_h_; ++i)
                        for (std::size_t j = 0; j < conv_w_; ++j) {
                            const double dc = delta_conv[i * conv_w_ + j];
                            if (dc == 0.0) continue;
                            gcb[f] += dc;
                            for (std::size_t a = 0; a < k; ++a)
                                for (std::size_t b = 0; b < k; ++b) gF[f](a, b) += img(i + a, j + b) * dc;
                        }
                }
            }

            // SGD update (batch-averaged gradients + L2 on weights).
            for (std::size_t f = 0; f < F; ++f) {
                for (std::size_t a = 0; a < k; ++a)
                    for (std::size_t b = 0; b < k; ++b)
                        filters_[f](a, b) -= options_.learning_rate * (gF[f](a, b) * inv + options_.l2 * filters_[f](a, b));
                conv_bias_[f] -= options_.learning_rate * gcb[f] * inv;
            }
            for (std::size_t o = 0; o < Hh; ++o) {
                for (std::size_t i = 0; i < flat_dim_; ++i)
                    W1_(o, i) -= options_.learning_rate * (gW1(o, i) * inv + options_.l2 * W1_(o, i));
                b1_[o] -= options_.learning_rate * gb1[o] * inv;
            }
            for (std::size_t o = 0; o < K; ++o) {
                for (std::size_t i = 0; i < Hh; ++i)
                    W2_(o, i) -= options_.learning_rate * (gW2(o, i) * inv + options_.l2 * W2_(o, i));
                b2_[o] -= options_.learning_rate * gb2[o] * inv;
            }
        }

        double loss = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            const auto probs = forward_probs(images[i]);
            loss -= std::log(std::max(probs[target[i]], 1e-12));
        }
        loss_curve_[epoch] = loss / static_cast<double>(n);
    }

    fitted_classes_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto probs = forward_probs(images[i]);
        fitted_classes_[i] =
            classes_[static_cast<std::size_t>(std::max_element(probs.begin(), probs.end()) - probs.begin())];
    }
}

double CNNClassifier::training_accuracy() const {
    std::size_t correct = 0;
    for (std::size_t i = 0; i < fitted_classes_.size(); ++i)
        if (fitted_classes_[i] == training_labels_[i]) ++correct;
    return fitted_classes_.empty() ? 0.0 : static_cast<double>(correct) / static_cast<double>(fitted_classes_.size());
}

linalg::DenseMatrix<double> CNNClassifier::confusion_matrix() const {
    const std::size_t k = classes_.size();
    linalg::DenseMatrix<double> matrix(k, k, 0.0);
    for (std::size_t i = 0; i < training_labels_.size(); ++i) {
        const auto a = std::lower_bound(classes_.begin(), classes_.end(), training_labels_[i]);
        const auto pidx = std::lower_bound(classes_.begin(), classes_.end(), fitted_classes_[i]);
        matrix(static_cast<std::size_t>(a - classes_.begin()), static_cast<std::size_t>(pidx - classes_.begin())) +=
            1.0;
    }
    return matrix;
}

CNNClassifierPrediction CNNClassifier::predict_detail(const std::vector<linalg::DenseMatrix<double>>& images) const {
    CNNClassifierPrediction result;
    result.class_label.reserve(images.size());
    result.probabilities.reserve(images.size());
    for (const auto& img : images) {
        const auto probs = forward_probs(img);
        const auto best = static_cast<std::size_t>(std::max_element(probs.begin(), probs.end()) - probs.begin());
        result.class_label.push_back(classes_[best]);
        result.probabilities.push_back(probs);
    }
    return result;
}

std::vector<std::string> CNNClassifier::predict(const std::vector<linalg::DenseMatrix<double>>& images) const {
    return predict_detail(images).class_label;
}

double CNNClassifier::accuracy(const std::vector<linalg::DenseMatrix<double>>& images,
                               const std::vector<std::string>& labels) const {
    const auto preds = predict(images);
    std::size_t correct = 0;
    for (std::size_t i = 0; i < preds.size(); ++i)
        if (preds[i] == labels[i]) ++correct;
    return preds.empty() ? 0.0 : static_cast<double>(correct) / static_cast<double>(preds.size());
}

linalg::DenseMatrix<double> CNNClassifier::relevance(const linalg::DenseMatrix<double>& image,
                                                     int target_class) const {
    const std::size_t k = options_.kernel_size, pool = options_.pool_size, F = filters_.size();
    const std::size_t K = classes_.size(), Hh = W1_.rows();
    const Cache c = forward(image, filters_, conv_bias_, W1_, b1_, W2_, b2_, k, pool, conv_h_, conv_w_, pool_h_, pool_w_);
    const std::size_t tc =
        (target_class >= 0 && static_cast<std::size_t>(target_class) < K)
            ? static_cast<std::size_t>(target_class)
            : static_cast<std::size_t>(std::max_element(c.probs.begin(), c.probs.end()) - c.probs.begin());
    constexpr double eps = 1e-9;

    // Initialize relevance at the output with the target class's (pre-softmax) score.
    double s_tc = b2_[tc];
    for (std::size_t i = 0; i < Hh; ++i) s_tc += W2_(tc, i) * c.h[i];

    // Output -> dense hidden (z+ rule: redistribute along positive weight * activation).
    std::vector<double> R_h(Hh, 0.0);
    {
        double denom = eps;
        for (std::size_t i = 0; i < Hh; ++i) denom += std::max(0.0, W2_(tc, i)) * c.h[i];
        for (std::size_t i = 0; i < Hh; ++i) R_h[i] = std::max(0.0, W2_(tc, i)) * c.h[i] / denom * s_tc;
    }

    // Dense hidden -> flattened feature maps (z+ rule).
    std::vector<double> R_flat(flat_dim_, 0.0);
    for (std::size_t o = 0; o < Hh; ++o) {
        double denom = eps;
        for (std::size_t i = 0; i < flat_dim_; ++i) denom += std::max(0.0, W1_(o, i)) * c.flat[i];
        const double scale = R_h[o] / denom;
        for (std::size_t i = 0; i < flat_dim_; ++i) R_flat[i] += std::max(0.0, W1_(o, i)) * c.flat[i] * scale;
    }

    // Unpool (winner-take-all): route each pooled cell's relevance to its argmax conv position.
    std::vector<std::vector<double>> R_conv(F, std::vector<double>(conv_h_ * conv_w_, 0.0));
    for (std::size_t f = 0; f < F; ++f)
        for (std::size_t pq = 0; pq < pool_h_ * pool_w_; ++pq)
            R_conv[f][c.pool_arg[f][pq]] += R_flat[f * (pool_h_ * pool_w_) + pq];

    // Convolution -> input pixels (z+ rule; pixels are non-negative).
    linalg::DenseMatrix<double> R_input(height_, width_, 0.0);
    for (std::size_t f = 0; f < F; ++f)
        for (std::size_t i = 0; i < conv_h_; ++i)
            for (std::size_t j = 0; j < conv_w_; ++j) {
                const double R = R_conv[f][i * conv_w_ + j];
                if (R == 0.0) continue;
                double denom = eps;
                for (std::size_t a = 0; a < k; ++a)
                    for (std::size_t b = 0; b < k; ++b) denom += std::max(0.0, filters_[f](a, b)) * image(i + a, j + b);
                const double scale = R / denom;
                for (std::size_t a = 0; a < k; ++a)
                    for (std::size_t b = 0; b < k; ++b)
                        R_input(i + a, j + b) += std::max(0.0, filters_[f](a, b)) * image(i + a, j + b) * scale;
            }
    return R_input;
}

std::string CNNClassifier::summary() const {
    std::ostringstream oss;
    print_summary(oss);
    return oss.str();
}

void CNNClassifier::print_summary() const { print_summary(std::cout); }

void CNNClassifier::print_summary(std::ostream& os) const {
    os << "Convolutional Neural Network Classifier\n";
    os << "Input: " << height_ << "x" << width_ << " grayscale, " << observations_ << " images, " << classes_.size()
       << " classes\n";
    os << "Architecture: conv(" << options_.n_filters << " x " << options_.kernel_size << "x" << options_.kernel_size
       << ") -> relu -> maxpool(" << options_.pool_size << ") -> flatten(" << flat_dim_ << ") -> dense("
       << options_.dense_hidden << ") -> softmax(" << classes_.size() << ")\n";
    os << "Trainable parameters: " << n_parameters() << "\n";
    os << "Epochs: " << options_.max_epochs << ", learning rate: " << format_stat(options_.learning_rate) << "\n";
    if (!loss_curve_.empty()) os << "Final training cross-entropy: " << format_stat(loss_curve_.back()) << "\n";
    os << "Training accuracy: " << format_stat(training_accuracy()) << "\n";
}

plot::RPlot CNNClassifier::plot_loss_curve() const {
    std::vector<double> epochs(loss_curve_.size());
    for (std::size_t i = 0; i < loss_curve_.size(); ++i) epochs[i] = static_cast<double>(i + 1);
    auto plot = plot::RPlot::create();
    plot.line(epochs, loss_curve_, "training cross-entropy", {37, 99, 235}, 2.0);
    plot.title("CNN Training Loss Curve").x_label("epoch").y_label("mean cross-entropy");
    return plot;
}

} // namespace datamunge::stats
