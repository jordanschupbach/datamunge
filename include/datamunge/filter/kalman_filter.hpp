#pragma once
/// @file kalman_filter.hpp
/// @brief Linear covariance-form filtering and actual forward/RTS smoothing states.

#include <datamunge/filter/detail/matrix_convert.hpp>
#include <datamunge/linalg/dense_matrix.hpp>
#include <datamunge/linalg/lu.hpp>
#include <stdexcept>
#include <vector>

namespace datamunge::filter {

/// @brief Owned state mean and covariance, shared by the native filter-family outputs.
/// Units follow the model; covariance is state-unit products. No input references.
/// Public edits are unchecked and do not mutate the producing filter.
/// @par Example
/// @code
/// const dm::filter::KalmanState s{{1},{{2}}};std::print("{} {}\n",s.x[0],s.P[0][0]); // 1 2
/// @endcode
/// @see KalmanFilter
struct KalmanState {
  std::vector<double> x; ///< Owned state mean, length n.
  std::vector<std::vector<double>>
      P; ///< Owned n-by-n covariance, no exact PSD certificate under rounding.
};

/// @brief Owned actual forward-filter and RTS backward-smoothed trajectories.
/// Equal lengths follow the input sequence. Final smoothed state/covariance equal
/// final filtered values because no future data remain. In an exact valid linear
/// Gaussian model, smoothing reduces conditional covariance; it need not reduce
/// realized error at every time or on every observed sequence. Rounding can impair
/// symmetry/PSD. No retained input views or prefix replay.
/// @par Example
/// @code
/// dm::filter::KalmanFilter k({{1}},{{1}},{{1}},{{1}},{0},{{1}});
/// std::print("{}\n",k.smooth({{1},{2}}).smoothed.size()); // 2
/// @endcode
/// @see KalmanFilter::smooth()
struct SmoothResult {
  std::vector<KalmanState> filtered; ///< Actual causal posterior states from the forward pass.
  std::vector<KalmanState> smoothed; ///< Actual RTS states using the complete sequence.
};

/// @brief Owned fixed-model linear covariance Kalman filter with optional full RTS smoothing.
/// Models x_k=F*x_(k-1)+w and z_k=H*x_k+v with caller-supplied Q/R covariance.
/// Inputs are copied; predict/update mutate this instance, independent copies
/// evolve separately. Concurrent mutation requires synchronization. Shape checks
/// do not establish finite values, covariance PSD, noise independence or valid
/// probability assumptions: those remain caller preconditions. Uses explicit LU
/// inverses and simplified covariance updates, not a Joseph/square-root method.
/// Under a valid linear Gaussian model the mean is conditional MMSE; floating
/// output is approximate. No universal realized-error or numerical PSD guarantee.
/// @par Example
/// @code
/// dm::filter::KalmanFilter k({{1}},{{1}},{{1}},{{3}},{0},{{4}});
/// k.predict();k.update({2});std::print("{}\n",k.state()[0]); // 1.25
/// @endcode
/// @par References
/// - Kalman(1960), A New Approach to Linear Filtering and Prediction Problems,
///   https://doi.org/10.1115/1.3662552.
/// - Rauch,Tung,Striebel(1965), Maximum likelihood estimates of linear dynamic
///   systems, https://doi.org/10.2514/3.3166.
/// @see filter(), smooth(), KalmanState
class KalmanFilter {
 public:
  /// @brief Copy a fixed linear model and initial state, validating matrix dimensions.
  /// No finite/covariance-definiteness checks are added; the caller supplies
  /// statistically valid finite matrices and invertible required innovation/prediction covariances.
  /// O(n*n+m*n+m*m) copied model/state storage for n state and m observation dimensions.
  /// @param F Owned n-by-n transition matrix.
  /// @param H Owned m-by-n observation matrix.
  /// @param Q Owned n-by-n process covariance.
  /// @param R Owned m-by-m observation covariance.
  /// @param x0 Owned initial length-n state mean.
  /// @param P0 Owned initial n-by-n state covariance.
  /// @throws std::invalid_argument On ragged/incompatible matrix shapes.
  /// @throws std::bad_alloc On model/state allocation failure.
  /// @par Example
  /// @code
  /// const dm::filter::KalmanFilter k({{1}},{{1}},{{1}},{{1}},{0},{{2}});
  /// std::print("{}\n",k.covariance()[0][0]); // 2
  /// @endcode
  KalmanFilter(
      std::vector<std::vector<double>> F,
      std::vector<std::vector<double>> H,
      std::vector<std::vector<double>> Q,
      std::vector<std::vector<double>> R,
      std::vector<double>              x0,
      std::vector<std::vector<double>> P0)
      : F_(detail::to_dense(F)),
        H_(detail::to_dense(H)),
        Q_(detail::to_dense(Q)),
        R_(detail::to_dense(R)),
        x_(std::move(x0)),
        P_(detail::to_dense(P0)) {
    const std::size_t n = x_.size();
    if (F_.rows() != n || F_.cols() != n)
      throw std::invalid_argument("KalmanFilter: F must be n x n where n = x0.size()");
    if (Q_.rows() != n || Q_.cols() != n)
      throw std::invalid_argument("KalmanFilter: Q must be n x n");
    if (P_.rows() != n || P_.cols() != n)
      throw std::invalid_argument("KalmanFilter: P0 must be n x n");
    if (H_.cols() != n)
      throw std::invalid_argument("KalmanFilter: H must have n columns");
    if (R_.rows() != H_.rows() || R_.cols() != H_.rows())
      throw std::invalid_argument("KalmanFilter: R must be m x m where m = H.rows()");
  }

  /// @brief Predict x=F*x and P=F*P*F^T+Q once, mutating the current state.
  /// Matrix arithmetic is ordinary floating point; nonfinite results propagate.
  /// O(n^3) dense work/O(n^2) temporaries; no rollback guarantee on exceptions.
  /// @throws std::bad_alloc On temporary allocation failure.
  /// @par Example
  /// @code
  /// dm::filter::KalmanFilter k({{1}},{{1}},{{1}},{{1}},{2},{{4}});k.predict();
  /// std::print("{}\n",k.covariance()[0][0]); // 5
  /// @endcode
  /// @see update()
  void predict() {
    x_ = matvec(F_, x_);
    P_ = F_ * P_ * F_.transpose() + Q_;
  }

  /// @brief Correct from z using innovation covariance, LU gain and simplified posterior covariance.
  /// Computes y=z-H*x,S=H*P*H^T+R,K=P*H^T*S^-1, then x+=K*y and
  /// P=(I-K*H)*P. Requires finite statistically valid inputs/invertible S;
  /// shape validation alone does not ensure those properties. Rounding may
  /// lose symmetry/PSD; not Joseph form. Mutates state without rollback guarantee.
  /// @param z Borrowed length-m observation, unchanged by this call.
  /// @throws std::invalid_argument On observation dimension mismatch.
  /// @throws std::runtime_error On LU failure under its implementation contract.
  /// @throws std::bad_alloc On temporary allocation failure.
  /// @par Example
  /// @code
  /// dm::filter::KalmanFilter k({{1}},{{1}},{{1}},{{3}},{0},{{5}});k.update({2});
  /// std::print("{}\n",k.state()[0]); // 1.25
  /// @endcode
  /// @see predict()
  void update(const std::vector<double>& z) {
    if (z.size() != H_.rows())
      throw std::invalid_argument("KalmanFilter::update: z size must match H.rows()");
    const std::vector<double>         y = subtract(z, matvec(H_, x_));
    const linalg::DenseMatrix<double> S = H_ * P_ * H_.transpose() + R_;
    const linalg::DenseMatrix<double> K = P_ * H_.transpose() * linalg::lu(S).inverse();
    x_                                  = add(x_, matvec(K, y));
    const linalg::DenseMatrix<double> I = linalg::DenseMatrix<double>::identity(x_.size());
    P_                                  = (I - K * H_) * P_;
  }

  /// @brief Copy the current state mean without another filter update.
  /// @return Owned length-n current mean.
  /// @throws std::bad_alloc On copy allocation failure.
  [[nodiscard]] std::vector<double> state() const { return x_; }
  /// @brief Copy the current covariance without recomputing a filter update.
  /// @return Owned n-by-n current covariance.
  /// @throws std::bad_alloc On copy allocation failure.
  [[nodiscard]] std::vector<std::vector<double>> covariance() const {
    return detail::to_vector2d(P_);
  }

  /// @brief Predict then correct every observation in order, owning actual posterior states.
  /// Empty input returns empty and leaves state unchanged. Ends at the final
  /// posterior, so repeated calls continue this instance rather than restarting.
  /// O(T) dense filter steps plus O(T*n^2) owned output; input remains unchanged.
  /// A later invalid measurement can follow already committed predictions/updates;
  /// failure does not roll the whole sequence back.
  /// @param measurements Borrowed ordered length-m observations.
  /// @return Owned actual causal mean/covariance at every completed step.
  /// @throws std::invalid_argument On a measurement dimension mismatch.
  /// @throws std::runtime_error On underlying LU failure.
  /// @throws std::bad_alloc On temporary/output allocation failure.
  /// @par Example
  /// @code
  /// dm::filter::KalmanFilter k({{1}},{{1}},{{1}},{{1}},{0},{{1}});
  /// std::print("{}\n",k.filter({{1},{2}}).size()); // 2
  /// @endcode
  /// @see smooth()
  [[nodiscard]] std::vector<KalmanState> filter(
      const std::vector<std::vector<double>>& measurements) {
    std::vector<KalmanState> result;
    result.reserve(measurements.size());
    for (const auto& z : measurements) {
      predict();
      update(z);
      result.push_back(KalmanState{x_, detail::to_vector2d(P_)});
    }
    return result;
  }

  /// @brief Execute one actual forward pass and an RTS backward pass on the full sequence.
  /// Owns actual prediction/posterior arrays during execution, then returns
  /// causal and smoothed means/covariances. No repeated-prefix runs. Leaves this
  /// filter at its final forward posterior. Empty input returns empty unchanged.
  /// Exact-model conditional covariance reduces; realized error need not decrease
  /// on every sequence. Ordinary matrix arithmetic can lose symmetry/PSD.
  /// O(T) dense steps/backward solves, O(T*n^2) intermediate/output storage.
  /// @param measurements Borrowed full ordered length-m sequence.
  /// @return Owned actual forward and full-sequence smoothed trajectories.
  /// @throws std::invalid_argument On measurement dimension mismatch.
  /// @throws std::runtime_error On underlying innovation/prediction LU failure.
  /// @throws std::bad_alloc On intermediate/output allocation failure.
  /// @par Example
  /// @code
  /// dm::filter::KalmanFilter k({{1}},{{1}},{{1}},{{1}},{0},{{1}});
  /// const auto r=k.smooth({{1},{2}});std::print("{}\n",r.filtered.back().x==r.smoothed.back().x); // true
  /// @endcode
  /// @par References
  /// - Rauch,Tung,Striebel(1965), https://doi.org/10.2514/3.3166.
  /// @see filter(), SmoothResult
  [[nodiscard]] SmoothResult smooth(const std::vector<std::vector<double>>& measurements) {
    const std::size_t n = measurements.size();
    if (n == 0) {
      return {};
    }

    std::vector<std::vector<double>>         x_pred(n), x_filt(n);
    std::vector<linalg::DenseMatrix<double>> P_pred(n), P_filt(n);
    for (std::size_t k = 0; k < n; ++k) {
      predict();
      x_pred[k] = x_;
      P_pred[k] = P_;
      update(measurements[k]);
      x_filt[k] = x_;
      P_filt[k] = P_;
    }

    std::vector<std::vector<double>>         x_smooth(n);
    std::vector<linalg::DenseMatrix<double>> P_smooth(n);
    x_smooth[n - 1] = x_filt[n - 1];
    P_smooth[n - 1] = P_filt[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) {
      const linalg::DenseMatrix<double> C =
          P_filt[i] * F_.transpose() * linalg::lu(P_pred[i + 1]).inverse();
      x_smooth[i] = add(x_filt[i], matvec(C, subtract(x_smooth[i + 1], x_pred[i + 1])));
      P_smooth[i] = P_filt[i] + C * (P_smooth[i + 1] - P_pred[i + 1]) * C.transpose();
    }

    SmoothResult out;
    out.filtered.reserve(n);
    out.smoothed.reserve(n);
    for (std::size_t k = 0; k < n; ++k) {
      out.filtered.push_back(KalmanState{x_filt[k], detail::to_vector2d(P_filt[k])});
      out.smoothed.push_back(KalmanState{x_smooth[k], detail::to_vector2d(P_smooth[k])});
    }
    return out;
  }

 private:
  /// @brief Multiply a dense matrix by a compatible state vector.
  /// @param m Borrowed dense matrix.
  /// @param v Borrowed compatible vector.
  /// @return Owned product vector.
  static std::vector<double> matvec(
      const linalg::DenseMatrix<double>& m,
      const std::vector<double>&         v) {
    return detail::to_vector(m * detail::to_column(v));
  }
  /// @brief Add two equally shaped state vectors elementwise.
  /// @param a Borrowed first vector.
  /// @param b Borrowed equally sized second vector.
  /// @return Owned elementwise sum.
  static std::vector<double> add(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      out[i] = a[i] + b[i];
    return out;
  }
  /// @brief Subtract equally shaped state vectors elementwise.
  /// @param a Borrowed minuend.
  /// @param b Borrowed equally sized subtrahend.
  /// @return Owned elementwise difference.
  static std::vector<double> subtract(const std::vector<double>& a, const std::vector<double>& b) {
    std::vector<double> out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i)
      out[i] = a[i] - b[i];
    return out;
  }

  linalg::DenseMatrix<double> F_; ///< Owned fixed transition n-by-n.
  linalg::DenseMatrix<double> H_; ///< Owned fixed observation m-by-n.
  linalg::DenseMatrix<double> Q_; ///< Owned fixed process covariance n-by-n.
  linalg::DenseMatrix<double> R_; ///< Owned fixed measurement covariance m-by-m.
  std::vector<double>         x_; ///< Actual mutable current state mean.
  linalg::DenseMatrix<double> P_; ///< Actual mutable current covariance.
};

} // namespace datamunge::filter
