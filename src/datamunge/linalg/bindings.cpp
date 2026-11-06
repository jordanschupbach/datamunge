/// @file bindings.cpp
/// @brief Runtime binding expressions dispatching to the canonical C++ matrix API.
#include <datamunge/format/matrix.hpp>
#include <datamunge/linalg/bindings.hpp>
#include <limits>
#include <variant>

namespace datamunge {
/// @brief Shared immutable runtime operation/leaf used by the binding facade.
struct LinalgMatrix::Node {
  /// @brief Native materialised storage alternatives.
  using Storage = std::variant<Mat<>, SpMat<>>;
  /// @brief Runtime expression operations.
  enum class Kind {
    Leaf,      ///< Already materialised native storage.
    Transpose, ///< Logical transpose.
    Product,   ///< Native matrix product.
    Inverse,   ///< Standalone deferred inverse action.
    Solve      ///< Fused coefficient-system solve.
  };
  Kind                           kind   = Kind::Leaf; ///< Operation performed at evaluation.
  std::size_t                    rows   = 0;          ///< Deferred result rows.
  std::size_t                    cols   = 0;          ///< Deferred result columns.
  bool                           sparse = false;      ///< Deferred result storage policy.
  std::shared_ptr<const Storage> storage;             ///< Shared immutable leaf storage.
  std::shared_ptr<const Node>    left;                ///< Owned coefficient or left operand.
  std::shared_ptr<const Node>    right;               ///< Owned right operand or solve RHS.

  /// @brief Wrap native storage without copying coefficients.
  /// @param value Owned dense or CSR storage.
  /// @return Immutable matrix leaf.
  static std::shared_ptr<const Node> leaf(Storage value) {
    auto node = std::make_shared<Node>();
    std::visit(
        [&](const auto& m) {
          node->rows = m.rows();
          node->cols = m.cols();
        },
        value);
    node->sparse  = std::holds_alternative<SpMat<>>(value);
    node->storage = std::make_shared<const Storage>(std::move(value));
    return node;
  }

  /// @brief Evaluate native kernels, preserving CSR products and fusing inverse actions.
  /// @return Shared immutable materialised storage.
  /// @throws std::invalid_argument For invalid solve inputs.
  /// @throws std::runtime_error For singular or rank-deficient solves.
  std::shared_ptr<const Storage> evaluate() const {
    if (kind == Kind::Leaf)
      return storage;
    if (kind == Kind::Transpose) {
      auto a = left->evaluate();
      return std::visit([](const auto& m) { return leaf(m.t().eval())->storage; }, *a);
    }
    if (kind == Kind::Inverse) {
      auto a = left->evaluate();
      return std::visit(
          [](const auto& m) { return leaf(datamunge::solve(m).eval())->storage; }, *a);
    }
    if (kind == Kind::Solve) {
      auto a = left->evaluate(), b = right->evaluate();
      return std::visit(
          [](const auto& x, const auto& y) { return leaf(datamunge::solve(x, y))->storage; },
          *a,
          *b);
    }
    // Treat transpose nodes as logical views, avoiding dense transpose temporaries.
    auto a_node = left, b_node = right;
    bool at = false, bt = false;
    while (a_node->kind == Kind::Transpose) {
      at     = !at;
      a_node = a_node->left;
    }
    while (b_node->kind == Kind::Transpose) {
      bt     = !bt;
      b_node = b_node->left;
    }
    auto a = a_node->evaluate(), b = b_node->evaluate();
    return std::visit(
        [&](const auto& x, const auto& y) {
          return leaf(linalg::matrix_detail::multiply(x, at, y, bt))->storage;
        },
        *a,
        *b);
  }

  /// @brief Build a product, reassociating inverse actions before numerical work.
  /// @param a Left expression.
  /// @param b Right expression.
  /// @return Deferred product or fused solve.
  /// @throws std::invalid_argument For incompatible inner dimensions.
  static std::shared_ptr<const Node> product(
      std::shared_ptr<const Node> a,
      std::shared_ptr<const Node> b) {
    if (a->cols != b->rows)
      throw std::invalid_argument("matrix product: incompatible dimensions");
    auto node  = std::make_shared<Node>();
    node->rows = a->rows;
    node->cols = b->cols;
    if (a->kind == Kind::Inverse || a->kind == Kind::Solve) {
      node->kind  = Kind::Solve;
      node->left  = a->left;
      node->right = a->kind == Kind::Solve ? product(a->right, b) : b;
    } else {
      node->kind   = Kind::Product;
      node->sparse = a->sparse && b->sparse;
      node->left   = std::move(a);
      node->right  = std::move(b);
    }
    return node;
  }
};

LinalgMatrix::LinalgMatrix(
    const std::vector<double>& values,
    std::size_t                rows,
    std::size_t                cols,
    bool                       sparse) {
  if ((cols && rows > std::numeric_limits<std::size_t>::max() / cols) ||
      values.size() != rows * cols)
    throw std::invalid_argument("matrix values must match rows * cols");
  if (sparse) {
    linalg::SparseCSR<double> m;
    m.nrows = rows;
    m.ncols = cols;
    m.row_ptr.push_back(0);
    for (std::size_t i = 0; i < rows; ++i) {
      for (std::size_t j = 0; j < cols; ++j)
        if (values[i * cols + j] != 0) {
          m.col_idx.push_back(j);
          m.values.push_back(values[i * cols + j]);
        }
      m.row_ptr.push_back(m.values.size());
    }
    node_ = Node::leaf(SpMat<>(std::move(m)));
  } else {
    Mat<> m(rows, cols);
    for (std::size_t i = 0; i < rows; ++i)
      for (std::size_t j = 0; j < cols; ++j)
        m.set(i, j, values[i * cols + j]);
    node_ = Node::leaf(std::move(m));
  }
}
LinalgMatrix::LinalgMatrix(std::shared_ptr<const Node> node) : node_(std::move(node)) {}
std::size_t LinalgMatrix::rows() const {
  return node_->rows;
}
std::size_t LinalgMatrix::cols() const {
  return node_->cols;
}
bool LinalgMatrix::is_sparse() const {
  return node_->sparse;
}
LinalgMatrix LinalgMatrix::transpose() const {
  auto node    = std::make_shared<Node>();
  node->kind   = Node::Kind::Transpose;
  node->rows   = cols();
  node->cols   = rows();
  node->sparse = is_sparse();
  node->left   = node_;
  return LinalgMatrix(node);
}
LinalgMatrix LinalgMatrix::multiply(const LinalgMatrix& other) const {
  return LinalgMatrix(Node::product(node_, other.node_));
}
LinalgMatrix LinalgMatrix::inverse_action() const {
  if (rows() != cols())
    throw std::invalid_argument("inverse action requires a square matrix");
  auto node  = std::make_shared<Node>();
  node->kind = Node::Kind::Inverse;
  node->rows = rows();
  node->cols = cols();
  node->left = node_;
  return LinalgMatrix(node);
}
LinalgMatrix LinalgMatrix::solve(const LinalgMatrix& rhs) const {
  auto node   = std::make_shared<Node>();
  node->kind  = Node::Kind::Solve;
  node->rows  = cols();
  node->cols  = rhs.cols();
  node->left  = node_;
  node->right = rhs.node_;
  return LinalgMatrix(node).eval();
}
LinalgMatrix LinalgMatrix::eval() const {
  auto node     = std::make_shared<Node>();
  node->rows    = rows();
  node->cols    = cols();
  node->sparse  = is_sparse();
  node->storage = node_->evaluate();
  return LinalgMatrix(node);
}
std::vector<double> LinalgMatrix::values() const {
  const auto          storage = node_->evaluate();
  std::vector<double> result;
  result.reserve(rows() * cols());
  std::visit(
      [&](const auto& m) {
        for (std::size_t i = 0; i < rows(); ++i)
          for (std::size_t j = 0; j < cols(); ++j)
            result.push_back(m(i, j));
      },
      *storage);
  return result;
}
std::string LinalgMatrix::text() const {
  return std::visit([](const auto& m) { return std::format("{}", m); }, *node_->evaluate());
}
} // namespace datamunge
