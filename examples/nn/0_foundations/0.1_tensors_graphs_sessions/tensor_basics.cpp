// CMD: just run-nn tensor_basics
// Tensor basics: shapes, dtypes, slicing, broadcasting, views.
//
// Port of tflow's tensor_basics. An nn::Tensor is a reference-counted,
// typed, N-dimensional buffer, like tensorflow::Tensor. There is no graph or
// session: the "graph ops" half of the TF example is the same code run
// eagerly. Shape errors are reported when the op runs, as an exception.
//
// String tensors live in linalg::Tensor; nn::Tensor is numeric only
// (bool, int32, int64, float32, float64).

#include <datamunge/nn/nn.hpp>
#include <iostream>
#include <string>
#include <vector>

#include "common/checks.hpp"

namespace nn = datamunge::nn;
using nn::DType;
using nn::Tensor;

namespace {

void Show(const std::string& label, const Tensor& t) { std::cout << "  " << label << ": " << t.str() << "\n"; }

// ---------------------------------------------------------------------------
// 1. Constructing tensors
// ---------------------------------------------------------------------------

void Construction() {
  Header("Constructing tensors");

  // A scalar: rank 0, shape [].
  Tensor scalar = nn::scalar(3.14);
  Show("scalar", scalar);

  // A vector, filled through the raw pointer.
  Tensor vec = nn::empty({4}, DType::Int32);
  for (int i = 0; i < 4; ++i) vec.data_ptr<std::int32_t>()[i] = 10 * i;
  Show("vec", vec);

  // A matrix. Storage is row-major.
  Tensor mat = nn::reshape(nn::arange(0, 6), {2, 3});
  Show("mat", mat);

  // Higher rank. at() reads one element; data_ptr() views any tensor as 1-D.
  Tensor cube = nn::reshape(nn::arange(0, 4, 0.5, DType::Float64), {2, 2, 2});
  std::cout << "  cube(1,0,1) = " << cube.at({1, 0, 1}) << "\n";
  Show("cube", cube);
  check::Near(cube.at({1, 0, 1}), 2.5, 0, "cube element");

  Tensor flags = nn::cast(nn::tensor({1, 0, 1}), DType::Bool);
  Show("flags", flags);

  Header("Shape and dtype metadata");
  std::cout << "  mat.dtype()     = " << nn::dtype_name(mat.dtype()) << "\n"
            << "  mat.shape()     = " << nn::shape_str(mat.shape()) << "\n"
            << "  mat.dim()       = " << mat.dim() << "\n"
            << "  mat.size(1)     = " << mat.size(1) << "\n"
            << "  mat.numel()     = " << mat.numel() << "\n"
            << "  bytes           = " << mat.numel() * static_cast<std::int64_t>(nn::dtype_size(mat.dtype())) << "\n"
            << "  mat.str()       = " << mat.str() << "\n";
  Tensor undefined;  // default-constructed: no buffer
  std::cout << "  default Tensor: " << undefined.str() << "\n";
}

// ---------------------------------------------------------------------------
// 2. Slicing and reshaping without copying
// ---------------------------------------------------------------------------

void Slicing() {
  Header("Slicing (views share storage)");
  Tensor mat = nn::reshape(nn::arange(0, 12), {4, 3});
  Show("mat", mat);

  // Rows [1, 3) of the first dimension. The result shares the buffer, so
  // writes are visible in both (TF Tensor::Slice).
  Tensor rows = nn::slice_axis(mat, 0, 1, 3);
  Show("rows 1..3", rows);
  rows.data_ptr<float>()[0] = -1.f;
  std::cout << "  after writing -1 into the slice, mat(1,0) = " << mat.at({1, 0}) << "\n";
  check::Near(mat.at({1, 0}), -1, 0, "slice shares storage");

  // select() drops the first dimension: a rank-1 row view (TF SubSlice).
  Show("select(mat, 0, 2)", nn::select(mat, 0, 2));

  // Reshape of a packed tensor is a view; clone() makes an independent copy.
  Tensor reshaped = nn::reshape(mat, {3, 4});
  Show("mat viewed as [3,4]", reshaped);
  Tensor copy = mat.clone();
  copy.data_ptr<float>()[0] = 100.f;
  std::cout << "  clone then write 100: copy(0,0)=" << copy.at({0, 0}) << ", mat(0,0)=" << mat.at({0, 0}) << "\n";

  // Transposes are strided views; contiguous() packs them.
  Tensor t = nn::transpose(mat);
  std::cout << "  transpose is contiguous? " << t.is_contiguous() << "; after contiguous(): "
            << t.contiguous().is_contiguous() << "\n";
}

// ---------------------------------------------------------------------------
// 3. Computing with tensors
// ---------------------------------------------------------------------------

void Compute() {
  Header("Computing");
  Tensor a = nn::tensor({{1, 2, 3}, {4, 5, 6}});
  Tensor b = nn::tensor({{1, 0}, {0, 1}, {1, 1}});
  Show("a", a);
  Show("b", b);
  std::cout << "  a.sum()            = " << nn::sum(a).item() << "\n";
  Show("a.sum(axis=1)", nn::sum(a, {1}));
  Show("a*2+1", a * 2.0 + 1.0);
  // Broadcasting is implicit (NumPy rules).
  Show("a + bias (broadcast)", a + nn::tensor({10, 20, 30}));
  Show("a @ b", nn::matmul(a, b));
  std::cout << "  Frobenius norm(a)  = " << nn::norm(a).item() << "\n";
  Show("a @ a^T", nn::matmul(a, a, false, true));
  check::Near(nn::norm(a).item(), std::sqrt(91.0), 1e-5, "Frobenius norm");
}

// ---------------------------------------------------------------------------
// 4. The TF graph ops, run eagerly
// ---------------------------------------------------------------------------

void ShapeOps() {
  Header("Shape ops");
  const Tensor x   = nn::tensor({{1, 2, 3}, {4, 5, 6}});
  const Tensor row = nn::tensor({10, 20, 30});
  const Tensor col = nn::tensor({{1}, {-1}});

  struct Named {
    std::string label;
    Tensor      value;
  };
  const std::vector<Named> results = {
      {"x", x},
      {"Shape(x)", nn::from_vector(std::vector<std::int32_t>(x.shape().begin(), x.shape().end()))},
      {"Reshape(x,[3,2])", nn::reshape(x, {3, 2})},
      {"Reshape(x,[-1])", nn::reshape(x, {-1})},
      {"Transpose(x,[1,0])", nn::transpose(x, {1, 0})},
      {"Slice(x,[0,1],[2,2])", nn::slice(x, {0, 1}, {2, 2})},
      {"StridedSlice(x,[0,0],[2,3],[1,2])", nn::strided_slice(x, {0, 0}, {2, 3}, {1, 2})},
      {"x[1] (select)", nn::select(x, 0, 1)},
      {"ExpandDims(x,0)", nn::expand_dims(x, 0)},
      {"Squeeze(ExpandDims(x,0))", nn::squeeze(nn::expand_dims(x, 0))},
      {"Cast(x*1.7, int32) truncates", nn::cast(x * 1.7, DType::Int32)},
      {"x + row  [2,3]+[3]", x + row},
      {"x * col  [2,3]*[2,1]", x * col},
      {"row + col [3]+[2,1]", row + col},
      {"BroadcastTo(row,[2,3])", nn::broadcast_to(row, {2, 3})},
      {"Concat([x,x],axis=0)", nn::concat({x, x}, 0)},
      {"Stack([x,x])", nn::stack({x, x})},
      {"Range(0,10,3)", nn::arange(0, 10, 3, DType::Int32)},
      {"Fill([2,2],7)", nn::full({2, 2}, 7, DType::Int32)},
  };
  for (const auto& r : results) Show(r.label, r.value);

  check::Require(nn::cast(x * 1.7, DType::Int32).to_vector<int>() == std::vector<int>{1, 3, 5, 6, 8, 10},
                 "Cast truncates toward zero like TF");
  check::Require((row + col).shape() == nn::Shape{2, 3}, "broadcast [3]+[2,1] -> [2,3]");

  Header("Shape mismatches are reported when the op runs");
  const std::string message = check::Throws([&] { (void)(x + nn::tensor({1, 2})); }, "AddV2([2,3],[2])");
  std::cout << "  [2,3] + [2] -> " << message << "\n";
}

}  // namespace

int main() {
  try {
    Construction();
    Slicing();
    Compute();
    ShapeOps();
    std::cout << "\nAll tensor basics checks passed.\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
  return 0;
}
