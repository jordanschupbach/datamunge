#include <datamunge/algorithms/cannon.hpp>
/// @file cannon.cpp
/// @brief Shared numeric Cannon kernel, checked shapes and actual transported ownership.
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Private checked products and optional-recording simulation kernel.
namespace cannon_detail {
/// @brief Multiply size/count terms before allocating or entering numeric loops.
/// @param a First nonnegative factor.
/// @param b Second nonnegative factor.
/// @return Exact size_t product.
/// @throws std::length_error Product exceeds size_t.
std::size_t product(std::size_t a,std::size_t b) {
  if(a && b>std::numeric_limits<std::size_t>::max()/a)
    throw std::length_error("Cannon: shape/work product exceeds size_t");
  return a*b;
}
/// @brief Add two residues modulo a positive modulus without overflowing their sum.
/// @param a First residue below modulus.
/// @param b Second residue below modulus.
/// @param modulus Positive modulus.
/// @return Sum modulo modulus.
std::size_t cyclic_add(std::size_t a,std::size_t b,std::size_t modulus) {
  return a>=modulus-b?a-(modulus-b):a+b;
}
/// @brief Simulate actual skewed numeric buffers and optionally carry original block ids.
/// @tparam Record Whether source ownership and per-round C snapshots are retained.
/// @param A Borrowed row-major left input.
/// @param B Borrowed row-major right input, possibly aliasing A.
/// @param dimension Nonnegative matrix side, validated before conversion.
/// @param mesh_dimension Valid mesh side, checked by cannon_work().
/// @param budget Whole-call block-update cap; zero is unlimited.
/// @return Owned product/work, with provenance only in recording mode.
/// @throws std::invalid_argument Shape/divisibility or input-length mismatch.
/// @throws std::length_error Size/count overflow or block-update cap exceeded.
template<bool Record>
CannonTrace run(const std::vector<double>& A,const std::vector<double>& B,
    int dimension,int mesh_dimension,std::size_t budget) {
  const auto expected=cannon_work(dimension,mesh_dimension);
  const auto n=static_cast<std::size_t>(dimension),q=static_cast<std::size_t>(mesh_dimension);
  const auto count=product(n,n);
  if(A.size()!=count || B.size()!=count) throw std::invalid_argument("Cannon: input length mismatch");
  if(budget && expected.block_updates>budget) throw std::length_error("Cannon: block-update budget exceeded");
  CannonTrace out;out.dimension=dimension;out.mesh_dimension=mesh_dimension;
  if(!n) return out;
  const auto b=n/q,cells=product(q,q);out.block_dimension=static_cast<int>(b);
  out.output.assign(count,0);
  std::vector<double> a(count),v(count),next_a(count),next_b(count);
  std::vector<std::size_t> owner_a,owner_b,next_owner_a,next_owner_b;
  if constexpr(Record) {
    owner_a.resize(cells);owner_b.resize(cells);next_owner_a.resize(cells);next_owner_b.resize(cells);
    out.rounds.reserve(q);
  }
  for(std::size_t i=0;i<q;++i) for(std::size_t j=0;j<q;++j) {
    const auto k=cyclic_add(i,j,q),cell=i*q+j;
    if constexpr(Record) {owner_a[cell]=i*q+k;owner_b[cell]=k*q+j;}
    for(std::size_t r=0;r<b;++r) for(std::size_t c=0;c<b;++c) {
      const auto destination=(i*b+r)*n+j*b+c;
      a[destination]=A[(i*b+r)*n+k*b+c];++out.work.skew_copies;
      v[destination]=B[(k*b+r)*n+j*b+c];++out.work.skew_copies;
    }
  }
  for(std::size_t step=0;step<q;++step) {
    for(std::size_t i=0;i<q;++i) for(std::size_t j=0;j<q;++j) {
      ++out.work.block_updates;
      for(std::size_t r=0;r<b;++r) for(std::size_t k=0;k<b;++k) for(std::size_t c=0;c<b;++c) {
        out.output[(i*b+r)*n+j*b+c]+=a[(i*b+r)*n+j*b+k]*v[(i*b+k)*n+j*b+c];
        ++out.work.multiply_adds;
      }
    }
    if constexpr(Record) out.rounds.push_back({step,owner_a,owner_b,out.output});
    for(std::size_t i=0;i<q;++i) for(std::size_t j=0;j<q;++j) {
      const auto right=cyclic_add(j,q==1?0:1,q),down=cyclic_add(i,q==1?0:1,q),cell=i*q+j;
      if constexpr(Record) {next_owner_a[cell]=owner_a[i*q+right];next_owner_b[cell]=owner_b[down*q+j];}
      for(std::size_t r=0;r<b;++r) for(std::size_t c=0;c<b;++c) {
        const auto destination=(i*b+r)*n+j*b+c;
        next_a[destination]=a[(i*b+r)*n+right*b+c];++out.work.shift_copies;
        next_b[destination]=v[(down*b+r)*n+j*b+c];++out.work.shift_copies;
      }
    }
    a.swap(next_a);v.swap(next_b);
    if constexpr(Record) {owner_a.swap(next_owner_a);owner_b.swap(next_owner_b);}
  }
  if constexpr(Record) {out.final_a_sources=std::move(owner_a);out.final_b_sources=std::move(owner_b);}
  return out;
}
} // namespace cannon_detail
CannonWork cannon_work(int n,int mesh_dimension) {
  if(n<0 || mesh_dimension<0 || (!n && mesh_dimension) || (n && (!mesh_dimension || n%mesh_dimension)))
    throw std::invalid_argument("Cannon: invalid matrix/mesh dimensions");
  if(!n) return {};
  const auto size=static_cast<std::size_t>(n),q=static_cast<std::size_t>(mesh_dimension);
  const auto square=cannon_detail::product(size,size),cells=cannon_detail::product(q,q);
  const auto skew=cannon_detail::product(2,square);
  return {cannon_detail::product(cells,q),cannon_detail::product(square,size),skew,cannon_detail::product(skew,q)};
}
std::vector<double> cannon_matmul(const std::vector<double>& A,const std::vector<double>& B,int n) {
  return cannon_detail::run<false>(A,B,n,n,0).output;
}
std::vector<double> cannon_block_matmul(const std::vector<double>& A,const std::vector<double>& B,
    int n,int mesh_dimension,std::size_t max_block_updates) {
  return cannon_detail::run<false>(A,B,n,mesh_dimension,max_block_updates).output;
}
CannonTrace cannon_matmul_trace(const std::vector<double>& A,const std::vector<double>& B,
    int n,int mesh_dimension,std::size_t max_block_updates) {
  return cannon_detail::run<true>(A,B,n,mesh_dimension,max_block_updates);
}
} // namespace datamunge::algorithms
