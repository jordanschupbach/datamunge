#include <datamunge/algorithms/dixon.hpp>

/// @file dixon.cpp
/// @brief Shared exact modular square scan and dynamic parity-elimination kernel.

#include <algorithm>
#include <bit>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Internal exact arithmetic and bounded capture; no shared mutable state.
namespace {
/// @brief Checked logical cell/counter addition.
/// @param a Current count.
/// @param b Increment.
/// @return Exact sum.
/// @throws std::length_error On size_t overflow.
std::size_t add_count(std::size_t a, std::size_t b) {
    if (b > std::numeric_limits<std::size_t>::max()-a)
        throw std::length_error("Dixon count overflow");
    return a+b;
}
/// @brief Check an optional logical capture cap before storing cells.
/// @param used Prior logical cells.
/// @param extra New logical cells.
/// @param cap Zero disables only the optional cap.
/// @return Exact new count.
/// @throws std::length_error On overflow or exceeded nonzero cap.
std::size_t capture(std::size_t used, std::size_t extra, std::size_t cap) {
    const auto next=add_count(used,extra);
    if (cap && next>cap) throw std::length_error("Dixon capture cap exceeded");
    return next;
}
/// @brief Compute floor square root of a uint64_t by exact bounded binary search.
/// @param n Nonnegative integer.
/// @return Unique r with r*r<=n<(r+1)*(r+1).
std::uint64_t root(std::uint64_t n) {
    std::uint64_t low=0,high=std::uint64_t{1}<<32;
    while (high-low>1) {
        const auto mid=low+(high-low)/2;
        if (static_cast<__uint128_t>(mid)*mid<=n) low=mid;
        else high=mid;
    }
    return low;
}
/// @brief Validate the full input domain and requested scan endpoint before allocating.
/// @param n Modulus.
/// @param k Prime count.
/// @param candidates Requested window, including candidates past an early relation target.
/// @return First scanned integer floor(sqrt(n))+1.
/// @throws std::invalid_argument For n<2 or k outside [1,4096].
/// @throws std::overflow_error If the requested final x is not representable.
std::uint64_t validate(std::uint64_t n, int k, std::size_t candidates) {
    if (n<2 || k<1 || k>4096) throw std::invalid_argument("Dixon needs n>=2 and 1<=base_size<=4096");
    const auto start=root(n)+1;
    if (candidates && static_cast<__uint128_t>(start)+candidates-1>std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error("Dixon candidate endpoint overflow");
    return start;
}
/// @brief Generate the first k primes using already discovered divisors.
/// @param k Validated count at most 4096.
/// @return Owned increasing primes.
std::vector<std::uint64_t> primes(std::size_t k) {
    std::vector<std::uint64_t> result;
    for (std::uint64_t c=2;result.size()<k;++c) {
        bool prime=true;
        for (auto p:result) {
            if (p>c/p) break;
            if (c%p==0) {prime=false;break;}
        }
        if (prime) result.push_back(c);
    }
    return result;
}
/// @brief Multiply modulo a nonzero uint64_t modulus without overflow.
/// @param a First factor.
/// @param b Second factor.
/// @param n Positive modulus.
/// @return Exact product residue.
std::uint64_t multiply(std::uint64_t a, std::uint64_t b, std::uint64_t n) {
    return static_cast<std::uint64_t>(static_cast<__uint128_t>(a)*b%n);
}
/// @brief Binary modular power with exact widened products.
/// @param a Base.
/// @param e Nonnegative exponent.
/// @param n Modulus at least two.
/// @return a to e modulo n.
std::uint64_t power(std::uint64_t a, std::uint64_t e, std::uint64_t n) {
    std::uint64_t r=1;
    while (e) {if(e&1)r=multiply(r,a,n);a=multiply(a,a,n);e>>=1;}
    return r;
}
/// @brief Shared collection loop, optionally stopping at a relation target.
/// @param n Validated modulus.
/// @param k Validated base count.
/// @param start Validated first candidate.
/// @param candidates Validated window length.
/// @param target Zero scans the full window; otherwise stop after target smooth residues.
/// @param cap Logical scalar-cell capture cap, zero disables.
/// @param t Receives collection, successful division count and captured cells.
/// @throws std::length_error On capture/count overflow; allocation can also fail.
void collect(std::uint64_t n, std::size_t k, std::uint64_t start,
             std::size_t candidates, std::size_t target, std::size_t cap, DixonTrace& t) {
    t.captured_cells=capture(0,k,cap);
    auto& c=t.collection;c.n=n;c.start=start;c.base=primes(k);
    for (std::size_t offset=0;offset<candidates && (!target || c.relations.size()<target);++offset) {
        const auto x=start+static_cast<std::uint64_t>(offset),q=multiply(x%n,x%n,n);
        ++c.scanned;
        if (!q) {++c.zero_residues;continue;}
        auto r=q;std::vector<unsigned> exponents(k,0);
        for (std::size_t i=0;i<k;++i) {
            while (r%c.base[i]==0) {
                r/=c.base[i];++exponents[i];
                t.successful_divisions=add_count(t.successful_divisions,1);
            }
        }
        if (r!=1) continue;
        t.captured_cells=capture(t.captured_cells,add_count(k,3),cap);
        c.relations.push_back({x,q,offset,std::move(exponents)});
    }
}
/// @brief Dynamic packed GF(2) vector, independently sized for primes or relations.
using Bits=std::vector<std::uint64_t>;
/// @brief Inspect one validated bit position in a packed vector.
/// @param row Packed words.
/// @param bit Valid position.
/// @return Whether the bit is set.
bool bit_at(const Bits& row, std::size_t bit) {return (row[bit/64]>>(bit%64))&1;}
/// @brief Set one validated bit position.
/// @param row Mutable packed words.
/// @param bit Valid position.
void set_bit(Bits& row, std::size_t bit) {row[bit/64]|=std::uint64_t{1}<<(bit%64);}
/// @brief XOR equal-length packed rows in place.
/// @param into Destination row.
/// @param from Source row, unchanged.
void xor_row(Bits& into, const Bits& from) {for(std::size_t i=0;i<into.size();++i)into[i]^=from[i];}
/// @brief Test whether a packed row is zero.
/// @param row Packed words.
/// @return True when every word is zero.
bool zero(const Bits& row) {for(auto word:row)if(word)return false;return true;}
/// @brief Recognize a proper divisor value already obtained from a GCD.
/// @param f Candidate GCD.
/// @param n Modulus.
/// @return Whether 1<f<n.
bool proper(std::uint64_t f, std::uint64_t n) {return f>1 && f<n;}
/// @brief Build one congruence from a zero parity row's original relation combination.
/// @param c Owned original relations and primes.
/// @param combo Packed selected original rows.
/// @return Owned even exponent sums, modular products and both exact GCDs.
DixonDependency dependency(const DixonCollection& c, const Bits& combo) {
    DixonDependency d;d.exponent_sums.resize(c.base.size());d.X=d.Y=1;
    for(std::size_t r=0;r<c.relations.size();++r)if(bit_at(combo,r)) {
        d.relation_indices.push_back(r);d.X=multiply(d.X,c.relations[r].x%c.n,c.n);
        for(std::size_t p=0;p<c.base.size();++p)d.exponent_sums[p]+=c.relations[r].exponents[p];
    }
    for(std::size_t p=0;p<c.base.size();++p)d.Y=multiply(d.Y,power(c.base[p],d.exponent_sums[p]/2,c.n),c.n);
    d.gcd_minus=std::gcd(d.X>d.Y?d.X-d.Y:d.Y-d.X,c.n);
    d.gcd_plus=std::gcd(static_cast<std::uint64_t>((static_cast<__uint128_t>(d.X)+d.Y)%c.n),c.n);
    return d;
}
/// @brief Execute the shared collection and batch all-row elimination kernel.
/// @param n Input modulus.
/// @param k Input prime count.
/// @param candidates Candidate budget, checked before allocations.
/// @param cap Optional logical capture cap; zero disables.
/// @param history Retain every actually tested dependency when true.
/// @return Actual trace; compact mode leaves dependency history empty.
/// @throws std::invalid_argument On invalid input domain.
/// @throws std::overflow_error On an unrepresentable requested endpoint.
/// @throws std::length_error On capture/count overflow; allocation can also fail.
DixonTrace run(std::uint64_t n, int k, std::size_t candidates, std::size_t cap, bool history) {
    const auto start=validate(n,k,candidates);DixonTrace t;
    collect(n,static_cast<std::size_t>(k),start,candidates,static_cast<std::size_t>(k)+1,cap,t);
    const auto& c=t.collection;const auto rows=c.relations.size(),cols=c.base.size();
    std::vector<Bits> matrix(rows,Bits((cols+63)/64)),combos(rows,Bits((rows+63)/64));
    for(std::size_t r=0;r<rows;++r) {
        set_bit(combos[r],r);
        for(std::size_t p=0;p<cols;++p)if(c.relations[r].exponents[p]&1)set_bit(matrix[r],p);
    }
    for(std::size_t col=0;col<cols && t.rank<rows;++col) {
        auto pivot=t.rank;
        while(pivot<rows && !bit_at(matrix[pivot],col))++pivot;
        if(pivot==rows)continue;
        std::swap(matrix[pivot],matrix[t.rank]);std::swap(combos[pivot],combos[t.rank]);
        for(std::size_t r=0;r<rows;++r)if(r!=t.rank && bit_at(matrix[r],col)) {
            xor_row(matrix[r],matrix[t.rank]);xor_row(combos[r],combos[t.rank]);++t.row_xors;
        }
        ++t.rank;
    }
    for(std::size_t r=0;r<rows;++r)if(zero(matrix[r])) {
        std::size_t selected=0;for(auto word:combos[r])selected+=std::popcount(word);
        if(history)t.captured_cells=capture(t.captured_cells,add_count(add_count(selected,cols),4),cap);
        auto d=dependency(c,combos[r]);
        if(proper(d.gcd_minus,n))t.factor=d.gcd_minus;
        else if(proper(d.gcd_plus,n))t.factor=d.gcd_plus;
        if(history)t.dependencies.push_back(std::move(d));
        if(t.factor)break;
    }
    return t;
}
} // namespace

DixonCollection dixon_collect(std::uint64_t n, int k, std::size_t candidates, std::size_t cap) {
    const auto start=validate(n,k,candidates);DixonTrace t;
    collect(n,static_cast<std::size_t>(k),start,candidates,0,cap,t);
    return std::move(t.collection);
}
DixonTrace dixon_factor_trace(std::uint64_t n, int k, std::size_t candidates, std::size_t cap) {
    return run(n,k,candidates,cap,true);
}
std::uint64_t dixon_factor(std::uint64_t n, int k) {return run(n,k,2000000,0,false).factor;}
} // namespace datamunge::algorithms
