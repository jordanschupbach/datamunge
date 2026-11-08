#include <datamunge/algorithms/polynomial_gf.hpp>
/// @file polynomial_gf.cpp
/// @brief Overflow-safe prime-field arithmetic with actual bounded Chien and Cantor-Zassenhaus provenance.
#include <datamunge/algorithms/miller_rabin.hpp>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>
namespace datamunge::algorithms {
/// @brief Private normalized polynomial kernels; public entry points validate the field once.
namespace gfpoly_detail {
/// @brief Validate a signed modulus with the existing deterministic 64-bit prime test.
/// @param p Candidate signed prime modulus.
/// @throws std::invalid_argument p is below two or composite.
void field(std::int64_t p) {
  if(p<2 || !is_probable_prime(static_cast<std::uint64_t>(p))) throw std::invalid_argument("GF polynomial: modulus must be prime");
}
/// @brief Normalize one signed coefficient without overflowing a+p.
/// @param a Arbitrary signed coefficient.
/// @param p Valid positive prime.
/// @return Residue in [0,p).
std::int64_t mod(std::int64_t a,std::int64_t p) {a%=p;return a<0?a+p:a;}
/// @brief Add two normalized residues through a widened intermediate.
/// @param a Residue in [0,p).
/// @param b Residue in [0,p).
/// @param p Valid positive prime.
/// @return Sum modulo p.
std::int64_t add(std::int64_t a,std::int64_t b,std::int64_t p) {return static_cast<std::int64_t>((static_cast<__int128>(a)+b)%p);}
/// @brief Multiply normalized residues through a signed 128-bit product.
/// @param a Residue in [0,p).
/// @param b Residue in [0,p).
/// @param p Valid positive prime, at most INT64_MAX.
/// @return Product modulo p.
std::int64_t mul(std::int64_t a,std::int64_t b,std::int64_t p) {return static_cast<std::int64_t>(static_cast<__int128>(a)*b%p);}
/// @brief Checked addition to a high-level work counter.
/// @param value Counter, unchanged on overflow.
/// @param amount Nonnegative increment.
/// @throws std::overflow_error Sum exceeds size_t.
void count(std::size_t& value,std::size_t amount=1) {
  if(amount>std::numeric_limits<std::size_t>::max()-value)throw std::overflow_error("GF polynomial: work count overflow");
  value+=amount;
}
/// @brief Binary modular scalar power for primality-valid field arithmetic.
/// @param a Arbitrary signed base.
/// @param e Nonnegative exponent.
/// @param p Valid prime modulus.
/// @param products Optional actual modular-product counter, default null for no counting.
/// @return a^e modulo p.
/// @throws std::overflow_error If a supplied product counter overflows.
std::int64_t scalar_power(std::int64_t a,std::uint64_t e,std::int64_t p,std::size_t* products=nullptr) {
  std::int64_t value=1;a=mod(a,p);
  while(e) {
    if(e&1) {value=mul(value,a,p);if(products)count(*products);}
    e>>=1;
    if(e) {a=mul(a,a,p);if(products)count(*products);}
  }
  return value;
}
/// @brief Remove trailing exact zero residues in place.
/// @param f Owned normalized coefficients to trim.
void trim(GFPoly& f) {while(!f.empty() && !f.back())f.pop_back();}
/// @brief Normalize and trim an owned coefficient vector.
/// @param f Owned arbitrary signed coefficients.
/// @param p Valid prime modulus.
/// @return Normalized owned coefficients.
GFPoly reduce(GFPoly f,std::int64_t p) {for(auto& c:f)c=mod(c,p);trim(f);return f;}
/// @brief Subtract already normalized polynomials without overflowing signed differences.
/// @param a Normalized left polynomial.
/// @param b Normalized right polynomial.
/// @param p Valid prime modulus.
/// @return Owned normalized difference.
GFPoly sub(const GFPoly& a,const GFPoly& b,std::int64_t p) {
  GFPoly out(std::max(a.size(),b.size()),0);
  std::copy(a.begin(),a.end(),out.begin());
  for(std::size_t i=0;i<b.size();++i)out[i]=mod(out[i]-b[i],p);
  trim(out);return out;
}
/// @brief Multiply normalized polynomials with checked output size and widened accumulation.
/// @param a Normalized left polynomial.
/// @param b Normalized right polynomial.
/// @param p Valid prime modulus.
/// @return Owned normalized schoolbook product.
/// @throws std::length_error Coefficient count exceeds size_t.
GFPoly multiply(const GFPoly& a,const GFPoly& b,std::int64_t p) {
  if(a.empty() || b.empty())return {};
  if(a.size()>std::numeric_limits<std::size_t>::max()-(b.size()-1))throw std::length_error("GF polynomial: product size overflow");
  GFPoly out(a.size()+(b.size()-1),0);
  for(std::size_t i=0;i<a.size();++i)if(a[i])for(std::size_t j=0;j<b.size();++j)out[i+j]=add(out[i+j],mul(a[i],b[j],p),p);
  trim(out);return out;
}
/// @brief Divide an owned normalized dividend by a normalized nonzero divisor.
/// @param a Owned normalized dividend, reduced in place to the remainder.
/// @param b Borrowed normalized divisor, nonempty.
/// @param p Valid prime modulus.
/// @return Owned quotient and remainder.
PolyDivMod divide(GFPoly a,const GFPoly& b,std::int64_t p) {
  if(a.size()<b.size())return {{},std::move(a)};
  GFPoly quotient(a.size()-b.size()+1,0);
  const auto inverse=scalar_power(b.back(),static_cast<std::uint64_t>(p-2),p);
  while(!a.empty() && a.size()>=b.size()) {
    const auto shift=a.size()-b.size();const auto coefficient=mul(a.back(),inverse,p);quotient[shift]=coefficient;
    for(std::size_t i=0;i<b.size();++i)a[i+shift]=mod(a[i+shift]-mul(coefficient,b[i],p),p);
    trim(a);
  }
  trim(quotient);return {std::move(quotient),std::move(a)};
}
/// @brief Reduce a normalized polynomial modulo a normalized nonzero modulus.
/// @param a Borrowed normalized polynomial.
/// @param modulus Borrowed nonempty normalized modulus.
/// @param p Valid prime modulus.
/// @return Owned normalized remainder.
GFPoly remainder(const GFPoly& a,const GFPoly& modulus,std::int64_t p) {return divide(a,modulus,p).remainder;}
/// @brief Divide a normalized nonzero polynomial by its leading coefficient.
/// @param f Owned normalized coefficients, zero allowed.
/// @param p Valid prime modulus.
/// @return Owned monic associate, or zero.
GFPoly monic(GFPoly f,std::int64_t p) {
  if(f.empty())return f;
  const auto inverse=scalar_power(f.back(),static_cast<std::uint64_t>(p-2),p);
  for(auto& c:f)c=mul(c,inverse,p);
  return f;
}
/// @brief Euclidean gcd on normalized owned polynomial operands.
/// @param a Owned normalized first operand.
/// @param b Owned normalized second operand.
/// @param p Valid prime modulus.
/// @return Monic gcd; zero for two zero operands.
GFPoly gcd(GFPoly a,GFPoly b,std::int64_t p) {
  while(!b.empty()) {auto r=remainder(a,b,p);a=std::move(b);b=std::move(r);}
  return monic(std::move(a),p);
}
/// @brief Formal derivative with safe modular reduction of the size_t degree.
/// @param f Normalized borrowed coefficients.
/// @param p Valid prime modulus.
/// @return Owned normalized derivative.
GFPoly derivative(const GFPoly& f,std::int64_t p) {
  if(f.size()<2)return {};
  GFPoly out(f.size()-1);
  for(std::size_t i=1;i<f.size();++i)out[i-1]=mul(f[i],static_cast<std::int64_t>(i%static_cast<std::uint64_t>(p)),p);
  trim(out);return out;
}
/// @brief Per-call field, options and checked arithmetic-request counters.
struct Context {
  std::int64_t p; ///< Validated prime field.
  CantorZassenhausOptions options; ///< Seed and complete-call budgets.
  CantorZassenhausWork work; ///< Actual counters, never shared across calls.

  /// @brief Multiply/reduce once after checking the whole-call product budget.
  /// @param a Normalized left polynomial.
  /// @param b Normalized right polynomial.
  /// @param modulus Nonzero normalized modulus.
  /// @return Owned normalized modular product.
  /// @throws std::length_error Product budget or coefficient size exceeded.
  /// @throws std::overflow_error Work count overflow.
  GFPoly modular_product(const GFPoly& a,const GFPoly& b,const GFPoly& modulus) {
    if(options.max_modular_products && work.modular_products>=options.max_modular_products)
      throw std::length_error("Cantor-Zassenhaus: modular-product budget exceeded");
    count(work.modular_products);return remainder(multiply(a,b,p),modulus,p);
  }
  /// @brief Binary polynomial powering with actual request counting and bounded products.
  /// @param base Normalized polynomial base, copied/reduced locally.
  /// @param exponent Nonnegative uint64 exponent.
  /// @param modulus Normalized nonzero polynomial modulus.
  /// @return Owned base^exponent modulo modulus.
  /// @throws std::length_error Product size or budget exceeded.
  /// @throws std::overflow_error Work count overflow.
  GFPoly power(GFPoly base,std::uint64_t exponent,const GFPoly& modulus) {
    count(work.modular_power_calls);base=remainder(base,modulus,p);GFPoly value=remainder({1},modulus,p);
    while(exponent) {
      if(exponent&1)value=modular_product(value,base,modulus);
      exponent>>=1;if(exponent)base=modular_product(base,base,modulus);
    }
    return value;
  }
  /// @brief Compute one actual gcd request and count it.
  /// @param a Normalized first polynomial, copied locally.
  /// @param b Normalized second polynomial, copied locally.
  /// @return Owned monic gcd.
  /// @throws std::overflow_error Gcd request count overflow.
  GFPoly common_divisor(const GFPoly& a,const GFPoly& b) {count(work.gcd_calls);return gcd(a,b,p);}
};
/// @brief Legacy xorshift64 sampler state, local to a factorization call.
struct XorShift {
  std::uint64_t state; ///< Nonzero current unsigned generator state.

  /// @brief Initialize the legacy stream, retaining its zero-seed mapping.
  /// @param seed Raw seed, zero maps to 0x1234567.
  explicit XorShift(std::uint64_t seed):state(seed?seed:0x1234567ULL) {}

  /// @brief Advance the unsigned xorshift stream by one raw word.
  /// @return Next deterministic 64-bit output/state.
  std::uint64_t next() {state^=state<<13;state^=state>>7;state^=state<<17;return state;}
};
/// @brief Compute h^((p^d-1)/2) modulo a bucket without ever storing p^d.
/// @param h Normalized random polynomial.
/// @param d Positive irreducible degree.
/// @param modulus Normalized bucket polynomial.
/// @param context Shared validated field and bounded work counters.
/// @return Actual modular character value.
/// @throws std::length_error Product size or complete-call product cap exceeded.
/// @throws std::overflow_error Work count overflow.
GFPoly character(const GFPoly& h,std::size_t d,const GFPoly& modulus,Context& context) {
  const auto half=static_cast<std::uint64_t>((context.p-1)/2);
  const auto first=context.power(h,half,modulus);auto value=first;
  for(std::size_t i=1;i<d;++i)value=context.modular_product(context.power(value,static_cast<std::uint64_t>(context.p),modulus),first,modulus);
  return value;
}
/// @brief Split a valid equal-degree bucket using actual iterative depth-first attempts.
/// @tparam Record Whether to own candidate/gcd/character/quotient records.
/// @param bucket Normalized monic squarefree degree-d bucket.
/// @param d Positive common irreducible degree.
/// @param rng Local legacy sampler, advanced by actual coefficient draws.
/// @param context Shared validated field, options and work counts.
/// @param out Owned factor/trace accumulator, never exposed partially.
/// @throws std::length_error Product size or complete-call trial/product budget exceeded.
/// @throws std::overflow_error Actual work count overflow.
template<bool Record>
void equal_degree(const GFPoly& bucket,std::size_t d,XorShift& rng,Context& context,CantorZassenhausTrace& out) {
  std::vector<GFPoly> pending{bucket};
  while(!pending.empty()) {
    auto target=std::move(pending.back());pending.pop_back();
    if(target.size()-1==d) {out.factors.push_back(std::move(target));continue;}
    bool accepted=false;
    while(!accepted) {
      if(context.options.max_random_trials && context.work.random_trials>=context.options.max_random_trials)
        throw std::length_error("Cantor-Zassenhaus: random-trial budget exceeded");
      CZEqualDegreeTrial trial;trial.attempt=context.work.random_trials;trial.degree=d;
      if constexpr(Record)trial.target=target;
      count(context.work.random_trials);
      GFPoly candidate(target.size()-1);
      for(auto& c:candidate) {count(context.work.random_words);c=static_cast<std::int64_t>(rng.next()%static_cast<std::uint64_t>(context.p));}
      trim(candidate);if constexpr(Record)trial.candidate=candidate;
      if(candidate.size()>1) {
        auto g=context.common_divisor(candidate,target);
        if constexpr(Record)trial.initial_gcd=g;
        if(g.size()<=1 || g.size()==target.size()) {
          auto value=character(candidate,d,target,context);
          if constexpr(Record) {trial.used_character=true;trial.character=value;}
          g=context.common_divisor(sub(value,{1},context.p),target);
        }
        if constexpr(Record)trial.split_gcd=g;
        if(g.size()>1 && g.size()<target.size()) {
          auto quotient=divide(target,g,context.p).quotient;
          if constexpr(Record) {trial.accepted=true;trial.quotient=quotient;}
          pending.push_back(std::move(quotient));pending.push_back(std::move(g));accepted=true;
        }
      }
      if constexpr(Record)out.equal_degree.push_back(std::move(trial));
    }
  }
}
/// @brief Shared checked factorization, optionally retaining actual DDF/EDF records.
/// @tparam Record Whether to retain owned input normalization and intermediate records.
/// @param input Borrowed arbitrary signed coefficients, nonzero squarefree modulo p.
/// @param p Odd prime modulus.
/// @param options Legacy seed and complete-call work budgets.
/// @return Owned complete factorization and actual counters/optional records.
/// @throws std::invalid_argument Invalid field, zero input or repeated factors.
/// @throws std::length_error Polynomial size or whole-call budget exceeded.
/// @throws std::overflow_error Work counters overflow size_t.
template<bool Record>
CantorZassenhausTrace factor(const GFPoly& input,std::int64_t p,CantorZassenhausOptions options) {
  field(p);if(p==2)throw std::invalid_argument("Cantor-Zassenhaus: odd prime required");
  auto normalized=reduce(input,p);if(normalized.empty())throw std::invalid_argument("Cantor-Zassenhaus: zero polynomial");
  CantorZassenhausTrace out;out.modulus=p;out.options=options;out.leading_coefficient=normalized.back();
  if constexpr(Record)out.input=normalized;
  auto remaining=monic(std::move(normalized),p);
  if constexpr(Record)out.monic_input=remaining;
  Context context{p,options,{}};
  auto squarefree=context.common_divisor(remaining,derivative(remaining,p));
  if(squarefree!=GFPoly{1})throw std::invalid_argument("Cantor-Zassenhaus: squarefree polynomial required");
  if constexpr(Record)out.squarefree_gcd=squarefree;
  XorShift rng(options.seed);GFPoly frobenius=remainder({0,1},remaining,p);
  for(std::size_t d=1;remaining.size()>1 && d<=(remaining.size()-1)/2;++d) {
    count(context.work.distinct_degree_steps);
    CZDistinctDegreeStep step;step.degree=d;if constexpr(Record)step.remaining_before=remaining;
    frobenius=context.power(std::move(frobenius),static_cast<std::uint64_t>(p),remaining);
    auto bucket=context.common_divisor(sub(frobenius,{0,1},p),remaining);
    if constexpr(Record) {step.frobenius=frobenius;step.bucket=bucket;}
    if(bucket.size()>1) {equal_degree<Record>(bucket,d,rng,context,out);remaining=divide(remaining,bucket,p).quotient;frobenius=remainder(frobenius,remaining,p);}
    if constexpr(Record) {step.remaining_after=remaining;out.distinct_degree.push_back(std::move(step));}
  }
  if constexpr(Record)out.residual=remaining;
  if(remaining.size()>1)out.factors.push_back(std::move(remaining));
  out.work=context.work;return out;
}
/// @brief Checked per-call Chien options and actual named operation counters.
struct ChienContext {
  std::int64_t p; ///< Validated prime modulus.
  ChienOptions options; ///< Complete-call point/product/generator-step caps.
  ChienWork work; ///< Actual counters, independent of other calls.

  /// @brief Admit one trial divisor or generator power test before performing it.
  /// @throws std::length_error If the positive complete-call step cap is full.
  /// @throws std::overflow_error If the step counter cannot be incremented.
  void generator_step() {
    if(options.max_generator_steps && work.generator_steps==options.max_generator_steps)
      throw std::length_error("Chien search: generator step budget exhausted");
    count(work.generator_steps);
  }
};
/// @brief Find the smallest primitive root while recording actual factor/power tests.
/// @param context Checked field, generator cap and actual counters.
/// @param trace Optional owned trace receiving factors and actual power results.
/// @return Smallest primitive root, one for GF(2).
/// @throws std::length_error If a generator-step cap or vector capacity is reached.
/// @throws std::overflow_error If a counter exceeds size_t.
/// @throws std::bad_alloc If factor/test storage allocation fails.
std::int64_t primitive_root(ChienContext& context,ChienTrace* trace) {
  const auto p=context.p;if(p==2)return 1;
  auto n=p-1;std::vector<std::int64_t> primes;
  for(std::int64_t f=2;f<=n/f;++f) {
    context.generator_step();count(context.work.generator_divisors);
    if(n%f==0) {primes.push_back(f);while(n%f==0)n/=f;}
  }
  if(n>1)primes.push_back(n);
  if(trace)trace->generator_factors=primes;
  for(std::int64_t g=2;g<p;++g) {
    bool valid=true;
    for(const auto q:primes) {
      context.generator_step();count(context.work.generator_tests);
      const auto value=scalar_power(g,static_cast<std::uint64_t>((p-1)/q),p,&context.work.generator_products);
      if(trace)trace->generator_tests.push_back({g,q,value});
      if(value==1) {valid=false;break;}
    }
    if(valid)return g;
  }
  return 1; // A valid prime field always has a primitive root.
}
/// @brief Shared legacy Chien shortcuts, actual recurrence and optional provenance.
/// @param f Borrowed arbitrary signed/trailing-zero coefficients.
/// @param p Candidate signed modulus, validated before any branch.
/// @param options Complete-call limits for named operations.
/// @param record Whether to retain generator tests and coefficient-state snapshots.
/// @return Owned roots/work, with full actual provenance only when requested.
/// @throws std::invalid_argument If the field is invalid.
/// @throws std::length_error If a cap or checked sweep/vector size is exceeded.
/// @throws std::overflow_error If an actual work counter overflows.
/// @throws std::bad_alloc If local coefficient/result allocation fails.
ChienTrace chien(const GFPoly& f,std::int64_t p,ChienOptions options,bool record) {
  field(p);auto normalized=reduce(f,p);ChienTrace out;out.modulus=p;
  if(record)out.input=normalized;
  ChienContext context{p,options,{}};
  if(normalized.empty())return out;
  if(normalized.size()==1) {out.path=ChienPath::Constant;return out;}
  if(normalized.size()==2) {
    out.path=ChienPath::Linear;
    const auto inverse=scalar_power(normalized[1],static_cast<std::uint64_t>(p-2),p,&context.work.linear_products);
    out.roots.push_back(mul(mod(-normalized[0],p),inverse,p));count(context.work.linear_products);
    out.work=context.work;return out;
  }
  out.path=ChienPath::Incremental;
  const auto points=static_cast<std::uint64_t>(p);
  if((options.max_points && points>options.max_points) || points>std::numeric_limits<std::size_t>::max())
    throw std::length_error("Chien search: complete field-point budget exceeded");
  const auto nonzero=static_cast<std::size_t>(points-1),terms_count=normalized.size();
  if(terms_count>std::numeric_limits<std::size_t>::max()/nonzero ||
      (options.max_term_products && terms_count>options.max_term_products/nonzero))
    throw std::length_error("Chien search: complete term-product budget exceeded");
  if(record) {
    if(nonzero>out.steps.max_size())throw std::length_error("Chien search: record capacity exceeded");
  }
  out.zero_checked=true;out.zero_value=normalized[0];count(context.work.field_points);
  if(!normalized[0])out.roots.push_back(0);
  out.generator=primitive_root(context,record?&out:nullptr);
  if(record)out.steps.reserve(nonzero);
  auto terms=normalized;GFPoly powers(terms_count);
  for(std::size_t j=0;j<terms_count;++j)
    powers[j]=scalar_power(out.generator,static_cast<std::uint64_t>(j),p,&context.work.setup_products);
  if(record)out.multipliers=powers;
  std::int64_t point=1;
  for(std::size_t i=0;i<nonzero;++i) {
    std::int64_t sum=0;
    for(const auto term:terms) {sum=add(sum,term,p);count(context.work.sum_additions);}
    count(context.work.field_points);
    if(!sum)out.roots.push_back(point);
    if(record)out.steps.push_back({i,point,terms,sum,sum==0});
    for(std::size_t j=0;j<terms_count;++j) {terms[j]=mul(terms[j],powers[j],p);count(context.work.term_products);}
    point=mul(point,out.generator,p);count(context.work.point_products);
  }
  if(record) {out.final_terms=std::move(terms);out.final_point=point;}
  out.work=context.work;return out;
}
} // namespace gfpoly_detail
GFPoly poly_normalize_gf(const GFPoly& f,std::int64_t p) {gfpoly_detail::field(p);return gfpoly_detail::reduce(f,p);}
GFPoly poly_multiply_gf(const GFPoly& a,const GFPoly& b,std::int64_t p) {
  gfpoly_detail::field(p);return gfpoly_detail::multiply(gfpoly_detail::reduce(a,p),gfpoly_detail::reduce(b,p),p);
}
PolyDivMod poly_divmod_gf(const GFPoly& a,const GFPoly& b,std::int64_t p) {
  gfpoly_detail::field(p);auto divisor=gfpoly_detail::reduce(b,p);
  if(divisor.empty())throw std::invalid_argument("GF polynomial: zero divisor");
  return gfpoly_detail::divide(gfpoly_detail::reduce(a,p),divisor,p);
}
GFPoly poly_gcd_gf(const GFPoly& a,const GFPoly& b,std::int64_t p) {
  gfpoly_detail::field(p);return gfpoly_detail::gcd(gfpoly_detail::reduce(a,p),gfpoly_detail::reduce(b,p),p);
}
std::int64_t poly_evaluate_gf(const GFPoly& f,std::int64_t x,std::int64_t p) {
  gfpoly_detail::field(p);x=gfpoly_detail::mod(x,p);std::int64_t value=0;
  for(std::size_t i=f.size();i-- >0;)value=gfpoly_detail::add(gfpoly_detail::mul(value,x,p),gfpoly_detail::mod(f[i],p),p);
  return value;
}
GFPoly poly_derivative_gf(const GFPoly& f,std::int64_t p) {gfpoly_detail::field(p);return gfpoly_detail::derivative(gfpoly_detail::reduce(f,p),p);}
bool poly_is_squarefree_gf(const GFPoly& f,std::int64_t p) {
  gfpoly_detail::field(p);auto normalized=gfpoly_detail::reduce(f,p);
  return !normalized.empty() && gfpoly_detail::gcd(normalized,gfpoly_detail::derivative(normalized,p),p)==GFPoly{1};
}
bool poly_is_irreducible_gf(const GFPoly& f,std::int64_t p) {
  gfpoly_detail::field(p);const auto normalized=gfpoly_detail::monic(gfpoly_detail::reduce(f,p),p);
  if(normalized.size()<2)return false;
  gfpoly_detail::Context context{p,{}, {}};GFPoly h=gfpoly_detail::remainder({0,1},normalized,p);
  for(std::size_t d=1;d<=(normalized.size()-1)/2;++d) {
    h=context.power(std::move(h),static_cast<std::uint64_t>(p),normalized);
    if(gfpoly_detail::gcd(gfpoly_detail::sub(h,{0,1},p),normalized,p).size()>1)return false;
  }
  return true;
}
std::string poly_string_gf(const GFPoly& f,std::int64_t p) {
  const auto normalized=poly_normalize_gf(f,p);std::string out;
  for(std::size_t i=normalized.size();i-- >0;)if(normalized[i]) {
    if(!out.empty())out+=" + ";
    if(!i || normalized[i]!=1)out+=std::to_string(normalized[i]);
    if(i) {out+="x";if(i>1)out+="^"+std::to_string(i);}
  }
  return out.empty()?"0":out;
}
std::vector<std::int64_t> chien_search(const GFPoly& f,std::int64_t p) {
  return gfpoly_detail::chien(f,p,{},false).roots;
}
std::vector<std::int64_t> chien_search_with_options(const GFPoly& f,std::int64_t p,ChienOptions options) {
  return gfpoly_detail::chien(f,p,options,false).roots;
}
ChienTrace chien_search_trace(const GFPoly& f,std::int64_t p,ChienOptions options) {
  return gfpoly_detail::chien(f,p,options,true);
}
std::vector<GFPoly> cantor_zassenhaus(const GFPoly& f,std::int64_t p,std::uint64_t seed) {
  CantorZassenhausOptions options;options.seed=seed;return gfpoly_detail::factor<false>(f,p,options).factors;
}
std::vector<GFPoly> cantor_zassenhaus_with_options(const GFPoly& f,std::int64_t p,CantorZassenhausOptions options) {
  return gfpoly_detail::factor<false>(f,p,options).factors;
}
CantorZassenhausTrace cantor_zassenhaus_trace(const GFPoly& f,std::int64_t p,CantorZassenhausOptions options) {
  return gfpoly_detail::factor<true>(f,p,options);
}
} // namespace datamunge::algorithms
