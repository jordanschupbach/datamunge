/// @file bisection.cpp
/// @brief Shared finite-value bracket loop with optional chronological recording.
#include <datamunge/algorithms/bisection.hpp>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace datamunge::algorithms {
/// @brief Private bisection execution, shared by recorded and unrecorded interfaces.
namespace {
/// @brief Evaluate f, incrementing the actual callback count and rejecting nonfinite results.
/// @param f Borrowed callable.
/// @param x Coordinate to evaluate.
/// @param count Borrowed callback counter, incremented before calling f.
/// @return Finite callback result.
/// @throws std::domain_error For a nonfinite result; callback exceptions propagate.
double evaluate(const std::function<double(double)>& f,double x,std::size_t& count) {
  ++count;
  const double value=f(x);
  if (!std::isfinite(value)) throw std::domain_error("bisection: callback returned a nonfinite value");
  return value;
}

/// @brief Execute the common midpoint loop, optionally recording each evaluated bracket.
/// @param f Nonempty callable.
/// @param a First finite endpoint.
/// @param b Second finite endpoint, reordered if necessary.
/// @param options Independent position/residual criteria and iteration budget.
/// @param record Whether to retain midpoint snapshots.
/// @return Owned result and diagnostic state; no steps are allocated when record is false.
/// @throws std::invalid_argument For invalid inputs or options.
/// @throws std::domain_error For nonfinite callback values; callback exceptions propagate.
BisectionTrace run(const std::function<double(double)>& f,double a,double b,
                   BisectionOptions options,bool record) {
  if (!f || !std::isfinite(a) || !std::isfinite(b) ||
      !std::isfinite(options.x_tolerance) || options.x_tolerance<=0 ||
      !std::isfinite(options.residual_tolerance) || options.residual_tolerance<0 ||
      options.max_iterations<=0)
    throw std::invalid_argument("bisection: finite endpoints and valid positive position/budget required");
  if (a>b) std::swap(a,b);
  BisectionTrace trace;
  trace.lower=a;
  trace.upper=b;
  double fa=evaluate(f,a,trace.evaluations);
  if (fa==0) {
    trace.result={true,a,0,0}; trace.stop=BisectionStop::Endpoint;
    trace.lower=trace.upper=a; trace.radius=0;
    return trace;
  }
  double fb=evaluate(f,b,trace.evaluations);
  if (fb==0) {
    trace.result={true,b,0,0}; trace.stop=BisectionStop::Endpoint;
    trace.lower=trace.upper=b; trace.radius=0;
    return trace;
  }
  if (std::signbit(fa)==std::signbit(fb)) return trace;
  for (int iteration=1;;++iteration) {
    const double c=std::midpoint(a,b);
    const double fc=evaluate(f,c,trace.evaluations);
    const double radius=std::max(c-a,b-c);
    trace.result={false,c,iteration,std::abs(fc)};
    trace.lower=a; trace.upper=b; trace.radius=radius;
    if (record) trace.steps.push_back({a,b,fa,fb,c,fc,radius});
    if (fc==0) { trace.stop=BisectionStop::ExactValue; trace.result.converged=true; return trace; }
    if (options.residual_tolerance>0 && std::abs(fc)<options.residual_tolerance) {
      trace.stop=BisectionStop::Residual; trace.result.converged=true; return trace;
    }
    if (radius<options.x_tolerance) {
      trace.stop=BisectionStop::Interval; trace.result.converged=true; return trace;
    }
    if (c==a || c==b) { trace.stop=BisectionStop::Stagnation; return trace; }
    if (iteration==options.max_iterations) { trace.stop=BisectionStop::IterationLimit; return trace; }
    if (std::signbit(fc)==std::signbit(fa)) { a=c; fa=fc; }
    else { b=c; fb=fc; }
  }
}
} // namespace

std::string_view bisection_stop_name(BisectionStop stop) {
  switch (stop) {
    case BisectionStop::NoBracket: return "no bracket";
    case BisectionStop::Endpoint: return "endpoint";
    case BisectionStop::ExactValue: return "zero callback";
    case BisectionStop::Residual: return "residual";
    case BisectionStop::Interval: return "interval";
    case BisectionStop::Stagnation: return "stagnation";
    case BisectionStop::IterationLimit: return "iteration limit";
  }
  return "unknown";
}

RootResult bisection(const std::function<double(double)>& f,double a,double b,double tol,int max_iter) {
  return run(f,a,b,{tol,tol,max_iter},false).result;
}

BisectionTrace bisection_trace(const std::function<double(double)>& f,double a,double b,BisectionOptions options) {
  return run(f,a,b,options,true);
}
} // namespace datamunge::algorithms
