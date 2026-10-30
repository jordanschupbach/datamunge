#include <datamunge/geometry/de_boor.hpp>

/// @file de_boor.cpp
/// @brief Validated B-spline span selection, shared de Boor recurrence and geometric diagnostics.

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace datamunge::geometry {
/// @brief Internal bounded interpolation and knot-ratio arithmetic.
namespace {

/// @brief Blends finite scalar coordinates without forming an overflowing opposite-sign difference.
/// @param a Left coordinate.
/// @param b Right coordinate.
/// @param weight Right-coordinate weight in [0,1].
/// @return Convex blend, exactly a or b when the weight is zero or one.
double blend(double a,double b,double weight) {
    if (weight==0) return a;
    if (weight==1) return b;
    if ((a<=0 && b>=0) || (a>=0 && b<=0)) return (1-weight)*a+weight*b;
    return a+weight*(b-a);
}

/// @brief Computes a finite interpolation weight even when the full knot interval overflows.
/// @param t Parameter between lo and hi, inclusive.
/// @param lo Finite lower knot.
/// @param hi Finite upper knot, strictly greater than lo.
/// @return Weight (t-lo)/(hi-lo), bounded to [0,1] against final-round rounding.
double knot_weight(double t,double lo,double hi) {
    if (t==lo) return 0;
    if (t==hi) return 1;
    const double width=hi-lo;
    const double value=std::isfinite(width) ? (t-lo)/width : (t*.5-lo*.5)/(hi*.5-lo*.5);
    return std::clamp(value,0.0,1.0);
}

} // namespace

DeBoorSpline::DeBoorSpline(int degree,std::vector<double> knots,std::vector<Point2D> control)
    : degree_(degree),knots_(std::move(knots)),control_(std::move(control)) {
    if (degree<0 || control_.empty() || static_cast<std::size_t>(degree)>=control_.size())
        throw std::invalid_argument("DeBoorSpline: require 0 <= degree < control count");
    const auto p=static_cast<std::size_t>(degree);
    if (knots_.size()<control_.size() || knots_.size()-control_.size()!=p+1)
        throw std::invalid_argument("DeBoorSpline: knot count must equal control count + degree + 1");
    std::size_t multiplicity=0;
    for (std::size_t i=0;i<knots_.size();++i) {
        if (!std::isfinite(knots_[i]) || (i && knots_[i]<knots_[i-1]))
            throw std::invalid_argument("DeBoorSpline: knots must be finite and nondecreasing");
        multiplicity=(i && knots_[i]==knots_[i-1]) ? multiplicity+1 : 1;
        if (multiplicity>p+1)
            throw std::invalid_argument("DeBoorSpline: knot multiplicity exceeds degree + 1");
    }
    const std::size_t n=control_.size();
    if (!(knots_[p]<knots_[p+1]) || !(knots_[n-1]<knots_[n]) || !(knots_[p]<knots_[n]))
        throw std::invalid_argument("DeBoorSpline: require a positive domain and nonempty end spans");
    for (const auto& point : control_)
        if (!std::isfinite(point.x) || !std::isfinite(point.y))
            throw std::invalid_argument("DeBoorSpline: control coordinates must be finite");
}

std::pair<double,double> DeBoorSpline::domain() const noexcept {
    return {knots_[static_cast<std::size_t>(degree_)],knots_[control_.size()]};
}

std::size_t DeBoorSpline::find_span(double t) const {
    const auto [lo,hi]=domain();
    if (!std::isfinite(t) || t<lo || t>hi)
        throw std::invalid_argument("DeBoorSpline: parameter must be finite and within the domain");
    if (t==hi) return control_.size()-1;
    return static_cast<std::size_t>(std::upper_bound(knots_.begin()+degree_,
        knots_.begin()+control_.size()+1,t)-knots_.begin()-1);
}

Point2D DeBoorSpline::evaluate_span(double t,std::size_t span,DeBoorTrace* trace) const {
    const auto p=static_cast<std::size_t>(degree_);
    std::vector<Point2D> work(control_.begin()+span-p,control_.begin()+span+1);
    if (trace) {
        trace->degree=degree_; trace->span=span; trace->parameter=t;
        const double nan=std::numeric_limits<double>::quiet_NaN();
        for (std::size_t j=0;j<=p;++j) trace->steps.push_back({0,j,span-p+j,nan,nan,nan,work[j]});
    }
    for (std::size_t round=1;round<=p;++round) {
        for (std::size_t j=p+1;j-- > round;) {
            const auto i=span-p+j;
            const double left=knots_[i],right=knots_[i+p-round+1];
            const double alpha=knot_weight(t,left,right);
            work[j]={blend(work[j-1].x,work[j].x,alpha),blend(work[j-1].y,work[j].y,alpha)};
            if (trace) trace->steps.push_back({round,j,i,left,right,alpha,work[j]});
        }
    }
    if (trace) trace->point=work[p];
    return work[p];
}

Point2D DeBoorSpline::evaluate(double t) const {
    return evaluate_span(t,find_span(t),nullptr);
}

DeBoorTrace DeBoorSpline::trace(double t) const {
    DeBoorTrace result;
    evaluate_span(t,find_span(t),&result);
    return result;
}

std::vector<Point2D> DeBoorSpline::sample(int samples) const {
    if (samples<2) throw std::invalid_argument("DeBoorSpline::sample: samples must be at least two");
    const auto [lo,hi]=domain();
    std::vector<Point2D> result;
    result.reserve(static_cast<std::size_t>(samples));
    for (int i=0;i<samples;++i) {
        const double t=blend(lo,hi,static_cast<double>(i)/(samples-1));
        result.push_back(evaluate(t));
    }
    return result;
}

dstruct::DataFrame DeBoorSpline::sample_table(const std::vector<double>& parameters) const {
    std::vector<double> span,first,last,x,y;
    for (double t : parameters) {
        const auto k=find_span(t);
        const auto point=evaluate_span(t,k,nullptr);
        span.push_back(static_cast<double>(k)); first.push_back(static_cast<double>(k-degree_));
        last.push_back(static_cast<double>(k)); x.push_back(point.x); y.push_back(point.y);
    }
    dstruct::DataFrame table;
    table.add_column("t",parameters); table.add_column("span",span);
    table.add_column("first_control",first); table.add_column("last_control",last);
    table.add_column("x",x); table.add_column("y",y);
    return table;
}

dstruct::DataFrame DeBoorTrace::table() const {
    std::vector<double> round,local,control,left,right,alpha,x,y;
    for (const auto& step : steps) {
        round.push_back(static_cast<double>(step.round)); local.push_back(static_cast<double>(step.local_index));
        control.push_back(static_cast<double>(step.control_index)); left.push_back(step.left_knot);
        right.push_back(step.right_knot); alpha.push_back(step.alpha); x.push_back(step.point.x); y.push_back(step.point.y);
    }
    dstruct::DataFrame table;
    table.add_column("round",round); table.add_column("local",local); table.add_column("control",control);
    table.add_column("left_knot",left); table.add_column("right_knot",right); table.add_column("alpha",alpha);
    table.add_column("x",x); table.add_column("y",y);
    return table;
}

graphics::RPlot DeBoorTrace::plot() const {
    const std::vector<graphics::RGB> colours{{110,110,110},{220,90,35},{25,145,105},{37,99,235}};
    auto plot=graphics::RPlot::create();
    for (int r=0;r<=degree;++r) {
        std::vector<double> x,y;
        for (const auto& step : steps) {
            if (step.round==static_cast<std::size_t>(r)) {x.push_back(step.point.x);y.push_back(step.point.y);}
        }
        if (x.empty()) continue;
        const auto colour=colours[static_cast<std::size_t>(r)%colours.size()];
        const auto label=r==0 ? "Active controls" : "Round "+std::to_string(r);
        if (x.size()>1) plot.lines(x,y,label,colour,2);
        plot.points(x,y,x.size()==1?label:"",colour,4);
    }
    plot.title("de Boor construction, t = "+std::to_string(parameter)).x_label("x").y_label("y");
    return plot;
}

graphics::RPlot DeBoorSpline::plot_curve(int samples_per_span) const {
    if (samples_per_span<2) throw std::invalid_argument("DeBoorSpline::plot_curve: samples per span must be at least two");
    auto plot=graphics::RPlot::create();
    std::vector<double> px,py;
    for (const auto& point : control_) {px.push_back(point.x);py.push_back(point.y);}
    if (control_.size()>1) plot.lines(px,py,"Control polygon",{145,145,145},1.5);
    plot.points(px,py,"Controls",{110,110,110},4);
    bool first=true;
    for (std::size_t k=static_cast<std::size_t>(degree_);k<control_.size();++k) {
        const double lo=knots_[k],hi=knots_[k+1];
        if (lo==hi) continue;
        std::vector<double> x,y;
        for (int j=0;j<samples_per_span;++j) {
            const double t=blend(lo,hi,static_cast<double>(j)/(samples_per_span-1));
            const auto point=evaluate_span(t,k,nullptr);
            x.push_back(point.x); y.push_back(point.y);
        }
        plot.lines(x,y,first?"Spline":"",{37,99,235},3);
        first=false;
    }
    plot.title("B-spline and its control polygon").x_label("x").y_label("y");
    return plot;
}

} // namespace datamunge::geometry
