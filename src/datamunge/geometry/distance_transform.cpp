/// @file distance_transform.cpp
/// @brief Infinity-aware parabola envelopes, separable feature fields and bounded field graphics.
#include <datamunge/geometry/distance_transform.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>
#include <stdexcept>
namespace datamunge::geometry {
/// @brief Stateless checked envelope and grid implementation helpers.
namespace distance_transform_detail {
/// @brief Checked logical-count addition.
/// @param a First count.
/// @param b Second count.
/// @return Exact sum.
/// @throws std::length_error Counter overflow.
std::size_t add(std::size_t a,std::size_t b){
    if(b>std::numeric_limits<std::size_t>::max()-a)throw std::length_error("distance transform: count overflow");return a+b;
}
/// @brief Checks rectangular cell-count arithmetic and optional capture/plot budget.
/// @param rows Row count.
/// @param columns Column count.
/// @param cap Optional nonzero cell budget.
/// @return Checked cell count, leaving room for worst-case linear sweep counters.
/// @throws std::length_error Count/container capacity or optional budget exceeded.
std::size_t cells(std::size_t rows,std::size_t columns,std::size_t cap){
    if(columns && rows>std::numeric_limits<std::size_t>::max()/columns)throw std::length_error("distance transform: cell count overflow");
    const auto count=rows*columns;
    if(count>std::numeric_limits<std::size_t>::max()/4 || (cap && count>cap) || rows>std::vector<std::vector<double>>().max_size() || columns>std::vector<double>().max_size())throw std::length_error("distance transform: complete shape/capacity limit");
    return count;
}
/// @brief Validates costs before any output or envelope allocation.
/// @param f Borrowed finite or positive-infinity values.
/// @throws std::invalid_argument NaN or negative infinity.
void validate_costs(const std::vector<double>& f){
    for(double value:f)if(std::isnan(value) || value==-std::numeric_limits<double>::infinity())throw std::invalid_argument("distance transform: NaN/negative infinity cost");
}
/// @brief Converts one wide evaluated cost to finite double.
/// @param value Wide finite cost.
/// @return Representable double.
/// @throws std::overflow_error Cost is nonfinite or outside double range.
double narrow(long double value){
    if(!std::isfinite(value) || std::abs(value)>std::numeric_limits<double>::max())throw std::overflow_error("distance transform: cost overflow");return static_cast<double>(value);
}
/// @brief Executes a single actual monotone envelope sweep and integer-query evaluation.
/// @param f Valid finite or positive-infinity costs.
/// @param capture Whether to own input, comparisons and final envelope.
/// @return Actual costs/argmins/work, with optional history.
/// @throws std::length_error Counter or container capacity exceeded.
/// @throws std::overflow_error Nonfinite crossing or output cost.
SquaredDistanceTrace line(const std::vector<double>& f,bool capture){
    const auto n=f.size();SquaredDistanceTrace out;if(capture)out.costs=f;
    out.values.assign(n,std::numeric_limits<double>::infinity());out.nearest_index.assign(n,n);
    out.work.lines=1;out.work.input_values=n;
    std::vector<std::size_t> envelope;std::vector<long double> begin;
    const auto inf=std::numeric_limits<long double>::infinity();
    for(std::size_t q=0;q<n;++q){if(!std::isfinite(f[q]))continue;
        out.work.finite_candidates=add(out.work.finite_candidates,1);long double crossing=-inf;
        while(!envelope.empty()){
            const auto p=envelope.back();const long double lp=p,lq=q;
            // Factor the difference of squares; divide costs before subtracting to avoid overflow.
            crossing=(lp+lq)/2+(static_cast<long double>(f[q])/2-static_cast<long double>(f[p])/2)/(lq-lp);
            if(!std::isfinite(crossing))throw std::overflow_error("distance transform: nonfinite crossing");
            out.work.intersections=add(out.work.intersections,1);const bool remove=crossing<=begin.back();
            if(capture)out.comparisons.push_back({q,p,crossing,begin.back(),remove});
            if(!remove)break;envelope.pop_back();begin.pop_back();out.work.envelope_pops=add(out.work.envelope_pops,1);
        }
        envelope.push_back(q);begin.push_back(envelope.size()==1?-inf:crossing);
        out.work.envelope_pushes=add(out.work.envelope_pushes,1);out.work.max_envelope_size=std::max(out.work.max_envelope_size,envelope.size());
    }
    if(!envelope.empty()){
        std::size_t selected=0;
        for(std::size_t q=0;q<n;++q){
            while(selected+1<envelope.size() && begin[selected+1]<static_cast<long double>(q)){++selected;out.work.selection_advances=add(out.work.selection_advances,1);}
            const auto p=envelope[selected];const long double delta=static_cast<long double>(q)-static_cast<long double>(p);
            out.values[q]=narrow(static_cast<long double>(f[p])+delta*delta);out.nearest_index[q]=p;out.work.evaluations=add(out.work.evaluations,1);
        }
        if(capture){out.envelope_sites=std::move(envelope);out.breakpoints=std::move(begin);out.breakpoints.push_back(inf);}
    }
    return out;
}
/// @brief Adds independent line work, retaining the largest envelope peak.
/// @param total Mutable aggregate.
/// @param work Completed line counters.
/// @throws std::length_error Counter overflow.
void accumulate(DistanceTransformWork& total,const DistanceTransformWork& work){
    for(auto member:{&DistanceTransformWork::lines,&DistanceTransformWork::input_values,&DistanceTransformWork::finite_candidates,&DistanceTransformWork::intersections,&DistanceTransformWork::envelope_pushes,&DistanceTransformWork::envelope_pops,&DistanceTransformWork::selection_advances,&DistanceTransformWork::evaluations})total.*member=add(total.*member,work.*member);
    total.max_envelope_size=std::max(total.max_envelope_size,work.max_envelope_size);
}
/// @brief Validates a rectangular input mask and its complete cell budget.
/// @param grid Borrowed input mask.
/// @param cap Optional cell limit.
/// @return Row and column counts.
/// @throws std::invalid_argument Ragged shape.
/// @throws std::length_error Count/capacity or optional cap exceeded.
std::pair<std::size_t,std::size_t> shape(const std::vector<std::vector<char>>& grid,std::size_t cap){
    const auto rows=grid.size(),columns=rows?grid.front().size():0;
    for(const auto& row:grid)if(row.size()!=columns)throw std::invalid_argument("distance transform: ragged grid");
    (void)cells(rows,columns,cap);return {rows,columns};
}
/// @brief Executes column then row transforms, optionally retaining features/intermediate matrices.
/// @param grid Validated rectangular mask.
/// @param rows Row count.
/// @param columns Column count.
/// @param features Whether to retain original-site witnesses.
/// @param capture Whether to retain column matrices too.
/// @return Owned actual field and optional intermediates.
/// @throws std::length_error Counter/container capacity exceeded.
/// @throws std::overflow_error Unrepresentable crossing or transformed value.
DistanceTransformTrace grid_run(const std::vector<std::vector<char>>& grid,std::size_t rows,std::size_t columns,bool features,bool capture){
    DistanceTransformTrace out;const double inf=std::numeric_limits<double>::infinity();
    out.result.distance.assign(rows,std::vector<double>(columns,inf));out.result.squared_distance=out.result.distance;
    std::vector<std::vector<std::size_t>> column_rows;
    if(features){column_rows.assign(rows,std::vector<std::size_t>(columns,rows));out.result.nearest_row=column_rows;out.result.nearest_column.assign(rows,std::vector<std::size_t>(columns,columns));}
    if(!rows || !columns){if(capture){out.column_squared=out.result.squared_distance;out.column_nearest_row=column_rows;}return out;}
    for(std::size_t c=0;c<columns;++c){std::vector<double> f(rows,inf);
        for(std::size_t r=0;r<rows;++r)if(grid[r][c])f[r]=0;
        const auto t=line(f,false);accumulate(out.work,t.work);
        for(std::size_t r=0;r<rows;++r){out.result.squared_distance[r][c]=t.values[r];if(features)column_rows[r][c]=t.nearest_index[r];}
    }
    if(capture){out.column_squared=out.result.squared_distance;out.column_nearest_row=column_rows;}
    for(std::size_t r=0;r<rows;++r){auto t=line(out.result.squared_distance[r],false);accumulate(out.work,t.work);
        out.result.squared_distance[r]=std::move(t.values);
        for(std::size_t c=0;c<columns;++c){out.result.distance[r][c]=std::sqrt(out.result.squared_distance[r][c]);
            if(features){const auto source_column=t.nearest_index[c];out.result.nearest_column[r][c]=source_column;out.result.nearest_row[r][c]=source_column<columns?column_rows[r][source_column]:rows;}}
    }
    return out;
}
/// @brief Formats bounded plot labels without multiplying extreme values.
/// @param value Finite nonnegative value.
/// @param precision Requested decimal/significant precision.
/// @param fixed Whether moderate values use fixed decimal notation.
/// @return Owned numeric label.
std::string numeric_label(double value,int precision,bool fixed){
    std::ostringstream out;if(fixed && value<10000)out<<std::fixed;
    out<<std::setprecision(precision)<<value;return out.str();
}
} // namespace distance_transform_detail
std::vector<double> squared_distance_transform_1d(const std::vector<double>& costs){
    distance_transform_detail::validate_costs(costs);return distance_transform_detail::line(costs,false).values;
}
SquaredDistanceTrace squared_distance_transform_1d_trace(const std::vector<double>& costs,std::size_t max_samples){
    distance_transform_detail::validate_costs(costs);if(max_samples && costs.size()>max_samples)throw std::length_error("distance transform: sample cap");return distance_transform_detail::line(costs,true);
}
std::vector<std::vector<double>> euclidean_distance_transform(const std::vector<std::vector<char>>& grid){
    const auto [rows,columns]=distance_transform_detail::shape(grid,0);return distance_transform_detail::grid_run(grid,rows,columns,false,false).result.distance;
}
EuclideanDistanceField euclidean_distance_transform_features(const std::vector<std::vector<char>>& grid){
    const auto [rows,columns]=distance_transform_detail::shape(grid,0);return distance_transform_detail::grid_run(grid,rows,columns,true,false).result;
}
DistanceTransformTrace euclidean_distance_transform_trace(const std::vector<std::vector<char>>& grid,std::size_t max_cells){
    const auto [rows,columns]=distance_transform_detail::shape(grid,max_cells);return distance_transform_detail::grid_run(grid,rows,columns,true,true);
}
std::vector<std::vector<char>> distance_site_grid(std::size_t rows,std::size_t columns,const std::vector<std::pair<std::size_t,std::size_t>>& sites,std::size_t max_cells){
    for(const auto& [r,c]:sites)if(r>=rows || c>=columns)throw std::invalid_argument("site grid: site outside shape");
    (void)distance_transform_detail::cells(rows,columns,max_cells);std::vector<std::vector<char>> grid(rows,std::vector<char>(columns,0));
    for(const auto& [r,c]:sites)grid[r][c]=1;return grid;
}
graphics::RPlot plot_distance_field(const EuclideanDistanceField& field,bool annotate,std::size_t max_cells){
    const auto& d=field.distance;const auto rows=d.size(),columns=rows?d.front().size():0;
    if(!rows || !columns)throw std::invalid_argument("distance plot: nonempty field required");
    for(const auto& row:d)if(row.size()!=columns)throw std::invalid_argument("distance plot: ragged field");
    (void)distance_transform_detail::cells(rows,columns,max_cells);
    std::vector<double> values,site_x,site_y;
    for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<columns;++c){const auto v=d[r][c];
        if(std::isnan(v) || v<0)throw std::invalid_argument("distance plot: invalid distance");
        if(std::isfinite(v))values.push_back(v);if(v==0){site_x.push_back(c);site_y.push_back(-static_cast<double>(r));}}
    const double maximum=values.empty()?0:*std::max_element(values.begin(),values.end());
    const std::vector<graphics::RGBA> palette{{248,250,252},{191,219,254},{96,165,250},{37,99,235},{30,58,138}};
    const auto colors=values.empty()?std::vector<graphics::RGBA>{}:graphics::value_colors(values,palette,0,maximum?maximum:1);std::size_t index=0;
    auto plot=graphics::RPlot::create();
    for(std::size_t r=0;r<rows;++r)for(std::size_t c=0;c<columns;++c){const auto v=d[r][c];const auto color=std::isfinite(v)?colors[index++]:graphics::RGBA{220,220,220};
        const double x=c,y=-static_cast<double>(r);plot.polygon({x-.5,x+.5,x+.5,x-.5},{y-.5,y-.5,y+.5,y+.5},color);
        if(annotate && std::isfinite(v) && v>0)plot.text(x,y,distance_transform_detail::numeric_label(v,1,true),v>maximum*.65?graphics::RGBA{255,255,255}:graphics::RGBA{17,24,39},12);
    }
    if(!site_x.empty())plot.points(site_x,site_y,"",{0,0,0},4);
    std::vector<double> ticks;std::vector<std::string> labels;const auto stride=std::max(std::size_t{1},rows/8);
    for(std::size_t r=0;r<rows;r+=stride){ticks.push_back(-static_cast<double>(r));labels.push_back(std::to_string(r));}
    plot.title("Distance field; black points are sites").x_label("column").y_label("row (downward)").x_limits(-.5,static_cast<double>(columns)-.5).y_limits(-static_cast<double>(rows)+.5,3).y_ticks(ticks,labels).show_grid(false);
    if(values.empty())plot.legend({"no site: infinity"},{{220,220,220}});
    else plot.legend({"distance 0","distance "+distance_transform_detail::numeric_label(maximum,3,false),"site"},{palette.front(),palette.back(),{0,0,0}});
    return plot;
}
} // namespace datamunge::geometry
