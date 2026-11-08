/// @file flow_diagram.cpp
/// @brief Per-input-edge native flow diagrams with clipped lanes, loops and supplied-cut colors.
#include <datamunge/algorithms/flow_diagram.hpp>
#include <algorithm>
#include <cmath>
#include <format>
#include <map>
#include <stdexcept>
namespace datamunge::algorithms {
/// @brief Native diagram geometry and complete local input validation.
namespace flow_diagram_detail {
/// @brief Validate shape and finite values without claiming a feasibility certificate.
/// @param g Original network.
/// @param r Caller flow/cut data.
/// @param p Caller diagram coordinates.
/// @throws std::invalid_argument On invalid graph, result or positions.
void validate(const FlowNetwork& g,const MaxFlowResult& r,const std::vector<graphics::DiagramPoint>& p){
    if(g.source>=g.vertex_count||g.sink>=g.vertex_count||g.source==g.sink||p.size()!=g.vertex_count
       ||r.edge_flow.size()!=g.edges.size()||!std::isfinite(r.max_flow))throw std::invalid_argument("flow diagram: invalid graph/result shape");
    for(const auto& [u,v,c]:g.edges)if(u>=g.vertex_count||v>=g.vertex_count||!std::isfinite(c)||c<0)throw std::invalid_argument("flow diagram: invalid capacity edge");
    for(auto f:r.edge_flow)if(!std::isfinite(f))throw std::invalid_argument("flow diagram: nonfinite edge flow");
    for(auto at:p)if(!std::isfinite(at.x)||!std::isfinite(at.y))throw std::invalid_argument("flow diagram: nonfinite position");
    std::vector<bool> cut(g.vertex_count,false);for(auto v:r.min_cut_source_side){if(v>=g.vertex_count||cut[v])throw std::invalid_argument("flow diagram: invalid cut ids");cut[v]=true;}
}
} // namespace flow_diagram_detail

graphics::Diagram plot_max_flow(const FlowNetwork& g,const MaxFlowResult& r,
                                const std::vector<graphics::DiagramPoint>& p){
    flow_diagram_detail::validate(g,r,p);
    std::vector<bool> cut(g.vertex_count,false);for(auto v:r.min_cut_source_side)cut[v]=true;
    std::vector<std::string> labels;std::vector<double> radii;
    for(std::size_t v=0;v<g.vertex_count;++v){labels.push_back(std::format("{}{}",v==g.source?"s=":v==g.sink?"t=":"",v));radii.push_back(std::max(.65,(12.*labels.back().size()+20)/128.));}
    std::map<std::pair<std::size_t,std::size_t>,std::vector<std::size_t>> groups;
    for(std::size_t i=0;i<g.edges.size();++i){const auto [u,v,c]=g.edges[i];groups[{std::min(u,v),std::max(u,v)}].push_back(i);}
    graphics::Diagram d(64,1100);
    for(const auto& [ends,ids]:groups){
        const auto [lo,hi]=ends;
        if(lo!=hi){const double distance=std::hypot(p[hi].x-p[lo].x,p[hi].y-p[lo].y);
            if(!std::isfinite(distance)||distance<=radii[lo]+radii[hi]+.2)throw std::invalid_argument("flow diagram: endpoints need finite separation and circle clearance");}
        for(std::size_t lane=0;lane<ids.size();++lane){const auto i=ids[lane];const auto [u,v,c]=g.edges[i];
            graphics::DiagramStroke stroke;stroke.arrow=true;stroke.width=r.edge_flow[i]==0?1.5:2.5;
            stroke.color=cut[u]&&!cut[v]?graphics::RGB{220,38,38}:graphics::RGB{71,85,105};
            const auto text=std::format("e{}: {:.4g}/{:.4g}",i,r.edge_flow[i],c);
            if(u==v){const double radius=radii[u],half=radius+.6+.45*lane,top=p[u].y+radius+1.4+.8*lane;
                const double q=radius/std::sqrt(2.);
                d.path({{p[u].x-q,p[u].y+q},{p[u].x-half,top},{p[u].x+half,top},{p[u].x+q,p[u].y+q}},stroke)
                 .label({p[u].x,top+.3},text,18,stroke.color);
            }else{
                const double dx=p[hi].x-p[lo].x,dy=p[hi].y-p[lo].y,distance=std::hypot(dx,dy),nx=-dy/distance,ny=dx/distance;
                const double offset=double(lane)-.5*double(ids.size()-1);
                const graphics::DiagramPoint mid{.5*p[u].x+.5*p[v].x+offset*nx,.5*p[u].y+.5*p[v].y+offset*ny};
                const double du=std::hypot(mid.x-p[u].x,mid.y-p[u].y),dv=std::hypot(mid.x-p[v].x,mid.y-p[v].y);
                if(!std::isfinite(du)||!std::isfinite(dv)||du<=radii[u]||dv<=radii[v])throw std::invalid_argument("flow diagram: lane geometry overflow or insufficient clearance");
                const graphics::DiagramPoint from{p[u].x+radii[u]*(mid.x-p[u].x)/du,p[u].y+radii[u]*(mid.y-p[u].y)/du};
                const graphics::DiagramPoint to{p[v].x+radii[v]*(mid.x-p[v].x)/dv,p[v].y+radii[v]*(mid.y-p[v].y)/dv};
                const double side=offset<0?-1:offset>0?1:u==lo?1:-1;
                d.path({from,mid,to},stroke).label({mid.x+side*.3*nx,mid.y+side*.3*ny},text,18,stroke.color);
            }
        }
    }
    for(std::size_t v=0;v<g.vertex_count;++v)d.ellipse(p[v],radii[v],radii[v],cut[v]?graphics::RGB{187,247,208}:graphics::RGB{254,215,170})
        .text(p[v],labels[v],20);
    return d;
}
} // namespace datamunge::algorithms
