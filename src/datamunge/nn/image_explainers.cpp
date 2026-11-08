/// @file image_explainers.cpp
/// @brief Detached tensor coalition masking, actual batched scores, weighted local fits and exact Shapley reductions.
#include <datamunge/nn/image_explainers.hpp>
#include <datamunge/nn/autograd.hpp>
#include <datamunge/nn/linalg.hpp>
#include <datamunge/nn/ops.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <format>
#include <limits>
#include <random>
#include <stdexcept>
namespace datamunge::nn {
/// @brief Validated geometry, metadata and shared tensor masking/prediction dispatch.
namespace image_explanation_detail {
/// @brief Require finite nonempty CPU floating data, without converting pixels to host arrays.
/// @param x Borrowed input or callback output.
/// @throws std::invalid_argument On undefined, empty, non-CPU, nonfloating or nonfinite data.
void finite(const Tensor& x){if(!x.defined()||x.device().backend!=compute::Backend::CPU||x.numel()<1||(x.dtype()!=DType::Float32&&x.dtype()!=DType::Float64)||!all(is_finite(x)).item())throw std::invalid_argument("image explanation: finite nonempty CPU floating tensor required");}
/// @brief Reject nonfinite computed tensor arithmetic.
/// @param x Computed tensor.
/// @return x unchanged.
/// @throws std::overflow_error On nonfinite values.
Tensor computed(const Tensor& x){if(!all(is_finite(x)).item())throw std::overflow_error("image explanation: nonfinite computed arithmetic");return x;}
/// @brief Validate scalar arithmetic after tensor reduction.
/// @param x Computed scalar.
/// @return Finite x unchanged.
/// @throws std::overflow_error On nonfinite result.
double scalar_checked(double x){if(!std::isfinite(x))throw std::overflow_error("image explanation: nonfinite scalar result");return x;}
/// @brief Check tensor element geometry and practical host-metadata count limits.
/// @param shape Positive requested dimensions.
/// @return Checked element count.
/// @throws std::length_error On dimension/container overflow.
std::int64_t geometry(const Shape& shape){
    try{auto n=shape_numel(shape);if(static_cast<std::uint64_t>(n)>std::vector<double>{}.max_size())throw std::length_error("image explanation: metadata container limit");return n;}
    catch(const std::overflow_error&){throw std::length_error("image explanation: shape overflow");}
}
/// @brief Validated spatial grid and channel shape, no image storage retained.
struct Grid {
    std::int64_t h=0; ///< Image height.
    std::int64_t w=0; ///< Image width.
    std::int64_t channels=0; ///< Positive image channel count.
    std::int64_t rows=0; ///< Positive nonempty grid rows.
    std::int64_t cols=0; ///< Positive nonempty grid columns.
    std::int64_t patches=0; ///< Checked rows times columns.
};
/// @brief Validate all spatial membership products before allocating patch metadata.
/// @param h Positive image height.
/// @param w Positive image width.
/// @param channels Positive image channels.
/// @param rows Positive nonempty grid rows.
/// @param cols Positive nonempty grid columns.
/// @return Checked scalar grid.
/// @throws std::invalid_argument On empty cells or nonpositive geometry.
/// @throws std::length_error On dimension or membership-product overflow.
Grid grid(std::int64_t h,std::int64_t w,std::int64_t channels,std::int64_t rows,std::int64_t cols){
    if(h<1||w<1||channels<1||rows<1||cols<1||rows>h||cols>w)throw std::invalid_argument("image explanation: every grid patch must contain pixels");
    geometry({h,w,channels});geometry({h,rows});geometry({w,cols});return {h,w,channels,rows,cols,geometry({rows,cols})};
}
/// @brief Validate one finite NHWC image and its baseline before any model call.
/// @param x Borrowed image.
/// @param o Grid and baseline controls.
/// @return Validated scalar grid.
/// @throws std::invalid_argument On rank, batch, data, baseline or grid domain errors.
/// @throws std::length_error On unsupported geometry.
Grid image_grid(const Tensor& x,const ImagePatchOptions& o){
    finite(x);if(x.dim()!=4||x.size(0)!=1)throw std::invalid_argument("image explanation: one NHWC image required");
    if(!std::isfinite(o.baseline)||(x.dtype()==DType::Float32&&!std::isfinite(static_cast<float>(o.baseline))))throw std::invalid_argument("image explanation: finite dtype-representable baseline required");
    return grid(x.size(1),x.size(2),x.size(3),o.patch_rows,o.patch_cols);
}
/// @brief Produce integer row-major patch indices for each spatial location.
/// @param g Validated grid with safe membership products.
/// @return Owned Int64 [H*W] metadata, with no numerical pixel values.
Tensor patch_indices(const Grid& g){
    std::vector<std::int64_t> index(geometry({g.h,g.w}));
    for(std::int64_t y=0;y<g.h;++y)for(std::int64_t x=0;x<g.w;++x)index[y*g.w+x]=(y*g.rows/g.h)*g.cols+x*g.cols/g.w;
    return from_vector(index);
}
/// @brief Construct independent masked images from already validated flags and geometry.
/// @param image Borrowed validated single image.
/// @param masks Bool [B,P] on/off flags.
/// @param index Precomputed Int64 spatial patch metadata.
/// @param g Validated scalar grid.
/// @param baseline Finite image-dtype baseline.
/// @return Owned detached [B,H,W,C] masked pixels.
Tensor mask(const Tensor& image,const Tensor& masks,const Tensor& index,const Grid& g,double baseline){
    const auto spatial=reshape(gather(masks,index,1),{masks.size(0),g.h,g.w,1});
    return where(spatial,image.detach(),scalar(baseline,image.dtype()));
}
/// @brief Preflight actual evaluation count, image batch and retained design geometry.
/// @param g Validated grid.
/// @param count Positive actual number of model images, not a requested upper limit.
/// @param o Batch and evaluation controls.
/// @param predict Required callback.
/// @param design Whether complete N-by-P mask storage is retained.
/// @throws std::invalid_argument On empty callback or zero batch size.
/// @throws std::length_error On quota or geometry limits; happens before callbacks.
void query_geometry(const Grid& g,std::size_t count,const ImagePatchOptions& o,const ImageScoreBatch& predict,bool design){
    if(!predict||!o.batch_size)throw std::invalid_argument("image explanation: callable and positive batch size required");
    if(o.max_evaluations&&count>o.max_evaluations)throw std::length_error("image explanation: evaluation quota exceeded");
    if(count>static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max()))throw std::length_error("image explanation: evaluation shape overflow");
    geometry({static_cast<std::int64_t>(count)});geometry({static_cast<std::int64_t>(std::min(count,o.batch_size)),g.h,g.w,g.channels});
    if(design){if(g.patches==std::numeric_limits<std::int64_t>::max())throw std::length_error("image explanation: design width overflow");geometry({static_cast<std::int64_t>(count),g.patches+1});geometry({g.patches+1,g.patches+1});}
}
/// @brief Invoke and validate one actual model batch, copying its independent scores.
/// @param images Owned masked-image batch.
/// @param predict Stable fixed-target callback.
/// @param work Actual dispatch counters.
/// @return Detached owned Float64 [B] score values.
/// @throws std::invalid_argument On malformed/nonfinite callback scores; callback exceptions propagate.
/// @throws std::bad_alloc On output copying.
Tensor score(const Tensor& images,const ImageScoreBatch& predict,ImageExplanationWork& work){
    ++work.predict_calls;work.evaluations+=images.size(0);auto out=predict(images);finite(out);
    if(out.dim()!=1||out.size(0)!=images.size(0))throw std::invalid_argument("image explanation: callback must return [B] scores");
    return cast(out.detach(),DType::Float64).clone();
}
/// @brief Score actual retained LIME masks in bounded chunks using the shared masker.
/// @param image Borrowed validated single image.
/// @param masks Owned actual Bool [N,P] perturbations.
/// @param index Spatial patch metadata.
/// @param g Validated grid.
/// @param o Common batch/baseline controls.
/// @param predict Fixed-target callback.
/// @param work Actual counters, updated during dispatch.
/// @return Owned actual Float64 [N] scores in original sample order.
Tensor score_masks(const Tensor& image,const Tensor& masks,const Tensor& index,const Grid& g,const ImagePatchOptions& o,const ImageScoreBatch& predict,ImageExplanationWork& work){
    std::vector<Tensor> chunks;const auto total=static_cast<std::size_t>(masks.size(0));
    for(std::size_t start=0;start<total;){const auto n=std::min(o.batch_size,total-start);
        chunks.push_back(score(mask(image,slice(masks,{static_cast<std::int64_t>(start),0},{static_cast<std::int64_t>(n),g.patches}),index,g,o.baseline),predict,work));start+=n;}
    return concat(chunks,0);
}
/// @brief Exact small combinatorial coefficient, requiring n<=19.
/// @param n Number of possible other players.
/// @param k Subset size no greater than n.
/// @return Integer binomial count, representable in size_t.
std::size_t choose(std::size_t n,std::size_t k){std::size_t value=1;for(std::size_t j=1;j<=k;++j)value=value*(n-j+1)/j;return value;}
} // namespace image_explanation_detail
Tensor mask_image_patches(const Tensor& image,const Tensor& masks,ImagePatchOptions options){
    NoGradGuard guard;const auto g=image_explanation_detail::image_grid(image,options);
    if(!masks.defined()||masks.device().backend!=compute::Backend::CPU||masks.dtype()!=DType::Bool||masks.dim()!=2||masks.size(0)<1||masks.size(1)!=g.patches)throw std::invalid_argument("image patches: positive Bool [B,P] flags required");
    image_explanation_detail::geometry({masks.size(0),g.h,g.w,g.channels});return image_explanation_detail::mask(image,masks,image_explanation_detail::patch_indices(g),g,options.baseline);
}
ExactImageShapley exact_image_shapley(const Tensor& image,const ImageScoreBatch& predict,ImagePatchOptions options){
    NoGradGuard guard;using namespace image_explanation_detail;const auto g=image_grid(image,options);
    if(g.patches>20)throw std::invalid_argument("exact image Shapley: at most 20 patches");
    const auto p=static_cast<std::size_t>(g.patches),m=std::size_t{1}<<p;query_geometry(g,m,options,predict,false);
    geometry({static_cast<std::int64_t>(std::min(m,options.batch_size)),g.patches});const auto index=patch_indices(g);ExactImageShapley out;
    std::vector<Tensor> chunks;
    for(std::size_t start=0;start<m;){const auto n=std::min(options.batch_size,m-start);std::vector<std::int32_t> flags(n*p);
        for(std::size_t row=0;row<n;++row)for(std::size_t j=0;j<p;++j)flags[row*p+j]=((start+row)>>j)&1;
        const auto z=cast(from_vector(flags,{static_cast<std::int64_t>(n),g.patches}),DType::Bool);
        chunks.push_back(score(mask(image,z,index,g,options.baseline),predict,out.work));start+=n;}
    out.coalition_scores=concat(chunks,0);out.base_value=select(out.coalition_scores,0,0).item();out.full_value=select(out.coalition_scores,0,m-1).item();
    std::vector<Tensor> phi;phi.reserve(p);
    for(std::size_t j=0;j<p;++j){const auto bit=std::size_t{1}<<j;std::vector<std::int64_t> absent,present;std::vector<double> weights;absent.reserve(m/2);present.reserve(m/2);weights.reserve(m/2);
        for(std::size_t s=0;s<m;++s)if(!(s&bit)){absent.push_back(s);present.push_back(s|bit);weights.push_back(1./double(p*choose(p-1,std::popcount(s))));}
        phi.push_back(computed(sum(from_vector(weights)*(gather(out.coalition_scores,from_vector(present))-gather(out.coalition_scores,from_vector(absent))))));out.work.marginal_terms+=absent.size();}
    out.patch_values=reshape(stack(phi),{g.rows,g.cols});out.efficiency_residual=scalar_checked(sum(out.patch_values).item()-scalar_checked(out.full_value-out.base_value));return out;
}
TensorImageLime lime_image(const Tensor& image,const ImageScoreBatch& predict,ImageLimeOptions options){
    NoGradGuard guard;using namespace image_explanation_detail;const auto g=image_grid(image,options.patches);
    if(!std::isfinite(options.kernel_width)||options.kernel_width<=0||!std::isfinite(options.ridge)||options.ridge<=0||options.samples<2||options.samples<=static_cast<std::size_t>(g.patches))throw std::invalid_argument("tensor image LIME: samples>=P+1, positive finite width/ridge required");
    query_geometry(g,options.samples,options.patches,predict,true);const auto n=static_cast<std::int64_t>(options.samples),p=g.patches;
    std::vector<std::int32_t> flags(geometry({n,p}));std::mt19937_64 rng(options.seed);
    for(std::int64_t row=0;row<n;++row)for(std::int64_t j=0;j<p;++j)flags[row*p+j]=row==0?1:row==1?0:rng()&1;
    TensorImageLime out;out.masks=cast(from_vector(flags,{n,p}),DType::Bool);
    out.scores=score_masks(image,out.masks,patch_indices(g),g,options.patches,predict,out.work);const auto z=cast(out.masks,DType::Float64);
    const auto distance=mean(1-z,{1})/options.kernel_width;out.proximity_weights=exp(-square(distance));
    const auto design=concat({ones({n,1},DType::Float64),z},1),weight=reshape(out.proximity_weights,{n,1});
    const auto penalty=concat({zeros({1},DType::Float64),full({p},options.ridge,DType::Float64)},0);
    const auto a=computed(matmul(design,design*weight,true,false)+matrix_diag(penalty));
    const auto rhs=computed(matmul(design,reshape(out.scores,{n,1})*weight,true,false));const auto beta=computed(cholesky_solve(cholesky(a),rhs));
    out.intercept=select(reshape(beta,{p+1}),0,0).item();out.patch_values=reshape(slice(beta,{1,0},{p,1}),{g.rows,g.cols}).clone();
    out.fitted_full=scalar_checked(out.intercept+sum(out.patch_values).item());out.full_value=select(out.scores,0,0).item();
    const auto fitted=reshape(computed(matmul(design,beta)),{n});const auto average=computed(out.full_value+sum(out.proximity_weights*(out.scores-out.full_value))/sum(out.proximity_weights));
    const auto sse=computed(sum(out.proximity_weights*square(out.scores-fitted))).item();const auto sst=computed(sum(out.proximity_weights*square(out.scores-average))).item();
    out.r_squared=sst>0?scalar_checked(1-sse/sst):0;return out;
}
Tensor expand_image_attributions(const Tensor& patch_values,std::int64_t height,std::int64_t width,bool distribute){
    NoGradGuard guard;using namespace image_explanation_detail;finite(patch_values);
    if(patch_values.dim()!=2)throw std::invalid_argument("image attribution map: [R,C] scores required");
    const auto g=grid(height,width,1,patch_values.size(0),patch_values.size(1));const auto index=patch_indices(g);auto flat=reshape(patch_values,{g.patches});
    if(distribute){const auto ids=index.to_vector<std::int64_t>();std::vector<double> areas(g.patches,0);for(auto id:ids)++areas[id];flat=flat/cast(from_vector(areas),patch_values.dtype());}
    return reshape(computed(gather(flat,index)),{1,height,width,1});
}
double attribution_correlation(const Tensor& first,const Tensor& second,bool positive_only){
    NoGradGuard guard;using namespace image_explanation_detail;finite(first);finite(second);
    if(first.shape()!=second.shape()||first.numel()<2)throw std::invalid_argument("attribution correlation: equal shapes with at least two entries required");
    auto a=cast(first,DType::Float64),b=cast(second,DType::Float64);if(positive_only){a=maximum(a,scalar(0,DType::Float64));b=maximum(b,scalar(0,DType::Float64));}
    const auto scale_a=max(abs(a)).item(),scale_b=max(abs(b)).item();if(scale_a==0||scale_b==0)return std::numeric_limits<double>::quiet_NaN();
    a=a/scale_a;b=b/scale_b;a=a-mean(a);b=b-mean(b);const auto norm_a=sum(square(a)).item(),norm_b=sum(square(b)).item();
    if(norm_a==0||norm_b==0)return std::numeric_limits<double>::quiet_NaN();return std::clamp(scalar_checked(sum(a*b).item()/std::sqrt(norm_a*norm_b)),-1.,1.);
}
graphics::Diagram diagram_image_maps(const Tensor& maps,std::int64_t columns,const std::vector<std::string>& labels,const std::vector<bool>& signed_maps,double max_width){
    NoGradGuard guard;using namespace image_explanation_detail;finite(maps);
    if(maps.dim()!=4||maps.size(3)!=1||columns<1||(!labels.empty()&&labels.size()!=static_cast<std::size_t>(maps.size(0)))||(!signed_maps.empty()&&signed_maps.size()!=static_cast<std::size_t>(maps.size(0)))||!std::isfinite(max_width)||max_width<=0)throw std::invalid_argument("image map diagram: invalid geometry, labels, flags or width");
    if(maps.numel()>1000000)throw std::length_error("image map diagram: one-million-pixel bound exceeded");
    const auto values=maps.to_vector<double>();const auto count=maps.size(0),h=maps.size(1),w=maps.size(2);graphics::Diagram figure(24,max_width);
    for(std::int64_t n=0;n<count;++n){const bool signed_map=signed_maps.empty()||signed_maps[n];double limit=0;
        if(signed_map){for(std::int64_t j=0;j<h*w;++j)limit=std::max(limit,std::abs(values[n*h*w+j]));if(limit==0)limit=1;}else limit=1;
        auto palette=graphics::diverging_colors(33);if(!signed_map){palette.clear();for(int grey=0;grey<256;++grey)palette.push_back({grey,grey,grey});}
        std::vector<double> normalized(h*w);for(std::int64_t j=0;j<h*w;++j)normalized[j]=signed_map?values[n*h*w+j]/limit:values[n*h*w+j];
        const auto colours=graphics::value_colors(normalized,palette,signed_map?-1:0,1);
        const double left=double(n%columns)*(double(w)+6),top=-double(n/columns)*(double(h)+7);
        figure.text({left+double(w)/2,top+1.5},labels.empty()?std::to_string(n):labels[n],19);
        for(std::int64_t row=0;row<h;++row)for(std::int64_t col=0;col<w;++col){const auto colour=colours[row*w+col];graphics::DiagramStroke stroke;stroke.color=colour;stroke.width=.05;const double x=left+col,y=top-row;figure.polygon({{x,y},{x+1,y},{x+1,y-1},{x,y-1}},colour,stroke);}
        const auto key= signed_map?graphics::diverging_colors(33):graphics::value_colors({0,.125,.25,.375,.5,.625,.75,.875,1},palette,0,1);
        for(std::size_t j=0;j<key.size();++j){const double x=left+double(w)*j/key.size(),next=left+double(w)*(j+1)/key.size(),y=top-h-1;graphics::DiagramStroke stroke;stroke.color=key[j];stroke.width=.05;figure.polygon({{x,y},{next,y},{next,y-.6},{x,y-.6}},key[j],stroke);}
        figure.text({left,top-h-2.5},signed_map?std::format("{:.3g}",-limit):"0",16).text({left+double(w)/2,top-h-2.5},signed_map?"0":"0.5",16).text({left+double(w),top-h-2.5},std::format("{:.3g}",limit),16);
    }
    return figure;
}
} // namespace datamunge::nn
