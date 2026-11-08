/// @file vision_plots.cpp
/// @brief Native tensor image/contact-sheet diagrams with square cells and no embedded raster payload.
#include <datamunge/nn/vision_plots.hpp>
namespace datamunge::nn {
graphics::Diagram diagram_tensor_images(const Tensor& batch,std::int64_t columns,const std::vector<std::string>& labels,double low,double high,double pixel_size,double max_width) {
  const long double span=static_cast<long double>(high)-low;
  if(!batch.defined() || batch.dim()!=4 || batch.numel()<1 || batch.size(0)<1 || batch.size(1)<1 || batch.size(2)<1 || (batch.size(3)!=1 && batch.size(3)!=3) || columns<1 || (!labels.empty() && labels.size()!=static_cast<std::size_t>(batch.size(0))) || !std::isfinite(low) || !std::isfinite(high) || !(high>low) || !std::isfinite(span) || !std::isfinite(pixel_size) || pixel_size<=0 || !std::isfinite(max_width) || max_width<=0)throw std::invalid_argument("tensor image diagram: incompatible geometry, labels, limits or scale");
  const auto values=batch.to_vector<double>();for(auto value:values)if(!std::isfinite(value))throw std::invalid_argument("tensor image diagram: finite values required");
  const auto count=batch.size(0),height=batch.size(1),width=batch.size(2),channels=batch.size(3);graphics::Diagram figure(pixel_size,max_width);
  /// @brief Convert a validated finite channel through clipping and rounded byte scaling.
  /// @param value Finite raw channel intensity.
  /// @return Byte intensity, with low black and high white.
  auto channel=[low,span](double value){return static_cast<std::uint8_t>(std::lround(255*std::clamp((static_cast<long double>(value)-low)/span,0.0L,1.0L)));};
  for(std::int64_t n=0;n<count;++n) {
    const double left=double(n%columns)*(double(width)+2),top=-double(n/columns)*(double(height)+4);
    figure.text({left+double(width)/2,top+1.25},labels.empty()?std::to_string(n):labels[n],16);
    for(std::int64_t row=0;row<height;++row)for(std::int64_t col=0;col<width;++col) {
      const auto offset=((n*height+row)*width+col)*channels;const auto red=channel(values[offset]);
      const graphics::RGB colour{red,channels==1?red:channel(values[offset+1]),channels==1?red:channel(values[offset+2])};graphics::DiagramStroke stroke;stroke.color=colour;stroke.width=.05;
      const auto x=left+col,y=top-row;figure.polygon({{x,y},{x+1,y},{x+1,y-1},{x,y-1}},colour,stroke);
    }
  }
  return figure;
}
}
