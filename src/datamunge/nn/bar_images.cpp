/// @file bar_images.cpp
/// @brief Validated noisy horizontal/vertical image generation with true masks and local shuffle state.
#include <datamunge/nn/bar_images.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
namespace datamunge::nn {
BarImageBatch make_bar_images(std::int64_t per_class,std::uint64_t seed,BarImageOptions options) {
  if(per_class<1 || options.side<1 || options.thickness<1 || options.thickness>options.side || !std::isfinite(options.noise_stddev) || options.noise_stddev<0 || (options.dtype!=DType::Float32 && options.dtype!=DType::Float64))throw std::invalid_argument("bar images: invalid count, geometry, noise or floating dtype");
  const auto last=options.max_position==-1?options.side-options.thickness:options.max_position;
  if(options.min_position<0 || last<options.min_position || last>options.side-options.thickness)throw std::invalid_argument("bar images: invalid position interval");
  if(per_class>std::numeric_limits<std::int64_t>::max()/2)throw std::length_error("bar images: sample count overflow");
  const auto rows=2*per_class;
  std::int64_t count;
  try { count=shape_numel({rows,options.side,options.side,1}); }
  catch(const std::overflow_error&) { throw std::length_error("bar images: tensor dimensions overflow"); }
  std::vector<double> pixels(static_cast<std::size_t>(count)),mask(static_cast<std::size_t>(count));std::vector<std::int64_t> labels(rows),positions(rows);
  std::mt19937_64 engine(seed);std::uniform_int_distribution<std::int64_t> position(options.min_position,last);
  for(std::int64_t row=0;row<rows;++row) {
    labels[row]=row%2;positions[row]=position(engine);std::normal_distribution<double> noise(0,options.noise_stddev>0?options.noise_stddev:1);
    for(std::int64_t y=0;y<options.side;++y)for(std::int64_t x=0;x<options.side;++x) {
      const auto coordinate=labels[row]?x:y;const bool occupied=coordinate>=positions[row] && coordinate<positions[row]+options.thickness;
      const auto index=(row*options.side+y)*options.side+x;mask[index]=occupied?1:0;
      pixels[index]=std::clamp(double(occupied)+(options.noise_stddev>0?noise(engine):0),0.0,1.0);
    }
  }
  BarImageBatch out;out.images=from_vector(pixels,{rows,options.side,options.side,1}).to(options.dtype);out.object_mask=from_vector(mask,{rows,options.side,options.side,1}).to(options.dtype);out.labels=from_vector(labels);out.positions=from_vector(positions);out.shuffled_rows.resize(rows);std::iota(out.shuffled_rows.begin(),out.shuffled_rows.end(),0);std::shuffle(out.shuffled_rows.begin(),out.shuffled_rows.end(),engine);return out;
}
}
