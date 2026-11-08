/// @file chacha20_analysis.cpp
/// @brief Native simulation-key expansion and preflighted actual ChaCha20 draw inspection.
#include <datamunge/random/chacha20_analysis.hpp>
#include <limits>
#include <stdexcept>
namespace datamunge::random {
std::array<std::uint32_t,8> chacha20_seed_key(std::uint64_t seed) {
  SplitMix64 seeder(seed);
  std::array<std::uint32_t,8> key{};
  for(std::size_t i=0;i<key.size();i+=2) {
    const auto word=seeder.next_u64();
    key[i]=static_cast<std::uint32_t>(word);
    key[i+1]=static_cast<std::uint32_t>(word>>32);
  }
  return key;
}
std::vector<ChaCha20Draw> chacha20_trace(ChaCha20& engine,std::size_t count) {
  std::vector<ChaCha20Draw> result;
  if(count>engine.remaining_u64() || count>result.max_size())
    throw std::length_error("chacha20_trace: requested raw draws exceed capacity");
  result.reserve(count);
  for(std::size_t i=0;i<count;++i) {
    const auto before=engine.remaining_u64();
    const auto counter=static_cast<std::uint32_t>(std::numeric_limits<std::uint32_t>::max()-(before-1)/8);
    const auto index=static_cast<std::size_t>((8-before%8)%8);
    result.push_back({counter,index,before,engine.next_u64()});
  }
  return result;
}
} // namespace datamunge::random
