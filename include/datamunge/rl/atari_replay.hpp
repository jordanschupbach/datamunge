#pragma once

/// @file atari_replay.hpp
/// @brief Byte-exact replay for frame-major Atari observations.

#include <datamunge/rl/environments/atari_frames.hpp>
#include <datamunge/rl/replay.hpp>

#include <array>
#include <memory>

namespace datamunge::rl::replay {

/// @brief Bounded ring of Atari transitions with shared immutable byte frames.
/// AtariFrameStack observations survive packing and unpacking exactly. The
/// source and successor stacks retain their frame order, including across resets.
/// Sampling draws ring slots with replacement, using the ordinary replay order.
/// @par Example
/// @code
/// dm::rl::replay::AtariReplay buffer(32);
/// dm::rl::environments::Obs pixels(4*84*84,0.f);
/// buffer.Add({pixels,0,0.f,pixels,0.f,{}});
/// std::mt19937_64 rng(0);
/// const auto batch=buffer.Sample(1,rng);
/// @endcode
/// @see ReplayBuffer, environments::AtariFrameStack
class AtariReplay {
public:
  /// @brief Construct an initially empty Atari ring.
  /// @param capacity Positive maximum stored transitions, up to 100,000.
  /// @throws std::invalid_argument If capacity is outside the supported range.
  /// @see AtariReplay
  explicit AtariReplay(int capacity) : capacity_(capacity) {
    if (capacity<1 || capacity>100000) throw std::invalid_argument("invalid Atari replay capacity");
    entries_.reserve(capacity);
  }

  /// @brief Store one discrete transition, overwriting the oldest when full.
  /// @param transition Source and successor from AtariFrameStack, action, reward and terminal flag.
  /// @throws std::invalid_argument For incompatible observations or transition metadata.
  /// @see AtariReplay
  void Add(const Experience& transition) {
    if (!transition.act_c.empty()) throw std::invalid_argument("Atari replay requires a discrete action");
    detail::ExperienceValues(transition,width);
    const Stack* previous=entries_.empty()?nullptr:&entries_[(next_+entries_.size()-1)%entries_.size()].next;
    auto state=pack(transition.s,previous,0);
    auto next=pack(transition.s2,&state,1);
    Entry entry{std::move(state),std::move(next),transition.a,transition.r,transition.done};
    if (static_cast<int>(entries_.size())<capacity_) entries_.push_back(std::move(entry));
    else entries_[next_]=std::move(entry);
    next_=(next_+1)%capacity_;
  }

  /// @brief Return the number of currently stored transitions.
  /// @return Occupied ring slots.
  /// @see AtariReplay
  int size() const { return static_cast<int>(entries_.size()); }

  /// @brief Sample uniformly with replacement and decode into owning Float32 tensors.
  /// @param count Number of draws, positive and no larger than occupancy or 2,267.
  /// @param rng Caller-owned random engine advanced once per draw.
  /// @return Standard replay minibatch with byte-exact Atari observations.
  /// @throws std::invalid_argument For an invalid sample count.
  /// @see ReplayBuffer::Sample()
  Batch Sample(int count,std::mt19937_64& rng) const {
    detail::SampleSize(count,size());
    if (count>64000000/width) throw std::invalid_argument("Atari minibatch exceeds replay tensor limit");
    std::vector<Experience> decoded;
    std::vector<int> indices;
    decoded.reserve(count);
    indices.reserve(count);
    for (int i=0;i<count;++i) {
      const auto slot=rng()%entries_.size();
      const auto& entry=entries_[slot];
      indices.push_back(static_cast<int>(slot));
      decoded.push_back({unpack(entry.state),entry.action,entry.reward,unpack(entry.next),entry.done,{}});
    }
    std::vector<const Experience*> rows;
    rows.reserve(count);
    for (const auto& transition:decoded) rows.push_back(&transition);
    return MakeBatch(rows,width,std::vector<float>(count,1.f),std::move(indices));
  }

private:
  /// @brief Pixel count in one preprocessed frame.
  static constexpr int pixels=environments::AtariFrameStack::kSize*environments::AtariFrameStack::kSize;
  /// @brief Number of consecutive frames in a stack.
  static constexpr int frames=environments::AtariFrameStack::kStack;
  /// @brief Number of Float32 values in a decoded observation.
  static constexpr int width=pixels*frames;
  /// @brief Row-major grayscale bytes from one observation frame.
  using Frame=std::array<unsigned char,pixels>;
  /// @brief Shared immutable physical frames in observation order.
  using Stack=std::array<std::shared_ptr<const Frame>,frames>;
  /// @brief One packed transition, independent of the lifetime of neighboring slots.
  struct Entry {
    Stack state; ///< Source frame references.
    Stack next; ///< Successor frame references.
    int action; ///< Discrete action ID.
    float reward; ///< Reward used for the update.
    float done; ///< Terminal-only bootstrap mask.
  };
  /// @brief Encode pixels while sharing unchanged frames at the expected shift.
  /// @param values Validated Float32 observation with width coordinates.
  /// @param previous Candidate frame stack, or null for a fresh observation.
  /// @param shift Candidate index offset, zero for a repeated stack and one for a successor.
  /// @return Stack of owned or shared immutable byte frames.
  static Stack pack(const environments::Obs& values,const Stack* previous,int shift) {
    Stack result;
    for (int i=0;i<frames;++i) {
      Frame frame;
      for (int j=0;j<pixels;++j) {
        const auto value=values[i*pixels+j];
        if (!(value>=0 && value<=1)) throw std::invalid_argument("Atari replay expects unit-scale pixels");
        const auto byte=static_cast<unsigned char>(255.f*value+.5f);
        if (byte/255.f!=value) throw std::invalid_argument("Atari replay expects exact byte normalization");
        frame[j]=byte;
      }
      if (previous && i+shift<frames && *(*previous)[i+shift]==frame) {
        result[i]=(*previous)[i+shift];
      } else if (i && *result[i-1]==frame) {
        result[i]=result[i-1];
      } else {
        result[i]=std::make_shared<Frame>(frame);
      }
    }
    return result;
  }
  /// @brief Restore one stack in the same order and scale as AtariFrameStack.
  /// @param stack Packed shared frame references.
  /// @return Independent normalized observation.
  static environments::Obs unpack(const Stack& stack) {
    environments::Obs pixels_out(width);
    for (int i=0;i<frames;++i)
      for (int j=0;j<pixels;++j) pixels_out[i*pixels+j]=(*stack[i])[j]/255.f;
    return pixels_out;
  }

  int capacity_; ///< Ring capacity.
  std::vector<Entry> entries_; ///< Occupied slots, in physical order.
  std::size_t next_=0; ///< Next insertion slot.
};

} // namespace datamunge::rl::replay
