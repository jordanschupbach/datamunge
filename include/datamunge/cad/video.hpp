#pragma once
/// @file video.hpp
/// @brief Bounded raw-RGBA streaming to an independently owned ffmpeg encoder process.
#include <datamunge/cad/render.hpp>
#include <memory>
#include <string>
namespace datamunge::cad {
/// @brief Stream rendered frames directly into a silent H.264 MP4 without intermediate PNGs.
/// Requires ffmpeg with libx264 on PATH. Uses a bounded OS stream with synchronous
/// backpressure; there is no unbounded queue or shell command interpolation.
/// Encoding uses baseline/yuv420p, CRF19, fast, no B frames and a two-second GOP.
/// Input alpha is discarded by the encoder. One writer is single-threaded;
/// independent writers may coexist. Call finish() to publish the completed file;
/// destruction closes the child and discards unfinalized temporary output.
/// Existing destinations are retained until successful finalization.
/// @par Example
/// @code
/// dm::cad::RenderResult frame;frame.width=64;frame.height=48;
/// frame.rgba.assign(64*48*4,255);
/// dm::cad::VideoWriter video("example.mp4",64,48,15);
/// video.append(frame);video.append(frame);video.finish();
/// @endcode
/// @par References
/// - FFmpeg's rawvideo input options and explicit pixel format/dimensions/frame rate:
///   https://github.com/FFmpeg/FFmpeg/blob/master/doc/demuxers.texi
/// @see RenderResult, render()
class VideoWriter {
 public:
  /// @brief Start a raw RGBA encoder, writing a temporary file beside the destination.
  /// @param path Destination MP4 filename; its parent directory must exist.
  /// @param width Even image width, 2..8192 pixels.
  /// @param height Even image height, 2..8192 pixels.
  /// @param fps Integer frame rate, 1..240; default 15.
  /// @throws std::invalid_argument Empty/NUL-containing filename or invalid dimensions/rate.
  /// @throws std::runtime_error Temporary-file or child-process startup failure.
  /// @par Example
  /// See the class example.
  VideoWriter(const std::string& path,int width,int height,int fps=15);
  /// @brief Close the child process without throwing; discard output unless finish() succeeded.
  ~VideoWriter();
  /// @brief Copying a live encoder is disabled.
  /// @param other Encoder that cannot be copied.
  VideoWriter(const VideoWriter& other)=delete;
  /// @brief Assignment of a live encoder is disabled.
  /// @param other Encoder that cannot be assigned.
  /// @return No value; deleted operation.
  VideoWriter& operator=(const VideoWriter& other)=delete;
  /// @brief Append one top-left row-major RGBA8 frame, blocking when the encoder falls behind.
  /// Depth/ID buffers are ignored. Invalid input leaves the stream/count unchanged.
  /// @param frame Exact configured dimensions and width*height*4 owned RGBA bytes.
  /// @throws std::invalid_argument Incorrect dimensions or RGBA buffer size.
  /// @throws std::logic_error Already finalized or closed encoder.
  /// @throws std::runtime_error Failed encoder or stream write, without a SIGPIPE process exit.
  /// @par Example
  /// See the class example.
  void append(const RenderResult& frame);
  /// @brief Close input, wait for successful encoding, and publish the completed MP4.
  /// Repeated successful calls are harmless. Failure preserves an existing target.
  /// @throws std::logic_error No complete frames have been appended.
  /// @throws std::runtime_error Encoder failure or inability to publish the temporary file.
  /// @par Example
  /// See the class example.
  void finish();
  /// @brief Count successfully appended complete frames.
  /// @return Number of frames written to the input stream.
  std::size_t frames_written()const;
 private:
  struct Impl; ///< Exclusive process, bounded transport and temporary-output ownership.
  std::unique_ptr<Impl> impl_; ///< Never shared across writers.
};
} // namespace datamunge::cad
