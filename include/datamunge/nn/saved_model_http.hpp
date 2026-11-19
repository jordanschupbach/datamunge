#pragma once

/// @file saved_model_http.hpp
/// @brief Local HTTP adapter for the two-feature affine SavedModel contract.

#include <cstddef>
#include <cstdint>
#include <memory>

#include <datamunge/nn/saved_model.hpp>

namespace datamunge::nn {

/// @brief Report whether the loopback HTTP serving adapter is compiled in.
/// @return True when TensorFlow and Libevent support are enabled.
/// @par Example
/// @code
/// std::println("{}", dm::nn::saved_model_http_available());
/// @endcode
/// @see SavedModelHttpServer
bool saved_model_http_available() noexcept;

/// @brief Serve the affine Float32 SavedModel signature on IPv4 loopback.
///
/// GET /healthz returns ok. POST /predict accepts whitespace-separated finite
/// feature pairs and returns one prediction per line. Requests are handled
/// sequentially with bounded headers, bodies, and row counts.
/// @par Example
/// @code
/// const dm::nn::SavedModel model("saved_model");
/// dm::nn::SavedModelHttpServer server(model);
/// std::println("{}", server.port());
/// @endcode
/// @see SavedModel, saved_model_http_available()
class SavedModelHttpServer {
 public:
  /// @brief Bind a local port and retain a copy of a loaded model.
  /// @param model Dynamic Float32 [batch, 2] to [batch, 1] SavedModel.
  /// @param port Local port, default zero to select an ephemeral port.
  /// @throws std::invalid_argument Incompatible serving shape.
  /// @throws std::runtime_error Disabled backend or listener setup failure.
  /// @par Example
  /// @code
  /// dm::nn::SavedModelHttpServer server(dm::nn::SavedModel("saved_model"), 8080);
  /// @endcode
  explicit SavedModelHttpServer(SavedModel model, std::uint16_t port = 0);

  /// @brief Close the listener and free the HTTP event loop.
  ~SavedModelHttpServer();
  /// @brief Return the actual bound IPv4 loopback port.
  /// @return Bound port, including the assigned port when constructed with zero.
  [[nodiscard]] std::uint16_t port() const noexcept;

  /// @brief Run the event loop on the calling thread.
  /// @param requests Completed routed replies before stopping, or zero to serve continuously.
  /// A finite run times out after ten seconds. Call from only one thread at a time.
  /// @throws std::runtime_error Event loop error or finite run timeout.
  /// @par Example
  /// @code
  /// dm::nn::SavedModelHttpServer server(dm::nn::SavedModel("saved_model"));
  /// // A separate client connects to server.port() and sends one request.
  /// server.serve(1);
  /// @endcode
  void serve(std::size_t requests = 0);

 private:
  /// @brief Own the listener, event loop and inference model.
  struct Impl;
  std::unique_ptr<Impl> impl_; ///< Exclusive server ownership.
};

} // namespace datamunge::nn
