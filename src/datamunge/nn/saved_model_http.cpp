/// @file saved_model_http.cpp
/// @brief Bounded loopback HTTP serving using Libevent.

#include <datamunge/nn/saved_model_http.hpp>

#include <stdexcept>
#include <utility>

#ifdef DATAMUNGE_ENABLE_SAVED_MODEL_HTTP
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <string_view>

#include <event2/buffer.h>
#include <event2/event.h>
#include <event2/http.h>
#include <event2/util.h>
#ifndef _WIN32
#include <netinet/in.h>
#include <sys/socket.h>
#endif
#endif

namespace datamunge::nn {

bool saved_model_http_available() noexcept {
#ifdef DATAMUNGE_ENABLE_SAVED_MODEL_HTTP
  return true;
#else
  return false;
#endif
}

/// @brief Retain the HTTP listener and model for callback dispatch.
struct SavedModelHttpServer::Impl {
#ifdef DATAMUNGE_ENABLE_SAVED_MODEL_HTTP
  std::unique_ptr<event_base, decltype(&event_base_free)> loop{event_base_new(), event_base_free}; ///< Event loop.
  std::unique_ptr<evhttp, decltype(&evhttp_free)> http{nullptr, evhttp_free}; ///< Listener owned by its loop.
  SavedModel model; ///< Loaded affine serving signature.
  std::size_t remaining{}; ///< Replies left before a bounded run stops.
#endif
  std::uint16_t port{}; ///< Actual loopback port.

#ifdef DATAMUNGE_ENABLE_SAVED_MODEL_HTTP
  /// @brief Create a listener with bounded request parsing on IPv4 loopback.
  Impl(SavedModel value, std::uint16_t requested_port) : model(std::move(value)) {
    if (model.input_shape() != Shape{-1, 2} || model.output_shape() != Shape{-1, 1})
      throw std::invalid_argument("SavedModel HTTP: incompatible affine signature");
    if (loop) http.reset(evhttp_new(loop.get()));
    if (!http) throw std::runtime_error("SavedModel HTTP: could not create listener");
    evhttp_set_max_headers_size(http.get(), 8192);
    evhttp_set_max_body_size(http.get(), 65536);
    evhttp_set_timeout(http.get(), 3);
    evhttp_set_gencb(http.get(), &Impl::handle, this);
    auto* bound = evhttp_bind_socket_with_handle(http.get(), "127.0.0.1", requested_port);
    if (!bound) throw std::runtime_error("SavedModel HTTP: could not bind loopback");
    sockaddr_in address{};
    ev_socklen_t size = sizeof(address);
    if (getsockname(evhttp_bound_socket_get_fd(bound), reinterpret_cast<sockaddr*>(&address), &size) != 0)
      throw std::runtime_error("SavedModel HTTP: could not determine bound port");
    port = ntohs(address.sin_port);
  }

  /// @brief Parse finite pairs and run one batched model invocation.
  std::string predict(std::string_view body) const {
    std::istringstream input{std::string(body)};
    input.imbue(std::locale::classic());
    std::vector<float> features;
    input >> std::ws;
    while (!input.eof()) {
      float a, b;
      if (!(input >> a >> b) || !std::isfinite(a) || !std::isfinite(b) || features.size() == 2048)
        throw std::invalid_argument("expected up to 1024 finite feature pairs");
      features.insert(features.end(), {a, b});
      input >> std::ws;
    }
    if (features.empty()) throw std::invalid_argument("at least one pair is required");
    const auto rows = static_cast<std::int64_t>(features.size() / 2);
    const auto prediction = model.predict(from_vector(features, {rows, 2}));
    if (prediction.shape() != Shape{rows, 1})
      throw std::runtime_error("SavedModel HTTP: unexpected prediction shape");
    std::ostringstream reply;
    reply.imbue(std::locale::classic());
    reply << std::setprecision(std::numeric_limits<float>::max_digits10);
    for (const auto value : prediction.to_vector<float>()) reply << value << '\n';
    return reply.str();
  }

  /// @brief Stop after the final response has been sent.
  static void complete(evhttp_request*, void* context) noexcept {
    auto& server = *static_cast<Impl*>(context);
    if (server.remaining && --server.remaining == 0)
      event_base_loopbreak(server.loop.get());
  }

  /// @brief Route the health and prediction endpoints with text responses.
  static void handle(evhttp_request* req, void* context) noexcept {
    const auto& server = *static_cast<Impl*>(context);
    evhttp_request_set_on_complete_cb(req, &Impl::complete, context);
    try {
      int status = 200;
      std::string response;
      const auto method = evhttp_request_get_command(req);
      const std::string_view target(evhttp_request_get_uri(req));
      if (method == EVHTTP_REQ_GET && target == "/healthz") {
        response = "ok\n";
      } else if (method == EVHTTP_REQ_POST && target == "/predict") {
        try {
          auto* headers = evhttp_request_get_input_headers(req);
          if (!evhttp_find_header(headers, "Content-Length") ||
              evhttp_find_header(headers, "Transfer-Encoding"))
            throw std::invalid_argument("use Content-Length");
          auto* input = evhttp_request_get_input_buffer(req);
          const auto size = evbuffer_get_length(input);
          if (!size) throw std::invalid_argument("at least one pair is required");
          const auto* data = evbuffer_pullup(input, -1);
          if (!data) throw std::runtime_error("could not read request body");
          response = server.predict({reinterpret_cast<const char*>(data), size});
        } catch (const std::invalid_argument& error) {
          status = 400;
          response = std::string(error.what()) + '\n';
        }
      } else {
        status = 404;
        response = "unknown route\n";
      }
      std::unique_ptr<evbuffer, decltype(&evbuffer_free)> output(evbuffer_new(), evbuffer_free);
      auto* headers = evhttp_request_get_output_headers(req);
      if (!output || evbuffer_add(output.get(), response.data(), response.size()) != 0 ||
          evhttp_add_header(headers, "Content-Type", "text/plain; charset=utf-8") != 0 ||
          evhttp_add_header(headers, "Connection", "close") != 0)
        throw std::runtime_error("could not prepare response");
      evhttp_send_reply(req, status, nullptr, output.get());
    } catch (...) {
      evhttp_send_error(req, 500, "inference failed");
    }
  }
#endif
};

SavedModelHttpServer::SavedModelHttpServer(SavedModel model, std::uint16_t port) {
#ifdef DATAMUNGE_ENABLE_SAVED_MODEL_HTTP
  impl_ = std::make_unique<Impl>(std::move(model), port);
#else
  (void)model;
  (void)port;
  throw std::runtime_error("SavedModel HTTP: rebuild with DATAMUNGE_ENABLE_SAVED_MODEL_HTTP=ON");
#endif
}

SavedModelHttpServer::~SavedModelHttpServer() = default;

std::uint16_t SavedModelHttpServer::port() const noexcept { return impl_->port; }

void SavedModelHttpServer::serve(std::size_t requests) {
#ifdef DATAMUNGE_ENABLE_SAVED_MODEL_HTTP
  impl_->remaining = requests;
  const timeval timeout{10, 0};
  // A scoped timer can be canceled when the reply limit ends the run.
  auto deadline_callback = [](evutil_socket_t, short, void* loop) {
    event_base_loopbreak(static_cast<event_base*>(loop));
  };
  std::unique_ptr<event, decltype(&event_free)> deadline(nullptr, event_free);
  if (requests) {
    deadline.reset(evtimer_new(impl_->loop.get(), deadline_callback, impl_->loop.get()));
    if (!deadline || evtimer_add(deadline.get(), &timeout) != 0)
      throw std::runtime_error("SavedModel HTTP: could not set test deadline");
  }
  if (event_base_dispatch(impl_->loop.get()) != 0 || impl_->remaining)
    throw std::runtime_error("SavedModel HTTP: event loop failed or timed out");
#else
  (void)requests;
  throw std::runtime_error("SavedModel HTTP: rebuild with DATAMUNGE_ENABLE_SAVED_MODEL_HTTP=ON");
#endif
}

} // namespace datamunge::nn
