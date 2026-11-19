#pragma once

/// @file board_replay.hpp
/// @brief Movie export from recorded grid-game actions.

#include <algorithm>
#include <format>
#include <stdexcept>
#include <string>
#include <vector>

#include "datamunge/cad/video.hpp"
#include "datamunge/rl/environments/board_games.hpp"

namespace datamunge::rl::environments {

/// @brief Replay grid-game actions on a clone and encode the board as an H.264 movie.
///
/// The initial board and every resulting board get one frame. Pieces have
/// distinct colors and symbols. The input game is unchanged. Requires ffmpeg
/// with libx264 on PATH.
/// @param initial Initial position of any grid game with at most 16 rows and columns.
/// @param actions Chronological legal actions, including pass actions where applicable.
/// @param destination MP4 path with an existing parent directory.
/// @param fps Frame rate in [1,240], default 4.
/// @throws std::invalid_argument Unsupported board dimensions or invalid movie settings.
/// @throws std::logic_error An action is illegal or follows a terminal position.
/// @throws std::runtime_error Encoder failure.
/// @par Example
/// @code
/// namespace e = dm::rl::environments;
/// e::TicTacToe game;
/// game.Reset();
/// e::save_board_replay(game, {4,0,8}, "game.mp4");
/// @endcode
/// @see GridGame, cad::VideoWriter
inline void save_board_replay(const GridGame& initial, const std::vector<int>& actions,
                             const std::string& destination, int fps = 4) {
  const int rows = initial.rows(), cols = initial.cols();
  if (rows < 1 || cols < 1 || rows > 16 || cols > 16)
    throw std::invalid_argument("unsupported replay board geometry");
  auto state = initial.Clone();
  auto& board = dynamic_cast<GridGame&>(*state);
  cad::RenderResult frame;
  frame.width = 480;
  frame.height = 480;
  frame.rgba.resize(frame.width * frame.height * 4);
  const auto rectangle = [&frame](int x, int y, int width, int height,
                                  unsigned char r, unsigned char g, unsigned char b) {
    for (int row = y; row < y + height; ++row)
      for (int col = x; col < x + width; ++col) {
        auto* pixel = frame.rgba.data() + 4 * (row * frame.width + col);
        pixel[0] = r;
        pixel[1] = g;
        pixel[2] = b;
        pixel[3] = 255;
      }
  };
  const int cell = std::min(420 / cols, 388 / rows);
  const int top = 65, left = (frame.width - (cols * cell)) / 2;
  cad::VideoWriter video(destination, frame.width, frame.height, fps);
  const auto capture = [&](std::size_t ply) {
    rectangle(0, 0, frame.width, frame.height, 15, 23, 42);
    const auto status = board.Terminal()
        ? (board.Winner() == 0 ? std::string("Draw") : std::format("Player {} wins", board.Winner()))
        : std::format("Next: player {}", board.CurrentPlayer());
    cad::draw_text(frame, std::format("Move {}   {}", ply, status), 20, 22, 2);
    for (int row = 0; row < rows; ++row)
      for (int col = 0; col < cols; ++col) {
        const int x = left + col * cell, y = top + row * cell;
        const int piece = board.At(row, col);
        if (piece == 1) rectangle(x, y, cell - 2, cell - 2, 29, 101, 177);
        else if (piece == 2) rectangle(x, y, cell - 2, cell - 2, 161, 63, 39);
        else rectangle(x, y, cell - 2, cell - 2, 57, 71, 88);
        if (piece != 0)
          cad::draw_text(frame, piece == 1 ? "X" : "O", x + (cell - 10) / 2,
                         y + (cell - 16) / 2);
      }
    video.append(frame);
  };
  capture(0);
  for (std::size_t i = 0; i < actions.size(); ++i) {
    const auto legal = board.LegalActions();
    if (board.Terminal() || std::find(legal.begin(), legal.end(), actions[i]) == legal.end())
      throw std::logic_error("illegal action in board replay");
    board.Apply(actions[i]);
    capture(i + 1);
  }
  video.finish();
}
}  // namespace datamunge::rl::environments
