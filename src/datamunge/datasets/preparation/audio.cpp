/// @file audio.cpp
/// @brief Native audio tensor preparation from verified source metadata.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <dm/dm.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>

#include "preparation.hpp"
/// @brief Convert exact original WAVs to tensor mu-law windows and held-out audio prompts.
/// @param argc Program name and pinned audio-source metadata root.
/// @param argv Directory containing source_files.tsv and original recordings.
/// @return Zero after all source-disjoint input/next-code arrays and true prompts are saved.
int datamunge::datasets::detail::prepare_audio(int argc, char** argv) {
  try {
    if (argc != 2)
      throw std::invalid_argument("expected original audio source root");
    namespace n = dm::nn;
    n::set_num_threads(2);
    n::NoGradGuard              guard;
    const std::filesystem::path root = argv[1];
    std::ifstream               file(root / "source_files.tsv");
    std::map<std::string, std::map<std::string, std::vector<n::Tensor>>> parts;
    std::vector<n::Tensor>                                               prompts;
    std::vector<int>                                                     prompt_ids;
    std::string                                                          split, path;
    int                                                                  source, speaker;
    while (file >> split >> source >> speaker >> path) {
      const auto wav = n::read_wav((root / path).string());
      if (wav.sample_rate != 8000 || wav.samples.dim() != 2 || wav.samples.size(1) != 1 ||
          wav.samples.size(0) < 1025)
        throw std::invalid_argument("original8kHz mono audio of at least1025samples required");
      const auto audio = n::slice_axis(wav.samples, 0, 0, 1025).reshape({1025});
      const auto codes = n::mu_law_encode(audio.to(n::DType::Float64), 256);
      for (int start : {0, 256, 512, 768}) {
        parts[split]["input"].push_back(n::slice_axis(codes, 0, start, start + 256));
        parts[split]["target"].push_back(n::slice_axis(codes, 0, start + 1, start + 257));
        parts[split]["source_id"].push_back(n::full({}, source, n::DType::Int32));
        parts[split]["speaker_id"].push_back(n::full({}, speaker, n::DType::Int32));
        parts[split]["start"].push_back(n::full({}, start, n::DType::Int32));
      }
      if (split == "test" && prompts.size() < 3) {
        prompts.push_back(n::slice_axis(codes, 0, 0, 384));
        prompt_ids.push_back(source);
      }
    }
    n::NamedTensors arrays;
    for (const std::string split : {"fit", "validation", "test"})
      for (const std::string key : {"input", "target", "source_id", "speaker_id", "start"})
        arrays.push_back({split + "_" + key, n::stack(parts.at(split).at(key), 0)});
    arrays.push_back({"prompts", n::stack(prompts, 0)});
    arrays.push_back({"prompt_source_ids", n::from_vector(prompt_ids)});
    n::save_npz((root / "wavenet_stack.npz").string(), arrays);
    std::cout << "Prepared genuine FSDD72/12/24speaker-disjoint windows and3original held-out "
                 "prompts\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
