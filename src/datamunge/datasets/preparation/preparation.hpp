#pragma once
/// @file preparation.hpp
/// @brief Private native tensor preparer entrypoints, with original ingress contracts.
namespace datamunge::datasets::detail {
/// @brief Convert verified audio ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_audio(int argc, char** argv);
/// @brief Convert verified depth ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_depth(int argc, char** argv);
/// @brief Convert verified flowers ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_flowers(int argc, char** argv);
/// @brief Convert verified medical ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_medical(int argc, char** argv);
/// @brief Convert verified mesh ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_mesh(int argc, char** argv);
/// @brief Convert verified multilabel ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_multilabel(int argc, char** argv);
/// @brief Convert verified ocr ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_ocr(int argc, char** argv);
/// @brief Convert verified optical_flow ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_optical_flow(int argc, char** argv);
/// @brief Convert verified pose ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_pose(int argc, char** argv);
/// @brief Convert verified signals ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_signals(int argc, char** argv);
/// @brief Convert verified style ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_style(int argc, char** argv);
/// @brief Convert verified text ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_text(int argc, char** argv);
/// @brief Convert verified two_stream ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_two_stream(int argc, char** argv);
/// @brief Convert verified video ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_video(int argc, char** argv);
/// @brief Convert verified voxel ingress metadata into native tensor archives.
/// @param argc Number of arguments, including the entrypoint name.
/// @param argv Entrypoint name, cache directory, and any dataset-specific paths.
/// @return Zero on success, nonzero after an ingress or tensor validation failure.
int prepare_voxel(int argc, char** argv);
/// @brief Convert human VQA source annotations into native tensor arrays.
/// @param argc Argument count including entrypoint name.
/// @param argv Entrypoint name and verified metadata directory.
/// @return Zero on success, nonzero on validation failure.
int prepare_vqa(int argc, char** argv);
} // namespace datamunge::datasets::detail
