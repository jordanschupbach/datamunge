// CMD: just run-nn learned_image_codec
// FrozenTHREEepochAE/0newupdates. Prepare:just autoencoder-assets;just run undercomplete_autoencoder.
//
// Port of tflow's learned_image_codec (helpers: examples/nn/common/learned_image_codec.hpp).
// Data: $DATAMUNGE_DATA_DIR/saliency. The frozen model is read from encoder=DIR: best.npz of
// our undercomplete_autoencoder port (default output/nn/models/undercomplete_autoencoder) or
// tflow's model_arrays.npz when pointed at its output/models/undercomplete_autoencoder. The
// TFLCODE1 bitstream format is unchanged, so files written by tflow decode here and vice versa
// (decode=FILE). The lossless PNG baseline uses datamunge's PNG writer, which emits STORED
// (uncompressed) deflate blocks: 880 bytes per 28x28 image, 176000 in total versus 55568 for
// tflow's zlib-compressed EncodePng files, so the PNG rate here is a much weaker baseline (the
// page says so). uint8 symbols are
// stored as int32 arrays in analysis.npz / quantizer_algebra.npz (nn has no uint8 dtype).
// Outputs (test.tflcodec, analysis.npz, model_arrays.npz, quantizer_algebra.npz, original_png/,
// png_sizes.tsv, metrics.json, index.html, case_<k>.png) go to output/nn/models/learned_image_codec.
// Options: data=PATH encoder=PATH out=PATH, or decode=FILE [out=PATH] (bitstream only).

#include <fstream>
#include <iostream>
#include <iterator>

#include "common/learned_image_codec.hpp"

namespace {

namespace nn = datamunge::nn;

void Unit(const std::filesystem::path& root) {
  std::vector<float> fit(64, .25f), probe(96, .25f);
  fit[0]       = 0;
  fit[32]      = 1;
  probe[0]     = -.2f;
  probe[32]    = .5f;
  probe[64]    = 1.2f;
  auto fit_t   = nn::from_vector(fit, {2, 32}), probe_t = nn::from_vector(probe, {3, 32});
  auto q       = ae::FitCodecQuantizer(fit_t);
  auto symbols = ae::QuantizeCodec(probe_t, q);
  auto dequantized = ae::DequantizeCodec(symbols, q);
  check::Require(symbols[0] == 0 && symbols[32] == 128 && symbols[64] == 255, "quantclipping/half-upfixture");
  const auto d = dequantized.to_vector<float>();
  for (int b = 0; b < 3; ++b)
    for (int j = 1; j < 32; ++j) {
      check::Require(symbols[b * 32 + j] == 0, "constantchannelzerosymbol");
      check::Near(d[b * 32 + j], .25, 0, "constantchanneldequantEXACT");
    }
  nn::save_npz((root / "quantizer_algebra.npz").string(), {{"fit", fit_t},
                                                            {"probe", probe_t},
                                                            {"minimum", nn::from_vector(q.minimum, {32})},
                                                            {"maximum", nn::from_vector(q.maximum, {32})},
                                                            {"symbols", ae::SymbolTensor(symbols)},
                                                            {"dequantized", dequantized}});
}

std::vector<std::uint8_t> ReadBytes(const std::filesystem::path& file) {
  std::ifstream stream(file, std::ios::binary);
  check::Require(bool(stream), "readexistingcodecfilewithoutsourceassets");
  return std::vector<std::uint8_t>((std::istreambuf_iterator<char>(stream)), {});
}

void Run(const std::filesystem::path& data, const std::filesystem::path& encoder, const std::filesystem::path& root) {
  auto pools = ae::LoadSaliency(data);
  std::filesystem::create_directories(root);
  Unit(root);
  ae::UndercompleteModel model(false);
  ae::LoadFrozen(model.parameters, encoder, "frozen3epochlearnedtransform");
  const auto before    = ae::Snapshot(model.parameters);
  auto       fit       = model.Evaluate(pools[0].images), test = model.Evaluate(pools[2].images);
  const auto quantizer = ae::FitCodecQuantizer(fit[0]);
  const auto symbols   = ae::QuantizeCodec(test[0], quantizer);
  auto       arrays    = ae::Arrays(model.parameters);
  ae::CodecPacket packet{quantizer, ae::Find(arrays, "decoder/W"), ae::Find(arrays, "decoder/b"),
                         pools[2].source.to_vector<int>(), symbols};
  const auto bytes = ae::PackCodec(packet);
  const auto file  = root / "test.tflcodec";
  {
    std::ofstream output(file, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
    check::Require(bool(output), "writeACTUALselfcontainedlearnedbitstream");
  }
  const auto reread = ReadBytes(file);
  check::Require(reread == bytes, "actualfilebytesroundtrip");
  const auto decoded_packet = ae::UnpackCodec(reread);
  check::Require(ae::PackCodec(decoded_packet) == bytes, "serialize-parse-serializeBYTEEXACT");
  auto dequantized = ae::DequantizeCodec(decoded_packet.symbols, decoded_packet.quantizer);
  auto decoded = ae::DecodeCodec(decoded_packet), reference = model.Decode(dequantized);
  const double decode_error = ae::MaxAbsDiff(decoded, reference, decoded.numel());
  check::Require(decode_error < 2e-6, "freshbitstreamONLYdecoderagreeswithoriginaldecoder");
  int rejected = 0;
  for (int kind = 0; kind < 3; ++kind) {
    auto broken = bytes;
    if (kind == 0)
      broken[0] = 'X';
    else if (kind == 1)
      broken.pop_back();
    else
      for (int j = 8; j < 12; ++j) broken[j] = 255;
    try {
      ae::UnpackCodec(broken);
    } catch (const std::exception&) {
      ++rejected;
    }
  }
  check::Require(rejected == 3, "badmagic/truncatedpayload/oversizedshapesrejected");
  int        clipped = 0, constant = 0;
  const auto code = test[0].to_vector<float>();
  for (int j = 0; j < 32; ++j) {
    constant += quantizer.minimum[j] == quantizer.maximum[j];
    for (int b = 0; b < 200; ++b) {
      const float v = code[b * 32 + j];
      clipped += v < quantizer.minimum[j] || v > quantizer.maximum[j];
    }
  }
  check::Require(before == ae::Snapshot(model.parameters), "encoding/referencecodecpathsleaveallpretrainedweightsimmutable");
  nn::save_npz((root / "analysis.npz").string(), {{"fit_images", pools[0].images},
                                                   {"fit_labels", pools[0].labels},
                                                   {"fit_source_ids", pools[0].source},
                                                   {"test_images", pools[2].images},
                                                   {"test_labels", pools[2].labels},
                                                   {"test_source_ids", pools[2].source},
                                                   {"fit_code", fit[0]},
                                                   {"test_code", test[0]},
                                                   {"unquantized_reconstruction", test[1]},
                                                   {"minimum", nn::from_vector(quantizer.minimum, {32})},
                                                   {"maximum", nn::from_vector(quantizer.maximum, {32})},
                                                   {"symbols", ae::SymbolTensor(symbols)},
                                                   {"dequantized", dequantized},
                                                   {"decoded_reconstruction", decoded}});
  nn::save_npz((root / "model_arrays.npz").string(), arrays);
  if (std::filesystem::exists(data / "manifest.json"))
    std::filesystem::copy_file(data / "manifest.json", root / "data_manifest.json",
                               std::filesystem::copy_options::overwrite_existing);
  std::filesystem::create_directories(root / "original_png");
  std::size_t   png_bytes = 0;
  std::ofstream sizes(root / "png_sizes.tsv");
  sizes << "source_id\tbytes\n";
  const auto source = pools[2].source.to_vector<int>();
  for (int b = 0; b < 200; ++b) {
    const auto path = root / "original_png" / (std::to_string(source[b]) + ".png");
    nn::save_png(nn::reshape(ae::Row(pools[2].images, b), {28, 28, 1}), path.string());
    const auto n = std::filesystem::file_size(path);
    png_bytes += n;
    sizes << source[b] << '\t' << n << '\n';
  }
  const std::size_t payload = 200 * 32, model_bytes = 25872 * 4, quantizer_bytes = 32 * 2 * 4, identity_bytes = 200 * 4,
                    header_bytes = 28;
  check::Require(bytes.size() == payload + model_bytes + quantizer_bytes + identity_bytes + header_bytes,
                 "allbytecategoriesaccountfortotalFILEsize");
  const double mse = ae::Mse(decoded, pools[2].images), unquantized = ae::Mse(test[1], pools[2].images);
  std::ofstream meta(root / "metrics.json");
  meta.precision(17);
  meta << "{\"additional_training_updates\":0,\"test_images\":200,\"latent_dimensions\":32,\"bits_per_symbol\":8,"
          "\"constant_channels\":"
       << constant << ",\"clipped_test_code_entries\":" << clipped << ",\"payload_bytes\":" << payload
       << ",\"decoder_parameter_bytes\":" << model_bytes << ",\"quantizer_bytes\":" << quantizer_bytes
       << ",\"source_identity_bytes\":" << identity_bytes << ",\"header_bytes\":" << header_bytes
       << ",\"total_file_bytes\":" << bytes.size() << ",\"original_raw_u8_bytes\":156800,\"original_png_collection_bytes\":"
       << png_bytes << ",\"payload_bits_per_pixel\":" << double(payload * 8) / (200 * 784)
       << ",\"total_bits_per_pixel\":" << double(bytes.size() * 8) / (200 * 784)
       << ",\"png_bits_per_pixel\":" << double(png_bytes * 8) / (200 * 784) << ",\"unquantized_test_mse\":" << unquantized
       << ",\"decoded_test_mse\":" << mse << ",\"decoded_global_psnr_db\":" << 10 * std::log10(1 / mse)
       << ",\"fresh_decoder_max_error\":" << decode_error << ",\"malformed_controls_rejected\":3}\n";
  std::ofstream html(root / "index.html");
  html << "<!doctype html><meta charset='utf-8'><title>Learned image codec baseline</title><style>body{font:16px "
          "system-ui;margin:30px;max-width:1100px}img{max-width:100%;image-rendering:pixelated}</style><h1>Actual "
          "learned-transform lossy codec baseline</h1><p>Frozen THREEepochMNIST784-32ReLU-784sigmoid AE,0newtraining. "
          "FitONLY1000codes set perchannelmin/max. Quantize32testcodecoordinates to8bituniform symbols,nearesthalf-up/clamp0..255;"
          "constantchannels encode0anddecodefitconstant. Testout-of-range codeentries clipped="
       << clipped << ",constantchannels=" << constant
       << ". Post-trainingquantization,NOentropy model/rate-distortiontraining/competitivecodec claim.</p><p><a "
          "href='test.tflcodec'>Actual self-contained bitstream</a>:8byteTFLCODE1magic;fiveU32LE "
          "n/latent32/height28/width28/channels1;32pairsF32LEmin/max;decoderW32x784+b784F32LE;NsourceIDsU32LE;N*32U8symbols. "
          "Allparamsneededbydecoder included. Fresh decoder decodes FROMFILEONLY,withoutoriginalmodel/encoder/cleanimages. "
          "Encoderweights not transmitted because receiver does not need them. Sharedsoftware/runtime bytes excluded,as for "
          "PNG.</p><p>Payload="
       << payload << "bytes(" << double(payload * 8) / (200 * 784) << "bpp),DECODERweights=" << model_bytes
       << ",quantizer=" << quantizer_bytes << ",IDs=" << identity_bytes << ",header=" << header_bytes
       << ",TOTAL=" << bytes.size() << "bytes(" << double(bytes.size() * 8) / (200 * 784)
       << "bpp). Raw8bitpixels156800bytes. ACTUAL200originalgrayscalePNGfiles total=" << png_bytes << "bytes("
       << double(png_bytes * 8) / (200 * 784)
       << "bpp),decodedlosslessly;individualPNGheaders included,filenames/container overhead excluded. NOTE:datamunge's PNG "
          "writer stores uncompressed deflate blocks,so this lossless baseline is much larger than zlib-compressed PNG "
          "(tflow's EncodePng files total55568bytes). PNG is a lossless "
          "baseline,not quality-matched;model overhead matters and is not hidden in payload-only rates.</p><p>"
          "UnquantizedAEtestMSE="
       << unquantized << ",FILEdecodedMSE=" << mse << ",GLOBALPSNR=" << 10 * std::log10(1 / mse)
       << "dB. Quantizationerror and reconstructionloss distinguished. Rate includes decoderparams amortized over these200"
          "images,not an arbitrary imaginary corpus. No32Dcoordinatecount-onlycompression claim. Parseguards "
          "rejectbadmagic/truncation/oversizedshapes;payload has no entropy coding or integritychecksum.</p><p>Ten "
          "evenlyspacedunfilteredheldoutcases,original/unquantizedAE/FILEdecoded/absoluteFILEerror,fixed0..1gray. "
          "Blurry/missingdigitdetails retained,not per-image normalized. Outputs are probabilities,not sampled pixels.</p>";
  const auto error = nn::abs(decoded - ae::Flatten(pools[2].images));
  for (int k = 0; k < 10; ++k) {
    const int b = k * 199 / 9;
    ae::CaseStrip3(root / ("case_" + std::to_string(k) + ".png"), {pools[2].images, test[1], decoded, error}, b);
    html << "<p>Source" << source[b] << "<br><img src='case_" << k << ".png'></p>";
  }
  check::Require(bool(meta) && bool(html) && bool(sizes), "completecodecfiles+byteaccounting+qualityartifacts");
  std::cout << "ACTUALfile=" << bytes.size() << "bytes,payload=" << payload << ",PNG=" << png_bytes << ",MSE=" << mse
            << ",unquantized=" << unquantized << ",clipped=" << clipped
            << ";selfContainedFreshDecode/byteRoundtrip/quantFixtures/malformedReject/immutable PASS\nArtifacts:"
            << (root / "index.html").string() << std::endl;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path data    = paths::Dataset("saliency"),
                          encoder = paths::OutputRoot() / "models/undercomplete_autoencoder",
                          out = paths::OutputRoot() / "models/learned_image_codec", decode_file;
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a.rfind("data=", 0) == 0)
        data = a.substr(5);
      else if (a.rfind("encoder=", 0) == 0)
        encoder = a.substr(8);
      else if (a.rfind("out=", 0) == 0)
        out = a.substr(4);
      else if (a.rfind("decode=", 0) == 0)
        decode_file = a.substr(7);
      else
        throw std::invalid_argument("expected data=PATH encoder=PATH out=PATH or decode=FILE");
    }
    if (!decode_file.empty()) {
      const auto packet         = ae::UnpackCodec(ReadBytes(decode_file));
      const auto reconstruction = ae::DecodeCodec(packet);
      std::filesystem::create_directories(out);
      nn::save_npz((out / "decoded.npz").string(),
                   {{"source_ids", nn::from_vector(packet.source_ids, {std::int64_t(packet.source_ids.size())})},
                    {"reconstruction", reconstruction}});
      std::cout << "decoded" << packet.source_ids.size()
                << "imagesFROMFILEONLY;datasetandencoderNOTLOADED;artifact:" << (out / "decoded.npz").string() << std::endl;
      return 0;
    }
    Run(data, encoder, out);
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
