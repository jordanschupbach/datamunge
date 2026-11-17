/// @file basic.cpp
/// @brief Canonical local memory, factor lattice, sentence grammar and original-MNIST preparation.
#ifdef DATAMUNGE_ENABLE_DATA_ASSETS
#include <datamunge/datasets/download.hpp>
#include <datamunge/nn/nn.hpp>
#include <map>
#include <numeric>
#include <random>
#include <set>

#include "support.hpp"
namespace datamunge::datasets::detail {
namespace nn = datamunge::nn;
/// @brief Generate deterministic fitting/validation/test copy or QA tensor records.
/// @param copy_task True selects binary copy, false selects fact/query answering.
/// @return All split observations, targets, lengths and source IDs.
static nn::NamedTensors MakeMemory(bool copy_task) {
  std::mt19937     generator(copy_task ? 45101 : 45301);
  nn::NamedTensors records;
  for (int split = 0; split < 3; ++split) {
    const std::string prefix = split == 0 ? "fit_" : split == 1 ? "validation_" : "test_";
    const int         count  = split == 0 ? 128 : 32;
    std::vector<int>  lengths(count), ids(count), facts(count * 12), queries(count), answers(count);
    std::vector<float> input(count * 36), target(count * 27);
    for (int row = 0; row < count; ++row) {
      ids[row]         = split * 10000 + row;
      const int length = 1 + generator() % (copy_task ? 4 : 6);
      lengths[row]     = length;
      if (copy_task) {
        for (int step = 0; step < length; ++step)
          for (int bit = 0; bit < 3; ++bit) {
            const float value                               = float(generator() & 1);
            input[(row * 9 + step) * 4 + bit]               = value;
            target[(row * 9 + length + 1 + step) * 3 + bit] = value;
          }
        input[(row * 9 + length) * 4 + 3] = 1;
      } else {
        int entities[7] = {1, 2, 3, 4, 5, 6, 7};
        for (int i = 6; i > 0; --i)
          std::swap(entities[i], entities[generator() % (i + 1)]);
        const int selected = generator() % length;
        for (int slot = 0; slot < length; ++slot) {
          const int answer                = generator() % 4;
          facts[(row * 6 + slot) * 2]     = entities[slot];
          facts[(row * 6 + slot) * 2 + 1] = 8 + answer;
          if (slot == selected) {
            queries[row] = entities[slot];
            answers[row] = answer;
          }
        }
      }
    }
    if (copy_task) {
      records.emplace_back(prefix + "input", nn::from_vector(input, {count, 9, 4}));
      records.emplace_back(prefix + "target", nn::from_vector(target, {count, 9, 3}));
    } else {
      records.emplace_back(prefix + "facts", nn::from_vector(facts, {count, 6, 2}));
      records.emplace_back(prefix + "query", nn::from_vector(queries, {count}));
      records.emplace_back(prefix + "target", nn::from_vector(answers, {count}));
    }
    records.emplace_back(prefix + "lengths", nn::from_vector(lengths, {count}));
    records.emplace_back(prefix + "source_ids", nn::from_vector(ids, {count}));
  }
  return records;
}

/// @brief Select balanced, byte-distinct original MNIST pools using seed1931.
/// @param directory Saliency cache with the original MNIST cache as its sibling.
static void Saliency(const std::string& directory) {
  namespace fs = std::filesystem;
  namespace nn = datamunge::nn;
  if (fs::is_regular_file(fs::path(directory) / "saliency_data.npz")) {
    if (!fs::is_regular_file(fs::path(directory) / "manifest.json"))
      save(fs::path(directory) / "manifest.json",
           {{"source", "caller-supplied saliency_data.npz"}, {"generation_verified", false}});
    return;
  }
  const auto mnist = fs::path(directory).parent_path() / "mnist";
  datamunge::datasets::fetch("mnist", mnist.string());
  const auto train = nn::load_mnist(mnist.string(), "train"),
             test  = nn::load_mnist(mnist.string(), "test");
  if (train.images.shape() != nn::Shape{60000, 28, 28, 1} ||
      test.images.shape() != nn::Shape{10000, 28, 28, 1})
    throw std::runtime_error("official MNIST dimensions required to prepare autoencoder pools");
  // read_idx retains exact original uint8 pixels (represented as int32).
  const auto locate = [&](const std::string& file) {
    return fs::is_regular_file(mnist / file) ? mnist / file : mnist / (file + ".gz");
  };
  const auto training =
      nn::reshape(nn::read_idx(locate("train-images-idx3-ubyte").string()), {60000, 28, 28, 1});
  const auto testing =
      nn::reshape(nn::read_idx(locate("t10k-images-idx3-ubyte").string()), {10000, 28, 28, 1});
  const auto   labels = train.labels.to_vector<int>(), test_labels = test.labels.to_vector<int>();
  const auto   pixels = training.to_vector<int>(), test_pixels = testing.to_vector<int>();
  std::mt19937 random(1931);
  std::set<std::string>                            seen;
  std::map<std::string, std::vector<std::int64_t>> rows;
  for (int digit = 0; digit < 10; ++digit) {
    for (bool held_out : {false, true}) {
      const auto&               y = held_out ? test_labels : labels;
      const auto&               x = held_out ? test_pixels : pixels;
      std::vector<std::int64_t> candidates;
      for (std::size_t i = 0; i < y.size(); ++i)
        if (y[i] == digit)
          candidates.push_back(i);
      std::shuffle(candidates.begin(), candidates.end(), random);
      std::size_t at = 0;
      for (const auto& [split, count] :
           (held_out
                ? std::vector<std::pair<std::string, int>>{{"test", 20}}
                : std::vector<std::pair<std::string, int>>{{"fit", 100}, {"validation", 20}})) {
        for (int i = 0; i < count; ++i) {
          bool added = false;
          while (at < candidates.size()) {
            const auto  row = candidates[at++];
            std::string bytes(784, '\0');
            for (int p = 0; p < 784; ++p)
              bytes[p] = static_cast<char>(x[row * 784 + p]);
            if (seen.insert(bytes).second) {
              rows[split].push_back(row);
              added = true;
              break;
            }
          }
          if (!added)
            throw std::runtime_error("insufficient distinct MNIST images");
        }
      }
    }
  }
  nn::NamedTensors records;
  for (const std::string split : {"fit", "validation", "test"}) {
    const auto& selected = rows[split];
    auto        ids      = selected;
    if (split == "test")
      for (auto& id : ids)
        id += 60000;
    records.emplace_back(
        split + "_images", nn::take_rows(split == "test" ? testing : training, selected));
    records.emplace_back(
        split + "_labels",
        nn::cast(
            nn::take_rows(split == "test" ? test.labels : train.labels, selected),
            nn::DType::Int32));
    records.emplace_back(split + "_source_ids", nn::cast(nn::from_vector(ids), nn::DType::Int32));
  }
  fs::create_directories(directory);
  Temporary stage(directory);
  nn::save_npz((stage.path / "saliency_data.npz").string(), records);
  std::ofstream manifest(stage.path / "manifest.json");
  manifest << "{\"dataset\":\"MNIST\",\"generator\":\"datamunge C++ balanced distinct images "
              "v1\",\"seed\":1931,"
              "\"selection\":\"std::mt19937, independent local "
              "selection\",\"test_source_offset\":60000,\"source_ids\":{";
  bool first = true;
  for (const std::string split : {"fit", "validation", "test"}) {
    if (!first)
      manifest << ',';
    first = false;
    manifest << '"' << split << "\":[";
    for (std::size_t i = 0; i < rows[split].size(); ++i) {
      if (i)
        manifest << ',';
      manifest << rows[split][i] + (split == "test" ? 60000 : 0);
    }
    manifest << ']';
  }
  manifest << "}}\n";
  manifest.close();
  if (!manifest)
    throw std::runtime_error("cannot write dataset manifest");
  fs::rename(stage.path / "manifest.json", fs::path(directory) / "manifest.json");
  fs::rename(stage.path / "saliency_data.npz", fs::path(directory) / "saliency_data.npz");
}

/// @brief Prepare the exact local known-factor circle/square lattice and its split provenance.
/// @param root Parent cache; outputs live below beta_vae/.
static void FactorLattice(const fs::path& root) {
  const auto directory=root/"beta_vae";
  fs::create_directories(directory);
  if(fs::is_regular_file(directory/"factors.npz")){
    if(!fs::is_regular_file(directory/"manifest.json"))
      save(directory/"manifest.json",{{"source","caller-supplied factors.npz"},{"generation_verified",false}});
    return;
  }
  Temporary stage(directory);
  auto records=nn::models::KnownFactorLattice();
  for(auto& [name,value]:records)if(name.ends_with("_images"))value=nn::cast(value,nn::DType::Int32);
  nn::save_npz((stage.path/"factors.npz").string(),records);
  save(stage.path/"manifest.json",{{"dataset","local known-factor circle/square lattice"},
    {"generator","datamunge C++ exhaustive lattice v1"},{"external_source",false},{"generation_verified",true},
    {"factor_order",{"shape","size","x","y"}},{"cardinalities",{2,3,7,7}},
    {"row_order","ascending source ID within split"},{"source_id","(((shape*3+size)*7+x)*7+y)"},
    {"split","residue=(x+2*y+size+3*shape)%7; test0,validation1,fit2..6"},
    {"fit",210},{"validation",42},{"test",42},{"image_shape",{28,28,1}},
    {"radius","size+2"},{"centre","(3*x+5,3*y+5)"},
    {"pixels","inclusive binary circle dx^2+dy^2<=r^2 or axis-aligned square |dx|,|dy|<=r"}});
  fs::rename(stage.path/"manifest.json",directory/"manifest.json");
  fs::rename(stage.path/"factors.npz",directory/"factors.npz");
}

/// @brief Enumerate the repository's complete six-factor sentence grammar in source-ID order.
/// @return Sixteen owning Int32 arrays: config plus five arrays for each of three splits.
static nn::NamedTensors SentenceGrammar() {
  nn::NamedTensors records{{"config", nn::from_vector(std::vector<int>{19, 7})}};
  for (int split = 0; split < 3; ++split) {
    const std::string prefix = split == 0 ? "fit_" : split == 1 ? "validation_" : "test_";
    const int count = split == 0 ? 288 : 72;
    std::vector<int> tokens, inputs, lengths, factors, sources;
    for (int id = 0; id < 432; ++id) {
      const std::array<int, 6> f{id / 216, (id / 72) % 3, (id / 24) % 3,
                                 (id / 12) % 2, (id / 3) % 4, id % 3};
      const int residue = 2 * ((f[1] + f[2] + f[5]) % 3) + (f[0] + f[3] + f[4]) % 2;
      if ((residue == 0 ? 2 : residue == 1 ? 1 : 0) != split)
        continue;
      std::vector<int> sentence{3 + f[0], 5 + f[1], 8 + f[2], 11 + f[3], 13 + f[4]};
      if (f[5])
        sentence.push_back(16 + f[5]);
      sentence.push_back(2);
      lengths.push_back(int(sentence.size()));
      sentence.resize(7, 0);
      tokens.insert(tokens.end(), sentence.begin(), sentence.end());
      inputs.push_back(1);
      inputs.insert(inputs.end(), sentence.begin(), sentence.end() - 1);
      factors.insert(factors.end(), f.begin(), f.end());
      sources.push_back(id);
    }
    records.emplace_back(prefix + "tokens", nn::from_vector(tokens, {count, 7}));
    records.emplace_back(prefix + "decoder_inputs", nn::from_vector(inputs, {count, 7}));
    records.emplace_back(prefix + "lengths", nn::from_vector(lengths, {count}));
    records.emplace_back(prefix + "factors", nn::from_vector(factors, {count, 6}));
    records.emplace_back(prefix + "source_ids", nn::from_vector(sources, {count}));
  }
  return records;
}

/// @brief Prepare the exhaustive local sentence grammar and explicit generation provenance.
/// @param root Parent cache; outputs live below sequence_vae/. Supplied archives are retained.
static void SentenceAssets(const fs::path& root) {
  const auto directory = root / "sequence_vae";
  fs::create_directories(directory);
  const std::string vocabulary =
      "[pad]\n[bos]\n[eos]\na\nthe\nred\ngreen\nblue\ncircle\nsquare\ntriangle\n"
      "moves\nrolls\nleft\nright\nup\ndown\nquickly\nslowly\n";
  if (fs::is_regular_file(directory / "sentences.npz")) {
    if (!fs::is_regular_file(directory / "vocabulary.txt"))
      write(directory / "vocabulary.txt", vocabulary);
    if (!fs::is_regular_file(directory / "manifest.json"))
      save(directory / "manifest.json",
           {{"source", "caller-supplied sentences.npz"}, {"generation_verified", false}});
    return;
  }
  Temporary stage(directory);
  nn::save_npz((stage.path / "sentences.npz").string(), SentenceGrammar());
  write(stage.path / "vocabulary.txt", vocabulary);
  save(stage.path / "manifest.json",
       {{"dataset", "local exhaustive compositional sentence grammar"},
        {"generator", "datamunge C++ sentence grammar v1"},
        {"external_source", false}, {"generation_verified", true},
        {"factor_order", {"article", "colour", "shape", "verb", "direction", "adverb"}},
        {"cardinalities", {2, 3, 3, 2, 4, 3}},
        {"row_order", "ascending source ID within split"},
        {"source_id", "(((((article*3+colour)*3+shape)*2+verb)*4+direction)*3+adverb)"},
        {"split", "residue=2*((colour+shape+adverb)%3)+(article+verb+direction)%2; test0,validation1,fit2..5"},
        {"fit", 288}, {"validation", 72}, {"test", 72}, {"vocabulary_size", 19}, {"steps", 7},
        {"special_ids", {{"pad", 0}, {"bos", 1}, {"eos", 2}}},
        {"tokens", "article,colour,shape,verb,direction,optional adverb,EOS; right PAD to seven"},
        {"adverb", "0 absent,1 quickly,2 slowly"},
        {"lengths", "six without adverb,seven otherwise; includes EOS, excludes PAD"},
        {"decoder_inputs", "BOS then previous target, including EOS before final PAD"},
        {"dtype", "Int32 for all sixteen arrays"}});
  fs::rename(stage.path / "vocabulary.txt", directory / "vocabulary.txt");
  fs::rename(stage.path / "manifest.json", directory / "manifest.json");
  fs::rename(stage.path / "sentences.npz", directory / "sentences.npz");
}

/// @brief Prepare checksummed original MovieLens100K warm-user teaching ratings.
/// @param root Dedicated autorec cache directory.
/// @throws std::runtime_error For source ingress, schema or fixed-count failures.
static void AutoRecRatings(const fs::path&root){
 const std::string digest="50d2a982c66986937beb9ffb3aa76efe955bf3d5c6b761f4e3a7cd717c6a3229";
 fetch_file("https://files.grouplens.org/datasets/movielens/ml-100k.zip",(root/"ml-100k.zip").string(),digest,4924029,"autorec");
 Temporary stage(root);extract_archive((root/"ml-100k.zip").string(),stage.path.string(),{"ml-100k/u.data"});
 fs::copy_file(stage.path/"ml-100k/u.data",root/"u.data",fs::copy_options::overwrite_existing);
 std::ifstream raw(root/"u.data");std::map<std::pair<int,int>,int> original;std::vector<int> counts(257);
 int user,item,rating,timestamp;
 while(raw>>user>>item>>rating>>timestamp){if(user<1||user>943||item<1||item>1682||rating<1||rating>5||!original.emplace(std::make_pair(user,item),rating).second)throw std::runtime_error("original MovieLens record invalid");if(user<=256&&item<=256)++counts[user];}
 if(!raw.eof()||original.size()!=100000)throw std::runtime_error("original MovieLens100K count mismatch");
 std::vector<int> users,items(256);std::iota(items.begin(),items.end(),1);for(int u=1;u<=256;++u)if(counts[u]>=10)users.push_back(u);
 if(users.size()!=199)throw std::runtime_error("original warm-user window changed");
 std::vector<float> input(199*256),ratings(input.size()),train(input.size()),validation(input.size()),test(input.size());
 MetadataRandom random(4601);std::array<int,3> totals{};
 for(int row=0;row<199;++row){std::vector<int> observed;for(int i=0;i<256;++i){auto found=original.find({users[row],i+1});if(found!=original.end()){ratings[row*256+i]=float(found->second);observed.push_back(i);}}
 random.shuffle(observed);const int heldout=std::max(1,int(observed.size())/10);
 for(int j=0;j<int(observed.size());++j){const int at=row*256+observed[j];if(j<heldout){validation[at]=1;++totals[1];}else if(j<2*heldout){test[at]=1;++totals[2];}else{train[at]=1;input[at]=ratings[at]/5.f;++totals[0];}}
 }
 if(totals!=std::array<int,3>{8079,898,898})throw std::runtime_error("AutoRec split support mismatch");
 const nn::Shape shape{199,256};nn::save_npz((root/"ratings.npz").string(),{{"input",nn::from_vector(input,shape)},{"ratings",nn::from_vector(ratings,shape)},{"train_mask",nn::from_vector(train,shape)},{"validation_mask",nn::from_vector(validation,shape)},{"test_mask",nn::from_vector(test,shape)},{"user_ids",nn::from_vector(users)},{"item_ids",nn::from_vector(items)}});
 save(root/"manifest.json",Json{{"generator","datamunge native AutoRec warm-user split v1"},{"source_url","https://files.grouplens.org/datasets/movielens/ml-100k.zip"},{"archive_sha256",digest},{"archive_bytes",4924029},{"source_ratings",100000},{"users",199},{"items",256},{"split_seed",4601},{"split_rng","Python-compatible MT19937 rejection shuffle"},{"split_rule","ascending users and items; shuffle each observed item list; first floor(n/10) validation, next test, remaining train"},{"counts",totals},{"input","training stars divided by five; all held-out and unknown entries zero"},{"scope","canonical native partition; external TensorFlow preparation archive is unavailable and split parity is not asserted"}});
}
/// @brief Prepare local teaching tasks or original MNIST plus its balanced teaching pools.
/// @param name "autorec", "memory", "mnist-assets", "beta-vae-assets" or "sequence-vae-assets".
/// @param root Parent dataset cache.
void prepare_basic(const std::string& name, const fs::path& root) {
  if(name=="autorec"){AutoRecRatings(root);return;}
  if(name=="beta-vae-assets"){FactorLattice(root);return;}
  if (name == "sequence-vae-assets") { SentenceAssets(root); return; }
  if (name == "mnist-assets") {
    datamunge::datasets::fetch("mnist", (root / "mnist").string());
    Saliency((root / "saliency").string());
    return;
  }
  for (bool copy : {false, true}) {
    const auto dir = root / (copy ? "ntm" : "memnet");
    fs::create_directories(dir);
    Temporary stage(dir);
    if (!fs::is_regular_file(dir / "data.npz")) {
      nn::save_npz((stage.path / "data.npz").string(), MakeMemory(copy));
      fs::rename(stage.path / "data.npz", dir / "data.npz");
      save(
          dir / "manifest.json",
          {{"generator", "datamunge local C++ memory task v1"},
           {"external_selection", false},
           {"seed", copy ? 45101 : 45301},
           {"fit", 128},
           {"validation", 32},
           {"test", 32},
           {"task",
            copy ? "binary copy with delimiter and recall-only targets"
                 : "unambiguous fact/query question answering"}});
    } else if (!fs::is_regular_file(dir / "manifest.json")) {
      save(
          dir / "manifest.json",
          {{"source", "caller-supplied data.npz"}, {"generation_verified", false}});
    }
  }
}
} // namespace datamunge::datasets::detail

#endif // DATAMUNGE_ENABLE_DATA_ASSETS
