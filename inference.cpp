#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include <sys/time.h>

#include "tensor.h"
#include "graph.h"
#include "utils.h"
#include "cuda_function.h"

#include "base64.h"
#include "tiktoken.h"

const int vocab_size = 151936;

float temp = 1, topp = 0.9;
const int topk = 300;

static const std::string PAT_STR = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

class QwenTokenizer {
  public:

    QwenTokenizer(const std::string & tiktoken_path);

    auto encode(const std::string &text, int max_length) const -> std::vector<int>;

    auto decode(const std::vector<int> &ids) const -> std::string;

    auto encode_history(const std::vector<std::string> &history, int max_length) const -> std::vector<int>;

    auto build_prompt(const std::vector<std::string> &history) const -> std::string;

    auto is_special_id(int id) const -> bool;

    tiktoken::tiktoken tokenizer;
    int eos_token_id;
    int im_start_id;
    int im_end_id;
};

static std::pair<std::string, int> _parse(const std::string &line) {
  auto pos = line.find(" ");
  if (pos == std::string::npos) {
    throw std::runtime_error("invalid encoder line: " + line);
  }

  auto token = base64::decode({line.data(), pos});
  int rank = 0;
  try {
    rank = std::stoul(line.substr(pos + 1));
  } catch (const std::exception &) {
    throw std::runtime_error("invalid encoder rank: " + line);
  }

  return {std::move(token), rank};
}

QwenTokenizer::QwenTokenizer(const std::string & tiktoken_path) {
  std::ifstream file(tiktoken_path);
  if (!file) {
    throw std::runtime_error("failed to open encoder file: " + tiktoken_path);
  }

  ankerl::unordered_dense::map<std::string, int> encoder;
  std::string line;
  while (std::getline(file, line)) {
    auto [token, rank] = _parse(line);

    if (!encoder.emplace(std::move(token), rank).second) {
      throw std::runtime_error("duplicate item: " + line);
    }
  }

  std::vector<std::string> special_tokens_s{"<|endoftext|>", "<|im_start|>", "<|im_end|>"};
  char buffer[14];
  for (size_t i = 0; i < 205; i++) {
    snprintf(buffer, 14, "<|extra_%zu|>", i);
    special_tokens_s.push_back(buffer);
  }
  size_t encoder_size = encoder.size();
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  special_tokens.reserve(special_tokens_s.size());
  for (size_t i = 0; i < special_tokens_s.size(); i++) {
    special_tokens[special_tokens_s[i]] = encoder_size + i;
  }

  tokenizer = tiktoken::tiktoken(std::move(encoder), special_tokens, PAT_STR);
  eos_token_id = 151643;
  im_start_id = 151644;
  im_end_id = 151645;
}

auto QwenTokenizer::encode(const std::string &text, int max_length) const -> std::vector<int> {
  auto ids = tokenizer.encode(text);
  if ((int)ids.size() > max_length) {
    ids.erase(ids.begin(), ids.end() - max_length);
  }
  return ids;
}

auto QwenTokenizer::decode(const std::vector<int> &ids) const -> std::string {
  std::vector<int> normal_ids(ids);
  normal_ids.erase(std::remove_if(normal_ids.begin(), normal_ids.end(), [this](int id) { return is_special_id(id); }),
                   normal_ids.end());
  auto text = tokenizer.decode(normal_ids);
  return text;
}

auto QwenTokenizer::is_special_id(int id) const -> bool {
  return id == eos_token_id || id == im_start_id || id == im_end_id;
}

int sample(const std::vector<float>& logits, float temp, float topp, int topk) {
    // return std::max_element(logits.begin(), logits.end()) - logits.begin();

    assert(logits.size() == vocab_size);

    if (fabsf(temp) < 1e-8)
        return std::max_element(logits.begin(), logits.end()) - logits.begin();

    struct timeval tv;
    gettimeofday(&tv, NULL);
    std::mt19937_64 rng(tv.tv_usec/100);  // haha
    std::uniform_real_distribution<float> dist(0, 1);

    std::vector<std::pair<float, int>> probs(vocab_size);
    for (int i = 0; i < vocab_size; i++) probs[i] = {logits[i] / temp, i};
    std::sort(probs.begin(), probs.end(),
              std::greater<std::pair<float, int>>());
    while (probs.size() > topk) probs.pop_back();

    // softmax
    auto maximum = probs[0].first;
    std::transform(probs.begin(), probs.end(), probs.begin(),
                   [maximum](auto x) {
                       return std::make_pair(expf(x.first - maximum), x.second);
                   });
    auto sum = std::accumulate(probs.begin(), probs.end(), 0.0f,
                               [](auto x, auto y) { return x + y.first; });
    std::transform(probs.begin(), probs.end(), probs.begin(), [sum](auto x) {
        return std::make_pair(x.first / sum, x.second);
    });

    sum = 0;
    int last = 0;
    for (int i = 0; i < (int)probs.size(); i++) {
        sum += probs[i].first;
        last = i;
        if (sum > topp) break;
    }

    float r = dist(rng) * sum;
    sum = 0;
    for (int i = 0; i <= last; i++) {
        sum += probs[i].first;
        if (sum > r) return probs[i].second;
    }
    return probs[last].second;
}

int get_model_params(string file_name, int* ctx_length, int* n_layers, int* dim, int* n_heads, int* kcache_start, int* vcache_start, int* kvcache_step)
{
    string param_file_name = file_name + ".ncnn.param";

    ifstream fin(param_file_name.c_str());

    int index = 0;
    string strline;
    while(getline(fin, strline))
    {
        if(index == 1)
        {
            vector<string> params = split(strline, " ");
            *ctx_length = atoi(params[0].c_str());
            *n_layers = atoi(params[1].c_str());
            *dim = atoi(params[2].c_str());
            *n_heads = atoi(params[3].c_str());
            *kcache_start = atoi(params[4].c_str());
            *vcache_start = atoi(params[5].c_str());
            *kvcache_step = atoi(params[6].c_str());
            break;
        }
        index++;
    }

    fin.close();

    return 0;
}

// ./inference MODEL PROMPT OUT-TOKEN-COUNT
int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " MODEL PROMPT OUT-TOKEN-COUNT TEMPERATURE"
                  << std::endl;
        return 1;
    }

    createCublas();

    std::string model_name = argv[1];
    std::string prompt = argv[2];
    std::string tokenizer_path = "qwen.tiktoken";
    int token_count = std::stoi(argv[3]);
    temp = std::atof(argv[4]);

    // 模型的基本参数
    int ctx_length, n_layers, dim, n_heads;
    int kcache_start, vcache_start, kvcache_step;
    get_model_params(model_name, &ctx_length, &n_layers, &dim, &n_heads, &kcache_start, &vcache_start, &kvcache_step);

    std::vector<Tensor*> kcache, vcache;
    std::vector<float> freqs_cos_table, freqs_sin_table;

    kcache.resize(n_layers);
    vcache.resize(n_layers);
    int head_dim = dim / n_heads;
    freqs_cos_table.resize(ctx_length * head_dim / 2);
    freqs_sin_table.resize(ctx_length * head_dim / 2);

    for (int i = 0; i < ctx_length; i++) {
        for (int j = 0; j < head_dim / 2; j++) {
            auto x = i / pow(1000000.0, j * 2 / (double)head_dim);
            freqs_cos_table[i * head_dim / 2 + j] = cos(x);
            freqs_sin_table[i * head_dim / 2 + j] = sin(x);
        }
    }

    vector<int> kvcache_init_shape;
    kvcache_init_shape.push_back(0);
    kvcache_init_shape.push_back(dim);
    for (int i = 0; i < n_layers; i++) {
        kcache[i] = new Tensor();
        vcache[i] = new Tensor();
        kcache[i]->set_shape(kvcache_init_shape);
        vcache[i]->set_shape(kvcache_init_shape);
    }

    // 加载tokenizer
    auto tokenizer = std::make_unique<QwenTokenizer>(tokenizer_path);

    // 加载模型
    Graph* graph = new Graph();
    graph->load_model(model_name);

    Tensor inputTensor;
    vector<int> inputShape;
    inputShape.push_back(1);
    inputTensor.set_shape(inputShape);
    vector<float> inputData;
    inputData.push_back(0);

    prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    auto tokens = tokenizer->encode(prompt, 4096);
    int prompt_end = tokens.size();
    tokens.resize(4096);

    std::chrono::steady_clock clk;
    auto t0 = clk.now();

    int pos = 0;

    Tensor freqs_cos;
    Tensor freqs_sin;

    while(1) {
      std::cout << "User: " << std::flush;
      string user_input;
      getline(cin, user_input);

      string format_input = "<|im_start|>user\n" + user_input + "<|im_end|>\n<|im_start|>assistant\n";

      // tokenize input
      auto tokens_input = tokenizer->encode(format_input, 4096);
      for(int i=0; i<tokens_input.size(); i++) {
        tokens[prompt_end] = tokens_input[i];
        prompt_end++;
      }

      std::cout << "AI: " << std::flush;
      int output_length = 0;

      // feed forward
      for (; pos < token_count; pos++) {
        inputData[0] = (float)tokens[pos];
        inputTensor.set_data(inputData);

        vector<int> posShape;
        posShape.push_back(pos + 1);
        posShape.push_back(head_dim / 2);
        freqs_cos.set_shape(posShape);
        freqs_sin.set_shape(posShape);

        int pos_len = (pos + 1) * head_dim / 2;
        memcpy(freqs_cos.get_data()->data(), freqs_cos_table.data(), pos_len*sizeof(float));
        memcpy(freqs_sin.get_data()->data(), freqs_sin_table.data(), pos_len*sizeof(float));

        graph->input("in", &inputTensor);
        graph->input("freqs_cos", &freqs_cos);
        graph->input("freqs_sin", &freqs_sin);

        for(int i = 0; i < n_layers; i++)
        {
          auto layer_name = std::to_string(i);
          auto kc_name = "k_cache_" + layer_name;
          auto vc_name = "v_cache_" + layer_name;
          graph->input(kc_name.c_str(), kcache[i]);
          graph->input(vc_name.c_str(), vcache[i]);
        }

        Tensor* output;
        graph->extract("output", output);

        //kvcache输出
        for(int i = 0; i < n_layers; i++)
        {
          Tensor* kcache_out;
          graph->get_result(std::to_string(kcache_start+i*kvcache_step), kcache_out);
          kcache[i] = kcache_out;

          Tensor* vcache_out;
          graph->get_result(std::to_string(vcache_start+i*kvcache_step), vcache_out);
          vcache[i] = vcache_out;
        }

        if (pos < prompt_end - 1) continue;
        tokens[pos+1] = sample(*output->get_data(), temp, topp, topk);
        output_length++;
        std::cout << tokenizer->decode({tokens[pos+1]}) << std::flush;
        if((151643 == tokens[pos+1]) || (151645 == tokens[pos+1])) {
          std::cout << std::endl;
          prompt_end += output_length;
          pos++;
          break;
        }
        if (pos == 0) t0 = clk.now();
      }
    }

    std::cout << std::endl;

    auto t1 = clk.now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cerr << (token_count - 1) * 1000 / elapsed.count() << " tokens/s"
              << std::endl;

    delete graph;

    destroyCublas();

    exit(0);
}
