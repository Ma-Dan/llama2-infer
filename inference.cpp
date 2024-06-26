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

#include "tensor.h"
#include "graph.h"
#include "utils.h"

const int vocab_size = 32000;

float temp = 1, topp = 0.9;
const int topk = 300;

struct bpe {
    int max_token_length;
    std::vector<std::string> vocab;
    std::unordered_map<std::string, int> lookup;
    std::vector<float> scores;

    void load(std::string path);
    std::vector<int> encode(std::string s);
};

void bpe::load(std::string path) {
    vocab.resize(vocab_size);
    scores.resize(vocab_size);
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) exit(1);
    fread(&max_token_length, sizeof(int), 1, f);
    std::vector<char> s(max_token_length + 1);
    for (int i = 0; i < vocab_size; i++) {
        fread(scores.data() + i, sizeof(float), 1, f);
        int len;
        fread(&len, sizeof(int), 1, f);
        fread(s.data(), sizeof(char) * len, 1, f);
        s[len] = 0;
        vocab[i] = s.data();
        lookup[vocab[i]] = i;
    }
    fclose(f);
}

typedef struct {
    const char *str;
    int id;
} TokenIndex;

int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = (TokenIndex *)bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

std::vector<int> bpe::encode(std::string s) {
    if (s.length() && s[0] != ' ') s = " " + s;

    // sort vocabulary
    TokenIndex *sorted_vocab = (TokenIndex *)malloc(vocab_size * sizeof(TokenIndex));
    for (int i = 0; i < vocab_size; i++) {
        sorted_vocab[i].str = vocab[i].c_str();
        sorted_vocab[i].id = i;
    }
    qsort(sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);

    char* str_buffer = (char *)malloc((s.length()*2 +1 +2) * sizeof(char));
    size_t str_len = 0;

    std::vector<int> tokens;
    for (const char *c = s.c_str(); *c != '\0'; c++) {
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, sorted_vocab, vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens.push_back(id);
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens.push_back((unsigned char)str_buffer[i] + 3);
            }
        }
        str_len = 0;
    }

    while (true) {
        float best_score = -1e10;
        int best_index = -1, best_token = -1;

        for (size_t i = 0; i + 1 < tokens.size(); i++) {
            auto merged = vocab[tokens[i]] + vocab[tokens[i + 1]];
            if (lookup.count(merged) && scores[lookup[merged]] > best_score) {
                best_score = scores[lookup[merged]];
                best_index = i;
                best_token = lookup[merged];
            }
        }

        if (best_token == -1) break;

        tokens[best_index] = best_token;
        tokens.erase(tokens.begin() + best_index + 1);
    }

    free(str_buffer);
    free(sorted_vocab);

    return tokens;
}

int sample(const std::vector<float>& logits, float temp, float topp, int topk) {
    // return std::max_element(logits.begin(), logits.end()) - logits.begin();

    assert(logits.size() == vocab_size);

    if (fabsf(temp) < 1e-8)
        return std::max_element(logits.begin(), logits.end()) - logits.begin();

    static std::mt19937_64 rng(3407);  // haha
    static std::uniform_real_distribution<float> dist(0, 1);

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

    std::string model_name = argv[1];
    std::string prompt = argv[2];
    std::string tokenizer_path = model_name + "_tokenizer.bin";
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
            auto x = i / pow(10000.0, j * 2 / (double)head_dim);
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

    // tokenize prompt
    bpe tokenizer;
    tokenizer.load(tokenizer_path);

    auto tokens = tokenizer.encode(prompt);
    tokens.insert(tokens.begin(), 1);  // bos
    int prompt_end = tokens.size();
    tokens.resize(token_count);

    // 加载模型
    Graph* graph = new Graph();
    graph->load_model(model_name);

    Tensor inputTensor;
    vector<int> inputShape;
    inputShape.push_back(1);
    inputTensor.set_shape(inputShape);
    vector<float> inputData;
    inputData.push_back(0);

    std::chrono::steady_clock clk;
    auto t0 = clk.now();

    // feed forward
    Tensor freqs_cos;
    Tensor freqs_sin;
    for (int pos = 0; pos < token_count; pos++) {
        std::cout << tokenizer.vocab[tokens[pos]] << std::flush;

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
        /*#pragma omp parallel for
        for (int i = 0; i < (pos + 1) * head_dim / 2; i++) {
          freqs_cos.get_data()->data()[i] = freqs_cos_table[i];
          freqs_sin.get_data()->data()[i] = freqs_sin_table[i];
        }*/

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
        tokens[pos + 1] = sample(*output->get_data(), temp, topp, topk);
        if (pos == 0) t0 = clk.now();
    }
    std::cout << std::endl;

    auto t1 = clk.now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0);
    std::cerr << (token_count - 1) * 1000 / elapsed.count() << " tokens/s"
              << std::endl;

    delete graph;

    exit(0);
}
