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

const int vocab_size = 32000;

const float temp = 1, topp = 0.9;
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

// ./inference MODEL PROMPT OUT-TOKEN-COUNT
int main(int argc, char** argv) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " MODEL PROMPT OUT-TOKEN-COUNT"
                  << std::endl;
        return 1;
    }

    std::string model_name = argv[1],
                tokenizer_path = "tokenizer.bin", prompt = argv[2];
    int token_count = std::stoi(argv[3]);

    // 模型的基本参数
    int ctx_length, n_layers, dim, n_heads;
    ctx_length = 256;
    n_layers = 8;
    dim = 512;
    n_heads = 8;

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

    // for (int i = 0; i < token_count; i++) std::cout << tokens[i] << " ";
    // std::cout << std::endl;

    // 加载模型
    Graph* graph = new Graph();
    graph->load_model(model_name);

    int pos = 0;

    Tensor freqs_cos;
    Tensor freqs_sin;
    vector<int> posShape;
    posShape.push_back(pos + 1);
    posShape.push_back(head_dim / 2);
    freqs_cos.set_shape(posShape);
    freqs_sin.set_shape(posShape);
    for (int i = 0; i < (pos + 1) * head_dim / 2; i++) {
        freqs_cos.get_data()->data()[i] = freqs_cos_table[i];
        freqs_sin.get_data()->data()[i] = freqs_sin_table[i];
    }

    Tensor inputTensor;
    vector<int> inputShape;
    vector<float> inputData;
    inputShape.push_back(1);
    inputData.push_back(1.0);
    inputTensor.set_shape(inputShape);
    inputTensor.set_data(inputData);

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
    graph->extract("17", output);

    delete graph;

    exit(0);
}
