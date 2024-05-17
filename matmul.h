#ifndef MATMUL_H
#define MATMUL_H

#include <vector>

#include "layer.h"
#include "tensor.h"

enum
{
    Matmul_CPU = 0,
    Matmul_GPU
};

class Matmul: public Layer
{
    public:
        Matmul();
        ~Matmul();

        int load_model(const vector<string> &params, FILE* fp);
        void forward(vector<Tensor*> &input, vector<Tensor*> &output);

        static int CreateInstance(Layer* &layer);
    private:
        int _matmul_type;
        float* _input_data;
        float* _output_data;

        float* _input_softmax;
        float* _input_v;
        float* _output_attention;
};

#endif