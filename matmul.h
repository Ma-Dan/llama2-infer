#ifndef MATMUL_H
#define MATMUL_H

#include <vector>

#include "layer.h"
#include "tensor.h"

class Matmul: public Layer
{
    public:
        Matmul();
        ~Matmul();

        int load_model(const vector<string> &params, FILE* fp);
        void forward(vector<Tensor*> &input, vector<Tensor*> &output);

        static int CreateInstance(Layer* &layer);
};

#endif