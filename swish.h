#ifndef SWISH_H
#define SWISH_H

#include <vector>

#include "layer.h"
#include "tensor.h"

class Swish: public Layer
{
    public:
        Swish();
        ~Swish();

        int load_model(const vector<string> &params, FILE* fp);
        void forward(vector<Tensor*> &input, vector<Tensor*> &output);

        static int CreateInstance(Layer* &layer);
};

#endif