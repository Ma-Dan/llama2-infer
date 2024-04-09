#ifndef REDUCTION_H
#define REDUCTION_H

#include <vector>

#include "layer.h"
#include "tensor.h"

class Softmax: public Layer
{
    public:
        Softmax();
        ~Softmax();

        int load_model(const vector<string> &params, FILE* fp);
        void forward(vector<Tensor*> &input, vector<Tensor*> &output);

        static int CreateInstance(Layer* &layer);

    private:
        int _dim;
};

#endif