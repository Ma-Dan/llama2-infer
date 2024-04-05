#ifndef LAYER_H
#define LAYER_H

#include <string>
#include <vector>

using namespace std;

#include "tensor.h"

class Layer
{
    public:
        Layer();
        virtual ~Layer();

        virtual int load_model(const vector<string> &params, FILE* fp);
        virtual void forward(vector<Tensor*> &input, Tensor* output);

        static int CreateInstance(Layer* &layer);
};

#endif