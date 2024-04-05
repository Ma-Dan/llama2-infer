#ifndef EMBED_H
#define EMBED_H

#include <vector>

#include "layer.h"
#include "tensor.h"

class Embed: public Layer
{
    public:
        Embed();
        ~Embed();

        int load_model(const vector<string> &params, FILE* fp);
        void forward(vector<Tensor*> &input, Tensor* output);

        static int CreateInstance(Layer* &layer);

    private:
        Tensor* weight;
};

#endif