#ifndef CONCAT_H
#define CONCAT_H

#include <vector>

#include "layer.h"
#include "tensor.h"

class Concat: public Layer
{
    public:
        Concat();
        ~Concat();

        int load_model(const vector<string> &params, FILE* fp);
        void forward(vector<Tensor*> &input, vector<Tensor*> &output);

        static int CreateInstance(Layer* &layer);

    private:
    int _dim;
};

#endif