#ifndef RESHAPE_H
#define RESHAPE_H

#include <vector>

#include "layer.h"
#include "tensor.h"

class Reshape: public Layer
{
    public:
        Reshape();
        ~Reshape();

        int load_model(const vector<string> &params, FILE* fp);
        void forward(vector<Tensor*> &input, vector<Tensor*> &output);

        static int CreateInstance(Layer* &layer);

    private:
        vector<int> _dims;
};

#endif