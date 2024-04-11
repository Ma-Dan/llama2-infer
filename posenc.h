#ifndef RESHAPE_H
#define RESHAPE_H

#include <vector>

#include "layer.h"
#include "tensor.h"

class Posenc: public Layer
{
    public:
        Posenc();
        ~Posenc();

        int load_model(const vector<string> &params, FILE* fp);
        void forward(vector<Tensor*> &input, vector<Tensor*> &output);

        static int CreateInstance(Layer* &layer);
    private:
        int _use_last;
};

#endif