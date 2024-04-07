#ifndef UNARYOP_H
#define UNARYOP_H

#include <vector>

#include "layer.h"
#include "tensor.h"

class UnaryOp: public Layer
{
    public:
        UnaryOp();
        ~UnaryOp();

        int load_model(const vector<string> &params, FILE* fp);
        void forward(vector<Tensor*> &input, vector<Tensor*> &output);

        static int CreateInstance(Layer* &layer);

    private:
        int _op_type;
};

#endif