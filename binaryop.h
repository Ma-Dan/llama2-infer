#ifndef BINARYOP_H
#define BINARYOP_H

#include <vector>

#include "layer.h"
#include "tensor.h"

class BinaryOp: public Layer
{
    public:
        BinaryOp();
        ~BinaryOp();

        int load_model(const vector<string> &params, FILE* fp);
        void forward(vector<Tensor*> &input, vector<Tensor*> &output);

        static int CreateInstance(Layer* &layer);

    private:
        int _op_type;
        int _param_type;
        float _param;
};

#endif