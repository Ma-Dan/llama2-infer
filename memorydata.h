#ifndef MEMORYDATA_H
#define MEMORYDATA_H

#include <vector>

#include "layer.h"
#include "tensor.h"

class MemoryData: public Layer
{
    public:
        MemoryData();
        ~MemoryData();

        int load_model(const vector<string> &params, FILE* fp);
        void forward(vector<Tensor*> &input, vector<Tensor*> &output);

        static int CreateInstance(Layer* &layer);

    private:
        Tensor* _weight;
};

#endif