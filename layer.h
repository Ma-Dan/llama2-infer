#ifndef LAYER_H
#define LAYER_H

#include <vector>

#include "tensor.h"

class Layer
{
    public:
        Layer();
        virtual void forward();

        static int CreateInstance(Layer& layer);

    protected:
        std::vector<Tensor> input;
        Tensor output;
};

#endif