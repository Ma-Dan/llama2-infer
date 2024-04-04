#ifndef LAYER_H
#define LAYER_H

#include <vector>

#include "tensor.h"

class Layer
{
    public:
        Layer();
        virtual void forward();

    private:
        std::vector<Tensor> input;
        Tensor output;
};

#endif