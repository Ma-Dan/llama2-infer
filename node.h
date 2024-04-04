#ifndef NODE_H
#define NODE_H

#include <vector>

#include "tensor.h"
#include "layer.h"

class Node
{
    public:
        Node();

        void forward();

    private:
        std::vector<Tensor> input;
        Tensor output;
        Layer operator;
};

#endif