#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

class Tensor
{
    public:
        Tensor();

    private:
        std::vector<int> shape;
        std::vector<float> data;
};

#endif