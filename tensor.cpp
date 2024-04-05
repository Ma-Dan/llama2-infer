#include "tensor.h"

Tensor::Tensor()
{

}

Tensor::~Tensor()
{
    clear();
}

int Tensor::set_shape(const vector<int> shape)
{
    clear();

    int size = 1;
    for(int i=0; i<shape.size(); i++)
    {
        shape_.push_back(shape[i]);
        size *= shape[i];
    }

    data_.resize(size);

    return 0;
}

int Tensor::load_data(FILE *fp)
{
    fread(data_.data(), data_.size(), sizeof(float), fp);
    return 0;
}

void Tensor::clear()
{
    shape_.clear();
    data_.clear();
}