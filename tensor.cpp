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
        _shape.push_back(shape[i]);
        size *= shape[i];
    }

    _data.resize(size);

    return 0;
}

void Tensor::set_data(const vector<float> data)
{
    _data.resize(data.size());
    memcpy(_data.data(), data.data(), data.size()*sizeof(float));
}

vector<float> Tensor::get_data()
{
    return _data;
}

vector<int> Tensor::get_shape()
{
    return _shape;
}

int Tensor::load_data(FILE *fp)
{
    fread(_data.data(), _data.size(), sizeof(float), fp);
    return 0;
}

void Tensor::clear()
{
    _shape.clear();
    _data.clear();
}