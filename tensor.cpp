#include "tensor.h"
#include "utils.h"

Tensor::Tensor()
{

}

Tensor::~Tensor()
{
    clear();
}

int Tensor::set_shape(const vector<int> shape)
{
    if(is_same_shape(shape, _shape))
    {
        return 0;
    }

    int oldSize = 1;
    for(int i=0; i<_shape.size(); i++)
    {
        oldSize *= _shape[i];
    }

    if(_shape.size() == 0)
    {
        oldSize = 0;
    }

    int newSize = 1;
    for(int i=0; i<shape.size(); i++)
    {
        newSize *= shape[i];
    }

    _shape.clear();
    for(int i=0; i<shape.size(); i++)
    {
        _shape.push_back(shape[i]);
    }

    if(newSize != oldSize)
    {
        _data.resize(newSize);
    }

    return 0;
}

void Tensor::set_data(const vector<float> data)
{
    _data.resize(data.size());
    memcpy(_data.data(), data.data(), data.size()*sizeof(float));
}

vector<float>* Tensor::get_data()
{
    return &_data;
}

vector<int> Tensor::get_shape()
{
    return _shape;
}

int Tensor::load_data(FILE *fp, long offset)
{
    fseek(fp, offset, SEEK_SET);
    fread(_data.data(), _data.size(), sizeof(float), fp);
    return 0;
}

void Tensor::clear()
{
    _shape.clear();
    _data.clear();
}