#include "tensor.h"
#include "utils.h"

Tensor::Tensor()
{
    _device_type = Tensor_CPU;

    _data = new vector<float>();
}

Tensor::~Tensor()
{
    clear();
}

int Tensor::get_device_type()
{
    return _device_type;
}

void Tensor::set_device_type(int device_type)
{
    _device_type = device_type;
}

float* Tensor::get_device_data()
{
    return _device_data;
}

int Tensor::set_shape(const vector<int> &shape)
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
        _data->resize(newSize);
    }

    return 0;
}

void Tensor::set_data(const vector<float> &data)
{
    _data->resize(data.size());
    memcpy(_data->data(), data.data(), data.size()*sizeof(float));
}

void Tensor::set_shape_data(const vector<int> &shape, const vector<float> *data)
{
    _shape.clear();
    for(int i=0; i<shape.size(); i++)
    {
        _shape.push_back(shape[i]);
    }

    _data = (vector<float>*)data;
}

vector<float>* Tensor::get_data()
{
    return _data;
}

vector<int> Tensor::get_shape()
{
    return _shape;
}

int Tensor::get_size()
{
    int size = 1;
    for(int i=0; i<_shape.size(); i++)
    {
        size *= _shape[i];
    }

    return size;
}

int Tensor::load_data(FILE *fp, long offset)
{
    fseek(fp, offset, SEEK_SET);
    fread(_data->data(), _data->size(), sizeof(float), fp);

    return 0;
}

void Tensor::clear()
{
    _shape.clear();
    _data->clear();

    delete _data;
}