#include "reshape.h"
#include "layer_register.h"
#include "utils.h"
#include "math.h"

Reshape::Reshape()
{
}

Reshape::~Reshape()
{
}

int Reshape::load_model(const vector<string> &params, FILE* fp)
{
    vector<string> dims_param = split(params[6], "=");
    int dims = atoi(dims_param[1].c_str());

    _dims.resize(dims);
    for(int i=0; i<dims; i++)
    {
        vector<string> size_param = split(params[7+i], "=");
        _dims[i] = atoi(size_param[1].c_str());
    }

    return 0;
}

void Reshape::forward(vector<Tensor*> &input, vector<Tensor*> &output)
{
    Tensor* result;
    if(output[0] == nullptr)
    {
        result = new Tensor();
    }
    else
    {
        result = output[0];
    }

    //TODO:进行更多形状检查
    vector<int> inputShape = input[0]->get_shape();
    vector<int> outputShape;
    for(int i=0; i<_dims.size(); i++)
    {
        if(_dims[i] == -1)
        {
            outputShape.push_back(inputShape[i]);
        }
        else
        {
            outputShape.push_back(_dims[i]);
        }
    }
    //result->set_shape(outputShape);
    //memcpy(result->get_data()->data(), input[0]->get_data()->data(), input[0]->get_data()->size()*sizeof(float));
    result->set_shape_data(outputShape, input[0]->get_data());

    output[0] = result;
}

int Reshape::CreateInstance(Layer* &layer)
{
    layer = new Reshape();
    return 0;
}

LayerRegistererWrapper reshapeCreateInstance("Reshape_t", Reshape::CreateInstance);