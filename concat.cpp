#include "concat.h"
#include "layer_register.h"
#include "utils.h"

Concat::Concat()
{
}

Concat::~Concat()
{
}

int Concat::load_model(const vector<string> &params, FILE* fp)
{
    vector<string> dim_param = split(params[7], "=");
    _dim = atoi(dim_param[1].c_str());

    return 0;
}

void Concat::forward(vector<Tensor*> &input, vector<Tensor*> &output)
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

    vector<float>* input0Data = input[0]->get_data();
    vector<float>* input1Data = input[1]->get_data();
    vector<int> input0Shape = input[0]->get_shape();
    vector<int> input1Shape = input[1]->get_shape();

    //判断维度是否合法
    if(input0Shape.size() != input1Shape.size())
    {
        return;
    }
    for(int i=0; i<input0Shape.size(); i++)
    {
        if(i == _dim)
        {
            continue;
        }
        if(input0Shape[i] != input1Shape[i])
        {
            return;
        }
    }

    vector<int> outputShape;
    int concatSize = input0Shape[_dim] + input1Shape[_dim];
    for(int i=0; i<input0Shape.size(); i++)
    {
        if(i == _dim)
        {
            outputShape.push_back(concatSize);
        }
        else
        {
            outputShape.push_back(input0Shape[i]);
        }
    }

    int offset = input0Data->size();

    result->set_shape(outputShape);
    vector<float>* outputData = result->get_data();

    //TODO:处理更多维度情况
    //本示例中只有0维度拼接，所以直接memcpy
    /*if(input0Data->size() > 0)
    {
        memcpy(result->get_data()->data(), input0Data->data(), input0Data->size()*sizeof(float));
    }*/

    if(input1Data->size() > 0)
    {
        memcpy(&result->get_data()->data()[offset], input1Data->data(), input1Data->size()*sizeof(float));
    }

    output[0] = result;
}

int Concat::CreateInstance(Layer* &layer)
{
    layer = new Concat();
    return 0;
}

LayerRegistererWrapper concatCreateInstance("Concat_t", Concat::CreateInstance);